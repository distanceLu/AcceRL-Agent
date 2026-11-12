"""Real CPU/Gloo coverage for fixed-episode FSDP window coordination."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from accerl_agent.test_fixed_episode_batch import _production, _sample


def _worker(rank, store):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        production = _production(dist)

        def make_worker(length):
            worker = production["Worker"]()
            worker.args = SimpleNamespace(
                global_batch_size=192, grad_accum_steps=None,
                train_max_sequences_per_pack=0, train_token_budget=8192,
                rl_algorithm="ppo", ppo_gae_mode="jit", max_length=4096,
            )
            worker.fsdp_world_size = 3
            worker.rank = rank
            worker.device = torch.device("cpu")
            worker.loss_pg = dist.group.WORLD
            worker.research = None
            worker.optimizer_step = 0
            samples = [_sample(rank * 64 + i, length) for i in range(64)]
            worker._sample_fixed_episode_batch = lambda: samples
            return worker

        # Initially [32, 16, 8] local packs: all ranks must execute 32 after
        # subdividing packs, while consuming exactly their original 64 samples.
        worker = make_worker([4096, 2048, 1024][rank])
        window = worker._prepare_varlen_optimizer_window(2)
        assert len(window) == 32
        assert sum(pack.sample_count for pack in window) == 64
        identifiers = []
        parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        valid_tokens, trajectories, stats = worker._reduce_varlen_window_counts(window)
        assert valid_tokens == trajectories == 192
        assert stats["global_pack_count"] == 96
        for pack in window:
            assert 0 < pack.total_token_count <= 8192
            starts = pack.batch["cu_seqlens"][:-1].long()
            ids = pack.batch["input_ids"][0, starts].double()
            identifiers.extend(ids.tolist())
            # Same global trajectory normalization as the production loss:
            # subdivision must not add an extra divide by microbatch count.
            ((parameter * (ids + 1) / 192).square().sum() / trajectories).backward()
        assert sorted(identifiers) == list(range(rank * 64, (rank + 1) * 64))
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        all_ids = torch.arange(1, 193, dtype=torch.float64) / 192
        torch.testing.assert_close(parameter.grad, 4 * all_ids.square().mean())

        # Next update gets a different pack count; no fixed accumulation
        # setting or previous window's MAX may leak into this window.
        worker = make_worker(2048)
        window = worker._prepare_varlen_optimizer_window(3)
        assert len(window) == 16
        assert worker._reduce_varlen_window_counts(window)[1] == 192

        # Failures on a single rank before MAX and after MAX must make every
        # rank exit the window, instead of entering mismatched collectives.
        for phase in ("sampling", "collation"):
            worker = make_worker(2048)
            if rank == 1:
                def fail(*args, **kwargs):
                    raise ValueError("injected single-rank failure")
                if phase == "sampling":
                    worker._sample_fixed_episode_batch = fail
                else:
                    worker._make_prepared_varlen_pack = fail
            try:
                worker._prepare_varlen_optimizer_window(4)
            except RuntimeError:
                failed = torch.tensor(1)
            else:
                failed = torch.tensor(0)
            dist.all_reduce(failed, op=dist.ReduceOp.SUM)
            assert failed.item() == 3
    finally:
        dist.destroy_process_group()


class FixedEpisodeDistributedTests(unittest.TestCase):
    def test_three_rank_alignment_normalization_and_failure_coordination(self):
        with tempfile.TemporaryDirectory(prefix="accerl-fixed-batch-") as directory:
            mp.spawn(_worker, args=(str(Path(directory) / "store"),), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
