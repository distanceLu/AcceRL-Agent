"""CPU/Gloo regression for AcceRL's actual PPO window and joint zero graph."""

import copy
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from accerl_agent.ppo_early_stop import K2EarlyStopState, graph_zero


def _worker(rank, store):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=3,
        timeout=timedelta(seconds=60),
    )
    try:
        from accerl_agent.test_accerl_early_stop import AcceRLEarlyStopTests

        fixture = AcceRLEarlyStopTests()
        worker = fixture.worker(clip_mode="gipo")
        worker.rank = rank
        worker.fsdp_world_size = 3
        worker.loss_pg = dist.group.WORLD
        worker._prepare_varlen_optimizer_window = lambda version: fixture.packs(bad=(rank == 1))
        parameters = [worker.logprob.detach().clone(), worker.value.detach().clone()]
        adam = copy.deepcopy(worker.optimizer.state_dict())
        ema = worker.ppo_adv_ema_mean, worker.ppo_adv_ema_variance
        result = worker._run_ppo_optimizer_step(2)
        assert result["early_stop"] == 1 and result["early_stop_microbatch"] == 2
        assert result["global_valid_token_count"] == 27
        assert result["global_valid_trajectory_count"] == 9
        assert result["global_pack_count"] == 9
        assert worker.optimizer_step == 0 and worker.train_micro_step == 3
        assert worker._finalize_optimizer_step.call_count == 0
        assert worker.scheduler.step.call_count == 0
        assert worker.ppo_adv_ema_mean == ema[0] and worker.ppo_adv_ema_variance == ema[1]
        assert worker._ppo_early_stop_state is None
        for parameter, initial in zip((worker.logprob, worker.value), parameters):
            assert torch.equal(parameter, initial) and parameter.grad is None
        current = worker.optimizer.state_dict()
        assert current["param_groups"] == adam["param_groups"]
        for key, values in adam["state"].items():
            assert all(torch.equal(current["state"][key][name], value) for name, value in values.items())

        worker._prepare_varlen_optimizer_window = lambda version: fixture.packs(bad=False)
        result = worker._run_ppo_optimizer_step(2)
        assert result["optimizer_step_skipped"] == 0
        assert worker.optimizer_step == 1 and worker.train_micro_step == 6

        # Response-token weighting and a remote stop also cover unequal
        # local response counts, rather than averaging the rank means.
        count = rank + 1
        zeros = torch.zeros(count)
        state = K2EarlyStopState(0.1)
        delta = torch.ones(count) if rank == 0 else zeros
        assert abs(state.observe(delta, zeros) - 1 / 12) < 1e-12
        assert not state.stop_ppo
        delta = torch.full_like(zeros, 2.0) if rank == 2 else zeros
        state.observe(delta, zeros)
        assert state.stop_ppo

        # Check real DDP backward traverses backbone and both heads, including
        # the value bootstrap branch, when subsequent loss is zero.
        class JointModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 3)
                self.policy = torch.nn.Linear(3, 1)
                self.value = torch.nn.Linear(3, 1)

            def forward(self, inputs):
                hidden = self.backbone(inputs)
                return self.policy(hidden), self.value(hidden)

        model = DistributedDataParallel(JointModel())
        optimizer = torch.optim.Adam(model.parameters())
        before = [parameter.detach().clone() for parameter in model.parameters()]
        state = K2EarlyStopState(0.1)
        for microbatch in range(3):
            logprobs, values = model(torch.ones(count, 2))
            bootstrap = values[-1:]
            delta = torch.full_like(logprobs, 2.0) if microbatch == 1 and rank == 1 else torch.zeros_like(logprobs)
            state.observe(logprobs, logprobs.detach() - delta)
            loss = graph_zero(logprobs, values, bootstrap) if state.stop_ppo else (
                logprobs.square().sum() + values.square().sum() + bootstrap.square().sum()
            )
            loss.backward()
        assert state.stop_ppo
        optimizer.zero_grad(set_to_none=True)
        assert all(torch.equal(parameter, initial) for parameter, initial in zip(model.parameters(), before))
        assert all(parameter.grad is None for parameter in model.parameters())
        assert not optimizer.state
    finally:
        dist.destroy_process_group()


class AcceRLEarlyStopDistributedTests(unittest.TestCase):
    def test_three_rank_latch_discard_reset_and_joint_ddp_backward(self):
        with tempfile.TemporaryDirectory(prefix="accerl-k2-gloo-") as directory:
            mp.spawn(_worker, args=(str(Path(directory) / "store"),), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
