"""CPU integration checks for production fixed-episode trainer methods.

The source loader keeps the real sampler, PPO collator, window coordinator,
and CLI code while replacing Ray RPCs and collectives with CPU stand-ins.
It does not import or initialize the model, vLLM, Ray, or CUDA runtime.
"""

import __future__
import argparse
import ast
import contextlib
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as torch_dist

from accerl_agent.rl_data import GRPOSample, RawPPOSample
from accerl_agent.varlen_batching import pack_sequence_indices, split_packs_to_count


SOURCE = Path(__file__).with_name("agent_textworld.py")


class _Collectives:
    ReduceOp = torch_dist.ReduceOp

    def __init__(self, world_size=3, target_pack_count=None, peer_failed=False):
        self.world_size = world_size
        self.target_pack_count = target_pack_count
        self.peer_failed = peer_failed
        self.operations = []

    def all_reduce(self, tensor, *, op, group):
        self.operations.append(op)
        if op == self.ReduceOp.SUM:
            tensor.mul_(self.world_size)
        elif op == self.ReduceOp.MAX and self.target_pack_count is not None:
            tensor.fill_(max(tensor.item(), self.target_pack_count))
        elif op == self.ReduceOp.MIN and self.peer_failed:
            tensor.zero_()


def _production(collectives):
    tree = ast.parse(SOURCE.read_text())
    worker_source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPTrainWorker")
    names = {
        "_prepare_rl_sample", "_collate_prepared_rl_samples", "_version_lag_stats",
        "_wait_for_replay_warmup", "_sample_fixed_episode_batch", "_make_prepared_varlen_pack",
        "_prepare_fixed_episode_window", "_prepare_varlen_optimizer_window", "_reduce_varlen_window_counts",
        "_validate_prepared_packed_ppo_batch", "_validate_ppo_flat_token_view", "_validate_ppo_boundary_layout",
    }
    methods = [n for n in worker_source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise AssertionError("Production fixed-episode trainer interface is incomplete")
    worker_class = ast.ClassDef(name="Worker", bases=[], keywords=[], body=methods, decorator_list=[])
    top_level = [n for n in tree.body if (
        isinstance(n, ast.ClassDef) and n.name in {"PreparedVarlenPack", "PPOFlatTokenView"}
    ) or (
        isinstance(n, ast.FunctionDef) and n.name in {"_make_ppo_varlen_batch_unchecked", "parse_args", "validate_args"}
    )]
    module = ast.fix_missing_locations(ast.Module(body=top_level + [worker_class], type_ignores=[]))
    namespace = dict(
        __name__=__name__, argparse=argparse, os=os, time=time, torch=torch, math=math,
        dataclass=dataclass, RawPPOSample=RawPPOSample, GRPOSample=GRPOSample,
        ray=SimpleNamespace(get=lambda result: result), dist=collectives,
        pack_sequence_indices=pack_sequence_indices, split_packs_to_count=split_packs_to_count,
        pick_dtype=lambda value: None, load_textworld_game_files=lambda args: [],
    )
    exec(compile(module, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def _sample(identifier, length=3000, frozen=False):
    return RawPPOSample(
        input_ids=(identifier,) * length,
        response_spans=((1, 2),),
        response_logprobs=(-0.5,), response_rewards=(1.0,),
        boundary_kind="terminated", behavior_version=2,
        telemetry_id=str(identifier), created_monotonic_ns=identifier,
        raw_advantages=(0.5,) if frozen else None,
        returns=(1.5,) if frozen else None,
        value_version=2 if frozen else None,
    )


class FixedEpisodeWindowTests(unittest.TestCase):
    def worker(self, samples=None, *, pack_target=None, peer_failed=False, **overrides):
        self.collectives = _Collectives(target_pack_count=pack_target, peer_failed=peer_failed)
        self.production = _production(self.collectives)
        worker = self.production["Worker"]()
        worker.args = SimpleNamespace(
            global_batch_size=192, grad_accum_steps=None, train_max_sequences_per_pack=0,
            train_token_budget=8192, max_length=4096, rl_algorithm="ppo", ppo_gae_mode="jit",
            replay_sample_timeout_seconds=0.0, replay_wait_sleep_seconds=0.0,
            min_replay_size_per_rank=64,
        )
        for key, value in overrides.items():
            setattr(worker.args, key, value)
        worker.fsdp_world_size = 3
        worker.rank = 0
        worker.device = torch.device("cpu")
        worker.loss_pg = object()
        worker.research = None
        worker.optimizer_step = 4
        worker.pending_prepared_samples = []
        worker.get_replay_stats = Mock(return_value={"size": 64, "capacity": 64})
        worker._wait_for_replay_warmup = Mock(return_value={"size": 64, "capacity": 64})
        samples = samples if samples is not None else [_sample(i) for i in range(64)]
        remote = Mock(side_effect=lambda count: samples[:count])
        worker.replay_buffer = SimpleNamespace(sample=SimpleNamespace(remote=remote))
        return worker

    def assert_window(self, window, expected_packs):
        self.assertEqual(len(window), expected_packs)
        self.assertEqual(sum(pack.sample_count for pack in window), 64)
        self.assertEqual(sum(pack.valid_trajectory_count for pack in window), 64)
        identifiers = []
        for pack in window:
            self.assertGreater(pack.sample_count, 0)
            self.assertLessEqual(pack.total_token_count, 8192)
            starts = pack.batch["cu_seqlens"][:-1].long()
            identifiers.extend(pack.batch["input_ids"][0, starts].tolist())
        self.assertEqual(sorted(identifiers), list(range(64)))

    def test_exact_rank_quota_is_selected_before_packing(self):
        worker = self.worker()
        samples = worker._sample_fixed_episode_batch()
        self.assertEqual(len(samples), 64)
        worker._wait_for_replay_warmup.assert_called_once_with()
        worker.replay_buffer.sample.remote.assert_called_once_with(64)

    def test_rejected_oversized_sample_is_replaced_without_exceeding_quota(self):
        worker = self.worker()
        worker.replay_buffer.sample.remote.side_effect = [
            [_sample(100, 4097)] + [_sample(i) for i in range(63)],
            [_sample(63)],
        ]
        samples = worker._sample_fixed_episode_batch()
        self.assertEqual([len(s.input_ids) for s in samples], [3000] * 64)
        self.assertEqual(sorted(int(s.telemetry_id) for s in samples), list(range(64)))
        self.assertEqual([c.args[0] for c in worker.replay_buffer.sample.remote.call_args_list], [64, 1])

    def test_short_buffer_fetches_are_topped_up_to_exact_quota(self):
        worker = self.worker()
        worker.replay_buffer.sample.remote.side_effect = [
            [_sample(i) for i in range(17)], [_sample(i) for i in range(17, 64)],
        ]
        self.assertEqual(len(worker._sample_fixed_episode_batch()), 64)
        self.assertEqual([c.args[0] for c in worker.replay_buffer.sample.remote.call_args_list], [64, 47])

    def test_two_episodes_per_pack_produces_32_microbatches(self):
        worker = self.worker()
        window = worker._prepare_varlen_optimizer_window(7)
        self.assert_window(window, 32)
        self.assertTrue(all(pack.sample_count == 2 for pack in window))
        self.assertEqual(self.collectives.operations, [torch_dist.ReduceOp.MIN, torch_dist.ReduceOp.MAX, torch_dist.ReduceOp.MIN])

    def test_four_episodes_per_pack_produces_16_microbatches(self):
        worker = self.worker([_sample(i, 2000) for i in range(64)])
        window = worker._prepare_varlen_optimizer_window(7)
        self.assert_window(window, 16)
        self.assertTrue(all(pack.sample_count == 4 for pack in window))

    def test_token_only_packing_can_exceed_two_episodes(self):
        worker = self.worker([_sample(i, 1000) for i in range(64)])
        window = worker._prepare_varlen_optimizer_window(7)
        self.assert_window(window, 8)
        self.assertTrue(all(pack.sample_count == 8 for pack in window))

    def test_rank_with_fewer_packs_splits_to_peer_max_without_duplication(self):
        worker = self.worker([_sample(i, 2000) for i in range(64)], pack_target=32)
        window = worker._prepare_varlen_optimizer_window(7)
        self.assert_window(window, 32)

    def test_collation_preserves_frozen_targets_and_telemetry(self):
        worker = self.worker([_sample(i, 2000, frozen=True) for i in range(64)], ppo_gae_mode="rollout")
        worker.research = SimpleNamespace(emit=Mock())
        window = worker._prepare_varlen_optimizer_window(7)
        self.assert_window(window, 16)
        self.assertEqual(sum(len(pack.telemetry_samples) for pack in window), 64)
        self.assertEqual(sum(pack.version_lag_sum for pack in window), 5 * 64)
        for pack in window:
            self.assertEqual(pack.batch["raw_advantages"].tolist(), [0.5] * pack.sample_count)
            self.assertEqual(pack.batch["returns"].tolist(), [1.5] * pack.sample_count)
        emitted = [call for call in worker.research.emit.call_args_list if call.args[0] == "training_window_ready"]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0].kwargs["samples"]), 64)

    def test_global_counts_match_192_and_actual_pack_count(self):
        worker = self.worker([_sample(i, 2000) for i in range(64)])
        window = worker._prepare_varlen_optimizer_window(7)
        tokens, trajectories, stats = worker._reduce_varlen_window_counts(window)
        self.assertEqual((tokens, trajectories), (192, 192))
        self.assertEqual(stats["global_pack_count"], 48)
        self.assertEqual(stats["global_pack_token_count"], 192 * 2000)

    def test_count_mismatch_fails_before_training(self):
        worker = self.worker()
        window = worker._prepare_varlen_optimizer_window(7)
        worker.args.global_batch_size = 195
        with self.assertRaisesRegex(RuntimeError, "batch size mismatch"):
            worker._reduce_varlen_window_counts(window)

    def test_peer_sampling_failure_does_not_enter_pack_count_collective(self):
        worker = self.worker(peer_failed=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed to prepare"):
            worker._prepare_varlen_optimizer_window(7)
        self.assertEqual(self.collectives.operations, [torch_dist.ReduceOp.MIN, torch_dist.ReduceOp.MIN])

    def test_local_collation_failure_is_coordinated_after_max_collective(self):
        worker = self.worker()
        worker._make_prepared_varlen_pack = Mock(side_effect=ValueError("bad metadata"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed to prepare"):
            worker._prepare_varlen_optimizer_window(7)
        self.assertEqual(self.collectives.operations, [torch_dist.ReduceOp.MIN, torch_dist.ReduceOp.MAX, torch_dist.ReduceOp.MIN])

    def test_legacy_mode_still_prepares_fixed_pack_count(self):
        worker = self.worker(global_batch_size=None, grad_accum_steps=5)
        one = worker._make_prepared_varlen_pack([_sample(0)], 7)
        worker._next_varlen_cpu_pack = Mock(return_value=one)
        window = worker._prepare_varlen_optimizer_window(7)
        self.assertEqual(len(window), 5)
        self.assertEqual(worker._next_varlen_cpu_pack.call_count, 5)
        worker.replay_buffer.sample.remote.assert_not_called()
        self.assertEqual(self.collectives.operations, [torch_dist.ReduceOp.MIN])


class FixedEpisodeCLITests(unittest.TestCase):
    def setUp(self):
        self.production = _production(_Collectives())
        self.base = [
            "test", "--model-path", "/tmp/model", "--tw-game-dir", "/tmp/games",
            "--train-token-budget", "8192", "--fsdp-world-size", "3",
        ]

    def parse(self, extra):
        with patch.object(sys, "argv", self.base + extra):
            return self.production["parse_args"]()

    def test_legacy_defaults_are_preserved(self):
        args = self.parse([])
        self.assertIsNone(args.global_batch_size)
        self.assertEqual((args.grad_accum_steps, args.train_max_sequences_per_pack), (8, 16))
        self.production["validate_args"](args)

    def test_global_mode_uses_dynamic_accumulation_and_unlimited_sequence_count(self):
        args = self.parse(["--global-batch-size", "192"])
        self.assertIsNone(args.grad_accum_steps)
        self.assertEqual(args.train_max_sequences_per_pack, 0)
        self.assertGreater(args.replay_capacity, 0)
        self.assertGreater(args.min_replay_size_per_rank, 0)
        self.production["validate_args"](args)

    def test_explicit_fixed_and_dynamic_batch_modes_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(["--global-batch-size", "192", "--grad-accum-steps", "32"])

    def test_invalid_global_batch_sizes_are_rejected(self):
        for batch_size in (0, -1, 193):
            with self.subTest(batch_size=batch_size):
                args = self.parse(["--global-batch-size", str(batch_size)])
                with self.assertRaisesRegex(ValueError, "global-batch-size"):
                    self.production["validate_args"](args)

    def test_unlimited_sequence_cap_has_positive_independent_defaults(self):
        args = self.parse(["--train-max-sequences-per-pack", "0", "--grad-accum-steps", "32"])
        self.assertEqual(args.train_max_sequences_per_pack, 0)
        self.assertGreater(args.replay_capacity, 0)
        self.assertGreater(args.train_pack_candidate_pool_size, 0)
        self.assertGreater(args.min_replay_size_per_rank, 0)
        self.production["validate_args"](args)


if __name__ == "__main__":
    unittest.main()
