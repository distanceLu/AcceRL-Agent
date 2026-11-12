from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from accerl_agent import agent_textworld as agent
from accerl_agent.rl_data import (
    RawPPOSample, freeze_rollout_ppo_targets, textworld_ppo_boundary_is_terminal,
)
from accerl_agent.rollout_critic import (
    RolloutCriticClient, RolloutCriticWorker, all_worker_rpc,
)


def sample(boundary="terminated"):
    return RawPPOSample(
        input_ids=tuple(range(7)), response_spans=((2, 4), (5, 6)),
        response_logprobs=(-0.2, -0.2, -0.2), response_rewards=(0.0, 1.0, 2.0),
        boundary_kind=boundary, behavior_version=2,
    )


def frozen(boundary="terminated"):
    return freeze_rollout_ppo_targets(
        sample(boundary), [10, 0.5, 0.8, 20, 1.2, 30, 2],
        value_version=7, gamma=0.9, gae_lambda=0.8,
    )


def pack(samples):
    batch = agent._make_ppo_varlen_batch_unchecked(samples)
    return agent.PreparedVarlenPack(
        batch=batch, total_token_count=sum(len(s.input_ids) for s in samples),
        valid_token_count=len(batch["target_indices"]),
        max_seqlen=max(len(s.input_ids) for s in samples),
        version_lag_sum=0, sample_count=len(samples), valid_trajectory_count=len(samples),
    )


class TargetTests(unittest.TestCase):
    def test_terminal_and_truncated_targets_match_scalar_reference(self):
        for boundary in ("terminated", "truncated"):
            with self.subTest(boundary=boundary):
                result = frozen(boundary)
                values = [0.5, 0.8, 1.2]
                bootstrap = 0 if boundary == "terminated" else 2
                expected = [0.0] * 3
                running = 0
                for index in reversed(range(3)):
                    following = values[index + 1] if index < 2 else bootstrap
                    delta = sample().response_rewards[index] + 0.9 * following - values[index]
                    running = delta + 0.9 * 0.8 * running
                    expected[index] = running
                torch.testing.assert_close(torch.tensor(result.raw_advantages), torch.tensor(expected))
                torch.testing.assert_close(torch.tensor(result.returns), torch.tensor(expected) + torch.tensor(values))
                self.assertEqual(result.value_version, 7)
                self.assertEqual(result.response_logprobs, sample().response_logprobs)

    def test_boundary_semantics_unchanged(self):
        for reason in ("won", "lost", "step_limit", "history_limit"):
            self.assertTrue(textworld_ppo_boundary_is_terminal(reason))
        self.assertFalse(textworld_ppo_boundary_is_terminal("environment_done_without_terminal_signal"))

    def test_targets_are_immutable_and_cannot_be_recomputed(self):
        result = frozen()
        with self.assertRaises(FrozenInstanceError):
            result.value_version = 8
        with self.assertRaisesRegex(ValueError, "already frozen"):
            freeze_rollout_ppo_targets(result, [0] * 7, value_version=8, gamma=1, gae_lambda=1)
        buffer = agent.ReplayBufferActor.__ray_metadata__.modified_class(8, 0)
        buffer.add_samples([result])
        self.assertEqual(buffer.sample(1)[0], buffer.sample(1)[0])
        self.assertEqual(buffer.get_stats()["size"], 1)

    def test_invalid_frozen_fields_rejected(self):
        for fields in (
            {"raw_advantages": (1, 2, 3)},
            {"raw_advantages": (1,), "returns": (2,), "value_version": 0},
            {"raw_advantages": (1, 2, float("nan")), "returns": (1, 2, 3), "value_version": 0},
            {"raw_advantages": (1, 2, 3), "returns": (1, 2, 3), "value_version": -1},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(sample(), **fields)

    def test_packing_preserves_targets_and_rejects_mixed_modes(self):
        examples = [frozen(), frozen("truncated")]
        batch = pack(examples).batch
        self.assertEqual(batch["raw_advantages"].tolist(), list(examples[0].raw_advantages + examples[1].raw_advantages))
        self.assertEqual(batch["returns"].tolist(), list(examples[0].returns + examples[1].returns))
        self.assertNotIn("raw_advantages", pack([sample()]).batch)
        with self.assertRaisesRegex(ValueError, "mix"):
            pack([sample(), frozen()])

    def test_cli_switch_selects_joint_ablation_and_keeps_default(self):
        base = ["test", "--model-path", "/tmp/model", "--tw-game-dir", "/tmp/games", "--train-token-budget", "8192", "--ppo-advantage-normalization", "ema_zscore"]
        with mock.patch("sys.argv", base):
            args = agent.parse_args()
        self.assertEqual((args.ppo_gae_mode, args.ppo_advantage_normalization), ("jit", "ema_zscore"))
        with mock.patch("sys.argv", base + ["--ppo-gae-mode", "rollout"]):
            args = agent.parse_args()
        self.assertEqual(args.ppo_advantage_normalization, "optimizer_window")
        self.assertGreater(args.vllm_max_model_len, args.tw_history_token_window)


class TrainerTests(unittest.TestCase):
    def worker(self, mode="rollout", normalization="optimizer_window"):
        worker = agent.FSDPTrainWorker.__new__(agent.FSDPTrainWorker)
        worker.device = torch.device("cpu")
        worker.rank = 0
        worker.fsdp_world_size = 1
        worker.loss_pg = None
        worker.research = None
        worker.optimizer_step = 0
        worker.train_micro_step = 0
        worker._ppo_forward_buffers_validated = True
        worker.ppo_adv_ema_mean = 0.25
        worker.ppo_adv_ema_variance = 2.0
        worker.args = SimpleNamespace(
            ppo_gae_mode=mode, ppo_advantage_normalization=normalization,
            train_token_budget=32, ppo_adv_norm_eps=1e-8, ppo_advantage_min_scale=1e-3,
            ppo_advantage_ema_beta=0.9, gae_gamma=0.9, gae_lambda=0.8,
            clip_mode="ppo", clip_eps=0.2, value_loss_coef=0.5, old_new_kl_coef=0,
            rl_algorithm="ppo", max_length=32,
        )
        worker.logprob = torch.nn.Parameter(torch.tensor(-0.2))
        worker.value = torch.nn.Parameter(torch.tensor(0.3))
        optimizer = torch.optim.SGD([worker.logprob, worker.value], lr=0.01)

        def forward(batch, **kwargs):
            count = len(batch["target_indices"])
            bootstrap_mask = torch.zeros(len(batch["response_counts"]), dtype=torch.bool)
            bootstrap_mask[batch["bootstrap_sample_indices"]] = True
            return agent.PPOFlatTokenView(
                current_logprobs=worker.logprob.expand(count),
                old_logprobs=batch["response_logprobs"], current_values=worker.value.expand(count),
                rewards=batch["response_rewards"], terminated=batch["response_terminated"],
                truncated=batch["response_truncated"], response_sample_indices=batch["response_sample_indices"],
                response_ordinals=batch["response_ordinals"], response_counts=batch["response_counts"],
                bootstrap_values=worker.value.expand(len(batch["response_counts"])), bootstrap_mask=bootstrap_mask,
            )

        def step():
            optimizer.step()
            optimizer.zero_grad()
            worker.optimizer_step += 1
            return 0.01

        worker._forward_ppo_token_view = mock.Mock(side_effect=forward)
        worker._finalize_optimizer_step = step
        worker._reduce_ep_dispatch_metrics = lambda: {}
        return worker

    def test_global_moments_weight_tokens_across_packs_and_ranks(self):
        worker = self.worker()
        packs = [pack([replace(frozen(), raw_advantages=(1, 2, 3))]),
                 pack([replace(frozen(), raw_advantages=(4, 5, 6)), frozen()])]
        local = torch.cat([p.batch["raw_advantages"].double() for p in packs])
        other_rank = torch.tensor([20.0, -10.0], dtype=torch.float64)
        remote = torch.tensor([other_rank.sum(), other_rank.square().sum(), 2], dtype=torch.float64)
        with mock.patch.object(agent.dist, "all_reduce", side_effect=lambda tensor, **kw: tensor.add_(remote)):
            mean, std = worker._rollout_ppo_optimizer_window_moments(packs, global_valid_token_count=len(local) + 2)
        reference = torch.cat((local, other_rank))
        torch.testing.assert_close(mean, reference.mean())
        torch.testing.assert_close(std, reference.std(unbiased=False))

    def test_rollout_step_uses_fixed_returns_one_forward_and_no_gae_or_ema(self):
        worker = self.worker()
        packs = [pack([frozen()]), pack([frozen("truncated")])]
        worker._prepare_varlen_optimizer_window = lambda version: packs
        worker._compute_ppo_raw_targets_unchecked = mock.Mock(side_effect=AssertionError("JIT called"))
        worker._precompute_ppo_optimizer_window_targets = mock.Mock(side_effect=AssertionError("prepass called"))
        original_loss = worker._compute_ppo_loss_from_targets_unchecked
        losses = []

        def loss(view, **targets):
            losses.append({name: tensor.clone() for name, tensor in targets.items()})
            return original_loss(view, **targets)

        worker._compute_ppo_loss_from_targets_unchecked = loss
        with mock.patch.object(agent.dist, "all_reduce"):
            worker._run_ppo_optimizer_step(8)
            # Replay the exact same samples after updating the current critic.
            worker._run_ppo_optimizer_step(9)
        self.assertEqual(worker._forward_ppo_token_view.call_count, 4)
        self.assertNotEqual(worker.value.item(), 0.3)
        for index, targets in enumerate(losses):
            torch.testing.assert_close(targets["returns"], packs[index % 2].batch["returns"])
        normalized = torch.cat([row["actor_advantages"] for row in losses[:2]])
        self.assertAlmostEqual(normalized.mean().item(), 0, places=6)
        self.assertAlmostEqual(normalized.std(unbiased=False).item(), 1, places=6)
        self.assertEqual((worker.ppo_adv_ema_mean, worker.ppo_adv_ema_variance), (0.25, 2.0))

    def test_existing_jit_window_and_ema_paths_still_compute_gae(self):
        for normalization, forwards in (("optimizer_window", 2), ("ema_zscore", 1)):
            with self.subTest(normalization=normalization):
                worker = self.worker("jit", normalization)
                worker._prepare_varlen_optimizer_window = lambda version: [pack([sample()])]
                with mock.patch.object(agent.dist, "all_reduce"), mock.patch.object(worker, "_compute_ppo_raw_targets_unchecked", wraps=worker._compute_ppo_raw_targets_unchecked) as gae:
                    worker._run_ppo_optimizer_step(3)
                self.assertEqual(worker._forward_ppo_token_view.call_count, forwards)
                self.assertEqual(gae.call_count, 1)

    def test_replay_mode_mismatch_fails(self):
        worker = self.worker()
        with self.assertRaisesRegex(ValueError, "ppo-gae-mode"):
            worker._prepare_rl_sample(sample())
        self.assertIs(worker._prepare_rl_sample(frozen()).__class__, RawPPOSample)


class WorkerTests(unittest.TestCase):
    def test_chunked_prefill_and_mixed_requests_use_correct_positions(self):
        runner = SimpleNamespace(
            device="cpu", model_config=SimpleNamespace(get_hidden_size=lambda: 2),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(pipeline_parallel_size=1, use_ubatching=False),
                speculative_config=None, model_config=SimpleNamespace(enforce_eager=True),
            ),
            _model_forward=lambda: None,
            input_batch=SimpleNamespace(req_ids=["normal", "accerl_value_test:-random"],
                                        num_computed_tokens_cpu=[0, 0], num_prompt_tokens=[1, 3]),
            query_start_loc=SimpleNamespace(np=[0, 1, 3]),
        )
        worker = RolloutCriticWorker(runner)
        worker.load({"weight": [[1, 2]], "bias": [0.5]}, 4)
        worker.capture(torch.tensor([[99., 99.], [1., 2.], [3., 4.]]))
        with self.assertRaisesRegex(RuntimeError, "unconsumed"):
            worker.load({"weight": [[0, 0]], "bias": [0]}, 5)
        runner.input_batch = SimpleNamespace(req_ids=["accerl_value_test:-random"], num_computed_tokens_cpu=[2], num_prompt_tokens=[3])
        runner.query_start_loc.np = [0, 1]
        worker.capture(torch.tensor([[5., 6.]]))
        self.assertEqual(worker.take("accerl_value_test"), {"version": 4, "values": [5.5, 11.5, 17.5]})
        self.assertIsNone(worker.take("accerl_value_test"))
        worker.load({"weight": [[0, 0]], "bias": [2]}, 5)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_drains_values_and_blocks_new_scoring_until_resume(self):
        client = RolloutCriticClient(SimpleNamespace())
        client.version = 1
        client.resume()
        started, finish = asyncio.Event(), asyncio.Event()

        async def score_one(tokens, version):
            started.set()
            await finish.wait()
            return [float(version)] * len(tokens)

        client._score_one = score_one
        first = asyncio.create_task(client.score([[1, 2]]))
        await started.wait()
        pause = asyncio.create_task(client.pause())
        await asyncio.sleep(0)
        second = asyncio.create_task(client.score([[3]]))
        await asyncio.sleep(0)
        self.assertFalse(pause.done())
        self.assertFalse(second.done())
        finish.set()
        self.assertEqual((await first)["value_version"], 1)
        await pause
        self.assertFalse(second.done())
        client.version = 2
        client.resume()
        self.assertEqual(await second, {"value_version": 2, "token_values": [[2.0]]})

    async def test_failed_batch_drains_siblings_before_update(self):
        client = RolloutCriticClient(SimpleNamespace())
        client.version = 1
        client.resume()
        finish = asyncio.Event()

        async def score_one(tokens, version):
            if tokens == [1]:
                raise RuntimeError("score failed")
            await finish.wait()
            return [2]

        client._score_one = score_one
        task = asyncio.create_task(client.score([[1], [2]]))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertFalse(client.idle.is_set())
        finish.set()
        with self.assertRaisesRegex(RuntimeError, "score failed"):
            await task
        self.assertTrue(client.idle.is_set())

    async def test_rpc_collects_all_data_parallel_engines(self):
        async def utility(*args, engine):
            return [engine]
        engine = SimpleNamespace(engine_core=SimpleNamespace(core_engines=[0, 1], _call_utility_async=utility))
        self.assertEqual(await all_worker_rpc(engine, "method"), [0, 1])


if __name__ == "__main__":
    unittest.main()
