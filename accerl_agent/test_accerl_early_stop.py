"""CPU integration coverage of AcceRL's real PPO loss and window coordinator."""

import contextlib
import copy
from dataclasses import replace
import io
import math
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from accerl_agent import agent_textworld as agent
from accerl_agent import test_rollout_gae as fixtures
from accerl_agent import test_fixed_episode_batch as fixed_batch_fixtures


def _assert_same_state(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            _assert_same_state(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            _assert_same_state(test, left, right)
    else:
        test.assertEqual(actual, expected)


class AcceRLEarlyStopTests(unittest.TestCase):
    def worker(self, mode="jit", normalization="ema_zscore", clip_mode="ppo", threshold=0.05):
        worker = fixtures.TrainerTests().worker(mode, normalization)
        worker.args.early_stop_kl = threshold
        worker.args.clip_mode = clip_mode
        worker.args.gipo_sigma = 0.3
        worker.args.max_steps = 1
        worker.args.stop_after_steps = 0
        worker.args.sync_every_optimizer_steps = 1
        worker.args.log_every = 1
        worker.expert_parallel_setup = None
        worker.train_window_step = 0
        worker.optimizer = torch.optim.Adam([worker.logprob, worker.value], lr=0.01)
        worker.scheduler = SimpleNamespace(step=mock.Mock())
        # Seed real Adam moment buffers: even a zero-gradient Adam step would
        # mutate this state and move parameters, so skipping step is essential.
        worker.logprob.grad = torch.tensor(0.25)
        worker.value.grad = torch.tensor(0.5)
        worker.optimizer.step()
        worker.optimizer.zero_grad(set_to_none=True)

        def commit():
            worker.optimizer.step()
            worker.optimizer.zero_grad(set_to_none=True)
            worker.optimizer_step += 1
            worker.scheduler.step()
            return worker.optimizer.param_groups[0]["lr"]

        worker._finalize_optimizer_step = mock.Mock(side_effect=commit)
        return worker

    def packs(self, mode="jit", bad=True):
        make_sample = fixtures.frozen if mode == "rollout" else fixtures.sample
        source = make_sample()
        middle = replace(source, response_logprobs=(-1000.0,) * 3) if bad else source
        return [fixtures.pack([source]), fixtures.pack([middle]), fixtures.pack([source])]

    def run_window(self, worker):
        with mock.patch.object(agent.dist, "all_reduce"), mock.patch.object(agent.dist, "is_initialized", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            return worker._run_ppo_optimizer_step(8)

    def test_middle_pack_rejects_entire_joint_update_and_keeps_adam_ema_and_lr(self):
        for clip_mode in ("ppo", "gipo"):
            with self.subTest(clip_mode=clip_mode):
                worker = self.worker(clip_mode=clip_mode)
                worker._prepare_varlen_optimizer_window = lambda version: self.packs()
                before_parameters = [worker.logprob.detach().clone(), worker.value.detach().clone()]
                before_adam = copy.deepcopy(worker.optimizer.state_dict())
                before_ema = worker.ppo_adv_ema_mean, worker.ppo_adv_ema_variance
                zero_gradients = []
                original_zero = worker._zero_ppo_loss

                def zero(view):
                    # The first rejected pack arrives after a real earlier
                    # pack already accumulated a nonzero Critic gradient.
                    zero_gradients.append(worker.value.grad.detach().clone())
                    return original_zero(view)

                worker._zero_ppo_loss = mock.Mock(side_effect=zero)
                result = self.run_window(worker)
                self.assertEqual(result["early_stop"], 1)
                self.assertEqual(result["optimizer_step_skipped"], 1)
                self.assertEqual(result["early_stop_microbatch"], 2)
                self.assertEqual(result["loss_mean"], 0)
                self.assertEqual(result["value_loss_mean"], 0)
                self.assertEqual(worker.optimizer_step, 0)
                self.assertEqual(worker.train_micro_step, 3)
                worker._finalize_optimizer_step.assert_not_called()
                worker.scheduler.step.assert_not_called()
                self.assertEqual(worker._zero_ppo_loss.call_count, 2)
                self.assertTrue(zero_gradients[0].abs().item() > 0)
                torch.testing.assert_close(zero_gradients[0], zero_gradients[1])
                self.assertIsNone(worker.logprob.grad)
                self.assertIsNone(worker.value.grad)
                self.assertIsNone(worker._ppo_early_stop_state)
                torch.testing.assert_close(worker.logprob, before_parameters[0], rtol=0, atol=0)
                torch.testing.assert_close(worker.value, before_parameters[1], rtol=0, atol=0)
                _assert_same_state(self, worker.optimizer.state_dict(), before_adam)
                self.assertEqual((worker.ppo_adv_ema_mean, worker.ppo_adv_ema_variance), before_ema)

                # The latch belongs to one window, not all future training.
                worker._prepare_varlen_optimizer_window = lambda version: self.packs(bad=False)
                result = self.run_window(worker)
                self.assertEqual(result["early_stop"], 0)
                self.assertEqual(result["optimizer_step_skipped"], 0)
                self.assertEqual(result["early_stop_microbatch"], 0)
                self.assertEqual(worker.optimizer_step, 1)
                worker._finalize_optimizer_step.assert_called_once_with()
                worker.scheduler.step.assert_called_once_with()
                self.assertNotEqual(worker.value.item(), before_parameters[1].item())

    def test_one_guard_observation_per_training_pack_and_none_in_target_prepass(self):
        cases = (("jit", "ema_zscore", False), ("jit", "optimizer_window", False),
                 ("jit", "ema_zscore", True), ("rollout", "optimizer_window", False))
        for mode, normalization, initialize_ema in cases:
            with self.subTest(mode=mode, normalization=normalization, initialize_ema=initialize_ema):
                worker = self.worker(mode, normalization)
                if initialize_ema:
                    worker.ppo_adv_ema_mean = worker.ppo_adv_ema_variance = None
                initial_ema = worker.ppo_adv_ema_mean, worker.ppo_adv_ema_variance
                worker._prepare_varlen_optimizer_window = lambda version: self.packs(mode)
                original_observe = agent.K2EarlyStopState.observe
                states = []

                def observe(state, *args, **kwargs):
                    self.assertIs(worker._ppo_early_stop_state, state)
                    states.append(state)
                    return original_observe(state, *args, **kwargs)

                with mock.patch.object(agent.K2EarlyStopState, "observe", new=observe):
                    result = self.run_window(worker)
                self.assertEqual(len(states), 3)
                self.assertEqual(len({id(state) for state in states}), 1)
                self.assertEqual(states[0].microbatches, 3)
                self.assertEqual(states[0].token_count, 9)
                self.assertEqual(result["early_stop_microbatch"], 2)
                self.assertEqual((worker.ppo_adv_ema_mean, worker.ppo_adv_ema_variance), initial_ema)
                expected_forwards = 3 if mode == "rollout" or (normalization == "ema_zscore" and not initialize_ema) else 6
                self.assertEqual(worker._forward_ppo_token_view.call_count, expected_forwards)

    def test_guard_gates_before_exp_and_before_jit_target_computation(self):
        worker = self.worker()
        worker._prepare_varlen_optimizer_window = lambda version: self.packs()
        original_targets = worker._compute_ppo_raw_targets_unchecked
        worker._compute_ppo_raw_targets_unchecked = mock.Mock(wraps=original_targets)
        original_exp = torch.exp

        def finite_exp(value, *args, **kwargs):
            self.assertLess(value.max().item(), 100, "overflowing ratio reached torch.exp")
            return original_exp(value, *args, **kwargs)

        with mock.patch.object(torch, "exp", side_effect=finite_exp):
            result = self.run_window(worker)
        self.assertEqual(result["early_stop"], 1)
        self.assertEqual(worker._compute_ppo_raw_targets_unchecked.call_count, 1)

    def test_disabled_guard_and_safe_guard_have_identical_updates(self):
        enabled = self.worker()
        disabled = self.worker(threshold=0)
        for worker in (enabled, disabled):
            worker._prepare_varlen_optimizer_window = lambda version: self.packs(bad=False)
        enabled_result = self.run_window(enabled)
        disabled_result = self.run_window(disabled)
        torch.testing.assert_close(enabled.logprob, disabled.logprob, rtol=0, atol=0)
        torch.testing.assert_close(enabled.value, disabled.value, rtol=0, atol=0)
        _assert_same_state(self, enabled.optimizer.state_dict(), disabled.optimizer.state_dict())
        self.assertEqual(enabled_result["loss_mean"], disabled_result["loss_mean"])
        self.assertEqual(enabled_result["early_stop"], 0)
        self.assertEqual(enabled_result["optimizer_step_skipped"], 0)
        self.assertEqual(disabled_result["optimizer_step_skipped"], 0)

    def test_zero_loss_keeps_actor_value_and_bootstrap_heads_connected(self):
        worker = self.worker()
        template = worker._forward_ppo_token_view(fixtures.pack([fixtures.sample()]).batch)
        logprobs = torch.tensor([float("nan")] * 3, requires_grad=True)
        values = torch.tensor([float("inf")] * 3, requires_grad=True)
        bootstrap = torch.tensor([float("-inf")], requires_grad=True)
        view = replace(template, current_logprobs=logprobs, current_values=values, bootstrap_values=bootstrap)
        loss, stats = worker._zero_ppo_loss(view)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(torch.count_nonzero(stats).item(), 0)
        loss.backward()
        for tensor in (logprobs, values, bootstrap):
            torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))

    def test_failed_window_clears_guard_state(self):
        worker = self.worker()
        worker._prepare_varlen_optimizer_window = mock.Mock(side_effect=RuntimeError("sample failure"))
        with self.assertRaisesRegex(RuntimeError, "sample failure"):
            self.run_window(worker)
        self.assertIsNone(worker._ppo_early_stop_state)
        worker._finalize_optimizer_step.assert_not_called()

    def test_all_skipped_sync_segment_is_bounded_and_can_return_zero_commits(self):
        worker = self.worker()
        worker._prepare_varlen_optimizer_window = lambda version: self.packs()
        original_run = worker._run_ppo_optimizer_step
        calls = []

        def run(version):
            calls.append(version)
            self.assertEqual(len(calls), 1, "a rejected window was retried without an attempt bound")
            return original_run(version)

        worker._run_ppo_optimizer_step = run
        with mock.patch.object(agent.dist, "all_reduce"), mock.patch.object(agent.dist, "barrier"), mock.patch.object(agent.dist, "is_initialized", return_value=False), mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            result = worker.train_until_next_sync(1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["optimizer_steps_run"], 0)
        self.assertEqual(worker.optimizer_step, 0)
        self.assertEqual(worker.train_window_step, 1)
        worker._finalize_optimizer_step.assert_not_called()

    def test_grpo_segment_does_not_construct_ppo_guard(self):
        worker = self.worker()
        worker.args.rl_algorithm = "grpo"

        def run(version):
            worker.optimizer_step += 1
            return dict(
                global_valid_token_count=3, global_valid_trajectory_count=1,
                global_version_lag_sum=0, global_sample_count=1,
                global_pack_count=1, global_pack_token_count=7,
                policy_trajectory_metric_sum=0.1, kl_trajectory_metric_sum=0,
                kl_token_metric_sum=0, clip_metric_sum=0,
                trajectory_metric_weight=1, token_metric_weight=3,
                loss_mean=0.1, kl_trajectory_mean=0, kl_token_mean=0,
                clip_fraction=0, current_lr=0.01,
            )

        worker._run_grpo_optimizer_step = run
        with mock.patch.object(agent.K2EarlyStopState, "__init__", side_effect=AssertionError("GRPO constructed a PPO guard")), mock.patch.object(agent.dist, "barrier"), mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            result = worker.train_until_next_sync(1)
        self.assertEqual(result["optimizer_steps_run"], 1)

    def test_mixed_sync_segment_counts_commits_and_does_not_dilute_losses(self):
        worker = self.worker()
        worker.args.max_steps = 10
        worker.args.sync_every_optimizer_steps = 3
        versions = []

        def step_stats(index):
            skipped = index == 2
            token_count = (3, 9, 100)[index]
            trajectory_count = (1, 3, 10)[index]
            stats = {name: 0.0 for name in agent.PPOReductionStats.names()}
            if skipped:
                # The coordinator must ignore skipped-window contributions,
                # even if a diagnostic payload happens to contain numbers.
                stats = {name: 9999.0 for name in stats}
            else:
                stats.update(
                    policy_trajectory_sum=(2, 10)[index],
                    value_loss_trajectory_sum=(3, 9)[index],
                    kl_trajectory_sum=(0.6, 3.4)[index],
                    value_loss_token_sum=(9, 27)[index],
                    kl_token_sum=(1.5, 4.5)[index],
                    valid_trajectory_count=trajectory_count,
                    clip_count=(1, 2)[index],
                    value_sum=token_count,
                    return_sum=2 * token_count,
                    return_sq_sum=5 * token_count,
                    raw_advantage_sum=token_count,
                    raw_advantage_sq_sum=5 * token_count / 3,
                    raw_advantage_count=token_count,
                    terminated_count=trajectory_count,
                )
            return dict(
                global_ppo_stats=stats, optimizer_step_skipped=float(skipped),
                global_valid_token_count=token_count,
                global_valid_trajectory_count=trajectory_count,
                global_version_lag_sum=0, global_sample_count=trajectory_count,
                global_pack_count=trajectory_count, global_pack_token_count=2 * token_count,
                loss_mean=0 if skipped else 1,
                value_loss_mean=0 if skipped else 3,
                kl_trajectory_mean=0, kl_token_mean=0, clip_fraction=0,
                current_lr=0.01,
            )

        def run(version):
            index = len(versions)
            self.assertLess(index, 3, "sync segment exceeded its attempt bound")
            versions.append(version)
            result = step_stats(index)
            if not result["optimizer_step_skipped"]:
                worker.optimizer_step += 1
            return result

        worker._run_ppo_optimizer_step = run
        with mock.patch.object(agent.dist, "barrier"), mock.patch.dict(os.environ, {"TW_EVENT_DIR": "/tmp/unused-k2-test-events"}, clear=True), mock.patch("framework_benchmark.events.emit") as emit, contextlib.redirect_stdout(io.StringIO()):
            result = worker.train_until_next_sync(3, trainer_version=7)
        self.assertEqual(len(versions), 3)
        for actual, expected in zip(versions, (7, 7 + 1 / 3, 7 + 2 / 3)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["train_windows_run"], 3)
        self.assertEqual(result["train_window_step"], 3)
        self.assertEqual(result["optimizer_steps_run"], 2)
        self.assertEqual(result["skipped_windows"], 1)
        self.assertEqual(result["segment_valid_tokens"], 112)
        self.assertAlmostEqual(result["segment_early_stop_fraction"], 1 / 3)
        # Only accepted data contributes to loss/value denominators:
        # 4 accepted trajectories and 12 accepted response tokens.
        self.assertAlmostEqual(result["segment_policy_loss_mean"], 12 / 4)
        self.assertAlmostEqual(result["segment_value_loss_mean"], 12 / 4)
        self.assertAlmostEqual(result["segment_loss_mean"], 4.5)
        self.assertAlmostEqual(result["segment_kl_trajectory_mean"], 4 / 4)
        self.assertAlmostEqual(result["segment_kl_token_mean"], 6 / 12)
        self.assertAlmostEqual(result["segment_clip_frac"], 3 / 12)
        self.assertAlmostEqual(result["segment_value_prediction_mean"], 1)
        self.assertAlmostEqual(result["segment_return_mean"], 2)
        self.assertAlmostEqual(result["segment_raw_advantage_std"], math.sqrt(2 / 3))
        events = [call.args[0] for call in emit.call_args_list]
        self.assertEqual([event["kind"] for event in events], ["optimizer_commit", "optimizer_commit", "optimizer_discard"])
        self.assertEqual([event["optimizer_step"] for event in events], [1, 2, 2])
        self.assertEqual([event["train_window_step"] for event in events], [1, 2, 3])


class AcceRLEarlyStopCLITests(unittest.TestCase):
    def setUp(self):
        self.production = fixed_batch_fixtures._production(fixed_batch_fixtures._Collectives())
        self.base = [
            "test", "--model-path", "/tmp/model", "--tw-game-dir", "/tmp/games",
            "--train-token-budget", "8192", "--fsdp-world-size", "3",
        ]

    def parse(self, extra):
        with mock.patch.object(sys, "argv", self.base + extra):
            return self.production["parse_args"]()

    def test_early_stop_is_disabled_by_default(self):
        args = self.parse([])
        self.assertEqual(args.early_stop_kl, 0)
        self.production["validate_args"](args)

    def test_positive_threshold_is_accepted_for_ppo(self):
        args = self.parse(["--rl-algorithm", "ppo", "--early-stop-kl", "0.01"])
        self.assertEqual(args.early_stop_kl, 0.01)
        self.production["validate_args"](args)

    def test_negative_and_nonfinite_thresholds_are_rejected(self):
        for threshold in ("-1", "nan", "inf", "-inf"):
            with self.subTest(threshold=threshold):
                args = self.parse(["--rl-algorithm", "ppo", f"--early-stop-kl={threshold}"])
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    self.production["validate_args"](args)

    def test_positive_threshold_is_rejected_for_grpo(self):
        args = self.parse(["--rl-algorithm", "grpo", "--early-stop-kl", "0.01"])
        with self.assertRaisesRegex(ValueError, "ppo only"):
            self.production["validate_args"](args)


if __name__ == "__main__":
    unittest.main()
