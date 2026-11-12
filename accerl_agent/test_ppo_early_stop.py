"""CPU tests for response-token K2 statistics and actor/critic zero graphs."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from accerl_agent import ppo_early_stop as guard


class K2EarlyStopTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(guard.dist, "is_initialized", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def observe_delta(state, delta):
        new = torch.tensor(delta, dtype=torch.float32)
        return state.observe(new, torch.zeros_like(new))

    def test_window_metrics_weight_tokens_and_latch_microbatch_stop(self):
        state = guard.K2EarlyStopState(0.4)
        self.assertAlmostEqual(self.observe_delta(state, [0.1]), 0.005)
        self.assertFalse(state.stop_ppo)
        self.assertEqual(self.observe_delta(state, [1.0]), 0.5)
        self.assertTrue(state.stop_ppo)
        self.assertEqual(self.observe_delta(state, [0.0, 0.0]), 0.0)
        metrics = state.metrics()
        self.assertAlmostEqual(metrics["k2_kl"], 0.505 / 4)
        self.assertEqual(metrics["k2_kl_max_microbatch"], 0.5)
        self.assertEqual(metrics["early_stop_microbatch"], 2)
        self.assertEqual(metrics["early_stop"], 1.0)
        self.assertEqual(metrics["optimizer_step_skipped"], 1.0)
        self.assertEqual(state.token_count, 4)
        self.assertEqual(state.microbatches, 3)

    def test_threshold_equality_does_not_stop(self):
        state = guard.K2EarlyStopState(0.5)
        self.assertEqual(self.observe_delta(state, [1.0]), 0.5)
        self.assertFalse(state.stop_ppo)
        self.observe_delta(state, [1.1])
        self.assertTrue(state.stop_ppo)
        self.assertEqual(state.first_stopped_microbatch, 2)

    def test_empty_response_tokens_do_not_stop_or_change_denominator(self):
        state = guard.K2EarlyStopState(0.05)
        self.assertEqual(self.observe_delta(state, []), 0.0)
        self.assertEqual(state.metrics()["k2_kl"], 0.0)
        self.assertFalse(state.stop_ppo)
        self.assertEqual(state.token_count, 0)
        self.assertAlmostEqual(self.observe_delta(state, [0.2]), 0.02)
        self.assertEqual(self.observe_delta(state, []), 0.0)
        self.assertAlmostEqual(state.metrics()["k2_kl"], 0.02)
        self.assertEqual(state.token_count, 1)

    def test_nonfinite_response_differences_stop(self):
        for value in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                state = guard.K2EarlyStopState(0.1)
                self.assertTrue(math.isinf(self.observe_delta(state, [0.0, value])))
                self.assertTrue(state.stop_ppo)
                self.assertEqual(state.nonfinite_tokens, 1)
                self.assertEqual(state.token_count, 2)
                self.assertTrue(math.isinf(state.metrics()["k2_kl"]))

    def test_float64_statistics_avoid_fp32_subtraction_and_square_overflow(self):
        state = guard.K2EarlyStopState(0.1)
        maximum = torch.finfo(torch.float32).max
        new = torch.tensor([maximum], requires_grad=True)
        old = torch.tensor([-maximum], requires_grad=True)
        observed = state.observe(new, old)
        self.assertTrue(math.isfinite(observed))
        self.assertTrue(math.isclose(observed, 0.5 * (2 * maximum) ** 2))
        self.assertEqual(state.nonfinite_tokens, 0)
        self.assertIsNone(new.grad)
        self.assertIsNone(old.grad)

    def test_nonfinite_squared_sum_also_stops_without_nonfinite_inputs(self):
        state = guard.K2EarlyStopState(0.1)
        result = state.observe(torch.tensor([1e200], dtype=torch.float64), torch.zeros(1))
        self.assertTrue(math.isinf(result))
        self.assertTrue(state.stop_ppo)
        self.assertEqual(state.nonfinite_tokens, 0)

    def test_distributed_microbatch_statistic_weights_tokens_not_ranks(self):
        group = object()
        cases = [([1.0], [0.5, 1.0, 0.0]), ([0.0, 0.0, 0.0], [0.0, 3.0, 0.0])]
        for values, local_stats in cases:
            with self.subTest(values=values):
                def reduce(stats, *, op, group):
                    self.assertIs(group, process_group)
                    self.assertEqual(op, guard.dist.ReduceOp.SUM)
                    self.assertEqual(stats.dtype, torch.float64)
                    torch.testing.assert_close(stats, torch.tensor(local_stats, dtype=torch.float64))
                    stats.copy_(torch.tensor([0.5, 4.0, 0.0], dtype=torch.float64))

                process_group = group
                with mock.patch.object(guard.dist, "is_initialized", return_value=True), \
                        mock.patch.object(guard.dist, "all_reduce", side_effect=reduce) as collective:
                    state = guard.K2EarlyStopState(0.2)
                    new = torch.tensor(values)
                    self.assertEqual(state.observe(new, torch.zeros_like(new), group=group), 0.125)
                collective.assert_called_once()
                self.assertFalse(state.stop_ppo)  # Averaging rank means would incorrectly yield 0.25.
                self.assertEqual(state.token_count, 4)

    def test_remote_rank_can_stop_a_rank_with_no_local_response_tokens(self):
        for nonfinite in [False, True]:
            with self.subTest(nonfinite=nonfinite):
                def reduce(stats, **kwargs):
                    torch.testing.assert_close(stats, torch.zeros(3, dtype=torch.float64))
                    stats.copy_(torch.tensor([0.5, 1.0, int(nonfinite)], dtype=torch.float64))

                with mock.patch.object(guard.dist, "is_initialized", return_value=True), \
                        mock.patch.object(guard.dist, "all_reduce", side_effect=reduce):
                    state = guard.K2EarlyStopState(0.1)
                    result = self.observe_delta(state, [])
                self.assertTrue(math.isinf(result) if nonfinite else result == 0.5)
                self.assertTrue(state.stop_ppo)
                self.assertEqual(state.nonfinite_tokens, int(nonfinite))

    def test_collectives_continue_after_stop_and_fresh_state_resets(self):
        def reduce(stats, **kwargs):
            return None

        with mock.patch.object(guard.dist, "is_initialized", return_value=True), \
                mock.patch.object(guard.dist, "all_reduce", side_effect=reduce) as collective:
            stopped = guard.K2EarlyStopState(0.1)
            self.observe_delta(stopped, [1.0])
            self.observe_delta(stopped, [0.0])
            self.observe_delta(stopped, [])
            self.assertEqual(collective.call_count, 3)
        self.assertTrue(stopped.stop_ppo)
        self.assertEqual(stopped.first_stopped_microbatch, 1)
        fresh = guard.K2EarlyStopState(0.1)
        self.assertEqual(fresh.metrics()["early_stop"], 0.0)
        self.assertEqual(fresh.metrics()["early_stop_microbatch"], 0)
        self.assertEqual(fresh.metrics()["k2_kl"], 0.0)
        self.observe_delta(fresh, [0.0])
        self.assertFalse(fresh.stop_ppo)

    def test_invalid_threshold_and_shape_are_rejected(self):
        for threshold in [-1.0, math.nan, math.inf]:
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    guard.K2EarlyStopState(threshold)
        state = guard.K2EarlyStopState(0.1)
        with self.assertRaisesRegex(ValueError, "same shape"):
            state.observe(torch.zeros(2), torch.zeros(1))

    def test_graph_zero_connects_actor_value_and_bootstrap_branches(self):
        actor = torch.tensor([-0.2, -0.5], requires_grad=True)
        value = torch.tensor([1.0, 2.0], requires_grad=True)
        bootstrap = torch.tensor([3.0], requires_grad=True)
        loss = guard.graph_zero(
            actor + torch.tensor([math.nan, math.inf]),
            2 * value + torch.tensor([math.inf, math.nan]),
            3 * bootstrap,
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for branch in [actor, value, bootstrap]:
            self.assertIsNotNone(branch.grad)
            torch.testing.assert_close(branch.grad, torch.zeros_like(branch))

    def test_graph_zero_accepts_empty_and_noncontiguous_tensors(self):
        values = torch.full((2, 3), math.nan, requires_grad=True)
        empty = torch.empty(0, requires_grad=True)
        loss = guard.graph_zero(values.transpose(0, 1), empty)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.zeros_like(values))
        torch.testing.assert_close(empty.grad, torch.zeros_like(empty))
        with self.assertRaisesRegex(ValueError, "at least one"):
            guard.graph_zero()


if __name__ == "__main__":
    unittest.main()
