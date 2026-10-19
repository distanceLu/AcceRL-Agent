from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest import mock

import torch

from test.partial_rollout import (
    Action, CostProfile, LengthObservation, RemainingEstimate,
    RemainingLengthEstimator, Request, Risk, RiskTable, Settings,
    decide, decide_engine,
)
from test.toy_cache import TinyAttentionLM, audit_continuations, sample_prefix


def profile() -> CostProfile:
    return CostProfile(((1, 0.0), (10, 3.0), (1000, 30.0)), 0.001, 0.001, "unit_test_fixture")


def request(remaining: float = 1000) -> Request:
    return Request("test", 10, RemainingEstimate(remaining, math.ceil(remaining), 10, False),
                   0, 1, 0, Risk(1.0), Risk(0.002))


class LengthTests(unittest.TestCase):
    def test_complete_history_matches_conditional_empirical_mean(self):
        estimator = RemainingLengthEstimator([
            LengthObservation("task", 10, True), LengthObservation("task", 20, True),
            LengthObservation("task", 30, True),
        ])
        result = estimator.estimate("task", generated=10, budget=40)
        self.assertAlmostEqual(result.mean, 15.0)
        self.assertEqual(result.p90, 20)
        self.assertEqual(result.support, 2)
        self.assertFalse(result.cold_start)

    def test_aborted_sample_is_censored_not_an_eos(self):
        censored = RemainingLengthEstimator([
            LengthObservation("task", 10, False), LengthObservation("task", 20, True),
        ])
        wrong_eos = RemainingLengthEstimator([
            LengthObservation("task", 10, True), LengthObservation("task", 20, True),
        ])
        self.assertAlmostEqual(censored.estimate("task", 0, 30).mean, 20.0)
        self.assertAlmostEqual(wrong_eos.estimate("task", 0, 30).mean, 15.0)

    def test_budget_caps_unobserved_tail(self):
        estimator = RemainingLengthEstimator([LengthObservation("task", 100, False)])
        result = estimator.estimate("task", 20, 12)
        self.assertEqual(result.mean, 12)
        self.assertEqual(result.p90, 12)

    def test_unknown_task_and_exhausted_support_are_explicit(self):
        estimator = RemainingLengthEstimator([LengthObservation("task", 10, True)])
        for task, generated in (("unknown", 0), ("task", 20)):
            result = estimator.estimate(task, generated, 30)
            self.assertTrue(result.cold_start)
            self.assertEqual(result.mean, 30)

    def test_events_and_censor_at_same_time_share_risk_set(self):
        estimator = RemainingLengthEstimator([
            LengthObservation("task", 10, True), LengthObservation("task", 10, False),
            LengthObservation("task", 20, True),
        ])
        self.assertAlmostEqual(estimator.estimate("task", 0, 30).mean, 10 + 10 * 2 / 3)


class DecisionTests(unittest.TestCase):
    def test_rebuild_threshold_matches_independent_analytic_boundary(self):
        # B minus C = R * .002 - 3. No decision implementation used to
        # construct this boundary; both sides have equal decode/queue costs.
        for remaining, expected in ((1499, Action.KEEP), (1501, Action.REBUILD)):
            self.assertEqual(decide(request(remaining), profile(), Settings()).action, expected)

    def test_nearly_complete_low_drift_request_can_finish_on_old_weights(self):
        row = replace(request(5), old_risk=Risk(0.001), cache_risk=Risk(0.1))
        self.assertEqual(decide(row, profile(), Settings()).action, Action.OLD)

    def test_queue_cost_can_outweigh_fresher_weights(self):
        row = replace(request(5), old_risk=Risk(0.1), cache_risk=Risk(0.001), new_queue_seconds=2)
        self.assertEqual(decide(row, profile(), Settings()).action, Action.OLD)

    def test_uncertain_cache_estimate_can_trigger_rebuild(self):
        row = replace(request(), cache_risk=Risk(0.001, 0.01))
        self.assertEqual(decide(row, profile(), Settings(uncertainty_weight=0)).action, Action.KEEP)
        self.assertEqual(decide(row, profile(), Settings(uncertainty_weight=1)).action, Action.REBUILD)

    def test_missing_old_weights_cannot_be_selected(self):
        row = replace(request(5), old_risk=Risk(0), old_weights_available=False)
        result = decide(row, profile(), Settings())
        self.assertNotEqual(result.action, Action.OLD)
        self.assertEqual(result.candidates[0].reason, "old_weights_unavailable")

    def test_evicted_cache_requires_rebuild(self):
        self.assertEqual(decide(replace(request(), cache_available=False), profile(), Settings()).action, Action.REBUILD)

    def test_old_route_rejects_cache_from_different_version(self):
        row = replace(request(), current_version=2, cache_version=1)
        result = decide(row, profile(), Settings())
        self.assertFalse(result.candidates[0].feasible)
        self.assertEqual(result.candidates[0].reason, "matching_old_cache_unavailable")

    def test_mixed_cache_is_not_misrepresented_as_old_snapshot_cache(self):
        result = decide(replace(request(), cache_is_mixed=True), profile(), Settings())
        self.assertFalse(result.candidates[0].feasible)
        self.assertEqual(result.candidates[0].reason, "matching_old_cache_unavailable")

    def test_gap_budgets_disable_old_and_stale_cache_routes(self):
        row = replace(request(), current_version=10)
        self.assertEqual(decide(row, profile(), Settings()).action, Action.REBUILD)

    def test_engine_deadline_uses_tail_length_not_only_mean(self):
        row = replace(request(5), remaining=RemainingEstimate(5, 100, 10, False),
                      old_risk=Risk(0), old_deadline_seconds=0.01)
        result = decide(row, profile(), Settings())
        self.assertEqual(result.candidates[0].reason, "old_engine_deadline")

    def test_bad_prefix_cannot_be_fixed_by_rebuilding_kv(self):
        result = decide(replace(request(), prefix_shift=2), profile(), Settings(max_prefix_shift=1))
        self.assertIsNone(result.action)
        self.assertTrue(all(c.reason == "prefix_shift_exceeds_budget" for c in result.candidates))

    def test_unknown_risk_falls_back_to_rebuild(self):
        row = replace(request(), old_risk=None, cache_risk=None)
        self.assertEqual(decide(row, profile(), Settings()).action, Action.REBUILD)

    def test_retaining_old_engine_has_opportunity_cost(self):
        row = replace(request(5), old_risk=Risk(0), cache_risk=Risk(0.001))
        self.assertEqual(decide(row, profile(), Settings()).action, Action.OLD)
        settings = Settings(old_retention_seconds_per_token=1)
        self.assertEqual(decide(row, profile(), settings).action, Action.KEEP)

    def test_one_engine_selects_one_action_even_with_different_preferences(self):
        short = replace(request(5), request_id="short", old_risk=Risk(0), cache_risk=Risk(0.1))
        long = replace(request(2000), request_id="long")
        self.assertEqual(decide(short, profile(), Settings()).action, Action.OLD)
        self.assertEqual(decide(long, profile(), Settings()).action, Action.REBUILD)
        # Rebuilding BOTH prefixes costs 6 seconds. The common KEEP action
        # costs 4.5 in risk, so the engine-level answer differs from the long
        # request's isolated answer. One shared weight assignment is enforced.
        self.assertEqual(decide_engine([short, long], profile(), Settings()), Action.KEEP)

    def test_engine_cannot_silently_mix_version_assignments(self):
        with self.assertRaisesRegex(ValueError, "compatible versions"):
            decide_engine([request(), replace(request(), current_version=3)], profile(), Settings())

    def test_cost_interpolation_and_no_silent_extrapolation(self):
        self.assertAlmostEqual(profile().prefill(5), 4 / 3)
        with self.assertRaisesRegex(ValueError, "outside profiled range"):
            profile().prefill(2000)

    def test_nonfinite_inputs_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                Risk(value)
            with self.assertRaises(ValueError):
                Settings(risk_seconds_per_nat=value)


class RiskTableTests(unittest.TestCase):
    def test_predicts_from_calibration_and_abstains_for_unseen_cohort(self):
        table = RiskTable()
        table.add(1, 12, 0.2, 0.01)
        self.assertIsNone(table.predict(1, 13))
        table.add(1, 14, 0.4, 0.03)
        old, cache = table.predict(1, 13)
        self.assertAlmostEqual(old.mean_kl, 0.3)
        self.assertAlmostEqual(old.standard_error, 0.1)
        self.assertAlmostEqual(cache.mean_kl, 0.02)
        self.assertIsNone(table.predict(3, 13))
        self.assertIsNone(table.predict(1, 48))


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_same_version_incremental_cache_matches_full_prefill(self):
        model = TinyAttentionLM(5)
        tokens = sample_prefix(model, 20, 7)
        cache = model.empty_cache()
        for index, token in enumerate(tokens):
            actual, cache = model.step(token, cache)
            reference, _ = model.prefill(tokens[:index + 1])
            torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-11)

    def test_identical_weights_have_no_cache_or_policy_drift(self):
        old = TinyAttentionLM(5)
        same_weights = old.changed(1, magnitude=0, seed=8)
        audit = audit_continuations(old, same_weights, sample_prefix(old, 20, 7), 8, 9)
        for action in Action:
            self.assertLess(audit.mean_risk(action), 1e-12)

    def test_real_weight_update_changes_old_and_mixed_cache_distributions(self):
        old = TinyAttentionLM(5)
        new = old.changed(2, magnitude=0.2, seed=8)
        audit = audit_continuations(old, new, sample_prefix(old, 20, 7), 8, 9)
        self.assertGreater(audit.mean_risk(Action.OLD), 1e-6)
        self.assertGreater(audit.mean_risk(Action.KEEP), 1e-6)
        self.assertEqual(audit.mean_risk(Action.REBUILD), 0)
        self.assertEqual(set(audit.final_cache_versions[Action.OLD]), {0})
        self.assertEqual(set(audit.final_cache_versions[Action.KEEP]), {0, 2})
        self.assertEqual(set(audit.final_cache_versions[Action.REBUILD]), {2})
        self.assertNotEqual(audit.behavior_logprobs[Action.OLD], audit.behavior_logprobs[Action.REBUILD])

    def test_audits_reproduce_with_fixed_seeds(self):
        old = TinyAttentionLM(5)
        new = old.changed(2, magnitude=0.2, seed=8)
        prefix = sample_prefix(old, 20, 7)
        self.assertEqual(audit_continuations(old, new, prefix, 8, 9), audit_continuations(old, new, prefix, 8, 9))

    def test_holdout_risk_cannot_leak_into_decisions(self):
        from pathlib import Path
        from test.run_partial_rollout_demo import calibrate, evaluate, load_fixture

        costs, lengths = load_fixture(Path(__file__).with_name("demo_traces.json"))
        table, _ = calibrate()
        rows, decisions = evaluate(costs, lengths, table, Settings())

        def altered_audit(*args, **kwargs):
            audit = audit_continuations(*args, **kwargs)
            changed_risks = {
                action: tuple(1000.0 for _ in values) if action != Action.REBUILD else values
                for action, values in audit.risks_by_token.items()
            }
            return replace(audit, risks_by_token=changed_risks)

        with mock.patch("test.run_partial_rollout_demo.audit_continuations", side_effect=altered_audit):
            altered_rows, altered_decisions = evaluate(costs, lengths, table, Settings())
        self.assertEqual(decisions, altered_decisions)
        self.assertNotEqual([row["objective"] for row in rows], [row["objective"] for row in altered_rows])


if __name__ == "__main__":
    unittest.main()
