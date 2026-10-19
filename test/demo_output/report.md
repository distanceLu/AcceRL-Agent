# Partial rollout scheduling demo

CPU toy-attention experiment. Time costs and length history are synthetic fixtures.
KL is measured on common teacher-forced continuations from untrained models.
This does not measure GPU throughput, actual EOS prediction, RL convergence, or task rewards.

| Policy | A / B / C | Fallbacks | Synthetic compute (s) | Toy KL sum | Proxy objective | Proxy regret |
|---|---|---:|---:|---:|---:|---:|
| adaptive | 4 / 25 / 19 | 0 | 2.6644 | 1.0916 | 9.2835 | 0.1604 |
| always_new_weights_old_kv | 0 / 45 / 3 | 3 | 1.1990 | 3.2493 | 12.0316 | 2.9084 |
| always_new_weights_rebuild_kv | 0 / 0 / 48 | 0 | 5.2869 | 0.0000 | 12.8701 | 3.7470 |
| always_old_weights_old_kv | 32 / 0 / 16 | 16 | 2.6965 | 91.7205 | 96.3459 | 87.2228 |
| oracle_proxy | 4 / 27 / 17 | 0 | 2.5069 | 1.0887 | 9.1231 | 0.0000 |

Oracle sees held-out KL and continuation lengths; it is an offline proxy lower bound, not a deployable policy.
Adaptive sees only calibration cohorts and the length estimator. It need not beat every fixed baseline.
All fixed policies rebuild when their preferred action fails a feasibility constraint.

## Adaptive decisions

| Request | Action | Predicted remaining | Audit horizon | Cold start |
|---|---|---:|---:|---|
| near_end_71 | old_weights_old_kv | 9.67 | 4 | False |
| mild_update_long_prefix_71 | new_weights_old_kv | 21.00 | 20 | False |
| large_update_short_prefix_71 | new_weights_rebuild_kv | 50.00 | 40 | False |
| large_update_long_prefix_71 | new_weights_rebuild_kv | 21.00 | 24 | False |
| old_engine_gone_71 | new_weights_old_kv | 38.00 | 28 | False |
| old_engine_expiring_71 | new_weights_old_kv | 9.67 | 8 | False |
| cache_evicted_71 | new_weights_rebuild_kv | 38.00 | 24 | False |
| length_cold_start_71 | new_weights_old_kv | 60.00 | 16 | True |
| random_0_71 | new_weights_rebuild_kv | 34.00 | 36 | False |
| random_1_71 | new_weights_rebuild_kv | 33.00 | 13 | False |
| random_2_71 | new_weights_rebuild_kv | 18.00 | 20 | False |
| random_3_71 | new_weights_old_kv | 38.00 | 44 | False |
| random_4_71 | new_weights_old_kv | 9.00 | 34 | False |
| random_5_71 | new_weights_old_kv | 28.00 | 40 | False |
| random_6_71 | new_weights_old_kv | 21.00 | 30 | False |
| random_7_71 | new_weights_rebuild_kv | 30.00 | 31 | False |
| near_end_72 | old_weights_old_kv | 9.67 | 4 | False |
| mild_update_long_prefix_72 | new_weights_old_kv | 21.00 | 20 | False |
| large_update_short_prefix_72 | new_weights_rebuild_kv | 50.00 | 40 | False |
| large_update_long_prefix_72 | new_weights_rebuild_kv | 21.00 | 24 | False |
| old_engine_gone_72 | new_weights_old_kv | 38.00 | 28 | False |
| old_engine_expiring_72 | new_weights_old_kv | 9.67 | 8 | False |
| cache_evicted_72 | new_weights_rebuild_kv | 38.00 | 24 | False |
| length_cold_start_72 | new_weights_old_kv | 60.00 | 16 | True |
| random_0_72 | new_weights_old_kv | 21.50 | 24 | False |
| random_1_72 | new_weights_old_kv | 17.00 | 28 | False |
| random_2_72 | new_weights_old_kv | 35.00 | 41 | False |
| random_3_72 | new_weights_old_kv | 22.00 | 17 | False |
| random_4_72 | new_weights_old_kv | 20.00 | 5 | False |
| random_5_72 | new_weights_rebuild_kv | 23.00 | 26 | False |
| random_6_72 | old_weights_old_kv | 12.67 | 16 | False |
| random_7_72 | new_weights_old_kv | 21.50 | 1 | False |
| near_end_73 | old_weights_old_kv | 9.67 | 4 | False |
| mild_update_long_prefix_73 | new_weights_old_kv | 21.00 | 20 | False |
| large_update_short_prefix_73 | new_weights_rebuild_kv | 50.00 | 40 | False |
| large_update_long_prefix_73 | new_weights_rebuild_kv | 21.00 | 24 | False |
| old_engine_gone_73 | new_weights_old_kv | 38.00 | 28 | False |
| old_engine_expiring_73 | new_weights_old_kv | 9.67 | 8 | False |
| cache_evicted_73 | new_weights_rebuild_kv | 38.00 | 24 | False |
| length_cold_start_73 | new_weights_old_kv | 60.00 | 16 | True |
| random_0_73 | new_weights_rebuild_kv | 8.67 | 20 | False |
| random_1_73 | new_weights_old_kv | 22.50 | 22 | False |
| random_2_73 | new_weights_rebuild_kv | 28.00 | 7 | False |
| random_3_73 | new_weights_rebuild_kv | 42.00 | 1 | False |
| random_4_73 | new_weights_old_kv | 27.00 | 5 | False |
| random_5_73 | new_weights_old_kv | 25.00 | 13 | False |
| random_6_73 | new_weights_rebuild_kv | 40.00 | 37 | False |
| random_7_73 | new_weights_rebuild_kv | 22.00 | 31 | False |

## Provenance

```json
{
  "cost_source": "synthetic_fixture_seconds",
  "calibration_seeds": [
    11,
    12,
    13
  ],
  "holdout_seeds": [
    71,
    72,
    73
  ],
  "calibration_samples": 18,
  "settings": {
    "risk_seconds_per_nat": 1.0,
    "uncertainty_weight": 1.0,
    "max_old_gap": 4,
    "max_cache_gap": 4,
    "max_prefix_shift": 1.0,
    "old_retention_seconds_per_token": 0.0
  },
  "holdout_requests": 48,
  "randomized_holdout_requests": 24,
  "measured_cpu_calibration_seconds": 0.13906427659094334,
  "measured_cpu_evaluation_seconds": 0.6152781168930233,
  "limitations": [
    "synthetic_time_costs",
    "untrained_single_head_model",
    "teacher_forced_conditional_KL_only",
    "controlled_holdout_horizons_not_EOS",
    "no_engine_migration_or_online_RL"
  ]
}
```
