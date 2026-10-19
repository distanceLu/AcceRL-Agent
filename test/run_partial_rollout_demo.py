"""Reproducible holdout evaluation with synthetic time and measured toy KL.

Run from the repository root with the absolute interpreter documented in README.
No model downloads, GPUs, network calls, or training-framework imports.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
from time import perf_counter

import torch

from test.partial_rollout import (
    Action, CostProfile, LengthObservation, RemainingLengthEstimator,
    Request, RiskTable, Settings, decide,
)
from test.toy_cache import TinyAttentionLM, audit_continuations, sample_prefix


def load_fixture(path: Path) -> tuple[CostProfile, RemainingLengthEstimator]:
    data = json.loads(path.read_text())
    raw = data["cost_profile"]
    profile = CostProfile(
        tuple((int(length), float(seconds)) for length, seconds in raw["prefill_points"]),
        raw["decode_old_seconds_per_token"], raw["decode_new_seconds_per_token"], raw["source"],
    )
    lengths = RemainingLengthEstimator([LengthObservation(**row) for row in data["length_observations"]])
    return profile, lengths


def calibrate() -> tuple[RiskTable, list[dict]]:
    table = RiskTable()
    rows = []
    for seed in (11, 12, 13):
        old = TinyAttentionLM(seed=seed)
        for gap in (1, 3):
            new = old.changed(gap, magnitude=0.04 * gap, seed=100 + seed)
            for prefix_length in (12, 24, 48):
                prefix = sample_prefix(old, prefix_length, seed=200 + seed)
                audit = audit_continuations(old, new, prefix, steps=12, seed=300 + seed)
                old_kl, cache_kl = audit.mean_risk(Action.OLD), audit.mean_risk(Action.KEEP)
                table.add(gap, prefix_length, old_kl, cache_kl)
                rows.append({
                    "seed": seed, "gap": gap, "prefix_tokens": prefix_length,
                    "old_mean_kl": old_kl, "cache_mean_kl": cache_kl,
                })
    return table, rows


def evaluate(profile: CostProfile, lengths: RemainingLengthEstimator, table: RiskTable, settings: Settings):
    rows, decisions = [], []
    # Audit horizons are controlled HELD-OUT continuation lengths, not observed
    # EOS times. They never enter the scheduling decision. This evaluates the
    # policy/cost proxy only, not a trained stopping model or task success.
    named_scenarios = (
        # name, prefix, gap, task, continuation length, new queue, old available, deadline, cache available
        ("near_end", 12, 1, "short", 4, 0.50, True, None, True),
        ("mild_update_long_prefix", 48, 1, "long", 20, 0.0, True, None, True),
        ("large_update_short_prefix", 12, 3, "long", 40, 0.0, True, None, True),
        ("large_update_long_prefix", 48, 3, "long", 24, 0.0, True, None, True),
        ("old_engine_gone", 24, 1, "long", 28, 0.0, False, None, True),
        ("old_engine_expiring", 12, 1, "short", 8, 0.20, True, 0.001, True),
        ("cache_evicted", 24, 3, "long", 24, 0.0, True, None, False),
        ("length_cold_start", 24, 1, "unseen_task", 16, 0.0, True, None, True),
    )
    for seed in (71, 72, 73):
        old = TinyAttentionLM(seed=seed)
        # Add untuned randomized holdout cases, not only easy hand-written
        # regimes. Contexts/queues/horizons differ from calibration and can
        # expose both risk-table error and length-estimation error.
        scenario_rng = random.Random(5000 + seed)
        scenarios = list(named_scenarios)
        for index in range(8):
            prefix_length = scenario_rng.randint(9, 55)
            horizon = scenario_rng.randint(1, min(50, 84 - prefix_length))
            scenarios.append((
                f"random_{index}", prefix_length, scenario_rng.choice((1, 3)),
                "short" if prefix_length <= 16 else "long", horizon,
                scenario_rng.uniform(0.0, 0.6), scenario_rng.random() > 0.2,
                None, True,
            ))
        for name, prefix_length, gap, task, horizon, queue, available, deadline, cache_available in scenarios:
            new = old.changed(gap, magnitude=0.04 * gap, seed=100 + seed)
            prefix = sample_prefix(old, prefix_length, seed=200 + seed)
            remaining = lengths.estimate(task, generated=prefix_length - 4, budget=80 - (prefix_length - 4))
            predicted = table.predict(gap, prefix_length)
            old_risk, cache_risk = predicted if predicted is not None else (None, None)
            request = Request(
                f"{name}_{seed}", prefix_length, remaining, 0, gap, 0,
                old_risk, cache_risk, old_weights_available=available,
                cache_available=cache_available, new_queue_seconds=queue,
                old_deadline_seconds=deadline,
            )
            # Decide BEFORE computing the held-out audit: no exact test KL/horizon
            # is visible to the scheduler. Holdout model seeds are disjoint.
            decision = decide(request, profile, settings)
            decisions.append({"request": asdict(request), "decision": asdict(decision)})
            audit = audit_continuations(old, new, prefix, steps=horizon, seed=300 + seed)
            eligible = {candidate.action: candidate for candidate in decision.candidates if candidate.feasible}
            action_metrics = {}
            for action in eligible:
                is_old = action == Action.OLD
                compute = horizon * (profile.decode_old_seconds_per_token if is_old else profile.decode_new_seconds_per_token)
                if action == Action.REBUILD:
                    compute += profile.prefill(prefix_length)
                queue_cost = request.old_queue_seconds if is_old else request.new_queue_seconds
                risk_nats = sum(audit.risks_by_token[action])
                retention = horizon * settings.old_retention_seconds_per_token if is_old else 0.0
                action_metrics[action] = {
                    "compute_seconds": compute,
                    "queue_seconds": queue_cost,
                    "conditional_kl_sum": risk_nats,
                    "objective": compute + queue_cost + retention + settings.risk_seconds_per_nat * risk_nats,
                }
            if not action_metrics:
                continue
            oracle = min(action_metrics, key=lambda action: action_metrics[action]["objective"])
            policies = {"adaptive": decision.action, "oracle_proxy": oracle}
            for action in Action:
                # Fixed-action baselines use the SAME constraints and rebuild if
                # their preferred action is unavailable. Report fallbacks openly.
                policies[f"always_{action.value}"] = action if action in eligible else Action.REBUILD
            for policy, action in policies.items():
                preferred = policy.removeprefix("always_")
                metrics = action_metrics[action]
                rows.append({
                    "request_id": request.request_id, "policy": policy,
                    "action": action.value, "fallback": policy.startswith("always_") and action.value != preferred,
                    "predicted_remaining": remaining.mean, "audit_horizon": horizon,
                    "cold_start": remaining.cold_start, **metrics,
                    "oracle_objective": action_metrics[oracle]["objective"],
                    "regret": max(0.0, metrics["objective"] - action_metrics[oracle]["objective"]),
                })
    return rows, decisions


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for policy in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        counts = {action.value: sum(row["action"] == action.value for row in selected) for action in Action}
        summaries[policy] = {
            "requests": len(selected), "action_counts": counts,
            "fallbacks": sum(row["fallback"] for row in selected),
            "synthetic_compute_seconds": sum(row["compute_seconds"] for row in selected),
            "synthetic_queue_seconds": sum(row["queue_seconds"] for row in selected),
            "measured_toy_conditional_kl_sum": sum(row["conditional_kl_sum"] for row in selected),
            "proxy_objective": sum(row["objective"] for row in selected),
            "proxy_regret": sum(row["regret"] for row in selected),
        }
    return summaries


def write_report(output: Path, metadata: dict, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# Partial rollout scheduling demo", "",
        "CPU toy-attention experiment. Time costs and length history are synthetic fixtures.",
        "KL is measured on common teacher-forced continuations from untrained models.",
        "This does not measure GPU throughput, actual EOS prediction, RL convergence, or task rewards.", "",
        "| Policy | A / B / C | Fallbacks | Synthetic compute (s) | Toy KL sum | Proxy objective | Proxy regret |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy, result in summary.items():
        counts = " / ".join(str(result["action_counts"][action.value]) for action in Action)
        lines.append(
            f"| {policy} | {counts} | {result['fallbacks']} | "
            f"{result['synthetic_compute_seconds']:.4f} | {result['measured_toy_conditional_kl_sum']:.4f} | "
            f"{result['proxy_objective']:.4f} | {result['proxy_regret']:.4f} |"
        )
    lines += ["", "Oracle sees held-out KL and continuation lengths; it is an offline proxy lower bound, not a deployable policy.",
              "Adaptive sees only calibration cohorts and the length estimator. It need not beat every fixed baseline.",
              "All fixed policies rebuild when their preferred action fails a feasibility constraint.", "",
              "## Adaptive decisions", "", "| Request | Action | Predicted remaining | Audit horizon | Cold start |",
              "|---|---|---:|---:|---|" ]
    for row in rows:
        if row["policy"] == "adaptive":
            lines.append(f"| {row['request_id']} | {row['action']} | {row['predicted_remaining']:.2f} | {row['audit_horizon']} | {row['cold_start']} |")
    lines += ["", "## Provenance", "", "```json", json.dumps(metadata, indent=2), "```", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("demo_traces.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("demo_output"))
    parser.add_argument("--risk-weight", type=float, default=1.0, help="Synthetic seconds per nat of conditional KL proxy")
    args = parser.parse_args()
    settings = Settings(risk_seconds_per_nat=args.risk_weight)
    torch.set_num_threads(1)
    profile, lengths = load_fixture(args.fixture)
    started = perf_counter()
    table, calibration = calibrate()
    calibration_seconds = perf_counter() - started
    started = perf_counter()
    rows, decisions = evaluate(profile, lengths, table, settings)
    evaluation_seconds = perf_counter() - started
    summary = summarize(rows)
    metadata = {
        "cost_source": profile.source, "calibration_seeds": [11, 12, 13], "holdout_seeds": [71, 72, 73],
        "calibration_samples": len(calibration), "settings": asdict(settings),
        "holdout_requests": len(decisions), "randomized_holdout_requests": 24,
        "measured_cpu_calibration_seconds": calibration_seconds,
        "measured_cpu_evaluation_seconds": evaluation_seconds,
        "limitations": ["synthetic_time_costs", "untrained_single_head_model", "teacher_forced_conditional_KL_only",
                        "controlled_holdout_horizons_not_EOS", "no_engine_migration_or_online_RL"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def dump(name, value):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    dump("summary.json", {"metadata": metadata, "policies": summary})
    dump("decisions.json", decisions)
    dump("calibration.json", calibration)
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir, metadata, summary, rows)
    for policy, result in summary.items():
        print(f"{policy:30s} proxy={result['proxy_objective']:.4f} regret={result['proxy_regret']:.4f} actions={result['action_counts']}")
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
