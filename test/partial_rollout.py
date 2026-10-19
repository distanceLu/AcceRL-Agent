"""A decision model, not a live vLLM/SGLang scheduler.

Time costs supplied by callers must share the same units. The demo uses
synthetic seconds. Conditional KL along a fixed continuation is a risk proxy,
not a guarantee about policy gradients or final task reward.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean, stdev


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


class Action(str, Enum):
    OLD = "old_weights_old_kv"
    KEEP = "new_weights_old_kv"
    REBUILD = "new_weights_rebuild_kv"


@dataclass(frozen=True)
class LengthObservation:
    task: str
    generated_tokens: int
    completed: bool

    def __post_init__(self) -> None:
        if self.generated_tokens < 0:
            raise ValueError("generated_tokens must be nonnegative")


@dataclass(frozen=True)
class RemainingEstimate:
    mean: float
    p90: int
    support: int
    cold_start: bool


class RemainingLengthEstimator:
    """Kaplan-Meier survival, conditioned on still running at this length.

    Aborted/token-limited observations are right-censored, not EOS events.
    The estimate is restricted to the caller's remaining generation budget.
    Unsupported tails conservatively return that full budget.
    """

    def __init__(self, observations: list[LengthObservation]):
        self.observations = tuple(observations)

    def estimate(self, task: str, generated: int, budget: int) -> RemainingEstimate:
        if generated < 0 or budget < 1:
            raise ValueError("generated must be nonnegative; budget must be positive")
        rows = [row for row in self.observations if row.task == task]
        support = sum(row.generated_tokens > generated for row in rows)
        if not support:
            return RemainingEstimate(float(budget), budget, 0, True)
        event_times = sorted({row.generated_tokens for row in rows if row.completed})
        curve = []
        survival = 1.0
        for time in event_times:
            at_risk = sum(row.generated_tokens >= time for row in rows)
            events = sum(row.generated_tokens == time and row.completed for row in rows)
            survival *= 1.0 - events / at_risk
            curve.append((time, survival))

        def surviving(time: int) -> float:
            result = 1.0
            for event_time, probability in curve:
                if event_time > time:
                    break
                result = probability
            return result

        base = surviving(generated)
        if base <= 0:
            return RemainingEstimate(float(budget), budget, support, True)
        # E[min(R, budget)] = sum_{k=1}^{budget} P(R >= k).
        expected = sum(surviving(generated + k - 1) / base for k in range(1, budget + 1))
        p90 = budget
        for k in range(1, budget + 1):
            if surviving(generated + k) / base <= 0.1 + 1e-12:
                p90 = k
                break
        return RemainingEstimate(expected, p90, support, False)


@dataclass(frozen=True)
class Risk:
    mean_kl: float
    standard_error: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative("mean_kl", self.mean_kl)
        _nonnegative("standard_error", self.standard_error)

    def upper(self, uncertainty_weight: float) -> float:
        return self.mean_kl + uncertainty_weight * self.standard_error


class RiskTable:
    """Small calibration table; unsupported cohorts explicitly abstain.

    Version gap is only a feature. KL observations, rather than version counts
    alone, determine the risk. There is no assumption that drift is monotonic.
    """

    def __init__(self, prefix_boundaries: tuple[int, ...] = (16, 32)):
        if tuple(sorted(set(prefix_boundaries))) != prefix_boundaries:
            raise ValueError("prefix_boundaries must be strictly increasing")
        self.boundaries = prefix_boundaries
        self.rows: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def _key(self, gap: int, prefix: int) -> tuple[int, int]:
        if gap < 0 or prefix < 1:
            raise ValueError("invalid version gap or prefix length")
        return gap, bisect_right(self.boundaries, prefix)

    def add(self, gap: int, prefix: int, old_kl: float, cache_kl: float) -> None:
        _nonnegative("old_kl", old_kl)
        _nonnegative("cache_kl", cache_kl)
        self.rows.setdefault(self._key(gap, prefix), []).append((old_kl, cache_kl))

    def predict(self, gap: int, prefix: int) -> tuple[Risk, Risk] | None:
        observations = self.rows.get(self._key(gap, prefix))
        if not observations or len(observations) < 2:
            return None

        def estimate(values: list[float]) -> Risk:
            error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            return Risk(mean(values), error)

        return estimate([x[0] for x in observations]), estimate([x[1] for x in observations])


@dataclass(frozen=True)
class CostProfile:
    prefill_points: tuple[tuple[int, float], ...]
    decode_old_seconds_per_token: float
    decode_new_seconds_per_token: float
    source: str

    def __post_init__(self) -> None:
        if len(self.prefill_points) < 2:
            raise ValueError("need at least two prefill measurements")
        last_length, last_seconds = 0, -1.0
        for length, seconds in self.prefill_points:
            _nonnegative("prefill seconds", seconds)
            if length <= last_length or seconds < last_seconds:
                raise ValueError("prefill measurements must increase monotonically")
            last_length, last_seconds = length, seconds
        _nonnegative("old decode cost", self.decode_old_seconds_per_token)
        _nonnegative("new decode cost", self.decode_new_seconds_per_token)

    def prefill(self, length: int) -> float:
        if length < self.prefill_points[0][0] or length > self.prefill_points[-1][0]:
            raise ValueError("prefix length is outside profiled range; profile it first")
        for (left, c_left), (right, c_right) in zip(self.prefill_points, self.prefill_points[1:]):
            if length <= right:
                return c_left + (c_right - c_left) * (length - left) / (right - left)
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class Request:
    request_id: str
    prefix_tokens: int
    remaining: RemainingEstimate
    old_version: int
    current_version: int
    cache_version: int
    old_risk: Risk | None
    cache_risk: Risk | None
    old_weights_available: bool = True
    cache_available: bool = True
    old_queue_seconds: float = 0.0
    new_queue_seconds: float = 0.0
    old_deadline_seconds: float | None = None
    prefix_shift: float = 0.0
    # cache_version is the oldest contributing version. A mixed cache must
    # not be presented as a cache computed entirely by the old snapshot.
    cache_is_mixed: bool = False

    def __post_init__(self) -> None:
        if self.prefix_tokens < 1 or self.current_version < self.old_version or self.old_version < 0:
            raise ValueError("invalid length or model versions")
        if self.cache_version < 0 or self.cache_version > self.current_version:
            raise ValueError("invalid cache version")
        if not math.isfinite(self.remaining.mean) or self.remaining.mean <= 0:
            raise ValueError("remaining mean must be finite and positive")
        if self.remaining.p90 < 1:
            raise ValueError("remaining p90 must be positive")
        for name in ("old_queue_seconds", "new_queue_seconds", "prefix_shift"):
            _nonnegative(name, getattr(self, name))
        if self.old_deadline_seconds is not None:
            _nonnegative("old_deadline_seconds", self.old_deadline_seconds)


@dataclass(frozen=True)
class Settings:
    risk_seconds_per_nat: float = 1.0
    uncertainty_weight: float = 1.0
    max_old_gap: int = 4
    max_cache_gap: int = 4
    max_prefix_shift: float = 1.0
    old_retention_seconds_per_token: float = 0.0

    def __post_init__(self) -> None:
        for name in ("risk_seconds_per_nat", "uncertainty_weight", "max_prefix_shift", "old_retention_seconds_per_token"):
            _nonnegative(name, getattr(self, name))
        if self.max_old_gap < 0 or self.max_cache_gap < 0:
            raise ValueError("version budgets must be nonnegative")


@dataclass(frozen=True)
class Candidate:
    action: Action
    feasible: bool
    reason: str
    compute_seconds: float = 0.0
    queue_seconds: float = 0.0
    risk_nats: float = 0.0
    retention_seconds: float = 0.0
    score: float | None = None


@dataclass(frozen=True)
class Decision:
    action: Action | None
    candidates: tuple[Candidate, ...]
    reason: str


def decide(request: Request, profile: CostProfile, settings: Settings) -> Decision:
    """Score a single request, assuming candidate engine routes exist.

    REBUILD has zero *additional cache/old-weight* mismatch to the current
    model on this fixed prefix. The immutable prefix shift is gated separately;
    zero here must never be interpreted as an entirely on-policy trajectory.
    """
    candidates = []
    for action in Action:
        reason = ""
        if request.prefix_shift > settings.max_prefix_shift:
            reason = "prefix_shift_exceeds_budget"
        elif action == Action.OLD:
            if not request.old_weights_available:
                reason = "old_weights_unavailable"
            elif not request.cache_available or request.cache_is_mixed or request.cache_version != request.old_version:
                reason = "matching_old_cache_unavailable"
            elif request.current_version - request.old_version > settings.max_old_gap:
                reason = "old_version_exceeds_budget"
            elif request.old_risk is None:
                reason = "old_risk_uncalibrated"
            elif request.old_deadline_seconds is not None and (
                request.old_queue_seconds + request.remaining.p90 * profile.decode_old_seconds_per_token
                > request.old_deadline_seconds
            ):
                reason = "old_engine_deadline"
        elif action == Action.KEEP:
            if not request.cache_available:
                reason = "cache_unavailable"
            elif request.current_version - request.cache_version > settings.max_cache_gap:
                reason = "cache_version_exceeds_budget"
            elif request.cache_risk is None:
                reason = "cache_risk_uncalibrated"
        if reason:
            candidates.append(Candidate(action, False, reason))
            continue

        remaining = request.remaining.mean
        old = action == Action.OLD
        compute = remaining * (profile.decode_old_seconds_per_token if old else profile.decode_new_seconds_per_token)
        if action == Action.REBUILD:
            compute += profile.prefill(request.prefix_tokens)
        queue = request.old_queue_seconds if old else request.new_queue_seconds
        risk = request.old_risk if old else request.cache_risk
        risk_nats = remaining * risk.upper(settings.uncertainty_weight) if action != Action.REBUILD else 0.0
        retention = remaining * settings.old_retention_seconds_per_token if old else 0.0
        score = compute + queue + settings.risk_seconds_per_nat * risk_nats + retention
        candidates.append(Candidate(action, True, "eligible", compute, queue, risk_nats, retention, score))
    eligible = [row for row in candidates if row.feasible]
    if not eligible:
        return Decision(None, tuple(candidates), "abstain: no eligible continuation")
    chosen = min(eligible, key=lambda row: (row.score, list(Action).index(row.action)))
    return Decision(chosen.action, tuple(candidates), "minimum estimated cost plus risk")


def decide_engine(requests: list[Request], profile: CostProfile, settings: Settings) -> Action | None:
    """A single engine chooses one common action, never per-request weights.

    This minimal grouped decision does not model migration or parallel queues.
    Routes needing different versions must use distinct engines in production.
    """
    if not requests:
        raise ValueError("engine request group must not be empty")
    if len({(r.old_version, r.current_version) for r in requests}) != 1:
        raise ValueError("requests on one engine must have compatible versions")
    decisions = [decide(request, profile, settings) for request in requests]
    scores = {}
    for action in Action:
        rows = [next(c for c in decision.candidates if c.action == action) for decision in decisions]
        if all(row.feasible for row in rows):
            scores[action] = sum(row.score for row in rows)
    return min(scores, key=scores.get) if scores else None
