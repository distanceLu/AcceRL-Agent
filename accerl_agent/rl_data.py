from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Literal, Tuple, TypeAlias

import torch


TerminationReason: TypeAlias = Literal[
    "won",
    "lost",
    "environment_done_without_terminal_signal",
    "step_limit",
    "history_limit",
]


def textworld_ppo_boundary_is_terminal(
    termination_reason: TerminationReason,
) -> bool:
    """Return whether PPO must stop return propagation without bootstrap."""
    return termination_reason in {
        "won",
        "lost",
        "step_limit",
        "history_limit",
    }


def classify_textworld_termination_reason(
    *,
    won: bool,
    lost: bool,
    done: bool,
    environment_steps: int,
    max_episode_steps: int,
) -> TerminationReason | None:
    """Classify TextWorld ``done`` while preserving time-limit semantics."""
    if won:
        return "won"
    if lost:
        return "lost"
    if not done:
        return None
    if environment_steps >= max_episode_steps:
        return "step_limit"
    return "environment_done_without_terminal_signal"


@dataclass(frozen=True)
class RawPPOSample:
    input_ids: Tuple[int, ...]
    # Half-open ranges of trainable response tokens within ``input_ids``.
    response_spans: Tuple[Tuple[int, int], ...]
    # These two fields align only with the flattened response spans, not with
    # the full input. Labels are derived directly from ``input_ids``.
    response_logprobs: Tuple[float, ...]
    response_rewards: Tuple[float, ...]
    # The boundary is implicitly on the final response token. A truncated
    # sample bootstraps from the final (non-response) input token.
    boundary_kind: Literal["terminated", "truncated"]
    # Maximum policy version among this sample's response tokens. Training
    # only uses the latest behavior version for replay-lag diagnostics.
    behavior_version: int
    telemetry_id: str | None = None
    created_monotonic_ns: int | None = None
    telemetry_oldest_version: int | None = None
    # Present together only for rollout GAE; aligned with response tokens.
    # These targets remain unchanged when the sample is replayed.
    raw_advantages: Tuple[float, ...] | None = None
    returns: Tuple[float, ...] | None = None
    # Optimizer step of the synchronized backbone + Value Head snapshot.
    value_version: int | None = None

    def __post_init__(self) -> None:
        # Canonical tuples make a validated replay sample deeply immutable;
        # downstream prepare/pack stages can trust its structural invariants.
        try:
            object.__setattr__(self, "input_ids", tuple(self.input_ids))
            object.__setattr__(
                self,
                "response_spans",
                tuple(tuple(span) for span in self.response_spans),
            )
            object.__setattr__(
                self,
                "response_logprobs",
                tuple(self.response_logprobs),
            )
            object.__setattr__(
                self,
                "response_rewards",
                tuple(self.response_rewards),
            )
            for name in ("raw_advantages", "returns"):
                values = getattr(self, name)
                if values is not None:
                    object.__setattr__(self, name, tuple(values))
        except TypeError as exc:
            raise ValueError(
                "PPO replay sequence fields must be iterable."
            ) from exc
        validate_raw_ppo_sample(self)


@dataclass(frozen=True)
class GRPOSample:
    input_ids: Tuple[int, ...]
    # Half-open ranges of trainable response tokens within ``input_ids``.
    response_spans: Tuple[Tuple[int, int], ...]
    # Aligns only with the flattened response spans.
    response_logprobs: Tuple[float, ...]
    # Reward is converted to a group-relative trajectory advantage before the
    # sample enters replay, so the raw reward is not retained.
    advantage: float
    # Maximum policy version among this sample's response tokens. It is used
    # only for replay-lag diagnostics.
    behavior_version: int
    telemetry_id: str | None = None
    created_monotonic_ns: int | None = None
    telemetry_oldest_version: int | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "input_ids", tuple(self.input_ids))
            object.__setattr__(
                self,
                "response_spans",
                tuple(tuple(span) for span in self.response_spans),
            )
            object.__setattr__(
                self,
                "response_logprobs",
                tuple(self.response_logprobs),
            )
        except TypeError as exc:
            raise ValueError(
                "GRPO replay sequence fields must be iterable."
            ) from exc
        validate_grpo_sample(self)


RLSample: TypeAlias = RawPPOSample | GRPOSample


def validate_raw_ppo_sample(sample: RawPPOSample) -> None:
    length = len(sample.input_ids)
    if length < 2:
        raise ValueError("A PPO sample must contain at least two tokens.")
    if not sample.response_spans:
        raise ValueError("A PPO sample must contain a response target.")
    response_count = 0
    previous_end = 0
    for span_index, span in enumerate(sample.response_spans):
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or not all(type(value) is int for value in span)
        ):
            raise ValueError(
                "PPO response spans must be (start, end) integer tuples."
            )
        start, end = span
        if start < 1:
            raise ValueError(
                "The first PPO token cannot be a causal response target."
            )
        if start >= end or end > length:
            raise ValueError(
                f"PPO response span {span_index} is outside the input range."
            )
        if start < previous_end:
            raise ValueError(
                "PPO response spans must be ordered and non-overlapping."
            )
        response_count += end - start
        previous_end = end

    response_fields = {
        "response_logprobs": sample.response_logprobs,
        "response_rewards": sample.response_rewards,
    }
    for name, values in response_fields.items():
        if len(values) != response_count:
            raise ValueError(
                f"{name} must align with response tokens: "
                f"{len(values)} != {response_count}"
            )
    frozen_fields = (sample.raw_advantages, sample.returns, sample.value_version)
    if any(value is not None for value in frozen_fields):
        if any(value is None for value in frozen_fields):
            raise ValueError("PPO rollout advantages, returns and value_version are required together.")
        if type(sample.value_version) is not int or sample.value_version < 0:
            raise ValueError("PPO value_version must be a non-negative optimizer step.")
        for name in ("raw_advantages", "returns"):
            values = getattr(sample, name)
            if len(values) != response_count or not all(math.isfinite(v) for v in values):
                raise ValueError(f"PPO {name} must contain one finite value per response token.")
    if sample.boundary_kind not in {"terminated", "truncated"}:
        raise ValueError(
            "PPO boundary_kind must be 'terminated' or 'truncated'."
        )
    if type(sample.behavior_version) is not int or sample.behavior_version < 0:
        raise ValueError("PPO behavior_version must be a non-negative integer.")
    if (
        sample.boundary_kind == "truncated"
        and sample.response_spans[-1][1] == length
    ):
        raise ValueError(
            "A truncated PPO sample requires final non-response context for "
            "bootstrap."
        )


def validate_grpo_sample(sample: GRPOSample) -> None:
    length = len(sample.input_ids)
    if length < 2:
        raise ValueError("A GRPO sample must contain at least two tokens.")
    if not sample.response_spans:
        raise ValueError("A GRPO sample must contain a response target.")
    response_count = 0
    previous_end = 0
    for span_index, span in enumerate(sample.response_spans):
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or not all(type(value) is int for value in span)
        ):
            raise ValueError(
                "GRPO response spans must be (start, end) integer tuples."
            )
        start, end = span
        if start < 1:
            raise ValueError(
                "The first GRPO token cannot be a causal response target."
            )
        if start >= end or end > length:
            raise ValueError(
                f"GRPO response span {span_index} is outside the input range."
            )
        if start < previous_end:
            raise ValueError(
                "GRPO response spans must be ordered and non-overlapping."
            )
        response_count += end - start
        previous_end = end
    if len(sample.response_logprobs) != response_count:
        raise ValueError(
            "GRPO response_logprobs must align with response tokens: "
            f"{len(sample.response_logprobs)} != {response_count}"
        )
    if type(sample.behavior_version) is not int or sample.behavior_version < 0:
        raise ValueError("GRPO behavior_version must be a non-negative integer.")


def _parallel_reverse_affine_scan(
    deltas: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``b_t + a_t * x_(t+1)`` recurrences in O(log T) stages."""
    accumulated_coefficients = coefficients
    accumulated_values = deltas
    width = deltas.shape[1]
    offset = 1
    while offset < width:
        composed_values = (
            accumulated_values[:, :-offset]
            + accumulated_coefficients[:, :-offset]
            * accumulated_values[:, offset:]
        )
        composed_coefficients = (
            accumulated_coefficients[:, :-offset]
            * accumulated_coefficients[:, offset:]
        )
        accumulated_values = torch.cat(
            (composed_values, accumulated_values[:, -offset:]),
            dim=1,
        )
        accumulated_coefficients = torch.cat(
            (
                composed_coefficients,
                accumulated_coefficients[:, -offset:],
            ),
            dim=1,
        )
        offset *= 2
    return accumulated_values


def compute_batched_token_gae(
    rewards: torch.Tensor,
    baseline_values: torch.Tensor,
    valid_mask: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Checked public API for detached batched token GAE."""
    dense_tensors = {
        "rewards": rewards,
        "baseline_values": baseline_values,
        "valid_mask": valid_mask,
        "terminated": terminated,
        "truncated": truncated,
    }
    for name, tensor in dense_tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, time].")
    if any(tensor.shape != rewards.shape for tensor in dense_tensors.values()):
        raise ValueError("All dense token GAE inputs must have the same shape.")
    if rewards.shape[0] < 1 or rewards.shape[1] < 1:
        raise ValueError("Batched token GAE requires a non-empty batch and time axis.")
    if bootstrap_values.shape != (rewards.shape[0],):
        raise ValueError("bootstrap_values must have shape [batch].")
    if not 0.0 <= gamma:
        raise ValueError("gamma must be non-negative.")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    valid = valid_mask.detach().bool()
    terminal = terminated.detach().bool()
    truncation = truncated.detach().bool()
    if terminal.logical_and(truncation).any():
        raise ValueError("A token cannot be both terminated and truncated.")
    if terminal.logical_or(truncation).logical_and(~valid).any():
        raise ValueError("Episode boundaries must be valid response tokens.")
    valid_counts = valid.sum(dim=1)
    if valid_counts.eq(0).any():
        raise ValueError("Every batched GAE row requires a response token.")
    expected_valid = (
        torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        < valid_counts.unsqueeze(1)
    )
    if not torch.equal(valid, expected_valid):
        raise ValueError("Batched GAE valid masks must be left-aligned.")
    boundary = terminal.logical_or(truncation)
    if boundary.sum(dim=1).ne(1).any():
        raise ValueError("Every batched GAE row requires exactly one boundary.")
    row_indices = torch.arange(valid.shape[0], device=valid.device)
    final_indices = valid_counts - 1
    if not boundary[row_indices, final_indices].all():
        raise ValueError("Each episode boundary must be on its final token.")

    return _compute_batched_token_gae_unchecked(
        rewards=rewards,
        baseline_values=baseline_values,
        valid_mask=valid_mask,
        terminated=terminated,
        truncated=truncated,
        bootstrap_values=bootstrap_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def _compute_batched_token_gae_unchecked(
    rewards: torch.Tensor,
    baseline_values: torch.Tensor,
    valid_mask: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GAE kernel for trainer-validated dense episode timelines."""

    values = baseline_values.detach().float()
    rewards_float = rewards.detach().float()
    bootstrap = bootstrap_values.detach().float()
    valid = valid_mask.detach().bool()
    terminal = terminated.detach().bool()
    truncation = truncated.detach().bool()
    valid_counts = valid.sum(dim=1)
    row_indices = torch.arange(valid.shape[0], device=valid.device)
    final_indices = valid_counts - 1
    next_values = torch.zeros_like(values)
    if values.shape[1] > 1:
        next_values[:, :-1] = values[:, 1:]
    next_values[row_indices, final_indices] = bootstrap

    bootstrap_mask = (~terminal).to(torch.float32)
    # 计算 TD delta: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    deltas = (
        rewards_float
        + float(gamma) * bootstrap_mask * next_values
        - values
    )
    # 仅在有效 token 上计算 GAE padding位置清零
    deltas = torch.where(valid, deltas, torch.zeros_like(deltas))
    trace_coefficients = (
        float(gamma)
        * float(gae_lambda)
        * (~(terminal | truncation)).to(torch.float32)
        * valid.to(torch.float32)
    )
    # 并行反向计算 GAE: A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
    advantages = _parallel_reverse_affine_scan(
        deltas,
        trace_coefficients,
    )
    advantages = torch.where(
        valid,
        advantages,
        torch.zeros_like(advantages),
    )
    returns = torch.where(
        valid,
        values + advantages,
        torch.zeros_like(values),
    )
    return advantages.detach(), returns.detach(), deltas.detach()


def freeze_rollout_ppo_targets(
    sample: RawPPOSample,
    token_values: List[float],
    *,
    value_version: int,
    gamma: float,
    gae_lambda: float,
) -> RawPPOSample:
    """Freeze GAE from one completed transcript and one critic snapshot.

    ``token_values[i]`` is V after consuming input_ids[i]. A response token
    at position p therefore uses V[p - 1], exactly as the trainer does.
    Observation tokens stay in the context, but not in the GAE timeline.
    """
    if sample.raw_advantages is not None:
        raise ValueError("Rollout PPO targets are already frozen.")
    if len(token_values) != len(sample.input_ids):
        raise ValueError("Critic values must align with the complete input transcript.")
    values = torch.tensor(token_values, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("Rollout critic values must be finite.")
    prediction_indices = [
        position - 1
        for start, end in sample.response_spans
        for position in range(start, end)
    ]
    baseline = values[prediction_indices].unsqueeze(0)
    valid = torch.ones_like(baseline, dtype=torch.bool)
    terminated = torch.zeros_like(valid)
    truncated = torch.zeros_like(valid)
    terminal = sample.boundary_kind == "terminated"
    (terminated if terminal else truncated)[0, -1] = True
    advantages, returns, _ = compute_batched_token_gae(
        rewards=torch.tensor([sample.response_rewards], dtype=torch.float32),
        baseline_values=baseline,
        valid_mask=valid,
        terminated=terminated,
        truncated=truncated,
        bootstrap_values=values.new_tensor([0.0]) if terminal else values[-1:],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return replace(
        sample,
        raw_advantages=tuple(advantages[0].tolist()),
        returns=tuple(returns[0].tolist()),
        value_version=value_version,
    )
