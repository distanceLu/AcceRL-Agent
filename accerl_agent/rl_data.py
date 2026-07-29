from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, TypeAlias

import torch


TerminationReason: TypeAlias = Literal[
    "won",
    "lost",
    "environment_done_without_terminal_signal",
    "step_limit",
    "history_limit",
]


@dataclass
class RawPPOSample:
    algorithm: Literal["ppo"]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    old_logprobs: List[float]
    token_rewards: List[float]
    token_terminated: List[bool]
    token_truncated: List[bool]
    response_indices: List[int]
    output_versions: List[int]
    bootstrap_prediction_position: int | None
    termination_reason: TerminationReason


@dataclass
class GRPOSample:
    algorithm: Literal["grpo"]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    old_logprobs: List[float]
    response_indices: List[int]
    output_versions: List[int]
    advantage: float


RLSample: TypeAlias = RawPPOSample | GRPOSample


def validate_raw_ppo_sample(sample: RawPPOSample) -> None:
    length = len(sample.input_ids)
    token_fields = {
        "attention_mask": sample.attention_mask,
        "labels": sample.labels,
        "old_logprobs": sample.old_logprobs,
        "token_rewards": sample.token_rewards,
        "token_terminated": sample.token_terminated,
        "token_truncated": sample.token_truncated,
        "response_indices": sample.response_indices,
        "output_versions": sample.output_versions,
    }
    if length < 2:
        raise ValueError("A PPO sample must contain at least two tokens.")
    for name, values in token_fields.items():
        if len(values) != length:
            raise ValueError(
                f"{name} must be token-aligned: {len(values)} != {length}"
            )
    if sample.labels[0] != -100:
        raise ValueError("The first PPO token cannot be a causal target.")
    if any(mask != 1 for mask in sample.attention_mask):
        raise ValueError("Raw PPO samples cannot contain padding.")

    valid_targets = [
        index for index, label in enumerate(sample.labels) if label != -100
    ]
    if not valid_targets:
        raise ValueError("A PPO sample must contain a response target.")
    for index in range(length):
        is_target = sample.labels[index] != -100
        if is_target:
            if sample.labels[index] != sample.input_ids[index]:
                raise ValueError(
                    "PPO response labels must equal their input token ids."
                )
            if sample.response_indices[index] < 0:
                raise ValueError(
                    "Response targets require non-negative response indices."
                )
            if sample.output_versions[index] < 0:
                raise ValueError(
                    "Response targets require non-negative behavior versions."
                )
        else:
            if sample.old_logprobs[index] != 0.0:
                raise ValueError("Ignored tokens must have zero old logprob.")
            if sample.token_rewards[index] != 0.0:
                raise ValueError("Ignored tokens must have zero token reward.")
            if sample.token_terminated[index] or sample.token_truncated[index]:
                raise ValueError("Ignored tokens cannot terminate or truncate.")
            if sample.response_indices[index] != -1:
                raise ValueError("Ignored tokens require response_index=-1.")
            if sample.output_versions[index] != -1:
                raise ValueError("Ignored tokens require output_version=-1.")

    terminated_indices = [
        index
        for index, value in enumerate(sample.token_terminated)
        if value
    ]
    truncated_indices = [
        index
        for index, value in enumerate(sample.token_truncated)
        if value
    ]
    if terminated_indices and truncated_indices:
        raise ValueError("terminated and truncated are mutually exclusive.")
    boundary_indices = terminated_indices + truncated_indices
    if len(boundary_indices) != 1:
        raise ValueError(
            "Each PPO episode must have exactly one terminal or truncation boundary."
        )
    if boundary_indices[0] != valid_targets[-1]:
        raise ValueError(
            "The episode boundary must be the final response target token."
        )

    is_terminated = bool(terminated_indices)
    if is_terminated:
        if sample.bootstrap_prediction_position is not None:
            raise ValueError("Terminated PPO samples cannot bootstrap.")
        if sample.termination_reason not in {"won", "lost"}:
            raise ValueError(
                "Only won/lost termination reasons may be terminal."
            )
    else:
        position = sample.bootstrap_prediction_position
        if position is None:
            raise ValueError("Truncated PPO samples require a bootstrap position.")
        if not 0 <= position < length:
            raise ValueError("PPO bootstrap position is out of range.")
        if position != length - 1:
            raise ValueError(
                "PPO bootstrap position must be the final context token."
            )
        if position <= valid_targets[-1]:
            raise ValueError(
                "PPO bootstrap context must follow the final response token."
            )
        if sample.labels[position] != -100:
            raise ValueError(
                "PPO bootstrap position must point to ignored final-state context."
            )
        if sample.termination_reason in {"won", "lost"}:
            raise ValueError("won/lost PPO samples must be terminal.")
        if sample.termination_reason not in {
            "environment_done_without_terminal_signal",
            "step_limit",
            "history_limit",
        }:
            raise ValueError("Invalid PPO truncation reason.")


def compute_token_gae(
    rewards: torch.Tensor,
    baseline_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute detached token GAE, lambda returns, and one-step TD deltas."""
    tensors = {
        "rewards": rewards,
        "baseline_values": baseline_values,
        "terminated": terminated,
        "truncated": truncated,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
    num_tokens = rewards.numel()
    if num_tokens < 1:
        raise ValueError("Token GAE requires at least one response token.")
    if any(tensor.numel() != num_tokens for tensor in tensors.values()):
        raise ValueError("All token GAE inputs must have the same length.")
    if terminated.bool().logical_and(truncated.bool()).any():
        raise ValueError("A token cannot be both terminated and truncated.")
    boundary = terminated.bool().logical_or(truncated.bool())
    if int(boundary.sum().item()) != 1 or not bool(boundary[-1].item()):
        raise ValueError(
            "Token GAE requires exactly one boundary on the final token."
        )
    if not 0.0 <= gamma:
        raise ValueError("gamma must be non-negative.")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    rewards = rewards.detach().float()
    values = baseline_values.detach().float()
    bootstrap_value = bootstrap_value.detach().float().reshape(())
    terminated = terminated.detach().bool()
    truncated = truncated.detach().bool()
    advantages = torch.zeros_like(rewards)
    deltas = torch.zeros_like(rewards)
    next_advantage = rewards.new_zeros(())
    next_value = bootstrap_value

    for index in range(num_tokens - 1, -1, -1):
        bootstrap_mask = (~terminated[index]).to(torch.float32)
        trace_mask = (~(terminated[index] | truncated[index])).to(torch.float32)
        delta = (
            rewards[index]
            + float(gamma) * bootstrap_mask * next_value
            - values[index]
        )
        advantage = (
            delta
            + float(gamma)
            * float(gae_lambda)
            * trace_mask
            * next_advantage
        )
        deltas[index] = delta
        advantages[index] = advantage
        next_value = values[index]
        next_advantage = advantage

    returns = values + advantages
    return advantages.detach(), returns.detach(), deltas.detach()


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
    """Compute detached token GAE for left-aligned dense episode timelines."""
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

    values = baseline_values.detach().float()
    rewards_float = rewards.detach().float()
    bootstrap = bootstrap_values.detach().float()
    next_values = torch.zeros_like(values)
    if values.shape[1] > 1:
        next_values[:, :-1] = values[:, 1:]
    next_values[row_indices, final_indices] = bootstrap

    bootstrap_mask = (~terminal).to(torch.float32)
    deltas = (
        rewards_float
        + float(gamma) * bootstrap_mask * next_values
        - values
    )
    deltas = torch.where(valid, deltas, torch.zeros_like(deltas))
    trace_coefficients = (
        float(gamma)
        * float(gae_lambda)
        * (~(terminal | truncation)).to(torch.float32)
        * valid.to(torch.float32)
    )
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
