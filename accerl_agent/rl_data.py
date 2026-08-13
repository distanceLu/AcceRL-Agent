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


PPO_TERMINAL_REASONS = frozenset({
    "won",
    "lost",
    "step_limit",
    "history_limit",
})


def textworld_ppo_boundary_is_terminal(
    termination_reason: TerminationReason,
) -> bool:
    """Return whether PPO must stop return propagation without bootstrap."""
    return termination_reason in PPO_TERMINAL_REASONS


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


@dataclass
class RawPPOSample:
    input_ids: List[int]
    labels: List[int]  # 模型生成的response/action token:label 等于对应的 input_ids, prompt、observation token:label 等于 -100
    old_logprobs: List[float]  # rollout时模型生成的response/action token的logprob
    token_rewards: List[float]  # 每个 token 对应的奖励
    token_terminated: List[bool]  # 每个 token 是否是终止状态的标记,终止token的回来回报为0
    token_truncated: List[bool]  # 每个 token 是否是截断状态的标记,截断token回来回报可能不为0,需要用bootstrap_value来计算gae
    output_versions: List[int]
    bootstrap_prediction_position: int | None  # 仅用于被截断的 PPO 样本，指出应该在哪个上下文位置预测最终状态价值 V(s_final)


@dataclass
class GRPOSample:
    input_ids: List[int]
    labels: List[int]
    old_logprobs: List[float]
    output_versions: List[int]
    advantage: float  # reward 在构造样本之前已经转换成了组内相对 advantage，所以训练样本不再需要保存原始 reward。


RLSample: TypeAlias = RawPPOSample | GRPOSample


def validate_raw_ppo_sample(sample: RawPPOSample) -> None:
    length = len(sample.input_ids)
    token_fields = {
        "labels": sample.labels,
        "old_logprobs": sample.old_logprobs,
        "token_rewards": sample.token_rewards,
        "token_terminated": sample.token_terminated,
        "token_truncated": sample.token_truncated,
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
