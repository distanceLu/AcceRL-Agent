"""Response-token KL monitoring for one distributed PPO optimizer window."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist


def graph_zero(*tensors: torch.Tensor) -> torch.Tensor:
    """Return a finite zero connected to every supplied backward graph.

    Empty slices avoid reading possibly nonfinite actor/value outputs. This
    preserves backward traversal without the NaN contamination of ``loss * 0``.
    All tensors must be on the same device, as they are in a PPO token view.
    """
    if not tensors:
        raise ValueError("graph_zero requires at least one tensor")
    loss = tensors[0].reshape(-1)[:0].float().sum()
    for tensor in tensors[1:]:
        loss = loss + tensor.reshape(-1)[:0].float().sum()
    return loss


@dataclass
class RawTokenKLState:
    """Monitor K2/K3 over the same unfiltered response tokens.

    The caller supplies the original log-ratio before constructing any safe
    values for the loss. Monitoring is detached FP64 work: even a finite
    log-ratio whose exponential overflows only makes the reported K3 infinite,
    without changing the loss, token filter, or optimizer decision.
    """

    stats: torch.Tensor | None = None

    @torch.no_grad()
    def observe(self, raw_logratio: torch.Tensor) -> None:
        """Accumulate local token sums without a distributed collective."""
        delta = raw_logratio.detach().double()
        finite = torch.isfinite(delta)
        finite_delta = delta[finite]
        stats = torch.stack((
            0.5 * finite_delta.square().sum(),
            (torch.expm1(finite_delta) - finite_delta).sum(),
            delta.new_tensor(delta.numel()),
            (~finite).double().sum(),
        ))
        self.stats = stats if self.stats is None else self.stats + stats

    @torch.no_grad()
    def finalize(self, *, group=None) -> dict[str, float | int]:
        """Sum across ranks once, then divide both estimates by token count."""
        if self.stats is None:
            raise RuntimeError("Raw KL monitoring requires an observed microbatch")
        stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        k2_sum, k3_sum, count, nonfinite = stats.tolist()
        denominator = max(count, 1)
        return {
            "raw_k2_kl": math.inf if nonfinite else k2_sum / denominator,
            "raw_k3_kl": math.inf if nonfinite else k3_sum / denominator,
            "raw_kl_token_count": int(count),
            "raw_kl_nonfinite_tokens": int(nonfinite),
        }


@dataclass
class K2EarlyStopState:
    """Latch stop decisions from global microbatch K2, retaining window metrics.

    Inputs to ``observe`` contain only response tokens. Every training rank
    must call it once per pack in the same order, including after a stop.
    Threshold zero is a valid numerical threshold here; the CLI decides
    whether to construct a state when early stopping is disabled.
    """

    threshold: float
    stop_ppo: bool = False
    squared_sum: float = 0.0
    token_count: int = 0
    max_microbatch_kl: float = 0.0
    microbatches: int = 0
    first_stopped_microbatch: int = 0
    nonfinite_tokens: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("K2 threshold must be finite and nonnegative")

    @torch.no_grad()
    def observe(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        *,
        group=None,
    ) -> float:
        """Globally reduce half squared log-ratio sum, token count, bad count.

        Each rank contributes token sums, rather than a rank-local mean, so
        unequal local response lengths do not bias the global statistic.
        FP64 subtraction and squaring avoid overflow for finite FP32 inputs.
        Nonfinite response-token differences always stop the whole window.
        """
        if new_logprobs.shape != old_logprobs.shape:
            raise ValueError("K2 new and old response logprobs must have the same shape")
        delta = new_logprobs.detach().double() - old_logprobs.detach().double()
        finite = torch.isfinite(delta)
        stats = torch.stack((
            0.5 * delta[finite].square().sum(),
            delta.new_tensor(delta.numel()),
            (~finite).double().sum(),
        ))
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        squared_sum, count, bad_count = stats.tolist()
        if bad_count or not math.isfinite(squared_sum):
            squared_sum = math.inf
        k2 = squared_sum / max(count, 1)

        self.microbatches += 1
        self.squared_sum += squared_sum
        self.token_count += int(count)
        self.nonfinite_tokens += int(bad_count)
        self.max_microbatch_kl = max(self.max_microbatch_kl, k2)
        if not math.isfinite(k2) or k2 > self.threshold:
            if not self.stop_ppo:
                self.first_stopped_microbatch = self.microbatches
            self.stop_ppo = True
        return k2

    def metrics(self) -> dict[str, float | int]:
        """Report the complete observed window, including packs after a stop."""
        return {
            "k2_kl": self.squared_sum / max(self.token_count, 1),
            "k2_kl_max_microbatch": self.max_microbatch_kl,
            "early_stop": float(self.stop_ppo),
            "early_stop_microbatch": self.first_stopped_microbatch,
            "early_stop_nonfinite_tokens": self.nonfinite_tokens,
            "optimizer_step_skipped": float(self.stop_ppo),
        }


@dataclass
class K2TokenFilterState:
    """Reject action tokens locally and reduce statistics once per window.

    The caller keeps the original loss denominators when applying the returned
    mask. Finite high-K2 tokens only lose their direct policy objective;
    raw K3 regularization remains active on every response token. They do not
    latch an early stop. Nonfinite action log-ratios and invalid model/loss
    inputs are reported separately so the shared optimizer can reject an
    unsafe window after every rank completes its backward schedule.
    """

    threshold: float = 0.01
    stats: torch.Tensor | None = None
    kept_tokens: int = 0
    nonfinite_tokens: int = 0
    invalid_microbatches: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("K2 threshold must be finite and nonnegative")

    @torch.no_grad()
    def observe(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the local keep mask without any distributed collective.

        A missing action mask means every input position is a response token.
        Detached FP64 subtraction/squaring avoids overflow on finite FP32
        logprobs and keeps monitoring out of the actor's backward graph.
        """
        if new_logprobs.shape != old_logprobs.shape or (
            mask is not None and new_logprobs.shape != mask.shape
        ):
            raise ValueError("K2 logprobs and action-token mask must have the same shape")
        action = (
            torch.ones_like(new_logprobs, dtype=torch.bool)
            if mask is None else mask.to(device=new_logprobs.device, dtype=torch.bool)
        )
        delta = new_logprobs.detach().double() - old_logprobs.detach().double()
        finite = torch.isfinite(delta)
        k2 = 0.5 * delta.square()
        keep = action & finite & (k2 <= self.threshold)
        stats = torch.stack((
            k2[action & finite].sum(),
            action.double().sum(),
            keep.double().sum(),
            (action & ~finite).double().sum(),
            k2[keep].sum(),
            k2.new_zeros(()),
        ))
        self.stats = stats if self.stats is None else self.stats + stats
        return keep

    @torch.no_grad()
    def invalidate(self) -> None:
        """Record an unsafe model/loss input in the current microbatch."""
        if self.stats is None:
            raise RuntimeError("Token filtering requires an observed microbatch before invalidation")
        self.stats[5] += 1

    @torch.no_grad()
    def finalize(self, *, group=None) -> dict[str, float | int]:
        """DP-sum complete window statistics, including empty local packs."""
        if self.stats is None:
            raise RuntimeError("Token filtering requires at least one observed microbatch")
        stats = self.stats.clone()
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        squared_sum, count, kept, nonfinite, kept_sum, invalid = stats.tolist()
        self.kept_tokens = int(kept)
        self.nonfinite_tokens = int(nonfinite)
        self.invalid_microbatches = int(invalid)
        raw_k2 = math.inf if nonfinite else squared_sum / max(count, 1)
        return {
            "k2_kl": raw_k2,
            "token_filter/action_tokens": int(count),
            "token_filter/kept_tokens": self.kept_tokens,
            "token_filter/masked_tokens": int(count - kept),
            "token_filter/masked_fraction": (count - kept) / max(count, 1),
            "token_filter/nonfinite_tokens": self.nonfinite_tokens,
            "token_filter/invalid_microbatches": self.invalid_microbatches,
            "token_filter/k2_kl": raw_k2,
            "token_filter/kept_k2_kl": kept_sum / max(kept, 1),
            "token_filter/threshold": self.threshold,
        }
