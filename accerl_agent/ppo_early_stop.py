"""Response-token K2 monitoring for one distributed PPO optimizer window."""

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
