# SPDX-License-Identifier: Apache-2.0
"""Inference request, state, result, and statistics contracts for VSI-QA."""

"""vsiqa"""
import time
"""vsiqa"""
from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class OnlineGenerationState:
    """Token-level state for one request across weight-update interruptions."""

    index: int
    """vsiqa"""
    llm_input: dict
    """vsiqa"""
    requested_max_tokens: int
    output_tokens: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
    attempts: int = 0

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.output_tokens))

    """vsiqa"""
    @property
    def restart_engine_input(self) -> dict:
        engine_input = dict(self.llm_input)
        engine_input["prompt_token_ids"] = (
            list(self.llm_input["prompt_token_ids"])
            + list(self.output_tokens)
        )
        engine_input["arrival_time"] = time.time()
        return engine_input
    """vsiqa"""


@dataclass
class RepeatingInferenceStats:
    total_requests: int = 0
    total_tokens: int = 0


@dataclass
class InferenceRequestItem:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    """vsiqa"""
    llm_input: dict
    """vsiqa"""
    requested_max_tokens: int


@dataclass
class InferenceResult:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    output_tokens: List[int]
    output_logprobs: List[float]
    output_versions: List[int]
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None
    attempts: int

    @property
    def version_range(self) -> str:
        if not self.output_versions:
            return "none"
        return f"{min(self.output_versions)}-{max(self.output_versions)}"
