# SPDX-License-Identifier: Apache-2.0
"""Task-independent results produced by asynchronous rollout workers."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RolloutResult:
    """One completed candidate from an asynchronous behavior policy."""

    rollout_worker_id: int
    rollout_group_id: int
    candidate_index: int
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    behavior_response_logprobs: list[float]
    response_token_policy_versions: list[int]
    generated_text: str
    sample_cost: int
    finish_reason: Literal["length", "stop", "tool_calls", "abort"] | None
    attempt_count: int
