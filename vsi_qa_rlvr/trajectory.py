# SPDX-License-Identifier: Apache-2.0
"""Trainer-facing trajectory contracts for VSI-QA reinforcement learning."""

from dataclasses import dataclass
from typing import List


@dataclass
class RLSample:
    prompt_ids: List[int]
    response_ids: List[int]
    old_response_logprobs: List[float]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    reward: float
    advantage: float
    question: str
    ground_truth: str
    format_reward: float
    answer_reward: float
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    output_versions: List[int]
    stop_reason: str | None
    generated_text: str


@dataclass
class VSIQARLSample(RLSample):
    prepared_media: dict
