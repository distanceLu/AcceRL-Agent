# SPDX-License-Identifier: Apache-2.0
"""Trainer-facing trajectory contracts for VSI-QA reinforcement learning."""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class RLSample:
    algorithm: Literal["ppo", "grpo"]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    old_logprobs: List[float]
    advantage: float
    token_advantages: List[float]
    response_indices: List[int]
    output_versions: List[int]
    """vsiqa"""
    prompt_ids: List[int]
    response_ids: List[int]
    old_response_logprobs: List[float]
    reward: float
    question: str
    ground_truth: str
    format_reward: float
    answer_reward: float
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    stop_reason: str | None
    generated_text: str
    prepared_media: dict
    """vsiqa"""
