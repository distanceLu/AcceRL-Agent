# SPDX-License-Identifier: Apache-2.0
"""VSI-QA rollout worker and rollout statistics actor."""

import argparse
import random
import time
from collections import deque
from typing import Dict, List

import ray
import torch

"""vsiqa"""
from vsi_qa_rlvr.dataloaders.qwen3vl_rollout_collator import (
    Qwen3VLRolloutDataCollator,
)
from vsi_qa_rlvr.dataloaders.vsi_qa_dataset import VSIQADataset
from vsi_qa_rlvr.inference import InferenceResult
from vsi_qa_rlvr.scannet_incremental_counting.reward import (
    IncrementalCountingReward,
)
from vsi_qa_rlvr.trajectory import RLSample
"""vsiqa"""


@ray.remote
class StatsActor:
    """Aggregates rollout metrics from async rollout workers."""

    def __init__(self, window_size: int, active_timeout_seconds: float):
        self.reward_sums = deque(maxlen=window_size)
        self.response_lengths = deque(maxlen=window_size)
        self.abort_flags = deque(maxlen=window_size)
        self.worker_last_active = {}
        self.active_timeout_seconds = active_timeout_seconds
        self.total_episodes = 0

    def add_rollout_batch(
        self,
        worker_id: int,
        rewards: List[float],
        response_lengths: List[int],
        abort_flags: List[bool],
    ) -> None:
        if not (len(rewards) == len(response_lengths) == len(abort_flags)):
            raise ValueError(
                "Rollout metric batch lengths must match: "
                f"rewards={len(rewards)} response_lengths={len(response_lengths)} "
                f"abort_flags={len(abort_flags)}"
            )
        self.reward_sums.extend(float(value) for value in rewards)
        self.response_lengths.extend(int(value) for value in response_lengths)
        self.abort_flags.extend(bool(value) for value in abort_flags)
        self.total_episodes += len(rewards)
        self.worker_last_active[int(worker_id)] = time.time()

    def get_stats(self) -> Dict[str, float]:
        now = time.time()
        active_cutoff = now - self.active_timeout_seconds
        active_workers = sum(
            1 for last_active in self.worker_last_active.values()
            if last_active >= active_cutoff
        )
        reward_count = len(self.reward_sums)
        response_count = len(self.response_lengths)
        abort_count = len(self.abort_flags)
        return {
            "global_reward_sum_mean": (
                sum(self.reward_sums) / reward_count if reward_count else 0.0
            ),
            "response_length_mean": (
                sum(self.response_lengths) / response_count if response_count else 0.0
            ),
            "abort_rate": (
                sum(1 for flag in self.abort_flags if flag) / abort_count
                if abort_count else 0.0
            ),
            "active_workers": active_workers,
            "total_episodes": self.total_episodes,
        }


class RolloutWorkerActor:
    """CPU Ray actor that builds prompts and feeds token IDs to InferActor."""

    def __init__(
        self,
        args: argparse.Namespace,
        infer_actor,
        replay_buffer,
        stats_actor,
        worker_id: int,
    ):
        self.args = args
        self.infer_actor = infer_actor
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor
        self.worker_id = int(worker_id)

        """vsiqa"""
        self.dataset = VSIQADataset(args.data_path)
        self.collator = Qwen3VLRolloutDataCollator(
            args.model_path,
            max_model_len=args.max_model_len,
        )
        self.tokenizer = self.collator.processor.tokenizer
        self.reward = IncrementalCountingReward(self.tokenizer)
        print(
            "[rollout] "
            f"worker={self.worker_id} loaded VSI-QA examples: "
            f"count={len(self.dataset)} path={args.data_path!r}"
        )
        """vsiqa"""
        self.batch_id = 0
        self.stopped = False

    async def stop(self):
        self.stopped = True

    """vsiqa"""
    def compute_reward_info(
        self,
        question: str,
        ground_truth: str,
        prompt: str,
        generated_text: str,
        result: InferenceResult,
    ) -> Dict[str, float]:
        del question, prompt, generated_text

        if result.stop_reason == "abort" or not result.output_tokens:
            return {
                "format_reward": 0.0,
                "answer_reward": 0.0,
                "reward": 0.0,
            }

        reward_info = self.reward.score(result.output_tokens, ground_truth)
        return {
            "format_reward": float(reward_info.get("r_format", 0.0)),
            "answer_reward": float(
                reward_info.get("answer_exact_reward", 0.0)
            ),
            "reward": float(reward_info.get("score", 0.0)),
        }
    """vsiqa"""

    def compute_group_advantages(self, rewards: List[float]) -> List[float]:
        if not rewards:
            return []

        valid_count = sum(1 for reward in rewards if reward > 0.0)
        if valid_count <= 1:
            return [0.0 for _ in rewards]

        reward_range = max(rewards) - min(rewards)
        if reward_range < 1e-6:
            return [0.0 for _ in rewards]

        rewards_t = torch.tensor(rewards, dtype=torch.float64)

        mean = rewards_t.mean()
        std = rewards_t.std(unbiased=False)
        if std.item() < 1e-6:
            return [0.0 for _ in rewards]

        advantages = (rewards_t - mean) / (std + 1e-6)
        return [float(value) for value in advantages.tolist()]

    """vsiqa"""
    def build_rl_samples(
        self,
        input_ids: List[int],
        question: str,
        ground_truth: str,
        prompt: str,
        results: List[InferenceResult],
        prepared_media: dict,
    ) -> List[RLSample]:
        decoded_texts = [
            self.tokenizer.decode(result.output_tokens, skip_special_tokens=True)
            for result in results
        ]
        reward_infos = [
            self.compute_reward_info(
                question,
                ground_truth,
                prompt,
                generated_text,
                result,
            )
            for generated_text, result in zip(decoded_texts, results)
        ]
        rewards = [info["reward"] for info in reward_infos]
        advantages = self.compute_group_advantages(rewards)

        return [
            RLSample(
                prompt_ids=list(input_ids),
                response_ids=list(result.output_tokens),
                old_response_logprobs=list(result.output_logprobs),
                input_ids=list(input_ids) + list(result.output_tokens),
                attention_mask=[1]
                * (len(input_ids) + len(result.output_tokens)),
                labels=[-100] * len(input_ids) + list(result.output_tokens),
                reward=float(reward),
                advantage=float(advantage),
                question=question,
                ground_truth=ground_truth,
                format_reward=float(reward_info["format_reward"]),
                answer_reward=float(reward_info["answer_reward"]),
                rollout_worker_id=self.worker_id,
                batch_id=result.batch_id,
                sample_id=result.sample_id,
                output_versions=list(result.output_versions),
                stop_reason=result.stop_reason,
                generated_text=generated_text,
                prepared_media=prepared_media,
            )
            for result, generated_text, reward_info, reward, advantage in zip(
                results,
                decoded_texts,
                reward_infos,
                rewards,
                advantages,
            )
        ]
    """vsiqa"""

    """vsiqa"""
    def sample_rollout_prompt(self):
        return self.collator([random.choice(self.dataset)])[0]
    """vsiqa"""

    async def run(self):
        while not self.stopped:
            """vsiqa"""
            rollout_item = self.sample_rollout_prompt()
            input_ids = rollout_item["prepared_media"][
                "prompt_token_ids"
            ].tolist()
            question = rollout_item["question"]
            ground_truth = rollout_item["reward_model"]["ground_truth"]
            prompt = rollout_item["prompt"]
            vllm_prompt = rollout_item["vllm_prompt"]
            prepared_media = rollout_item["prepared_media"]
            """vsiqa"""
            current_batch_id = self.batch_id
            self.batch_id += 1
            """vsiqa"""
            results = await self.infer_actor.request_batch.remote(
                self.worker_id,
                current_batch_id,
                vllm_prompt,
                self.args.infer_max_tokens,
                self.args.rollout_batch_size,
            )
            """vsiqa"""
            if not results:
                continue

            """vsiqa"""
            rl_samples = self.build_rl_samples(
                input_ids=list(input_ids),
                question=question,
                ground_truth=ground_truth,
                prompt=prompt,
                results=results,
                prepared_media=prepared_media,
            )
            """vsiqa"""
            self.replay_buffer.add_samples.remote(rl_samples)
            self.stats_actor.add_rollout_batch.remote(
                self.worker_id,
                [sample.reward for sample in rl_samples],
                [len(sample.response_ids) for sample in rl_samples],
                [sample.stop_reason == "abort" for sample in rl_samples],
            )

            rewards = [sample.reward for sample in rl_samples]
            advantages = [sample.advantage for sample in rl_samples]
            response_lengths = [len(sample.response_ids) for sample in rl_samples]
            reward_t = torch.tensor(rewards, dtype=torch.float32)
            advantage_t = torch.tensor(advantages, dtype=torch.float32)
            response_length_t = torch.tensor(response_lengths, dtype=torch.float32)
            version_ranges = sorted({result.version_range for result in results})
            stop_reasons = sorted({
                str(sample.stop_reason)
                for sample in rl_samples
            })
            print(
                "[rollout] "
                f"worker={self.worker_id} "
                f"batch={current_batch_id} "
                f"samples={len(rl_samples)} "
                f"reward_mean={reward_t.mean().item():.4f} "
                f"reward_std={reward_t.std(unbiased=False).item():.4f} "
                f"adv_mean={advantage_t.mean().item():.4f} "
                f"adv_std={advantage_t.std(unbiased=False).item():.4f} "
                f"response_len_mean={response_length_t.mean().item():.1f} "
                f"versions={','.join(version_ranges)} "
                f"stops={','.join(stop_reasons)}"
            )

        print(f"[rollout] worker={self.worker_id} stopped.")
        return {"worker_id": self.worker_id, "batches": self.batch_id}
