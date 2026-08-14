# SPDX-License-Identifier: Apache-2.0
"""VSI-QA rollout worker and rollout statistics actor."""

import argparse
import asyncio
import math
import random
import time
from collections import deque
from typing import Dict, List, Tuple

import ray
import torch

"""vsiqa"""
import os

from PIL import Image
import pyarrow.parquet as pq
from transformers import AutoProcessor

from vsi_qa_rlvr.inference import InferenceResult
from vsi_qa_rlvr.P3_reward import P3Reward
from vsi_qa_rlvr.reward import IncrementalCountingReward
from vsi_qa_rlvr.trajectory import RLSample
"""vsiqa"""


"""vsiqa"""
def load_train_data(
    data_path: str,
    limit: int | None = None,
) -> List[dict]:
    examples = pq.read_table(
        data_path,
        columns=[
            "sample_id",
            "data_source",
            "prompt",
            "images",
            "reward_model",
            "extra_info",
        ],
        memory_map=True,
    ).to_pylist()
    examples = [
        example
        for example in examples
        if all(
            os.path.isfile(image_record["image"])
            for image_record in example["images"]
        )
    ]
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No ScanNet examples loaded from {data_path!r}.")
    return examples
"""vsiqa"""


@ray.remote
class StatsActor:
    """Aggregates rollout metrics from async rollout workers."""

    def __init__(self, window_size: int, active_timeout_seconds: float):
        self.worker_last_active = {}
        self.active_timeout_seconds = active_timeout_seconds
        """vsiqa"""
        self.reward_sums = deque(maxlen=window_size)
        self.response_lengths = deque(maxlen=window_size)
        self.abort_flags = deque(maxlen=window_size)
        self.total_episodes = 0
        """vsiqa"""

    """vsiqa"""
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
    """vsiqa"""

    def get_stats(self) -> Dict[str, float]:
        active_cutoff = time.time() - self.active_timeout_seconds
        active_workers = sum(
            last_active >= active_cutoff
            for last_active in self.worker_last_active.values()
        )
        """vsiqa"""
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
        """vsiqa"""


class RolloutWorkerActor:
    """CPU Ray actor that generates rollouts and writes RL samples to Replay."""

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
        self.worker_id = int(worker_id)
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor

        """vsiqa"""
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        if args.reward_type == "p3":
            self.reward = P3Reward(self.tokenizer)
        else:
            self.reward = IncrementalCountingReward(self.tokenizer)
        self.train_examples = load_train_data(args.data_path)
        if self._log_detail:
            print(
                "[rollout] "
                f"worker={self.worker_id} loaded ScanNet examples: "
                f"count={len(self.train_examples)} "
                "image_pixels=from_parquet "
                f"path={args.data_path!r}"
            )
        self.batch_id = 0
        self.stopped = False
        """vsiqa"""

    @property
    def _log_detail(self) -> bool:
        return self.worker_id == 0

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

        if self.args.reward_type == "p3":
            reward_info = self.reward.score(
                result.output_tokens,
                ground_truth,
                stop_reason=result.stop_reason,
            )
        else:
            reward_info = self.reward.score(
                result.output_tokens,
                ground_truth,
            )
        return {
            "format_reward": float(reward_info.get("r_format", 0.0)),
            "answer_reward": float(
                reward_info.get("answer_exact_reward", 0.0)
            ),
            "reward": float(reward_info.get("score", 0.0)),
        }
    """vsiqa"""

    def _compute_token_level_advantages(
        self,
        labels: List[int],
        token_rewards: List[float],
    ) -> List[float]:
        if len(labels) != len(token_rewards):
            raise ValueError("labels and token_rewards must have the same length.")
        advantages = [0.0] * len(labels)
        running_return = 0.0
        for index in range(len(labels) - 1, -1, -1):
            if labels[index] == -100:
                continue
            running_return = (
                float(token_rewards[index])
                + self.args.ppo_gamma * running_return
            )
            advantages[index] = running_return
        return advantages

    def _compute_grpo_group_advantages(
        self,
        rewards: List[float],
    ) -> Tuple[float, float, List[float]]:
        if not rewards:
            return 0.0, 0.0, []
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(variance)
        if len(rewards) <= 1 or std <= self.args.grpo_adv_eps:
            return mean, std, [0.0 for _ in rewards]
        return mean, std, [
            (reward - mean) / (std + self.args.grpo_adv_eps)
            for reward in rewards
        ]

    """vsiqa"""
    def build_rl_samples(
        self,
        input_ids: List[int],
        question: str,
        ground_truth: str,
        prompt: str,
        batch_id: int,
        results: List[InferenceResult],
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
        if self.args.rl_algorithm == "grpo":
            _, _, advantages = self._compute_grpo_group_advantages(rewards)
        elif self.args.rl_algorithm == "ppo":
            advantages = [0.0 for _ in rewards]
        else:
            raise ValueError(
                f"Unsupported rl_algorithm: {self.args.rl_algorithm}"
            )

        samples = []
        for sample_id, (
            result,
            generated_text,
            reward_info,
            reward,
            advantage,
        ) in enumerate(
            zip(
                results,
                decoded_texts,
                reward_infos,
                rewards,
                advantages,
            )
        ):
            if (
                not result.output_tokens
                or result.stop_reason == "abort"
                or len(result.output_logprobs) != len(result.output_tokens)
                or len(result.output_versions) != len(result.output_tokens)
            ):
                continue
            response_length = len(result.output_tokens)
            labels = [-100] * len(input_ids) + list(result.output_tokens)
            if self.args.rl_algorithm == "ppo":
                token_rewards = [0.0] * len(labels)
                token_rewards[-1] = float(reward)
                token_advantages = self._compute_token_level_advantages(
                    labels,
                    token_rewards,
                )
                sample_advantage = 0.0
            else:
                token_advantages = (
                    [0.0] * len(input_ids)
                    + [float(advantage)] * response_length
                )
                sample_advantage = float(advantage)
            samples.append(RLSample(
                algorithm=self.args.rl_algorithm,
                input_ids=list(input_ids) + list(result.output_tokens),
                attention_mask=[1]
                * (len(input_ids) + response_length),
                labels=labels,
                old_logprobs=(
                    [0.0] * len(input_ids)
                    + list(result.output_logprobs)
                ),
                advantage=sample_advantage,
                token_advantages=token_advantages,
                response_indices=(
                    [-1] * len(input_ids) + [0] * response_length
                ),
                output_versions=list(result.output_versions),
                prompt_ids=list(input_ids),
                response_ids=list(result.output_tokens),
                old_response_logprobs=list(result.output_logprobs),
                reward=float(reward),
                question=question,
                ground_truth=ground_truth,
                format_reward=float(reward_info["format_reward"]),
                answer_reward=float(reward_info["answer_reward"]),
                rollout_worker_id=self.worker_id,
                batch_id=batch_id,
                sample_id=sample_id,
                stop_reason=result.stop_reason,
                generated_text=generated_text,
                prepared_media=self.prepared_media,
            ))
        return samples
    """vsiqa"""

    """vsiqa"""
    def sample_rollout_prompt(self) -> Tuple[str, str, str]:
        row = random.choice(self.train_examples)
        system_message, user_message = row["prompt"]
        image_records = row["images"]
        min_pixels = int(image_records[0]["min_pixels"])
        max_pixels = int(image_records[0]["max_pixels"])
        if any(
            int(image_record["min_pixels"]) != min_pixels
            or int(image_record["max_pixels"]) != max_pixels
            for image_record in image_records
        ):
            raise ValueError(
                "All images in one rollout row must use the same "
                "min_pixels and max_pixels."
            )
        question = user_message["content"][
            len("<image>\n") * len(image_records) :
        ]
        messages = [
            system_message,
            {
                "role": user_message["role"],
                "content": [
                    {
                        "type": "image",
                        "image": image_record["image"],
                        "min_pixels": int(image_record["min_pixels"]),
                        "max_pixels": int(image_record["max_pixels"]),
                    }
                    for image_record in image_records
                ]
                + [{"type": "text", "text": question}],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images = []
        for image_record in image_records:
            with Image.open(image_record["image"]) as image:
                images.append(image.convert("RGB").copy())

        encoded = self.processor(
            text=[prompt],
            images=images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            return_mm_token_type_ids=True,
            return_tensors="pt",
            padding=True,
        )
        valid = encoded["attention_mask"][0].bool()
        self.prepared_media = {
            "prompt_token_ids": encoded["input_ids"][0][valid].contiguous(),
            "prompt_mm_token_type_ids": (
                encoded["mm_token_type_ids"][0][valid].contiguous()
            ),
            "pixel_values": encoded["pixel_values"].contiguous(),
            "image_grid_thw": encoded["image_grid_thw"].contiguous(),
            "pad_token_id": int(self.processor.tokenizer.pad_token_id),
        }
        self.vllm_prompt = {
            "prompt_token_ids": self.processor.tokenizer.encode(
                prompt,
                add_special_tokens=True,
            ),
            "multi_modal_data": {"image": images},
            "multi_modal_uuids": {
                "image": [
                    f"scannet:{row['sample_id']}:{image_index}"
                    for image_index in range(len(images))
                ]
            },
            "mm_processor_kwargs": {
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
        }
        ground_truth = row["reward_model"]["ground_truth"]
        return question, ground_truth, prompt
    """vsiqa"""

    async def run(self) -> Dict[str, int]:
        """vsiqa"""
        while not self.stopped:
            question, ground_truth, prompt = self.sample_rollout_prompt()
            input_ids = self.prepared_media["prompt_token_ids"].tolist()
            current_batch_id = self.batch_id
            self.batch_id += 1
            num_samples = (
                self.args.grpo_group_size
                if self.args.rl_algorithm == "grpo"
                else self.args.rollout_batch_size
            )
            request_refs = [
                self.infer_actor.request_batch.remote(
                    self.vllm_prompt,
                    self.args.infer_max_tokens,
                )
                for _ in range(num_samples)
            ]
            results = await asyncio.gather(*request_refs)
            if not results:
                continue

            rl_samples = self.build_rl_samples(
                input_ids=list(input_ids),
                question=question,
                ground_truth=ground_truth,
                prompt=prompt,
                batch_id=current_batch_id,
                results=results,
            )
            if not rl_samples:
                continue
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
            version_ranges = sorted({
                (
                    f"{min(result.output_versions)}-"
                    f"{max(result.output_versions)}"
                    if result.output_versions
                    else "none"
                )
                for result in results
            })
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
        """vsiqa"""
