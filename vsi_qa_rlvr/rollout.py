# SPDX-License-Identifier: Apache-2.0
"""VSI-QA rollout worker and rollout statistics actor."""

import time
from collections import deque
from typing import Dict, List

import ray
import torch
from torch.utils.data import DataLoader, RandomSampler

from vsi_qa_rlvr.dataloaders.qwen3vl_rollout_collator import (
    Qwen3VLRolloutDataCollator,
)
from vsi_qa_rlvr.dataloaders.vsi_qa_dataset import VSIQADataset
from vsi_qa_rlvr.exp02_v4_reward import score_vsi_qa_group
from vsi_qa_rlvr.trajectory import VSIQARLSample


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
            1
            for last_active in self.worker_last_active.values()
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
                sum(self.response_lengths) / response_count
                if response_count
                else 0.0
            ),
            "abort_rate": (
                sum(1 for flag in self.abort_flags if flag) / abort_count
                if abort_count
                else 0.0
            ),
            "active_workers": active_workers,
            "total_episodes": self.total_episodes,
        }


class VSIQARolloutWorkerActor:
    """CPU Ray actor that feeds VSI-QA rollout requests to vLLM."""

    def __init__(
        self,
        args,
        infer_actor,
        replay_buffer,
        stats_actor,
        worker_id,
    ):
        self.args = args
        self.infer_actor = infer_actor
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor
        self.worker_id = int(worker_id)
        self.dataset = VSIQADataset(args.data_path)
        self.collator = Qwen3VLRolloutDataCollator(
            args.model_path,
            max_model_len=args.max_model_len,
        )
        self.sampler = RandomSampler(
            self.dataset,
            replacement=True,
            num_samples=len(self.dataset),
        )
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=args.rollout_data_batch_size,
            sampler=self.sampler,
            collate_fn=self.collator,
            num_workers=args.rollout_data_workers,
            prefetch_factor=args.rollout_prefetch_factor,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        self.data_iterator = None
        self.batch_id = 0
        self.stopped = False
        print(
            "[rollout] "
            f"worker={self.worker_id} loaded VSI-QA examples: "
            f"count={len(self.dataset)} path={args.data_path!r} "
            f"data_workers={args.rollout_data_workers} "
            f"prefetch_factor={args.rollout_prefetch_factor}"
        )

    def next_rollout_batch(self):
        try:
            if self.data_iterator is None:
                self.data_iterator = iter(self.data_loader)
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            return next(self.data_iterator)

    async def stop(self):
        self.stopped = True

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

    def build_rl_samples(self, rollout_item, results):
        generated_texts = [
            self.collator.processor.tokenizer.decode(
                result.output_tokens,
                skip_special_tokens=True,
            )
            for result in results
        ]
        valid_indices = [
            index
            for index, result in enumerate(results)
            if result.stop_reason != "abort" and result.output_tokens
        ]
        reward_details = [None] * len(results)
        if valid_indices:
            valid_reward_details = score_vsi_qa_group(
                [generated_texts[index] for index in valid_indices],
                rollout_item["reward_model"]["ground_truth"],
                rollout_item["extra_info"],
                history_path=self.args.reward_history_path,
            )
            for index, details in zip(valid_indices, valid_reward_details):
                reward_details[index] = details
        for index, details in enumerate(reward_details):
            if details is None:
                reward_details[index] = {
                    "score": 0.0,
                    "r_format": 0.0,
                    "answer_exact_reward": 0.0,
                }

        rewards = [float(details["score"]) for details in reward_details]
        advantages = self.compute_group_advantages(rewards)
        prompt_ids = rollout_item["prepared_media"][
            "prompt_token_ids"
        ].tolist()
        return [
            VSIQARLSample(
                prompt_ids=list(prompt_ids),
                response_ids=list(result.output_tokens),
                old_response_logprobs=list(result.output_logprobs),
                input_ids=list(prompt_ids) + list(result.output_tokens),
                attention_mask=[1]
                * (len(prompt_ids) + len(result.output_tokens)),
                labels=[-100] * len(prompt_ids) + list(result.output_tokens),
                reward=float(reward),
                advantage=float(advantage),
                question=rollout_item["question"],
                ground_truth=rollout_item["reward_model"]["ground_truth"],
                format_reward=float(details["r_format"]),
                answer_reward=float(details["answer_exact_reward"]),
                rollout_worker_id=self.worker_id,
                batch_id=result.batch_id,
                sample_id=result.sample_id,
                output_versions=list(result.output_versions),
                stop_reason=result.stop_reason,
                generated_text=generated_text,
                prepared_media=rollout_item["prepared_media"],
            )
            for result, generated_text, details, reward, advantage in zip(
                results,
                generated_texts,
                reward_details,
                rewards,
                advantages,
            )
        ]

    async def run(self):
        while not self.stopped:
            rollout_batch = self.next_rollout_batch()
            for rollout_item in rollout_batch:
                current_batch_id = self.batch_id
                self.batch_id += 1
                results = await self.infer_actor.request_batch.remote(
                    self.worker_id,
                    current_batch_id,
                    rollout_item["llm_input"],
                    self.args.infer_max_tokens,
                    self.args.rollout_batch_size,
                )
                if not results:
                    continue
                rl_samples = self.build_rl_samples(rollout_item, results)
                self.replay_buffer.add_samples.remote(rl_samples)
                self.stats_actor.add_rollout_batch.remote(
                    self.worker_id,
                    [sample.reward for sample in rl_samples],
                    [len(sample.response_ids) for sample in rl_samples],
                    [sample.stop_reason == "abort" for sample in rl_samples],
                )

                rewards = [sample.reward for sample in rl_samples]
                advantages = [sample.advantage for sample in rl_samples]
                response_lengths = [
                    len(sample.response_ids) for sample in rl_samples
                ]
                reward_t = torch.tensor(rewards, dtype=torch.float32)
                advantage_t = torch.tensor(advantages, dtype=torch.float32)
                response_length_t = torch.tensor(
                    response_lengths,
                    dtype=torch.float32,
                )
                version_ranges = sorted(
                    {result.version_range for result in results}
                )
                stop_reasons = sorted(
                    {str(sample.stop_reason) for sample in rl_samples}
                )
                print(
                    "[rollout] "
                    f"worker={self.worker_id} "
                    f"batch={current_batch_id} "
                    f"samples={len(rl_samples)} "
                    f"reward_mean={reward_t.mean().item():.4f} "
                    f"reward_std={reward_t.std(unbiased=False).item():.4f} "
                    f"adv_mean={advantage_t.mean().item():.4f} "
                    f"adv_std={advantage_t.std(unbiased=False).item():.4f} "
                    f"response_len_mean="
                    f"{response_length_t.mean().item():.1f} "
                    f"versions={','.join(version_ranges)} "
                    f"stops={','.join(stop_reasons)}"
                )

        print(f"[rollout] worker={self.worker_id} stopped.")
        return {"worker_id": self.worker_id, "batches": self.batch_id}
