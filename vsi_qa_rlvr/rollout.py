# SPDX-License-Identifier: Apache-2.0
"""VSI-QA Dataset rollout adapter for AcceRL's original Replay workflow."""

from dataclasses import dataclass

from torch.utils.data import DataLoader, RandomSampler

from accerl_agent.vllm_fsdp import RLSample, RolloutWorkerActor
from vsi_qa_rlvr.dataloaders.qwen3vl_rollout_collator import (
    Qwen3VLRolloutDataCollator,
)
from vsi_qa_rlvr.dataloaders.vsi_qa_dataset import VSIQADataset
from vsi_qa_rlvr.exp02_v4_reward import score_vsi_qa_group


@dataclass
class VSIQARLSample(RLSample):
    prepared_media: dict


class VSIQARolloutWorkerActor(RolloutWorkerActor):
    """Replace GSM8K prompt loading with a VSI-QA DataLoader."""

    def __init__(self, args, infer_actor, replay_buffer, worker_id):
        self.args = args
        self.infer_actor = infer_actor
        self.replay_buffer = replay_buffer
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

    async def run(self):
        while not self.stopped:
            rollout_batch = self.next_rollout_batch()
            for rollout_item in rollout_batch:
                current_batch_id = self.batch_id
                self.batch_id += 1
                results = await self.infer_actor.generate_group.remote(
                    rollout_worker_id=self.worker_id,
                    batch_id=current_batch_id,
                    llm_input=rollout_item["llm_input"],
                    num_samples=self.args.rollout_n,
                    max_tokens=self.args.infer_max_tokens,
                )
                results = [
                    result for result in results if result.output_tokens
                ]
                if not results:
                    continue
                group_samples = []
                generated_texts = [
                    self.collator.processor.tokenizer.decode(
                        result.output_tokens,
                        skip_special_tokens=True,
                    )
                    for result in results
                ]
                reward_details = score_vsi_qa_group(
                    generated_texts,
                    rollout_item["reward_model"]["ground_truth"],
                    rollout_item["extra_info"],
                    history_path=self.args.reward_history_path,
                )
                rewards = [
                    float(details["score"]) for details in reward_details
                ]
                advantages = self.compute_group_advantages(rewards)
                prompt_ids = rollout_item["prepared_media"][
                    "prompt_token_ids"
                ].tolist()
                for candidate_index, (
                    result,
                    generated_text,
                    details,
                    reward,
                    advantage,
                ) in enumerate(
                    zip(
                        results,
                        generated_texts,
                        reward_details,
                        rewards,
                        advantages,
                    )
                ):
                    response_ids = list(result.output_tokens)
                    sample = VSIQARLSample(
                        prompt_ids=list(prompt_ids),
                        response_ids=response_ids,
                        old_response_logprobs=list(result.output_logprobs),
                        input_ids=list(prompt_ids) + response_ids,
                        attention_mask=[1] * (len(prompt_ids) + len(response_ids)),
                        labels=[-100] * len(prompt_ids) + response_ids,
                        reward=reward,
                        advantage=advantage,
                        question=rollout_item["sample_id"],
                        ground_truth=rollout_item["reward_model"]["ground_truth"],
                        format_reward=float(details["r_format"]),
                        answer_reward=float(details["answer_exact_reward"]),
                        rollout_worker_id=self.worker_id,
                        batch_id=current_batch_id,
                        sample_id=candidate_index,
                        output_versions=list(result.output_versions),
                        stop_reason=result.stop_reason,
                        generated_text=generated_text,
                        prepared_media=rollout_item["prepared_media"],
                    )
                    group_samples.append(sample)

                self.replay_buffer.add_samples.remote(group_samples)
                print(
                    "[vsi-rollout] "
                    f"worker={self.worker_id} input_batch=1 "
                    f"samples={len(group_samples)} "
                    f"reward_mean={sum(rewards) / len(rewards):.4f}",
                    flush=True,
                )

        return {"worker_id": self.worker_id, "batches": self.batch_id}
