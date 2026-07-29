# SPDX-License-Identifier: Apache-2.0
"""Batch Qwen3-VL trajectories for the reinforcement-learning path.

Dataset reading and Qwen3-VL processor calls do not belong here. This collator
only pads AcceRL Replay samples and preserves behavior-policy log-probabilities.
"""

import torch


class Qwen3VLRLDataCollator:
    """Pad RL trajectories and preserve behavior-policy log-probabilities."""

    def __call__(self, samples):
        max_length = max(len(sample.input_ids) for sample in samples)
        pad_token_id = int(samples[0].prepared_media["pad_token_id"])
        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        old_logprobs = torch.zeros(
            (batch_size, max_length),
            dtype=torch.float32,
        )
        mm_token_type_ids = torch.zeros_like(input_ids)

        for index, sample in enumerate(samples):
            offset = max_length - len(sample.input_ids)
            input_ids[index, offset:] = torch.tensor(
                sample.input_ids,
                dtype=torch.long,
            )
            attention_mask[index, offset:] = 1
            response_start = offset + len(sample.prompt_ids)
            response_end = response_start + len(sample.response_ids)
            loss_mask[index, response_start:response_end] = True
            old_logprobs[index, offset:] = torch.tensor(
                [0.0] * len(sample.prompt_ids)
                + sample.old_response_logprobs,
                dtype=torch.float32,
            )
            prompt_types = sample.prepared_media[
                "prompt_mm_token_type_ids"
            ]
            response_types = torch.zeros(
                len(sample.response_ids),
                dtype=torch.long,
            )
            mm_token_type_ids[index, offset:] = torch.cat(
                (prompt_types, response_types)
            )

        labels = input_ids.masked_fill(~loss_mask, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "old_logprobs": old_logprobs,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values_videos": torch.cat(
                [
                    sample.prepared_media["pixel_values_videos"]
                    for sample in samples
                ],
                dim=0,
            ),
            "video_grid_thw": torch.cat(
                [
                    sample.prepared_media["video_grid_thw"]
                    for sample in samples
                ],
                dim=0,
            ),
        }


__all__ = ["Qwen3VLRLDataCollator"]
