# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL batch and forward adapter for AcceRL's FSDPTrainWorker."""

import os
from importlib import import_module

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForImageTextToText

from accerl_agent.vllm_fsdp import (
    FSDPTrainWorker,
    get_vllm_weight_metadata,
    set_seed,
)
from vsi_qa_rlvr.dataloaders.qwen3vl_rl_collator import Qwen3VLRLDataCollator


TRAIN_ATTENTION_BACKENDS = ("flash_attention_2", "sdpa")


def require_training_attention_backend(backend):
    if backend not in TRAIN_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported training attention backend: {backend!r}. "
            f"Expected one of {TRAIN_ATTENTION_BACKENDS}."
        )
    if backend != "flash_attention_2":
        return
    try:
        import_module("flash_attn")
        import_module("flash_attn_2_cuda")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TRAIN_ATTENTION_BACKEND=flash_attention_2 requires the official "
            "Dao-AILab training package and its flash_attn_2_cuda extension. "
            "Refusing to fall back to SDPA or a Transformers kernel substitute."
        ) from error


class VSIQAFSDPTrainWorker(FSDPTrainWorker):
    """Keep AcceRL Replay polling and optimizer steps; add Qwen3-VL inputs."""

    def __init__(
        self,
        args,
        rank,
        fsdp_world_size,
        fsdp_master_addr,
        fsdp_master_port,
        replay_buffer,
    ):
        self.args = args
        self.rank = int(rank)
        self.fsdp_world_size = int(fsdp_world_size)
        self.replay_buffer = replay_buffer
        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.fsdp_world_size,
        )
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        set_seed(args.seed + self.rank)
        self.forward_dtype = torch.bfloat16
        self.data_collator = Qwen3VLRLDataCollator()

        require_training_attention_backend(args.train_attention_backend)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=self.forward_dtype,
            attn_implementation=args.train_attention_backend,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.to(self.device)
        model.train()

        named_parameters = list(model.named_parameters())
        self.vllm_is_checkpoint_format = True
        weight_metadata = get_vllm_weight_metadata(named_parameters)
        self.weight_metadata_by_scope = {
            "all": weight_metadata,
            "trainable": weight_metadata,
        }
        for layer in model.model.language_model.layers:
            fully_shard(layer)
        fully_shard(model)
        self.model = model
        sharded_parameters = dict(model.named_parameters())
        self.trainable_parameter_list = list(sharded_parameters.values())
        self.params_by_scope = {
            "all": [
                (name, sharded_parameters[name]) for name, _ in named_parameters
            ],
            "trainable": [
                (name, sharded_parameters[name]) for name, _ in named_parameters
            ],
        }
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameter_list,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.train_micro_step = 0
        self.optimizer_step = 0
        self.last_loss = 0.0
        self.last_reward_mean = 0.0
        self.last_advantage_mean = 0.0
        self.last_response_tokens = 0.0
        self.last_replay_size = 0
        self.last_total_sampled = 0
        self.last_ppo_clip_frac = 0.0
        self.transfer_port = None
        self.transfer_master_address = None
        self.model_update_group = None
        print(
            f"[rank {self.rank}] FSDP worker ready: "
            f"training_attention_backend={args.train_attention_backend}."
        )

    def _prepare_rl_sample(self, sample):
        if not sample.response_ids:
            return None
        if len(sample.input_ids) > self.args.max_length:
            return None
        if len(sample.old_response_logprobs) != len(sample.response_ids):
            return None
        return sample

    def _collate_prepared_rl_samples(self, prepared_samples, trainer_version):
        samples = [sample for sample, _ in prepared_samples]
        batch = self.data_collator(samples)
        for key, value in batch.items():
            if key == "pixel_values_videos":
                batch[key] = value.to(
                    device=self.device,
                    dtype=self.forward_dtype,
                    non_blocking=True,
                )
            else:
                batch[key] = value.to(self.device, non_blocking=True)
        advantages = torch.tensor(
            [sample.advantage for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        version_lags = [
            max(
                float(trainer_version)
                - float(max(sample.output_versions) if sample.output_versions else 0),
                0.0,
            )
            for sample in samples
        ]
        return batch, advantages, {
            "reward_mean": sum(sample.reward for sample in samples) / len(samples),
            "advantage_mean": (
                sum(sample.advantage for sample in samples) / len(samples)
            ),
            "response_tokens": float(
                sum(len(sample.response_ids) for sample in samples)
            ),
            "trainer_version_lag_mean": sum(version_lags) / len(version_lags),
        }

    def _compute_rl_loss(self, batch, advantages):
        labels = batch["labels"][:, 1:]
        response_mask = batch["loss_mask"][:, 1:]
        response_token_counts = response_mask.sum(dim=-1).clamp_min(1)
        valid_positions = response_mask.nonzero(as_tuple=False)
        if valid_positions.numel() == 0:
            raise RuntimeError("No valid response tokens found for RL loss.")
        valid_sample_indices = valid_positions[:, 0]

        with torch.autocast(device_type="cuda", dtype=self.forward_dtype):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values_videos=batch["pixel_values_videos"],
                video_grid_thw=batch["video_grid_thw"],
                mm_token_type_ids=batch["mm_token_type_ids"],
                use_cache=False,
            )
        valid_token_log_probs = self._valid_token_log_probs_from_full_logits(
            outputs.logits[:, :-1, :],
            labels,
            response_mask,
        )
        valid_advantages = advantages[valid_sample_indices].to(
            valid_token_log_probs.dtype
        )
        if self.args.clip_mode == "none":
            valid_objective = valid_advantages * valid_token_log_probs
            ratio = None
        else:
            old_logprobs = batch["old_logprobs"][:, 1:].to(
                valid_token_log_probs.dtype
            )
            ratio = torch.exp(
                valid_token_log_probs - old_logprobs[response_mask]
            )
            valid_objective = torch.minimum(
                ratio * valid_advantages,
                torch.clamp(
                    ratio,
                    1.0 - self.args.clip_eps,
                    1.0 + self.args.clip_eps,
                )
                * valid_advantages,
            )

        sample_objective = self._aggregate_valid_objective(
            valid_objective,
            valid_sample_indices,
            response_token_counts,
            batch_size=labels.shape[0],
        )
        loss_stats = {}
        if ratio is not None:
            loss_stats["ppo_clip_frac"] = float(
                (
                    (ratio < 1.0 - self.args.clip_eps)
                    | (ratio > 1.0 + self.args.clip_eps)
                )
                .float()
                .mean()
                .item()
            )
        return -sample_objective.mean(), response_token_counts, loss_stats
