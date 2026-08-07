# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL FSDP trainer following AcceRL's asynchronous Replay workflow."""

import os
import random
import socket
from importlib import import_module

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForImageTextToText

from accerl_agent.vllm_fsdp import (
    FSDPTrainWorker as AcceRLFSDPTrainWorker,
    validate_weight_scope,
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


def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def find_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_vllm_loadable_weights(name, tensor):
    """Yield checkpoint-format weights accepted by the vLLM model loader."""
    if name.endswith(".mlp.experts.gate_up_proj"):
        prefix = name.removesuffix(".gate_up_proj")
        gate_proj, up_proj = tensor.chunk(2, dim=1)
        for expert_idx in range(tensor.shape[0]):
            yield f"{prefix}.{expert_idx}.gate_proj.weight", gate_proj[expert_idx]
            yield f"{prefix}.{expert_idx}.up_proj.weight", up_proj[expert_idx]
    elif name.endswith(".mlp.experts.down_proj"):
        prefix = name.removesuffix(".down_proj")
        for expert_idx in range(tensor.shape[0]):
            yield f"{prefix}.{expert_idx}.down_proj.weight", tensor[expert_idx]
    else:
        yield name, tensor


def get_vllm_weight_metadata(named_parameters):
    names = []
    dtype_names = []
    shapes = []
    for name, parameter in named_parameters:
        for load_name, load_tensor in iter_vllm_loadable_weights(name, parameter):
            names.append(load_name)
            dtype_names.append(str(load_tensor.dtype).split(".")[-1])
            shapes.append(list(load_tensor.shape))
    return names, dtype_names, shapes


class FSDPTrainWorker(AcceRLFSDPTrainWorker):
    """One Qwen3-VL FSDP2 training worker per training GPU."""

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
        if hasattr(torch, "accelerator"):
            torch.accelerator.set_device_index(0)
        else:
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
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.to(self.device)
        model.train()

        named_parameters = list(model.named_parameters())
        all_parameter_names = [name for name, _ in named_parameters]
        trainable_parameter_names = [
            name for name, parameter in named_parameters if parameter.requires_grad
        ]
        self.vllm_is_checkpoint_format = True
        self.weight_metadata_by_scope = {
            "all": get_vllm_weight_metadata(named_parameters),
            "trainable": get_vllm_weight_metadata(
                [
                    (name, parameter)
                    for name, parameter in named_parameters
                    if parameter.requires_grad
                ]
            ),
        }

        for layer in model.model.language_model.layers:
            fully_shard(layer)
        fully_shard(model)

        self.model = model
        sharded_parameters = dict(self.model.named_parameters())
        self.params_by_scope = {
            "all": [
                (name, sharded_parameters[name]) for name in all_parameter_names
            ],
            "trainable": [
                (name, sharded_parameters[name])
                for name in trainable_parameter_names
            ],
        }
        self.trainable_parameter_list = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not self.trainable_parameter_list:
            raise RuntimeError("No trainable parameters found for Qwen3-VL.")

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
            f"training_attention_backend={args.train_attention_backend} "
            f"gradient_checkpointing={args.gradient_checkpointing}."
        )

    def _prepare_rl_sample(self, sample):
        if not sample.response_ids:
            return None
        if len(sample.input_ids) > self.args.max_length:
            return None
        if (
            self.args.clip_mode != "none"
            and len(sample.old_response_logprobs) != len(sample.response_ids)
        ):
            return None
        return sample

    def _collate_prepared_rl_samples(self, prepared_samples, trainer_version):
        samples = [sample for sample, _ in prepared_samples]
        if not samples:
            raise RuntimeError("No valid RL samples were available for training.")

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
        version_lags = []
        for sample in samples:
            sample_version = max(sample.output_versions) if sample.output_versions else 0
            version_lags.append(
                max(float(trainer_version) - float(sample_version), 0.0)
            )
        stats = {
            "sample_count": float(len(samples)),
            "reward_mean": sum(sample.reward for sample in samples) / len(samples),
            "advantage_mean": (
                sum(sample.advantage for sample in samples) / len(samples)
            ),
            "response_tokens": float(
                sum(len(sample.response_ids) for sample in samples)
            ),
            "trainer_version_lag_mean": (
                sum(version_lags) / len(version_lags) if version_lags else 0.0
            ),
        }
        return batch, advantages, stats

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
            sample_objective = self._aggregate_valid_objective(
                valid_objective,
                valid_sample_indices,
                response_token_counts,
                batch_size=labels.shape[0],
            )
            return -sample_objective.mean(), response_token_counts, {}

        old_logprobs = batch["old_logprobs"][:, 1:].to(
            valid_token_log_probs.dtype
        )
        valid_ratio = torch.exp(
            valid_token_log_probs - old_logprobs[response_mask]
        )
        if self.args.clip_mode == "ppo":
            surrogate_1 = valid_ratio * valid_advantages
            surrogate_2 = torch.clamp(
                valid_ratio,
                1.0 - self.args.clip_eps,
                1.0 + self.args.clip_eps,
            ) * valid_advantages
            valid_objective = torch.minimum(surrogate_1, surrogate_2)
        elif self.args.clip_mode == "gipo":
            detached_ratio = valid_ratio.clamp_min(1e-9).detach()
            coefficient = torch.exp(
                -0.5
                * (torch.log(detached_ratio) / self.args.gipo_sigma) ** 2
            )
            valid_objective = valid_ratio * valid_advantages * coefficient
        elif self.args.clip_mode == "sapo":
            ratio = valid_ratio.clamp(1e-6, 1e6)
            tau = torch.where(
                valid_advantages > 0,
                torch.full_like(valid_advantages, self.args.sapo_tau_pos),
                torch.full_like(valid_advantages, self.args.sapo_tau_neg),
            )
            gate = torch.sigmoid(tau * (ratio - 1.0)) * (4.0 / tau)
            valid_objective = gate * valid_advantages
        else:
            raise ValueError(f"Unsupported clip_mode: {self.args.clip_mode}")

        sample_objective = self._aggregate_valid_objective(
            valid_objective,
            valid_sample_indices,
            response_token_counts,
            batch_size=labels.shape[0],
        )
        loss = -sample_objective.mean()
        loss_stats = {}
        if self.args.clip_mode == "ppo" and valid_ratio.numel() > 0:
            clipped_mask = (
                (valid_ratio < 1.0 - self.args.clip_eps)
                | (valid_ratio > 1.0 + self.args.clip_eps)
            )
            loss_stats["ppo_clip_frac"] = float(
                clipped_mask.float().mean().item()
            )
        return loss, response_token_counts, loss_stats

__all__ = [
    "TRAIN_ATTENTION_BACKENDS",
    "FSDPTrainWorker",
    "find_open_port",
    "get_local_ip",
    "get_vllm_weight_metadata",
    "iter_vllm_loadable_weights",
    "require_training_attention_backend",
    "set_seed",
    "validate_weight_scope",
]
