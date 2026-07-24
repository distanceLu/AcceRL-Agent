from __future__ import annotations

import argparse
import asyncio
import glob
import inspect
import json
import math
import os
import random
import shlex
import socket
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Tuple

import ray
import textworld
import textworld.gym
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.v1.executor import Executor

def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class EncodedExample:
    def __init__(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        labels: List[int],
        old_logprobs: List[float],
        sample_reward: float,
        sample_advantage: float,
        token_rewards: List[float],
        token_advantages: List[float],
        response_indices: List[int],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.old_logprobs = old_logprobs
        self.sample_reward = float(sample_reward)
        self.sample_advantage = float(sample_advantage)
        self.token_rewards = token_rewards
        self.token_advantages = token_advantages
        self.response_indices = response_indices


@dataclass
class PreparedVarlenPack:
    """One CPU-resident pack prepared for a Varlen optimizer window."""

    batch: Dict[str, torch.Tensor]
    train_stats: Dict[str, float]
    replay_stats: Dict[str, int]
    valid_token_count: int
    max_seqlen: int


VARLEN_TOKEN_STAT_NAMES = (
    "policy_token_sum",
    "old_new_kl_k3_sum",
    "token_reward_sum",
    "raw_advantage_sum",
    "raw_advantage_sum_sq",
    "used_advantage_sum",
    "used_advantage_sum_sq",
    "ppo_clip_count",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def make_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(examples: List[EncodedExample]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example.input_ids) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []
        old_logprobs = []
        sample_rewards = []
        sample_advantages = []
        token_rewards = []
        token_advantages = []
        response_indices = []

        for example in examples:
            pad_len = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [pad_token_id] * pad_len)
            attention_mask.append(example.attention_mask + [0] * pad_len)
            labels.append(example.labels + [-100] * pad_len)
            old_logprobs.append(example.old_logprobs + [0.0] * pad_len)
            sample_rewards.append(example.sample_reward)
            sample_advantages.append(example.sample_advantage)
            token_rewards.append(example.token_rewards + [0.0] * pad_len)
            token_advantages.append(example.token_advantages + [0.0] * pad_len)
            response_indices.append(example.response_indices + [-1] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
            "sample_rewards": torch.tensor(sample_rewards, dtype=torch.float32),
            "sample_advantages": torch.tensor(sample_advantages, dtype=torch.float32),
            "token_rewards": torch.tensor(token_rewards, dtype=torch.float32),
            "token_advantages": torch.tensor(token_advantages, dtype=torch.float32),
            "response_indices": torch.tensor(response_indices, dtype=torch.long),
        }

    return collate


def make_varlen_batch(examples: List[EncodedExample]) -> Dict[str, torch.Tensor]:
    """Flatten examples while retaining their causal and RL sample boundaries."""
    if not examples:
        raise ValueError("At least one example is required for varlen packing.")

    input_ids = []
    labels = []
    old_logprobs = []
    token_rewards = []
    token_advantages = []
    response_indices = []
    position_ids = []
    sequence_ids = []
    cu_seqlens = [0]

    for sequence_id, example in enumerate(examples):
        length = len(example.input_ids)
        if length < 2:
            raise ValueError("Each packed sequence must contain at least two tokens.")
        if example.labels[0] != -100:
            raise ValueError("The first token in a packed sequence cannot be an RL target.")

        input_ids.extend(example.input_ids)
        labels.extend(example.labels)
        old_logprobs.extend(example.old_logprobs)
        token_rewards.extend(example.token_rewards)
        token_advantages.extend(example.token_advantages)
        response_indices.extend(example.response_indices)
        position_ids.extend(range(length))
        sequence_ids.extend([sequence_id] * length)
        cu_seqlens.append(cu_seqlens[-1] + length)

    target_indices = [
        index for index, label in enumerate(labels) if label != -100
    ]
    if not target_indices:
        raise ValueError("A varlen batch must contain at least one RL target.")
    prediction_indices = [index - 1 for index in target_indices]
    for target_index, prediction_index in zip(
        target_indices,
        prediction_indices,
    ):
        if sequence_ids[target_index] != sequence_ids[prediction_index]:
            raise ValueError("A packed target cannot cross a sequence boundary.")
        if position_ids[target_index] != position_ids[prediction_index] + 1:
            raise ValueError(
                "A packed target must immediately follow its prediction position."
            )

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
        "sample_rewards": torch.tensor(
            [example.sample_reward for example in examples], dtype=torch.float32
        ),
        "sample_advantages": torch.tensor(
            [example.sample_advantage for example in examples], dtype=torch.float32
        ),
        "token_rewards": torch.tensor(token_rewards, dtype=torch.float32),
        "token_advantages": torch.tensor(token_advantages, dtype=torch.float32),
        "response_indices": torch.tensor(response_indices, dtype=torch.long),
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "prediction_indices": torch.tensor(prediction_indices, dtype=torch.long),
    }


def select_varlen_pack(
    prepared_samples: List[Tuple[Any, EncodedExample]],
    token_budget: int,
    max_sequences: int,
) -> Tuple[List[Tuple[Any, EncodedExample]], List[Tuple[Any, EncodedExample]]]:
    """First-fit a random replay candidate pool after a local length sort."""
    ordered = sorted(
        enumerate(prepared_samples),
        key=lambda item: len(item[1][1].input_ids),
        reverse=True,
    )
    selected_indices = []
    selected = []
    total_tokens = 0
    for original_index, prepared in ordered:
        length = len(prepared[1].input_ids)
        if len(selected) >= max_sequences:
            break
        if total_tokens + length > token_budget:
            continue
        selected_indices.append(original_index)
        selected.append(prepared)
        total_tokens += length

    selected_index_set = set(selected_indices)
    remaining = [
        prepared
        for index, prepared in enumerate(prepared_samples)
        if index not in selected_index_set
    ]
    return selected, remaining


def configure_trainable_parameters(model, train_mode: str) -> None:
    if train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    if train_mode == "lm_head":
        target_keywords = ("lm_head",)
    elif train_mode == "last_layer":
        num_layers = len(getattr(model.model, "layers"))
        target_keywords = (f"model.layers.{num_layers - 1}.", "lm_head")
    else:
        raise ValueError(f"Unsupported train mode: {train_mode}")

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in target_keywords):
            param.requires_grad = True


def iter_trainable_parameters(model) -> Iterable:
    return (param for param in model.parameters() if param.requires_grad)


def count_parameters(model) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return total, trainable


def log_parameter_count(model, train_mode: str, rank: int = 0):
    total_params, trainable_params = count_parameters(model)
    trainable_parameter_list = list(iter_trainable_parameters(model))
    if not trainable_parameter_list:
        raise RuntimeError(f"No trainable parameters found for mode: {train_mode}")

    if rank == 0:
        print(
            "[train] Parameter count: "
            f"trainable={trainable_params:,} / total={total_params:,} "
            f"({trainable_params / total_params:.4%})"
        )
    return trainable_parameter_list


def move_batch_to_device(batch: Dict, device) -> Dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_tokenizer(args: argparse.Namespace, log: bool = True):
    if log:
        print(f"[init] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token or eos_token.")
    return tokenizer


def build_model(args: argparse.Namespace, device, torch_dtype, log: bool = True):
    if log:
        print(
            f"[init] Loading model from {args.model_path} "
            f"(device={device}, dtype={torch_dtype})"
        )
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.train_packing == "varlen":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.train_packing == "varlen":
        attention_implementation = getattr(
            model.config,
            "_attn_implementation",
            None,
        )
        if attention_implementation != "flash_attention_2":
            raise RuntimeError(
                "Varlen training requires the loaded model to use "
                "flash_attention_2; got "
                f"{attention_implementation!r}."
            )
    model.to(device)
    model.train()
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def iter_vllm_loadable_weights(name: str, tensor: torch.Tensor):
    """Yield checkpoint-style weights accepted by vLLM's Qwen2-MoE loader.

    Recent Transformers stores routed expert weights as fused 3D parameters:
    ``experts.gate_up_proj`` and ``experts.down_proj``. vLLM's checkpoint
    loader expects the original per-expert HF names and performs its own
    loading into FusedMoE kernel parameters, so we split them before transfer.
    """
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
    """Return names, dtypes, and shapes matching iter_vllm_loadable_weights."""
    names = []
    dtype_names = []
    shapes = []
    for name, param in named_parameters:
        for load_name, load_tensor in iter_vllm_loadable_weights(name, param):
            names.append(load_name)
            dtype_names.append(str(load_tensor.dtype).split(".")[-1])
            shapes.append(list(load_tensor.shape))
    return names, dtype_names, shapes


def validate_weight_scope(scope: str) -> None:
    if scope not in {"all", "trainable"}:
        raise ValueError(f"Unsupported weight scope: {scope!r}")


def dtype_nbytes(dtype_name: str) -> int:
    """Return bytes per element for dtype names emitted by get_vllm_weight_metadata."""
    return {
        "float64": 8,
        "double": 8,
        "float32": 4,
        "float": 4,
        "bfloat16": 2,
        "float16": 2,
        "half": 2,
        "int64": 8,
        "long": 8,
        "int32": 4,
        "int": 4,
        "int16": 2,
        "short": 2,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }[dtype_name]


def numel_from_shape(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


class FSDPTrainWorker:
    """
    One FSDP2 training worker per GPU.  Four of these form the FSDP group.
    Rank 0 additionally handles weight transfer to the vLLM engine.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
        replay_buffer,
    ):
        self.args = args
        self.rank = rank
        self.fsdp_world_size = fsdp_world_size
        self.replay_buffer = replay_buffer

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        if hasattr(torch, "accelerator"):
            torch.accelerator.set_device_index(0)
        else:
            torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")

        set_seed(args.seed + rank)

        self.tokenizer = build_tokenizer(args, log=rank == 0)
        self.collate_fn = make_collate_fn(self.tokenizer)
        torch_dtype = pick_dtype(args.dtype)
        model = build_model(args, self.device, torch_dtype, log=rank == 0)
        configure_trainable_parameters(model, args.train_mode)
        log_parameter_count(model, args.train_mode, rank=rank)

        named_parameters = list(model.named_parameters())
        all_param_names = [name for name, _ in named_parameters]
        trainable_param_names = [
            name for name, param in named_parameters if param.requires_grad
        ]
        self.weight_metadata_by_scope = {
            "all": get_vllm_weight_metadata(named_parameters),
            "trainable": get_vllm_weight_metadata(
                [
                    (name, param)
                    for name, param in named_parameters
                    if param.requires_grad
                ]
            ),
        }

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        self.model = model
        fsdp_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "set_gradient_divide_factor")
        ]
        if not fsdp_modules:
            raise RuntimeError(
                "The pinned FSDP2 runtime must expose "
                "set_gradient_divide_factor()."
            )
        for module in fsdp_modules:
            module.set_gradient_divide_factor(float(self.fsdp_world_size))
        if self.rank == 0:
            print(
                "[train] FSDP gradient divide factor fixed at "
                f"{self.fsdp_world_size}; Varlen loss pre-scale uses "
                "world_size / global_valid_token_count."
            )
        sharded_params_by_name = dict(self.model.named_parameters())
        self.params_by_scope = {
            "all": [
                (name, sharded_params_by_name[name])
                for name in all_param_names
            ],
            "trainable": [
                (name, sharded_params_by_name[name])
                for name in trainable_param_names
            ],
        }
        self.trainable_parameter_list = list(iter_trainable_parameters(self.model))
        if not self.trainable_parameter_list:
            raise RuntimeError(f"No trainable parameters found for mode: {args.train_mode}")

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
        self.last_token_reward_mean = 0.0
        self.last_raw_advantage_mean = 0.0
        self.last_raw_advantage_std = 0.0
        self.last_used_advantage_mean = 0.0
        self.last_used_advantage_std = 0.0
        self.last_old_new_kl_k3_token_mean = 0.0
        self.last_old_new_kl_k3_sample_sum_mean = 0.0
        self.last_old_new_kl_k3_loss = 0.0
        self.last_pack_token_utilization = 0.0
        self.last_pack_sample_count = 0.0
        self.last_pack_max_sequence_length = 0.0
        self.last_pack_cpu_seconds = 0.0
        self.last_global_training_samples = 0
        self.cumulative_global_training_samples = 0
        self.cumulative_global_valid_response_tokens = 0
        self.total_replay_candidates_fetched = 0
        self.total_valid_samples_prepared = 0
        self.total_training_samples_consumed = 0
        self.total_invalid_candidates = 0
        self.pending_prepared_samples: List[Tuple["RLSample", EncodedExample]] = []

        self.transfer_port = None
        self.transfer_master_address = None
        self.model_update_group = None
        print(f"[rank {rank}] FSDP worker ready.")

    def get_rank(self):
        return self.rank

    def get_replay_stats(self):
        return ray.get(self.replay_buffer.get_stats.remote())

    def _get_current_lr(
        self,
        current_step: int,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        start_step: int = 0,
    ) -> float:
        if current_step < start_step:
            return 0.0

        effective_step = current_step - start_step
        if warmup_steps > 0 and effective_step < warmup_steps:
            return peak_lr * (effective_step / warmup_steps)

        decay_steps = total_steps - start_step - warmup_steps
        if decay_steps <= 0:
            return peak_lr

        progress = (effective_step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * cosine_decay

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _prepare_rl_sample(self, sample: "RLSample") -> EncodedExample | None:
        if sample.algorithm != self.args.rl_algorithm:
            return None
        input_ids = list(sample.input_ids)
        labels = list(sample.labels)
        attention_mask = list(sample.attention_mask)
        if (
            not input_ids
            or len(input_ids) != len(labels)
            or len(input_ids) != len(attention_mask)
        ):
            return None
        old_logprobs = list(sample.old_logprobs)
        token_rewards = list(sample.token_rewards)
        token_advantages = list(sample.token_advantages)
        response_indices = list(sample.response_indices)
        if (
            len(old_logprobs) != len(input_ids)
            or len(token_rewards) != len(input_ids)
            or len(token_advantages) != len(input_ids)
            or len(response_indices) != len(input_ids)
        ):
            return None
        response_old_logprobs = [
            old_logprob
            for old_logprob, label in zip(old_logprobs, labels)
            if label != -100
        ]
        if len(response_old_logprobs) != len(sample.response_ids):
            return None

        max_length = self.args.max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            labels = labels[-max_length:]
            old_logprobs = old_logprobs[-max_length:]
            token_rewards = token_rewards[-max_length:]
            token_advantages = token_advantages[-max_length:]
            response_indices = response_indices[-max_length:]

        # A left-truncated first token has no in-window predecessor, so it
        # cannot be a causal LM target. The padded path already ignored it via
        # labels[:, 1:]; make that boundary explicit for flattened batches.
        labels[0] = -100
        old_logprobs[0] = 0.0
        token_rewards[0] = 0.0
        token_advantages[0] = 0.0
        response_indices[0] = -1

        if len(input_ids) < 2:
            return None
        if all(label == -100 for label in labels[1:]):
            return None
        if any(
            response_index < 0
            for response_index, label in zip(response_indices[1:], labels[1:])
            if label != -100
        ):
            return None

        return EncodedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            old_logprobs=old_logprobs,
            sample_reward=sample.reward,
            sample_advantage=sample.advantage,
            token_rewards=token_rewards,
            token_advantages=token_advantages,
            response_indices=response_indices,
        )

    def _collate_prepared_rl_samples(
        self,
        prepared_samples: List[Tuple["RLSample", EncodedExample]],
        trainer_version: float,
        *,
        move_to_device: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        kept_samples = [sample for sample, _ in prepared_samples]
        examples = [example for _, example in prepared_samples]

        if not examples:
            raise RuntimeError("No valid RL samples were available for training.")

        if self.args.train_packing == "varlen":
            batch = make_varlen_batch(examples)
        else:
            batch = self.collate_fn(examples)
        if move_to_device:
            batch = move_batch_to_device(batch, self.device)
        response_token_counts = [
            sum(1 for label in example.labels[1:] if label != -100)
            for example in examples
        ]
        version_lags = []
        for sample in kept_samples:
            sample_version = max(sample.output_versions) if sample.output_versions else 0
            version_lags.append(max(float(trainer_version) - float(sample_version), 0.0))

        sample_rewards = [float(sample.reward) for sample in kept_samples]
        sample_advantages = [float(sample.advantage) for sample in kept_samples]
        sample_advantage_mean = (
            sum(sample_advantages) / len(sample_advantages)
            if sample_advantages else 0.0
        )
        sample_advantage_std = (
            math.sqrt(
                sum(
                    (advantage - sample_advantage_mean) ** 2
                    for advantage in sample_advantages
                )
                / len(sample_advantages)
            )
            if sample_advantages else 0.0
        )

        stats = {
            "sample_count": float(len(kept_samples)),
            "reward_sum": float(sum(sample_rewards)),
            "episode_return_sum": float(
                sum(sample.episode_return for sample in kept_samples)
            ),
            "reward_mean": (
                sum(sample_rewards) / len(sample_rewards)
                if sample_rewards else 0.0
            ),
            "episode_return_mean": (
                sum(sample.episode_return for sample in kept_samples)
                / len(kept_samples)
            ),
            "advantage_mean": sample_advantage_mean,
            "advantage_std": sample_advantage_std,
            "response_tokens": float(sum(response_token_counts)),
            "trainer_version_lag_mean": (
                sum(version_lags) / len(version_lags) if version_lags else 0.0
            ),
            "trainer_version_lag_sum": float(sum(version_lags)),
            "pack_tokens": float(sum(len(example.input_ids) for example in examples)),
            "pack_token_utilization": (
                sum(len(example.input_ids) for example in examples)
                / self.args.train_token_budget
                if self.args.train_packing == "varlen"
                else 0.0
            ),
            "pack_sample_count": float(len(examples)),
            "pack_max_sequence_length": float(
                max(len(example.input_ids) for example in examples)
            ),
        }
        if self.args.rl_algorithm == "ppo":
            valid_token_rewards = []
            valid_token_advantages = []
            for example in examples:
                for label, reward, advantage in zip(
                    example.labels[1:],
                    example.token_rewards[1:],
                    example.token_advantages[1:],
                ):
                    if label != -100:
                        valid_token_rewards.append(float(reward))
                        valid_token_advantages.append(float(advantage))

            token_reward_mean = (
                sum(valid_token_rewards) / len(valid_token_rewards)
                if valid_token_rewards else 0.0
            )
            token_advantage_mean = (
                sum(valid_token_advantages) / len(valid_token_advantages)
                if valid_token_advantages else 0.0
            )
            token_advantage_std = (
                math.sqrt(
                    sum(
                        (advantage - token_advantage_mean) ** 2
                        for advantage in valid_token_advantages
                    )
                    / len(valid_token_advantages)
                )
                if valid_token_advantages else 0.0
            )
            stats.update(
                {
                    "reward_mean": stats["episode_return_mean"],
                    "token_reward_mean": token_reward_mean,
                    "advantage_mean": token_advantage_mean,
                    "advantage_std": token_advantage_std,
                    "raw_advantage_mean": token_advantage_mean,
                    "raw_advantage_std": token_advantage_std,
                    "token_reward_sum": float(sum(valid_token_rewards)),
                    "raw_advantage_sum": float(sum(valid_token_advantages)),
                    "raw_advantage_sum_sq": float(
                        sum(value * value for value in valid_token_advantages)
                    ),
                    "advantage_token_count": float(len(valid_token_advantages)),
                }
            )
        else:
            stats.update(
                {
                    "token_reward_mean": 0.0,
                    "token_reward_sum": 0.0,
                    "raw_advantage_mean": sample_advantage_mean,
                    "raw_advantage_std": sample_advantage_std,
                    "raw_advantage_sum": float(sum(sample_advantages)),
                    "raw_advantage_sum_sq": float(
                        sum(value * value for value in sample_advantages)
                    ),
                    "advantage_token_count": float(len(sample_advantages)),
                }
            )
        return batch, stats

    def _select_varlen_pack(
        self,
    ) -> List[Tuple["RLSample", EncodedExample]]:
        """Select a length-aware pack and retain non-selected candidates."""
        if not self.pending_prepared_samples:
            return []

        selected, remaining = select_varlen_pack(
            self.pending_prepared_samples,
            token_budget=self.args.train_token_budget,
            max_sequences=self.args.batch_size,
        )

        if not selected:
            longest = max(
                len(example.input_ids)
                for _, example in self.pending_prepared_samples
            )
            raise RuntimeError(
                "No replay sample fits in --train-token-budget: "
                f"longest_pending={longest} budget={self.args.train_token_budget}"
            )

        self.pending_prepared_samples = remaining
        return selected

    def _next_rl_training_batch(
        self,
        trainer_version: float,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, int]]:
        collected = []
        replay_stats = self.get_replay_stats()
        warmup_deadline = None
        if self.args.replay_sample_timeout_seconds > 0:
            warmup_deadline = time.monotonic() + self.args.replay_sample_timeout_seconds
        while replay_stats["size"] < self.args.min_replay_size_per_rank:
            if warmup_deadline is not None and time.monotonic() >= warmup_deadline:
                raise TimeoutError(
                    "Timed out waiting for replay warmup: "
                    f"rank={self.rank} "
                    f"size={replay_stats['size']} "
                    f"min_replay_size_per_rank={self.args.min_replay_size_per_rank} "
                    f"stats={replay_stats}"
                )
            time.sleep(self.args.replay_wait_sleep_seconds)
            replay_stats = self.get_replay_stats()

        if self.args.train_packing == "varlen":
            prepared = self._next_varlen_cpu_pack(
                trainer_version,
                replay_stats=replay_stats,
            )
            return (
                move_batch_to_device(prepared.batch, self.device),
                prepared.train_stats,
                prepared.replay_stats,
            )

        while len(collected) < self.args.batch_size:
            need = self.args.batch_size - len(collected)
            deadline = None
            if self.args.replay_sample_timeout_seconds > 0:
                deadline = time.monotonic() + self.args.replay_sample_timeout_seconds

            sampled = []
            while not sampled:
                sampled = ray.get(self.replay_buffer.sample.remote(need))
                replay_stats = self.get_replay_stats()
                if sampled:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for replay samples: "
                        f"rank={self.rank} have={len(collected)} "
                        f"need={self.args.batch_size} stats={replay_stats}"
                    )
                time.sleep(self.args.replay_wait_sleep_seconds)

            for sample in sampled:
                example = self._prepare_rl_sample(sample)
                if example is not None:
                    collected.append((sample, example))
                    if len(collected) >= self.args.batch_size:
                        break

        batch, train_stats = self._collate_prepared_rl_samples(
            collected[: self.args.batch_size],
            trainer_version,
        )
        train_stats["pack_cpu_seconds"] = 0.0
        return batch, train_stats, replay_stats

    def _next_varlen_cpu_pack(
        self,
        trainer_version: float,
        *,
        replay_stats: Dict[str, int] | None = None,
    ) -> PreparedVarlenPack:
        """Prepare one Varlen pack without moving any tensor to the GPU."""
        if replay_stats is None:
            replay_stats = self.get_replay_stats()
            warmup_deadline = None
            if self.args.replay_sample_timeout_seconds > 0:
                warmup_deadline = (
                    time.monotonic() + self.args.replay_sample_timeout_seconds
                )
            while replay_stats["size"] < self.args.min_replay_size_per_rank:
                if (
                    warmup_deadline is not None
                    and time.monotonic() >= warmup_deadline
                ):
                    raise TimeoutError(
                        "Timed out waiting for replay warmup: "
                        f"rank={self.rank} size={replay_stats['size']} "
                        "min_replay_size_per_rank="
                        f"{self.args.min_replay_size_per_rank} "
                        f"stats={replay_stats}"
                    )
                time.sleep(self.args.replay_wait_sleep_seconds)
                replay_stats = self.get_replay_stats()

        candidate_target = self.args.train_pack_candidate_pool_size
        deadline = None
        if self.args.replay_sample_timeout_seconds > 0:
            deadline = time.monotonic() + self.args.replay_sample_timeout_seconds

        while not self.pending_prepared_samples:
            sampled = ray.get(self.replay_buffer.sample.remote(candidate_target))
            self.total_replay_candidates_fetched += len(sampled)
            replay_stats = self.get_replay_stats()
            valid_count = 0
            for sample in sampled:
                example = self._prepare_rl_sample(sample)
                if example is None:
                    self.total_invalid_candidates += 1
                    continue
                self.pending_prepared_samples.append((sample, example))
                valid_count += 1
            self.total_valid_samples_prepared += valid_count
            if self.pending_prepared_samples:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for valid replay candidates: "
                    f"rank={self.rank} target={candidate_target} "
                    f"stats={replay_stats}"
                )
            time.sleep(self.args.replay_wait_sleep_seconds)

        packing_start = time.perf_counter()
        collected = self._select_varlen_pack()
        batch, train_stats = self._collate_prepared_rl_samples(
            collected,
            trainer_version,
            move_to_device=False,
        )
        train_stats["pack_cpu_seconds"] = time.perf_counter() - packing_start
        valid_token_count = int(batch["target_indices"].numel())
        if valid_token_count <= 0:
            raise RuntimeError("A Varlen pack must contain at least one target.")
        max_seqlen = max(len(example.input_ids) for _, example in collected)
        return PreparedVarlenPack(
            batch=batch,
            train_stats=train_stats,
            replay_stats=replay_stats,
            valid_token_count=valid_token_count,
            max_seqlen=max_seqlen,
        )

    def _compute_rl_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        varlen_max_seqlen: int | None = None,
        return_varlen_token_sums: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, float | torch.Tensor],
    ]:
        is_varlen = self.args.train_packing == "varlen"
        batch_size = int(batch["sample_rewards"].shape[0])
        if is_varlen:
            if return_varlen_token_sums and varlen_max_seqlen is None:
                raise ValueError(
                    "Varlen token-sum training requires an explicit "
                    "max sequence length and cumulative sequence boundaries."
                )
            target_indices = batch["target_indices"]
            prediction_indices = batch["prediction_indices"]
            if target_indices.numel() == 0:
                raise RuntimeError("No valid response tokens found for RL loss.")
            valid_sample_indices = batch["sequence_ids"][target_indices]
            response_token_counts = torch.bincount(
                valid_sample_indices,
                minlength=batch_size,
            ).clamp_min(1)
            valid_labels = batch["labels"][target_indices]
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "position_ids": batch["position_ids"],
                "attention_mask": None,
                "use_cache": False,
            }
            if varlen_max_seqlen is not None:
                model_kwargs.update(
                    {
                        "cu_seq_lens_q": batch["cu_seqlens"],
                        "cu_seq_lens_k": batch["cu_seqlens"],
                        "max_length_q": int(varlen_max_seqlen),
                        "max_length_k": int(varlen_max_seqlen),
                    }
                )
            model_type = str(getattr(self.model.config, "model_type", ""))
            if "moe" in model_type or hasattr(self.model.config, "num_experts"):
                model_kwargs["output_router_logits"] = False

            if self.args.train_logprob_mode == "full_logits_ce":
                outputs = self.model(**model_kwargs)
                valid_logits = outputs.logits[0, prediction_indices, :]
            elif self.args.train_logprob_mode == "response_only_lm_head":
                outputs = self.model(
                    **model_kwargs,
                    logits_to_keep=prediction_indices,
                )
                valid_logits = outputs.logits.squeeze(0)
            else:
                raise ValueError(
                    f"Unsupported train_logprob_mode: {self.args.train_logprob_mode}"
                )
            valid_token_log_probs = -F.cross_entropy(
                valid_logits,
                valid_labels,
                reduction="none",
            )
            valid_old_token_log_probs = batch["old_logprobs"][target_indices].to(
                torch.float32
            )
        else:
            labels = batch["labels"][:, 1:]
            response_mask = labels.ne(-100)
            response_token_counts = response_mask.sum(dim=-1).clamp_min(1)
            valid_positions = response_mask.nonzero(as_tuple=False)
            if valid_positions.numel() == 0:
                raise RuntimeError("No valid response tokens found for RL loss.")
            valid_sample_indices = valid_positions[:, 0]
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits[:, :-1, :]
            valid_token_log_probs = self._valid_token_log_probs_from_full_logits(
                logits,
                labels,
                response_mask,
            )
            old_token_log_probs = batch["old_logprobs"][:, 1:].to(
                torch.float32
            )
            valid_old_token_log_probs = old_token_log_probs[response_mask]

        # Ratio-based RL objectives are numerically sensitive. Keep the
        # subtraction, exponentiation, clipping/gating, and KL construction in
        # FP32 even when the model forward and logits use BF16/FP16.
        valid_token_log_probs = valid_token_log_probs.float()
        valid_old_token_log_probs = valid_old_token_log_probs.float()
        valid_log_ratio = valid_token_log_probs - valid_old_token_log_probs
        valid_ratio = torch.exp(valid_log_ratio)

        valid_response_indices = None
        valid_token_rewards = None
        if self.args.rl_algorithm == "ppo":
            if is_varlen:
                valid_response_indices = batch["response_indices"][target_indices]
                raw_valid_adv = batch["token_advantages"][target_indices].to(
                    torch.float32
                )
                valid_token_rewards = batch["token_rewards"][target_indices].to(
                    torch.float32
                )
            else:
                response_indices = batch["response_indices"][:, 1:]
                valid_response_indices = response_indices[response_mask]
                raw_valid_adv = batch["token_advantages"][:, 1:][response_mask].to(
                    torch.float32
                )
                valid_token_rewards = batch["token_rewards"][:, 1:][response_mask].to(
                    torch.float32
                )
            if valid_response_indices.lt(0).any():
                raise RuntimeError(
                    "Valid response tokens must have non-negative response indices."
                )
            if self.args.ppo_normalize_advantages:
                adv_mean = raw_valid_adv.mean()
                adv_std = raw_valid_adv.std(unbiased=False)
                normalized_valid_adv = (
                    (raw_valid_adv - adv_mean)
                    / (adv_std + self.args.ppo_adv_norm_eps)
                )
                valid_adv = normalized_valid_adv
            else:
                valid_adv = raw_valid_adv
        elif self.args.rl_algorithm == "grpo":
            raw_valid_adv = batch["sample_advantages"][valid_sample_indices].to(
                torch.float32
            )
            valid_adv = raw_valid_adv
        else:
            raise ValueError(f"Unsupported rl_algorithm: {self.args.rl_algorithm}")

        if self.args.clip_mode == "ppo":
            surr1 = valid_ratio * valid_adv
            surr2 = torch.clamp(
                valid_ratio,
                1.0 - self.args.clip_eps,
                1.0 + self.args.clip_eps,
            ) * valid_adv
            valid_objective = torch.minimum(surr1, surr2)
        elif self.args.clip_mode == "gipo":
            r_detach = valid_ratio.clamp_min(1e-9).detach()
            coeff = torch.exp(
                -0.5 * (torch.log(r_detach) / self.args.gipo_sigma) ** 2
            )
            valid_objective = valid_ratio * valid_adv * coeff
        elif self.args.clip_mode == "sapo":
            r = valid_ratio.clamp(1e-6, 1e6)
            tau_pos = torch.full_like(valid_adv, self.args.sapo_tau_pos)
            tau_neg = torch.full_like(valid_adv, self.args.sapo_tau_neg)
            tau = torch.where(valid_adv > 0, tau_pos, tau_neg)
            gate = torch.sigmoid(tau * (r - 1.0)) * (4.0 / tau)
            valid_objective = gate * valid_adv
        else:
            raise ValueError(f"Unsupported clip_mode: {self.args.clip_mode}")

        objective_for_diagnostics = (
            valid_objective.detach()
            if is_varlen and return_varlen_token_sums
            else valid_objective
        )
        if self.args.rl_algorithm == "ppo":
            assert valid_response_indices is not None
            sample_objective = self._aggregate_valid_objective_by_response(
                objective_for_diagnostics,
                valid_sample_indices,
                valid_response_indices,
                batch_size=batch_size,
            )
        else:
            sample_objective = self._aggregate_valid_objective(
                objective_for_diagnostics,
                valid_sample_indices,
                response_token_counts,
                batch_size=batch_size,
            )
        old_new_kl_k3 = valid_ratio - 1.0 - valid_log_ratio
        policy_token_sum = -valid_objective.float().sum()
        old_new_kl_k3_sum = old_new_kl_k3.float().sum()
        valid_token_count = int(valid_objective.numel())
        if is_varlen and return_varlen_token_sums:
            # Varlen intentionally optimizes a global valid-response-token
            # mean. It does not preserve the padded path's per-response or
            # per-episode weighting, so longer responses carry more weight.
            policy_loss = policy_token_sum / valid_token_count
            old_new_kl_k3_token_mean = old_new_kl_k3_sum / valid_token_count
            loss = policy_token_sum + (
                self.args.old_new_kl_coef * old_new_kl_k3_sum
            )
        else:
            policy_loss = -sample_objective.mean()
            old_new_kl_k3_token_mean = old_new_kl_k3.mean()
            loss = policy_loss + (
                self.args.old_new_kl_coef * old_new_kl_k3_token_mean
            )
        old_new_kl_k3_loss = (
            self.args.old_new_kl_coef * old_new_kl_k3_token_mean
        )

        with torch.no_grad():
            if is_varlen and return_varlen_token_sums:
                if valid_token_rewards is not None:
                    token_reward_sum = valid_token_rewards.sum()
                else:
                    token_reward_sum = policy_token_sum.new_zeros(())
                if self.args.clip_mode == "ppo":
                    clipped_mask = (
                        (valid_ratio < (1.0 - self.args.clip_eps))
                        | (valid_ratio > (1.0 + self.args.clip_eps))
                    )
                    ppo_clip_count = clipped_mask.sum().float()
                else:
                    ppo_clip_count = policy_token_sum.new_zeros(())

                # Keep per-pack statistics on the accelerator. The optimizer
                # window accumulates this detached vector and transfers it to
                # Python only once after the cross-rank all-reduce.
                varlen_token_stats = torch.stack(
                    [
                        policy_token_sum,
                        old_new_kl_k3_sum,
                        token_reward_sum,
                        raw_valid_adv.sum(),
                        raw_valid_adv.square().sum(),
                        valid_adv.sum(),
                        valid_adv.square().sum(),
                        ppo_clip_count,
                    ]
                ).detach().to(dtype=torch.float64)
                return (
                    loss,
                    response_token_counts,
                    {"varlen_token_stats": varlen_token_stats},
                )

            loss_stats = {}
            if valid_ratio.numel() > 0:
                sample_kl_sum = torch.zeros(
                    batch_size,
                    device=old_new_kl_k3.device,
                    dtype=old_new_kl_k3.dtype,
                )
                sample_kl_sum.index_add_(
                    0,
                    valid_sample_indices,
                    old_new_kl_k3,
                )
                loss_stats["policy_loss"] = float(policy_loss.item())
                loss_stats["policy_token_sum"] = float(policy_token_sum.item())
                loss_stats["old_new_kl_k3_sum"] = float(
                    old_new_kl_k3_sum.item()
                )
                loss_stats["valid_token_count"] = float(valid_token_count)
                loss_stats["episode_objective_mean"] = float(
                    sample_objective.mean().item()
                )
                if valid_token_rewards is not None:
                    loss_stats["token_reward_mean"] = float(
                        valid_token_rewards.mean().item()
                    )
                    loss_stats["token_reward_sum"] = float(
                        valid_token_rewards.sum().item()
                    )
                else:
                    loss_stats["token_reward_mean"] = 0.0
                    loss_stats["token_reward_sum"] = 0.0
                loss_stats["raw_advantage_mean"] = float(raw_valid_adv.mean().item())
                loss_stats["raw_advantage_std"] = float(
                    raw_valid_adv.std(unbiased=False).item()
                )
                loss_stats["raw_advantage_sum"] = float(raw_valid_adv.sum().item())
                loss_stats["raw_advantage_sum_sq"] = float(
                    raw_valid_adv.square().sum().item()
                )
                loss_stats["used_advantage_mean"] = float(
                    valid_adv.float().mean().item()
                )
                loss_stats["used_advantage_std"] = float(
                    valid_adv.float().std(unbiased=False).item()
                )
                loss_stats["used_advantage_sum"] = float(
                    valid_adv.float().sum().item()
                )
                loss_stats["used_advantage_sum_sq"] = float(
                    valid_adv.float().square().sum().item()
                )
                loss_stats["ppo_advantages_normalized"] = float(
                    self.args.rl_algorithm == "ppo"
                    and bool(self.args.ppo_normalize_advantages)
                )
                loss_stats["old_new_kl_k3_loss"] = float(
                    old_new_kl_k3_loss.item()
                )
                loss_stats["old_new_kl_k3_token_mean"] = float(
                    old_new_kl_k3_token_mean.item()
                )
                loss_stats["old_new_kl_k3_sample_sum_mean"] = float(
                    sample_kl_sum.mean().item()
                )
                loss_stats["old_new_kl_k3_sample_sum"] = float(
                    sample_kl_sum.sum().item()
                )
            if self.args.clip_mode == "ppo" and valid_ratio.numel() > 0:
                clipped_mask = (
                    (valid_ratio < (1.0 - self.args.clip_eps))
                    | (valid_ratio > (1.0 + self.args.clip_eps))
                )
                loss_stats["ppo_clip_frac"] = float(
                    clipped_mask.float().mean().item()
                )
                loss_stats["ppo_clip_count"] = float(clipped_mask.sum().item())
            return loss, response_token_counts, loss_stats

    def _valid_token_log_probs_from_full_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_logits = logits[response_mask]
        valid_labels = labels[response_mask]
        if valid_logits.numel() == 0:
            raise RuntimeError("No valid response logits found for RL loss.")
        return -F.cross_entropy(
            valid_logits,
            valid_labels,
            reduction="none",
        )

    def _aggregate_valid_objective(
        self,
        valid_objective: torch.Tensor,
        valid_sample_indices: torch.Tensor,
        response_token_counts: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        sample_objective_sum = torch.zeros(
            batch_size,
            device=valid_objective.device,
            dtype=valid_objective.dtype,
        )
        sample_objective_sum.index_add_(
            0,
            valid_sample_indices,
            valid_objective,
        )
        return sample_objective_sum / response_token_counts.to(valid_objective.dtype)

    def _aggregate_valid_objective_by_response(
        self,
        valid_objective: torch.Tensor,
        valid_sample_indices: torch.Tensor,
        valid_response_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        response_stride = int(valid_response_indices.max().item()) + 1
        group_ids = (
            valid_sample_indices.to(torch.long) * response_stride
            + valid_response_indices.to(torch.long)
        )
        unique_group_ids, inverse = torch.unique(
            group_ids,
            sorted=False,
            return_inverse=True,
        )
        response_objective_sums = torch.zeros(
            unique_group_ids.numel(),
            device=valid_objective.device,
            dtype=valid_objective.dtype,
        )
        response_objective_sums.index_add_(
            0,
            inverse,
            valid_objective,
        )
        response_token_counts = torch.zeros_like(response_objective_sums)
        response_token_counts.index_add_(
            0,
            inverse,
            torch.ones_like(valid_objective),
        )
        response_objective_means = (
            response_objective_sums / response_token_counts.clamp_min(1)
        )

        response_sample_indices = torch.div(
            unique_group_ids,
            response_stride,
            rounding_mode="floor",
        ).to(torch.long)
        sample_objective_sums = torch.zeros(
            batch_size,
            device=valid_objective.device,
            dtype=valid_objective.dtype,
        )
        sample_response_counts = torch.zeros_like(sample_objective_sums)
        sample_objective_sums.index_add_(
            0,
            response_sample_indices,
            response_objective_means,
        )
        sample_response_counts.index_add_(
            0,
            response_sample_indices,
            torch.ones_like(response_objective_means),
        )
        valid_sample_mask = sample_response_counts.gt(0)
        if not valid_sample_mask.any():
            raise RuntimeError("No valid response groups found for RL loss.")
        return (
            sample_objective_sums[valid_sample_mask]
            / sample_response_counts[valid_sample_mask]
        )

    def _prepare_varlen_optimizer_window(
        self,
        trainer_version: float,
    ) -> List[PreparedVarlenPack]:
        """Prepare a fixed-size CPU window and synchronize preparation errors."""
        window = None
        local_error = None
        try:
            window = [
                self._next_varlen_cpu_pack(trainer_version)
                for _ in range(self.args.grad_accum_steps)
            ]
            if len(window) != self.args.grad_accum_steps:
                raise RuntimeError(
                    "Varlen optimizer window has an unexpected pack count: "
                    f"{len(window)} != {self.args.grad_accum_steps}"
                )
        except Exception as exc:
            local_error = repr(exc)
            print(
                f"[rank {self.rank}] Varlen window preparation failed: "
                f"{local_error}"
            )

        success = torch.tensor(
            0 if local_error is not None else 1,
            device=self.device,
            dtype=torch.int32,
        )
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
        if int(success.item()) != 1:
            raise RuntimeError(
                "At least one FSDP rank failed to prepare its Varlen "
                "optimizer window; see per-rank logs for the original error."
            )
        assert window is not None
        return window

    def _run_varlen_optimizer_step(
        self,
        trainer_version: float,
    ) -> Dict[str, float]:
        """Run one globally token-normalized Varlen optimizer step."""
        window = self._prepare_varlen_optimizer_window(trainer_version)
        local_valid_token_count = sum(
            pack.valid_token_count for pack in window
        )
        global_valid_token_count_tensor = torch.tensor(
            local_valid_token_count,
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(
            global_valid_token_count_tensor,
            op=dist.ReduceOp.SUM,
        )
        global_valid_token_count = int(global_valid_token_count_tensor.item())
        if global_valid_token_count <= 0:
            raise RuntimeError(
                "Global Varlen optimizer window contains no valid "
                "response tokens."
            )

        local_token_stats = torch.zeros(
            len(VARLEN_TOKEN_STAT_NAMES),
            device=self.device,
            dtype=torch.float64,
        )
        local_episode_return_sum = 0.0
        local_version_lag_sum = 0.0
        local_sample_count = 0.0
        local_pack_utilization_sum = 0.0
        local_pack_sample_count_sum = 0.0
        local_pack_max_seqlen_sum = 0.0
        local_pack_cpu_seconds_sum = 0.0
        last_replay_stats = window[-1].replay_stats

        for prepared_pack in window:
            batch = move_batch_to_device(prepared_pack.batch, self.device)
            token_loss_sum, response_token_counts, loss_stats = (
                self._compute_rl_loss(
                    batch,
                    varlen_max_seqlen=prepared_pack.max_seqlen,
                    return_varlen_token_sums=True,
                )
            )
            backward_loss = token_loss_sum * (
                float(self.fsdp_world_size)
                / float(global_valid_token_count)
            )
            backward_loss.backward()

            train_stats = prepared_pack.train_stats
            varlen_token_stats = loss_stats["varlen_token_stats"]
            if not isinstance(varlen_token_stats, torch.Tensor):
                raise TypeError(
                    "Varlen loss statistics must remain an accelerator tensor."
                )
            if varlen_token_stats.shape != local_token_stats.shape:
                raise RuntimeError(
                    "Unexpected Varlen loss statistics shape: "
                    f"{tuple(varlen_token_stats.shape)} != "
                    f"{tuple(local_token_stats.shape)}"
                )
            local_token_stats.add_(varlen_token_stats)
            local_episode_return_sum += train_stats["episode_return_sum"]
            local_version_lag_sum += train_stats["trainer_version_lag_sum"]
            local_sample_count += train_stats["sample_count"]
            local_pack_utilization_sum += train_stats["pack_token_utilization"]
            local_pack_sample_count_sum += train_stats["pack_sample_count"]
            local_pack_max_seqlen_sum += train_stats[
                "pack_max_sequence_length"
            ]
            local_pack_cpu_seconds_sum += train_stats["pack_cpu_seconds"]
            self.train_micro_step += 1
            del (
                batch,
                token_loss_sum,
                response_token_counts,
                backward_loss,
                varlen_token_stats,
            )

        global_stats = torch.cat(
            (
                local_token_stats,
                torch.tensor(
                    [
                        local_episode_return_sum,
                        local_version_lag_sum,
                        local_sample_count,
                    ],
                    device=self.device,
                    dtype=torch.float64,
                ),
            )
        )
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
        (
            global_policy_sum,
            global_kl_sum,
            global_token_reward_sum,
            global_raw_advantage_sum,
            global_raw_advantage_sum_sq,
            global_used_advantage_sum,
            global_used_advantage_sum_sq,
            global_clip_count,
            global_episode_return_sum,
            global_version_lag_sum,
            global_sample_count_float,
        ) = global_stats.tolist()
        global_sample_count = int(global_sample_count_float)
        if global_sample_count <= 0:
            raise RuntimeError(
                "Global Varlen optimizer window contains no training samples."
            )

        torch.nn.utils.clip_grad_norm_(
            self.trainable_parameter_list,
            max_norm=1.0,
        )
        current_lr = self._get_current_lr(
            self.optimizer_step,
            self.args.learning_rate,
            self.args.lr_warmup_steps,
            self.args.max_steps,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step += 1

        token_count = float(global_valid_token_count)
        sample_count = float(global_sample_count)
        raw_advantage_mean = global_raw_advantage_sum / token_count
        used_advantage_mean = global_used_advantage_sum / token_count
        raw_advantage_variance = max(
            global_raw_advantage_sum_sq / token_count
            - raw_advantage_mean**2,
            0.0,
        )
        used_advantage_variance = max(
            global_used_advantage_sum_sq / token_count
            - used_advantage_mean**2,
            0.0,
        )
        policy_loss_token_mean = global_policy_sum / token_count
        kl_token_mean = global_kl_sum / token_count
        weighted_kl_loss = self.args.old_new_kl_coef * kl_token_mean
        total_loss_token_mean = policy_loss_token_mean + weighted_kl_loss

        self.last_loss = total_loss_token_mean
        self.last_reward_mean = global_episode_return_sum / sample_count
        self.last_advantage_mean = raw_advantage_mean
        self.last_response_tokens = token_count
        self.last_global_training_samples = global_sample_count
        self.last_replay_size = int(last_replay_stats["size"])
        self.last_total_sampled = int(
            last_replay_stats["total_samples_sampled"]
        )
        self.last_ppo_clip_frac = global_clip_count / token_count
        self.last_token_reward_mean = global_token_reward_sum / token_count
        self.last_raw_advantage_mean = raw_advantage_mean
        self.last_raw_advantage_std = math.sqrt(raw_advantage_variance)
        self.last_used_advantage_mean = used_advantage_mean
        self.last_used_advantage_std = math.sqrt(used_advantage_variance)
        self.last_old_new_kl_k3_token_mean = kl_token_mean
        self.last_old_new_kl_k3_sample_sum_mean = (
            global_kl_sum / sample_count
        )
        self.last_old_new_kl_k3_loss = weighted_kl_loss
        pack_count = float(len(window))
        self.last_pack_token_utilization = (
            local_pack_utilization_sum / pack_count
        )
        self.last_pack_sample_count = (
            local_pack_sample_count_sum / pack_count
        )
        self.last_pack_max_sequence_length = (
            local_pack_max_seqlen_sum / pack_count
        )
        self.last_pack_cpu_seconds = (
            local_pack_cpu_seconds_sum / pack_count
        )
        self.cumulative_global_valid_response_tokens += (
            global_valid_token_count
        )
        self.cumulative_global_training_samples += global_sample_count
        self.total_training_samples_consumed += int(local_sample_count)

        return {
            "global_policy_sum": global_policy_sum,
            "global_kl_sum": global_kl_sum,
            "global_token_reward_sum": global_token_reward_sum,
            "global_raw_advantage_sum": global_raw_advantage_sum,
            "global_raw_advantage_sum_sq": global_raw_advantage_sum_sq,
            "global_used_advantage_sum": global_used_advantage_sum,
            "global_used_advantage_sum_sq": global_used_advantage_sum_sq,
            "global_clip_count": global_clip_count,
            "global_episode_return_sum": global_episode_return_sum,
            "global_version_lag_sum": global_version_lag_sum,
            "global_valid_token_count": token_count,
            "global_sample_count": sample_count,
            "loss_token_mean": total_loss_token_mean,
            "policy_loss_token_mean": policy_loss_token_mean,
            "kl_token_mean": kl_token_mean,
            "weighted_kl_loss": weighted_kl_loss,
            "pack_utilization_sum": local_pack_utilization_sum,
            "pack_sample_count_sum": local_pack_sample_count_sum,
            "pack_max_seqlen_sum": local_pack_max_seqlen_sum,
            "pack_cpu_seconds_sum": local_pack_cpu_seconds_sum,
            "pack_count": pack_count,
            "current_lr": current_lr,
        }

    def train_until_next_sync(
        self,
        num_optimizer_steps: int = 100,
    ) -> Dict[str, float]:
        """
        Continue the persistent training loop until this worker finishes the
        requested number of optimizer steps, or reaches args.max_steps.

        args.max_steps is interpreted as optimizer steps.
        """
        if num_optimizer_steps < 1:
            raise ValueError("num_optimizer_steps must be >= 1")

        start_optimizer_step = self.optimizer_step
        target_optimizer_step = min(
            self.optimizer_step + num_optimizer_steps,
            self.args.max_steps,
        )
        segment_losses = []
        segment_version_lags = []
        segment_episode_return_means = []
        segment_token_reward_means = []
        segment_raw_advantage_means = []
        segment_raw_advantage_stds = []
        segment_used_advantage_means = []
        segment_used_advantage_stds = []
        segment_old_new_kl_k3_token_means = []
        segment_old_new_kl_k3_sample_sum_means = []
        segment_old_new_kl_k3_losses = []
        segment_pack_token_utilizations = []
        segment_pack_sample_counts = []
        segment_pack_max_sequence_lengths = []
        segment_pack_cpu_seconds = []
        segment_varlen_steps = []

        while self.optimizer_step < target_optimizer_step:
            trainer_version = self.optimizer_step / self.args.sync_every_optimizer_steps
            if self.args.train_packing == "varlen":
                step_stats = self._run_varlen_optimizer_step(trainer_version)
                segment_varlen_steps.append(step_stats)
                if self.rank == 0 and self.optimizer_step % self.args.log_every == 0:
                    print(
                        "[train] "
                        f"optimizer_step={self.optimizer_step} "
                        f"micro_step={self.train_micro_step} "
                        f"rl_loss={self.last_loss:.6f} "
                        "policy_loss_token_mean="
                        f"{step_stats['policy_loss_token_mean']:.6f} "
                        f"kl_token_mean={step_stats['kl_token_mean']:.6f} "
                        f"global_valid_tokens={self.last_response_tokens:.0f} "
                        "global_training_samples="
                        f"{self.last_global_training_samples} "
                        "cumulative_valid_tokens="
                        f"{self.cumulative_global_valid_response_tokens} "
                        "cumulative_training_samples="
                        f"{self.cumulative_global_training_samples} "
                        f"pack_utilization={self.last_pack_token_utilization:.4f} "
                        f"pack_samples={self.last_pack_sample_count:.2f} "
                        f"pack_max_seqlen={self.last_pack_max_sequence_length:.0f} "
                        f"pack_cpu_ms={self.last_pack_cpu_seconds * 1000.0:.3f} "
                        f"lr={step_stats['current_lr']:.8g}"
                    )
                continue

            batch, train_stats, replay_stats = self._next_rl_training_batch(
                trainer_version
            )
            raw_loss, response_token_counts, loss_stats = self._compute_rl_loss(
                batch,
            )

            loss = raw_loss / self.args.grad_accum_steps
            loss.backward()
            segment_losses.append(float(raw_loss.item()))
            segment_version_lags.append(float(train_stats["trainer_version_lag_mean"]))
            segment_episode_return_means.append(
                float(train_stats.get("episode_return_mean", 0.0))
            )
            segment_token_reward_means.append(
                float(loss_stats.get("token_reward_mean", 0.0))
            )
            segment_raw_advantage_means.append(
                float(loss_stats.get("raw_advantage_mean", 0.0))
            )
            segment_raw_advantage_stds.append(
                float(loss_stats.get("raw_advantage_std", 0.0))
            )
            segment_used_advantage_means.append(
                float(loss_stats.get("used_advantage_mean", 0.0))
            )
            segment_used_advantage_stds.append(
                float(loss_stats.get("used_advantage_std", 0.0))
            )
            segment_old_new_kl_k3_token_means.append(
                float(loss_stats.get("old_new_kl_k3_token_mean", 0.0))
            )
            segment_old_new_kl_k3_sample_sum_means.append(
                float(loss_stats.get("old_new_kl_k3_sample_sum_mean", 0.0))
            )
            segment_old_new_kl_k3_losses.append(
                float(loss_stats.get("old_new_kl_k3_loss", 0.0))
            )
            segment_pack_token_utilizations.append(
                float(train_stats.get("pack_token_utilization", 0.0))
            )
            segment_pack_sample_counts.append(
                float(train_stats.get("pack_sample_count", 0.0))
            )
            segment_pack_max_sequence_lengths.append(
                float(train_stats.get("pack_max_sequence_length", 0.0))
            )
            segment_pack_cpu_seconds.append(
                float(train_stats.get("pack_cpu_seconds", 0.0))
            )

            self.train_micro_step += 1
            self.last_loss = float(raw_loss.item())
            self.last_reward_mean = float(train_stats["reward_mean"])
            self.last_advantage_mean = float(train_stats["advantage_mean"])
            self.last_response_tokens = float(response_token_counts.sum().item())
            self.last_replay_size = int(replay_stats["size"])
            self.last_total_sampled = int(replay_stats["total_samples_sampled"])
            self.last_ppo_clip_frac = float(loss_stats.get("ppo_clip_frac", 0.0))
            self.last_token_reward_mean = float(
                loss_stats.get("token_reward_mean", 0.0)
            )
            self.last_raw_advantage_mean = float(
                loss_stats.get("raw_advantage_mean", 0.0)
            )
            self.last_raw_advantage_std = float(
                loss_stats.get("raw_advantage_std", 0.0)
            )
            self.last_used_advantage_mean = float(
                loss_stats.get("used_advantage_mean", 0.0)
            )
            self.last_used_advantage_std = float(
                loss_stats.get("used_advantage_std", 0.0)
            )
            self.last_old_new_kl_k3_token_mean = float(
                loss_stats.get("old_new_kl_k3_token_mean", 0.0)
            )
            self.last_old_new_kl_k3_sample_sum_mean = float(
                loss_stats.get("old_new_kl_k3_sample_sum_mean", 0.0)
            )
            self.last_old_new_kl_k3_loss = float(
                loss_stats.get("old_new_kl_k3_loss", 0.0)
            )
            self.last_pack_token_utilization = float(
                train_stats.get("pack_token_utilization", 0.0)
            )
            self.last_pack_sample_count = float(
                train_stats.get("pack_sample_count", 0.0)
            )
            self.last_pack_max_sequence_length = float(
                train_stats.get("pack_max_sequence_length", 0.0)
            )
            self.last_pack_cpu_seconds = float(
                train_stats.get("pack_cpu_seconds", 0.0)
            )
            should_step = self.train_micro_step % self.args.grad_accum_steps == 0
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameter_list,
                max_norm=1.0,
            )
            current_lr = self._get_current_lr(
                self.optimizer_step,
                self.args.learning_rate,
                self.args.lr_warmup_steps,
                self.args.max_steps,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1
            if self.rank == 0 and self.optimizer_step % self.args.log_every == 0:
                print(
                    "[train] "
                    f"optimizer_step={self.optimizer_step} "
                    f"micro_step={self.train_micro_step} "
                    f"rl_loss={self.last_loss:.6f} "
                    f"reward_mean={self.last_reward_mean:.4f} "
                    f"adv_mean={self.last_advantage_mean:.4f} "
                    f"response_tokens={self.last_response_tokens:.0f} "
                    f"replay_size={self.last_replay_size} "
                    f"total_sampled={self.last_total_sampled} "
                    f"ppo_clip_frac={self.last_ppo_clip_frac:.4f} "
                    f"token_reward_mean={self.last_token_reward_mean:.4f} "
                    f"raw_adv_mean={self.last_raw_advantage_mean:.4f} "
                    f"raw_adv_std={self.last_raw_advantage_std:.4f} "
                    f"used_adv_mean={self.last_used_advantage_mean:.4f} "
                    f"used_adv_std={self.last_used_advantage_std:.4f} "
                    "old_new_kl_k3_token_mean="
                    f"{self.last_old_new_kl_k3_token_mean:.6f} "
                    "old_new_kl_k3_sample_sum_mean="
                    f"{self.last_old_new_kl_k3_sample_sum_mean:.6f} "
                    f"old_new_kl_k3_loss={self.last_old_new_kl_k3_loss:.6f} "
                    f"pack_utilization={self.last_pack_token_utilization:.4f} "
                    f"pack_samples={self.last_pack_sample_count:.0f} "
                    f"pack_max_seqlen={self.last_pack_max_sequence_length:.0f} "
                    f"pack_cpu_ms={self.last_pack_cpu_seconds * 1000.0:.3f} "
                    f"lr={current_lr:.8g}"
                )

        varlen_segment = None
        if segment_varlen_steps:
            global_valid_tokens = sum(
                step["global_valid_token_count"]
                for step in segment_varlen_steps
            )
            global_samples = sum(
                step["global_sample_count"] for step in segment_varlen_steps
            )
            global_policy_sum = sum(
                step["global_policy_sum"] for step in segment_varlen_steps
            )
            global_kl_sum = sum(
                step["global_kl_sum"] for step in segment_varlen_steps
            )
            global_token_reward_sum = sum(
                step["global_token_reward_sum"]
                for step in segment_varlen_steps
            )
            global_raw_advantage_sum = sum(
                step["global_raw_advantage_sum"]
                for step in segment_varlen_steps
            )
            global_raw_advantage_sum_sq = sum(
                step["global_raw_advantage_sum_sq"]
                for step in segment_varlen_steps
            )
            global_used_advantage_sum = sum(
                step["global_used_advantage_sum"]
                for step in segment_varlen_steps
            )
            global_used_advantage_sum_sq = sum(
                step["global_used_advantage_sum_sq"]
                for step in segment_varlen_steps
            )
            global_episode_return_sum = sum(
                step["global_episode_return_sum"]
                for step in segment_varlen_steps
            )
            global_version_lag_sum = sum(
                step["global_version_lag_sum"]
                for step in segment_varlen_steps
            )
            global_clip_count = sum(
                step["global_clip_count"] for step in segment_varlen_steps
            )
            pack_count = sum(
                step["pack_count"] for step in segment_varlen_steps
            )
            raw_advantage_mean = (
                global_raw_advantage_sum / global_valid_tokens
            )
            used_advantage_mean = (
                global_used_advantage_sum / global_valid_tokens
            )
            raw_advantage_std = math.sqrt(
                max(
                    global_raw_advantage_sum_sq / global_valid_tokens
                    - raw_advantage_mean**2,
                    0.0,
                )
            )
            used_advantage_std = math.sqrt(
                max(
                    global_used_advantage_sum_sq / global_valid_tokens
                    - used_advantage_mean**2,
                    0.0,
                )
            )
            policy_loss_token_mean = global_policy_sum / global_valid_tokens
            kl_token_mean = global_kl_sum / global_valid_tokens
            weighted_kl_loss = self.args.old_new_kl_coef * kl_token_mean
            varlen_segment = {
                "loss_mean": policy_loss_token_mean + weighted_kl_loss,
                "episode_return_mean": (
                    global_episode_return_sum / global_samples
                ),
                "token_reward_mean": (
                    global_token_reward_sum / global_valid_tokens
                ),
                "raw_advantage_mean": raw_advantage_mean,
                "raw_advantage_std": raw_advantage_std,
                "used_advantage_mean": used_advantage_mean,
                "used_advantage_std": used_advantage_std,
                "kl_token_mean": kl_token_mean,
                "kl_sample_sum_mean": global_kl_sum / global_samples,
                "weighted_kl_loss": weighted_kl_loss,
                "clip_fraction": global_clip_count / global_valid_tokens,
                "version_lag_mean": global_version_lag_sum / global_samples,
                "pack_utilization_mean": sum(
                    step["pack_utilization_sum"]
                    for step in segment_varlen_steps
                )
                / pack_count,
                "pack_sample_count_mean": sum(
                    step["pack_sample_count_sum"]
                    for step in segment_varlen_steps
                )
                / pack_count,
                "pack_max_seqlen_mean": sum(
                    step["pack_max_seqlen_sum"]
                    for step in segment_varlen_steps
                )
                / pack_count,
                "pack_cpu_seconds_mean": sum(
                    step["pack_cpu_seconds_sum"]
                    for step in segment_varlen_steps
                )
                / pack_count,
                "global_valid_tokens": global_valid_tokens,
                "global_training_samples": global_samples,
            }

        dist.barrier()
        optimizer_steps_run = self.optimizer_step - start_optimizer_step
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "rank": self.rank,
            "optimizer_steps_run": optimizer_steps_run,
            "optimizer_step": self.optimizer_step,
            "micro_step": self.train_micro_step,
            "reached_max_steps": self.optimizer_step >= self.args.max_steps,
            "last_loss": self.last_loss,
            "last_reward_mean": self.last_reward_mean,
            "last_advantage_mean": self.last_advantage_mean,
            "last_response_tokens": self.last_response_tokens,
            "last_replay_size": self.last_replay_size,
            "last_total_sampled": self.last_total_sampled,
            "last_ppo_clip_frac": self.last_ppo_clip_frac,
            "last_token_reward_mean": self.last_token_reward_mean,
            "last_raw_advantage_mean": self.last_raw_advantage_mean,
            "last_raw_advantage_std": self.last_raw_advantage_std,
            "last_used_advantage_mean": self.last_used_advantage_mean,
            "last_used_advantage_std": self.last_used_advantage_std,
            "last_old_new_kl_k3_token_mean": self.last_old_new_kl_k3_token_mean,
            "last_old_new_kl_k3_sample_sum_mean": (
                self.last_old_new_kl_k3_sample_sum_mean
            ),
            "last_old_new_kl_k3_loss": self.last_old_new_kl_k3_loss,
            "last_pack_token_utilization": self.last_pack_token_utilization,
            "last_pack_sample_count": self.last_pack_sample_count,
            "last_pack_max_sequence_length": self.last_pack_max_sequence_length,
            "last_pack_cpu_seconds": self.last_pack_cpu_seconds,
            "last_global_training_samples": self.last_global_training_samples,
            "cumulative_global_valid_response_tokens": (
                self.cumulative_global_valid_response_tokens
            ),
            "cumulative_global_training_samples": (
                self.cumulative_global_training_samples
            ),
            "total_replay_candidates_fetched": (
                self.total_replay_candidates_fetched
            ),
            "total_valid_samples_prepared": self.total_valid_samples_prepared,
            "total_training_samples_consumed": (
                self.total_training_samples_consumed
            ),
            "total_invalid_candidates": self.total_invalid_candidates,
            "segment_global_valid_response_tokens": (
                varlen_segment["global_valid_tokens"]
                if varlen_segment is not None
                else 0.0
            ),
            "segment_global_training_samples": (
                varlen_segment["global_training_samples"]
                if varlen_segment is not None
                else 0.0
            ),
            "segment_policy_loss_token_mean": (
                varlen_segment["loss_mean"]
                - varlen_segment["weighted_kl_loss"]
                if varlen_segment is not None
                else 0.0
            ),
            "segment_ppo_clip_frac": (
                varlen_segment["clip_fraction"]
                if varlen_segment is not None
                else 0.0
            ),
            "segment_loss_mean": (
                varlen_segment["loss_mean"]
                if varlen_segment is not None
                else sum(segment_losses) / len(segment_losses)
                if segment_losses
                else 0.0
            ),
            "segment_episode_return_mean": (
                varlen_segment["episode_return_mean"]
                if varlen_segment is not None
                else sum(segment_episode_return_means) / len(segment_episode_return_means)
                if segment_episode_return_means else 0.0
            ),
            "segment_token_reward_mean": (
                varlen_segment["token_reward_mean"]
                if varlen_segment is not None
                else sum(segment_token_reward_means) / len(segment_token_reward_means)
                if segment_token_reward_means else 0.0
            ),
            "segment_raw_advantage_mean": (
                varlen_segment["raw_advantage_mean"]
                if varlen_segment is not None
                else sum(segment_raw_advantage_means) / len(segment_raw_advantage_means)
                if segment_raw_advantage_means else 0.0
            ),
            "segment_raw_advantage_std": (
                varlen_segment["raw_advantage_std"]
                if varlen_segment is not None
                else sum(segment_raw_advantage_stds) / len(segment_raw_advantage_stds)
                if segment_raw_advantage_stds else 0.0
            ),
            "segment_used_advantage_mean": (
                varlen_segment["used_advantage_mean"]
                if varlen_segment is not None
                else sum(segment_used_advantage_means) / len(segment_used_advantage_means)
                if segment_used_advantage_means else 0.0
            ),
            "segment_used_advantage_std": (
                varlen_segment["used_advantage_std"]
                if varlen_segment is not None
                else sum(segment_used_advantage_stds) / len(segment_used_advantage_stds)
                if segment_used_advantage_stds else 0.0
            ),
            "segment_old_new_kl_k3_token_mean": (
                varlen_segment["kl_token_mean"]
                if varlen_segment is not None
                else sum(segment_old_new_kl_k3_token_means)
                / len(segment_old_new_kl_k3_token_means)
                if segment_old_new_kl_k3_token_means else 0.0
            ),
            "segment_old_new_kl_k3_sample_sum_mean": (
                varlen_segment["kl_sample_sum_mean"]
                if varlen_segment is not None
                else sum(segment_old_new_kl_k3_sample_sum_means)
                / len(segment_old_new_kl_k3_sample_sum_means)
                if segment_old_new_kl_k3_sample_sum_means else 0.0
            ),
            "segment_old_new_kl_k3_loss": (
                varlen_segment["weighted_kl_loss"]
                if varlen_segment is not None
                else sum(segment_old_new_kl_k3_losses)
                / len(segment_old_new_kl_k3_losses)
                if segment_old_new_kl_k3_losses else 0.0
            ),
            "segment_pack_token_utilization_mean": (
                varlen_segment["pack_utilization_mean"]
                if varlen_segment is not None
                else sum(segment_pack_token_utilizations)
                / len(segment_pack_token_utilizations)
                if segment_pack_token_utilizations else 0.0
            ),
            "segment_pack_sample_count_mean": (
                varlen_segment["pack_sample_count_mean"]
                if varlen_segment is not None
                else sum(segment_pack_sample_counts) / len(segment_pack_sample_counts)
                if segment_pack_sample_counts else 0.0
            ),
            "segment_pack_max_sequence_length_mean": (
                varlen_segment["pack_max_seqlen_mean"]
                if varlen_segment is not None
                else sum(segment_pack_max_sequence_lengths)
                / len(segment_pack_max_sequence_lengths)
                if segment_pack_max_sequence_lengths else 0.0
            ),
            "segment_pack_cpu_seconds_mean": (
                varlen_segment["pack_cpu_seconds_mean"]
                if varlen_segment is not None
                else sum(segment_pack_cpu_seconds) / len(segment_pack_cpu_seconds)
                if segment_pack_cpu_seconds else 0.0
            ),
            "train_sample_trainer_version_lag_mean": (
                varlen_segment["version_lag_mean"]
                if varlen_segment is not None
                else sum(segment_version_lags) / len(segment_version_lags)
                if segment_version_lags else 0.0
            ),
            "learning_rate": current_lr,
        }

    # ---- weight-transfer setup (rank 0 only) ----

    def setup_transfer_endpoint(self):
        """Create the NCCL rendezvous endpoint for weight transfer."""
        assert self.rank == 0
        self.transfer_port = find_open_port()
        self.transfer_master_address = get_local_ip()
        return self.transfer_master_address, self.transfer_port

    def init_weight_transfer_group(self, transfer_world_size: int):
        """Join the weight-transfer NCCL group as rank 0 (the source)."""
        assert self.rank == 0
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.transfer_master_address,
                master_port=self.transfer_port,
                world_size=transfer_world_size,
            ),
        )

    def get_weight_metadata(self, scope: str = "all"):
        """Return scoped weight names, dtypes, and shapes from pre-FSDP params."""
        validate_weight_scope(scope)
        return self.weight_metadata_by_scope[scope]

    # ---- collective ops (ALL FSDP ranks must call concurrently) ----

    def gather_and_broadcast_weights(self, scope: str = "all", packed: bool = True):
        """
        All-gather scoped full parameters and broadcast them to vLLM.
        Only rank 0 performs the actual NCCL broadcast; others just
        participate in the FSDP all-gather.

        full_tensor() is a collective — all FSDP ranks must call it
        for each parameter in the same order.  Rank 0 additionally
        feeds each gathered tensor to the weight-transfer engine.
        """
        validate_weight_scope(scope)
        params = self.params_by_scope[scope]
        if self.rank == 0:
            def _full_param_iter():
                for name, param in params:
                    full_param = param.full_tensor().detach()
                    yield from iter_vllm_loadable_weights(name, full_param)

            trainer_args = NCCLTrainerSendWeightsArgs(
                group=self.model_update_group,
                packed=packed,
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=_full_param_iter(),
                trainer_args=trainer_args,
            )
        else:
            for _, param in params:
                param.full_tensor()

    def save_checkpoint(self, checkpoint_dir: str, tag: str) -> Dict[str, Any]:
        """
        Save a HuggingFace-format full-model checkpoint.

        ``full_tensor()`` is collective for FSDP2 sharded parameters, so all
        ranks must call this method together. Rank 0 materializes CPU tensors
        and writes the checkpoint; other ranks only participate in all-gather.
        """
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be non-empty")
        if not tag:
            raise ValueError("tag must be non-empty")

        output_dir = os.path.join(checkpoint_dir, tag)
        dist.barrier()
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            print(
                "[checkpoint] "
                f"Saving HuggingFace checkpoint to {output_dir} "
                f"(optimizer_step={self.optimizer_step})"
            )
        dist.barrier()

        state_dict = None
        if self.rank == 0:
            state_dict = {}

        with torch.no_grad():
            for name, param in self.params_by_scope["all"]:
                full_param = param.full_tensor().detach()
                if self.rank == 0:
                    assert state_dict is not None
                    state_dict[name] = full_param.cpu()
                del full_param

        result = {
            "rank": self.rank,
            "checkpoint_dir": output_dir,
            "optimizer_step": self.optimizer_step,
            "saved": False,
        }
        if self.rank == 0:
            assert state_dict is not None
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            self.tokenizer.save_pretrained(output_dir)
            trainer_state = {
                "optimizer_step": self.optimizer_step,
                "train_micro_step": self.train_micro_step,
                "fsdp_world_size": self.fsdp_world_size,
                "train_mode": self.args.train_mode,
                "rl_algorithm": self.args.rl_algorithm,
                "max_steps": self.args.max_steps,
                "sync_every_optimizer_steps": self.args.sync_every_optimizer_steps,
            }
            state_path = os.path.join(output_dir, "trainer_state.json")
            with open(state_path, "w", encoding="utf-8") as file:
                json.dump(trainer_state, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
            del state_dict
            result["saved"] = True
            print(f"[checkpoint] Saved checkpoint to {output_dir}")

        dist.barrier()
        return result


def create_async_engine(**kwargs):
    """Create an AsyncLLMEngine directly (no subclass needed)."""
    kwargs = _filter_async_engine_args(kwargs)
    engine_args = vllm.AsyncEngineArgs(**kwargs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return vllm.AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )


def _filter_async_engine_args(kwargs: Dict) -> Dict:
    signature = inspect.signature(vllm.AsyncEngineArgs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[vllm] Ignoring unsupported AsyncEngineArgs: {dropped}")
    return filtered


@dataclass
class OnlineGenerationState:
    """Token-level state for one request across weight-update interruptions."""

    index: int
    input_ids: List[int]
    requested_max_tokens: int
    output_tokens: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
    attempts: int = 0

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.output_tokens))

    @property
    def restart_prompt_token_ids(self) -> List[int]:
        return self.input_ids + self.output_tokens


@dataclass
class RepeatingInferenceStats:
    total_requests: int = 0
    total_tokens: int = 0


@dataclass
class InferenceRequestItem:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    input_ids: List[int]
    requested_max_tokens: int


@dataclass
class InferenceResult:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    output_tokens: List[int]
    output_logprobs: List[float]
    output_versions: List[int]
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None
    attempts: int

    @property
    def version_range(self) -> str:
        if not self.output_versions:
            return "none"
        return f"{min(self.output_versions)}-{max(self.output_versions)}"


@dataclass
class RLSample:
    algorithm: Literal["ppo", "grpo"]
    response_ids: List[int]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    old_logprobs: List[float]
    reward: float
    advantage: float
    token_rewards: List[float]
    token_advantages: List[float]
    response_indices: List[int]
    episode_return: float
    output_versions: List[int]


@ray.remote
class StatsActor:
    """Aggregates rollout metrics from async rollout workers."""

    def __init__(self, window_size: int, active_timeout_seconds: float):
        self.reward_sums = deque(maxlen=window_size)
        self.worker_last_active = {}
        self.active_timeout_seconds = active_timeout_seconds
        self.total_episodes = 0
        self.tw_scores = deque(maxlen=window_size)
        self.tw_max_scores = deque(maxlen=window_size)
        self.tw_wins = deque(maxlen=window_size)
        self.tw_steps = deque(maxlen=window_size)
        self.tw_env_steps = deque(maxlen=window_size)
        self.tw_invalid_actions = deque(maxlen=window_size)
        self.tw_selected_response_lengths = deque(maxlen=window_size)
        self.ppo_token_reward_means = deque(maxlen=window_size)
        self.ppo_raw_advantage_means = deque(maxlen=window_size)
        self.ppo_raw_advantage_stds = deque(maxlen=window_size)
        self.ppo_invalid_actions = deque(maxlen=window_size)
        self.grpo_group_reward_means = deque(maxlen=window_size)
        self.grpo_group_reward_stds = deque(maxlen=window_size)
        self.grpo_advantage_stds = deque(maxlen=window_size)
        self.grpo_invalid_actions = deque(maxlen=window_size)

    def add_textworld_episode(
        self,
        worker_id: int,
        trajectory_return: float,
        score: float,
        max_score: float,
        won: bool,
        steps: int,
        env_steps: int,
        invalid_actions: int,
        selected_response_lengths: List[int] | None = None,
        ppo_token_reward_mean: float | None = None,
        ppo_raw_advantage_mean: float | None = None,
        ppo_raw_advantage_std: float | None = None,
        ppo_invalid_actions_mean: float | None = None,
        grpo_group_reward_mean: float | None = None,
        grpo_group_reward_std: float | None = None,
        grpo_advantage_std: float | None = None,
        grpo_invalid_actions_mean: float | None = None,
    ) -> None:
        selected_response_lengths = selected_response_lengths or []
        self.reward_sums.append(float(trajectory_return))
        self.tw_scores.append(float(score))
        self.tw_max_scores.append(float(max_score))
        self.tw_wins.append(bool(won))
        self.tw_steps.append(int(steps))
        self.tw_env_steps.append(int(env_steps))
        self.tw_invalid_actions.append(int(invalid_actions))
        self.tw_selected_response_lengths.extend(
            int(value) for value in selected_response_lengths
        )
        if ppo_token_reward_mean is not None:
            self.ppo_token_reward_means.append(float(ppo_token_reward_mean))
        if ppo_raw_advantage_mean is not None:
            self.ppo_raw_advantage_means.append(float(ppo_raw_advantage_mean))
        if ppo_raw_advantage_std is not None:
            self.ppo_raw_advantage_stds.append(float(ppo_raw_advantage_std))
        if ppo_invalid_actions_mean is not None:
            self.ppo_invalid_actions.append(float(ppo_invalid_actions_mean))
        if grpo_group_reward_mean is not None:
            self.grpo_group_reward_means.append(float(grpo_group_reward_mean))
        if grpo_group_reward_std is not None:
            self.grpo_group_reward_stds.append(float(grpo_group_reward_std))
        if grpo_advantage_std is not None:
            self.grpo_advantage_stds.append(float(grpo_advantage_std))
        if grpo_invalid_actions_mean is not None:
            self.grpo_invalid_actions.append(float(grpo_invalid_actions_mean))
        self.total_episodes += 1
        self.worker_last_active[int(worker_id)] = time.time()

    def get_stats(self) -> Dict[str, float]:
        now = time.time()
        active_cutoff = now - self.active_timeout_seconds
        active_workers = sum(
            1 for last_active in self.worker_last_active.values()
            if last_active >= active_cutoff
        )
        reward_count = len(self.reward_sums)
        tw_selected_response_count = len(self.tw_selected_response_lengths)
        ppo_token_reward_mean_count = len(self.ppo_token_reward_means)
        ppo_raw_advantage_mean_count = len(self.ppo_raw_advantage_means)
        ppo_raw_advantage_std_count = len(self.ppo_raw_advantage_stds)
        ppo_invalid_action_count = len(self.ppo_invalid_actions)
        grpo_group_reward_mean_count = len(self.grpo_group_reward_means)
        grpo_group_reward_std_count = len(self.grpo_group_reward_stds)
        grpo_advantage_std_count = len(self.grpo_advantage_stds)
        grpo_invalid_action_count = len(self.grpo_invalid_actions)
        tw_episode_count = len(self.tw_scores)
        tw_total_steps = sum(self.tw_steps)
        tw_total_env_steps = sum(self.tw_env_steps)
        tw_total_max_score = sum(self.tw_max_scores)
        return {
            "global_reward_sum_mean": (
                sum(self.reward_sums) / reward_count if reward_count else 0.0
            ),
            "active_workers": active_workers,
            "total_episodes": self.total_episodes,
            "tw_episode_count": tw_episode_count,
            "tw_win_rate": (
                sum(1 for won in self.tw_wins if won) / tw_episode_count
                if tw_episode_count else 0.0
            ),
            "tw_mean_score": (
                sum(self.tw_scores) / tw_episode_count if tw_episode_count else 0.0
            ),
            "tw_normalized_score": (
                sum(self.tw_scores) / tw_total_max_score
                if tw_total_max_score > 0 else 0.0
            ),
            "tw_invalid_action_rate": (
                sum(self.tw_invalid_actions) / tw_total_steps
                if tw_total_steps else 0.0
            ),
            "tw_env_steps_mean": (
                tw_total_env_steps / tw_episode_count if tw_episode_count else 0.0
            ),
            "tw_selected_response_length_mean": (
                sum(self.tw_selected_response_lengths) / tw_selected_response_count
                if tw_selected_response_count else 0.0
            ),
            "ppo_token_reward_mean": (
                sum(self.ppo_token_reward_means) / ppo_token_reward_mean_count
                if ppo_token_reward_mean_count else 0.0
            ),
            "ppo_raw_advantage_mean": (
                sum(self.ppo_raw_advantage_means) / ppo_raw_advantage_mean_count
                if ppo_raw_advantage_mean_count else 0.0
            ),
            "ppo_raw_advantage_std": (
                sum(self.ppo_raw_advantage_stds) / ppo_raw_advantage_std_count
                if ppo_raw_advantage_std_count else 0.0
            ),
            "ppo_invalid_actions_mean": (
                sum(self.ppo_invalid_actions) / ppo_invalid_action_count
                if ppo_invalid_action_count else 0.0
            ),
            "grpo_group_reward_mean": (
                sum(self.grpo_group_reward_means) / grpo_group_reward_mean_count
                if grpo_group_reward_mean_count else 0.0
            ),
            "grpo_group_reward_std": (
                sum(self.grpo_group_reward_stds) / grpo_group_reward_std_count
                if grpo_group_reward_std_count else 0.0
            ),
            "grpo_advantage_std": (
                sum(self.grpo_advantage_stds) / grpo_advantage_std_count
                if grpo_advantage_std_count else 0.0
            ),
            "grpo_invalid_actions_mean": (
                sum(self.grpo_invalid_actions) / grpo_invalid_action_count
                if grpo_invalid_action_count else 0.0
            ),
        }


@ray.remote
class ReplayBufferActor:
    """Replay buffer that stores rollout-produced RL samples."""

    def __init__(self, capacity: int):
        self.samples = deque(maxlen=capacity)
        self.total_samples_added = 0
        self.total_samples_sampled = 0
        self.total_samples_evicted = 0
        self.total_batches_added = 0
        self.total_batches_sampled = 0

    def add_samples(self, samples: List[RLSample]) -> Dict[str, int]:
        capacity = self.samples.maxlen or 0
        if capacity > 0:
            self.total_samples_evicted += max(
                0,
                len(self.samples) + len(samples) - capacity,
            )
        self.samples.extend(samples)
        self.total_samples_added += len(samples)
        self.total_batches_added += 1
        return self.get_stats()

    def sample(self, batch_size: int) -> List[RLSample]:
        if batch_size < 1:
            return []
        sample_count = min(batch_size, len(self.samples))
        if sample_count == 0:
            return []
        samples = random.sample(list(self.samples), sample_count)
        self.total_samples_sampled += len(samples)
        self.total_batches_sampled += 1
        return samples

    def get_stats(self) -> Dict[str, int]:
        return {
            "size": len(self.samples),
            "capacity": self.samples.maxlen or 0,
            "total_samples_added": self.total_samples_added,
            "total_samples_sampled": self.total_samples_sampled,
            "total_candidates_fetched": self.total_samples_sampled,
            "total_samples_evicted": self.total_samples_evicted,
            "total_batches_added": self.total_batches_added,
            "total_batches_sampled": self.total_batches_sampled,
            "total_candidate_batches_fetched": self.total_batches_sampled,
        }


def _tokens_from_output(request_output) -> List[int]:
    if not getattr(request_output, "outputs", None):
        return []
    return list(getattr(request_output.outputs[0], "token_ids", []) or [])


def _logprobs_from_output(request_output, token_ids: List[int]) -> List[float]:
    if not getattr(request_output, "outputs", None):
        return []
    output = request_output.outputs[0]
    output_logprobs = getattr(output, "logprobs", None) or []
    logprobs = []
    for token_id, token_logprobs in zip(token_ids, output_logprobs):
        value = None
        if isinstance(token_logprobs, dict):
            value = token_logprobs.get(token_id)
            if value is None:
                value = token_logprobs.get(str(token_id))
        else:
            value = token_logprobs

        if hasattr(value, "logprob"):
            value = value.logprob
        if value is None:
            return []
        logprobs.append(float(value))
    return logprobs


def _finish_reason_from_output(request_output):
    if not getattr(request_output, "outputs", None):
        return None
    return getattr(request_output.outputs[0], "finish_reason", None)


def _normalize_stop_reason(stop_reason) -> Literal["length", "stop", "tool_calls", "abort"]:
    if stop_reason in ("length", "stop", "tool_calls", "abort"):
        return stop_reason
    if stop_reason in ("eos", "stop_token", "stop_sequence"):
        return "stop"
    return "abort"


class InterruptibleGenerationRunner:
    """Run vLLM requests that survive abort-based weight-update pauses."""

    def __init__(
        self,
        engine,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: List[str] | None = None,
        collect_logprobs: bool = False,
        max_resubmit_retries: int = 200,
    ):
        self.engine = engine
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences or ["</answer>"]
        self.collect_logprobs = collect_logprobs
        self.max_resubmit_retries = max_resubmit_retries
        self.version = 0
        self.resume_event = asyncio.Event()
        self.resume_event.set()
        self._active_attempts = 0
        self._active_changed = asyncio.Condition()

    def pause(self) -> None:
        self.resume_event.clear()

    def resume(self) -> None:
        self.resume_event.set()

    @property
    def is_resumed(self) -> bool:
        return self.resume_event.is_set()

    # 新的engine.generate() attempt开始 +1
    async def _increment_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts += 1
            self._active_changed.notify_all()

    # 一个engine.generate() attempt 结束 -1
    async def _decrement_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts -= 1
            self._active_changed.notify_all()

    # 等待正在跑的 generate attempt 都结束,可能是正常结束，也可能是被abort打断
    async def wait_for_idle(self) -> None:
        async with self._active_changed:
            await self._active_changed.wait_for(lambda: self._active_attempts == 0)

    async def generate(self, state: OnlineGenerationState) -> OnlineGenerationState:
        for attempt in range(1, self.max_resubmit_retries + 1):
            # 如果当前正在weight update的attempt还没结束，就等着，不要开始新的generate attempt
            await self.resume_event.wait()

            remaining = state.remaining_max_tokens
            if remaining <= 0:
                state.stop_reason = "length"
                return state

            state.attempts = attempt
            attempt_version = self.version
            sampling_kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": remaining,
                "stop": self.stop_sequences,
            }
            if self.collect_logprobs:
                sampling_kwargs["logprobs"] = 1
            sampling_params = SamplingParams(**sampling_kwargs)
            request_id = (
                f"online-sync-{state.index}-v{attempt_version}-"
                f"try{attempt}-{uuid.uuid4()}"
            )
            final_output = None
            request_finished = False

            await self._increment_active_attempts()
            try:
                # 调用vllm生成接口，拿到输出后更新state，如果生成过程中被weight update打断了，engine.generate()会抛出异常，直接进入finally块结束这个attempt
                async for request_output in self.engine.generate(
                    {"prompt_token_ids": state.restart_prompt_token_ids},
                    sampling_params,
                    request_id=request_id,
                ):
                    final_output = request_output
                    request_finished = bool(
                        getattr(request_output, "finished", False)
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.resume_event.is_set():
                    raise
                final_output = None
            finally:
                await self._decrement_active_attempts()

            if final_output is None:
                state.stop_reason = "abort"
                continue

            attempt_tokens = _tokens_from_output(final_output)[:remaining]
            if attempt_tokens:
                attempt_logprobs = []
                if self.collect_logprobs:
                    attempt_logprobs = _logprobs_from_output(
                        final_output,
                        attempt_tokens,
                    )[: len(attempt_tokens)]
                    if len(attempt_logprobs) != len(attempt_tokens):
                        state.stop_reason = "abort"
                        continue
                state.output_tokens.extend(attempt_tokens)
                state.output_logprobs.extend(attempt_logprobs)
                state.output_versions.extend(
                    [attempt_version] * len(attempt_tokens)
                )

            stop_reason = _normalize_stop_reason(
                _finish_reason_from_output(final_output)
            )
            if len(state.output_tokens) >= state.requested_max_tokens:
                stop_reason = "length"

            state.stop_reason = stop_reason
            if stop_reason in ("stop", "tool_calls", "length"):
                return state

            if not request_finished or stop_reason == "abort":
                await asyncio.sleep(0)
                continue

            return state

        state.stop_reason = (
            "length" if state.remaining_max_tokens <= 0 else "abort"
        )
        print(
            "[generate] Request "
            f"{state.index} reached max_resubmit_retries="
            f"{self.max_resubmit_retries}; keeping partial output."
        )
        return state


class VLLMInferenceActor:
    """GPU Ray actor that owns vLLM and consumes tokenized rollout requests."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        engine_kwargs = dict(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=True,
            tensor_parallel_size=args.infer_tp_size,
            data_parallel_size=args.infer_size,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            gpu_memory_utilization=0.8,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            enable_prefix_caching=not args.disable_vllm_prefix_caching,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            load_format="dummy",
        )
        if args.vllm_max_model_len is not None:
            engine_kwargs["max_model_len"] = args.vllm_max_model_len
        self.engine = create_async_engine(**engine_kwargs)
        self.runner = InterruptibleGenerationRunner(
            self.engine,
            temperature=args.infer_temperature,
            top_p=args.infer_top_p,
            stop_sequences=["\n"],
            collect_logprobs=True,
        )
        self.active_generation_tasks = set()
        self.stats = RepeatingInferenceStats()
        self.next_request_index = 0
        self.pending_futures = set()
        self.stopped = False

    async def start(self):
        self.stopped = False
        return {"continuous_submit": True, "inference_loop_task": 0}

    async def request_batch(
        self,
        rollout_worker_id: int,
        batch_id: int,
        input_ids: List[int],
        infer_max_tokens: int,
    ) -> InferenceResult:
        if self.stopped:
            raise RuntimeError("VLLMInferenceActor is stopped.")
        request_index = self.next_request_index
        self.next_request_index += 1
        item = InferenceRequestItem(
            request_index=request_index,
            rollout_worker_id=int(rollout_worker_id),
            batch_id=int(batch_id),
            input_ids=list(input_ids),
            requested_max_tokens=int(infer_max_tokens),
        )
        return await self._run_generation_item(item)

    async def _run_generation_item(
        self,
        item: InferenceRequestItem,
    ) -> InferenceResult:
        current_call = asyncio.current_task()
        if current_call is not None:
            self.pending_futures.add(current_call)

        try:
            state = OnlineGenerationState(
                index=item.request_index,
                input_ids=item.input_ids,
                requested_max_tokens=item.requested_max_tokens,
            )
            generation_task = asyncio.create_task(self.runner.generate(state))
            self.active_generation_tasks.add(generation_task)
            generation_task.add_done_callback(self.active_generation_tasks.discard)
            try:
                completed_state = await generation_task
            except asyncio.CancelledError:
                if not generation_task.done():
                    generation_task.cancel()
                await asyncio.gather(generation_task, return_exceptions=True)
                raise

            result = InferenceResult(
                request_index=item.request_index,
                rollout_worker_id=item.rollout_worker_id,
                batch_id=item.batch_id,
                output_tokens=list(completed_state.output_tokens),
                output_logprobs=list(completed_state.output_logprobs),
                output_versions=list(completed_state.output_versions),
                stop_reason=completed_state.stop_reason,
                attempts=completed_state.attempts,
            )
            await self._record_completed_state(result)
            return result
        finally:
            if current_call is not None:
                self.pending_futures.discard(current_call)

    async def _record_completed_state(
        self,
        result: InferenceResult,
    ) -> None:
        self.stats.total_requests += 1
        self.stats.total_tokens += len(result.output_tokens)

    async def pause_and_wait_idle(self):
        self.runner.pause()
        await self.engine.pause_generation(mode="abort", clear_cache=True)
        await self.runner.wait_for_idle()

    async def resume_generation(self, increment_version: bool = False):
        if increment_version:
            self.runner.version += 1
        await self.engine.resume_generation()
        self.runner.resume()
        return self.runner.version

    async def init_weight_transfer_engine(
        self,
        master_address: str,
        master_port: int,
        transfer_world_size: int,
    ):
        await self.engine.init_weight_transfer_engine(
            WeightTransferInitRequest(
                init_info=asdict(
                    NCCLWeightTransferInitInfo(
                        master_address=master_address,
                        master_port=master_port,
                        rank_offset=1,
                        world_size=transfer_world_size,
                    )
                )
            )
        )

    async def start_weight_update(self):
        await self.engine.start_weight_update()

    async def update_weights(
        self,
        names: List[str],
        dtype_names: List[str],
        shapes: List[List[int]],
        packed: bool = True,
    ):
        await self.engine.update_weights(
            WeightTransferUpdateRequest(
                update_info=asdict(
                    NCCLWeightTransferUpdateInfo(
                        names=names,
                        dtype_names=dtype_names,
                        shapes=shapes,
                        packed=packed,
                    )
                )
            )
        )

    async def finish_weight_update(self):
        await self.engine.finish_weight_update()

    def get_stats(self):
        return {
            "pending_futures": len(self.pending_futures),
            "total_requests": self.stats.total_requests,
            "total_tokens": self.stats.total_tokens,
            "active_attempts": self.runner._active_attempts,
            "active_generation_tasks": len(self.active_generation_tasks),
            "vllm_max_num_seqs": self.args.vllm_max_num_seqs,
            "vllm_max_num_batched_tokens": self.args.vllm_max_num_batched_tokens,
            "vllm_max_model_len": self.args.vllm_max_model_len,
            "weight_version": self.runner.version,
            "resumed": self.runner.is_resumed,
        }

    async def shutdown(self):
        self.stopped = True
        self.runner.resume()
        for task in list(self.active_generation_tasks):
            if not task.done():
                task.cancel()
        if self.active_generation_tasks:
            await asyncio.gather(*self.active_generation_tasks, return_exceptions=True)
        await shutdown_vllm_engine(self.engine)
        return self.get_stats()


@dataclass
class TextWorldStepRecord:
    training_result: InferenceResult | None
    prompt_ids: List[int]
    reward: float


@dataclass
class TextWorldTrajectoryState:
    group_index: int
    env: Any
    obs: str
    infos: Dict
    latest_score: float
    step_records: List[TextWorldStepRecord] = field(default_factory=list)
    transcript_ids: List[int] = field(default_factory=list)
    invalid_actions: int = 0
    env_steps: int = 0
    done: bool = False
    won: bool = False
    lost: bool = False
    selected_response_lengths: List[int] = field(default_factory=list)


@dataclass
class TextWorldPendingRequest:
    state: TextWorldTrajectoryState
    prompt_obs: str
    prompt_infos: Dict
    input_ids: List[int]
    score_before: float


def load_textworld_game_files(args: argparse.Namespace) -> List[str]:
    pattern = os.path.join(args.tw_game_dir, args.tw_game_pattern)
    game_files = sorted(glob.glob(pattern))
    game_files = [path for path in game_files if os.path.isfile(path)]
    random.Random(args.seed).shuffle(game_files)
    if args.tw_game_limit is not None:
        game_files = game_files[: args.tw_game_limit]
    if not game_files:
        raise ValueError(
            "No TextWorld game files found: "
            f"tw_game_dir={args.tw_game_dir!r} "
            f"tw_game_pattern={args.tw_game_pattern!r}"
        )
    return game_files


def make_textworld_request_infos() -> textworld.EnvInfos:
    return textworld.EnvInfos(
        objective=True,
        inventory=True,
        admissible_commands=True,
        score=True,
        max_score=True,
        won=True,
        lost=True,
        moves=True,
    )


TEXTWORLD_SYSTEM_PROMPT = (
    "You are a TextWorld action selector. Reply with exactly one command "
    "from the admissible commands. Do not add explanations, punctuation, "
    "quotes, or a leading prompt marker."
)


@dataclass
class ParsedAction:
    raw_text: str
    normalized: str
    valid: bool
    action: str | None


def _clean_action_text(text: str) -> str:
    first_line = text.splitlines()[0] if text.splitlines() else text
    cleaned = first_line.strip()
    for _ in range(4):
        cleaned = cleaned.strip()
        cleaned = cleaned.lstrip(">")
        cleaned = cleaned.strip()
        lower_cleaned = cleaned.lower()
        for prefix in (
            "action:",
            "command:",
            "assistant:",
            "answer:",
            "output:",
            "input:",
        ):
            if lower_cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        else:
            break
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.strip("'\"`")
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.strip("'\"`")
    return " ".join(cleaned.lower().split())


def parse_model_action(raw_text: str, admissible_commands: List[str]) -> ParsedAction:
    normalized = _clean_action_text(raw_text)
    command_by_normalized = {
        " ".join(command.lower().split()): command
        for command in admissible_commands
    }
    action = command_by_normalized.get(normalized)
    return ParsedAction(
        raw_text=raw_text,
        normalized=normalized,
        valid=action is not None,
        action=action,
    )


def format_textworld_user_content(obs: str, infos: Dict) -> str:
    objective = infos.get("objective") or ""
    inventory = infos.get("inventory") or ""
    admissible_commands = infos.get("admissible_commands", []) or []
    command_lines = "\n".join(f"- {command}" for command in admissible_commands)
    return (
        "Objective:\n"
        f"{objective}\n\n"
        "Observation:\n"
        f"{obs}\n\n"
        "Inventory:\n"
        f"{inventory}\n\n"
        "Admissible commands:\n"
        f"{command_lines}\n\n"
        "Return one command only."
    )


def format_textworld_illegal_action_feedback(action: str, obs: str, infos: Dict) -> str:
    return (
        f'Illegal action: "{action}" is not an admissible command.\n\n'
        + format_textworld_user_content(obs, infos)
    )


def encode_tokenizer_fragment(tokenizer, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def format_textworld_transcript_user_suffix(
    user_content: str,
    tokenizer=None,
) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        if "<|im_start|>" not in chat_template or "<|im_end|>" not in chat_template:
            raise ValueError(
                "--tw-history-token-window requires a Qwen-style "
                "<|im_start|>/<|im_end|> chat template when using "
                "apply_chat_template."
            )
        return (
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    return f"\n\nUser:\n{user_content}\n\nAssistant:"


def encode_textworld_transcript_user_suffix(
    user_content: str,
    tokenizer=None,
) -> List[int]:
    return encode_tokenizer_fragment(
        tokenizer,
        format_textworld_transcript_user_suffix(user_content, tokenizer=tokenizer),
    )


def format_textworld_prompt(obs: str, infos: Dict, tokenizer=None) -> str:
    user_prompt = format_textworld_user_content(obs, infos)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": TEXTWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"System: {TEXTWORLD_SYSTEM_PROMPT}\n\n"
        f"User:\n{user_prompt}\n\n"
        "Assistant:"
    )


def _textworld_score(step_score, infos: Dict) -> float:
    value = infos.get("score", step_score)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _textworld_max_score(infos: Dict) -> float:
    value = infos.get("max_score", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class TextWorldRolloutWorkerActor:
    """CPU Ray actor that collects TextWorld episodes and optional RL samples."""

    def __init__(
        self,
        args: argparse.Namespace,
        infer_actor,
        worker_id: int,
        replay_buffer,
        stats_actor,
    ):
        self.args = args
        self.infer_actor = infer_actor
        self.worker_id = int(worker_id)
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor
        self.tokenizer = build_tokenizer(args, log=False)
        self.game_files = load_textworld_game_files(args)
        self.batch_id = 0
        self.stopped = False
        if self._log_detail:
            print(
                "[tw-rollout] "
                f"worker={self.worker_id} loaded TextWorld games: "
                f"count={len(self.game_files)} "
                f"dir={args.tw_game_dir!r} pattern={args.tw_game_pattern!r}"
            )

    @property
    def _log_detail(self) -> bool:
        return self.worker_id == 0

    async def stop(self):
        self.stopped = True

    def _compute_step_reward(
        self,
        score_before: float,
        score_after: float,
        won: bool,
        lost: bool,
    ) -> float:
        reward = score_after - score_before - self.args.tw_step_penalty
        if won:
            reward += self.args.tw_win_bonus
        if lost:
            reward -= self.args.tw_lost_penalty
        return float(reward)

    def _initial_transcript_ids(self, obs: str, infos: Dict) -> List[int]:
        prompt = format_textworld_prompt(obs, infos, tokenizer=self.tokenizer)
        return list(self.tokenizer.encode(prompt))

    def _append_transcript_user_suffix(
        self,
        transcript_ids: List[int],
        user_content: str,
    ) -> None:
        transcript_ids.extend(
            encode_textworld_transcript_user_suffix(
                user_content,
                tokenizer=self.tokenizer,
            )
        )

    def _append_next_observation_to_transcript(
        self,
        transcript_ids: List[int],
        obs: str,
        infos: Dict,
    ) -> None:
        self._append_transcript_user_suffix(
            transcript_ids,
            format_textworld_user_content(obs, infos),
        )

    def _append_illegal_feedback_to_transcript(
        self,
        transcript_ids: List[int],
        action: str,
        obs: str,
        infos: Dict,
    ) -> None:
        self._append_transcript_user_suffix(
            transcript_ids,
            format_textworld_illegal_action_feedback(action, obs, infos),
        )

    def _apply_textworld_action_result(
        self,
        pending: TextWorldPendingRequest,
        result: InferenceResult,
        invalid_reward_mode: Literal["zero", "ppo_penalty"],
    ) -> None:
        state = pending.state
        admissible_commands = state.infos.get("admissible_commands", []) or []
        raw_text = self.tokenizer.decode(
            result.output_tokens,
            skip_special_tokens=True,
        )
        parsed_action = parse_model_action(raw_text, admissible_commands)
        model_action_valid = parsed_action.valid and parsed_action.action is not None

        state.selected_response_lengths.append(len(result.output_tokens))
        state.transcript_ids.extend(result.output_tokens)
        if model_action_valid:
            selected_action = parsed_action.action
            assert selected_action is not None
            obs, step_score, done, infos = state.env.step(selected_action)
            state.env_steps += 1
            state.obs = obs
            state.infos = dict(infos)
            state.latest_score = _textworld_score(step_score, infos)
            state.done = bool(done)
            state.won = bool(infos.get("won", False))
            state.lost = bool(infos.get("lost", False))
            if not state.done:
                self._append_next_observation_to_transcript(
                    state.transcript_ids,
                    state.obs,
                    state.infos,
                )
            reward = self._compute_step_reward(
                pending.score_before,
                state.latest_score,
                state.won,
                state.lost,
            )
        else:
            state.invalid_actions += 1
            invalid_action = (
                parsed_action.normalized
                if parsed_action.normalized
                else raw_text.strip()
            )
            self._append_illegal_feedback_to_transcript(
                state.transcript_ids,
                invalid_action,
                pending.prompt_obs,
                pending.prompt_infos,
            )
            if invalid_reward_mode == "zero":
                reward = 0.0
            elif invalid_reward_mode == "ppo_penalty":
                if result.output_tokens and result.stop_reason != "abort":
                    reward = -self.args.tw_invalid_action_penalty
                else:
                    reward = 0.0
            else:
                raise ValueError(
                    f"Unsupported invalid_reward_mode: {invalid_reward_mode}"
                )

        state.step_records.append(
            TextWorldStepRecord(
                training_result=result,
                prompt_ids=list(pending.input_ids),
                reward=reward,
            )
        )

    async def _run_textworld_step_batch(
        self,
        states: List[TextWorldTrajectoryState],
        invalid_reward_mode: Literal["zero", "ppo_penalty"],
    ) -> int:
        if self.stopped:
            return 0

        active_states = [
            state
            for state in states
            if not state.done
            and len(state.step_records) < self.args.tw_max_episode_steps
        ]
        if not active_states:
            return 0

        pending = []
        request_refs = []
        for state in active_states:
            prompt_obs = state.obs
            prompt_infos = dict(state.infos)
            input_ids = list(state.transcript_ids)
            if len(input_ids) >= self.args.tw_history_token_window:
                state.done = True
                continue
            infer_max_tokens = min(
                self.args.infer_max_tokens,
                self.args.tw_history_token_window - len(input_ids),
            )
            current_batch_id = self.batch_id
            self.batch_id += 1
            request_refs.append(
                self.infer_actor.request_batch.remote(
                    self.worker_id,
                    current_batch_id,
                    list(input_ids),
                    infer_max_tokens,
                )
            )
            pending.append(
                TextWorldPendingRequest(
                    state=state,
                    prompt_obs=prompt_obs,
                    prompt_infos=prompt_infos,
                    input_ids=input_ids,
                    score_before=state.latest_score,
                )
            )

        if not request_refs:
            return len(active_states)

        generation_results = await asyncio.gather(*request_refs)
        for pending_request, result in zip(pending, generation_results):
            self._apply_textworld_action_result(
                pending_request,
                result,
                invalid_reward_mode=invalid_reward_mode,
            )

        return len(active_states)

    def _build_textworld_episode_rl_sample(
        self,
        step_records: List[TextWorldStepRecord],
        algorithm: Literal["ppo", "grpo"] | None = None,
        trajectory_return: float | None = None,
        sample_reward: float | None = None,
        sample_advantage: float | None = None,
    ) -> RLSample | None:
        algorithm = self.args.rl_algorithm if algorithm is None else algorithm
        input_ids: List[int] = []
        labels: List[int] = []
        old_logprobs: List[float] = []
        token_rewards: List[float] = []
        response_indices: List[int] = []
        response_ids: List[int] = []
        output_versions: List[int] = []
        next_response_index = 0

        for record in step_records:
            prompt_ids = list(record.prompt_ids)
            if len(input_ids) > len(prompt_ids):
                return None
            if prompt_ids[: len(input_ids)] != input_ids:
                return None
            prompt_delta = prompt_ids[len(input_ids):]
            if prompt_delta:
                input_ids.extend(prompt_delta)
                labels.extend([-100] * len(prompt_delta))
                old_logprobs.extend([0.0] * len(prompt_delta))
                token_rewards.extend([0.0] * len(prompt_delta))
                response_indices.extend([-1] * len(prompt_delta))

            result = record.training_result
            if result is None:
                continue
            if not result.output_tokens:
                continue
            output_tokens = list(result.output_tokens)
            if result.stop_reason == "abort":
                input_ids.extend(output_tokens)
                labels.extend([-100] * len(output_tokens))
                old_logprobs.extend([0.0] * len(output_tokens))
                token_rewards.extend([0.0] * len(output_tokens))
                response_indices.extend([-1] * len(output_tokens))
                continue
            result_logprobs = list(result.output_logprobs)
            if len(result_logprobs) != len(result.output_tokens):
                return None
            if len(result.output_versions) != len(result.output_tokens):
                return None

            input_ids.extend(output_tokens)
            labels.extend(output_tokens)
            old_logprobs.extend(result_logprobs)
            if algorithm == "ppo":
                response_token_rewards = [0.0] * len(output_tokens)
                response_token_rewards[-1] = float(record.reward)
            elif algorithm == "grpo":
                response_token_rewards = [0.0] * len(output_tokens)
            else:
                raise ValueError(f"Unsupported rl_algorithm: {algorithm}")
            token_rewards.extend(response_token_rewards)
            response_indices.extend([next_response_index] * len(output_tokens))
            response_ids.extend(output_tokens)
            output_versions.extend(result.output_versions)
            next_response_index += 1

        if not response_ids:
            return None
        if (
            not input_ids
            or len(input_ids) != len(labels)
            or len(input_ids) != len(old_logprobs)
            or len(input_ids) != len(token_rewards)
            or len(input_ids) != len(response_indices)
        ):
            return None
        if len(input_ids) > self.args.tw_history_token_window:
            return None

        episode_return = (
            sum(float(record.reward) for record in step_records)
            if trajectory_return is None
            else float(trajectory_return)
        )
        if algorithm == "ppo":
            token_advantages = self._compute_token_level_advantages(
                labels,
                token_rewards,
            )
            valid_token_advantages = [
                advantage
                for advantage, label in zip(token_advantages, labels)
                if label != -100
            ]
            sample_reward = float(episode_return)
            sample_advantage = (
                sum(valid_token_advantages) / len(valid_token_advantages)
                if valid_token_advantages else 0.0
            )
        elif algorithm == "grpo":
            sample_reward = (
                float(episode_return) if sample_reward is None else float(sample_reward)
            )
            sample_advantage = (
                float(sample_reward)
                if sample_advantage is None
                else float(sample_advantage)
            )
            token_advantages = [
                float(sample_advantage) if label != -100 else 0.0
                for label in labels
            ]
        else:
            raise ValueError(f"Unsupported rl_algorithm: {algorithm}")

        return RLSample(
            algorithm=algorithm,
            response_ids=response_ids,
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            old_logprobs=old_logprobs,
            reward=float(sample_reward),
            advantage=float(sample_advantage),
            token_rewards=token_rewards,
            token_advantages=token_advantages,
            response_indices=response_indices,
            episode_return=float(episode_return),
            output_versions=output_versions,
        )

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
                + self.args.tw_gamma * running_return
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

    async def _run_textworld_grpo_group(
        self,
        game_file: str,
    ) -> None:
        request_infos = make_textworld_request_infos()
        env_id = textworld.gym.register_game(
            game_file,
            request_infos=request_infos,
            max_episode_steps=self.args.tw_max_episode_steps,
        )
        group_size = int(self.args.grpo_group_size)
        states: List[TextWorldTrajectoryState] = []

        try:
            for group_index in range(group_size):
                env = textworld.gym.make(env_id)
                obs, infos = env.reset()
                transcript_ids = self._initial_transcript_ids(obs, infos)
                states.append(
                    TextWorldTrajectoryState(
                        group_index=group_index,
                        env=env,
                        obs=obs,
                        infos=dict(infos),
                        latest_score=_textworld_score(0, infos),
                        transcript_ids=transcript_ids,
                        won=bool(infos.get("won", False)),
                        lost=bool(infos.get("lost", False)),
                    )
                )

            for _ in range(self.args.tw_max_episode_steps):
                active_count = await self._run_textworld_step_batch(
                    states=states,
                    invalid_reward_mode="zero",
                )
                if active_count == 0:
                    break

            raw_returns = [
                sum(record.reward for record in state.step_records)
                for state in states
            ]
            grpo_rewards = [
                raw_return
                - self.args.tw_invalid_action_penalty * state.invalid_actions
                for raw_return, state in zip(raw_returns, states)
            ]
            group_reward_mean, group_reward_std, advantages = (
                self._compute_grpo_group_advantages(grpo_rewards)
            )
            _, advantage_std, _ = self._compute_grpo_group_advantages(advantages)

            samples = []
            for state, raw_return, grpo_reward, advantage in zip(
                states,
                raw_returns,
                grpo_rewards,
                advantages,
            ):
                sample = self._build_textworld_episode_rl_sample(
                    step_records=state.step_records,
                    algorithm="grpo",
                    trajectory_return=raw_return,
                    sample_reward=grpo_reward,
                    sample_advantage=advantage,
                )
                if sample is not None:
                    samples.append(sample)

            if samples:
                self.replay_buffer.add_samples.remote(samples)

            invalid_actions_mean = (
                sum(state.invalid_actions for state in states) / len(states)
                if states else 0.0
            )
            max_score = _textworld_max_score(states[0].infos) if states else 0.0
            for state, grpo_reward in zip(states, grpo_rewards):
                self.stats_actor.add_textworld_episode.remote(
                    self.worker_id,
                    grpo_reward,
                    state.latest_score,
                    max_score,
                    bool(state.won),
                    len(state.step_records),
                    state.env_steps,
                    state.invalid_actions,
                    list(state.selected_response_lengths),
                    grpo_group_reward_mean=group_reward_mean,
                    grpo_group_reward_std=group_reward_std,
                    grpo_advantage_std=advantage_std,
                    grpo_invalid_actions_mean=invalid_actions_mean,
                )

        finally:
            for state in states:
                state.env.close()

    async def _run_textworld_ppo_batch(
        self,
        game_file: str,
    ) -> None:
        request_infos = make_textworld_request_infos()
        env_id = textworld.gym.register_game(
            game_file,
            request_infos=request_infos,
            max_episode_steps=self.args.tw_max_episode_steps,
        )
        batch_size = int(self.args.rollout_batch_size)
        states: List[TextWorldTrajectoryState] = []

        try:
            for batch_index in range(batch_size):
                env = textworld.gym.make(env_id)
                obs, infos = env.reset()
                transcript_ids = self._initial_transcript_ids(obs, infos)
                states.append(
                    TextWorldTrajectoryState(
                        group_index=batch_index,
                        env=env,
                        obs=obs,
                        infos=dict(infos),
                        latest_score=_textworld_score(0, infos),
                        transcript_ids=transcript_ids,
                        won=bool(infos.get("won", False)),
                        lost=bool(infos.get("lost", False)),
                    )
                )

            for _ in range(self.args.tw_max_episode_steps):
                active_count = await self._run_textworld_step_batch(
                    states=states,
                    invalid_reward_mode="ppo_penalty",
                )
                if active_count == 0:
                    break

            samples = []
            sample_by_episode_index = {}
            for state in states:
                sample = self._build_textworld_episode_rl_sample(
                    step_records=state.step_records,
                )
                if sample is not None:
                    samples.append(sample)
                    sample_by_episode_index[state.group_index] = sample

            if samples:
                self.replay_buffer.add_samples.remote(samples)

            invalid_actions_mean = (
                sum(state.invalid_actions for state in states) / len(states)
                if states else 0.0
            )
            max_score = _textworld_max_score(states[0].infos) if states else 0.0
            episode_returns = [
                sum(record.reward for record in state.step_records)
                for state in states
            ]
            for state, episode_return in zip(states, episode_returns):
                sample = sample_by_episode_index.get(state.group_index)
                valid_token_rewards = []
                valid_token_advantages = []
                if sample is not None:
                    valid_token_rewards = [
                        reward
                        for reward, label in zip(sample.token_rewards[1:], sample.labels[1:])
                        if label != -100
                    ]
                    valid_token_advantages = [
                        advantage
                        for advantage, label in zip(
                            sample.token_advantages[1:],
                            sample.labels[1:],
                        )
                        if label != -100
                    ]
                token_reward_mean = (
                    sum(valid_token_rewards) / len(valid_token_rewards)
                    if valid_token_rewards else None
                )
                raw_advantage_mean = (
                    sum(valid_token_advantages) / len(valid_token_advantages)
                    if valid_token_advantages else None
                )
                raw_advantage_std = None
                if valid_token_advantages:
                    assert raw_advantage_mean is not None
                    raw_advantage_std = math.sqrt(
                        sum(
                            (advantage - raw_advantage_mean) ** 2
                            for advantage in valid_token_advantages
                        )
                        / len(valid_token_advantages)
                    )
                self.stats_actor.add_textworld_episode.remote(
                    self.worker_id,
                    episode_return,
                    state.latest_score,
                    max_score,
                    bool(state.won),
                    len(state.step_records),
                    state.env_steps,
                    state.invalid_actions,
                    list(state.selected_response_lengths),
                    ppo_token_reward_mean=token_reward_mean,
                    ppo_raw_advantage_mean=raw_advantage_mean,
                    ppo_raw_advantage_std=raw_advantage_std,
                    ppo_invalid_actions_mean=invalid_actions_mean,
                )

        finally:
            for state in states:
                state.env.close()

    async def run(self) -> Dict[str, int]:
        episode_index = 0
        while not self.stopped:
            game_file = self.game_files[
                (self.worker_id + episode_index) % len(self.game_files)
            ]
            if self.args.rl_algorithm == "ppo":
                await self._run_textworld_ppo_batch(game_file)
            elif self.args.rl_algorithm == "grpo":
                await self._run_textworld_grpo_group(game_file)
            else:
                raise ValueError(
                    f"Unsupported rl_algorithm: {self.args.rl_algorithm}"
                )
            episode_index += 1

        print(
            f"[tw-rollout] worker={self.worker_id} stopped; "
            f"episodes={episode_index}"
        )
        return {"episodes": episode_index}


def summarize_weight_payload(dtype_names: List[str], shapes: List[List[int]]) -> float:
    total_weight_bytes = sum(
        numel_from_shape(shape) * dtype_nbytes(dtype_name)
        for dtype_name, shape in zip(dtype_names, shapes)
    )
    return total_weight_bytes / 1024**3


async def sync_weights_to_vllm(
    infer_actor,
    fsdp_workers,
    scope: str,
    transfer_world_size: int,
    packed: bool = True,
):
    validate_weight_scope(scope)
    names, dtype_names, shapes = ray.get(
        fsdp_workers[0].get_weight_metadata.remote(scope)
    )
    model_gib = summarize_weight_payload(dtype_names, shapes)
    infer_payload_gib = model_gib * (transfer_world_size - 1)
    print(
        f"[sync] {scope} metadata: tensors={len(names)}, "
        f"logical_payload={model_gib:.3f} GiB, "
        f"aggregate_infer_payload={infer_payload_gib:.3f} GiB"
    )

    ray.get(infer_actor.start_weight_update.remote())
    t0 = time.perf_counter()
    broadcast_handles = [
        worker.gather_and_broadcast_weights.remote(scope=scope, packed=packed)
        for worker in fsdp_workers
    ]
    ray.get(
        infer_actor.update_weights.remote(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=packed,
        )
    )
    ray.get(broadcast_handles)
    ray.get(infer_actor.finish_weight_update.remote())
    elapsed = time.perf_counter() - t0
    print(
        f"[sync] {scope} weight update complete: {elapsed:.3f}s, "
        f"model-sync throughput={model_gib / elapsed:.3f} GiB/s, "
        f"aggregate-infer throughput={infer_payload_gib / elapsed:.3f} GiB/s"
    )
    return elapsed


async def shutdown_vllm_engine(engine) -> None:
    """Shut down vLLM workers before Ray is torn down."""
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if shutdown is None:
        return

    try:
        result = shutdown()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            await result
        print("[cleanup] vLLM engine shut down.")
    except Exception as exc:
        print(f"[cleanup] Ignoring vLLM engine shutdown error: {exc!r}")


def save_run_config(args: argparse.Namespace) -> None:
    os.makedirs(args.log_dir, exist_ok=True)

    args_path = os.path.join(args.log_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    command_path = os.path.join(args.log_dir, "command.txt")
    command = format_shell_command(sys.argv)
    with open(command_path, "w", encoding="utf-8") as file:
        file.write(command)
        file.write("\n")


def format_shell_command(argv: List[str]) -> str:
    if not argv:
        return "python"

    command = f"python {shlex.quote(argv[0])}"
    if len(argv) == 1:
        return command

    lines = [f"{command} \\"]
    parts: List[List[str]] = []
    idx = 1
    while idx < len(argv):
        part = argv[idx]
        if part.startswith("-") and idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
            parts.append([part, argv[idx + 1]])
            idx += 2
        else:
            parts.append([part])
            idx += 1

    for idx, part_group in enumerate(parts):
        line = "  " + " ".join(shlex.quote(part) for part in part_group)
        if idx + 1 < len(parts):
            line += " \\"
        lines.append(line)

    return "\n".join(lines)


def checkpoint_tag(optimizer_step: int, suffix: str | None = None) -> str:
    tag = f"step-{optimizer_step:06d}"
    if suffix:
        tag = f"{tag}-{suffix}"
    return tag


def save_fsdp_checkpoint(
    fsdp_workers: List[Any],
    checkpoint_dir: str,
    optimizer_step: int,
    suffix: str | None = None,
    fixed_tag: str | None = None,
) -> str:
    tag = fixed_tag or checkpoint_tag(optimizer_step, suffix=suffix)
    results = ray.get([
        worker.save_checkpoint.remote(checkpoint_dir, tag)
        for worker in fsdp_workers
    ])
    rank0_result = next(result for result in results if result["rank"] == 0)
    return str(rank0_result["checkpoint_dir"])


async def run_textworld_train(args: argparse.Namespace):
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    save_run_config(args)
    writer = SummaryWriter(args.log_dir)
    print(f"[metrics] TensorBoard log dir: {args.log_dir}")
    print(
        "[data] "
        "dataset=textworld "
        f"tw_game_dir={args.tw_game_dir!r} "
        f"tw_game_pattern={args.tw_game_pattern!r} "
        f"tw_game_limit={args.tw_game_limit} "
        f"tw_max_episode_steps={args.tw_max_episode_steps} "
        f"tw_history_token_window={args.tw_history_token_window} "
        f"max_length={args.max_length} "
        f"train_packing={args.train_packing} "
        f"train_token_budget={args.train_token_budget} "
        f"train_pack_candidate_pool_size={args.train_pack_candidate_pool_size} "
        f"train_logprob_mode={args.train_logprob_mode} "
        f"infer_tp_size={args.infer_tp_size} "
        f"infer_size={args.infer_size} "
        f"infer_max_tokens={args.infer_max_tokens} "
        f"vllm_max_model_len={args.vllm_max_model_len} "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"rl_algorithm={args.rl_algorithm} "
        f"grpo_group_size={args.grpo_group_size} "
        f"tw_gamma={args.tw_gamma} "
        f"ppo_normalize_advantages={args.ppo_normalize_advantages} "
        f"tw_invalid_action_penalty={args.tw_invalid_action_penalty} "
        f"tw_lost_penalty={args.tw_lost_penalty}"
    )
    print(
        "[data] TextWorld full-history token-window transcript mode enabled: "
        f"window={args.tw_history_token_window}; rollout requests and "
        "episode-level training samples share one growing token transcript."
    )
    fsdp_workers = []
    replay_buffers = []
    infer_actor = None
    rollout_workers = []
    rollout_refs = []
    try:
        # Use local/shared model weights directly.
        print(f"[init] Loading local model from {args.model_path}")

        # FSDP rendezvous address (single-node)
        fsdp_master_addr = args.fsdp_master_addr or get_local_ip()
        fsdp_master_port = args.fsdp_master_port or find_open_port()

        replay_buffers = [
            ReplayBufferActor.remote(capacity=args.replay_capacity)
            for _ in range(args.fsdp_world_size)
        ]
        stats_actor = StatsActor.remote(
            window_size=args.metrics_window_size,
            active_timeout_seconds=args.metrics_active_timeout_seconds,
        )
        print(
            "[replay] "
            f"Created {len(replay_buffers)} ReplayBufferActor instances "
            f"(capacity={args.replay_capacity} samples each, "
            f"min_replay_size_per_rank={args.min_replay_size_per_rank})."
        )

        # Launch FSDP training workers. Ray allocates 1 GPU per worker; vLLM's
        # internal DP placement groups will land on the remaining GPUs.
        remote_worker = ray.remote(num_gpus=1)(FSDPTrainWorker)
        fsdp_workers = [
            remote_worker.remote(
                args,
                rank,
                args.fsdp_world_size,
                fsdp_master_addr,
                fsdp_master_port,
                replay_buffers[rank],
            )
            for rank in range(args.fsdp_world_size)
        ]
        ray.get([w.get_rank.remote() for w in fsdp_workers])
        print(f"[init] {args.fsdp_world_size} FSDP training workers ready.")

        remote_infer_actor = ray.remote(
            num_gpus=args.infer_tp_size * args.infer_size,
        )(VLLMInferenceActor)
        infer_actor = remote_infer_actor.remote(args)

        remote_rollout_worker = ray.remote(
            num_gpus=0,
            max_concurrency=2, # 允许 run() 正在跑的时候，stop() 还能被执行。
        )(TextWorldRolloutWorkerActor)

        def check_rollout_workers() -> None:
            if not rollout_refs:
                return
            ready, _ = ray.wait(rollout_refs, num_returns=1, timeout=0.0)
            if ready:
                ray.get(ready[0])
                raise RuntimeError("A TextWorldRolloutWorkerActor exited unexpectedly.")

        # --- Weight-transfer setup ---
        print("[transfer] Setting up weight-transfer endpoint...")
        transfer_addr, transfer_port = ray.get(
            fsdp_workers[0].setup_transfer_endpoint.remote()
        )
        print(f"[transfer] Endpoint ready at {transfer_addr}:{transfer_port}")

        transfer_world_size = args.infer_tp_size * args.infer_size + 1
        print(
            f"[transfer] World size: {transfer_world_size} "
            f"(1 trainer + {args.infer_tp_size * args.infer_size} vLLM workers)"
        )

        print("[transfer] Initializing NCCL groups...")
        train_handle = fsdp_workers[0].init_weight_transfer_group.remote(
            transfer_world_size
        )
        ray.get(
            infer_actor.init_weight_transfer_engine.remote(
                master_address=transfer_addr,
                master_port=transfer_port,
                transfer_world_size=transfer_world_size,
            )
        )
        ray.get(train_handle)
        print("[transfer] NCCL groups initialized.")

        print("[sync] Initial full sync from FSDP to vLLM...")
        ray.get(infer_actor.pause_and_wait_idle.remote())
        await sync_weights_to_vllm(
            infer_actor=infer_actor,
            fsdp_workers=fsdp_workers,
            scope="all",
            transfer_world_size=transfer_world_size,
            packed=True,
        )
        ray.get(infer_actor.resume_generation.remote(increment_version=False))
        print("[sync] Initial full sync complete; generation can start.")

        ray.get(infer_actor.start.remote())
        rollout_workers = [
            remote_rollout_worker.remote(
                args,
                infer_actor,
                worker_id,
                replay_buffers[worker_id % args.fsdp_world_size],
                stats_actor,
            )
            for worker_id in range(args.num_rollout_workers)
        ]
        rollout_refs = [worker.run.remote() for worker in rollout_workers]
        print(
            "[rollout] Started rollout workers while trainer runs; "
            f"num_rollout_workers={args.num_rollout_workers} "
            f"num_replay_buffers={len(replay_buffers)} "
            "replay_assignment=worker_id_mod_fsdp_world_size "
            f"rollout_batch_size={args.rollout_batch_size} "
            f"rl_algorithm={args.rl_algorithm} "
            f"grpo_group_size={args.grpo_group_size} "
            f"tw_gamma={args.tw_gamma} "
            f"ppo_normalize_advantages={args.ppo_normalize_advantages} "
            f"infer_tp_size={args.infer_tp_size} "
            f"infer_size={args.infer_size} "
            f"infer_max_tokens={args.infer_max_tokens} "
            f"infer_temperature={args.infer_temperature} "
            f"infer_top_p={args.infer_top_p} "
            f"vllm_max_num_seqs={args.vllm_max_num_seqs} "
            f"vllm_max_num_batched_tokens={args.vllm_max_num_batched_tokens} "
            f"vllm_max_model_len={args.vllm_max_model_len}."
        )

        sync_rounds = 0
        last_sync_elapsed_seconds = None
        latest_optimizer_step = 0
        last_checkpoint_step = None
        training_reached_max = False
        while not training_reached_max:
            check_rollout_workers()
            print(
                "[train] Launching trainer segment: "
                f"sync_every_optimizer_steps={args.sync_every_optimizer_steps}"
            )
            infer_stats_start = ray.get(infer_actor.get_stats.remote())
            infer_t0 = time.perf_counter()
            train_segment_start_time = time.perf_counter()
            train_handles = [
                worker.train_until_next_sync.remote(
                    args.sync_every_optimizer_steps
                )
                for worker in fsdp_workers
            ]
            train_future = asyncio.create_task(
                asyncio.to_thread(ray.get, train_handles)
            )
            summaries = await train_future
            train_segment_elapsed = time.perf_counter() - train_segment_start_time
            infer_elapsed = time.perf_counter() - infer_t0
            infer_stats_end = ray.get(infer_actor.get_stats.remote())
            check_rollout_workers()
            rank0_summary = next(item for item in summaries if item["rank"] == 0)
            training_reached_max = bool(rank0_summary["reached_max_steps"])
            latest_optimizer_step = int(rank0_summary["optimizer_step"])
            if rank0_summary["optimizer_steps_run"] <= 0:
                print("[train] No optimizer steps left; stopping sync loop.")
                break

            print(
                "[train] Trainer segment complete: "
                f"optimizer_step={rank0_summary['optimizer_step']} "
                f"steps_run={rank0_summary['optimizer_steps_run']} "
                f"last_loss={rank0_summary['last_loss']:.6f} "
                f"reward_mean={rank0_summary['last_reward_mean']:.4f} "
                f"adv_mean={rank0_summary['last_advantage_mean']:.4f} "
                f"response_tokens={rank0_summary['last_response_tokens']:.0f} "
                f"replay_size={rank0_summary['last_replay_size']} "
                "replay_candidates_fetched="
                f"{rank0_summary['last_total_sampled']} "
                f"ppo_clip_frac={rank0_summary['last_ppo_clip_frac']:.4f} "
                f"token_reward_mean={rank0_summary['last_token_reward_mean']:.4f} "
                f"raw_adv_mean={rank0_summary['last_raw_advantage_mean']:.4f} "
                f"raw_adv_std={rank0_summary['last_raw_advantage_std']:.4f} "
                f"used_adv_mean={rank0_summary['last_used_advantage_mean']:.4f} "
                f"used_adv_std={rank0_summary['last_used_advantage_std']:.4f} "
                "old_new_kl_k3_token_mean="
                f"{rank0_summary['last_old_new_kl_k3_token_mean']:.6f} "
                "old_new_kl_k3_sample_sum_mean="
                f"{rank0_summary['last_old_new_kl_k3_sample_sum_mean']:.6f} "
                "old_new_kl_k3_loss="
                f"{rank0_summary['last_old_new_kl_k3_loss']:.6f} "
                f"pack_utilization={rank0_summary['last_pack_token_utilization']:.4f} "
                f"pack_samples={rank0_summary['last_pack_sample_count']:.0f} "
                "pack_max_seqlen="
                f"{rank0_summary['last_pack_max_sequence_length']:.0f} "
                f"pack_cpu_ms={rank0_summary['last_pack_cpu_seconds'] * 1000.0:.3f}"
            )
            infer_delta_tokens = (
                infer_stats_end["total_tokens"] - infer_stats_start["total_tokens"]
            )
            infer_delta_requests = (
                infer_stats_end["total_requests"]
                - infer_stats_start["total_requests"]
            )
            infer_tokens_per_sec = infer_delta_tokens / max(infer_elapsed, 1e-9)
            infer_requests_per_sec = infer_delta_requests / max(infer_elapsed, 1e-9)
            print(
                "[infer-throughput] "
                f"sync_round={sync_rounds} "
                f"optimizer_step={rank0_summary['optimizer_step']} "
                f"elapsed={infer_elapsed:.3f}s "
                f"tokens={infer_delta_tokens} "
                f"requests={infer_delta_requests} "
                f"tokens_per_sec={infer_tokens_per_sec:.2f} "
                f"requests_per_sec={infer_requests_per_sec:.2f} "
                f"total_tokens={infer_stats_end['total_tokens']} "
                f"total_requests={infer_stats_end['total_requests']}"
            )
            replay_stats = ray.get([
                worker.get_replay_stats.remote()
                for worker in fsdp_workers
            ])
            rollout_stats = ray.get(stats_actor.get_stats.remote())
            replay_sizes = ",".join(str(stats["size"]) for stats in replay_stats)
            replay_received = ",".join(
                str(stats["total_samples_added"])
                for stats in replay_stats
            )
            replay_sampled = ",".join(
                str(stats["total_candidates_fetched"])
                for stats in replay_stats
            )
            replay_evicted = ",".join(
                str(stats["total_samples_evicted"])
                for stats in replay_stats
            )
            print(
                "[replay] "
                f"sync_round={sync_rounds} "
                f"sizes=[{replay_sizes}] "
                f"total_received=[{replay_received}] "
                f"total_candidates_fetched=[{replay_sampled}] "
                f"total_evicted=[{replay_evicted}]"
            )
            print(
                "[tw-train-stats] "
                f"episodes={rollout_stats['tw_episode_count']:.0f} "
                f"win_rate={rollout_stats['tw_win_rate']:.4f} "
                f"normalized_score={rollout_stats['tw_normalized_score']:.4f} "
                f"invalid_action_rate={rollout_stats['tw_invalid_action_rate']:.4f} "
                f"env_steps_mean={rollout_stats['tw_env_steps_mean']:.2f} "
                "selected_response_length_mean="
                f"{rollout_stats['tw_selected_response_length_mean']:.2f}"
            )
            if args.rl_algorithm == "ppo":
                print(
                    "[ppo-rollout-stats] "
                    f"token_reward_mean={rollout_stats['ppo_token_reward_mean']:.4f} "
                    f"raw_advantage_mean={rollout_stats['ppo_raw_advantage_mean']:.4f} "
                    f"raw_advantage_std={rollout_stats['ppo_raw_advantage_std']:.4f} "
                    "invalid_actions_mean="
                    f"{rollout_stats['ppo_invalid_actions_mean']:.2f}"
                )
            elif args.rl_algorithm == "grpo":
                print(
                    "[grpo-rollout-stats] "
                    f"group_reward_mean={rollout_stats['grpo_group_reward_mean']:.4f} "
                    f"group_reward_std={rollout_stats['grpo_group_reward_std']:.4f} "
                    f"advantage_std={rollout_stats['grpo_advantage_std']:.4f} "
                    "invalid_actions_mean="
                    f"{rollout_stats['grpo_invalid_actions_mean']:.2f}"
                )
            total_replay_size = sum(int(stats["size"]) for stats in replay_stats)
            total_replay_capacity = sum(int(stats["capacity"]) for stats in replay_stats)
            replay_fill_ratio = (
                total_replay_size / total_replay_capacity
                if total_replay_capacity > 0 else 0.0
            )
            train_loss_mean = (
                sum(float(summary["segment_loss_mean"]) for summary in summaries)
                / len(summaries)
            )
            episode_return_mean = (
                sum(float(summary["segment_episode_return_mean"]) for summary in summaries)
                / len(summaries)
            )
            old_new_kl_k3_token_mean = (
                sum(
                    float(summary["segment_old_new_kl_k3_token_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            old_new_kl_k3_sample_sum_mean = (
                sum(
                    float(summary["segment_old_new_kl_k3_sample_sum_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            old_new_kl_k3_loss = (
                sum(
                    float(summary["segment_old_new_kl_k3_loss"])
                    for summary in summaries
                )
                / len(summaries)
            )
            version_lag_mean = (
                sum(
                    float(summary["train_sample_trainer_version_lag_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            pack_token_utilization_mean = sum(
                float(summary["segment_pack_token_utilization_mean"])
                for summary in summaries
            ) / len(summaries)
            pack_sample_count_mean = sum(
                float(summary["segment_pack_sample_count_mean"])
                for summary in summaries
            ) / len(summaries)
            pack_max_sequence_length_mean = sum(
                float(summary["segment_pack_max_sequence_length_mean"])
                for summary in summaries
            ) / len(summaries)
            pack_cpu_seconds_mean = sum(
                float(summary["segment_pack_cpu_seconds_mean"])
                for summary in summaries
            ) / len(summaries)
            optimizer_steps_per_sec = (
                rank0_summary["optimizer_steps_run"] / max(train_segment_elapsed, 1e-9)
            )
            segment_global_valid_response_tokens = float(
                rank0_summary.get("segment_global_valid_response_tokens", 0.0)
            )
            valid_response_tokens_per_sec = (
                segment_global_valid_response_tokens
                / max(train_segment_elapsed, 1e-9)
            )
            tb_step = rank0_summary["optimizer_step"]
            writer.add_scalar(
                "Rollout/GlobalRewardSumMean",
                rollout_stats["global_reward_sum_mean"],
                tb_step,
            )
            writer.add_scalar(
                "TextWorld/WinRate",
                rollout_stats["tw_win_rate"],
                tb_step,
            )
            writer.add_scalar(
                "TextWorld/NormalizedScore",
                rollout_stats["tw_normalized_score"],
                tb_step,
            )
            writer.add_scalar(
                "TextWorld/InvalidActionRate",
                rollout_stats["tw_invalid_action_rate"],
                tb_step,
            )
            writer.add_scalar(
                "TextWorld/EnvStepsMean",
                rollout_stats["tw_env_steps_mean"],
                tb_step,
            )
            if args.rl_algorithm == "ppo":
                writer.add_scalar(
                    "PPO/RolloutTokenRewardMean",
                    rollout_stats["ppo_token_reward_mean"],
                    tb_step,
                )
                writer.add_scalar(
                    "PPO/RolloutRawAdvantageMean",
                    rollout_stats["ppo_raw_advantage_mean"],
                    tb_step,
                )
                writer.add_scalar(
                    "PPO/RolloutRawAdvantageStd",
                    rollout_stats["ppo_raw_advantage_std"],
                    tb_step,
                )
                writer.add_scalar(
                    "PPO/RolloutInvalidActionsMean",
                    rollout_stats["ppo_invalid_actions_mean"],
                    tb_step,
                )
            elif args.rl_algorithm == "grpo":
                writer.add_scalar(
                    "GRPO/GroupRewardMean",
                    rollout_stats["grpo_group_reward_mean"],
                    tb_step,
                )
                writer.add_scalar(
                    "GRPO/GroupRewardStd",
                    rollout_stats["grpo_group_reward_std"],
                    tb_step,
                )
                writer.add_scalar(
                    "GRPO/AdvantageStd",
                    rollout_stats["grpo_advantage_std"],
                    tb_step,
                )
                writer.add_scalar(
                    "GRPO/InvalidActionsMean",
                    rollout_stats["grpo_invalid_actions_mean"],
                    tb_step,
                )
            writer.add_scalar(
                "Rollout/ActiveWorkers",
                rollout_stats["active_workers"],
                tb_step,
            )
            writer.add_scalar("Replay/FillRatio", replay_fill_ratio, tb_step)
            writer.add_scalar(
                "Replay/TrainSampleTrainerVersionLagMean",
                version_lag_mean,
                tb_step,
            )
            writer.add_scalar("Train/LossMeanAcrossRanks", train_loss_mean, tb_step)
            if args.rl_algorithm == "ppo":
                writer.add_scalar(
                    "PPO/TrainEpisodeReturnMeanAcrossRanks",
                    episode_return_mean,
                    tb_step,
                )
            elif args.rl_algorithm == "grpo":
                writer.add_scalar(
                    "GRPO/TrainEpisodeReturnMeanAcrossRanks",
                    episode_return_mean,
                    tb_step,
                )
            writer.add_scalar(
                "KL/OldNewK3TokenMean",
                old_new_kl_k3_token_mean,
                tb_step,
            )
            writer.add_scalar(
                "KL/OldNewK3SampleSumMean",
                old_new_kl_k3_sample_sum_mean,
                tb_step,
            )
            writer.add_scalar(
                "KL/OldNewK3Loss",
                old_new_kl_k3_loss,
                tb_step,
            )
            writer.add_scalar(
                "Train/LearningRate",
                rank0_summary["learning_rate"],
                tb_step,
            )
            writer.add_scalar(
                "Train/OptimizerStepsPerSec",
                optimizer_steps_per_sec,
                tb_step,
            )
            if args.train_packing == "varlen":
                writer.add_scalar(
                    "Train/PolicyLossTokenMean",
                    rank0_summary["segment_policy_loss_token_mean"],
                    tb_step,
                )
                writer.add_scalar(
                    "Train/SegmentGlobalValidResponseTokens",
                    segment_global_valid_response_tokens,
                    tb_step,
                )
                writer.add_scalar(
                    "Train/SegmentGlobalTrainingSamples",
                    rank0_summary["segment_global_training_samples"],
                    tb_step,
                )
                writer.add_scalar(
                    "Train/GlobalValidResponseTokensPerOptimizerStep",
                    segment_global_valid_response_tokens
                    / rank0_summary["optimizer_steps_run"],
                    tb_step,
                )
                writer.add_scalar(
                    "Train/GlobalTrainingSamplesPerOptimizerStep",
                    rank0_summary["segment_global_training_samples"]
                    / rank0_summary["optimizer_steps_run"],
                    tb_step,
                )
                writer.add_scalar(
                    "Train/CumulativeGlobalValidResponseTokens",
                    rank0_summary[
                        "cumulative_global_valid_response_tokens"
                    ],
                    tb_step,
                )
                writer.add_scalar(
                    "Train/CumulativeGlobalTrainingSamples",
                    rank0_summary["cumulative_global_training_samples"],
                    tb_step,
                )
                writer.add_scalar(
                    "TextWorld/NormalizedScoreByValidResponseToken",
                    rollout_stats["tw_normalized_score"],
                    rank0_summary[
                        "cumulative_global_valid_response_tokens"
                    ],
                )
                writer.add_scalar(
                    "TextWorld/NormalizedScoreByTrainingSample",
                    rollout_stats["tw_normalized_score"],
                    rank0_summary["cumulative_global_training_samples"],
                )
                writer.add_scalar(
                    "Train/ValidResponseTokensPerSec",
                    valid_response_tokens_per_sec,
                    tb_step,
                )
                writer.add_scalar(
                    "Replay/CandidatesFetched",
                    sum(
                        int(summary["total_replay_candidates_fetched"])
                        for summary in summaries
                    ),
                    tb_step,
                )
                writer.add_scalar(
                    "Replay/ValidSamplesPrepared",
                    sum(
                        int(summary["total_valid_samples_prepared"])
                        for summary in summaries
                    ),
                    tb_step,
                )
                writer.add_scalar(
                    "Replay/TrainingSamplesConsumed",
                    sum(
                        int(summary["total_training_samples_consumed"])
                        for summary in summaries
                    ),
                    tb_step,
                )
                writer.add_scalar(
                    "Replay/InvalidCandidates",
                    sum(
                        int(summary["total_invalid_candidates"])
                        for summary in summaries
                    ),
                    tb_step,
                )
                writer.add_scalar(
                    "Train/PackTokenUtilization",
                    pack_token_utilization_mean,
                    tb_step,
                )
                writer.add_scalar(
                    "Train/PackSampleCount",
                    pack_sample_count_mean,
                    tb_step,
                )
                writer.add_scalar(
                    "Train/PackMaxSequenceLength",
                    pack_max_sequence_length_mean,
                    tb_step,
                )
                writer.add_scalar(
                    "Train/PackCpuMilliseconds",
                    pack_cpu_seconds_mean * 1000.0,
                    tb_step,
                )
            if args.clip_mode == "ppo":
                writer.add_scalar(
                    "Clip/PPOClipFrac",
                    (
                        rank0_summary["segment_ppo_clip_frac"]
                        if args.train_packing == "varlen"
                        else rank0_summary["last_ppo_clip_frac"]
                    ),
                    tb_step,
                )
            writer.add_scalar("Infer/TokensPerSec", infer_tokens_per_sec, tb_step)
            writer.flush()

            if (
                args.save_checkpoint
                and args.checkpoint_every_sync_rounds > 0
                and (sync_rounds + 1) % args.checkpoint_every_sync_rounds == 0
            ):
                checkpoint_path = save_fsdp_checkpoint(
                    fsdp_workers=fsdp_workers,
                    checkpoint_dir=args.checkpoint_dir,
                    optimizer_step=latest_optimizer_step,
                    fixed_tag=args.checkpoint_name,
                )
                last_checkpoint_step = latest_optimizer_step
                print(f"[checkpoint] Periodic checkpoint ready: {checkpoint_path}")

            if (
                args.max_sync_rounds is not None
                and sync_rounds >= args.max_sync_rounds
            ):
                print(
                    "[sync] max_sync_rounds reached; letting inference finish "
                    "without more trainable updates."
                )
                break

            sync_rounds += 1
            print(
                f"[sync] Round {sync_rounds}: pausing generation for "
                "trainable-only weight update..."
            )
            ray.get(infer_actor.pause_and_wait_idle.remote())

            last_sync_elapsed_seconds = await sync_weights_to_vllm(
                infer_actor=infer_actor,
                fsdp_workers=fsdp_workers,
                scope="trainable",
                transfer_world_size=transfer_world_size,
                packed=True,
            )
            writer.add_scalar("Sync/ElapsedSeconds", last_sync_elapsed_seconds, tb_step)
            writer.flush()
            next_version = ray.get(
                infer_actor.resume_generation.remote(increment_version=True)
            )
            print(
                f"[sync] Round {sync_rounds}: resumed generation with "
                f"weight version {next_version}."
            )

        print(
            "[rollout] Trainer finished or sync loop stopped; stopping "
            "rollout workers after their current batch."
        )
        ray.get([worker.stop.remote() for worker in rollout_workers])
        ready, pending_rollouts = ray.wait(
            rollout_refs,
            num_returns=len(rollout_refs),
            timeout=args.rollout_stop_timeout,
        )
        if pending_rollouts:
            print(
                "[cleanup] Cancelling rollout worker run refs: "
                f"ready={len(ready)} pending={len(pending_rollouts)}"
            )
            for ref in pending_rollouts:
                try:
                    ray.cancel(ref)
                except Exception as exc:
                    print(f"[cleanup] Ignoring rollout cancel error: {exc!r}")
        if (
            args.save_checkpoint
            and latest_optimizer_step > 0
            and last_checkpoint_step != latest_optimizer_step
        ):
            checkpoint_path = save_fsdp_checkpoint(
                fsdp_workers=fsdp_workers,
                checkpoint_dir=args.checkpoint_dir,
                optimizer_step=latest_optimizer_step,
                suffix="final",
                fixed_tag=args.checkpoint_name,
            )
            last_checkpoint_step = latest_optimizer_step
            print(f"[checkpoint] Final checkpoint ready: {checkpoint_path}")
    finally:
        if rollout_workers:
            try:
                ray.get([worker.stop.remote() for worker in rollout_workers])
            except Exception as exc:
                print(f"[cleanup] Ignoring rollout stop error: {exc!r}")
        for ref in rollout_refs:
            try:
                ray.cancel(ref)
            except Exception as exc:
                print(f"[cleanup] Ignoring rollout cancel error: {exc!r}")
        if infer_actor is not None:
            try:
                ray.get(infer_actor.shutdown.remote())
            except Exception as exc:
                print(f"[cleanup] Ignoring InferActor shutdown error: {exc!r}")
        if fsdp_workers:
            try:
                ray.get([worker.close.remote() for worker in fsdp_workers])
            except Exception as exc:
                print(f"[cleanup] Ignoring FSDP worker close error: {exc!r}")
        writer.close()
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal Ray FSDP trainer + vLLM interruptible inference "
            "NCCL weight-sync demo."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Required local HuggingFace model path.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--train-mode",
        default="full",
        choices=("lm_head", "last_layer", "full"),
        help="Default lm_head mode is intended to validate the training loop.",
    )
    parser.add_argument(
        "--tw-game-dir",
        required=True,
        help="Directory containing TextWorld .z8 games.",
    )
    parser.add_argument(
        "--tw-game-pattern",
        default="*.z8",
        help="Glob pattern for TextWorld games under --tw-game-dir.",
    )
    parser.add_argument(
        "--tw-game-limit",
        type=int,
        default=None,
        help="Optional maximum number of TextWorld games to load.",
    )
    parser.add_argument(
        "--tw-max-episode-steps",
        type=int,
        default=20,
        help="Maximum TextWorld environment steps per episode.",
    )
    parser.add_argument(
        "--tw-history-token-window",
        type=int,
        default=2048,
        help=(
            "Total token window for TextWorld full-history episode transcripts."
        ),
    )
    parser.add_argument(
        "--rl-algorithm",
        type=str,
        default="grpo",
        choices=("ppo", "grpo"),
        help=(
            "RL rollout/advantage algorithm. 'ppo' uses token-level Monte Carlo "
            "advantages; 'grpo' uses group-normalized trajectory advantages."
        ),
    )
    parser.add_argument(
        "--tw-gamma",
        type=float,
        default=1.0,
        help="Discount factor for token-level Monte Carlo reward-to-go.",
    )
    parser.add_argument(
        "--tw-step-penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from each TextWorld environment step reward.",
    )
    parser.add_argument(
        "--tw-win-bonus",
        type=float,
        default=1.0,
        help="Extra reward added to the terminal winning TextWorld step.",
    )
    parser.add_argument(
        "--tw-lost-penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from the terminal losing TextWorld step.",
    )
    parser.add_argument(
        "--tw-invalid-action-penalty",
        type=float,
        default=0.0,
        help=(
            "Penalty assigned to the last token of a non-aborted invalid "
            "TextWorld action response for PPO, or subtracted from the "
            "trajectory reward per invalid action for GRPO."
        ),
    )
    parser.add_argument(
        "--grpo-group-size",
        type=int,
        default=None,
        help=(
            "Number of complete trajectories sampled per GRPO group. Defaults "
            "to --rollout-batch-size."
        ),
    )
    parser.add_argument(
        "--grpo-adv-eps",
        type=float,
        default=1e-8,
        help="Epsilon used when normalizing GRPO group advantages.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--train-packing",
        choices=["padded", "varlen"],
        default="padded",
        help=(
            "Trainer batch layout. 'varlen' flattens independently sampled "
            "RLSamples, uses FlashAttention 2 sequence boundaries, and "
            "optimizes a global valid-response-token weighted objective."
        ),
    )
    parser.add_argument(
        "--train-token-budget",
        type=int,
        default=None,
        help=(
            "Maximum real tokens in one varlen training micro-batch. Required "
            "with --train-packing varlen and must be >= --max-length."
        ),
    )
    parser.add_argument(
        "--train-pack-candidate-pool-size",
        type=int,
        default=None,
        help=(
            "Replay candidates retained locally for length-aware varlen packing. "
            "Defaults to 4 * --batch-size."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum optimizer steps to run before stopping the demo.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=500,
        help=(
            "Optimizer steps used for linear learning-rate warmup before "
            "cosine decay to 0."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=8,
        help=(
            "Gradient accumulation microsteps. In varlen mode this is the "
            "fixed number of packs prepared per rank for each optimizer step."
        ),
    )
    parser.add_argument(
        "--train-logprob-mode",
        type=str,
        default="full_logits_ce",
        choices=["full_logits_ce", "response_only_lm_head"],
        help=(
            "How the trainer computes per-token logprobs. "
            "'full_logits_ce' keeps the standard model forward but avoids "
            "materializing full log_softmax; "
            "'response_only_lm_head' passes packed prediction indices through "
            "the model-native logits_to_keep API and requires varlen packing."
        ),
    )
    parser.add_argument(
        "--clip-mode",
        type=str,
        default="ppo",
        choices=["ppo", "gipo", "sapo"],
        help="Policy objective clipping mode.",
    )
    parser.add_argument(
        "--ppo-normalize-advantages",
        action="store_true",
        help=(
            "Normalize token advantages over valid response tokens in each "
            "training micro-batch before computing the policy loss."
        ),
    )
    parser.add_argument(
        "--ppo-adv-norm-eps",
        type=float,
        default=1e-8,
        help="Epsilon used by --ppo-normalize-advantages.",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="PPO clipping epsilon and diagnostic outside-clip threshold.",
    )
    parser.add_argument(
        "--old-new-kl-coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient for the KL(old || new) k3 token-mean penalty. "
            "0 disables the penalty while keeping KL metrics."
        ),
    )
    parser.add_argument(
        "--gipo-sigma",
        type=float,
        default=1.0,
        help="Sigma for GIPO log-Gaussian soft clipping.",
    )
    parser.add_argument(
        "--sapo-tau-pos",
        type=float,
        default=1.0,
        help="SAPO gate temperature for positive advantages.",
    )
    parser.add_argument(
        "--sapo-tau-neg",
        type=float,
        default=2.0,
        help="SAPO gate temperature for negative advantages.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=None,
        help=(
            "Maximum number of RL samples kept in each ReplayBufferActor. "
            "Defaults to batch_size * grad_accum_steps * 4."
        ),
    )
    parser.add_argument(
        "--replay-wait-sleep-seconds",
        type=float,
        default=0.2,
        help="Seconds a trainer waits between ReplayBufferActor sample polls.",
    )
    parser.add_argument(
        "--replay-sample-timeout-seconds",
        type=float,
        default=None,
        help="Maximum seconds to wait for replay samples; 0 means wait forever.",
    )
    parser.add_argument(
        "--min-replay-size-per-rank",
        type=int,
        default=None,
        help=(
            "Minimum samples required in each trainer's replay buffer before "
            "that trainer starts sampling. Defaults to --batch-size."
        ),
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--metrics-window-size",
        type=int,
        default=1000,
        help="Sliding window size for rollout metrics.",
    )
    parser.add_argument(
        "--metrics-active-timeout-seconds",
        type=float,
        default=600.0,
        help="Seconds before a rollout worker is considered inactive.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="TensorBoard log directory. Defaults to runs/TextWorld_FSDP/<timestamp>.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help=(
            "Save a HuggingFace-format full-model checkpoint from FSDP rank 0. "
            "All FSDP ranks participate in parameter all-gather."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint root directory. Defaults to <log-dir>/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="latest",
        help=(
            "Checkpoint subdirectory name under --checkpoint-dir. The default "
            "'latest' overwrites the previous model instead of creating "
            "step-XXXXXX directories. Set to an empty string to keep per-step "
            "checkpoint directories."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-sync-rounds",
        type=int,
        default=0,
        help=(
            "Save a periodic checkpoint every N completed train/sync segments. "
            "0 disables periodic saves; --save-checkpoint still saves at the end."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp-world-size", type=int, default=6)
    parser.add_argument("--fsdp-master-addr", default=None)
    parser.add_argument("--fsdp-master-port", type=int, default=None)
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray cluster address. Defaults to local ray.init().",
    )
    parser.add_argument(
        "--sync-every-optimizer-steps",
        type=int,
        default=8,
        help="Sync trainable weights after this many optimizer steps.",
    )
    parser.add_argument(
        "--infer-size",
        type=int,
        default=2,
        help="vLLM inference data-parallel size.",
    )
    parser.add_argument(
        "--infer-tp-size",
        type=int,
        default=1,
        help="vLLM inference tensor-parallel size.",
    )
    parser.add_argument(
        "--infer-max-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens per TextWorld action prompt.",
    )
    parser.add_argument(
        "--infer-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for rollout generation.",
    )
    parser.add_argument(
        "--infer-top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p for rollout generation.",
    )
    parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=None,
        help=(
            "Number of CPU TextWorldRolloutWorkerActor instances to launch. "
            "Defaults to --fsdp-world-size."
        ),
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help=(
            "Number of parallel TextWorld PPO episodes per rollout worker batch; "
            "also the default GRPO group size."
        ),
    )
    parser.add_argument(
        "--rollout-stop-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for TextWorldRolloutWorkerActor run loops to stop cleanly.",
    )
    # vllm 最多同时调度多少条sequence，也就是最多256个active请求
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences vLLM may schedule concurrently.",
    )
    # vllm 最多同时调度多少个batched tokens，也就是最多16384个batched tokens
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum number of batched tokens vLLM may schedule.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help=(
            "Optional vLLM max_model_len override. When unset, vLLM infers "
            "the model context length from the model config."
        ),
    )
    parser.add_argument(
        "--disable-vllm-prefix-caching",
        action="store_true",
        help="Disable vLLM automatic prefix caching for history prompts.",
    )
    # 可选的最大同步轮数，超过这个轮数后即使训练还没结束也停止同步，让推理继续跑下去，适合验证推理在不同版本权重下的表现差异
    parser.add_argument(
        "--max-sync-rounds",
        type=int,
        default=None,
        help="Optional maximum number of trainable-only sync rounds in the demo.",
    )
    args = parser.parse_args()
    if args.clip_eps <= 0:
        raise ValueError(f"--clip-eps must be > 0, got {args.clip_eps}")
    if args.old_new_kl_coef < 0:
        raise ValueError(
            "--old-new-kl-coef must be >= 0, got "
            f"{args.old_new_kl_coef}"
        )
    if args.gipo_sigma <= 0:
        raise ValueError(f"--gipo-sigma must be > 0, got {args.gipo_sigma}")
    if args.sapo_tau_pos <= 0 or args.sapo_tau_neg <= 0:
        raise ValueError(
            "--sapo-tau-pos and --sapo-tau-neg must be > 0, got "
            f"{args.sapo_tau_pos}, {args.sapo_tau_neg}"
        )
    if args.replay_capacity is None:
        # args.replay_capacity = args.batch_size * 4
        args.replay_capacity = args.batch_size * args.grad_accum_steps * 4
    if args.train_pack_candidate_pool_size is None:
        args.train_pack_candidate_pool_size = args.batch_size * 4
    if args.replay_sample_timeout_seconds is None:
        args.replay_sample_timeout_seconds = 0.0
    if args.min_replay_size_per_rank is None:
        args.min_replay_size_per_rank = args.batch_size
    if args.num_rollout_workers is None:
        args.num_rollout_workers = args.fsdp_world_size
    if args.grpo_group_size is None:
        args.grpo_group_size = args.rollout_batch_size
    if args.log_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join("runs", "TextWorld_FSDP", timestamp)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    return args


def validate_varlen_training_options(args: argparse.Namespace) -> None:
    if args.train_packing != "varlen":
        return
    if args.ppo_normalize_advantages:
        raise ValueError(
            "--ppo-normalize-advantages is not supported with "
            "--train-packing varlen because the Varlen baseline uses a "
            "global valid-response-token weighted objective."
        )


def validate_args(args: argparse.Namespace) -> None:
    validate_varlen_training_options(args)
    if args.tw_game_limit is not None and args.tw_game_limit < 1:
        raise ValueError("--tw-game-limit must be >= 1 when set")
    if args.tw_max_episode_steps < 1:
        raise ValueError("--tw-max-episode-steps must be >= 1")
    if args.tw_history_token_window < 1:
        raise ValueError("--tw-history-token-window must be >= 1")
    if args.tw_gamma < 0:
        raise ValueError("--tw-gamma must be >= 0")
    if args.max_length < args.tw_history_token_window:
        raise ValueError(
            "--max-length must be >= --tw-history-token-window when "
            "TextWorld token-window transcript mode is enabled"
        )
    load_textworld_game_files(args)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.learning_rate < 0:
        raise ValueError("--learning-rate must be >= 0")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.train_pack_candidate_pool_size < 1:
        raise ValueError("--train-pack-candidate-pool-size must be >= 1")
    if args.train_packing == "varlen":
        if args.train_token_budget is None:
            raise ValueError(
                "--train-token-budget is required with --train-packing varlen"
            )
        if args.train_token_budget < args.max_length:
            raise ValueError(
                "--train-token-budget must be >= --max-length in varlen mode; "
                f"got {args.train_token_budget} < {args.max_length}"
            )
        if args.dtype == "float32":
            raise ValueError(
                "FlashAttention 2 varlen training requires float16, bfloat16, "
                "or auto dtype"
            )
    elif args.train_token_budget is not None and args.train_token_budget < 1:
        raise ValueError("--train-token-budget must be >= 1 when set")
    if (
        args.train_logprob_mode == "response_only_lm_head"
        and args.train_packing != "varlen"
    ):
        raise ValueError(
            "--train-logprob-mode response_only_lm_head requires "
            "--train-packing varlen so tensor logits_to_keep can address "
            "per-sample prediction positions"
        )
    if args.replay_capacity < 1:
        raise ValueError("--replay-capacity must be >= 1")
    if args.min_replay_size_per_rank < 1:
        raise ValueError("--min-replay-size-per-rank must be >= 1")
    if args.min_replay_size_per_rank > args.replay_capacity:
        raise ValueError(
            "--min-replay-size-per-rank must be <= --replay-capacity; "
            f"got {args.min_replay_size_per_rank} > {args.replay_capacity}"
        )
    if args.replay_wait_sleep_seconds <= 0:
        raise ValueError("--replay-wait-sleep-seconds must be > 0")
    if args.replay_sample_timeout_seconds < 0:
        raise ValueError("--replay-sample-timeout-seconds must be >= 0")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    if args.metrics_window_size < 1:
        raise ValueError("--metrics-window-size must be >= 1")
    if args.metrics_active_timeout_seconds <= 0:
        raise ValueError("--metrics-active-timeout-seconds must be > 0")
    if args.checkpoint_every_sync_rounds < 0:
        raise ValueError("--checkpoint-every-sync-rounds must be >= 0")
    if args.fsdp_world_size < 1:
        raise ValueError("--fsdp-world-size must be >= 1")
    if args.sync_every_optimizer_steps < 1:
        raise ValueError("--sync-every-optimizer-steps must be >= 1")
    if args.infer_size < 1:
        raise ValueError("--infer-size must be >= 1")
    if args.infer_tp_size < 1:
        raise ValueError("--infer-tp-size must be >= 1")
    if args.infer_max_tokens < 1:
        raise ValueError("--infer-max-tokens must be >= 1")
    if args.infer_temperature < 0.0:
        raise ValueError("--infer-temperature must be >= 0")
    if not 0.0 < args.infer_top_p <= 1.0:
        raise ValueError("--infer-top-p must be in (0, 1]")
    if args.num_rollout_workers < 1:
        raise ValueError("--num-rollout-workers must be >= 1")
    if args.num_rollout_workers < args.fsdp_world_size:
        raise ValueError(
            "TextWorld training requires --num-rollout-workers >= "
            "--fsdp-world-size so every trainer replay buffer receives samples; "
            f"got num_rollout_workers={args.num_rollout_workers}, "
            f"fsdp_world_size={args.fsdp_world_size}"
        )
    if args.rollout_batch_size < 1:
        raise ValueError("--rollout-batch-size must be >= 1")
    if args.grpo_group_size < 1:
        raise ValueError("--grpo-group-size must be >= 1")
    if args.grpo_adv_eps <= 0:
        raise ValueError("--grpo-adv-eps must be > 0")
    if args.ppo_adv_norm_eps <= 0:
        raise ValueError("--ppo-adv-norm-eps must be > 0")
    if args.tw_invalid_action_penalty < 0:
        raise ValueError("--tw-invalid-action-penalty must be >= 0")
    if args.tw_lost_penalty < 0:
        raise ValueError("--tw-lost-penalty must be >= 0")
    if args.rollout_stop_timeout <= 0:
        raise ValueError("--rollout-stop-timeout must be > 0")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be >= 1")
    if args.vllm_max_num_batched_tokens < 1:
        raise ValueError("--vllm-max-num-batched-tokens must be >= 1")
    if args.vllm_max_model_len is not None and args.vllm_max_model_len < 1:
        raise ValueError("--vllm-max-model-len must be >= 1 when set")
    if args.max_sync_rounds is not None and args.max_sync_rounds < 0:
        raise ValueError("--max-sync-rounds must be >= 0 when set")


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run_textworld_train(args))


if __name__ == "__main__":
    main()
