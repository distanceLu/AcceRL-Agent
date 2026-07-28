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


@dataclass
class PreparedVarlenPack:
    """One CPU-resident pack prepared for a Varlen optimizer window."""

    batch: Dict[str, torch.Tensor]
    valid_token_count: int
    max_seqlen: int
    version_lag_sum: float
    sample_count: int


VARLEN_TOKEN_STAT_NAMES = (
    "policy_token_sum",
    "old_new_kl_k3_sum",
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

    def collate(examples: List[RLSample]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example.input_ids) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []
        old_logprobs = []
        sample_advantages = []
        token_advantages = []
        response_indices = []

        for example in examples:
            pad_len = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [pad_token_id] * pad_len)
            attention_mask.append(example.attention_mask + [0] * pad_len)
            labels.append(example.labels + [-100] * pad_len)
            old_logprobs.append(example.old_logprobs + [0.0] * pad_len)
            sample_advantages.append(example.advantage)
            token_advantages.append(example.token_advantages + [0.0] * pad_len)
            response_indices.append(example.response_indices + [-1] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
            "sample_advantages": torch.tensor(sample_advantages, dtype=torch.float32),
            "token_advantages": torch.tensor(token_advantages, dtype=torch.float32),
            "response_indices": torch.tensor(response_indices, dtype=torch.long),
        }

    return collate


def make_varlen_batch(examples: List[RLSample]) -> Dict[str, torch.Tensor]:
    """Flatten examples while retaining their causal and RL sample boundaries."""
    if not examples:
        raise ValueError("At least one example is required for varlen packing.")

    input_ids = []
    labels = []
    old_logprobs = []
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
        "sample_advantages": torch.tensor(
            [example.advantage for example in examples], dtype=torch.float32
        ),
        "token_advantages": torch.tensor(token_advantages, dtype=torch.float32),
        "response_indices": torch.tensor(response_indices, dtype=torch.long),
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "prediction_indices": torch.tensor(prediction_indices, dtype=torch.long),
    }


def select_varlen_pack(
    prepared_samples: List[RLSample],
    token_budget: int,
    max_sequences: int,
) -> Tuple[List[RLSample], List[RLSample]]:
    """First-fit a random replay candidate pool after a local length sort."""
    ordered = sorted(
        enumerate(prepared_samples),
        key=lambda item: len(item[1].input_ids),
        reverse=True,
    )
    selected_indices = []
    selected = []
    total_tokens = 0
    for original_index, prepared in ordered:
        length = len(prepared.input_ids)
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
        self.pending_prepared_samples: List[RLSample] = []

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

    def _prepare_rl_sample(self, sample: RLSample) -> RLSample | None:
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
        token_advantages = list(sample.token_advantages)
        response_indices = list(sample.response_indices)
        if (
            len(old_logprobs) != len(input_ids)
            or len(token_advantages) != len(input_ids)
            or len(response_indices) != len(input_ids)
        ):
            return None
        max_length = self.args.max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            labels = labels[-max_length:]
            old_logprobs = old_logprobs[-max_length:]
            token_advantages = token_advantages[-max_length:]
            response_indices = response_indices[-max_length:]

        # A left-truncated first token has no in-window predecessor, so it
        # cannot be a causal LM target. The padded path already ignored it via
        # labels[:, 1:]; make that boundary explicit for flattened batches.
        labels[0] = -100
        old_logprobs[0] = 0.0
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

        return RLSample(
            algorithm=sample.algorithm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            old_logprobs=old_logprobs,
            advantage=sample.advantage,
            token_advantages=token_advantages,
            response_indices=response_indices,
            output_versions=list(sample.output_versions),
        )

    def _collate_prepared_rl_samples(
        self,
        prepared_samples: List[RLSample],
        *,
        move_to_device: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not prepared_samples:
            raise RuntimeError("No valid RL samples were available for training.")

        if self.args.train_packing == "varlen":
            batch = make_varlen_batch(prepared_samples)
        else:
            batch = self.collate_fn(prepared_samples)
        if move_to_device:
            batch = move_batch_to_device(batch, self.device)
        return batch

    @staticmethod
    def _version_lag_stats(
        samples: List[RLSample],
        trainer_version: float,
    ) -> Tuple[float, int]:
        lag_sum = sum(
            max(
                trainer_version
                - (max(sample.output_versions) if sample.output_versions else 0),
                0.0,
            )
            for sample in samples
        )
        return float(lag_sum), len(samples)

    def _select_varlen_pack(
        self,
    ) -> List[RLSample]:
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
                len(sample.input_ids)
                for sample in self.pending_prepared_samples
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
    ) -> Tuple[Dict[str, torch.Tensor], float]:
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
                prepared_sample = self._prepare_rl_sample(sample)
                if prepared_sample is not None:
                    collected.append(prepared_sample)
                    if len(collected) >= self.args.batch_size:
                        break

        samples = collected[: self.args.batch_size]
        lag_sum, sample_count = self._version_lag_stats(samples, trainer_version)
        return (
            self._collate_prepared_rl_samples(samples),
            lag_sum / sample_count,
        )

    def _next_varlen_cpu_pack(
        self,
        trainer_version: float,
    ) -> PreparedVarlenPack:
        """Prepare one Varlen pack without moving any tensor to the GPU."""
        replay_stats = self.get_replay_stats()
        warmup_deadline = None
        if self.args.replay_sample_timeout_seconds > 0:
            warmup_deadline = time.monotonic() + self.args.replay_sample_timeout_seconds
        while replay_stats["size"] < self.args.min_replay_size_per_rank:
            if warmup_deadline is not None and time.monotonic() >= warmup_deadline:
                raise TimeoutError(
                    "Timed out waiting for replay warmup: "
                    f"rank={self.rank} size={replay_stats['size']} "
                    f"min_replay_size_per_rank={self.args.min_replay_size_per_rank} "
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
            replay_stats = self.get_replay_stats()
            for sample in sampled:
                prepared_sample = self._prepare_rl_sample(sample)
                if prepared_sample is None:
                    continue
                self.pending_prepared_samples.append(prepared_sample)
            if self.pending_prepared_samples:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for valid replay candidates: "
                    f"rank={self.rank} target={candidate_target} "
                    f"stats={replay_stats}"
                )
            time.sleep(self.args.replay_wait_sleep_seconds)

        collected = self._select_varlen_pack()
        batch = self._collate_prepared_rl_samples(
            collected,
            move_to_device=False,
        )
        valid_token_count = int(batch["target_indices"].numel())
        if valid_token_count <= 0:
            raise RuntimeError("A Varlen pack must contain at least one target.")
        max_seqlen = max(len(sample.input_ids) for sample in collected)
        version_lag_sum, sample_count = self._version_lag_stats(
            collected,
            trainer_version,
        )
        return PreparedVarlenPack(
            batch=batch,
            valid_token_count=valid_token_count,
            max_seqlen=max_seqlen,
            version_lag_sum=version_lag_sum,
            sample_count=sample_count,
        )

    def _compute_rl_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        varlen_max_seqlen: int | None = None,
        return_varlen_token_sums: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor | None,
        Dict[str, float | torch.Tensor],
    ]:
        is_varlen = self.args.train_packing == "varlen"
        use_token_sum_loss = is_varlen and return_varlen_token_sums
        batch_size = int(batch["sample_advantages"].shape[0])
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
            valid_sample_indices = None
            if self.args.rl_algorithm == "grpo" or not use_token_sum_loss:
                valid_sample_indices = batch["sequence_ids"][target_indices]
            response_token_counts = None
            if not use_token_sum_loss:
                assert valid_sample_indices is not None
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
        if self.args.rl_algorithm == "ppo":
            if is_varlen:
                raw_valid_adv = batch["token_advantages"][target_indices].to(
                    torch.float32
                )
                if not use_token_sum_loss:
                    valid_response_indices = batch["response_indices"][
                        target_indices
                    ]
            else:
                response_indices = batch["response_indices"][:, 1:]
                valid_response_indices = response_indices[response_mask]
                raw_valid_adv = batch["token_advantages"][:, 1:][response_mask].to(
                    torch.float32
                )
            if (
                valid_response_indices is not None
                and valid_response_indices.lt(0).any()
            ):
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
            assert valid_sample_indices is not None
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

        sample_objective = None
        if not use_token_sum_loss:
            assert valid_sample_indices is not None
            assert response_token_counts is not None
            if self.args.rl_algorithm == "ppo":
                assert valid_response_indices is not None
                sample_objective = self._aggregate_valid_objective_by_response(
                    valid_objective,
                    valid_sample_indices,
                    valid_response_indices,
                    batch_size=batch_size,
                )
            else:
                sample_objective = self._aggregate_valid_objective(
                    valid_objective,
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
            old_new_kl_k3_token_mean = old_new_kl_k3_sum / valid_token_count
            loss = policy_token_sum + (
                self.args.old_new_kl_coef * old_new_kl_k3_sum
            )
        else:
            assert sample_objective is not None
            policy_loss = -sample_objective.mean()
            old_new_kl_k3_token_mean = old_new_kl_k3.mean()
            loss = policy_loss + (
                self.args.old_new_kl_coef * old_new_kl_k3_token_mean
            )
        with torch.no_grad():
            if is_varlen and return_varlen_token_sums:
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
                        ppo_clip_count,
                    ]
                ).detach().to(dtype=torch.float64)
                return (
                    loss,
                    None,
                    {"varlen_token_stats": varlen_token_stats},
                )

            loss_stats = {}
            if valid_ratio.numel() > 0:
                loss_stats["old_new_kl_k3_token_mean"] = float(
                    old_new_kl_k3_token_mean.item()
                )
            if self.args.clip_mode == "ppo" and valid_ratio.numel() > 0:
                clipped_mask = (
                    (valid_ratio < (1.0 - self.args.clip_eps))
                    | (valid_ratio > (1.0 + self.args.clip_eps))
                )
                loss_stats["ppo_clip_frac"] = float(
                    clipped_mask.float().mean().item()
                )
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
        local_version_stats = torch.zeros(
            2,
            device=self.device,
            dtype=torch.float64,
        )
        for prepared_pack in window:
            batch = move_batch_to_device(prepared_pack.batch, self.device)
            token_loss_sum, _, loss_stats = self._compute_rl_loss(
                batch,
                varlen_max_seqlen=prepared_pack.max_seqlen,
                return_varlen_token_sums=True,
            )
            backward_loss = token_loss_sum * (
                float(self.fsdp_world_size)
                / float(global_valid_token_count)
            )
            backward_loss.backward()

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
            local_version_stats[0] += prepared_pack.version_lag_sum
            local_version_stats[1] += prepared_pack.sample_count
            self.train_micro_step += 1
            del batch, token_loss_sum, backward_loss, varlen_token_stats

        dist.all_reduce(local_token_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_version_stats, op=dist.ReduceOp.SUM)
        global_policy_sum, global_kl_sum, global_clip_count = (
            local_token_stats.tolist()
        )
        global_version_lag_sum, global_sample_count = local_version_stats.tolist()

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
        policy_loss_token_mean = global_policy_sum / token_count
        kl_token_mean = global_kl_sum / token_count

        return {
            "global_policy_sum": global_policy_sum,
            "global_kl_sum": global_kl_sum,
            "global_clip_count": global_clip_count,
            "global_valid_token_count": token_count,
            "global_version_lag_sum": global_version_lag_sum,
            "global_sample_count": global_sample_count,
            "loss_mean": (
                policy_loss_token_mean
                + self.args.old_new_kl_coef * kl_token_mean
            ),
            "kl_token_mean": kl_token_mean,
            "clip_fraction": global_clip_count / token_count,
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
        segment_kls = []
        segment_clips = []
        segment_version_lags = []
        segment_valid_tokens = 0.0
        segment_varlen_steps = []

        while self.optimizer_step < target_optimizer_step:
            trainer_version = (
                self.optimizer_step / self.args.sync_every_optimizer_steps
            )
            if self.args.train_packing == "varlen":
                step_stats = self._run_varlen_optimizer_step(trainer_version)
                segment_varlen_steps.append(step_stats)
                if self.rank == 0 and self.optimizer_step % self.args.log_every == 0:
                    print(
                        "[train] "
                        f"optimizer_step={self.optimizer_step} "
                        f"loss={step_stats['loss_mean']:.6f} "
                        f"kl_token_mean={step_stats['kl_token_mean']:.6f} "
                        f"clip_frac={step_stats['clip_fraction']:.4f} "
                        f"tokens={step_stats['global_valid_token_count']:.0f} "
                        f"lr={step_stats['current_lr']:.8g}"
                    )
                continue

            batch, version_lag = self._next_rl_training_batch(trainer_version)
            raw_loss, response_token_counts, loss_stats = self._compute_rl_loss(
                batch,
            )

            loss = raw_loss / self.args.grad_accum_steps
            loss.backward()
            segment_losses.append(float(raw_loss.item()))
            segment_kls.append(
                float(loss_stats.get("old_new_kl_k3_token_mean", 0.0))
            )
            segment_clips.append(
                float(loss_stats.get("ppo_clip_frac", 0.0))
            )
            segment_version_lags.append(version_lag)
            segment_valid_tokens += float(response_token_counts.sum().item())
            self.train_micro_step += 1
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
                    f"loss={raw_loss.item():.6f} "
                    "kl_token_mean="
                    f"{loss_stats.get('old_new_kl_k3_token_mean', 0.0):.6f} "
                    f"clip_frac={loss_stats.get('ppo_clip_frac', 0.0):.4f} "
                    f"lr={current_lr:.8g}"
                )

        if segment_varlen_steps:
            segment_valid_tokens = sum(
                step["global_valid_token_count"]
                for step in segment_varlen_steps
            )
            policy_sum = sum(
                step["global_policy_sum"] for step in segment_varlen_steps
            )
            kl_sum = sum(step["global_kl_sum"] for step in segment_varlen_steps)
            clip_count = sum(
                step["global_clip_count"] for step in segment_varlen_steps
            )
            segment_kls = [kl_sum / segment_valid_tokens]
            segment_losses = [
                (policy_sum + self.args.old_new_kl_coef * kl_sum)
                / segment_valid_tokens
            ]
            segment_clips = [clip_count / segment_valid_tokens]
            version_lag_sum = sum(
                step["global_version_lag_sum"] for step in segment_varlen_steps
            )
            sample_count = sum(
                step["global_sample_count"] for step in segment_varlen_steps
            )
            segment_version_lags = [version_lag_sum / sample_count]

        dist.barrier()
        optimizer_steps_run = self.optimizer_step - start_optimizer_step
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "rank": self.rank,
            "optimizer_steps_run": optimizer_steps_run,
            "optimizer_step": self.optimizer_step,
            "micro_step": self.train_micro_step,
            "reached_max_steps": self.optimizer_step >= self.args.max_steps,
            "segment_loss_mean": (
                sum(segment_losses) / len(segment_losses)
                if segment_losses
                else 0.0
            ),
            "segment_kl_mean": (
                sum(segment_kls) / len(segment_kls) if segment_kls else 0.0
            ),
            "segment_clip_frac": (
                sum(segment_clips) / len(segment_clips)
                if segment_clips else 0.0
            ),
            "segment_valid_tokens": segment_valid_tokens,
            "segment_version_lag_mean": (
                sum(segment_version_lags) / len(segment_version_lags)
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

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.output_tokens))

    @property
    def restart_prompt_token_ids(self) -> List[int]:
        return self.input_ids + self.output_tokens


@dataclass
class InferenceResult:
    output_tokens: List[int]
    output_logprobs: List[float]
    output_versions: List[int]
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None


@ray.remote
class StatsActor:
    """Aggregates rollout metrics from async rollout workers."""

    def __init__(self, window_size: int, active_timeout_seconds: float):
        self.worker_last_active = {}
        self.active_timeout_seconds = active_timeout_seconds
        self.tw_scores = deque(maxlen=window_size)
        self.tw_max_scores = deque(maxlen=window_size)
        self.tw_wins = deque(maxlen=window_size)
        self.tw_steps = deque(maxlen=window_size)
        self.tw_invalid_actions = deque(maxlen=window_size)

    def add_textworld_episode(
        self,
        worker_id: int,
        score: float,
        max_score: float,
        won: bool,
        steps: int,
        invalid_actions: int,
    ) -> None:
        self.tw_scores.append(float(score))
        self.tw_max_scores.append(float(max_score))
        self.tw_wins.append(bool(won))
        self.tw_steps.append(int(steps))
        self.tw_invalid_actions.append(int(invalid_actions))
        self.worker_last_active[int(worker_id)] = time.time()

    def get_stats(self) -> Dict[str, float]:
        active_cutoff = time.time() - self.active_timeout_seconds
        active_workers = sum(
            last_active >= active_cutoff
            for last_active in self.worker_last_active.values()
        )
        tw_episode_count = len(self.tw_scores)
        tw_total_steps = sum(self.tw_steps)
        tw_total_max_score = sum(self.tw_max_scores)
        return {
            "active_workers": active_workers,
            "tw_win_rate": (
                sum(1 for won in self.tw_wins if won) / tw_episode_count
                if tw_episode_count else 0.0
            ),
            "tw_normalized_score": (
                sum(self.tw_scores) / tw_total_max_score
                if tw_total_max_score > 0 else 0.0
            ),
            "tw_invalid_action_rate": (
                sum(self.tw_invalid_actions) / tw_total_steps
                if tw_total_steps else 0.0
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

    def add_samples(self, samples: List[RLSample]) -> Dict[str, int]:
        capacity = self.samples.maxlen or 0
        if capacity > 0:
            self.total_samples_evicted += max(
                0,
                len(self.samples) + len(samples) - capacity,
            )
        self.samples.extend(samples)
        self.total_samples_added += len(samples)
        return self.get_stats()

    def sample(self, batch_size: int) -> List[RLSample]:
        if batch_size < 1:
            return []
        sample_count = min(batch_size, len(self.samples))
        if sample_count == 0:
            return []
        samples = random.sample(list(self.samples), sample_count)
        self.total_samples_sampled += len(samples)
        return samples

    def get_stats(self) -> Dict[str, int]:
        return {
            "size": len(self.samples),
            "capacity": self.samples.maxlen or 0,
            "total_samples_added": self.total_samples_added,
            "total_samples_sampled": self.total_samples_sampled,
            "total_samples_evicted": self.total_samples_evicted,
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
        self.total_tokens = 0
        self.next_request_index = 0
        self.stopped = False

    async def request_batch(
        self,
        input_ids: List[int],
        infer_max_tokens: int,
    ) -> InferenceResult:
        if self.stopped:
            raise RuntimeError("VLLMInferenceActor is stopped.")
        request_index = self.next_request_index
        self.next_request_index += 1
        state = OnlineGenerationState(
            index=request_index,
            input_ids=list(input_ids),
            requested_max_tokens=int(infer_max_tokens),
        )
        return await self._run_generation(state)

    async def _run_generation(
        self,
        state: OnlineGenerationState,
    ) -> InferenceResult:
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
            output_tokens=list(completed_state.output_tokens),
            output_logprobs=list(completed_state.output_logprobs),
            output_versions=list(completed_state.output_versions),
            stop_reason=completed_state.stop_reason,
        )
        self.total_tokens += len(result.output_tokens)
        return result

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
        return {"total_tokens": self.total_tokens}

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
    training_result: InferenceResult
    prompt_ids: List[int]
    reward: float


@dataclass
class TextWorldTrajectoryState:
    env: Any
    obs: str
    infos: Dict
    latest_score: float
    step_records: List[TextWorldStepRecord] = field(default_factory=list)
    transcript_ids: List[int] = field(default_factory=list)
    invalid_actions: int = 0
    done: bool = False
    won: bool = False
    lost: bool = False


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
    normalized: str
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
        normalized=normalized,
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
        state.transcript_ids.extend(result.output_tokens)
        if parsed_action.action is not None:
            selected_action = parsed_action.action
            obs, step_score, done, infos = state.env.step(selected_action)
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
            request_refs.append(
                self.infer_actor.request_batch.remote(
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
        sample_advantage: float | None = None,
    ) -> RLSample | None:
        algorithm = self.args.rl_algorithm if algorithm is None else algorithm
        input_ids: List[int] = []
        labels: List[int] = []
        old_logprobs: List[float] = []
        token_rewards: List[float] = []
        response_indices: List[int] = []
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
            output_versions.extend(result.output_versions)
            next_response_index += 1

        if all(label == -100 for label in labels):
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

        if algorithm == "ppo":
            token_advantages = self._compute_token_level_advantages(
                labels,
                token_rewards,
            )
            sample_advantage = 0.0
        elif algorithm == "grpo":
            if sample_advantage is None:
                raise ValueError("GRPO samples require a sample advantage.")
            token_advantages = [
                float(sample_advantage) if label != -100 else 0.0
                for label in labels
            ]
        else:
            raise ValueError(f"Unsupported rl_algorithm: {algorithm}")

        return RLSample(
            algorithm=algorithm,
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            old_logprobs=old_logprobs,
            advantage=float(sample_advantage),
            token_advantages=token_advantages,
            response_indices=response_indices,
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
            for _ in range(group_size):
                env = textworld.gym.make(env_id)
                obs, infos = env.reset()
                transcript_ids = self._initial_transcript_ids(obs, infos)
                states.append(
                    TextWorldTrajectoryState(
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
            _, _, advantages = (
                self._compute_grpo_group_advantages(grpo_rewards)
            )

            samples = []
            for state, advantage in zip(
                states,
                advantages,
            ):
                sample = self._build_textworld_episode_rl_sample(
                    step_records=state.step_records,
                    algorithm="grpo",
                    sample_advantage=advantage,
                )
                if sample is not None:
                    samples.append(sample)

            if samples:
                self.replay_buffer.add_samples.remote(samples)

            max_score = _textworld_max_score(states[0].infos) if states else 0.0
            for state in states:
                self.stats_actor.add_textworld_episode.remote(
                    self.worker_id,
                    state.latest_score,
                    max_score,
                    bool(state.won),
                    len(state.step_records),
                    state.invalid_actions,
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
            for _ in range(batch_size):
                env = textworld.gym.make(env_id)
                obs, infos = env.reset()
                transcript_ids = self._initial_transcript_ids(obs, infos)
                states.append(
                    TextWorldTrajectoryState(
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
            for state in states:
                sample = self._build_textworld_episode_rl_sample(
                    step_records=state.step_records,
                )
                if sample is not None:
                    samples.append(sample)

            if samples:
                self.replay_buffer.add_samples.remote(samples)

            max_score = _textworld_max_score(states[0].infos) if states else 0.0
            for state in states:
                self.stats_actor.add_textworld_episode.remote(
                    self.worker_id,
                    state.latest_score,
                    max_score,
                    bool(state.won),
                    len(state.step_records),
                    state.invalid_actions,
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

            infer_delta_tokens = (
                infer_stats_end["total_tokens"] - infer_stats_start["total_tokens"]
            )
            infer_tokens_per_sec = infer_delta_tokens / max(infer_elapsed, 1e-9)
            replay_stats = ray.get([
                worker.get_replay_stats.remote()
                for worker in fsdp_workers
            ])
            rollout_stats = ray.get(stats_actor.get_stats.remote())
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
            kl_token_mean = (
                sum(
                    float(summary["segment_kl_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            clip_fraction = (
                sum(
                    float(summary["segment_clip_frac"])
                    for summary in summaries
                )
                / len(summaries)
            )
            version_lag_mean = (
                sum(
                    float(summary["segment_version_lag_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            segment_valid_tokens = (
                float(rank0_summary["segment_valid_tokens"])
                if args.train_packing == "varlen"
                else sum(
                    float(summary["segment_valid_tokens"])
                    for summary in summaries
                )
            )
            train_tokens_per_sec = (
                segment_valid_tokens
                / max(train_segment_elapsed, 1e-9)
            )
            optimizer_steps_per_sec = (
                rank0_summary["optimizer_steps_run"]
                / max(train_segment_elapsed, 1e-9)
            )
            tb_step = rank0_summary["optimizer_step"]
            writer.add_scalar(
                "TextWorld/NormalizedScore",
                rollout_stats["tw_normalized_score"],
                tb_step,
            )
            writer.add_scalar(
                "TextWorld/WinRate",
                rollout_stats["tw_win_rate"],
                tb_step,
            )
            writer.add_scalar(
                "TextWorld/InvalidActionRate",
                rollout_stats["tw_invalid_action_rate"],
                tb_step,
            )
            writer.add_scalar("Replay/FillRatio", replay_fill_ratio, tb_step)
            writer.add_scalar(
                "Replay/TrainSampleTrainerVersionLagMean",
                version_lag_mean,
                tb_step,
            )
            writer.add_scalar(
                "Rollout/ActiveWorkers",
                rollout_stats["active_workers"],
                tb_step,
            )
            writer.add_scalar("Train/Loss", train_loss_mean, tb_step)
            writer.add_scalar(
                "KL/OldNewK3TokenMean",
                kl_token_mean,
                tb_step,
            )
            writer.add_scalar(
                "Train/LearningRate",
                rank0_summary["learning_rate"],
                tb_step,
            )
            writer.add_scalar("Train/TokensPerSec", train_tokens_per_sec, tb_step)
            writer.add_scalar(
                "Train/OptimizerStepsPerSec",
                optimizer_steps_per_sec,
                tb_step,
            )
            if args.clip_mode == "ppo":
                writer.add_scalar(
                    "Clip/PPOClipFrac",
                    clip_fraction,
                    tb_step,
                )
            writer.add_scalar("Infer/TokensPerSec", infer_tokens_per_sec, tb_step)
            print(
                "[metrics] "
                f"step={tb_step} loss={train_loss_mean:.6f} "
                f"kl={kl_token_mean:.6f} clip={clip_fraction:.4f} "
                f"train_tokens_per_sec={train_tokens_per_sec:.2f} "
                f"infer_tokens_per_sec={infer_tokens_per_sec:.2f} "
                f"score={rollout_stats['tw_normalized_score']:.4f}"
            )
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

            sync_elapsed_seconds = await sync_weights_to_vllm(
                infer_actor=infer_actor,
                fsdp_workers=fsdp_workers,
                scope="trainable",
                transfer_world_size=transfer_world_size,
                packed=True,
            )
            writer.add_scalar("Sync/ElapsedSeconds", sync_elapsed_seconds, tb_step)
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
