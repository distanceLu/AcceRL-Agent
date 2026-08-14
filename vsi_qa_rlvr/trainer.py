# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL FSDP trainer following AcceRL's asynchronous Replay workflow."""

import argparse
import json
import math
import os
import random
import socket
import time
from dataclasses import dataclass
"""vsiqa"""
from importlib import import_module
"""vsiqa"""
from typing import Any, Dict, Iterable, List, Tuple

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard

from transformers import AutoTokenizer

"""vsiqa"""
from transformers import Qwen3VLForConditionalGeneration
"""vsiqa"""

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

"""vsiqa"""
from vsi_qa_rlvr.trajectory import RLSample
"""vsiqa"""


"""vsiqa"""
TRAIN_ATTENTION_BACKENDS = ("flash_attention_2", "sdpa")


def require_training_attention_backend(backend: str) -> None:
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
"""vsiqa"""


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
        """vsiqa"""
        mm_token_type_ids = []
        """vsiqa"""

        for example in examples:
            pad_len = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [pad_token_id] * pad_len)
            attention_mask.append(example.attention_mask + [0] * pad_len)
            labels.append(example.labels + [-100] * pad_len)
            old_logprobs.append(example.old_logprobs + [0.0] * pad_len)
            sample_advantages.append(example.advantage)
            token_advantages.append(example.token_advantages + [0.0] * pad_len)
            response_indices.append(example.response_indices + [-1] * pad_len)

        """vsiqa"""
        for example in examples:
            pad_len = max_len - len(example.input_ids)
            prompt_mm_token_type_ids = example.prepared_media[
                "prompt_mm_token_type_ids"
            ]
            response_types = torch.zeros(
                len(example.input_ids) - len(prompt_mm_token_type_ids),
                dtype=torch.long,
            )
            mm_token_type_ids.append(
                torch.cat((prompt_mm_token_type_ids, response_types)).tolist()
                + [0] * pad_len
            )
        """vsiqa"""

        """vsiqa"""
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
            "sample_advantages": torch.tensor(sample_advantages, dtype=torch.float32),
            "token_advantages": torch.tensor(token_advantages, dtype=torch.float32),
            "response_indices": torch.tensor(response_indices, dtype=torch.long),
            "mm_token_type_ids": torch.tensor(mm_token_type_ids, dtype=torch.long),
            "pixel_values": torch.cat(
                [
                    example.prepared_media["pixel_values"]
                    for example in examples
                ],
                dim=0,
            ),
            "image_grid_thw": torch.cat(
                [
                    example.prepared_media["image_grid_thw"]
                    for example in examples
                ],
                dim=0,
            ),
        }
        """vsiqa"""

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
    """vsiqa"""
    qwen_position_ids = []
    mm_token_type_ids = []
    """vsiqa"""

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

    """vsiqa"""
    for example in examples:
        length = len(example.input_ids)
        example_position_ids = example.prepared_media["position_ids"]
        if tuple(example_position_ids.shape) != (3, length):
            raise ValueError(
                "Qwen3-VL position_ids must have shape "
                f"(3, {length}); got {tuple(example_position_ids.shape)}."
            )
        qwen_position_ids.append(example_position_ids)
        prompt_mm_token_type_ids = example.prepared_media[
            "prompt_mm_token_type_ids"
        ]
        response_types = torch.zeros(
            length - len(prompt_mm_token_type_ids),
            dtype=torch.long,
        )
        mm_token_type_ids.append(
            torch.cat((prompt_mm_token_type_ids, response_types))
        )
    packed_position_ids = torch.cat(qwen_position_ids, dim=1)
    """vsiqa"""

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

    """vsiqa"""
    for target_index, prediction_index in zip(
        target_indices,
        prediction_indices,
    ):
        if not torch.equal(
            packed_position_ids[:, target_index],
            packed_position_ids[:, prediction_index] + 1,
        ):
            raise ValueError(
                "A packed target must immediately follow its prediction "
                "position in Qwen3-VL MRoPE coordinates."
            )
    """vsiqa"""

    """vsiqa"""
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "position_ids": packed_position_ids.unsqueeze(1),
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
        "mm_token_type_ids": torch.cat(mm_token_type_ids).unsqueeze(0),
        "pixel_values": torch.cat(
            [example.prepared_media["pixel_values"] for example in examples],
            dim=0,
        ),
        "image_grid_thw": torch.cat(
            [example.prepared_media["image_grid_thw"] for example in examples],
            dim=0,
        ),
    }
    """vsiqa"""


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


"""vsiqa"""
def configure_trainable_parameters(model, train_mode: str) -> None:
    """Enable all model parameters for full-parameter training."""
    if train_mode != "full":
        raise ValueError(
            "Only full-parameter training is supported: "
            f"train_mode={train_mode!r}"
        )

    for param in model.parameters():
        param.requires_grad = True
"""vsiqa"""


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


"""vsiqa"""
def build_model(args: argparse.Namespace, device, torch_dtype, log: bool = True):
    if log:
        print(
            f"[init] Loading model from {args.model_path} "
            f"(device={device}, dtype={torch_dtype})"
        )
    require_training_attention_backend(args.train_attention_backend)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        attn_implementation=args.train_attention_backend,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    model.to(device)
    model.train()
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model
"""vsiqa"""


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
        """vsiqa"""
        self.forward_dtype = torch_dtype
        """vsiqa"""
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

        """vsiqa"""
        for layer in model.model.language_model.layers:
            fully_shard(layer)
        """vsiqa"""
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
                (name, sharded_params_by_name[name]) for name in all_param_names
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

        """vsiqa"""
        prompt_length = len(sample.prompt_ids)
        truncated_token_count = max(0, len(sample.input_ids) - max_length)
        removed_prompt_token_count = min(truncated_token_count, prompt_length)
        removed_prompt_types = sample.prepared_media[
            "prompt_mm_token_type_ids"
        ][:removed_prompt_token_count]
        if removed_prompt_types.ne(0).any():
            return None
        truncated_prompt_length = max(
            0,
            prompt_length - truncated_token_count,
        )
        prepared_media = dict(sample.prepared_media)
        original_prompt_mm_token_type_ids = sample.prepared_media[
            "prompt_mm_token_type_ids"
        ]
        prompt_mm_token_type_ids = (
            original_prompt_mm_token_type_ids[-truncated_prompt_length:]
            if truncated_prompt_length > 0
            else original_prompt_mm_token_type_ids[:0]
        )
        prepared_media["prompt_token_ids"] = torch.tensor(
            input_ids[:truncated_prompt_length],
            dtype=torch.long,
        )
        prepared_media["prompt_mm_token_type_ids"] = (
            prompt_mm_token_type_ids.contiguous()
        )
        if self.args.train_packing == "varlen":
            response_types = torch.zeros(
                len(input_ids) - len(prompt_mm_token_type_ids),
                dtype=torch.long,
            )
            mm_token_type_ids = torch.cat(
                (prompt_mm_token_type_ids, response_types)
            ).unsqueeze(0)
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
            attention_mask_tensor = torch.tensor(
                [attention_mask],
                dtype=torch.long,
            )
            position_ids, _ = self.model.model.get_rope_index(
                input_ids_tensor,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=sample.prepared_media["image_grid_thw"],
                attention_mask=attention_mask_tensor,
            )
            prepared_media["position_ids"] = position_ids[:, 0].contiguous()
        return RLSample(
            algorithm=sample.algorithm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            old_logprobs=old_logprobs,
            advantage=float(sample.advantage),
            token_advantages=token_advantages,
            response_indices=response_indices,
            output_versions=list(sample.output_versions),
            prompt_ids=input_ids[:truncated_prompt_length],
            response_ids=input_ids[truncated_prompt_length:],
            old_response_logprobs=old_logprobs[truncated_prompt_length:],
            reward=float(sample.reward),
            question=sample.question,
            ground_truth=sample.ground_truth,
            format_reward=float(sample.format_reward),
            answer_reward=float(sample.answer_reward),
            rollout_worker_id=int(sample.rollout_worker_id),
            batch_id=int(sample.batch_id),
            sample_id=int(sample.sample_id),
            stop_reason=sample.stop_reason,
            generated_text=sample.generated_text,
            prepared_media=prepared_media,
        )
        """vsiqa"""

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
            """vsiqa"""
            batch["pixel_values"] = batch["pixel_values"].to(
                dtype=self.forward_dtype,
            )
            """vsiqa"""
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
            """vsiqa"""
            model_kwargs.update({
                "pixel_values": batch["pixel_values"],
                "image_grid_thw": batch["image_grid_thw"],
                "mm_token_type_ids": batch["mm_token_type_ids"],
            })
            """vsiqa"""
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
                """vsiqa"""
                with torch.autocast(device_type="cuda", dtype=self.forward_dtype):
                    outputs = self.model(**model_kwargs)
                """vsiqa"""
                valid_logits = outputs.logits[0, prediction_indices, :]
            elif self.args.train_logprob_mode == "response_only_lm_head":
                """vsiqa"""
                with torch.autocast(device_type="cuda", dtype=self.forward_dtype):
                    outputs = self.model(
                        **model_kwargs,
                        logits_to_keep=prediction_indices,
                    )
                """vsiqa"""
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
            """vsiqa"""
            with torch.autocast(device_type="cuda", dtype=self.forward_dtype):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    mm_token_type_ids=batch["mm_token_type_ids"],
                    use_cache=False,
                )
            """vsiqa"""
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
            """vsiqa"""
            prepared_pack.batch["pixel_values"] = prepared_pack.batch[
                "pixel_values"
            ].to(
                dtype=self.forward_dtype,
            )
            """vsiqa"""
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
                if segment_version_lags
                else 0.0
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
