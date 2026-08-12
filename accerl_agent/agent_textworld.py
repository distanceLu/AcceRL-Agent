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
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from typing import Any, Dict, Iterable, List, Literal, Tuple

# Keep direct script execution import-compatible, then delegate to the
# canonical package module in the __main__ block below.
if __package__ in (None, ""):
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

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

from accerl_agent.interval_diagnostics import (
    IntervalDiagnostics,
    TimeWeightedGauge,
    distribution_scalar,
    merge_interval_snapshots,
)
from accerl_agent.ppo_value import (
    TokenValueHead,
    load_value_head_checkpoint,
    save_value_head_checkpoint,
)
from accerl_agent.rl_data import (
    GRPOSample,
    RLSample,
    RawPPOSample,
    TerminationReason,
    compute_batched_token_gae,
    validate_raw_ppo_sample,
)


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


def wait_for_selected_ray_actor_debugger(role: str, rank: int) -> None:
    """Wait for debugpy in the Ray actor selected through the environment."""
    selected_role_value = os.environ.get("ACCERL_DEBUG_ROLE", "").strip().lower()
    if not selected_role_value:
        return

    valid_roles = {"trainer", "infer", "rollout", "replay"}
    if selected_role_value == "all":
        selected_roles = valid_roles
    else:
        selected_roles = {
            "replay" if item.strip() == "replaybuffer" else item.strip()
            for item in selected_role_value.split(",")
            if item.strip()
        }
    invalid_roles = selected_roles - valid_roles
    if not selected_roles or invalid_roles:
        raise ValueError(
            "ACCERL_DEBUG_ROLE must be trainer, infer, rollout, replay "
            "(or replaybuffer), all, or a comma-separated combination; "
            f"got {selected_role_value!r}"
        )
    if role not in selected_roles:
        return

    if role == "infer":
        selected_rank = 0
    else:
        debug_rank_value = os.environ.get("ACCERL_DEBUG_RANK", "0")
        try:
            selected_rank = int(debug_rank_value)
        except ValueError as exc:
            raise ValueError(
                "ACCERL_DEBUG_RANK must be an integer, "
                f"got {debug_rank_value!r}"
            ) from exc
    if rank != selected_rank:
        return

    try:
        import debugpy
    except ImportError as exc:
        raise RuntimeError(
            "ACCERL_DEBUG_ROLE is set, but debugpy is not installed in the "
            "Ray actor environment. Install it with `python -m pip install "
            "debugpy`."
        ) from exc

    debug_host = os.environ.get("ACCERL_DEBUG_HOST", "127.0.0.1")
    default_ports = {
        "trainer": 5678,
        "infer": 5679,
        "rollout": 5680,
        "replay": 5681,
    }
    role_port_variable = f"ACCERL_DEBUG_{role.upper()}_PORT"
    debug_port_value = os.environ.get(
        role_port_variable,
        os.environ.get("ACCERL_DEBUG_PORT", str(default_ports[role])),
    )
    try:
        debug_port = int(debug_port_value)
    except ValueError as exc:
        raise ValueError(
            "ACCERL_DEBUG_PORT must be an integer, "
            f"got {debug_port_value!r}"
        ) from exc

    debugpy.listen((debug_host, debug_port))
    print(
        f"[debug] {role} rank {rank} waiting for debugger at "
        f"{debug_host}:{debug_port}...",
        flush=True,
    )
    debugpy.wait_for_client()
    print(f"[debug] {role} rank {rank} debugger attached.", flush=True)


@dataclass
class PreparedVarlenPack:
    """One CPU-resident pack prepared for a Varlen optimizer window."""

    batch: Dict[str, torch.Tensor]
    total_token_count: int
    valid_token_count: int
    max_seqlen: int
    version_lag_sum: float
    sample_count: int
    cpu_milliseconds: float
    valid_trajectory_count: int = 0


@dataclass
class PPOFlatTokenView:
    """Layout-independent token tensors consumed by the PPO objective."""

    current_logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    current_values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    response_sample_indices: torch.Tensor
    response_ordinals: torch.Tensor
    response_counts: torch.Tensor
    bootstrap_values: torch.Tensor
    bootstrap_mask: torch.Tensor


@dataclass(frozen=True)
class FrozenPPOTargets:
    """CPU-resident PPO targets frozen at optimizer-window start."""

    raw_advantages: torch.Tensor
    returns: torch.Tensor
    cpu_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor | None


class PackedTensorStats:
    """Named scalar statistics packed into one tensor for one all-reduce."""

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        return tuple(item.name for item in dataclass_fields(cls))

    @classmethod
    def zeros(cls, device) -> torch.Tensor:
        return torch.zeros(
            len(dataclass_fields(cls)),
            device=device,
            dtype=torch.float64,
        )

    def pack(self) -> torch.Tensor:
        values = [
            getattr(self, item.name)
            for item in dataclass_fields(self)
        ]
        if any(
            not isinstance(value, torch.Tensor) or value.ndim != 0
            for value in values
        ):
            raise TypeError("Packed statistics fields must be scalar tensors.")
        return torch.stack(values).detach().to(dtype=torch.float64)

    @classmethod
    def unpack(cls, packed: torch.Tensor):
        expected_shape = (len(dataclass_fields(cls)),)
        if packed.shape != expected_shape:
            raise ValueError(
                f"{cls.__name__} expected packed shape {expected_shape}, "
                f"got {tuple(packed.shape)}."
            )
        return cls(**{
            item.name: packed[index]
            for index, item in enumerate(dataclass_fields(cls))
        })

    def to_float_dict(self) -> Dict[str, float]:
        values = self.pack().tolist()
        return {
            item.name: float(values[index])
            for index, item in enumerate(dataclass_fields(self))
        }


@dataclass(frozen=True)
class GRPOReductionStats(PackedTensorStats):
    policy_trajectory_sum: torch.Tensor
    old_new_kl_k3_trajectory_sum: torch.Tensor
    old_new_kl_k3_token_sum: torch.Tensor
    ppo_clip_count: torch.Tensor
    valid_trajectory_count: torch.Tensor


@dataclass(frozen=True)
class PPOReductionStats(PackedTensorStats):
    policy_trajectory_sum: torch.Tensor
    value_loss_trajectory_sum: torch.Tensor
    kl_trajectory_sum: torch.Tensor
    policy_token_sum: torch.Tensor
    value_loss_token_sum: torch.Tensor
    kl_token_sum: torch.Tensor
    valid_trajectory_count: torch.Tensor
    clip_count: torch.Tensor
    value_sum: torch.Tensor
    value_sq_sum: torch.Tensor
    return_sum: torch.Tensor
    return_sq_sum: torch.Tensor
    raw_advantage_sum: torch.Tensor
    raw_advantage_sq_sum: torch.Tensor
    raw_advantage_count: torch.Tensor
    terminated_count: torch.Tensor
    truncated_count: torch.Tensor


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


def sum_token_values_by_trajectory(
    token_values: torch.Tensor,
    token_sample_indices: torch.Tensor,
    sample_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return trajectory-mean sum, token sum, and valid trajectory count.

    Importance ratios remain token-level. This reducer only changes how the
    resulting PPO/GRPO token objectives contribute to optimization: every
    trajectory with at least one valid response token contributes one
    within-trajectory token mean.
    """
    if token_values.ndim != 1 or token_sample_indices.ndim != 1:
        raise ValueError("RL token values and sample indices must be 1-D.")
    if token_values.numel() != token_sample_indices.numel():
        raise ValueError(
            "RL token values and sample indices must have equal lengths."
        )
    if sample_count < 0:
        raise ValueError("RL sample count must be non-negative.")
    if token_sample_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("RL sample indices must use an integer dtype.")
    if token_sample_indices.device != token_values.device:
        raise ValueError(
            "RL token values and sample indices must share a device."
        )

    response_counts = torch.bincount(
        token_sample_indices.long(),
        minlength=sample_count,
    )
    trajectory_sums = token_values.new_zeros(sample_count)
    trajectory_sums.scatter_add_(
        0,
        token_sample_indices.long(),
        token_values,
    )
    valid_trajectory_mask = response_counts > 0
    trajectory_mean_sum = (
        trajectory_sums
        / response_counts.clamp_min(1).to(dtype=token_values.dtype)
    )[valid_trajectory_mask].sum()
    return (
        trajectory_mean_sum,
        token_values.sum(),
        valid_trajectory_mask.sum(),
    )


def make_grpo_varlen_batch(
    examples: List[GRPOSample],
) -> Dict[str, torch.Tensor]:
    """Flatten examples while retaining their causal and RL sample boundaries."""
    if not examples:
        raise ValueError("At least one example is required for varlen packing.")

    input_ids = []
    labels = []
    old_logprobs = []
    output_versions = []
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
        output_versions.extend(example.output_versions)
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
        "output_versions": torch.tensor(output_versions, dtype=torch.long),
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
    }


def make_ppo_varlen_batch(
    examples: List[RawPPOSample],
) -> Dict[str, torch.Tensor]:
    """Pack PPO samples and derive all physical-layout metadata on CPU."""
    if not examples:
        raise ValueError("At least one PPO sample is required for varlen packing.")
    if not all(isinstance(example, RawPPOSample) for example in examples):
        raise TypeError("PPO varlen packing only accepts RawPPOSample inputs.")

    input_ids: List[int] = []
    labels: List[int] = []
    old_logprobs: List[float] = []
    token_rewards: List[float] = []
    token_terminated: List[bool] = []
    token_truncated: List[bool] = []
    output_versions: List[int] = []
    position_ids: List[int] = []
    sequence_ids: List[int] = []
    cu_seqlens = [0]
    bootstrap_sample_indices: List[int] = []
    bootstrap_prediction_indices: List[int] = []

    for sequence_id, example in enumerate(examples):
        validate_raw_ppo_sample(example)
        length = len(example.input_ids)
        if length < 2:
            raise ValueError("Each packed PPO sequence requires at least two tokens.")
        if example.labels[0] != -100:
            raise ValueError(
                "The first token in a packed PPO sequence cannot be an RL target."
            )
        sequence_start = cu_seqlens[-1]
        input_ids.extend(example.input_ids)
        labels.extend(example.labels)
        old_logprobs.extend(example.old_logprobs)
        token_rewards.extend(example.token_rewards)
        token_terminated.extend(example.token_terminated)
        token_truncated.extend(example.token_truncated)
        output_versions.extend(example.output_versions)
        position_ids.extend(range(length))
        sequence_ids.extend([sequence_id] * length)
        cu_seqlens.append(sequence_start + length)

        local_bootstrap = example.bootstrap_prediction_position
        if local_bootstrap is not None:
            packed_bootstrap = sequence_start + local_bootstrap
            bootstrap_sample_indices.append(sequence_id)
            bootstrap_prediction_indices.append(packed_bootstrap)
            if packed_bootstrap != cu_seqlens[-1] - 1:
                raise ValueError(
                    "A packed PPO bootstrap must be its sequence's final token."
                )
            if labels[packed_bootstrap] != -100:
                raise ValueError(
                    "A packed PPO bootstrap must point to an ignored context token."
                )

    target_indices = [
        index for index, label in enumerate(labels) if label != -100
    ]
    if not target_indices:
        raise ValueError("A PPO varlen pack must contain at least one RL target.")
    prediction_indices = [index - 1 for index in target_indices]
    for target_index, prediction_index in zip(
        target_indices,
        prediction_indices,
    ):
        if prediction_index < 0:
            raise ValueError("A packed PPO prediction position cannot be negative.")
        if sequence_ids[target_index] != sequence_ids[prediction_index]:
            raise ValueError("A packed PPO target cannot cross a sequence boundary.")
        if position_ids[target_index] != position_ids[prediction_index] + 1:
            raise ValueError(
                "A packed PPO target must immediately follow its prediction position."
            )
    # 每个有效 response token 属于 varlen pack 中哪一条原始 PPO trajectory。
    response_sample_indices = [
        sequence_ids[target_index] for target_index in target_indices
    ]
    response_counts = [0] * len(examples)
    response_ordinals = []
    for sample_index in response_sample_indices:
        response_ordinals.append(response_counts[sample_index])
        response_counts[sample_index] += 1
    if any(count <= 0 for count in response_counts):
        raise ValueError("Every PPO varlen sample requires a response token.")

    # 找出模型需要计算的位置
    selected_positions = sorted(
        set(prediction_indices + bootstrap_prediction_indices)
    )
    total_tokens = len(input_ids)
    if (
        not selected_positions
        or selected_positions[0] < 0
        or selected_positions[-1] >= total_tokens
    ):
        raise ValueError("PPO selected positions exceed the packed token range.")
    # 建立原始位置到精简列号的映射
    selected_column_by_position = {
        position: column for column, position in enumerate(selected_positions)
    }
    # 找到 response logits 对应的精简列
    response_selected_columns = [
        selected_column_by_position[position] for position in prediction_indices
    ]
    # 找到 bootstrap hidden 对应的精简列
    bootstrap_selected_columns = [
        selected_column_by_position[position]
        for position in bootstrap_prediction_indices
    ]
    if any(
        selected_positions[column] != position
        for column, position in zip(
            response_selected_columns,
            prediction_indices,
        )
    ):
        raise ValueError("PPO response selected-position mapping failed.")
    if any(
        selected_positions[column] != position
        for column, position in zip(
            bootstrap_selected_columns,
            bootstrap_prediction_indices,
        )
    ):
        raise ValueError("PPO bootstrap selected-position mapping failed.")

    # Derived packing metadata below is Trainer-local and is never persisted
    # in Replay.
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
        "token_rewards": torch.tensor(token_rewards, dtype=torch.float32),
        "token_terminated": torch.tensor(token_terminated, dtype=torch.bool),
        "token_truncated": torch.tensor(token_truncated, dtype=torch.bool),
        "output_versions": torch.tensor(output_versions, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "response_sample_indices": torch.tensor(
            response_sample_indices,
            dtype=torch.long,
        ),
        "response_ordinals": torch.tensor(
            response_ordinals,
            dtype=torch.long,
        ),
        "response_counts": torch.tensor(response_counts, dtype=torch.long),
        "bootstrap_sample_indices": torch.tensor(
            bootstrap_sample_indices,
            dtype=torch.long,
        ),
        "selected_positions": torch.tensor(
            selected_positions,
            dtype=torch.long,
        ),
        "response_selected_columns": torch.tensor(
            response_selected_columns,
            dtype=torch.long,
        ),
        "bootstrap_selected_columns": torch.tensor(
            bootstrap_selected_columns,
            dtype=torch.long,
        ),
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
    if train_mode == "lora":
        # TODO(lora): Inject LoRA adapters before FSDP wrapping, freeze the
        # base policy, expose only adapter parameters to the optimizer, save
        # adapter checkpoints, and add a vLLM-compatible adapter sync path.
        # Remove the matching validate_args guard once that path is complete.
        raise NotImplementedError("LoRA training is not implemented yet.")
    raise ValueError(f"Unsupported train mode: {train_mode}")


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
    model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    attention_implementation = getattr(
        model.config,
        "_attn_implementation",
        None,
    )
    if attention_implementation != "flash_attention_2":
        raise RuntimeError(
            "Packed training requires the loaded model to use "
            "flash_attention_2; got "
            f"{attention_implementation!r}."
        )
    model.to(device)
    model.train()
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def validate_ppo_selected_forward_support(model, args) -> None:
    """Reject PPO policies that cannot provide selected logits and hidden."""
    if args.rl_algorithm != "ppo":
        return
    forward_parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" not in forward_parameters:
        raise RuntimeError(
            "PPO requires a model forward with explicit tensor "
            "logits_to_keep support; no full-logits fallback is provided."
        )
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Module):
        raise RuntimeError(
            "PPO requires a model-native lm_head module so the "
            "selected final hidden states can be captured."
        )


def run_selected_causal_lm_forward(
    model,
    *,
    selected_positions: torch.Tensor,
    model_kwargs: Dict[str, Any],
):
    """Run the full CausalLM root and retain only selected logits/hidden."""
    forbidden_keys = {
        "logits_to_keep",
        "use_cache",
        "output_hidden_states",
        "return_dict",
    }
    conflicts = forbidden_keys.intersection(model_kwargs)
    if conflicts:
        raise ValueError(
            "Selected PPO model kwargs contain reserved keys: "
            f"{sorted(conflicts)}"
        )
    input_ids = model_kwargs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Selected PPO forward requires 2D input_ids.")
    if selected_positions.ndim != 1 or selected_positions.numel() == 0:
        raise ValueError("Selected PPO positions must be a non-empty 1D tensor.")

    captured_lm_head_inputs = []

    def capture_lm_head_input(_module, module_inputs):
        if len(module_inputs) != 1:
            raise RuntimeError(
                "PPO expected the model-native lm_head to receive one "
                "selected hidden-state tensor."
            )
        captured_lm_head_inputs.append(module_inputs[0])

    hook_handle = model.lm_head.register_forward_pre_hook(
        capture_lm_head_input
    )
    try:
        forward_kwargs = {
            **model_kwargs,
            "use_cache": False,
            "output_hidden_states": False,
            "return_dict": True,
            "logits_to_keep": selected_positions,
        }
        if "output_router_logits" in inspect.signature(model.forward).parameters:
            forward_kwargs["output_router_logits"] = False
        outputs = model(**forward_kwargs)
    finally:
        hook_handle.remove()

    if len(captured_lm_head_inputs) != 1:
        raise RuntimeError(
            "PPO selected-position forward requires exactly one "
            "model-native lm_head invocation; got "
            f"{len(captured_lm_head_inputs)}."
        )
    if outputs.hidden_states is not None:
        raise RuntimeError(
            "PPO selected-position forward unexpectedly retained all "
            "model hidden states."
        )
    selected_hidden = captured_lm_head_inputs[0]
    if outputs.logits.ndim != 3 or selected_hidden.ndim != 3:
        raise RuntimeError(
            "Selected PPO logits and hidden states must both be rank-3 tensors."
        )
    expected_prefix = (
        input_ids.shape[0],
        selected_positions.numel(),
    )
    if tuple(outputs.logits.shape[:2]) != expected_prefix:
        raise RuntimeError(
            "Unexpected selected PPO logits shape: "
            f"{tuple(outputs.logits.shape[:2])} != {expected_prefix}"
        )
    if tuple(selected_hidden.shape[:2]) != expected_prefix:
        raise RuntimeError(
            "Unexpected selected PPO hidden shape: "
            f"{tuple(selected_hidden.shape[:2])} != {expected_prefix}"
        )
    return outputs, selected_hidden


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


def validate_vllm_policy_weight_names(names: Iterable[str]) -> None:
    forbidden = [
        name
        for name in names
        if "value_head" in name.lower() or "critic" in name.lower()
    ]
    if forbidden:
        raise ValueError(
            "Critic tensors must not be included in the vLLM policy payload: "
            f"{forbidden[:5]}"
        )


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

        wait_for_selected_ray_actor_debugger("trainer", rank)

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
        torch_dtype = pick_dtype(args.dtype)
        model = build_model(args, self.device, torch_dtype, log=rank == 0)
        # 验证取hidden_state和logits的forward是否支持selected_positions参数,同时计算 Policy logits 和 PPO value
        validate_ppo_selected_forward_support(model, args)
        configure_trainable_parameters(model, args.train_mode)
        log_parameter_count(model, args.train_mode, rank=rank)

        hidden_size = getattr(model.config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(
                "The policy model config must define a positive integer "
                f"hidden_size; got {hidden_size!r}."
            )
        value_head = TokenValueHead(hidden_size=hidden_size, bias=True).to(
            device=self.device
        )
        value_head_param_names = [
            name for name, _ in value_head.named_parameters()
        ]
        self.value_head_loaded = load_value_head_checkpoint(
            value_head,
            args.model_path,
        )
        if rank == 0:
            policy_total, policy_trainable = count_parameters(model)
            value_total, value_trainable = count_parameters(value_head)
            print(
                "[train] Parameter groups: "
                f"policy_total={policy_total:,} "
                f"policy_trainable={policy_trainable:,} "
                f"value_head_total={value_total:,} "
                f"value_head_trainable={value_trainable:,} "
                f"total_trainable={policy_trainable + value_trainable:,} "
                f"value_head_loaded={self.value_head_loaded}"
            )

        # vLLM metadata must remain policy-only. Keep this list rooted at the
        # Hugging Face policy model rather than any future actor-critic wrapper.
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
        for metadata in self.weight_metadata_by_scope.values():
            validate_vllm_policy_weight_names(metadata[0])

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(value_head)
        fully_shard(model)

        self.model = model
        self.value_head = value_head
        fsdp_modules = [
            module
            for root in (self.model, self.value_head)
            for module in root.modules()
            if hasattr(module, "set_gradient_divide_factor")
        ]
        if not fsdp_modules:
            raise RuntimeError(
                "The pinned FSDP2 runtime must expose "
                "set_gradient_divide_factor()."
            )
        # 所有 FSDP 模块设置“梯度聚合后的除数”
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
        sharded_value_params_by_name = dict(self.value_head.named_parameters())
        self.value_head_params = [
            (name, sharded_value_params_by_name[name])
            for name in value_head_param_names
        ]
        self.policy_trainable_parameters = list(
            iter_trainable_parameters(self.model)
        )
        self.value_head_parameters = [
            param for _, param in self.value_head_params
        ]
        if not self.policy_trainable_parameters:
            raise RuntimeError(f"No trainable parameters found for mode: {args.train_mode}")
        if not self.value_head_parameters:
            raise RuntimeError("Value Head has no trainable parameters.")
        self.trainable_parameter_list = (
            self.policy_trainable_parameters + self.value_head_parameters
        )

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.policy_trainable_parameters,
                    "lr": args.learning_rate,
                    "group_name": "policy",
                },
                {
                    "params": self.value_head_parameters,
                    "lr": args.learning_rate,
                    "group_name": "value_head",
                },
            ],
            weight_decay=args.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)

        self.train_micro_step = 0
        self.optimizer_step = 0
        self.ppo_adv_ema_mean: float | None = None
        self.ppo_adv_ema_variance: float | None = None
        self.pending_prepared_samples: List[RLSample] = []
        self._ppo_forward_buffers_validated = False

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
        if self.args.rl_algorithm == "ppo" and isinstance(sample, RawPPOSample):
            return self._prepare_ppo_sample(sample)
        if self.args.rl_algorithm == "grpo" and isinstance(sample, GRPOSample):
            return self._prepare_grpo_sample(sample)
        return None

    def _prepare_ppo_sample(
        self,
        sample: RawPPOSample,
    ) -> RawPPOSample | None:
        input_ids = list(sample.input_ids)
        labels = list(sample.labels)
        old_logprobs = list(sample.old_logprobs)
        token_rewards = list(sample.token_rewards)
        token_terminated = list(sample.token_terminated)
        token_truncated = list(sample.token_truncated)
        output_versions = list(sample.output_versions)
        bootstrap_position = sample.bootstrap_prediction_position
        original_length = len(input_ids)
        token_fields = (
            labels,
            old_logprobs,
            token_rewards,
            token_terminated,
            token_truncated,
            output_versions,
        )
        if not input_ids or any(
            len(field) != original_length for field in token_fields
        ):
            return None
        max_length = self.args.max_length
        if original_length > max_length:
            truncate_offset = original_length - max_length
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
            old_logprobs = old_logprobs[-max_length:]
            token_rewards = token_rewards[-max_length:]
            token_terminated = token_terminated[-max_length:]
            token_truncated = token_truncated[-max_length:]
            output_versions = output_versions[-max_length:]
            if bootstrap_position is not None:
                bootstrap_position -= truncate_offset
                if not 0 <= bootstrap_position < max_length:
                    return None

        if len(input_ids) < 2:
            return None
        labels[0] = -100
        old_logprobs[0] = 0.0
        token_rewards[0] = 0.0
        token_terminated[0] = False
        token_truncated[0] = False
        output_versions[0] = -1

        prepared = RawPPOSample(
            input_ids=input_ids,
            labels=labels,
            old_logprobs=old_logprobs,
            token_rewards=token_rewards,
            token_terminated=token_terminated,
            token_truncated=token_truncated,
            output_versions=output_versions,
            bootstrap_prediction_position=bootstrap_position,
        )
        try:
            validate_raw_ppo_sample(prepared)
        except ValueError:
            return None
        return prepared

    def _prepare_grpo_sample(
        self,
        sample: GRPOSample,
    ) -> GRPOSample | None:
        input_ids = list(sample.input_ids)
        labels = list(sample.labels)
        old_logprobs = list(sample.old_logprobs)
        output_versions = list(sample.output_versions)
        original_length = len(input_ids)
        if not input_ids or any(
            len(field) != original_length
            for field in (
                labels,
                old_logprobs,
                output_versions,
            )
        ):
            return None
        max_length = self.args.max_length
        if original_length > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
            old_logprobs = old_logprobs[-max_length:]
            output_versions = output_versions[-max_length:]
        if len(input_ids) < 2:
            return None
        labels[0] = -100
        old_logprobs[0] = 0.0
        output_versions[0] = -1
        if all(label == -100 for label in labels[1:]):
            return None
        if any(
            output_version < 0
            for output_version, label in zip(output_versions[1:], labels[1:])
            if label != -100
        ):
            return None
        return GRPOSample(
            input_ids=input_ids,
            labels=labels,
            old_logprobs=old_logprobs,
            advantage=sample.advantage,
            output_versions=output_versions,
        )

    def _collate_prepared_rl_samples(
        self,
        prepared_samples: List[RLSample],
        *,
        move_to_device: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not prepared_samples:
            raise RuntimeError("No valid RL samples were available for training.")

        if self.args.rl_algorithm == "ppo":
            if not all(
                isinstance(sample, RawPPOSample)
                for sample in prepared_samples
            ):
                raise TypeError("PPO pack requires RawPPOSample inputs.")
            batch = make_ppo_varlen_batch(prepared_samples)
        else:
            if not all(
                isinstance(sample, GRPOSample)
                for sample in prepared_samples
            ):
                raise TypeError("GRPO pack requires GRPOSample inputs.")
            batch = make_grpo_varlen_batch(prepared_samples)
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
                - max(
                    (
                        version
                        for version in sample.output_versions
                        if version >= 0
                    ),
                    default=0,
                ),
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
            max_sequences=self.args.train_max_sequences_per_pack,
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

        pack_start_time = time.perf_counter()
        collected = self._select_varlen_pack()
        batch = self._collate_prepared_rl_samples(
            collected,
            move_to_device=False,
        )
        cpu_milliseconds = (
            time.perf_counter() - pack_start_time
        ) * 1000.0
        total_token_count = int(batch["input_ids"].numel())
        if total_token_count <= 0:
            raise RuntimeError("A Varlen pack must contain at least one token.")
        valid_token_count = int(batch["target_indices"].numel())
        if valid_token_count <= 0:
            raise RuntimeError("A Varlen pack must contain at least one target.")
        if self.args.rl_algorithm == "grpo":
            valid_sample_indices = batch["sequence_ids"][
                batch["target_indices"]
            ]
            valid_trajectory_count = int(
                torch.unique(valid_sample_indices).numel()
            )
            if valid_trajectory_count <= 0:
                raise RuntimeError(
                    "A GRPO Varlen pack must contain at least one valid "
                    "trajectory."
                )
        else:
            response_counts = batch["response_counts"]
            valid_trajectory_count = int(response_counts.gt(0).sum().item())
            if valid_trajectory_count <= 0:
                raise RuntimeError(
                    "A PPO Varlen pack must contain at least one valid "
                    "trajectory."
                )
        max_seqlen = max(len(sample.input_ids) for sample in collected)
        version_lag_sum, sample_count = self._version_lag_stats(
            collected,
            trainer_version,
        )
        return PreparedVarlenPack(
            batch=batch,
            total_token_count=total_token_count,
            valid_token_count=valid_token_count,
            max_seqlen=max_seqlen,
            version_lag_sum=version_lag_sum,
            sample_count=sample_count,
            cpu_milliseconds=cpu_milliseconds,
            valid_trajectory_count=valid_trajectory_count,
        )

    def _compute_rl_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        max_seqlen: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.args.rl_algorithm != "grpo":
            raise RuntimeError(
                "_compute_rl_loss is the GRPO-only training path; PPO must "
                "use its token-sum path."
            )
        target_indices = batch["target_indices"]
        prediction_indices = target_indices - 1
        if target_indices.numel() == 0:
            raise RuntimeError("No valid response tokens found for RL loss.")
        valid_sample_indices = batch["sequence_ids"][target_indices]
        valid_labels = batch["labels"][target_indices]
        model_kwargs = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
            "attention_mask": None,
            "use_cache": False,
            "cu_seq_lens_q": batch["cu_seqlens"],
            "cu_seq_lens_k": batch["cu_seqlens"],
            "max_length_q": int(max_seqlen),
            "max_length_k": int(max_seqlen),
        }
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

        # Ratio-based RL objectives are numerically sensitive. Keep the
        # subtraction, exponentiation, clipping/gating, and KL construction in
        # FP32 even when the model forward and logits use BF16/FP16.
        valid_token_log_probs = valid_token_log_probs.float()
        valid_old_token_log_probs = valid_old_token_log_probs.float()
        valid_log_ratio = valid_token_log_probs - valid_old_token_log_probs
        valid_ratio = torch.exp(valid_log_ratio)

        raw_valid_adv = batch["sample_advantages"][valid_sample_indices].to(
            torch.float32
        )
        valid_adv = raw_valid_adv

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

        sample_count = int(batch["sample_advantages"].numel())
        policy_trajectory_sum, _, valid_trajectory_count = (
            sum_token_values_by_trajectory(
                -valid_objective.float(),
                valid_sample_indices,
                sample_count,
            )
        )
        old_new_kl_k3 = valid_ratio - 1.0 - valid_log_ratio
        (
            old_new_kl_k3_trajectory_sum,
            old_new_kl_k3_token_sum,
            _,
        ) = sum_token_values_by_trajectory(
            old_new_kl_k3.float(),
            valid_sample_indices,
            sample_count,
        )
        loss = policy_trajectory_sum + (
            self.args.old_new_kl_coef * old_new_kl_k3_trajectory_sum
        )
        with torch.no_grad():
            if self.args.clip_mode == "ppo":
                clipped_mask = (
                    (valid_ratio < (1.0 - self.args.clip_eps))
                    | (valid_ratio > (1.0 + self.args.clip_eps))
                )
                ppo_clip_count = clipped_mask.sum().float()
            else:
                ppo_clip_count = policy_trajectory_sum.new_zeros(())

            reduction_stats = GRPOReductionStats(
                policy_trajectory_sum=policy_trajectory_sum,
                old_new_kl_k3_trajectory_sum=(
                    old_new_kl_k3_trajectory_sum
                ),
                old_new_kl_k3_token_sum=old_new_kl_k3_token_sum,
                ppo_clip_count=ppo_clip_count,
                valid_trajectory_count=valid_trajectory_count,
            ).pack()
        return loss, {"reduction_stats": reduction_stats}

    def _forward_ppo_token_view(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        max_seqlen: int,
    ) -> PPOFlatTokenView:
        target_indices = batch["target_indices"]
        response_sample_indices = batch["response_sample_indices"]
        response_columns = batch["response_selected_columns"]
        bootstrap_sample_indices = batch["bootstrap_sample_indices"]
        bootstrap_columns = batch["bootstrap_selected_columns"]
        outputs, selected_hidden = run_selected_causal_lm_forward(
            self.model,
            selected_positions=batch["selected_positions"],
            model_kwargs={
                "input_ids": batch["input_ids"],
                "position_ids": batch["position_ids"],
                "attention_mask": None,
                "cu_seq_lens_q": batch["cu_seqlens"],
                "cu_seq_lens_k": batch["cu_seqlens"],
                "max_length_q": int(max_seqlen),
                "max_length_k": int(max_seqlen),
            },
        )
        valid_logits = outputs.logits[0, response_columns]
        current_logprobs = -F.cross_entropy(
            valid_logits,
            batch["labels"][target_indices],
            reduction="none",
        ).float()
        response_hidden = selected_hidden[0, response_columns]
        if bootstrap_columns.numel() > 0:
            bootstrap_hidden = selected_hidden[0, bootstrap_columns]
        else:
            bootstrap_hidden = response_hidden.new_empty(
                (0, response_hidden.shape[-1])
            )
        all_value_hidden = torch.cat(
            (response_hidden, bootstrap_hidden),
            dim=0,
        )
        if all_value_hidden.shape[-1] != self.value_head.hidden_size:
            raise RuntimeError(
                "PPO selected hidden size does not match the Value Head: "
                f"{all_value_hidden.shape[-1]} != {self.value_head.hidden_size}"
            )
        all_values = self.value_head(all_value_hidden)
        response_value_count = int(response_hidden.shape[0])
        current_values = all_values[:response_value_count]
        sample_count = int(batch["response_counts"].numel())
        bootstrap_values = torch.zeros(
            sample_count,
            device=all_values.device,
            dtype=torch.float32,
        )
        bootstrap_mask = torch.zeros(
            sample_count,
            device=all_values.device,
            dtype=torch.bool,
        )
        if bootstrap_sample_indices.numel() > 0:
            bootstrap_values[bootstrap_sample_indices] = all_values[
                response_value_count:
            ]
            bootstrap_mask[bootstrap_sample_indices] = True
        return PPOFlatTokenView(
            current_logprobs=current_logprobs,
            old_logprobs=batch["old_logprobs"][target_indices].float(),
            current_values=current_values,
            rewards=batch["token_rewards"][target_indices].float(),
            terminated=batch["token_terminated"][target_indices],
            truncated=batch["token_truncated"][target_indices],
            response_sample_indices=response_sample_indices,
            response_ordinals=batch["response_ordinals"],
            response_counts=batch["response_counts"],
            bootstrap_values=bootstrap_values,
            bootstrap_mask=bootstrap_mask,
        )

    @staticmethod
    def _validate_ppo_flat_token_view(view: PPOFlatTokenView) -> None:
        num_tokens = int(view.current_logprobs.numel())
        sample_count = int(view.response_counts.numel())
        if num_tokens <= 0 or sample_count <= 0:
            raise RuntimeError("PPO flat token view must contain tokens and samples.")
        flat_fields = {
            "current_logprobs": view.current_logprobs,
            "old_logprobs": view.old_logprobs,
            "current_values": view.current_values,
            "rewards": view.rewards,
            "terminated": view.terminated,
            "truncated": view.truncated,
            "response_sample_indices": view.response_sample_indices,
            "response_ordinals": view.response_ordinals,
        }
        for name, tensor in flat_fields.items():
            if tensor.ndim != 1 or tensor.numel() != num_tokens:
                raise RuntimeError(
                    f"PPO flat field {name} must have shape [{num_tokens}]."
                )
        if view.response_counts.ndim != 1:
            raise RuntimeError("PPO response_counts must be one-dimensional.")
        if view.bootstrap_values.shape != (sample_count,):
            raise RuntimeError("PPO bootstrap_values must have shape [samples].")
        if view.bootstrap_mask.shape != (sample_count,):
            raise RuntimeError("PPO bootstrap_mask must have shape [samples].")
        if view.bootstrap_mask.dtype != torch.bool:
            raise RuntimeError("PPO bootstrap_mask must use bool dtype.")
        if view.terminated.dtype != torch.bool or view.truncated.dtype != torch.bool:
            raise RuntimeError("PPO boundary fields must use bool dtype.")
        if view.terminated.logical_and(view.truncated).any():
            raise RuntimeError("PPO tokens cannot terminate and truncate together.")
        if view.response_counts.le(0).any():
            raise RuntimeError("Every PPO sample must contain a response token.")
        if int(view.response_counts.sum().item()) != num_tokens:
            raise RuntimeError("PPO response counts do not match flat token count.")
        if (
            view.response_sample_indices.min().item() < 0
            or view.response_sample_indices.max().item() >= sample_count
        ):
            raise RuntimeError("PPO response sample index is out of range.")
        expected_counts = view.response_counts[
            view.response_sample_indices
        ]
        if (
            view.response_ordinals.lt(0).any()
            or view.response_ordinals.ge(expected_counts).any()
        ):
            raise RuntimeError("PPO response ordinal is out of range.")
        actual_counts = torch.bincount(
            view.response_sample_indices.long(),
            minlength=sample_count,
        )
        if not torch.equal(actual_counts, view.response_counts.long()):
            raise RuntimeError(
                "PPO response sample indices do not match response counts."
            )
        dense_width = int(view.response_counts.max().item())
        timeline_keys = (
            view.response_sample_indices.long() * dense_width
            + view.response_ordinals.long()
        )
        if torch.unique(timeline_keys).numel() != num_tokens:
            raise RuntimeError("PPO response timeline coordinates must be unique.")

    @staticmethod
    def _validate_ppo_boundary_layout(view: PPOFlatTokenView) -> None:
        expected_final = view.response_ordinals.eq(
            view.response_counts[view.response_sample_indices] - 1
        )
        boundaries = view.terminated.logical_or(view.truncated)
        if not torch.equal(boundaries, expected_final):
            raise RuntimeError(
                "Every PPO sample must have exactly one boundary on its "
                "final response token."
            )
        final_truncated = torch.zeros_like(view.bootstrap_mask)
        final_sample_indices = view.response_sample_indices[expected_final]
        final_truncated[final_sample_indices] = (
            view.truncated[expected_final]
        )
        if not torch.equal(view.bootstrap_mask, final_truncated):
            raise RuntimeError(
                "PPO bootstrap mask must exactly match truncated samples."
            )
        final_terminated = torch.zeros_like(view.bootstrap_mask)
        final_terminated[final_sample_indices] = view.terminated[expected_final]
        if view.bootstrap_mask.logical_and(final_terminated).any():
            raise RuntimeError("Terminated PPO samples cannot bootstrap.")

    def _compute_ppo_raw_targets(
        self,
        view: PPOFlatTokenView,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute detached FP32 token advantages and returns for one pack."""
        self._validate_ppo_flat_token_view(view)
        self._validate_ppo_boundary_layout(view)
        sample_count = int(view.response_counts.numel())
        dense_width = int(view.response_counts.max().item())
        dense_valid_mask = (
            torch.arange(
                dense_width,
                device=view.response_counts.device,
            ).unsqueeze(0)
            < view.response_counts.unsqueeze(1)
        )
        dense_values = view.current_values.new_zeros(
            (sample_count, dense_width)
        )
        dense_values[
            view.response_sample_indices,
            view.response_ordinals,
        ] = view.current_values
        dense_rewards = dense_values.new_zeros(dense_values.shape)
        dense_terminated = torch.zeros_like(
            dense_valid_mask,
            dtype=torch.bool,
        )
        dense_truncated = torch.zeros_like(
            dense_valid_mask,
            dtype=torch.bool,
        )
        dense_rewards[
            view.response_sample_indices,
            view.response_ordinals,
        ] = view.rewards
        dense_terminated[
            view.response_sample_indices,
            view.response_ordinals,
        ] = view.terminated
        dense_truncated[
            view.response_sample_indices,
            view.response_ordinals,
        ] = view.truncated
        row_indices = torch.arange(
            sample_count,
            device=view.response_counts.device,
        )
        final_ordinals = view.response_counts - 1
        final_terminated = dense_terminated[row_indices, final_ordinals]
        final_truncated = dense_truncated[row_indices, final_ordinals]
        if not torch.equal(view.bootstrap_mask, final_truncated):
            raise RuntimeError(
                "PPO bootstrap mask must exactly match truncated samples."
            )
        if view.bootstrap_mask.logical_and(final_terminated).any():
            raise RuntimeError("Terminated PPO samples cannot bootstrap.")
        dense_advantages, dense_returns, _ = (
            compute_batched_token_gae(
                rewards=dense_rewards,
                baseline_values=dense_values,
                valid_mask=dense_valid_mask,
                terminated=dense_terminated,
                truncated=dense_truncated,
                bootstrap_values=view.bootstrap_values,
                gamma=self.args.gae_gamma,
                gae_lambda=self.args.gae_lambda,
            )
        )
        raw_advantages = dense_advantages[
            view.response_sample_indices,
            view.response_ordinals,
        ]
        returns = dense_returns[
            view.response_sample_indices,
            view.response_ordinals,
        ]
        if raw_advantages.numel() != view.current_values.numel():
            raise RuntimeError("PPO batched GAE value/token alignment mismatch.")
        if raw_advantages.requires_grad or returns.requires_grad:
            raise RuntimeError("PPO GAE targets must be detached.")
        return raw_advantages.float().detach(), returns.float().detach()

    @staticmethod
    def _capture_ppo_rng_state(device: torch.device) -> Tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        cpu_rng_state = torch.get_rng_state().clone()
        cuda_rng_state = None
        if device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(device).clone()
        return cpu_rng_state, cuda_rng_state

    @staticmethod
    def _restore_ppo_rng_state(
        device: torch.device,
        cpu_rng_state: torch.Tensor,
        cuda_rng_state: torch.Tensor | None,
    ) -> None:
        torch.set_rng_state(cpu_rng_state)
        if device.type == "cuda":
            if cuda_rng_state is None:
                raise RuntimeError("Frozen PPO target is missing CUDA RNG state.")
            torch.cuda.set_rng_state(cuda_rng_state, device)

    @staticmethod
    def _finalize_ppo_advantage_moments(
        moments: torch.Tensor,
        *,
        expected_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if moments.shape != (3,) or moments.dtype != torch.float64:
            raise ValueError(
                "PPO advantage moments must be a three-element FP64 tensor."
            )
        global_sum, global_sq_sum, global_count = moments
        advantage_count = int(global_count.item())
        if advantage_count != expected_count:
            raise RuntimeError(
                "Global PPO advantage count does not match the optimizer-window "
                "valid response-token count: "
                f"{advantage_count} != {expected_count}."
            )
        if advantage_count <= 0:
            raise RuntimeError("Global PPO advantage count must be positive.")
        mean = global_sum / global_count
        variance = (
            global_sq_sum / global_count - mean.square()
        ).clamp_min(0.0)
        return mean, variance.sqrt()

    @staticmethod
    def _normalize_ppo_advantages(
        raw_advantages: torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if mean.ndim != 0 or std.ndim != 0:
            raise ValueError("PPO advantage mean and std must be scalars.")
        return (
            raw_advantages.detach().float()
            - mean.to(device=raw_advantages.device, dtype=torch.float32)
        ) / (
            std.to(device=raw_advantages.device, dtype=torch.float32)
            + float(eps)
        )

    @staticmethod
    def _freeze_ppo_targets(
        raw_advantages: torch.Tensor,
        returns: torch.Tensor,
        *,
        cpu_rng_state: torch.Tensor,
        cuda_rng_state: torch.Tensor | None,
    ) -> FrozenPPOTargets:
        return FrozenPPOTargets(
            raw_advantages=raw_advantages.detach().float().cpu().contiguous(),
            returns=returns.detach().float().cpu().contiguous(),
            cpu_rng_state=cpu_rng_state.detach().cpu().clone(),
            cuda_rng_state=(
                cuda_rng_state.detach().cpu().clone()
                if cuda_rng_state is not None
                else None
            ),
        )

    @staticmethod
    def _validate_frozen_ppo_targets(
        view: PPOFlatTokenView,
        targets: FrozenPPOTargets,
    ) -> None:
        expected_shape = view.current_values.shape
        if (
            targets.raw_advantages.shape != expected_shape
            or targets.returns.shape != expected_shape
        ):
            raise RuntimeError(
                "Frozen PPO advantages and returns must match the current "
                f"value shape {tuple(expected_shape)}."
            )

    def _compute_ppo_loss_from_targets(
        self,
        view: PPOFlatTokenView,
        *,
        raw_advantages: torch.Tensor,
        actor_advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute trajectory-equal PPO loss and dual-reduction statistics."""
        self._validate_ppo_flat_token_view(view)
        expected_shape = view.current_values.shape
        target_fields = {
            "raw_advantages": raw_advantages,
            "actor_advantages": actor_advantages,
            "returns": returns,
        }
        for name, tensor in target_fields.items():
            if tensor.shape != expected_shape:
                raise RuntimeError(
                    f"PPO {name} shape must be {tuple(expected_shape)}, "
                    f"got {tuple(tensor.shape)}."
                )
            if tensor.requires_grad:
                raise RuntimeError(f"PPO {name} must be detached.")
        raw_advantages = raw_advantages.detach().float()
        actor_advantages = actor_advantages.detach().float()
        returns = returns.detach().float()

        valid_log_ratio = view.current_logprobs.float() - view.old_logprobs.float()
        valid_ratio = torch.exp(valid_log_ratio)
        if self.args.clip_mode == "ppo":
            surrogate_1 = valid_ratio * actor_advantages
            surrogate_2 = torch.clamp(
                valid_ratio,
                1.0 - self.args.clip_eps,
                1.0 + self.args.clip_eps,
            ) * actor_advantages
            valid_objective = torch.minimum(surrogate_1, surrogate_2)
            clipped_mask = (
                (valid_ratio < 1.0 - self.args.clip_eps)
                | (valid_ratio > 1.0 + self.args.clip_eps)
            )
        elif self.args.clip_mode == "gipo":
            ratio_detached = valid_ratio.clamp_min(1e-9).detach()
            coefficient = torch.exp(
                -0.5
                * (torch.log(ratio_detached) / self.args.gipo_sigma) ** 2
            )
            valid_objective = (
                valid_ratio * actor_advantages * coefficient
            )
            clipped_mask = torch.zeros_like(valid_ratio, dtype=torch.bool)
        elif self.args.clip_mode == "sapo":
            ratio = valid_ratio.clamp(1e-6, 1e6)
            tau = torch.where(
                actor_advantages > 0,
                torch.full_like(actor_advantages, self.args.sapo_tau_pos),
                torch.full_like(actor_advantages, self.args.sapo_tau_neg),
            )
            gate = torch.sigmoid(tau * (ratio - 1.0)) * (4.0 / tau)
            valid_objective = gate * actor_advantages
            clipped_mask = torch.zeros_like(valid_ratio, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported clip_mode: {self.args.clip_mode}")

        sample_count = int(view.response_counts.numel())
        (
            policy_trajectory_sum,
            policy_token_sum,
            valid_trajectory_count,
        ) = sum_token_values_by_trajectory(
            -valid_objective.float(),
            view.response_sample_indices,
            sample_count,
        )
        residuals = returns.detach().float() - view.current_values.float()
        value_loss_tokens = 0.5 * residuals.float().square()
        value_loss_trajectory_sum, value_loss_token_sum, _ = (
            sum_token_values_by_trajectory(
                value_loss_tokens,
                view.response_sample_indices,
                sample_count,
            )
        )
        old_new_kl = valid_ratio - 1.0 - valid_log_ratio
        kl_trajectory_sum, kl_token_sum, _ = (
            sum_token_values_by_trajectory(
                old_new_kl.float(),
                view.response_sample_indices,
                sample_count,
            )
        )
        trajectory_loss_sum = (
            policy_trajectory_sum
            + self.args.value_loss_coef * value_loss_trajectory_sum
            + self.args.old_new_kl_coef * kl_trajectory_sum
        )

        with torch.no_grad():
            values_detached = view.current_values.detach().float()
            returns_float = returns.float()
            raw_advantages64 = raw_advantages.double()
            stats = PPOReductionStats(
                policy_trajectory_sum=policy_trajectory_sum,
                value_loss_trajectory_sum=value_loss_trajectory_sum,
                kl_trajectory_sum=kl_trajectory_sum,
                policy_token_sum=policy_token_sum,
                value_loss_token_sum=value_loss_token_sum,
                kl_token_sum=kl_token_sum,
                valid_trajectory_count=valid_trajectory_count,
                clip_count=clipped_mask.sum(),
                value_sum=values_detached.sum(),
                value_sq_sum=values_detached.square().sum(),
                return_sum=returns_float.sum(),
                return_sq_sum=returns_float.square().sum(),
                raw_advantage_sum=raw_advantages64.sum(),
                raw_advantage_sq_sum=raw_advantages64.square().sum(),
                raw_advantage_count=raw_advantages64.new_tensor(
                    raw_advantages64.numel()
                ),
                terminated_count=view.terminated.sum(),
                truncated_count=view.truncated.sum(),
            ).pack()
        return trajectory_loss_sum, stats

    def _compute_ppo_reduction_sums(
        self,
        view: PPOFlatTokenView,
        *,
        ema_mean: float | None = None,
        ema_scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a single-forward trajectory-equal PPO objective."""
        mode = self.args.ppo_advantage_normalization
        if mode == "optimizer_window":
            raise RuntimeError(
                "Optimizer-window PPO must use frozen two-pass targets."
            )
        raw_advantages, returns = self._compute_ppo_raw_targets(view)
        if mode == "none":
            actor_advantages = raw_advantages
        elif mode == "ema_rms":
            if ema_scale is None:
                raise RuntimeError("EMA-RMS PPO requires a frozen EMA scale.")
            actor_advantages = raw_advantages / ema_scale
        elif mode == "ema_zscore":
            if ema_mean is None or ema_scale is None:
                raise RuntimeError(
                    "EMA-ZScore PPO requires frozen EMA mean and scale."
                )
            actor_advantages = (raw_advantages - ema_mean) / ema_scale
        else:
            raise ValueError(
                "Unsupported PPO advantage normalization mode: "
                f"{mode!r}."
            )
        if not torch.isfinite(actor_advantages).all():
            raise RuntimeError(
                "PPO actor advantages contain non-finite values after "
                f"{mode} normalization."
            )
        return self._compute_ppo_loss_from_targets(
            view,
            raw_advantages=raw_advantages,
            actor_advantages=actor_advantages,
            returns=returns,
        )

    def _compute_packed_ppo_reduction_sums(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        max_seqlen: int,
        ema_mean: float | None = None,
        ema_scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._compute_ppo_reduction_sums(
            self._forward_ppo_token_view(
                batch,
                max_seqlen=max_seqlen,
            ),
            ema_mean=ema_mean,
            ema_scale=ema_scale,
        )

    def _validate_prepared_packed_ppo_batch(
        self,
        prepared: PreparedVarlenPack,
    ) -> None:
        """Validate PPO timeline metadata on CPU before any FSDP forward."""
        batch = prepared.batch
        target_indices = batch["target_indices"]
        valid_token_count = int(target_indices.numel())
        if valid_token_count != prepared.valid_token_count:
            raise RuntimeError(
                "Prepared PPO valid-token count does not match target indices: "
                f"{prepared.valid_token_count} != {valid_token_count}."
            )
        sample_count = int(batch["response_counts"].numel())
        valid_trajectory_count = int(
            batch["response_counts"].gt(0).sum().item()
        )
        if valid_trajectory_count != prepared.valid_trajectory_count:
            raise RuntimeError(
                "Prepared PPO valid-trajectory count does not match response "
                f"counts: {prepared.valid_trajectory_count} != "
                f"{valid_trajectory_count}."
            )
        bootstrap_mask = torch.zeros(sample_count, dtype=torch.bool)
        bootstrap_sample_indices = batch["bootstrap_sample_indices"].long()
        if bootstrap_sample_indices.numel() > 0:
            bootstrap_mask[bootstrap_sample_indices] = True
        zeros = torch.zeros(valid_token_count, dtype=torch.float32)
        bootstrap_values = torch.zeros(sample_count, dtype=torch.float32)
        view = PPOFlatTokenView(
            current_logprobs=zeros,
            old_logprobs=batch["old_logprobs"][target_indices].float(),
            current_values=zeros,
            rewards=batch["token_rewards"][target_indices].float(),
            terminated=batch["token_terminated"][target_indices],
            truncated=batch["token_truncated"][target_indices],
            response_sample_indices=batch["response_sample_indices"].long(),
            response_ordinals=batch["response_ordinals"].long(),
            response_counts=batch["response_counts"].long(),
            bootstrap_values=bootstrap_values,
            bootstrap_mask=bootstrap_mask,
        )
        self._validate_ppo_flat_token_view(view)
        self._validate_ppo_boundary_layout(view)

    def _snapshot_ppo_forward_buffers(self) -> Dict[str, torch.Tensor]:
        snapshots = {}
        for root_name, root in (
            ("model", self.model),
            ("value_head", self.value_head),
        ):
            for name, buffer in root.named_buffers():
                snapshots[f"{root_name}.{name}"] = buffer.detach().clone()
        return snapshots

    def _validate_ppo_forward_buffers_unchanged(
        self,
        snapshots: Dict[str, torch.Tensor],
    ) -> None:
        current = {
            f"{root_name}.{name}": buffer
            for root_name, root in (
                ("model", self.model),
                ("value_head", self.value_head),
            )
            for name, buffer in root.named_buffers()
        }
        if current.keys() != snapshots.keys():
            raise RuntimeError(
                "PPO forward changed the model buffer set; two-pass target "
                "replay does not support mutable forward state."
            )
        changed = [
            name
            for name, before in snapshots.items()
            if not torch.equal(before, current[name])
        ]
        if changed:
            raise RuntimeError(
                "PPO forward mutated persistent model buffers; two-pass "
                "target replay is unsupported. Changed buffers: "
                f"{changed[:8]}"
            )

    def _precompute_ppo_optimizer_window_targets(
        self,
        window: List[PreparedVarlenPack],
        *,
        global_valid_token_count: int,
    ) -> Tuple[
        List[FrozenPPOTargets],
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """Freeze one optimizer window and compute its cross-rank moments."""
        start_time = time.perf_counter()
        local_moments = torch.zeros(
            3,
            device=self.device,
            dtype=torch.float64,
        )
        frozen_targets = []
        validate_buffers = not self._ppo_forward_buffers_validated

        for pack_index, prepared in enumerate(window):
            local_error = None
            frozen = None
            batch = None
            try:
                batch = move_batch_to_device(prepared.batch, self.device)
                cpu_rng_state, cuda_rng_state = self._capture_ppo_rng_state(
                    self.device
                )
                buffer_snapshots = (
                    self._snapshot_ppo_forward_buffers()
                    if validate_buffers and pack_index == 0
                    else None
                )
                with torch.no_grad():
                    view = self._forward_ppo_token_view(
                        batch,
                        max_seqlen=prepared.max_seqlen,
                    )
                    raw_advantages, returns = self._compute_ppo_raw_targets(view)
                if buffer_snapshots is not None:
                    self._validate_ppo_forward_buffers_unchanged(
                        buffer_snapshots
                    )
                frozen = self._freeze_ppo_targets(
                    raw_advantages,
                    returns,
                    cpu_rng_state=cpu_rng_state,
                    cuda_rng_state=cuda_rng_state,
                )
                advantages64 = raw_advantages.double()
                local_moments[0] += advantages64.sum()
                local_moments[1] += advantages64.square().sum()
                local_moments[2] += advantages64.numel()
            except Exception as exc:
                local_error = repr(exc)
                print(
                    f"[rank {self.rank}] PPO target prepass failed at "
                    f"pack {pack_index}: {local_error}"
                )
            finally:
                del batch

            success = torch.tensor(
                0 if local_error is not None else 1,
                device=self.device,
                dtype=torch.int32,
            )
            dist.all_reduce(success, op=dist.ReduceOp.MIN)
            if int(success.item()) != 1:
                raise RuntimeError(
                    "At least one FSDP rank failed during PPO target "
                    f"prepass pack {pack_index}; see rank logs."
                )
            assert frozen is not None
            frozen_targets.append(frozen)

        self._ppo_forward_buffers_validated = True
        dist.all_reduce(local_moments, op=dist.ReduceOp.SUM)
        advantage_mean, advantage_std = (
            self._finalize_ppo_advantage_moments(
                local_moments,
                expected_count=global_valid_token_count,
            )
        )
        elapsed_milliseconds = (time.perf_counter() - start_time) * 1000.0
        return (
            frozen_targets,
            advantage_mean,
            advantage_std,
            elapsed_milliseconds,
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
            if self.args.rl_algorithm == "ppo":
                for prepared in window:
                    self._validate_prepared_packed_ppo_batch(prepared)
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

    def _reduce_varlen_pack_stats(
        self,
        window: List[PreparedVarlenPack],
    ) -> Dict[str, float]:
        """Aggregate pack-shape and CPU-construction statistics across ranks."""
        pack_sums = torch.tensor(
            [
                sum(pack.total_token_count for pack in window),
                len(window),
                sum(pack.sample_count for pack in window),
                sum(pack.cpu_milliseconds for pack in window),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        max_seqlen = torch.tensor(
            max((pack.max_seqlen for pack in window), default=0),
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(pack_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_seqlen, op=dist.ReduceOp.MAX)
        (
            global_token_count,
            global_pack_count,
            global_sample_count,
            global_cpu_milliseconds,
        ) = pack_sums.tolist()
        return {
            "global_pack_token_count": global_token_count,
            "global_pack_count": global_pack_count,
            "global_pack_sample_count": global_sample_count,
            "global_pack_cpu_milliseconds": global_cpu_milliseconds,
            "global_pack_max_seqlen": float(max_seqlen.item()),
        }

    def _run_ppo_optimizer_step(
        self,
        trainer_version: float,
    ) -> Dict[str, object]:
        """Run one globally trajectory-normalized packed PPO optimizer step."""
        window = self._prepare_varlen_optimizer_window(trainer_version)
        pack_stats = self._reduce_varlen_pack_stats(window)
        local_valid_token_count = sum(
            prepared.valid_token_count for prepared in window
        )
        local_valid_trajectory_count = sum(
            prepared.valid_trajectory_count for prepared in window
        )
        global_reduction_counts = torch.tensor(
            [local_valid_token_count, local_valid_trajectory_count],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(
            global_reduction_counts,
            op=dist.ReduceOp.SUM,
        )
        (
            global_valid_token_count,
            global_valid_trajectory_count,
        ) = (int(value) for value in global_reduction_counts.tolist())
        if global_valid_token_count <= 0:
            raise RuntimeError(
                "Global packed PPO optimizer window contains no valid "
                "response tokens."
            )
        if global_valid_trajectory_count <= 0:
            raise RuntimeError(
                "Global packed PPO optimizer window contains no valid "
                "trajectories."
            )

        normalization_mode = self.args.ppo_advantage_normalization
        ema_mode = normalization_mode in ("ema_rms", "ema_zscore")
        ema_mean = self.ppo_adv_ema_mean
        ema_variance = self.ppo_adv_ema_variance
        if (ema_mean is None) != (ema_variance is None):
            raise RuntimeError(
                "PPO advantage EMA mean and variance must be initialized "
                "together."
            )
        ema_initialized = ema_mean is not None
        use_optimizer_window = (
            normalization_mode == "optimizer_window"
            or (ema_mode and not ema_initialized)
        )
        ema_scale = None
        ema_scale_clamped = False
        if ema_mode and ema_initialized:
            assert ema_mean is not None
            assert ema_variance is not None
            if not math.isfinite(ema_mean) or not math.isfinite(ema_variance):
                raise RuntimeError("PPO advantage EMA state must be finite.")
            if ema_variance < 0.0:
                raise RuntimeError("PPO advantage EMA variance must be >= 0.")
            if normalization_mode == "ema_rms":
                raw_ema_scale = math.sqrt(
                    max(ema_variance + ema_mean * ema_mean, 0.0)
                )
            else:
                raw_ema_scale = math.sqrt(max(ema_variance, 0.0))
            ema_scale = max(
                raw_ema_scale,
                self.args.ppo_advantage_min_scale,
            )
            ema_scale_clamped = (
                raw_ema_scale < self.args.ppo_advantage_min_scale
            )

        frozen_window = None
        advantage_mean = None
        advantage_std = None
        target_prepass_milliseconds = 0.0
        if use_optimizer_window:
            (
                frozen_window,
                advantage_mean,
                advantage_std,
                target_prepass_milliseconds,
            ) = self._precompute_ppo_optimizer_window_targets(
                window,
                global_valid_token_count=global_valid_token_count,
            )

        local_reduction_stats = PPOReductionStats.zeros(self.device)
        local_version_stats = torch.zeros(
            2,
            device=self.device,
            dtype=torch.float64,
        )
        for pack_index, prepared in enumerate(window):
            batch = move_batch_to_device(prepared.batch, self.device)
            if frozen_window is not None:
                frozen = frozen_window[pack_index]
                self._restore_ppo_rng_state(
                    self.device,
                    frozen.cpu_rng_state,
                    frozen.cuda_rng_state,
                )
                view = self._forward_ppo_token_view(
                    batch,
                    max_seqlen=prepared.max_seqlen,
                )
                self._validate_frozen_ppo_targets(view, frozen)
                raw_advantages = frozen.raw_advantages.to(
                    device=self.device,
                    dtype=torch.float32,
                )
                returns = frozen.returns.to(
                    device=self.device,
                    dtype=torch.float32,
                )
                assert advantage_mean is not None
                assert advantage_std is not None
                actor_advantages = self._normalize_ppo_advantages(
                    raw_advantages,
                    mean=advantage_mean,
                    std=advantage_std,
                    eps=self.args.ppo_adv_norm_eps,
                )
                trajectory_loss_sum, reduction_stats = (
                    self._compute_ppo_loss_from_targets(
                        view,
                        raw_advantages=raw_advantages,
                        actor_advantages=actor_advantages,
                        returns=returns,
                    )
                )
            else:
                trajectory_loss_sum, reduction_stats = (
                    self._compute_packed_ppo_reduction_sums(
                        batch,
                        max_seqlen=prepared.max_seqlen,
                        ema_mean=ema_mean,
                        ema_scale=ema_scale,
                    )
                )
            backward_loss = trajectory_loss_sum * (
                float(self.fsdp_world_size)
                / float(global_valid_trajectory_count)
            )
            backward_loss.backward()
            if reduction_stats.shape != local_reduction_stats.shape:
                raise RuntimeError(
                    "Unexpected packed PPO statistics shape: "
                    f"{tuple(reduction_stats.shape)} != "
                    f"{tuple(local_reduction_stats.shape)}"
                )
            local_reduction_stats.add_(reduction_stats)
            local_version_stats[0] += prepared.version_lag_sum
            local_version_stats[1] += prepared.sample_count
            self.train_micro_step += 1
            del batch, trajectory_loss_sum, reduction_stats, backward_loss
            if frozen_window is not None:
                del view, raw_advantages, actor_advantages, returns, frozen

        dist.all_reduce(local_reduction_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_version_stats, op=dist.ReduceOp.SUM)

        stats = PPOReductionStats.unpack(
            local_reduction_stats
        ).to_float_dict()
        stats_valid_trajectory_count = int(
            stats["valid_trajectory_count"]
        )
        if stats_valid_trajectory_count != global_valid_trajectory_count:
            raise RuntimeError(
                "Prepared and forward PPO valid-trajectory counts differ: "
                f"{global_valid_trajectory_count} != "
                f"{stats_valid_trajectory_count}."
            )
        raw_advantage_count = stats["raw_advantage_count"]
        if raw_advantage_count != float(global_valid_token_count):
            raise RuntimeError(
                "Global raw advantage count does not match global valid-token "
                f"count: {raw_advantage_count} != "
                f"{global_valid_token_count}."
            )
        raw_advantage_mean = (
            stats["raw_advantage_sum"] / raw_advantage_count
        )
        raw_advantage_variance = max(
            stats["raw_advantage_sq_sum"] / raw_advantage_count
            - raw_advantage_mean * raw_advantage_mean,
            0.0,
        )
        if not math.isfinite(raw_advantage_mean) or not math.isfinite(
            raw_advantage_variance
        ):
            raise RuntimeError(
                "Global PPO raw advantage moments must be finite."
            )

        candidate_ema_mean = None
        candidate_ema_variance = None
        if ema_mode:
            if not ema_initialized:
                candidate_ema_mean = raw_advantage_mean
                candidate_ema_variance = raw_advantage_variance
            else:
                assert ema_mean is not None
                assert ema_variance is not None
                alpha = 1.0 - self.args.ppo_advantage_ema_beta
                delta = raw_advantage_mean - ema_mean
                candidate_ema_mean = (
                    self.args.ppo_advantage_ema_beta * ema_mean
                    + alpha * raw_advantage_mean
                )
                candidate_ema_variance = max(
                    self.args.ppo_advantage_ema_beta * ema_variance
                    + alpha * raw_advantage_variance
                    + self.args.ppo_advantage_ema_beta
                    * alpha
                    * delta
                    * delta,
                    0.0,
                )
            if not math.isfinite(candidate_ema_mean) or not math.isfinite(
                candidate_ema_variance
            ):
                raise RuntimeError(
                    "Updated PPO advantage EMA moments must be finite."
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
        if ema_mode:
            assert candidate_ema_mean is not None
            assert candidate_ema_variance is not None
            self.ppo_adv_ema_mean = candidate_ema_mean
            self.ppo_adv_ema_variance = candidate_ema_variance
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step += 1

        token_count = float(global_valid_token_count)
        trajectory_count = float(global_valid_trajectory_count)
        policy_loss_mean = (
            stats["policy_trajectory_sum"] / trajectory_count
        )
        value_loss_mean = (
            stats["value_loss_trajectory_sum"] / trajectory_count
        )
        kl_trajectory_mean = (
            stats["kl_trajectory_sum"] / trajectory_count
        )
        policy_loss_token_mean = stats["policy_token_sum"] / token_count
        value_loss_token_mean = (
            stats["value_loss_token_sum"] / token_count
        )
        kl_token_mean = stats["kl_token_sum"] / token_count
        global_version_lag_sum, global_sample_count = (
            local_version_stats.tolist()
        )
        return {
            **pack_stats,
            "global_ppo_stats": stats,
            "global_valid_token_count": token_count,
            "global_valid_trajectory_count": trajectory_count,
            "global_version_lag_sum": global_version_lag_sum,
            "global_sample_count": global_sample_count,
            "policy_loss_mean": policy_loss_mean,
            "value_loss_mean": value_loss_mean,
            "loss_mean": (
                policy_loss_mean
                + self.args.value_loss_coef * value_loss_mean
                + self.args.old_new_kl_coef * kl_trajectory_mean
            ),
            "policy_loss_token_mean": policy_loss_token_mean,
            "value_loss_token_mean": value_loss_token_mean,
            "kl_trajectory_mean": kl_trajectory_mean,
            "kl_token_mean": kl_token_mean,
            "clip_fraction": stats["clip_count"] / token_count,
            "current_lr": current_lr,
            "ppo_target_prepass_milliseconds": target_prepass_milliseconds,
            "ppo_ema_active": float(ema_mode and ema_initialized),
            "ppo_ema_mean_used": (
                float(ema_mean) if ema_mode and ema_initialized else 0.0
            ),
            "ppo_ema_scale_used": (
                float(ema_scale) if ema_scale is not None else 0.0
            ),
            "ppo_ema_scale_clamped": float(ema_scale_clamped),
        }

    def _run_grpo_optimizer_step(
        self,
        trainer_version: float,
    ) -> Dict[str, float]:
        """Run one globally trajectory-normalized packed GRPO step."""
        window = self._prepare_varlen_optimizer_window(trainer_version)
        pack_stats = self._reduce_varlen_pack_stats(window)
        local_valid_token_count = sum(
            pack.valid_token_count for pack in window
        )
        local_valid_trajectory_count = sum(
            pack.valid_trajectory_count for pack in window
        )
        global_reduction_counts = torch.tensor(
            [local_valid_token_count, local_valid_trajectory_count],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(
            global_reduction_counts,
            op=dist.ReduceOp.SUM,
        )
        (
            global_valid_token_count,
            global_valid_trajectory_count,
        ) = (int(value) for value in global_reduction_counts.tolist())
        if global_valid_token_count <= 0:
            raise RuntimeError(
                "Global Varlen optimizer window contains no valid "
                "response tokens."
            )
        if global_valid_trajectory_count <= 0:
            raise RuntimeError(
                "Global GRPO optimizer window contains no valid "
                "trajectories."
            )

        local_reduction_stats = GRPOReductionStats.zeros(self.device)
        local_version_stats = torch.zeros(
            2,
            device=self.device,
            dtype=torch.float64,
        )
        for prepared_pack in window:
            batch = move_batch_to_device(prepared_pack.batch, self.device)
            trajectory_loss_sum, loss_stats = self._compute_rl_loss(
                batch,
                max_seqlen=prepared_pack.max_seqlen,
            )
            backward_loss = trajectory_loss_sum * (
                float(self.fsdp_world_size)
                / float(global_valid_trajectory_count)
            )
            backward_loss.backward()

            reduction_stats = loss_stats["reduction_stats"]
            if not isinstance(reduction_stats, torch.Tensor):
                raise TypeError(
                    "Packed loss statistics must remain an accelerator tensor."
                )
            if reduction_stats.shape != local_reduction_stats.shape:
                raise RuntimeError(
                    "Unexpected packed loss statistics shape: "
                    f"{tuple(reduction_stats.shape)} != "
                    f"{tuple(local_reduction_stats.shape)}"
                )
            local_reduction_stats.add_(reduction_stats)
            local_version_stats[0] += prepared_pack.version_lag_sum
            local_version_stats[1] += prepared_pack.sample_count
            self.train_micro_step += 1
            del batch, trajectory_loss_sum, backward_loss, reduction_stats

        dist.all_reduce(local_reduction_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_version_stats, op=dist.ReduceOp.SUM)
        reduced_stats = (
            GRPOReductionStats.unpack(
                local_reduction_stats
            ).to_float_dict()
        )
        global_policy_trajectory_sum = reduced_stats[
            "policy_trajectory_sum"
        ]
        global_kl_trajectory_sum = reduced_stats[
            "old_new_kl_k3_trajectory_sum"
        ]
        global_kl_token_sum = reduced_stats["old_new_kl_k3_token_sum"]
        global_clip_count = reduced_stats["ppo_clip_count"]
        stats_valid_trajectory_count = int(
            reduced_stats["valid_trajectory_count"]
        )
        if stats_valid_trajectory_count != global_valid_trajectory_count:
            raise RuntimeError(
                "Prepared and forward GRPO valid-trajectory counts differ: "
                f"{global_valid_trajectory_count} != "
                f"{stats_valid_trajectory_count}."
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
        trajectory_count = float(global_valid_trajectory_count)
        policy_loss_trajectory_mean = (
            global_policy_trajectory_sum / trajectory_count
        )
        kl_trajectory_mean = global_kl_trajectory_sum / trajectory_count
        kl_token_mean = global_kl_token_sum / token_count

        return {
            **pack_stats,
            "policy_trajectory_metric_sum": global_policy_trajectory_sum,
            "kl_trajectory_metric_sum": global_kl_trajectory_sum,
            "kl_token_metric_sum": global_kl_token_sum,
            "clip_metric_sum": global_clip_count,
            "trajectory_metric_weight": trajectory_count,
            "token_metric_weight": token_count,
            "global_valid_token_count": token_count,
            "global_valid_trajectory_count": trajectory_count,
            "global_version_lag_sum": global_version_lag_sum,
            "global_sample_count": global_sample_count,
            "loss_mean": (
                policy_loss_trajectory_mean
                + self.args.old_new_kl_coef * kl_trajectory_mean
            ),
            "policy_loss_mean": policy_loss_trajectory_mean,
            "kl_trajectory_mean": kl_trajectory_mean,
            "kl_token_mean": kl_token_mean,
            "clip_fraction": global_clip_count / token_count,
            "current_lr": current_lr,
        }

    def train_until_next_sync(
        self,
        num_optimizer_steps: int = 100,
    ) -> Dict[str, float]:
        """Train through one sync segment using the algorithm-specific path."""
        if num_optimizer_steps < 1:
            raise ValueError("num_optimizer_steps must be >= 1")
        start_optimizer_step = self.optimizer_step
        target_optimizer_step = min(
            self.optimizer_step + num_optimizer_steps,
            self.args.max_steps,
        )

        global_steps = []
        aggregate_ppo_stats = {
            name: 0.0 for name in PPOReductionStats.names()
        }
        while self.optimizer_step < target_optimizer_step:
            trainer_version = (
                self.optimizer_step / self.args.sync_every_optimizer_steps
            )
            if self.args.rl_algorithm == "ppo":
                step_stats = self._run_ppo_optimizer_step(trainer_version)
                for name, value in step_stats["global_ppo_stats"].items():
                    aggregate_ppo_stats[name] += value
            else:
                step_stats = self._run_grpo_optimizer_step(trainer_version)
            global_steps.append(step_stats)
            if (
                self.rank == 0
                and self.optimizer_step % self.args.log_every == 0
            ):
                value_text = (
                    f" value={step_stats['value_loss_mean']:.6f}"
                    if self.args.rl_algorithm == "ppo"
                    else ""
                )
                trajectory_kl_text = (
                    " kl_trajectory_mean="
                    f"{step_stats['kl_trajectory_mean']:.6f}"
                )
                print(
                    "[train] "
                    f"optimizer_step={self.optimizer_step} "
                    f"loss={step_stats['loss_mean']:.6f}"
                    f"{value_text} "
                    f"{trajectory_kl_text} "
                    f"kl_token_mean={step_stats['kl_token_mean']:.6f} "
                    f"clip_frac={step_stats['clip_fraction']:.4f} "
                    f"tokens={step_stats['global_valid_token_count']:.0f} "
                    f"lr={step_stats['current_lr']:.8g}"
                )

        valid_tokens = sum(
            float(step["global_valid_token_count"])
            for step in global_steps
        )
        version_lag_sum = sum(
            float(step["global_version_lag_sum"])
            for step in global_steps
        )
        sample_count = sum(
            float(step["global_sample_count"])
            for step in global_steps
        )
        result = {
            "rank": self.rank,
            "optimizer_steps_run": self.optimizer_step - start_optimizer_step,
            "optimizer_step": self.optimizer_step,
            "micro_step": self.train_micro_step,
            "reached_max_steps": self.optimizer_step >= self.args.max_steps,
            "segment_valid_tokens": valid_tokens,
            "segment_version_lag_mean": (
                version_lag_sum / sample_count if sample_count else 0.0
            ),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        pack_token_count = sum(
            float(step["global_pack_token_count"])
            for step in global_steps
        )
        pack_count = sum(
            float(step["global_pack_count"])
            for step in global_steps
        )
        pack_sample_count = sum(
            float(step["global_pack_sample_count"])
            for step in global_steps
        )
        pack_cpu_milliseconds = sum(
            float(step["global_pack_cpu_milliseconds"])
            for step in global_steps
        )
        pack_capacity = pack_count * float(self.args.train_token_budget)
        result.update(
            {
                "segment_pack_token_utilization": (
                    pack_token_count / pack_capacity
                    if pack_capacity > 0
                    else 0.0
                ),
                "segment_pack_sample_count": (
                    pack_sample_count / pack_count
                    if pack_count > 0
                    else 0.0
                ),
                "segment_pack_max_sequence_length": max(
                    (
                        float(step["global_pack_max_seqlen"])
                        for step in global_steps
                    ),
                    default=0.0,
                ),
                "segment_pack_cpu_milliseconds": (
                    pack_cpu_milliseconds / pack_count
                    if pack_count > 0
                    else 0.0
                ),
            }
        )

        if self.args.rl_algorithm == "grpo":
            trajectory_metric_weight = sum(
                float(step["trajectory_metric_weight"])
                for step in global_steps
            )
            token_metric_weight = sum(
                float(step["token_metric_weight"])
                for step in global_steps
            )
            policy_trajectory_metric_sum = sum(
                float(step["policy_trajectory_metric_sum"])
                for step in global_steps
            )
            kl_trajectory_metric_sum = sum(
                float(step["kl_trajectory_metric_sum"])
                for step in global_steps
            )
            kl_token_metric_sum = sum(
                float(step["kl_token_metric_sum"])
                for step in global_steps
            )
            clip_metric_sum = sum(
                float(step["clip_metric_sum"]) for step in global_steps
            )
            trajectory_denominator = max(trajectory_metric_weight, 1.0)
            token_denominator = max(token_metric_weight, 1.0)
            policy_mean = (
                policy_trajectory_metric_sum / trajectory_denominator
            )
            kl_trajectory_mean = (
                kl_trajectory_metric_sum / trajectory_denominator
            )
            kl_token_mean = kl_token_metric_sum / token_denominator
            result.update(
                {
                    "segment_loss_mean": (
                        policy_mean
                        + self.args.old_new_kl_coef * kl_trajectory_mean
                    ),
                    "segment_policy_loss_mean": policy_mean,
                    "segment_value_loss_mean": 0.0,
                    "segment_kl_mean": kl_trajectory_mean,
                    "segment_kl_trajectory_mean": kl_trajectory_mean,
                    "segment_kl_token_mean": kl_token_mean,
                    "segment_clip_frac": (
                        clip_metric_sum / token_denominator
                    ),
                    "segment_valid_trajectories": trajectory_metric_weight,
                }
            )
            dist.barrier()
            return result

        token_denominator = max(valid_tokens, 1.0)
        valid_trajectories = sum(
            float(step["global_valid_trajectory_count"])
            for step in global_steps
        )
        trajectory_denominator = max(valid_trajectories, 1.0)
        policy_mean = (
            aggregate_ppo_stats["policy_trajectory_sum"]
            / trajectory_denominator
        )
        value_loss_mean = (
            aggregate_ppo_stats["value_loss_trajectory_sum"]
            / trajectory_denominator
        )
        kl_trajectory_mean = (
            aggregate_ppo_stats["kl_trajectory_sum"]
            / trajectory_denominator
        )
        policy_token_mean = (
            aggregate_ppo_stats["policy_token_sum"] / token_denominator
        )
        value_loss_token_mean = (
            aggregate_ppo_stats["value_loss_token_sum"]
            / token_denominator
        )
        kl_token_mean = (
            aggregate_ppo_stats["kl_token_sum"] / token_denominator
        )

        def moments(prefix):
            mean = (
                aggregate_ppo_stats[f"{prefix}_sum"] / token_denominator
            )
            variance = max(
                aggregate_ppo_stats[f"{prefix}_sq_sum"]
                / token_denominator
                - mean * mean,
                0.0,
            )
            return mean, math.sqrt(variance)

        value_mean, value_std = moments("value")
        return_mean, return_std = moments("return")
        raw_advantage_mean, raw_advantage_std = moments("raw_advantage")
        raw_advantage_rms = math.sqrt(
            max(
                aggregate_ppo_stats["raw_advantage_sq_sum"]
                / token_denominator,
                0.0,
            )
        )
        last_ema_step = next(
            (
                step
                for step in reversed(global_steps)
                if float(step["ppo_ema_active"]) > 0.0
            ),
            None,
        )
        residual_mean = return_mean - value_mean
        residual_variance = max(
            2.0 * value_loss_token_mean - residual_mean * residual_mean,
            0.0,
        )
        return_variance = return_std * return_std
        explained_variance = (
            1.0 - residual_variance / return_variance
            if return_variance > 1e-12
            else 0.0
        )
        boundary_count = (
            aggregate_ppo_stats["terminated_count"]
            + aggregate_ppo_stats["truncated_count"]
        )
        result.update(
            {
                "segment_loss_mean": (
                    policy_mean
                    + self.args.value_loss_coef * value_loss_mean
                    + self.args.old_new_kl_coef * kl_trajectory_mean
                ),
                "segment_policy_loss_mean": policy_mean,
                "segment_value_loss_mean": value_loss_mean,
                "segment_policy_loss_token_mean": policy_token_mean,
                "segment_value_loss_token_mean": value_loss_token_mean,
                "segment_kl_mean": kl_trajectory_mean,
                "segment_kl_trajectory_mean": kl_trajectory_mean,
                "segment_kl_token_mean": kl_token_mean,
                "segment_clip_frac": (
                    aggregate_ppo_stats["clip_count"] / token_denominator
                ),
                "segment_valid_trajectories": valid_trajectories,
                "segment_value_prediction_mean": value_mean,
                "segment_value_prediction_std": value_std,
                "segment_return_mean": return_mean,
                "segment_return_std": return_std,
                "segment_raw_advantage_mean": raw_advantage_mean,
                "segment_raw_advantage_std": raw_advantage_std,
                "segment_raw_advantage_rms": raw_advantage_rms,
                "segment_ppo_ema_mean_used": (
                    float(last_ema_step["ppo_ema_mean_used"])
                    if last_ema_step is not None
                    else 0.0
                ),
                "segment_ppo_ema_scale_used": (
                    float(last_ema_step["ppo_ema_scale_used"])
                    if last_ema_step is not None
                    else 0.0
                ),
                "segment_ppo_ema_scale_clamped": (
                    float(last_ema_step["ppo_ema_scale_clamped"])
                    if last_ema_step is not None
                    else 0.0
                ),
                "segment_ppo_target_prepass_milliseconds": (
                    sum(
                        float(step["ppo_target_prepass_milliseconds"])
                        for step in global_steps
                    )
                    / max(len(global_steps), 1)
                ),
                "segment_value_mse": (
                    2.0
                    * aggregate_ppo_stats["value_loss_token_sum"]
                    / token_denominator
                ),
                "segment_explained_variance": explained_variance,
                "segment_terminated_token_count": (
                    aggregate_ppo_stats["terminated_count"]
                ),
                "segment_truncated_token_count": (
                    aggregate_ppo_stats["truncated_count"]
                ),
                "segment_bootstrap_fraction": (
                    aggregate_ppo_stats["truncated_count"]
                    / max(boundary_count, 1.0)
                ),
            }
        )
        dist.barrier()
        return result

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
        value_head_state_dict = None
        if self.rank == 0:
            state_dict = {}
            value_head_state_dict = {}

        with torch.no_grad():
            for name, param in self.params_by_scope["all"]:
                full_param = param.full_tensor().detach()
                if self.rank == 0:
                    assert state_dict is not None
                    state_dict[name] = full_param.cpu().contiguous()
                del full_param

            # full_tensor() is collective. Every rank must traverse the
            # independently sharded Value Head parameters in identical order.
            for name, param in self.value_head_params:
                full_param = param.full_tensor().detach()
                if self.rank == 0:
                    assert value_head_state_dict is not None
                    value_head_state_dict[name] = (
                        full_param.cpu().float().contiguous()
                    )
                del full_param

        result = {
            "rank": self.rank,
            "checkpoint_dir": output_dir,
            "optimizer_step": self.optimizer_step,
            "saved": False,
            "critic_saved": False,
        }
        if self.rank == 0:
            assert state_dict is not None
            assert value_head_state_dict is not None
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            self.tokenizer.save_pretrained(output_dir)
            value_weights_path, value_config_path = save_value_head_checkpoint(
                output_dir,
                value_head_state_dict,
                hidden_size=self.value_head.hidden_size,
                bias=self.value_head.bias is not None,
            )
            trainer_state = {
                "optimizer_step": self.optimizer_step,
                "train_micro_step": self.train_micro_step,
                "fsdp_world_size": self.fsdp_world_size,
                "train_mode": self.args.train_mode,
                "rl_algorithm": self.args.rl_algorithm,
                "max_steps": self.args.max_steps,
                "sync_every_optimizer_steps": self.args.sync_every_optimizer_steps,
                "critic": {
                    "architecture": "TokenValueHead",
                    "hidden_size": self.value_head.hidden_size,
                    "dtype": "float32",
                    "weights": os.path.relpath(
                        value_weights_path,
                        output_dir,
                    ),
                    "config": os.path.relpath(
                        value_config_path,
                        output_dir,
                    ),
                },
            }
            state_path = os.path.join(output_dir, "trainer_state.json")
            with open(state_path, "w", encoding="utf-8") as file:
                json.dump(trainer_state, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
            del state_dict
            del value_head_state_dict
            result["saved"] = True
            result["critic_saved"] = True
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
    attempt_count: int = 0
    sync_interrupted_attempts: int = 0
    pending_retry_reason: Literal["sync"] | None = None

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

    def __init__(self, capacity: int, rank: int):
        self.rank = int(rank)
        wait_for_selected_ray_actor_debugger("replay", self.rank)
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


def record_sync_retry_probe(
    diagnostics: IntervalDiagnostics,
    request_output,
    *,
    prompt_tokens: int,
) -> None:
    """Record vLLM timing/token proxies for one sync-retried first token."""
    metrics = getattr(request_output, "metrics", None)
    try:
        queued_ts = float(getattr(metrics, "queued_ts"))
        scheduled_ts = float(getattr(metrics, "scheduled_ts"))
        first_token_ts = float(getattr(metrics, "first_token_ts"))
    except (AttributeError, TypeError, ValueError):
        diagnostics.increment("sync_retry_invalid_timing_metric_count")
    else:
        timestamps_valid = (
            math.isfinite(queued_ts)
            and math.isfinite(scheduled_ts)
            and math.isfinite(first_token_ts)
            and queued_ts > 0.0
            and queued_ts <= scheduled_ts <= first_token_ts
        )
        if timestamps_valid:
            diagnostics.observe(
                "sync_retry_scheduled_to_first_token_ms",
                (first_token_ts - scheduled_ts) * 1000.0,
            )
            diagnostics.observe(
                "sync_retry_queue_ms",
                (scheduled_ts - queued_ts) * 1000.0,
            )
        else:
            diagnostics.increment("sync_retry_invalid_timing_metric_count")

    cached_tokens = getattr(request_output, "num_cached_tokens", None)
    cached_tokens_valid = (
        isinstance(cached_tokens, int)
        and not isinstance(cached_tokens, bool)
        and 0 <= cached_tokens <= prompt_tokens
    )
    if cached_tokens_valid:
        diagnostics.observe(
            "sync_retry_recomputed_tokens",
            prompt_tokens - cached_tokens,
        )
    else:
        diagnostics.increment("sync_retry_invalid_cached_tokens_metric_count")


class InterruptibleGenerationRunner:
    """Run vLLM requests that survive abort-based weight-update pauses."""

    def __init__(
        self,
        engine,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_sequences: List[str] | None = None,
        collect_logprobs: bool = False,
        max_resubmit_retries: int = 200,
        diagnostics: IntervalDiagnostics | None = None,
        active_attempts_gauge: TimeWeightedGauge | None = None,
        clock=time.perf_counter,
    ):
        self.engine = engine
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences or ["</answer>"]
        self.collect_logprobs = collect_logprobs
        self.max_resubmit_retries = max_resubmit_retries
        self.diagnostics = diagnostics
        self.active_attempts_gauge = active_attempts_gauge
        self.clock = clock
        self.version = 0
        self.resume_event = asyncio.Event()
        self.resume_event.set()
        self._active_attempts = 0
        self.total_sync_interrupted_attempts = 0
        self._active_changed = asyncio.Condition()

    def pause(self) -> None:
        self.resume_event.clear()

    def resume(self) -> None:
        self.resume_event.set()

    # 新的engine.generate() attempt开始 +1
    async def _increment_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts += 1
            if self.active_attempts_gauge is not None:
                self.active_attempts_gauge.set(self._active_attempts)
            self._active_changed.notify_all()

    # 一个engine.generate() attempt 结束 -1
    async def _decrement_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts -= 1
            if self.active_attempts_gauge is not None:
                self.active_attempts_gauge.set(self._active_attempts)
            self._active_changed.notify_all()

    # 等待正在跑的 generate attempt 都结束,可能是正常结束，也可能是被abort打断
    async def wait_for_idle(self) -> None:
        async with self._active_changed:
            await self._active_changed.wait_for(lambda: self._active_attempts == 0)

    async def generate(self, state: OnlineGenerationState) -> OnlineGenerationState:
        for attempt in range(1, self.max_resubmit_retries + 1):
            # 如果当前正在weight update的attempt还没结束，就等着，不要开始新的generate attempt
            await self.resume_event.wait()

            retry_reason = state.pending_retry_reason
            state.pending_retry_reason = None
            is_sync_retry = retry_reason == "sync"

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
            attempt_started_at = self.clock()
            first_token_at = None
            first_token_output = None
            attempt_prompt_token_ids = state.restart_prompt_token_ids
            exception_was_sync_interrupt = False
            state.attempt_count += 1
            if self.diagnostics is not None:
                self.diagnostics.increment("attempt_count")

            await self._increment_active_attempts()
            try:
                # 调用vllm生成接口，拿到输出后更新state，如果生成过程中被weight update打断了，engine.generate()会抛出异常，直接进入finally块结束这个attempt
                async for request_output in self.engine.generate(
                    {"prompt_token_ids": attempt_prompt_token_ids},
                    sampling_params,
                    request_id=request_id,
                ):
                    final_output = request_output
                    if (
                        first_token_at is None
                        and _tokens_from_output(request_output)
                    ):
                        first_token_at = self.clock()
                        first_token_output = request_output
                    request_finished = bool(
                        getattr(request_output, "finished", False)
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.resume_event.is_set():
                    raise
                exception_was_sync_interrupt = True
                final_output = None
            finally:
                await self._decrement_active_attempts()

            if final_output is None:
                state.stop_reason = "abort"
                if exception_was_sync_interrupt:
                    state.pending_retry_reason = "sync"
                    state.sync_interrupted_attempts += 1
                    self.total_sync_interrupted_attempts += 1
                    if self.diagnostics is not None:
                        self.diagnostics.increment(
                            "sync_interrupted_attempt_count"
                        )
                continue

            attempt_tokens = _tokens_from_output(final_output)[:remaining]
            attempt_finished_at = self.clock()
            stop_reason = _normalize_stop_reason(
                _finish_reason_from_output(final_output)
            )
            sync_interrupted = (
                stop_reason == "abort" and not self.resume_event.is_set()
            )
            if sync_interrupted:
                state.pending_retry_reason = "sync"
                state.sync_interrupted_attempts += 1
                self.total_sync_interrupted_attempts += 1
                if self.diagnostics is not None:
                    self.diagnostics.increment("sync_interrupted_attempt_count")
            if self.diagnostics is not None and first_token_at is not None:
                self.diagnostics.observe(
                    "ttft_ms",
                    (first_token_at - attempt_started_at) * 1000.0,
                )
                if is_sync_retry and first_token_output is not None:
                    record_sync_retry_probe(
                        self.diagnostics,
                        first_token_output,
                        prompt_tokens=len(attempt_prompt_token_ids),
                    )
                if not sync_interrupted and len(attempt_tokens) >= 2:
                    decode_elapsed = attempt_finished_at - first_token_at
                    self.diagnostics.observe(
                        "tpot_ms",
                        decode_elapsed * 1000.0 / (len(attempt_tokens) - 1),
                    )
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
        wait_for_selected_ray_actor_debugger("infer", 0)
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
            max_model_len=args.vllm_max_model_len,
            enable_prefix_caching=True,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            load_format="dummy",
        )
        self.engine = create_async_engine(**engine_kwargs)
        self.diagnostics = IntervalDiagnostics()
        self.active_requests_gauge = TimeWeightedGauge()
        self.active_attempts_gauge = TimeWeightedGauge()
        self.runner = InterruptibleGenerationRunner(
            self.engine,
            temperature=args.infer_temperature,
            top_p=args.infer_top_p,
            stop_sequences=["\n"],
            collect_logprobs=True,
            diagnostics=self.diagnostics,
            active_attempts_gauge=self.active_attempts_gauge,
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
        request_started_at = time.perf_counter()
        self.active_requests_gauge.increment()
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
        finally:
            self.active_requests_gauge.increment(-1.0)

        result = InferenceResult(
            output_tokens=list(completed_state.output_tokens),
            output_logprobs=list(completed_state.output_logprobs),
            output_versions=list(completed_state.output_versions),
            stop_reason=completed_state.stop_reason,
        )
        request_latency_ms = (time.perf_counter() - request_started_at) * 1000.0
        self.diagnostics.increment("request_count")
        self.diagnostics.increment("output_token_count", len(result.output_tokens))
        self.diagnostics.observe("prompt_tokens", len(state.input_ids))
        self.diagnostics.observe("output_tokens_per_request", len(result.output_tokens))
        self.diagnostics.observe("request_latency_ms", request_latency_ms)
        self.diagnostics.observe("attempts_per_request", state.attempt_count)
        if state.attempt_count > 1:
            self.diagnostics.increment("resubmitted_request_count")
        stop_reason = result.stop_reason or "abort"
        if stop_reason == "tool_calls":
            stop_reason = "stop"
        self.diagnostics.increment(f"stop_reason_{stop_reason}_count")
        self.total_tokens += len(result.output_tokens)
        return result

    def begin_diagnostics_interval(self) -> Dict[str, int]:
        self.diagnostics.reset()
        self.active_requests_gauge.reset_interval()
        self.active_attempts_gauge.reset_interval()
        return {"total_tokens": self.total_tokens}

    def end_diagnostics_interval(self) -> Dict[str, object]:
        result = self.diagnostics.snapshot_and_reset()
        result["gauges"] = {
            "active_requests": self.active_requests_gauge.snapshot(),
            "active_attempts": self.active_attempts_gauge.snapshot(),
        }
        result["total_tokens"] = self.total_tokens
        return result

    async def pause_and_wait_idle(self):
        active_attempts_before_pause = self.runner._active_attempts
        interrupted_before_pause = self.runner.total_sync_interrupted_attempts
        self.runner.pause()
        await self.engine.pause_generation(mode="abort", clear_cache=False)
        await self.runner.wait_for_idle()
        return {
            "active_attempts": active_attempts_before_pause,
            "interrupted_attempts": (
                self.runner.total_sync_interrupted_attempts
                - interrupted_before_pause
            ),
        }

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
    termination_reason: TerminationReason | None = None


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
        wait_for_selected_ray_actor_debugger("rollout", self.worker_id)
        self.tokenizer = build_tokenizer(args, log=False)
        self.game_files = load_textworld_game_files(args)
        self.stopped = False
        self.diagnostics = IntervalDiagnostics()
        self.diagnostics_interval_started_at = time.perf_counter()
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

    def begin_diagnostics_interval(self) -> None:
        self.diagnostics.reset()
        self.diagnostics_interval_started_at = time.perf_counter()

    def end_diagnostics_interval(self) -> Dict[str, object]:
        result = self.diagnostics.snapshot_and_reset()
        result["counters"]["interval_elapsed_seconds"] = max(
            0.0,
            time.perf_counter() - self.diagnostics_interval_started_at,
        )
        return result

    def _compute_step_reward(
        self,
        score_before: float,
        score_after: float,
        won: bool,
        lost: bool,
    ) -> float:
        reward = score_after - score_before
        if won:
            reward += self.args.tw_win_bonus
        if lost:
            reward -= self.args.tw_lost_penalty
        return float(reward)

    def _initial_transcript_ids(self, obs: str, infos: Dict) -> List[int]:
        prompt = format_textworld_prompt(obs, infos, tokenizer=self.tokenizer)
        return list(self.tokenizer.encode(prompt))

    def _append_transcript_user_content(
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

    def _apply_textworld_action_result(
        self,
        pending: TextWorldPendingRequest,
        result: InferenceResult,
    ) -> None:
        postprocess_started_at = time.perf_counter()
        env_step_seconds = 0.0
        state = pending.state
        admissible_commands = state.infos.get("admissible_commands", []) or []
        raw_text = self.tokenizer.decode(
            result.output_tokens,
            skip_special_tokens=True,
        )
        parsed_action = parse_model_action(raw_text, admissible_commands)
        if result.stop_reason == "abort":
            parsed_action = ParsedAction(
                normalized=parsed_action.normalized,
                action=None,
            )
        state.transcript_ids.extend(result.output_tokens)
        if parsed_action.action is not None:
            selected_action = parsed_action.action
            env_step_started_at = time.perf_counter()
            obs, step_score, done, infos = state.env.step(selected_action)
            env_step_seconds = time.perf_counter() - env_step_started_at
            self.diagnostics.observe("env_step_ms", env_step_seconds * 1000.0)
            state.obs = obs
            state.infos = dict(infos)
            state.latest_score = _textworld_score(step_score, infos)
            state.done = bool(done)
            state.won = bool(infos.get("won", False))
            state.lost = bool(infos.get("lost", False))
            if state.won:
                state.termination_reason = "won"
            elif state.lost:
                state.termination_reason = "lost"
            elif state.done:
                state.termination_reason = (
                    "environment_done_without_terminal_signal"
                )
            if not state.done or state.termination_reason not in {"won", "lost"}:
                self._append_transcript_user_content(
                    state.transcript_ids,
                    format_textworld_user_content(state.obs, state.infos),
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
            self._append_transcript_user_content(
                state.transcript_ids,
                format_textworld_illegal_action_feedback(
                    invalid_action,
                    pending.prompt_obs,
                    pending.prompt_infos,
                ),
            )
            reward = (
                0.0
                if result.stop_reason == "abort"
                else -self.args.tw_invalid_action_penalty
            )

        state.step_records.append(
            TextWorldStepRecord(
                training_result=result,
                prompt_ids=list(pending.input_ids),
                reward=reward,
            )
        )
        postprocess_seconds = time.perf_counter() - postprocess_started_at
        self.diagnostics.observe(
            "postprocess_ms",
            max(0.0, postprocess_seconds - env_step_seconds) * 1000.0,
        )

    async def _run_textworld_step_batch(
        self,
        states: List[TextWorldTrajectoryState],
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

        pending = []  # 保存每个请求对应的业务上下文，用于结果返回后更新正确的 trajectory。
        request_refs = []  # 保存异步推理请求的“引用/句柄”，用于等待 vLLM 返回结果。
        for state in active_states:
            prompt_obs = state.obs
            prompt_infos = dict(state.infos)
            input_ids = list(state.transcript_ids)
            if len(input_ids) >= self.args.tw_history_token_window:
                state.done = True
                state.termination_reason = "history_limit"
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
            # 当前 TextWorld step 中“已经提交推理、但结果尚未返回”的请求上下文列表。
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

        inference_wait_started_at = time.perf_counter()
        generation_results = await asyncio.gather(*request_refs)
        self.diagnostics.increment(
            "inference_wait_seconds",
            time.perf_counter() - inference_wait_started_at,
        )
        for pending_request, result in zip(pending, generation_results):
            self._apply_textworld_action_result(
                pending_request,
                result,
            )

        return len(active_states)

    def _build_textworld_episode_rl_sample(
        self,
        state: TextWorldTrajectoryState,
        algorithm: Literal["ppo", "grpo"] | None = None,
        sample_advantage: float | None = None,
    ) -> RLSample | None:
        algorithm = self.args.rl_algorithm if algorithm is None else algorithm
        input_ids: List[int] = []
        labels: List[int] = []
        old_logprobs: List[float] = []
        token_rewards: List[float] = []
        token_terminated: List[bool] = []
        token_truncated: List[bool] = []
        output_versions: List[int] = []

        for record in state.step_records:
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
                token_terminated.extend([False] * len(prompt_delta))
                token_truncated.extend([False] * len(prompt_delta))
                output_versions.extend([-1] * len(prompt_delta))

            result = record.training_result
            if not result.output_tokens:
                continue
            output_tokens = list(result.output_tokens)
            if result.stop_reason == "abort":
                input_ids.extend(output_tokens)
                labels.extend([-100] * len(output_tokens))
                old_logprobs.extend([0.0] * len(output_tokens))
                token_rewards.extend([0.0] * len(output_tokens))
                token_terminated.extend([False] * len(output_tokens))
                token_truncated.extend([False] * len(output_tokens))
                output_versions.extend([-1] * len(output_tokens))
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
            token_terminated.extend([False] * len(output_tokens))
            token_truncated.extend([False] * len(output_tokens))
            output_versions.extend(result.output_versions)

        bootstrap_prediction_position = None
        if algorithm == "ppo":
            if state.termination_reason is None:
                return None
            final_transcript_ids = list(state.transcript_ids)
            if len(input_ids) > len(final_transcript_ids):
                return None
            if final_transcript_ids[: len(input_ids)] != input_ids:
                return None
            final_delta = final_transcript_ids[len(input_ids):]
            if final_delta:
                input_ids.extend(final_delta)
                labels.extend([-100] * len(final_delta))
                old_logprobs.extend([0.0] * len(final_delta))
                token_rewards.extend([0.0] * len(final_delta))
                token_terminated.extend([False] * len(final_delta))
                token_truncated.extend([False] * len(final_delta))
                output_versions.extend([-1] * len(final_delta))

        if all(label == -100 for label in labels):
            return None
        if (
            not input_ids
            or len(input_ids) != len(labels)
            or len(input_ids) != len(old_logprobs)
            or len(input_ids) != len(token_rewards)
            or len(input_ids) != len(token_terminated)
            or len(input_ids) != len(token_truncated)
            or len(input_ids) != len(output_versions)
        ):
            return None
        if len(input_ids) > self.args.tw_history_token_window:
            return None

        if algorithm == "ppo":
            valid_target_indices = [
                index for index, label in enumerate(labels) if label != -100
            ]
            if not valid_target_indices:
                return None
            boundary_index = valid_target_indices[-1]
            if state.termination_reason in {"won", "lost"}:
                token_terminated[boundary_index] = True
            else:
                token_truncated[boundary_index] = True
                if not final_delta:
                    return None
                bootstrap_prediction_position = len(input_ids) - 1
            sample = RawPPOSample(
                input_ids=input_ids,
                labels=labels,
                old_logprobs=old_logprobs,
                token_rewards=token_rewards,
                token_terminated=token_terminated,
                token_truncated=token_truncated,
                output_versions=output_versions,
                bootstrap_prediction_position=bootstrap_prediction_position,
            )
            try:
                validate_raw_ppo_sample(sample)
            except ValueError:
                return None
            return sample
        elif algorithm == "grpo":
            if sample_advantage is None:
                raise ValueError("GRPO samples require a sample advantage.")
            return GRPOSample(
                input_ids=input_ids,
                labels=labels,
                old_logprobs=old_logprobs,
                output_versions=output_versions,
                advantage=float(sample_advantage),
            )
        else:
            raise ValueError(f"Unsupported rl_algorithm: {algorithm}")

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

    async def _run_textworld_batch(self, game_file: str) -> None:
        algorithm = self.args.rl_algorithm
        if algorithm == "ppo":
            batch_size = int(self.args.rollout_batch_size)
        elif algorithm == "grpo":
            batch_size = int(self.args.grpo_group_size)
        else:
            raise ValueError(f"Unsupported rl_algorithm: {algorithm}")

        request_infos = make_textworld_request_infos()
        env_id = textworld.gym.register_game(
            game_file,
            request_infos=request_infos,
            max_episode_steps=self.args.tw_max_episode_steps,
        )
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
                )
                if active_count == 0:
                    break

            if algorithm == "ppo":
                for state in states:
                    if state.termination_reason is None and state.step_records:
                        state.done = True
                        state.termination_reason = "step_limit"
                advantages = [None] * len(states)
            else:
                raw_returns = [
                    sum(record.reward for record in state.step_records)
                    for state in states
                ]
                _, _, advantages = self._compute_grpo_group_advantages(
                    raw_returns
                )

            samples = []
            for state, advantage in zip(states, advantages):
                sample = self._build_textworld_episode_rl_sample(
                    state=state,
                    algorithm=algorithm,
                    sample_advantage=advantage,
                )
                if sample is not None:
                    samples.append(sample)

            if samples:
                self.replay_buffer.add_samples.remote(samples)

            max_score = _textworld_max_score(states[0].infos) if states else 0.0
            for state in states:
                self.diagnostics.increment("episode_count")
                if state.termination_reason == "history_limit":
                    self.diagnostics.increment("history_limit_count")
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
            await self._run_textworld_batch(game_file)
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
    validate_vllm_policy_weight_names(names)
    if not (len(names) == len(dtype_names) == len(shapes)):
        raise ValueError(
            "vLLM policy metadata lengths do not match: "
            f"names={len(names)} dtypes={len(dtype_names)} shapes={len(shapes)}"
        )
    model_gib = summarize_weight_payload(dtype_names, shapes)
    infer_payload_gib = model_gib * (transfer_world_size - 1)
    print(
        f"[sync] {scope} metadata: tensors={len(names)}, "
        f"logical_payload={model_gib:.3f} GiB, "
        f"aggregate_infer_payload={infer_payload_gib:.3f} GiB, "
        "critic_in_vllm_payload=False"
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
    command = format_shell_command(
        sys.argv,
        module="accerl_agent.run_agent_textworld",
    )
    with open(command_path, "w", encoding="utf-8") as file:
        file.write(command)
        file.write("\n")


def format_shell_command(
    argv: List[str],
    *,
    module: str | None = None,
) -> str:
    if not argv and module is None:
        return "python"

    if module is None:
        command = f"python {shlex.quote(argv[0])}"
        argument_start = 1
    else:
        command = f"python -m {shlex.quote(module)}"
        argument_start = 1 if argv else 0

    if len(argv) == argument_start:
        return command

    lines = [f"{command} \\"]
    parts: List[List[str]] = []
    idx = argument_start
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


def save_fsdp_checkpoint(
    fsdp_workers: List[Any],
    checkpoint_dir: str,
) -> str:
    results = ray.get([
        worker.save_checkpoint.remote(checkpoint_dir, "latest")
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
        f"train_token_budget={args.train_token_budget} "
        f"train_pack_candidate_pool_size={args.train_pack_candidate_pool_size} "
        f"train_logprob_mode={args.train_logprob_mode} "
        "ppo_forward_mode="
        f"{'selected_positions' if args.rl_algorithm == 'ppo' else 'inactive'} "
        "ppo_layout="
        f"{'packed' if args.rl_algorithm == 'ppo' else 'inactive'} "
        f"infer_tp_size={args.infer_tp_size} "
        f"infer_size={args.infer_size} "
        f"infer_max_tokens={args.infer_max_tokens} "
        f"vllm_max_model_len={args.vllm_max_model_len} "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"rl_algorithm={args.rl_algorithm} "
        f"grpo_group_size={args.grpo_group_size} "
        f"gae_gamma={args.gae_gamma} "
        f"gae_lambda={args.gae_lambda} "
        f"value_loss_coef={args.value_loss_coef} "
        "ppo_advantage_normalization="
        f"{args.ppo_advantage_normalization} "
        f"tw_lost_penalty={args.tw_lost_penalty} "
        f"tw_invalid_action_penalty={args.tw_invalid_action_penalty}"
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
            ReplayBufferActor.remote(
                capacity=args.replay_capacity,
                rank=rank,
            )
            for rank in range(args.fsdp_world_size)
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
            f"gae_gamma={args.gae_gamma} "
            f"gae_lambda={args.gae_lambda} "
            f"value_loss_coef={args.value_loss_coef} "
            "ppo_advantage_normalization="
            f"{args.ppo_advantage_normalization} "
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
            ray.get([
                worker.begin_diagnostics_interval.remote()
                for worker in rollout_workers
            ])
            infer_stats_start = ray.get(
                infer_actor.begin_diagnostics_interval.remote()
            )
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
            infer_diagnostics = ray.get(
                infer_actor.end_diagnostics_interval.remote()
            )
            infer_stats_end = infer_diagnostics
            rollout_diagnostics = merge_interval_snapshots(
                ray.get([
                    worker.end_diagnostics_interval.remote()
                    for worker in rollout_workers
                ])
            )
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
            infer_counters = infer_diagnostics["counters"]
            infer_gauges = infer_diagnostics["gauges"]
            infer_request_count = float(infer_counters.get("request_count", 0.0))
            rollout_counters = rollout_diagnostics["counters"]
            rollout_episode_count = float(
                rollout_counters.get("episode_count", 0.0)
            )
            rollout_total_worker_seconds = float(
                rollout_counters.get("interval_elapsed_seconds", 0.0)
            )
            infer_dropped_samples = sum(
                int(distribution.get("dropped", 0))
                for distribution in infer_diagnostics["distributions"].values()
            )
            rollout_dropped_samples = sum(
                int(distribution.get("dropped", 0))
                for distribution in rollout_diagnostics["distributions"].values()
            )
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
            policy_loss_mean = (
                sum(
                    float(summary["segment_policy_loss_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            value_loss_mean = (
                sum(
                    float(summary["segment_value_loss_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            kl_token_mean = (
                sum(
                    float(summary["segment_kl_token_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            kl_trajectory_mean = (
                sum(
                    float(summary["segment_kl_trajectory_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            policy_loss_token_mean = None
            value_loss_token_mean = None
            if args.rl_algorithm == "ppo":
                policy_loss_token_mean = (
                    sum(
                        float(summary["segment_policy_loss_token_mean"])
                        for summary in summaries
                    )
                    / len(summaries)
                )
                value_loss_token_mean = (
                    sum(
                        float(summary["segment_value_loss_token_mean"])
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
            ppo_target_prepass_milliseconds = (
                sum(
                    float(
                        summary.get(
                            "segment_ppo_target_prepass_milliseconds",
                            0.0,
                        )
                    )
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
            segment_valid_tokens = float(
                rank0_summary["segment_valid_tokens"]
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
            writer.add_scalar(
                "Train/TotalLoss",
                train_loss_mean,
                tb_step,
            )
            writer.add_scalar(
                "Train/PolicyLoss",
                policy_loss_mean,
                tb_step,
            )
            writer.add_scalar(
                "Train/ValueLoss",
                value_loss_mean,
                tb_step,
            )
            if policy_loss_token_mean is not None:
                writer.add_scalar(
                    "Train/PolicyLossTokenMean",
                    policy_loss_token_mean,
                    tb_step,
                )
            if value_loss_token_mean is not None:
                writer.add_scalar(
                    "Train/ValueLossTokenMean",
                    value_loss_token_mean,
                    tb_step,
                )
            writer.add_scalar(
                "KL/OldNewK3TokenMean",
                kl_token_mean,
                tb_step,
            )
            writer.add_scalar(
                "KL/OldNewK3TrajectoryMean",
                kl_trajectory_mean,
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
            writer.add_scalar(
                "Train/PackTokenUtilization",
                rank0_summary["segment_pack_token_utilization"],
                tb_step,
            )
            writer.add_scalar(
                "Train/PackSampleCount",
                rank0_summary["segment_pack_sample_count"],
                tb_step,
            )
            writer.add_scalar(
                "Train/PackMaxSequenceLength",
                rank0_summary["segment_pack_max_sequence_length"],
                tb_step,
            )
            writer.add_scalar(
                "Train/PackCpuMilliseconds",
                rank0_summary["segment_pack_cpu_milliseconds"],
                tb_step,
            )
            if args.clip_mode == "ppo":
                writer.add_scalar(
                    "Clip/PPOClipFrac",
                    clip_fraction,
                    tb_step,
                )
            if args.rl_algorithm == "ppo":
                writer.add_scalar(
                    "Train/PPOTargetPrepassMilliseconds",
                    ppo_target_prepass_milliseconds,
                    tb_step,
                )
                ppo_metric_tags = {
                    "Value/PredictionMean": "segment_value_prediction_mean",
                    "Value/PredictionStd": "segment_value_prediction_std",
                    "Value/ReturnMean": "segment_return_mean",
                    "Value/ReturnStd": "segment_return_std",
                    "Value/MSE": "segment_value_mse",
                    "Value/ExplainedVariance": "segment_explained_variance",
                    "PPO/TerminatedTokenCount": (
                        "segment_terminated_token_count"
                    ),
                    "PPO/TruncatedTokenCount": (
                        "segment_truncated_token_count"
                    ),
                    "PPO/BootstrapFraction": "segment_bootstrap_fraction",
                    "PPO/RawAdvantageMean": (
                        "segment_raw_advantage_mean"
                    ),
                    "PPO/RawAdvantageStd": "segment_raw_advantage_std",
                    "PPO/RawAdvantageRMS": "segment_raw_advantage_rms",
                }
                if args.ppo_advantage_normalization in (
                    "ema_rms",
                    "ema_zscore",
                ):
                    ppo_metric_tags.update(
                        {
                            "PPO/EMAMeanUsed": "segment_ppo_ema_mean_used",
                            "PPO/EMAScaleUsed": "segment_ppo_ema_scale_used",
                            "PPO/EMAScaleClamped": (
                                "segment_ppo_ema_scale_clamped"
                            ),
                        }
                    )
                for tag, summary_key in ppo_metric_tags.items():
                    writer.add_scalar(
                        tag,
                        float(rank0_summary[summary_key]),
                        tb_step,
                    )
            writer.add_scalar("Infer/TokensPerSec", infer_tokens_per_sec, tb_step)
            infer_metric_values = {
                "Infer/RequestsPerSec": (
                    infer_request_count / max(infer_elapsed, 1e-9)
                ),
                "Infer/RequestCount": infer_request_count,
                "Infer/OutputTokensPerRequest": distribution_scalar(
                    infer_diagnostics, "output_tokens_per_request", "mean"
                ),
                "Infer/PromptTokensMean": distribution_scalar(
                    infer_diagnostics, "prompt_tokens", "mean"
                ),
                "Infer/PromptTokensP50": distribution_scalar(
                    infer_diagnostics, "prompt_tokens", "p50"
                ),
                "Infer/PromptTokensP95": distribution_scalar(
                    infer_diagnostics, "prompt_tokens", "p95"
                ),
                "Infer/RequestLatencyMsMean": distribution_scalar(
                    infer_diagnostics, "request_latency_ms", "mean"
                ),
                "Infer/RequestLatencyMsP50": distribution_scalar(
                    infer_diagnostics, "request_latency_ms", "p50"
                ),
                "Infer/RequestLatencyMsP95": distribution_scalar(
                    infer_diagnostics, "request_latency_ms", "p95"
                ),
                "Infer/TTFTMsMean": distribution_scalar(
                    infer_diagnostics, "ttft_ms", "mean"
                ),
                "Infer/TTFTMsP95": distribution_scalar(
                    infer_diagnostics, "ttft_ms", "p95"
                ),
                "Sync/RetryScheduledToFirstTokenMsMean": (
                    distribution_scalar(
                        infer_diagnostics,
                        "sync_retry_scheduled_to_first_token_ms",
                        "mean",
                    )
                ),
                "Sync/RetryScheduledToFirstTokenMsCount": (
                    distribution_scalar(
                        infer_diagnostics,
                        "sync_retry_scheduled_to_first_token_ms",
                        "count",
                    )
                ),
                "Sync/RetryRecomputedTokensMean": distribution_scalar(
                    infer_diagnostics,
                    "sync_retry_recomputed_tokens",
                    "mean",
                ),
                "Sync/RetryRecomputedTokensCount": distribution_scalar(
                    infer_diagnostics,
                    "sync_retry_recomputed_tokens",
                    "count",
                ),
                "Sync/RetryQueueMsMean": distribution_scalar(
                    infer_diagnostics,
                    "sync_retry_queue_ms",
                    "mean",
                ),
                "Sync/RetryQueueMsCount": distribution_scalar(
                    infer_diagnostics,
                    "sync_retry_queue_ms",
                    "count",
                ),
                "Sync/RetryInvalidTimingMetricCount": float(
                    infer_counters.get(
                        "sync_retry_invalid_timing_metric_count",
                        0.0,
                    )
                ),
                "Sync/RetryInvalidCachedTokensMetricCount": float(
                    infer_counters.get(
                        "sync_retry_invalid_cached_tokens_metric_count",
                        0.0,
                    )
                ),
                "Infer/TPOTMsMean": distribution_scalar(
                    infer_diagnostics, "tpot_ms", "mean"
                ),
                "Infer/TPOTMsP95": distribution_scalar(
                    infer_diagnostics, "tpot_ms", "p95"
                ),
                "Infer/ActiveRequestsMean": float(
                    infer_gauges["active_requests"]["mean"]
                ),
                "Infer/ActiveRequestsMax": float(
                    infer_gauges["active_requests"]["max"]
                ),
                "Infer/ActiveAttemptsMean": float(
                    infer_gauges["active_attempts"]["mean"]
                ),
                "Infer/ActiveAttemptsMax": float(
                    infer_gauges["active_attempts"]["max"]
                ),
                "Infer/AttemptsPerRequest": distribution_scalar(
                    infer_diagnostics, "attempts_per_request", "mean"
                ),
                "Infer/ResubmittedRequestRate": (
                    float(infer_counters.get("resubmitted_request_count", 0.0))
                    / max(infer_request_count, 1.0)
                ),
                "Infer/StopRate": (
                    float(infer_counters.get("stop_reason_stop_count", 0.0))
                    / max(infer_request_count, 1.0)
                ),
                "Infer/LengthRate": (
                    float(infer_counters.get("stop_reason_length_count", 0.0))
                    / max(infer_request_count, 1.0)
                ),
                "Infer/AbortRate": (
                    float(infer_counters.get("stop_reason_abort_count", 0.0))
                    / max(infer_request_count, 1.0)
                ),
                "Infer/DiagnosticsDroppedSamples": infer_dropped_samples,
            }
            rollout_metric_values = {
                "Rollout/EpisodesPerSec": (
                    rollout_episode_count / max(infer_elapsed, 1e-9)
                ),
                "Rollout/InferenceWaitFraction": (
                    float(rollout_counters.get("inference_wait_seconds", 0.0))
                    / max(rollout_total_worker_seconds, 1e-9)
                ),
                "Rollout/EnvStepMsMean": distribution_scalar(
                    rollout_diagnostics, "env_step_ms", "mean"
                ),
                "Rollout/EnvStepMsP95": distribution_scalar(
                    rollout_diagnostics, "env_step_ms", "p95"
                ),
                "Rollout/PostprocessMsMean": distribution_scalar(
                    rollout_diagnostics, "postprocess_ms", "mean"
                ),
                "Rollout/PostprocessMsP95": distribution_scalar(
                    rollout_diagnostics, "postprocess_ms", "p95"
                ),
                "Rollout/HistoryLimitRate": (
                    float(rollout_counters.get("history_limit_count", 0.0))
                    / max(rollout_episode_count, 1.0)
                ),
                "Rollout/DiagnosticsDroppedSamples": rollout_dropped_samples,
            }
            for tag, value in {
                **infer_metric_values,
                **rollout_metric_values,
            }.items():
                writer.add_scalar(tag, float(value), tb_step)
            print(
                "[metrics] "
                f"step={tb_step} loss={train_loss_mean:.6f} "
                f"kl_token={kl_token_mean:.6f} "
                + (
                    f"kl_trajectory={kl_trajectory_mean:.6f} "
                    if kl_trajectory_mean is not None
                    else ""
                )
                + f"clip={clip_fraction:.4f} "
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
                )
                last_checkpoint_step = latest_optimizer_step
                print(f"[checkpoint] Periodic checkpoint ready: {checkpoint_path}")

            if (
                args.max_sync_rounds is not None
                and sync_rounds >= args.max_sync_rounds
            ):
                writer.add_scalar(
                    "Infer/SyncInterruptedAttemptRate", 0.0, tb_step
                )
                writer.add_scalar("Infer/PauseActiveAttempts", 0.0, tb_step)
                writer.flush()
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
            pause_diagnostics = ray.get(
                infer_actor.pause_and_wait_idle.remote()
            )
            pause_active_attempts = float(
                pause_diagnostics["active_attempts"]
            )
            writer.add_scalar(
                "Infer/SyncInterruptedAttemptRate",
                float(pause_diagnostics["interrupted_attempts"])
                / max(pause_active_attempts, 1.0),
                tb_step,
            )
            writer.add_scalar(
                "Infer/PauseActiveAttempts",
                pause_active_attempts,
                tb_step,
            )

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
        choices=("full", "lora"),
        help=(
            "Policy training mode. 'full' is supported; 'lora' is reserved "
            "for the forthcoming adapter-training implementation."
        ),
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
            "RL algorithm. 'ppo' computes token TD(lambda) targets from the "
            "current Value Head when replay is sampled; 'grpo' uses "
            "group-normalized trajectory advantages."
        ),
    )
    parser.add_argument(
        "--gae-gamma",
        type=float,
        default=1.0,
        help="PPO discount factor applied once per valid response token.",
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
            "Penalty subtracted when the model emits an invalid TextWorld "
            "command. Generation aborts caused by synchronization are not "
            "penalized."
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
    parser.add_argument(
        "--train-max-sequences-per-pack",
        type=int,
        default=16,
        help=(
            "Maximum number of independent RLSamples in one packed training "
            "microbatch; this is not a padded tensor batch dimension."
        ),
    )
    parser.add_argument(
        "--train-token-budget",
        type=int,
        required=True,
        help=(
            "Maximum real tokens in one packed training microbatch; must be "
            ">= --max-length."
        ),
    )
    parser.add_argument(
        "--train-pack-candidate-pool-size",
        type=int,
        default=None,
        help=(
            "Replay candidates retained locally for length-aware packing. "
            "Defaults to 4 * --train-max-sequences-per-pack."
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
            "Fixed number of packs prepared per rank for each optimizer step."
        ),
    )
    parser.add_argument(
        "--train-logprob-mode",
        type=str,
        default="full_logits_ce",
        choices=["full_logits_ce", "response_only_lm_head"],
        help=(
            "How GRPO computes per-token logprobs. PPO uses "
            "the native selected-position logits_to_keep path. "
            "'full_logits_ce' keeps the standard model forward but avoids "
            "materializing full log_softmax; "
            "'response_only_lm_head' passes packed prediction indices through "
            "the model-native logits_to_keep API."
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
        "--ppo-advantage-normalization",
        choices=("none", "optimizer_window", "ema_rms", "ema_zscore"),
        default="optimizer_window",
        help=(
            "PPO actor-advantage normalization: exact full optimizer-window "
            "moments, historical EMA RMS/Z-score moments, or none. Defaults "
            "to optimizer_window."
        ),
    )
    parser.add_argument(
        "--ppo-advantage-ema-beta",
        type=float,
        default=0.9,
        help="Per-optimizer-step decay for historical PPO advantage moments.",
    )
    parser.add_argument(
        "--ppo-advantage-min-scale",
        type=float,
        default=1e-3,
        help="Positive scale floor used by EMA PPO normalization modes.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Trace parameter for on-the-fly token TD(lambda) targets.",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help=(
            "Coefficient for unclipped per-token Value MSE, reduced equally "
            "over valid PPO trajectories."
        ),
    )
    parser.add_argument(
        "--ppo-adv-norm-eps",
        type=float,
        default=1e-8,
        help="Epsilon used by PPO advantage normalization.",
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
            "Coefficient for the KL(old || new) k3 penalty. PPO and GRPO "
            "reduce it equally over valid trajectories while retaining a "
            "global-token diagnostic. 0 disables the penalty while keeping "
            "KL metrics."
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
            "Defaults to train_max_sequences_per_pack * "
            "grad_accum_steps * 4."
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
            "that trainer starts sampling. Defaults to "
            "--train-max-sequences-per-pack."
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
    # vLLM 同时调度的最大 active sequence 数。
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences vLLM may schedule concurrently.",
    )
    # vLLM 每次调度迭代最多处理的 token 数。
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
            "vLLM max_model_len override. Defaults to "
            "--tw-history-token-window."
        ),
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
        args.replay_capacity = (
            args.train_max_sequences_per_pack
            * args.grad_accum_steps
            * 4
        )
    if args.train_pack_candidate_pool_size is None:
        args.train_pack_candidate_pool_size = (
            args.train_max_sequences_per_pack * 4
        )
    if args.replay_sample_timeout_seconds is None:
        args.replay_sample_timeout_seconds = 0.0
    if args.min_replay_size_per_rank is None:
        args.min_replay_size_per_rank = args.train_max_sequences_per_pack
    if args.num_rollout_workers is None:
        args.num_rollout_workers = args.fsdp_world_size
    if args.grpo_group_size is None:
        args.grpo_group_size = args.rollout_batch_size
    if args.vllm_max_model_len is None:
        args.vllm_max_model_len = args.tw_history_token_window
    if args.log_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join("runs", "TextWorld_FSDP", timestamp)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.train_mode == "lora":
        raise NotImplementedError(
            "--train-mode lora is reserved but not implemented yet."
        )
    if args.tw_game_limit is not None and args.tw_game_limit < 1:
        raise ValueError("--tw-game-limit must be >= 1 when set")
    if args.tw_max_episode_steps < 1:
        raise ValueError("--tw-max-episode-steps must be >= 1")
    if args.tw_history_token_window < 1:
        raise ValueError("--tw-history-token-window must be >= 1")
    if args.gae_gamma < 0:
        raise ValueError("--gae-gamma must be >= 0")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1]")
    if args.value_loss_coef < 0:
        raise ValueError("--value-loss-coef must be >= 0")
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
    if args.train_max_sequences_per_pack < 1:
        raise ValueError("--train-max-sequences-per-pack must be >= 1")
    if args.train_pack_candidate_pool_size < 1:
        raise ValueError("--train-pack-candidate-pool-size must be >= 1")
    if args.train_token_budget < args.max_length:
        raise ValueError(
            "--train-token-budget must be >= --max-length; "
            f"got {args.train_token_budget} < {args.max_length}"
        )
    if args.dtype == "float32":
        raise ValueError(
            "FlashAttention 2 packed training requires float16, bfloat16, "
            "or auto dtype"
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
    if not 0.0 <= args.ppo_advantage_ema_beta < 1.0:
        raise ValueError("--ppo-advantage-ema-beta must be in [0, 1)")
    if args.ppo_advantage_min_scale <= 0.0:
        raise ValueError("--ppo-advantage-min-scale must be > 0")
    if args.tw_lost_penalty < 0:
        raise ValueError("--tw-lost-penalty must be >= 0")
    if args.tw_invalid_action_penalty < 0:
        raise ValueError("--tw-invalid-action-penalty must be >= 0")
    if args.rollout_stop_timeout <= 0:
        raise ValueError("--rollout-stop-timeout must be > 0")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be >= 1")
    if args.vllm_max_num_batched_tokens < 1:
        raise ValueError("--vllm-max-num-batched-tokens must be >= 1")
    if args.vllm_max_model_len < args.tw_history_token_window:
        raise ValueError(
            "--vllm-max-model-len must be >= --tw-history-token-window; "
            f"got {args.vllm_max_model_len} < {args.tw_history_token_window}"
        )
    if args.max_sync_rounds is not None and args.max_sync_rounds < 0:
        raise ValueError("--max-sync-rounds must be >= 0 when set")


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run_textworld_train(args))


if __name__ == "__main__":
    # Ray serializes classes defined by __main__ by value. Re-import this file
    # through its canonical package name so remote workers are referenced as
    # accerl_agent.agent_textworld.FSDPTrainWorker instead.
    from accerl_agent.agent_textworld import main as canonical_main

    canonical_main()
