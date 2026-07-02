# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal FSDP trainer + vLLM interruptible inference sync demo.

8-GPU layout:
  Training  - 4 GPUs, PyTorch FSDP2 (fully_shard)
  Inference - 4 GPUs, vLLM AsyncLLMEngine with EP+DP

This script launches Ray FSDP trainer workers and a vLLM inference engine.
The trainer runs in short optimizer-step segments; at each sync boundary,
generation is paused/aborted, in-flight requests are drained, FSDP weights are
sent to vLLM over NCCL, and interruptible inference resumes with resubmitted
requests.

Assumes a single-node cluster with 8 GPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shlex
import socket
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Literal, Tuple

import ray
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scripts.drgrpo_grader import r1_zero_reward_fn

MODEL_NAME = "/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat"
GSM8K_TRAIN_PATH = "/mnt/data/lcx4/AcceRL/accerl_vllm/data/gsm8k/gsm8k_train.jsonl"
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

FSDP_WORLD_SIZE = 6
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 2
INFER_LOG_EVERY_REQUESTS = 128


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
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.old_logprobs = old_logprobs


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


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the final GSM8K answer after the #### marker."""
    if "####" in answer:
        answer = answer.rsplit("####", maxsplit=1)[-1]
    return answer.strip().replace(",", "")


def load_gsm8k_train_data(
    data_path: str,
    limit: int | None = None,
) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                (
                    item["question"],
                    extract_gsm8k_final_answer(item["answer"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No GSM8K examples loaded from {data_path!r}.")
    return examples


def format_r1_zero_prompt(question: str) -> str:
    return R1_ZERO_PROMPT.replace("{question}", question)


def make_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(examples: List[EncodedExample]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example.input_ids) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []
        old_logprobs = []

        for example in examples:
            pad_len = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [pad_token_id] * pad_len)
            attention_mask.append(example.attention_mask + [0] * pad_len)
            labels.append(example.labels + [-100] * pad_len)
            old_logprobs.append(example.old_logprobs + [0.0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
        }

    return collate


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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
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

        self.transfer_port = None
        self.transfer_master_address = None
        self.model_update_group = None
        print(f"[rank {rank}] FSDP worker ready.")

    def get_rank(self):
        return self.rank

    def get_replay_stats(self):
        return ray.get(self.replay_buffer.get_stats.remote())

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _prepare_rl_sample(self, sample: "RLSample") -> EncodedExample | None:
        input_ids = list(sample.input_ids)
        labels = list(sample.labels)
        attention_mask = list(sample.attention_mask)
        old_response_logprobs = list(sample.old_response_logprobs)
        if (
            self.args.clip_mode != "none"
            and len(sample.response_ids) != len(old_response_logprobs)
        ):
            return None
        if len(sample.response_ids) != len(old_response_logprobs):
            old_response_logprobs = [0.0] * len(sample.response_ids)
        if (
            not input_ids
            or len(input_ids) != len(labels)
            or len(input_ids) != len(attention_mask)
        ):
            return None
        old_logprobs = [0.0] * len(sample.prompt_ids) + old_response_logprobs
        if len(old_logprobs) != len(input_ids):
            return None

        max_length = self.args.max_length
        if len(input_ids) > max_length:
            prompt_ids = list(sample.prompt_ids)
            response_ids = list(sample.response_ids)
            response_logprobs = list(sample.old_response_logprobs)
            if (
                self.args.clip_mode != "none"
                and len(response_ids) != len(response_logprobs)
            ):
                return None
            if len(response_ids) != len(response_logprobs):
                response_logprobs = [0.0] * len(response_ids)
            if not response_ids:
                return None
            response_keep = min(len(response_ids), max_length)
            if prompt_ids and response_keep == max_length and max_length > 1:
                response_keep = max_length - 1
            prompt_keep = min(len(prompt_ids), max_length - response_keep)
            if response_keep < 1:
                return None
            kept_prompt_ids = prompt_ids[:prompt_keep]
            kept_response_ids = response_ids[:response_keep]
            kept_response_logprobs = response_logprobs[:response_keep]
            input_ids = kept_prompt_ids + kept_response_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(kept_prompt_ids) + kept_response_ids
            old_logprobs = [0.0] * len(kept_prompt_ids) + kept_response_logprobs

        if len(input_ids) < 2:
            return None
        if all(label == -100 for label in labels[1:]):
            return None

        return EncodedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            old_logprobs=old_logprobs,
        )

    def _collate_prepared_rl_samples(
        self,
        prepared_samples: List[Tuple["RLSample", EncodedExample]],
        trainer_version: float,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
        kept_samples = [sample for sample, _ in prepared_samples]
        examples = [example for _, example in prepared_samples]

        if not examples:
            raise RuntimeError("No valid RL samples were available for training.")

        batch = self.collate_fn(examples)
        batch = move_batch_to_device(batch, self.device)
        advantages = torch.tensor(
            [sample.advantage for sample in kept_samples],
            dtype=torch.float32,
            device=self.device,
        )
        response_token_counts = [
            sum(1 for label in example.labels[1:] if label != -100)
            for example in examples
        ]
        version_lags = []
        for sample in kept_samples:
            sample_version = max(sample.output_versions) if sample.output_versions else 0
            version_lags.append(max(float(trainer_version) - float(sample_version), 0.0))
        stats = {
            "sample_count": float(len(kept_samples)),
            "reward_mean": (
                sum(sample.reward for sample in kept_samples) / len(kept_samples)
            ),
            "advantage_mean": (
                sum(sample.advantage for sample in kept_samples) / len(kept_samples)
            ),
            "response_tokens": float(sum(response_token_counts)),
            "trainer_version_lag_mean": (
                sum(version_lags) / len(version_lags) if version_lags else 0.0
            ),
        }
        return batch, advantages, stats

    def _next_rl_training_batch(
        self,
        trainer_version: float,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float], Dict[str, int]]:
        collected = []
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
                example = self._prepare_rl_sample(sample)
                if example is not None:
                    collected.append((sample, example))
                    if len(collected) >= self.args.batch_size:
                        break

        batch, advantages, train_stats = self._collate_prepared_rl_samples(
            collected[: self.args.batch_size],
            trainer_version,
        )
        return batch, advantages, train_stats, replay_stats

    def _compute_rl_loss(
        self,
        batch: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        labels = batch["labels"][:, 1:] # labels 对齐 logits 的时间步，去掉第一个 token 的标签（通常是 -100），因为它不对应任何预测
        response_mask = labels.ne(-100) # response_mask 标记哪些位置是有效的响应 token（标签不为 -100），这些位置对应的 log_probs 会被用来计算 loss
        response_token_counts = response_mask.sum(dim=-1).clamp_min(1) # 计算每个样本的响应 token 数量，形状为 [batch_size]，最小值为 1 以避免除零
        valid_positions = response_mask.nonzero(as_tuple=False)
        if valid_positions.numel() == 0:
            raise RuntimeError("No valid response tokens found for RL loss.")

        valid_sample_indices = valid_positions[:, 0]
        if self.args.train_logprob_mode == "full_logits_ce":
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits[:, :-1, :] # logits 对应 input_ids 的每个 token 预测下一个 token，所以去掉最后一个时间步
            valid_token_log_probs = self._valid_token_log_probs_from_full_logits(
                logits,
                labels,
                response_mask,
            )
        elif self.args.train_logprob_mode == "response_only_lm_head":
            valid_token_log_probs = self._valid_token_log_probs_from_response_only_lm_head(
                batch,
                labels,
                response_mask,
            )
        else:
            raise ValueError(
                f"Unsupported train_logprob_mode: {self.args.train_logprob_mode}"
            )

        if self.args.clip_mode == "none":
            valid_adv = advantages[valid_sample_indices].to(valid_token_log_probs.dtype)
            valid_objective = valid_adv * valid_token_log_probs
            sample_objective = self._aggregate_valid_objective(
                valid_objective,
                valid_sample_indices,
                response_token_counts,
                batch_size=labels.shape[0],
            )
            loss = -sample_objective.mean() # 无重要性采样loss，保持每个样本内部按 response token 平均，再对 batch 平均
            return loss, response_token_counts, {}

        old_token_log_probs = batch["old_logprobs"][:, 1:].to(valid_token_log_probs.dtype)
        valid_old_token_log_probs = old_token_log_probs[response_mask]
        valid_ratio = torch.exp(valid_token_log_probs - valid_old_token_log_probs)
        valid_adv = advantages[valid_sample_indices].to(valid_token_log_probs.dtype)

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

        sample_objective = self._aggregate_valid_objective(
            valid_objective,
            valid_sample_indices,
            response_token_counts,
            batch_size=labels.shape[0],
        )
        loss = -sample_objective.mean()

        with torch.no_grad():
            loss_stats = {}
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

    def _valid_token_log_probs_from_response_only_lm_head(
        self,
        batch: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        backbone_outputs = self.model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        if hasattr(backbone_outputs, "last_hidden_state"):
            hidden_states = backbone_outputs.last_hidden_state
        else:
            hidden_states = backbone_outputs[0]

        valid_hidden_states = hidden_states[:, :-1, :][response_mask]
        valid_labels = labels[response_mask]
        if valid_hidden_states.numel() == 0:
            raise RuntimeError("No valid response hidden states found for RL loss.")

        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Model does not define output embeddings for LM logits.")
        valid_logits = output_embeddings(valid_hidden_states)
        # 算rl的log_probs, 不是监督学习的交叉熵，之所以用这个是因为算子优化得好，且结果正好为log_probs
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

        while self.optimizer_step < target_optimizer_step:
            trainer_version = self.optimizer_step / self.args.sync_every_optimizer_steps
            batch, advantages, train_stats, replay_stats = self._next_rl_training_batch(
                trainer_version
            )
            raw_loss, response_token_counts, loss_stats = self._compute_rl_loss(
                batch,
                advantages,
            )

            loss = raw_loss / self.args.grad_accum_steps
            loss.backward()
            segment_losses.append(float(raw_loss.item()))
            segment_version_lags.append(float(train_stats["trainer_version_lag_mean"]))

            self.train_micro_step += 1
            self.last_loss = float(raw_loss.item())
            self.last_reward_mean = float(train_stats["reward_mean"])
            self.last_advantage_mean = float(train_stats["advantage_mean"])
            self.last_response_tokens = float(response_token_counts.sum().item())
            self.last_replay_size = int(replay_stats["size"])
            self.last_total_sampled = int(replay_stats["total_samples_sampled"])
            self.last_ppo_clip_frac = float(loss_stats.get("ppo_clip_frac", 0.0))
            should_step = self.train_micro_step % self.args.grad_accum_steps == 0
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameter_list,
                max_norm=1.0,
            )
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
                    f"ppo_clip_frac={self.last_ppo_clip_frac:.4f}"
                )

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
            "segment_loss_mean": (
                sum(segment_losses) / len(segment_losses) if segment_losses else 0.0
            ),
            "train_sample_trainer_version_lag_mean": (
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


def create_async_engine(**kwargs):
    """Create an AsyncLLMEngine directly (no subclass needed)."""
    engine_args = vllm.AsyncEngineArgs(**kwargs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return vllm.AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )


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
    sample_id: int
    input_ids: List[int]
    requested_max_tokens: int


@dataclass
class InferenceResult:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
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
    prompt_ids: List[int]
    response_ids: List[int]
    old_response_logprobs: List[float]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    reward: float
    advantage: float
    question: str
    ground_truth: str
    format_reward: float
    answer_reward: float
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    output_versions: List[int]
    stop_reason: str | None
    generated_text: str


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
            1 for last_active in self.worker_last_active.values()
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
                sum(self.response_lengths) / response_count if response_count else 0.0
            ),
            "abort_rate": (
                sum(1 for flag in self.abort_flags if flag) / abort_count
                if abort_count else 0.0
            ),
            "active_workers": active_workers,
            "total_episodes": self.total_episodes,
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

    # def size(self) -> int:
    #     return len(self.samples)

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
            "total_samples_evicted": self.total_samples_evicted,
            "total_batches_added": self.total_batches_added,
            "total_batches_sampled": self.total_batches_sampled,
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
        collect_logprobs: bool = False,
        max_resubmit_retries: int = 200,
    ):
        self.engine = engine
        self.temperature = temperature
        self.top_p = top_p
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
                "stop": ["</answer>"],
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
        self.engine = create_async_engine(
            model=args.model_path,
            enforce_eager=True,
            tensor_parallel_size=INFERENCE_TP_SIZE,
            data_parallel_size=INFERENCE_DP_SIZE,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            load_format="dummy",
            gpu_memory_utilization=0.8,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        )
        self.runner = InterruptibleGenerationRunner(
            self.engine,
            temperature=args.infer_temperature,
            top_p=args.infer_top_p,
            collect_logprobs=args.clip_mode != "none",
        )
        self.active_generation_tasks = set()
        self.stats = RepeatingInferenceStats()
        self.next_request_index = 0
        self.pending_futures = set()
        self.stopped = False
        print(
            "[infer-actor] AsyncLLMEngine ready: "
            f"tp={INFERENCE_TP_SIZE} dp={INFERENCE_DP_SIZE} "
            f"continuous_submit=True "
            f"vllm_max_num_seqs={args.vllm_max_num_seqs} "
            f"vllm_max_num_batched_tokens={args.vllm_max_num_batched_tokens} "
            f"infer_temperature={args.infer_temperature} "
            f"infer_top_p={args.infer_top_p}"
        )

    async def start(self):
        self.stopped = False
        print(
            "[infer-actor] Continuous submit mode enabled; "
            "vLLM handles batching internally."
        )
        return {"continuous_submit": True, "inference_loop_task": 0}

    async def request_batch(
        self,
        rollout_worker_id: int,
        batch_id: int,
        input_ids: List[int],
        infer_max_tokens: int,
        num_samples: int, # rollout_worker一次性请求的样本数量
    ) -> List[InferenceResult]:
        if self.stopped:
            raise RuntimeError("VLLMInferenceActor is stopped.")
        if num_samples < 1:
            return []

        requests_to_process = []
        # Infer将RolloutWorker发送的批量请求每个请求发给vLLM一次，vLLM内部负责批量处理和调度，生成结果后再合并返回给RolloutWorker
        for sample_id in range(num_samples):
            request_index = self.next_request_index
            self.next_request_index += 1
            item = InferenceRequestItem(
                request_index=request_index,
                rollout_worker_id=int(rollout_worker_id),
                batch_id=int(batch_id),
                sample_id=int(sample_id),
                input_ids=list(input_ids),
                requested_max_tokens=int(infer_max_tokens),
            )
            requests_to_process.append(item)

        return await self._run_generation_items(requests_to_process)

    async def _run_generation_items(
        self,
        requests_to_process: List[InferenceRequestItem],
    ) -> List[InferenceResult]:
        current_call = asyncio.current_task()
        if current_call is not None:
            self.pending_futures.add(current_call)

        generation_tasks = []
        try:
            for item in requests_to_process:
                state = OnlineGenerationState(
                    index=item.request_index,
                    input_ids=item.input_ids,
                    requested_max_tokens=item.requested_max_tokens,
                )
                task = asyncio.create_task(self.runner.generate(state))
                self.active_generation_tasks.add(task)
                task.add_done_callback(self.active_generation_tasks.discard)
                generation_tasks.append(task)

            try:
                completed_states = await asyncio.gather(
                    *generation_tasks,
                    return_exceptions=True,
                )
            except asyncio.CancelledError:
                for task in generation_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*generation_tasks, return_exceptions=True)
                raise

            results = []
            first_exception = None
            for item, completed_state in zip(requests_to_process, completed_states):
                if isinstance(completed_state, BaseException):
                    if first_exception is None:
                        first_exception = completed_state
                    continue

                result = InferenceResult(
                    request_index=item.request_index,
                    rollout_worker_id=item.rollout_worker_id,
                    batch_id=item.batch_id,
                    sample_id=item.sample_id,
                    output_tokens=list(completed_state.output_tokens),
                    output_logprobs=list(completed_state.output_logprobs),
                    output_versions=list(completed_state.output_versions),
                    stop_reason=completed_state.stop_reason,
                    attempts=completed_state.attempts,
                )
                await self._record_completed_state(result)
                results.append(result)

            if first_exception is not None:
                raise first_exception
            return results
        finally:
            if current_call is not None:
                self.pending_futures.discard(current_call)

    async def _record_completed_state(
        self,
        result: InferenceResult,
    ) -> None:
        self.stats.total_requests += 1
        self.stats.total_tokens += len(result.output_tokens)

        if self.stats.total_requests % INFER_LOG_EVERY_REQUESTS == 0:
            print(
                "[infer-actor] Batch progress: "
                f"completed_requests={self.stats.total_requests} "
                f"total_tokens={self.stats.total_tokens} "
                f"active_generation_tasks={len(self.active_generation_tasks)} "
                f"active_attempts={self.runner._active_attempts} "
                f"latest_worker={result.rollout_worker_id} "
                f"latest_batch={result.batch_id} "
                f"latest_sample={result.sample_id} "
                f"latest_request={result.request_index} "
                f"latest_tokens={len(result.output_tokens)} "
                f"latest_versions={result.version_range}"
            )

    async def pause_and_wait_idle(self):
        self.runner.pause()
        await self.engine.pause_generation(mode="abort", clear_cache=True)
        await self.runner.wait_for_idle()
        print("[infer-actor] Generation paused and in-flight attempts drained.")

    async def resume_generation(self, increment_version: bool = False):
        if increment_version:
            self.runner.version += 1
        await self.engine.resume_generation()
        self.runner.resume()
        print(
            "[infer-actor] Generation resumed: "
            f"weight_version={self.runner.version}"
        )
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


class RolloutWorkerActor:
    """CPU Ray actor that builds prompts and feeds token IDs to InferActor."""

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
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor
        self.worker_id = int(worker_id)
        self.tokenizer = build_tokenizer(args, log=False)
        self.gsm8k_examples = load_gsm8k_train_data(
            args.gsm8k_train_path,
            limit=args.gsm8k_limit,
        )
        print(
            "[rollout] "
            f"worker={self.worker_id} loaded GSM8K examples: "
            f"count={len(self.gsm8k_examples)} "
            f"path={args.gsm8k_train_path!r}"
        )
        self.batch_id = 0
        self.stopped = False

    async def stop(self):
        self.stopped = True

    def compute_reward_info(
        self,
        question: str,
        ground_truth: str,
        prompt: str,
        generated_text: str,
        result: InferenceResult,
    ) -> Dict[str, float]:
        del question, prompt

        if result.stop_reason == "abort" or not result.output_tokens:
            return {
                "format_reward": 0.0,
                "answer_reward": 0.0,
                "reward": 0.0,
            }

        reward_info = r1_zero_reward_fn(generated_text, ground_truth)
        return {
            "format_reward": float(reward_info.get("format_reward", 0.0)),
            "answer_reward": float(reward_info.get("answer_reward", 0.0)),
            "reward": float(reward_info.get("reward", 0.0)),
        }

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

    def build_rl_samples(
        self,
        input_ids: List[int],
        question: str,
        ground_truth: str,
        prompt: str,
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
        advantages = self.compute_group_advantages(rewards)

        return [
            RLSample(
                prompt_ids=list(input_ids),
                response_ids=list(result.output_tokens),
                old_response_logprobs=list(result.output_logprobs),
                input_ids=list(input_ids) + list(result.output_tokens),
                attention_mask=[1] * (len(input_ids) + len(result.output_tokens)),
                labels=[-100] * len(input_ids) + list(result.output_tokens),
                reward=float(reward),
                advantage=float(advantage),
                question=question,
                ground_truth=ground_truth,
                format_reward=float(reward_info["format_reward"]),
                answer_reward=float(reward_info["answer_reward"]),
                rollout_worker_id=self.worker_id,
                batch_id=result.batch_id,
                sample_id=result.sample_id,
                output_versions=list(result.output_versions),
                stop_reason=result.stop_reason,
                generated_text=generated_text,
            )
            for result, generated_text, reward_info, reward, advantage in zip(
                results,
                decoded_texts,
                reward_infos,
                rewards,
                advantages,
            )
        ]

    def sample_rollout_prompt(self) -> Tuple[str, str, str]:
        question, ground_truth = random.choice(self.gsm8k_examples)
        prompt = format_r1_zero_prompt(question)
        return question, ground_truth, prompt

    async def run(self):
        while not self.stopped:
            question, ground_truth, prompt = self.sample_rollout_prompt()
            input_ids = self.tokenizer.encode(prompt)

            current_batch_id = self.batch_id
            self.batch_id += 1
            results = await self.infer_actor.request_batch.remote(
                self.worker_id,
                current_batch_id,
                list(input_ids),
                self.args.infer_max_tokens,
                self.args.rollout_batch_size,
            )
            if not results:
                continue
            
            rl_samples = self.build_rl_samples(
                input_ids=list(input_ids),
                question=question,
                ground_truth=ground_truth,
                prompt=prompt,
                results=results,
            )
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
            version_ranges = sorted({result.version_range for result in results})
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


async def run_weight_sync_demo(args: argparse.Namespace):
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    save_run_config(args)
    writer = SummaryWriter(args.log_dir)
    print(f"[metrics] TensorBoard log dir: {args.log_dir}")
    print(
        "[data] "
        "dataset=gsm8k "
        f"gsm8k_train_path={args.gsm8k_train_path!r} "
        f"gsm8k_limit={args.gsm8k_limit} "
        f"max_length={args.max_length} "
        f"infer_max_tokens={args.infer_max_tokens} "
        f"rollout_batch_size={args.rollout_batch_size}"
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
            f"(capacity={args.replay_capacity} samples each)."
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

        print(
            "[infer-actor] Creating AsyncLLMEngine actor with dummy weights "
            # 调度器同一时刻最多允许多少条 sequence 处于 active/running 状态。
            f"(max_num_seqs={args.vllm_max_num_seqs}, "
            # 一次调度最大的toekn总量
            f"max_num_batched_tokens={args.vllm_max_num_batched_tokens})..."
        )
        remote_infer_actor = ray.remote(
            num_gpus=INFERENCE_TP_SIZE * INFERENCE_DP_SIZE,
            max_concurrency=args.infer_actor_max_concurrency,
        )(VLLMInferenceActor)
        infer_actor = remote_infer_actor.remote(args)
        print("[infer-actor] Actor created.")

        remote_rollout_worker = ray.remote(
            num_gpus=0,
            max_concurrency=2, # 允许 run() 正在跑的时候，stop() 还能被执行。
        )(RolloutWorkerActor)

        def check_rollout_workers() -> None:
            if not rollout_refs:
                return
            ready, _ = ray.wait(rollout_refs, num_returns=1, timeout=0.0)
            if ready:
                ray.get(ready[0])
                raise RuntimeError("A RolloutWorkerActor exited unexpectedly.")

        # --- Weight-transfer setup ---
        print("[transfer] Setting up weight-transfer endpoint...")
        transfer_addr, transfer_port = ray.get(
            fsdp_workers[0].setup_transfer_endpoint.remote()
        )
        print(f"[transfer] Endpoint ready at {transfer_addr}:{transfer_port}")

        transfer_world_size = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE + 1
        print(
            f"[transfer] World size: {transfer_world_size} "
            f"(1 trainer + {INFERENCE_TP_SIZE * INFERENCE_DP_SIZE} vLLM workers)"
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
                replay_buffers[worker_id % args.fsdp_world_size],
                stats_actor,
                worker_id,
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
            f"infer_max_tokens={args.infer_max_tokens} "
            f"infer_temperature={args.infer_temperature} "
            f"infer_top_p={args.infer_top_p} "
            f"vllm_max_num_seqs={args.vllm_max_num_seqs} "
            f"vllm_max_num_batched_tokens={args.vllm_max_num_batched_tokens}."
        )

        sync_rounds = 0
        last_sync_elapsed_seconds = None
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
                f"total_sampled={rank0_summary['last_total_sampled']} "
                f"ppo_clip_frac={rank0_summary['last_ppo_clip_frac']:.4f}"
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
                str(stats["total_samples_sampled"])
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
                f"total_sampled=[{replay_sampled}] "
                f"total_evicted=[{replay_evicted}]"
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
            version_lag_mean = (
                sum(
                    float(summary["train_sample_trainer_version_lag_mean"])
                    for summary in summaries
                )
                / len(summaries)
            )
            optimizer_steps_per_sec = (
                rank0_summary["optimizer_steps_run"] / max(train_segment_elapsed, 1e-9)
            )
            tb_step = rank0_summary["optimizer_step"]
            writer.add_scalar(
                "Rollout/GlobalRewardSumMean",
                rollout_stats["global_reward_sum_mean"],
                tb_step,
            )
            writer.add_scalar(
                "Rollout/ActiveWorkers",
                rollout_stats["active_workers"],
                tb_step,
            )
            writer.add_scalar(
                "Rollout/ResponseLengthMean",
                rollout_stats["response_length_mean"],
                tb_step,
            )
            writer.add_scalar(
                "Rollout/AbortRate",
                rollout_stats["abort_rate"],
                tb_step,
            )
            writer.add_scalar("Replay/FillRatio", replay_fill_ratio, tb_step)
            writer.add_scalar(
                "Replay/TrainSampleTrainerVersionLagMean",
                version_lag_mean,
                tb_step,
            )
            writer.add_scalar("Train/LossMeanAcrossRanks", train_loss_mean, tb_step)
            writer.add_scalar("Train/OptimizerStep", tb_step, tb_step)
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
            if args.clip_mode == "ppo":
                writer.add_scalar(
                    "Clip/PPOClipFrac",
                    rank0_summary["last_ppo_clip_frac"],
                    tb_step,
                )
            writer.add_scalar("Infer/TokensPerSec", infer_tokens_per_sec, tb_step)
            writer.flush()

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
    parser.add_argument("--model-path", default=MODEL_NAME)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--train-mode",
        default="lm_head",
        choices=("lm_head", "last_layer", "full"),
        help="Default lm_head mode is intended to validate the training loop.",
    )
    parser.add_argument(
        "--gsm8k-train-path",
        default=GSM8K_TRAIN_PATH,
        help="Path to GSM8K train jsonl.",
    )
    parser.add_argument(
        "--gsm8k-limit",
        type=int,
        default=None,
        help="Optional number of GSM8K examples to load for sanity checks.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum optimizer steps to run before stopping the demo.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument(
        "--train-logprob-mode",
        type=str,
        default="full_logits_ce",
        choices=["full_logits_ce", "response_only_lm_head"],
        help=(
            "How the trainer computes per-token logprobs. "
            "'full_logits_ce' keeps the standard model forward but avoids "
            "materializing full log_softmax; "

            "TODO: It hasn't been correctly implemented yet."
            "'response_only_lm_head' is an "
            "experimental path that applies the LM head only to response tokens."
        ),
    )
    parser.add_argument(
        "--clip-mode",
        type=str,
        default="none",
        choices=["none", "ppo", "gipo", "sapo"],
        help="Policy objective clipping mode. 'none' keeps the original loss.",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="PPO clipping epsilon and diagnostic outside-clip threshold.",
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
        help="TensorBoard log directory. Defaults to runs/VLLM_FSDP/<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp-world-size", type=int, default=FSDP_WORLD_SIZE)
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
        "--infer-max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per prompt in the weight-sync demo.",
    )
    parser.add_argument(
        "--infer-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rollout generation.",
    )
    parser.add_argument(
        "--infer-top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p for rollout generation.",
    )
    parser.add_argument(
        "--infer-actor-max-concurrency",
        type=int,
        default=1024,
        help="Maximum concurrent async method calls allowed on InferActor.",
    )
    parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=8,
        help="Number of CPU RolloutWorkerActor instances to launch.",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help="Number of duplicate prompts each RolloutWorker submits per batch.",
    )
    parser.add_argument(
        "--rollout-stop-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for RolloutWorkerActor run loops to stop cleanly.",
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
    if args.replay_sample_timeout_seconds is None:
        args.replay_sample_timeout_seconds = 0.0
    if args.log_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.log_dir = os.path.join("runs", "VLLM_FSDP", timestamp)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.gsm8k_train_path):
        raise ValueError(
            "--gsm8k-train-path must point to an existing jsonl file: "
            f"{args.gsm8k_train_path!r}"
        )
    if args.gsm8k_limit is not None and args.gsm8k_limit < 1:
        raise ValueError("--gsm8k-limit must be >= 1 when set")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.replay_capacity < 1:
        raise ValueError("--replay-capacity must be >= 1")
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
    if args.fsdp_world_size < 1:
        raise ValueError("--fsdp-world-size must be >= 1")
    if args.sync_every_optimizer_steps < 1:
        raise ValueError("--sync-every-optimizer-steps must be >= 1")
    if args.infer_max_tokens < 1:
        raise ValueError("--infer-max-tokens must be >= 1")
    if args.infer_temperature < 0.0:
        raise ValueError("--infer-temperature must be >= 0")
    if not 0.0 < args.infer_top_p <= 1.0:
        raise ValueError("--infer-top-p must be in (0, 1]")
    if args.infer_actor_max_concurrency < 1:
        raise ValueError("--infer-actor-max-concurrency must be >= 1")
    if args.num_rollout_workers < 1:
        raise ValueError("--num-rollout-workers must be >= 1")
    if args.rollout_batch_size < 1:
        raise ValueError("--rollout-batch-size must be >= 1")
    if args.rollout_stop_timeout <= 0:
        raise ValueError("--rollout-stop-timeout must be > 0")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be >= 1")
    if args.vllm_max_num_batched_tokens < 1:
        raise ValueError("--vllm-max-num-batched-tokens must be >= 1")
    if args.max_sync_rounds is not None and args.max_sync_rounds < 0:
        raise ValueError("--max-sync-rounds must be >= 0 when set")


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run_weight_sync_demo(args))


if __name__ == "__main__":
    main()
