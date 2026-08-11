# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL FSDP trainer following AcceRL's asynchronous Replay workflow."""

import argparse
import os
import random
import socket
import time
"""vsiqa"""
from importlib import import_module
"""vsiqa"""
from typing import Dict, Iterable, List, Tuple

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard

"""vsiqa"""
from transformers import Qwen3VLForConditionalGeneration
"""vsiqa"""

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

"""vsiqa"""
from vsi_qa_rlvr.dataloaders.qwen3vl_rl_collator import Qwen3VLRLDataCollator
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
        """vsiqa"""
        num_layers = len(model.model.language_model.layers)
        target_keywords = (
            f"model.language_model.layers.{num_layers - 1}.",
            "lm_head",
        )
        """vsiqa"""
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

        """vsiqa"""
        self.forward_dtype = pick_dtype(args.dtype)
        self.data_collator = Qwen3VLRLDataCollator()

        require_training_attention_backend(args.train_attention_backend)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=self.forward_dtype,
            attn_implementation=args.train_attention_backend,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.to(self.device)
        model.train()
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

    """vsiqa"""
    def _prepare_rl_sample(self, sample: RLSample) -> RLSample | None:
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
    """vsiqa"""

    """vsiqa"""
    def _collate_prepared_rl_samples(
        self,
        prepared_samples: List[Tuple["RLSample", "RLSample"]],
        trainer_version: float,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
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
    """vsiqa"""

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
        labels = batch["labels"][:, 1:]
        """vsiqa"""
        response_mask = batch["loss_mask"][:, 1:]
        """vsiqa"""
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
                pixel_values_videos=batch["pixel_values_videos"],
                video_grid_thw=batch["video_grid_thw"],
                mm_token_type_ids=batch["mm_token_type_ids"],
                use_cache=False,
            )
        logits = outputs.logits[:, :-1, :]
        valid_token_log_probs = self._valid_token_log_probs_from_full_logits(
            logits,
            labels,
            response_mask,
        )
        """vsiqa"""

        if self.args.clip_mode == "none":
            valid_adv = advantages[valid_sample_indices].to(
                valid_token_log_probs.dtype
            )
            valid_objective = valid_adv * valid_token_log_probs
            sample_objective = self._aggregate_valid_objective(
                valid_objective,
                valid_sample_indices,
                response_token_counts,
                batch_size=labels.shape[0],
            )
            loss = -sample_objective.mean()
            return loss, response_token_counts, {}

        old_token_log_probs = batch["old_logprobs"][:, 1:].to(
            valid_token_log_probs.dtype
        )
        valid_old_token_log_probs = old_token_log_probs[response_mask]
        valid_ratio = torch.exp(
            valid_token_log_probs - valid_old_token_log_probs
        )
        valid_adv = advantages[valid_sample_indices].to(
            valid_token_log_probs.dtype
        )

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
