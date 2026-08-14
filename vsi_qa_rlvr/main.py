# SPDX-License-Identifier: Apache-2.0
"""Run VSI-QA on AcceRL's original asynchronous FSDP/vLLM workflow."""

import argparse
import asyncio
import json
import os
import shlex
import sys
import time
from typing import Any, List

import ray
from torch.utils.tensorboard import SummaryWriter

from vsi_qa_rlvr.replay import ReplayBufferActor
from vsi_qa_rlvr.rollout import RolloutWorkerActor, StatsActor
from vsi_qa_rlvr.synchronization import sync_weights_to_vllm
from vsi_qa_rlvr.trainer import (
    FSDPTrainWorker,
    TRAIN_ATTENTION_BACKENDS,
    find_open_port,
    get_local_ip,
)
from vsi_qa_rlvr.vllm_rollout_actor import (
    ROLLOUT_ATTENTION_BACKENDS,
    VLLMInferenceActor,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run asynchronous RLVR training with Ray FSDP, vLLM rollout, "
            "and NCCL weight synchronization."
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
    """vsiqa"""
    parser.add_argument(
        "--train-mode",
        default="full",
        choices=("full",),
        help="Train all model parameters.",
    )
    """vsiqa"""
    """vsiqa"""
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit-images", type=int, default=16)
    """vsiqa"""
    parser.add_argument(
        "--rl-algorithm",
        type=str,
        default="grpo",
        choices=("ppo", "grpo"),
        help=(
            "RL rollout/advantage algorithm. 'ppo' uses token-level Monte "
            "Carlo advantages; 'grpo' uses group-normalized trajectory "
            "advantages."
        ),
    )
    parser.add_argument(
        "--ppo-gamma",
        type=float,
        default=1.0,
        help="Discount factor for token-level Monte Carlo reward-to-go.",
    )
    """vsiqa"""
    parser.add_argument(
        "--reward-type",
        choices=("incremental_counting", "p3"),
        default="incremental_counting",
    )
    """vsiqa"""
    parser.add_argument(
        "--grpo-group-size",
        type=int,
        default=None,
        help=(
            "Number of complete trajectories sampled per GRPO group. "
            "Defaults to --rollout-batch-size."
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
        help="Maximum optimizer steps to run before stopping training.",
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
        help="TensorBoard log directory. Defaults to runs/Async_RLVR/<timestamp>.",
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
        help="Maximum generated tokens per rollout response.",
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
            "Number of CPU RolloutWorkerActor instances to launch. "
            "Defaults to --fsdp-world-size."
        ),
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help=(
            "Number of parallel PPO rollout samples generated per prompt; "
            "also the default value of --grpo-group-size."
        ),
    )
    parser.add_argument(
        "--rollout-stop-timeout",
        type=float,
        default=10.0,
        help=(
            "Seconds to wait for RolloutWorkerActor run loops to "
            "stop cleanly."
        ),
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
        help="Optional maximum number of trainable-only sync rounds in training.",
    )
    """vsiqa"""
    parser.add_argument(
        "--train-attention-backend",
        choices=TRAIN_ATTENTION_BACKENDS,
        default="flash_attention_2",
    )
    parser.add_argument(
        "--rollout-attention-backend",
        choices=ROLLOUT_ATTENTION_BACKENDS,
        default="FLASH_ATTN",
    )
    """vsiqa"""
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
        args.log_dir = os.path.join("runs", "Async_RLVR", timestamp)
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
    """vsiqa"""
    if args.limit_images < 1:
        raise ValueError("--limit-images must be positive")
    """vsiqa"""
    if args.ppo_gamma < 0:
        raise ValueError("--ppo-gamma must be >= 0")
    """vsiqa"""
    if not os.path.isdir(args.model_path):
        raise ValueError(
            f"--model-path must be an existing directory: {args.model_path!r}"
        )
    if not os.path.isfile(args.data_path):
        raise ValueError(
            f"--data-path must be an existing parquet file: {args.data_path!r}"
        )
    """vsiqa"""
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
    """vsiqa"""
    if (
        args.train_packing == "varlen"
        and args.train_attention_backend != "flash_attention_2"
    ):
        raise ValueError(
            "--train-packing varlen requires "
            "--train-attention-backend flash_attention_2"
        )
    """vsiqa"""
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
            "Async RLVR training requires --num-rollout-workers >= "
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
    """vsiqa"""
    if args.reward_type not in {"incremental_counting", "p3"}:
        raise ValueError(f"Unsupported reward type: {args.reward_type!r}")
    """vsiqa"""
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
        if (
            part.startswith("-")
            and idx + 1 < len(argv)
            and not argv[idx + 1].startswith("-")
        ):
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


async def run_async_rlvr(args: argparse.Namespace):
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    save_run_config(args)
    writer = SummaryWriter(args.log_dir)
    print(f"[metrics] TensorBoard log dir: {args.log_dir}")
    """vsiqa"""
    print(
        "[data] "
        "dataset=vsi_qa "
        f"data_path={args.data_path!r} "
        f"max_length={args.max_length} "
        f"infer_max_tokens={args.infer_max_tokens} "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"grpo_group_size={args.grpo_group_size}"
    )
    """vsiqa"""
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
        )(RolloutWorkerActor)

        def check_rollout_workers() -> None:
            if not rollout_refs:
                return
            ready, _ = ray.wait(rollout_refs, num_returns=1, timeout=0.0)
            if ready:
                ray.get(ready[0])
                raise RuntimeError(
                    "A RolloutWorkerActor exited unexpectedly."
                )

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
        await sync_weights_to_vllm(  # 第一次完整权重传输
            infer_actor=infer_actor,
            fsdp_workers=fsdp_workers,
            scope="all",
            transfer_world_size=transfer_world_size,
            packed=True,
        )
        ray.get(
            infer_actor.resume_generation.remote(increment_version=False)
        )
        print("[sync] Initial full sync complete; generation can start.")

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
            f"rl_algorithm={args.rl_algorithm} "
            f"grpo_group_size={args.grpo_group_size} "
            f"ppo_gamma={args.ppo_gamma} "
            f"ppo_normalize_advantages={args.ppo_normalize_advantages} "
            f"infer_tp_size={args.infer_tp_size} "
            f"infer_size={args.infer_size} "
            f"infer_max_tokens={args.infer_max_tokens} "
            f"infer_temperature={args.infer_temperature} "
            f"infer_top_p={args.infer_top_p} "
            f"vllm_max_num_seqs={args.vllm_max_num_seqs} "
            f"vllm_max_num_batched_tokens="
            f"{args.vllm_max_num_batched_tokens} "
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

            rank0_summary = next(
                item for item in summaries if item["rank"] == 0
            )
            training_reached_max = bool(rank0_summary["reached_max_steps"])
            latest_optimizer_step = int(rank0_summary["optimizer_step"])
            if rank0_summary["optimizer_steps_run"] <= 0:
                print("[train] No optimizer steps left; stopping sync loop.")
                break

            infer_delta_tokens = (
                infer_stats_end["total_tokens"]
                - infer_stats_start["total_tokens"]
            )
            infer_tokens_per_sec = infer_delta_tokens / max(
                infer_elapsed,
                1e-9,
            )

            replay_stats = ray.get([
                worker.get_replay_stats.remote()
                for worker in fsdp_workers
            ])
            rollout_stats = ray.get(stats_actor.get_stats.remote())

            total_replay_size = sum(
                int(stats["size"]) for stats in replay_stats
            )
            total_replay_capacity = sum(
                int(stats["capacity"]) for stats in replay_stats
            )
            replay_fill_ratio = (
                total_replay_size / total_replay_capacity
                if total_replay_capacity > 0
                else 0.0
            )
            train_loss_mean = (
                sum(
                    float(summary["segment_loss_mean"])
                    for summary in summaries
                )
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
            """vsiqa"""
            writer.add_scalar(
                "Rollout/GlobalRewardSumMean",
                rollout_stats["global_reward_sum_mean"],
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
            """vsiqa"""
            writer.add_scalar(
                "Replay/FillRatio",
                replay_fill_ratio,
                tb_step,
            )
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
                "Train/Loss",
                train_loss_mean,
                tb_step,
            )
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
            writer.add_scalar(
                "Train/TokensPerSec",
                train_tokens_per_sec,
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
                    clip_fraction,
                    tb_step,
                )
            writer.add_scalar(
                "Infer/TokensPerSec",
                infer_tokens_per_sec,
                tb_step,
            )
            """vsiqa"""
            print(
                "[metrics] "
                f"step={tb_step} loss={train_loss_mean:.6f} "
                f"kl={kl_token_mean:.6f} clip={clip_fraction:.4f} "
                f"train_tokens_per_sec={train_tokens_per_sec:.2f} "
                f"infer_tokens_per_sec={infer_tokens_per_sec:.2f} "
                f"reward={rollout_stats['global_reward_sum_mean']:.4f}"
            )
            """vsiqa"""
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
                    "[sync] max_sync_rounds reached; letting inference "
                    "finish without more trainable updates."
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
            writer.add_scalar(
                "Sync/ElapsedSeconds",
                sync_elapsed_seconds,
                tb_step,
            )
            writer.flush()
            next_version = ray.get(
                infer_actor.resume_generation.remote(
                    increment_version=True
                )
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


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run_async_rlvr(args))


if __name__ == "__main__":
    main()
