# SPDX-License-Identifier: Apache-2.0
"""Run VSI-QA on AcceRL's original asynchronous FSDP/vLLM workflow."""

import argparse
import asyncio
import json
import os
import shlex
import sys
import time

import ray
from torch.utils.tensorboard import SummaryWriter

from vsi_qa_rlvr.replay import ReplayBufferActor
from vsi_qa_rlvr.rollout import StatsActor, VSIQARolloutWorkerActor
from vsi_qa_rlvr.synchronization import sync_weights_to_vllm
from vsi_qa_rlvr.trainer import (
    TRAIN_ATTENTION_BACKENDS,
    VSIQAFSDPTrainWorker,
    find_open_port,
    get_local_ip,
)
from vsi_qa_rlvr.vllm_rollout_actor import (
    ROLLOUT_ATTENTION_BACKENDS,
    VSIQAVLLMInferenceActor,
)


def parse_args():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Run VSI-QA with AcceRL's original asynchronous workflow."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--output-dir",
        default=(
            "/data/all/luck/runs/AcceRL-Agent/"
            f"vsi_qa_async_rlvr/{timestamp}"
        ),
    )
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-num-cpus", type=int, default=16)

    parser.add_argument("--fsdp-world-size", type=int, default=4)
    parser.add_argument("--infer-tp-size", type=int, default=1)
    parser.add_argument("--infer-dp-size", type=int, default=4)
    parser.add_argument("--infer-actor-max-concurrency", type=int, default=1024)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--sync-every-optimizer-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument(
        "--train-attention-backend",
        choices=TRAIN_ATTENTION_BACKENDS,
        default="flash_attention_2",
    )
    parser.add_argument(
        "--clip-mode",
        choices=("none", "ppo", "gipo", "sapo"),
        default="ppo",
    )
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gipo-sigma", type=float, default=1.0)
    parser.add_argument("--sapo-tau-pos", type=float, default=1.0)
    parser.add_argument("--sapo-tau-neg", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--metrics-window-size", type=int, default=1000)
    parser.add_argument(
        "--metrics-active-timeout-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--replay-capacity", type=int, default=8192)
    parser.add_argument("--replay-wait-sleep-seconds", type=float, default=0.01)
    parser.add_argument(
        "--replay-sample-timeout-seconds",
        type=float,
        default=1800.0,
    )
    parser.add_argument("--num-rollout-workers", type=int, default=40)
    parser.add_argument("--rollout-data-batch-size", type=int, default=1)
    parser.add_argument("--rollout-data-workers", type=int, default=1)
    parser.add_argument("--rollout-prefetch-factor", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--rollout-stop-timeout", type=float, default=600.0)

    parser.add_argument("--infer-max-tokens", type=int, default=512)
    parser.add_argument("--infer-temperature", type=float, default=1.0)
    parser.add_argument("--infer-top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--rollout-attention-backend",
        choices=ROLLOUT_ATTENTION_BACKENDS,
        default="TRITON_ATTN",
    )
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
        default=0.55,
    )
    parser.add_argument("--max-sync-rounds", type=int, default=None)
    return parser.parse_args()


def validate_args(args):
    if not os.path.isdir(args.model_path):
        raise ValueError(
            f"--model-path must be an existing directory: {args.model_path!r}"
        )
    if not os.path.isfile(args.data_path):
        raise ValueError(
            f"--data-path must be an existing parquet file: {args.data_path!r}"
        )
    positive_integer_fields = (
        "fsdp_world_size",
        "ray_num_cpus",
        "infer_tp_size",
        "infer_dp_size",
        "infer_actor_max_concurrency",
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "sync_every_optimizer_steps",
        "max_length",
        "log_every",
        "metrics_window_size",
        "replay_capacity",
        "num_rollout_workers",
        "rollout_data_batch_size",
        "rollout_data_workers",
        "rollout_prefetch_factor",
        "rollout_batch_size",
        "infer_max_tokens",
        "max_model_len",
        "vllm_max_num_batched_tokens",
        "vllm_max_num_seqs",
    )
    for field_name in positive_integer_fields:
        if int(getattr(args, field_name)) < 1:
            raise ValueError(f"--{field_name.replace('_', '-')} must be positive")
    if args.replay_wait_sleep_seconds <= 0:
        raise ValueError("--replay-wait-sleep-seconds must be positive")
    if args.metrics_active_timeout_seconds <= 0:
        raise ValueError("--metrics-active-timeout-seconds must be positive")
    if args.replay_sample_timeout_seconds < 0:
        raise ValueError("--replay-sample-timeout-seconds must be non-negative")
    if args.rollout_stop_timeout <= 0:
        raise ValueError("--rollout-stop-timeout must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.clip_eps <= 0:
        raise ValueError("--clip-eps must be positive")
    if args.gipo_sigma <= 0:
        raise ValueError("--gipo-sigma must be positive")
    if args.sapo_tau_pos <= 0 or args.sapo_tau_neg <= 0:
        raise ValueError("--sapo-tau-pos and --sapo-tau-neg must be positive")
    if args.infer_temperature < 0:
        raise ValueError("--infer-temperature must be non-negative")
    if not 0 < args.infer_top_p <= 1:
        raise ValueError("--infer-top-p must be in (0, 1]")
    if not 0 < args.rollout_gpu_memory_utilization <= 1:
        raise ValueError("--rollout-gpu-memory-utilization must be in (0, 1]")
    if args.max_sync_rounds is not None and args.max_sync_rounds < 0:
        raise ValueError("--max-sync-rounds must be non-negative")


def format_shell_command(argv):
    if not argv:
        return "python"

    command = f"python {shlex.quote(argv[0])}"
    if len(argv) == 1:
        return command

    lines = [f"{command} \\"]
    parts = []
    index = 1
    while index < len(argv):
        part = argv[index]
        if (
            part.startswith("-")
            and index + 1 < len(argv)
            and not argv[index + 1].startswith("-")
        ):
            parts.append([part, argv[index + 1]])
            index += 2
        else:
            parts.append([part])
            index += 1

    for index, part_group in enumerate(parts):
        line = "  " + " ".join(shlex.quote(part) for part in part_group)
        if index + 1 < len(parts):
            line += " \\"
        lines.append(line)
    return "\n".join(lines)


def save_run_config(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "args.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    with open(
        os.path.join(args.output_dir, "command.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(format_shell_command(sys.argv))
        file.write("\n")


async def run_vsi_qa(args):
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(
            num_cpus=args.ray_num_cpus,
            include_dashboard=False,
        )

    inference_gpu_count = args.infer_tp_size * args.infer_dp_size
    cluster_gpu_count = int(ray.cluster_resources().get("GPU", 0))

    os.makedirs(args.output_dir, exist_ok=True)
    args.reward_history_path = os.path.join(
        args.output_dir,
        "reward_history.jsonl",
    )
    save_run_config(args)
    writer = SummaryWriter(args.output_dir)
    print(f"[metrics] TensorBoard log dir: {args.output_dir}")
    print(
        "[data] "
        "dataset=vsi_qa "
        f"data_path={args.data_path!r} "
        f"max_length={args.max_length} "
        f"infer_max_tokens={args.infer_max_tokens} "
        f"rollout_batch_size={args.rollout_batch_size}"
    )
    print(
        "[topology] "
        f"visible_gpus={cluster_gpu_count} "
        f"ray_cpus={args.ray_num_cpus} "
        f"fsdp={args.fsdp_world_size} "
        f"infer_tp={args.infer_tp_size} "
        f"infer_dp={args.infer_dp_size} "
        f"infer_gpus={inference_gpu_count}"
    )

    fsdp_workers = []
    infer_actor = None
    rollout_workers = []
    rollout_refs = []
    try:
        replay_buffers = [
            ReplayBufferActor.remote(capacity=args.replay_capacity)
            for _ in range(args.fsdp_world_size)
        ]
        stats_actor = StatsActor.remote(
            window_size=args.metrics_window_size,
            active_timeout_seconds=args.metrics_active_timeout_seconds,
        )
        print(
            f"[replay] Created {len(replay_buffers)} ReplayBufferActor "
            f"instances (capacity={args.replay_capacity} samples each)."
        )

        fsdp_master_address = get_local_ip()
        fsdp_master_port = find_open_port()
        remote_train_worker = ray.remote(num_gpus=1)(VSIQAFSDPTrainWorker)
        fsdp_workers = [
            remote_train_worker.remote(
                args,
                rank,
                args.fsdp_world_size,
                fsdp_master_address,
                fsdp_master_port,
                replay_buffers[rank],
            )
            for rank in range(args.fsdp_world_size)
        ]
        ray.get([worker.get_rank.remote() for worker in fsdp_workers])
        print(f"[init] {args.fsdp_world_size} FSDP training workers ready.")

        print(
            "[infer-actor] Creating AsyncLLMEngine actor with dummy weights "
            f"(tp={args.infer_tp_size}, dp={args.infer_dp_size}, "
            f"max_num_seqs={args.vllm_max_num_seqs}, "
            f"max_num_batched_tokens={args.vllm_max_num_batched_tokens})..."
        )
        remote_infer_actor = ray.remote(
            num_gpus=inference_gpu_count,
            max_concurrency=args.infer_actor_max_concurrency,
        )(VSIQAVLLMInferenceActor)
        infer_actor = remote_infer_actor.remote(args)
        print("[infer-actor] Actor created.")

        remote_rollout_worker = ray.remote(
            num_gpus=0,
            max_concurrency=2,
        )(VSIQARolloutWorkerActor)

        def check_rollout_workers():
            if not rollout_refs:
                return
            ready, _ = ray.wait(
                rollout_refs,
                num_returns=1,
                timeout=0.0,
            )
            if ready:
                ray.get(ready[0])
                raise RuntimeError(
                    "A VSIQARolloutWorkerActor exited unexpectedly."
                )

        print("[transfer] Setting up weight-transfer endpoint...")
        transfer_address, transfer_port = ray.get(
            fsdp_workers[0].setup_transfer_endpoint.remote()
        )
        print(
            f"[transfer] Endpoint ready at "
            f"{transfer_address}:{transfer_port}"
        )
        transfer_world_size = inference_gpu_count + 1
        print(
            f"[transfer] World size: {transfer_world_size} "
            f"(1 trainer + {inference_gpu_count} vLLM workers)"
        )
        print("[transfer] Initializing NCCL groups...")
        trainer_transfer_ref = (
            fsdp_workers[0].init_weight_transfer_group.remote(
                transfer_world_size
            )
        )
        ray.get(
            infer_actor.init_weight_transfer_engine.remote(
                master_address=transfer_address,
                master_port=transfer_port,
                transfer_world_size=transfer_world_size,
            )
        )
        ray.get(trainer_transfer_ref)
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
        ray.get(
            infer_actor.resume_generation.remote(increment_version=False)
        )
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
            f"vllm_max_num_batched_tokens="
            f"{args.vllm_max_num_batched_tokens}."
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
            infer_started = time.perf_counter()
            train_segment_started = time.perf_counter()
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
            train_segment_elapsed = (
                time.perf_counter() - train_segment_started
            )
            infer_elapsed = time.perf_counter() - infer_started
            infer_stats_end = ray.get(infer_actor.get_stats.remote())
            check_rollout_workers()

            rank0_summary = next(
                item for item in summaries if item["rank"] == 0
            )
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
                infer_stats_end["total_tokens"]
                - infer_stats_start["total_tokens"]
            )
            infer_delta_requests = (
                infer_stats_end["total_requests"]
                - infer_stats_start["total_requests"]
            )
            infer_tokens_per_sec = infer_delta_tokens / max(
                infer_elapsed,
                1e-9,
            )
            infer_requests_per_sec = infer_delta_requests / max(
                infer_elapsed,
                1e-9,
            )
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
            replay_sizes = ",".join(
                str(stats["size"]) for stats in replay_stats
            )
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
            version_lag_mean = (
                sum(
                    float(
                        summary[
                            "train_sample_trainer_version_lag_mean"
                        ]
                    )
                    for summary in summaries
                )
                / len(summaries)
            )
            optimizer_steps_per_sec = (
                rank0_summary["optimizer_steps_run"]
                / max(train_segment_elapsed, 1e-9)
            )
            tensorboard_step = rank0_summary["optimizer_step"]
            writer.add_scalar(
                "Rollout/GlobalRewardSumMean",
                rollout_stats["global_reward_sum_mean"],
                tensorboard_step,
            )
            writer.add_scalar(
                "Rollout/ActiveWorkers",
                rollout_stats["active_workers"],
                tensorboard_step,
            )
            writer.add_scalar(
                "Rollout/ResponseLengthMean",
                rollout_stats["response_length_mean"],
                tensorboard_step,
            )
            writer.add_scalar(
                "Rollout/AbortRate",
                rollout_stats["abort_rate"],
                tensorboard_step,
            )
            writer.add_scalar(
                "Replay/FillRatio",
                replay_fill_ratio,
                tensorboard_step,
            )
            writer.add_scalar(
                "Replay/TrainSampleTrainerVersionLagMean",
                version_lag_mean,
                tensorboard_step,
            )
            writer.add_scalar(
                "Train/LossMeanAcrossRanks",
                train_loss_mean,
                tensorboard_step,
            )
            writer.add_scalar(
                "Train/OptimizerStep",
                tensorboard_step,
                tensorboard_step,
            )
            writer.add_scalar(
                "Train/LearningRate",
                rank0_summary["learning_rate"],
                tensorboard_step,
            )
            writer.add_scalar(
                "Train/OptimizerStepsPerSec",
                optimizer_steps_per_sec,
                tensorboard_step,
            )
            if args.clip_mode == "ppo":
                writer.add_scalar(
                    "Clip/PPOClipFrac",
                    rank0_summary["last_ppo_clip_frac"],
                    tensorboard_step,
                )
            writer.add_scalar(
                "Infer/TokensPerSec",
                infer_tokens_per_sec,
                tensorboard_step,
            )
            writer.flush()

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

            last_sync_elapsed_seconds = await sync_weights_to_vllm(
                infer_actor=infer_actor,
                fsdp_workers=fsdp_workers,
                scope="trainable",
                transfer_world_size=transfer_world_size,
                packed=True,
            )
            writer.add_scalar(
                "Sync/ElapsedSeconds",
                last_sync_elapsed_seconds,
                tensorboard_step,
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
                except Exception as error:
                    print(
                        "[cleanup] Ignoring rollout cancel error: "
                        f"{error!r}"
                    )
    finally:
        if rollout_workers:
            try:
                ray.get([worker.stop.remote() for worker in rollout_workers])
            except Exception as error:
                print(f"[cleanup] Ignoring rollout stop error: {error!r}")
        for ref in rollout_refs:
            try:
                ray.cancel(ref)
            except Exception as error:
                print(f"[cleanup] Ignoring rollout cancel error: {error!r}")
        if infer_actor is not None:
            try:
                ray.get(infer_actor.shutdown.remote())
            except Exception as error:
                print(f"[cleanup] Ignoring InferActor shutdown error: {error!r}")
        if fsdp_workers:
            try:
                ray.get([worker.close.remote() for worker in fsdp_workers])
            except Exception as error:
                print(f"[cleanup] Ignoring FSDP worker close error: {error!r}")
        writer.close()
        ray.shutdown()


def main():
    args = parse_args()
    validate_args(args)
    asyncio.run(run_vsi_qa(args))


if __name__ == "__main__":
    main()
