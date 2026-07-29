# SPDX-License-Identifier: Apache-2.0
"""AcceRL-compatible FSDP-to-vLLM weight synchronization transaction."""

import json
import time

import ray

from vsi_qa_rlvr.trainer import validate_weight_scope


def dtype_nbytes(dtype_name):
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
    for dimension in shape:
        numel *= dimension
    return numel


def summarize_weight_payload(dtype_names, shapes):
    total_weight_bytes = sum(
        numel_from_shape(shape) * dtype_nbytes(dtype_name)
        for dtype_name, shape in zip(dtype_names, shapes)
    )
    return total_weight_bytes / 1024**3


async def sync_weights_to_vllm(
    infer_actor,
    fsdp_workers,
    scope,
    transfer_world_size,
    packed=True,
):
    validate_weight_scope(scope)
    end_to_end_started = time.perf_counter()
    metadata_started = time.perf_counter()
    weight_metadata = ray.get(
        fsdp_workers[0].get_weight_metadata.remote(scope)
    )
    names = weight_metadata["names"]
    dtype_names = weight_metadata["dtype_names"]
    shapes = weight_metadata["shapes"]
    is_checkpoint_format = weight_metadata["is_checkpoint_format"]
    metadata_rpc_seconds = time.perf_counter() - metadata_started
    model_gib = summarize_weight_payload(dtype_names, shapes)
    infer_payload_gib = model_gib * (transfer_world_size - 1)
    print(
        f"[sync] {scope} metadata: tensors={len(names)}, "
        f"format={'checkpoint' if is_checkpoint_format else 'kernel'}, "
        f"logical_payload={model_gib:.3f} GiB, "
        f"aggregate_infer_payload={infer_payload_gib:.3f} GiB"
    )

    start_update_started = time.perf_counter()
    start_update_result = ray.get(infer_actor.start_weight_update.remote())
    start_update_rpc_seconds = time.perf_counter() - start_update_started
    start_update_actor_seconds = (
        float(start_update_result["elapsed_seconds"])
        if isinstance(start_update_result, dict)
        and "elapsed_seconds" in start_update_result
        else None
    )

    t0 = time.perf_counter()
    transfer_load_started = time.perf_counter()
    broadcast_handles = [
        worker.gather_and_broadcast_weights.remote(scope=scope, packed=packed)
        for worker in fsdp_workers
    ]
    update_handle = infer_actor.update_weights.remote(
        names=names,
        dtype_names=dtype_names,
        shapes=shapes,
        packed=packed,
    )
    update_result = ray.get(update_handle)
    vllm_update_rpc_seconds = time.perf_counter() - transfer_load_started
    broadcast_results = ray.get(broadcast_handles)
    transfer_load_wall_seconds = time.perf_counter() - transfer_load_started

    vllm_update_actor_seconds = (
        float(update_result["elapsed_seconds"])
        if isinstance(update_result, dict)
        and "elapsed_seconds" in update_result
        else None
    )
    fsdp_rank_timings = sorted(
        [
            {
                "rank": int(result["rank"]),
                "role": str(result["role"]),
                "elapsed_s": round(float(result["elapsed_seconds"]), 6),
            }
            for result in broadcast_results
            if isinstance(result, dict)
            and {"rank", "role", "elapsed_seconds"}.issubset(result)
        ],
        key=lambda item: item["rank"],
    )
    fsdp_rank_max_seconds = max(
        (item["elapsed_s"] for item in fsdp_rank_timings),
        default=0.0,
    )

    finish_update_started = time.perf_counter()
    finish_update_result = ray.get(infer_actor.finish_weight_update.remote())
    finish_update_rpc_seconds = time.perf_counter() - finish_update_started
    finish_update_actor_seconds = (
        float(finish_update_result["elapsed_seconds"])
        if isinstance(finish_update_result, dict)
        and "elapsed_seconds" in finish_update_result
        else None
    )
    elapsed = time.perf_counter() - t0
    end_to_end_seconds = time.perf_counter() - end_to_end_started

    def _format_optional_seconds(value):
        return "na" if value is None else f"{value:.6f}"

    print(
        "[sync-timing] "
        f"scope={scope} "
        f"metadata_rpc_s={metadata_rpc_seconds:.6f} "
        f"start_update_rpc_s={start_update_rpc_seconds:.6f} "
        "start_update_actor_s="
        f"{_format_optional_seconds(start_update_actor_seconds)} "
        f"transfer_load_wall_s={transfer_load_wall_seconds:.6f} "
        f"vllm_update_rpc_s={vllm_update_rpc_seconds:.6f} "
        "vllm_update_actor_s="
        f"{_format_optional_seconds(vllm_update_actor_seconds)} "
        f"fsdp_rank_max_s={fsdp_rank_max_seconds:.6f} "
        "fsdp_rank_timings="
        f"{json.dumps(fsdp_rank_timings, separators=(',', ':'))} "
        f"finish_update_rpc_s={finish_update_rpc_seconds:.6f} "
        "finish_update_actor_s="
        f"{_format_optional_seconds(finish_update_actor_seconds)} "
        f"timed_update_s={elapsed:.6f} "
        f"end_to_end_s={end_to_end_seconds:.6f}"
    )
    print(
        f"[sync] {scope} weight update complete: {elapsed:.3f}s, "
        f"model-sync throughput={model_gib / elapsed:.3f} GiB/s, "
        f"aggregate-infer throughput={infer_payload_gib / elapsed:.3f} GiB/s"
    )
    return elapsed


__all__ = ["sync_weights_to_vllm"]
