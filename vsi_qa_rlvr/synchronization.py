# SPDX-License-Identifier: Apache-2.0
"""AcceRL-compatible FSDP-to-vLLM weight synchronization transaction."""

import time
from typing import List

import ray

from vsi_qa_rlvr.trainer import validate_weight_scope


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


def summarize_weight_payload(
    dtype_names: List[str],
    shapes: List[List[int]],
) -> float:
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


__all__ = ["sync_weights_to_vllm"]
