# SPDX-License-Identifier: Apache-2.0
"""Model-specific weight conversion for AcceRL-to-vLLM synchronization."""

import torch


# Mirrors vLLM Qwen3VLForConditionalGeneration.hf_to_vllm_mapper.
_QWEN3VL_PREFIX_MAP = (
    ("model.visual.", "visual."),
    ("model.language_model.", "language_model.model."),
    ("lm_head.", "language_model.lm_head."),
)

_QWEN3VL_FUSED_PROJECTIONS = (
    (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
    (("gate_proj", "up_proj"), "gate_up_proj"),
)


def _qwen3vl_weight_name(name):
    for source, target in _QWEN3VL_PREFIX_MAP:
        if name.startswith(source):
            return target + name.removeprefix(source)
    return name


def iter_qwen3vl_kernel_weights(named_tensors):
    """Pre-fuse dense Qwen3-VL weights into vLLM TP=1 kernel format."""
    pending = {}
    for name, tensor in named_tensors:
        projection = None
        for parts, fused in _QWEN3VL_FUSED_PROJECTIONS:
            part = next((part for part in parts if f".{part}." in name), None)
            if part is not None:
                projection = parts, part, fused
                break

        if projection is None:
            yield _qwen3vl_weight_name(name), tensor
            continue

        parts, part, fused = projection
        fused_name = _qwen3vl_weight_name(
            name.replace(f".{part}.", f".{fused}.")
        )
        tensors = pending.setdefault(fused_name, [None] * len(parts))
        tensors[parts.index(part)] = tensor
        if all(value is not None for value in tensors):
            yield fused_name, torch.cat(tensors, dim=0)
            del pending[fused_name]


def get_qwen3vl_weight_metadata(named_parameters):
    """Return names, dtypes, and shapes for Qwen3-VL kernel weights."""
    names = []
    dtype_names = []
    shapes = []
    for name, tensor in iter_qwen3vl_kernel_weights(named_parameters):
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
    return names, dtype_names, shapes


VLLM_WEIGHT_CONVERTER_REGISTRY = {
    "qwen3_vl": iter_qwen3vl_kernel_weights,
}


def get_vllm_weight_converter(model_type):
    """Return the registered model-specific converter, if any."""
    return VLLM_WEIGHT_CONVERTER_REGISTRY.get(model_type)
