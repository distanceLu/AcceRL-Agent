from __future__ import annotations

import json
import os
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


VALUE_HEAD_FORMAT_VERSION = 1
VALUE_HEAD_ARCHITECTURE = "TokenValueHead"
VALUE_HEAD_DIRECTORY = "critic"
VALUE_HEAD_WEIGHTS_NAME = "value_head.safetensors"
VALUE_HEAD_CONFIG_NAME = "value_head_config.json"
VALUE_HEAD_STATE_KEYS = frozenset({"weight", "bias"})
VALUE_HEAD_CONFIG_KEYS = frozenset(
    {
        "format_version",
        "architecture",
        "hidden_size",
        "out_features",
        "bias",
        "saved_dtype",
    }
)


class TokenValueHead(torch.nn.Module):
    """A zero-initialized FP32 scalar head over policy hidden states."""

    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")

        # Construct Parameters directly instead of instantiating nn.Linear.
        # This preserves the caller's CPU/CUDA RNG state exactly.
        self.hidden_size = int(hidden_size)
        self.weight = torch.nn.Parameter(
            torch.zeros((1, self.hidden_size), dtype=torch.float32)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 1:
            raise ValueError("hidden_states must have at least one dimension")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states last dimension must equal hidden_size: "
                f"got {hidden_states.shape[-1]} and {self.hidden_size}"
            )
        return F.linear(
            hidden_states.to(dtype=torch.float32),
            self.weight,
            self.bias,
        ).squeeze(-1)


def value_head_checkpoint_paths(checkpoint_root: str) -> tuple[str, str]:
    critic_dir = os.path.join(checkpoint_root, VALUE_HEAD_DIRECTORY)
    return (
        os.path.join(critic_dir, VALUE_HEAD_WEIGHTS_NAME),
        os.path.join(critic_dir, VALUE_HEAD_CONFIG_NAME),
    )


def build_value_head_config(
    hidden_size: int,
    *,
    bias: bool,
    saved_dtype: str = "float32",
) -> dict[str, Any]:
    return {
        "format_version": VALUE_HEAD_FORMAT_VERSION,
        "architecture": VALUE_HEAD_ARCHITECTURE,
        "hidden_size": int(hidden_size),
        "out_features": 1,
        "bias": bool(bias),
        "saved_dtype": str(saved_dtype),
    }


def _validate_value_head_config(
    config: Mapping[str, Any],
    *,
    hidden_size: int,
    bias: bool,
) -> None:
    if frozenset(config) != VALUE_HEAD_CONFIG_KEYS:
        raise ValueError(
            "Invalid Value Head config keys: "
            f"expected {sorted(VALUE_HEAD_CONFIG_KEYS)}, "
            f"got {sorted(config)}"
        )
    expected = {
        "format_version": VALUE_HEAD_FORMAT_VERSION,
        "architecture": VALUE_HEAD_ARCHITECTURE,
        "hidden_size": int(hidden_size),
        "out_features": 1,
        "bias": bool(bias),
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"Invalid Value Head config field {key!r}: "
                f"expected {expected_value!r}, got {config.get(key)!r}"
            )
    if config["saved_dtype"] not in {"bfloat16", "float32"}:
        raise ValueError(
            "Invalid Value Head saved_dtype: "
            f"{config['saved_dtype']!r}"
        )


def _validate_value_head_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    hidden_size: int,
    bias: bool,
) -> None:
    expected_keys = VALUE_HEAD_STATE_KEYS if bias else frozenset({"weight"})
    actual_keys = frozenset(state_dict)
    if actual_keys != expected_keys:
        raise ValueError(
            "Invalid Value Head tensor keys: "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )

    expected_shapes = {"weight": (1, int(hidden_size))}
    if bias:
        expected_shapes["bias"] = (1,)
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Invalid Value Head tensor shape for {key!r}: "
                f"expected {expected_shape}, got {actual_shape}"
            )


def load_value_head_checkpoint(
    value_head: TokenValueHead,
    checkpoint_root: str,
) -> bool:
    """Load a Critic checkpoint when present; return False for legacy policies."""
    weights_path, config_path = value_head_checkpoint_paths(checkpoint_root)
    weights_exists = os.path.isfile(weights_path)
    config_exists = os.path.isfile(config_path)
    if not weights_exists and not config_exists:
        return False
    if weights_exists != config_exists:
        raise ValueError(
            "Incomplete Value Head checkpoint: both files are required under "
            f"{os.path.dirname(weights_path)}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Value Head config must be a JSON object.")
    bias = value_head.bias is not None
    _validate_value_head_config(
        config,
        hidden_size=value_head.hidden_size,
        bias=bias,
    )

    state_dict = load_file(weights_path)
    _validate_value_head_state(
        state_dict,
        hidden_size=value_head.hidden_size,
        bias=bias,
    )
    value_head.load_state_dict(state_dict)
    return True


def save_value_head_checkpoint(
    checkpoint_root: str,
    state_dict: Mapping[str, torch.Tensor],
    *,
    hidden_size: int,
    bias: bool,
) -> tuple[str, str]:
    """Save an already full, CPU-resident Value Head state dictionary."""
    _validate_value_head_state(
        state_dict,
        hidden_size=hidden_size,
        bias=bias,
    )
    weights_path, config_path = value_head_checkpoint_paths(checkpoint_root)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    cpu_state = {
        key: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key, tensor in state_dict.items()
    }
    save_file(cpu_state, weights_path)
    config = build_value_head_config(
        hidden_size,
        bias=bias,
        saved_dtype="float32",
    )
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return weights_path, config_path
