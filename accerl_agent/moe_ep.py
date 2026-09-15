from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_to_all_single
from torch.distributed.tensor import DTensor, Shard


@dataclass(frozen=True)
class ExpertParallelSetup:
    """Expert-parallel state installed into one Qwen2-MoE policy."""

    expert_params_by_layer: Dict[nn.Module, frozenset[nn.Parameter]]
    expert_params: Tuple[nn.Parameter, ...]
    expert_modules: Tuple[nn.Module, ...]
    ep_size: int
    local_num_experts: int
    num_experts: int


def estimate_ep_dispatch_temporary_bytes(
    *,
    per_rank_token_budget: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    element_size: int = 2,
) -> dict[str, int]:
    """Estimate dispatcher peaks from the forward tensor lifetimes.

    The estimate deliberately models both balanced routing and the legal
    worst case where one owner receives every rank's assignments. It includes
    the dominant expert MLP intermediates but excludes persistent weights.
    """
    values = {
        "per_rank_token_budget": per_rank_token_budget,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "ep_size": ep_size,
        "element_size": element_size,
    }
    invalid = {name: value for name, value in values.items() if value < 1}
    if invalid:
        raise ValueError(f"EP dispatch estimate inputs must be positive: {invalid}")

    local_assignments = per_rank_token_budget * top_k

    def estimate(received_assignments: int) -> int:
        # Direct owner-ordered gather avoids a separate [T, K, H] expansion.
        dispatch_a2a = (
            local_assignments + received_assignments
        ) * hidden_size * element_size
        expert_regroup = 2 * received_assignments * hidden_size * element_size
        # Input/output plus gate+up and activation/multiply intermediates.
        expert_compute = received_assignments * (
            2 * hidden_size + 4 * intermediate_size
        ) * element_size
        reverse_a2a = (
            received_assignments + local_assignments
        ) * hidden_size * element_size
        source_restore = 2 * local_assignments * hidden_size * element_size
        metadata = (
            local_assignments + received_assignments
        ) * torch.int64.itemsize
        return max(
            dispatch_a2a,
            expert_regroup,
            expert_compute,
            reverse_a2a,
            source_restore,
        ) + metadata

    balanced = estimate(local_assignments)
    worst_case = estimate(local_assignments * ep_size)
    return {
        "balanced_bytes": balanced,
        "worst_case_bytes": worst_case,
        "local_assignments": local_assignments,
        "balanced_received_assignments": local_assignments,
        "worst_case_received_assignments": local_assignments * ep_size,
    }


def _inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse.scatter_(
        0,
        permutation,
        torch.arange(permutation.numel(), device=permutation.device),
    )
    return inverse


def _exchange_split_sizes(
    input_split_sizes: torch.Tensor,
    *,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    if input_split_sizes.ndim != 1:
        raise ValueError("input_split_sizes must be rank-1")
    output_split_sizes = torch.empty_like(input_split_sizes)
    dist.all_to_all_single(output_split_sizes, input_split_sizes, group=group)
    return output_split_sizes


def _ragged_all_to_all_autograd(
    input_tensor: torch.Tensor,
    *,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    output_shape = (sum(output_split_sizes), *input_tensor.shape[1:])
    output = torch.empty(
        output_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    return all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )


def _ragged_all_to_all_metadata(
    input_tensor: torch.Tensor,
    *,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    output_shape = (sum(output_split_sizes), *input_tensor.shape[1:])
    output = torch.empty(
        output_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _parameter_from_expert_shard(
    full_parameter: nn.Parameter,
    *,
    expert_begin: int,
    local_num_experts: int,
    device: torch.device,
    mesh: DeviceMesh,
    run_check: bool,
) -> nn.Parameter:
    if isinstance(full_parameter, DTensor):
        raise TypeError("Expected an unsharded expert parameter before EP setup")
    global_shape = full_parameter.shape
    global_stride = full_parameter.stride()
    local = (
        full_parameter.detach()
        .narrow(0, expert_begin, local_num_experts)
        .clone()
        .to(device=device, dtype=torch.float32)
    )
    distributed = DTensor.from_local(
        local,
        device_mesh=mesh,
        placements=(Shard(0),),
        shape=global_shape,
        stride=global_stride,
        run_check=run_check,
    )
    return nn.Parameter(
        distributed,
        requires_grad=full_parameter.requires_grad,
    )


class Qwen2MoeEPExperts(nn.Module):
    """Qwen2-MoE fused experts sharded by global expert ID."""

    def __init__(
        self,
        source: nn.Module,
        *,
        mesh: DeviceMesh,
        group: dist.ProcessGroup,
        device: torch.device,
        compute_dtype: torch.dtype,
        run_check: bool,
    ) -> None:
        super().__init__()
        if compute_dtype is not torch.bfloat16:
            raise ValueError(
                "Qwen2-MoE EP compute dtype must be torch.bfloat16"
            )
        self.num_experts = int(source.num_experts)
        self.hidden_dim = int(source.hidden_dim)
        self.intermediate_dim = int(source.intermediate_dim)
        self.act_fn = source.act_fn
        self.ep_group = group
        self.ep_size = dist.get_world_size(group)
        self.ep_rank = dist.get_rank(group)
        self.compute_dtype = compute_dtype
        if self.ep_size < 2:
            raise ValueError("Qwen2MoeEPExperts requires EP size >= 2")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"ep_size={self.ep_size}"
            )
        self.local_num_experts = self.num_experts // self.ep_size
        expert_begin = self.ep_rank * self.local_num_experts

        gate_up = source.gate_up_proj
        down = source.down_proj
        expected_gate_up = (
            self.num_experts,
            2 * self.intermediate_dim,
            self.hidden_dim,
        )
        expected_down = (
            self.num_experts,
            self.hidden_dim,
            self.intermediate_dim,
        )
        if tuple(gate_up.shape) != expected_gate_up:
            raise ValueError(
                "Unexpected Qwen2-MoE gate_up_proj shape: "
                f"{tuple(gate_up.shape)} != {expected_gate_up}"
            )
        if tuple(down.shape) != expected_down:
            raise ValueError(
                "Unexpected Qwen2-MoE down_proj shape: "
                f"{tuple(down.shape)} != {expected_down}"
            )

        self.gate_up_proj = _parameter_from_expert_shard(
            gate_up,
            expert_begin=expert_begin,
            local_num_experts=self.local_num_experts,
            device=device,
            mesh=mesh,
            run_check=run_check,
        )
        self.down_proj = _parameter_from_expert_shard(
            down,
            expert_begin=expert_begin,
            local_num_experts=self.local_num_experts,
            device=device,
            mesh=mesh,
            run_check=run_check,
        )
        self.train(source.training)
        self.last_dispatch_stats: dict[str, object] = {}

    def _compute_local_experts(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        expert_permutation = torch.argsort(local_expert_ids, stable=True)
        expert_inverse = _inverse_permutation(expert_permutation)
        expert_hidden = hidden_states.index_select(0, expert_permutation)
        expert_ids = local_expert_ids.index_select(0, expert_permutation)
        expert_counts = torch.bincount(
            expert_ids,
            minlength=self.local_num_experts,
        )

        # FSDP mixed precision does not apply to ignored parameters. Keep the
        # registered DTensor/master shard in FP32 and create differentiable
        # BF16 compute weights for this forward.
        gate_up = self.gate_up_proj.to_local().to(self.compute_dtype)
        down = self.down_proj.to_local().to(self.compute_dtype)
        outputs = []
        offset = 0
        for expert_idx, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue
            current = expert_hidden.narrow(0, offset, count)
            gate, up = F.linear(current, gate_up[expert_idx]).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = F.linear(current, down[expert_idx])
            outputs.append(current)
            offset += count
        if offset != hidden_states.shape[0]:
            raise RuntimeError(
                "Local expert counts do not cover all received assignments"
            )
        expert_output = torch.cat(outputs, dim=0)
        # Restore the A2A received/source-major order before reverse A2A.
        return expert_output.index_select(0, expert_inverse)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                "EP expert hidden_states must have shape [tokens, hidden]"
            )
        if hidden_states.shape[1] != self.hidden_dim:
            raise ValueError(
                f"hidden size {hidden_states.shape[1]} != {self.hidden_dim}"
            )
        if top_k_index.shape != top_k_weights.shape:
            raise ValueError("top_k_index and top_k_weights must have one shape")
        if top_k_index.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("routing tensors must have shape [tokens, top_k]")
        if hidden_states.dtype is not self.compute_dtype:
            raise TypeError(
                f"EP hidden states must be {self.compute_dtype}, got "
                f"{hidden_states.dtype}"
            )

        token_count, top_k = top_k_index.shape
        assignment_experts = top_k_index.reshape(-1).to(torch.int64)
        if assignment_experts.numel() > 0:
            min_expert = int(assignment_experts.min().item())
            max_expert = int(assignment_experts.max().item())
            if min_expert < 0 or max_expert >= self.num_experts:
                raise ValueError(
                    f"Expert IDs must be in [0, {self.num_experts}); got "
                    f"[{min_expert}, {max_expert}]"
                )
        assignment_tokens = (
            torch.arange(token_count, device=hidden_states.device)
            .unsqueeze(1)
            .expand(token_count, top_k)
            .reshape(-1)
        )
        owners = torch.div(
            assignment_experts,
            self.local_num_experts,
            rounding_mode="floor",
        )
        owner_permutation = torch.argsort(owners, stable=True)
        owner_inverse = _inverse_permutation(owner_permutation)
        sorted_owners = owners.index_select(0, owner_permutation)
        sorted_tokens = assignment_tokens.index_select(0, owner_permutation)
        sorted_local_experts = (
            assignment_experts.index_select(0, owner_permutation)
            .remainder(self.local_num_experts)
            .contiguous()
        )
        send_hidden = hidden_states.index_select(0, sorted_tokens).contiguous()
        input_counts = torch.bincount(
            sorted_owners,
            minlength=self.ep_size,
        ).to(device=hidden_states.device, dtype=torch.int64)
        output_counts = _exchange_split_sizes(input_counts, group=self.ep_group)
        input_splits = [int(value) for value in input_counts.tolist()]
        output_splits = [int(value) for value in output_counts.tolist()]

        received_hidden = _ragged_all_to_all_autograd(
            send_hidden,
            input_split_sizes=input_splits,
            output_split_sizes=output_splits,
            group=self.ep_group,
        )
        received_local_experts = _ragged_all_to_all_metadata(
            sorted_local_experts,
            input_split_sizes=input_splits,
            output_split_sizes=output_splits,
            group=self.ep_group,
        )
        received_output = self._compute_local_experts(
            received_hidden,
            received_local_experts,
        )
        returned_sorted_output = _ragged_all_to_all_autograd(
            received_output,
            input_split_sizes=output_splits,
            output_split_sizes=input_splits,
            group=self.ep_group,
        )
        assignment_output = returned_sorted_output.index_select(
            0,
            owner_inverse,
        )
        weighted_output = assignment_output * top_k_weights.reshape(
            -1, 1
        ).to(dtype=assignment_output.dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        final_hidden_states.index_add_(0, assignment_tokens, weighted_output)

        self.last_dispatch_stats = {
            "send_assignments": int(assignment_experts.numel()),
            "recv_assignments": int(received_local_experts.numel()),
            "input_splits": tuple(input_splits),
            "output_splits": tuple(output_splits),
            "local_expert_counts": tuple(
                int(value)
                for value in torch.bincount(
                    received_local_experts,
                    minlength=self.local_num_experts,
                ).tolist()
            ),
        }
        return final_hidden_states


def install_qwen2_moe_expert_parallel(
    model: nn.Module,
    *,
    mesh: DeviceMesh,
    group: dist.ProcessGroup,
    device: torch.device,
    compute_dtype: torch.dtype,
    run_check: bool = True,
) -> ExpertParallelSetup:
    """Replace fused Qwen2-MoE expert modules with EP-owned equivalents."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen2_moe":
        raise ValueError(
            "Training expert parallelism currently supports only "
            f"model_type='qwen2_moe'; got {model_type!r}"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Qwen2-MoE policy must expose model.layers")

    by_layer: Dict[nn.Module, frozenset[nn.Parameter]] = {}
    all_params = []
    expert_modules = []
    num_experts = None
    local_num_experts = None
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        source = getattr(mlp, "experts", None)
        if source is None:
            continue
        required = (
            "num_experts",
            "hidden_dim",
            "intermediate_dim",
            "gate_up_proj",
            "down_proj",
            "act_fn",
        )
        if not all(hasattr(source, name) for name in required):
            raise ValueError(
                "Unsupported Qwen2-MoE expert module; expected fused "
                "gate_up_proj/down_proj parameters"
            )
        replacement = Qwen2MoeEPExperts(
            source,
            mesh=mesh,
            group=group,
            device=device,
            compute_dtype=compute_dtype,
            run_check=run_check,
        )
        mlp.experts = replacement
        layer_params = frozenset(
            (replacement.gate_up_proj, replacement.down_proj)
        )
        by_layer[layer] = layer_params
        all_params.extend(layer_params)
        expert_modules.append(replacement)
        if num_experts is None:
            num_experts = replacement.num_experts
            local_num_experts = replacement.local_num_experts
        elif num_experts != replacement.num_experts:
            raise ValueError("All MoE layers must use the same expert count")
    if not by_layer or num_experts is None or local_num_experts is None:
        raise ValueError("The policy contains no fused Qwen2-MoE expert layers")
    return ExpertParallelSetup(
        expert_params_by_layer=by_layer,
        expert_params=tuple(all_params),
        expert_modules=tuple(expert_modules),
        ep_size=dist.get_world_size(group),
        local_num_experts=local_num_experts,
        num_experts=num_experts,
    )


def materialize_missing_expert_grads(
    expert_params: Iterable[nn.Parameter],
) -> int:
    """Make AdamW step every expert shard, including zero-token owners."""
    materialized = 0
    for parameter in expert_params:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
            materialized += 1
    return materialized


def validate_expert_parallel_setup(
    setup: ExpertParallelSetup,
    *,
    device: torch.device,
) -> None:
    """Validate the public DTensor invariants after all FSDP wrapping."""
    expected_local_experts = setup.num_experts // setup.ep_size
    if setup.local_num_experts != expected_local_experts:
        raise RuntimeError("Expert-parallel local expert count changed")
    for parameter in setup.expert_params:
        if not isinstance(parameter, DTensor):
            raise RuntimeError("Expert parameter is no longer a DTensor")
        if parameter.placements != (Shard(0),):
            raise RuntimeError(
                f"Expert parameter must retain Shard(0), got {parameter.placements}"
            )
        if parameter.dtype is not torch.float32:
            raise RuntimeError(
                f"Expert master parameter must be FP32, got {parameter.dtype}"
            )
        local = parameter.to_local()
        if local.device != device:
            raise RuntimeError(
                f"Expert local shard is on {local.device}, expected {device}"
            )
        if local.shape[0] != setup.local_num_experts:
            raise RuntimeError(
                "Expert local shard does not contain the configured number "
                f"of experts: {local.shape[0]} != {setup.local_num_experts}"
            )
