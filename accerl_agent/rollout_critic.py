"""Completed-trajectory value inference on the synchronized vLLM backbone.

The runner adapter targets the vLLM version pinned in requirement.txt. It
captures prefill hidden states, including chunked prefills, without loading
another backbone. Unique cache salts force every scoring request to evaluate
its entire transcript. Weight updates drain these requests before refitting.
"""
from __future__ import annotations

import asyncio
import functools
import uuid

import torch

from accerl_agent.ppo_value import TokenValueHead


VALUE_REQUEST_PREFIX = "accerl_value_"


class RolloutCriticWorker:
    """Installed inside each vLLM GPU worker through its extension class."""

    def __init__(self, runner):
        # vLLM 0.24 can select the V2 runner for MoE models. V1 exposes the
        # input batch directly and calls ``_model_forward``; V2 stores the
        # completed forward in ExecuteModelState until sample_tokens().
        is_v1 = hasattr(runner, "_model_forward") and hasattr(runner, "input_batch")
        is_v2 = hasattr(runner, "execute_model_state") and hasattr(runner, "prepare_inputs")
        if not (is_v1 or is_v2):
            raise RuntimeError(
                "Rollout critic requires a supported vLLM V1 or V2 model runner."
            )
        config = runner.vllm_config
        if (
            config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.use_ubatching
            or config.speculative_config is not None
            or not config.model_config.enforce_eager
        ):
            raise ValueError("Rollout critic requires eager execution without PP, DBO or speculation.")
        self.runner = runner
        self.head = None
        self.version = None
        self.records = {}
        self._runner_api = "v1" if is_v1 else "v2"
        if is_v1:
            original = runner._model_forward

            @functools.wraps(original)
            def forward_with_values(*args, **kwargs):
                hidden = original(*args, **kwargs)
                self.capture_v1(hidden)
                return hidden

            runner._model_forward = forward_with_values
        else:
            original = runner.execute_model

            @functools.wraps(original)
            def execute_with_values(*args, **kwargs):
                result = original(*args, **kwargs)
                self.capture_v2()
                return result

            runner.execute_model = execute_with_values

    def load(self, state, version):
        if self.records:
            raise RuntimeError("Cannot replace critic weights with unconsumed value requests.")
        if type(version) is not int or version < 0:
            raise ValueError("Critic snapshot version must be a non-negative optimizer step.")
        tensors = {
            name: torch.tensor(value, dtype=torch.float32, device=self.runner.device)
            for name, value in state.items()
        }
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Critic snapshot contains non-finite weights.")
        if self.head is None:
            hidden_size = self.runner.model_config.get_hidden_size()
            self.head = TokenValueHead(hidden_size).to(self.runner.device).eval()
            self.head.requires_grad_(False)
        self.head.load_state_dict(tensors, strict=True)
        self.version = version
        return version

    @torch.no_grad()
    def capture_v1(self, hidden):
        batch = self.runner.input_batch
        for index, request_id in enumerate(batch.req_ids):
            if not request_id.startswith(VALUE_REQUEST_PREFIX):
                continue
            if self.head is None or not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
                raise RuntimeError("Rollout critic did not receive final token hidden states.")
            # vLLM appends a random suffix to external IDs; ':' is our delimiter.
            key = request_id.split(":", 1)[0]
            first = int(batch.num_computed_tokens_cpu[index])
            prompt_length = int(batch.num_prompt_tokens[index])
            offset = int(self.runner.query_start_loc.np[index])
            end = int(self.runner.query_start_loc.np[index + 1])
            count = min(end - offset, prompt_length - first)
            if count <= 0:
                continue
            record = self.records.setdefault(
                key, {"version": self.version, "values": [None] * prompt_length}
            )
            if record["version"] != self.version or len(record["values"]) != prompt_length:
                raise RuntimeError("A value request crossed critic snapshots or changed length.")
            values = self.head(hidden[offset:offset + count]).float().cpu().tolist()
            record["values"][first:first + count] = values

    @torch.no_grad()
    def capture_v2(self):
        state = self.runner.execute_model_state
        if state is None or state.hidden_states is None:
            return
        input_batch = state.input_batch
        hidden = state.hidden_states
        query_start = input_batch.query_start_loc_np
        for index, request_id in enumerate(input_batch.req_ids):
            if not request_id.startswith(VALUE_REQUEST_PREFIX):
                continue
            key = request_id.split(":", 1)[0]
            first = int(input_batch.num_computed_tokens_np[index])
            prompt_length = int(input_batch.prefill_len_np[index])
            offset = int(query_start[index])
            end = int(query_start[index + 1])
            count = min(end - offset, prompt_length - first)
            if count <= 0:
                continue
            record = self.records.setdefault(
                key, {"version": self.version, "values": [None] * prompt_length}
            )
            if record["version"] != self.version or len(record["values"]) != prompt_length:
                raise RuntimeError("A value request crossed critic snapshots or changed length.")
            values = self.head(hidden[offset:offset + count]).float().cpu().tolist()
            record["values"][first:first + count] = values

    # Test and compatibility helper for the V1 path.
    capture = capture_v1

    def take(self, key):
        return self.records.pop(key, None)


class RolloutCriticWorkerExtension:
    """Named RPCs keep the engine's default safe serialization enabled."""

    def load_rollout_critic(self, state, version):
        if not hasattr(self, "_accerl_rollout_critic"):
            self._accerl_rollout_critic = RolloutCriticWorker(self.model_runner)
        return self._accerl_rollout_critic.load(state, version)

    def take_rollout_values(self, key):
        return self._accerl_rollout_critic.take(key)


async def all_worker_rpc(engine, method, **kwargs):
    # vLLM's DP frontend broadcasts collective_rpc but returns only engine 0's
    # result. Read every engine explicitly so trajectories routed to other DP
    # replicas are not lost. This adapter uses the pinned vLLM client API.
    client = engine.engine_core
    engines = getattr(client, "core_engines", ())
    if len(engines) > 1:
        groups = await asyncio.gather(*[
            client._call_utility_async(
                "collective_rpc", method, None, (), kwargs, engine=core
            )
            for core in engines
        ])
        return [item for group in groups for item in group]
    return await engine.collective_rpc(method, kwargs=kwargs)


class RolloutCriticClient:
    """Admission gate shared by scoring and the actor's weight-update path."""

    def __init__(self, engine):
        self.engine = engine
        self.version = None
        self.admission = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()
        self.active = 0
        self.closed = False

    async def pause(self):
        self.admission.clear()
        # Drain while the engine is still running. Aborting prefills here
        # would leave partial value vectors or deadlock the update barrier.
        await self.idle.wait()

    async def load(self, state, version):
        if self.admission.is_set() or self.active:
            raise RuntimeError("Pause and drain rollout values before updating critic weights.")
        versions = await all_worker_rpc(
            self.engine, "load_rollout_critic", state=state, version=version
        )
        if not versions or any(item != version for item in versions):
            raise RuntimeError("vLLM workers did not acknowledge the same critic snapshot.")
        self.version = version

    def resume(self):
        if self.version is None:
            raise RuntimeError("Cannot score trajectories before the first critic weight sync.")
        self.admission.set()

    def close(self):
        self.closed = True
        self.admission.set()  # Wake callers waiting for a now impossible sync.

    async def score(self, transcripts):
        while not self.admission.is_set():
            await self.admission.wait()
        if self.closed:
            raise RuntimeError("Rollout critic is closed.")
        # No await between checking admission and incrementing active: pause
        # either sees this batch or prevents it from entering.
        version = self.version
        self.active += 1
        self.idle.clear()
        try:
            results = await asyncio.gather(
                *(self._score_one(tokens, version) for tokens in transcripts),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return {"value_version": version, "token_values": results}
        finally:
            self.active -= 1
            if self.active == 0:
                self.idle.set()

    async def _score_one(self, tokens, version):
        from vllm import SamplingParams

        key = VALUE_REQUEST_PREFIX + uuid.uuid4().hex
        request_id = key + ":"
        complete = False
        consumed = False
        try:
            final = None
            async for output in self.engine.generate(
                {"prompt_token_ids": list(tokens), "cache_salt": key},
                SamplingParams(
                    max_tokens=1, temperature=0.0, detokenize=False,
                    ignore_eos=True,
                ),
                request_id=request_id,
            ):
                final = output
            complete = True
            if final is None or not final.outputs or final.outputs[0].finish_reason not in {"length", "stop"}:
                raise RuntimeError("Rollout value prefill did not finish normally.")
            records = await all_worker_rpc(self.engine, "take_rollout_values", key=key)
            consumed = True
            records = [record for record in records if record is not None]
            if not records:
                raise RuntimeError("No vLLM worker returned rollout values.")
            record = records[0]
            values = record["values"]
            if (
                any(item["version"] != version for item in records)
                or len(values) != len(tokens)
                or any(value is None for value in values)
            ):
                raise RuntimeError("Incomplete rollout values or mismatched critic snapshot.")
            # TP replicas see the same final hidden states and must agree.
            for other in records[1:]:
                if other["values"] != values:
                    raise RuntimeError("vLLM tensor-parallel ranks disagree on rollout values.")
            if not torch.isfinite(torch.tensor(values)).all():
                raise RuntimeError("Rollout critic returned non-finite values.")
            return values
        finally:
            if not complete:
                await self.engine.abort(request_id)
            if not consumed:
                await all_worker_rpc(self.engine, "take_rollout_values", key=key)
