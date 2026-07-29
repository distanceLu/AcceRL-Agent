# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vLLM actor following AcceRL's interruptible inference workflow."""

import asyncio
import time
import uuid
from dataclasses import asdict
from typing import List

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.v1.executor import Executor

from vsi_qa_rlvr.inference import (
    InferenceRequestItem,
    InferenceResult,
    OnlineGenerationState,
    RepeatingInferenceStats,
)


ROLLOUT_ATTENTION_BACKENDS = ("TRITON_ATTN", "FLASH_ATTN")
INFER_LOG_EVERY_REQUESTS = 128


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


def _tokens_from_output(request_output) -> List[int]:
    if not getattr(request_output, "outputs", None):
        return []
    return list(getattr(request_output.outputs[0], "token_ids", []) or [])


def _logprobs_from_output(
    request_output,
    token_ids: List[int],
) -> List[float]:
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


def _normalize_stop_reason(stop_reason):
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
        temperature=0.7,
        top_p=0.9,
        collect_logprobs=False,
        max_resubmit_retries=200,
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

    def pause(self):
        self.resume_event.clear()

    def resume(self):
        self.resume_event.set()

    @property
    def is_resumed(self):
        return self.resume_event.is_set()

    async def _increment_active_attempts(self):
        async with self._active_changed:
            self._active_attempts += 1
            self._active_changed.notify_all()

    async def _decrement_active_attempts(self):
        async with self._active_changed:
            self._active_attempts -= 1
            self._active_changed.notify_all()

    async def wait_for_idle(self):
        async with self._active_changed:
            await self._active_changed.wait_for(
                lambda: self._active_attempts == 0
            )

    async def generate(self, state):
        for attempt in range(1, self.max_resubmit_retries + 1):
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
                async for request_output in self.engine.generate(
                    state.restart_engine_input,
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


class VSIQAVLLMInferenceActor:
    """GPU Ray actor that owns Qwen3-VL vLLM inference."""

    def __init__(self, args):
        self.args = args
        self.engine = create_async_engine(
            model=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            tensor_parallel_size=args.infer_tp_size,
            data_parallel_size=args.infer_dp_size,
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            load_format="dummy",
            gpu_memory_utilization=args.rollout_gpu_memory_utilization,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
            max_model_len=args.max_model_len,
            attention_backend=args.rollout_attention_backend,
            mm_encoder_attn_backend="TORCH_SDPA",
            logprobs_mode="raw_logprobs",
            allowed_local_media_path="/",
            limit_mm_per_prompt={"video": 1},
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
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
            f"tp={args.infer_tp_size} dp={args.infer_dp_size} "
            "continuous_submit=True "
            f"vllm_max_num_seqs={args.vllm_max_num_seqs} "
            f"vllm_max_num_batched_tokens="
            f"{args.vllm_max_num_batched_tokens} "
            f"attention_backend={args.rollout_attention_backend} "
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
        rollout_worker_id,
        batch_id,
        llm_input,
        infer_max_tokens,
        num_samples,
    ) -> List[InferenceResult]:
        if self.stopped:
            raise RuntimeError("VSIQAVLLMInferenceActor is stopped.")
        if num_samples < 1:
            return []

        requests_to_process = []
        for sample_id in range(num_samples):
            request_index = self.next_request_index
            self.next_request_index += 1
            item = InferenceRequestItem(
                request_index=request_index,
                rollout_worker_id=int(rollout_worker_id),
                batch_id=int(batch_id),
                sample_id=int(sample_id),
                llm_input=llm_input,
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
                    llm_input=item.llm_input,
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
                await asyncio.gather(
                    *generation_tasks,
                    return_exceptions=True,
                )
                raise

            results = []
            first_exception = None
            for item, completed_state in zip(
                requests_to_process,
                completed_states,
            ):
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

    async def _record_completed_state(self, result):
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

    async def resume_generation(self, increment_version=False):
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
        master_address,
        master_port,
        transfer_world_size,
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
        names,
        dtype_names,
        shapes,
        packed=True,
    ):
        started = time.perf_counter()
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
        return {"elapsed_seconds": time.perf_counter() - started}

    async def finish_weight_update(self):
        started = time.perf_counter()
        await self.engine.finish_weight_update()
        return {"elapsed_seconds": time.perf_counter() - started}

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
            await asyncio.gather(
                *self.active_generation_tasks,
                return_exceptions=True,
            )
        await shutdown_vllm_engine(self.engine)
        return self.get_stats()


async def shutdown_vllm_engine(engine):
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
    except Exception as error:
        print(f"[cleanup] Ignoring vLLM engine shutdown error: {error!r}")
