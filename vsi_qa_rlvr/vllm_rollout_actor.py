# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vLLM actor following AcceRL's interruptible inference workflow."""

import asyncio
import argparse
import uuid
from dataclasses import asdict
from typing import List, Literal

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


INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 2
INFER_LOG_EVERY_REQUESTS = 128
"""vsiqa"""
ROLLOUT_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")
"""vsiqa"""


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


def _normalize_stop_reason(
    stop_reason,
) -> Literal["length", "stop", "tool_calls", "abort"]:
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
        temperature: float = 0.7,
        top_p: float = 0.9,
        collect_logprobs: bool = False,
        max_resubmit_retries: int = 200,
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

    def pause(self) -> None:
        self.resume_event.clear()

    def resume(self) -> None:
        self.resume_event.set()

    @property
    def is_resumed(self) -> bool:
        return self.resume_event.is_set()

    # 新的engine.generate() attempt开始 +1
    async def _increment_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts += 1
            self._active_changed.notify_all()

    # 一个engine.generate() attempt 结束 -1
    async def _decrement_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts -= 1
            self._active_changed.notify_all()

    # 等待正在跑的 generate attempt 都结束,可能是正常结束，也可能是被abort打断
    async def wait_for_idle(self) -> None:
        async with self._active_changed:
            await self._active_changed.wait_for(lambda: self._active_attempts == 0)

    """vsiqa"""
    async def generate(
        self,
        state: OnlineGenerationState,
    ) -> OnlineGenerationState:
        for attempt in range(1, self.max_resubmit_retries + 1):
            # 如果当前正在weight update的attempt还没结束，就等着，不要开始新的generate attempt
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
                engine_input = state.restart_prompt_token_ids
                # 调用vllm生成接口，拿到输出后更新state，如果生成过程中被weight update打断了，engine.generate()会抛出异常，直接进入finally块结束这个attempt
                async for request_output in self.engine.generate(
                    engine_input,
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
    """vsiqa"""


class VLLMInferenceActor:
    """GPU Ray actor that owns vLLM and consumes tokenized rollout requests."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        """vsiqa"""
        self.engine = create_async_engine(
            model=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=False,
            tensor_parallel_size=INFERENCE_TP_SIZE,
            data_parallel_size=INFERENCE_DP_SIZE,
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            load_format="dummy",
            gpu_memory_utilization=args.rollout_gpu_memory_utilization,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
            max_model_len=args.max_model_len,
            attention_backend=args.rollout_attention_backend,
            mm_encoder_attn_backend=args.rollout_attention_backend,
            logprobs_mode="raw_logprobs",
            allowed_local_media_path="/",
            limit_mm_per_prompt={"image": args.limit_images},
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )
        """vsiqa"""
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
        """vsiqa"""
        print(
            "[infer-actor] AsyncLLMEngine ready: "
            f"tp={INFERENCE_TP_SIZE} dp={INFERENCE_DP_SIZE} "
            "continuous_submit=True "
            f"vllm_max_num_seqs={args.vllm_max_num_seqs} "
            f"vllm_max_num_batched_tokens="
            f"{args.vllm_max_num_batched_tokens} "
            f"attention_backend={args.rollout_attention_backend} "
            f"infer_temperature={args.infer_temperature} "
            f"infer_top_p={args.infer_top_p}"
        )
        """vsiqa"""

    async def start(self):
        self.stopped = False
        print(
            "[infer-actor] Continuous submit mode enabled; "
            "vLLM handles batching internally."
        )
        return {"continuous_submit": True, "inference_loop_task": 0}

    """vsiqa"""
    async def request_batch(
        self,
        rollout_worker_id: int,
        batch_id: int,
        input_ids: dict,  # Qwen3-VL receives the multimodal vLLM prompt dictionary.
        infer_max_tokens: int,
        num_samples: int,
    ) -> List[InferenceResult]:
        if self.stopped:
            raise RuntimeError("VLLMInferenceActor is stopped.")
        if num_samples < 1:
            return []

        engine_inputs = await self.engine.renderer.render_cmpl_async(
            [input_ids]
        )
        llm_input = engine_inputs[0]

        requests_to_process = []
        for sample_id in range(num_samples):
            request_index = self.next_request_index
            self.next_request_index += 1
            item = InferenceRequestItem(
                request_index=request_index,
                rollout_worker_id=int(rollout_worker_id),
                batch_id=int(batch_id),
                sample_id=int(sample_id),
                input_ids=llm_input,
                requested_max_tokens=int(infer_max_tokens),
            )
            requests_to_process.append(item)

        return await self._run_generation_items(requests_to_process)
    """vsiqa"""

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
                """vsiqa"""
                state = OnlineGenerationState(
                    index=item.request_index,
                    input_ids=item.input_ids,
                    requested_max_tokens=item.requested_max_tokens,
                )
                """vsiqa"""
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
                await asyncio.gather(*generation_tasks, return_exceptions=True)
                raise

            results = []
            first_exception = None
            for item, completed_state in zip(requests_to_process, completed_states):
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

    async def _record_completed_state(
        self,
        result: InferenceResult,
    ) -> None:
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

    async def resume_generation(self, increment_version: bool = False):
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
        master_address: str,
        master_port: int,
        transfer_world_size: int,
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
        names: List[str],
        dtype_names: List[str],
        shapes: List[List[int]],
        packed: bool = True,
    ):
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

    async def finish_weight_update(self):
        await self.engine.finish_weight_update()

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
            await asyncio.gather(*self.active_generation_tasks, return_exceptions=True)
        await shutdown_vllm_engine(self.engine)
        return self.get_stats()


async def shutdown_vllm_engine(engine) -> None:
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
    except Exception as exc:
        print(f"[cleanup] Ignoring vLLM engine shutdown error: {exc!r}")
