# SPDX-License-Identifier: Apache-2.0
"""Pass VSI-QA multimodal inputs through AcceRL's vLLM actor protocol."""

import asyncio
import time
import uuid

from vllm import SamplingParams
from vllm.config import WeightTransferConfig

from accerl_agent.vllm_fsdp import (
    InferenceResult,
    RepeatingInferenceStats,
    VLLMInferenceActor,
    _finish_reason_from_output,
    _logprobs_from_output,
    _normalize_stop_reason,
    _tokens_from_output,
    create_async_engine,
    shutdown_vllm_engine,
)


ROLLOUT_ATTENTION_BACKENDS = ("TRITON_ATTN", "FLASH_ATTN")


def _disable_model_gradients(model):
    model.requires_grad_(False)


def _build_attempt_engine_input(llm_input, output_tokens):
    """Restart one generation attempt from an already rendered EngineInput."""
    if llm_input.get("type") != "multimodal":
        raise ValueError(
            "VSI-QA Infer Actor requires a multimodal vLLM EngineInput."
        )
    request_input = dict(llm_input)
    request_input["prompt_token_ids"] = (
        list(llm_input["prompt_token_ids"])
        + list(output_tokens)
    )
    request_input["arrival_time"] = time.time()
    return request_input


class VSIQAVLLMInferenceActor(VLLMInferenceActor):
    """Use AcceRL generation and weight sync with Qwen3-VL inputs."""

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
        self.resume_event = asyncio.Event()
        self.resume_event.set()
        self.active_changed = asyncio.Condition()
        self.active_attempts = 0
        self.active_generation_tasks = set()
        self.stats = RepeatingInferenceStats()
        self.next_request_index = 0
        self.policy_version = 0
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

    async def start_weight_update(self):
        if self.args.infer_tp_size != 1:
            raise RuntimeError(
                "Qwen3-VL kernel-format weight sync currently requires "
                f"infer TP=1, got TP={self.args.infer_tp_size}."
            )
        if not getattr(self, "_kernel_weight_update_ready", False):
            await self.engine.collective_rpc(
                "apply_model",
                args=(_disable_model_gradients,),
            )
            self._kernel_weight_update_ready = True
        await self.engine.start_weight_update(is_checkpoint_format=False)

    async def _change_active_attempts(self, delta):
        async with self.active_changed:
            self.active_attempts += delta
            self.active_changed.notify_all()

    async def _wait_for_idle(self):
        async with self.active_changed:
            await self.active_changed.wait_for(
                lambda: self.active_attempts == 0
            )

    async def _generate_candidate(
        self,
        llm_input,
        rollout_worker_id,
        batch_id,
        sample_id,
        max_tokens,
    ):
        request_index = self.next_request_index
        self.next_request_index += 1
        output_tokens = []
        output_logprobs = []
        output_versions = []
        stop_reason = "abort"

        for attempt in range(1, self.args.max_resubmit_retries + 1):
            await self.resume_event.wait()
            remaining = int(max_tokens) - len(output_tokens)
            if remaining <= 0:
                stop_reason = "length"
                break
            attempt_version = self.policy_version
            sampling_params = SamplingParams(
                temperature=self.args.infer_temperature,
                top_p=self.args.infer_top_p,
                max_tokens=remaining,
                stop=["</answer>"],
                logprobs=1,
            )
            request_input = _build_attempt_engine_input(
                llm_input,
                output_tokens,
            )
            request_id = (
                f"vsi-{request_index}-v{attempt_version}-"
                f"try{attempt}-{uuid.uuid4()}"
            )
            final_output = None
            await self._change_active_attempts(1)
            try:
                async for request_output in self.engine.generate(
                    request_input,
                    sampling_params,
                    request_id=request_id,
                ):
                    final_output = request_output
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.resume_event.is_set():
                    raise
            finally:
                await self._change_active_attempts(-1)

            if final_output is None:
                continue
            attempt_tokens = _tokens_from_output(final_output)[:remaining]
            attempt_logprobs = _logprobs_from_output(
                final_output,
                attempt_tokens,
            )[: len(attempt_tokens)]
            if len(attempt_logprobs) != len(attempt_tokens):
                continue
            output_tokens.extend(attempt_tokens)
            output_logprobs.extend(attempt_logprobs)
            output_versions.extend([attempt_version] * len(attempt_tokens))
            stop_reason = _normalize_stop_reason(
                _finish_reason_from_output(final_output)
            )
            if len(output_tokens) >= int(max_tokens):
                stop_reason = "length"
            if stop_reason in ("stop", "tool_calls", "length"):
                break

        result = InferenceResult(
            request_index=request_index,
            rollout_worker_id=int(rollout_worker_id),
            batch_id=int(batch_id),
            sample_id=int(sample_id),
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
            stop_reason=stop_reason,
            attempts=attempt,
        )
        self.stats.total_requests += 1
        self.stats.total_tokens += len(output_tokens)
        return result

    async def generate_group(
        self,
        rollout_worker_id,
        batch_id,
        llm_input,
        num_samples,
        max_tokens,
    ):
        tasks = [
            asyncio.create_task(
                self._generate_candidate(
                    llm_input,
                    rollout_worker_id,
                    batch_id,
                    sample_id,
                    max_tokens,
                )
            )
            for sample_id in range(int(num_samples))
        ]
        self.active_generation_tasks.update(tasks)
        for task in tasks:
            task.add_done_callback(self.active_generation_tasks.discard)
        return await asyncio.gather(*tasks)

    async def pause_and_wait_idle(self):
        self.resume_event.clear()
        await self.engine.pause_generation(mode="abort", clear_cache=True)
        await self._wait_for_idle()
        print("[infer-actor] Generation paused and in-flight attempts drained.")

    async def resume_generation(self, increment_version=False):
        if increment_version:
            self.policy_version += 1
        await self.engine.resume_generation()
        self.resume_event.set()
        print(
            "[infer-actor] Generation resumed: "
            f"weight_version={self.policy_version}"
        )
        return self.policy_version

    def get_stats(self):
        return {
            "total_requests": self.stats.total_requests,
            "total_tokens": self.stats.total_tokens,
            "active_attempts": self.active_attempts,
            "active_generation_tasks": len(self.active_generation_tasks),
            "weight_version": self.policy_version,
        }

    async def shutdown(self):
        self.resume_event.set()
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
