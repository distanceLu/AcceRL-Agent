# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL rollout, Replay, and asynchronous vLLM actors for VSI-QA."""

import asyncio
import random
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal

import ray
import torch
import vllm
from torch.utils.data import DataLoader, RandomSampler
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

from vsi_qa_rlvr.dataloaders.qwen3vl_rollout_collator import (
    Qwen3VLRolloutDataCollator,
)
from vsi_qa_rlvr.dataloaders.vsi_qa_dataset import VSIQADataset
from vsi_qa_rlvr.exp02_v4_reward import score_vsi_qa_group


ROLLOUT_ATTENTION_BACKENDS = ("TRITON_ATTN", "FLASH_ATTN")
INFER_LOG_EVERY_REQUESTS = 128


@dataclass
class OnlineGenerationState:
    """Token-level state for one request across weight-update interruptions."""

    index: int
    llm_input: dict
    requested_max_tokens: int
    output_tokens: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
    attempts: int = 0

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.output_tokens))

    @property
    def restart_engine_input(self) -> dict:
        engine_input = dict(self.llm_input)
        engine_input["prompt_token_ids"] = (
            list(self.llm_input["prompt_token_ids"])
            + list(self.output_tokens)
        )
        engine_input["arrival_time"] = time.time()
        return engine_input


@dataclass
class RepeatingInferenceStats:
    total_requests: int = 0
    total_tokens: int = 0


@dataclass
class InferenceRequestItem:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    llm_input: dict
    requested_max_tokens: int


@dataclass
class InferenceResult:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    output_tokens: List[int]
    output_logprobs: List[float]
    output_versions: List[int]
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None
    attempts: int

    @property
    def version_range(self) -> str:
        if not self.output_versions:
            return "none"
        return f"{min(self.output_versions)}-{max(self.output_versions)}"


@dataclass
class RLSample:
    prompt_ids: List[int]
    response_ids: List[int]
    old_response_logprobs: List[float]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    reward: float
    advantage: float
    question: str
    ground_truth: str
    format_reward: float
    answer_reward: float
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    output_versions: List[int]
    stop_reason: str | None
    generated_text: str


@dataclass
class VSIQARLSample(RLSample):
    prepared_media: dict


@ray.remote
class StatsActor:
    """Aggregates rollout metrics from async rollout workers."""

    def __init__(self, window_size: int, active_timeout_seconds: float):
        self.reward_sums = deque(maxlen=window_size)
        self.response_lengths = deque(maxlen=window_size)
        self.abort_flags = deque(maxlen=window_size)
        self.worker_last_active = {}
        self.active_timeout_seconds = active_timeout_seconds
        self.total_episodes = 0

    def add_rollout_batch(
        self,
        worker_id: int,
        rewards: List[float],
        response_lengths: List[int],
        abort_flags: List[bool],
    ) -> None:
        if not (len(rewards) == len(response_lengths) == len(abort_flags)):
            raise ValueError(
                "Rollout metric batch lengths must match: "
                f"rewards={len(rewards)} response_lengths={len(response_lengths)} "
                f"abort_flags={len(abort_flags)}"
            )
        self.reward_sums.extend(float(value) for value in rewards)
        self.response_lengths.extend(int(value) for value in response_lengths)
        self.abort_flags.extend(bool(value) for value in abort_flags)
        self.total_episodes += len(rewards)
        self.worker_last_active[int(worker_id)] = time.time()

    def get_stats(self) -> Dict[str, float]:
        now = time.time()
        active_cutoff = now - self.active_timeout_seconds
        active_workers = sum(
            1
            for last_active in self.worker_last_active.values()
            if last_active >= active_cutoff
        )
        reward_count = len(self.reward_sums)
        response_count = len(self.response_lengths)
        abort_count = len(self.abort_flags)
        return {
            "global_reward_sum_mean": (
                sum(self.reward_sums) / reward_count if reward_count else 0.0
            ),
            "response_length_mean": (
                sum(self.response_lengths) / response_count
                if response_count
                else 0.0
            ),
            "abort_rate": (
                sum(1 for flag in self.abort_flags if flag) / abort_count
                if abort_count
                else 0.0
            ),
            "active_workers": active_workers,
            "total_episodes": self.total_episodes,
        }


@ray.remote
class ReplayBufferActor:
    """Replay buffer that stores rollout-produced RL samples."""

    def __init__(self, capacity: int):
        self.samples = deque(maxlen=capacity)
        self.total_samples_added = 0
        self.total_samples_sampled = 0
        self.total_samples_evicted = 0
        self.total_batches_added = 0
        self.total_batches_sampled = 0

    def add_samples(self, samples: List[RLSample]) -> Dict[str, int]:
        capacity = self.samples.maxlen or 0
        if capacity > 0:
            self.total_samples_evicted += max(
                0,
                len(self.samples) + len(samples) - capacity,
            )
        self.samples.extend(samples)
        self.total_samples_added += len(samples)
        self.total_batches_added += 1
        return self.get_stats()

    def sample(self, batch_size: int) -> List[RLSample]:
        if batch_size < 1:
            return []
        sample_count = min(batch_size, len(self.samples))
        if sample_count == 0:
            return []
        samples = random.sample(list(self.samples), sample_count)
        self.total_samples_sampled += len(samples)
        self.total_batches_sampled += 1
        return samples

    def get_stats(self) -> Dict[str, int]:
        return {
            "size": len(self.samples),
            "capacity": self.samples.maxlen or 0,
            "total_samples_added": self.total_samples_added,
            "total_samples_sampled": self.total_samples_sampled,
            "total_samples_evicted": self.total_samples_evicted,
            "total_batches_added": self.total_batches_added,
            "total_batches_sampled": self.total_batches_sampled,
        }


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


class VSIQARolloutWorkerActor:
    """CPU Ray actor that feeds VSI-QA rollout requests to vLLM."""

    def __init__(
        self,
        args,
        infer_actor,
        replay_buffer,
        stats_actor,
        worker_id,
    ):
        self.args = args
        self.infer_actor = infer_actor
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor
        self.worker_id = int(worker_id)
        self.dataset = VSIQADataset(args.data_path)
        self.collator = Qwen3VLRolloutDataCollator(
            args.model_path,
            max_model_len=args.max_model_len,
        )
        self.sampler = RandomSampler(
            self.dataset,
            replacement=True,
            num_samples=len(self.dataset),
        )
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=args.rollout_data_batch_size,
            sampler=self.sampler,
            collate_fn=self.collator,
            num_workers=args.rollout_data_workers,
            prefetch_factor=args.rollout_prefetch_factor,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        self.data_iterator = None
        self.batch_id = 0
        self.stopped = False
        print(
            "[rollout] "
            f"worker={self.worker_id} loaded VSI-QA examples: "
            f"count={len(self.dataset)} path={args.data_path!r} "
            f"data_workers={args.rollout_data_workers} "
            f"prefetch_factor={args.rollout_prefetch_factor}"
        )

    def next_rollout_batch(self):
        try:
            if self.data_iterator is None:
                self.data_iterator = iter(self.data_loader)
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            return next(self.data_iterator)

    async def stop(self):
        self.stopped = True

    def compute_group_advantages(self, rewards: List[float]) -> List[float]:
        if not rewards:
            return []

        valid_count = sum(1 for reward in rewards if reward > 0.0)
        if valid_count <= 1:
            return [0.0 for _ in rewards]

        reward_range = max(rewards) - min(rewards)
        if reward_range < 1e-6:
            return [0.0 for _ in rewards]

        rewards_t = torch.tensor(rewards, dtype=torch.float64)
        mean = rewards_t.mean()
        std = rewards_t.std(unbiased=False)
        if std.item() < 1e-6:
            return [0.0 for _ in rewards]

        advantages = (rewards_t - mean) / (std + 1e-6)
        return [float(value) for value in advantages.tolist()]

    def build_rl_samples(self, rollout_item, results):
        generated_texts = [
            self.collator.processor.tokenizer.decode(
                result.output_tokens,
                skip_special_tokens=True,
            )
            for result in results
        ]
        valid_indices = [
            index
            for index, result in enumerate(results)
            if result.stop_reason != "abort" and result.output_tokens
        ]
        reward_details = [None] * len(results)
        if valid_indices:
            valid_reward_details = score_vsi_qa_group(
                [generated_texts[index] for index in valid_indices],
                rollout_item["reward_model"]["ground_truth"],
                rollout_item["extra_info"],
                history_path=self.args.reward_history_path,
            )
            for index, details in zip(valid_indices, valid_reward_details):
                reward_details[index] = details
        for index, details in enumerate(reward_details):
            if details is None:
                reward_details[index] = {
                    "score": 0.0,
                    "r_format": 0.0,
                    "answer_exact_reward": 0.0,
                }

        rewards = [float(details["score"]) for details in reward_details]
        advantages = self.compute_group_advantages(rewards)
        prompt_ids = rollout_item["prepared_media"][
            "prompt_token_ids"
        ].tolist()
        return [
            VSIQARLSample(
                prompt_ids=list(prompt_ids),
                response_ids=list(result.output_tokens),
                old_response_logprobs=list(result.output_logprobs),
                input_ids=list(prompt_ids) + list(result.output_tokens),
                attention_mask=[1]
                * (len(prompt_ids) + len(result.output_tokens)),
                labels=[-100] * len(prompt_ids) + list(result.output_tokens),
                reward=float(reward),
                advantage=float(advantage),
                question=rollout_item["question"],
                ground_truth=rollout_item["reward_model"]["ground_truth"],
                format_reward=float(details["r_format"]),
                answer_reward=float(details["answer_exact_reward"]),
                rollout_worker_id=self.worker_id,
                batch_id=result.batch_id,
                sample_id=result.sample_id,
                output_versions=list(result.output_versions),
                stop_reason=result.stop_reason,
                generated_text=generated_text,
                prepared_media=rollout_item["prepared_media"],
            )
            for result, generated_text, details, reward, advantage in zip(
                results,
                generated_texts,
                reward_details,
                rewards,
                advantages,
            )
        ]

    async def run(self):
        while not self.stopped:
            rollout_batch = self.next_rollout_batch()
            for rollout_item in rollout_batch:
                current_batch_id = self.batch_id
                self.batch_id += 1
                results = await self.infer_actor.request_batch.remote(
                    self.worker_id,
                    current_batch_id,
                    rollout_item["llm_input"],
                    self.args.infer_max_tokens,
                    self.args.rollout_batch_size,
                )
                if not results:
                    continue
                rl_samples = self.build_rl_samples(rollout_item, results)
                self.replay_buffer.add_samples.remote(rl_samples)
                self.stats_actor.add_rollout_batch.remote(
                    self.worker_id,
                    [sample.reward for sample in rl_samples],
                    [len(sample.response_ids) for sample in rl_samples],
                    [sample.stop_reason == "abort" for sample in rl_samples],
                )

                rewards = [sample.reward for sample in rl_samples]
                advantages = [sample.advantage for sample in rl_samples]
                response_lengths = [
                    len(sample.response_ids) for sample in rl_samples
                ]
                reward_t = torch.tensor(rewards, dtype=torch.float32)
                advantage_t = torch.tensor(advantages, dtype=torch.float32)
                response_length_t = torch.tensor(
                    response_lengths,
                    dtype=torch.float32,
                )
                version_ranges = sorted(
                    {result.version_range for result in results}
                )
                stop_reasons = sorted(
                    {str(sample.stop_reason) for sample in rl_samples}
                )
                print(
                    "[rollout] "
                    f"worker={self.worker_id} "
                    f"batch={current_batch_id} "
                    f"samples={len(rl_samples)} "
                    f"reward_mean={reward_t.mean().item():.4f} "
                    f"reward_std={reward_t.std(unbiased=False).item():.4f} "
                    f"adv_mean={advantage_t.mean().item():.4f} "
                    f"adv_std={advantage_t.std(unbiased=False).item():.4f} "
                    f"response_len_mean="
                    f"{response_length_t.mean().item():.1f} "
                    f"versions={','.join(version_ranges)} "
                    f"stops={','.join(stop_reasons)}"
                )

        print(f"[rollout] worker={self.worker_id} stopped.")
        return {"worker_id": self.worker_id, "batches": self.batch_id}
