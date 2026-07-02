# SPDX-License-Identifier: Apache-2.0
"""Run local vLLM action inference in TextWorld environments.

This script is intentionally shaped like ``agent_textworld.py``'s InferActor
path: tokenized prompt -> OnlineGenerationState -> InferenceResult.  It does
not train, build replay samples, use Ray, or reload weights.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import glob
import inspect
import os
import random
import sys
import uuid
from dataclasses import dataclass, field
from typing import Literal

import textworld
import textworld.gym
import vllm
from vllm import AsyncLLMEngine, SamplingParams
from vllm.v1.executor import Executor

DEFAULT_EVAL_EPISODES = 500
DEFAULT_GAME_LIMIT = 500
VLLM_MAX_NUM_BATCHED_TOKENS = 4096
VLLM_MAX_NUM_SEQS = 16
MAX_RESUBMIT_RETRIES = 20


@dataclass
class OnlineGenerationState:
    """Token-level state for one request across abort/resubmit boundaries."""

    index: int
    input_ids: list[int]
    requested_max_tokens: int
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
    attempts: int = 0

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.output_tokens))

    @property
    def restart_prompt_token_ids(self) -> list[int]:
        return self.input_ids + self.output_tokens


@dataclass
class InferenceRequestItem:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    input_ids: list[int]
    requested_max_tokens: int


@dataclass
class InferenceResult:
    request_index: int
    rollout_worker_id: int
    batch_id: int
    sample_id: int
    output_tokens: list[int]
    output_logprobs: list[float]
    output_versions: list[int]
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None
    attempts: int

    @property
    def version_range(self) -> str:
        if not self.output_versions:
            return "none"
        return f"{min(self.output_versions)}-{max(self.output_versions)}"


@dataclass
class ParsedAction:
    raw_text: str
    normalized: str
    valid: bool
    action: str | None


@dataclass
class StepRecord:
    step_index: int
    raw_outputs: list[str]
    parsed_actions: list[str]
    executed_action: str
    valid: bool
    score: float
    done: bool
    won: bool
    lost: bool


@dataclass
class EpisodeResult:
    episode_index: int
    game_file: str
    won: bool
    lost: bool
    score: float
    max_score: float
    normalized_score: float
    steps: int
    env_steps: int
    invalid_actions: int
    trajectory: list[str]


def _tokens_from_output(request_output) -> list[int]:
    if not getattr(request_output, "outputs", None):
        return []
    return list(getattr(request_output.outputs[0], "token_ids", []) or [])


def _logprobs_from_output(request_output, token_ids: list[int]) -> list[float]:
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
    """Run vLLM requests with the same result shape as agent_textworld.py."""

    def __init__(
        self,
        engine: AsyncLLMEngine,
        temperature: float = 0.0,
        top_p: float = 1.0,
        collect_logprobs: bool = False,
        max_resubmit_retries: int = MAX_RESUBMIT_RETRIES,
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

    async def _increment_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts += 1
            self._active_changed.notify_all()

    async def _decrement_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts -= 1
            self._active_changed.notify_all()

    async def wait_for_idle(self) -> None:
        async with self._active_changed:
            await self._active_changed.wait_for(lambda: self._active_attempts == 0)

    async def generate(self, state: OnlineGenerationState) -> OnlineGenerationState:
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
                "stop": ["\n"],
            }
            if self.collect_logprobs:
                sampling_kwargs["logprobs"] = 1

            sampling_params = SamplingParams(**sampling_kwargs)
            request_id = (
                f"textworld-local-{state.index}-v{attempt_version}-"
                f"try{attempt}-{uuid.uuid4()}"
            )
            final_output = None
            request_finished = False

            await self._increment_active_attempts()
            try:
                async for request_output in self.engine.generate(
                    {"prompt_token_ids": state.restart_prompt_token_ids},
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


class TextWorldLocalInfer:
    """Local non-Ray equivalent of VLLMInferenceActor.request_batch()."""

    def __init__(
        self,
        engine: AsyncLLMEngine,
        temperature: float,
        top_p: float,
        collect_logprobs: bool,
    ):
        self.runner = InterruptibleGenerationRunner(
            engine,
            temperature=temperature,
            top_p=top_p,
            collect_logprobs=collect_logprobs,
        )
        self.next_request_index = 0

    async def request_batch(
        self,
        rollout_worker_id: int,
        batch_id: int,
        input_ids: list[int],
        infer_max_tokens: int,
        num_samples: int,
    ) -> list[InferenceResult]:
        if num_samples < 1:
            return []

        requests_to_process = []
        for sample_id in range(num_samples):
            request_index = self.next_request_index
            self.next_request_index += 1
            requests_to_process.append(
                InferenceRequestItem(
                    request_index=request_index,
                    rollout_worker_id=int(rollout_worker_id),
                    batch_id=int(batch_id),
                    sample_id=int(sample_id),
                    input_ids=list(input_ids),
                    requested_max_tokens=int(infer_max_tokens),
                )
            )

        tasks = [
            asyncio.create_task(
                self.runner.generate(
                    OnlineGenerationState(
                        index=item.request_index,
                        input_ids=item.input_ids,
                        requested_max_tokens=item.requested_max_tokens,
                    )
                )
            )
            for item in requests_to_process
        ]
        completed_states = await asyncio.gather(*tasks)

        return [
            InferenceResult(
                request_index=item.request_index,
                rollout_worker_id=item.rollout_worker_id,
                batch_id=item.batch_id,
                sample_id=item.sample_id,
                output_tokens=list(state.output_tokens),
                output_logprobs=list(state.output_logprobs),
                output_versions=list(state.output_versions),
                stop_reason=state.stop_reason,
                attempts=state.attempts,
            )
            for item, state in zip(requests_to_process, completed_states)
        ]


def create_async_engine(args: argparse.Namespace) -> AsyncLLMEngine:
    """Create an AsyncLLMEngine for local TextWorld action inference."""
    env_bin_dir = os.path.dirname(sys.executable)
    if env_bin_dir and env_bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = env_bin_dir + os.pathsep + os.environ.get("PATH", "")

    engine_kwargs = _filter_async_engine_args(
        dict(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend="mp",
            enable_expert_parallel=args.enable_expert_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
            disable_custom_all_reduce=True,
            max_model_len=args.max_model_len,
            enable_prefix_caching=not args.disable_prefix_caching,
        )
    )
    engine_args = vllm.AsyncEngineArgs(**engine_kwargs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )


def _filter_async_engine_args(kwargs: dict) -> dict:
    signature = inspect.signature(vllm.AsyncEngineArgs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[vllm] Ignoring unsupported AsyncEngineArgs: {dropped}")
    return filtered


def load_game_files(args: argparse.Namespace) -> list[str]:
    pattern = os.path.join(args.game_dir, args.game_pattern)
    game_files = sorted(glob.glob(pattern))
    game_files = [path for path in game_files if os.path.isfile(path)]
    if args.game_limit is not None:
        game_files = game_files[: args.game_limit]
    if not game_files:
        raise ValueError(
            "No TextWorld game files found: "
            f"game_dir={args.game_dir!r} pattern={args.game_pattern!r}"
        )
    return game_files


def make_request_infos() -> textworld.EnvInfos:
    return textworld.EnvInfos(
        objective=True,
        inventory=True,
        admissible_commands=True,
        score=True,
        max_score=True,
        won=True,
        lost=True,
        moves=True,
    )


TEXTWORLD_SYSTEM_PROMPT = (
    "You are a TextWorld action selector. Reply with exactly one command "
    "from the admissible commands. Do not add explanations, punctuation, "
    "quotes, or a leading prompt marker."
)


def format_textworld_user_content(obs: str, infos: dict) -> str:
    objective = infos.get("objective") or ""
    inventory = infos.get("inventory") or ""
    admissible_commands = infos.get("admissible_commands", []) or []
    command_lines = "\n".join(f"- {command}" for command in admissible_commands)
    return (
        "Objective:\n"
        f"{objective}\n\n"
        "Observation:\n"
        f"{obs}\n\n"
        "Inventory:\n"
        f"{inventory}\n\n"
        "Admissible commands:\n"
        f"{command_lines}\n\n"
        "Return one command only."
    )


def format_illegal_action_feedback(action: str, obs: str, infos: dict) -> str:
    return (
        f'Illegal action: "{action}" is not an admissible command.\n\n'
        + format_textworld_user_content(obs, infos)
    )


def format_textworld_prompt(obs: str, infos: dict, tokenizer=None) -> str:
    user_prompt = format_textworld_user_content(obs, infos)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": TEXTWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"System: {TEXTWORLD_SYSTEM_PROMPT}\n\n"
        f"User:\n{user_prompt}\n\n"
        "Assistant:"
    )


def format_textworld_history_prompt(
    initial_obs: str,
    initial_infos: dict,
    history_turns: list[dict],
    tokenizer=None,
    history_steps: int = 8,
) -> str:
    if history_steps <= 0:
        if history_turns:
            latest_turn = history_turns[-1]
            return format_textworld_prompt(
                latest_turn["obs"],
                latest_turn["infos"],
                tokenizer=tokenizer,
            )
        return format_textworld_prompt(initial_obs, initial_infos, tokenizer=tokenizer)

    retained_turns = history_turns[-history_steps:]
    if not retained_turns or len(retained_turns) == len(history_turns):
        first_obs = initial_obs
        first_infos = initial_infos
    else:
        first_turn = retained_turns[0]
        first_obs = first_turn["pre_obs"]
        first_infos = first_turn["pre_infos"]

    messages = [
        {"role": "system", "content": TEXTWORLD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_textworld_user_content(first_obs, first_infos),
        },
    ]
    for turn in retained_turns:
        messages.append({"role": "assistant", "content": turn["action"]})
        if turn.get("kind") == "illegal":
            messages.append(
                {
                    "role": "user",
                    "content": format_illegal_action_feedback(
                        turn["action"],
                        turn["obs"],
                        turn["infos"],
                    ),
                }
            )
            continue
        messages.append(
            {
                "role": "user",
                "content": format_textworld_user_content(turn["obs"], turn["infos"]),
            }
        )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = [f"System: {TEXTWORLD_SYSTEM_PROMPT}"]
    for message in messages[1:]:
        if message["role"] == "user":
            rendered.append(f"User:\n{message['content']}")
        elif message["role"] == "assistant":
            rendered.append(f"Assistant:\n{message['content']}")
    rendered.append("Assistant:")
    return "\n\n".join(rendered)


def _clean_action_text(text: str) -> str:
    first_line = text.splitlines()[0] if text.splitlines() else text
    cleaned = first_line.strip()
    for _ in range(4):
        cleaned = cleaned.strip()
        cleaned = cleaned.lstrip(">")
        cleaned = cleaned.strip()
        lower_cleaned = cleaned.lower()
        for prefix in (
            "action:",
            "command:",
            "assistant:",
            "answer:",
            "output:",
            "input:",
        ):
            if lower_cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        else:
            break
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.strip("'\"`")
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.strip("'\"`")
    return " ".join(cleaned.lower().split())


def parse_model_action(raw_text: str, admissible_commands: list[str]) -> ParsedAction:
    normalized = _clean_action_text(raw_text)
    command_by_normalized = {
        " ".join(command.lower().split()): command for command in admissible_commands
    }
    action = command_by_normalized.get(normalized)
    return ParsedAction(
        raw_text=raw_text,
        normalized=normalized,
        valid=action is not None,
        action=action,
    )


def _normalize_command(command: str) -> str:
    return " ".join(command.lower().split())


def score_from_step(step_score, infos: dict) -> float:
    value = infos.get("score", step_score)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def max_score_from_infos(infos: dict) -> float:
    value = infos.get("max_score", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_eval_summary(results: list[EpisodeResult], game_count: int) -> None:
    episode_count = len(results)
    if episode_count == 0:
        print("[eval-summary] episodes=0")
        return

    wins = sum(1 for result in results if result.won)
    total_score = sum(result.score for result in results)
    total_normalized_score = sum(result.normalized_score for result in results)
    total_steps = sum(result.steps for result in results)
    total_env_steps = sum(result.env_steps for result in results)
    total_invalid_actions = sum(result.invalid_actions for result in results)
    invalid_action_rate = (
        total_invalid_actions / total_steps if total_steps else 0.0
    )

    print("=" * 80)
    print(
        "[eval-summary] "
        f"episodes={episode_count} "
        f"wins={wins} "
        f"win_rate={wins / episode_count:.4f} "
        f"avg_score={total_score / episode_count:.4f} "
        f"avg_normalized_score={total_normalized_score / episode_count:.4f} "
        f"avg_steps={total_steps / episode_count:.2f} "
        f"avg_env_steps={total_env_steps / episode_count:.2f} "
        f"invalid_action_rate={invalid_action_rate:.4f} "
        f"game_count={game_count}"
    )


async def run_episode(
    episode_index: int,
    game_file: str,
    local_infer: TextWorldLocalInfer,
    tokenizer,
    args: argparse.Namespace,
) -> EpisodeResult:
    request_infos = make_request_infos()
    env_id = textworld.gym.register_game(
        game_file,
        request_infos=request_infos,
        max_episode_steps=args.max_episode_steps,
    )
    env = textworld.gym.make(env_id)
    basename = os.path.basename(game_file)
    trajectory: list[StepRecord] = []
    history_turns: list[dict] = []
    invalid_count = 0
    env_steps = 0
    batch_id = 0

    try:
        obs, infos = env.reset()
        initial_obs = obs
        initial_infos = dict(infos)
        if args.verbose:
            print("=" * 80)
            print(f"[episode] start index={episode_index} game={basename}")
            print(f"[episode] objective={infos.get('objective')!r}")

        done = False
        step_index = 0
        latest_score = score_from_step(0, infos)
        while not done and step_index < args.max_episode_steps:
            prompt_obs = obs
            prompt_infos = dict(infos)
            if args.history_steps > 0:
                prompt = format_textworld_history_prompt(
                    initial_obs,
                    initial_infos,
                    history_turns,
                    tokenizer=tokenizer,
                    history_steps=args.history_steps,
                )
            else:
                prompt = format_textworld_prompt(obs, infos, tokenizer=tokenizer)
            if args.verbose:
                print("[prompt]")
                print(prompt)
            input_ids = tokenizer.encode(prompt)
            results = await local_infer.request_batch(
                rollout_worker_id=0,
                batch_id=batch_id,
                input_ids=list(input_ids),
                infer_max_tokens=args.max_action_tokens,
                num_samples=args.num_samples,
            )
            batch_id += 1

            admissible_commands = infos.get("admissible_commands", []) or []
            raw_texts = [
                tokenizer.decode(
                    result.output_tokens,
                    skip_special_tokens=True,
                )
                for result in results
            ]
            parsed_actions = [
                parse_model_action(raw_text, admissible_commands)
                for raw_text in raw_texts
            ]
            selected = next(
                (
                    parsed
                    for parsed in parsed_actions
                    if parsed.valid and parsed.action is not None
                ),
                None,
            )
            selected_valid = selected is not None
            if selected_valid:
                executed_action = selected.action
                obs, step_score, done, infos = env.step(executed_action)
                env_steps += 1
                if args.history_steps > 0:
                    history_turns.append(
                        {
                            "kind": "env",
                            "action": executed_action,
                            "pre_obs": prompt_obs,
                            "pre_infos": dict(prompt_infos),
                            "obs": obs,
                            "infos": dict(infos),
                        }
                    )
                latest_score = score_from_step(step_score, infos)
                won = bool(infos.get("won", False))
                lost = bool(infos.get("lost", False))
            else:
                invalid_count += 1
                first_raw_text = raw_texts[0] if raw_texts else ""
                first_parsed = parsed_actions[0] if parsed_actions else None
                executed_action = (
                    first_parsed.normalized
                    if first_parsed is not None and first_parsed.normalized
                    else first_raw_text.strip()
                )
                if args.history_steps > 0:
                    history_turns.append(
                        {
                            "kind": "illegal",
                            "action": executed_action,
                            "pre_obs": prompt_obs,
                            "pre_infos": dict(prompt_infos),
                            "obs": prompt_obs,
                            "infos": dict(prompt_infos),
                        }
                    )
                won = bool(infos.get("won", False))
                lost = bool(infos.get("lost", False))

            trajectory.append(
                StepRecord(
                    step_index=step_index,
                    raw_outputs=raw_texts,
                    parsed_actions=[parsed.normalized for parsed in parsed_actions],
                    executed_action=executed_action,
                    valid=selected_valid,
                    score=latest_score,
                    done=bool(done),
                    won=won,
                    lost=lost,
                )
            )

            if args.verbose:
                print("-" * 80)
                print(f"[step] episode={episode_index} step={step_index}")
                print(
                    "[admissible] "
                    + ", ".join(repr(command) for command in admissible_commands)
                )
                for result, raw_text, parsed in zip(results, raw_texts, parsed_actions):
                    print(
                        "[sample] "
                        f"sample_id={result.sample_id} "
                        f"raw={raw_text!r} "
                        f"parsed={parsed.normalized!r} "
                        f"valid={parsed.valid} "
                        f"tokens={len(result.output_tokens)} "
                        f"versions={result.version_range} "
                        f"stop={result.stop_reason} "
                        f"attempts={result.attempts}"
                    )
                if selected_valid:
                    print(
                        "[env] "
                        f"executed={executed_action!r} "
                        f"valid={selected_valid} "
                        f"env_advanced=True "
                        f"score={latest_score:g} "
                        f"done={done} "
                        f"won={won} "
                        f"lost={lost}"
                    )
                else:
                    print(
                        "[illegal-action] "
                        f"attempted={executed_action!r} "
                        f"valid={selected_valid} "
                        f"env_advanced=False "
                        f"score={latest_score:g} "
                        f"done={done} "
                        f"won={won} "
                        f"lost={lost}"
                    )

            step_index += 1

        max_score = max_score_from_infos(infos)
        normalized_score = latest_score / max_score if max_score > 0 else 0.0
        won = bool(infos.get("won", False))
        lost = bool(infos.get("lost", False))
        action_trajectory = [record.executed_action for record in trajectory]
        print("=" * 80)
        print(
            "[episode-summary] "
            f"index={episode_index} "
            f"game={basename} "
            f"steps={len(trajectory)} "
            f"env_steps={env_steps} "
            f"score={latest_score:g}/{max_score:g} "
            f"normalized_score={normalized_score:.4f} "
            f"won={won} "
            f"lost={lost} "
            f"invalid_actions={invalid_count}"
        )
        actions = " -> ".join(action_trajectory)
        if args.verbose:
            print(f"[trajectory] {actions}")
        return EpisodeResult(
            episode_index=episode_index,
            game_file=game_file,
            won=won,
            lost=lost,
            score=latest_score,
            max_score=max_score,
            normalized_score=normalized_score,
            steps=len(trajectory),
            env_steps=env_steps,
            invalid_actions=invalid_count,
            trajectory=action_trajectory,
        )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local vLLM action inference over TextWorld games."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Required local HuggingFace model path.",
    )
    parser.add_argument(
        "--game-dir",
        required=True,
        help="Required directory containing TextWorld .z8 games.",
    )
    parser.add_argument("--game-pattern", default="*.z8")
    parser.add_argument("--game-limit", type=int, default=DEFAULT_GAME_LIMIT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-episode-steps", type=int, default=20)
    parser.add_argument("--max-action-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--collect-logprobs", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print prompts, per-step samples, trajectories, and every game path.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=8,
        help=(
            "Number of recent TextWorld action/observation turns to keep in "
            "the chat prompt. 0 preserves the previous single-state prompt."
        ),
    )
    parser.add_argument(
        "--fallback-action",
        default="auto",
        help=(
            "Legacy option retained for CLI compatibility. Invalid model "
            "actions are now fed back to the model instead of falling back."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM automatic prefix caching for history prompts.",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=VLLM_MAX_NUM_SEQS,
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=VLLM_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.game_limit is not None and args.game_limit < 1:
        raise ValueError("--game-limit must be >= 1 when set")
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be >= 1")
    if args.max_action_tokens < 1:
        raise ValueError("--max-action-tokens must be >= 1")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.history_steps < 0:
        raise ValueError("--history-steps must be >= 0")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len < 1:
        raise ValueError("--max-model-len must be >= 1")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be >= 1")
    if args.vllm_max_num_batched_tokens < 1:
        raise ValueError("--vllm-max-num-batched-tokens must be >= 1")
    return args


async def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    game_files = load_game_files(args)

    print(
        "[init] Loaded TextWorld games: "
        f"count={len(game_files)} "
        f"episodes={args.episodes} "
        f"game_dir={args.game_dir!r} "
        f"game_pattern={args.game_pattern!r}"
    )
    if args.verbose:
        for path in game_files:
            print(f"[init] game={path}")

    print(f"[init] Loading local model from {args.model_path}")
    engine = create_async_engine(args)
    tokenizer = engine.get_tokenizer()
    local_infer = TextWorldLocalInfer(
        engine,
        temperature=args.temperature,
        top_p=args.top_p,
        collect_logprobs=args.collect_logprobs,
    )

    try:
        results: list[EpisodeResult] = []
        for episode_index in range(args.episodes):
            game_file = game_files[episode_index % len(game_files)]
            results.append(
                await run_episode(
                    episode_index=episode_index,
                    game_file=game_file,
                    local_infer=local_infer,
                    tokenizer=tokenizer,
                    args=args,
                )
            )
        print_eval_summary(results, game_count=len(game_files))
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if asyncio.iscoroutine(result):
                await result


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
