# AcceRL-Agent

AcceRL-Agent is an asynchronous framework for online reinforcement learning with language-model agents. The main example in this repository is a TextWorld training loop: the model samples actions online through vLLM, the environment returns rewards, an FSDP trainer asynchronously consumes rollout samples, and updated weights are hot-synced back into vLLM.

Main entry point:

```text
accerl_agent/agent_textworld.py
```

End-to-end loop:

```text
TextWorld episode -> vLLM action generation -> environment step -> RLSample
       -> Replay Buffer -> FSDP trainer parameter update
       -> NCCL weight sync -> vLLM samples with newer weights
```

If you only want to run the smallest working flow first, see [QUICKSTART.md](QUICKSTART.md).

## Features

- TextWorld environment rollout.
- Online vLLM inference and action sampling.
- Distributed PyTorch FSDP training.
- Asynchronous replay buffer for caching and reusing training samples.
- Trainer-side token-budget packing with FlashAttention 2 varlen attention.
- Response-only LM-head projection through the model-native `logits_to_keep` API.
- NCCL weight transfer from FSDP to vLLM.
- Raw token PPO replay with trainer-side current-value TD(λ), plus GRPO.

## Repository Layout

| Path | Description |
| --- | --- |
| `accerl_agent/run_agent_textworld.py` | Canonical launcher for Ray-safe TextWorld training startup. |
| `accerl_agent/agent_textworld.py` | Full Ray + vLLM + FSDP online RL training implementation. |
| `accerl_agent/rl_data.py` | Canonical replay schemas, strict PPO validation, and detached scalar/batched token GAE. |
| `accerl_agent/ppo_value.py` | Shared-backbone FP32 Value Head and Critic checkpoint helpers. |
| `accerl_agent/textworld_local_infer.py` | Checks vLLM inference and TextWorld environment interaction without training. |
| `accerl_agent/local_trainer.py` | Local dummy SFT smoke test for tokenizer/model/FSDP training paths. |
| `accerl_agent/vllm_*.py` | Experimental scripts for vLLM, NCCL, and rollout-engine work. |
| `QUICKSTART.md` | Minimal run path, recommended scaling order, and troubleshooting checklist. |

## Requirements

This project targets Linux GPU environments. Exact package versions must match your CUDA, PyTorch, and vLLM stack. The main training path requires at least:

- Python >=3.10,<3.15. Python 3.10 is recommended for conda environments.
- NVIDIA GPUs and a CUDA-enabled PyTorch build.
- PyTorch with FSDP2 support.
- vLLM with the weight-transfer API around `WeightTransferConfig`.
- Ray, Transformers, TensorBoard, and TextWorld.
- A local HuggingFace Causal LM model directory.
- TextWorld `.z8` game files.

Several parts of the current code assume a model layout close to Qwen/Qwen-MoE, such as `model.model.layers` and `lm_head`. If you use another HuggingFace model family, carefully check `build_model()`, `configure_trainable_parameters()`, the FSDP wrapping path, and `iter_vllm_loadable_weights()`.

Packed training requires a GPU-supported `flash-attn`
build. The `response_only_lm_head` logprob mode also requires a Transformers
model that implements tensor `logits_to_keep`; packed GRPO with `full_logits_ce`
does not require that API. Transformers 5.12.1 with Qwen/Qwen-MoE is the
currently verified combination for response-only LM-head projection. Other
Transformers versions or model classes may not support this API.

## Installation

The repository is not packaged as a pip package yet. Run scripts directly from the repository root.

```bash
git clone <REPO_URL>
cd AcceRL-Agent

conda create -n accerl-agent python=3.10 -y
conda activate accerl-agent
python -m pip install --upgrade pip setuptools wheel ninja packaging

python -m pip install torch==2.11.0
python -m pip install --no-build-isolation -r requirements.txt

python -c "import flash_attn, ray, torch, transformers, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('flash-attn', flash_attn.__version__); print('ray', ray.__version__)"
```

The required stack uses PyTorch 2.11.0, Transformers 5.12.1, vLLM 0.21.0 or
newer, FlashAttention 2.8.3.post1, and Ray 2.56.0. The current local
environment uses vLLM 0.24.0. Install PyTorch first because FlashAttention
imports it while building, and keep `--no-build-isolation` on the requirements
installation command. These packages are tightly coupled to CUDA; update the
related pins together when using a different cluster-provided stack.

## Quickstart

Set the model and TextWorld game paths first:

```bash
export MODEL_PATH=<LOCAL_HF_MODEL_PATH>
export TEXTWORLD_GAME_DIR=<TEXTWORLD_Z8_GAME_DIR>
```

First, run the local multi-trainer smoke test. It does not depend on vLLM or TextWorld; it only checks tokenizer/model loading, response-only labels, Ray FSDP multi-trainer initialization, forward/backward, and optimizer steps. The example below starts 2 FSDP trainers and needs at least 2 visible GPUs.

```bash
python accerl_agent/local_trainer.py \
  --model-path "$MODEL_PATH" \
  --train-mode full \
  --use-fsdp \
  --fsdp-world-size 2 \
  --max-steps 5 \
  --batch-size 1 \
  --max-length 128 \
  --trust-remote-code
```

Second, run local TextWorld inference without training:

```bash
python accerl_agent/textworld_local_infer.py \
  --model-path "$MODEL_PATH" \
  --game-dir "$TEXTWORLD_GAME_DIR" \
  --game-pattern "*.z8" \
  --episodes 2 \
  --game-limit 2 \
  --max-episode-steps 10 \
  --num-samples 1 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --vllm-max-num-seqs 4 \
  --vllm-max-num-batched-tokens 2048
```

Third, run the current long-running end-to-end configuration. It uses 3 FSDP
trainer GPUs and 1 vLLM inference GPU, so it needs at least 4 visible GPUs. The
rollout algorithm is GRPO, while the policy objective uses PPO-style clipping:

```bash
python -m accerl_agent.run_agent_textworld \
  --model-path "$MODEL_PATH" \
  --tw-game-dir "$TEXTWORLD_GAME_DIR" \
  --tw-game-pattern "*.z8" \
  --tw-max-episode-steps 50 \
  --tw-history-token-window 8192 \
  --tw-game-limit 400 \
  --max-length 8192 \
  --gae-gamma 1.0 \
  --tw-lost-penalty 0.0 \
  --fsdp-world-size 3 \
  --infer-size 1 \
  --infer-tp-size 1 \
  --num-rollout-workers 24 \
  --rollout-batch-size 8 \
  --infer-max-tokens 16 \
  --infer-temperature 1.0 \
  --infer-top-p 1.0 \
  --train-max-sequences-per-pack 2 \
  --grad-accum-steps 32 \
  --max-steps 500000 \
  --lr-warmup-steps 500 \
  --train-mode full \
  --sync-every-optimizer-steps 1 \
  --clip-mode ppo \
  --trust-remote-code \
  --replay-capacity 256 \
  --min-replay-size-per-rank 32 \
  --rl-algorithm grpo \
  --train-token-budget 16384 \
  --train-pack-candidate-pool-size 64 \
  --train-logprob-mode response_only_lm_head \
  --dtype bfloat16
```

The command above is the full GRPO packed configuration and is intended for
long-running training. For a much smaller 2-GPU PPO validation run, use:

```bash
python -m accerl_agent.run_agent_textworld \
  --model-path "$MODEL_PATH" \
  --tw-game-dir "$TEXTWORLD_GAME_DIR" \
  --tw-game-pattern "*.z8" \
  --tw-game-limit 2 \
  --tw-max-episode-steps 10 \
  --tw-history-token-window 1024 \
  --max-length 1024 \
  --fsdp-world-size 1 \
  --infer-size 1 \
  --infer-tp-size 1 \
  --num-rollout-workers 1 \
  --rollout-batch-size 2 \
  --train-max-sequences-per-pack 2 \
  --grad-accum-steps 1 \
  --replay-capacity 8 \
  --min-replay-size-per-rank 2 \
  --max-steps 2 \
  --max-sync-rounds 1 \
  --sync-every-optimizer-steps 1 \
  --train-mode full \
  --rl-algorithm ppo \
  --clip-mode ppo \
  --train-token-budget 2048 \
  --gae-lambda 0.95 \
  --value-loss-coef 0.5 \
  --dtype bfloat16 \
  --trust-remote-code
```

Packed training selects at most `--train-max-sequences-per-pack` samples and packs no more than
`--train-token-budget` real tokens into one microbatch. PPO and GRPO prepare
`--grad-accum-steps` CPU packs before each optimizer step. Both algorithms use
trajectory-equal optimization reduction: PPO first averages policy, unclipped
Value MSE, and KL token objectives within each complete `RawPPOSample`; GRPO
does the same for policy and KL within each complete `GRPOSample`. They then
average trajectories with at least one valid response token across the complete
distributed optimizer window. PPO ratios, clipping, GAE, returns, value
predictions, and advantage-normalization moments remain token-level. A
trajectory is one replay sample, not one turn/action, and this reduction is not
sequence-level importance sampling or prompt/group-equal weighting. Unlike
slime's default per-sample metric reducer, AcceRL intentionally keeps PPO clip
fraction as a global valid-token ratio so that it continues to answer how many
token ratios crossed the clip boundary.

GPU requirement for full training:

```text
total GPUs >= fsdp_world_size + infer_tp_size * infer_size
```

Open TensorBoard:

```bash
tensorboard --logdir runs/TextWorld_FSDP
```

Each run writes `args.json`, `command.txt`, and TensorBoard event files under:

```text
runs/TextWorld_FSDP/<timestamp>
```

## Architecture

`agent_textworld.py` creates five main Ray actor types:

```mermaid
flowchart LR
    Main["run_textworld_train"]
    Trainer["FSDPTrainWorker x fsdp_world_size"]
    Infer["VLLMInferenceActor"]
    Rollout["TextWorldRolloutWorkerActor x num_rollout_workers"]
    Replay["ReplayBufferActor x fsdp_world_size"]
    Stats["StatsActor"]
    Env["TextWorld .z8 games"]

    Main --> Trainer
    Main --> Infer
    Main --> Rollout
    Main --> Replay
    Main --> Stats

    Rollout --> Env
    Rollout -->|"tokenized prompt"| Infer
    Infer -->|"tokens + old logprobs + version"| Rollout
    Rollout -->|"RLSample"| Replay
    Rollout -->|"episode metrics"| Stats
    Trainer -->|"sample batch"| Replay
    Trainer -->|"NCCL trainable weights"| Infer
```

`FSDPTrainWorker` loads the tokenizer and `AutoModelForCausalLM`, trains the full policy model, and samples independent replay objects. It also owns a separately sharded FP32 Value Head. PPO uses packed FlashAttention 2 boundaries, native tensor `logits_to_keep`, and a temporary model-native LM-head hook, so only response/bootstrap logits and final hidden states are retained. It computes current values and batched detached TD(λ) targets when replay is sampled, then optimizes policy and Value losses with trajectory-equal reduction while retaining token-weighted advantage normalization and diagnostics. GRPO uses the same token-budget packing and trajectory-equal reduction pipeline. Weight synchronization to vLLM remains policy-only.

`VLLMInferenceActor` handles rollout inference. It starts vLLM with dummy weights, waits for the initial full weight sync, pauses generation during later syncs, aborts requests when needed, updates weights, and then resumes generation.

`TextWorldRolloutWorkerActor` is a CPU actor. It loads `.z8` games, builds prompts, asks vLLM to generate actions, parses actions, steps the environment, computes rewards, builds `RLSample` objects, and writes them to replay.

`ReplayBufferActor` stores samples sharded by FSDP rank. Rollout workers write to `replay_buffers[worker_id % fsdp_world_size]`, so `--num-rollout-workers` must be at least `--fsdp-world-size`.

`StatsActor` maintains sliding-window metrics and feeds TensorBoard with win rate, normalized score, invalid action rate, replay fill, reward, advantage, throughput, and sync-latency statistics.

## TextWorld Rollout

The TextWorld prompt includes the objective, observation, inventory, and admissible commands. It asks the model to return exactly one command. The parser:

1. Takes the first line of model output.
2. Removes common prefixes such as `action:`, `command:`, and `assistant:`.
3. Removes a leading `>` marker, trailing periods, and wrapping quotes or
   backticks.
4. Lowercases and normalizes whitespace.
5. Exact-matches the normalized action against the current admissible commands.

`TextWorld/InvalidActionRate` is one of the most important early metrics. If it is high, first check the prompt, `--infer-max-tokens`, temperature, parser strictness, and whether the admissible commands are fully included in the prompt.

## Rewards and Algorithms

TextWorld step reward is computed in `_compute_step_reward()` from score deltas:

```python
reward = score_after - score_before
if won:
    reward += tw_win_bonus
if lost:
    reward -= tw_lost_penalty
```

PPO mode is enabled with `--rl-algorithm ppo`. Rollout stores token-aligned rewards, behavior
logprobs, terminal/truncation boundaries, and optional final-state bootstrap
context, but no values, returns, or advantages. The trainer recomputes current
values and detached token TD(λ) targets on every replay sample. `--gae-gamma`
discounts once per valid response token; configure the trace and Critic weight
with `--gae-lambda` and `--value-loss-coef`. PPO advantage normalization
defaults to exact moments over the complete optimizer accumulation window on
all FSDP ranks. The `ema_rms` and `ema_zscore` modes use historical global
moments so initialized EMA steps need only one forward per pack; their first
step uses the exact window to initialize those moments.

GRPO mode is enabled with `--rl-algorithm grpo`. A group of full trajectories is sampled from the same game, then rewards are normalized within the group:

```python
advantage = (reward - group_mean) / (group_std + eps)
```

The training-side policy objective is controlled by `--clip-mode`:

- `ppo`: standard clipped surrogate using `--clip-eps`.
- `gipo`: log-ratio Gaussian soft clipping using `--gipo-sigma`.
- `sapo`: separate gate temperatures for positive and negative advantages using `--sapo-tau-pos` and `--sapo-tau-neg`.

## Replay Sample Contract

Replay stores `RawPPOSample | GRPOSample`. Every token field is aligned to
`input_ids`:

```text
len(input_ids) == len(labels)
len(input_ids) == len(old_logprobs)
len(input_ids) == len(response_indices)
len(input_ids) == len(output_versions)
```

`RawPPOSample` additionally aligns `token_rewards`, `token_terminated`, and
`token_truncated`. Ignored tokens use `label=-100`, zero reward/logprob,
`False` boundaries, and `response_index=output_version=-1`. Exactly one
terminal or truncation boundary appears on the final response token.
Truncations include ignored final-state prompt context and a valid bootstrap
prediction position. TextWorld `step_limit` and `history_limit` boundaries are
treated as terminal failures, so their final-state value is zero and they do
not bootstrap. The rollout tracks successful environment steps separately
from action attempts so the TextWorld time-limit wrapper is also classified as
`step_limit`. PPO rollout never stores values, returns, or advantages.
`GRPOSample` instead stores one trajectory-level advantage.

## Important Arguments

| Argument | Description |
| --- | --- |
| `--model-path` | Local HuggingFace model path. |
| `--dtype` | `auto`, `bfloat16`, `float16`, or `float32`. |
| `--train-mode` | `full` is supported; `lora` is reserved for a future adapter-training implementation. |
| `--tw-game-dir` | Directory containing TextWorld `.z8` games. |
| `--tw-history-token-window` | Token limit for the episode transcript. |
| `--max-length` | Maximum trainer-side sequence length; must be at least `--tw-history-token-window`. |
| `--fsdp-world-size` | Number of FSDP trainer GPUs. |
| `--infer-size` | vLLM data-parallel size. |
| `--infer-tp-size` | vLLM tensor-parallel size. |
| `--num-rollout-workers` | Number of CPU rollout actors; must be at least `--fsdp-world-size`. |
| `--rollout-batch-size` | Episode batch size per rollout worker in PPO mode; also the default GRPO group size. |
| `--train-max-sequences-per-pack` | Maximum number of independent `RLSample` objects in one packed microbatch, per FSDP rank. |
| `--gae-lambda` | PPO token TD(λ) trace parameter; defaults to `0.95`. |
| `--value-loss-coef` | Coefficient for unclipped per-token Value MSE with trajectory-equal PPO reduction; defaults to `0.5`. |
| `--ppo-advantage-normalization` | PPO advantage mode: `optimizer_window` (default), `ema_rms`, `ema_zscore`, or `none`. |
| `--ppo-advantage-ema-beta` | Per-optimizer-step EMA decay for historical advantage moments; defaults to `0.9`. |
| `--ppo-advantage-min-scale` | Positive scale floor for EMA normalization; defaults to `1e-3`. |
| `--train-token-budget` | Maximum real tokens in a pack; required and must be at least `--max-length`. |
| `--train-pack-candidate-pool-size` | Replay candidate pool used for length-aware packing; defaults to four times `--train-max-sequences-per-pack`. |
| `--train-logprob-mode` | GRPO logprob mode. PPO always uses its native selected-position forward. |
| `--grad-accum-steps` | Gradient accumulation steps. |
| `--replay-capacity` | Maximum number of samples in each replay buffer. |
| `--min-replay-size-per-rank` | Minimum replay size required before a trainer rank starts training. |
| `--sync-every-optimizer-steps` | Number of optimizer steps between vLLM weight syncs. |
| `--max-sync-rounds` | Maximum number of train/sync segments; useful for smoke tests. |
| `--save-checkpoint` | Save a HuggingFace-format model checkpoint. |
| `--checkpoint-every-sync-rounds` | Periodic checkpoint interval; `0` disables periodic saves. |

## Checkpoints

Enable checkpoint saving:

```bash
--save-checkpoint
```

Default output path:

```text
<log-dir>/checkpoints/latest
```

Periodic and final saves both overwrite `latest`, so only the newest model is
retained.

Saved checkpoint contents include model weights, config, tokenizer files, and `trainer_state.json`. Optimizer state and replay-buffer contents are not saved yet, so these checkpoints are mainly for inference/evaluation rather than full training resume.

## Key Metrics

| Metric | Meaning |
| --- | --- |
| `TextWorld/WinRate` | Episode win rate over the recent window. |
| `TextWorld/NormalizedScore` | Normalized score over the recent window. |
| `TextWorld/InvalidActionRate` | Fraction of invalid actions. |
| `TextWorld/EnvStepsMean` | Average number of valid environment steps per episode. |
| `Replay/FillRatio` | Replay-buffer fill ratio. |
| `Replay/TrainSampleTrainerVersionLagMean` | Version lag between training samples and the current trainer. |
| `Train/LossMeanAcrossRanks` | Average loss across FSDP ranks. |
| `Train/PackTokenUtilization` | Fraction of `--train-token-budget` occupied by real tokens in a pack. |
| `Train/PackSampleCount` | Number of independent samples in a pack. |
| `Train/PackMaxSequenceLength` | Longest sequence in the current pack. |
| `Train/PackCpuMilliseconds` | CPU time spent selecting and constructing a pack. |
| `Train/PolicyLossTokenMean` | PPO global valid-token policy-loss diagnostic; `Train/PolicyLoss` is the trajectory mean used by optimization. |
| `Train/ValueLossTokenMean` | PPO global valid-token Value-loss diagnostic; `Train/ValueLoss` is the trajectory mean used by optimization. |
| `KL/OldNewK3TrajectoryMean` | PPO/GRPO KL penalty used by optimization: valid-token mean within each trajectory, then an equal mean across valid trajectories. |
| `KL/OldNewK3TokenMean` | Global valid-token KL diagnostic reported alongside the trajectory-mean KL. |
| `Clip/PPOClipFrac` | Fraction of global valid response tokens outside the PPO ratio clip interval; intentionally remains token-level for both algorithms. |
| `Infer/TokensPerSec` | vLLM generation throughput. |
| `Sync/ElapsedSeconds` | Weight-sync latency. |
| `Sync/RetryScheduledToFirstTokenMsMean` | Compute-side first-token latency proxy for attempts explicitly resubmitted after sync. |
| `Sync/RetryRecomputedTokensMean` | Prompt tokens not served by prefix cache for successful sync-retry probes. |
| `Sync/RetryQueueMsMean` | Scheduler queue time before a sync-retry attempt is scheduled. |

### TextWorld inference throughput diagnostics

`Infer/TokensPerSec` is the number of completed output tokens during a trainer
segment divided by that segment's wall-clock duration. It is a system-level
rollout rate, not a pure vLLM decode benchmark. TextWorld runs also report the
following per-segment diagnostics:

| Metric family | Meaning |
| --- | --- |
| `Infer/RequestsPerSec`, `Infer/RequestCount` | Logical request supply rate and count. |
| `Infer/OutputTokensPerRequest` | Mean completed output length. |
| `Infer/PromptTokensMean\|P50\|P95` | Logical-request prompt-length distribution. |
| `Infer/RequestLatencyMsMean\|P50\|P95` | End-to-end logical-request latency, including resubmits. |
| `Infer/TTFTMsMean\|P95` | Engine-attempt time to first generated token. |
| `Sync/RetryScheduledToFirstTokenMsMean\|Count` | From vLLM scheduling to first token for explicit sync retries; a prefill/KV-materialization plus first-decode proxy, not pure KV-cache write time. |
| `Sync/RetryRecomputedTokensMean\|Count` | Retry prompt tokens minus the top-level `RequestOutput.num_cached_tokens`; invalid cache counts are skipped. |
| `Sync/RetryQueueMsMean\|Count` | Time from vLLM queueing to scheduling for explicit sync retries. |
| `Infer/TPOTMsMean\|P95` | Time per output token for successful attempts with at least two tokens. |
| `Infer/ActiveRequestsMean\|Max`, `Infer/ActiveAttemptsMean\|Max` | Time-weighted logical-request and engine-attempt concurrency. |
| `Infer/AttemptsPerRequest`, `Infer/SyncInterruptedAttemptRate`, `Infer/ResubmittedRequestRate`, `Infer/PauseActiveAttempts` | Weight-sync interruption and retry pressure; the interruption rate is measured across attempts active when the following sync pause begins. |
| `Infer/StopRate\|LengthRate\|AbortRate` | Final logical-request stop-reason fractions. |
| `Rollout/EpisodesPerSec`, `Rollout/InferenceWaitFraction` | Episode production and fraction of aggregate worker time waiting for inference. |
| `Rollout/EnvStepMsMean\|P95`, `Rollout/PostprocessMsMean\|P95` | TextWorld environment and decode/parse/history-update CPU costs. |
| `Rollout/HistoryLimitRate` | Fraction of completed episodes stopped by the history-token limit. |

Use the metrics together to classify a throughput drop:

| Observation | Likely bottleneck |
| --- | --- |
| Requests/sec falls while request latency is stable | Rollout or environment request supply. |
| Output tokens/request falls proportionally with tokens/sec | Shorter model outputs rather than slower inference. |
| Prompt P95 and TTFT rise while TPOT stays stable | Longer-context prefill. |
| TPOT rises at stable prompt lengths and concurrency | Decode throughput. |
| Active requests fall while rollout CPU timings rise | Environment, parsing, or tokenization. |
| Interrupted-attempt and resubmitted-request rates rise | Weight-sync interruption overhead. |
| Retry recomputed tokens and scheduled-to-first-token time rise while retry queue time stays low | Sync-triggered prefill/KV reconstruction is a likely recovery bottleneck. |
| Retry queue time dominates scheduled-to-first-token time | Concurrent retry scheduler backlog is a more likely recovery bottleneck. |
| Request latency rises while TTFT and TPOT stay stable | Queueing or scheduling delay. |

`Infer/DiagnosticsDroppedSamples` and
`Rollout/DiagnosticsDroppedSamples` should remain zero. A non-zero value means
that a segment exceeded the bounded 20,000-sample percentile buffer; counts,
sums, means, and maxima remain exact, but percentiles cover only retained
samples.

## Troubleshooting

If the trainer keeps waiting for replay, check `--num-rollout-workers`, `--min-replay-size-per-rank`, `--replay-capacity`, the TextWorld game path, and `TextWorld/InvalidActionRate`.

If many actions are invalid, lower the temperature, reduce `--infer-max-tokens`, enable more detailed rollout logs, and confirm that the parser matches the task output format.

If vLLM weight sync fails, check GPU counts, vLLM weight-transfer API support, the NCCL environment, the names/shapes/dtypes returned by `iter_vllm_loadable_weights()`, and whether the trainable parameter set is empty.

If loss or KL is unstable, lower the learning rate, reduce replay staleness,
increase the KL penalty, and confirm that invalid, aborted, or empty outputs
are not mistakenly labeled as trainable tokens. Compare the default exact
`optimizer_window` normalization with the single-forward `ema_rms` mode, or
disable normalization with `--ppo-advantage-normalization none`.

If packed model loading fails, verify that `flash_attn` imports in the trainer
environment, the model supports `flash_attention_2`, and the dtype is
`bfloat16`, `float16`, or `auto`. If
`--train-logprob-mode response_only_lm_head` fails, also confirm that the model
forward accepts tensor `logits_to_keep`; alternatively, use
`--train-logprob-mode full_logits_ce` for GRPO.
