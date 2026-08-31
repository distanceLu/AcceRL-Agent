# Quickstart

This guide takes a fresh checkout through the currently used end-to-end
TextWorld training configuration. It also gives a recommended checklist for
changing models or replacing the task.

The canonical training entry point is
`python -m accerl_agent.run_agent_textworld`; `agent_textworld.py` is its
implementation module.

All commands are intended to run from the repository root:

```bash
cd AcceRL-Agent
export ACCERL_ROOT="$PWD"
```

## 1. Install Dependencies

Create a conda environment first:

```bash
conda create -n accerl-agent python=3.10 -y
conda activate accerl-agent
python -m pip install --upgrade pip setuptools wheel ninja packaging

python -m pip install torch==2.11.0
python -m pip install --no-build-isolation -r requirement.txt

python -c "import flash_attn, ray, torch, transformers, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('flash-attn', flash_attn.__version__); print('ray', ray.__version__)"
```

`requirement.txt` pins the tested stack: PyTorch 2.11.0, Transformers 5.12.1,
vLLM 0.24.0, FlashAttention 2.8.3.post1, Ray 2.56.0, TextWorld 1.7.0,
TensorBoard 2.21.0, Safetensors 0.8.0, and Hugging Face Hub 1.21.0. PyTorch is
installed first because FlashAttention imports it during its build. Keep
`--no-build-isolation` on the second command. These packages are tightly
coupled to CUDA and vLLM's weight-transfer API; update and validate the related
pins together rather than changing only one package.

## 2. Download the Model and TextWorld Dataset

The verified model is
[Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat).
It has approximately 14B total BF16 parameters and occupies about 27 GB after
download. The pinned Hugging Face Hub dependency provides the `hf` CLI. Use it
to download the model into a repository-local artifact directory:

```bash
command -v curl
command -v unzip
command -v hf

mkdir -p "$ACCERL_ROOT/artifacts/models"

hf download Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --local-dir "$ACCERL_ROOT/artifacts/models/Qwen1.5-MoE-A2.7B-Chat"

export MODEL_PATH="$ACCERL_ROOT/artifacts/models/Qwen1.5-MoE-A2.7B-Chat"
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors.index.json"
```

`hf download` reuses already-downloaded files when the command is run again.
If Hugging Face requires authentication in your environment, run
`hf auth login` first. Review the model card and license before use. Install
the `curl` and `unzip` system utilities first if either check above fails.

The training examples use
[The First TextWorld Problems dataset](https://www.microsoft.com/en-us/download/details.aspx?id=100932).
The official archive is about 1.6 GB and contains training, validation, and
test games. Download and extract it as follows:

```bash
mkdir -p "$ACCERL_ROOT/artifacts/textworld"
export FTWP_ARCHIVE="$ACCERL_ROOT/artifacts/textworld/cog2019_ftwp.zip"

if ! unzip -tq "$FTWP_ARCHIVE" >/dev/null 2>&1; then
  curl --fail --location --continue-at - \
    "https://download.microsoft.com/download/e/8/0/e80a789f-ed4a-443b-9bd8-a1cf297c1a70/cog2019_ftwp.zip" \
    --output "$FTWP_ARCHIVE"
fi
unzip -tq "$FTWP_ARCHIVE"

unzip -q -o \
  "$FTWP_ARCHIVE" \
  -d "$ACCERL_ROOT/artifacts/textworld/ftwp"

export TEXTWORLD_GAME_DIR="$ACCERL_ROOT/artifacts/textworld/ftwp/games/train"
test -d "$TEXTWORLD_GAME_DIR"
find "$TEXTWORLD_GAME_DIR" -maxdepth 1 -type f -name "*.z8" | wc -l
```

The final command must report a non-zero number. Training uses the `train`
split; switch the environment variable to `games/valid` or `games/test` for
evaluation. The archive also contains `.json` and `.ulx` files, but the
commands in this guide intentionally select only `*.z8`. Reserve roughly
40 GB of free disk space for the model, compressed dataset, and extracted
games.

If the model and games already exist elsewhere, skip both downloads and set
absolute paths instead:

```bash
export MODEL_PATH=/absolute/path/to/Qwen1.5-MoE-A2.7B-Chat
export TEXTWORLD_GAME_DIR=/absolute/path/to/ftwp/games/train

test -f "$MODEL_PATH/config.json"
find "$TEXTWORLD_GAME_DIR" -maxdepth 1 -type f -name "*.z8" | head
```

Keep these two variables exported in the shell used to start the Ray driver;
the commands below pass their resolved paths to trainer, inference, and
rollout actors.

## 3. Run the Local Trainer Smoke Test

Start with the multi-trainer smoke test in `local_trainer.py`. It does not use TextWorld, vLLM, or rollout workers. It only validates tokenizer/model loading, response-only labels, Ray FSDP multi-trainer initialization, forward/backward, and optimizer steps.

The example below starts 2 FSDP trainers and needs at least 2 visible GPUs.
Each trainer first loads the complete model onto one GPU and only then shards
the Transformer layers and root model with FSDP, so each trainer GPU must also
fit the unsharded model initialization peak:

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

Success criteria: all Ray FSDP workers start, complete a few optimizer steps, produce finite loss values, and report no tokenizer/model/FSDP initialization errors.

## 4. Run Local TextWorld Inference

Next, run `textworld_local_infer.py` to validate only vLLM inference and environment interaction:

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
  --vllm-max-num-batched-tokens 2048 \
  --verbose
```

Success criteria: episodes run until completion or the step limit. With
`--verbose`, logs show prompts, model outputs, parsed actions, selected actions,
scores, and terminal states; the episode summary reports the final score and
invalid-action count.

## 5. Run the Current End-to-End Training Configuration

The full framework uses both FSDP training GPUs and vLLM inference GPUs:

```text
total GPUs >= fsdp_world_size + infer_tp_size * infer_size
```

The commonly used configuration below uses 6 FSDP trainer GPUs and one vLLM
inference GPU, so at least 7 visible GPUs are required. It trains the full
Qwen-MoE model with GRPO advantages, PPO-style policy clipping, FlashAttention
packed training, and response-only LM-head projection:

Unlike the separate local-trainer smoke test, the TextWorld trainer loads FP32
policy storage on CPU and lets FSDP2 move and shard it. Forward/backward uses
BF16, while gradients and AdamW states remain FP32. The verified 27 GiB
BF16 checkpoint has roughly 14.3B parameters, so 3-way FP32 optimizer sharding
is too small for typical 80 GiB GPUs once gradients, moments, and temporary
buffers are included. Each trainer rank also holds a roughly 53 GiB FP32 CPU
policy copy (about 320 GiB across six ranks), excluding loading overhead.
Startup enforces the configured GPU static-memory limit.

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
  --fsdp-world-size 6 \
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
  --sync-every-optimizer-steps 1 \
  --clip-mode ppo \
  --trust-remote-code \
  --replay-capacity 256 \
  --min-replay-size-per-rank 32 \
  --rl-algorithm grpo \
  --train-token-budget 16384 \
  --train-pack-candidate-pool-size 64 \
  --dtype bfloat16
```

Packed training argument rules:

- `--train-max-sequences-per-pack` is the maximum number of independent `RLSample` objects in
  one pack, rather than a padded tensor batch dimension.
- `--train-token-budget` is the maximum total number of real tokens in a pack
  and must be at least `--max-length`.
- Replay stores independent samples; only the trainer constructs packed
  micro-batches.
- `response_only_lm_head` uses model-native
  `logits_to_keep=prediction_indices`.

This is a long-running configuration: `--max-steps` counts optimizer steps,
and no `--max-sync-rounds` limit is set. For a short end-to-end check, append
`--max-sync-rounds 1 --max-steps 2`, lower
`--min-replay-size-per-rank`, and use a smaller `--tw-game-limit`.

Success criteria:

- All 3 FSDP ranks and the vLLM actor initialize on separate GPUs.
- Rollout workers keep producing samples and each replay shard reaches its
  minimum size.
- `Train/PackTokenUtilization` becomes non-zero.
- The trainer completes optimizer steps and vLLM receives weight updates.
- TensorBoard event files appear under `runs/TextWorld_FSDP/<timestamp>`.

Open TensorBoard:

```bash
tensorboard --logdir runs/TextWorld_FSDP
```

## 6. Scale the Experiment

After the smoke test passes, scale one dimension at a time:

1. Increase `--tw-game-limit` and `--tw-max-episode-steps`.
2. Increase `--rollout-batch-size` and `--num-rollout-workers`.
3. Tune `--train-token-budget`, `--train-max-sequences-per-pack`, and
   `--train-pack-candidate-pool-size` together.
4. Increase `--grad-accum-steps` if more effective batch size is needed.
5. Tune `--sync-every-optimizer-steps` and `--replay-capacity` to control sample staleness.

Watch these metrics first:

| Metric | Purpose |
| --- | --- |
| `TextWorld/InvalidActionRate` | Checks whether the model/parser produces valid commands. |
| `Replay/FillRatio` | Checks whether rollout keeps the trainer fed. |
| `Replay/TrainSampleTrainerVersionLagMean` | Checks whether training samples are too stale. |
| `Train/PolicyLoss` | Checks policy-optimization stability. |
| `Train/PackTokenUtilization` | Checks how much of the packed token budget is used. |
| `KL/OldNewK3TrajectoryMean` | Checks whether policy updates are too large. |
| `Infer/TokensPerSec` | Checks vLLM throughput. |
| `Sync/ElapsedSeconds` | Checks whether weight sync is a bottleneck. |

## 7. Save Checkpoints

Enable HuggingFace-format checkpoint saving:

```bash
--save-checkpoint
```

Default path:

```text
<log-dir>/checkpoints/latest
```

Periodic saving:

```bash
--save-checkpoint --checkpoint-every-sync-rounds 5
```

Periodic and final saves both overwrite `latest`, so only the newest model is
retained.

The current checkpoint contains policy weights, config, tokenizer files, and
`trainer_state.json`. PPO checkpoints additionally contain Value Head weights
under `critic/`; GRPO checkpoints do not create a critic. Checkpoints do not
include optimizer state or the replay buffer, so they can restore model
parameters but are not full training-resume checkpoints.

## 8. Change Models

The TextWorld trainer always performs full-policy training.

For non-Qwen or non-Qwen-MoE style models, carefully check:

- `build_tokenizer()`
- `build_model()`
- `configure_full_training()`
- The loop that applies `fully_shard(layer)` to every
  `model.model.layers` entry, then shards the PPO Value Head when present and
  finally the root model
- `iter_vllm_loadable_weights()`

Common issues:

- Transformer layers are not under `model.model.layers`.
- The output head is not named `lm_head`.
- HuggingFace parameter names do not match the names expected by the vLLM loader.
- The tokenizer does not have a pad token. When it implements
  `apply_chat_template`, the history transcript path specifically requires
  Qwen `<|im_start|>` and `<|im_end|>` markers.

## 9. Change Tasks

For non-TextWorld tasks, prefer creating a new rollout actor instead of editing all TextWorld-specific logic inside the existing class.

The main system boundaries to keep are:

```python
result = await infer_actor.request_generation.remote(input_ids, infer_max_tokens)
replay_buffer.add_samples.remote(samples)
```

The new rollout actor should:

1. Build model input tokens.
2. Call vLLM generation.
3. Parse the model output into a task action or answer.
4. Compute task reward.
5. Build valid `RLSample` objects.
6. Write samples to the replay buffer.

The TextWorld functions most commonly replaced are:

- `TEXTWORLD_SYSTEM_PROMPT`
- `format_textworld_user_content()`
- `format_textworld_prompt()`
- `parse_model_action()`
- `_compute_step_reward()`
- `_build_textworld_episode_rl_sample()`
- `_compute_grpo_group_advantages()`

## 10. Check RLSample Alignment

This is the most important stability check. Every sample must guarantee:

- PPO response spans are ordered, non-overlapping, and never include input
  position zero; labels are derived from the corresponding `input_ids`.
- PPO response logprobs and rewards align with the flattened response spans.
- PPO stores one non-negative maximum behavior version and one boundary kind;
  the trainer derives the boundary mask on the final response token.
- Aborted or non-trainable outputs may stay in `input_ids` outside the spans.
- Truncated PPO samples end with non-response final-state context; the trainer
  uses its final token as the bootstrap prediction position. TextWorld
  `step_limit` and `history_limit`
  samples are terminal failures and do not bootstrap.
- GRPO stores compact `response_spans`, response-aligned `response_logprobs`,
  one trajectory-level `advantage`, and the maximum response-token
  `behavior_version`. Labels are derived from `input_ids`; full-length labels,
  logprobs, and per-token versions are not stored.
- Rollout samples do not exceed `--tw-history-token-window`, and argument
  validation requires `--max-length >= --tw-history-token-window`.

## Troubleshooting

### The trainer keeps waiting for replay

- Confirm `--num-rollout-workers >= --fsdp-world-size`.
- Lower `--min-replay-size-per-rank`.
- Check `TextWorld/InvalidActionRate`.
- Confirm `--tw-game-dir` and `--tw-game-pattern` match real `.z8` files.

### Invalid action rate is high

- Lower the temperature.
- Reduce `--infer-max-tokens` or `--max-action-tokens`.
- Re-run `textworld_local_infer.py` with `--verbose`; the distributed training
  launcher has no `--verbose` flag.
- Confirm that the parser matches the output format.
- Confirm that admissible commands are fully included in the prompt.

### vLLM weight sync fails

- Check GPU count using the formula above.
- Check the NCCL environment and node communication.
- Confirm that the vLLM version supports the current weight-transfer API.
- Check names, shapes, and dtypes emitted by `iter_vllm_loadable_weights()`.
- Confirm that the full policy metadata contains the expected parameters.

### Loss or KL is unstable

- Lower the learning rate.
- Reduce `--sync-every-optimizer-steps` or `--replay-capacity` to reduce sample staleness.
- PPO uses packed training, on-the-fly current values, and token TD(λ).
  `--ppo-advantage-normalization optimizer_window` is the default exact
  full-window mode. Use `ema_rms` or `ema_zscore` for historical-moment,
  single-forward steps, or `none` to disable normalization.
- Increase `--old-new-kl-coef`.
- Confirm that invalid, aborted, or empty outputs are not mistakenly marked as trainable tokens.

### FP32 FSDP static-memory preflight fails

- Increase `--fsdp-world-size`; the TextWorld trainer keeps policy shards,
  gradients, and both AdamW moments in FP32.
- Treat `--fsdp-static-memory-fraction-limit` as a capacity guard, not a way to
  reclaim memory. Raise it only after measuring activation and optimizer peaks.
- The first implementation does not automatically offload optimizer state.

### BF16 training reports non-finite gradients

- BF16 does not use loss scaling or recoverable overflow skips. Non-finite
  gradients fail the optimizer step immediately.
- Check the loss, rewards, advantages, learning rate, and input data before
  restarting the run.

### Packed training or FlashAttention initialization fails

- Confirm that `flash_attn` imports in the same environment used by Ray
  trainers.
- Use `bfloat16`; every trainer GPU must support native BF16.
- Confirm that the model supports Transformers `flash_attention_2`.
- Confirm that the CausalLM forward accepts tensor `logits_to_keep`; this is verified with
  Transformers 5.12.1 Qwen/Qwen-MoE. PPO and GRPO both require this API.
