# Quickstart

This guide takes a fresh checkout to the smallest end-to-end TextWorld training loop. It also gives a recommended checklist for changing models or replacing the task.

All commands are intended to run from the repository root:

```bash
cd AcceRL-Agent
```

Set two paths first:

```bash
export MODEL_PATH=<LOCAL_HF_MODEL_PATH>
export TEXTWORLD_GAME_DIR=<TEXTWORLD_Z8_GAME_DIR>
```

`MODEL_PATH` should point to a local HuggingFace Causal LM model directory. `TEXTWORLD_GAME_DIR` should point to a directory containing TextWorld `.z8` files.

## 1. Install Dependencies

Create a conda environment first:

```bash
conda create -n accerl-agent python=3.10 -y
conda activate accerl-agent
python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt

python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__)"
```

The current `requirements.txt` pins `vllm==0.21.0` and `torch==2.11.0`. vLLM, PyTorch, and CUDA versions are tightly coupled. If your cluster already provides validated PyTorch or vLLM modules, prefer those versions and update the `torch`/`vllm` constraints in `requirements.txt` accordingly.

## 2. Run the Local Trainer Smoke Test

Start with the multi-trainer smoke test in `local_trainer.py`. It does not use TextWorld, vLLM, or rollout workers. It only validates tokenizer/model loading, response-only labels, Ray FSDP multi-trainer initialization, forward/backward, and optimizer steps.

The example below starts 2 FSDP trainers and needs at least 2 visible GPUs:

```bash
python accerl_agent/local_trainer.py \
  --model-path "$MODEL_PATH" \
  --train-mode lm_head \
  --use-fsdp \
  --fsdp-world-size 2 \
  --max-steps 5 \
  --batch-size 1 \
  --max-length 128 \
  --trust-remote-code
```

Success criteria: all Ray FSDP workers start, complete a few optimizer steps, produce finite loss values, and report no tokenizer/model/FSDP initialization errors.

## 3. Run Local TextWorld Inference

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
  --vllm-max-num-batched-tokens 2048
```

Success criteria: episodes run until completion or the step limit, and logs show observations, model outputs, parsed actions, rewards, and done states.

## 4. Run the Smallest End-to-End RL Loop

The full framework uses both FSDP training GPUs and vLLM inference GPUs:

```text
total GPUs >= fsdp_world_size + infer_tp_size * infer_size
```

The command below uses 1 FSDP GPU and 1 vLLM GPU, so it is suitable for a smoke test:

```bash
python accerl_agent/agent_textworld.py \
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
  --rollout-batch-size 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --replay-capacity 8 \
  --min-replay-size-per-rank 1 \
  --max-steps 2 \
  --max-sync-rounds 1 \
  --sync-every-optimizer-steps 1 \
  --train-mode lm_head \
  --rl-algorithm ppo \
  --clip-mode ppo \
  --trust-remote-code
```

Success criteria:

- Rollout workers keep producing samples.
- The replay buffer is non-empty.
- The trainer completes optimizer steps.
- vLLM receives the initial full sync and later trainable-weight syncs.
- TensorBoard event files appear under `runs/TextWorld_FSDP/<timestamp>`.

Open TensorBoard:

```bash
tensorboard --logdir runs/TextWorld_FSDP
```

## 5. Scale the Experiment

After the smoke test passes, scale one dimension at a time:

1. Increase `--tw-game-limit` and `--tw-max-episode-steps`.
2. Increase `--rollout-batch-size` and `--num-rollout-workers`.
3. Increase `--batch-size` or `--grad-accum-steps`.
4. Move from `--train-mode lm_head` to `last_layer`, then finally to `full`.
5. Tune `--sync-every-optimizer-steps` and `--replay-capacity` to control sample staleness.

Watch these metrics first:

| Metric | Purpose |
| --- | --- |
| `TextWorld/InvalidActionRate` | Checks whether the model/parser produces valid commands. |
| `Replay/FillRatio` | Checks whether rollout keeps the trainer fed. |
| `Replay/TrainSampleTrainerVersionLagMean` | Checks whether training samples are too stale. |
| `Train/LossMeanAcrossRanks` | Checks training stability. |
| `KL/OldNewK3TokenMean` | Checks whether policy updates are too large. |
| `Infer/TokensPerSec` | Checks vLLM throughput. |
| `Sync/ElapsedSeconds` | Checks whether weight sync is a bottleneck. |

## 6. Save Checkpoints

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

By default, periodic and final saves both overwrite `latest`. To keep independent directories for each step, set:

```bash
--checkpoint-name ""
```

The current checkpoint contains model weights, config, tokenizer files, and `trainer_state.json`. It does not include optimizer state or the replay buffer, so it is mainly intended for inference/evaluation rather than full training resume.

## 7. Change Models

When switching models, start with the smallest trainable scope:

1. `--train-mode lm_head`
2. `--train-mode last_layer`
3. `--train-mode full`

For non-Qwen or non-Qwen-MoE style models, carefully check:

- `build_tokenizer()`
- `build_model()`
- `configure_trainable_parameters()`
- The `fully_shard(model.model.layers)` path in `FSDPTrainWorker.__init__()`
- `iter_vllm_loadable_weights()`

Common issues:

- Transformer layers are not under `model.model.layers`.
- The output head is not named `lm_head`.
- HuggingFace parameter names do not match the names expected by the vLLM loader.
- The tokenizer does not have a suitable chat template or pad token.

## 8. Change Tasks

For non-TextWorld tasks, prefer creating a new rollout actor instead of editing all TextWorld-specific logic inside the existing class.

The main system boundaries to keep are:

```python
result = await infer_actor.request_batch.remote(...)
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
- `_compute_token_level_advantages()`
- `_compute_grpo_group_advantages()`

## 9. Check RLSample Alignment

This is the most important stability check. Every sample must guarantee:

- Prompt-token labels are `-100`.
- Trainable response-token labels equal the token ids.
- Trainable response tokens have old-policy logprobs.
- Aborted or non-trainable outputs may stay in `input_ids`, but their labels must be `-100`.
- `response_ids` contains exactly all tokens where `labels != -100`.
- `output_versions` has the same length as `response_ids`.
- Total sequence length does not exceed `--max-length` or `--tw-history-token-window`.

## Troubleshooting

### The trainer keeps waiting for replay

- Confirm `--num-rollout-workers >= --fsdp-world-size`.
- Lower `--min-replay-size-per-rank`.
- Check `TextWorld/InvalidActionRate`.
- Confirm `--tw-game-dir` and `--tw-game-pattern` match real `.z8` files.

### Invalid action rate is high

- Lower the temperature.
- Reduce `--infer-max-tokens` or `--max-action-tokens`.
- Enable more detailed inference logs.
- Confirm that the parser matches the output format.
- Confirm that admissible commands are fully included in the prompt.

### vLLM weight sync fails

- Check GPU count using the formula above.
- Check the NCCL environment and node communication.
- Confirm that the vLLM version supports the current weight-transfer API.
- Check names, shapes, and dtypes emitted by `iter_vllm_loadable_weights()`.
- Confirm that the current trainable parameter set is not empty.

### Loss or KL is unstable

- Lower the learning rate.
- Reduce `--sync-every-optimizer-steps` or `--replay-capacity` to reduce sample staleness.
- Try `--ppo-normalize-advantages`.
- Increase `--old-new-kl-coef`.
- Confirm that invalid, aborted, or empty outputs are not mistakenly marked as trainable tokens.
