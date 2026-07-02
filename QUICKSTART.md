# Quickstart

这份文档用于从新 checkout 的仓库跑到一个最小端到端 TextWorld 训练闭环，并给出换模型、换任务时的检查顺序。

所有命令默认从仓库根目录执行：

```bash
cd AcceRL-Agent
```

先设置两个路径：

```bash
export MODEL_PATH=<LOCAL_HF_MODEL_PATH>
export TEXTWORLD_GAME_DIR=<TEXTWORLD_Z8_GAME_DIR>
```

`MODEL_PATH` 指向本地 HuggingFace Causal LM 模型目录。`TEXTWORLD_GAME_DIR` 指向包含 TextWorld `.z8` 文件的目录。

## 1. 安装依赖

建议先创建 conda 环境：

```bash
conda create -n accerl-agent python=3.10 -y
conda activate accerl-agent
python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt

python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__)"
```

当前 `requirements.txt` 锁定了 `vllm==0.21.0`、`torch==2.11.0`。vLLM 和 PyTorch/CUDA 版本强相关；如果集群已经提供 PyTorch 或 vLLM 模块，优先使用集群环境中验证过的版本，并同步调整 `requirements.txt` 里的 `torch`/`vllm` 约束。

## 2. 跑本地 Trainer Smoke Test

先跑 `local_trainer.py` 的多 trainer smoke test。它不接 TextWorld、不接 vLLM、不接 rollout worker，只验证 tokenizer/model 加载、response-only labels、Ray FSDP 多 trainer 初始化、forward/backward 和 optimizer step。

下面示例启动 2 个 FSDP trainer，需要至少 2 张可见 GPU：

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

验收标准：多个 Ray FSDP worker 都能启动，完成几个 optimizer steps，loss 是有限值，没有 tokenizer/model/FSDP 初始化错误。

## 3. 跑 TextWorld 本地推理

再跑 `textworld_local_infer.py`，只验证 vLLM 推理和环境交互：

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

验收标准：episode 能正常运行到结束或达到 step limit，日志中能看到 observation、model output、parsed action、reward 和 done。

## 4. 跑最小端到端 RL 闭环

完整框架会同时占用 FSDP 训练 GPU 和 vLLM 推理 GPU：

```text
total GPUs >= fsdp_world_size + infer_tp_size * infer_size
```

下面命令使用 1 张 FSDP GPU 和 1 张 vLLM GPU，适合 smoke test：

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

验收标准：

- rollout worker 能持续产出样本。
- replay buffer 非空。
- trainer 能执行 optimizer step。
- vLLM 能收到初始全量同步和后续 trainable 权重同步。
- `runs/TextWorld_FSDP/<timestamp>` 下出现 TensorBoard event 文件。

查看指标：

```bash
tensorboard --logdir runs/TextWorld_FSDP
```

## 5. 放大实验

smoke test 通过后，建议一次只放大一个维度：

1. 增大 `--tw-game-limit` 和 `--tw-max-episode-steps`。
2. 增大 `--rollout-batch-size` 和 `--num-rollout-workers`。
3. 增大 `--batch-size` 或 `--grad-accum-steps`。
4. 从 `--train-mode lm_head` 切到 `last_layer`，最后再切到 `full`。
5. 调整 `--sync-every-optimizer-steps` 和 `--replay-capacity` 控制样本滞后。

优先观察这些指标：

| 指标 | 用途 |
| --- | --- |
| `TextWorld/InvalidActionRate` | 判断模型/parser 是否产生有效命令。 |
| `Replay/FillRatio` | 判断 rollout 是否能喂饱 trainer。 |
| `Replay/TrainSampleTrainerVersionLagMean` | 判断训练样本是否太旧。 |
| `Train/LossMeanAcrossRanks` | 判断训练是否稳定。 |
| `KL/OldNewK3TokenMean` | 判断 policy update 是否过大。 |
| `Infer/TokensPerSec` | 判断 vLLM 吞吐是否正常。 |
| `Sync/ElapsedSeconds` | 判断权重同步是否成为瓶颈。 |

## 6. 保存 Checkpoint

启用 HuggingFace 格式 checkpoint：

```bash
--save-checkpoint
```

默认路径：

```text
<log-dir>/checkpoints/latest
```

周期性保存：

```bash
--save-checkpoint --checkpoint-every-sync-rounds 5
```

默认情况下，周期保存和最终保存都会覆盖 `latest`。如果想保留每个 step 的独立目录，设置：

```bash
--checkpoint-name ""
```

当前 checkpoint 包含模型权重、config、tokenizer 文件和 `trainer_state.json`，不包含 optimizer state 或 replay buffer，因此主要用于推理/评估，不是完整 resume-training checkpoint。

## 7. 换模型

换模型时，先从最小训练范围开始：

1. `--train-mode lm_head`
2. `--train-mode last_layer`
3. `--train-mode full`

非 Qwen/Qwen-MoE 风格模型需要重点检查：

- `build_tokenizer()`
- `build_model()`
- `configure_trainable_parameters()`
- `FSDPTrainWorker.__init__()` 中的 `fully_shard(model.model.layers)` 路径
- `iter_vllm_loadable_weights()`

常见问题：

- transformer layers 不在 `model.model.layers`。
- 输出头不叫 `lm_head`。
- HuggingFace 参数名和 vLLM loader 期望的名字不一致。
- tokenizer 没有合适的 chat template 或 pad token。

## 8. 换任务

如果不是 TextWorld，建议新建一个 rollout actor，而不是在原类里硬改所有 TextWorld 逻辑。

需要保留的系统边界主要是：

```python
result = await infer_actor.request_batch.remote(...)
replay_buffer.add_samples.remote(samples)
```

新的 rollout actor 需要完成：

1. 构造模型输入 token。
2. 调用 vLLM generation。
3. 把模型输出解析成任务动作或答案。
4. 计算 task reward。
5. 构造合法的 `RLSample`。
6. 写入 replay buffer。

TextWorld 任务里最常替换的函数：

- `TEXTWORLD_SYSTEM_PROMPT`
- `format_textworld_user_content()`
- `format_textworld_prompt()`
- `parse_model_action()`
- `_compute_step_reward()`
- `_build_textworld_episode_rl_sample()`
- `_compute_token_level_advantages()`
- `_compute_grpo_group_advantages()`

## 9. 检查 RLSample 对齐

这是最重要的稳定性检查。每条样本都要保证：

- prompt token 的 label 是 `-100`。
- 参与训练的 response token 的 label 等于 token id。
- 参与训练的 response token 有 old-policy logprob。
- abort 或不训练的输出可以保留在 `input_ids`，但 label 必须是 `-100`。
- `response_ids` 正好包含所有 `labels != -100` 的 token。
- `output_versions` 和 `response_ids` 等长。
- 总长度不超过 `--max-length` 和 `--tw-history-token-window`。

## 常见问题

### trainer 一直等 replay

- 确认 `--num-rollout-workers >= --fsdp-world-size`。
- 降低 `--min-replay-size-per-rank`。
- 查看 `TextWorld/InvalidActionRate`。
- 确认 `--tw-game-dir` 和 `--tw-game-pattern` 能匹配真实 `.z8` 文件。

### invalid action rate 很高

- 降低 temperature。
- 减小 `--infer-max-tokens` 或 `--max-action-tokens`。
- 打开更详细的推理日志。
- 确认 parser 和输出格式一致。
- 确认 admissible commands 已完整放进 prompt。

### vLLM 权重同步失败

- 按上面的公式检查 GPU 数量。
- 检查 NCCL 环境和节点通信。
- 确认 vLLM 版本支持当前 weight-transfer API。
- 检查 `iter_vllm_loadable_weights()` 输出的名字、shape 和 dtype。
- 确认当前 trainable 参数集合不是空的。

### loss 或 KL 不稳定

- 降低学习率。
- 减小 `--sync-every-optimizer-steps` 或 `--replay-capacity`，减少样本滞后。
- 尝试 `--ppo-normalize-advantages`。
- 增大 `--old-new-kl-coef`。
- 确认非法、abort 或空输出没有被错误地设成可训练 token。
