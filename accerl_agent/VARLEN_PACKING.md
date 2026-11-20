# 固定 global batch 的变长打包

启动入口是 `train_textworld_ppo_gipo_varlenPack.sh`。它保留现有
`train_textworld_ppo_gipo.sh` 的 PPO/GIPO、JIT GAE、EMA 优势归一化、
3 张训练卡和 1 张推理卡配置，改用固定轨迹使用次数决定一次参数更新的数据量。

```bash
bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh
```

先检查实际命令而不启动 Python、Ray 或 GPU：

```bash
DRY_RUN=1 bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh
```

dry run 只打印命令，不创建日志、Ray 或缓存目录。

## batch 与 pack 的对应关系

新入口的主要参数为：

```text
--global-batch-size 192
--fsdp-world-size 3
--train-max-sequences-per-pack 0
--train-token-budget 8192
--replay-capacity 64
--min-replay-size-per-rank 64
```

一次 optimizer update 使用的全局轨迹次数为 `global_batch_size`。
它必须能被训练 rank 数 `fsdp_world_size` 整除；每个 rank 为当前窗口准备
`global_batch_size / fsdp_world_size` 条有效轨迹。本例为每 rank 64 条，
全局合计 192 次轨迹使用。

`--train-max-sequences-per-pack 0` 表示不限制 pack 内的序列条数，
同时仍要求 pack 中所有序列的真实 token 总数不超过 `train_token_budget`。
每条轨迹的 `max_length` 上限仍然有效。较短轨迹可以多条装入同一个 pack；
8192 是预算上限，不会把每个 pack 补齐到这个长度。

每个 rank 收齐其当前窗口的轨迹后，先按长度排序，再在 token 预算内打包。
不同 rank 的轨迹长度不同，得到的 pack 数也可能不同。训练端通过
all-reduce MAX 得到本窗口各 rank 的最大 pack 数，把其他 rank 的已有 pack
继续拆分到相同数量，从而保持 FSDP 前向、反向的执行次数一致。
拆分保留全部真实轨迹及其训练内容，不添加虚假样本，也不重复填充样本。

因此，梯度累积次数由本窗口对齐后的 pack 数决定，每个窗口都可能变化。
每个 pack 前向、反向后立即累积梯度，全部 pack 完成后更新一次参数。
新脚本不再传固定的 `--grad-accum-steps 32`，也不传旧的
`--train-pack-candidate-pool-size 2`；固定 global batch 模式使用完整的本地
轨迹窗口进行打包，而不是依赖该小候选池决定一次更新的数据量。

## 回放与训练目标

固定 batch 控制的是轨迹参与计算的次数，不保证每次更新都是新产生的轨迹，
也不保证全局 192 条轨迹互不重复。Replay 中的样本不会因采样而删除，
后续更新仍可能复用历史轨迹；容量和 warmup 条数不是无重复遍历保证。
这个模式也没有增加固定数据集的 `ppo_epoch` 循环。

JIT GAE 仍使用当前训练模型重新计算价值、优势和 return；EMA 优势统计
仍在一次成功参数更新后提交。pack 的边界不会改变轨迹内的因果注意力边界，
也不改变轨迹等权的损失归约。当前配置每更新一次参数，就向推理端发布一次权重。

新脚本还默认启用 `--early-stop-kl 0.01`。超限会丢弃整个联合更新窗口，
而不是保留触发前的部分梯度；被拒绝窗口不增加成功更新计数、不提交 EMA、
也不发布权重。详见 [K2 KL 窗口保护](EARLY_STOP_KL.md)。

可在脚本末尾添加 `--token-level-filter`，改为逐 token 过滤：

```bash
bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
# 将逐 token K2 阈值设为 0.02：
TOKEN_FILTER_KL_THRESHOLD=0.02 \
  bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
# 也可用尾部 CLI 参数覆盖：
bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh \
  --token-level-filter --token-filter-kl-threshold 0.02
```

该选项默认关闭；启用后优先于整窗口保护，每个 rank 独立屏蔽
`0.5 * (new_logprob - old_logprob)^2 > token_filter_kl_threshold` 的 Actor
policy 项，保留原有 loss 分母。`old_new_kl` 项使用过滤前、带梯度的原始
K3（`expm1(raw_logratio) - raw_logratio`），覆盖全部原始有效 response token。
阈值由独立参数
`--token-filter-kl-threshold` 控制，默认 `0.01`；脚本环境变量为
`TOKEN_FILTER_KL_THRESHOLD`，不随 `EARLY_STOP_KL` 改变。阈值须有限且非负，
`0` 只保留 logratio 恰为 `0` 的 token；省略 `--token-level-filter` 才关闭
逐 token 过滤（如需同时关闭窗口保护，再传 `--early-stop-kl 0`）。
其它 token、此前梯度和后续 pack 继续参与本次更新，
Critic 仍使用全部有效 token。由于共享优化器，即使 policy token 全部被屏蔽，
有限的 Critic loss 或原始 K3 正则仍可更新参数；非有限 loss（含原始 K3）、
模型输出或目标会取消整个联合窗口。

## 长度预算分别控制什么

| 参数 | 控制范围 |
| --- | --- |
| `--tw-history-token-window 4096` | rollout 中单条 episode 的累计输入历史预算 |
| `--max-length 4096` | 训练接收的单条轨迹长度上限 |
| `--train-token-budget 8192` | 每卡一个训练 pack 的真实 token 总数上限 |
| `--vllm-max-model-len 8192` | 推理端单条序列的最大上下文长度 |
| `--vllm-max-num-batched-tokens 8192` | vLLM 调度的 token 预算 |

训练 pack 预算和推理调度预算不要求相等。本入口沿用原脚本中的值，
不改变单条 episode 的长度限制或推理配置。

## 兼容与覆盖参数

不提供 `--global-batch-size` 的现有入口继续使用原来的固定 pack 累积模式，
即由 `--grad-accum-steps` 指定每 rank 每次更新处理多少个 pack。
新的固定 global batch 模式优先用轨迹数决定窗口大小，不能再通过调整
`grad_accum_steps` 改变其每次更新的轨迹数。

脚本尾部的额外 CLI 参数覆盖前面同名参数，例如：

```bash
DRY_RUN=1 bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh \
  --global-batch-size 96 --stop-after-steps 100
```

如需改变训练卡数，还需要协调 GPU 分配、每 rank 的 batch、replay 容量及
warmup 条数；global batch 必须仍能被训练 rank 数整除。修改 batch 时，
默认 run name 里的 `gbs192` 只是启动入口标识，以保存的 `launch_command.sh`
中的实际参数为准。

## 检查运行日志

固定 global batch 模式应同时观察以下量，而不是根据 pack 数反推样本数：

- 每次更新各 rank 及全局的有效轨迹使用次数，确认 global batch 达到配置值。
- 跨 rank 对齐后的 pack 数，了解实际梯度累积次数。
- pack 的真实 token 总量、可由轨迹数和 pack 数计算的平均轨迹数，以及 token 预算利用率。

控制台每次更新输出 `episode_uses` 和 `microbatches_per_rank`；TensorBoard
记录 `Train/EpisodeUsesPerUpdate`、`Train/MicrobatchesPerRank`、
`Train/InputTokenUsesPerUpdate` 和 `Train/PackTokenUtilization`。
若一个权重同步分段包含多次 optimizer update，以上按更新记录的量为该分段的平均值。

跨 rank 对齐可能拆分原本更满的 pack，因此最终利用率不保证每轮都提高。
这些日志反映真实样本、实际 pack 和预算消耗，不将对齐过程计成新增样本。
