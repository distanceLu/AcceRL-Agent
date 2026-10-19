# Partial rollout 自适应续写：CPU 最小原型

这个目录验证前文讨论的决策机制，不改动 AcceRL 的训练主循环。只依赖标准库和当前环境已有的 PyTorch，不下载模型、不安装依赖、不需要 CUDA。

## 运行

从 `/mnt/data/lcx4/AcceRL-Agent` 运行：

```bash
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m unittest test.test_partial_rollout -v
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m test.run_partial_rollout_demo
```

改变风险与时间的折算权重，观察动作如何变化：

```bash
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m test.run_partial_rollout_demo --risk-weight 0 --output-dir test/demo_compute_only
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m test.run_partial_rollout_demo --risk-weight 10 --output-dir test/demo_risk_sensitive
```

默认输出位于 `test/demo_output/`：

- `report.md`：固定策略、自适应策略和离线 oracle 的对比，以及逐请求选择。
- `decisions.json`：所有候选的可行性、拒绝原因、预测成本、风险和打分。
- `results.csv`：独立评估样本上的实际 toy KL、合成成本和代理目标。
- `calibration.json`：校准模型产生的实测条件 KL。
- `summary.json`：汇总、随机种子、实际 CPU 审计耗时和实验边界。

重复执行默认命令会覆盖该输出目录内的同名结果。调度、toy KL 和打分可复现；CPU 耗时会变化。无需运行其他需要 GPU/Ray/vLLM 的项目测试。

## 三个动作

| 动作 | 权重 | 前缀 KV | 约束 |
|---|---|---|---|
| A `old_weights_old_kv` | 旧版本 | 匹配旧版本 | 旧模型和缓存必须存在，版本差和引擎退役期限允许 |
| B `new_weights_old_kv` | 新版本 | 原有旧缓存 | 缓存存在、年龄未超预算、风险有校准数据 |
| C `new_weights_rebuild_kv` | 新版本 | 完整重新 prefill | 前缀长度必须在成本 profile 范围内 |

旧模型不可用、缓存被淘汰、版本过旧、p90 完成时间超过旧引擎期限时，相应动作不可选。混合版本 KV 不能伪装成匹配旧快照的 KV；`cache_version` 表示最早来源版本，`cache_is_mixed` 需要调用方提供。风险校准 cohort 少于两条记录时不猜测为零，A/B 被禁用，回退 C。前缀分布偏差超过预算时所有动作均拒绝：重建 KV 不能改变旧策略已经生成的文字。

`decide()` 比较单个请求的候选路线，假定各路线引擎已经可用；不包含模型加载、KV 迁移的实现。`decide_engine()` 给同一组请求选一个共同动作，防止假设单个引擎可以同时使用不同版本权重。它是串行成本相加的最小分组近似，不是支持连续批处理的全局调度器。

## 长度、风险和成本

**剩余长度**：`RemainingLengthEstimator` 用 Kaplan–Meier 生存曲线，条件化于已经生成 L 个 token 且尚未结束。中断和 token 上限作为右删失处理，不把它们当 EOS。输出剩余长度的受预算限制均值和 p90。未知任务、已无支持样本时返回完整剩余预算并显式标记 `cold_start`。只有删失尾部时会保守保留生存概率，不能凭空推断何时结束。

**风险校准**：`RiskTable` 按版本差、前缀长度分桶，用独立校准模型产生的 KL 均值和标准误估计候选风险。默认风险代理是 `mean + standard_error`；它只是保守启发式，不是统计保证的置信上界。版本差只用于分组，不被当作真正 KL；未知 cohort 不自动外推。

**成本 profile**：读取 `demo_traces.json` 中单调的 prefill 测量点，区间内线性插值；超出 profile 范围报错，防止静默外推。decode 使用每 token 时间，另计候选引擎排队成本和旧版本保留成本。

打分：

```text
J = compute_seconds + queue_seconds
    + risk_seconds_per_nat * remaining_mean * estimated_per_token_KL
    + old_version_retention_cost
```

C 的额外旧参数/KV 风险为零，只表示“同一既有前缀上的新模型一致性”，不表示整个轨迹 on-policy。既有前缀偏差通过独立的 `prefix_shift` 输入及预算检查处理；原型没有实现这个量的在线估计，默认 demo 为零。

当 B/C 的 decode、队列成本相同时，理论切换点为：

```text
remaining * risk_weight * (risk_B - risk_C) > prefill_cost
```

测试在阈值两侧独立构造样例，检查是否按预期从 B 切换至 C。其他测试覆盖不同偏好请求的共同引擎决策、未知风险、过期模型、KV 丢失和 prefix gate。

## KV 实验真正做了什么

`toy_cache.py` 是一个确定种子的单头因果注意力模型，有真实 Q/K/V 投影、因果 softmax、位置嵌入和输出分布。参数都在 CPU FP64 上计算。

先用旧模型采样前缀，在解码边界保留前缀最后一个 token 为 pending：

```text
cache 覆盖 prefix[:-1]
pending token = prefix[-1]

A：旧参数处理 pending token + 旧 cache
B：新参数处理 pending token + 旧 cache
C：新参数完整 prefill prefix
```

这种边界避免把旧参数算出的最后一个 logits 误当作新参数续写。此后用新参数、完整 KV 的模型采样一条公共 continuation，各候选在相同 token 历史上 teacher forcing，逐位置测量：

```text
KL(fresh_new_distribution || candidate_distribution)
```

同时记录每个候选对被选中 token 的 logprob、KV 各位置的来源版本。B 的后续 token 使用新权重，但初始旧 KV 被保留，因此缓存包含多个版本。

可验证性质：

- 同权重逐 token cached forward 与完整 causal prefill 数值一致。
- 权重不变时，A/B/C 的条件 KL 接近零。
- 指定的非零参数更新产生可测的旧参数偏差和旧 KV 偏差。
- 重建后与新参数完整前缀计算一致，缓存来源全部为新版本。
- 相同随机种子重复产生相同 audit。

## 如何避免自我验证

校准使用模型种子 11/12/13，评估使用 71/72/73。调度器先做决定，然后才运行 holdout 审计；它看不到评估样本的实际 KL 或 continuation 长度。校准窗口固定 12 token，而评估窗口有不同长度，因此预测可能出错，报告保留这部分 regret。

评估包括 24 个明确边界场景和 24 个固定种子的随机场景；随机场景的前缀长度、队列和审计长度独立于校准，保留误判结果。评估的 continuation 长度是人为设置的隐藏审计长度，不是模型生成 EOS 的时间。长度模块由独立的合成完成/删失记录验证；本实验没有验证真实 LLM 的长度预测准确率。

三个固定策略与自适应策略共用可行性约束。固定策略偏好的动作不可行时，回退 C，并报告 fallback 数量。oracle 事后使用 holdout KL 和审计长度，在相同可行候选中挑最小代理目标，仅作为离线下界。自适应方法不保证胜过每个固定策略。

## 结果可以说明什么

可以说明决策逻辑可执行、解析阈值成立、KV 不一致在真实注意力运算中可测、风险/成本变化会影响选择，以及估计误差如何产生代理目标 regret。

**不能用结果宣称 GPU 加速比、真实任务奖励提升或 RL 收敛更好。** prefill/decode/queue 时间及长度历史全部是显式合成数据；实测部分仅为 toy 模型条件 KL 和 CPU 审计耗时。公共 teacher-forced continuation 也不是各候选自行生成的轨迹分布，KL 不是整条轨迹的分布 KL。

未实现：在线 Trainer/rollout 集成、真正模型多版本驻留、KV 导出/迁移、持续 batching 成本、全局资源调度、真实多轮环境恢复、前缀分布校正、token 策略损失或 V-trace。未来接入时应替换成本 profile、长度历史和风险校准数据，先保留本目录的数值与可行性测试，再验证真实任务的 time-to-reward。

## 文件

| 文件 | 用途 |
|---|---|
| `partial_rollout.py` | 估计器、成本插值、约束及 A/B/C 打分 |
| `toy_cache.py` | CPU 注意力、KV 与条件分布审计 |
| `run_partial_rollout_demo.py` | 独立校准/评估、固定策略对比、结果导出 |
| `demo_traces.json` | 可替换的合成成本和长度记录 |
| `test_partial_rollout.py` | 有意义的边界、约束和数值一致性测试 |
