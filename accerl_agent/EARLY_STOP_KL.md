# AcceRL PPO 的 K2 KL 窗口保护与 token 过滤

`--early-stop-kl` 在 PPO 训练的每个 pack 上检查当前策略相对样本行为策略的
K2 近似 KL。该保护适用于 PPO、GIPO 和 SAPO 的 Actor 目标；GRPO
入口不启用此选项。命令行参数本身默认 `0`，表示关闭。

以下三个 PPO 启动脚本默认传入 `${EARLY_STOP_KL:-0.01}`：

- `train_textworld_ppo.sh`
- `train_textworld_ppo_gipo.sh`
- `train_textworld_ppo_gipo_varlenPack.sh`

可以通过环境变量或脚本尾部的同名参数覆盖阈值：

```bash
EARLY_STOP_KL=0.02 bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh
bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --early-stop-kl 0
```

尾部 CLI 参数优先于脚本中通过环境变量生成的默认参数。阈值为正时启用，
设为 `0` 关闭；阈值必须有限且非负。

## 可选的逐 token 过滤

在上述任一 PPO 脚本末尾加 `--token-level-filter`，即可切换为逐 token
过滤。`--token-filter-kl-threshold` 独立控制 token K2 阈值，默认 `0.01`：

```bash
bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
# 通过环境变量把 token 阈值设为 0.02：
TOKEN_FILTER_KL_THRESHOLD=0.02 \
  bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
# 或通过尾部 CLI 参数覆盖（优先于环境变量）：
bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh \
  --token-level-filter --token-filter-kl-threshold 0.02
# 仅预览命令：
DRY_RUN=1 TOKEN_FILTER_KL_THRESHOLD=0.02 \
  bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
```

该选项默认关闭；不传时仍执行本文后续说明的整窗口保护。传入后优先使用
token 过滤，即使脚本同时传入 `--early-stop-kl 0.01`，也不会因为有限 K2
超限而取消整窗口。三个 PPO 脚本通过 `${TOKEN_FILTER_KL_THRESHOLD:-0.01}`
传入 token 阈值，不受 `EARLY_STOP_KL` 或 `--early-stop-kl` 的数值影响。
阈值必须有限且非负；设为 `0` 只保留 logratio 恰为 `0` 的 token，
不是关闭过滤。关闭 token 过滤应省略 `--token-level-filter`；如需同时
关闭整窗口保护，再设置 `--early-stop-kl 0`。

每个训练 rank 在每个 pack 内独立屏蔽
`0.5 * (new_logprob - old_logprob)^2 > token_filter_kl_threshold`
的有效 response token。
屏蔽发生在 policy ratio 的指数计算前；这些位置不贡献 Actor policy loss，
但仍作为上下文参与前向，并保留原始、带梯度的 K3 正则：
`expm1(raw_logratio) - raw_logratio`，其中
`raw_logratio = new_logprob - old_logprob` 在屏蔽前独立保存，未 detach。
`old_new_kl` loss 使用全部原始有效 response token。正常 token、此前累积的梯度、
后续 pack 和其它 rank 的有效训练项都保留。原有轨迹内 token 分母及
全局轨迹分母不变，不按过滤后的 token 数重新平均。

Critic 继续对所有有效 response token 做价值回归。AcceRL 的 Actor 与
Critic 共享优化器，因此即使全窗口的 policy token 都被屏蔽，有限的
Critic loss 或原始 K3 正则仍可提交参数更新；过滤后的 policy 直接梯度为零，并不表示
共享骨干参数被冻结，优化器动量也可能移动 Actor 参数。若同时设置
`--value-loss-coef 0` 和 `--old-new-kl-coef 0`，且全窗口没有保留的 policy token，则跳过优化器、
学习率和 EMA，避免只有动量或权重衰减的更新。
JIT GAE、rollout GAE 及各优势归一化模式均支持此选项。

所有 rank 仍完成原有 pack 的前向、反向和通信，再统一决定是否提交更新。
训练 pack 的动作 log-ratio、原始 K3 loss、模型输出或训练目标出现非有限值时，会完成本窗口的执行并
丢弃整个联合更新，optimizer、学习率和 EMA 不推进；有限 K2 超限本身不会
触发跳步。目标预计算阶段保留原有的跨 rank 错误检查：若该阶段的目标非有限，
会在更新前报错退出。

窗口返回指标中的 `token_filter/` 字段按训练 rank 汇总：

| 指标 | 含义 |
| --- | --- |
| `action_tokens` / `kept_tokens` / `masked_tokens` | 原始、保留和屏蔽的 response token 数 |
| `masked_fraction` | 屏蔽 token 占原始 token 的比例 |
| `k2_kl` / `kept_k2_kl` | 过滤前与仅保留 token 的 K2 均值 |
| `nonfinite_tokens` | 动作 log-ratio 非有限的 token 数 |
| `invalid_microbatches` | 存在非有限 loss（含原始 K3）、其它模型输出或目标的 rank-local pack 次数 |
| `threshold` | 本次配置的 `--token-filter-kl-threshold`，默认 `0.01` |

`k2_kl` 顶层指标同样保留过滤前的均值。token 模式的 `early_stop` 为 `0`；
`optimizer_step_skipped` 记录数值保护或没有有效训练项而取消的窗口，不能将
局部 token 过滤等同于整窗口早停。同步段汇总同样提供 `token_filter/` 字段，
TensorBoard 记录在 `Train/token_filter/` 下；`Train/K2KL` 保留过滤前的 K2 均值。

所有 PPO 窗口还独立保留过滤前的 `raw_logratio = new_logprob - old_logprob`，
用同一批原始有效 response token 同时记录 `Train/K2KL` 与 `Train/K3KL`：
分别为 `0.5 * raw_logratio^2` 和 `expm1(raw_logratio) - raw_logratio`。
两者都先按 token 求和，跨 pack、训练 rank 及本同步段全部尝试窗口汇总后，
除以相同的原始 response-token 总数；包含被 token filter 屏蔽的 token、
early-stop 触发及之后的 pack，也包含最终取消更新的窗口。目标预计算不重复计数。
关闭 token filter 和 early-stop 时，仍记录这两个原始指标。

原始监控使用无梯度的 FP64 独立计算，不改变 policy 的安全占位值、过滤决定、
KL 正则或优化器更新。监控中的原始 K3 指数溢出时如实记录 `inf`，
监控本身不触发跳步；训练中带梯度的原始 K3 loss 若非有限，则按上述数值保护取消窗口。
`KL/OldNewK3TrajectoryMean` 表示参与优化的原始 K3 项：token filter 模式下
也包括全部原始有效 response token，先在轨迹内按 token 平均，再按轨迹等权平均。
对同一次有效 loss 计算，它与 `Train/K3KL` 都覆盖原始 token，但分母权重不同；
同步段的 `Train/K3KL` 还包含取消更新的窗口，因此不能直接视为同一均值。

## 检查的统计量

对 pack 中有效的 response token，定义

\[
d_t=\log\pi_\theta(a_t\mid s_t)
    -\log\mu_t(a_t\mid s_t),\qquad
k_{2,t}=\tfrac12 d_t^2.
\]

这里的旧 log probability 来自 replay 样本记录的实际行为策略；它不是固定
reference model 的概率。初始 prompt、环境 observation 等非 response
位置不计入分子或分母。

每个 pack 检查时，训练端在 `loss_pg` 中对全部训练 rank 的有效 token
平方差之和与 token 数分别归约，再计算

\[
\widehat D_{\mathrm{K2}}
=\frac{\sum_r\sum_{t\in\mathcal M_r}\tfrac12d_t^2}
       {\sum_r|\mathcal M_r|}.
\]

这是按 token 加权的全局均值，不是各 rank 均值的简单平均，也不是轨迹等权
loss 归约或 EMA 优势归一化。它检查各 rank 当前对应 pack 的联合数据，
不是等整个梯度累积窗口结束后才检查窗口平均值。

## 触发后取消整个参数更新

任一 pack 的全局 K2 严格大于阈值时，本窗口的停止状态会锁存。
在构造 ratio 的指数和 PPO/GIPO surrogate、执行对应 backward 之前，
该 pack 改用连接模型计算图的零 loss；本窗口后续 pack 也使用零 loss
完成反向执行，使所有训练 rank 保持相同的 collective 顺序。

AcceRL 的 Actor 与 Critic 共享骨干，Value Head 也在这次参数更新中。
取消窗口会清空本窗口已累积的全部梯度，包括触发之前的 pack 所产生的
骨干和 Value Head 梯度；不会保留一个部分更新。该窗口不执行
`optimizer.step()`，不会修改 Adam 状态，不推进学习率调度，也不提交 EMA
优势统计。零 loss 是用于完成被取消窗口的执行，不表示其价值回归已经拟合成功。

下一窗口开始时停止状态重新初始化。Replay 中的轨迹不会因为一次窗口被拒绝
而永久删除；后续仍可能抽到相同轨迹。Rollout 持续产生新轨迹并更新缓冲区，
但这一保护不要求下一窗口的数据全部是新生成的。

K2 是近似 KL 的批次均值，并不是每 token ratio 的硬上限。它也不替代
原有的非有限值检查；模型内部已经出现 NaN/Inf 时，现有数值检查仍可能直接报错。

## 更新次数、同步与终止条件

- `optimizer_step` 只计成功提交的参数更新；被拒绝窗口不增加它。
- `train_window_step` 计已经尝试的训练窗口，包括被拒绝的窗口。
- `--max-steps`、`--stop-after-steps` 和学习率调度仍按成功更新计数。

`--sync-every-optimizer-steps K` 的每个训练分段至多尝试 `K` 个窗口。
如果其中存在成功更新，分段结束后发布最新权重；如果全部窗口被拒绝，
则记录此次尝试后继续采样，不发布相同权重、不增加推理策略版本，也不把
零次成功更新误判成训练结束。因此，开启保护后，一个含拒绝窗口的分段
不保证恰好提交 `K` 次更新。

TensorBoard 使用窗口计数作为横轴，成功更新计数另行记录，使连续拒绝的窗口
也有独立日志位置。应结合 K2、触发标志、跳过窗口和成功提交次数解释训练曲线，
而不是把取消窗口的零 loss 当成学习进展。

| TensorBoard 指标 | 含义 |
| --- | --- |
| `Train/K2KL` | 分段全部尝试窗口的有效 response-token 加权 K2 均值 |
| `Train/K3KL` | 与 `Train/K2KL` 使用完全相同原始 token 和全局 token 分母的 K3 均值 |
| `Train/K2KLMaxMicrobatch` | 分段内各 pack 的全局 K2 均值的最大值，即阈值检查对象的最大值 |
| `Train/EarlyStopFraction` | 本分段被拒绝的窗口比例 |
| `Train/SkippedWindows` | 本分段被拒绝的窗口数量 |
| `Train/OptimizerStep` | 实际成功提交的更新总数 |

loss、value 和优势统计仅用成功窗口归约，避免拒绝窗口稀释指标；采样量、
pack 使用量和吞吐统计包含所有尝试。控制台的 `early_stop=1` 表示该窗口
没有参数更新，`train_window_step` 与 `optimizer_step` 分别显示尝试和提交计数。

如果所有窗口持续被拒绝，成功更新上限本身不会结束训练；训练会继续从
replay 采样。已配置的研究运行 wall-time/GPU 时间预算仍可终止运行。

## 安全预览

变长打包入口提供仅打印命令的 dry run，不会启动 Python、Ray 或 GPU，
也不会创建日志目录：

```bash
DRY_RUN=1 EARLY_STOP_KL=0.02 \
  bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh
```

原有两个 PPO 脚本不提供 `DRY_RUN` 分支；不要将该环境变量视为它们的预览保护。
