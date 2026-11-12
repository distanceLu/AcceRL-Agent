# Slime / AcceRL TextWorld PPO comparison

范围已经确定：四张 H20-3e（0–3），各三训练卡加一推理卡作为初始布局；
AcceRL PPO + JIT GAE + EMA，Slime 原生独立 Actor/Critic PPO，分别使用
FSDP/vLLM 与 Megatron/SGLang。种子 42/43/44 顺序运行，比较完整系统。
本目录没有自动启动训练、轮询后启动或终止现有 GPU 作业的行为。

## 当前可运行部分

```bash
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m framework_benchmark.prepare --output framework_benchmark/prepared_v1
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m unittest framework_benchmark.test_protocol -v
```

prepare 冻结 seed42 选出的400个训练游戏以及完整222 validation/514 test，
保存游戏内容哈希、两个仓库的commit/diff、当前Python源码哈希与协议。
不覆盖已有目录。不同训练seed复用同一manifest。
AcceRL增加可选 `--tw-game-manifest .../games.json`；未指定时行为不变。
源码哈希不是源码备份，正式启动前还需归档未跟踪文件并校验代码未改变。

Slime的generate hook使用持久CPU子进程调用现有AcceRL环境的TextWorld，
复用当前动作解析与prompt内容函数；不需要安装TextWorld到Slime环境。
子进程禁用CUDA，使用指定AcceRL解释器。设置 `TW_EVENT_DIR` 保存原始事件；
`TW_CPU_WORKERS` 默认24。Slime prompt-data使用生成的train.jsonl，不再套chat template。
采样token IDs和logprob直接取SGLang结果，不重新tokenize模型回答。

## PPO 接入尚需验收

`slime_adapter.compute_advantages` 按动作token计算GAE，观察token既不参与loss也不推进折扣；
逐动作奖励位于该动作末token。需要在Actor与Critic两条路径都保留该hook。
上下文并行目前限制CP=1。非终止环境边界需要bootstrap，当前明确拒绝这种样本，
不能默默把它当终止。对真实数据检查后再实现/验收该分支。

指定Slime版本将Sample.train_metadata转成metadata后，没有在DP分发中保留它。
`slime_integration.patch` 已应用到指定Slime目录，保留DP元数据与Critic的custom advantage hook，
并为Qwen2MoE启用已有的专家/共享专家权重导出映射。`slime_logging.patch` 也已应用，
通过公共事件文件记录原生训练指标。`slime_metadata_logging.patch` 跳过日志中的字典元数据数值聚合。
这些补丁均只修改指定Slime源码，没有安装依赖。
必须通过跨DP数据传输、Actor/Critic目标和一次真实更新检查后才能启用训练。

config.json是待验收配置，不是可以直接启动的完整命令。模型已成功转换为27 GiB的Megatron
checkpoint，4,659个权重张量往返检查全部逐元素相等。但完整前向误差仍需通过验收。
Qwen2MoE权重热同步和一次Actor/Critic更新已通过四卡容量测试；
全局batch/样本复用/LR schedule、行为KL定义和独立评估GPU测试尚待完成。
不把AcceRL行为KL系数直接填成Slime固定reference KL系数。
不要套用另一个slime_runtime checkout里的7卡GRPO TextWorld脚本。

## 指标协议

- 完整episode/s：按唯一episode ID计数，胜/负/动作上限/历史上限分开报告，shutdown半局除外。
- env.step/s：每次真实返回的环境transition计一次，非法动作未执行env.step不计。
- 新生成有效action tokens/s：完整且可训练轨迹的response spans去重计数。
  Replay重复消费、观察、padding、prefill及最终abort token不计；负奖励非法动作token仍计。
- 训练token消费吞吐单列，不作为新生成吞吐。同步等待、重建和更新在训练时间分母内。
- GPU小时计入全部四张分配GPU的全程墙钟，包括初始化、保存、评估与清理。
- policy/value loss、行为KL、reference KL（启用时）、clip fraction、entropy、grad norm、
  advantage统计、版本年龄、复用率、同步/训练等待使用明确字段与分母，不能按日志同名直接合并。
- 独立评估冻结权重，greedy，共用后端；validation用于pilot和周期评估，test用于终评。
  主指标宏平均score/max_score，辅指标胜率。目标阈值与最大预算在pilot后冻结。
  time-to-target采用首次实际测量达标时间，未达标记为未达到，不插值、不回填。

两边均已接入episode/env事件；AcceRL还记录成功optimizer commit与完整step metrics。
Slime记录其原生loss/KL/clip等指标，但训练token消费计数尚未接线，分析器以null表示缺失。
evaluate.py已实现共用SGLang对冻结HF checkpoint的独立评估入口，尚未完成GPU验收。
analyze.py要求明确作业与训练区间，按事件计算吞吐；不能将GPU检查与转换计入正式训练结果。

## 已执行验证

公共协议与现有rollout GAE回归共20项通过；Slime接口与跨Python环境TextWorld共6项通过。
首次完整模型转换成功，产物位于artifacts/conversion_v1/checkpoint，仅表示可转换，不表示前向验收通过。
artifacts/conversion_weight_audit/weight_parity.json记录4,659个张量全部一致。
原始前向检查未通过（原阈值mean KL<0.02、最大token logprob差<0.2），报告完整保留。
发现Megatron在FP32 softmax之后、top-k之前转BF16，而Qwen先top-k再转BF16。
qwen2_router.install为本实验保留Qwen的选择顺序，修正后的完整GPU检查另存新目录。
runtime.py修复了两套cuDNN混用的加载顺序，不修改任何已安装包。

## 四卡容量实测

常用启动脚本：`bash framework_benchmark/train_textworld_ppo.sh`。
默认EP=3、3训练+1推理、192 episodes/更新、2500次更新，每100轮保存checkpoint。
`DRY_RUN=1`仅打印命令；尾部Slime参数覆盖默认值，`SEED=43`切换种子。
日志、原始指标事件和checkpoint保存到`runs/TextWorld_Slime/`。
TensorBoard记录`textworld/train/WinRate`与`textworld/train/NormalizedScore`，按每轮选中批次的完整episode统计，
横轴沿用Slime的rollout step；胜率为0–1比例，归一化分数为逐局score/max_score的均值。
步数/历史上限结束计入，abort与缺失结果不计入；max_score=0不参与分数均值，并记录有效分母。
这是训练采样曲线，不是独立validation/test评估；原始episode事件仍保留在events/。
启动脚本默认`TW_RATIO_DIAGNOSTICS=1`，需要已应用的`slime_ratio_diagnostics.patch`。
`train/ratio_diag/`记录每次更新所有DP rank/microbatch有效action token的精确全局统计：
ratio、log_ratio、old/new/rollout logprob、advantage的均值、最小/最大、p50/p90/p99/p999。
`at_max_ratio/`各字段来自同一个最大ratio token；events/的`ppo_ratio_diagnostics`保存top20及rank/microbatch/response位置。
这里new logprob来自参与反向传播的前向，尚未执行该次optimizer.step；不是更新后另跑一次的logprob。
非有限值单独计数，存在时对应统计改用`*_finite_*`字段，不默默截断；原始top token用字符串记录inf/nan。
诊断仅支持当前TP=PP=CP=1的PPO；精确统计有CPU拷贝和通信开销，正式吞吐对比可用`TW_RATIO_DIAGNOSTICS=0`关闭。
已运行的进程不会自动加载此改动，需要下次启动生效。
脚本不会自动运行；192批量及长期训练尚未验收，不能视为已经完成公平对比配置。

启动脚本默认启用 Actor K2 KL 早停：`--early-stop-kl 0.01`。
可用 `EARLY_STOP_KL=0.02 bash framework_benchmark/train_textworld_ppo.sh` 调整，
或在脚本末尾传 `--early-stop-kl 0` 关闭。Slime 参数本身默认关闭，以保持其它任务的原有行为。
每个 microbatch 在 backward 前，先对所有普通 DP rank 的有效动作 token 归约
`sum(0.5 * (new_logprob - old_logprob)^2)` 与 token 数，再计算全局加权均值；
任一 microbatch 的全局 K2 超阈值或有效 log-ratio 非有限，就锁存 `stop_ppo=True`。
当前及后续 microbatch 返回保留模型计算图的零损失，继续完成 Megatron 反传和通信；
窗口结束后丢弃包括触发前已积累的全部 Actor 梯度，跳过 optimizer 与 LR scheduler。
该训练窗口不会重试，下一窗口重置状态；Critic 更新流程不变。
不使用 `loss * 0`，避免已有非有限 loss 污染零损失；模型内部已非有限时仍保留原有 NaN 保护。

`train/k2_kl` 是整窗口按有效动作 token 加权的均值，
`train/k2_kl_max_microbatch` 是触发检查所用的 microbatch 全局均值的最大值；
后者超限即可丢弃整窗，即使前者较小。`train/early_stop` 和
`train/optimizer_step_skipped` 为 1 表示该次 Actor 更新被丢弃，
此时有效 loss 与 grad_norm 记录为 0，不代表完成了一次正常更新。
`train/early_stop_microbatch` 记录首次触发的位置（从 1 开始）。
guard 支持当前 TP=PP=CP=1、DP/EP 配置及模型层重计算；
暂不支持 loss 函数重计算、MTP 和 TIS/mismatch metrics，启用时会明确报错。
该改动保存在 `slime_k2_early_stop.patch`，已应用到指定 Slime checkout；运行中的训练需重新启动才会加载。

EP=1在Critic首次Adam状态初始化时OOM；EP=3使用60专家分片后，3训练+1推理完整跑通。
结果见`artifacts/four_gpu_ep3_v2/result.json`和同目录console.log、gpu.jsonl、events/。
仅测试3条轨迹、4096上下文、一次Actor/Critic更新；正式训练仍需通过前向与协议验收。
复现容量测试（需要空闲GPU 0–3，输出目录必须不存在）：

```bash
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m framework_benchmark.smoke_four_gpu --expert-parallel-size 3 --output framework_benchmark/artifacts/four_gpu_new
```
