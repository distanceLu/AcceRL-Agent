# JIT GAE + EMA 联合消融

默认 `--ppo-gae-mode jit` 保留原有实现和优势归一化选项。

在原启动命令末尾增加 `--ppo-gae-mode rollout`，启用联合消融：

```bash
bash accerl_agent/train_textworld_ppo.sh --ppo-gae-mode rollout
# 使用 GIPO objective 时：
bash accerl_agent/train_textworld_ppo_gipo.sh --ppo-gae-mode rollout
```

该开关会将 `ppo_advantage_normalization` 设为 `optimizer_window`，包括原脚本
已经传入 `ema_zscore` 的情况。最终生效的配置记录在运行参数 JSON 和启动日志中。
原启动脚本的目录名称仍沿用其命名规则；比较实验时以保存的参数为准。

## 数据与更新语义

1. rollout worker 完成轨迹后，请求 vLLM 对完整轨迹做 value 推理。
   推理使用最近一次同步的共享 backbone 和配套的 FP32 Value Head，
   不增加另一份 backbone。每次同步均从同一 optimizer step 收集这两部分权重。
2. value 请求使用独立的 prefix-cache salt，重新计算完整上下文。
   分块 prefill 的所有块使用同一 critic 快照。同步前停止接收新 value 请求，
   等待已经接收的请求全部完成，然后才暂停推理、更新权重。
3. rollout worker 在 CPU 上调用原有 GAE kernel。response 位置 `p` 使用
   hidden state `p-1` 的 value。prompt、observation、abort token 仅作为上下文，
   不加入 GAE 时间轴。保留原有奖励、gamma、lambda、terminated/truncated 语义，
   包括 `step_limit` 和 `history_limit` 不 bootstrap 的现有行为。
4. replay 保存 response-token 对齐的 `raw_advantages`、`returns` 和
   `value_version`（critic 快照的 optimizer step）。`returns = A + V_snapshot`。
   再次抽样时保持这些数据不变；原有 behavior logprob 不变。
5. trainer 汇总本次 optimizer step 内所有 rank、所有 pack 的有效 response
   token，以 FP64 汇总 `sum(A)`、`sum(A²)`、`count`，使用总体方差计算
   `(A - mean) / (std + ppo_adv_norm_eps)`。3 个 rank 各 32 pack 时，共统计 96 pack。
   仅 actor 使用归一化优势；critic 使用原始冻结 return。
6. trainer 仍计算当前 logprob 和当前 value 并反向传播；不运行 GAE，不进行
   target prepass，不读取或更新优势 EMA。每个 pack 只有一次训练前向。

这是同时移除 JIT GAE 和 EMA 的实验。额外的完整轨迹 value 推理会增加 rollout
成本，并可能增加同步前等待时间；比较吞吐时需要计入这部分工作。

## 验证

```bash
/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m unittest accerl_agent.test_rollout_gae -v

# 可选：使用空闲 GPU，创建临时小模型，不下载模型或安装依赖。
ACCERL_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=4 \
  /mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python -m unittest accerl_agent.test_rollout_critic_gpu -v
```

vLLM 适配器针对 `requirement.txt` 中固定的 0.24.0 版本，支持当前 eager generation
路径、chunked prefill 和 TP/DP。它不支持 pipeline parallel、DBO 或 speculative
decoding；这些不是当前训练入口启用的配置。
