# 当前状态

实验范围：AcceRL PPO + JIT GAE + EMA 对 Slime PPO；各自使用原生FSDP/vLLM与Megatron/SGLang。
没有运行正式学习曲线、没有比较胜率或吞吐结果。

已完成：
- 固定400 train / 222 validation / 514 test清单；跨训练种子保持集合不变。
- Slime逐动作奖励与action mask接入、DP元数据传递、Actor/Critic目标hook保留。
- 两边生成吞吐事件、AcceRL optimizer commit和Slime原生训练指标事件。
- 公共分析器与冻结HF checkpoint的SGLang评估入口（评估器尚未GPU验收）。
- 27 GiB Megatron checkpoint转换成功：artifacts/conversion_v1/checkpoint。
- 完整4,659个权重张量逐元素往返一致，见artifacts/conversion_weight_audit/weight_parity.json。
- 20项公共协议/AcceRL回归、6项Slime接口/真实TextWorld跨环境测试通过。
- 后续批量GAE与路由精度各一项定向测试通过。

前向验收仍未通过：
- 参考来自AcceRL实际Transformers/FlashAttention2环境，BF16。
- 保留预设阈值：mean KL < 0.02，最大token logprob差 < 0.2。
- 路由修正前，三个短输入的mean KL约0.01360/0.00118/0.00339，最大logprob差约1.213/0.167/0.176。
- Megatron先将softmax概率转BF16再top-k，而Qwen先FP32 top-k；qwen2_router.py修正这一顺序。
- 修正后mean KL约0.02556/0.00185/0.00593，最大差约0.440/0.170/0.264，仍不满足阈值。
- 不能将路由差异认定为唯一根因；下一步需逐层hidden-state和专家路由诊断，区分BF16执行差异与模型语义问题。

完整报告：artifacts/conversion_router_fp32/forward_parity.json。
config.json的launch_enabled保持false；不使用未通过验收的checkpoint启动正式训练。

仍需完成：模型前向一致性、统一独立评估GPU测试、
全局batch/复用/LR与KL口径、Slime训练token消费计数，以及pilot阈值与预算。
0–3卡在本轮开始时空闲；转换/前向诊断使用0卡，容量测试使用0–3卡，未操作4–7卡。
未安装或升级任何依赖。

四卡容量实测（2026-11-06；不作为质量/吞吐基准）：
- `artifacts/four_gpu_v1`：3张训练 + 1张SGLang，EP=1，分布式Adam，重计算，3条轨迹，context=4096。
- 模型初始化、首次Megatron→SGLang权重同步、TextWorld采样与Critic前后向成功。
- Critic首次Adam状态初始化OOM，Actor尚未更新；退出码1，总计268.96秒。
- 训练卡峰值141683/142103/141139 MiB，推理卡88486 MiB。
- 仅据此不能断定必须增加GPU；下一轮使用EP=3分片60个专家，仍为3+1布局。
- `artifacts/four_gpu_ep3_v1`：EP=3成功完成Critic更新，但Actor日志聚合metadata字典报TypeError；没有OOM。
- 应用`slime_metadata_logging.patch`，排除metadata的数值日志聚合。
- `artifacts/four_gpu_ep3_v2`：EP=3完整通过，退出码0，422.43秒（包含启动/清理，不能用作稳态性能）。
- TextWorld采样、Critic Adam更新、Actor PPO Adam更新、更新后SGLang权重同步均完成。
- GPU 0/1/2/3峰值分别104221/103797/102709/88488 MiB，约101.78/101.36/100.30/86.41 GiB。
- Actor loss=0.0836084、clipfrac=0.0144928；Critic value_loss=7.51525，均为单次容量测试指标。
- 结论：EP=3 + Actor/Critic轮流卸载，可在四张H20-3e上运行这个小批量PPO流程；不证明正式批量/长时间训练稳定。
- Actor一次卸载38.2秒，切换与同步成本必须计入后续端到端性能比较。
- 前向一致性门槛仍未通过，本次仅证明容量和接口闭环，不启用正式学习曲线比较。
