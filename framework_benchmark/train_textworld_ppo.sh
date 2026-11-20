#!/usr/bin/env bash
# Slime Megatron/SGLang PPO: Qwen1.5-MoE-A2.7B-Chat + existing FTWP games.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export RAY_ADDRESS=local
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TW_CPU_WORKERS="${TW_CPU_WORKERS:-24}"
export TW_RATIO_DIAGNOSTICS="${TW_RATIO_DIAGNOSTICS:-1}"

SEED="${SEED:-42}"
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_ppo_ep3_seed${SEED}_$$"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runs/TextWorld_Slime/$RUN_NAME}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat}"
MEGATRON_CHECKPOINT="${MEGATRON_CHECKPOINT:-$SCRIPT_DIR/artifacts/conversion_v1/checkpoint}"
PROMPT_DATA="${PROMPT_DATA:-$SCRIPT_DIR/prepared_v2/train.jsonl}"
RESUME_FROM="${RESUME_FROM:-}"
export TW_EVENT_DIR="$LOG_DIR/events"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$LOG_DIR/tensorboard}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -r "$HF_CHECKPOINT/config.json" ]] || { echo "Missing HF config" >&2; exit 1; }
if [[ ! -d "$MEGATRON_CHECKPOINT" ]]; then
  # Conversion refuses an existing output directory; preserve earlier logs.
  conversion_output="$SCRIPT_DIR/artifacts/conversion_rebuilt_$(date +%Y%m%d_%H%M%S)_$$"
  {
    printf 'Missing converted Megatron checkpoint: %s\n' "$MEGATRON_CHECKPOINT"
    printf 'Set MEGATRON_CHECKPOINT to an existing initial conversion, or rebuild from HF (requires free GPUs 0–3):\n'
    printf '  cd %q\n' "$PROJECT_ROOT"
    printf '  %q -m framework_benchmark.convert_checkpoint --output %q &&\n' "$PYTHON_BIN" "$conversion_output"
    printf '  MEGATRON_CHECKPOINT=%q bash %q' "$conversion_output/checkpoint" "$SCRIPT_DIR/train_textworld_ppo.sh"
    if (( $# )); then printf ' %q' "$@"; fi
    printf '\n'
  } >&2
  exit 1
fi
[[ -r "$PROMPT_DATA" ]] || { echo "Missing frozen TextWorld prompt manifest: $PROMPT_DATA" >&2; exit 1; }
source "$SCRIPT_DIR/model_qwen15_moe.sh"

# Model runtime uses the existing Slime interpreter via launch_slime.py;
# TextWorld CPU workers use the AcceRL interpreter. No dependency installation.
# EP=3 + optimizer sharding + recomputation passed the four-GPU capacity test.
# Independent Actor/Critic alternate on GPUs 0–2 through Slime's PPO offload;
# GPU 3 runs SGLang. Do not replace this with EP=1 (first Adam step OOM).
# Adapter: 400 frozen training games, 40 attempts, 4096 history tokens,
# action length <=16, score delta + win 1.0 / lost -0.5 / invalid -0.1.
# Slime uses rollout GAE + batch advantage normalization, not AcceRL JIT/EMA.
# KL below is reference KL; AcceRL old-new-kl-coef=0.05 is not equivalent.
# Batch=192 is a starting configuration, not proven equal to AcceRL replay usage.
# Only batch=3 / one update has been GPU-tested; forward parity remains unresolved.
cmd=("$PYTHON_BIN" -m framework_benchmark.launch_slime
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$MEGATRON_CHECKPOINT"
  --prompt-data "$PROMPT_DATA"
  --input-key prompt --metadata-key metadata
  --actor-num-nodes 1 --actor-num-gpus-per-node 3
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 3
  --custom-megatron-init-path framework_benchmark.qwen2_router.install
  --custom-generate-function-path framework_benchmark.slime_adapter.generate
  --custom-rollout-log-function-path framework_benchmark.textworld_logging.log_rollout
  --custom-advantage-function-path framework_benchmark.slime_adapter.compute_advantages
  --advantage-estimator ppo
  --gamma 0.9975 --lambd 0.9873
  --eps-clip 0.1 --eps-clip-high 0.1
  --early-stop-kl "${EARLY_STOP_KL:-0.01}" # Reject the actor update on excessive microbatch K2; 0 disables.
  --normalize-advantages --kl-coef 0
  --num-critic-only-steps 0
  --num-rollout 2500                 # One actor and critic update per rollout.
  --rollout-batch-size 192            # Episodes collected per rollout, not worker count.
  --global-batch-size 192
  --num-steps-per-rollout 1
  --n-samples-per-prompt 1
  --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1
  --rollout-max-response-len 16 --rollout-max-context-len 4096
  --rollout-data-transport object-store
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0
  --use-distributed-optimizer
  --use-dynamic-batch-size --max-tokens-per-gpu 8192
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --sglang-mem-fraction-static 0.6 --sglang-max-running-requests 128
  --sglang-disable-cuda-graph --sglang-enable-metrics
  --attention-backend flash --bf16 --seed "$SEED"
  --use-tensorboard
  --save "$LOG_DIR/checkpoints" --save-interval 100 --save-total-limit 1
  --save-full-ppo-checkpoint
)
if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--load "$RESUME_FROM")
fi
cmd+=("$@")

# Preview without starting Ray or touching GPUs:
# DRY_RUN=1 bash framework_benchmark/train_textworld_ppo.sh
# Use rank-local action-token K2 filtering instead of the window early-stop guard:
# bash framework_benchmark/train_textworld_ppo.sh --token-level-filter
# K2 > 0.01 tokens contribute zero loss; original episode/batch denominators remain.
# The number of packed microbatches is dynamic, independent of the 192-episode batch.
# Small pilot (same batch/context as the successful capacity test):
# bash framework_benchmark/train_textworld_ppo.sh --num-rollout 1 --rollout-batch-size 3 --global-batch-size 3
# Additional Slime arguments override earlier defaults.
# Resume the most recent complete Actor/Critic checkpoint in a new log directory:
# RESUME_FROM=/absolute/path/to/previous/run/checkpoints bash framework_benchmark/train_textworld_ppo.sh --token-level-filter
# --num-rollout remains the total target; checkpoint 799 resumes at rollout 800.
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$LOG_DIR"
printf 'Project: %s\nGPUs: %s\nLogs: %s\nTensorBoard: %s\n' "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$LOG_DIR" "$TENSORBOARD_DIR"
printf '%q ' "${cmd[@]}" > "$LOG_DIR/launch_command.sh"
printf '\n' >> "$LOG_DIR/launch_command.sh"
"${cmd[@]}" 2>&1 | tee "$LOG_DIR/console.log"

# 默认batch级过滤，开启token级过滤参数为--token-level-filter
# bash /mnt/data/lcx4/AcceRL-Agent/framework_benchmark/train_textworld_ppo.sh --token-level-filter

# 前向一致性导致old / new 比率ratio不为1，bash /mnt/data/lcx4/AcceRL-Agent/framework_benchmark/train_textworld_ppo.sh --token-level-filter --moe-permute-fusion

# RESUME_FROM=/之前运行/checkpoints \
# bash /mnt/data/lcx4/AcceRL-Agent/framework_benchmark/train_textworld_ppo.sh --token-level-filter


# 11-09 10:50 ratio爆炸，moe专家合并前向不一致
# 11-10 18:03 batch级过滤
# 11-13 17:44 token级过滤