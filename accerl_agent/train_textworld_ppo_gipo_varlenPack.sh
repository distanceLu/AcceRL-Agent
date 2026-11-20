#!/usr/bin/env bash
# PPO/GIPO with a fixed global trajectory batch and variable-length packs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export RAY_ADDRESS=local
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_ppo_gipo_varlenPack_gbs192_seed42_ctx4096_t1.0_ema_$$"
LOG_DIR="$PROJECT_ROOT/runs/TextWorld_FSDP/$RUN_NAME"
RAY_TEMP_DIR="$PROJECT_ROOT/ray_tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"

cmd=("$PYTHON_BIN" -m accerl_agent.run_agent_textworld
  --model-path /mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat
  --tw-game-dir /mnt/data/lcx4/AcceRL/accerl_vllm/data/tw_games/ftwp/games/train
  --tw-game-pattern '*.z8'
  --tw-game-limit 400
  --tw-max-episode-steps 40
  --tw-history-token-window 4096
  --tw-win-bonus 1.0
  --tw-lost-penalty 0.5
  --tw-invalid-action-penalty 0.1
  --max-length 4096
  --fsdp-world-size 3
  --infer-size 1
  --infer-tp-size 1
  --num-rollout-workers 15
  --rollout-batch-size 8
  --infer-max-tokens 16
  --infer-temperature 1
  --infer-top-p 1.0
  --vllm-max-model-len 8192
  --vllm-max-num-seqs 128
  --vllm-max-num-batched-tokens 8192
  --global-batch-size 192             # Trajectory uses per optimizer update across all training ranks.
  --train-max-sequences-per-pack 0    # No sequence-count cap; the token budget still applies.
  --train-token-budget 8192
  --replay-capacity 64
  --min-replay-size-per-rank 64
  --replay-sample-timeout-seconds 0.0
  --replay-wait-sleep-seconds 0.2
  --max-steps 500000
  --stop-after-steps 2500
  --learning-rate 1e-6
  --lr-warmup-steps 500
  --weight-decay 0.0
  --sync-every-optimizer-steps 1
  --rl-algorithm ppo
  --ppo-gae-mode jit
  --gae-gamma 0.9975
  --gae-lambda 0.9873
  --value-loss-coef 0.25
  --ppo-advantage-normalization ema_zscore
  --clip-mode gipo
  --gipo-sigma 0.3
  --clip-eps 0.1
  --early-stop-kl "${EARLY_STOP_KL:-0.01}"
  --token-filter-kl-threshold "${TOKEN_FILTER_KL_THRESHOLD:-0.01}"
  --old-new-kl-coef 0.05
  --dtype bfloat16
  --seed 42
  --metrics-window-size 192
  --metrics-active-timeout-seconds 600
  --log-dir "$LOG_DIR"
  --trust-remote-code
  --ray-temp-dir "$RAY_TEMP_DIR"
  "$@"
)

# Preview without creating output directories, starting Python, or using GPUs.
# Trailing CLI arguments override earlier defaults.
# Add --token-level-filter to mask policy terms above TOKEN_FILTER_KL_THRESHOLD; raw differentiable K3 still covers all response tokens.
# DRY_RUN=1 bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
# TOKEN_FILTER_KL_THRESHOLD=0.02 bash accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ -r /mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat/config.json ]] || { echo "Model config is not readable" >&2; exit 1; }
compgen -G "/mnt/data/lcx4/AcceRL/accerl_vllm/data/tw_games/ftwp/games/train/*.z8" >/dev/null || { echo "No readable TextWorld game directory" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$RAY_TEMP_DIR" "$XDG_CACHE_HOME"

printf "Project: %s\nGPUs: %s\nLog directory: %s\n" "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$LOG_DIR"
printf "%q " "${cmd[@]}" > "$LOG_DIR/launch_command.sh"
printf "\n" >> "$LOG_DIR/launch_command.sh"
"${cmd[@]}" 2>&1 | tee "$LOG_DIR/console.log"

# bash /mnt/data/lcx4/AcceRL-Agent/accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --token-level-filter

# bash /mnt/data/lcx4/AcceRL-Agent/accerl_agent/train_textworld_ppo_gipo_varlenPack.sh --early-stop-kl 0 不启用过滤 batch/token级

# bash /mnt/data/lcx4/AcceRL-Agent/accerl_agent/train_textworld_ppo_gipo_varlenPack.sh \
#  --token-level-filter \
#  --token-filter-kl-threshold 0.02




# 11-16 17:17 token kl 0.01 过滤后k3算kl_loss
# 11-17 11:09 无batch/token 过滤
# 11-18 10:41 优化rollout策略，减缓震荡
# 11-19 10:55 过滤前k3，token kl 0.02