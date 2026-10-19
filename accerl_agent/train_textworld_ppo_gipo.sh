#!/usr/bin/env bash
# Reproduce the reference PPO/GIPO run on GPUs 0,1,2,3.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# This existing environment matches requirement.txt; override with PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
# Always create a local Ray cluster limited to the four selected GPUs.
export RAY_ADDRESS=local
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_ppo_gipo_seed42_ctx4096_t1.0_ema_$$"
LOG_DIR="$PROJECT_ROOT/runs/TextWorld_FSDP/$RUN_NAME"
RAY_TEMP_DIR="$PROJECT_ROOT/ray_tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ -r /mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat/config.json ]] || { echo "Model config is not readable" >&2; exit 1; }
compgen -G "/mnt/data/lcx4/AcceRL/accerl_vllm/data/tw_games/ftwp/games/train/*.z8" >/dev/null || { echo "No readable TextWorld game directory" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$RAY_TEMP_DIR" "$XDG_CACHE_HOME"

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
  --train-max-sequences-per-pack 2
  --train-token-budget 8192
  --train-pack-candidate-pool-size 2
  --grad-accum-steps 32
  --replay-capacity 64
  --min-replay-size-per-rank 32
  --replay-sample-timeout-seconds 0.0
  --replay-wait-sleep-seconds 0.2
  --max-steps 500000
  --learning-rate 1e-6
  --lr-warmup-steps 500
  --weight-decay 0.0
  --sync-every-optimizer-steps 1
  --rl-algorithm ppo
  --gae-gamma 0.9975
  --gae-lambda 0.9873
  --value-loss-coef 0.25
  --ppo-advantage-normalization ema_zscore
  --clip-mode gipo
  --gipo-sigma 0.3
  --clip-eps 0.1
  --old-new-kl-coef 0.05
  --dtype bfloat16
  --seed 42
  --metrics-window-size 1000
  --metrics-active-timeout-seconds 600
  --log-dir "$LOG_DIR"
  --trust-remote-code
  --ray-temp-dir "$RAY_TEMP_DIR"
  "$@"
)

# Extra CLI arguments are appended, e.g. --max-steps 2 for a short run.
printf "Project: %s\nGPUs: %s\nLog directory: %s\n" "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$LOG_DIR"
printf "%q " "${cmd[@]}" > "$LOG_DIR/launch_command.sh"
printf "\n" >> "$LOG_DIR/launch_command.sh"
"${cmd[@]}" 2>&1 | tee "$LOG_DIR/console.log"
