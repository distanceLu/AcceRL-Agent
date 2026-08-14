#!/usr/bin/env bash
set -euo pipefail

if [[ -f /cpfs01/luck_workspace/software/bashrc_shared.sh ]]; then
  source /cpfs01/luck_workspace/software/bashrc_shared.sh
fi
source /cpfs01/qw_workspace/anaconda3/etc/profile.d/conda.sh
conda activate luck_312_torch_211_vllm_21_rl

cd /cpfs01/luck_workspace/repositiory/AcceRL-Agent

export LD_LIBRARY_PATH="${CONDA_PREFIX:?}/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cusparse/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cublas/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cufft/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/curand/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cusolver/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvtx/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/cpfs01/luck_workspace/repositiory/AcceRL-Agent${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/data/all/luck/cache/huggingface
export HF_HUB_CACHE=/data/all/luck/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/all/luck/cache/huggingface/transformers
export XDG_CACHE_HOME=/data/all/luck/accerl_p3/xdg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export TMPDIR=/data/all/luck/accerl_p3/tmp
export RAY_TMPDIR=/data/all/luck/accerl_p3/ray

BASE_MODEL=/data/all/luck/models/verl_rlvr/qwen3vl8b_three_frame_sft_checkpoint80
DATA_ROOT=/data/all/luck/derived_dataset/VSI_590K_derived/scannet_chair_incremental_counting/three_frame_complete_capability
TRAIN_FILE=${DATA_ROOT}/train.parquet
VAL_FILE=${DATA_ROOT}/test.parquet
OUTPUT_DIR="/data/all/luck/runs/AcceRL-Agent/three_frame/P3/$(date +%Y%m%d_%H%M%S)_accerl_qwen3vl8b"

for required_path in "${BASE_MODEL}" "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "required P3 input does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory is not empty: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p \
  "${OUTPUT_DIR}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${RAY_TMPDIR}"

echo "output_dir=${OUTPUT_DIR}"
echo "train_file=${TRAIN_FILE}"
echo "validation_file=${VAL_FILE}"
echo "validation_note=AcceRL currently trains from TRAIN_FILE; VAL_FILE is reserved for the validation workflow."

python -m vsi_qa_rlvr.main \
  --model-path "${BASE_MODEL}" \
  --dtype bfloat16 \
  --train-mode full \
  --trust-remote-code \
  --data-path "${TRAIN_FILE}" \
  --reward-type p3 \
  --limit-images 3 \
  --log-dir "${OUTPUT_DIR}" \
  --fsdp-world-size 4 \
  --infer-size 2 \
  --infer-tp-size 1 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --max-steps 400 \
  --sync-every-optimizer-steps 32 \
  --learning-rate 1e-6 \
  --weight-decay 0.0 \
  --max-length 8192 \
  --train-attention-backend flash_attention_2 \
  --gradient-checkpointing \
  --clip-mode ppo \
  --clip-eps 0.2 \
  --log-every 1 \
  --seed 7 \
  --replay-capacity 8192 \
  --replay-wait-sleep-seconds 0.01 \
  --replay-sample-timeout-seconds 1800 \
  --num-rollout-workers 144 \
  --rollout-batch-size 8 \
  --rollout-stop-timeout 600 \
  --infer-max-tokens 128 \
  --infer-temperature 1.0 \
  --infer-top-p 1.0 \
  --vllm-max-model-len 8192 \
  --vllm-max-num-batched-tokens 262144 \
  --vllm-max-num-seqs 384 \
  --rollout-attention-backend FLASH_ATTN \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "train_log=${OUTPUT_DIR}/train.log"
echo "tensorboard_log=${OUTPUT_DIR}"
