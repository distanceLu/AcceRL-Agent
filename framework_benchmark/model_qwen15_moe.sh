# Exact Qwen1.5-MoE-A2.7B-Chat HF config. No QK norm; QKV bias is present.
MODEL_ARGS=(
  --num-layers 24 --hidden-size 2048 --ffn-hidden-size 5632
  --num-attention-heads 16 --num-query-groups 16 --group-query-attention --kv-channels 128
  --disable-bias-linear --add-qkv-bias --normalization RMSNorm --norm-epsilon 1e-6
  --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
  --swiglu --untie-embeddings-and-output-weights --vocab-size 151936
  --num-experts 60 --moe-ffn-hidden-size 1408 --moe-router-topk 4
  --moe-router-score-function softmax --moe-router-pre-softmax
  --moe-shared-expert-intermediate-size 5632 --moe-shared-expert-gate
  --moe-grouped-gemm --moe-token-dispatcher-type alltoall
  --moe-router-load-balancing-type none --moe-aux-loss-coeff 0
  --attention-dropout 0 --hidden-dropout 0
)
