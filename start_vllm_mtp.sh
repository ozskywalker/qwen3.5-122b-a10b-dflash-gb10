#!/bin/bash
# Start Qwen3.5-122B-A10B with built-in MTP speculative decoding
# Uses the same aeon-vllm-tvm-fixed image as DFlash (required for SM121a B12X MoE kernels)
# MTP uses the model's own hidden states — no separate draft model needed

CONTAINER_NAME="qwen3.5-122b-mtp"
IMAGE="aeon-vllm-tvm-fixed:latest"

MAIN_MODEL="/models/models--nvidia--Qwen3.5-122B-A10B-NVFP4/snapshots/98915d837c4e7c87ac8296d02e89de19b3207e6d"

MTP_TOKENS="${1:-2}"    # MTP spec tokens; default 2 (typical for MTP)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vLLM 0.24.0+aeon │ Qwen3.5-122B │ MTP tokens=${MTP_TOKENS}"
echo " Base: aeon-vllm-tvm-fixed (SM121a, CUDA 13.0, B12X, tvm-ffi 0.1.12)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Stop/remove old container if running
sudo docker stop "${CONTAINER_NAME}" 2>/dev/null && \
    sudo docker rm  "${CONTAINER_NAME}" 2>/dev/null || true

sudo docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --net=host \
    -v "/models:/models:ro" \
    -v "/home/luser/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "/home/luser/.cache/vllm:/root/.cache/vllm" \
    -e CUTE_DSL_ARCH=sm_121a \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e FLASHINFER_CACHE_DIR=/root/.cache/flashinfer \
    --entrypoint vllm \
    "${IMAGE}" \
        serve "${MAIN_MODEL}" \
        --served-model-name "qwen3.5-122B-A10B" \
        --trust-remote-code \
        --quantization modelopt_fp4 \
        --kv-cache-dtype fp8 \
        --tensor-parallel-size 1 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 196608 \
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}" \
        --gpu-memory-utilization 0.88 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 8192 \
        --enable-prefix-caching

echo ""
echo "Container '${CONTAINER_NAME}' started. Tailing logs..."
echo "(First run: FlashInfer SM121 JIT warmup + torch.compile may take 5-10 min)"
echo ""
sudo docker logs -f "${CONTAINER_NAME}"
