#!/bin/bash
# Start Qwen3.5-122B-A10B with DFlash speculative decoding
# Uses ghcr.io/aeon-7/aeon-vllm-ultimate (vLLM 0.24.0+aeon.sm121a.dflash, CUDA 13.0)
# This image is purpose-built for NVIDIA GB10 DGX Spark (SM121a) with:
#   - DFlash speculative decoding
#   - FLASHINFER_B12X NVFP4 MoE (requires CUDA 13.0 runtime, baked into image)
#   - No mixed-attention guard (works with z-lab/Qwen3.5-122B-A10B-DFlash)
#
# DFlash benchmarks (concurrency-1):
#   block=8  → ~3.2-3.7x speedup   block=16 → ~3.4-4.2x speedup
# Set DFLASH_BLOCK_SIZE below; 8=better concurrency, 16=best single-stream

CONTAINER_NAME="qwen3.5-122b-dflash"
# aeon-vllm-tvm-fixed adds apache-tvm-ffi==0.1.12 on top of aeon-vllm-ultimate.
# Fixes: TypeError: make_kwargs_wrapper() got an unexpected keyword argument
# 'map_dataclass_to_tuple' (nvidia-cutlass-dsl 4.6.0 requires tvm-ffi>=0.1.11)
IMAGE="aeon-vllm-tvm-fixed:latest"

MAIN_MODEL="/models/models--nvidia--Qwen3.5-122B-A10B-NVFP4/snapshots/98915d837c4e7c87ac8296d02e89de19b3207e6d"
DRAFT_MODEL="/models/models--z-lab--Qwen3.5-122B-A10B-DFlash/snapshots/bce6f76cef2027552bed4a8a1bc9c449def48f05"

DFLASH_BLOCK_SIZE="${1:-8}"    # pass as first arg, defaults to 8 (AccLen≈2 → better yield than 16)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vLLM 0.24.0+aeon │ Qwen3.5-122B │ DFlash block=${DFLASH_BLOCK_SIZE}"
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
    -v "/home/luser/aeon-tvm-fix/patches/kv_cache_utils.py:/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py:ro" \
    -v "/home/luser/aeon-tvm-fix/patches/flashinfer.py:/usr/local/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py:ro" \
    -v "/home/luser/aeon-tvm-fix/patches/dflash.py:/usr/local/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py:ro" \
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
        --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${DFLASH_BLOCK_SIZE}}" \
        --moe-backend flashinfer_b12x \
        --gpu-memory-utilization 0.88 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 8192 \
        --enable-prefix-caching

echo ""
echo "Container '${CONTAINER_NAME}' started. Tailing logs..."
echo "(First run: FlashInfer SM121 JIT warmup + torch.compile may take 5-10 min)"
echo ""
sudo docker logs -f "${CONTAINER_NAME}"
