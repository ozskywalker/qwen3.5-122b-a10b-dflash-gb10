# CLAUDE.md — DFlash GB10 research project

This file gives Claude context for working in this repository. Read it at the start of any
session before making changes or running experiments.

---

## What this repo is

Working configuration for `nvidia/Qwen3.5-122B-A10B-NVFP4` with DFlash speculative decoding
on a DGX Spark GB10. Five bugs were fixed to get here (see README.md). The patches are
applied as a Dockerfile layer on top of `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`, pinned
to digest `sha256:d74b35b7aeeebe9e5c386fe27241eada8732055ccc6e856a6fd000df00c6ca1a`.

This is an active research/tuning project. RESEARCH.md tracks open experiments.

---

## Hardware

- **Machine:** NVIDIA DGX Spark, GB10 Superchip, SM121a, 128 GB unified memory, aarch64
- **OS:** Ubuntu 24.04
- **CUDA driver:** 13.0 (host). Container image bakes in CUDA 13.0 toolkit.
- **Important:** SM121a requires `CUTE_DSL_ARCH=sm_121a` and the `flashinfer_b12x` MoE
  backend. Standard vLLM images fail with `cudaErrorNoKernelImageForDevice`.

---

## Key paths on host

| What | Path |
|---|---|
| Main model | `/models/models--nvidia--Qwen3.5-122B-A10B-NVFP4/snapshots/98915d837c4e7c87ac8296d02e89de19b3207e6d` |
| Draft model | `/models/models--z-lab--Qwen3.5-122B-A10B-DFlash/snapshots/bce6f76cef2027552bed4a8a1bc9c449def48f05` |
| FlashInfer JIT cache | `/home/luser/.cache/flashinfer` (volume-mounted into container) |
| vLLM compile cache | `/home/luser/.cache/vllm` (volume-mounted into container) |
| Benchmark tool | `/home/luser/venvs/vllm-bench/bin/vllm bench serve` |
| Bench results | `./bench-results/` |

---

## Container management

```bash
# Build (only needed when Dockerfile or patches change)
cd /path/to/this/repo
sudo docker build -t aeon-vllm-tvm-fixed:latest .

# Start DFlash server (num_speculative_tokens defaults to 8 via DFLASH_BLOCK_SIZE)
./start_vllm_dflash.sh          # speclen=8 (recommended)
./start_vllm_dflash.sh 16       # speclen=16 (first arg overrides)

# Start MTP baseline server
./start_vllm_mtp.sh

# Check server health
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
# → 200 means up (empty body is normal)
# Note: curl -sf fails silently here; use -w "%{http_code}" instead

# Tail logs
sudo docker logs -f qwen3.5-122b-dflash

# Stop
sudo docker stop qwen3.5-122b-dflash && sudo docker rm qwen3.5-122b-dflash
```

**Startup time:** ~10 min cold (JIT + torch.compile). ~3 min warm (caches hit).
Watch for `Application startup complete.` in logs.

---

## Current best config

As of 2026-07-14, `start_vllm_dflash.sh` runs:

| Flag | Value | Why |
|---|---|---|
| `--max-num-batched-tokens` | 8192 | Doubled from 4096; +47% on long-context |
| `num_speculative_tokens` | 8 (default arg) | Halved from 16; AccLen≈2 on random → better yield |
| `enforce_eager` | not set | Draft CUDA graphs active; +9% single-stream |
| `--enable-prefix-caching` | set | Fixed in kv_cache_utils.py (Fix 6); recommended for interactive use |
| `--gpu-memory-utilization` | 0.88 | ~543k token KV cache; 27 GiB available |
| `--max-num-seqs` | 8 | Limits concurrent sequences |
| `--max-model-len` | 196608 | Full 192k context |

---

## Running benchmarks

Standard benchmark command (all scenarios):

```bash
BENCH=/home/luser/venvs/vllm-bench/bin/vllm
TOK=/models/models--nvidia--Qwen3.5-122B-A10B-NVFP4/snapshots/98915d837c4e7c87ac8296d02e89de19b3207e6d

$BENCH bench serve \
  --backend openai-chat --model qwen3.5-122B-A10B \
  --host localhost --port 8000 --endpoint /v1/chat/completions \
  --tokenizer $TOK \
  --dataset-name random \
  --random-input-len INPUT_LEN --random-output-len OUTPUT_LEN \
  --num-prompts 30 --max-concurrency CONCURRENCY --request-rate inf \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
  --save-result --result-dir ./bench-results \
  --result-filename LABEL.json
```

**Standard 4-scenario matrix:**

| Label suffix | INPUT_LEN | OUTPUT_LEN | CONCURRENCY |
|---|---|---|---|
| `-short-c1` | 512 | 128 | 1 |
| `-code-c1` | 1024 | 512 | 1 |
| `-code-c4` | 1024 | 512 | 4 |
| `-long-c1` | 4096 | 512 | 1 |

**Label prefix convention:** `NN-description` where NN is a two-digit sequence number.
Current: `00-baseline`, `01-batched8k`, `02-no-eager`, `03-speclen8`.

**Key metrics to track per run:**
- `output_throughput` (tok/s)
- `p50_ttft_ms`, `p95_ttft_ms`
- `p50_itl_ms`, `p95_itl_ms`
- `spec_decode_acceptance_rate` (%), `spec_decode_acceptance_length`

**Parse results:**
```bash
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob('./bench-results/*.json')):
    d = json.load(open(f))
    print(f"{f:<50} {d['output_throughput']:>6.1f} tok/s  "
          f"TTFT={d['p50_ttft_ms']:.0f}ms  ITL={d['p50_itl_ms']:.1f}ms  "
          f"Acc={d.get('spec_decode_acceptance_rate',0):.1f}%  "
          f"AccLen={d.get('spec_decode_acceptance_length',0):.2f}")
EOF
```

---

## Live spec decoding metrics

While the server is running under real load:

```bash
sudo docker exec qwen3.5-122b-dflash curl -s localhost:8000/metrics \
  | grep -E "spec_decode|gpu_cache"
```

Key Prometheus metrics:
- `vllm:spec_decode_draft_acceptance_rate` — real-time acceptance rate
- `vllm:spec_decode_num_accepted_tokens_total` — cumulative accepted
- `vllm:spec_decode_num_draft_tokens_total` — cumulative proposed
- `vllm:gpu_cache_usage_perc` — KV cache fill level

---

## Patched files

The three files in `patches/` are copied over the container's site-packages at build time
AND volume-mounted at runtime (for live editing without rebuild):

| File | Fixes |
|---|---|
| `patches/kv_cache_utils.py` | Fix 3a (page_size_padded scaling), Fix 3b (preserve sliding_window), Fix 6 (prefix caching GCD hash_block_size) |
| `patches/flashinfer.py` | Fix 4a (normalize mixed window_left), Fix 4b (relax forward assertion) |
| `patches/dflash.py` | Fix 5 (multi-group KV, config, slot-mapping) |

If you edit a patch file, changes take effect on next container start — no rebuild needed
because the volume mount overrides the baked-in copy.

---

## Known issues

**`--enable-prefix-caching` — FIXED (Fix 6 in `patches/kv_cache_utils.py`).**
Previously crashed with `AssertionError: block_size must be divisible by hash_block_size`.
Root cause: `resolve_kv_cache_block_sizes()` returned `hash_block_size = lcm(all groups) = 8736`,
but one FullAttentionSpec group had `block_size = 4368` (4368 % 8736 ≠ 0).
Fix: use `math.gcd(*group_block_sizes)` as hash_block_size in the Mamba bail-out path.
With GCD=4368, all 9 group block sizes (4368 and 8736) divide evenly.
The `start_vllm_dflash.sh` launch script now includes `--enable-prefix-caching`.

**`max_num_scheduled_tokens` warning at startup.**
```
WARNING: max_num_scheduled_tokens is set to 8072 based on speculative decoding settings.
```
Expected. With `max_num_batched_tokens=8192`, `max_num_seqs=8`, `num_speculative_tokens=8`:
overhead = 8 × 7 = 56 (or similar) → effective = 8136. Not a problem in practice.
Silence it by increasing `--max-num-batched-tokens` to 16384 or reducing `--max-num-seqs`.

**`CUDAGraphMode.FULL_AND_PIECEWISE` downgraded.**
```
INFO: setting cudagraph_mode=PIECEWISE
```
Expected. FlashInfer backend with spec-decode does not support the FULL variant. PIECEWISE
is still used and provides most of the benefit.

**`Mamba cache mode set to 'align'`** — appears when `--enable-prefix-caching` is set
(i.e., before the crash). Harmless warning, but moot since prefix caching crashes anyway.

---

## Rebuild policy

The Dockerfile base is pinned to a specific digest. To update the base image:
1. Pull: `sudo docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest`
2. Get new digest: `sudo docker inspect ghcr.io/aeon-7/aeon-vllm-ultimate:latest --format '{{.Id}}'`
3. Update the `FROM` line in `Dockerfile`
4. Rebuild: `sudo docker build -t aeon-vllm-tvm-fixed:latest .`
5. Verify each of the 5 fixes still applies cleanly (the Dockerfile's `assert` in Fix 2
   will fail loudly if the target line changes; Fixes 3–5 apply silently via COPY).

---

## Research agenda

See `RESEARCH.md` for the full list. Top priorities:
1. Measure AccLen on real OpenCode traffic (may justify bumping speclen to 10–12)
2. MTP vs DFlash head-to-head comparison
3. Fix prefix caching incompatibility (high value, medium effort)
4. Add Prometheus scraping for live acceptance rate visibility
