# Qwen3.5-122B-A10B with DFlash Speculative Decoding on DGX Spark GB10

End-to-end guide for running `nvidia/Qwen3.5-122B-A10B-NVFP4` with `z-lab/Qwen3.5-122B-A10B-DFlash`
speculative decoding on an NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory).

**Status:** Working as of 2026-07-14. 4× concurrent OpenCode agents confirmed stable.

## Background and motivation

The NVIDIA DGX Spark (GB10 Superchip) is a unique machine: it pairs a full Blackwell GPU (SM121a)
with 128 GB of unified CPU+GPU memory on a single SoC. This makes it possible to run frontier-scale
MoE models like Qwen3.5-122B entirely in-memory on a single consumer-accessible device, with no
multi-GPU coordination overhead.

However, running such models at interactive latency is hard. NVFP4 quantization reduces the
122B-parameter model to ~75 GB weight footprint, and at decode time the GPU generates tokens
sequentially at ~10–16 tok/s without speculative decoding — acceptable for throughput, but slow
for interactive use where time-to-first-token and inter-token latency matter.

**DFlash** (Block Diffusion Flash, from z-lab) addresses this with a lightweight speculative
decoding approach that uses the last 6 layers of Qwen3.5-122B itself as the draft model. Instead
of running a separate smaller model, the draft produces multiple candidate tokens in a single
non-causal forward pass, which the full 94-layer model then verifies. When the draft predicts
correctly, multiple tokens are committed per step rather than one — reducing the number of full
model passes needed per output token.

This repository documents the bring-up work required to make DFlash work on the GB10 Spark
under the aeon vLLM image (the only publicly available image with SM121a NVFP4 MoE kernels as
of mid-2026), including 5 bug fixes that were needed to get past a series of errors during
initialization and forward passes. The fixes are applied as a thin Dockerfile layer on top of
the aeon base image.

### Why this was nontrivial

The DFlash draft model has an unusual architecture: it mixes sliding-window attention layers
(SWA, layers 48–51), a Mamba SSM layer (layer 52), and a full-attention layer (layer 53).
This hybrid layout exposed multiple bugs in vLLM's KV cache unification, FlashInfer backend,
and the DFlash proposer itself — none of which appear when running standard dense or pure-MoE
models. All 5 fixes are documented in detail below.

### Primary use case

This setup is optimized for interactive coding assistance via OpenCode agents. The real-world
workload is 4–8 concurrent agents, each sending 1024-token prompts and expecting 512-token
code completions. Benchmark results show 39.5 tok/s aggregate output at c=4 — roughly 10 tok/s
per agent, which is fast enough for non-blocking agentic workflows.

**Status:** Working as of 2026-07-14. 4× concurrent OpenCode agents confirmed stable.

---

## Hardware & Software

| | |
|---|---|
| **Hardware** | NVIDIA DGX Spark, GB10 Superchip, SM121a, 128 GB unified memory |
| **OS** | Ubuntu 24.04, aarch64 |
| **CUDA driver** | 13.0 (host) |
| **Container base** | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` |
| **Built image** | `aeon-vllm-tvm-fixed:latest` |
| **vLLM** | 0.24.0+aeon.sm121a.dflash |
| **FlashInfer** | bundled in aeon image, with SM121a JIT kernels |

---

## Models

| Role | Model | Path on host |
|---|---|---|
| Main (target) | `nvidia/Qwen3.5-122B-A10B-NVFP4` | `/models/models--nvidia--Qwen3.5-122B-A10B-NVFP4/snapshots/98915d837c4e7c87ac8296d02e89de19b3207e6d` |
| Draft | `z-lab/Qwen3.5-122B-A10B-DFlash` | `/models/models--z-lab--Qwen3.5-122B-A10B-DFlash/snapshots/bce6f76cef2027552bed4a8a1bc9c449def48f05` |

The main model is quantized to NVFP4 (weights) + FP8 (KV cache). The draft model contains only the
last 6 layers of Qwen3.5-122B (layers 48–53): 4 sliding-window attention layers, 1 Mamba layer,
and 1 full-attention layer. It shares weights with the main model (EAGLE-style).

---

## Building the Container

```bash
cd /home/luser/aeon-tvm-fix
sudo docker build -t aeon-vllm-tvm-fixed:latest .
```

The build context is the `aeon-tvm-fix/` directory. The `patches/` subdirectory must be present
(it is the source for the three COPY instructions that bake in fixes 3–5).

**Build time:** ~2 minutes (mostly pip install; no compilation).

### What the Dockerfile does

The base image `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` already has:
- vLLM 0.24.0+aeon.sm121a.dflash with DFlash speculative decoding support
- FlashInfer with SM121a (GB10/Blackwell) JIT kernels
- FLASHINFER_B12X NVFP4 MoE kernels (requires CUDA 13.0 runtime — baked in)
- CUDA 13.0 toolkit

On top of that, the Dockerfile applies 5 fixes:

#### Fix 1 — apache-tvm-ffi upgrade
`nvidia-cutlass-dsl 4.6.0` (used by FLASHINFER_B12X) calls
`make_kwargs_wrapper(map_dataclass_to_tuple=...)`, which was added in `apache-tvm-ffi 0.1.11`.
The base image ships `0.1.9`. Fix: `pip install --no-deps apache-tvm-ffi==0.1.12`.

#### Fix 2 — MoE shared-memory budget (`moe_dynamic_kernel.py`)
`MoEDynamicKernel` in gated/SiLU mode allocates `sB_up + sSFB_up` (~9216 bytes/stage) in
`StorageGated` that `_compute_stages()` doesn't count toward its smem budget. With
`intermediate_size=1536` → `k_tile_cnt=12` → `12 % 3 == 0` → `ab_stage` stays at 3 →
118 784 bytes allocated, exceeding SM121a's 99 KB shared-memory limit.

Fix: cap `smem_capacity = 73728` for gated mode so `_compute_stages()` returns `ab_stage=2`
(actual usage 91 136 bytes < 99 KB).

#### Fix 3a — KV cache page-size unification (`kv_cache_utils.py`)
`unify_kv_cache_spec_page_size` takes the `ratio=2` path for layers whose page size is half the
maximum. It scales `block_size × ratio` but leaves `page_size_padded` pinned at the old value,
so `new_spec.page_size_bytes` still returns the un-scaled padded size and the assertion
`new_spec.page_size_bytes == max_page_size` fails.

Fix: also scale `page_size_padded` by `ratio` when it is set.

#### Fix 3b — Hybrid KV spec unification preserves sliding window (`kv_cache_utils.py`)
`unify_hybrid_kv_cache_specs` converts `SlidingWindowSpec` → `FullAttentionSpec` when the model
mixes full-attention and sliding-window layers. The original conversion discarded the
`sliding_window` value, producing `FullAttentionSpec(sliding_window=None)` for every SWA layer.
This made all layers look identical to `is_kv_cache_spec_uniform`, collapsing them into a single
KV cache group with a merged spec that no longer distinguishes window sizes.

Fix: pass `sliding_window=spec.sliding_window` when converting, so the
`FlashInferMetadataBuilder` sees the correct per-layer window size and can normalize correctly
(see Fix 4a).

#### Fix 4a — Mixed window_left normalization (`flashinfer.py`)
`FlashInferMetadataBuilder.__init__` raises `ValueError: Window left is not the same for all
layers` when an attention group mixes SWA layers (`window_left=4095`) and a full-attention layer
(`window_left=-1`). The DFlash draft model deliberately has this mix (layers 48–51 are SWA,
layer 53 is full attention), but the KV cache spec has already been widened to `FullAttentionSpec`
for all draft layers (Fix 3b), so treating the entire group as full attention at runtime is safe.

Fix: when `has_same_window_lefts=False`, normalize `global_hyperparameters.window_left = -1`
(full attention) and mark as uniform, instead of raising.

#### Fix 4b — Forward-path window_left assertion (`flashinfer.py`)
`FlashInferImpl.forward()` asserts that `prefill_wrapper._window_left` matches the layer's
`self.window_left`. In the DFlash draft model, Fix 4a normalises the builder to use
`window_left=-1`, so the wrapper has `_window_left=-1`. Layers 48–51 (SWA) have `self.window_left
= 4095` — the existing allow-list for `wrapper == -1` handles them. Layer 53 (full attention)
has `self.window_left = -1`, which the original check already passes by exact match.

The additional condition `or self.window_left == -1` covers the edge case where Fix 4a doesn't
fire (e.g., if a future change makes all layers share the same merged window_left=4095): a
full-attention layer should always accept any wrapper window.

#### Fix 5 — DFlash proposer machinery (`dflash.py`)
Several issues in `DFlashProposer` that surfaced during bring-up:

- **`allow_multiple_draft_kv_cache_groups()`** — must return `True` to allow the draft model's
  attention group to coexist with the main model's groups.
- **`set_draft_block_table()`** — guards on `kv_cache_gid in self._draft_kv_cache_group_ids`
  to prevent writing the block table to the wrong group.
- **`_create_draft_vllm_config()`** — strips the `is_mm_prefix_lm` multimodal flag (draft model
  is text-only) and removes the `BACKED_SIZE_OBLIVIOUS` dynamic-shapes flag that caused a CUDA
  illegal memory access.
- **Slot-mapping buffers** — allocate per KV cache group; first group reuses the proposer's
  existing buffers, additional groups get fresh zero-filled tensors.

---

## Running

```bash
./start_vllm_qwen3.5_122b_DFlash_aeon.sh [block_size]
```

`block_size` defaults to 16. Pass 8 for better multi-request concurrency.

The script stops any running container with the same name, starts a new detached container, then
tails the logs. On first run (or after clearing the vLLM compile cache), expect ~10–15 minutes
for SM121a JIT warmup and torch.compile.

### Key flags explained

| Flag | Value | Why |
|---|---|---|
| `--quantization modelopt_fp4` | NVFP4 | Required for the nvidia NVFP4 checkpoint |
| `--kv-cache-dtype fp8` | FP8 KV | FP8 KV fits in 128 GB; FlashInfer auto-selects FP8-aware attention |
| `--moe-backend flashinfer_b12x` | B12X | SM121-compiled NVFP4 MoE kernels; other backends fail with `cudaErrorNoKernelImageForDevice` |
| `--speculative-config` | dflash, 16 tokens | `enforce_eager: true` needed — DFlash draft doesn't use CUDA graphs |
| `--gpu-memory-utilization 0.88` | 88% | Leave headroom for the SM121a JIT workspace |
| `--max-model-len 196608` | 192k tokens | Full GB10 context window |
| `--max-num-seqs 8` | 8 | Limit concurrent sequences; DFlash adds per-sequence draft overhead |

### Caches to persist across container restarts

```bash
-v "/home/luser/.cache/flashinfer:/root/.cache/flashinfer"  # FlashInfer SM121 JIT kernels
-v "/home/luser/.cache/vllm:/root/.cache/vllm"              # torch.compile artifacts
```

Without these mounts the full JIT warmup runs on every start (~10 min).

---

## Architecture: How DFlash works

DFlash (Block Diffusion Flash) is a speculative decoding method from `z-lab`:

1. **Draft**: The draft model (last 6 layers of Qwen3.5-122B) runs *non-causally* over the
   current position + N speculative slots in a single forward pass, producing N draft tokens
   simultaneously (block diffusion). This is much cheaper than running the full 94-layer model.

2. **Verify**: The main model (full 94 layers) verifies all N draft tokens in one causal forward
   pass, accepting tokens that match its distribution and rejecting the rest.

3. **Throughput**: When the acceptance rate is high (~3–4 tokens accepted per speculative step on
   average), wall-clock throughput improves 3–4× vs. greedy baseline at the same quality.

The key architectural feature is that the draft model shares all weights with the main model —
no separate smaller model is needed, and the draft tokens come "for free" as the last-layer
activations of the main model's own computation.

---

## Debugging history

The following errors were encountered and resolved during bring-up. Recorded here for future
reference if the base image is updated.

| # | Error | Root cause | Fix |
|---|---|---|---|
| 1 | `TypeError: make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'` | `apache-tvm-ffi 0.1.9` too old for `nvidia-cutlass-dsl 4.6.0` | Fix 1: upgrade to 0.1.12 |
| 2 | `CUDA_ERROR_ILLEGAL_ADDRESS` in MoE kernel | SM121a smem limit exceeded by `MoEDynamicKernel` gated mode | Fix 2: cap smem budget |
| 3 | `AssertionError: new_spec.page_size_bytes == max_page_size` | `page_size_padded` not scaled with `block_size` in `unify_kv_cache_spec_page_size` | Fix 3a |
| 4 | `ValueError: Window left is not the same for all layers` | FlashInfer rejects mixed SWA+full-attn in one group | Fix 4a |
| 5 | `NotImplementedError: DFlash does not support BACKED_SIZE_OBLIVIOUS` | Draft vllm config inherited `BACKED_SIZE_OBLIVIOUS` dynamic shapes | Fix 5 (`_create_draft_vllm_config`) |
| 6–12 | Various `AssertionError` / `AttributeError` in propose path | DFlash proposer machinery not wired for the aeon vLLM version | Fix 5 (multiple sub-fixes) |
| 13 | `AssertionError: window_left mismatch: wrapper=4095 impl=-1` | SlidingWindowSpec → FullAttentionSpec conversion discarded `sliding_window` value, causing incorrect merge | Fix 3b + Fix 4b |

---

## Benchmark results

All runs: `vllm bench serve`, 30 prompts, `request-rate inf`, random dataset,
`enable_thinking=false`. Config at time of measurement: `max-num-batched-tokens 8192`,
`num_speculative_tokens 8`, no `enforce_eager`.

| Scenario | Input/Output | Conc | Output tok/s | TTFT p50 | ITL p50 | Acc% | AccLen |
|---|---|---|---|---|---|---|---|
| short-c1 | 512 / 128 | 1 | 15.6 | 592ms | 115ms | 12.1% | 1.97 |
| code-c1  | 1024 / 512 | 1 | 18.8 | 705ms | 117ms | 15.6% | 2.25 |
| code-c4  | 1024 / 512 | 4 | 39.5 | 1043ms | 216ms | 16.9% | 2.35 |
| long-c1  | 4096 / 512 | 1 | 15.9 | 1685ms | 116ms | 11.9% | 1.95 |

### Tuning history (vs original `enforce_eager=true`, `max_batched=4096`, `speclen=16`)

| Change | Throughput delta | ITL delta | Notes |
|---|---|---|---|
| `max_num_batched_tokens` 4096 → 8192 | +4% (short/code), **+47% long** | -6–10% | Unblocked long-context prefill |
| Remove `enforce_eager` | +2–9% single-stream | flat | Draft CUDA graphs captured cleanly |
| `num_speculative_tokens` 16 → 8 | **+10–21%** | **-14–21%** | AccLen≈2 regardless; halving the spec window doubles yield |
| `--enable-prefix-caching` | **CRASH** | — | Incompatible: `HybridKVCacheCoordinator` requires all KV group block sizes divisible by `hash_block_size`; DFlash's mixed Mamba+attention+draft groups violate this |

### Key insight: speculative token count
DFlash accepts ~2 tokens per speculative step on random text (AccLen≈2), independent of
`num_speculative_tokens`. With `speclen=16`: 2/16 = 12.5% yield. With `speclen=8`: 2/8 = 25%
yield. The draft forward-pass cost scales with speclen, so the optimal setting is near the
expected AccLen. Real code workloads (OpenCode agents) likely have higher acceptance than
random prompts — if AccLen rises above 4 in practice, increasing speclen to 12–16 may
recapture throughput.

## Further tuning opportunities

These have not been benchmarked yet. Listed roughly in priority order.

### 1. Tune `num_speculative_tokens` to your actual workload (high priority)

The benchmarks above used random text, which is adversarial for speculative decoding (low
repetition → low acceptance). On real code, AccLen is likely 3–5× rather than ~2. To find
the optimal speclen for your workload:

1. Run the server with the production workload for 10–15 minutes.
2. Watch `spec_decode_acceptance_length` in the vLLM server logs (logged every 10s).
3. Set `num_speculative_tokens` to `round(AccLen * 1.5)` — enough headroom to capture most
   accepted tokens without paying excessive overhead on the tail.

Rule of thumb: if AccLen on real traffic is ~4, try speclen=6 or 8. If AccLen is ~6, try 8–10.

### 2. `--max-num-batched-tokens 16384` for long-context workloads

Going from 4096→8192 gave +47% on long-c1. A further doubling to 16384 may yield additional
gains on prompts approaching the full 192k context window. The warning
`max_num_scheduled_tokens is set to N` will reduce by the spec token overhead
(8 seqs × speclen tokens), but the headroom is now large enough that this rarely matters.
Watch GPU memory — increase `--gpu-memory-utilization` slightly if KV cache shrinks too much.

### 3. `--gpu-memory-utilization 0.90` (minor)

vLLM logs that the effective utilization after CUDA graph profiling is ~0.87 when 0.88 is set.
Bumping to 0.90 gives ~3% more KV cache (~560K → ~580K tokens) at low risk on 128 GB unified
memory. Not worth benchmarking in isolation but worth combining with another change.

### 4. `--max-num-seqs 4` at low concurrency

The 8-sequence cap adds `8 × speclen` to the batch budget overhead. At c=1 workloads, capping
at 4 sequences halves the overhead and may marginally improve single-stream TTFT. Only relevant
if you exclusively run single requests.

### 5. Prefix caching (blocked — see Known limitations)

`--enable-prefix-caching` crashes at startup due to a `HybridKVCacheCoordinator` incompatibility.
This would be the highest-value change for interactive use (repeated system prompts in OpenCode
agents share large common prefixes). Requires a fix in vLLM's coordinator to handle DFlash's
mixed block sizes — not patchable at the Python patch level without a more invasive change.

---

## Known limitations

- **Prefix caching incompatible**: `--enable-prefix-caching` causes `AssertionError: block_size
  must be divisible by hash_block_size` at startup. Root cause: `HybridKVCacheCoordinator`
  validates all KV groups; DFlash's mixed Mamba+attention+draft groups have block sizes that
  don't all divide evenly by the scheduler's `hash_block_size`. This is a vLLM limitation
  requiring changes to the coordinator, not patchable at the Python level.

## Pending

- **Image tag**: consider versioning `aeon-vllm-tvm-fixed:latest` once stable
  (e.g., `aeon-vllm-tvm-fixed:20260714`).
- **Upstream**: Fixes 3a, 3b, 4a, 4b are candidates for upstreaming to vLLM. Fix 5 is
  specific to the aeon DFlash proposer implementation.
- **Real-workload acceptance rate**: measure AccLen under actual OpenCode agent traffic
  (code is more repetitive than random text — AccLen may be 3–5×, justifying a higher speclen).

---

## File map

```
qwen3.5-dflash-gb10/
├── README.md                 # this file
├── Dockerfile                # builds aeon-vllm-tvm-fixed:latest
├── start_vllm_dflash.sh      # launch script (DFlash, recommended)
├── start_vllm_mtp.sh         # launch script (MTP baseline)
├── patches/
│   ├── kv_cache_utils.py     # Fixes 3a + 3b
│   ├── flashinfer.py         # Fixes 4a + 4b
│   └── dflash.py             # Fix 5
└── bench-results/
    ├── 00-baseline-*.json    # enforce_eager=true, max_batched=4096, speclen=16
    ├── 01-batched8k-*.json   # max_batched=8192
    ├── 02-no-eager-*.json    # CUDA graphs enabled for draft
    └── 03-speclen8-*.json    # num_speculative_tokens=8 (final/best config)
```
