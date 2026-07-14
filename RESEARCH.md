# Research areas: Qwen3.5-122B DFlash on GB10

Open questions and experiments worth pursuing, roughly in priority order.
Each item includes the hypothesis, what to measure, and the expected difficulty.

---

## 1. Speclen calibration on real code traffic (highest priority)

**Hypothesis:** AccLen on real OpenCode agent traffic is significantly higher than the ~2
measured on random prompts. Code is repetitive and predictable; the DFlash draft may accept
4–6 tokens per step, shifting the optimal `num_speculative_tokens` upward.

**What to measure:**
- Enable vLLM's observability metrics: add `--collect-spec-decoding-metrics` and scrape
  `vllm:spec_decode_draft_acceptance_rate` and `vllm:spec_decode_num_accepted_tokens` from
  the `/metrics` Prometheus endpoint while running actual OpenCode sessions.
- Record AccLen at sustained c=1, c=4 under real agent workload.
- Sweep `num_speculative_tokens` ∈ {4, 6, 8, 10, 12} and benchmark each against the
  code-c4 scenario — the primary production case.

**Expected outcome:** If AccLen on real traffic is 4–6, speclen=8–10 will outperform speclen=8
on ITL and throughput. If AccLen is still ~2, speclen=4 may be even better than speclen=8.

**Difficulty:** Low — config change only. Run `docker exec qwen3.5-122b-dflash curl localhost:8000/metrics | grep spec_decode` during live load.

---

## 2. MTP vs DFlash head-to-head comparison

**Hypothesis:** DFlash's block-diffusion approach may not be straightforwardly better than
vLLM's built-in MTP (Multi-Token Prediction) speculative decoding for this specific model
on GB10. MTP uses the model's own hidden states and a small auxiliary head — very cheap —
and may have higher acceptance on code where token distributions are sharp.

**What to measure:**
- Run the full 4-scenario benchmark matrix against `start_vllm_mtp.sh` (MTP with
  `num_speculative_tokens=2`, which is currently configured).
- Compare: throughput, TTFT, ITL, AccLen.
- Note: MTP supports `--enable-prefix-caching`, which DFlash does not. This is a
  significant advantage for interactive use.

**Key question:** Does DFlash's parallel block proposal (non-causal) beat MTP's sequential
2-token proposal on code generation latency? On random prompts, DFlash got AccLen≈2 at
speclen=8; MTP with speclen=2 would accept ~2 tokens sequentially. The forward-pass cost
differs — DFlash proposes 8 at once vs MTP proposes 2 per main-model step.

**Difficulty:** Low. MTP script already exists. Use `02-mtp-baseline` label prefix.

---

## 3. Prefix caching for DFlash (high value, medium difficulty)

**Hypothesis:** OpenCode agents share long system prompts (tool definitions, instructions).
Prefix caching would allow the KV cache for the system prompt to be reused across requests,
potentially cutting TTFT by 50–70% on repeated-prefix traffic.

**The blocker:** `HybridKVCacheCoordinator.__init__` asserts `block_size % hash_block_size == 0`
across all KV cache groups. DFlash's hybrid layout (Mamba + SWA attention + full attention +
draft) produces multiple groups with heterogeneous block sizes, at least one of which fails
the divisibility check.

**What to investigate:**
1. Identify exactly which group fails and what its block_size is (add logging to
   `kv_cache_coordinator.py` before the failing assertion).
2. The `hash_block_size` defaults to `scheduler_block_size` (16). Check whether rounding the
   failing group's block_size up to the next multiple of 16 is safe — this may be a one-line
   fix in `kv_cache_utils.py`.
3. Alternatively, check if `hash_block_size` can be set to the GCD of all group block sizes
   rather than defaulting to the scheduler block size.

**Expected impact:** TTFT reduction of 40–70% for second and subsequent turns in a
multi-turn conversation. This is the single highest-value unblocked feature.

**Difficulty:** Medium. Requires understanding the coordinator block-size math and verifying
that hash collisions don't occur with the modified block size.

---

## 4. Extended context benchmarks (64k, 128k tokens)

**Hypothesis:** The `max_num_batched_tokens` bottleneck was most visible at long-c1 (4096
token input), where going from 4096→8192 gave +47% throughput. At 64k or 128k token inputs —
which the model supports up to 192k — this effect may be even more pronounced or may plateau.

**What to measure:**
- Add `long-64k-c1` (64k input, 512 output, c=1) and `long-128k-c1` benchmarks.
- Test with `--max-num-batched-tokens` at 8192 and 16384.
- Watch KV cache utilization — at 64k+ inputs, the 543k-token KV cache (~8 full 64k contexts)
  may become the binding constraint before batch size.

**Practical note:** 64k-token random prompts are not representative of real workloads.
Consider using real document inputs (code repos, long documents) for this benchmark.

**Difficulty:** Low for the benchmark setup; requires sourcing representative long-context
test inputs.

---

## 5. Two-group DFlash architecture (correctness + future-proofing)

**Current workaround (Fix 4a):** When the DFlash draft group has mixed window sizes (SWA
layers 48–51 at window=4095, full-attention layer 53 at window=-1), Fix 4a normalizes all
to window=-1 (full attention). This is safe today because DFlash only proposes 8–16 tokens —
far below the 4096-token SWA window — so attending slightly wider than necessary has zero
quality impact.

**The correct architecture:** Split the DFlash draft's FlashInfer groups:
- Group A: layers 48–51 (SWA, window_left=4095, causal=True)
- Group B: layer 53 (full attention, window_left=-1, causal=False)

This would require `build_per_group_and_layer_attn_metadata()` in `dflash.py` to produce two
`attn_metadata` objects and route each layer to the correct group during the draft forward pass.

**Why it matters:** The current workaround is a correctness approximation. If speclen were
ever pushed to 4096+ tokens, SWA layers would silently attend to wrong context. More
practically, upstreaming Fix 4a to vLLM requires either this proper fix or a clear documented
assumption about max speclen.

**Difficulty:** High. Requires deep changes to `dflash.py` and the FlashInfer metadata
construction path. Not needed for current usage, but important for any upstream contribution.

---

## 6. Speculative decoding metrics via Prometheus

**Current state:** No runtime visibility into acceptance rate or draft quality. We only see
`spec_decode_acceptance_rate` in `vllm bench serve` JSON output post-run.

**What to set up:**
- Add a Prometheus scrape endpoint (vLLM exposes `/metrics` by default).
- Key metrics: `vllm:spec_decode_draft_acceptance_rate`, `vllm:spec_decode_num_accepted_tokens`,
  `vllm:spec_decode_num_draft_tokens`, `vllm:gpu_cache_usage_perc`.
- A Grafana dashboard with acceptance rate, ITL p50/p95, and KV cache utilization gives
  real-time visibility into draft quality under production load.

**Why it matters:** Without live metrics, we can't tell if acceptance rate degrades under
concurrent load, if certain request types cause low acceptance (hurting latency for others
in the batch), or if the KV cache is filling up. This is table stakes for production operation.

**Difficulty:** Low. vLLM exposes `/metrics` by default; just needs a scrape config and
a dashboard. No code changes required.

---

## 7. Thinking mode acceptance rate

**Current state:** All benchmarks used `enable_thinking=false`. Qwen3.5 with thinking enabled
produces a different token distribution (chain-of-thought tokens have different statistical
properties than code or direct answers).

**Hypothesis:** Thinking-mode output may have lower acceptance rate than code — reasoning
tokens tend to be more varied and creative — but the longer output lengths (512–4096 thinking
tokens per response) mean any per-token latency improvement is amplified.

**What to measure:**
- Repeat code-c1 and code-c4 benchmarks with `enable_thinking=true` and speclen=4, 8, 12.
- Compare AccLen and ITL with thinking vs non-thinking.

**Difficulty:** Low. One flag change; may require longer bench runs to get stable statistics
given the variable output length.

---

## 8. Upstreaming fixes to vLLM

Fixes 3a, 3b, 4a, and 4b are general correctness fixes for hybrid models in vLLM and are
not specific to the aeon image or GB10. They are candidates for upstream PRs.

**Fix 3a** (`page_size_padded` scaling): Clean correctness fix. No behavior change for
models where `page_size_padded` is not set. Low controversy.

**Fix 3b** (preserve `sliding_window` in hybrid KV spec conversion): Needs careful review —
the original code may have discarded `sliding_window` deliberately for other hybrid models.
Requires a regression test on a non-DFlash hybrid model (e.g., Mistral with SWA).

**Fix 4a** (window_left normalization): Ideally replaced by the two-group architecture
(item 5 above) before upstreaming. As a standalone fix it's too DFlash-specific.

**Fix 4b** (forward assertion relaxation): Small and defensible. Pairs naturally with 4a.

**Process:** File issues in the vLLM repo referencing the specific error messages, then
submit PRs with the patches. The aeon team may also be interested.

---

## 9. `--max-num-batched-tokens 16384` for long-context

We went 4096→8192 and saw the largest gains on long-c1 (+47%). The hypothesis is that a
second doubling to 16384 would again benefit long-context most. This is a one-line change.

**Risk:** Memory pressure. With 8192 tokens of batch budget, the KV cache sits at 543k
tokens. At 16384, the profiling run uses more memory, potentially reducing the KV cache
budget. Watch the startup log: `GPU KV cache size: N tokens` — if it drops below ~400k,
the tradeoff may not be worth it.

**Mitigation:** Combine with `--gpu-memory-utilization 0.90` to compensate.

---

## 10. Multi-instance serving for request diversity

**Observation:** DFlash throughput at c=4 is 39.5 tok/s aggregate. A second vLLM instance
is not possible on a single GB10 (model is 75+ GB, leaving ~50 GB for KV cache; a second
copy would not fit). However, at lower quantization or with a smaller model, it might be
possible to run two processes sharing the GPU.

**Alternative angle:** For mixed-length workloads (some requests short, some very long),
a single DFlash instance with `max_num_seqs=8` may serialize short requests behind long
ones. Running two processes — one with `max_model_len=8192` (for short/code), one with
`max_model_len=196608` (for long-context) — on separate ports could improve P95 latency
for short requests when long ones are in-flight.

**Difficulty:** High (memory constraints), speculative. Worth revisiting if GB10 unified
memory management improves with future drivers.
