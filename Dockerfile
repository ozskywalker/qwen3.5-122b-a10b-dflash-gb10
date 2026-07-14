FROM ghcr.io/aeon-7/aeon-vllm-ultimate:latest@sha256:d74b35b7aeeebe9e5c386fe27241eada8732055ccc6e856a6fd000df00c6ca1a

# Fix 1: nvidia-cutlass-dsl 4.6.0 calls make_kwargs_wrapper(map_dataclass_to_tuple=...)
# which was added in apache-tvm-ffi 0.1.11.  The base image ships 0.1.9.
RUN pip install --no-deps "apache-tvm-ffi==0.1.12"

# Fix 2: MoEDynamicKernel gated (SiLU) SharedStorage allocates sB_up+sSFB_up
# (~9216 bytes/stage) that _compute_stages() does not count.  With
# intermediate_size=1536 → k_tile_cnt=12 → 12%3==0, ab_stage stays at 3 →
# 118784 bytes allocated, exceeding SM121a's 99 KB shared-memory limit.
# Cap the smem budget to 73728 bytes so _compute_stages returns ab_stage=2
# (actual usage 91136 bytes < 99 KB).
RUN python3 - <<'EOF'
import pathlib, re
p = pathlib.Path(
    "/usr/local/lib/python3.12/site-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
src = p.read_text()
old = "        self.smem_capacity = utils.get_smem_capacity_in_bytes(\"sm_120\")"
new = (
    "        _raw_smem = utils.get_smem_capacity_in_bytes(\"sm_120\")\n"
    "        # StorageGated allocates sB_up+sSFB_up (~9216 bytes/stage extra) that\n"
    "        # _compute_stages() doesn't count, so ab_stage=3 → 118784 bytes which\n"
    "        # exceeds SM121a's 99 KB shared-memory limit.  Cap the budget so\n"
    "        # _compute_stages returns ab_stage=2 (actual usage 91136 bytes < 99 KB).\n"
    "        self.smem_capacity = 73728 if self.is_gated else _raw_smem"
)
assert old in src, "patch target not found — source may have changed"
p.write_text(src.replace(old, new, 1))
print("patched moe_dynamic_kernel.py")
EOF

# Fixes 3-5: replace patched vLLM Python files.
# Each file carries inline comments explaining its changes.
#
# kv_cache_utils.py:
#   Fix 3a — unify_kv_cache_spec_page_size: scale page_size_padded by ratio
#             when unifying DFlash draft (2x page size) with main model pages.
#   Fix 3b — unify_hybrid_kv_cache_specs: preserve sliding_window when converting
#             SlidingWindowSpec → FullAttentionSpec for the DFlash draft group.
#
# flashinfer.py:
#   Fix 4a — FlashInferMetadataBuilder.__init__: when DFlash draft group has mixed
#             window_lefts (SWA=4095, full-attn=-1), normalise to -1 (full attention)
#             instead of raising ValueError.
#   Fix 4b — FlashInferImpl.forward(): allow full-attention layers (window_left=-1)
#             to accept any wrapper window_left, covering the edge case where 4a
#             doesn't normalise (all layers share the same merged spec).
#
# dflash.py:
#   Fix 5  — DFlashProposer: allow_multiple_draft_kv_cache_groups returns True;
#             set_draft_block_table guards on correct gid; _create_draft_vllm_config
#             strips BACKED_SIZE_OBLIVIOUS and multimodal flag; slot-mapping buffers
#             allocated per KV cache group.
COPY patches/kv_cache_utils.py /usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
COPY patches/flashinfer.py /usr/local/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py
COPY patches/dflash.py /usr/local/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py
