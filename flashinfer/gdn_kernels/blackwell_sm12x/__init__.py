"""
GDN Blackwell SM12x Kernels
===========================

CuTe-DSL chunked prefill kernel entry point for Gated Delta Net on SM12x
GPUs such as SM120 and SM121 / GB10.
"""

try:
    from .gdn_prefill import chunk_gated_delta_rule_sm12x
except (ImportError, RuntimeError):
    chunk_gated_delta_rule_sm12x = None  # type: ignore

# Enables the correctness-first SM12x recurrent prefill kernel. The first
# implementation is intentionally simple and should be replaced by a chunked
# warp-MMA path once correctness coverage is broad enough.
_has_sm12x_prefill = chunk_gated_delta_rule_sm12x is not None

__all__ = [
    "chunk_gated_delta_rule_sm12x",
    "_has_sm12x_prefill",
]
