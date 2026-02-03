"""
VOID (FlashSparse) - Cache-aware block tiling for sparse matrix operations.

This package provides:
- VOIDTensor: Efficient sparse matrix format using dense tiles with Morton ordering
- void_spmm/void_spmv: Triton kernels for sparse-dense multiplication
- sparse_attention: FlashAttention-style sparse attention
- VOIDSpMM/SparseLinear: Autograd-enabled modules for training
- void_spmm_stream_k: Load-balanced SpMM for power-law distributions
"""

from .format import VOIDTensor, csr_to_void, dense_to_void
from .ops import void_spmm, void_spmv, void_spmm_autotuned
from .autotune import void_spmm_with_autotune, KernelConfig, clear_autotune_cache
from .adaptive import (
    csr_to_void_adaptive,
    select_adaptive_tile_size,
    analyze_sparsity_pattern,
    TileSizeMetrics,
)

# Attention
from .attention import (
    sparse_attention,
    local_attention,
    block_sparse_attention,
    SparseAttentionMask,
    create_local_attention_mask,
    create_causal_local_mask,
    create_strided_attention_mask,
    create_block_sparse_mask,
)

# Autograd
from .autograd import (
    VOIDSpMM,
    SparseLinear,
    sparse_attention_with_grad,
)

# Stream-K load balancing
from .stream_k import (
    void_spmm_stream_k,
    void_spmm_work_stealing,
    compute_stream_k_workload,
    analyze_workload_balance,
)

__all__ = [
    # Core format
    "VOIDTensor",
    "csr_to_void",
    "csr_to_void_adaptive",
    "dense_to_void",
    # Basic ops
    "void_spmm",
    "void_spmv",
    "void_spmm_autotuned",
    "void_spmm_with_autotune",
    # Attention
    "sparse_attention",
    "local_attention",
    "block_sparse_attention",
    "SparseAttentionMask",
    "create_local_attention_mask",
    "create_causal_local_mask",
    "create_strided_attention_mask",
    "create_block_sparse_mask",
    # Autograd
    "VOIDSpMM",
    "SparseLinear",
    "sparse_attention_with_grad",
    # Stream-K
    "void_spmm_stream_k",
    "void_spmm_work_stealing",
    "compute_stream_k_workload",
    "analyze_workload_balance",
    # Autotuning
    "KernelConfig",
    "clear_autotune_cache",
    # Adaptive tile selection
    "select_adaptive_tile_size",
    "analyze_sparsity_pattern",
    "TileSizeMetrics",
]

__version__ = "0.1.0"
