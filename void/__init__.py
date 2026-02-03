"""
VOID (FlashSparse) - Cache-aware block tiling for sparse matrix operations.

This package provides:
- VOIDTensor: Efficient sparse matrix format using dense tiles with Morton ordering
- void_spmm/void_spmv: Triton kernels for sparse-dense multiplication
- sparse_attention: FlashAttention-style sparse attention
- VOIDSpMM/SparseLinear: Autograd-enabled modules for training
- void_spmm_stream_k: Load-balanced SpMM for power-law distributions

2025 SOTA Features:
- FP8 support (FlashAttention-3 style)
- Data-affinity reordering (Acc-SpMM style)
- Enhanced pipelining with 3-5 stages
- Operation fusion framework (SpMM + activation)
- Dynamic kernel selection
- Hybrid core scheduling (Hopper+)
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

# FP8 support (2025 SOTA)
from .fp8 import (
    FP8Format,
    FP8Config,
    FP8ScalingInfo,
    FP8VOIDTensor,
    quantize_to_fp8,
    dequantize_from_fp8,
    void_spmm_fp8,
    void_tensor_to_fp8,
)

# Block reordering (2025 SOTA)
from .reorder import (
    OrderingStrategy,
    AffinityInfo,
    ReorderingResult,
    reorder_blocks,
    compute_block_affinity,
    analyze_ordering_quality,
    reorder_for_spmm,
    reorder_to_row_major,
    reorder_to_hilbert,
)

# Operation fusion (2025 SOTA)
from .fusion import (
    FusedOpType,
    ActivationType,
    FusedSparseOp,
    void_spmm_gelu,
    void_spmm_relu,
    void_spmm_silu,
    fused_sparse_mlp,
    FusedSparseLinear,
    create_fused_spmm_gelu,
    create_fused_spmm_relu,
    create_sparse_mlp,
)

# Dynamic dispatch (2025 SOTA)
from .dispatch import (
    KernelVariant,
    DispatchDecision,
    WorkloadCharacteristics,
    void_spmm_auto,
    get_recommended_kernel,
    analyze_dispatch_decision,
    KernelDispatcher,
)

# Hybrid scheduling (2025 SOTA)
from .scheduler import (
    CoreType,
    SchedulingStrategy,
    WorkloadProfile,
    WorkPartition,
    HybridScheduler,
    get_execution_plan,
    should_use_tensor_cores,
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
    # FP8 support (2025 SOTA)
    "FP8Format",
    "FP8Config",
    "FP8ScalingInfo",
    "FP8VOIDTensor",
    "quantize_to_fp8",
    "dequantize_from_fp8",
    "void_spmm_fp8",
    "void_tensor_to_fp8",
    # Block reordering (2025 SOTA)
    "OrderingStrategy",
    "AffinityInfo",
    "ReorderingResult",
    "reorder_blocks",
    "compute_block_affinity",
    "analyze_ordering_quality",
    "reorder_for_spmm",
    "reorder_to_row_major",
    "reorder_to_hilbert",
    # Operation fusion (2025 SOTA)
    "FusedOpType",
    "ActivationType",
    "FusedSparseOp",
    "void_spmm_gelu",
    "void_spmm_relu",
    "void_spmm_silu",
    "fused_sparse_mlp",
    "FusedSparseLinear",
    "create_fused_spmm_gelu",
    "create_fused_spmm_relu",
    "create_sparse_mlp",
    # Dynamic dispatch (2025 SOTA)
    "KernelVariant",
    "DispatchDecision",
    "WorkloadCharacteristics",
    "void_spmm_auto",
    "get_recommended_kernel",
    "analyze_dispatch_decision",
    "KernelDispatcher",
    # Hybrid scheduling (2025 SOTA)
    "CoreType",
    "SchedulingStrategy",
    "WorkloadProfile",
    "WorkPartition",
    "HybridScheduler",
    "get_execution_plan",
    "should_use_tensor_cores",
]

__version__ = "0.2.0"  # 2025 SOTA release
