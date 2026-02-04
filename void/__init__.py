"""
VOID (FlashSparse) - Cache-aware block tiling for sparse matrix operations.

This package provides:
- VOIDTensor: Efficient sparse matrix format using dense tiles with Morton ordering
- void_spmm/void_spmv: Triton kernels for sparse-dense multiplication
- sparse_attention: FlashAttention-style sparse attention
- VOIDSpMM/SparseLinear: Autograd-enabled modules for training
- void_spmm_stream_k: Load-balanced SpMM for power-law distributions

2025/2026 SOTA Features:
- FP8 support (FlashAttention-3 style)
- Data-affinity reordering (Acc-SpMM style)
- Enhanced pipelining with 3-5 stages
- Operation fusion framework (SpMM + activation)
- Dynamic kernel selection
- Hybrid core scheduling (Hopper+)
- Tensor Core configuration and alignment
- 2:4 Structured sparsity (NVIDIA Sparse Tensor Cores)
- INT8/INT4 quantization support
- Dynamic sparsity patterns
- Multi-GPU distributed computation
"""

from .format import VOIDTensor, csr_to_void, dense_to_void
from .ops import void_spmm, void_spmv, void_spmm_autotuned, void_spmm_pipelined, compute_optimal_tile_n, MIN_TENSOR_CORE_K
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
    sparse_attention_pipelined,
    local_attention,
    block_sparse_attention,
    SparseAttentionMask,
    create_local_attention_mask,
    create_causal_local_mask,
    create_strided_attention_mask,
    create_block_sparse_mask,
)

# Tensor Core configuration (2026 SOTA)
from .tensor_core import (
    TCPrecision,
    TensorCoreConfig,
    TCAlignmentResult,
    check_tensor_core_alignment,
    get_optimal_tc_tile_size,
    should_use_tensor_cores as tc_should_use_tensor_cores,
    get_gpu_tensor_core_info,
    get_triton_tc_config,
)

# 2:4 Structured sparsity (2026 SOTA)
from .structured import (
    PruneMethod,
    StructuredSparsityMetadata,
    VOIDStructuredTensor,
    check_2_4_compatible,
    prune_to_2_4,
    compress_2_4,
    decompress_2_4,
    compress_2_4_vectorized,
    decompress_2_4_vectorized,
    void_to_structured,
    structured_to_dense,
    get_structured_sparsity_info,
)

# INT8/INT4 quantization (2026 SOTA)
from .int_quant import (
    IntFormat,
    ScaleMode,
    IntQuantConfig,
    QuantizationParams,
    Int8VOIDTensor,
    quantize_tensor,
    dequantize_tensor,
    quantize_void_tensor,
    dequantize_void_tensor,
    void_spmm_int8,
    get_quantization_error,
)

# Dynamic sparsity (2026 SOTA)
from .dynamic import (
    UpdateStrategy,
    DynamicVOIDTensor,
    DynamicAttentionConfig,
    create_dynamic_void_tensor,
    from_void_tensor,
    update_from_topk_scores,
    update_from_mask,
    update_from_threshold,
    compute_topk_attention_mask,
    dynamic_topk_attention,
    progressive_sparsification,
)

# Multi-GPU distributed (2026 SOTA)
from .distributed import (
    ShardingStrategy,
    DistributedVOIDTensor,
    PipelineStage,
    is_distributed,
    get_world_size,
    get_rank,
    shard_void_tensor,
    distributed_void_spmm,
    gather_void_tensor,
    broadcast_void_tensor,
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
    "void_spmm_pipelined",
    "void_spmm_with_autotune",
    "compute_optimal_tile_n",
    "MIN_TENSOR_CORE_K",
    # Attention
    "sparse_attention",
    "sparse_attention_pipelined",
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
    # Tensor Core configuration (2026 SOTA)
    "TCPrecision",
    "TensorCoreConfig",
    "TCAlignmentResult",
    "check_tensor_core_alignment",
    "get_optimal_tc_tile_size",
    "tc_should_use_tensor_cores",
    "get_gpu_tensor_core_info",
    "get_triton_tc_config",
    # 2:4 Structured sparsity (2026 SOTA)
    "PruneMethod",
    "StructuredSparsityMetadata",
    "VOIDStructuredTensor",
    "check_2_4_compatible",
    "prune_to_2_4",
    "compress_2_4",
    "decompress_2_4",
    "compress_2_4_vectorized",
    "decompress_2_4_vectorized",
    "void_to_structured",
    "structured_to_dense",
    "get_structured_sparsity_info",
    # INT8/INT4 quantization (2026 SOTA)
    "IntFormat",
    "ScaleMode",
    "IntQuantConfig",
    "QuantizationParams",
    "Int8VOIDTensor",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_void_tensor",
    "dequantize_void_tensor",
    "void_spmm_int8",
    "get_quantization_error",
    # Dynamic sparsity (2026 SOTA)
    "UpdateStrategy",
    "DynamicVOIDTensor",
    "DynamicAttentionConfig",
    "create_dynamic_void_tensor",
    "from_void_tensor",
    "update_from_topk_scores",
    "update_from_mask",
    "update_from_threshold",
    "compute_topk_attention_mask",
    "dynamic_topk_attention",
    "progressive_sparsification",
    # Multi-GPU distributed (2026 SOTA)
    "ShardingStrategy",
    "DistributedVOIDTensor",
    "PipelineStage",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "shard_void_tensor",
    "distributed_void_spmm",
    "gather_void_tensor",
    "broadcast_void_tensor",
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
]

__version__ = "0.3.0"  # 2026 SOTA release with structured sparsity, INT8, dynamic, and distributed
