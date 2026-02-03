"""
Hybrid Core Scheduling for VOID Sparse Operations

Implements workload partitioning and scheduling strategies for modern GPUs
with heterogeneous compute capabilities (e.g., NVIDIA Hopper with tensor cores).

Key features:
- Workload profiling for memory vs compute intensity
- Partitioning for hybrid execution (tensor core + CUDA core)
- Regularity analysis for optimal kernel assignment

Note: Full benefits require Hopper-class GPUs or newer.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum


class CoreType(Enum):
    """Types of GPU cores for scheduling."""
    CUDA = "cuda"           # Standard CUDA/streaming multiprocessor cores
    TENSOR = "tensor"       # Tensor cores (Volta+)
    SPARSE_TENSOR = "sparse_tensor"  # Sparse tensor cores (Ampere+)


class SchedulingStrategy(Enum):
    """Scheduling strategies for hybrid execution."""
    CUDA_ONLY = "cuda_only"         # Use only CUDA cores
    TENSOR_ONLY = "tensor_only"     # Use only tensor cores
    HYBRID = "hybrid"               # Partition between core types
    ADAPTIVE = "adaptive"           # Dynamically adapt based on workload


@dataclass
class WorkloadProfile:
    """Profile of a sparse workload's characteristics.

    Used to determine optimal scheduling strategy.
    """
    # Memory characteristics
    memory_intensity: float  # Ratio of memory ops to compute ops (higher = memory bound)

    # Regularity characteristics
    regularity_score: float  # 0.0 = highly irregular, 1.0 = perfectly regular

    # Tensor core eligibility
    tensor_core_eligible: bool  # True if dimensions are multiples of tensor core sizes

    # Block statistics
    mean_blocks_per_row: float
    block_size_optimal: bool  # True if block size matches tensor core requirements

    # Recommended execution
    recommended_strategy: SchedulingStrategy
    estimated_efficiency: float  # 0.0-1.0, estimated hardware utilization


@dataclass
class WorkPartition:
    """Partition of work between different core types."""
    # Regular portion (suitable for tensor cores)
    regular_blocks: torch.Tensor  # Indices of regular blocks
    n_regular_blocks: int

    # Irregular portion (better on CUDA cores)
    irregular_blocks: torch.Tensor  # Indices of irregular blocks
    n_irregular_blocks: int

    # Partition ratio
    regular_fraction: float


def analyze_workload(
    void_tensor: 'VOIDTensor',
    b_shape: Optional[Tuple[int, int]] = None,
) -> WorkloadProfile:
    """Analyze workload characteristics for scheduling decisions.

    Args:
        void_tensor: VOID sparse matrix to analyze
        b_shape: Shape of dense matrix B (K, N) if known

    Returns:
        WorkloadProfile with computed characteristics
    """
    from .format import VOIDTensor

    M, K = void_tensor.shape
    N = b_shape[1] if b_shape else K
    tile_m, tile_k = void_tensor.tile_size
    n_blocks = void_tensor.n_blocks

    # Memory intensity: bytes loaded / FLOPs
    # For SpMM: Load A (n_blocks * tile_m * tile_k), B (K * N), store C (M * N)
    # FLOPs: 2 * n_blocks * tile_m * tile_k * N (multiply-add for each element)
    bytes_loaded = (n_blocks * tile_m * tile_k + K * N + M * N) * 4  # Assume FP32
    flops = 2 * n_blocks * tile_m * tile_k * N

    memory_intensity = bytes_loaded / max(flops, 1)

    # Regularity: analyze variance in blocks per row
    from .stream_k import analyze_workload_balance
    balance = analyze_workload_balance(void_tensor)

    if balance.get("empty", False):
        regularity = 1.0
    else:
        # Lower CoV = more regular
        cov = balance.get("coefficient_of_variation", 0.0)
        regularity = max(0.0, 1.0 - cov / 2.0)

    # Tensor core eligibility: dimensions should be multiples of 8 (for FP16) or 16 (for FP32)
    tensor_core_size = 16  # Conservative
    tc_eligible = (
        tile_m % tensor_core_size == 0 and
        tile_k % tensor_core_size == 0 and
        N % tensor_core_size == 0
    )

    # Block size optimal for tensor cores?
    optimal_sizes = {16, 32, 64, 128}
    block_optimal = tile_m in optimal_sizes and tile_k in optimal_sizes

    # Mean blocks per row
    n_block_rows = (M + tile_m - 1) // tile_m
    mean_blocks = n_blocks / max(n_block_rows, 1)

    # Determine recommended strategy
    if not tc_eligible:
        strategy = SchedulingStrategy.CUDA_ONLY
        efficiency = 0.6  # CUDA cores less efficient for dense-like ops
    elif memory_intensity > 2.0:
        # Memory bound - tensor cores won't help much
        strategy = SchedulingStrategy.CUDA_ONLY
        efficiency = 0.5
    elif regularity > 0.8:
        # Highly regular - tensor cores ideal
        strategy = SchedulingStrategy.TENSOR_ONLY
        efficiency = 0.9
    elif regularity > 0.5:
        # Mixed regularity - hybrid approach
        strategy = SchedulingStrategy.HYBRID
        efficiency = 0.75
    else:
        # Highly irregular - CUDA cores better
        strategy = SchedulingStrategy.CUDA_ONLY
        efficiency = 0.55

    return WorkloadProfile(
        memory_intensity=memory_intensity,
        regularity_score=regularity,
        tensor_core_eligible=tc_eligible,
        mean_blocks_per_row=mean_blocks,
        block_size_optimal=block_optimal,
        recommended_strategy=strategy,
        estimated_efficiency=efficiency,
    )


class HybridScheduler:
    """
    Hybrid scheduler for partitioning work between core types.

    Analyzes block patterns and divides work to maximize hardware utilization.
    """

    # Thresholds
    REGULARITY_THRESHOLD = 0.7  # Blocks with regularity above this go to tensor cores
    MIN_TENSOR_FRACTION = 0.1   # Minimum work fraction for tensor core path
    MIN_CUDA_FRACTION = 0.1     # Minimum work fraction for CUDA core path

    def __init__(self, prefer_tensor_cores: bool = True):
        """
        Initialize the hybrid scheduler.

        Args:
            prefer_tensor_cores: If True, bias towards tensor cores when possible
        """
        self.prefer_tensor_cores = prefer_tensor_cores

    def partition_work(
        self,
        void_tensor: 'VOIDTensor',
        profile: Optional[WorkloadProfile] = None,
    ) -> WorkPartition:
        """Partition work between core types.

        Identifies which blocks are suitable for tensor cores (regular patterns)
        versus CUDA cores (irregular patterns).

        Args:
            void_tensor: VOID sparse matrix
            profile: Pre-computed workload profile (optional)

        Returns:
            WorkPartition with block assignments
        """
        if profile is None:
            profile = analyze_workload(void_tensor)

        n_blocks = void_tensor.n_blocks
        device = void_tensor.values.device

        if n_blocks == 0:
            return WorkPartition(
                regular_blocks=torch.empty(0, dtype=torch.int64, device=device),
                n_regular_blocks=0,
                irregular_blocks=torch.empty(0, dtype=torch.int64, device=device),
                n_irregular_blocks=0,
                regular_fraction=0.0,
            )

        # Analyze block-level regularity
        block_regularity = self._compute_block_regularity(void_tensor)

        # Partition based on regularity scores
        regular_mask = block_regularity >= self.REGULARITY_THRESHOLD
        irregular_mask = ~regular_mask

        regular_blocks = torch.where(regular_mask)[0]
        irregular_blocks = torch.where(irregular_mask)[0]

        n_regular = len(regular_blocks)
        n_irregular = len(irregular_blocks)

        # Ensure minimum fractions
        regular_fraction = n_regular / max(n_blocks, 1)

        if regular_fraction < self.MIN_TENSOR_FRACTION and n_regular > 0:
            # Too few regular blocks, merge with irregular
            irregular_blocks = torch.arange(n_blocks, device=device, dtype=torch.int64)
            regular_blocks = torch.empty(0, dtype=torch.int64, device=device)
            n_regular, n_irregular = 0, n_blocks
            regular_fraction = 0.0

        elif regular_fraction > (1 - self.MIN_CUDA_FRACTION) and n_irregular > 0:
            # Too few irregular blocks, merge with regular
            regular_blocks = torch.arange(n_blocks, device=device, dtype=torch.int64)
            irregular_blocks = torch.empty(0, dtype=torch.int64, device=device)
            n_regular, n_irregular = n_blocks, 0
            regular_fraction = 1.0

        return WorkPartition(
            regular_blocks=regular_blocks,
            n_regular_blocks=n_regular,
            irregular_blocks=irregular_blocks,
            n_irregular_blocks=n_irregular,
            regular_fraction=regular_fraction,
        )

    def _compute_block_regularity(
        self,
        void_tensor: 'VOIDTensor',
    ) -> torch.Tensor:
        """Compute per-block regularity scores.

        Blocks in rows with consistent block counts are more "regular".

        Args:
            void_tensor: VOID sparse matrix

        Returns:
            Tensor of regularity scores [n_blocks], range [0, 1]
        """
        device = void_tensor.values.device
        n_blocks = void_tensor.n_blocks

        if n_blocks == 0:
            return torch.empty(0, dtype=torch.float32, device=device)

        # Count blocks per row
        n_block_rows = void_tensor.block_grid[0]
        row_counts = torch.zeros(n_block_rows, dtype=torch.float32, device=device)
        row_counts.scatter_add_(
            0,
            void_tensor.block_rows.long(),
            torch.ones(n_blocks, dtype=torch.float32, device=device)
        )

        # Compute mean and std
        non_empty_mask = row_counts > 0
        if non_empty_mask.sum() == 0:
            return torch.ones(n_blocks, dtype=torch.float32, device=device)

        mean_count = row_counts[non_empty_mask].mean()
        std_count = row_counts[non_empty_mask].std()

        if std_count < 1e-6:
            # All rows have same count - perfectly regular
            return torch.ones(n_blocks, dtype=torch.float32, device=device)

        # Regularity = how close each block's row count is to the mean
        # Normalize to [0, 1]
        block_row_counts = row_counts[void_tensor.block_rows.long()]
        deviation = torch.abs(block_row_counts - mean_count) / (std_count + 1e-6)
        regularity = torch.clamp(1.0 - deviation / 3.0, min=0.0, max=1.0)

        return regularity

    def get_execution_plan(
        self,
        void_tensor: 'VOIDTensor',
        b_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Get a detailed execution plan for a sparse operation.

        Args:
            void_tensor: VOID sparse matrix
            b_shape: Shape of dense matrix B (K, N)

        Returns:
            Dictionary with execution plan details
        """
        profile = analyze_workload(void_tensor, b_shape)
        partition = self.partition_work(void_tensor, profile)

        return {
            "strategy": profile.recommended_strategy.value,
            "profile": {
                "memory_intensity": f"{profile.memory_intensity:.3f}",
                "regularity_score": f"{profile.regularity_score:.2f}",
                "tensor_core_eligible": profile.tensor_core_eligible,
                "estimated_efficiency": f"{profile.estimated_efficiency:.0%}",
            },
            "partition": {
                "regular_blocks": partition.n_regular_blocks,
                "irregular_blocks": partition.n_irregular_blocks,
                "regular_fraction": f"{partition.regular_fraction:.1%}",
            },
            "recommendations": self._get_recommendations(profile, partition),
        }

    def _get_recommendations(
        self,
        profile: WorkloadProfile,
        partition: WorkPartition,
    ) -> List[str]:
        """Generate optimization recommendations."""
        recs = []

        if not profile.tensor_core_eligible:
            recs.append("Consider tile sizes that are multiples of 16 for tensor core usage")

        if profile.memory_intensity > 1.5:
            recs.append("Workload is memory-bound; consider FP16 or FP8 for bandwidth reduction")

        if profile.regularity_score < 0.5:
            recs.append("Irregular sparsity pattern; Stream-K may improve load balance")

        if partition.regular_fraction < 0.3:
            recs.append("Few regular blocks; CUDA-only execution recommended")

        if profile.estimated_efficiency < 0.6:
            recs.append("Low estimated efficiency; consider matrix reordering")

        if not recs:
            recs.append("Workload is well-suited for current configuration")

        return recs


def create_scheduler(prefer_tensor_cores: bool = True) -> HybridScheduler:
    """Create a hybrid scheduler instance.

    Args:
        prefer_tensor_cores: If True, prefer tensor cores when possible

    Returns:
        HybridScheduler instance
    """
    return HybridScheduler(prefer_tensor_cores=prefer_tensor_cores)


def get_execution_plan(
    void_tensor: 'VOIDTensor',
    b_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Get execution plan using default scheduler.

    Convenience function for quick analysis.

    Args:
        void_tensor: VOID sparse matrix
        b_shape: Shape of dense matrix B (K, N)

    Returns:
        Execution plan dictionary
    """
    scheduler = HybridScheduler()
    return scheduler.get_execution_plan(void_tensor, b_shape)


def should_use_tensor_cores(
    void_tensor: 'VOIDTensor',
    b_shape: Optional[Tuple[int, int]] = None,
) -> bool:
    """Quick check if tensor cores would be beneficial.

    Args:
        void_tensor: VOID sparse matrix
        b_shape: Shape of dense matrix B (K, N) if known

    Returns:
        True if tensor cores are recommended
    """
    profile = analyze_workload(void_tensor, b_shape)
    return profile.recommended_strategy in (
        SchedulingStrategy.TENSOR_ONLY,
        SchedulingStrategy.HYBRID,
    )


# Export public API
__all__ = [
    # Enums
    "CoreType",
    "SchedulingStrategy",
    # Dataclasses
    "WorkloadProfile",
    "WorkPartition",
    # Functions
    "analyze_workload",
    "create_scheduler",
    "get_execution_plan",
    "should_use_tensor_cores",
    # Classes
    "HybridScheduler",
]
