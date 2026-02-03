"""
Adaptive Tile Size Selection for VOID

Automatically selects the optimal tile size for a given sparse matrix to:
1. Minimize padding overhead (wasted storage from partially-filled tiles)
2. Maximize block sparsity (fraction of all-zero tiles that can be skipped)
3. Balance computation efficiency vs memory overhead

The optimal tile size depends on the sparsity pattern:
- Highly clustered matrices → larger tiles (32x32 or 64x64)
- Scattered sparse matrices → smaller tiles (8x8 or 16x16)
- Block-structured matrices → match the block structure
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TileSizeMetrics:
    """Metrics for a candidate tile size."""
    tile_size: int
    block_sparsity: float  # Fraction of empty blocks (higher is better)
    overhead_ratio: float  # Stored elements / nnz (lower is better)
    score: float  # Combined score (higher is better)
    n_blocks: int
    nnz_in_blocks: int  # Total elements stored (including padding)


def estimate_block_sparsity(
    matrix: sp.csr_matrix,
    tile_size: int,
) -> float:
    """
    Estimate block sparsity without full conversion.

    Block sparsity = fraction of all-zero tiles.
    """
    M, N = matrix.shape

    # Calculate block grid dimensions
    n_block_rows = (M + tile_size - 1) // tile_size
    n_block_cols = (N + tile_size - 1) // tile_size
    total_blocks = n_block_rows * n_block_cols

    # Convert to COO for easier block identification
    coo = matrix.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data

    # Compute block indices for each non-zero
    block_row_idx = rows // tile_size
    block_col_idx = cols // tile_size

    # Find unique blocks
    block_keys = block_row_idx * n_block_cols + block_col_idx
    unique_blocks = np.unique(block_keys)
    n_active_blocks = len(unique_blocks)

    # Block sparsity = fraction of empty blocks
    block_sparsity = 1.0 - (n_active_blocks / max(total_blocks, 1))

    return block_sparsity


def estimate_overhead(
    matrix: sp.csr_matrix,
    tile_size: int,
) -> float:
    """
    Estimate storage overhead ratio.

    Overhead = total_stored_elements / original_nnz
    Values > 1.0 indicate padding overhead.
    """
    M, N = matrix.shape
    nnz = matrix.nnz

    if nnz == 0:
        return 0.0

    # Calculate block grid dimensions
    n_block_rows = (M + tile_size - 1) // tile_size
    n_block_cols = (N + tile_size - 1) // tile_size

    # Convert to COO
    coo = matrix.tocoo()
    rows, cols = coo.row, coo.col

    # Compute block indices
    block_row_idx = rows // tile_size
    block_col_idx = cols // tile_size

    # Find unique blocks
    block_keys = block_row_idx * n_block_cols + block_col_idx
    unique_blocks = np.unique(block_keys)
    n_blocks = len(unique_blocks)

    # Total stored elements (each block stores tile_size^2 elements)
    total_stored = n_blocks * tile_size * tile_size

    # Overhead ratio
    overhead = total_stored / max(nnz, 1)

    return overhead


def compute_tile_metrics(
    matrix: sp.csr_matrix,
    tile_size: int,
) -> TileSizeMetrics:
    """
    Compute comprehensive metrics for a tile size.

    Args:
        matrix: Sparse matrix in CSR format
        tile_size: Candidate tile size

    Returns:
        TileSizeMetrics with all metrics
    """
    block_sparsity = estimate_block_sparsity(matrix, tile_size)
    overhead = estimate_overhead(matrix, tile_size)

    # Compute score: favor high block sparsity and low overhead
    # Score formula: block_sparsity / (1 + overhead - 1)^2
    # This penalizes overhead quadratically while rewarding block sparsity
    if overhead < 1.0:
        overhead = 1.0  # Can't have overhead < 1.0

    overhead_penalty = (overhead - 1.0) + 1.0  # Shift so 1.0 overhead → no penalty
    score = block_sparsity / (overhead_penalty ** 1.5)

    # Get number of blocks
    M, N = matrix.shape
    n_block_rows = (M + tile_size - 1) // tile_size
    n_block_cols = (N + tile_size - 1) // tile_size

    coo = matrix.tocoo()
    block_row_idx = coo.row // tile_size
    block_col_idx = coo.col // tile_size
    block_keys = block_row_idx * n_block_cols + block_col_idx
    n_blocks = len(np.unique(block_keys))

    nnz_in_blocks = n_blocks * tile_size * tile_size

    return TileSizeMetrics(
        tile_size=tile_size,
        block_sparsity=block_sparsity,
        overhead_ratio=overhead,
        score=score,
        n_blocks=n_blocks,
        nnz_in_blocks=nnz_in_blocks,
    )


def select_adaptive_tile_size(
    matrix: sp.csr_matrix,
    candidate_sizes: Optional[List[int]] = None,
    max_overhead: float = 1.5,
    verbose: bool = False,
) -> int:
    """
    Automatically select the best tile size for a sparse matrix.

    Args:
        matrix: Sparse matrix in CSR format
        candidate_sizes: List of tile sizes to consider (default: [8, 16, 32, 64])
        max_overhead: Maximum acceptable overhead ratio (default: 1.5 = 50% padding)
        verbose: Print analysis (default: False)

    Returns:
        Optimal tile size

    Algorithm:
        1. Compute metrics for each candidate tile size
        2. Filter out sizes with overhead > max_overhead
        3. Select the size with highest score (block_sparsity / overhead_penalty)
    """
    if candidate_sizes is None:
        candidate_sizes = [8, 16, 32, 64]

    if not sp.isspmatrix_csr(matrix):
        matrix = sp.csr_matrix(matrix)

    # Compute metrics for all candidates
    metrics = []
    for tile_size in candidate_sizes:
        m = compute_tile_metrics(matrix, tile_size)
        metrics.append(m)

    if verbose:
        print(f"\nAdaptive Tile Size Selection")
        print(f"Matrix: {matrix.shape}, nnz: {matrix.nnz}, sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.1%}")
        print(f"{'Tile':>6} {'Blocks':>8} {'BlkSparse':>10} {'Overhead':>10} {'Score':>10}")
        print("-" * 56)
        for m in metrics:
            print(f"{m.tile_size:>6} {m.n_blocks:>8} {m.block_sparsity:>10.1%} "
                  f"{m.overhead_ratio:>10.2f}x {m.score:>10.4f}")

    # Filter by max overhead
    valid_metrics = [m for m in metrics if m.overhead_ratio <= max_overhead]

    if not valid_metrics:
        # All sizes exceed overhead limit, choose smallest (least overhead)
        if verbose:
            print(f"\nWarning: All tile sizes exceed max_overhead={max_overhead:.1f}x")
            print(f"Choosing smallest tile size to minimize overhead")
        return min(metrics, key=lambda m: m.overhead_ratio).tile_size

    # Select best score among valid options
    best = max(valid_metrics, key=lambda m: m.score)

    if verbose:
        print(f"\nSelected: {best.tile_size}x{best.tile_size} "
              f"(score: {best.score:.4f}, overhead: {best.overhead_ratio:.2f}x, "
              f"block sparsity: {best.block_sparsity:.1%})")

    return best.tile_size


def analyze_sparsity_pattern(
    matrix: sp.csr_matrix,
    tile_size: int = 32,
) -> dict:
    """
    Analyze the sparsity pattern of a matrix.

    Returns insights about whether the matrix is suitable for VOID format.

    Args:
        matrix: Sparse matrix
        tile_size: Tile size for analysis (default: 32)

    Returns:
        Dictionary with analysis results
    """
    M, N = matrix.shape
    nnz = matrix.nnz
    density = nnz / (M * N) if M * N > 0 else 0
    sparsity = 1.0 - density

    metrics = compute_tile_metrics(matrix, tile_size)

    # Compute clustering metric: how clustered are non-zeros?
    # Higher clustering → better for VOID
    # Measure: compare block sparsity to element sparsity
    # If block_sparsity >> element_sparsity, non-zeros are clustered
    clustering_ratio = metrics.block_sparsity / max(sparsity, 0.01)

    # Recommendation
    if metrics.overhead_ratio > 2.0:
        recommendation = "NOT RECOMMENDED - High overhead (>2x). Consider CSR format."
        suitable = False
    elif metrics.block_sparsity < 0.3:
        recommendation = "MARGINAL - Low block sparsity. VOID may not help much."
        suitable = False
    elif clustering_ratio < 0.5:
        recommendation = "MARGINAL - Non-zeros are scattered. Consider smaller tile size."
        suitable = False
    else:
        recommendation = "RECOMMENDED - Good candidate for VOID format."
        suitable = True

    return {
        "suitable_for_void": suitable,
        "recommendation": recommendation,
        "element_sparsity": sparsity,
        "block_sparsity": metrics.block_sparsity,
        "overhead_ratio": metrics.overhead_ratio,
        "clustering_ratio": clustering_ratio,
        "n_blocks": metrics.n_blocks,
        "tile_size": tile_size,
        "matrix_shape": matrix.shape,
        "nnz": nnz,
    }


# =============================================================================
# Enhanced csr_to_void with Adaptive Tile Size
# =============================================================================

def csr_to_void_adaptive(
    matrix: sp.csr_matrix,
    target_overhead: float = 1.3,
    candidate_sizes: Optional[List[int]] = None,
    dtype = None,
    device: str = 'cpu',
    verbose: bool = False,
):
    """
    Convert CSR matrix to VOID format with adaptive tile size selection.

    This is a drop-in replacement for csr_to_void that automatically
    chooses the best tile size.

    Args:
        matrix: Sparse matrix in CSR format
        target_overhead: Target maximum overhead (default: 1.3 = 30% padding)
        candidate_sizes: Tile sizes to consider (default: [8, 16, 32, 64])
        dtype: Output dtype (default: float32)
        device: Output device (default: 'cpu')
        verbose: Print selection process (default: False)

    Returns:
        VOIDTensor with automatically selected tile size
    """
    import torch
    from .format import csr_to_void

    if dtype is None:
        dtype = torch.float32

    # Select optimal tile size
    tile_size = select_adaptive_tile_size(
        matrix,
        candidate_sizes=candidate_sizes,
        max_overhead=target_overhead,
        verbose=verbose
    )

    # Convert using selected tile size
    return csr_to_void(matrix, tile_size=tile_size, dtype=dtype, device=device)
