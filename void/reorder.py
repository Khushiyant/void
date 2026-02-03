"""
Data-Affinity Block Reordering for VOID Sparse Matrices

Implements block reordering strategies inspired by Acc-SpMM for improved
cache locality and memory access patterns.

Key strategies:
- Morton (Z-order): Default VOID ordering for 2D spatial locality
- Row-major: Simple row-major ordering
- Affinity: Data-affinity based reordering (Acc-SpMM style)
- Hilbert: Hilbert curve for better cache locality than Morton

Reference: Acc-SpMM paper for affinity-based reordering concepts.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from enum import Enum


class OrderingStrategy(Enum):
    """Block ordering strategies for VOID sparse matrices."""
    MORTON = "morton"       # Z-order curve (default VOID)
    ROW_MAJOR = "row_major"  # Simple row-major
    COL_MAJOR = "col_major"  # Column-major
    AFFINITY = "affinity"   # Data-affinity based (Acc-SpMM style)
    HILBERT = "hilbert"     # Hilbert curve


@dataclass
class AffinityInfo:
    """Information about block affinity relationships.

    Used for affinity-based reordering that groups blocks accessed together.
    """
    # Permutations for rows and columns
    row_permutation: torch.Tensor  # [n_rows] maps original row -> new row
    col_permutation: torch.Tensor  # [n_cols] maps original col -> new col

    # Inverse permutations for reverse mapping
    row_inv_perm: torch.Tensor
    col_inv_perm: torch.Tensor

    # Affinity scores between blocks
    affinity_scores: Optional[torch.Tensor] = None  # [n_blocks, n_blocks] or None

    # Statistics
    locality_score: float = 0.0  # Higher = better locality


@dataclass
class ReorderingResult:
    """Result of block reordering operation."""
    # Reordered VOID tensor
    void_tensor: 'VOIDTensor'

    # Mapping information
    original_to_new: torch.Tensor  # [n_blocks] old block idx -> new block idx
    new_to_original: torch.Tensor  # [n_blocks] new block idx -> old block idx

    # Strategy used
    strategy: OrderingStrategy

    # Locality metrics
    locality_score: float = 0.0


# =============================================================================
# Morton (Z-order) Encoding - Already in format.py but included for completeness
# =============================================================================

def morton_encode(x: int, y: int) -> int:
    """Encode 2D coordinates into Morton code (Z-order curve)."""
    def spread_bits(v: int) -> int:
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    return spread_bits(x) | (spread_bits(y) << 1)


def morton_encode_batch(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Vectorized Morton encoding."""
    def spread_bits_vec(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64)
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    return spread_bits_vec(rows) | (spread_bits_vec(cols) << 1)


# =============================================================================
# Hilbert Curve Encoding
# =============================================================================

def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert curve index to 2D coordinates.

    Args:
        n: Size of grid (must be power of 2)
        d: Hilbert curve index

    Returns:
        (x, y) coordinates
    """
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def hilbert_xy2d(n: int, x: int, y: int) -> int:
    """Convert 2D coordinates to Hilbert curve index.

    Args:
        n: Size of grid (must be power of 2)
        x, y: Coordinates

    Returns:
        Hilbert curve index
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_encode_batch(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Vectorized Hilbert curve encoding.

    Args:
        rows: Row coordinates
        cols: Column coordinates
        n: Grid size (will be rounded up to power of 2)

    Returns:
        Hilbert curve indices
    """
    # Round n up to power of 2
    n = int(2 ** np.ceil(np.log2(max(n, 1))))

    result = np.zeros(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        result[i] = hilbert_xy2d(n, int(rows[i]), int(cols[i]))
    return result


# =============================================================================
# Affinity-Based Reordering (Acc-SpMM Style)
# =============================================================================

def compute_block_affinity(
    void_tensor: 'VOIDTensor',
    dense_cols: int,
) -> AffinityInfo:
    """Compute block affinity for reordering.

    Analyzes which blocks are likely to be accessed together during SpMM
    and computes optimal row/column permutations.

    Args:
        void_tensor: Source VOID tensor
        dense_cols: Number of columns in dense matrix B (affects access patterns)

    Returns:
        AffinityInfo with computed permutations
    """
    n_block_rows, n_block_cols = void_tensor.block_grid
    device = void_tensor.block_rows.device

    block_rows = void_tensor.block_rows.cpu().numpy()
    block_cols = void_tensor.block_cols.cpu().numpy()

    # Build adjacency: which rows share column blocks
    # Rows that access the same K-blocks have affinity
    row_to_cols = {}
    for br, bc in zip(block_rows, block_cols):
        if br not in row_to_cols:
            row_to_cols[br] = set()
        row_to_cols[br].add(bc)

    # Compute row affinity matrix (Jaccard similarity of column access)
    unique_rows = sorted(row_to_cols.keys())
    n_unique = len(unique_rows)

    if n_unique <= 1:
        # Trivial case: identity permutation
        row_perm = torch.arange(n_block_rows, dtype=torch.int64, device=device)
        col_perm = torch.arange(n_block_cols, dtype=torch.int64, device=device)
        return AffinityInfo(
            row_permutation=row_perm,
            col_permutation=col_perm,
            row_inv_perm=row_perm,
            col_inv_perm=col_perm,
            locality_score=1.0,
        )

    # Greedy reordering: place rows with high affinity adjacent
    # Use nearest-neighbor heuristic starting from row with most blocks
    row_block_counts = {r: len(cols) for r, cols in row_to_cols.items()}
    start_row = max(row_block_counts.keys(), key=lambda r: row_block_counts[r])

    ordered_rows = [start_row]
    remaining = set(unique_rows) - {start_row}

    while remaining:
        last_row = ordered_rows[-1]
        last_cols = row_to_cols[last_row]

        # Find most similar remaining row (Jaccard)
        best_row = None
        best_sim = -1

        for r in remaining:
            r_cols = row_to_cols[r]
            intersection = len(last_cols & r_cols)
            union = len(last_cols | r_cols)
            sim = intersection / union if union > 0 else 0

            if sim > best_sim:
                best_sim = sim
                best_row = r

        ordered_rows.append(best_row)
        remaining.remove(best_row)

    # Create row permutation (maps old row -> new position)
    row_perm = torch.arange(n_block_rows, dtype=torch.int64, device=device)
    row_inv_perm = torch.arange(n_block_rows, dtype=torch.int64, device=device)

    for new_idx, old_idx in enumerate(ordered_rows):
        row_perm[old_idx] = new_idx
        row_inv_perm[new_idx] = old_idx

    # For columns: order by access frequency (most accessed first)
    col_counts = {}
    for bc in block_cols:
        col_counts[bc] = col_counts.get(bc, 0) + 1

    unique_cols = sorted(col_counts.keys(), key=lambda c: col_counts[c], reverse=True)

    col_perm = torch.arange(n_block_cols, dtype=torch.int64, device=device)
    col_inv_perm = torch.arange(n_block_cols, dtype=torch.int64, device=device)

    for new_idx, old_idx in enumerate(unique_cols):
        col_perm[old_idx] = new_idx
        col_inv_perm[new_idx] = old_idx

    # Compute locality score (average distance between consecutively accessed blocks)
    locality_score = compute_locality_score(ordered_rows, row_to_cols)

    return AffinityInfo(
        row_permutation=row_perm,
        col_permutation=col_perm,
        row_inv_perm=row_inv_perm,
        col_inv_perm=col_inv_perm,
        locality_score=locality_score,
    )


def compute_locality_score(ordered_rows: List[int], row_to_cols: dict) -> float:
    """Compute locality score for a row ordering.

    Higher score = better locality (adjacent rows share more columns).
    """
    if len(ordered_rows) <= 1:
        return 1.0

    total_sim = 0.0
    for i in range(len(ordered_rows) - 1):
        r1, r2 = ordered_rows[i], ordered_rows[i + 1]
        cols1, cols2 = row_to_cols[r1], row_to_cols[r2]
        intersection = len(cols1 & cols2)
        union = len(cols1 | cols2)
        total_sim += intersection / union if union > 0 else 0

    return total_sim / (len(ordered_rows) - 1)


# =============================================================================
# Main Reordering Function
# =============================================================================

def reorder_blocks(
    void_tensor: 'VOIDTensor',
    strategy: OrderingStrategy = OrderingStrategy.MORTON,
    dense_cols: Optional[int] = None,
    **kwargs,
) -> ReorderingResult:
    """Reorder blocks in a VOID tensor according to specified strategy.

    Args:
        void_tensor: Source VOID tensor
        strategy: Ordering strategy to use
        dense_cols: For AFFINITY strategy, number of columns in dense B matrix
        **kwargs: Additional strategy-specific arguments

    Returns:
        ReorderingResult with reordered tensor and mapping info
    """
    from .format import VOIDTensor

    n_blocks = void_tensor.n_blocks
    if n_blocks == 0:
        return ReorderingResult(
            void_tensor=void_tensor,
            original_to_new=torch.empty(0, dtype=torch.int64),
            new_to_original=torch.empty(0, dtype=torch.int64),
            strategy=strategy,
            locality_score=1.0,
        )

    device = void_tensor.values.device
    block_rows = void_tensor.block_rows.cpu().numpy()
    block_cols = void_tensor.block_cols.cpu().numpy()

    # Compute ordering based on strategy
    if strategy == OrderingStrategy.MORTON:
        # Z-order curve (already default in VOID)
        codes = morton_encode_batch(block_rows, block_cols)
        sort_order = np.argsort(codes)
        locality_score = 1.0  # Baseline

    elif strategy == OrderingStrategy.ROW_MAJOR:
        # Simple row-major: sort by (row, col)
        n_cols = void_tensor.block_grid[1]
        codes = block_rows * n_cols + block_cols
        sort_order = np.argsort(codes)
        locality_score = 0.8  # Generally good for row-wise access

    elif strategy == OrderingStrategy.COL_MAJOR:
        # Column-major: sort by (col, row)
        n_rows = void_tensor.block_grid[0]
        codes = block_cols * n_rows + block_rows
        sort_order = np.argsort(codes)
        locality_score = 0.7  # Good for column-wise access

    elif strategy == OrderingStrategy.HILBERT:
        # Hilbert curve for better locality than Morton
        n = max(void_tensor.block_grid)
        codes = hilbert_encode_batch(block_rows, block_cols, n)
        sort_order = np.argsort(codes)
        locality_score = 1.1  # Slightly better than Morton typically

    elif strategy == OrderingStrategy.AFFINITY:
        # Data-affinity based reordering
        if dense_cols is None:
            dense_cols = void_tensor.shape[1]  # Default: assume square

        affinity_info = compute_block_affinity(void_tensor, dense_cols)

        # Apply row/col permutations to get new block coordinates
        row_perm = affinity_info.row_permutation.cpu().numpy()
        col_perm = affinity_info.col_permutation.cpu().numpy()

        new_block_rows = row_perm[block_rows]
        new_block_cols = col_perm[block_cols]

        # Sort by new row-major order
        n_cols = void_tensor.block_grid[1]
        codes = new_block_rows * n_cols + new_block_cols
        sort_order = np.argsort(codes)

        locality_score = affinity_info.locality_score

    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")

    # Apply reordering
    sort_order_tensor = torch.tensor(sort_order, dtype=torch.int64, device=device)
    inverse_order = torch.argsort(sort_order_tensor)

    # Reorder all block data
    new_values = void_tensor.values[sort_order_tensor]
    new_block_rows = void_tensor.block_rows[sort_order_tensor]
    new_block_cols = void_tensor.block_cols[sort_order_tensor]

    # Recompute Morton codes for new ordering
    new_morton = torch.tensor(
        morton_encode_batch(
            new_block_rows.cpu().numpy(),
            new_block_cols.cpu().numpy()
        ),
        dtype=torch.int64,
        device=device
    )

    # Create new VOIDTensor
    new_void_tensor = VOIDTensor(
        values=new_values,
        block_rows=new_block_rows,
        block_cols=new_block_cols,
        morton_codes=new_morton,
        shape=void_tensor.shape,
        tile_size=void_tensor.tile_size,
        nnz_original=void_tensor.nnz_original,
        n_blocks=void_tensor.n_blocks,
        density=void_tensor.density,
        dtype=void_tensor.dtype,
    )

    return ReorderingResult(
        void_tensor=new_void_tensor,
        original_to_new=sort_order_tensor,
        new_to_original=inverse_order,
        strategy=strategy,
        locality_score=locality_score,
    )


def analyze_ordering_quality(void_tensor: 'VOIDTensor') -> dict:
    """Analyze the quality of current block ordering.

    Returns metrics about spatial locality and access patterns.
    """
    n_blocks = void_tensor.n_blocks
    if n_blocks <= 1:
        return {
            "n_blocks": n_blocks,
            "mean_neighbor_distance": 0.0,
            "max_row_gap": 0,
            "max_col_gap": 0,
            "row_locality_score": 1.0,
        }

    block_rows = void_tensor.block_rows.cpu().numpy()
    block_cols = void_tensor.block_cols.cpu().numpy()

    # Compute distances between consecutive blocks
    row_diffs = np.abs(np.diff(block_rows))
    col_diffs = np.abs(np.diff(block_cols))

    # Euclidean distance in block space
    distances = np.sqrt(row_diffs**2 + col_diffs**2)

    # Row locality: how often consecutive blocks are in same/adjacent rows
    same_row = np.sum(row_diffs == 0)
    adjacent_row = np.sum(row_diffs == 1)
    row_locality = (same_row + 0.5 * adjacent_row) / max(n_blocks - 1, 1)

    return {
        "n_blocks": n_blocks,
        "mean_neighbor_distance": float(np.mean(distances)),
        "max_neighbor_distance": float(np.max(distances)),
        "max_row_gap": int(np.max(row_diffs)),
        "max_col_gap": int(np.max(col_diffs)),
        "row_locality_score": float(row_locality),
        "same_row_fraction": float(same_row / max(n_blocks - 1, 1)),
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def reorder_for_spmm(
    void_tensor: 'VOIDTensor',
    b_cols: int,
) -> 'VOIDTensor':
    """Reorder blocks optimally for SpMM with dense matrix of given width.

    Uses affinity-based reordering for best SpMM performance.

    Args:
        void_tensor: Source tensor
        b_cols: Number of columns in dense B matrix

    Returns:
        Reordered VOIDTensor
    """
    result = reorder_blocks(
        void_tensor,
        strategy=OrderingStrategy.AFFINITY,
        dense_cols=b_cols,
    )
    return result.void_tensor


def reorder_to_row_major(void_tensor: 'VOIDTensor') -> 'VOIDTensor':
    """Reorder blocks to simple row-major ordering.

    Args:
        void_tensor: Source tensor

    Returns:
        Reordered VOIDTensor
    """
    result = reorder_blocks(void_tensor, strategy=OrderingStrategy.ROW_MAJOR)
    return result.void_tensor


def reorder_to_hilbert(void_tensor: 'VOIDTensor') -> 'VOIDTensor':
    """Reorder blocks using Hilbert curve for improved locality.

    Args:
        void_tensor: Source tensor

    Returns:
        Reordered VOIDTensor
    """
    result = reorder_blocks(void_tensor, strategy=OrderingStrategy.HILBERT)
    return result.void_tensor


# Export public API
__all__ = [
    # Enums
    "OrderingStrategy",
    # Dataclasses
    "AffinityInfo",
    "ReorderingResult",
    # Main functions
    "reorder_blocks",
    "compute_block_affinity",
    "analyze_ordering_quality",
    # Convenience functions
    "reorder_for_spmm",
    "reorder_to_row_major",
    "reorder_to_hilbert",
    # Encoding utilities
    "hilbert_xy2d",
    "hilbert_d2xy",
    "hilbert_encode_batch",
]
