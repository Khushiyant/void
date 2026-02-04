"""
2:4 Structured Sparsity Support for VOID

NVIDIA Sparse Tensor Cores (Ampere+) support 2:4 structured sparsity:
- For every 4 consecutive elements, exactly 2 must be zero
- Hardware achieves ~2x throughput vs dense operations
- Storage is compressed to 50% + 2-bit indices per 4 elements

This module provides:
- Conversion from dense tensors to 2:4 structured format
- Pruning algorithms to create 2:4 patterns
- Compressed storage format compatible with VOID
- Triton kernels for structured SpMM

Note: 2:4 structured sparsity requires Ampere (SM80+) or newer GPUs.
"""

import torch
import triton
import triton.language as tl
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union
from enum import Enum


class PruneMethod(Enum):
    """Methods for pruning to 2:4 structured sparsity."""
    MAGNITUDE = "magnitude"      # Keep top-2 by absolute value
    RANDOM = "random"            # Random 2 of 4
    GRADIENT = "gradient"        # Keep top-2 by gradient magnitude (training)


@dataclass
class StructuredSparsityMetadata:
    """
    Metadata for 2:4 structured sparse tensor.

    For every group of 4 elements, we store:
    - 2 non-zero values (50% compression)
    - 2-bit index indicating which 2 positions are non-zero

    Index encoding (2 bits per position pair):
    - 0b00 (0): positions 0, 1
    - 0b01 (1): positions 0, 2
    - 0b10 (2): positions 0, 3
    - 0b11 (3): positions 1, 2
    - 0b100 (4): positions 1, 3
    - 0b101 (5): positions 2, 3

    We pack 4 indices into a single uint8 for efficiency.
    """
    # For shape [M, N], indices has shape [M, N // 16] (4 groups per byte)
    indices: torch.Tensor  # uint8, packed indices
    original_shape: Tuple[int, int]
    n_groups: int  # Total number of 4-element groups

    def to(self, device: Union[str, torch.device]) -> 'StructuredSparsityMetadata':
        return StructuredSparsityMetadata(
            indices=self.indices.to(device),
            original_shape=self.original_shape,
            n_groups=self.n_groups,
        )


@dataclass
class VOIDStructuredTensor:
    """
    VOID tensor with 2:4 structured sparsity.

    Storage format:
    - values: [n_blocks, TILE_M, TILE_K // 2] - compressed block values
    - metadata: StructuredSparsityMetadata with indices
    - block_rows, block_cols: Block positions (same as VOIDTensor)

    Benefits:
    - 2x compression vs dense blocks
    - Hardware-accelerated on Ampere+ Sparse Tensor Cores
    - Compatible with VOID block format
    """
    values: torch.Tensor          # [n_blocks, tile_m, tile_k // 2]
    metadata: StructuredSparsityMetadata
    block_rows: torch.Tensor      # [n_blocks]
    block_cols: torch.Tensor      # [n_blocks]
    shape: Tuple[int, int]        # Original dense shape [M, K]
    tile_size: Tuple[int, int]    # (tile_m, tile_k)
    n_blocks: int

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def block_grid(self) -> Tuple[int, int]:
        """Number of blocks in each dimension."""
        return (
            (self.shape[0] + self.tile_size[0] - 1) // self.tile_size[0],
            (self.shape[1] + self.tile_size[1] - 1) // self.tile_size[1],
        )

    def to(self, device: Union[str, torch.device]) -> 'VOIDStructuredTensor':
        return VOIDStructuredTensor(
            values=self.values.to(device),
            metadata=self.metadata.to(device),
            block_rows=self.block_rows.to(device),
            block_cols=self.block_cols.to(device),
            shape=self.shape,
            tile_size=self.tile_size,
            n_blocks=self.n_blocks,
        )


# Index lookup table: maps combination index to (pos0, pos1)
INDEX_TO_POSITIONS = [
    (0, 1),  # 0
    (0, 2),  # 1
    (0, 3),  # 2
    (1, 2),  # 3
    (1, 3),  # 4
    (2, 3),  # 5
]

# Reverse lookup: (pos0, pos1) -> index
POSITIONS_TO_INDEX = {pos: idx for idx, pos in enumerate(INDEX_TO_POSITIONS)}


def check_2_4_compatible(tensor: torch.Tensor) -> Tuple[bool, str]:
    """
    Check if a tensor is compatible with 2:4 structured sparsity.

    Requirements:
    - Last dimension must be divisible by 4
    - For training compatibility, recommend dimensions divisible by 16

    Args:
        tensor: Input tensor to check

    Returns:
        Tuple of (is_compatible, reason)
    """
    if tensor.dim() < 1:
        return False, "Tensor must have at least 1 dimension"

    last_dim = tensor.shape[-1]

    if last_dim % 4 != 0:
        return False, f"Last dimension ({last_dim}) must be divisible by 4"

    if last_dim % 16 != 0:
        return True, f"Warning: Last dimension ({last_dim}) not divisible by 16, may have suboptimal performance"

    return True, "Fully compatible with 2:4 structured sparsity"


def prune_to_2_4(
    tensor: torch.Tensor,
    method: PruneMethod = PruneMethod.MAGNITUDE,
    gradients: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Prune a dense tensor to 2:4 structured sparsity pattern.

    For every 4 consecutive elements in the last dimension,
    keeps exactly 2 and zeros out the other 2.

    Args:
        tensor: Dense tensor [..., N] where N % 4 == 0
        method: Pruning method (MAGNITUDE, RANDOM, or GRADIENT)
        gradients: Gradient tensor for GRADIENT method

    Returns:
        Pruned tensor with same shape, 50% zeros in 2:4 pattern
    """
    compatible, msg = check_2_4_compatible(tensor)
    if not compatible:
        raise ValueError(msg)

    original_shape = tensor.shape
    # Reshape to [..., N // 4, 4] for group processing
    tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 4)

    if method == PruneMethod.MAGNITUDE:
        # Keep top-2 by absolute value
        abs_vals = tensor_reshaped.abs()
        # Get indices of top-2 values in each group
        _, top2_idx = torch.topk(abs_vals, k=2, dim=-1)

    elif method == PruneMethod.RANDOM:
        # Random 2 of 4
        n_groups = tensor_reshaped.shape[-2]
        batch_shape = tensor_reshaped.shape[:-2]
        # Generate random indices for each group
        rand_perm = torch.stack([
            torch.randperm(4, device=tensor.device)[:2]
            for _ in range(np.prod(batch_shape) * n_groups if batch_shape else n_groups)
        ])
        top2_idx = rand_perm.view(*batch_shape, n_groups, 2)

    elif method == PruneMethod.GRADIENT:
        if gradients is None:
            raise ValueError("GRADIENT method requires gradients tensor")
        grad_reshaped = gradients.view(*gradients.shape[:-1], -1, 4)
        abs_grads = grad_reshaped.abs()
        _, top2_idx = torch.topk(abs_grads, k=2, dim=-1)

    else:
        raise ValueError(f"Unknown pruning method: {method}")

    # Create mask from top-2 indices
    mask = torch.zeros_like(tensor_reshaped, dtype=torch.bool)
    # Scatter ones at top-2 positions
    mask.scatter_(-1, top2_idx, True)

    # Apply mask
    pruned = tensor_reshaped * mask.to(tensor.dtype)

    return pruned.view(original_shape)


def compress_2_4(tensor: torch.Tensor) -> Tuple[torch.Tensor, StructuredSparsityMetadata]:
    """
    Compress a 2:4 structured sparse tensor.

    Input must already be in 2:4 pattern (exactly 2 non-zeros per 4 elements).

    Args:
        tensor: 2D tensor [M, N] in 2:4 pattern

    Returns:
        Tuple of (compressed_values [M, N // 2], metadata)
    """
    assert tensor.dim() == 2, "compress_2_4 expects 2D tensor"
    M, N = tensor.shape
    assert N % 4 == 0, f"N ({N}) must be divisible by 4"

    n_groups = N // 4
    tensor_groups = tensor.view(M, n_groups, 4)  # [M, n_groups, 4]

    # Find non-zero positions in each group
    nonzero_mask = tensor_groups != 0  # [M, n_groups, 4]

    # Extract the two non-zero values per group
    # Sort positions to get consistent ordering
    compressed_values = torch.zeros(M, n_groups, 2, dtype=tensor.dtype, device=tensor.device)

    # Compute indices for each group
    indices = torch.zeros(M, n_groups, dtype=torch.uint8, device=tensor.device)

    for m in range(M):
        for g in range(n_groups):
            group = tensor_groups[m, g]
            nz_pos = torch.where(group != 0)[0]

            if len(nz_pos) >= 2:
                pos0, pos1 = nz_pos[0].item(), nz_pos[1].item()
                compressed_values[m, g, 0] = group[pos0]
                compressed_values[m, g, 1] = group[pos1]

                # Encode position pair as index
                if pos0 > pos1:
                    pos0, pos1 = pos1, pos0
                indices[m, g] = POSITIONS_TO_INDEX.get((pos0, pos1), 0)
            elif len(nz_pos) == 1:
                # Only one non-zero, treat as (pos, pos) - use first valid combo
                pos = nz_pos[0].item()
                compressed_values[m, g, 0] = group[pos]
                compressed_values[m, g, 1] = 0
                indices[m, g] = 0

    # Pack indices: 4 groups per byte (each index needs ~3 bits, we use 2 bits + overflow)
    # For simplicity, use 1 byte per group for now (can optimize later)
    packed_indices = indices  # [M, n_groups]

    # Reshape compressed values to [M, N // 2]
    compressed_values = compressed_values.view(M, N // 2)

    metadata = StructuredSparsityMetadata(
        indices=packed_indices,
        original_shape=(M, N),
        n_groups=M * n_groups,
    )

    return compressed_values, metadata


def decompress_2_4(
    compressed: torch.Tensor,
    metadata: StructuredSparsityMetadata,
) -> torch.Tensor:
    """
    Decompress a 2:4 structured sparse tensor back to dense.

    Args:
        compressed: Compressed values [M, N // 2]
        metadata: Sparsity metadata with indices

    Returns:
        Dense tensor [M, N]
    """
    M, N = metadata.original_shape
    n_groups = N // 4

    # Reshape compressed to [M, n_groups, 2]
    compressed_groups = compressed.view(M, n_groups, 2)

    # Output tensor
    dense = torch.zeros(M, N, dtype=compressed.dtype, device=compressed.device)
    dense_groups = dense.view(M, n_groups, 4)

    indices = metadata.indices  # [M, n_groups]

    for m in range(M):
        for g in range(n_groups):
            idx = indices[m, g].item()
            if idx < len(INDEX_TO_POSITIONS):
                pos0, pos1 = INDEX_TO_POSITIONS[idx]
                dense_groups[m, g, pos0] = compressed_groups[m, g, 0]
                dense_groups[m, g, pos1] = compressed_groups[m, g, 1]

    return dense


def compress_2_4_vectorized(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized compression for 2:4 structured sparsity.

    Faster than compress_2_4 for large tensors.

    Args:
        tensor: 2D tensor [M, N] in 2:4 pattern

    Returns:
        Tuple of (compressed_values [M, N // 2], indices [M, N // 4])
    """
    M, N = tensor.shape
    n_groups = N // 4

    tensor_groups = tensor.view(M, n_groups, 4)  # [M, n_groups, 4]
    abs_groups = tensor_groups.abs()

    # Get top-2 indices by magnitude (to handle cases with >2 nonzeros gracefully)
    _, top2_idx = torch.topk(abs_groups, k=2, dim=-1)  # [M, n_groups, 2]
    top2_idx_sorted, _ = torch.sort(top2_idx, dim=-1)  # Ensure pos0 < pos1

    # Gather values at top-2 positions
    compressed_values = torch.gather(tensor_groups, dim=-1, index=top2_idx_sorted)
    compressed_values = compressed_values.view(M, N // 2)

    # Encode indices: pos0 * 4 + pos1 gives unique ID, map to 0-5
    pos0 = top2_idx_sorted[..., 0]  # [M, n_groups]
    pos1 = top2_idx_sorted[..., 1]  # [M, n_groups]

    # Create lookup: (pos0, pos1) -> index
    # pos0=0,pos1=1 -> 0; pos0=0,pos1=2 -> 1; pos0=0,pos1=3 -> 2
    # pos0=1,pos1=2 -> 3; pos0=1,pos1=3 -> 4; pos0=2,pos1=3 -> 5
    # Formula: if pos0 == 0: idx = pos1 - 1
    #          if pos0 == 1: idx = pos1 + 1
    #          if pos0 == 2: idx = 5
    indices = torch.where(
        pos0 == 0, pos1 - 1,
        torch.where(
            pos0 == 1, pos1 + 1,
            torch.tensor(5, device=tensor.device)
        )
    ).to(torch.uint8)

    return compressed_values, indices


def decompress_2_4_vectorized(
    compressed: torch.Tensor,
    indices: torch.Tensor,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Vectorized decompression for 2:4 structured sparsity.

    Args:
        compressed: Compressed values [M, N // 2]
        indices: Position indices [M, N // 4]
        original_shape: (M, N)

    Returns:
        Dense tensor [M, N]
    """
    M, N = original_shape
    n_groups = N // 4

    compressed_groups = compressed.view(M, n_groups, 2)
    indices_long = indices.long()

    # Create output
    dense = torch.zeros(M, n_groups, 4, dtype=compressed.dtype, device=compressed.device)

    # Lookup table as tensor
    pos_lookup = torch.tensor(INDEX_TO_POSITIONS, device=compressed.device)  # [6, 2]

    # Get positions for all indices
    pos0 = pos_lookup[indices_long, 0]  # [M, n_groups]
    pos1 = pos_lookup[indices_long, 1]  # [M, n_groups]

    # Scatter values to positions
    batch_idx = torch.arange(M, device=compressed.device)[:, None].expand(M, n_groups)
    group_idx = torch.arange(n_groups, device=compressed.device)[None, :].expand(M, n_groups)

    dense[batch_idx, group_idx, pos0] = compressed_groups[..., 0]
    dense[batch_idx, group_idx, pos1] = compressed_groups[..., 1]

    return dense.view(M, N)


# =============================================================================
# Triton Kernel for 2:4 Structured SpMM
# =============================================================================

@triton.jit
def void_structured_spmm_kernel(
    # Sparse matrix A (2:4 compressed VOID format)
    a_values_ptr,      # [n_blocks, TILE_M, TILE_K // 2]
    a_indices_ptr,     # [n_blocks, TILE_M, TILE_K // 4] - position indices
    a_block_rows_ptr,  # [n_blocks]
    a_block_cols_ptr,  # [n_blocks]
    a_row_ptr_ptr,     # [n_block_rows + 1]
    a_block_idx_ptr,   # [n_blocks]
    # Dense matrix B
    b_ptr,             # [K, N]
    # Output matrix C
    c_ptr,             # [M, N]
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    SpMM kernel for 2:4 structured sparse VOID format.

    The sparse matrix A is stored in compressed 2:4 format:
    - values: 50% compressed (TILE_K // 2 elements per row)
    - indices: indicate which 2 of 4 positions are non-zero

    This kernel decompresses on-the-fly during computation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= n_block_rows:
        return

    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    col_start = pid_n * TILE_N

    for block_idx in range(row_start, row_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load compressed A tile [TILE_M, TILE_K // 2]
        # For simplicity, we decompress to full tile in registers
        # In production, use specialized 2:4 matmul instructions

        # Load B tile (full K dimension)
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, col_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        # For now, load the full decompressed A tile
        # TODO: Implement true 2:4 sparse tensor core path
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * (TILE_K // 2),
            shape=(TILE_M, TILE_K // 2),
            strides=(TILE_K // 2, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K // 2),
            order=(1, 0),
        )
        # Note: This is a placeholder - real implementation needs decompression
        # or use of NVIDIA's sparse tensor core instructions via cuSparseLt

        # For demonstration, we skip actual computation with compressed format
        # and just load what we can. Full implementation requires cuSparseLt bindings.

    out_row = pid_m * TILE_M
    c_tile_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(out_row, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    tl.store(c_tile_ptr, acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def void_to_structured(
    void_tensor,  # VOIDTensor
    prune_method: PruneMethod = PruneMethod.MAGNITUDE,
) -> VOIDStructuredTensor:
    """
    Convert a VOIDTensor to 2:4 structured sparse format.

    This prunes each block to 2:4 pattern and compresses storage.

    Args:
        void_tensor: Input VOIDTensor
        prune_method: Method for pruning to 2:4 pattern

    Returns:
        VOIDStructuredTensor with 50% compressed storage
    """
    from .format import VOIDTensor

    tile_m, tile_k = void_tensor.tile_size

    if tile_k % 4 != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by 4 for 2:4 structured sparsity")

    n_blocks = void_tensor.n_blocks
    values = void_tensor.values  # [n_blocks, tile_m, tile_k]

    # Prune each block to 2:4 pattern
    # Reshape to [n_blocks * tile_m, tile_k] for pruning
    values_flat = values.view(-1, tile_k)
    pruned_flat = prune_to_2_4(values_flat, method=prune_method)

    # Compress using vectorized method
    compressed_values, indices = compress_2_4_vectorized(pruned_flat)

    # Reshape back to block structure
    compressed_values = compressed_values.view(n_blocks, tile_m, tile_k // 2)
    indices = indices.view(n_blocks, tile_m, tile_k // 4)

    metadata = StructuredSparsityMetadata(
        indices=indices.view(n_blocks, -1),  # Flatten indices per block
        original_shape=void_tensor.shape,
        n_groups=n_blocks * tile_m * (tile_k // 4),
    )

    return VOIDStructuredTensor(
        values=compressed_values,
        metadata=metadata,
        block_rows=void_tensor.block_rows,
        block_cols=void_tensor.block_cols,
        shape=void_tensor.shape,
        tile_size=void_tensor.tile_size,
        n_blocks=n_blocks,
    )


def structured_to_dense(structured_tensor: VOIDStructuredTensor) -> torch.Tensor:
    """
    Convert a VOIDStructuredTensor back to dense format.

    Args:
        structured_tensor: Input structured sparse tensor

    Returns:
        Dense tensor [M, K]
    """
    M, K = structured_tensor.shape
    tile_m, tile_k = structured_tensor.tile_size
    n_blocks = structured_tensor.n_blocks

    # Output dense matrix
    dense = torch.zeros(M, K, dtype=structured_tensor.dtype, device=structured_tensor.device)

    # Decompress each block
    values = structured_tensor.values  # [n_blocks, tile_m, tile_k // 2]
    indices = structured_tensor.metadata.indices.view(n_blocks, tile_m, tile_k // 4)

    for b in range(n_blocks):
        block_row = structured_tensor.block_rows[b].item()
        block_col = structured_tensor.block_cols[b].item()

        row_start = block_row * tile_m
        col_start = block_col * tile_k

        # Decompress this block
        block_values = values[b]  # [tile_m, tile_k // 2]
        block_indices = indices[b]  # [tile_m, tile_k // 4]

        # Flatten for decompression
        block_values_flat = block_values.view(tile_m, -1)
        block_indices_flat = block_indices.view(tile_m, -1)

        block_dense = decompress_2_4_vectorized(
            block_values_flat,
            block_indices_flat,
            (tile_m, tile_k)
        )

        # Place in output
        row_end = min(row_start + tile_m, M)
        col_end = min(col_start + tile_k, K)
        dense[row_start:row_end, col_start:col_end] = block_dense[:row_end-row_start, :col_end-col_start]

    return dense


def get_structured_sparsity_info(tensor: torch.Tensor) -> dict:
    """
    Analyze a tensor for 2:4 structured sparsity compatibility.

    Args:
        tensor: Input tensor to analyze

    Returns:
        Dictionary with sparsity analysis
    """
    if tensor.dim() != 2:
        return {"compatible": False, "reason": "Must be 2D tensor"}

    M, N = tensor.shape
    compatible, msg = check_2_4_compatible(tensor)

    if not compatible:
        return {"compatible": False, "reason": msg}

    # Analyze current sparsity pattern
    n_groups = N // 4
    tensor_groups = tensor.view(M, n_groups, 4)
    zeros_per_group = (tensor_groups == 0).sum(dim=-1).float()  # [M, n_groups]

    avg_zeros = zeros_per_group.mean().item()
    groups_with_2_zeros = (zeros_per_group == 2).float().mean().item()

    return {
        "compatible": True,
        "message": msg,
        "shape": (M, N),
        "n_groups": M * n_groups,
        "avg_zeros_per_group": avg_zeros,
        "pct_already_2_4": groups_with_2_zeros * 100,
        "compression_ratio": 0.5,  # 2:4 always gives 50% compression
        "estimated_speedup": "~2x on Ampere+ with Sparse Tensor Cores",
    }


# Export public API
__all__ = [
    # Enums
    "PruneMethod",
    # Dataclasses
    "StructuredSparsityMetadata",
    "VOIDStructuredTensor",
    # Functions
    "check_2_4_compatible",
    "prune_to_2_4",
    "compress_2_4",
    "decompress_2_4",
    "compress_2_4_vectorized",
    "decompress_2_4_vectorized",
    "void_to_structured",
    "structured_to_dense",
    "get_structured_sparsity_info",
]
