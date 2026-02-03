"""
VOID Sparse Matrix Format

The VOID format stores sparse matrices as a collection of dense tiles with:
1. Block decomposition: Fixed-size tiles (e.g., 32x32)
2. Thresholding: Only tiles with non-zeros are stored
3. Morton ordering: Tiles sorted by Z-curve for spatial locality
4. Compressed metadata: Block table for tile lookup
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import numpy as np
from scipy import sparse


def morton_encode(x: int, y: int) -> int:
    """
    Encode 2D coordinates into Morton code (Z-order curve).

    Morton codes interleave the bits of x and y coordinates,
    ensuring that spatially close tiles are also close in memory.
    """
    def spread_bits(v: int) -> int:
        # Spread bits of a 16-bit integer to even positions
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    return spread_bits(x) | (spread_bits(y) << 1)


def morton_decode(code: int) -> Tuple[int, int]:
    """Decode Morton code back to 2D coordinates."""
    def compact_bits(v: int) -> int:
        v = v & 0x55555555
        v = (v | (v >> 1)) & 0x33333333
        v = (v | (v >> 2)) & 0x0F0F0F0F
        v = (v | (v >> 4)) & 0x00FF00FF
        v = (v | (v >> 8)) & 0x0000FFFF
        return v

    return compact_bits(code), compact_bits(code >> 1)


def morton_encode_batch(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Vectorized Morton encoding for arrays of coordinates."""
    def spread_bits_vec(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64)
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    return spread_bits_vec(rows) | (spread_bits_vec(cols) << 1)


@dataclass
class VOIDTensor:
    """
    VOID sparse matrix format.

    Attributes:
        values: Dense tile data, shape [n_blocks, tile_m, tile_n]
        block_rows: Block row indices, shape [n_blocks]
        block_cols: Block column indices, shape [n_blocks]
        morton_codes: Morton codes for each block (for debugging/validation)
        shape: Original matrix shape (M, N)
        tile_size: Tile dimensions (tile_m, tile_n)
        nnz_original: Original number of non-zeros
        n_blocks: Number of active blocks
        density: Fraction of tiles that are active
        dtype: Data type of values tensor
    """
    values: torch.Tensor          # [n_blocks, tile_m, tile_n]
    block_rows: torch.Tensor      # [n_blocks] int32
    block_cols: torch.Tensor      # [n_blocks] int32
    morton_codes: torch.Tensor    # [n_blocks] int64
    shape: Tuple[int, int]        # Original (M, N)
    tile_size: Tuple[int, int]    # (tile_m, tile_n)
    nnz_original: int
    n_blocks: int
    density: float
    dtype: torch.dtype = torch.float32

    @property
    def padded_shape(self) -> Tuple[int, int]:
        """Shape after padding to tile boundaries."""
        tile_m, tile_n = self.tile_size
        m = ((self.shape[0] + tile_m - 1) // tile_m) * tile_m
        n = ((self.shape[1] + tile_n - 1) // tile_n) * tile_n
        return (m, n)

    @property
    def block_grid(self) -> Tuple[int, int]:
        """Number of blocks in each dimension."""
        tile_m, tile_n = self.tile_size
        return (
            (self.shape[0] + tile_m - 1) // tile_m,
            (self.shape[1] + tile_n - 1) // tile_n,
        )

    @property
    def sparsity(self) -> float:
        """Fraction of zeros in original matrix."""
        total = self.shape[0] * self.shape[1]
        return 1.0 - (self.nnz_original / total)

    @property
    def block_sparsity(self) -> float:
        """Fraction of empty blocks."""
        total_blocks = self.block_grid[0] * self.block_grid[1]
        return 1.0 - (self.n_blocks / total_blocks)

    @property
    def overhead_ratio(self) -> float:
        """
        Ratio of stored elements to original non-zeros.

        Values > 1.0 indicate padding overhead.
        """
        stored = self.n_blocks * self.tile_size[0] * self.tile_size[1]
        return stored / max(self.nnz_original, 1)

    def to(self, device: Union[str, torch.device]) -> 'VOIDTensor':
        """Move tensor to specified device."""
        return VOIDTensor(
            values=self.values.to(device),
            block_rows=self.block_rows.to(device),
            block_cols=self.block_cols.to(device),
            morton_codes=self.morton_codes.to(device),
            shape=self.shape,
            tile_size=self.tile_size,
            nnz_original=self.nnz_original,
            n_blocks=self.n_blocks,
            density=self.density,
            dtype=self.dtype,
        )

    def cuda(self) -> 'VOIDTensor':
        return self.to('cuda')

    def cpu(self) -> 'VOIDTensor':
        return self.to('cpu')

    def to_dtype(self, dtype: torch.dtype) -> 'VOIDTensor':
        """Convert values to specified dtype."""
        return VOIDTensor(
            values=self.values.to(dtype),
            block_rows=self.block_rows,
            block_cols=self.block_cols,
            morton_codes=self.morton_codes,
            shape=self.shape,
            tile_size=self.tile_size,
            nnz_original=self.nnz_original,
            n_blocks=self.n_blocks,
            density=self.density,
            dtype=dtype,
        )

    def half(self) -> 'VOIDTensor':
        """Convert to FP16."""
        return self.to_dtype(torch.float16)

    def bfloat16(self) -> 'VOIDTensor':
        """Convert to BF16."""
        return self.to_dtype(torch.bfloat16)

    def float(self) -> 'VOIDTensor':
        """Convert to FP32."""
        return self.to_dtype(torch.float32)

    def to_dense(self) -> torch.Tensor:
        """Convert back to dense matrix (for validation)."""
        tile_m, tile_n = self.tile_size
        result = torch.zeros(self.padded_shape, dtype=self.values.dtype, device=self.values.device)

        for i in range(self.n_blocks):
            br = self.block_rows[i].item()
            bc = self.block_cols[i].item()
            row_start = br * tile_m
            col_start = bc * tile_n
            result[row_start:row_start + tile_m, col_start:col_start + tile_n] = self.values[i]

        # Trim to original shape
        return result[:self.shape[0], :self.shape[1]]

    def get_row_block_info(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get block information organized by row for SpMM/SpMV.

        Returns:
            row_ptr: CSR-style row pointer for blocks, shape [n_block_rows + 1]
            block_indices: Index into values for each row's blocks
        """
        n_block_rows = self.block_grid[0]

        # Count blocks per row
        row_counts = torch.zeros(n_block_rows, dtype=torch.int32, device=self.block_rows.device)
        row_counts.scatter_add_(0, self.block_rows.long(), torch.ones_like(self.block_rows))

        # Build row pointer
        row_ptr = torch.zeros(n_block_rows + 1, dtype=torch.int32, device=self.block_rows.device)
        row_ptr[1:] = torch.cumsum(row_counts, dim=0)

        # Blocks are already sorted by Morton code, but we need row-major order for SpMM
        # Sort by (block_row, block_col)
        sort_keys = self.block_rows.long() * self.block_grid[1] + self.block_cols.long()
        block_indices = torch.argsort(sort_keys).to(torch.int32)

        return row_ptr, block_indices

    def __repr__(self) -> str:
        return (
            f"VOIDTensor(shape={self.shape}, tile_size={self.tile_size}, "
            f"n_blocks={self.n_blocks}, sparsity={self.sparsity:.1%}, "
            f"block_sparsity={self.block_sparsity:.1%}, overhead={self.overhead_ratio:.2f}x, "
            f"dtype={self.dtype})"
        )


def csr_to_void(
    matrix: Union[sparse.csr_matrix, sparse.csr_array],
    tile_size: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
) -> VOIDTensor:
    """
    Convert CSR sparse matrix to VOID format.

    Args:
        matrix: scipy CSR sparse matrix
        tile_size: Size of square tiles (default 32)
        dtype: Output data type
        device: Output device

    Returns:
        VOIDTensor in the specified format
    """
    if not sparse.isspmatrix_csr(matrix):
        matrix = sparse.csr_matrix(matrix)

    M, N = matrix.shape
    nnz = matrix.nnz

    # Calculate block grid dimensions
    n_block_rows = (M + tile_size - 1) // tile_size
    n_block_cols = (N + tile_size - 1) // tile_size

    # Find active blocks (those containing at least one non-zero)
    # Convert to COO for easier block identification
    coo = matrix.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data

    # Compute block indices for each non-zero
    block_row_idx = rows // tile_size
    block_col_idx = cols // tile_size

    # Find unique blocks
    block_keys = block_row_idx * n_block_cols + block_col_idx
    unique_blocks, inverse_indices = np.unique(block_keys, return_inverse=True)

    n_blocks = len(unique_blocks)

    # Decode block coordinates
    active_block_rows = unique_blocks // n_block_cols
    active_block_cols = unique_blocks % n_block_cols

    # Compute Morton codes for sorting
    morton_codes = morton_encode_batch(active_block_rows, active_block_cols)

    # Sort blocks by Morton code
    sort_order = np.argsort(morton_codes)
    sorted_block_rows = active_block_rows[sort_order]
    sorted_block_cols = active_block_cols[sort_order]
    sorted_morton = morton_codes[sort_order]

    # Create reverse mapping: old block index -> new sorted index
    reverse_mapping = np.zeros(n_blocks, dtype=np.int64)
    reverse_mapping[sort_order] = np.arange(n_blocks)

    # Allocate tile storage
    values = np.zeros((n_blocks, tile_size, tile_size), dtype=np.float32)

    # Fill tiles with data
    for i in range(len(data)):
        old_block_idx = inverse_indices[i]
        new_block_idx = reverse_mapping[old_block_idx]

        local_row = rows[i] % tile_size
        local_col = cols[i] % tile_size
        values[new_block_idx, local_row, local_col] = data[i]

    # Convert to tensors
    # Keep FP32 for numerical stability, then convert to target dtype
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if dtype != torch.float32:
        values_tensor = values_tensor.to(dtype)
    block_rows_tensor = torch.tensor(sorted_block_rows, dtype=torch.int32, device=device)
    block_cols_tensor = torch.tensor(sorted_block_cols, dtype=torch.int32, device=device)
    morton_tensor = torch.tensor(sorted_morton, dtype=torch.int64, device=device)

    return VOIDTensor(
        values=values_tensor,
        block_rows=block_rows_tensor,
        block_cols=block_cols_tensor,
        morton_codes=morton_tensor,
        shape=(M, N),
        tile_size=(tile_size, tile_size),
        nnz_original=nnz,
        n_blocks=n_blocks,
        density=n_blocks / (n_block_rows * n_block_cols),
        dtype=dtype,
    )


def dense_to_void(
    matrix: torch.Tensor,
    tile_size: int = 32,
    threshold: float = 0.0,
) -> VOIDTensor:
    """
    Convert dense matrix to VOID format, pruning tiles below threshold.

    Args:
        matrix: Dense 2D tensor
        tile_size: Size of square tiles
        threshold: Minimum L1 norm for a tile to be kept (0 = keep all non-empty)

    Returns:
        VOIDTensor
    """
    assert matrix.dim() == 2, "Input must be 2D tensor"

    M, N = matrix.shape
    device = matrix.device
    dtype = matrix.dtype

    # Pad to tile boundaries
    pad_m = (tile_size - M % tile_size) % tile_size
    pad_n = (tile_size - N % tile_size) % tile_size

    if pad_m > 0 or pad_n > 0:
        matrix = torch.nn.functional.pad(matrix, (0, pad_n, 0, pad_m))

    padded_m, padded_n = matrix.shape
    n_block_rows = padded_m // tile_size
    n_block_cols = padded_n // tile_size

    # Reshape into tiles
    tiles = matrix.reshape(n_block_rows, tile_size, n_block_cols, tile_size)
    tiles = tiles.permute(0, 2, 1, 3)  # [n_block_rows, n_block_cols, tile_size, tile_size]

    # Compute tile norms to find non-empty tiles
    tile_norms = tiles.abs().sum(dim=(2, 3))  # [n_block_rows, n_block_cols]

    # Find active tiles
    active_mask = tile_norms > threshold
    active_indices = active_mask.nonzero(as_tuple=False)  # [n_active, 2]

    if len(active_indices) == 0:
        # Fully sparse - return empty VOID tensor
        return VOIDTensor(
            values=torch.empty((0, tile_size, tile_size), dtype=dtype, device=device),
            block_rows=torch.empty(0, dtype=torch.int32, device=device),
            block_cols=torch.empty(0, dtype=torch.int32, device=device),
            morton_codes=torch.empty(0, dtype=torch.int64, device=device),
            shape=(M, N),
            tile_size=(tile_size, tile_size),
            nnz_original=int((matrix[:M, :N] != 0).sum().item()),
            n_blocks=0,
            density=0.0,
            dtype=dtype,
        )

    n_blocks = len(active_indices)
    block_rows = active_indices[:, 0]
    block_cols = active_indices[:, 1]

    # Compute Morton codes (on CPU for now)
    morton_codes_np = morton_encode_batch(
        block_rows.cpu().numpy(),
        block_cols.cpu().numpy()
    )

    # Sort by Morton code
    sort_order = np.argsort(morton_codes_np)
    sort_order_tensor = torch.tensor(sort_order, device=device)

    sorted_block_rows = block_rows[sort_order_tensor].to(torch.int32)
    sorted_block_cols = block_cols[sort_order_tensor].to(torch.int32)
    sorted_morton = torch.tensor(morton_codes_np[sort_order], dtype=torch.int64, device=device)

    # Extract tile values in sorted order
    values = tiles[sorted_block_rows.long(), sorted_block_cols.long()]  # [n_blocks, tile_size, tile_size]

    # Count original non-zeros
    nnz = int((matrix[:M, :N] != 0).sum().item())

    return VOIDTensor(
        values=values,
        block_rows=sorted_block_rows,
        block_cols=sorted_block_cols,
        morton_codes=sorted_morton,
        shape=(M, N),
        tile_size=(tile_size, tile_size),
        nnz_original=nnz,
        n_blocks=n_blocks,
        density=n_blocks / (n_block_rows * n_block_cols),
        dtype=dtype,
    )


def void_to_csr(void_tensor: VOIDTensor) -> sparse.csr_matrix:
    """Convert VOID format back to CSR (for validation)."""
    dense = void_tensor.to_dense().cpu().numpy()
    return sparse.csr_matrix(dense)
