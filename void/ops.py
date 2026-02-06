"""Triton SpMM and SpMV kernels for VOID sparse matrices."""

import torch
import triton
import triton.language as tl
from typing import Optional

from .format import VOIDTensor


def get_triton_dtype(torch_dtype: torch.dtype):
    """Map PyTorch dtype to Triton dtype."""
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    # Add FP8 types if available (PyTorch 2.1+)
    if hasattr(torch, 'float8_e4m3fn'):
        dtype_map[torch.float8_e4m3fn] = tl.float8e4nv
    if hasattr(torch, 'float8_e5m2'):
        dtype_map[torch.float8_e5m2] = tl.float8e5

    return dtype_map.get(torch_dtype, tl.float32)


# Minimum K dimension for Tensor Core operations
MIN_TENSOR_CORE_K = 16


def compute_optimal_tile_n(N: int) -> int:
    """Select TILE_N based on output dimension N."""
    if N >= 128:
        return 128
    elif N >= 64:
        return 64
    elif N >= 32:
        return 32
    elif N >= 16:
        return 16
    else:
        return triton.next_power_of_2(N)


# =============================================================================
# SpMM Kernel: VOID Sparse @ Dense -> Dense
# =============================================================================

@triton.jit
def void_spmm_kernel(
    # Sparse matrix A (VOID format)
    a_values_ptr,      # [n_blocks, TILE_M, TILE_K]
    a_block_rows_ptr,  # [n_blocks]
    a_block_cols_ptr,  # [n_blocks]
    a_row_ptr_ptr,     # [n_block_rows + 1] - CSR-style row pointers for blocks
    a_block_idx_ptr,   # [n_blocks] - block indices sorted by row
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
    # Tile sizes (constexpr)
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Compute C = A @ B where A is in VOID sparse format.

    Grid: (n_block_rows, cdiv(N, TILE_N))
    Each program computes a TILE_M x TILE_N output tile.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # Which block row of A
    pid_n = tl.program_id(1)  # Which tile column of output

    # Bounds check for block row
    if pid_m >= n_block_rows:
        return

    # Get the range of blocks in this row
    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)

    # Initialize accumulator in FP32 for numerical stability
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Output column range
    col_start = pid_n * TILE_N
    col_offsets = col_start + tl.arange(0, TILE_N)
    col_mask = col_offsets < N

    # Iterate over blocks in this row
    for block_idx in range(row_start, row_end):
        # Get the actual block index (sorted by row)
        actual_idx = tl.load(a_block_idx_ptr + block_idx)

        # Get block column (which determines K offset)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load A tile using block pointer
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile = tl.load(a_tile_ptr).to(tl.float32)

        # Load B tile
        # B is [K, N], we need [TILE_K, TILE_N] starting at (k_offset, col_start)
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, col_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        # Accumulate: C[m, n] += A[m, k] * B[k, n]
        acc += tl.dot(a_tile, b_tile)

    # Store output tile with target dtype
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


def void_spmm(
    a: VOIDTensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sparse-Dense Matrix Multiplication: C = A @ B

    Args:
        a: VOID sparse matrix [M, K]
        b: Dense matrix [K, N]
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N]
    """
    # Fast path assertions (minimal overhead)
    M, K = a.shape
    N = b.shape[1]

    # Early exit for empty matrix
    if a.n_blocks == 0:
        if out is None:
            return torch.zeros(M, N, dtype=b.dtype, device=b.device)
        out.zero_()
        return out

    # Ensure B is contiguous (usually already is)
    if not b.is_contiguous():
        b = b.contiguous()

    # Allocate output
    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    # Get cached row-organized block info
    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]
    tile_m, tile_k = a.tile_size

    # Choose TILE_N: larger is better for memory coalescing
    # Use 128 for large N, 64 for medium, match N for small
    if N >= 128:
        TILE_N = 128
    elif N >= 64:
        TILE_N = 64
    else:
        TILE_N = triton.next_power_of_2(N)

    # Grid: one program per (block_row, output_tile_column)
    grid = (n_block_rows, triton.cdiv(N, TILE_N))

    void_spmm_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, out,
        M, N, K,
        a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
        OUTPUT_DTYPE=get_triton_dtype(b.dtype),
    )

    return out


# =============================================================================
# SpMV Kernel: VOID Sparse @ Vector -> Vector
# =============================================================================

@triton.jit
def void_spmv_kernel(
    # Sparse matrix A (VOID format)
    a_values_ptr,
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Input vector x
    x_ptr,
    # Output vector y
    y_ptr,
    # Dimensions
    M, K,
    n_blocks,
    n_block_rows,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Compute y = A @ x where A is VOID sparse and x is dense vector.

    Optimized version using vectorized matrix-vector multiply with broadcast.
    """
    pid = tl.program_id(0)  # Block row

    if pid >= n_block_rows:
        return

    # Get block range for this row
    row_start = tl.load(a_row_ptr_ptr + pid)
    row_end = tl.load(a_row_ptr_ptr + pid + 1)

    # Initialize accumulator [TILE_M] in FP32 for stability
    acc = tl.zeros((TILE_M,), dtype=tl.float32)

    # Iterate over blocks in this row
    for block_idx in range(row_start, row_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load x segment [TILE_K] and cast to FP32
        x_offsets = k_offset + tl.arange(0, TILE_K)
        x_mask = x_offsets < K
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)

        # Load entire A tile [TILE_M, TILE_K] using block pointer
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile = tl.load(a_tile_ptr).to(tl.float32)  # [TILE_M, TILE_K]

        # Vectorized matrix-vector multiply using broadcast:
        # a_tile [TILE_M, TILE_K] * x_vals[None, :] [1, TILE_K] -> [TILE_M, TILE_K]
        # then sum along K dimension -> [TILE_M]
        acc += tl.sum(a_tile * x_vals[None, :], axis=1)

    # Store output with target dtype
    out_offset = pid * TILE_M
    out_offsets = out_offset + tl.arange(0, TILE_M)
    out_mask = out_offsets < M
    tl.store(y_ptr + out_offsets, acc.to(OUTPUT_DTYPE), mask=out_mask)


def void_spmv(
    a: VOIDTensor,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sparse-Dense Matrix-Vector Multiplication: y = A @ x

    Args:
        a: VOID sparse matrix [M, K]
        x: Dense vector [K] or [K, 1]
        out: Optional output buffer [M]

    Returns:
        Dense vector y [M]
    """
    # Handle [K, 1] input
    squeeze_output = False
    if x.dim() == 2 and x.shape[1] == 1:
        x = x.squeeze(1)
        squeeze_output = True

    assert x.dim() == 1, "x must be 1D vector"
    assert a.shape[1] == x.shape[0], f"Dimension mismatch: A is {a.shape}, x is {x.shape}"
    assert x.is_cuda, "x must be on CUDA"
    assert a.values.is_cuda, "A must be on CUDA"

    M, K = a.shape
    tile_m, tile_k = a.tile_size

    if not x.is_contiguous():
        x = x.contiguous()

    if out is None:
        out = torch.zeros(M, dtype=x.dtype, device=x.device)
    else:
        assert out.shape == (M,)
        out.zero_()

    if a.n_blocks == 0:
        return out.unsqueeze(1) if squeeze_output else out

    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]

    # Get output dtype for Triton
    output_dtype = get_triton_dtype(x.dtype)

    grid = (n_block_rows,)

    void_spmv_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        x, out,
        M, K, a.n_blocks, n_block_rows,
        TILE_M=tile_m, TILE_K=tile_k,
        OUTPUT_DTYPE=output_dtype,
    )

    return out.unsqueeze(1) if squeeze_output else out


# =============================================================================
# Autotuned SpMM Kernel (for optimal performance)
# =============================================================================

def get_autotune_configs():
    """Generate autotuning configurations."""
    configs = []
    for TILE_N in [32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append(
                    triton.Config(
                        {'TILE_N': TILE_N},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K', 'n_blocks'],
)
@triton.jit
def void_spmm_autotuned_kernel(
    a_values_ptr, a_block_rows_ptr, a_block_cols_ptr,
    a_row_ptr_ptr, a_block_idx_ptr,
    b_ptr, c_ptr,
    M, N, K, n_blocks, n_block_rows,
    stride_bk, stride_bn, stride_cm, stride_cn,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """Autotuned version of SpMM kernel."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= n_block_rows:
        return

    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)

    # FP32 accumulator for stability
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    col_start = pid_n * TILE_N

    for block_idx in range(row_start, row_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile = tl.load(a_tile_ptr).to(tl.float32)

        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, col_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        acc += tl.dot(a_tile, b_tile)

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


def void_spmm_autotuned(
    a: VOIDTensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SpMM with autotuning for optimal tile sizes."""
    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert b.is_cuda and a.values.is_cuda

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        return out

    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]

    # Get output dtype for Triton
    output_dtype = get_triton_dtype(b.dtype)

    # Let autotuner pick TILE_N
    TILE_N = 64  # Will be overridden by autotuner

    grid = lambda meta: (n_block_rows, triton.cdiv(N, meta['TILE_N']))

    void_spmm_autotuned_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m, TILE_K=tile_k,
        OUTPUT_DTYPE=output_dtype,
    )

    return out


# =============================================================================
# Pipelined SpMM Kernel (async memory prefetching)
# =============================================================================

def get_pipelined_autotune_configs():
    """Generate autotuning configurations for pipelined kernel."""
    configs = []
    # More aggressive pipelining configurations
    for TILE_N in [64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [3, 4, 5]:  # Higher stages for better pipelining
                configs.append(
                    triton.Config(
                        {'TILE_N': TILE_N, 'NUM_STAGES': num_stages},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_pipelined_autotune_configs(),
    key=['M', 'N', 'K', 'n_blocks'],
)
@triton.jit
def void_spmm_pipelined_kernel(
    # Sparse matrix A (VOID format)
    a_values_ptr,
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B
    b_ptr,
    # Output matrix C
    c_ptr,
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes (constexpr)
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Pipelined SpMM kernel with async memory prefetching.

    Uses software pipelining (num_stages) to overlap memory loads
    with computation, improving performance for irregular sparse patterns.
    """
    pid_m = tl.program_id(0)  # Which block row of A
    pid_n = tl.program_id(1)  # Which tile column of output

    if pid_m >= n_block_rows:
        return

    # Get the range of blocks in this row
    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)
    n_row_blocks = row_end - row_start

    # Initialize accumulator in FP32 for numerical stability
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    col_start = pid_n * TILE_N

    # Use pipelined loop with explicit num_stages hint
    # This tells Triton to prefetch future iterations
    for block_offset in tl.range(0, n_row_blocks, num_stages=NUM_STAGES):
        block_idx = row_start + block_offset

        # Get the actual block index (sorted by row)
        actual_idx = tl.load(a_block_idx_ptr + block_idx)

        # Get block column (which determines K offset)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load A tile using block pointer
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile = tl.load(a_tile_ptr).to(tl.float32)

        # Load B tile with prefetch hint via block pointer
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, col_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        # Accumulate: C[m, n] += A[m, k] * B[k, n]
        acc += tl.dot(a_tile, b_tile)

    # Store output tile with target dtype
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


def void_spmm_pipelined(
    a: VOIDTensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pipelined Sparse-Dense Matrix Multiplication with async memory prefetching.

    This variant uses software pipelining to overlap memory loads with
    computation, which can improve performance for irregular sparse patterns
    where memory latency is the bottleneck.

    Args:
        a: VOID sparse matrix [M, K]
        b: Dense matrix [K, N]
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N]
    """
    assert b.dim() == 2, "B must be 2D matrix"
    assert a.shape[1] == b.shape[0], f"Dimension mismatch: A is {a.shape}, B is {b.shape}"
    assert b.is_cuda and a.values.is_cuda, "Tensors must be on CUDA"

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        return out

    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]

    output_dtype = get_triton_dtype(b.dtype)

    # Grid is determined by autotuner's TILE_N
    grid = lambda meta: (n_block_rows, triton.cdiv(N, meta['TILE_N']))

    void_spmm_pipelined_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m, TILE_K=tile_k,
        OUTPUT_DTYPE=output_dtype,
    )

    return out
