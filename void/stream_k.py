"""
Stream-K Load Balancing for VOID SpMM

Handles power-law row distributions where some rows have many more
non-zero blocks than others. Standard row-parallel mapping causes
severe load imbalance.

Stream-K approach:
1. Estimate total work (sum of blocks across all rows)
2. Divide work evenly among thread blocks
3. Each thread block processes a "budget" of work
4. Heavy rows are split across multiple thread blocks
5. Partial results combined via atomic operations
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
from dataclasses import dataclass

from .format import VOIDTensor


def get_triton_dtype(torch_dtype: torch.dtype):
    """Map PyTorch dtype to Triton dtype."""
    return {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }.get(torch_dtype, tl.float32)


@dataclass
class StreamKWorkload:
    """
    Pre-computed workload distribution for Stream-K SpMM.

    Attributes:
        work_assignments: [n_workers, 3] - (row_idx, block_start, block_end) per worker
        n_workers: Number of parallel workers (thread blocks)
        total_work: Total number of block computations
        max_work_per_worker: Maximum work assigned to any worker
    """
    # Per-worker assignments: which row and which block range
    worker_rows: torch.Tensor       # [n_workers]
    worker_block_start: torch.Tensor  # [n_workers]
    worker_block_end: torch.Tensor    # [n_workers]

    # Whether this worker needs atomic accumulation (partial row)
    worker_needs_atomic: torch.Tensor  # [n_workers] bool

    n_workers: int
    total_work: int


def compute_stream_k_workload(
    void_tensor: VOIDTensor,
    n_workers: Optional[int] = None,
    max_work_per_worker: int = 16,
) -> StreamKWorkload:
    """
    Compute Stream-K work distribution.

    Args:
        void_tensor: VOID sparse matrix
        n_workers: Number of workers (default: auto based on SM count)
        max_work_per_worker: Maximum blocks per worker

    Returns:
        StreamKWorkload with assignments
    """
    row_ptr, block_indices = void_tensor.get_row_block_info()
    n_block_rows = void_tensor.block_grid[0]

    # Count work per row
    row_work = row_ptr[1:] - row_ptr[:-1]  # Blocks per row

    total_work = void_tensor.n_blocks

    # Determine number of workers
    if n_workers is None:
        # Heuristic: enough workers to keep GPU busy, but not too fragmented
        n_workers = min(total_work, 1024)

    # Target work per worker
    target_work = (total_work + n_workers - 1) // n_workers
    target_work = min(target_work, max_work_per_worker)

    # Assign work to workers
    worker_rows = []
    worker_block_start = []
    worker_block_end = []
    worker_needs_atomic = []

    current_worker = 0
    for row in range(n_block_rows):
        row_start = row_ptr[row].item()
        row_end = row_ptr[row + 1].item()
        row_blocks = row_end - row_start

        if row_blocks == 0:
            continue

        # Split row if needed
        pos = row_start
        while pos < row_end:
            blocks_to_assign = min(target_work, row_end - pos)

            worker_rows.append(row)
            worker_block_start.append(pos)
            worker_block_end.append(pos + blocks_to_assign)

            # Needs atomic if not processing the complete row
            is_partial = (pos > row_start) or (pos + blocks_to_assign < row_end)
            worker_needs_atomic.append(is_partial)

            pos += blocks_to_assign

    actual_n_workers = len(worker_rows)

    device = void_tensor.values.device

    return StreamKWorkload(
        worker_rows=torch.tensor(worker_rows, dtype=torch.int32, device=device),
        worker_block_start=torch.tensor(worker_block_start, dtype=torch.int32, device=device),
        worker_block_end=torch.tensor(worker_block_end, dtype=torch.int32, device=device),
        worker_needs_atomic=torch.tensor(worker_needs_atomic, dtype=torch.bool, device=device),
        n_workers=actual_n_workers,
        total_work=total_work,
    )


@triton.jit
def stream_k_spmm_kernel(
    # Sparse matrix A
    a_values_ptr,
    a_block_cols_ptr,
    a_block_idx_ptr,
    # Stream-K workload
    worker_rows_ptr,
    worker_block_start_ptr,
    worker_block_end_ptr,
    worker_needs_atomic_ptr,
    # Dense matrix B
    b_ptr,
    # Output matrix C
    c_ptr,
    # Lock for atomics (one per output row-tile)
    locks_ptr,
    # Dimensions
    M, N, K,
    n_workers,
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
    Stream-K SpMM kernel with load balancing.

    Each program (worker) processes its assigned block range.
    Partial results are accumulated atomically.
    """
    pid = tl.program_id(0)  # Worker ID
    pid_n = tl.program_id(1)  # Output tile column

    if pid >= n_workers:
        return

    # Get this worker's assignment
    row_idx = tl.load(worker_rows_ptr + pid)
    block_start = tl.load(worker_block_start_ptr + pid)
    block_end = tl.load(worker_block_end_ptr + pid)
    needs_atomic = tl.load(worker_needs_atomic_ptr + pid)

    # Initialize accumulator in FP32 for stability
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    col_start = pid_n * TILE_N

    # Process assigned blocks
    for block_idx in range(block_start, block_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load A tile
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

    # Store result with target dtype
    out_row = row_idx * TILE_M

    if needs_atomic:
        # Use spin-lock for atomic accumulation
        lock_id = row_idx * ((N + TILE_N - 1) // TILE_N) + pid_n

        # Acquire lock
        while tl.atomic_cas(locks_ptr + lock_id, 0, 1) == 1:
            pass

        # Read-modify-write
        c_tile_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(out_row, col_start),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        existing = tl.load(c_tile_ptr, boundary_check=(0, 1)).to(tl.float32)
        tl.store(c_tile_ptr, (existing + acc).to(OUTPUT_DTYPE), boundary_check=(0, 1))

        # Release lock
        tl.atomic_xchg(locks_ptr + lock_id, 0)
    else:
        # Direct store (no conflict)
        c_tile_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(out_row, col_start),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        tl.store(c_tile_ptr, acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def void_spmm_stream_k(
    a: VOIDTensor,
    b: torch.Tensor,
    workload: Optional[StreamKWorkload] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Stream-K load-balanced SpMM.

    Args:
        a: VOID sparse matrix
        b: Dense matrix [K, N]
        workload: Pre-computed workload (optional, will compute if None)
        out: Output buffer (optional)

    Returns:
        Dense output [M, N]
    """
    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert b.is_cuda and a.values.is_cuda

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    # Allocate output
    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        return out

    # Compute workload if not provided
    if workload is None:
        workload = compute_stream_k_workload(a)

    # Get block indices (sorted by row)
    _, block_indices = a.get_row_block_info()

    # Allocate locks for atomic accumulation
    n_output_tiles = ((M + tile_m - 1) // tile_m) * ((N + 64 - 1) // 64)
    locks = torch.zeros(n_output_tiles, dtype=torch.int32, device=b.device)

    TILE_N = min(64, triton.next_power_of_2(N))

    # Get output dtype for Triton
    output_dtype = get_triton_dtype(b.dtype)

    grid = (workload.n_workers, triton.cdiv(N, TILE_N))

    stream_k_spmm_kernel[grid](
        a.values, a.block_cols, block_indices,
        workload.worker_rows, workload.worker_block_start,
        workload.worker_block_end, workload.worker_needs_atomic,
        b, out, locks,
        M, N, K, workload.n_workers,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
        OUTPUT_DTYPE=output_dtype,
    )

    return out


# =============================================================================
# Work Stealing Variant (Alternative to Stream-K)
# =============================================================================

@triton.jit
def work_stealing_spmm_kernel(
    # Sparse matrix
    a_values_ptr,
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B
    b_ptr,
    # Output C
    c_ptr,
    # Work queue
    work_counter_ptr,  # Atomic counter for work stealing
    # Dimensions
    M, N, K,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Work-stealing SpMM kernel.

    Thread blocks dynamically claim rows from a shared work queue.
    """
    pid_n = tl.program_id(1)
    col_start = pid_n * TILE_N

    # Work stealing loop
    while True:
        # Atomically claim next row
        row_idx = tl.atomic_add(work_counter_ptr, 1)

        if row_idx >= n_block_rows:
            break

        # Get blocks for this row
        row_start = tl.load(a_row_ptr_ptr + row_idx)
        row_end = tl.load(a_row_ptr_ptr + row_idx + 1)

        if row_start == row_end:
            continue

        # Initialize accumulator in FP32 for stability
        acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        # Process all blocks in this row
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

        # Store output with target dtype
        out_row = row_idx * TILE_M
        c_tile_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(out_row, col_start),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        tl.store(c_tile_ptr, acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def void_spmm_work_stealing(
    a: VOIDTensor,
    b: torch.Tensor,
    n_workers: int = 256,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Work-stealing SpMM for highly imbalanced workloads.

    Args:
        a: VOID sparse matrix
        b: Dense matrix
        n_workers: Number of parallel workers
        out: Output buffer

    Returns:
        Dense output
    """
    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]

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

    # Work counter (atomic)
    work_counter = torch.zeros(1, dtype=torch.int32, device=b.device)

    TILE_N = min(64, triton.next_power_of_2(N))

    # Get output dtype for Triton
    output_dtype = get_triton_dtype(b.dtype)

    # Grid: n_workers x output_tile_columns
    grid = (n_workers, triton.cdiv(N, TILE_N))

    work_stealing_spmm_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, out, work_counter,
        M, N, K, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
        OUTPUT_DTYPE=output_dtype,
    )

    return out


# =============================================================================
# Utility: Analyze workload imbalance
# =============================================================================

def analyze_workload_balance(void_tensor: VOIDTensor) -> dict:
    """
    Analyze the workload distribution of a VOID tensor.

    Returns statistics about row imbalance.
    """
    row_ptr, _ = void_tensor.get_row_block_info()
    row_work = (row_ptr[1:] - row_ptr[:-1]).float()

    non_empty = row_work > 0
    row_work_nonzero = row_work[non_empty]

    if len(row_work_nonzero) == 0:
        return {"empty": True}

    return {
        "empty": False,
        "n_rows": len(row_work),
        "n_nonempty_rows": non_empty.sum().item(),
        "min_blocks_per_row": row_work_nonzero.min().item(),
        "max_blocks_per_row": row_work_nonzero.max().item(),
        "mean_blocks_per_row": row_work_nonzero.mean().item(),
        "std_blocks_per_row": row_work_nonzero.std().item(),
        "imbalance_ratio": (row_work_nonzero.max() / row_work_nonzero.mean()).item(),
        "coefficient_of_variation": (row_work_nonzero.std() / row_work_nonzero.mean()).item(),
    }
