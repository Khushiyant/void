"""
Hardware Bandwidth Feasibility Benchmark for Project VOID

This benchmark validates the core hypothesis:
- Block-gather memory access patterns can achieve >70% peak HBM bandwidth
- Even with non-contiguous tile access, coalesced block loads outperform scalar loads

Tests:
1. Sequential (baseline): Fully coalesced linear memory read
2. Block-gather: Load random 2D tiles (simulating sparse block access)
3. Scalar-gather: Random element access (simulating CSR pointer-chasing)

Success criteria: Block-gather achieves >70% of sequential bandwidth
"""

import torch
import triton
import triton.language as tl
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BenchmarkResult:
    name: str
    tile_size: int
    bandwidth_gbps: float
    efficiency_pct: float
    time_ms: float


def get_gpu_info() -> dict:
    """Get GPU specifications for bandwidth calculations."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    props = torch.cuda.get_device_properties(0)

    # Theoretical peak bandwidth (GB/s)
    # memory_clock is in kHz, bus_width in bits
    peak_bandwidth = (props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) * 2) / 1e9

    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / 1e9,
        "peak_bandwidth_gbps": peak_bandwidth,
        "sm_count": props.multi_processor_count,
    }


# =============================================================================
# Triton Kernels for Bandwidth Testing
# =============================================================================

@triton.jit
def sequential_read_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Baseline: Sequential coalesced memory read."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


@triton.jit
def block_gather_kernel(
    input_ptr,
    output_ptr,
    block_indices_ptr,  # Which blocks to load
    n_blocks,
    matrix_stride,  # Stride between rows in the source matrix
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Block-gather: Load 2D tiles from random locations."""
    pid = tl.program_id(0)

    if pid >= n_blocks:
        return

    # Get the block index (row, col) for this program
    block_idx = tl.load(block_indices_ptr + pid * 2)      # block_row
    block_idy = tl.load(block_indices_ptr + pid * 2 + 1)  # block_col

    # Compute base offset in source matrix
    base_row = block_idx * TILE_M
    base_col = block_idy * TILE_N

    # Load the entire tile using 2D indexing
    row_offsets = tl.arange(0, TILE_M)
    col_offsets = tl.arange(0, TILE_N)

    # 2D tile load - this is the key operation we're benchmarking
    for i in range(TILE_M):
        row_offset = (base_row + i) * matrix_stride + base_col
        cols = tl.arange(0, TILE_N)
        data = tl.load(input_ptr + row_offset + cols)

        # Store to output (contiguous for simplicity)
        out_offset = pid * TILE_M * TILE_N + i * TILE_N
        tl.store(output_ptr + out_offset + cols, data)


@triton.jit
def block_gather_optimized_kernel(
    input_ptr,
    output_ptr,
    block_row_ptr,
    block_col_ptr,
    n_blocks,
    matrix_rows,
    matrix_cols,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Optimized block-gather using block pointers (Triton 2.0+)."""
    pid = tl.program_id(0)

    if pid >= n_blocks:
        return

    block_row = tl.load(block_row_ptr + pid)
    block_col = tl.load(block_col_ptr + pid)

    base_row = block_row * TILE_M
    base_col = block_col * TILE_N

    # Use make_block_ptr for efficient 2D tile loading
    tile_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(matrix_rows, matrix_cols),
        strides=(matrix_cols, 1),
        offsets=(base_row, base_col),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    tile_data = tl.load(tile_ptr, boundary_check=(0, 1))

    # Store output contiguously
    out_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_blocks * TILE_M, TILE_N),
        strides=(TILE_N, 1),
        offsets=(pid * TILE_M, 0),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    tl.store(out_ptr, tile_data)


@triton.jit
def scalar_gather_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Scalar-gather: Random element access (simulates CSR)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Random indices - worst case for coalescing
    indices = tl.load(indices_ptr + offsets, mask=mask)
    data = tl.load(input_ptr + indices, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_sequential(
    data_size_mb: int = 256,
    n_iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark sequential memory read (baseline)."""
    n_elements = (data_size_mb * 1024 * 1024) // 4  # float32

    input_tensor = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Warmup
    for _ in range(warmup):
        sequential_read_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        sequential_read_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_transferred = n_elements * 4 * 2 * n_iterations  # read + write
    bandwidth = bytes_transferred / elapsed / 1e9

    gpu_info = get_gpu_info()
    efficiency = (bandwidth / gpu_info['peak_bandwidth_gbps']) * 100

    return BenchmarkResult(
        name="Sequential",
        tile_size=0,
        bandwidth_gbps=bandwidth,
        efficiency_pct=efficiency,
        time_ms=(elapsed / n_iterations) * 1000,
    )


def benchmark_block_gather(
    tile_size: int = 32,
    matrix_size: int = 4096,
    n_blocks: int = 1024,
    n_iterations: int = 100,
    warmup: int = 10,
    use_optimized: bool = True,
) -> BenchmarkResult:
    """Benchmark block-gather memory access pattern."""
    # Create source matrix
    source = torch.randn(matrix_size, matrix_size, device='cuda', dtype=torch.float32)

    # Generate random block indices (non-overlapping for fair comparison)
    max_blocks_per_dim = matrix_size // tile_size
    n_blocks = min(n_blocks, max_blocks_per_dim * max_blocks_per_dim)

    # Random unique block positions
    all_positions = torch.randperm(max_blocks_per_dim * max_blocks_per_dim, device='cuda')[:n_blocks]
    block_rows = (all_positions // max_blocks_per_dim).to(torch.int32)
    block_cols = (all_positions % max_blocks_per_dim).to(torch.int32)

    # Output buffer
    output = torch.empty(n_blocks * tile_size * tile_size, device='cuda', dtype=torch.float32)

    grid = (n_blocks,)

    if use_optimized:
        kernel = block_gather_optimized_kernel
        kernel_args = (
            source, output, block_rows, block_cols,
            n_blocks, matrix_size, matrix_size, tile_size, tile_size
        )
    else:
        # Pack indices for non-optimized kernel
        block_indices = torch.stack([block_rows, block_cols], dim=1).flatten().contiguous()
        kernel = block_gather_kernel
        kernel_args = (
            source, output, block_indices,
            n_blocks, matrix_size, tile_size, tile_size
        )

    # Warmup
    for _ in range(warmup):
        kernel[grid](*kernel_args)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        kernel[grid](*kernel_args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate bandwidth
    bytes_per_tile = tile_size * tile_size * 4
    bytes_transferred = n_blocks * bytes_per_tile * 2 * n_iterations  # read + write
    bandwidth = bytes_transferred / elapsed / 1e9

    gpu_info = get_gpu_info()
    efficiency = (bandwidth / gpu_info['peak_bandwidth_gbps']) * 100

    return BenchmarkResult(
        name=f"Block-Gather-{tile_size}x{tile_size}",
        tile_size=tile_size,
        bandwidth_gbps=bandwidth,
        efficiency_pct=efficiency,
        time_ms=(elapsed / n_iterations) * 1000,
    )


def benchmark_scalar_gather(
    data_size_mb: int = 256,
    n_iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark scalar-gather (CSR-like random access)."""
    n_elements = (data_size_mb * 1024 * 1024) // 4

    input_tensor = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty(n_elements, device='cuda', dtype=torch.float32)

    # Random indices - simulates CSR column indices
    indices = torch.randint(0, n_elements, (n_elements,), device='cuda', dtype=torch.int64)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Warmup
    for _ in range(warmup):
        scalar_gather_kernel[grid](input_tensor, output_tensor, indices, n_elements, BLOCK_SIZE)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        scalar_gather_kernel[grid](input_tensor, output_tensor, indices, n_elements, BLOCK_SIZE)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_transferred = n_elements * 4 * 2 * n_iterations
    bandwidth = bytes_transferred / elapsed / 1e9

    gpu_info = get_gpu_info()
    efficiency = (bandwidth / gpu_info['peak_bandwidth_gbps']) * 100

    return BenchmarkResult(
        name="Scalar-Gather",
        tile_size=1,
        bandwidth_gbps=bandwidth,
        efficiency_pct=efficiency,
        time_ms=(elapsed / n_iterations) * 1000,
    )


def run_full_benchmark() -> Tuple[List[BenchmarkResult], bool]:
    """
    Run the complete feasibility benchmark suite.

    Returns:
        results: List of benchmark results
        feasible: Whether the approach is feasible (block-gather > 70% efficiency)
    """
    print("=" * 70)
    print("Project VOID - Hardware Bandwidth Feasibility Benchmark")
    print("=" * 70)

    # GPU Info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"Peak Bandwidth: {gpu_info['peak_bandwidth_gbps']:.1f} GB/s")
    print(f"SM Count: {gpu_info['sm_count']}")
    print()

    # Adjust data size based on GPU memory
    if gpu_info['total_memory_gb'] < 8:
        data_size_mb = 64
        matrix_size = 2048
        n_blocks = 512
    elif gpu_info['total_memory_gb'] < 16:
        data_size_mb = 128
        matrix_size = 4096
        n_blocks = 1024
    else:
        data_size_mb = 256
        matrix_size = 4096
        n_blocks = 1024

    print(f"Using data_size={data_size_mb}MB, matrix_size={matrix_size}")
    print()

    results = []

    # 1. Sequential baseline
    print("Running Sequential (baseline)...")
    try:
        seq_result = benchmark_sequential(data_size_mb=data_size_mb)
        results.append(seq_result)
        print(f"  Bandwidth: {seq_result.bandwidth_gbps:.1f} GB/s ({seq_result.efficiency_pct:.1f}%)")
    except Exception as e:
        print(f"  Failed: {e}")
        seq_result = None

    # 2. Scalar-gather (CSR simulation)
    print("Running Scalar-Gather (CSR simulation)...")
    try:
        scalar_result = benchmark_scalar_gather(data_size_mb=data_size_mb)
        results.append(scalar_result)
        print(f"  Bandwidth: {scalar_result.bandwidth_gbps:.1f} GB/s ({scalar_result.efficiency_pct:.1f}%)")
    except Exception as e:
        print(f"  Failed: {e}")
        scalar_result = BenchmarkResult("Scalar-Gather", 1, 0.0, 0.0, 0.0)

    torch.cuda.empty_cache()

    # 3. Block-gather with different tile sizes
    tile_sizes = [16, 32, 64]
    for tile_size in tile_sizes:
        print(f"Running Block-Gather ({tile_size}x{tile_size})...")
        try:
            block_result = benchmark_block_gather(
                tile_size=tile_size,
                matrix_size=matrix_size,
                n_blocks=n_blocks
            )
            results.append(block_result)
            print(f"  Bandwidth: {block_result.bandwidth_gbps:.1f} GB/s ({block_result.efficiency_pct:.1f}%)")
        except Exception as e:
            print(f"  Failed: {e}")
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<30} {'Bandwidth (GB/s)':<20} {'Efficiency':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<30} {r.bandwidth_gbps:<20.1f} {r.efficiency_pct:<15.1f}%")

    # Feasibility check
    block_results = [r for r in results if r.name.startswith("Block-Gather")]
    best_block = max(block_results, key=lambda r: r.efficiency_pct) if block_results else None

    print("\n" + "=" * 70)
    print("FEASIBILITY ASSESSMENT")
    print("=" * 70)

    if best_block:
        feasible = best_block.efficiency_pct >= 70.0
        if scalar_result and scalar_result.bandwidth_gbps > 0:
            speedup_vs_scalar = best_block.bandwidth_gbps / scalar_result.bandwidth_gbps
        else:
            speedup_vs_scalar = 0.0

        print(f"Best Block-Gather: {best_block.name}")
        print(f"  Efficiency: {best_block.efficiency_pct:.1f}% (target: >=70%)")
        print(f"  Speedup vs Scalar-Gather: {speedup_vs_scalar:.1f}x")

        if feasible:
            print(f"\n[PASS] Block-gather achieves {best_block.efficiency_pct:.1f}% bandwidth efficiency")
            print(f"       VOID approach is FEASIBLE with {best_block.tile_size}x{best_block.tile_size} tiles")
        else:
            print(f"\n[WARN] Block-gather only achieves {best_block.efficiency_pct:.1f}% efficiency")
            print(f"       Consider smaller tiles or different memory layout")

        return results, feasible
    else:
        print("[FAIL] No block-gather results available")
        return results, False


if __name__ == "__main__":
    results, feasible = run_full_benchmark()
    exit(0 if feasible else 1)
