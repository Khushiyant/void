"""
Validation and Performance Comparison: VOID vs cuSPARSE vs cuBLAS

This script validates correctness and compares performance across:
1. VOID (our implementation)
2. cuSPARSE (torch.sparse.mm)
3. cuBLAS (dense matmul baseline)
"""

import torch
import scipy.sparse as sp
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import csr_to_void, void_spmm


@dataclass
class BenchmarkConfig:
    M: int
    K: int
    N: int
    sparsity: float
    tile_size: int = 32


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    void_time_ms: float
    cusparse_time_ms: float
    dense_time_ms: float
    void_error: float
    cusparse_error: float
    void_blocks: int
    void_overhead: float


def create_sparse_matrix(M: int, K: int, sparsity: float) -> sp.csr_matrix:
    """Create a random sparse matrix."""
    density = 1.0 - sparsity
    return sp.random(M, K, density=density, format='csr', dtype=np.float32)


def benchmark_single(
    config: BenchmarkConfig,
    n_iterations: int = 100,
    warmup: int = 20,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    device = torch.device("cuda")

    # Create sparse matrix
    sparse_np = create_sparse_matrix(config.M, config.K, config.sparsity)

    # Convert to various formats
    void_tensor = csr_to_void(sparse_np, tile_size=config.tile_size).cuda()

    # PyTorch sparse CSR
    crow_indices = torch.tensor(sparse_np.indptr, dtype=torch.int32, device=device)
    col_indices = torch.tensor(sparse_np.indices, dtype=torch.int32, device=device)
    values = torch.tensor(sparse_np.data, dtype=torch.float32, device=device)
    torch_sparse = torch.sparse_csr_tensor(
        crow_indices, col_indices, values,
        size=(config.M, config.K), device=device
    )

    # Dense reference
    A_dense = torch.tensor(sparse_np.toarray(), device=device)

    # Dense B matrix
    B = torch.randn(config.K, config.N, device=device, dtype=torch.float32)

    # Reference result
    C_ref = A_dense @ B

    # =========================================================================
    # Correctness validation
    # =========================================================================

    C_void = void_spmm(void_tensor, B)
    void_error = (C_void - C_ref).abs().max().item() / C_ref.abs().max().item()

    C_cusparse = torch.sparse.mm(torch_sparse, B)
    cusparse_error = (C_cusparse - C_ref).abs().max().item() / C_ref.abs().max().item()

    # =========================================================================
    # Performance benchmarking
    # =========================================================================

    # Warmup
    for _ in range(warmup):
        _ = void_spmm(void_tensor, B)
        _ = torch.sparse.mm(torch_sparse, B)
        _ = A_dense @ B
    torch.cuda.synchronize()

    # VOID
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm(void_tensor, B)
    torch.cuda.synchronize()
    void_time = (time.perf_counter() - start) / n_iterations * 1000

    # cuSPARSE
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = torch.sparse.mm(torch_sparse, B)
    torch.cuda.synchronize()
    cusparse_time = (time.perf_counter() - start) / n_iterations * 1000

    # Dense (cuBLAS)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = A_dense @ B
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / n_iterations * 1000

    return BenchmarkResult(
        config=config,
        void_time_ms=void_time,
        cusparse_time_ms=cusparse_time,
        dense_time_ms=dense_time,
        void_error=void_error,
        cusparse_error=cusparse_error,
        void_blocks=void_tensor.n_blocks,
        void_overhead=void_tensor.overhead_ratio,
    )


def run_sparsity_sweep(
    M: int = 2048,
    K: int = 2048,
    N: int = 512,
    sparsities: Optional[List[float]] = None,
    tile_size: int = 32,
) -> List[BenchmarkResult]:
    """Sweep across sparsity levels."""
    if sparsities is None:
        sparsities = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    results = []
    for sparsity in sparsities:
        config = BenchmarkConfig(M, K, N, sparsity, tile_size)
        print(f"Testing sparsity {sparsity:.0%}...", end=" ", flush=True)
        try:
            result = benchmark_single(config)
            results.append(result)
            print(f"VOID: {result.void_time_ms:.2f}ms, cuSPARSE: {result.cusparse_time_ms:.2f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
        torch.cuda.empty_cache()

    return results


def run_size_sweep(
    sizes: Optional[List[Tuple[int, int, int]]] = None,
    sparsity: float = 0.9,
    tile_size: int = 32,
) -> List[BenchmarkResult]:
    """Sweep across matrix sizes."""
    if sizes is None:
        sizes = [
            (512, 512, 256),
            (1024, 1024, 256),
            (2048, 2048, 512),
            (4096, 4096, 512),
            (8192, 8192, 256),
        ]

    results = []
    for M, K, N in sizes:
        config = BenchmarkConfig(M, K, N, sparsity, tile_size)
        print(f"Testing {M}x{K} @ {K}x{N}...", end=" ", flush=True)
        try:
            result = benchmark_single(config)
            results.append(result)
            print(f"VOID: {result.void_time_ms:.2f}ms, cuSPARSE: {result.cusparse_time_ms:.2f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
        torch.cuda.empty_cache()

    return results


def print_results_table(results: List[BenchmarkResult], title: str = "Results"):
    """Print results in a formatted table."""
    print(f"\n{'=' * 90}")
    print(f"{title}")
    print(f"{'=' * 90}")

    header = f"{'Size':<20} {'Sparsity':>8} {'VOID (ms)':>10} {'cuSPARSE':>10} {'Dense':>10} {'Speedup':>10} {'Error':>10}"
    print(header)
    print("-" * 90)

    for r in results:
        size_str = f"{r.config.M}x{r.config.K}x{r.config.N}"
        speedup = r.cusparse_time_ms / r.void_time_ms if r.void_time_ms > 0 else 0
        print(
            f"{size_str:<20} {r.config.sparsity:>7.0%} "
            f"{r.void_time_ms:>10.2f} {r.cusparse_time_ms:>10.2f} "
            f"{r.dense_time_ms:>10.2f} {speedup:>9.2f}x "
            f"{r.void_error:>10.2e}"
        )

    print("-" * 90)

    # Summary
    avg_speedup = np.mean([r.cusparse_time_ms / r.void_time_ms for r in results if r.void_time_ms > 0])
    max_error = max(r.void_error for r in results)
    print(f"Average speedup vs cuSPARSE: {avg_speedup:.2f}x")
    print(f"Maximum relative error: {max_error:.2e}")


def main():
    print("=" * 90)
    print("VOID Validation and Performance Comparison")
    print("=" * 90)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Check GPU memory and adjust sizes
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB")
    print()

    # Adjust sizes based on GPU memory
    if gpu_mem_gb < 8:
        base_M, base_K, base_N = 1024, 1024, 256
        sizes = [
            (512, 512, 128),
            (1024, 1024, 256),
            (2048, 2048, 256),
        ]
        print("Using reduced sizes for low-memory GPU")
    elif gpu_mem_gb < 16:
        base_M, base_K, base_N = 2048, 2048, 512
        sizes = [
            (512, 512, 256),
            (1024, 1024, 256),
            (2048, 2048, 512),
            (4096, 4096, 256),
        ]
        print("Using medium sizes")
    else:
        base_M, base_K, base_N = 2048, 2048, 512
        sizes = [
            (512, 512, 256),
            (1024, 1024, 256),
            (2048, 2048, 512),
            (4096, 4096, 512),
            (8192, 8192, 256),
        ]

    # Sparsity sweep
    print(f"\nSparsity Sweep ({base_M}x{base_K} @ {base_K}x{base_N})")
    print("-" * 50)
    sparsity_results = run_sparsity_sweep(M=base_M, K=base_K, N=base_N)
    if sparsity_results:
        print_results_table(sparsity_results, "Sparsity Sweep Results")

    torch.cuda.empty_cache()

    # Size sweep
    print("\n\nSize Sweep (90% sparsity)")
    print("-" * 50)
    size_results = run_size_sweep(sizes=sizes)
    if size_results:
        print_results_table(size_results, "Size Sweep Results")

    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    if sparsity_results:
        # Find crossover point
        for r in sparsity_results:
            if r.void_time_ms < r.cusparse_time_ms:
                print(f"VOID beats cuSPARSE at {r.config.sparsity:.0%} sparsity "
                      f"({r.void_time_ms:.2f}ms vs {r.cusparse_time_ms:.2f}ms)")

        # Find where VOID overhead is too high
        high_overhead = [r for r in sparsity_results if r.void_overhead > 2.0]
        if high_overhead:
            print(f"\nWarning: VOID padding overhead >2x at sparsities: "
                  f"{[f'{r.config.sparsity:.0%}' for r in high_overhead]}")
    else:
        print("No sparsity sweep results to analyze")

    print("\nDone!")


if __name__ == "__main__":
    main()
