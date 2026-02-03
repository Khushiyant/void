"""
Structured Sparsity Benchmark

Tests VOID performance on realistic sparsity patterns:
1. Block-diagonal: Common in scientific computing (FEM, PDEs)
2. Clustered: Common in graph neural networks
3. N:M Structured: Common in pruned neural networks
"""

import torch
import scipy.sparse as sp
import numpy as np
import time
from dataclasses import dataclass
from typing import List

import sys
sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import csr_to_void, void_spmm


@dataclass
class StructuredBenchmarkResult:
    pattern: str
    sparsity: float
    block_sparsity: float
    void_time_ms: float
    cusparse_time_ms: float
    dense_time_ms: float
    speedup_vs_cusparse: float
    speedup_vs_dense: float


def create_block_diagonal(M: int, K: int, block_size: int = 64, fill_density: float = 0.5) -> sp.csr_matrix:
    """
    Create block-diagonal sparse matrix.

    Each diagonal block is filled with `fill_density` non-zeros.
    This is common in FEM/scientific computing.
    """
    n_blocks = min(M, K) // block_size
    rows, cols, data = [], [], []

    for b in range(n_blocks):
        block_start_row = b * block_size
        block_start_col = b * block_size

        # Fill the block with random non-zeros
        for i in range(block_size):
            for j in range(block_size):
                if np.random.random() < fill_density:
                    rows.append(block_start_row + i)
                    cols.append(block_start_col + j)
                    data.append(np.random.randn())

    return sp.csr_matrix((data, (rows, cols)), shape=(M, K), dtype=np.float32)


def create_clustered(M: int, K: int, n_clusters: int = 32, cluster_size: int = 64, density: float = 0.8) -> sp.csr_matrix:
    """
    Create clustered sparse matrix.

    Non-zeros are concentrated in random cluster regions.
    This simulates graph adjacency patterns (communities).
    """
    rows, cols, data = [], [], []

    for _ in range(n_clusters):
        # Random cluster position
        row_start = np.random.randint(0, max(1, M - cluster_size))
        col_start = np.random.randint(0, max(1, K - cluster_size))

        for i in range(cluster_size):
            for j in range(cluster_size):
                if np.random.random() < density:
                    r = row_start + i
                    c = col_start + j
                    if r < M and c < K:
                        rows.append(r)
                        cols.append(c)
                        data.append(np.random.randn())

    # Remove duplicates by using COO's duplicate handling
    coo = sp.coo_matrix((data, (rows, cols)), shape=(M, K), dtype=np.float32)
    return coo.tocsr()


def create_nm_structured(M: int, K: int, n: int = 2, m: int = 4) -> sp.csr_matrix:
    """
    Create N:M structured sparse matrix.

    Every group of M consecutive elements has exactly N non-zeros.
    This is the sparsity pattern used in Ampere Tensor Cores.
    """
    # Vectorized implementation for speed
    n_groups_per_row = (K + m - 1) // m
    total_nnz = M * n_groups_per_row * n

    rows = np.repeat(np.arange(M), n_groups_per_row * n)
    cols = np.zeros(total_nnz, dtype=np.int32)
    data = np.random.randn(total_nnz).astype(np.float32)

    idx = 0
    for i in range(M):
        for group_start in range(0, K, m):
            group_size = min(m, K - group_start)
            positions = np.random.choice(group_size, size=min(n, group_size), replace=False)
            for j in positions:
                cols[idx] = group_start + j
                idx += 1

    # Trim to actual size
    rows = rows[:idx]
    cols = cols[:idx]
    data = data[:idx]

    return sp.csr_matrix((data, (rows, cols)), shape=(M, K), dtype=np.float32)


def create_banded(M: int, K: int, bandwidth: int = 64) -> sp.csr_matrix:
    """
    Create banded sparse matrix.

    Common in tridiagonal/pentadiagonal systems from PDEs.
    """
    rows, cols, data = [], [], []

    for i in range(M):
        j_start = max(0, i - bandwidth // 2)
        j_end = min(K, i + bandwidth // 2 + 1)
        for j in range(j_start, j_end):
            rows.append(i)
            cols.append(j)
            data.append(np.random.randn())

    return sp.csr_matrix((data, (rows, cols)), shape=(M, K), dtype=np.float32)


def benchmark_pattern(
    name: str,
    sparse_matrix: sp.csr_matrix,
    N: int = 512,
    tile_size: int = 32,
    n_iterations: int = 100,
    warmup: int = 20,
) -> StructuredBenchmarkResult:
    """Benchmark a specific sparsity pattern."""
    device = torch.device("cuda")
    M, K = sparse_matrix.shape

    # VOID format
    void_tensor = csr_to_void(sparse_matrix, tile_size=tile_size).cuda()

    # PyTorch sparse
    crow = torch.tensor(sparse_matrix.indptr, dtype=torch.int32, device=device)
    col = torch.tensor(sparse_matrix.indices, dtype=torch.int32, device=device)
    val = torch.tensor(sparse_matrix.data, dtype=torch.float32, device=device)
    torch_sparse = torch.sparse_csr_tensor(crow, col, val, size=(M, K), device=device)

    # Dense
    A_dense = torch.tensor(sparse_matrix.toarray(), device=device)

    # B matrix
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = void_spmm(void_tensor, B)
        _ = torch.sparse.mm(torch_sparse, B)
        _ = A_dense @ B
    torch.cuda.synchronize()

    # Benchmark VOID
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm(void_tensor, B)
    torch.cuda.synchronize()
    void_time = (time.perf_counter() - start) / n_iterations * 1000

    # Benchmark cuSPARSE
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = torch.sparse.mm(torch_sparse, B)
    torch.cuda.synchronize()
    cusparse_time = (time.perf_counter() - start) / n_iterations * 1000

    # Benchmark Dense
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = A_dense @ B
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / n_iterations * 1000

    sparsity = 1.0 - sparse_matrix.nnz / (M * K)

    return StructuredBenchmarkResult(
        pattern=name,
        sparsity=sparsity,
        block_sparsity=void_tensor.block_sparsity,
        void_time_ms=void_time,
        cusparse_time_ms=cusparse_time,
        dense_time_ms=dense_time,
        speedup_vs_cusparse=cusparse_time / void_time if void_time > 0 else 0,
        speedup_vs_dense=dense_time / void_time if void_time > 0 else 0,
    )


def main():
    print("=" * 90)
    print("Structured Sparsity Benchmark")
    print("=" * 90)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Check available GPU memory and adjust sizes accordingly
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB")

    # Use smaller sizes for GPUs with less memory
    if gpu_mem_gb < 8:
        M, K, N = 1024, 1024, 128
        print("Using reduced sizes for low-memory GPU")
    elif gpu_mem_gb < 16:
        M, K, N = 2048, 2048, 256
        print("Using medium sizes")
    else:
        M, K, N = 4096, 4096, 512

    print(f"Matrix sizes: M={M}, K={K}, N={N}")
    print()
    results: List[StructuredBenchmarkResult] = []

    def run_benchmark_safe(name, matrix_fn):
        """Run benchmark with error handling and memory cleanup."""
        try:
            print(f"Creating {name} matrix...", flush=True)
            matrix = matrix_fn()
            result = benchmark_pattern(name, matrix, N)
            print(f"  Sparsity: {result.sparsity:.1%}, Block sparsity: {result.block_sparsity:.1%}")
            print(f"  VOID: {result.void_time_ms:.2f}ms, cuSPARSE: {result.cusparse_time_ms:.2f}ms")
            # Clean up GPU memory
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            print(f"  FAILED: {e}")
            torch.cuda.empty_cache()
            return None

    # Block-diagonal
    result = run_benchmark_safe("Block-Diagonal",
        lambda: create_block_diagonal(M, K, block_size=64, fill_density=0.5))
    if result:
        results.append(result)

    # Clustered
    result = run_benchmark_safe("Clustered",
        lambda: create_clustered(M, K, n_clusters=32, cluster_size=64, density=0.8))
    if result:
        results.append(result)

    # N:M Structured (2:4)
    result = run_benchmark_safe("2:4 Structured",
        lambda: create_nm_structured(M, K, n=2, m=4))
    if result:
        results.append(result)

    # Banded
    result = run_benchmark_safe("Banded",
        lambda: create_banded(M, K, bandwidth=64))
    if result:
        results.append(result)

    # Random (for comparison)
    result = run_benchmark_safe("Random",
        lambda: sp.random(M, K, density=0.1, format='csr', dtype=np.float32))
    if result:
        results.append(result)

    # Results table
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"{'Pattern':<20} {'Sparsity':>10} {'Block Sp.':>10} {'VOID (ms)':>10} {'cuSPARSE':>10} {'Speedup':>10}")
    print("-" * 90)

    for r in results:
        print(
            f"{r.pattern:<20} {r.sparsity:>9.1%} {r.block_sparsity:>9.1%} "
            f"{r.void_time_ms:>10.2f} {r.cusparse_time_ms:>10.2f} "
            f"{r.speedup_vs_cusparse:>9.2f}x"
        )

    print("-" * 90)

    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    best = max(results, key=lambda r: r.speedup_vs_cusparse)
    print(f"Best speedup: {best.pattern} with {best.speedup_vs_cusparse:.2f}x over cuSPARSE")

    structured_results = [r for r in results if r.pattern != "Random"]
    if structured_results:
        avg_structured = np.mean([r.speedup_vs_cusparse for r in structured_results])
        print(f"Average speedup on structured patterns: {avg_structured:.2f}x")

    print("\nKey insight: VOID excels when non-zeros cluster into blocks")
    print("(block_sparsity > 0 means some tiles are entirely empty)")


if __name__ == "__main__":
    main()
