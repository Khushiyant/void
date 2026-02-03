"""
BSR (Block Sparse Row) Comparison Benchmark

This is the CRITICAL benchmark - compares VOID against cuSPARSE's BSR format,
which is the actual competitor (not CSR).

BSR stores sparse matrices as blocks, just like VOID, so this is an apples-to-apples
comparison to validate whether VOID's optimizations (Morton ordering, Triton kernels)
actually provide speedup over the standard cuSPARSE BSR implementation.
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
class BSRBenchmarkResult:
    """Results from BSR vs VOID comparison."""
    matrix_size: Tuple[int, int, int]  # (M, K, N)
    sparsity: float
    block_size: int
    void_time_ms: float
    bsr_time_ms: float
    csr_time_ms: float
    dense_time_ms: float
    void_vs_bsr_speedup: float
    void_vs_csr_speedup: float
    void_blocks: int
    bsr_blocks: int
    void_overhead: float


def create_block_sparse_matrix(M, K, block_sparsity, block_size=32):
    """Create a matrix with actual block sparsity (not element sparsity)."""
    n_block_rows = M // block_size
    n_block_cols = K // block_size
    data = np.zeros((M, K), dtype=np.float32)

    for br in range(n_block_rows):
        for bc in range(n_block_cols):
            if np.random.random() > block_sparsity:
                r_start, r_end = br * block_size, (br + 1) * block_size
                c_start, c_end = bc * block_size, (bc + 1) * block_size
                data[r_start:r_end, c_start:c_end] = np.random.randn(block_size, block_size) * 0.1

    return sp.csr_matrix(data)


def benchmark_bsr_vs_void(
    M: int,
    K: int,
    N: int,
    sparsity: float,
    block_size: int = 32,
    n_iterations: int = 100,
    warmup: int = 20,
) -> BSRBenchmarkResult:
    """
    Compare VOID against cuSPARSE BSR format.

    This is the key benchmark to prove VOID's value.
    Uses BLOCK sparsity (not element sparsity) for fair comparison.
    """
    device = torch.device("cuda")

    # Create block-sparse matrix (this is what VOID is designed for)
    sparse_csr = create_block_sparse_matrix(M, K, sparsity, block_size)

    # Convert to BSR format (this is VOID's competitor)
    try:
        sparse_bsr = sparse_csr.tobsr(blocksize=(block_size, block_size))
    except Exception as e:
        print(f"  Warning: BSR conversion failed: {e}, using CSR blocks")
        # Fallback: manually create block structure
        sparse_bsr = sparse_csr.tobsr(blocksize=(block_size, block_size))

    # VOID format
    void_tensor = csr_to_void(sparse_csr, tile_size=block_size).cuda()

    # PyTorch BSR tensor (PyTorch >= 2.0)
    # Note: PyTorch's BSR support may be limited, handle gracefully
    bsr_tensor = None
    try:
        bsr_crow = torch.from_numpy(sparse_bsr.indptr).to(torch.int32)
        bsr_col = torch.from_numpy(sparse_bsr.indices).to(torch.int32)
        bsr_values = torch.from_numpy(sparse_bsr.data).to(torch.float32)

        bsr_tensor = torch.sparse_bsr_tensor(
            crow_indices=bsr_crow,
            col_indices=bsr_col,
            values=bsr_values,
            size=(M, K),
            device=device
        )
    except (AttributeError, RuntimeError) as e:
        print(f"  Warning: PyTorch BSR not available ({e}), skipping BSR benchmark")
        bsr_tensor = None

    # PyTorch CSR tensor (for comparison)
    csr_crow = torch.tensor(sparse_csr.indptr, dtype=torch.int32, device=device)
    csr_col = torch.tensor(sparse_csr.indices, dtype=torch.int32, device=device)
    csr_values = torch.tensor(sparse_csr.data, dtype=torch.float32, device=device)
    csr_tensor = torch.sparse_csr_tensor(csr_crow, csr_col, csr_values, size=(M, K), device=device)

    # Dense reference
    A_dense = torch.tensor(sparse_csr.toarray(), device=device, dtype=torch.float32)

    # Dense B matrix
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = void_spmm(void_tensor, B)
        if bsr_tensor is not None:
            _ = torch.sparse.mm(bsr_tensor, B)
        _ = torch.sparse.mm(csr_tensor, B)
        _ = A_dense @ B
    torch.cuda.synchronize()

    # =========================================================================
    # Benchmark VOID
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm(void_tensor, B)
    torch.cuda.synchronize()
    void_time = (time.perf_counter() - start) / n_iterations * 1000

    # =========================================================================
    # Benchmark cuSPARSE BSR (THE CRITICAL COMPARISON)
    # =========================================================================
    bsr_time = None
    if bsr_tensor is not None:
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = torch.sparse.mm(bsr_tensor, B)
        torch.cuda.synchronize()
        bsr_time = (time.perf_counter() - start) / n_iterations * 1000
    else:
        bsr_time = float('inf')  # Mark as unavailable

    # =========================================================================
    # Benchmark cuSPARSE CSR (for reference)
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = torch.sparse.mm(csr_tensor, B)
    torch.cuda.synchronize()
    csr_time = (time.perf_counter() - start) / n_iterations * 1000

    # =========================================================================
    # Benchmark Dense cuBLAS (baseline)
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = A_dense @ B
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / n_iterations * 1000

    return BSRBenchmarkResult(
        matrix_size=(M, K, N),
        sparsity=sparsity,
        block_size=block_size,
        void_time_ms=void_time,
        bsr_time_ms=bsr_time,
        csr_time_ms=csr_time,
        dense_time_ms=dense_time,
        void_vs_bsr_speedup=bsr_time / void_time if bsr_time and bsr_time != float('inf') else 0,
        void_vs_csr_speedup=csr_time / void_time,
        void_blocks=void_tensor.n_blocks,
        bsr_blocks=sparse_bsr.nnz // (block_size * block_size) if sparse_bsr.nnz > 0 else 0,
        void_overhead=void_tensor.overhead_ratio,
    )


def run_sparsity_sweep_bsr(
    M: int = 4096,
    K: int = 4096,
    N: int = 512,
    block_size: int = 32,
    sparsities: Optional[List[float]] = None,
) -> List[BSRBenchmarkResult]:
    """Sweep sparsity levels comparing VOID vs BSR."""
    if sparsities is None:
        sparsities = [0.7, 0.8, 0.9, 0.95, 0.98]

    results = []
    for sparsity in sparsities:
        print(f"Testing sparsity {sparsity:.0%}...", end=" ", flush=True)
        try:
            result = benchmark_bsr_vs_void(M, K, N, sparsity, block_size)
            results.append(result)

            if result.bsr_time_ms != float('inf'):
                print(f"VOID: {result.void_time_ms:.2f}ms, BSR: {result.bsr_time_ms:.2f}ms, "
                      f"Speedup: {result.void_vs_bsr_speedup:.2f}x")
            else:
                print(f"VOID: {result.void_time_ms:.2f}ms, BSR: N/A, CSR: {result.csr_time_ms:.2f}ms")

        except Exception as e:
            print(f"FAILED: {e}")

        torch.cuda.empty_cache()

    return results


def run_block_size_sweep(
    M: int = 4096,
    K: int = 4096,
    N: int = 512,
    sparsity: float = 0.9,
    block_sizes: Optional[List[int]] = None,
) -> List[BSRBenchmarkResult]:
    """Test different block sizes."""
    if block_sizes is None:
        block_sizes = [8, 16, 32, 64]

    results = []
    for block_size in block_sizes:
        print(f"Testing block size {block_size}x{block_size}...", end=" ", flush=True)
        try:
            result = benchmark_bsr_vs_void(M, K, N, sparsity, block_size)
            results.append(result)

            if result.bsr_time_ms != float('inf'):
                print(f"VOID: {result.void_time_ms:.2f}ms, BSR: {result.bsr_time_ms:.2f}ms, "
                      f"Speedup: {result.void_vs_bsr_speedup:.2f}x")
            else:
                print(f"VOID: {result.void_time_ms:.2f}ms, CSR: {result.csr_time_ms:.2f}ms")

        except Exception as e:
            print(f"FAILED: {e}")

        torch.cuda.empty_cache()

    return results


def print_comparison_table(results: List[BSRBenchmarkResult], title: str = "VOID vs BSR"):
    """Print detailed comparison table."""
    print(f"\n{'=' * 100}")
    print(f"{title}")
    print(f"{'=' * 100}")

    header = (f"{'Config':<25} {'VOID (ms)':>12} {'BSR (ms)':>12} {'CSR (ms)':>12} "
              f"{'Dense (ms)':>12} {'vs BSR':>10} {'vs CSR':>10}")
    print(header)
    print("-" * 100)

    for r in results:
        M, K, N = r.matrix_size
        config = f"{M}x{K}, sp={r.sparsity:.0%}, blk={r.block_size}"

        bsr_str = f"{r.bsr_time_ms:.2f}" if r.bsr_time_ms != float('inf') else "N/A"
        speedup_bsr_str = f"{r.void_vs_bsr_speedup:.2f}x" if r.void_vs_bsr_speedup > 0 else "N/A"

        print(
            f"{config:<25} {r.void_time_ms:>12.2f} {bsr_str:>12} {r.csr_time_ms:>12.2f} "
            f"{r.dense_time_ms:>12.2f} {speedup_bsr_str:>10} {r.void_vs_csr_speedup:>9.2f}x"
        )

    print("-" * 100)

    # Statistics
    bsr_results = [r for r in results if r.void_vs_bsr_speedup > 0]
    if bsr_results:
        avg_bsr_speedup = np.mean([r.void_vs_bsr_speedup for r in bsr_results])
        max_bsr_speedup = max([r.void_vs_bsr_speedup for r in bsr_results])
        min_bsr_speedup = min([r.void_vs_bsr_speedup for r in bsr_results])

        print(f"\nVOID vs BSR Speedup Statistics:")
        print(f"  Average: {avg_bsr_speedup:.2f}x")
        print(f"  Best:    {max_bsr_speedup:.2f}x")
        print(f"  Worst:   {min_bsr_speedup:.2f}x")

        if avg_bsr_speedup > 1.5:
            print(f"\n[SUCCESS] VOID is {avg_bsr_speedup:.2f}x faster than cuSPARSE BSR on average!")
        elif avg_bsr_speedup > 1.0:
            print(f"\n[MARGINAL] VOID is {avg_bsr_speedup:.2f}x faster than BSR (needs more optimization)")
        else:
            print(f"\n[FAILED] VOID is slower than BSR by {1.0/avg_bsr_speedup:.2f}x (needs major work)")
    else:
        print("\n[WARNING] No BSR comparison available (PyTorch version too old?)")

    avg_csr_speedup = np.mean([r.void_vs_csr_speedup for r in results])
    print(f"\nVOID vs CSR: {avg_csr_speedup:.2f}x average speedup")


def main():
    print("=" * 100)
    print("CRITICAL BENCHMARK: VOID vs cuSPARSE BSR")
    print("=" * 100)
    print("\nThis benchmark compares VOID against its actual competitor: Block Sparse Row (BSR) format.")
    print("If VOID doesn't beat BSR by >1.5x, it's not providing real value.\n")

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Check PyTorch version for BSR support
    pytorch_version = torch.__version__
    print(f"PyTorch version: {pytorch_version}")

    if not hasattr(torch, 'sparse_bsr_tensor'):
        print("\n[WARNING] PyTorch BSR not available in this version.")
        print("BSR format was added in PyTorch 2.0. You're running an older version.")
        print("Will still run benchmarks but only compare against CSR.\n")

    # Check GPU memory and adjust sizes
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB\n")

    # Adjust sizes based on GPU memory
    if gpu_mem_gb < 8:
        M, K, N = 2048, 2048, 256
        print("Using reduced sizes for low-memory GPU")
    elif gpu_mem_gb < 16:
        M, K, N = 4096, 4096, 512
        print("Using medium sizes")
    else:
        M, K, N = 8192, 8192, 512
        print("Using large sizes")

    # =========================================================================
    # Test 1: Sparsity Sweep
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"Test 1: Sparsity Sweep (matrix size {M}x{K}, block 32x32)")
    print("=" * 100)

    sparsity_results = run_sparsity_sweep_bsr(M=M, K=K, N=N, block_size=32)
    if sparsity_results:
        print_comparison_table(sparsity_results, "Sparsity Sweep: VOID vs BSR")

    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: Block Size Sweep
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"Test 2: Block Size Sweep (matrix {M}x{K}, 90% sparsity)")
    print("=" * 100)

    block_size_results = run_block_size_sweep(M=M, K=K, N=N, sparsity=0.9)
    if block_size_results:
        print_comparison_table(block_size_results, "Block Size Sweep: VOID vs BSR")

    # =========================================================================
    # Final Verdict
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("FINAL VERDICT")
    print("=" * 100)

    all_results = sparsity_results + block_size_results
    bsr_results = [r for r in all_results if r.void_vs_bsr_speedup > 0]

    if bsr_results:
        avg_speedup = np.mean([r.void_vs_bsr_speedup for r in bsr_results])

        if avg_speedup >= 2.0:
            print(f"[EXCELLENT] VOID is {avg_speedup:.2f}x faster than BSR!")
            print("This is a significant speedup that justifies VOID's existence.")
        elif avg_speedup >= 1.5:
            print(f"[GOOD] VOID is {avg_speedup:.2f}x faster than BSR.")
            print("This is a meaningful speedup for production use.")
        elif avg_speedup >= 1.2:
            print(f"[MARGINAL] VOID is {avg_speedup:.2f}x faster than BSR.")
            print("Speedup exists but may not justify switching from BSR.")
        elif avg_speedup >= 1.0:
            print(f"[WEAK] VOID is only {avg_speedup:.2f}x faster than BSR.")
            print("Not enough speedup to be worth the complexity.")
        else:
            print(f"[FAILED] VOID is SLOWER than BSR by {1.0/avg_speedup:.2f}x!")
            print("VOID needs major optimizations before it's useful.")
    else:
        print("[INCONCLUSIVE] Could not compare against BSR (not available in PyTorch).")
        print("Upgrade to PyTorch >= 2.0 for BSR support.")

    print("\nDone!")


if __name__ == "__main__":
    main()
