"""
Fusion Operations Benchmark

Compare fused vs unfused sparse operations to measure kernel fusion benefits.
"""

import torch
import time
import numpy as np
from scipy import sparse

import sys
sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import csr_to_void, void_spmm
from void.fusion import (
    void_spmm_gelu,
    void_spmm_relu,
    fused_sparse_mlp,
    FusedSparseLinear,
)


def benchmark_fused_activation(
    M: int = 2048,
    K: int = 2048,
    N: int = 512,
    sparsity: float = 0.9,
    n_iterations: int = 100,
    warmup: int = 20,
):
    """Benchmark fused vs unfused SpMM + activation."""
    device = torch.device("cuda")

    # Create sparse matrix
    density = 1.0 - sparsity
    sparse_csr = sparse.random(M, K, density=density, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(sparse_csr, tile_size=32).cuda()

    B = torch.randn(K, N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    results = {}

    # Warmup
    for _ in range(warmup):
        _ = void_spmm_gelu(void_tensor, B)
        _ = void_spmm(void_tensor, B)
    torch.cuda.synchronize()

    # =========================================================================
    # Fused SpMM + GELU
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm_gelu(void_tensor, B)
    torch.cuda.synchronize()
    fused_gelu_time = (time.perf_counter() - start) / n_iterations * 1000
    results["fused_gelu"] = fused_gelu_time

    # =========================================================================
    # Unfused SpMM + GELU
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = void_spmm(void_tensor, B)
        out = torch.nn.functional.gelu(out)
    torch.cuda.synchronize()
    unfused_gelu_time = (time.perf_counter() - start) / n_iterations * 1000
    results["unfused_gelu"] = unfused_gelu_time

    # =========================================================================
    # Fused SpMM + GELU + bias
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm_gelu(void_tensor, B, bias=bias)
    torch.cuda.synchronize()
    fused_gelu_bias_time = (time.perf_counter() - start) / n_iterations * 1000
    results["fused_gelu_bias"] = fused_gelu_bias_time

    # =========================================================================
    # Unfused SpMM + bias + GELU
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = void_spmm(void_tensor, B)
        out = out + bias
        out = torch.nn.functional.gelu(out)
    torch.cuda.synchronize()
    unfused_gelu_bias_time = (time.perf_counter() - start) / n_iterations * 1000
    results["unfused_gelu_bias"] = unfused_gelu_bias_time

    # =========================================================================
    # Fused SpMM + ReLU
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm_relu(void_tensor, B)
    torch.cuda.synchronize()
    fused_relu_time = (time.perf_counter() - start) / n_iterations * 1000
    results["fused_relu"] = fused_relu_time

    # =========================================================================
    # Unfused SpMM + ReLU
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = void_spmm(void_tensor, B)
        out = torch.relu(out)
    torch.cuda.synchronize()
    unfused_relu_time = (time.perf_counter() - start) / n_iterations * 1000
    results["unfused_relu"] = unfused_relu_time

    return results


def benchmark_fused_mlp(
    batch: int = 64,
    in_features: int = 1024,
    hidden_features: int = 4096,
    out_features: int = 1024,
    sparsity: float = 0.9,
    n_iterations: int = 100,
    warmup: int = 20,
):
    """Benchmark fused vs unfused sparse MLP."""
    device = torch.device("cuda")

    # Create sparse weight matrices
    density = 1.0 - sparsity
    W1_sparse = sparse.random(hidden_features, in_features, density=density, format='csr', dtype=np.float32)
    W2_sparse = sparse.random(out_features, hidden_features, density=density, format='csr', dtype=np.float32)

    W1 = csr_to_void(W1_sparse, tile_size=32).cuda()
    W2 = csr_to_void(W2_sparse, tile_size=32).cuda()

    bias1 = torch.randn(hidden_features, device=device)
    bias2 = torch.randn(out_features, device=device)

    x = torch.randn(batch, in_features, device=device)

    results = {}

    # Warmup
    for _ in range(warmup):
        _ = fused_sparse_mlp(x, W1, W2, activation="gelu")
    torch.cuda.synchronize()

    # =========================================================================
    # Fused MLP (2 kernel launches)
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fused_sparse_mlp(x, W1, W2, activation="gelu")
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iterations * 1000
    results["fused_mlp"] = fused_time

    # =========================================================================
    # Fused MLP with biases
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fused_sparse_mlp(x, W1, W2, activation="gelu", bias1=bias1, bias2=bias2)
    torch.cuda.synchronize()
    fused_bias_time = (time.perf_counter() - start) / n_iterations * 1000
    results["fused_mlp_bias"] = fused_bias_time

    # =========================================================================
    # Unfused MLP (6 operations)
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        hidden = void_spmm(W1, x.T)  # [hidden, batch]
        hidden = hidden + bias1[:, None]
        hidden = torch.nn.functional.gelu(hidden)
        out = void_spmm(W2, hidden)  # [out, batch]
        out = out + bias2[:, None]
        out = out.T
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / n_iterations * 1000
    results["unfused_mlp"] = unfused_time

    return results


def benchmark_fused_linear(
    batch: int = 64,
    in_features: int = 2048,
    out_features: int = 2048,
    sparsity: float = 0.9,
    n_iterations: int = 100,
    warmup: int = 20,
):
    """Benchmark FusedSparseLinear module."""
    device = torch.device("cuda")

    # Create sparse weight matrix
    density = 1.0 - sparsity
    W_sparse = sparse.random(out_features, in_features, density=density, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(W_sparse, tile_size=32).cuda()

    # Create fused linear layer
    layer = FusedSparseLinear(
        in_features=in_features,
        out_features=out_features,
        void_tensor=void_tensor,
        activation="gelu",
        bias=True,
    ).to(device)

    x = torch.randn(batch, in_features, device=device)

    results = {}

    # Warmup
    for _ in range(warmup):
        _ = layer(x)
    torch.cuda.synchronize()

    # =========================================================================
    # Fused Linear (1 kernel)
    # =========================================================================
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = layer(x)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iterations * 1000
    results["fused_linear"] = fused_time

    # =========================================================================
    # Unfused Linear (3 operations)
    # =========================================================================
    bias = layer.bias
    start = time.perf_counter()
    for _ in range(n_iterations):
        y = void_spmm(void_tensor, x.T)  # [out, batch]
        y = y + bias[:, None]
        y = torch.nn.functional.gelu(y)
        y = y.T
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / n_iterations * 1000
    results["unfused_linear"] = unfused_time

    return results


def print_results(results: dict, title: str):
    """Print benchmark results."""
    print(f"\n{title}")
    print("-" * 60)

    for name, time_ms in results.items():
        print(f"  {name:<25}: {time_ms:.3f} ms")

    # Calculate speedups
    if "fused_gelu" in results and "unfused_gelu" in results:
        speedup = results["unfused_gelu"] / results["fused_gelu"]
        print(f"\n  GELU Fusion Speedup: {speedup:.2f}x")

    if "fused_gelu_bias" in results and "unfused_gelu_bias" in results:
        speedup = results["unfused_gelu_bias"] / results["fused_gelu_bias"]
        print(f"  GELU+Bias Fusion Speedup: {speedup:.2f}x")

    if "fused_relu" in results and "unfused_relu" in results:
        speedup = results["unfused_relu"] / results["fused_relu"]
        print(f"  ReLU Fusion Speedup: {speedup:.2f}x")

    if "fused_mlp" in results and "unfused_mlp" in results:
        speedup = results["unfused_mlp"] / results["fused_mlp_bias"]
        print(f"\n  MLP Fusion Speedup: {speedup:.2f}x")

    if "fused_linear" in results and "unfused_linear" in results:
        speedup = results["unfused_linear"] / results["fused_linear"]
        print(f"\n  Linear Fusion Speedup: {speedup:.2f}x")


def main():
    print("=" * 60)
    print("VOID Fusion Operations Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB")

    # Adjust sizes based on GPU memory
    if gpu_mem_gb < 8:
        M, K, N = 1024, 1024, 256
        batch, hidden = 32, 2048
    else:
        M, K, N = 2048, 2048, 512
        batch, hidden = 64, 4096

    # =========================================================================
    # Test 1: Fused Activation
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Test 1: Fused SpMM + Activation ({M}x{K} @ {K}x{N})")
    print("=" * 60)

    results1 = benchmark_fused_activation(M=M, K=K, N=N, sparsity=0.9)
    print_results(results1, "Activation Fusion Results (90% sparsity)")

    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: Fused MLP
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Test 2: Fused Sparse MLP (batch={batch}, {1024}->{hidden}->{1024})")
    print("=" * 60)

    results2 = benchmark_fused_mlp(
        batch=batch,
        in_features=1024,
        hidden_features=hidden,
        out_features=1024,
        sparsity=0.9
    )
    print_results(results2, "MLP Fusion Results (90% sparsity)")

    torch.cuda.empty_cache()

    # =========================================================================
    # Test 3: Fused Linear Layer
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Test 3: FusedSparseLinear ({batch}x{M} @ {M}x{M})")
    print("=" * 60)

    results3 = benchmark_fused_linear(
        batch=batch,
        in_features=M,
        out_features=M,
        sparsity=0.9
    )
    print_results(results3, "Linear Fusion Results (90% sparsity)")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print("\nKernel Fusion Benefits:")
    print("  - Eliminates memory round-trips between operations")
    print("  - Single kernel launch instead of multiple")
    print("  - Better GPU occupancy and cache utilization")

    # Overall speedups
    speedups = []
    if "fused_gelu" in results1 and "unfused_gelu" in results1:
        speedups.append(results1["unfused_gelu"] / results1["fused_gelu"])
    if "fused_mlp_bias" in results2 and "unfused_mlp" in results2:
        speedups.append(results2["unfused_mlp"] / results2["fused_mlp_bias"])
    if "fused_linear" in results3 and "unfused_linear" in results3:
        speedups.append(results3["unfused_linear"] / results3["fused_linear"])

    if speedups:
        avg_speedup = np.mean(speedups)
        print(f"\nAverage Fusion Speedup: {avg_speedup:.2f}x")

        if avg_speedup >= 1.5:
            print("[SUCCESS] Kernel fusion provides significant speedup!")
        elif avg_speedup >= 1.2:
            print("[GOOD] Kernel fusion provides meaningful speedup.")
        else:
            print("[MARGINAL] Kernel fusion shows limited benefit.")


if __name__ == "__main__":
    main()
