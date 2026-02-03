"""
Sparse Attention Benchmark

Compare VOID sparse attention against dense attention.
"""

import torch
import time
import math

import sys
sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import (
    local_attention,
    block_sparse_attention,
    create_local_attention_mask,
    create_block_sparse_mask,
    sparse_attention,
)


def dense_attention(q, k, v, scale=None):
    """Reference dense attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return attn @ v


def benchmark_attention(
    batch: int = 4,
    n_heads: int = 8,
    seq_len: int = 2048,
    head_dim: int = 64,
    n_iterations: int = 50,
    warmup: int = 10,
):
    """Benchmark sparse vs dense attention."""
    device = torch.device("cuda")

    print(f"Config: batch={batch}, heads={n_heads}, seq={seq_len}, dim={head_dim}")
    print("-" * 60)

    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

    results = {}

    # Dense attention
    print("Dense attention...", end=" ", flush=True)
    for _ in range(warmup):
        _ = dense_attention(q, k, v)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = dense_attention(q, k, v)
    torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / n_iterations * 1000
    results["dense"] = dense_time
    print(f"{dense_time:.2f}ms")

    # Local attention (window=256)
    print("Local attention (window=256)...", end=" ", flush=True)
    for _ in range(warmup):
        _ = local_attention(q, k, v, window_size=256, block_size=64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = local_attention(q, k, v, window_size=256, block_size=64)
    torch.cuda.synchronize()
    local_time = (time.perf_counter() - start) / n_iterations * 1000
    results["local_256"] = local_time
    print(f"{local_time:.2f}ms (speedup: {dense_time/local_time:.2f}x)")

    # Local attention (window=512)
    print("Local attention (window=512)...", end=" ", flush=True)
    for _ in range(warmup):
        _ = local_attention(q, k, v, window_size=512, block_size=64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = local_attention(q, k, v, window_size=512, block_size=64)
    torch.cuda.synchronize()
    local512_time = (time.perf_counter() - start) / n_iterations * 1000
    results["local_512"] = local512_time
    print(f"{local512_time:.2f}ms (speedup: {dense_time/local512_time:.2f}x)")

    # Block sparse (90% sparse)
    print("Block-sparse (90% sparse)...", end=" ", flush=True)
    mask = create_block_sparse_mask(seq_len, block_size=64, sparsity=0.9, device='cuda')

    for _ in range(warmup):
        _ = sparse_attention(q, k, v, mask)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = sparse_attention(q, k, v, mask)
    torch.cuda.synchronize()
    sparse_time = (time.perf_counter() - start) / n_iterations * 1000
    results["block_sparse_90"] = sparse_time
    print(f"{sparse_time:.2f}ms (speedup: {dense_time/sparse_time:.2f}x)")

    return results


def main():
    print("=" * 60)
    print("VOID Sparse Attention Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Check GPU memory and adjust sizes
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB\n")

    # Adjust sequence lengths based on GPU memory
    if gpu_mem_gb < 8:
        seq_lengths = [512, 1024]
        batch = 2
        n_heads = 4
    elif gpu_mem_gb < 16:
        seq_lengths = [1024, 2048]
        batch = 2
        n_heads = 8
    else:
        seq_lengths = [1024, 2048, 4096]
        batch = 4
        n_heads = 8

    # Different sequence lengths
    for seq_len in seq_lengths:
        try:
            print(f"\n{'=' * 60}")
            benchmark_attention(batch=batch, n_heads=n_heads, seq_len=seq_len)
            # Clean up GPU memory between benchmarks
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed for seq_len={seq_len}: {e}")
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Key insight: Sparse attention scales better with sequence length")
    print("Dense: O(n^2), Local: O(n*w), Block-sparse: O(n*k)")


if __name__ == "__main__":
    main()
