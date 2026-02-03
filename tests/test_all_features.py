"""
Quick test for all VOID features.
"""

import pytest
import torch
import scipy.sparse as sp
import numpy as np

import sys
sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import (
    # Core
    csr_to_void, void_spmm, void_spmv,
    # Attention
    sparse_attention, local_attention, create_local_attention_mask,
    # Autograd
    VOIDSpMM, SparseLinear,
    # Stream-K
    void_spmm_stream_k, analyze_workload_balance,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def test_basic_spmm():
    """Test basic SpMM."""
    print("Testing basic SpMM...", end=" ")

    M, K, N = 512, 512, 128
    sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(sparse_np, tile_size=32).cuda()

    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = void_spmm(void_tensor, B)

    # Reference
    A_dense = torch.tensor(sparse_np.toarray(), device='cuda')
    C_ref = A_dense @ B

    error = (C - C_ref).abs().max().item() / C_ref.abs().max().item()
    assert error < 1e-3, f"SpMM error too high: {error}"
    print(f"PASS (error={error:.2e})")


def test_spmv():
    """Test SpMV."""
    print("Testing SpMV...", end=" ")

    M, K = 512, 512
    sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(sparse_np, tile_size=32).cuda()

    x = torch.randn(K, device='cuda', dtype=torch.float32)
    y = void_spmv(void_tensor, x)

    A_dense = torch.tensor(sparse_np.toarray(), device='cuda')
    y_ref = A_dense @ x

    error = (y - y_ref).abs().max().item() / y_ref.abs().max().item()
    assert error < 1e-3, f"SpMV error too high: {error}"
    print(f"PASS (error={error:.2e})")


def test_sparse_attention():
    """Test sparse attention."""
    print("Testing sparse attention...", end=" ")

    batch, n_heads, seq_len, head_dim = 2, 4, 256, 64
    q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')

    # Local attention with window=64
    out = local_attention(q, k, v, window_size=64, block_size=32)

    assert out.shape == q.shape, f"Shape mismatch: {out.shape} vs {q.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print("PASS")


def test_autograd():
    """Test autograd support."""
    print("Testing autograd...", end=" ")

    M, K, N = 256, 256, 64
    sparse_np = sp.random(M, K, density=0.2, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(sparse_np, tile_size=32).cuda()

    # Create module
    module = VOIDSpMM(void_tensor, requires_grad=True)

    x = torch.randn(K, N, device='cuda', requires_grad=True)
    y = module(x)

    # Backward
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "No gradient for x"
    assert module.values.grad is not None, "No gradient for sparse values"
    print("PASS")


def test_sparse_linear():
    """Test SparseLinear layer."""
    print("Testing SparseLinear...", end=" ")

    in_features, out_features = 256, 128
    batch_size = 32

    sparse_np = sp.random(out_features, in_features, density=0.2, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(sparse_np, tile_size=32).cuda()

    layer = SparseLinear(in_features, out_features, void_tensor, bias=True).cuda()

    x = torch.randn(batch_size, in_features, device='cuda')
    y = layer(x)

    assert y.shape == (batch_size, out_features), f"Shape mismatch: {y.shape}"

    # Test backward
    y.sum().backward()
    print("PASS")


def test_stream_k():
    """Test Stream-K load balancing."""
    print("Testing Stream-K...", end=" ")

    M, K, N = 512, 512, 128
    sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)
    void_tensor = csr_to_void(sparse_np, tile_size=32).cuda()

    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    # Analyze workload
    balance = analyze_workload_balance(void_tensor)
    assert "imbalance_ratio" in balance, "Balance analysis failed"

    # Run Stream-K
    C = void_spmm_stream_k(void_tensor, B)

    # Reference
    A_dense = torch.tensor(sparse_np.toarray(), device='cuda')
    C_ref = A_dense @ B

    error = (C - C_ref).abs().max().item() / C_ref.abs().max().item()
    assert error < 1e-3, f"Stream-K error too high: {error}"
    print(f"PASS (error={error:.2e})")


def main():
    print("=" * 50)
    print("VOID Feature Tests")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}\n")

    test_basic_spmm()
    test_spmv()
    test_sparse_attention()
    test_autograd()
    test_sparse_linear()
    test_stream_k()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
