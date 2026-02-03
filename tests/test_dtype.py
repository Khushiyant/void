"""
Tests for FP16/BF16 dtype support in VOID.
"""

import pytest
import torch
import numpy as np
import scipy.sparse as sp

import sys
sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import (
    csr_to_void, void_spmm, void_spmv,
    sparse_attention, local_attention, create_local_attention_mask,
    void_spmm_stream_k,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestVOIDTensorDtype:
    """Tests for VOIDTensor dtype handling."""

    def test_csr_to_void_fp32(self):
        """Test conversion with FP32 dtype."""
        sparse_np = sp.random(256, 256, density=0.1, format='csr', dtype=np.float32)
        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.float32)

        assert void_tensor.dtype == torch.float32
        assert void_tensor.values.dtype == torch.float32

    def test_csr_to_void_fp16(self):
        """Test conversion with FP16 dtype."""
        sparse_np = sp.random(256, 256, density=0.1, format='csr', dtype=np.float32)
        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.float16)

        assert void_tensor.dtype == torch.float16
        assert void_tensor.values.dtype == torch.float16

    def test_csr_to_void_bf16(self):
        """Test conversion with BF16 dtype."""
        sparse_np = sp.random(256, 256, density=0.1, format='csr', dtype=np.float32)
        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.bfloat16)

        assert void_tensor.dtype == torch.bfloat16
        assert void_tensor.values.dtype == torch.bfloat16

    def test_to_dtype_conversion(self):
        """Test dtype conversion methods."""
        sparse_np = sp.random(256, 256, density=0.1, format='csr', dtype=np.float32)
        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.float32).cuda()

        # Test half()
        void_fp16 = void_tensor.half()
        assert void_fp16.dtype == torch.float16
        assert void_fp16.values.dtype == torch.float16

        # Test bfloat16()
        void_bf16 = void_tensor.bfloat16()
        assert void_bf16.dtype == torch.bfloat16
        assert void_bf16.values.dtype == torch.bfloat16

        # Test float()
        void_fp32 = void_fp16.float()
        assert void_fp32.dtype == torch.float32
        assert void_fp32.values.dtype == torch.float32

    def test_to_preserves_dtype(self):
        """Test that to() preserves dtype when moving devices."""
        sparse_np = sp.random(256, 256, density=0.1, format='csr', dtype=np.float32)
        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.float16)

        void_cuda = void_tensor.cuda()
        assert void_cuda.dtype == torch.float16

        void_cpu = void_cuda.cpu()
        assert void_cpu.dtype == torch.float16


class TestSpMMDtype:
    """Tests for SpMM with different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_spmm_dtype_correctness(self, dtype):
        """Test SpMM correctness with various dtypes."""
        M, K, N = 256, 256, 64
        sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)

        # Create VOID tensor with target dtype
        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=dtype).cuda()

        # Dense matrix with matching dtype
        B = torch.randn(K, N, device='cuda', dtype=dtype)

        # SpMM
        C = void_spmm(void_tensor, B)

        # Reference - compute in target dtype to match VOID's precision
        # (VOID loads values in target dtype, converts to FP32 for accumulation, then converts back)
        A_dense = torch.tensor(sparse_np.toarray(), device='cuda', dtype=dtype)
        C_ref = A_dense @ B

        # Check output dtype
        assert C.dtype == dtype

        # Check correctness with appropriate tolerance
        # FP32: Tensor cores in Triton may have different rounding than PyTorch cuBLAS
        # FP16/BF16: Additional precision loss from lower precision formats
        if dtype == torch.float32:
            rtol, atol = 1e-2, 5e-3
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-3
        else:  # bfloat16
            rtol, atol = 2e-2, 2e-3

        torch.testing.assert_close(C, C_ref, rtol=rtol, atol=atol)

    def test_spmm_fp16_numerical_stability(self):
        """Test that FP16 SpMM uses FP32 accumulation for stability."""
        M, K, N = 512, 512, 128
        # Create sparse matrix with values that would overflow in FP16
        sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)
        sparse_np.data *= 100  # Scale up values

        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.float16).cuda()
        B = torch.randn(K, N, device='cuda', dtype=torch.float16) * 10

        C = void_spmm(void_tensor, B)

        # Check no NaN/Inf
        assert not torch.isnan(C).any(), "SpMM produced NaN values"
        assert not torch.isinf(C).any(), "SpMM produced Inf values"


class TestSpMVDtype:
    """Tests for SpMV with different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_spmv_dtype_correctness(self, dtype):
        """Test SpMV correctness with various dtypes."""
        M, K = 256, 256
        sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)

        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=dtype).cuda()
        x = torch.randn(K, device='cuda', dtype=dtype)

        y = void_spmv(void_tensor, x)

        # Reference - compute in target dtype to match VOID's precision
        A_dense = torch.tensor(sparse_np.toarray(), device='cuda', dtype=dtype)
        y_ref = A_dense @ x

        assert y.dtype == dtype

        if dtype == torch.float32:
            rtol, atol = 1e-2, 5e-3
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-3
        else:  # bfloat16
            rtol, atol = 2e-2, 2e-3

        torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)


class TestStreamKDtype:
    """Tests for Stream-K SpMM with different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_stream_k_dtype_correctness(self, dtype):
        """Test Stream-K SpMM correctness with various dtypes."""
        M, K, N = 256, 256, 64
        sparse_np = sp.random(M, K, density=0.1, format='csr', dtype=np.float32)

        void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=dtype).cuda()
        B = torch.randn(K, N, device='cuda', dtype=dtype)

        C = void_spmm_stream_k(void_tensor, B)

        # Reference - compute in target dtype to match VOID's precision
        A_dense = torch.tensor(sparse_np.toarray(), device='cuda', dtype=dtype)
        C_ref = A_dense @ B

        assert C.dtype == dtype

        # Stream-K uses atomic operations which can introduce additional precision loss
        if dtype == torch.float32:
            rtol, atol = 1e-2, 5e-3
        else:  # float16
            rtol, atol = 3e-2, 5e-3

        torch.testing.assert_close(C, C_ref, rtol=rtol, atol=atol)


class TestAttentionDtype:
    """Tests for sparse attention with different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_local_attention_dtype(self, dtype):
        """Test local attention with various dtypes."""
        batch, n_heads, seq_len, head_dim = 2, 4, 128, 64

        q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        v = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=dtype)

        out = local_attention(q, k, v, window_size=32, block_size=32)

        assert out.dtype == dtype
        assert out.shape == q.shape
        assert not torch.isnan(out).any(), "Attention produced NaN values"

    def test_attention_fp16_stability(self):
        """Test FP16 attention stability with longer sequences."""
        batch, n_heads, seq_len, head_dim = 1, 2, 256, 64

        q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

        out = local_attention(q, k, v, window_size=64, block_size=32)

        assert not torch.isnan(out).any(), "FP16 attention produced NaN"
        assert not torch.isinf(out).any(), "FP16 attention produced Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
