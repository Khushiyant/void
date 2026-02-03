"""
Tests for FP8 Support in VOID

Tests FP8 quantization, dequantization, and SpMM correctness.
"""

import pytest
import torch
import numpy as np
from scipy import sparse

# Skip all tests if CUDA not available
pytestmark = pytest.mark.cuda


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def device(cuda_available):
    """Get the device to use."""
    return 'cuda' if cuda_available else 'cpu'


@pytest.fixture
def sparse_matrix_small():
    """Create a small sparse matrix for testing."""
    np.random.seed(42)
    density = 0.1
    M, N = 256, 256
    matrix = sparse.random(M, N, density=density, format='csr', dtype=np.float32)
    return matrix


@pytest.fixture
def sparse_matrix_medium():
    """Create a medium sparse matrix for testing."""
    np.random.seed(42)
    density = 0.05
    M, N = 512, 512
    matrix = sparse.random(M, N, density=density, format='csr', dtype=np.float32)
    return matrix


class TestFP8Config:
    """Tests for FP8 configuration."""

    def test_fp8_format_e4m3(self):
        """Test E4M3 format configuration."""
        from void.fp8 import FP8Config, FP8Format

        config = FP8Config(format=FP8Format.E4M3)
        assert config.max_value == 448.0
        if hasattr(torch, 'float8_e4m3fn'):
            assert config.torch_dtype == torch.float8_e4m3fn

    def test_fp8_format_e5m2(self):
        """Test E5M2 format configuration."""
        from void.fp8 import FP8Config, FP8Format

        config = FP8Config(format=FP8Format.E5M2)
        assert config.max_value == 57344.0
        if hasattr(torch, 'float8_e5m2'):
            assert config.torch_dtype == torch.float8_e5m2

    def test_scale_modes(self):
        """Test different scaling modes."""
        from void.fp8 import FP8Config

        for mode in ["per_tensor", "per_block", "per_row"]:
            config = FP8Config(scale_mode=mode)
            assert config.scale_mode == mode


class TestFP8Quantization:
    """Tests for FP8 quantization/dequantization."""

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_quantize_dequantize_roundtrip(self, device):
        """Test that quantize->dequantize preserves values approximately."""
        from void.fp8 import quantize_to_fp8, dequantize_from_fp8, FP8Config

        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        # Create test tensor
        tensor = torch.randn(64, 64, device=device) * 10

        # Quantize
        config = FP8Config()
        fp8_tensor, scaling_info = quantize_to_fp8(tensor, config)

        # Dequantize
        recovered = dequantize_from_fp8(fp8_tensor, scaling_info, output_dtype=torch.float32)

        # Check approximate equality (FP8 has limited precision)
        # E4M3 has ~3 bits of mantissa, so expect ~12.5% relative error
        relative_error = (tensor - recovered).abs() / (tensor.abs() + 1e-6)
        mean_error = relative_error.mean().item()

        assert mean_error < 0.2, f"Mean relative error {mean_error:.3f} too high"

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_per_block_scaling(self, device):
        """Test per-block scaling mode."""
        from void.fp8 import quantize_to_fp8, dequantize_from_fp8, FP8Config

        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        # Create tensor with varying magnitudes per block
        tensor = torch.randn(4, 64, 64, device=device)
        tensor[0] *= 0.01  # Very small
        tensor[1] *= 1.0   # Normal
        tensor[2] *= 100   # Large
        tensor[3] *= 1000  # Very large

        config = FP8Config(scale_mode="per_block")
        fp8_tensor, scaling_info = quantize_to_fp8(tensor, config)

        # Should have separate scales
        assert scaling_info.scale.shape == (4,)

        recovered = dequantize_from_fp8(fp8_tensor, scaling_info, output_dtype=torch.float32)

        # Per-block scaling should handle varying magnitudes better
        relative_error = (tensor - recovered).abs() / (tensor.abs() + 1e-6)
        assert relative_error.mean() < 0.3

    def test_compute_amax(self, device):
        """Test AMAX computation."""
        from void.fp8 import compute_amax

        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ], device=device)

        # Per-tensor
        amax = compute_amax(tensor, "per_tensor")
        assert amax.item() == 8.0

        # Per-block (first dim)
        amax = compute_amax(tensor, "per_block")
        assert amax.shape == (2,)
        assert amax[0].item() == 4.0
        assert amax[1].item() == 8.0


class TestFP8VOIDTensor:
    """Tests for FP8VOIDTensor."""

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_void_tensor_to_fp8(self, sparse_matrix_small, device):
        """Test conversion from VOIDTensor to FP8."""
        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        from void import csr_to_void
        from void.fp8 import FP8VOIDTensor

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        fp8_tensor = FP8VOIDTensor.from_void_tensor(void_tensor)

        assert fp8_tensor.n_blocks == void_tensor.n_blocks
        assert fp8_tensor.shape == void_tensor.shape
        assert fp8_tensor.values_fp8.dtype == torch.float8_e4m3fn

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_fp8_to_void_tensor(self, sparse_matrix_small, device):
        """Test conversion back from FP8 to VOIDTensor."""
        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        from void import csr_to_void
        from void.fp8 import FP8VOIDTensor

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        fp8_tensor = FP8VOIDTensor.from_void_tensor(void_tensor)
        recovered = fp8_tensor.to_void_tensor(output_dtype=torch.float32)

        # Check structure preserved
        assert recovered.n_blocks == void_tensor.n_blocks
        assert recovered.shape == void_tensor.shape

        # Check values approximately equal
        error = (void_tensor.values - recovered.values).abs().mean()
        assert error < void_tensor.values.abs().mean() * 0.3

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_void_tensor_to_fp8_method(self, sparse_matrix_small, device):
        """Test VOIDTensor.to_fp8() method."""
        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        from void import csr_to_void

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        fp8_tensor = void_tensor.to_fp8(format="e4m3", scale_mode="per_block")

        assert fp8_tensor.n_blocks == void_tensor.n_blocks


class TestFP8SpMM:
    """Tests for FP8 SpMM kernel."""

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_fp8_spmm_correctness(self, sparse_matrix_small, device):
        """Test FP8 SpMM produces correct results."""
        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        from void import csr_to_void, void_spmm
        from void.fp8 import FP8VOIDTensor, void_spmm_fp8

        # Convert to VOID
        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        # Create dense B
        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        # Reference: FP32 SpMM
        C_ref = void_spmm(void_tensor, B)

        # FP8 SpMM
        fp8_tensor = FP8VOIDTensor.from_void_tensor(void_tensor)
        C_fp8 = void_spmm_fp8(fp8_tensor, B, output_dtype=torch.float32)

        # Compare (expect some error due to FP8 precision)
        relative_error = (C_ref - C_fp8).abs() / (C_ref.abs() + 1e-6)
        mean_error = relative_error.mean().item()

        # FP8 should be within 20% of FP32 for typical cases
        assert mean_error < 0.3, f"FP8 SpMM error {mean_error:.3f} too high"

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_fp8_spmm_output_dtype(self, sparse_matrix_small, device):
        """Test FP8 SpMM with different output dtypes."""
        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        from void import csr_to_void
        from void.fp8 import FP8VOIDTensor, void_spmm_fp8

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        fp8_tensor = FP8VOIDTensor.from_void_tensor(void_tensor)

        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        # Test FP32 output
        C_fp32 = void_spmm_fp8(fp8_tensor, B, output_dtype=torch.float32)
        assert C_fp32.dtype == torch.float32

        # Test FP16 output
        C_fp16 = void_spmm_fp8(fp8_tensor, B, output_dtype=torch.float16)
        assert C_fp16.dtype == torch.float16


class TestFP8MemoryReduction:
    """Tests for FP8 memory benefits."""

    @pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason="FP8 not supported")
    def test_fp8_memory_footprint(self, sparse_matrix_medium, device):
        """Test that FP8 reduces memory footprint."""
        if device == 'cpu':
            pytest.skip("FP8 requires CUDA")

        from void import csr_to_void
        from void.fp8 import FP8VOIDTensor

        void_tensor = csr_to_void(sparse_matrix_medium, device=device)

        # FP32 memory
        fp32_bytes = void_tensor.values.numel() * 4

        # FP8 memory
        fp8_tensor = FP8VOIDTensor.from_void_tensor(void_tensor)
        fp8_bytes = fp8_tensor.memory_bytes

        # FP8 should use significantly less memory
        # Values: 1 byte vs 4 bytes, plus some overhead for scales
        assert fp8_bytes < fp32_bytes * 0.5, "FP8 should reduce memory by >50%"
