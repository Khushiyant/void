"""
Tests for Operation Fusion in VOID

Tests fused SpMM + activation kernels and sparse MLP.
"""

import pytest
import torch
import numpy as np
from scipy import sparse

pytestmark = pytest.mark.cuda


@pytest.fixture
def device():
    """Get device for tests."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sparse_matrix_small():
    """Create a small sparse matrix."""
    np.random.seed(42)
    return sparse.random(256, 256, density=0.1, format='csr', dtype=np.float32)


@pytest.fixture
def sparse_matrix_medium():
    """Create a medium sparse matrix."""
    np.random.seed(42)
    return sparse.random(512, 512, density=0.05, format='csr', dtype=np.float32)


class TestFusedSpMMActivation:
    """Tests for fused SpMM + activation kernels."""

    def test_spmm_gelu(self, sparse_matrix_small, device):
        """Test fused SpMM + GELU."""
        from void import csr_to_void, void_spmm
        from void.fusion import void_spmm_gelu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device)

        # Fused kernel
        C_fused = void_spmm_gelu(void_tensor, B)

        # Reference: separate ops
        C_ref = void_spmm(void_tensor, B)
        C_ref = torch.nn.functional.gelu(C_ref)

        # GELU approximation has some tolerance
        error = (C_fused - C_ref).abs().max().item()
        assert error < 0.05, f"Fused GELU error: {error}"

    def test_spmm_relu(self, sparse_matrix_small, device):
        """Test fused SpMM + ReLU."""
        from void import csr_to_void, void_spmm
        from void.fusion import void_spmm_relu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device)

        # Fused kernel
        C_fused = void_spmm_relu(void_tensor, B)

        # Reference
        C_ref = void_spmm(void_tensor, B)
        C_ref = torch.relu(C_ref)

        error = (C_fused - C_ref).abs().max().item()
        assert error < 1e-5, f"Fused ReLU error: {error}"

    def test_spmm_silu(self, sparse_matrix_small, device):
        """Test fused SpMM + SiLU."""
        from void import csr_to_void, void_spmm
        from void.fusion import void_spmm_silu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device)

        # Fused kernel
        C_fused = void_spmm_silu(void_tensor, B)

        # Reference
        C_ref = void_spmm(void_tensor, B)
        C_ref = torch.nn.functional.silu(C_ref)

        # SiLU uses sigmoid which may have small numerical differences
        error = (C_fused - C_ref).abs().max().item()
        assert error < 0.01, f"Fused SiLU error: {error}"


class TestFusedSpMMBias:
    """Tests for fused SpMM + bias + activation."""

    def test_spmm_bias_gelu(self, sparse_matrix_small, device):
        """Test fused SpMM + bias + GELU."""
        from void import csr_to_void, void_spmm
        from void.fusion import void_spmm_gelu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device)
        bias = torch.randn(N, device=device)

        # Fused kernel with bias
        C_fused = void_spmm_gelu(void_tensor, B, bias=bias)

        # Reference
        C_ref = void_spmm(void_tensor, B)
        C_ref = C_ref + bias
        C_ref = torch.nn.functional.gelu(C_ref)

        error = (C_fused - C_ref).abs().max().item()
        assert error < 0.01, f"Fused bias GELU error: {error}"

    def test_spmm_bias_relu(self, sparse_matrix_small, device):
        """Test fused SpMM + bias + ReLU."""
        from void import csr_to_void, void_spmm
        from void.fusion import void_spmm_relu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device)
        bias = torch.randn(N, device=device)

        C_fused = void_spmm_relu(void_tensor, B, bias=bias)

        C_ref = void_spmm(void_tensor, B)
        C_ref = torch.relu(C_ref + bias)

        error = (C_fused - C_ref).abs().max().item()
        assert error < 1e-5, f"Fused bias ReLU error: {error}"


class TestFusedSparseOp:
    """Tests for FusedSparseOp class."""

    def test_fused_op_gelu(self, sparse_matrix_small, device):
        """Test FusedSparseOp with GELU."""
        from void import csr_to_void
        from void.fusion import create_fused_spmm_gelu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        fused_op = create_fused_spmm_gelu(void_tensor)

        K, N = void_tensor.shape[1], 64
        x = torch.randn(K, N, device=device)

        result = fused_op(x)
        assert result.shape == (void_tensor.shape[0], N)

    def test_fused_op_relu_with_bias(self, sparse_matrix_small, device):
        """Test FusedSparseOp with ReLU and bias."""
        from void import csr_to_void
        from void.fusion import create_fused_spmm_relu

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        bias = torch.randn(void_tensor.shape[1], device=device)  # Actually would be output dim

        # For SpMM, output dim is M, so bias should be length M
        bias = torch.randn(void_tensor.shape[0], device=device)

        # Note: Current impl applies bias to output columns, not rows
        # This test just verifies the API works
        fused_op = create_fused_spmm_relu(void_tensor)

        K, N = void_tensor.shape[1], 64
        x = torch.randn(K, N, device=device)

        result = fused_op(x)
        assert result.shape == (void_tensor.shape[0], N)


class TestFusedSparseMLP:
    """Tests for fused sparse MLP."""

    def test_sparse_mlp_basic(self, device):
        """Test basic sparse MLP."""
        from void import csr_to_void
        from void.fusion import fused_sparse_mlp

        np.random.seed(42)

        # Create two sparse weight matrices
        # W1: [hidden, in] = [512, 256]
        # W2: [out, hidden] = [128, 512]
        W1_sparse = sparse.random(512, 256, density=0.1, format='csr', dtype=np.float32)
        W2_sparse = sparse.random(128, 512, density=0.1, format='csr', dtype=np.float32)

        W1 = csr_to_void(W1_sparse, device=device)
        W2 = csr_to_void(W2_sparse, device=device)

        # Input: [batch, in_features] = [32, 256]
        x = torch.randn(32, 256, device=device)

        # Fused MLP
        y = fused_sparse_mlp(x, W1, W2, activation="gelu")

        # Output should be [batch, out_features] = [32, 128]
        assert y.shape == (32, 128)

    def test_sparse_mlp_with_bias(self, device):
        """Test sparse MLP with biases."""
        from void import csr_to_void
        from void.fusion import fused_sparse_mlp

        np.random.seed(42)

        W1_sparse = sparse.random(512, 256, density=0.1, format='csr', dtype=np.float32)
        W2_sparse = sparse.random(128, 512, density=0.1, format='csr', dtype=np.float32)

        W1 = csr_to_void(W1_sparse, device=device)
        W2 = csr_to_void(W2_sparse, device=device)

        bias1 = torch.randn(512, device=device)
        bias2 = torch.randn(128, device=device)

        x = torch.randn(32, 256, device=device)

        y = fused_sparse_mlp(x, W1, W2, activation="gelu", bias1=bias1, bias2=bias2)

        assert y.shape == (32, 128)

    def test_sparse_mlp_different_activations(self, device):
        """Test sparse MLP with different activations."""
        from void import csr_to_void
        from void.fusion import fused_sparse_mlp

        np.random.seed(42)

        W1_sparse = sparse.random(256, 128, density=0.1, format='csr', dtype=np.float32)
        W2_sparse = sparse.random(64, 256, density=0.1, format='csr', dtype=np.float32)

        W1 = csr_to_void(W1_sparse, device=device)
        W2 = csr_to_void(W2_sparse, device=device)

        x = torch.randn(16, 128, device=device)

        for activation in ["gelu", "relu", "silu"]:
            y = fused_sparse_mlp(x, W1, W2, activation=activation)
            assert y.shape == (16, 64)


class TestFusedSparseLinear:
    """Tests for FusedSparseLinear module."""

    def test_fused_linear_forward(self, sparse_matrix_small, device):
        """Test FusedSparseLinear forward pass."""
        from void import csr_to_void
        from void.fusion import FusedSparseLinear

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        M, K = void_tensor.shape

        layer = FusedSparseLinear(
            in_features=K,
            out_features=M,
            void_tensor=void_tensor,
            activation="gelu",
            bias=True,
        ).to(device)

        x = torch.randn(32, K, device=device)
        y = layer(x)

        assert y.shape == (32, M)

    def test_fused_linear_batched(self, sparse_matrix_small, device):
        """Test FusedSparseLinear with batched input."""
        from void import csr_to_void
        from void.fusion import FusedSparseLinear

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        M, K = void_tensor.shape

        layer = FusedSparseLinear(
            in_features=K,
            out_features=M,
            void_tensor=void_tensor,
            activation="relu",
            bias=False,
        ).to(device)

        # Batched input [batch, seq, features]
        x = torch.randn(4, 8, K, device=device)
        y = layer(x)

        assert y.shape == (4, 8, M)

    def test_fused_linear_no_activation(self, sparse_matrix_small, device):
        """Test FusedSparseLinear without activation."""
        from void import csr_to_void
        from void.fusion import FusedSparseLinear

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        M, K = void_tensor.shape

        layer = FusedSparseLinear(
            in_features=K,
            out_features=M,
            void_tensor=void_tensor,
            activation="none",  # No activation
            bias=True,
        ).to(device)

        x = torch.randn(16, K, device=device)
        y = layer(x)

        assert y.shape == (16, M)


class TestActivationCodes:
    """Tests for activation code conversion."""

    def test_activation_codes(self):
        """Test activation name to code conversion."""
        from void.fusion import get_activation_code, ActivationType

        assert get_activation_code("none") == 0
        assert get_activation_code("relu") == 1
        assert get_activation_code("gelu") == 2
        assert get_activation_code("silu") == 3
        assert get_activation_code("swish") == 3  # Alias

        assert get_activation_code(ActivationType.RELU) == 1
        assert get_activation_code(ActivationType.GELU) == 2


class TestCreateHelpers:
    """Tests for fused operation creation helpers."""

    def test_create_sparse_mlp(self, device):
        """Test create_sparse_mlp helper."""
        from void import csr_to_void
        from void.fusion import create_sparse_mlp, FusedOpType

        np.random.seed(42)

        W1_sparse = sparse.random(256, 128, density=0.1, format='csr', dtype=np.float32)
        W2_sparse = sparse.random(64, 256, density=0.1, format='csr', dtype=np.float32)

        W1 = csr_to_void(W1_sparse, device=device)
        W2 = csr_to_void(W2_sparse, device=device)

        mlp_op = create_sparse_mlp(W1, W2, activation="gelu")

        assert mlp_op.op_type == FusedOpType.SPARSE_MLP
        assert len(mlp_op.sparse_tensors) == 2
        assert mlp_op.activation == "gelu"

        # Test calling the op
        x = torch.randn(16, 128, device=device)
        y = mlp_op(x)
        assert y.shape == (16, 64)
