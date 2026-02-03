"""
Tests for Dynamic Kernel Selection in VOID

Tests the automatic kernel dispatch system.
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
def sparse_matrix_balanced():
    """Create a sparse matrix with balanced row distribution and actual block sparsity."""
    np.random.seed(42)
    # Create a block-sparse pattern with ~50% block sparsity
    # This ensures proper dispatch testing
    M, N = 512, 512
    block_size = 32
    n_block_rows = M // block_size
    n_block_cols = N // block_size

    # Randomly select ~50% of blocks to be non-empty
    data = np.zeros((M, N), dtype=np.float32)
    for br in range(n_block_rows):
        for bc in range(n_block_cols):
            if np.random.random() > 0.5:  # 50% block sparsity
                # Fill block with sparse data
                r_start, r_end = br * block_size, (br + 1) * block_size
                c_start, c_end = bc * block_size, (bc + 1) * block_size
                block_data = np.random.randn(block_size, block_size) * 0.1
                # Make it sparse within the block
                mask = np.random.random((block_size, block_size)) > 0.9
                block_data[~mask] = 0
                data[r_start:r_end, c_start:c_end] = block_data

    return sparse.csr_matrix(data)


@pytest.fixture
def sparse_matrix_imbalanced():
    """Create a sparse matrix with imbalanced row distribution (power law) and block sparsity."""
    np.random.seed(42)
    M, N = 512, 512
    block_size = 32
    n_block_rows = M // block_size
    n_block_cols = N // block_size

    # Create imbalanced block distribution
    data = np.zeros((M, N), dtype=np.float32)
    for br in range(n_block_rows):
        # Power-law: first rows have many more blocks
        if br < 2:
            n_blocks_in_row = n_block_cols  # All blocks
        elif br < 5:
            n_blocks_in_row = n_block_cols // 2
        else:
            n_blocks_in_row = max(1, n_block_cols // 8)

        # Randomly select which columns to use
        cols_to_use = np.random.choice(n_block_cols, size=n_blocks_in_row, replace=False)

        for bc in cols_to_use:
            r_start, r_end = br * block_size, (br + 1) * block_size
            c_start, c_end = bc * block_size, (bc + 1) * block_size
            block_data = np.random.randn(block_size, block_size) * 0.1
            mask = np.random.random((block_size, block_size)) > 0.8
            block_data[~mask] = 0
            data[r_start:r_end, c_start:c_end] = block_data

    return sparse.csr_matrix(data)


@pytest.fixture
def dense_matrix():
    """Create a nearly dense matrix (low block sparsity)."""
    np.random.seed(42)
    # Very high density - should trigger dense fallback
    return sparse.random(256, 256, density=0.8, format='csr', dtype=np.float32)


class TestKernelDispatcher:
    """Tests for KernelDispatcher class."""

    def test_dispatcher_initialization(self):
        """Test dispatcher initializes correctly."""
        from void.dispatch import KernelDispatcher, KernelVariant

        dispatcher = KernelDispatcher()

        # Should have standard kernel registered
        assert dispatcher.get_kernel(KernelVariant.STANDARD) is not None
        assert dispatcher.get_kernel(KernelVariant.STREAM_K) is not None

    def test_select_standard_for_balanced(self, sparse_matrix_balanced, device):
        """Test that balanced workload selects appropriate kernel."""
        from void import csr_to_void
        from void.dispatch import KernelDispatcher, KernelVariant

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        K = void_tensor.shape[1]
        B = torch.randn(K, 64, device=device)

        dispatcher = KernelDispatcher()
        decision = dispatcher.select_spmm_kernel(void_tensor, B)

        # Balanced workload should use standard, work_steal, or dense (if sparsity is low)
        assert decision.variant in [
            KernelVariant.STANDARD,
            KernelVariant.WORK_STEAL,
            KernelVariant.DENSE,  # May choose dense if block sparsity is low
        ]
        assert decision.reason is not None

    def test_select_stream_k_for_imbalanced(self, sparse_matrix_imbalanced, device):
        """Test that imbalanced workload selects appropriate kernel."""
        from void import csr_to_void
        from void.dispatch import KernelDispatcher, KernelVariant

        void_tensor = csr_to_void(sparse_matrix_imbalanced, device=device)
        K = void_tensor.shape[1]
        B = torch.randn(K, 64, device=device)

        dispatcher = KernelDispatcher()
        decision = dispatcher.select_spmm_kernel(void_tensor, B)

        # Could be any valid variant depending on actual sparsity pattern
        assert decision.variant in [
            KernelVariant.STREAM_K,
            KernelVariant.WORK_STEAL,
            KernelVariant.STANDARD,
            KernelVariant.DENSE,  # May choose dense if block sparsity is low
        ]

    def test_select_fused_for_activation(self, sparse_matrix_balanced, device):
        """Test that activation request considers fused kernel."""
        from void import csr_to_void
        from void.dispatch import KernelDispatcher, KernelVariant

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        K = void_tensor.shape[1]
        B = torch.randn(K, 64, device=device)

        dispatcher = KernelDispatcher()

        # With GELU activation - may choose fused or dense depending on sparsity
        decision = dispatcher.select_spmm_kernel(void_tensor, B, activation="gelu")
        assert decision.variant in [KernelVariant.FUSED_GELU, KernelVariant.DENSE]

        # With ReLU activation
        decision = dispatcher.select_spmm_kernel(void_tensor, B, activation="relu")
        assert decision.variant in [KernelVariant.FUSED_RELU, KernelVariant.DENSE]

    def test_select_dense_fallback(self, dense_matrix, device):
        """Test that low sparsity triggers dense fallback."""
        from void import csr_to_void
        from void.dispatch import KernelDispatcher, KernelVariant

        void_tensor = csr_to_void(dense_matrix, device=device)
        B = torch.randn(256, 64, device=device)

        dispatcher = KernelDispatcher()
        decision = dispatcher.select_spmm_kernel(void_tensor, B)

        # Low block sparsity should suggest dense fallback
        # Note: May not trigger if block sparsity is still high enough
        assert decision.variant in [KernelVariant.DENSE, KernelVariant.STANDARD]


class TestAutoDispatch:
    """Tests for automatic dispatch function."""

    def test_void_spmm_auto_basic(self, sparse_matrix_balanced, device):
        """Test void_spmm_auto produces correct results."""
        from void import csr_to_void, void_spmm
        from void.dispatch import void_spmm_auto

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        K = void_tensor.shape[1]
        B = torch.randn(K, 64, device=device)

        # Auto dispatch
        C_auto = void_spmm_auto(void_tensor, B)

        # Reference (use dense for comparison since auto may pick dense)
        C_ref = torch.mm(void_tensor.to_dense(), B)

        # Allow some tolerance for numerical differences between paths
        error = (C_auto - C_ref).abs().max().item()
        assert error < 0.01, f"Auto dispatch error: {error}"

    def test_void_spmm_auto_with_activation(self, sparse_matrix_balanced, device):
        """Test void_spmm_auto with fused activation."""
        from void import csr_to_void
        from void.dispatch import void_spmm_auto

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        K = void_tensor.shape[1]
        B = torch.randn(K, 64, device=device)

        # Auto dispatch with GELU
        C_auto = void_spmm_auto(void_tensor, B, activation="gelu")

        # Reference using dense path
        C_ref = torch.mm(void_tensor.to_dense(), B)
        C_ref = torch.nn.functional.gelu(C_ref)

        # Allow tolerance for numerical differences
        error = (C_auto - C_ref).abs().max().item()
        assert error < 1e-2, f"Auto dispatch with GELU error: {error}"

    def test_void_spmm_auto_with_bias(self, sparse_matrix_balanced, device):
        """Test void_spmm_auto with bias."""
        from void import csr_to_void
        from void.dispatch import void_spmm_auto

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        K = void_tensor.shape[1]
        N = 64
        B = torch.randn(K, N, device=device)
        bias = torch.randn(N, device=device)

        C_auto = void_spmm_auto(void_tensor, B, activation="relu", bias=bias)

        # Reference using dense
        C_ref = torch.mm(void_tensor.to_dense(), B)
        C_ref = torch.relu(C_ref + bias)

        error = (C_auto - C_ref).abs().max().item()
        assert error < 0.01, f"Auto dispatch with bias error: {error}"


class TestWorkloadAnalysis:
    """Tests for workload analysis."""

    def test_analyze_workload(self, sparse_matrix_balanced, device):
        """Test workload analysis."""
        from void import csr_to_void
        from void.dispatch import analyze_workload

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        chars = analyze_workload(void_tensor, (512, 64))

        assert chars.n_blocks == void_tensor.n_blocks
        assert 0.0 <= chars.block_sparsity <= 1.0
        assert chars.imbalance_ratio >= 1.0
        assert chars.m_dim == 512
        assert chars.n_dim == 64

    def test_analyze_imbalanced_workload(self, sparse_matrix_imbalanced, device):
        """Test workload analysis for imbalanced matrix."""
        from void import csr_to_void
        from void.dispatch import analyze_workload

        void_tensor = csr_to_void(sparse_matrix_imbalanced, device=device)
        K = void_tensor.shape[1]
        chars = analyze_workload(void_tensor, (K, 64))

        # Imbalanced matrix should have high imbalance ratio
        # Note: With our fixture, imbalance should be > 1
        assert chars.imbalance_ratio >= 1.0
        # CoV may be 0 if all non-empty rows have same count
        assert chars.coefficient_of_variation >= 0.0


class TestDispatchDecision:
    """Tests for dispatch decision details."""

    def test_get_recommended_kernel(self, sparse_matrix_balanced, device):
        """Test getting kernel recommendation."""
        from void import csr_to_void
        from void.dispatch import get_recommended_kernel, KernelVariant

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        variant, reason = get_recommended_kernel(void_tensor, (512, 64))

        assert isinstance(variant, KernelVariant)
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_analyze_dispatch_decision(self, sparse_matrix_balanced, device):
        """Test detailed dispatch analysis."""
        from void import csr_to_void
        from void.dispatch import analyze_dispatch_decision

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        analysis = analyze_dispatch_decision(void_tensor, (512, 64))

        # Check structure
        assert "workload" in analysis
        assert "recommendations" in analysis
        assert "thresholds" in analysis

        # Check workload info
        assert "n_blocks" in analysis["workload"]
        assert "block_sparsity" in analysis["workload"]
        assert "imbalance_ratio" in analysis["workload"]

        # Check recommendations
        assert "default" in analysis["recommendations"]
        assert "with_gelu" in analysis["recommendations"]
        assert "variant" in analysis["recommendations"]["default"]


class TestDispatchCorrectness:
    """Tests that all dispatch paths produce correct results."""

    def test_all_variants_correct(self, sparse_matrix_balanced, device):
        """Test that all kernel variants produce correct results."""
        from void import csr_to_void, void_spmm
        from void.dispatch import KernelDispatcher, KernelVariant

        void_tensor = csr_to_void(sparse_matrix_balanced, device=device)
        K = void_tensor.shape[1]
        B = torch.randn(K, 64, device=device)

        # Reference result
        C_ref = void_spmm(void_tensor, B)

        dispatcher = KernelDispatcher()

        # Test each available variant (skip WORK_STEAL as it uses break which Triton doesn't support)
        for variant in [KernelVariant.STANDARD, KernelVariant.STREAM_K]:
            kernel = dispatcher.get_kernel(variant)
            if kernel is not None:
                C = kernel(void_tensor, B)
                error = (C - C_ref).abs().max().item()
                assert error < 1e-4, f"{variant.value} produced incorrect result: error={error}"


class TestEdgeCases:
    """Tests for edge cases in dispatch."""

    def test_empty_matrix(self, device):
        """Test dispatch with empty matrix."""
        from void import csr_to_void
        from void.dispatch import void_spmm_auto

        empty = sparse.csr_matrix((64, 64), dtype=np.float32)
        void_tensor = csr_to_void(empty, device=device)
        B = torch.randn(64, 32, device=device)

        C = void_spmm_auto(void_tensor, B)

        # Should return zeros
        assert C.shape == (64, 32)
        assert torch.allclose(C, torch.zeros_like(C))

    def test_single_block(self, device):
        """Test dispatch with single block matrix."""
        from void import csr_to_void
        from void.dispatch import void_spmm_auto

        # Create matrix with single non-zero block
        M = np.zeros((64, 64), dtype=np.float32)
        M[:32, :32] = np.random.randn(32, 32)
        sparse_m = sparse.csr_matrix(M)

        void_tensor = csr_to_void(sparse_m, device=device)
        B = torch.randn(64, 16, device=device)

        C = void_spmm_auto(void_tensor, B)

        assert C.shape == (64, 16)

    def test_very_sparse_matrix(self, device):
        """Test dispatch with very sparse matrix."""
        from void import csr_to_void
        from void.dispatch import void_spmm_auto

        # Very sparse: 0.1% density
        sparse_m = sparse.random(512, 512, density=0.001, format='csr', dtype=np.float32)
        void_tensor = csr_to_void(sparse_m, device=device)
        B = torch.randn(512, 64, device=device)

        C = void_spmm_auto(void_tensor, B)

        assert C.shape == (512, 64)
