"""
Tests for Block Reordering in VOID

Tests different ordering strategies and their effect on correctness and locality.
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
def sparse_matrix_irregular():
    """Create a sparse matrix with irregular sparsity pattern."""
    np.random.seed(42)
    M, N = 512, 512

    # Create power-law distributed rows (some very sparse, some dense)
    rows, cols, data = [], [], []
    for i in range(M):
        # Power-law: row i has roughly 1/sqrt(i+1) * base_nnz elements
        nnz = int(50 / np.sqrt(i + 1)) + 1
        nnz = min(nnz, N)

        col_indices = np.random.choice(N, size=nnz, replace=False)
        rows.extend([i] * nnz)
        cols.extend(col_indices)
        data.extend(np.random.randn(nnz))

    return sparse.csr_matrix((data, (rows, cols)), shape=(M, N), dtype=np.float32)


class TestOrderingStrategies:
    """Tests for different ordering strategies."""

    def test_morton_ordering(self, sparse_matrix_small, device):
        """Test Morton (Z-order) ordering."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        result = reorder_blocks(void_tensor, OrderingStrategy.MORTON)

        # Morton ordering should preserve all blocks
        assert result.void_tensor.n_blocks == void_tensor.n_blocks

        # Original should already be Morton ordered, so minimal change
        assert result.strategy == OrderingStrategy.MORTON

    def test_row_major_ordering(self, sparse_matrix_small, device):
        """Test row-major ordering."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        result = reorder_blocks(void_tensor, OrderingStrategy.ROW_MAJOR)

        # Should preserve all blocks
        assert result.void_tensor.n_blocks == void_tensor.n_blocks

        # Check that blocks are sorted by row
        block_rows = result.void_tensor.block_rows.cpu().numpy()
        assert np.all(block_rows[:-1] <= block_rows[1:]), "Blocks should be sorted by row"

    def test_col_major_ordering(self, sparse_matrix_small, device):
        """Test column-major ordering."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        result = reorder_blocks(void_tensor, OrderingStrategy.COL_MAJOR)

        # Should preserve all blocks
        assert result.void_tensor.n_blocks == void_tensor.n_blocks

        # Check that blocks are sorted by column
        block_cols = result.void_tensor.block_cols.cpu().numpy()
        assert np.all(block_cols[:-1] <= block_cols[1:]), "Blocks should be sorted by column"

    def test_hilbert_ordering(self, sparse_matrix_small, device):
        """Test Hilbert curve ordering."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        result = reorder_blocks(void_tensor, OrderingStrategy.HILBERT)

        # Should preserve all blocks
        assert result.void_tensor.n_blocks == void_tensor.n_blocks
        assert result.strategy == OrderingStrategy.HILBERT

    def test_affinity_ordering(self, sparse_matrix_irregular, device):
        """Test affinity-based ordering."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_irregular, device=device)
        result = reorder_blocks(
            void_tensor,
            OrderingStrategy.AFFINITY,
            dense_cols=64
        )

        # Should preserve all blocks
        assert result.void_tensor.n_blocks == void_tensor.n_blocks
        assert result.strategy == OrderingStrategy.AFFINITY


class TestReorderingInvariance:
    """Tests that reordering preserves correctness."""

    def test_spmm_invariance(self, sparse_matrix_small, device):
        """Test that SpMM produces same results regardless of ordering."""
        from void import csr_to_void, void_spmm
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        # Dense matrix B
        K, N = void_tensor.shape[1], 64
        B = torch.randn(K, N, device=device)

        # Original result
        C_original = void_spmm(void_tensor, B)

        # Test each ordering strategy
        for strategy in [
            OrderingStrategy.MORTON,
            OrderingStrategy.ROW_MAJOR,
            OrderingStrategy.COL_MAJOR,
            OrderingStrategy.HILBERT,
        ]:
            result = reorder_blocks(void_tensor, strategy)
            C_reordered = void_spmm(result.void_tensor, B)

            # Results should be identical (or very close due to floating point)
            error = (C_original - C_reordered).abs().max().item()
            assert error < 1e-5, f"{strategy.value} ordering changed SpMM result: error={error}"

    def test_to_dense_invariance(self, sparse_matrix_small, device):
        """Test that to_dense produces same matrix regardless of ordering."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        dense_original = void_tensor.to_dense()

        for strategy in OrderingStrategy:
            try:
                result = reorder_blocks(void_tensor, strategy)
                dense_reordered = result.void_tensor.to_dense()

                error = (dense_original - dense_reordered).abs().max().item()
                assert error < 1e-6, f"{strategy.value} ordering changed dense conversion"
            except ValueError:
                # Some strategies may need additional args
                pass


class TestAffinityComputation:
    """Tests for affinity-based reordering."""

    def test_compute_block_affinity(self, sparse_matrix_irregular, device):
        """Test affinity computation."""
        from void import csr_to_void
        from void.reorder import compute_block_affinity

        void_tensor = csr_to_void(sparse_matrix_irregular, device=device)
        affinity_info = compute_block_affinity(void_tensor, dense_cols=64)

        # Should have valid permutations
        n_block_rows = void_tensor.block_grid[0]
        n_block_cols = void_tensor.block_grid[1]

        assert affinity_info.row_permutation.shape == (n_block_rows,)
        assert affinity_info.col_permutation.shape == (n_block_cols,)

        # Permutation should be a valid permutation (bijective)
        row_perm_sorted = torch.sort(affinity_info.row_permutation[:void_tensor.block_grid[0]])[0]
        assert torch.all(row_perm_sorted == torch.arange(n_block_rows, device=device))

    def test_locality_score(self, sparse_matrix_irregular, device):
        """Test locality score computation."""
        from void import csr_to_void
        from void.reorder import compute_block_affinity

        void_tensor = csr_to_void(sparse_matrix_irregular, device=device)
        affinity_info = compute_block_affinity(void_tensor, dense_cols=64)

        # Locality score should be between 0 and 1
        assert 0.0 <= affinity_info.locality_score <= 1.0


class TestOrderingQuality:
    """Tests for ordering quality analysis."""

    def test_analyze_ordering_quality(self, sparse_matrix_small, device):
        """Test ordering quality analysis."""
        from void import csr_to_void
        from void.reorder import analyze_ordering_quality

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        quality = analyze_ordering_quality(void_tensor)

        # Should return valid metrics
        assert "n_blocks" in quality
        assert "mean_neighbor_distance" in quality
        assert "row_locality_score" in quality

        assert quality["n_blocks"] == void_tensor.n_blocks
        assert quality["row_locality_score"] >= 0.0
        assert quality["row_locality_score"] <= 1.0

    def test_hilbert_vs_morton_locality(self, sparse_matrix_small, device):
        """Test that Hilbert often has better locality than Morton."""
        from void import csr_to_void
        from void.reorder import reorder_blocks, OrderingStrategy, analyze_ordering_quality

        void_tensor = csr_to_void(sparse_matrix_small, device=device)

        morton_result = reorder_blocks(void_tensor, OrderingStrategy.MORTON)
        hilbert_result = reorder_blocks(void_tensor, OrderingStrategy.HILBERT)

        morton_quality = analyze_ordering_quality(morton_result.void_tensor)
        hilbert_quality = analyze_ordering_quality(hilbert_result.void_tensor)

        # Hilbert should have similar or better locality
        # (Not always better, but should be comparable)
        assert hilbert_quality["mean_neighbor_distance"] < morton_quality["mean_neighbor_distance"] * 2


class TestConvenienceFunctions:
    """Tests for convenience reordering functions."""

    def test_reorder_for_spmm(self, sparse_matrix_irregular, device):
        """Test reorder_for_spmm convenience function."""
        from void import csr_to_void
        from void.reorder import reorder_for_spmm

        void_tensor = csr_to_void(sparse_matrix_irregular, device=device)
        reordered = reorder_for_spmm(void_tensor, b_cols=64)

        assert reordered.n_blocks == void_tensor.n_blocks

    def test_reorder_to_row_major(self, sparse_matrix_small, device):
        """Test reorder_to_row_major convenience function."""
        from void import csr_to_void
        from void.reorder import reorder_to_row_major

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        reordered = reorder_to_row_major(void_tensor)

        # Check row-major order
        block_rows = reordered.block_rows.cpu().numpy()
        assert np.all(block_rows[:-1] <= block_rows[1:])

    def test_reorder_to_hilbert(self, sparse_matrix_small, device):
        """Test reorder_to_hilbert convenience function."""
        from void import csr_to_void
        from void.reorder import reorder_to_hilbert

        void_tensor = csr_to_void(sparse_matrix_small, device=device)
        reordered = reorder_to_hilbert(void_tensor)

        assert reordered.n_blocks == void_tensor.n_blocks


class TestHilbertCurve:
    """Tests for Hilbert curve encoding."""

    def test_hilbert_encode_decode(self):
        """Test Hilbert curve encoding/decoding."""
        from void.reorder import hilbert_xy2d, hilbert_d2xy

        n = 16  # Grid size

        # Test roundtrip for several points
        for x in range(n):
            for y in range(n):
                d = hilbert_xy2d(n, x, y)
                x2, y2 = hilbert_d2xy(n, d)
                assert x == x2 and y == y2, f"Hilbert roundtrip failed for ({x}, {y})"

    def test_hilbert_encode_batch(self):
        """Test batch Hilbert encoding."""
        from void.reorder import hilbert_encode_batch, hilbert_xy2d

        rows = np.array([0, 1, 2, 3, 4])
        cols = np.array([0, 1, 2, 3, 4])
        n = 8

        codes = hilbert_encode_batch(rows, cols, n)

        # Verify against single encoding
        for i, (r, c) in enumerate(zip(rows, cols)):
            expected = hilbert_xy2d(n, r, c)
            assert codes[i] == expected
