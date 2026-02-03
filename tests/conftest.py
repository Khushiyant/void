"""
Pytest fixtures for VOID tests.
"""

import pytest
import torch
import numpy as np
import scipy.sparse as sp


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "cuda: marks tests as requiring CUDA (deselect with '-m \"not cuda\"')"
    )


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def device(cuda_available):
    """Return the appropriate device."""
    return torch.device('cuda' if cuda_available else 'cpu')


@pytest.fixture
def sparse_matrix_small():
    """Small sparse matrix for quick tests."""
    np.random.seed(42)
    return sp.random(256, 256, density=0.1, format='csr', dtype=np.float32)


@pytest.fixture
def sparse_matrix_medium():
    """Medium sparse matrix for standard tests."""
    np.random.seed(42)
    return sp.random(512, 512, density=0.1, format='csr', dtype=np.float32)


@pytest.fixture
def sparse_matrix_large():
    """Large sparse matrix for stress tests."""
    np.random.seed(42)
    return sp.random(2048, 2048, density=0.05, format='csr', dtype=np.float32)


@pytest.fixture
def dense_vector(sparse_matrix_small):
    """Dense vector matching sparse matrix dimensions."""
    return torch.randn(sparse_matrix_small.shape[1], dtype=torch.float32)


@pytest.fixture
def dense_matrix(sparse_matrix_small):
    """Dense matrix matching sparse matrix dimensions."""
    return torch.randn(sparse_matrix_small.shape[1], 64, dtype=torch.float32)
