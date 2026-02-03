"""
Real-World Benchmarks for VOID

Tests VOID against cuSPARSE on production workloads:
1. SuiteSparse matrices (scientific computing)
2. Pruned neural network weights (ML inference)
3. Graph adjacency matrices (GNN workloads)
"""

import torch
import scipy.sparse as sp
import numpy as np
import time
import os
import urllib.request
import tarfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from void import csr_to_void, void_spmm
from void.stream_k import void_spmm_stream_k, analyze_workload_balance


# =============================================================================
# Data Download Utilities
# =============================================================================

CACHE_DIR = Path.home() / ".cache" / "void_benchmarks"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> Path:
    """Download file if not cached."""
    if dest.exists():
        return dest
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    return dest


# =============================================================================
# SuiteSparse Matrix Collection
# =============================================================================

SUITESPARSE_MATRICES = {
    # Small matrices for quick testing
    "bcsstk17": "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz",
    "nos4": "https://suitesparse-collection-website.herokuapp.com/MM/HB/nos4.tar.gz",

    # Medium matrices
    "thermomech_dK": "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_dK.tar.gz",

    # Large matrices (download only if needed)
    "af_shell3": "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell3.tar.gz",
}


def load_suitesparse_matrix(name: str) -> Optional[sp.csr_matrix]:
    """
    Load a matrix from the SuiteSparse collection.

    Downloads and caches the matrix.
    """
    if name not in SUITESPARSE_MATRICES:
        print(f"Unknown matrix: {name}")
        print(f"Available: {list(SUITESPARSE_MATRICES.keys())}")
        return None

    url = SUITESPARSE_MATRICES[name]
    cache_path = CACHE_DIR / f"{name}.tar.gz"

    try:
        download_file(url, cache_path)

        # Extract
        extract_dir = CACHE_DIR / name
        if not extract_dir.exists():
            with tarfile.open(cache_path, 'r:gz') as tar:
                tar.extractall(CACHE_DIR)

        # Find .mtx file
        mtx_files = list(extract_dir.rglob("*.mtx"))
        if not mtx_files:
            print(f"No .mtx file found in {extract_dir}")
            return None

        # Load matrix
        from scipy.io import mmread
        matrix = mmread(mtx_files[0]).tocsr().astype(np.float32)
        return matrix

    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None


# =============================================================================
# Pruned Neural Network Weights
# =============================================================================

def create_pruned_linear_weight(
    out_features: int,
    in_features: int,
    sparsity: float = 0.9,
    pattern: str = "unstructured",
) -> sp.csr_matrix:
    """
    Create a pruned linear layer weight matrix.

    Patterns:
    - unstructured: Random element-wise pruning
    - 2:4: NVIDIA structured sparsity (2 non-zeros per 4 elements)
    - block: Block-wise pruning
    """
    if pattern == "unstructured":
        # Random unstructured pruning
        density = 1.0 - sparsity
        return sp.random(out_features, in_features, density=density,
                         format='csr', dtype=np.float32)

    elif pattern == "2:4":
        # 2:4 structured sparsity
        rows, cols, data = [], [], []
        for i in range(out_features):
            for group_start in range(0, in_features, 4):
                group_size = min(4, in_features - group_start)
                positions = np.random.choice(group_size, size=min(2, group_size), replace=False)
                for j in positions:
                    rows.append(i)
                    cols.append(group_start + j)
                    data.append(np.random.randn())
        return sp.csr_matrix((data, (rows, cols)),
                            shape=(out_features, in_features), dtype=np.float32)

    elif pattern == "block":
        # Block-wise pruning (entire blocks of 32x32 are zero or non-zero)
        block_size = 32
        n_row_blocks = (out_features + block_size - 1) // block_size
        n_col_blocks = (in_features + block_size - 1) // block_size

        rows, cols, data = [], [], []
        for br in range(n_row_blocks):
            for bc in range(n_col_blocks):
                if np.random.random() > sparsity:
                    # Fill this block
                    for i in range(block_size):
                        for j in range(block_size):
                            r = br * block_size + i
                            c = bc * block_size + j
                            if r < out_features and c < in_features:
                                rows.append(r)
                                cols.append(c)
                                data.append(np.random.randn())

        return sp.csr_matrix((data, (rows, cols)),
                            shape=(out_features, in_features), dtype=np.float32)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def create_transformer_layer_weights(
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    sparsity: float = 0.9,
    pattern: str = "2:4",
) -> Dict[str, sp.csr_matrix]:
    """
    Create pruned weights for a transformer layer.

    Returns dict with:
    - query, key, value projections
    - output projection
    - FFN layers
    """
    return {
        "query": create_pruned_linear_weight(hidden_size, hidden_size, sparsity, pattern),
        "key": create_pruned_linear_weight(hidden_size, hidden_size, sparsity, pattern),
        "value": create_pruned_linear_weight(hidden_size, hidden_size, sparsity, pattern),
        "output": create_pruned_linear_weight(hidden_size, hidden_size, sparsity, pattern),
        "ffn_up": create_pruned_linear_weight(intermediate_size, hidden_size, sparsity, pattern),
        "ffn_down": create_pruned_linear_weight(hidden_size, intermediate_size, sparsity, pattern),
    }


# =============================================================================
# Graph Adjacency Matrices
# =============================================================================

def create_power_law_graph(
    n_nodes: int,
    avg_degree: int = 10,
    alpha: float = 2.5,
) -> sp.csr_matrix:
    """
    Create a power-law graph adjacency matrix.

    Simulates social networks, citation graphs, etc.
    """
    # Generate degree sequence from power law
    degrees = np.random.pareto(alpha - 1, n_nodes) + 1
    degrees = (degrees / degrees.sum() * n_nodes * avg_degree).astype(int)
    degrees = np.clip(degrees, 1, n_nodes - 1)

    rows, cols = [], []
    for node, degree in enumerate(degrees):
        # Connect to random nodes
        neighbors = np.random.choice(n_nodes, size=degree, replace=False)
        for neighbor in neighbors:
            if neighbor != node:
                rows.append(node)
                cols.append(neighbor)

    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def create_community_graph(
    n_nodes: int,
    n_communities: int = 10,
    intra_prob: float = 0.3,
    inter_prob: float = 0.01,
) -> sp.csr_matrix:
    """
    Create a graph with community structure.

    Nodes within a community are densely connected,
    nodes between communities are sparsely connected.
    """
    community_size = n_nodes // n_communities
    community_assignment = np.repeat(np.arange(n_communities), community_size)

    # Pad if needed
    if len(community_assignment) < n_nodes:
        community_assignment = np.concatenate([
            community_assignment,
            np.full(n_nodes - len(community_assignment), n_communities - 1)
        ])

    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if community_assignment[i] == community_assignment[j]:
                if np.random.random() < intra_prob:
                    rows.extend([i, j])
                    cols.extend([j, i])
            else:
                if np.random.random() < inter_prob:
                    rows.extend([i, j])
                    cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


# =============================================================================
# Benchmark Functions
# =============================================================================

@dataclass
class RealWorldResult:
    name: str
    category: str
    shape: Tuple[int, int]
    nnz: int
    sparsity: float
    block_sparsity: float
    void_time_ms: float
    cusparse_time_ms: float
    stream_k_time_ms: float
    speedup_vs_cusparse: float
    imbalance_ratio: float


def benchmark_matrix(
    name: str,
    category: str,
    matrix: sp.csr_matrix,
    N: int = 512,
    tile_size: int = 32,
    n_iterations: int = 50,
    warmup: int = 10,
) -> RealWorldResult:
    """Benchmark a single matrix."""
    device = torch.device("cuda")
    M, K = matrix.shape

    # Convert to VOID
    void_tensor = csr_to_void(matrix, tile_size=tile_size).cuda()

    # PyTorch sparse
    crow = torch.tensor(matrix.indptr, dtype=torch.int32, device=device)
    col = torch.tensor(matrix.indices, dtype=torch.int32, device=device)
    val = torch.tensor(matrix.data, dtype=torch.float32, device=device)
    torch_sparse = torch.sparse_csr_tensor(crow, col, val, size=(M, K), device=device)

    # Dense B
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # Analyze workload
    balance = analyze_workload_balance(void_tensor)
    imbalance = balance.get("imbalance_ratio", 1.0) if not balance.get("empty", True) else 1.0

    # Warmup
    for _ in range(warmup):
        _ = void_spmm(void_tensor, B)
        _ = torch.sparse.mm(torch_sparse, B)
        if imbalance > 2.0:
            _ = void_spmm_stream_k(void_tensor, B)
    torch.cuda.synchronize()

    # VOID
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = void_spmm(void_tensor, B)
    torch.cuda.synchronize()
    void_time = (time.perf_counter() - start) / n_iterations * 1000

    # cuSPARSE
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = torch.sparse.mm(torch_sparse, B)
    torch.cuda.synchronize()
    cusparse_time = (time.perf_counter() - start) / n_iterations * 1000

    # Stream-K (for imbalanced workloads)
    if imbalance > 2.0:
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = void_spmm_stream_k(void_tensor, B)
        torch.cuda.synchronize()
        stream_k_time = (time.perf_counter() - start) / n_iterations * 1000
    else:
        stream_k_time = void_time

    sparsity = 1.0 - matrix.nnz / (M * K)

    return RealWorldResult(
        name=name,
        category=category,
        shape=(M, K),
        nnz=matrix.nnz,
        sparsity=sparsity,
        block_sparsity=void_tensor.block_sparsity,
        void_time_ms=void_time,
        cusparse_time_ms=cusparse_time,
        stream_k_time_ms=stream_k_time,
        speedup_vs_cusparse=cusparse_time / void_time if void_time > 0 else 0,
        imbalance_ratio=imbalance,
    )


def run_suitesparse_benchmarks() -> List[RealWorldResult]:
    """Benchmark SuiteSparse matrices."""
    results = []

    for name in ["bcsstk17", "nos4"]:  # Start with smaller ones
        print(f"Loading {name}...", end=" ", flush=True)
        matrix = load_suitesparse_matrix(name)
        if matrix is not None:
            print(f"({matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz})")
            result = benchmark_matrix(name, "SuiteSparse", matrix)
            results.append(result)
            print(f"  VOID: {result.void_time_ms:.2f}ms, cuSPARSE: {result.cusparse_time_ms:.2f}ms")
        else:
            print("Failed")

    return results


def run_pruned_nn_benchmarks() -> List[RealWorldResult]:
    """Benchmark pruned neural network weights."""
    results = []

    # Check GPU memory and adjust sizes
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if gpu_mem_gb < 8:
        configs = [
            ("BERT-base 2:4", 768, 3072, 0.5, "2:4"),
            ("GPT2-small 90%", 768, 3072, 0.9, "unstructured"),
        ]
        batch_size = 32
    elif gpu_mem_gb < 16:
        configs = [
            ("BERT-base 2:4", 768, 3072, 0.5, "2:4"),
            ("BERT-large 2:4", 1024, 4096, 0.5, "2:4"),
            ("LLaMA-7B 90%", 4096, 11008, 0.9, "unstructured"),
        ]
        batch_size = 64
    else:
        configs = [
            ("BERT-base 2:4", 768, 3072, 0.5, "2:4"),
            ("BERT-large 2:4", 1024, 4096, 0.5, "2:4"),
            ("LLaMA-7B 90%", 4096, 11008, 0.9, "unstructured"),
            ("LLaMA-7B block", 4096, 11008, 0.9, "block"),
        ]
        batch_size = 128

    for name, hidden, intermediate, sparsity, pattern in configs:
        try:
            print(f"Creating {name}...", end=" ", flush=True)

            # Test FFN up projection (most compute-heavy)
            matrix = create_pruned_linear_weight(intermediate, hidden, sparsity, pattern)
            print(f"({matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz})")

            result = benchmark_matrix(name, "PrunedNN", matrix, N=batch_size)
            results.append(result)
            print(f"  VOID: {result.void_time_ms:.2f}ms, cuSPARSE: {result.cusparse_time_ms:.2f}ms, speedup: {result.speedup_vs_cusparse:.2f}x")

            # Clean up GPU memory
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED: {e}")
            torch.cuda.empty_cache()

    return results


def run_graph_benchmarks() -> List[RealWorldResult]:
    """Benchmark graph adjacency matrices."""
    results = []

    # Check GPU memory and adjust sizes
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem_gb < 8:
        configs = [
            ("PowerLaw-5k", 5000, "power_law", {"avg_degree": 15, "alpha": 2.5}),
            ("Community-5k", 5000, "community", {"n_communities": 10, "intra_prob": 0.2}),
        ]
    elif gpu_mem_gb < 16:
        configs = [
            ("PowerLaw-10k", 10000, "power_law", {"avg_degree": 20, "alpha": 2.5}),
            ("Community-10k", 10000, "community", {"n_communities": 20, "intra_prob": 0.2}),
        ]
    else:
        configs = [
            ("PowerLaw-10k", 10000, "power_law", {"avg_degree": 20, "alpha": 2.5}),
            ("PowerLaw-50k", 50000, "power_law", {"avg_degree": 15, "alpha": 2.5}),
            ("Community-10k", 10000, "community", {"n_communities": 20, "intra_prob": 0.2}),
            ("Community-50k", 50000, "community", {"n_communities": 50, "intra_prob": 0.1}),
        ]

    for name, n_nodes, graph_type, kwargs in configs:
        try:
            print(f"Creating {name}...", end=" ", flush=True)

            if graph_type == "power_law":
                matrix = create_power_law_graph(n_nodes, **kwargs)
            else:
                matrix = create_community_graph(n_nodes, **kwargs)

            print(f"({matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz})")

            # GNN typically uses feature dimension 64-256
            result = benchmark_matrix(name, "Graph", matrix, N=128)
            results.append(result)
            print(f"  VOID: {result.void_time_ms:.2f}ms, cuSPARSE: {result.cusparse_time_ms:.2f}ms, speedup: {result.speedup_vs_cusparse:.2f}x")

            # Clean up GPU memory
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED: {e}")
            torch.cuda.empty_cache()

    return results


def print_results_table(results: List[RealWorldResult], title: str):
    """Print formatted results table."""
    print(f"\n{'=' * 100}")
    print(f"{title}")
    print(f"{'=' * 100}")

    header = (
        f"{'Name':<25} {'Shape':<15} {'Sparsity':>8} {'BlockSp':>8} "
        f"{'VOID':>8} {'cuSPARSE':>8} {'StreamK':>8} {'Speedup':>8}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        shape_str = f"{r.shape[0]}x{r.shape[1]}"
        print(
            f"{r.name:<25} {shape_str:<15} {r.sparsity:>7.1%} {r.block_sparsity:>7.1%} "
            f"{r.void_time_ms:>7.2f}ms {r.cusparse_time_ms:>7.2f}ms "
            f"{r.stream_k_time_ms:>7.2f}ms {r.speedup_vs_cusparse:>7.2f}x"
        )


def main():
    print("=" * 100)
    print("VOID Real-World Benchmarks")
    print("=" * 100)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}\n")

    all_results = []

    # Pruned NN benchmarks (most important for ML use case)
    print("\n" + "=" * 50)
    print("PRUNED NEURAL NETWORK WEIGHTS")
    print("=" * 50)
    nn_results = run_pruned_nn_benchmarks()
    all_results.extend(nn_results)

    # Graph benchmarks
    print("\n" + "=" * 50)
    print("GRAPH ADJACENCY MATRICES")
    print("=" * 50)
    graph_results = run_graph_benchmarks()
    all_results.extend(graph_results)

    # SuiteSparse (scientific)
    print("\n" + "=" * 50)
    print("SUITESPARSE MATRICES")
    print("=" * 50)
    suite_results = run_suitesparse_benchmarks()
    all_results.extend(suite_results)

    # Summary tables
    print_results_table([r for r in all_results if r.category == "PrunedNN"], "Pruned Neural Networks")
    print_results_table([r for r in all_results if r.category == "Graph"], "Graph Neural Networks")
    print_results_table([r for r in all_results if r.category == "SuiteSparse"], "Scientific Computing")

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    for category in ["PrunedNN", "Graph", "SuiteSparse"]:
        cat_results = [r for r in all_results if r.category == category]
        if cat_results:
            avg_speedup = np.mean([r.speedup_vs_cusparse for r in cat_results])
            wins = sum(1 for r in cat_results if r.speedup_vs_cusparse > 1.0)
            print(f"{category}: {avg_speedup:.2f}x avg speedup, {wins}/{len(cat_results)} wins")

    # Best and worst cases
    if all_results:
        best = max(all_results, key=lambda r: r.speedup_vs_cusparse)
        worst = min(all_results, key=lambda r: r.speedup_vs_cusparse)
        print(f"\nBest: {best.name} ({best.speedup_vs_cusparse:.2f}x)")
        print(f"Worst: {worst.name} ({worst.speedup_vs_cusparse:.2f}x)")


if __name__ == "__main__":
    main()
