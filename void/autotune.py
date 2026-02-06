"""Kernel autotuning for tile sizes and warp configurations."""

import torch
import triton
import triton.language as tl
from typing import Dict, Tuple, Optional, List
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

from .ops import get_triton_dtype


@dataclass
class KernelConfig:
    """Configuration for a VOID kernel."""
    TILE_M: int
    TILE_K: int
    TILE_N: int
    num_warps: int
    num_stages: int = 3

    def __hash__(self):
        return hash((self.TILE_M, self.TILE_K, self.TILE_N, self.num_warps, self.num_stages))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# =============================================================================
# Autotune Cache
# =============================================================================

class AutotuneCache:
    """
    Persistent cache for autotuned kernel configurations.

    Stores best configurations per (GPU, operation, matrix_shape) to avoid
    re-tuning on every run.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/void_autotune")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get GPU identifier
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_name = gpu_name.replace(" ", "_").replace("/", "_")
            self.cache_file = self.cache_dir / f"cache_{gpu_name}.json"
        else:
            self.cache_file = self.cache_dir / "cache_cpu.json"

        self.cache: Dict = {}
        self._load()

    def _load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert dict back to KernelConfig
                    self.cache = {k: KernelConfig.from_dict(v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Failed to load autotune cache: {e}")
                self.cache = {}

    def _save(self):
        """Save cache to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save autotune cache: {e}")

    def make_key(self, op: str, M: int, K: int, N: int, tile_m: int, tile_k: int) -> str:
        """Create cache key from operation and dimensions."""
        # Round to nearest power of 2 for generalization
        M_key = 2 ** max(0, int(torch.log2(torch.tensor(float(M))).item()))
        K_key = 2 ** max(0, int(torch.log2(torch.tensor(float(K))).item()))
        N_key = 2 ** max(0, int(torch.log2(torch.tensor(float(N))).item()))
        return f"{op}_M{M_key}_K{K_key}_N{N_key}_TM{tile_m}_TK{tile_k}"

    def get(self, key: str) -> Optional[KernelConfig]:
        """Get cached configuration."""
        return self.cache.get(key)

    def set(self, key: str, config: KernelConfig):
        """Set and persist configuration."""
        self.cache[key] = config
        self._save()


# Global cache instance
_autotune_cache = AutotuneCache()


# =============================================================================
# Autotuned SpMM Kernel
# =============================================================================

@triton.jit
def void_spmm_kernel_autotuned(
    # Sparse matrix A (VOID format)
    a_values_ptr,
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B
    b_ptr,
    # Output matrix C
    c_ptr,
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes (constexpr for performance)
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Autotuned SpMM kernel with configurable tile sizes.

    This is the same as void_spmm_kernel but with TILE_N as a parameter
    that can be tuned at runtime.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= n_block_rows:
        return

    # Get the range of blocks in this row
    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)

    # Initialize accumulator in FP32
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Output column range
    col_start = pid_n * TILE_N

    # Iterate over blocks in this row
    for block_idx in range(row_start, row_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load A tile
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile = tl.load(a_tile_ptr).to(tl.float32)

        # Load B tile
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, col_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Store output
    out_row = pid_m * TILE_M
    c_tile_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(out_row, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    tl.store(c_tile_ptr, acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def benchmark_config(
    config: KernelConfig,
    a_values: torch.Tensor,
    a_block_rows: torch.Tensor,
    a_block_cols: torch.Tensor,
    a_row_ptr: torch.Tensor,
    a_block_idx: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    M: int, K: int, N: int,
    n_blocks: int,
    n_block_rows: int,
    tile_m: int,
    tile_k: int,
    n_iterations: int = 10,
) -> float:
    """
    Benchmark a specific kernel configuration.

    Returns average time in milliseconds.
    """
    output_dtype = get_triton_dtype(b.dtype)

    grid = (n_block_rows, triton.cdiv(N, config.TILE_N))

    # Warmup
    for _ in range(3):
        void_spmm_kernel_autotuned[grid](
            a_values, a_block_rows, a_block_cols, a_row_ptr, a_block_idx,
            b, out,
            M, N, K, n_blocks, n_block_rows,
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            TILE_M=tile_m,
            TILE_K=tile_k,
            TILE_N=config.TILE_N,
            OUTPUT_DTYPE=output_dtype,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    torch.cuda.synchronize()

    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(n_iterations):
        void_spmm_kernel_autotuned[grid](
            a_values, a_block_rows, a_block_cols, a_row_ptr, a_block_idx,
            b, out,
            M, N, K, n_blocks, n_block_rows,
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            TILE_M=tile_m,
            TILE_K=tile_k,
            TILE_N=config.TILE_N,
            OUTPUT_DTYPE=output_dtype,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / n_iterations) * 1000  # ms


def autotune_spmm(
    void_tensor: 'VOIDTensor',
    b: torch.Tensor,
    use_cache: bool = True,
) -> KernelConfig:
    """
    Automatically find the best kernel configuration for SpMM.

    Args:
        void_tensor: VOID sparse matrix
        b: Dense matrix
        use_cache: Use persistent cache (default: True)

    Returns:
        Best KernelConfig
    """
    M, K = void_tensor.shape
    _, N = b.shape
    tile_m, tile_k = void_tensor.tile_size

    # Check cache first
    if use_cache:
        cache_key = _autotune_cache.make_key("spmm", M, K, N, tile_m, tile_k)
        cached_config = _autotune_cache.get(cache_key)
        if cached_config is not None:
            return cached_config

    # Generate candidate configurations
    configs = generate_spmm_configs(N, tile_m, tile_k)

    # Prepare kernel arguments
    row_ptr, block_indices = void_tensor.get_row_block_info()
    n_block_rows = void_tensor.block_grid[0]
    n_blocks = void_tensor.n_blocks

    # Allocate output
    out = torch.zeros(M, N, dtype=b.dtype, device=b.device)

    # Benchmark each configuration
    best_config = None
    best_time = float('inf')

    for config in configs:
        try:
            time_ms = benchmark_config(
                config,
                void_tensor.values, void_tensor.block_rows, void_tensor.block_cols,
                row_ptr, block_indices,
                b, out,
                M, K, N, n_blocks, n_block_rows, tile_m, tile_k
            )

            if time_ms < best_time:
                best_time = time_ms
                best_config = config

        except Exception as e:
            # Configuration failed (e.g., too many registers)
            continue

    if best_config is None:
        # Fallback to safe default
        best_config = KernelConfig(
            TILE_M=tile_m,
            TILE_K=tile_k,
            TILE_N=64,
            num_warps=4,
            num_stages=3
        )

    # Cache the result
    if use_cache:
        _autotune_cache.set(cache_key, best_config)

    return best_config


def generate_spmm_configs(N: int, tile_m: int, tile_k: int) -> List[KernelConfig]:
    """
    Generate candidate configurations for SpMM based on problem size.

    Returns list of configs sorted by likelihood of being good.

    2025 Update: Expanded num_stages to [2,3,4,5] for better memory hiding
    on modern GPUs with larger register files.
    """
    configs = []

    # TILE_N candidates (powers of 2 from 16 to 128)
    tile_n_candidates = [16, 32, 64, 128]

    # Filter to reasonable sizes for N
    tile_n_candidates = [tn for tn in tile_n_candidates if tn <= N and tn <= 128]

    # Warp counts (4 or 8 is usually best)
    warp_candidates = [4, 8]

    # Pipeline stages - expanded for 2025 SOTA
    # More stages = better memory latency hiding on modern GPUs
    stage_candidates = [2, 3, 4, 5]

    # Generate all combinations
    for tile_n in tile_n_candidates:
        for num_warps in warp_candidates:
            for num_stages in stage_candidates:
                configs.append(KernelConfig(
                    TILE_M=tile_m,
                    TILE_K=tile_k,
                    TILE_N=tile_n,
                    num_warps=num_warps,
                    num_stages=num_stages
                ))

    return configs


def generate_attention_configs(head_dim: int, block_size: int) -> List[dict]:
    """
    Generate candidate configurations for sparse attention.

    Returns list of config dicts for attention kernel tuning.
    """
    configs = []

    # Warp counts
    warp_candidates = [4, 8]

    # Pipeline stages - expanded for 2025 SOTA
    stage_candidates = [2, 3, 4, 5]

    for num_warps in warp_candidates:
        for num_stages in stage_candidates:
            configs.append({
                "num_warps": num_warps,
                "num_stages": num_stages,
                "head_dim": head_dim,
                "block_size": block_size,
            })

    return configs


# =============================================================================
# Public API
# =============================================================================

def void_spmm_with_autotune(
    a: 'VOIDTensor',
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    config: Optional[KernelConfig] = None,
) -> torch.Tensor:
    """
    SpMM with automatic kernel configuration tuning.

    On first call, benchmarks different configurations and caches the best one.
    Subsequent calls use the cached configuration for fast execution.

    Args:
        a: VOID sparse matrix
        b: Dense matrix [K, N]
        out: Optional output buffer
        config: Optional manual configuration (skips autotuning)

    Returns:
        Dense output [M, N]
    """
    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert b.is_cuda and a.values.is_cuda

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    # Allocate output
    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        return out

    # Get or autotune configuration
    if config is None:
        config = autotune_spmm(a, b, use_cache=True)

    # Get block indices
    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]

    # Get output dtype
    output_dtype = get_triton_dtype(b.dtype)

    # Launch kernel
    grid = (n_block_rows, triton.cdiv(N, config.TILE_N))

    void_spmm_kernel_autotuned[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m,
        TILE_K=tile_k,
        TILE_N=config.TILE_N,
        OUTPUT_DTYPE=output_dtype,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return out


def clear_autotune_cache():
    """Clear the persistent autotune cache."""
    global _autotune_cache
    _autotune_cache = AutotuneCache()
    if _autotune_cache.cache_file.exists():
        _autotune_cache.cache_file.unlink()
