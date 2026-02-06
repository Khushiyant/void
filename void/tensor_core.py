"""Tensor Core configuration and alignment utilities."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, Dict, Any
import torch


class TCPrecision(Enum):
    """Tensor Core precision modes."""
    TF32 = "tf32"          # TensorFloat-32 (Ampere+): fast FP32-like precision
    FP16 = "fp16"          # Half precision (Volta+)
    BF16 = "bf16"          # Brain float 16 (Ampere+)
    FP8_E4M3 = "fp8_e4m3"  # FP8 E4M3 format (Hopper/Ada)
    FP8_E5M2 = "fp8_e5m2"  # FP8 E5M2 format (Hopper/Ada)


# Minimum dimensions for Tensor Core activation
TC_MIN_DIMS = {
    TCPrecision.TF32: {"m": 16, "n": 16, "k": 16},
    TCPrecision.FP16: {"m": 8, "n": 8, "k": 16},
    TCPrecision.BF16: {"m": 8, "n": 8, "k": 16},
    TCPrecision.FP8_E4M3: {"m": 16, "n": 16, "k": 16},
    TCPrecision.FP8_E5M2: {"m": 16, "n": 16, "k": 16},
}


@dataclass
class TensorCoreConfig:
    """
    Configuration for Tensor Core operations.

    Attributes:
        precision: Tensor Core precision mode
        min_k: Minimum K dimension for TC activation
        min_m: Minimum M dimension
        min_n: Minimum N dimension
        alignment: Memory alignment in bytes
        acc_dtype: Accumulator data type (always FP32 for stability)
        allow_tf32: Allow TF32 for FP32 inputs (faster, slightly less precise)
    """
    precision: TCPrecision = TCPrecision.TF32
    min_k: int = 16
    min_m: int = 16
    min_n: int = 16
    alignment: int = 16
    acc_dtype: str = "fp32"
    allow_tf32: bool = True

    def __post_init__(self):
        """Update min dimensions based on precision mode."""
        if self.precision in TC_MIN_DIMS:
            dims = TC_MIN_DIMS[self.precision]
            self.min_k = max(self.min_k, dims["k"])
            self.min_m = max(self.min_m, dims["m"])
            self.min_n = max(self.min_n, dims["n"])

    @classmethod
    def for_dtype(cls, dtype: torch.dtype) -> 'TensorCoreConfig':
        """Create config optimized for a specific dtype."""
        if dtype == torch.float32:
            return cls(precision=TCPrecision.TF32, allow_tf32=True)
        elif dtype == torch.float16:
            return cls(precision=TCPrecision.FP16)
        elif dtype == torch.bfloat16:
            return cls(precision=TCPrecision.BF16)
        elif hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
            return cls(precision=TCPrecision.FP8_E4M3)
        elif hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
            return cls(precision=TCPrecision.FP8_E5M2)
        else:
            return cls()  # Default TF32


@dataclass
class TCAlignmentResult:
    """Result of Tensor Core alignment check."""
    is_aligned: bool
    tile_m: int
    tile_k: int
    tile_n: int
    issues: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


def check_tensor_core_alignment(
    tile_m: int,
    tile_k: int,
    tile_n: int,
    config: Optional[TensorCoreConfig] = None,
) -> TCAlignmentResult:
    """
    Check if tile dimensions are properly aligned for Tensor Cores.

    Args:
        tile_m: M dimension of tile
        tile_k: K dimension of tile (inner dimension for matmul)
        tile_n: N dimension of tile
        config: TC configuration (uses default if None)

    Returns:
        TCAlignmentResult with alignment status and any issues
    """
    if config is None:
        config = TensorCoreConfig()

    issues = []
    recommendations = []

    # Check minimum dimensions
    if tile_k < config.min_k:
        issues.append(f"tile_k={tile_k} < min_k={config.min_k}")
        recommendations.append(f"Increase tile_k to at least {config.min_k}")

    if tile_m < config.min_m:
        issues.append(f"tile_m={tile_m} < min_m={config.min_m}")
        recommendations.append(f"Increase tile_m to at least {config.min_m}")

    if tile_n < config.min_n:
        issues.append(f"tile_n={tile_n} < min_n={config.min_n}")
        recommendations.append(f"Increase tile_n to at least {config.min_n}")

    # Check power-of-2 alignment for optimal performance
    for name, val in [("tile_m", tile_m), ("tile_k", tile_k), ("tile_n", tile_n)]:
        if val & (val - 1) != 0:  # Not power of 2
            issues.append(f"{name}={val} is not a power of 2")
            next_pow2 = 1 << (val - 1).bit_length()
            recommendations.append(f"Consider {name}={next_pow2} for optimal alignment")

    is_aligned = len([i for i in issues if "min_" in i]) == 0  # Only critical issues

    return TCAlignmentResult(
        is_aligned=is_aligned,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        issues=issues,
        recommendations=recommendations,
    )


def get_optimal_tc_tile_size(
    M: int,
    K: int,
    N: int,
    config: Optional[TensorCoreConfig] = None,
    max_tile: int = 128,
) -> Tuple[int, int, int]:
    """
    Compute optimal tile sizes for Tensor Core operations.

    Args:
        M: Matrix M dimension
        K: Matrix K dimension
        N: Matrix N dimension
        config: TC configuration
        max_tile: Maximum tile size to consider

    Returns:
        Tuple of (tile_m, tile_k, tile_n)
    """
    if config is None:
        config = TensorCoreConfig()

    def best_tile(dim: int, min_size: int, max_size: int = max_tile) -> int:
        """Find best power-of-2 tile size for a dimension."""
        # Start with largest power-of-2 that fits
        for size in [128, 64, 32, 16]:
            if size <= max_size and size >= min_size:
                if dim >= size:
                    return size
        return min_size

    tile_m = best_tile(M, config.min_m)
    tile_k = best_tile(K, config.min_k)
    tile_n = best_tile(N, config.min_n)

    return tile_m, tile_k, tile_n


def should_use_tensor_cores(
    tile_m: int,
    tile_k: int,
    tile_n: int,
    n_blocks: int,
    config: Optional[TensorCoreConfig] = None,
) -> Tuple[bool, str]:
    """
    Determine if Tensor Cores should be used for an operation.

    Args:
        tile_m, tile_k, tile_n: Tile dimensions
        n_blocks: Number of blocks in sparse matrix
        config: TC configuration

    Returns:
        Tuple of (should_use, reason)
    """
    if config is None:
        config = TensorCoreConfig()

    result = check_tensor_core_alignment(tile_m, tile_k, tile_n, config)

    if not result.is_aligned:
        return False, f"Dimensions not aligned: {', '.join(result.issues[:2])}"

    # TC overhead isn't worth it for very small workloads
    min_blocks_for_tc = 10  # Empirical threshold
    if n_blocks < min_blocks_for_tc:
        return False, f"Workload too small ({n_blocks} blocks < {min_blocks_for_tc})"

    return True, "Dimensions aligned and workload sufficient for Tensor Cores"


def get_gpu_tensor_core_info() -> Dict[str, Any]:
    """
    Get information about Tensor Core support on current GPU.

    Returns:
        Dictionary with GPU capabilities
    """
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    compute_cap = (props.major, props.minor)

    info = {
        "available": True,
        "device_name": props.name,
        "compute_capability": compute_cap,
        "total_memory_gb": props.total_memory / (1024**3),
    }

    # Determine TC support based on compute capability
    if compute_cap >= (9, 0):
        info["tc_generation"] = "Hopper"
        info["supported_precisions"] = ["TF32", "FP16", "BF16", "FP8_E4M3", "FP8_E5M2"]
        info["sparse_tc_support"] = True  # 2:4 structured sparsity
    elif compute_cap >= (8, 0):
        info["tc_generation"] = "Ampere"
        info["supported_precisions"] = ["TF32", "FP16", "BF16"]
        info["sparse_tc_support"] = True  # 2:4 structured sparsity
    elif compute_cap >= (7, 5):
        info["tc_generation"] = "Turing"
        info["supported_precisions"] = ["FP16"]
        info["sparse_tc_support"] = False
    elif compute_cap >= (7, 0):
        info["tc_generation"] = "Volta"
        info["supported_precisions"] = ["FP16"]
        info["sparse_tc_support"] = False
    else:
        info["tc_generation"] = None
        info["supported_precisions"] = []
        info["sparse_tc_support"] = False

    return info


# =============================================================================
# Triton Kernel Configuration Helpers
# =============================================================================

def get_triton_tc_config(
    dtype: torch.dtype,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> Dict[str, Any]:
    """
    Get Triton kernel configuration for Tensor Core operations.

    Args:
        dtype: Input data type
        tile_m, tile_k, tile_n: Tile dimensions

    Returns:
        Dictionary with kernel configuration
    """
    config = TensorCoreConfig.for_dtype(dtype)
    result = check_tensor_core_alignment(tile_m, tile_k, tile_n, config)

    return {
        "use_tensor_cores": result.is_aligned,
        "allow_tf32": config.allow_tf32 and dtype == torch.float32,
        "acc_dtype": "float32",  # Always accumulate in FP32
        "precision": config.precision.value,
        "tile_m": tile_m,
        "tile_k": tile_k,
        "tile_n": tile_n,
        "issues": result.issues,
    }


# Export public API
__all__ = [
    # Enums
    "TCPrecision",
    # Dataclasses
    "TensorCoreConfig",
    "TCAlignmentResult",
    # Functions
    "check_tensor_core_alignment",
    "get_optimal_tc_tile_size",
    "should_use_tensor_cores",
    "get_gpu_tensor_core_info",
    "get_triton_tc_config",
]
