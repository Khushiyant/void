"""Dynamic kernel selection based on sparsity pattern and workload."""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple
from enum import Enum

from .format import VOIDTensor
from .stream_k import analyze_workload_balance


class KernelVariant(Enum):
    """Available kernel variants for sparse operations."""
    STANDARD = "standard"       # Basic row-parallel kernel
    STREAM_K = "stream_k"       # Load-balanced Stream-K
    WORK_STEAL = "work_steal"   # Work-stealing variant
    FP8 = "fp8"                 # FP8 quantized kernel
    FUSED_GELU = "fused_gelu"   # Fused with GELU activation
    FUSED_RELU = "fused_relu"   # Fused with ReLU activation
    DENSE = "dense"             # Dense fallback for low sparsity


@dataclass
class DispatchDecision:
    """Result of kernel dispatch decision."""
    variant: KernelVariant
    reason: str
    estimated_speedup: float = 1.0
    config: Optional[Dict[str, Any]] = None


@dataclass
class WorkloadCharacteristics:
    """Characteristics of a sparse workload for dispatch decisions."""
    # Basic metrics
    n_blocks: int
    block_sparsity: float  # Fraction of empty blocks
    element_sparsity: float  # Fraction of zeros

    # Distribution metrics
    imbalance_ratio: float  # Max blocks per row / mean blocks per row
    coefficient_of_variation: float  # std / mean of blocks per row

    # Size metrics
    m_dim: int
    k_dim: int
    n_dim: int  # Output columns (from dense matrix B)

    # Data type
    dtype: torch.dtype


def analyze_workload(
    void_tensor: VOIDTensor,
    b_shape: Optional[Tuple[int, int]] = None,
) -> WorkloadCharacteristics:
    """Analyze workload characteristics for dispatch decisions.

    Args:
        void_tensor: VOID sparse matrix
        b_shape: Shape of dense matrix B (K, N) if known

    Returns:
        WorkloadCharacteristics with computed metrics
    """
    from .stream_k import analyze_workload_balance

    # Get balance analysis
    balance = analyze_workload_balance(void_tensor)

    # Compute additional metrics
    M, K = void_tensor.shape
    N = b_shape[1] if b_shape else K

    return WorkloadCharacteristics(
        n_blocks=void_tensor.n_blocks,
        block_sparsity=void_tensor.block_sparsity,
        element_sparsity=void_tensor.sparsity,
        imbalance_ratio=balance.get("imbalance_ratio", 1.0),
        coefficient_of_variation=balance.get("coefficient_of_variation", 0.0),
        m_dim=M,
        k_dim=K,
        n_dim=N,
        dtype=void_tensor.dtype,
    )


class KernelDispatcher:
    """
    Dynamic kernel dispatcher for VOID sparse operations.

    Analyzes workload characteristics and selects the best kernel variant.
    """

    # Thresholds for kernel selection
    DENSE_FALLBACK_THRESHOLD = 0.3  # Use dense if block sparsity < 30%
    STREAM_K_IMBALANCE_THRESHOLD = 4.0  # Use Stream-K if imbalance > 4x
    FP8_SIZE_THRESHOLD = 1024  # Consider FP8 for matrices > 1024x1024

    def __init__(self):
        # Kernel registry
        self._spmm_kernels: Dict[KernelVariant, Callable] = {}
        self._attention_kernels: Dict[KernelVariant, Callable] = {}
        self._register_kernels()

    def _register_kernels(self):
        """Register available kernel implementations."""
        from .ops import void_spmm
        from .stream_k import void_spmm_stream_k, void_spmm_work_stealing

        self._spmm_kernels = {
            KernelVariant.STANDARD: void_spmm,
            KernelVariant.STREAM_K: void_spmm_stream_k,
            KernelVariant.WORK_STEAL: void_spmm_work_stealing,
        }

        # Try to register FP8 kernel
        try:
            from .fp8 import void_spmm_fp8
            self._spmm_kernels[KernelVariant.FP8] = void_spmm_fp8
        except ImportError:
            pass

        # Register fused kernels
        try:
            from .fusion import void_spmm_gelu, void_spmm_relu
            self._spmm_kernels[KernelVariant.FUSED_GELU] = void_spmm_gelu
            self._spmm_kernels[KernelVariant.FUSED_RELU] = void_spmm_relu
        except ImportError:
            pass

    def select_spmm_kernel(
        self,
        void_tensor: VOIDTensor,
        b: torch.Tensor,
        activation: Optional[str] = None,
        force_variant: Optional[KernelVariant] = None,
    ) -> DispatchDecision:
        """Select optimal kernel variant for SpMM.

        Args:
            void_tensor: VOID sparse matrix
            b: Dense matrix B
            activation: Optional activation to fuse ("gelu", "relu")
            force_variant: Force a specific variant (for testing)

        Returns:
            DispatchDecision with selected kernel and reasoning
        """
        if force_variant is not None:
            return DispatchDecision(
                variant=force_variant,
                reason="Forced by user",
                estimated_speedup=1.0,
            )

        # Analyze workload
        chars = analyze_workload(void_tensor, b.shape)

        # Decision tree for kernel selection

        # 1. Check for dense fallback
        if chars.block_sparsity < self.DENSE_FALLBACK_THRESHOLD:
            return DispatchDecision(
                variant=KernelVariant.DENSE,
                reason=f"Low block sparsity ({chars.block_sparsity:.1%}), dense is faster",
                estimated_speedup=0.8,  # Dense usually wins at low sparsity
            )

        # 2. Check for fused activation
        if activation == "gelu" and KernelVariant.FUSED_GELU in self._spmm_kernels:
            return DispatchDecision(
                variant=KernelVariant.FUSED_GELU,
                reason="Fused GELU activation requested",
                estimated_speedup=1.3,  # Fusion typically gives 30% speedup
            )

        if activation == "relu" and KernelVariant.FUSED_RELU in self._spmm_kernels:
            return DispatchDecision(
                variant=KernelVariant.FUSED_RELU,
                reason="Fused ReLU activation requested",
                estimated_speedup=1.25,
            )

        # 3. Check for FP8 (if dtype supports or large matrices)
        if (self._is_fp8_dtype(chars.dtype) and
                KernelVariant.FP8 in self._spmm_kernels):
            return DispatchDecision(
                variant=KernelVariant.FP8,
                reason="FP8 dtype detected",
                estimated_speedup=1.5,
            )

        # 4. Check for load imbalance -> Stream-K
        if chars.imbalance_ratio > self.STREAM_K_IMBALANCE_THRESHOLD:
            return DispatchDecision(
                variant=KernelVariant.STREAM_K,
                reason=f"High workload imbalance ({chars.imbalance_ratio:.1f}x)",
                estimated_speedup=1.4,  # Stream-K helps significantly with imbalance
            )

        # 5. Check coefficient of variation for moderate imbalance
        if chars.coefficient_of_variation > 1.5:
            return DispatchDecision(
                variant=KernelVariant.WORK_STEAL,
                reason=f"Moderate imbalance (CV={chars.coefficient_of_variation:.2f})",
                estimated_speedup=1.2,
            )

        # 6. Default to standard kernel
        return DispatchDecision(
            variant=KernelVariant.STANDARD,
            reason="Standard kernel suitable for balanced workload",
            estimated_speedup=1.0,
        )

    def _is_fp8_dtype(self, dtype: torch.dtype) -> bool:
        """Check if dtype is an FP8 format."""
        fp8_types = []
        if hasattr(torch, 'float8_e4m3fn'):
            fp8_types.append(torch.float8_e4m3fn)
        if hasattr(torch, 'float8_e5m2'):
            fp8_types.append(torch.float8_e5m2)
        return dtype in fp8_types

    def get_kernel(self, variant: KernelVariant) -> Optional[Callable]:
        """Get kernel function for a variant."""
        return self._spmm_kernels.get(variant)


# Global dispatcher instance
_dispatcher = KernelDispatcher()


def void_spmm_auto(
    a: VOIDTensor,
    b: torch.Tensor,
    activation: Optional[str] = None,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Automatic kernel-selecting SpMM: C = A @ B (with optional activation and bias)

    Automatically selects the best kernel variant based on workload characteristics.

    Args:
        a: VOID sparse matrix [M, K]
        b: Dense matrix [K, N]
        activation: Optional activation ("gelu", "relu", "silu")
        bias: Optional bias vector [N] (only used with activation)
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N]
    """
    # Get dispatch decision
    decision = _dispatcher.select_spmm_kernel(a, b, activation)

    # Handle dense fallback
    if decision.variant == KernelVariant.DENSE:
        result = torch.mm(a.to_dense(), b)
        if activation:
            result = _apply_activation_torch(result, activation)
        if bias is not None:
            result = result + bias
        return result

    # Handle fused activation kernels
    if decision.variant == KernelVariant.FUSED_GELU:
        from .fusion import void_spmm_gelu
        return void_spmm_gelu(a, b, bias, out)

    if decision.variant == KernelVariant.FUSED_RELU:
        from .fusion import void_spmm_relu
        return void_spmm_relu(a, b, bias, out)

    # Handle FP8 kernel (requires FP8VOIDTensor)
    if decision.variant == KernelVariant.FP8:
        from .fp8 import FP8VOIDTensor, void_spmm_fp8

        # Check if already FP8
        if not isinstance(a, FP8VOIDTensor):
            # This path shouldn't normally be hit unless dtype is FP8
            # but the tensor is not FP8VOIDTensor
            from .ops import void_spmm
            result = void_spmm(a, b, out)
        else:
            result = void_spmm_fp8(a, b)

        if activation:
            result = _apply_activation_torch(result, activation)
        if bias is not None:
            result = result + bias
        return result

    # Handle Stream-K
    if decision.variant == KernelVariant.STREAM_K:
        from .stream_k import void_spmm_stream_k
        result = void_spmm_stream_k(a, b, out=out)
        if activation:
            result = _apply_activation_torch(result, activation)
        if bias is not None:
            result = result + bias
        return result

    # Handle work-stealing
    if decision.variant == KernelVariant.WORK_STEAL:
        from .stream_k import void_spmm_work_stealing
        result = void_spmm_work_stealing(a, b, out=out)
        if activation:
            result = _apply_activation_torch(result, activation)
        if bias is not None:
            result = result + bias
        return result

    # Default: standard kernel
    from .ops import void_spmm
    result = void_spmm(a, b, out)
    if activation:
        result = _apply_activation_torch(result, activation)
    if bias is not None:
        result = result + bias
    return result


def _apply_activation_torch(x: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply activation function using PyTorch (for non-fused paths)."""
    if activation == "gelu":
        return torch.nn.functional.gelu(x)
    elif activation == "relu":
        return torch.relu(x)
    elif activation == "silu" or activation == "swish":
        return torch.nn.functional.silu(x)
    elif activation == "tanh":
        return torch.tanh(x)
    elif activation == "sigmoid":
        return torch.sigmoid(x)
    else:
        return x


def get_recommended_kernel(
    void_tensor: VOIDTensor,
    b_shape: Tuple[int, int],
    activation: Optional[str] = None,
) -> Tuple[KernelVariant, str]:
    """Get recommended kernel variant without executing.

    Useful for debugging or understanding dispatch decisions.

    Args:
        void_tensor: VOID sparse matrix
        b_shape: Shape of dense matrix B (K, N)
        activation: Optional activation to fuse

    Returns:
        Tuple of (KernelVariant, reason string)
    """
    # Create dummy tensor for analysis
    b_dummy = torch.empty(b_shape, device=void_tensor.values.device)

    decision = _dispatcher.select_spmm_kernel(void_tensor, b_dummy, activation)
    return decision.variant, decision.reason


def analyze_dispatch_decision(
    void_tensor: VOIDTensor,
    b_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Get detailed analysis of dispatch decision.

    Args:
        void_tensor: VOID sparse matrix
        b_shape: Shape of dense matrix B (K, N)

    Returns:
        Dictionary with workload characteristics and recommendations
    """
    chars = analyze_workload(void_tensor, b_shape)
    b_dummy = torch.empty(b_shape, device=void_tensor.values.device)

    # Get decisions for different scenarios
    decision_no_act = _dispatcher.select_spmm_kernel(void_tensor, b_dummy, None)
    decision_gelu = _dispatcher.select_spmm_kernel(void_tensor, b_dummy, "gelu")
    decision_relu = _dispatcher.select_spmm_kernel(void_tensor, b_dummy, "relu")

    return {
        "workload": {
            "n_blocks": chars.n_blocks,
            "block_sparsity": f"{chars.block_sparsity:.1%}",
            "element_sparsity": f"{chars.element_sparsity:.1%}",
            "imbalance_ratio": f"{chars.imbalance_ratio:.2f}",
            "coefficient_of_variation": f"{chars.coefficient_of_variation:.2f}",
            "dimensions": f"({chars.m_dim}, {chars.k_dim}) @ ({chars.k_dim}, {chars.n_dim})",
            "dtype": str(chars.dtype),
        },
        "recommendations": {
            "default": {
                "variant": decision_no_act.variant.value,
                "reason": decision_no_act.reason,
                "estimated_speedup": decision_no_act.estimated_speedup,
            },
            "with_gelu": {
                "variant": decision_gelu.variant.value,
                "reason": decision_gelu.reason,
            },
            "with_relu": {
                "variant": decision_relu.variant.value,
                "reason": decision_relu.reason,
            },
        },
        "thresholds": {
            "dense_fallback": f"<{KernelDispatcher.DENSE_FALLBACK_THRESHOLD:.0%} block sparsity",
            "stream_k": f">{KernelDispatcher.STREAM_K_IMBALANCE_THRESHOLD}x imbalance",
        },
    }


# Export public API
__all__ = [
    # Enums
    "KernelVariant",
    # Dataclasses
    "DispatchDecision",
    "WorkloadCharacteristics",
    # Main functions
    "void_spmm_auto",
    "get_recommended_kernel",
    "analyze_dispatch_decision",
    "analyze_workload",
    # Classes
    "KernelDispatcher",
]
