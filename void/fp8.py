"""FP8 (E4M3/E5M2) quantization for VOID sparse matrices."""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Literal
from enum import Enum


# =============================================================================
# FP8 Format Constants
# =============================================================================

class FP8Format(Enum):
    """FP8 format types supported by modern GPUs."""
    E4M3 = "e4m3"  # 4 exponent bits, 3 mantissa bits - higher precision
    E5M2 = "e5m2"  # 5 exponent bits, 2 mantissa bits - higher dynamic range


# Maximum representable values for FP8 formats
FP8_MAX = {
    FP8Format.E4M3: 448.0,    # Max value for E4M3
    FP8Format.E5M2: 57344.0,  # Max value for E5M2
}

FP8_MIN = {
    FP8Format.E4M3: -448.0,
    FP8Format.E5M2: -57344.0,
}


# =============================================================================
# Configuration and Scaling Info
# =============================================================================

@dataclass
class FP8Config:
    """Configuration for FP8 quantization.

    Attributes:
        format: FP8 format to use (E4M3 or E5M2)
        scale_mode: How to compute scaling factors
            - "per_tensor": Single scale for entire tensor
            - "per_block": Separate scale per VOID block
            - "per_row": Separate scale per row (for attention)
        margin: Safety margin for dynamic scaling (to avoid overflow)
        amax_history_len: Length of AMAX history for smoothing
    """
    format: FP8Format = FP8Format.E4M3
    scale_mode: Literal["per_tensor", "per_block", "per_row"] = "per_tensor"
    margin: float = 0.0
    amax_history_len: int = 16

    @property
    def max_value(self) -> float:
        """Maximum representable value for this format."""
        return FP8_MAX[self.format]

    @property
    def torch_dtype(self) -> torch.dtype:
        """Corresponding PyTorch dtype."""
        if self.format == FP8Format.E4M3:
            return torch.float8_e4m3fn
        else:
            return torch.float8_e5m2


@dataclass
class FP8ScalingInfo:
    """Scaling information for FP8 tensors.

    Stores scale factors and AMAX history for dynamic quantization.
    """
    scale: torch.Tensor  # Scale factor(s) - shape depends on scale_mode
    scale_inv: torch.Tensor  # Inverse scale for dequantization
    amax: torch.Tensor  # Current absolute max value(s)
    amax_history: torch.Tensor  # History of AMAX values for smoothing
    config: FP8Config = field(default_factory=FP8Config)

    def update_amax(self, new_amax: torch.Tensor) -> None:
        """Update AMAX history with new values."""
        # Roll history and insert new value
        self.amax_history = torch.roll(self.amax_history, shifts=1, dims=-1)
        if self.amax_history.dim() == 1:
            self.amax_history[0] = new_amax
        else:
            self.amax_history[..., 0] = new_amax
        self.amax = new_amax

    def compute_scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute new scale factors from AMAX history."""
        # Use max of history for stable scaling
        if self.amax_history.dim() == 1:
            smoothed_amax = self.amax_history.max()
        else:
            smoothed_amax = self.amax_history.max(dim=-1).values

        # Compute scale to map values to FP8 range
        fp8_max = self.config.max_value * (1.0 - self.config.margin)
        self.scale = smoothed_amax / fp8_max
        self.scale = torch.clamp(self.scale, min=1e-12)  # Avoid division by zero
        self.scale_inv = 1.0 / self.scale

        return self.scale, self.scale_inv


def create_scaling_info(
    shape: Tuple[int, ...],
    config: FP8Config,
    device: Union[str, torch.device] = 'cuda',
) -> FP8ScalingInfo:
    """Create initial FP8ScalingInfo for a tensor shape.

    Args:
        shape: Shape of the tensor to be quantized
        config: FP8 configuration
        device: Device for tensors

    Returns:
        Initialized FP8ScalingInfo
    """
    if config.scale_mode == "per_tensor":
        scale_shape = ()
    elif config.scale_mode == "per_block":
        # Assume first dimension is n_blocks
        scale_shape = (shape[0],)
    elif config.scale_mode == "per_row":
        scale_shape = (shape[0],)
    else:
        raise ValueError(f"Unknown scale_mode: {config.scale_mode}")

    # Initialize with unit scale
    scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
    scale_inv = torch.ones(scale_shape, dtype=torch.float32, device=device)
    amax = torch.zeros(scale_shape, dtype=torch.float32, device=device)

    # AMAX history
    if scale_shape == ():
        history_shape = (config.amax_history_len,)
    else:
        history_shape = scale_shape + (config.amax_history_len,)

    amax_history = torch.zeros(history_shape, dtype=torch.float32, device=device)

    return FP8ScalingInfo(
        scale=scale,
        scale_inv=scale_inv,
        amax=amax,
        amax_history=amax_history,
        config=config,
    )


# =============================================================================
# Quantization / Dequantization Functions
# =============================================================================

def compute_amax(
    tensor: torch.Tensor,
    scale_mode: str = "per_tensor",
) -> torch.Tensor:
    """Compute absolute maximum values for scaling.

    Args:
        tensor: Input tensor
        scale_mode: How to compute AMAX
            - "per_tensor": Single value for entire tensor
            - "per_block": Per first dimension (for VOID blocks)
            - "per_row": Per first dimension

    Returns:
        AMAX tensor with appropriate shape
    """
    if scale_mode == "per_tensor":
        return tensor.abs().max()
    elif scale_mode in ("per_block", "per_row"):
        # Flatten all but first dimension
        flat = tensor.view(tensor.shape[0], -1)
        return flat.abs().max(dim=1).values
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")


def quantize_to_fp8(
    tensor: torch.Tensor,
    config: Optional[FP8Config] = None,
    scaling_info: Optional[FP8ScalingInfo] = None,
) -> Tuple[torch.Tensor, FP8ScalingInfo]:
    """Quantize a tensor to FP8 format.

    Args:
        tensor: Input tensor (float16, bfloat16, or float32)
        config: FP8 configuration (creates new scaling_info if not provided)
        scaling_info: Existing scaling info to update (optional)

    Returns:
        Tuple of (FP8 tensor, updated scaling info)
    """
    if config is None:
        config = FP8Config()

    # Create or update scaling info
    if scaling_info is None:
        scaling_info = create_scaling_info(tensor.shape, config, tensor.device)

    # Compute current AMAX
    current_amax = compute_amax(tensor, config.scale_mode)
    scaling_info.update_amax(current_amax)

    # Compute scale factors
    scale, scale_inv = scaling_info.compute_scale()

    # Scale the tensor
    if config.scale_mode == "per_tensor":
        scaled = tensor * scale_inv
    else:
        # Broadcast scale along first dimension
        shape = (tensor.shape[0],) + (1,) * (tensor.dim() - 1)
        scaled = tensor * scale_inv.view(shape)

    # Clamp to FP8 range and convert
    fp8_max = config.max_value
    scaled = torch.clamp(scaled, -fp8_max, fp8_max)

    # Convert to FP8 dtype
    fp8_tensor = scaled.to(config.torch_dtype)

    return fp8_tensor, scaling_info


def dequantize_from_fp8(
    tensor: torch.Tensor,
    scaling_info: FP8ScalingInfo,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to higher precision.

    Args:
        tensor: FP8 tensor
        scaling_info: Scaling info from quantization
        output_dtype: Output dtype (float16, bfloat16, or float32)

    Returns:
        Dequantized tensor
    """
    # Convert to float first
    float_tensor = tensor.to(torch.float32)

    # Apply inverse scale
    if scaling_info.config.scale_mode == "per_tensor":
        result = float_tensor * scaling_info.scale
    else:
        shape = (tensor.shape[0],) + (1,) * (tensor.dim() - 1)
        result = float_tensor * scaling_info.scale.view(shape)

    return result.to(output_dtype)


# =============================================================================
# Triton Kernels for FP8 Operations
# =============================================================================

def get_triton_fp8_dtype(config: FP8Config):
    """Map FP8Config to Triton dtype."""
    if config.format == FP8Format.E4M3:
        return tl.float8e4nv
    else:
        return tl.float8e5


@triton.jit
def _quantize_fp8_kernel(
    # Input
    input_ptr,
    # Output
    output_ptr,
    # Scale
    scale_inv_ptr,
    # Dimensions
    n_elements,
    # Scale mode
    elements_per_scale,  # 0 for per-tensor
    # FP8 max
    fp8_max,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Fused quantize to FP8 kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Load scale (per-tensor or per-block)
    if elements_per_scale == 0:
        scale_inv = tl.load(scale_inv_ptr)
    else:
        scale_idx = offsets // elements_per_scale
        scale_inv = tl.load(scale_inv_ptr + scale_idx, mask=mask, other=1.0)

    # Scale and clamp
    scaled = x * scale_inv
    clamped = tl.minimum(tl.maximum(scaled, -fp8_max), fp8_max)

    # Store as FP8 (Triton handles the conversion)
    tl.store(output_ptr + offsets, clamped.to(tl.float8e4nv), mask=mask)


@triton.jit
def _dequantize_fp8_kernel(
    # Input (FP8)
    input_ptr,
    # Output
    output_ptr,
    # Scale
    scale_ptr,
    # Dimensions
    n_elements,
    # Scale mode
    elements_per_scale,  # 0 for per-tensor
    # Block size
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float16,
):
    """Fused dequantize from FP8 kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load FP8 input and convert to float32
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Load scale
    if elements_per_scale == 0:
        scale = tl.load(scale_ptr)
    else:
        scale_idx = offsets // elements_per_scale
        scale = tl.load(scale_ptr + scale_idx, mask=mask, other=1.0)

    # Dequantize
    result = x * scale

    # Store
    tl.store(output_ptr + offsets, result.to(OUTPUT_DTYPE), mask=mask)


def quantize_fp8_fused(
    tensor: torch.Tensor,
    scaling_info: FP8ScalingInfo,
) -> torch.Tensor:
    """Fused FP8 quantization using Triton kernel.

    Args:
        tensor: Input tensor
        scaling_info: Pre-computed scaling info

    Returns:
        FP8 quantized tensor
    """
    # Flatten for kernel
    flat = tensor.view(-1)
    n_elements = flat.numel()

    # Output tensor
    output = torch.empty(n_elements, dtype=scaling_info.config.torch_dtype, device=tensor.device)

    # Compute elements per scale
    if scaling_info.config.scale_mode == "per_tensor":
        elements_per_scale = 0
    else:
        elements_per_scale = n_elements // scaling_info.scale_inv.numel()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _quantize_fp8_kernel[grid](
        flat, output,
        scaling_info.scale_inv,
        n_elements, elements_per_scale,
        scaling_info.config.max_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(tensor.shape)


def dequantize_fp8_fused(
    tensor: torch.Tensor,
    scaling_info: FP8ScalingInfo,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Fused FP8 dequantization using Triton kernel.

    Args:
        tensor: FP8 input tensor
        scaling_info: Scaling info from quantization
        output_dtype: Output dtype

    Returns:
        Dequantized tensor
    """
    from .ops import get_triton_dtype

    flat = tensor.view(-1)
    n_elements = flat.numel()

    output = torch.empty(n_elements, dtype=output_dtype, device=tensor.device)

    if scaling_info.config.scale_mode == "per_tensor":
        elements_per_scale = 0
    else:
        elements_per_scale = n_elements // scaling_info.scale.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    triton_dtype = get_triton_dtype(output_dtype)

    _dequantize_fp8_kernel[grid](
        flat, output,
        scaling_info.scale,
        n_elements, elements_per_scale,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_DTYPE=triton_dtype,
    )

    return output.view(tensor.shape)


# =============================================================================
# FP8 VOID Tensor Extension
# =============================================================================

@dataclass
class FP8VOIDTensor:
    """VOID sparse tensor with FP8 storage.

    Wraps a VOIDTensor with FP8 quantization for reduced memory and faster compute.
    """
    # FP8 values [n_blocks, tile_m, tile_n] in FP8 format
    values_fp8: torch.Tensor

    # Scaling info
    scaling_info: FP8ScalingInfo

    # Block indices (same as VOIDTensor)
    block_rows: torch.Tensor
    block_cols: torch.Tensor
    morton_codes: torch.Tensor

    # Metadata
    shape: Tuple[int, int]
    tile_size: Tuple[int, int]
    nnz_original: int
    n_blocks: int
    density: float

    @classmethod
    def from_void_tensor(
        cls,
        void_tensor: 'VOIDTensor',
        config: Optional[FP8Config] = None,
    ) -> 'FP8VOIDTensor':
        """Convert a VOIDTensor to FP8 format.

        Args:
            void_tensor: Source VOIDTensor
            config: FP8 configuration

        Returns:
            FP8VOIDTensor with quantized values
        """
        if config is None:
            config = FP8Config(scale_mode="per_block")

        # Quantize values
        values_fp8, scaling_info = quantize_to_fp8(
            void_tensor.values,
            config=config,
        )

        return cls(
            values_fp8=values_fp8,
            scaling_info=scaling_info,
            block_rows=void_tensor.block_rows,
            block_cols=void_tensor.block_cols,
            morton_codes=void_tensor.morton_codes,
            shape=void_tensor.shape,
            tile_size=void_tensor.tile_size,
            nnz_original=void_tensor.nnz_original,
            n_blocks=void_tensor.n_blocks,
            density=void_tensor.density,
        )

    def to_void_tensor(
        self,
        output_dtype: torch.dtype = torch.float16,
    ) -> 'VOIDTensor':
        """Convert back to regular VOIDTensor.

        Args:
            output_dtype: Dtype for dequantized values

        Returns:
            VOIDTensor with dequantized values
        """
        from .format import VOIDTensor

        values = dequantize_from_fp8(
            self.values_fp8,
            self.scaling_info,
            output_dtype=output_dtype,
        )

        return VOIDTensor(
            values=values,
            block_rows=self.block_rows,
            block_cols=self.block_cols,
            morton_codes=self.morton_codes,
            shape=self.shape,
            tile_size=self.tile_size,
            nnz_original=self.nnz_original,
            n_blocks=self.n_blocks,
            density=self.density,
            dtype=output_dtype,
        )

    def to(self, device: Union[str, torch.device]) -> 'FP8VOIDTensor':
        """Move to device."""
        return FP8VOIDTensor(
            values_fp8=self.values_fp8.to(device),
            scaling_info=FP8ScalingInfo(
                scale=self.scaling_info.scale.to(device),
                scale_inv=self.scaling_info.scale_inv.to(device),
                amax=self.scaling_info.amax.to(device),
                amax_history=self.scaling_info.amax_history.to(device),
                config=self.scaling_info.config,
            ),
            block_rows=self.block_rows.to(device),
            block_cols=self.block_cols.to(device),
            morton_codes=self.morton_codes.to(device),
            shape=self.shape,
            tile_size=self.tile_size,
            nnz_original=self.nnz_original,
            n_blocks=self.n_blocks,
            density=self.density,
        )

    def cuda(self) -> 'FP8VOIDTensor':
        return self.to('cuda')

    @property
    def memory_bytes(self) -> int:
        """Memory footprint in bytes."""
        # FP8 values + scales + indices
        fp8_bytes = self.values_fp8.numel()  # 1 byte per FP8 element
        scale_bytes = self.scaling_info.scale.numel() * 4
        index_bytes = (self.block_rows.numel() + self.block_cols.numel()) * 4
        return fp8_bytes + scale_bytes + index_bytes

    def __repr__(self) -> str:
        return (
            f"FP8VOIDTensor(shape={self.shape}, tile_size={self.tile_size}, "
            f"n_blocks={self.n_blocks}, format={self.scaling_info.config.format.value}, "
            f"scale_mode={self.scaling_info.config.scale_mode})"
        )


# =============================================================================
# FP8 SpMM Kernel
# =============================================================================

@triton.jit
def void_spmm_fp8_kernel(
    # Sparse matrix A (FP8 format)
    a_values_ptr,      # [n_blocks, TILE_M, TILE_K] in FP8
    a_scales_ptr,      # [n_blocks] or [1] for per-tensor
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B
    b_ptr,             # [K, N]
    # Output matrix C
    c_ptr,             # [M, N]
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scale mode (0=per_tensor, 1=per_block)
    scale_mode,
    # Tile sizes
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    FP8 SpMM kernel: C = A @ B where A is FP8 VOID sparse.

    Uses FP32 accumulation for numerical stability.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= n_block_rows:
        return

    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)

    # FP32 accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    col_start = pid_n * TILE_N

    for block_idx in range(row_start, row_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load A tile (FP8) and convert to FP32
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile_fp8 = tl.load(a_tile_ptr)
        a_tile = a_tile_fp8.to(tl.float32)

        # Apply scale
        if scale_mode == 0:  # per-tensor
            scale = tl.load(a_scales_ptr)
        else:  # per-block
            scale = tl.load(a_scales_ptr + actual_idx)
        a_tile = a_tile * scale

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


def void_spmm_fp8(
    a: FP8VOIDTensor,
    b: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    FP8 Sparse-Dense Matrix Multiplication: C = A @ B

    Args:
        a: FP8VOIDTensor sparse matrix [M, K]
        b: Dense matrix [K, N] (any dtype, will be converted to FP32 internally)
        output_dtype: Output dtype (default: float16)

    Returns:
        Dense matrix C [M, N] in output_dtype
    """
    from .ops import get_triton_dtype

    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert b.is_cuda

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    # Output tensor
    out = torch.zeros(M, N, dtype=output_dtype, device=b.device)

    if a.n_blocks == 0:
        return out

    # Get row block info (need to compute from block_rows)
    n_block_rows = (M + tile_m - 1) // tile_m

    # Build row pointer
    row_counts = torch.zeros(n_block_rows, dtype=torch.int32, device=a.block_rows.device)
    row_counts.scatter_add_(0, a.block_rows.long(), torch.ones_like(a.block_rows))
    row_ptr = torch.zeros(n_block_rows + 1, dtype=torch.int32, device=a.block_rows.device)
    row_ptr[1:] = torch.cumsum(row_counts, dim=0)

    # Sort by row
    sort_keys = a.block_rows.long() * ((K + tile_k - 1) // tile_k) + a.block_cols.long()
    block_indices = torch.argsort(sort_keys).to(torch.int32)

    TILE_N = min(64, triton.next_power_of_2(N))
    triton_output_dtype = get_triton_dtype(output_dtype)

    # Determine scale mode
    scale_mode = 0 if a.scaling_info.config.scale_mode == "per_tensor" else 1

    grid = (n_block_rows, triton.cdiv(N, TILE_N))

    void_spmm_fp8_kernel[grid](
        a.values_fp8, a.scaling_info.scale,
        a.block_rows, a.block_cols, row_ptr, block_indices,
        b, out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        scale_mode,
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
        OUTPUT_DTYPE=triton_output_dtype,
    )

    return out


# =============================================================================
# Convenience Functions
# =============================================================================

def void_tensor_to_fp8(
    void_tensor: 'VOIDTensor',
    format: str = "e4m3",
    scale_mode: str = "per_block",
) -> FP8VOIDTensor:
    """Convert VOIDTensor to FP8 format.

    Args:
        void_tensor: Source tensor
        format: "e4m3" or "e5m2"
        scale_mode: "per_tensor" or "per_block"

    Returns:
        FP8VOIDTensor
    """
    fp8_format = FP8Format.E4M3 if format == "e4m3" else FP8Format.E5M2
    config = FP8Config(format=fp8_format, scale_mode=scale_mode)
    return FP8VOIDTensor.from_void_tensor(void_tensor, config)


# Export all public APIs
__all__ = [
    # Enums and configs
    "FP8Format",
    "FP8Config",
    "FP8ScalingInfo",
    # Tensor class
    "FP8VOIDTensor",
    # Functions
    "create_scaling_info",
    "compute_amax",
    "quantize_to_fp8",
    "dequantize_from_fp8",
    "quantize_fp8_fused",
    "dequantize_fp8_fused",
    "void_spmm_fp8",
    "void_tensor_to_fp8",
]
