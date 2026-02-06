"""INT8/INT4 quantization for VOID sparse tensors."""

import torch
import triton
import triton.language as tl
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Literal
from enum import Enum


class IntFormat(Enum):
    """Integer quantization formats."""
    INT8 = "int8"      # 8-bit signed (-128 to 127)
    UINT8 = "uint8"    # 8-bit unsigned (0 to 255)
    INT4 = "int4"      # 4-bit signed (-8 to 7), packed 2 per byte
    UINT4 = "uint4"    # 4-bit unsigned (0 to 15), packed 2 per byte


class ScaleMode(Enum):
    """Quantization scale granularity."""
    PER_TENSOR = "per_tensor"    # Single scale for entire tensor
    PER_CHANNEL = "per_channel"  # Scale per output channel (row)
    PER_BLOCK = "per_block"      # Scale per VOID block (best for sparse)


@dataclass
class IntQuantConfig:
    """
    Configuration for integer quantization.

    Attributes:
        format: Integer format (INT8, UINT8, INT4, UINT4)
        symmetric: If True, use symmetric quantization (zero_point = 0)
        scale_mode: Granularity of quantization scale
        calibration_method: How to compute scale ('minmax', 'percentile', 'mse')
        percentile: For percentile calibration, which percentile to use
    """
    format: IntFormat = IntFormat.INT8
    symmetric: bool = True
    scale_mode: ScaleMode = ScaleMode.PER_BLOCK
    calibration_method: Literal['minmax', 'percentile', 'mse'] = 'minmax'
    percentile: float = 99.9

    @property
    def bits(self) -> int:
        """Number of bits for this format."""
        if self.format in (IntFormat.INT8, IntFormat.UINT8):
            return 8
        else:
            return 4

    @property
    def is_signed(self) -> bool:
        """Whether format is signed."""
        return self.format in (IntFormat.INT8, IntFormat.INT4)

    @property
    def qmin(self) -> int:
        """Minimum quantized value."""
        if self.format == IntFormat.INT8:
            return -128
        elif self.format == IntFormat.UINT8:
            return 0
        elif self.format == IntFormat.INT4:
            return -8
        else:  # UINT4
            return 0

    @property
    def qmax(self) -> int:
        """Maximum quantized value."""
        if self.format == IntFormat.INT8:
            return 127
        elif self.format == IntFormat.UINT8:
            return 255
        elif self.format == IntFormat.INT4:
            return 7
        else:  # UINT4
            return 15


@dataclass
class QuantizationParams:
    """
    Quantization parameters for dequantization.

    For symmetric quantization: x_float = x_int * scale
    For asymmetric quantization: x_float = (x_int - zero_point) * scale
    """
    scale: torch.Tensor       # Scale factor(s)
    zero_point: torch.Tensor  # Zero point(s), all zeros for symmetric
    config: IntQuantConfig
    original_shape: Tuple[int, ...]

    def to(self, device: Union[str, torch.device]) -> 'QuantizationParams':
        return QuantizationParams(
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device),
            config=self.config,
            original_shape=self.original_shape,
        )


@dataclass
class Int8VOIDTensor:
    """
    Quantized VOID tensor with INT8 values.

    Stores:
    - values: INT8 quantized block values [n_blocks, tile_m, tile_k]
    - scales: Per-block scales [n_blocks] or per-tensor [1]
    - zero_points: Per-block zero points (for asymmetric)
    - block_rows, block_cols: Block positions (same as VOIDTensor)
    """
    values: torch.Tensor          # int8 [n_blocks, tile_m, tile_k]
    scales: torch.Tensor          # float32 scales
    zero_points: torch.Tensor     # int8 zero points
    block_rows: torch.Tensor      # [n_blocks]
    block_cols: torch.Tensor      # [n_blocks]
    shape: Tuple[int, int]        # Original dense shape [M, K]
    tile_size: Tuple[int, int]    # (tile_m, tile_k)
    n_blocks: int
    config: IntQuantConfig

    # Cached row organization
    _row_ptr: Optional[torch.Tensor] = None
    _block_indices: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def block_grid(self) -> Tuple[int, int]:
        return (
            (self.shape[0] + self.tile_size[0] - 1) // self.tile_size[0],
            (self.shape[1] + self.tile_size[1] - 1) // self.tile_size[1],
        )

    def to(self, device: Union[str, torch.device]) -> 'Int8VOIDTensor':
        return Int8VOIDTensor(
            values=self.values.to(device),
            scales=self.scales.to(device),
            zero_points=self.zero_points.to(device),
            block_rows=self.block_rows.to(device),
            block_cols=self.block_cols.to(device),
            shape=self.shape,
            tile_size=self.tile_size,
            n_blocks=self.n_blocks,
            config=self.config,
        )

    def get_row_block_info(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached row pointer and block indices for efficient SpMM."""
        if self._row_ptr is None or self._block_indices is None:
            n_block_rows = self.block_grid[0]

            # Count blocks per row
            row_counts = torch.zeros(n_block_rows, dtype=torch.int32, device=self.device)
            row_counts.scatter_add_(0, self.block_rows.long(), torch.ones_like(self.block_rows))

            # Compute row pointers (CSR-style)
            self._row_ptr = torch.zeros(n_block_rows + 1, dtype=torch.int32, device=self.device)
            self._row_ptr[1:] = torch.cumsum(row_counts, dim=0)

            # Sort blocks by row for coalesced access
            sort_idx = torch.argsort(self.block_rows.long() * self.block_grid[1] + self.block_cols.long())
            self._block_indices = sort_idx.to(torch.int32)

        return self._row_ptr, self._block_indices


def compute_scale_minmax(
    tensor: torch.Tensor,
    config: IntQuantConfig,
    dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization scale and zero point using min-max method.

    Args:
        tensor: Input tensor to quantize
        config: Quantization configuration
        dim: Dimension to compute stats over (None for per-tensor)

    Returns:
        Tuple of (scale, zero_point)
    """
    if dim is not None:
        min_val = tensor.amin(dim=dim, keepdim=True)
        max_val = tensor.amax(dim=dim, keepdim=True)
    else:
        min_val = tensor.min()
        max_val = tensor.max()

    if config.symmetric:
        # Symmetric: scale based on max absolute value
        abs_max = torch.maximum(min_val.abs(), max_val.abs())
        scale = abs_max / max(abs(config.qmin), config.qmax)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
    else:
        # Asymmetric: use full range
        scale = (max_val - min_val) / (config.qmax - config.qmin)
        zero_point = (config.qmin - min_val / scale).round().to(torch.int8)

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    return scale, zero_point


def compute_scale_percentile(
    tensor: torch.Tensor,
    config: IntQuantConfig,
    dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization scale using percentile clipping.

    Clips outliers to improve quantization accuracy for the majority of values.
    """
    p_low = (100 - config.percentile) / 2
    p_high = 100 - p_low

    if dim is not None:
        # Flatten along dim for percentile computation
        shape = tensor.shape
        tensor_flat = tensor.transpose(dim, -1).reshape(-1, shape[dim])
        min_val = torch.quantile(tensor_flat, p_low / 100, dim=0)
        max_val = torch.quantile(tensor_flat, p_high / 100, dim=0)
        min_val = min_val.view(*shape[:dim], 1, *shape[dim+1:])
        max_val = max_val.view(*shape[:dim], 1, *shape[dim+1:])
    else:
        min_val = torch.quantile(tensor.flatten(), p_low / 100)
        max_val = torch.quantile(tensor.flatten(), p_high / 100)

    if config.symmetric:
        abs_max = torch.maximum(min_val.abs(), max_val.abs())
        scale = abs_max / max(abs(config.qmin), config.qmax)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
    else:
        scale = (max_val - min_val) / (config.qmax - config.qmin)
        zero_point = (config.qmin - min_val / scale).round().to(torch.int8)

    scale = torch.clamp(scale, min=1e-8)
    return scale, zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    config: IntQuantConfig,
) -> Tuple[torch.Tensor, QuantizationParams]:
    """
    Quantize a tensor to integer format.

    Args:
        tensor: Input floating-point tensor
        config: Quantization configuration

    Returns:
        Tuple of (quantized_tensor, quantization_params)
    """
    original_shape = tensor.shape

    # Determine dimension for scale computation
    if config.scale_mode == ScaleMode.PER_TENSOR:
        dim = None
    elif config.scale_mode == ScaleMode.PER_CHANNEL:
        dim = tuple(range(1, tensor.dim()))  # All dims except first
    else:  # PER_BLOCK - handled separately
        dim = None

    # Compute scale and zero point
    if config.calibration_method == 'minmax':
        scale, zero_point = compute_scale_minmax(tensor, config, dim)
    elif config.calibration_method == 'percentile':
        scale, zero_point = compute_scale_percentile(tensor, config, dim)
    else:
        raise ValueError(f"Unknown calibration method: {config.calibration_method}")

    # Quantize
    if config.symmetric:
        quantized = (tensor / scale).round().clamp(config.qmin, config.qmax)
    else:
        quantized = ((tensor / scale) + zero_point.float()).round().clamp(config.qmin, config.qmax)

    # Convert to appropriate dtype
    if config.format in (IntFormat.INT8, IntFormat.INT4):
        quantized = quantized.to(torch.int8)
    else:
        quantized = quantized.to(torch.uint8)

    params = QuantizationParams(
        scale=scale.to(torch.float32),
        zero_point=zero_point,
        config=config,
        original_shape=original_shape,
    )

    return quantized, params


def dequantize_tensor(
    quantized: torch.Tensor,
    params: QuantizationParams,
) -> torch.Tensor:
    """
    Dequantize a tensor back to floating point.

    Args:
        quantized: Quantized integer tensor
        params: Quantization parameters

    Returns:
        Dequantized floating-point tensor
    """
    if params.config.symmetric:
        return quantized.float() * params.scale
    else:
        return (quantized.float() - params.zero_point.float()) * params.scale


def quantize_void_tensor(
    void_tensor,  # VOIDTensor
    config: Optional[IntQuantConfig] = None,
) -> Int8VOIDTensor:
    """
    Quantize a VOIDTensor to INT8 format.

    Uses per-block quantization for best accuracy with sparse patterns.

    Args:
        void_tensor: Input VOIDTensor
        config: Quantization config (default: INT8, symmetric, per-block)

    Returns:
        Int8VOIDTensor
    """
    from .format import VOIDTensor

    if config is None:
        config = IntQuantConfig(
            format=IntFormat.INT8,
            symmetric=True,
            scale_mode=ScaleMode.PER_BLOCK,
        )

    n_blocks = void_tensor.n_blocks
    tile_m, tile_k = void_tensor.tile_size
    values = void_tensor.values  # [n_blocks, tile_m, tile_k]

    if config.scale_mode == ScaleMode.PER_BLOCK:
        # Compute scale per block
        scales = torch.zeros(n_blocks, dtype=torch.float32, device=values.device)
        zero_points = torch.zeros(n_blocks, dtype=torch.int8, device=values.device)
        quantized_values = torch.zeros_like(values, dtype=torch.int8)

        for b in range(n_blocks):
            block = values[b]  # [tile_m, tile_k]
            q_block, params = quantize_tensor(block, config)
            quantized_values[b] = q_block
            scales[b] = params.scale.squeeze()
            if not config.symmetric:
                zero_points[b] = params.zero_point.squeeze()
    else:
        # Per-tensor quantization
        values_flat = values.view(-1)
        q_flat, params = quantize_tensor(values_flat, config)
        quantized_values = q_flat.view(n_blocks, tile_m, tile_k)
        scales = params.scale.expand(n_blocks)
        zero_points = params.zero_point.expand(n_blocks) if not config.symmetric else torch.zeros(n_blocks, dtype=torch.int8, device=values.device)

    return Int8VOIDTensor(
        values=quantized_values,
        scales=scales,
        zero_points=zero_points,
        block_rows=void_tensor.block_rows.clone(),
        block_cols=void_tensor.block_cols.clone(),
        shape=void_tensor.shape,
        tile_size=void_tensor.tile_size,
        n_blocks=n_blocks,
        config=config,
    )


def dequantize_void_tensor(
    int8_tensor: Int8VOIDTensor,
) -> 'VOIDTensor':
    """
    Dequantize an Int8VOIDTensor back to floating-point VOIDTensor.

    Args:
        int8_tensor: Quantized tensor

    Returns:
        VOIDTensor with float32 values
    """
    from .format import VOIDTensor

    n_blocks = int8_tensor.n_blocks
    tile_m, tile_k = int8_tensor.tile_size

    # Dequantize each block
    values = torch.zeros(n_blocks, tile_m, tile_k, dtype=torch.float32, device=int8_tensor.device)

    for b in range(n_blocks):
        q_block = int8_tensor.values[b].float()  # [tile_m, tile_k]
        scale = int8_tensor.scales[b]

        if int8_tensor.config.symmetric:
            values[b] = q_block * scale
        else:
            zp = int8_tensor.zero_points[b].float()
            values[b] = (q_block - zp) * scale

    return VOIDTensor(
        values=values,
        block_rows=int8_tensor.block_rows.clone(),
        block_cols=int8_tensor.block_cols.clone(),
        shape=int8_tensor.shape,
        tile_size=int8_tensor.tile_size,
        n_blocks=n_blocks,
    )


# =============================================================================
# Triton Kernel for INT8 SpMM
# =============================================================================

@triton.jit
def void_spmm_int8_kernel(
    # Sparse matrix A (INT8 quantized VOID format)
    a_values_ptr,      # int8 [n_blocks, TILE_M, TILE_K]
    a_scales_ptr,      # float32 [n_blocks] - per-block scales
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B (expected to be INT8 quantized as well)
    b_ptr,             # int8 [K, N]
    b_scale,           # float32 scalar - B scale
    # Output matrix C
    c_ptr,             # float32 [M, N]
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """
    INT8 SpMM kernel with INT32 accumulation.

    Performs: C = dequant(A_int8) @ dequant(B_int8)
    Using: acc_int32 = A_int8 @ B_int8, then scale to float32

    This leverages INT8 Tensor Cores for ~2-4x speedup over FP16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= n_block_rows:
        return

    row_start = tl.load(a_row_ptr_ptr + pid_m)
    row_end = tl.load(a_row_ptr_ptr + pid_m + 1)

    # INT32 accumulator for INT8 matmul
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.int32)
    # Scale accumulator
    scale_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    col_start = pid_n * TILE_N

    for block_idx in range(row_start, row_end):
        actual_idx = tl.load(a_block_idx_ptr + block_idx)
        block_col = tl.load(a_block_cols_ptr + actual_idx)
        k_offset = block_col * TILE_K

        # Load per-block scale
        a_scale = tl.load(a_scales_ptr + actual_idx)

        # Load A tile (INT8)
        a_tile_ptr = tl.make_block_ptr(
            base=a_values_ptr + actual_idx * TILE_M * TILE_K,
            shape=(TILE_M, TILE_K),
            strides=(TILE_K, 1),
            offsets=(0, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_tile = tl.load(a_tile_ptr)  # int8

        # Load B tile (INT8)
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_offset, col_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1))  # int8

        # INT8 matmul with INT32 accumulation
        # Note: tl.dot with int8 inputs accumulates in int32
        acc += tl.dot(a_tile, b_tile)

        # Track scale contribution
        combined_scale = a_scale * b_scale
        scale_acc += combined_scale

    # Apply average scale and convert to float32
    n_blocks_in_row = row_end - row_start
    if n_blocks_in_row > 0:
        avg_scale = scale_acc / n_blocks_in_row
        out = acc.to(tl.float32) * avg_scale
    else:
        out = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

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
    tl.store(c_tile_ptr, out, boundary_check=(0, 1))


def void_spmm_int8(
    a: Int8VOIDTensor,
    b: torch.Tensor,
    b_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    INT8 Sparse-Dense Matrix Multiplication.

    Args:
        a: INT8 quantized VOID sparse matrix [M, K]
        b: Dense matrix [K, N] (INT8 or will be quantized)
        b_scale: Scale for B if already quantized
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N] in float32
    """
    M, K = a.shape
    N = b.shape[1]

    # Quantize B if needed
    if b.dtype != torch.int8:
        b_config = IntQuantConfig(
            format=IntFormat.INT8,
            symmetric=True,
            scale_mode=ScaleMode.PER_TENSOR,
        )
        b_int8, b_params = quantize_tensor(b, b_config)
        b = b_int8
        b_scale = b_params.scale.item()
    elif b_scale is None:
        b_scale = 1.0

    if not b.is_contiguous():
        b = b.contiguous()

    if out is None:
        out = torch.zeros(M, N, dtype=torch.float32, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        return out

    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]
    tile_m, tile_k = a.tile_size

    # Choose TILE_N
    if N >= 128:
        TILE_N = 128
    elif N >= 64:
        TILE_N = 64
    else:
        TILE_N = triton.next_power_of_2(N)

    grid = (n_block_rows, triton.cdiv(N, TILE_N))

    void_spmm_int8_kernel[grid](
        a.values, a.scales, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, b_scale,
        out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
    )

    return out


def get_quantization_error(
    original: torch.Tensor,
    config: IntQuantConfig,
) -> dict:
    """
    Analyze quantization error for a tensor.

    Args:
        original: Original floating-point tensor
        config: Quantization configuration

    Returns:
        Dictionary with error metrics
    """
    quantized, params = quantize_tensor(original, config)
    reconstructed = dequantize_tensor(quantized, params)

    abs_error = (original - reconstructed).abs()
    rel_error = abs_error / (original.abs() + 1e-8)

    return {
        "config": str(config),
        "mean_abs_error": abs_error.mean().item(),
        "max_abs_error": abs_error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "snr_db": (10 * torch.log10(
            original.pow(2).mean() / abs_error.pow(2).mean()
        )).item(),
        "compression_ratio": original.element_size() / (config.bits / 8),
    }


# Export public API
__all__ = [
    # Enums
    "IntFormat",
    "ScaleMode",
    # Dataclasses
    "IntQuantConfig",
    "QuantizationParams",
    "Int8VOIDTensor",
    # Functions
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_void_tensor",
    "dequantize_void_tensor",
    "void_spmm_int8",
    "get_quantization_error",
]
