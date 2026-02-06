"""Fused SpMM + activation kernels (GELU, ReLU, SiLU)."""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Callable
from enum import Enum

from .format import VOIDTensor
from .ops import get_triton_dtype, compute_optimal_tile_n


class FusedOpType(Enum):
    """Types of fused operations supported."""
    SPMM_GELU = "spmm_gelu"           # SpMM followed by GELU activation
    SPMM_RELU = "spmm_relu"           # SpMM followed by ReLU activation
    SPMM_SILU = "spmm_silu"           # SpMM followed by SiLU/Swish activation
    SPMM_BIAS_GELU = "spmm_bias_gelu" # SpMM + bias + GELU
    SPMM_BIAS_RELU = "spmm_bias_relu" # SpMM + bias + ReLU
    SPARSE_MLP = "sparse_mlp"         # Full MLP: SpMM + act + SpMM


class ActivationType(Enum):
    """Activation functions for fusion."""
    NONE = "none"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"       # Also known as Swish
    TANH = "tanh"
    SIGMOID = "sigmoid"


# =============================================================================
# Activation Function Implementations for Triton
# =============================================================================

@triton.jit
def _gelu(x):
    """GELU activation using tanh approximation."""
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    tanh_approx = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    return x * 0.5 * (1.0 + tanh_approx)


@triton.jit
def _relu(x):
    """ReLU activation: max(0, x)"""
    return tl.maximum(x, 0.0)


@triton.jit
def _silu(x):
    """SiLU/Swish activation: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.jit
def _apply_activation(x, ACTIVATION: tl.constexpr):
    """Apply activation function based on constexpr flag."""
    if ACTIVATION == 0:  # None
        return x
    elif ACTIVATION == 1:  # ReLU
        return _relu(x)
    elif ACTIVATION == 2:  # GELU
        return _gelu(x)
    elif ACTIVATION == 3:  # SiLU
        return _silu(x)
    elif ACTIVATION == 4:  # Tanh
        return tl.libdevice.tanh(x)
    elif ACTIVATION == 5:  # Sigmoid
        return tl.sigmoid(x)
    else:
        return x


def get_activation_code(activation: Union[str, ActivationType]) -> int:
    """Convert activation name to integer code for kernel."""
    if isinstance(activation, ActivationType):
        activation = activation.value

    codes = {
        "none": 0,
        "relu": 1,
        "gelu": 2,
        "silu": 3,
        "swish": 3,  # Alias for SiLU
        "tanh": 4,
        "sigmoid": 5,
    }
    return codes.get(activation.lower(), 0)


# =============================================================================
# Fused SpMM + Activation Kernels
# =============================================================================

@triton.jit
def void_spmm_activation_kernel(
    # Sparse matrix A (VOID format)
    a_values_ptr,
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B
    b_ptr,
    # Optional bias
    bias_ptr,
    # Output matrix C
    c_ptr,
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Flags
    has_bias,
    # Tile sizes
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=gelu, 3=silu
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Fused SpMM + optional bias + activation kernel.

    C = activation(A @ B + bias)

    Grid: (n_block_rows, cdiv(N, TILE_N))
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

    # SpMM computation
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

        acc += tl.dot(a_tile, b_tile)

    # Add bias if present (broadcast along M dimension)
    if has_bias:
        bias_offsets = col_start + tl.arange(0, TILE_N)
        bias_mask = bias_offsets < N
        bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # Apply activation (fused - no memory round-trip)
    acc = _apply_activation(acc, ACTIVATION)

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


@triton.jit
def void_spmm_activation_row_bias_kernel(
    # Sparse matrix A (VOID format)
    a_values_ptr,
    a_block_rows_ptr,
    a_block_cols_ptr,
    a_row_ptr_ptr,
    a_block_idx_ptr,
    # Dense matrix B
    b_ptr,
    # Optional bias (per-row, i.e., per output feature)
    bias_ptr,
    # Output matrix C
    c_ptr,
    # Dimensions
    M, N, K,
    n_blocks,
    n_block_rows,
    # Strides
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Flags
    has_bias,
    # Tile sizes
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=gelu, 3=silu
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Fused SpMM + row-wise bias + activation kernel.

    C = activation(A @ B + bias[:, None])

    This kernel applies bias per-row (M dimension), which is correct for
    linear layer semantics: y = Wx + b where b has shape [out_features].

    Grid: (n_block_rows, cdiv(N, TILE_N))
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
    out_row = pid_m * TILE_M

    # SpMM computation
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

        acc += tl.dot(a_tile, b_tile)

    # Add bias if present (broadcast along N dimension - row-wise bias)
    if has_bias:
        bias_offsets = out_row + tl.arange(0, TILE_M)
        bias_mask = bias_offsets < M
        bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0).to(tl.float32)
        acc = acc + bias[:, None]  # [TILE_M, 1] broadcast to [TILE_M, TILE_N]

    # Apply activation (fused - no memory round-trip)
    acc = _apply_activation(acc, ACTIVATION)

    # Store output
    c_tile_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(out_row, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    tl.store(c_tile_ptr, acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def _void_spmm_activation_row_bias(
    a: VOIDTensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    activation: str,
) -> torch.Tensor:
    """
    Internal implementation for fused SpMM + row-wise bias + activation.

    This version applies bias per-row (M dimension), suitable for linear layers.
    """
    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert b.is_cuda and a.values.is_cuda

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        if bias is not None:
            out[:, :] = bias[:, None]  # Row-wise bias
            # Apply activation to bias
            if activation == "relu":
                out = torch.relu(out)
            elif activation == "gelu":
                out = torch.nn.functional.gelu(out)
            elif activation == "silu":
                out = torch.nn.functional.silu(out)
        return out

    # Prepare bias (row-wise: shape [M])
    if bias is not None:
        assert bias.shape == (M,), f"Row-wise bias must have shape ({M},), got {bias.shape}"
        if not bias.is_contiguous():
            bias = bias.contiguous()
        has_bias = True
    else:
        bias = torch.empty(1, device=b.device)  # Dummy
        has_bias = False

    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]

    TILE_N = compute_optimal_tile_n(N)
    output_dtype = get_triton_dtype(b.dtype)
    activation_code = get_activation_code(activation)

    grid = (n_block_rows, triton.cdiv(N, TILE_N))

    void_spmm_activation_row_bias_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, bias, out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        has_bias,
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
        ACTIVATION=activation_code,
        OUTPUT_DTYPE=output_dtype,
    )

    return out


def void_spmm_gelu(
    a: VOIDTensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused SpMM + GELU: C = GELU(A @ B + bias)

    Args:
        a: VOID sparse matrix [M, K]
        b: Dense matrix [K, N]
        bias: Optional bias vector [N]
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N]
    """
    return _void_spmm_activation(a, b, bias, out, activation="gelu")


def void_spmm_relu(
    a: VOIDTensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused SpMM + ReLU: C = ReLU(A @ B + bias)

    Args:
        a: VOID sparse matrix [M, K]
        b: Dense matrix [K, N]
        bias: Optional bias vector [N]
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N]
    """
    return _void_spmm_activation(a, b, bias, out, activation="relu")


def void_spmm_silu(
    a: VOIDTensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused SpMM + SiLU: C = SiLU(A @ B + bias)

    Args:
        a: VOID sparse matrix [M, K]
        b: Dense matrix [K, N]
        bias: Optional bias vector [N]
        out: Optional output buffer [M, N]

    Returns:
        Dense matrix C [M, N]
    """
    return _void_spmm_activation(a, b, bias, out, activation="silu")


def _void_spmm_activation(
    a: VOIDTensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    activation: str,
) -> torch.Tensor:
    """Internal implementation for fused SpMM + activation."""
    assert b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert b.is_cuda and a.values.is_cuda

    M, K = a.shape
    _, N = b.shape
    tile_m, tile_k = a.tile_size

    if not b.is_contiguous():
        b = b.contiguous()

    if out is None:
        out = torch.zeros(M, N, dtype=b.dtype, device=b.device)
    else:
        out.zero_()

    if a.n_blocks == 0:
        if bias is not None:
            out[:, :] = bias[None, :]
            # Apply activation to bias
            if activation == "relu":
                out = torch.relu(out)
            elif activation == "gelu":
                out = torch.nn.functional.gelu(out)
            elif activation == "silu":
                out = torch.nn.functional.silu(out)
        return out

    # Prepare bias
    if bias is not None:
        assert bias.shape == (N,)
        if not bias.is_contiguous():
            bias = bias.contiguous()
        has_bias = True
    else:
        bias = torch.empty(1, device=b.device)  # Dummy
        has_bias = False

    row_ptr, block_indices = a.get_row_block_info()
    n_block_rows = a.block_grid[0]

    TILE_N = compute_optimal_tile_n(N)
    output_dtype = get_triton_dtype(b.dtype)
    activation_code = get_activation_code(activation)

    grid = (n_block_rows, triton.cdiv(N, TILE_N))

    void_spmm_activation_kernel[grid](
        a.values, a.block_rows, a.block_cols, row_ptr, block_indices,
        b, bias, out,
        M, N, K, a.n_blocks, n_block_rows,
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        has_bias,
        TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
        ACTIVATION=activation_code,
        OUTPUT_DTYPE=output_dtype,
    )

    return out


# =============================================================================
# Fused Sparse MLP
# =============================================================================

@dataclass
class FusedSparseOp:
    """Configuration for a fused sparse operation.

    Can represent complex fused patterns like MLP layers.
    """
    op_type: FusedOpType
    sparse_tensors: List[VOIDTensor]
    activation: Optional[str] = "gelu"
    biases: Optional[List[torch.Tensor]] = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the fused operation."""
        if self.op_type == FusedOpType.SPMM_GELU:
            bias = self.biases[0] if self.biases else None
            return void_spmm_gelu(self.sparse_tensors[0], x, bias)

        elif self.op_type == FusedOpType.SPMM_RELU:
            bias = self.biases[0] if self.biases else None
            return void_spmm_relu(self.sparse_tensors[0], x, bias)

        elif self.op_type == FusedOpType.SPMM_SILU:
            bias = self.biases[0] if self.biases else None
            return void_spmm_silu(self.sparse_tensors[0], x, bias)

        elif self.op_type == FusedOpType.SPARSE_MLP:
            return fused_sparse_mlp(
                x,
                self.sparse_tensors[0],
                self.sparse_tensors[1],
                activation=self.activation,
                bias1=self.biases[0] if self.biases and len(self.biases) > 0 else None,
                bias2=self.biases[1] if self.biases and len(self.biases) > 1 else None,
            )

        else:
            raise ValueError(f"Unknown fused op type: {self.op_type}")


def fused_sparse_mlp(
    x: torch.Tensor,
    w1: VOIDTensor,
    w2: VOIDTensor,
    activation: str = "gelu",
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused sparse MLP: y = W2 @ activation(W1 @ x + b1) + b2

    Minimizes memory traffic by fusing SpMM + bias + activation in single kernel.

    Args:
        x: Input tensor [*, in_features]
        w1: First sparse weight [hidden, in_features]
        w2: Second sparse weight [out_features, hidden]
        activation: Activation function name
        bias1: Optional first bias [hidden]
        bias2: Optional second bias [out_features]

    Returns:
        Output tensor [*, out_features]
    """
    # Handle batched input
    input_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])  # [batch, in_features]

    # First layer: FUSED SpMM + bias + activation in single kernel
    # W1 @ x^T => [hidden, in] @ [in, batch] => [hidden, batch]
    hidden = _void_spmm_activation_row_bias(
        w1, x_2d.T, bias1, None, activation
    )  # [hidden, batch]

    # Second layer: FUSED SpMM + bias (no activation, or use "none")
    # W2 @ hidden => [out, hidden] @ [hidden, batch] => [out, batch]
    out = _void_spmm_activation_row_bias(
        w2, hidden, bias2, None, "none"
    )  # [out_features, batch]

    # Transpose to [batch, out_features]
    out = out.T

    # Restore shape
    output_shape = input_shape[:-1] + (out.shape[-1],)
    return out.view(output_shape)


# =============================================================================
# Fused Attention Operations
# =============================================================================

@triton.jit
def _sparse_attention_fused_softmax_kernel(
    Q, K, V, Out,
    # Scale factors for Q and K (for FP8 or quantized attention)
    q_scale, k_scale,
    # Mask info
    block_rows_ptr, block_cols_ptr,
    block_offsets_ptr,
    # Dimensions
    seq_len, head_dim, n_blocks, n_heads,
    # Strides
    stride_batch, stride_head, stride_seq, stride_dim,
    # Softmax scale
    sm_scale,
    # Options
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_SCALE: tl.constexpr,  # Whether to use q_scale/k_scale
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Sparse attention with fused input scaling (for FP8 inputs).

    When USE_SCALE is True, applies: softmax(q_scale * Q @ (k_scale * K)^T / sqrt(d))
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    batch_idx = pid_bh // n_heads
    head_idx = pid_bh % n_heads

    base_offset = batch_idx * stride_batch + head_idx * stride_head

    block_start = tl.load(block_offsets_ptr + pid_q)
    block_end = tl.load(block_offsets_ptr + pid_q + 1)

    m_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)

    q_seq_start = pid_q * BLOCK_SIZE
    q_ptrs = Q + base_offset + (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
    q_mask = (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Apply Q scale if using scaled inputs
    if USE_SCALE:
        q = q * q_scale

    for block_idx in range(block_start, block_end):
        k_block = tl.load(block_cols_ptr + block_idx)
        k_seq_start = k_block * BLOCK_SIZE

        k_ptrs = K + base_offset + (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
        k_mask = (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Apply K scale if using scaled inputs
        if USE_SCALE:
            k = k * k_scale

        v_ptrs = V + base_offset + (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        q_pos = q_seq_start + tl.arange(0, BLOCK_SIZE)
        k_pos = k_seq_start + tl.arange(0, BLOCK_SIZE)
        qk_mask = (q_pos[:, None] < seq_len) & (k_pos[None, :] < seq_len)
        qk = tl.where(qk_mask, qk, float('-inf'))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_ij[:, None]) * beta[:, None], axis=1)

        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_new

    out = acc / l_i[:, None]

    o_ptrs = Out + base_offset + (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
    o_mask = (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
    tl.store(o_ptrs, out.to(OUTPUT_DTYPE), mask=o_mask)


# =============================================================================
# Helper: Create Fused Operations
# =============================================================================

def create_fused_spmm_gelu(
    weight: VOIDTensor,
    bias: Optional[torch.Tensor] = None,
) -> FusedSparseOp:
    """Create a fused SpMM + GELU operation."""
    return FusedSparseOp(
        op_type=FusedOpType.SPMM_GELU,
        sparse_tensors=[weight],
        activation="gelu",
        biases=[bias] if bias is not None else None,
    )


def create_fused_spmm_relu(
    weight: VOIDTensor,
    bias: Optional[torch.Tensor] = None,
) -> FusedSparseOp:
    """Create a fused SpMM + ReLU operation."""
    return FusedSparseOp(
        op_type=FusedOpType.SPMM_RELU,
        sparse_tensors=[weight],
        activation="relu",
        biases=[bias] if bias is not None else None,
    )


def create_sparse_mlp(
    w1: VOIDTensor,
    w2: VOIDTensor,
    activation: str = "gelu",
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
) -> FusedSparseOp:
    """Create a fused sparse MLP operation."""
    biases = []
    if bias1 is not None:
        biases.append(bias1)
    if bias2 is not None:
        biases.append(bias2)

    return FusedSparseOp(
        op_type=FusedOpType.SPARSE_MLP,
        sparse_tensors=[w1, w2],
        activation=activation,
        biases=biases if biases else None,
    )


# =============================================================================
# Module Wrapper for Fused Operations
# =============================================================================

class FusedSparseLinear(torch.nn.Module):
    """
    Sparse linear layer with fused activation.

    Equivalent to: y = activation(W @ x + b)
    But computed in a single fused kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        void_tensor: VOIDTensor,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()

        assert void_tensor.shape == (out_features, in_features)

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Store VOID tensor components
        self.register_buffer('values', void_tensor.values.clone())
        self.register_buffer('block_rows', void_tensor.block_rows)
        self.register_buffer('block_cols', void_tensor.block_cols)
        self.register_buffer('morton_codes', void_tensor.morton_codes)

        self.shape = void_tensor.shape
        self.tile_size = void_tensor.tile_size
        self.n_blocks = void_tensor.n_blocks

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def get_void_tensor(self) -> VOIDTensor:
        """Reconstruct VOIDTensor from module state."""
        return VOIDTensor(
            values=self.values,
            block_rows=self.block_rows,
            block_cols=self.block_cols,
            morton_codes=self.morton_codes,
            shape=self.shape,
            tile_size=self.tile_size,
            nnz_original=0,
            n_blocks=self.n_blocks,
            density=0.0,
            dtype=self.values.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused SpMM + bias + activation in single kernel."""
        input_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        void_tensor = self.get_void_tensor()

        # FUSED: SpMM + bias + activation in single kernel pass
        # W @ x^T -> [out, batch] with row-wise bias
        y = _void_spmm_activation_row_bias(
            void_tensor,
            x_flat.T,
            self.bias,
            None,
            self.activation,
        )  # [out_features, batch]

        y = y.T  # [batch, out]

        output_shape = input_shape[:-1] + (self.out_features,)
        return y.view(output_shape)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'activation={self.activation}, bias={self.bias is not None}'
        )


# Export public API
__all__ = [
    # Enums
    "FusedOpType",
    "ActivationType",
    # Dataclasses
    "FusedSparseOp",
    # Fused kernels
    "void_spmm_gelu",
    "void_spmm_relu",
    "void_spmm_silu",
    "fused_sparse_mlp",
    # Creation helpers
    "create_fused_spmm_gelu",
    "create_fused_spmm_relu",
    "create_sparse_mlp",
    # Modules
    "FusedSparseLinear",
    # Utilities
    "get_activation_code",
]
