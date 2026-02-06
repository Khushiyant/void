"""Autograd functions for SpMM, SpMV, and sparse attention."""

import torch
import torch.autograd as autograd
from torch.autograd import Function
from typing import Optional, Tuple, Any
import triton
import triton.language as tl

from .format import VOIDTensor
from .ops import void_spmm, void_spmv, compute_optimal_tile_n, get_triton_dtype


@triton.jit
def void_spmm_backward_a_kernel(
    # Inputs
    grad_output_ptr,    # [M, N]
    b_ptr,              # [K, N]
    # Sparse block info
    a_block_rows_ptr,   # [n_blocks]
    a_block_cols_ptr,   # [n_blocks]
    # Output
    grad_a_values_ptr,  # [n_blocks, TILE_M, TILE_K]
    # Dimensions
    M, N, K,
    n_blocks,
    # Strides
    stride_gm, stride_gn,
    stride_bk, stride_bn,
    # Tile sizes
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
):
    """
    Compute gradient w.r.t. sparse A values.

    For each sparse block (br, bc):
        grad_A[br, bc] = grad_output[br*TILE_M : (br+1)*TILE_M, :] @ B[bc*TILE_K : (bc+1)*TILE_K, :].T

    Grid: (n_blocks,) - each program handles one sparse block independently.
    """
    block_idx = tl.program_id(0)

    if block_idx >= n_blocks:
        return

    # Load block indices
    br = tl.load(a_block_rows_ptr + block_idx)
    bc = tl.load(a_block_cols_ptr + block_idx)

    # Initialize accumulator in FP32 for numerical stability
    acc = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Iterate over N dimension in tiles
    for n_tile in range((N + TILE_N - 1) // TILE_N):
        n_start = n_tile * TILE_N

        # Load grad_output tile [TILE_M, TILE_N]
        # grad_output[br*TILE_M : (br+1)*TILE_M, n_start : n_start+TILE_N]
        grad_tile_ptr = tl.make_block_ptr(
            base=grad_output_ptr,
            shape=(M, N),
            strides=(stride_gm, stride_gn),
            offsets=(br * TILE_M, n_start),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        grad_tile = tl.load(grad_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        # Load B tile [TILE_K, TILE_N]
        # B[bc*TILE_K : (bc+1)*TILE_K, n_start : n_start+TILE_N]
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(bc * TILE_K, n_start),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )
        b_tile = tl.load(b_tile_ptr, boundary_check=(0, 1)).to(tl.float32)

        # Accumulate: grad_tile @ b_tile.T -> [TILE_M, TILE_K]
        acc += tl.dot(grad_tile, tl.trans(b_tile))

    # Store result to grad_a_values[block_idx]
    grad_a_ptr = tl.make_block_ptr(
        base=grad_a_values_ptr + block_idx * TILE_M * TILE_K,
        shape=(TILE_M, TILE_K),
        strides=(TILE_K, 1),
        offsets=(0, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )
    tl.store(grad_a_ptr, acc.to(OUTPUT_DTYPE))


class VOIDSpMMFunction(Function):
    """
    Autograd function for VOID Sparse Matrix Multiplication.

    Forward: C = A @ B where A is VOID sparse, B is dense
    Backward:
        dL/dB = A^T @ dL/dC
        dL/dA_values = dL/dC @ B^T (only for non-zero blocks)
    """

    @staticmethod
    def forward(
        ctx: Any,
        a_values: torch.Tensor,
        a_block_rows: torch.Tensor,
        a_block_cols: torch.Tensor,
        a_shape: Tuple[int, int],
        a_tile_size: Tuple[int, int],
        a_n_blocks: int,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for SpMM.

        We take individual components to allow gradients to flow to a_values.
        """
        # Reconstruct VOIDTensor for forward computation
        void_tensor = VOIDTensor(
            values=a_values,
            block_rows=a_block_rows,
            block_cols=a_block_cols,
            morton_codes=torch.empty(0, device=a_values.device),  # Not needed for forward
            shape=a_shape,
            tile_size=a_tile_size,
            nnz_original=0,
            n_blocks=a_n_blocks,
            density=0.0,
        )

        c = void_spmm(void_tensor, b)

        # Save for backward
        ctx.save_for_backward(a_values, a_block_rows, a_block_cols, b)
        ctx.a_shape = a_shape
        ctx.a_tile_size = a_tile_size
        ctx.a_n_blocks = a_n_blocks

        return c

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for SpMM.

        grad_output: dL/dC [M, N]

        Returns:
            grad_a_values: dL/dA_values [n_blocks, tile_m, tile_k]
            None for non-differentiable inputs
            grad_b: dL/dB [K, N]
        """
        a_values, a_block_rows, a_block_cols, b = ctx.saved_tensors
        a_shape = ctx.a_shape
        a_tile_size = ctx.a_tile_size
        a_n_blocks = ctx.a_n_blocks

        M, K = a_shape
        tile_m, tile_k = a_tile_size
        _, N = b.shape

        grad_a_values = None
        grad_b = None

        # Gradient w.r.t. B: dL/dB = A^T @ dL/dC
        if ctx.needs_input_grad[6]:  # b is input 6
            # Create transposed VOID tensor
            void_tensor_t = VOIDTensor(
                values=a_values.permute(0, 2, 1).contiguous(),  # Transpose each tile
                block_rows=a_block_cols,  # Swap rows and cols for transpose
                block_cols=a_block_rows,
                morton_codes=torch.empty(0, device=a_values.device),
                shape=(K, M),
                tile_size=(tile_k, tile_m),
                nnz_original=0,
                n_blocks=a_n_blocks,
                density=0.0,
            )
            grad_b = void_spmm(void_tensor_t, grad_output)

        # Gradient w.r.t. A values using fused Triton kernel
        if ctx.needs_input_grad[0]:  # a_values is input 0
            grad_a_values = torch.zeros_like(a_values)

            if a_n_blocks > 0:
                # Ensure inputs are contiguous
                grad_output_contig = grad_output.contiguous()
                b_contig = b.contiguous()

                # Choose TILE_N based on N (use consistent selection across all kernels)
                TILE_N = compute_optimal_tile_n(N)

                # Get output dtype for Triton
                output_dtype = get_triton_dtype(a_values.dtype)

                # Launch fused backward kernel
                grid = (a_n_blocks,)
                void_spmm_backward_a_kernel[grid](
                    grad_output_contig, b_contig,
                    a_block_rows, a_block_cols,
                    grad_a_values,
                    M, N, K, a_n_blocks,
                    grad_output_contig.stride(0), grad_output_contig.stride(1),
                    b_contig.stride(0), b_contig.stride(1),
                    TILE_M=tile_m, TILE_K=tile_k, TILE_N=TILE_N,
                    OUTPUT_DTYPE=output_dtype,
                )

        return grad_a_values, None, None, None, None, None, grad_b


class VOIDSpMM(torch.nn.Module):
    """
    PyTorch module wrapper for VOID SpMM with autograd support.

    Holds a VOID sparse matrix and provides a differentiable forward pass.
    """

    def __init__(self, void_tensor: VOIDTensor, requires_grad: bool = True):
        super().__init__()

        # Store as parameters/buffers
        if requires_grad:
            self.values = torch.nn.Parameter(void_tensor.values.clone())
        else:
            self.register_buffer('values', void_tensor.values.clone())

        self.register_buffer('block_rows', void_tensor.block_rows)
        self.register_buffer('block_cols', void_tensor.block_cols)

        self.shape = void_tensor.shape
        self.tile_size = void_tensor.tile_size
        self.n_blocks = void_tensor.n_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = A @ x

        Args:
            x: Dense input [K, N] or [K]

        Returns:
            Dense output [M, N] or [M]
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        y = VOIDSpMMFunction.apply(
            self.values,
            self.block_rows,
            self.block_cols,
            self.shape,
            self.tile_size,
            self.n_blocks,
            x,
        )

        if squeeze:
            y = y.squeeze(1)

        return y

    def get_void_tensor(self) -> VOIDTensor:
        """Reconstruct VOIDTensor from module state."""
        return VOIDTensor(
            values=self.values if isinstance(self.values, torch.Tensor) else self.values.data,
            block_rows=self.block_rows,
            block_cols=self.block_cols,
            morton_codes=torch.empty(0, device=self.values.device),
            shape=self.shape,
            tile_size=self.tile_size,
            nnz_original=0,
            n_blocks=self.n_blocks,
            density=0.0,
        )


class SparseLinear(torch.nn.Module):
    """
    Sparse linear layer using VOID format.

    Equivalent to nn.Linear but with sparse weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        void_tensor: VOIDTensor,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert void_tensor.shape == (out_features, in_features)

        self.in_features = in_features
        self.out_features = out_features

        self.sparse_weight = VOIDSpMM(void_tensor, requires_grad=True)

        # Use void_tensor dtype if not explicitly specified
        dtype = dtype or void_tensor.dtype

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = xW^T + b

        Args:
            x: Input [*, in_features]

        Returns:
            Output [*, out_features]
        """
        # Handle batched input
        input_shape = x.shape
        x_flat = x.view(-1, self.in_features)  # [batch, in_features]

        # Sparse matmul: [out, in] @ [in, batch]^T -> [out, batch]
        # We need: [batch, in] @ [in, out] -> [batch, out]
        # So compute: ([out, in] @ [batch, in]^T)^T = [batch, out]

        y = self.sparse_weight(x_flat.T).T  # [batch, out_features]

        if self.bias is not None:
            y = y + self.bias

        # Restore shape
        output_shape = input_shape[:-1] + (self.out_features,)
        return y.view(output_shape)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# =============================================================================
# Sparse Attention Backward
# =============================================================================

class SparseAttentionFunction(Function):
    """
    Autograd function for sparse attention.

    Uses checkpointing-style recomputation to save memory.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_rows: torch.Tensor,
        block_cols: torch.Tensor,
        block_offsets: torch.Tensor,
        seq_len: int,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        """Forward pass stores inputs for backward recomputation."""
        from .attention import _sparse_attention_fwd_kernel

        batch, n_heads, _, head_dim = q.shape
        out = torch.empty_like(q)

        n_query_blocks = (seq_len + block_size - 1) // block_size
        n_blocks = len(block_rows)

        grid = (batch * n_heads, n_query_blocks)

        _sparse_attention_fwd_kernel[grid](
            q, k, v, out,
            block_rows, block_cols, block_offsets,
            seq_len, head_dim, n_blocks,
            q.stride(0) * n_heads, q.stride(1), q.stride(2), q.stride(3),
            k.stride(0) * n_heads, k.stride(1), k.stride(2), k.stride(3),
            v.stride(0) * n_heads, v.stride(1), v.stride(2), v.stride(3),
            out.stride(0) * n_heads, out.stride(1), out.stride(2), out.stride(3),
            scale,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, block_rows, block_cols, block_offsets)
        ctx.seq_len = seq_len
        ctx.block_size = block_size
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for sparse attention.

        Computes gradients for Q, K, V using the sparse pattern.
        """
        q, k, v, out, block_rows, block_cols, block_offsets = ctx.saved_tensors
        seq_len = ctx.seq_len
        block_size = ctx.block_size
        scale = ctx.scale

        batch, n_heads, _, head_dim = q.shape

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        # Recompute attention weights and compute gradients
        # This is done block-by-block to match the sparse pattern

        n_query_blocks = (seq_len + block_size - 1) // block_size

        for q_block in range(n_query_blocks):
            block_start = block_offsets[q_block].item()
            block_end = block_offsets[q_block + 1].item()

            q_start = q_block * block_size
            q_end = min(q_start + block_size, seq_len)

            q_slice = q[:, :, q_start:q_end, :]  # [B, H, block, D]
            grad_out_slice = grad_output[:, :, q_start:q_end, :]
            out_slice = out[:, :, q_start:q_end, :]

            # Accumulate attention scores and values for this query block
            for block_idx in range(block_start, block_end):
                k_block = block_cols[block_idx].item()

                k_start = k_block * block_size
                k_end = min(k_start + block_size, seq_len)

                k_slice = k[:, :, k_start:k_end, :]
                v_slice = v[:, :, k_start:k_end, :]

                # Recompute attention for this block pair
                attn_scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) * scale
                attn_probs = torch.softmax(attn_scores, dim=-1)

                # Gradient of attention output w.r.t. attention probs
                # grad_attn_probs = grad_out @ V^T
                grad_attn_probs = torch.matmul(grad_out_slice, v_slice.transpose(-2, -1))

                # Gradient of softmax
                # grad_scores = attn_probs * (grad_attn_probs - sum(attn_probs * grad_attn_probs))
                sum_grad = (attn_probs * grad_attn_probs).sum(dim=-1, keepdim=True)
                grad_scores = attn_probs * (grad_attn_probs - sum_grad) * scale

                # Gradients
                grad_q[:, :, q_start:q_end, :] += torch.matmul(grad_scores, k_slice)
                grad_k[:, :, k_start:k_end, :] += torch.matmul(grad_scores.transpose(-2, -1), q_slice)
                grad_v[:, :, k_start:k_end, :] += torch.matmul(attn_probs.transpose(-2, -1), grad_out_slice)

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


def sparse_attention_with_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: 'SparseAttentionMask',
    scale: Optional[float] = None,
    use_optimized: bool = True,
) -> torch.Tensor:
    """
    Sparse attention with full autograd support.

    This version supports backward pass for training.

    Args:
        q, k, v: Query, key, value tensors
        mask: Sparse attention mask
        scale: Softmax scale (default: 1/sqrt(head_dim))
        use_optimized: Use fast Triton backward (default: True)
                       Set to False to use slow reference implementation

    Returns:
        Output tensor with gradients enabled
    """
    import math
    from .attention import SparseAttentionMask

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Use optimized Triton backward by default (100x faster)
    if use_optimized:
        try:
            from .attention_backward import sparse_attention_optimized
            return sparse_attention_optimized(q, k, v, mask, scale)
        except ImportError:
            print("Warning: Optimized backward not available, using slow reference")
            use_optimized = False

    # Fallback to slow reference implementation
    if not use_optimized:
        # Build CSR-style offsets
        n_query_blocks = mask.n_seq_blocks
        block_counts = torch.zeros(n_query_blocks, dtype=torch.int32, device=q.device)
        block_counts.scatter_add_(0, mask.block_rows.long(), torch.ones_like(mask.block_rows))
        block_offsets = torch.zeros(n_query_blocks + 1, dtype=torch.int32, device=q.device)
        block_offsets[1:] = torch.cumsum(block_counts, dim=0)

        # Sort mask
        sort_idx = torch.argsort(mask.block_rows.long() * mask.n_seq_blocks + mask.block_cols.long())
        sorted_block_cols = mask.block_cols[sort_idx]

        return SparseAttentionFunction.apply(
            q, k, v,
            mask.block_rows[sort_idx], sorted_block_cols, block_offsets,
            mask.seq_len, mask.block_size, scale,
        )
