"""Sparse attention kernels (local, block-sparse, strided patterns)."""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple, Union, Literal
from dataclasses import dataclass

from .format import VOIDTensor, morton_encode_batch
from .ops import get_triton_dtype
import numpy as np


@dataclass
class SparseAttentionMask:
    """
    Represents a sparse attention pattern in VOID format.

    The mask indicates which (query_block, key_block) pairs should attend.
    """
    block_rows: torch.Tensor      # [n_blocks] - query block indices
    block_cols: torch.Tensor      # [n_blocks] - key block indices
    n_blocks: int
    seq_len: int
    block_size: int

    @property
    def n_seq_blocks(self) -> int:
        return (self.seq_len + self.block_size - 1) // self.block_size

    def to(self, device: Union[str, torch.device]) -> 'SparseAttentionMask':
        return SparseAttentionMask(
            block_rows=self.block_rows.to(device),
            block_cols=self.block_cols.to(device),
            n_blocks=self.n_blocks,
            seq_len=self.seq_len,
            block_size=self.block_size,
        )

    def cuda(self) -> 'SparseAttentionMask':
        return self.to('cuda')


def create_local_attention_mask(
    seq_len: int,
    window_size: int,
    block_size: int = 64,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create a local (sliding window) attention mask.

    Each query attends to keys within `window_size` positions.
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    window_blocks = (window_size + block_size - 1) // block_size

    block_rows = []
    block_cols = []

    for q_block in range(n_blocks):
        # Attend to blocks within window
        k_start = max(0, q_block - window_blocks)
        k_end = min(n_blocks, q_block + window_blocks + 1)

        for k_block in range(k_start, k_end):
            block_rows.append(q_block)
            block_cols.append(k_block)

    return SparseAttentionMask(
        block_rows=torch.tensor(block_rows, dtype=torch.int32, device=device),
        block_cols=torch.tensor(block_cols, dtype=torch.int32, device=device),
        n_blocks=len(block_rows),
        seq_len=seq_len,
        block_size=block_size,
    )


def create_strided_attention_mask(
    seq_len: int,
    stride: int,
    block_size: int = 64,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create a strided attention mask (attend every `stride` blocks).

    Used in Longformer-style global + local attention.
    """
    n_blocks = (seq_len + block_size - 1) // block_size

    block_rows = []
    block_cols = []

    for q_block in range(n_blocks):
        # Local: attend to self
        block_rows.append(q_block)
        block_cols.append(q_block)

        # Strided: attend to every `stride` blocks
        for k_block in range(0, n_blocks, stride):
            if k_block != q_block:
                block_rows.append(q_block)
                block_cols.append(k_block)

    return SparseAttentionMask(
        block_rows=torch.tensor(block_rows, dtype=torch.int32, device=device),
        block_cols=torch.tensor(block_cols, dtype=torch.int32, device=device),
        n_blocks=len(block_rows),
        seq_len=seq_len,
        block_size=block_size,
    )


def create_block_sparse_mask(
    seq_len: int,
    block_size: int = 64,
    sparsity: float = 0.9,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create a random block-sparse attention mask.

    Randomly selects which block pairs attend to each other.
    Always includes the diagonal (self-attention).
    """
    n_blocks = (seq_len + block_size - 1) // block_size

    block_rows = []
    block_cols = []

    for q_block in range(n_blocks):
        for k_block in range(n_blocks):
            # Always include diagonal
            if q_block == k_block:
                block_rows.append(q_block)
                block_cols.append(k_block)
            # Randomly include off-diagonal
            elif np.random.random() > sparsity:
                block_rows.append(q_block)
                block_cols.append(k_block)

    return SparseAttentionMask(
        block_rows=torch.tensor(block_rows, dtype=torch.int32, device=device),
        block_cols=torch.tensor(block_cols, dtype=torch.int32, device=device),
        n_blocks=len(block_rows),
        seq_len=seq_len,
        block_size=block_size,
    )


def create_causal_local_mask(
    seq_len: int,
    window_size: int,
    block_size: int = 64,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create a causal local attention mask.

    Each query attends to keys within `window_size` positions,
    but only to past positions (causal).
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    window_blocks = (window_size + block_size - 1) // block_size

    block_rows = []
    block_cols = []

    for q_block in range(n_blocks):
        # Only attend to current and past blocks
        k_start = max(0, q_block - window_blocks)
        k_end = q_block + 1  # Causal: only up to current block

        for k_block in range(k_start, k_end):
            block_rows.append(q_block)
            block_cols.append(k_block)

    return SparseAttentionMask(
        block_rows=torch.tensor(block_rows, dtype=torch.int32, device=device),
        block_cols=torch.tensor(block_cols, dtype=torch.int32, device=device),
        n_blocks=len(block_rows),
        seq_len=seq_len,
        block_size=block_size,
    )


# =============================================================================
# Sparse Attention Triton Kernel
# =============================================================================

@triton.jit
def _sparse_attention_fwd_kernel(
    Q, K, V, Out,
    # Mask info
    block_rows_ptr, block_cols_ptr,
    block_offsets_ptr,  # [n_query_blocks + 1] - CSR-style offsets
    # Dimensions
    seq_len, head_dim, n_blocks, n_heads,
    # Strides (for batch dimension)
    stride_batch, stride_head, stride_seq, stride_dim,
    # Softmax scale
    sm_scale,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    # Enhanced pipelining (2025 SOTA)
    NUM_STAGES: tl.constexpr = 3,
):
    """
    Fused sparse attention forward kernel.

    Grid: (batch * n_heads, n_query_blocks)
    """
    # Program IDs
    pid_bh = tl.program_id(0)  # batch * head index
    pid_q = tl.program_id(1)   # query block

    # Decompose batch and head
    batch_idx = pid_bh // n_heads
    head_idx = pid_bh % n_heads

    # Base offset for this batch/head combination
    base_offset = batch_idx * stride_batch + head_idx * stride_head

    # Get the range of key blocks for this query block
    block_start = tl.load(block_offsets_ptr + pid_q)
    block_end = tl.load(block_offsets_ptr + pid_q + 1)

    # Initialize accumulators in FP32 for numerical stability
    m_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')  # max for softmax
    l_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)  # sum for softmax
    acc = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)  # output accumulator

    # Load Q block and cast to FP32
    q_seq_start = pid_q * BLOCK_SIZE
    q_ptrs = Q + base_offset + (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
    q_mask = (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Iterate over key blocks
    for block_idx in range(block_start, block_end):
        k_block = tl.load(block_cols_ptr + block_idx)
        k_seq_start = k_block * BLOCK_SIZE

        # Load K block and cast to FP32
        k_ptrs = K + base_offset + (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
        k_mask = (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load V block and cast to FP32
        v_ptrs = V + base_offset + (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Compute attention scores: Q @ K^T
        qk = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_SIZE, BLOCK_SIZE]

        # Mask out invalid positions
        q_pos = q_seq_start + tl.arange(0, BLOCK_SIZE)
        k_pos = k_seq_start + tl.arange(0, BLOCK_SIZE)
        qk_mask = (q_pos[:, None] < seq_len) & (k_pos[None, :] < seq_len)
        qk = tl.where(qk_mask, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # [BLOCK_SIZE]
        m_new = tl.maximum(m_i, m_ij)

        # Correction factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        # Update running sum
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_ij[:, None]) * beta[:, None], axis=1)

        # Update accumulator
        p = tl.exp(qk - m_new[:, None])  # [BLOCK_SIZE, BLOCK_SIZE]
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_new

    # Normalize output
    out = acc / l_i[:, None]

    # Store output with target dtype
    o_ptrs = Out + base_offset + (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
    o_mask = (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
    tl.store(o_ptrs, out.to(OUTPUT_DTYPE), mask=o_mask)


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: SparseAttentionMask,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute sparse attention using VOID-style block masking.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        v: Value tensor [batch, n_heads, seq_len, head_dim]
        mask: SparseAttentionMask defining which blocks attend
        scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    batch, n_heads, seq_len, head_dim = q.shape

    assert k.shape == q.shape and v.shape == q.shape
    assert seq_len == mask.seq_len
    assert q.is_cuda and k.is_cuda and v.is_cuda

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Output
    out = torch.empty_like(q)

    # Build CSR-style offsets for query blocks
    n_query_blocks = mask.n_seq_blocks
    block_counts = torch.zeros(n_query_blocks, dtype=torch.int32, device=q.device)
    block_counts.scatter_add_(0, mask.block_rows.long(), torch.ones_like(mask.block_rows))
    block_offsets = torch.zeros(n_query_blocks + 1, dtype=torch.int32, device=q.device)
    block_offsets[1:] = torch.cumsum(block_counts, dim=0)

    # Sort mask by query block for coalesced access
    sort_idx = torch.argsort(mask.block_rows.long() * mask.n_seq_blocks + mask.block_cols.long())
    sorted_block_rows = mask.block_rows[sort_idx]
    sorted_block_cols = mask.block_cols[sort_idx]

    # Get output dtype for Triton
    output_dtype = get_triton_dtype(q.dtype)

    # Grid
    grid = (batch * n_heads, n_query_blocks)

    # Use enhanced pipelining with 3-4 stages for better memory hiding
    num_stages = 3 if head_dim <= 64 else 4

    _sparse_attention_fwd_kernel[grid](
        q, k, v, out,
        sorted_block_rows, sorted_block_cols, block_offsets,
        seq_len, head_dim, mask.n_blocks, n_heads,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        scale,
        BLOCK_SIZE=mask.block_size,
        HEAD_DIM=head_dim,
        OUTPUT_DTYPE=output_dtype,
        NUM_STAGES=num_stages,
    )

    return out


# =============================================================================
# Dense Attention Reference (for validation)
# =============================================================================

def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[SparseAttentionMask] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference dense attention implementation.

    If mask is provided, applies the sparse mask for validation.
    """
    batch, n_heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply sparse mask if provided
    if mask is not None:
        # Create dense mask from sparse representation
        dense_mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=torch.bool)
        for i in range(mask.n_blocks):
            br = mask.block_rows[i].item()
            bc = mask.block_cols[i].item()
            r_start = br * mask.block_size
            r_end = min(r_start + mask.block_size, seq_len)
            c_start = bc * mask.block_size
            c_end = min(c_start + mask.block_size, seq_len)
            dense_mask[r_start:r_end, c_start:c_end] = True

        attn = attn.masked_fill(~dense_mask, float('-inf'))

    # Softmax and output
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)

    return out


# =============================================================================
# Convenience Functions
# =============================================================================

def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """
    Compute local (sliding window) attention.

    Args:
        q, k, v: [batch, n_heads, seq_len, head_dim]
        window_size: Size of attention window
        block_size: Block size for sparse computation
        causal: If True, only attend to past positions

    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    seq_len = q.shape[2]

    if causal:
        mask = create_causal_local_mask(seq_len, window_size, block_size, q.device)
    else:
        mask = create_local_attention_mask(seq_len, window_size, block_size, q.device)

    return sparse_attention(q, k, v, mask)


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float = 0.9,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Compute block-sparse attention with random sparsity pattern.

    Args:
        q, k, v: [batch, n_heads, seq_len, head_dim]
        sparsity: Fraction of blocks to drop (0.9 = keep 10%)
        block_size: Block size

    Returns:
        Output tensor
    """
    seq_len = q.shape[2]
    mask = create_block_sparse_mask(seq_len, block_size, sparsity, q.device)
    return sparse_attention(q, k, v, mask)


# =============================================================================
# Pipelined Sparse Attention Kernel (async memory prefetching)
# =============================================================================

def get_attention_autotune_configs():
    """Generate autotuning configurations for pipelined attention."""
    configs = []
    for BLOCK_SIZE in [32, 64]:
        for num_warps in [4, 8]:
            for num_stages in [3, 4, 5]:
                configs.append(
                    triton.Config(
                        {'BLOCK_SIZE': BLOCK_SIZE, 'NUM_STAGES': num_stages},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_attention_autotune_configs(),
    key=['seq_len', 'head_dim', 'n_blocks'],
)
@triton.jit
def _sparse_attention_pipelined_kernel(
    Q, K, V, Out,
    # Mask info
    block_rows_ptr, block_cols_ptr,
    block_offsets_ptr,  # [n_query_blocks + 1] - CSR-style offsets
    # Dimensions
    seq_len, head_dim, n_blocks, n_heads,
    # Strides (for batch dimension)
    stride_batch, stride_head, stride_seq, stride_dim,
    # Softmax scale
    sm_scale,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    NUM_STAGES: tl.constexpr = 3,
):
    """
    Pipelined fused sparse attention forward kernel.

    Uses software pipelining (num_stages) to overlap memory loads
    with computation for better memory latency hiding.

    Grid: (batch * n_heads, n_query_blocks)
    """
    # Program IDs
    pid_bh = tl.program_id(0)  # batch * head index
    pid_q = tl.program_id(1)   # query block

    # Decompose batch and head
    batch_idx = pid_bh // n_heads
    head_idx = pid_bh % n_heads

    # Base offset for this batch/head combination
    base_offset = batch_idx * stride_batch + head_idx * stride_head

    # Get the range of key blocks for this query block
    block_start = tl.load(block_offsets_ptr + pid_q)
    block_end = tl.load(block_offsets_ptr + pid_q + 1)
    n_key_blocks = block_end - block_start

    # Initialize accumulators in FP32 for numerical stability
    m_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)

    # Load Q block and cast to FP32
    q_seq_start = pid_q * BLOCK_SIZE
    q_ptrs = Q + base_offset + (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
    q_mask = (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Pipelined iteration over key blocks with explicit prefetching
    for block_offset in tl.range(0, n_key_blocks, num_stages=NUM_STAGES):
        block_idx = block_start + block_offset
        k_block = tl.load(block_cols_ptr + block_idx)
        k_seq_start = k_block * BLOCK_SIZE

        # Load K block and cast to FP32
        k_ptrs = K + base_offset + (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
        k_mask = (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load V block and cast to FP32
        v_ptrs = V + base_offset + (k_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Compute attention scores: Q @ K^T
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Mask out invalid positions
        q_pos = q_seq_start + tl.arange(0, BLOCK_SIZE)
        k_pos = k_seq_start + tl.arange(0, BLOCK_SIZE)
        qk_mask = (q_pos[:, None] < seq_len) & (k_pos[None, :] < seq_len)
        qk = tl.where(qk_mask, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Correction factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        # Update running sum
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_ij[:, None]) * beta[:, None], axis=1)

        # Update accumulator
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_new

    # Normalize output
    out = acc / l_i[:, None]

    # Store output with target dtype
    o_ptrs = Out + base_offset + (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim
    o_mask = (q_seq_start + tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len
    tl.store(o_ptrs, out.to(OUTPUT_DTYPE), mask=o_mask)


def sparse_attention_pipelined(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: SparseAttentionMask,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute sparse attention with pipelined memory prefetching.

    This variant uses autotuned software pipelining to overlap memory
    loads with computation for better performance on irregular patterns.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        v: Value tensor [batch, n_heads, seq_len, head_dim]
        mask: SparseAttentionMask defining which blocks attend
        scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    batch, n_heads, seq_len, head_dim = q.shape

    assert k.shape == q.shape and v.shape == q.shape
    assert seq_len == mask.seq_len
    assert q.is_cuda and k.is_cuda and v.is_cuda

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Output
    out = torch.empty_like(q)

    # Build CSR-style offsets for query blocks
    n_query_blocks = mask.n_seq_blocks
    block_counts = torch.zeros(n_query_blocks, dtype=torch.int32, device=q.device)
    block_counts.scatter_add_(0, mask.block_rows.long(), torch.ones_like(mask.block_rows))
    block_offsets = torch.zeros(n_query_blocks + 1, dtype=torch.int32, device=q.device)
    block_offsets[1:] = torch.cumsum(block_counts, dim=0)

    # Sort mask by query block for coalesced access
    sort_idx = torch.argsort(mask.block_rows.long() * mask.n_seq_blocks + mask.block_cols.long())
    sorted_block_rows = mask.block_rows[sort_idx]
    sorted_block_cols = mask.block_cols[sort_idx]

    # Get output dtype for Triton
    output_dtype = get_triton_dtype(q.dtype)

    # Grid is determined by autotuner's BLOCK_SIZE
    grid = lambda meta: (batch * n_heads, (seq_len + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    _sparse_attention_pipelined_kernel[grid](
        q, k, v, out,
        sorted_block_rows, sorted_block_cols, block_offsets,
        seq_len, head_dim, mask.n_blocks, n_heads,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        scale,
        HEAD_DIM=head_dim,
        OUTPUT_DTYPE=output_dtype,
    )

    return out
