"""
Dynamic Sparsity Support for VOID

Enables runtime-adaptive sparse patterns that can change during inference/training:
- Top-k attention: dynamically select most important key-value pairs
- Threshold-based: prune values below dynamic threshold
- Pattern-based: update sparsity pattern from learned masks

Key features:
- Pre-allocated buffers for efficient pattern updates
- Batch-aware dynamic patterns (different patterns per batch element)
- Integration with attention mechanisms for dynamic sparse attention

Use cases:
- Adaptive inference where sparsity varies by input
- Progressive pruning during training
- Dynamic routing in mixture-of-experts
"""

import torch
import triton
import triton.language as tl
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List
from enum import Enum


class UpdateStrategy(Enum):
    """Strategies for updating dynamic sparse patterns."""
    TOPK = "topk"              # Keep top-k elements by score
    THRESHOLD = "threshold"    # Keep elements above threshold
    RANDOM = "random"          # Random subset (for dropout-style sparsity)
    LEARNED = "learned"        # Update from learned binary mask


@dataclass
class DynamicVOIDTensor:
    """
    VOID tensor with mutable sparsity pattern.

    Unlike regular VOIDTensor, this supports efficient pattern updates
    by pre-allocating buffers for maximum expected blocks.

    Attributes:
        values: Block values [max_blocks, tile_m, tile_k]
        block_rows: Block row indices [max_blocks]
        block_cols: Block column indices [max_blocks]
        active_mask: Boolean mask indicating which blocks are active [max_blocks]
        shape: Original dense shape (M, K)
        tile_size: Block dimensions (tile_m, tile_k)
        max_blocks: Maximum number of blocks (pre-allocated capacity)
        n_active: Current number of active blocks
    """
    values: torch.Tensor           # [max_blocks, tile_m, tile_k]
    block_rows: torch.Tensor       # [max_blocks]
    block_cols: torch.Tensor       # [max_blocks]
    active_mask: torch.Tensor      # [max_blocks] bool
    shape: Tuple[int, int]
    tile_size: Tuple[int, int]
    max_blocks: int

    # Cached active block info
    _active_indices: Optional[torch.Tensor] = None
    _row_ptr: Optional[torch.Tensor] = None

    @property
    def n_active(self) -> int:
        """Number of currently active blocks."""
        return self.active_mask.sum().item()

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

    @property
    def sparsity(self) -> float:
        """Current block sparsity (fraction of inactive blocks)."""
        total_possible = self.block_grid[0] * self.block_grid[1]
        return 1.0 - (self.n_active / total_possible)

    def invalidate_cache(self):
        """Invalidate cached indices after pattern update."""
        self._active_indices = None
        self._row_ptr = None

    def get_active_indices(self) -> torch.Tensor:
        """Get indices of active blocks."""
        if self._active_indices is None:
            self._active_indices = torch.where(self.active_mask)[0]
        return self._active_indices

    def get_active_values(self) -> torch.Tensor:
        """Get values of active blocks only."""
        indices = self.get_active_indices()
        return self.values[indices]

    def get_active_block_rows(self) -> torch.Tensor:
        """Get row indices of active blocks."""
        indices = self.get_active_indices()
        return self.block_rows[indices]

    def get_active_block_cols(self) -> torch.Tensor:
        """Get column indices of active blocks."""
        indices = self.get_active_indices()
        return self.block_cols[indices]

    def to(self, device: Union[str, torch.device]) -> 'DynamicVOIDTensor':
        return DynamicVOIDTensor(
            values=self.values.to(device),
            block_rows=self.block_rows.to(device),
            block_cols=self.block_cols.to(device),
            active_mask=self.active_mask.to(device),
            shape=self.shape,
            tile_size=self.tile_size,
            max_blocks=self.max_blocks,
        )

    def to_void_tensor(self):
        """Convert to regular VOIDTensor (active blocks only)."""
        from .format import VOIDTensor

        indices = self.get_active_indices()
        n_active = len(indices)

        return VOIDTensor(
            values=self.values[indices].clone(),
            block_rows=self.block_rows[indices].clone(),
            block_cols=self.block_cols[indices].clone(),
            shape=self.shape,
            tile_size=self.tile_size,
            n_blocks=n_active,
        )


def create_dynamic_void_tensor(
    shape: Tuple[int, int],
    tile_size: int = 32,
    max_sparsity: float = 0.95,
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
) -> DynamicVOIDTensor:
    """
    Create an empty DynamicVOIDTensor with pre-allocated buffers.

    Args:
        shape: Dense matrix shape (M, K)
        tile_size: Block size
        max_sparsity: Maximum expected sparsity (determines buffer size)
        dtype: Data type for values
        device: Device to create tensor on

    Returns:
        DynamicVOIDTensor with pre-allocated buffers
    """
    M, K = shape
    tile_m = tile_k = tile_size

    n_block_rows = (M + tile_m - 1) // tile_m
    n_block_cols = (K + tile_k - 1) // tile_k
    total_blocks = n_block_rows * n_block_cols

    # Pre-allocate for (1 - max_sparsity) fraction of blocks
    max_blocks = max(1, int(total_blocks * (1 - max_sparsity) * 1.5))  # 1.5x buffer

    # Allocate buffers
    values = torch.zeros(max_blocks, tile_m, tile_k, dtype=dtype, device=device)
    block_rows = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    block_cols = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    active_mask = torch.zeros(max_blocks, dtype=torch.bool, device=device)

    return DynamicVOIDTensor(
        values=values,
        block_rows=block_rows,
        block_cols=block_cols,
        active_mask=active_mask,
        shape=shape,
        tile_size=(tile_m, tile_k),
        max_blocks=max_blocks,
    )


def from_void_tensor(
    void_tensor,  # VOIDTensor
    max_blocks: Optional[int] = None,
) -> DynamicVOIDTensor:
    """
    Create a DynamicVOIDTensor from a regular VOIDTensor.

    Args:
        void_tensor: Source VOIDTensor
        max_blocks: Maximum blocks (default: 2x current)

    Returns:
        DynamicVOIDTensor
    """
    n_blocks = void_tensor.n_blocks
    if max_blocks is None:
        max_blocks = max(n_blocks * 2, 16)

    tile_m, tile_k = void_tensor.tile_size

    # Create buffers
    values = torch.zeros(max_blocks, tile_m, tile_k,
                         dtype=void_tensor.values.dtype,
                         device=void_tensor.values.device)
    block_rows = torch.zeros(max_blocks, dtype=torch.int32,
                             device=void_tensor.block_rows.device)
    block_cols = torch.zeros(max_blocks, dtype=torch.int32,
                             device=void_tensor.block_cols.device)
    active_mask = torch.zeros(max_blocks, dtype=torch.bool,
                              device=void_tensor.values.device)

    # Copy existing data
    values[:n_blocks] = void_tensor.values
    block_rows[:n_blocks] = void_tensor.block_rows
    block_cols[:n_blocks] = void_tensor.block_cols
    active_mask[:n_blocks] = True

    return DynamicVOIDTensor(
        values=values,
        block_rows=block_rows,
        block_cols=block_cols,
        active_mask=active_mask,
        shape=void_tensor.shape,
        tile_size=void_tensor.tile_size,
        max_blocks=max_blocks,
    )


def update_from_topk_scores(
    tensor: DynamicVOIDTensor,
    scores: torch.Tensor,
    k: int,
    per_row: bool = True,
) -> None:
    """
    Update sparsity pattern keeping top-k blocks by score.

    Args:
        tensor: DynamicVOIDTensor to update (modified in-place)
        scores: Score tensor [n_block_rows, n_block_cols] or [max_blocks]
        k: Number of blocks to keep (total or per row)
        per_row: If True, keep k blocks per row; else keep k total
    """
    tensor.invalidate_cache()

    if per_row:
        n_block_rows, n_block_cols = tensor.block_grid

        # Reshape scores to per-row format
        if scores.dim() == 1:
            # Assume scores are for active blocks, need to map
            scores_2d = torch.full((n_block_rows, n_block_cols), float('-inf'),
                                   device=tensor.device)
            for i, active in enumerate(tensor.active_mask):
                if active and i < len(scores):
                    r = tensor.block_rows[i].item()
                    c = tensor.block_cols[i].item()
                    scores_2d[r, c] = scores[i]
            scores = scores_2d

        # Get top-k per row
        k_per_row = min(k, n_block_cols)
        topk_scores, topk_cols = torch.topk(scores, k_per_row, dim=-1)

        # Reset active mask
        tensor.active_mask.fill_(False)

        # Set new active blocks
        block_idx = 0
        for row in range(n_block_rows):
            for col_idx in range(k_per_row):
                if topk_scores[row, col_idx] > float('-inf'):
                    col = topk_cols[row, col_idx].item()
                    if block_idx < tensor.max_blocks:
                        tensor.block_rows[block_idx] = row
                        tensor.block_cols[block_idx] = col
                        tensor.active_mask[block_idx] = True
                        block_idx += 1
    else:
        # Global top-k
        if scores.dim() > 1:
            scores = scores.flatten()

        k = min(k, len(scores), tensor.max_blocks)
        topk_scores, topk_indices = torch.topk(scores, k)

        # Reset and update
        tensor.active_mask.fill_(False)
        tensor.active_mask[:k] = True


def update_from_mask(
    tensor: DynamicVOIDTensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    """
    Update sparsity pattern from a dense or block mask.

    Args:
        tensor: DynamicVOIDTensor to update (modified in-place)
        mask: Mask tensor [M, K] or [n_block_rows, n_block_cols]
        threshold: Threshold for binarizing soft masks
    """
    tensor.invalidate_cache()

    n_block_rows, n_block_cols = tensor.block_grid
    tile_m, tile_k = tensor.tile_size

    # Convert to block mask if needed
    if mask.shape[0] == tensor.shape[0]:
        # Dense mask - pool to block level
        M, K = tensor.shape
        mask_padded = torch.nn.functional.pad(
            mask.float(),
            (0, n_block_cols * tile_k - K, 0, n_block_rows * tile_m - M)
        )
        mask_blocks = mask_padded.view(n_block_rows, tile_m, n_block_cols, tile_k)
        mask_blocks = mask_blocks.permute(0, 2, 1, 3).reshape(n_block_rows, n_block_cols, -1)
        block_mask = mask_blocks.mean(dim=-1) > threshold
    else:
        block_mask = mask > threshold

    # Update active pattern
    tensor.active_mask.fill_(False)
    block_idx = 0

    for row in range(n_block_rows):
        for col in range(n_block_cols):
            if block_mask[row, col] and block_idx < tensor.max_blocks:
                tensor.block_rows[block_idx] = row
                tensor.block_cols[block_idx] = col
                tensor.active_mask[block_idx] = True
                block_idx += 1


def update_from_threshold(
    tensor: DynamicVOIDTensor,
    values: torch.Tensor,
    threshold: float,
    per_block: bool = True,
) -> None:
    """
    Update sparsity pattern based on value threshold.

    Blocks with all values below threshold become inactive.

    Args:
        tensor: DynamicVOIDTensor to update
        values: Value tensor (same shape as tensor.values)
        threshold: Threshold for keeping blocks
        per_block: If True, threshold based on block max; else element-wise
    """
    tensor.invalidate_cache()

    n_active = tensor.n_active

    if per_block:
        # Check max value per block
        for i in range(min(n_active, tensor.max_blocks)):
            if tensor.active_mask[i]:
                block_max = values[i].abs().max()
                if block_max < threshold:
                    tensor.active_mask[i] = False
    else:
        # Update values and mask elements
        new_values = values.clone()
        new_values[values.abs() < threshold] = 0

        # Check if blocks are still active
        for i in range(min(n_active, tensor.max_blocks)):
            if tensor.active_mask[i]:
                if new_values[i].abs().sum() == 0:
                    tensor.active_mask[i] = False
                else:
                    tensor.values[i] = new_values[i]


# =============================================================================
# Dynamic Top-K Sparse Attention
# =============================================================================

@dataclass
class DynamicAttentionConfig:
    """Configuration for dynamic sparse attention."""
    k_per_query: int = 64          # Top-k keys per query
    block_size: int = 64           # Block size for sparse computation
    score_scale: float = 1.0       # Scale for attention scores
    use_causal: bool = False       # Apply causal masking


@triton.jit
def _topk_indices_kernel(
    scores_ptr,          # [batch, n_heads, seq_len, seq_len]
    indices_ptr,         # [batch, n_heads, seq_len, k]
    seq_len,
    k,
    stride_batch, stride_head, stride_q, stride_k,
    BLOCK_SIZE: tl.constexpr,
):
    """Extract top-k indices per query position."""
    pid_bh = tl.program_id(0)  # batch * head
    pid_q = tl.program_id(1)   # query position

    batch_idx = pid_bh // stride_head
    head_idx = pid_bh % stride_head

    # This is a simplified placeholder
    # Real implementation would use efficient top-k selection
    pass


def compute_topk_attention_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    top_k: int,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Compute top-k attention pattern by selecting most important key positions.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        top_k: Number of key positions to attend to per query
        block_size: Block size for sparse attention

    Returns:
        Block mask tensor [batch, n_heads, n_q_blocks, n_k_blocks]
    """
    batch, n_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Compute full attention scores (for small sequences)
    # For large sequences, use approximate methods
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, n_heads, seq, seq]

    # Get top-k per query
    _, topk_indices = torch.topk(scores, min(top_k, seq_len), dim=-1)

    # Convert to block mask
    n_blocks = (seq_len + block_size - 1) // block_size
    block_mask = torch.zeros(batch, n_heads, n_blocks, n_blocks,
                             device=q.device, dtype=torch.bool)

    # Map top-k indices to blocks
    topk_blocks = topk_indices // block_size
    query_blocks = torch.arange(seq_len, device=q.device).unsqueeze(-1) // block_size

    # Set block mask where top-k falls
    for b in range(batch):
        for h in range(n_heads):
            for q_pos in range(seq_len):
                q_block = q_pos // block_size
                for k_idx in range(topk_indices.shape[-1]):
                    k_block = topk_blocks[b, h, q_pos, k_idx].item()
                    block_mask[b, h, q_block, k_block] = True

    return block_mask


def dynamic_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    block_size: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Dynamic top-k sparse attention.

    Computes attention while only attending to top-k most similar
    key positions per query, using block-sparse computation.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        v: Value tensor [batch, n_heads, seq_len, head_dim]
        top_k: Number of keys to attend to per query
        block_size: Block size for sparse computation
        scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    from .attention import sparse_attention, SparseAttentionMask

    batch, n_heads, seq_len, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # For each batch/head, compute top-k pattern and apply sparse attention
    # This is a simplified implementation - production would batch this

    outputs = torch.zeros_like(q)

    for b in range(batch):
        for h in range(n_heads):
            # Get single head
            q_bh = q[b:b+1, h:h+1]
            k_bh = k[b:b+1, h:h+1]
            v_bh = v[b:b+1, h:h+1]

            # Compute top-k mask for this head
            scores = torch.matmul(q_bh, k_bh.transpose(-2, -1)) * scale
            _, topk_idx = torch.topk(scores.squeeze(0).squeeze(0), min(top_k, seq_len), dim=-1)

            # Convert to block mask
            n_blocks = (seq_len + block_size - 1) // block_size
            block_rows = []
            block_cols = []

            for q_pos in range(seq_len):
                q_block = q_pos // block_size
                for k_idx in range(topk_idx.shape[-1]):
                    k_pos = topk_idx[q_pos, k_idx].item()
                    k_block = k_pos // block_size
                    if (q_block, k_block) not in zip(block_rows, block_cols):
                        block_rows.append(q_block)
                        block_cols.append(k_block)

            # Create sparse attention mask
            mask = SparseAttentionMask(
                block_rows=torch.tensor(block_rows, dtype=torch.int32, device=q.device),
                block_cols=torch.tensor(block_cols, dtype=torch.int32, device=q.device),
                n_blocks=len(block_rows),
                seq_len=seq_len,
                block_size=block_size,
            )

            # Apply sparse attention
            out_bh = sparse_attention(q_bh, k_bh, v_bh, mask, scale)
            outputs[b, h] = out_bh.squeeze(0).squeeze(0)

    return outputs


def progressive_sparsification(
    tensor: DynamicVOIDTensor,
    target_sparsity: float,
    current_step: int,
    total_steps: int,
    schedule: str = 'linear',
) -> float:
    """
    Progressively increase sparsity over training steps.

    Args:
        tensor: DynamicVOIDTensor to sparsify
        target_sparsity: Final target sparsity
        current_step: Current training step
        total_steps: Total steps for sparsification
        schedule: 'linear', 'cubic', or 'exponential'

    Returns:
        Current sparsity level
    """
    progress = min(1.0, current_step / total_steps)

    if schedule == 'linear':
        current_sparsity = progress * target_sparsity
    elif schedule == 'cubic':
        current_sparsity = (progress ** 3) * target_sparsity
    elif schedule == 'exponential':
        current_sparsity = (1 - math.exp(-3 * progress)) * target_sparsity
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # Compute threshold to achieve target sparsity
    if tensor.n_active > 0:
        values_flat = tensor.get_active_values().abs().flatten()
        n_to_keep = int(len(values_flat) * (1 - current_sparsity))
        if n_to_keep > 0:
            threshold = torch.kthvalue(values_flat, len(values_flat) - n_to_keep + 1).values.item()
            update_from_threshold(tensor, tensor.values, threshold)

    return current_sparsity


# Export public API
__all__ = [
    # Enums
    "UpdateStrategy",
    # Dataclasses
    "DynamicVOIDTensor",
    "DynamicAttentionConfig",
    # Functions
    "create_dynamic_void_tensor",
    "from_void_tensor",
    "update_from_topk_scores",
    "update_from_mask",
    "update_from_threshold",
    "compute_topk_attention_mask",
    "dynamic_topk_attention",
    "progressive_sparsification",
]
