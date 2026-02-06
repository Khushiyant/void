"""Multi-GPU sharding and distributed SpMM for VOID tensors."""

import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
from enum import Enum


class ShardingStrategy(Enum):
    """Strategies for distributing VOID tensors across GPUs."""
    ROW_WISE = "row_wise"          # Each GPU has subset of rows
    COLUMN_WISE = "column_wise"    # Each GPU has subset of columns
    BLOCK_WISE = "block_wise"      # Distribute blocks round-robin


@dataclass
class DistributedVOIDTensor:
    """
    VOID tensor distributed across multiple GPUs.

    Each rank holds a local shard of the full tensor.
    Metadata tracks sharding strategy and communication patterns.

    Attributes:
        local_tensor: Local VOIDTensor shard on this rank
        strategy: How the tensor is sharded
        world_size: Number of processes/GPUs
        rank: This process's rank
        global_shape: Shape of the full (unsharded) tensor
        local_block_ranges: Which blocks this rank owns
    """
    local_tensor: 'VOIDTensor'  # type: ignore
    strategy: ShardingStrategy
    world_size: int
    rank: int
    global_shape: Tuple[int, int]
    local_row_range: Tuple[int, int]   # (start_row, end_row) in block indices
    local_col_range: Tuple[int, int]   # (start_col, end_col) in block indices

    @property
    def device(self) -> torch.device:
        return self.local_tensor.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.local_tensor.values.dtype

    @property
    def n_local_blocks(self) -> int:
        return self.local_tensor.n_blocks

    def to(self, device: Union[str, torch.device]) -> 'DistributedVOIDTensor':
        return DistributedVOIDTensor(
            local_tensor=self.local_tensor.to(device),
            strategy=self.strategy,
            world_size=self.world_size,
            rank=self.rank,
            global_shape=self.global_shape,
            local_row_range=self.local_row_range,
            local_col_range=self.local_col_range,
        )


def is_distributed() -> bool:
    """Check if distributed is initialized."""
    return dist.is_initialized()


def get_world_size() -> int:
    """Get world size or 1 if not distributed."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank or 0 if not distributed."""
    if is_distributed():
        return dist.get_rank()
    return 0


def shard_void_tensor(
    void_tensor,  # VOIDTensor
    strategy: ShardingStrategy = ShardingStrategy.ROW_WISE,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> DistributedVOIDTensor:
    """
    Shard a VOIDTensor across multiple GPUs.

    This function should be called on all ranks with the same input tensor.
    Each rank will keep only its assigned shard.

    Args:
        void_tensor: Full VOIDTensor to shard
        strategy: Sharding strategy
        world_size: Number of processes (default: from dist)
        rank: This process's rank (default: from dist)

    Returns:
        DistributedVOIDTensor with local shard
    """
    from .format import VOIDTensor

    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()

    n_block_rows, n_block_cols = void_tensor.block_grid
    tile_m, tile_k = void_tensor.tile_size

    if strategy == ShardingStrategy.ROW_WISE:
        # Divide rows among ranks
        rows_per_rank = (n_block_rows + world_size - 1) // world_size
        start_row = rank * rows_per_rank
        end_row = min(start_row + rows_per_rank, n_block_rows)

        # Filter blocks belonging to this rank
        mask = (void_tensor.block_rows >= start_row) & (void_tensor.block_rows < end_row)

        local_row_range = (start_row, end_row)
        local_col_range = (0, n_block_cols)

    elif strategy == ShardingStrategy.COLUMN_WISE:
        # Divide columns among ranks
        cols_per_rank = (n_block_cols + world_size - 1) // world_size
        start_col = rank * cols_per_rank
        end_col = min(start_col + cols_per_rank, n_block_cols)

        # Filter blocks
        mask = (void_tensor.block_cols >= start_col) & (void_tensor.block_cols < end_col)

        local_row_range = (0, n_block_rows)
        local_col_range = (start_col, end_col)

    elif strategy == ShardingStrategy.BLOCK_WISE:
        # Round-robin block assignment
        block_indices = torch.arange(void_tensor.n_blocks, device=void_tensor.values.device)
        mask = (block_indices % world_size) == rank

        local_row_range = (0, n_block_rows)
        local_col_range = (0, n_block_cols)

    else:
        raise ValueError(f"Unknown sharding strategy: {strategy}")

    # Create local shard
    indices = torch.where(mask)[0]
    n_local = len(indices)

    if n_local > 0:
        local_values = void_tensor.values[indices]
        local_block_rows = void_tensor.block_rows[indices]
        local_block_cols = void_tensor.block_cols[indices]

        # Adjust block indices for row-wise sharding
        if strategy == ShardingStrategy.ROW_WISE:
            local_block_rows = local_block_rows - start_row
    else:
        local_values = torch.zeros(0, tile_m, tile_k,
                                   dtype=void_tensor.values.dtype,
                                   device=void_tensor.values.device)
        local_block_rows = torch.zeros(0, dtype=torch.int32, device=void_tensor.values.device)
        local_block_cols = torch.zeros(0, dtype=torch.int32, device=void_tensor.values.device)

    # Compute local shape
    if strategy == ShardingStrategy.ROW_WISE:
        local_M = (end_row - start_row) * tile_m
        local_K = void_tensor.shape[1]
    elif strategy == ShardingStrategy.COLUMN_WISE:
        local_M = void_tensor.shape[0]
        local_K = (end_col - start_col) * tile_k
    else:
        local_M, local_K = void_tensor.shape

    local_tensor = VOIDTensor(
        values=local_values,
        block_rows=local_block_rows,
        block_cols=local_block_cols,
        shape=(local_M, local_K),
        tile_size=void_tensor.tile_size,
        n_blocks=n_local,
    )

    return DistributedVOIDTensor(
        local_tensor=local_tensor,
        strategy=strategy,
        world_size=world_size,
        rank=rank,
        global_shape=void_tensor.shape,
        local_row_range=local_row_range,
        local_col_range=local_col_range,
    )


def distributed_void_spmm(
    dist_tensor: DistributedVOIDTensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    Distributed Sparse-Dense Matrix Multiplication.

    Performs C = A @ B where A is distributed across GPUs.
    Communication pattern depends on sharding strategy:
    - ROW_WISE: All-gather output rows
    - COLUMN_WISE: All-reduce partial sums
    - BLOCK_WISE: All-reduce partial sums

    Args:
        dist_tensor: Distributed VOID sparse matrix [M, K]
        b: Dense matrix [K, N] (must be replicated on all ranks)
        out: Optional output buffer
        async_op: Whether to return async handle

    Returns:
        Dense matrix C [M, N] on all ranks
    """
    from .ops import void_spmm

    M, K = dist_tensor.global_shape
    N = b.shape[1]

    # Compute local result
    local_out = void_spmm(dist_tensor.local_tensor, b)

    if dist_tensor.world_size == 1:
        return local_out

    if dist_tensor.strategy == ShardingStrategy.ROW_WISE:
        # All-gather rows from all ranks
        if out is None:
            out = torch.zeros(M, N, dtype=b.dtype, device=b.device)

        # Gather all local outputs
        gathered = [torch.zeros_like(local_out) for _ in range(dist_tensor.world_size)]
        dist.all_gather(gathered, local_out)

        # Concatenate along row dimension
        tile_m = dist_tensor.local_tensor.tile_size[0]
        for rank_idx, g in enumerate(gathered):
            start_row = rank_idx * (dist_tensor.local_row_range[1] - dist_tensor.local_row_range[0]) * tile_m
            end_row = min(start_row + g.shape[0], M)
            out[start_row:end_row] = g[:end_row - start_row]

        return out

    elif dist_tensor.strategy == ShardingStrategy.COLUMN_WISE:
        # All-reduce partial sums
        if out is None:
            out = torch.zeros(M, N, dtype=b.dtype, device=b.device)

        # Each rank has partial result for all rows
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out

    elif dist_tensor.strategy == ShardingStrategy.BLOCK_WISE:
        # All-reduce partial sums
        if out is None:
            out = torch.zeros(M, N, dtype=b.dtype, device=b.device)

        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out

    else:
        raise ValueError(f"Unknown strategy: {dist_tensor.strategy}")


def gather_void_tensor(
    dist_tensor: DistributedVOIDTensor,
    dst_rank: int = 0,
) -> Optional['VOIDTensor']:  # type: ignore
    """
    Gather distributed VOID tensor to a single rank.

    Args:
        dist_tensor: Distributed tensor
        dst_rank: Destination rank (default: 0)

    Returns:
        Full VOIDTensor on dst_rank, None on other ranks
    """
    from .format import VOIDTensor

    if dist_tensor.world_size == 1:
        return dist_tensor.local_tensor

    rank = dist_tensor.rank
    world_size = dist_tensor.world_size
    device = dist_tensor.device

    # Gather block counts
    local_count = torch.tensor([dist_tensor.n_local_blocks], device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    all_counts = [c.item() for c in all_counts]

    total_blocks = sum(all_counts)
    tile_m, tile_k = dist_tensor.local_tensor.tile_size

    if rank == dst_rank:
        # Allocate full tensor
        all_values = torch.zeros(total_blocks, tile_m, tile_k,
                                 dtype=dist_tensor.dtype, device=device)
        all_block_rows = torch.zeros(total_blocks, dtype=torch.int32, device=device)
        all_block_cols = torch.zeros(total_blocks, dtype=torch.int32, device=device)

        # Copy local data
        offset = sum(all_counts[:rank])
        n_local = all_counts[rank]
        all_values[offset:offset+n_local] = dist_tensor.local_tensor.values
        all_block_rows[offset:offset+n_local] = dist_tensor.local_tensor.block_rows
        all_block_cols[offset:offset+n_local] = dist_tensor.local_tensor.block_cols

        # Receive from other ranks
        for src in range(world_size):
            if src != dst_rank and all_counts[src] > 0:
                offset = sum(all_counts[:src])
                n_blocks = all_counts[src]

                recv_values = torch.zeros(n_blocks, tile_m, tile_k,
                                          dtype=dist_tensor.dtype, device=device)
                recv_rows = torch.zeros(n_blocks, dtype=torch.int32, device=device)
                recv_cols = torch.zeros(n_blocks, dtype=torch.int32, device=device)

                dist.recv(recv_values, src=src)
                dist.recv(recv_rows, src=src)
                dist.recv(recv_cols, src=src)

                all_values[offset:offset+n_blocks] = recv_values
                all_block_rows[offset:offset+n_blocks] = recv_rows
                all_block_cols[offset:offset+n_blocks] = recv_cols

        # Adjust block indices for row-wise sharding
        if dist_tensor.strategy == ShardingStrategy.ROW_WISE:
            for src in range(world_size):
                offset = sum(all_counts[:src])
                n_blocks = all_counts[src]
                rows_per_rank = dist_tensor.local_row_range[1] - dist_tensor.local_row_range[0]
                all_block_rows[offset:offset+n_blocks] += src * rows_per_rank

        return VOIDTensor(
            values=all_values,
            block_rows=all_block_rows,
            block_cols=all_block_cols,
            shape=dist_tensor.global_shape,
            tile_size=dist_tensor.local_tensor.tile_size,
            n_blocks=total_blocks,
        )
    else:
        # Send to dst_rank
        if dist_tensor.n_local_blocks > 0:
            dist.send(dist_tensor.local_tensor.values, dst=dst_rank)
            dist.send(dist_tensor.local_tensor.block_rows, dst=dst_rank)
            dist.send(dist_tensor.local_tensor.block_cols, dst=dst_rank)
        return None


def broadcast_void_tensor(
    void_tensor: Optional['VOIDTensor'],  # type: ignore
    src_rank: int = 0,
) -> 'VOIDTensor':  # type: ignore
    """
    Broadcast VOID tensor from source rank to all ranks.

    Args:
        void_tensor: VOIDTensor on src_rank, can be None on others
        src_rank: Source rank

    Returns:
        VOIDTensor replicated on all ranks
    """
    from .format import VOIDTensor

    if not is_distributed() or get_world_size() == 1:
        return void_tensor

    rank = get_rank()
    device = torch.device('cuda', rank)

    # Broadcast metadata
    if rank == src_rank:
        metadata = torch.tensor([
            void_tensor.n_blocks,
            void_tensor.shape[0],
            void_tensor.shape[1],
            void_tensor.tile_size[0],
            void_tensor.tile_size[1],
        ], device=device)
    else:
        metadata = torch.zeros(5, dtype=torch.long, device=device)

    dist.broadcast(metadata, src=src_rank)

    n_blocks = metadata[0].item()
    shape = (metadata[1].item(), metadata[2].item())
    tile_size = (metadata[3].item(), metadata[4].item())

    # Broadcast tensors
    if rank == src_rank:
        values = void_tensor.values
        block_rows = void_tensor.block_rows
        block_cols = void_tensor.block_cols
    else:
        values = torch.zeros(n_blocks, tile_size[0], tile_size[1],
                             dtype=torch.float32, device=device)
        block_rows = torch.zeros(n_blocks, dtype=torch.int32, device=device)
        block_cols = torch.zeros(n_blocks, dtype=torch.int32, device=device)

    dist.broadcast(values, src=src_rank)
    dist.broadcast(block_rows, src=src_rank)
    dist.broadcast(block_cols, src=src_rank)

    return VOIDTensor(
        values=values,
        block_rows=block_rows,
        block_cols=block_cols,
        shape=shape,
        tile_size=tile_size,
        n_blocks=n_blocks,
    )


# =============================================================================
# Pipeline Parallelism Support
# =============================================================================

class PipelineStage:
    """
    Represents a pipeline stage for distributed sparse computation.

    Enables pipeline parallelism where different ranks process
    different stages of a computation (e.g., layers of a neural network).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        num_micro_batches: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = num_micro_batches

        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor: torch.Tensor, async_op: bool = False):
        """Send activation to next stage."""
        if self.next_rank is not None:
            return dist.isend(tensor, dst=self.next_rank) if async_op else dist.send(tensor, dst=self.next_rank)
        return None

    def recv_forward(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Receive activation from previous stage."""
        if self.prev_rank is not None:
            tensor = torch.zeros(shape, dtype=dtype, device=f'cuda:{self.rank}')
            dist.recv(tensor, src=self.prev_rank)
            return tensor
        return None

    def send_backward(self, tensor: torch.Tensor, async_op: bool = False):
        """Send gradient to previous stage."""
        if self.prev_rank is not None:
            return dist.isend(tensor, dst=self.prev_rank) if async_op else dist.send(tensor, dst=self.prev_rank)
        return None

    def recv_backward(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Receive gradient from next stage."""
        if self.next_rank is not None:
            tensor = torch.zeros(shape, dtype=dtype, device=f'cuda:{self.rank}')
            dist.recv(tensor, src=self.next_rank)
            return tensor
        return None


# Export public API
__all__ = [
    # Enums
    "ShardingStrategy",
    # Dataclasses
    "DistributedVOIDTensor",
    "PipelineStage",
    # Functions
    "is_distributed",
    "get_world_size",
    "get_rank",
    "shard_void_tensor",
    "distributed_void_spmm",
    "gather_void_tensor",
    "broadcast_void_tensor",
]
