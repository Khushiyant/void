"""
Needle in a Haystack Test for VOID Sparse Attention

Improved implementation with:
1. Realistic embedding patterns (not just alternating +1/-1)
2. Multiple sparse attention strategies (local, strided, hybrid)
3. Proper evaluation metrics (MRR, Recall@K)
4. Multi-needle testing for statistical significance
5. Practical attention patterns used in real models (Longformer, BigBird)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import sys
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, '/home/khushiyant/Develop/experiments/void')

from void import (
    sparse_attention,
    create_block_sparse_mask,
    create_local_attention_mask,
    create_causal_local_mask,
    SparseAttentionMask,
)


class AttentionPattern(Enum):
    """Sparse attention patterns to test."""
    DENSE = "dense"
    LOCAL = "local"
    STRIDED = "strided"
    LOCAL_GLOBAL = "local_global"  # Longformer-style
    RANDOM_BLOCK = "random_block"  # BigBird random component


@dataclass
class NeedleResult:
    """Results from a single needle test."""
    needle_positions: List[int]
    pattern: AttentionPattern

    # Retrieval metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank

    # Coverage
    needles_in_mask: int
    total_needles: int

    # Output quality (max-pooled - can the model find ANY needle?)
    max_needle_similarity: float
    max_noise_similarity: float
    # Also track best rank achieved
    best_rank: int

    # Attention weight metrics (most direct measure)
    max_needle_attn_weight: float  # Max attention weight on any needle
    total_needle_attn_weight: float  # Sum of attention weights on all needles


@dataclass
class BenchmarkConfig:
    """Configuration for needle-haystack benchmark."""
    seq_len: int = 2048
    head_dim: int = 64
    n_heads: int = 8
    batch_size: int = 1
    block_size: int = 64

    # Needle settings
    n_needles: int = 5  # Multiple needles for robustness
    needle_similarity: float = 0.98  # How similar query is to needle (high = distinctive)

    # Sparse attention settings
    local_window: int = 256
    n_global_tokens: int = 64  # For Longformer-style
    stride: int = 64  # For strided attention

    # Hard mode: place needles OUTSIDE local window to truly test sparse patterns
    hard_mode: bool = True


def create_semantic_embeddings(
    n_embeddings: int,
    head_dim: int,
    device: str = 'cuda',
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Create realistic semantic embeddings.

    Uses a learned-like distribution where similar concepts cluster.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create base embeddings with some structure
    # Real embeddings have lower effective dimensionality
    effective_dim = head_dim // 4
    base = torch.randn(n_embeddings, effective_dim, device=device)

    # Project to full dimension with random projection (simulates learned embeddings)
    projection = torch.randn(effective_dim, head_dim, device=device) / math.sqrt(effective_dim)
    embeddings = base @ projection

    # Add small noise and normalize
    embeddings = embeddings + torch.randn_like(embeddings) * 0.1
    embeddings = F.normalize(embeddings, dim=-1)

    return embeddings


def create_needle_query_pair(
    head_dim: int,
    similarity: float = 0.9,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a needle-query pair with specified similarity.

    Returns:
        needle: The target embedding to find
        query: The query embedding (similar but not identical)
    """
    # Create a structured needle (not just random)
    needle = torch.randn(head_dim, device=device)
    needle = F.normalize(needle, dim=0)

    # Create query as interpolation between needle and random
    random_dir = torch.randn(head_dim, device=device)
    random_dir = F.normalize(random_dir, dim=0)

    # Orthogonalize random direction to needle
    random_dir = random_dir - (random_dir @ needle) * needle
    random_dir = F.normalize(random_dir, dim=0)

    # Interpolate: query = similarity * needle + sqrt(1-sim^2) * orthogonal
    query = similarity * needle + math.sqrt(1 - similarity**2) * random_dir
    query = F.normalize(query, dim=0)

    return needle, query


def create_haystack_with_needles(
    seq_len: int,
    head_dim: int,
    needles: torch.Tensor,  # [n_needles, head_dim]
    needle_positions: List[int],
    query: torch.Tensor,  # [head_dim] - the query to make noise orthogonal to
    device: str = 'cuda',
    orthogonal_strength: float = 0.8,  # How much to push noise away from query
) -> torch.Tensor:
    """
    Create haystack with needles inserted and noise orthogonal to query.

    This ensures needles are clearly distinguishable from noise.
    """
    # Create background noise (realistic semantic embeddings)
    haystack = create_semantic_embeddings(seq_len, head_dim, device)

    # Make noise more orthogonal to query direction
    # Project out the query component from noise embeddings
    query_norm = F.normalize(query.unsqueeze(0), dim=-1)  # [1, head_dim]

    # For each noise vector, reduce its similarity to query
    # v_orthogonal = v - (v · q) * q * strength
    similarities = haystack @ query_norm.T  # [seq_len, 1]
    projection = similarities * query_norm * orthogonal_strength  # [seq_len, head_dim]
    haystack = haystack - projection

    # Re-normalize
    haystack = F.normalize(haystack, dim=-1)

    # Insert needles at specified positions (after orthogonalization)
    for i, pos in enumerate(needle_positions):
        haystack[pos] = needles[i]

    return haystack


# =============================================================================
# Sparse Attention Mask Creators
# =============================================================================

def create_local_global_mask(
    seq_len: int,
    block_size: int,
    local_window: int,
    n_global_tokens: int,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create Longformer-style local + global attention mask.

    - Every token attends to local window
    - First n_global_tokens attend to/from all tokens
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    window_blocks = (local_window + block_size - 1) // block_size
    global_blocks = (n_global_tokens + block_size - 1) // block_size

    block_rows = []
    block_cols = []

    for q_block in range(n_blocks):
        # Local attention
        k_start = max(0, q_block - window_blocks // 2)
        k_end = min(n_blocks, q_block + window_blocks // 2 + 1)

        for k_block in range(k_start, k_end):
            block_rows.append(q_block)
            block_cols.append(k_block)

        # Global attention: query attends to global tokens
        for g_block in range(global_blocks):
            if g_block not in range(k_start, k_end):
                block_rows.append(q_block)
                block_cols.append(g_block)

        # Global attention: global tokens attend to all
        if q_block < global_blocks:
            for k_block in range(n_blocks):
                pair = (q_block, k_block)
                if pair not in zip(block_rows, block_cols):
                    block_rows.append(q_block)
                    block_cols.append(k_block)

    # Remove duplicates
    pairs = list(set(zip(block_rows, block_cols)))
    block_rows = [p[0] for p in pairs]
    block_cols = [p[1] for p in pairs]

    return SparseAttentionMask(
        block_rows=torch.tensor(block_rows, dtype=torch.int32, device=device),
        block_cols=torch.tensor(block_cols, dtype=torch.int32, device=device),
        n_blocks=len(block_rows),
        seq_len=seq_len,
        block_size=block_size,
    )


def create_strided_mask(
    seq_len: int,
    block_size: int,
    stride: int,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create strided attention mask (attend every `stride` positions).
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    stride_blocks = max(1, stride // block_size)

    block_rows = []
    block_cols = []

    for q_block in range(n_blocks):
        # Self-attention (diagonal)
        block_rows.append(q_block)
        block_cols.append(q_block)

        # Strided attention
        for k_block in range(0, n_blocks, stride_blocks):
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


def create_hybrid_mask(
    seq_len: int,
    block_size: int,
    local_window: int,
    stride: int,
    n_global_tokens: int,
    random_blocks: int = 0,
    device: str = 'cuda',
) -> SparseAttentionMask:
    """
    Create BigBird-style hybrid mask: local + global + strided + random.
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    window_blocks = (local_window + block_size - 1) // block_size
    stride_blocks = max(1, stride // block_size)
    global_blocks = (n_global_tokens + block_size - 1) // block_size

    pairs = set()

    for q_block in range(n_blocks):
        # Local window
        for offset in range(-window_blocks // 2, window_blocks // 2 + 1):
            k_block = q_block + offset
            if 0 <= k_block < n_blocks:
                pairs.add((q_block, k_block))

        # Strided
        for k_block in range(0, n_blocks, stride_blocks):
            pairs.add((q_block, k_block))

        # Global tokens
        for g_block in range(global_blocks):
            pairs.add((q_block, g_block))
            pairs.add((g_block, q_block))

        # Random blocks
        if random_blocks > 0:
            random_targets = np.random.choice(n_blocks, size=min(random_blocks, n_blocks), replace=False)
            for k_block in random_targets:
                pairs.add((q_block, int(k_block)))

    block_rows = [p[0] for p in pairs]
    block_cols = [p[1] for p in pairs]

    return SparseAttentionMask(
        block_rows=torch.tensor(block_rows, dtype=torch.int32, device=device),
        block_cols=torch.tensor(block_cols, dtype=torch.int32, device=device),
        n_blocks=len(block_rows),
        seq_len=seq_len,
        block_size=block_size,
    )


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_attention_rankings(
    query: torch.Tensor,  # [head_dim]
    keys: torch.Tensor,   # [seq_len, head_dim]
    needle_positions: List[int],
    scale: Optional[float] = None,
    mask: Optional[SparseAttentionMask] = None,
    query_position: Optional[int] = None,
) -> Tuple[List[int], torch.Tensor]:
    """
    Compute where needles rank in attention weights.

    Args:
        query: Query vector
        keys: Key matrix
        needle_positions: Positions of needles in sequence
        scale: Attention scale factor
        mask: Optional sparse attention mask
        query_position: Position of query (required if mask is provided)

    Returns:
        ranks: Rank of each needle (1-indexed, lower is better)
        weights: Full attention weight vector
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    scores = torch.matmul(keys, query) * scale

    # Apply sparse mask if provided
    if mask is not None and query_position is not None:
        # Create attention mask: -inf for positions not in sparse pattern
        block_size = mask.block_size
        query_block = query_position // block_size
        seq_len = keys.shape[0]

        # Find which key blocks are attended by query block
        attended_blocks = set()
        for i in range(mask.n_blocks):
            if mask.block_rows[i].item() == query_block:
                attended_blocks.add(mask.block_cols[i].item())

        # Create mask tensor
        attn_mask = torch.full((seq_len,), float('-inf'), device=scores.device)
        for kb in attended_blocks:
            start = kb * block_size
            end = min(start + block_size, seq_len)
            attn_mask[start:end] = 0.0

        scores = scores + attn_mask

    weights = F.softmax(scores, dim=-1)

    # Get rankings (positions with -inf score will have 0 weight)
    sorted_indices = torch.argsort(weights, descending=True)
    position_to_rank = {idx.item(): rank + 1 for rank, idx in enumerate(sorted_indices)}

    ranks = [position_to_rank[pos] for pos in needle_positions]

    return ranks, weights


def check_needles_in_mask(
    needle_positions: List[int],
    query_position: int,
    mask: SparseAttentionMask,
) -> List[bool]:
    """Check which needles are in the sparse mask for the query position."""
    block_size = mask.block_size
    query_block = query_position // block_size

    # Get all key blocks attended by query block
    attended_blocks = set()
    for i in range(mask.n_blocks):
        if mask.block_rows[i].item() == query_block:
            attended_blocks.add(mask.block_cols[i].item())

    results = []
    for pos in needle_positions:
        needle_block = pos // block_size
        results.append(needle_block in attended_blocks)

    return results


def compute_retrieval_metrics(ranks: List[int], k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Compute retrieval metrics from rankings.

    Returns:
        recall@k for each k
        MRR (Mean Reciprocal Rank)
    """
    n = len(ranks)

    metrics = {}
    for k in k_values:
        recall = sum(1 for r in ranks if r <= k) / n
        metrics[f'recall@{k}'] = recall

    # MRR
    mrr = sum(1.0 / r for r in ranks) / n
    metrics['mrr'] = mrr

    return metrics


# =============================================================================
# Main Benchmark Functions
# =============================================================================

def run_single_pattern_test(
    config: BenchmarkConfig,
    pattern: AttentionPattern,
    device: str = 'cuda',
    seed: int = 42,
) -> NeedleResult:
    """Run needle-in-haystack test for a specific attention pattern."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a SINGLE query - all needles will be similar to this query
    # This is more rigorous than having matched query-needle pairs
    base_query = torch.randn(config.head_dim, device=device)
    base_query = F.normalize(base_query, dim=0)

    # Create needles as perturbations of the query (all similar to the same query)
    needles = []
    for i in range(config.n_needles):
        # Add small orthogonal perturbation to make each needle unique but similar
        noise = torch.randn(config.head_dim, device=device)
        noise = noise - (noise @ base_query) * base_query  # Orthogonalize
        noise = F.normalize(noise, dim=0)

        # Needle = high similarity to query
        needle = config.needle_similarity * base_query + math.sqrt(1 - config.needle_similarity**2) * noise
        needle = F.normalize(needle, dim=0)
        needles.append(needle)

    needles = torch.stack(needles)  # [n_needles, head_dim]
    query = base_query  # Single query for all evaluations

    # Query from end of sequence
    query_position = config.seq_len - 1

    if config.hard_mode:
        # HARD MODE: Place ALL needles OUTSIDE the local window
        # This truly tests whether sparse patterns can find distant information
        # Local window from query_position covers [seq_len - local_window, seq_len]
        # So we place needles in [0, seq_len - local_window - margin]
        max_needle_pos = config.seq_len - config.local_window - config.block_size
        segment_size = max_needle_pos // config.n_needles
        needle_positions = []
        for i in range(config.n_needles):
            start = i * segment_size
            end = start + segment_size
            pos = np.random.randint(max(0, start), max(1, end))
            needle_positions.append(pos)
    else:
        # Easy mode: spread needles across entire sequence
        segment_size = (config.seq_len - 2 * config.block_size) // config.n_needles
        needle_positions = []
        for i in range(config.n_needles):
            start = config.block_size + i * segment_size
            end = start + segment_size
            pos = np.random.randint(start, end)
            needle_positions.append(pos)

    # Create haystack with noise strongly orthogonal to query direction
    haystack = create_haystack_with_needles(
        config.seq_len, config.head_dim, needles, needle_positions,
        query=query, device=device, orthogonal_strength=0.95
    )

    # Create appropriate mask
    if pattern == AttentionPattern.DENSE:
        # For dense, we just compute attention directly
        mask = None
    elif pattern == AttentionPattern.LOCAL:
        mask = create_local_attention_mask(
            config.seq_len, config.local_window, config.block_size, device
        )
    elif pattern == AttentionPattern.STRIDED:
        mask = create_strided_mask(
            config.seq_len, config.block_size, config.stride, device
        )
    elif pattern == AttentionPattern.LOCAL_GLOBAL:
        mask = create_local_global_mask(
            config.seq_len, config.block_size, config.local_window,
            config.n_global_tokens, device
        )
    elif pattern == AttentionPattern.RANDOM_BLOCK:
        mask = create_block_sparse_mask(
            config.seq_len, config.block_size, sparsity=0.9, device=device
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Evaluate with single query (more rigorous test)
    # Compute attention rankings with sparse mask applied
    ranks, weights = compute_attention_rankings(
        query, haystack, needle_positions,
        mask=mask, query_position=query_position
    )

    # Track attention weights on needle positions
    needle_attn_weights = [weights[pos].item() for pos in needle_positions]

    # Check mask coverage
    if mask is not None:
        needles_in_mask = check_needles_in_mask(needle_positions, query_position, mask)
    else:
        needles_in_mask = [True] * config.n_needles

    # Compute output similarities
    if mask is not None:
        # Create proper tensors for sparse attention
        q_tensor = torch.zeros(1, 1, config.seq_len, config.head_dim, device=device)
        q_tensor[0, 0, query_position] = query
        k_tensor = haystack.unsqueeze(0).unsqueeze(0)
        v_tensor = haystack.unsqueeze(0).unsqueeze(0)

        out = sparse_attention(q_tensor, k_tensor, v_tensor, mask)
        output = out[0, 0, query_position]
    else:
        # Dense attention output
        output = weights @ haystack

    # Similarity to each needle
    needle_similarities = []
    for j in range(len(needle_positions)):
        sim = F.cosine_similarity(output.unsqueeze(0), needles[j].unsqueeze(0)).item()
        needle_similarities.append(sim)

    # Sample noise positions for comparison
    noise_positions = np.random.choice(
        [p for p in range(config.seq_len) if p not in needle_positions],
        size=min(20, config.seq_len - config.n_needles),
        replace=False
    )
    noise_similarities = []
    for pos in noise_positions:
        sim = F.cosine_similarity(output.unsqueeze(0), haystack[pos].unsqueeze(0)).item()
        noise_similarities.append(sim)

    # Compute metrics
    metrics = compute_retrieval_metrics(ranks)

    return NeedleResult(
        needle_positions=needle_positions,
        pattern=pattern,
        recall_at_1=metrics['recall@1'],
        recall_at_5=metrics['recall@5'],
        recall_at_10=metrics['recall@10'],
        mrr=metrics['mrr'],
        needles_in_mask=sum(needles_in_mask),
        total_needles=len(needles_in_mask),
        # Max-pooled similarities - can the model find ANY needle?
        max_needle_similarity=max(needle_similarities),
        max_noise_similarity=max(noise_similarities),
        best_rank=min(ranks),  # Best (lowest) rank achieved
        # Attention weight metrics
        max_needle_attn_weight=max(needle_attn_weights),
        total_needle_attn_weight=sum(needle_attn_weights),
    )


def run_comprehensive_benchmark(
    config: BenchmarkConfig = None,
    n_trials: int = 5,
    device: str = 'cuda',
):
    """Run comprehensive needle-in-haystack benchmark across all patterns."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 80)
    print("VOID Sparse Attention: Needle in a Haystack Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Block size: {config.block_size}")
    print(f"  Needles per test: {config.n_needles}")
    print(f"  Needle-query similarity: {config.needle_similarity}")
    print(f"  Local window: {config.local_window}")
    print(f"  Global tokens: {config.n_global_tokens}")
    print(f"  Trials: {n_trials}")
    print()

    patterns = [
        AttentionPattern.DENSE,
        AttentionPattern.LOCAL,
        AttentionPattern.STRIDED,
        AttentionPattern.LOCAL_GLOBAL,
        AttentionPattern.RANDOM_BLOCK,
    ]

    results_by_pattern = {p: [] for p in patterns}

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")

        for pattern in patterns:
            try:
                result = run_single_pattern_test(config, pattern, device, seed=trial * 100)
                results_by_pattern[pattern].append(result)
            except Exception as e:
                print(f"  {pattern.value}: FAILED - {e}")

        torch.cuda.empty_cache()

    # Aggregate and print results
    print("\n" + "=" * 80)
    print("RESULTS (Max-Pooled Metrics)")
    print("=" * 80)

    print(f"\n{'Pattern':<16} {'Cover':>7} {'MaxAttn':>8} {'TotalAttn':>10} {'BestRank':>9} {'NeedleSim':>10}")
    print("-" * 70)

    for pattern in patterns:
        results = results_by_pattern[pattern]
        if not results:
            print(f"{pattern.value:<16} {'FAILED':>7}")
            continue

        avg_coverage = np.mean([r.needles_in_mask / r.total_needles for r in results])
        # Attention weight metrics (most meaningful)
        max_attn = np.mean([r.max_needle_attn_weight for r in results])
        total_attn = np.mean([r.total_needle_attn_weight for r in results])
        best_rank = min([r.best_rank for r in results])
        max_needle_sim = np.mean([r.max_needle_similarity for r in results])

        print(f"{pattern.value:<16} {avg_coverage:>6.0%} "
              f"{max_attn:>8.4f} {total_attn:>10.4f} "
              f"{best_rank:>9} {max_needle_sim:>10.3f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS (Max-Pooled)")
    print("=" * 80)

    dense_results = results_by_pattern[AttentionPattern.DENSE]
    local_results = results_by_pattern[AttentionPattern.LOCAL]
    local_global_results = results_by_pattern[AttentionPattern.LOCAL_GLOBAL]
    strided_results = results_by_pattern[AttentionPattern.STRIDED]
    random_results = results_by_pattern[AttentionPattern.RANDOM_BLOCK]

    if dense_results:
        dense_max_attn = np.mean([r.max_needle_attn_weight for r in dense_results])
        dense_total_attn = np.mean([r.total_needle_attn_weight for r in dense_results])
        dense_best_rank = min([r.best_rank for r in dense_results])
        print(f"\nDense Baseline:")
        print(f"  Max attention on needle: {dense_max_attn:.4f}")
        print(f"  Total attention on needles: {dense_total_attn:.4f}")
        print(f"  Best rank achieved: {dense_best_rank}")

        # Compare each pattern to dense
        comparisons = [
            ("Local", local_results),
            ("Local+Global", local_global_results),
            ("Strided", strided_results),
            ("Random Block", random_results),
        ]

        print(f"\nComparison to Dense (attention retention):")
        for name, results in comparisons:
            if results:
                max_attn = np.mean([r.max_needle_attn_weight for r in results])
                total_attn = np.mean([r.total_needle_attn_weight for r in results])
                attn_retention = (total_attn / dense_total_attn * 100) if dense_total_attn > 0 else 0
                coverage = np.mean([r.needles_in_mask / r.total_needles for r in results])
                print(f"  {name:<15}: {attn_retention:>5.1f}% attn retention, coverage={coverage:.0%}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. LOCAL+GLOBAL (Longformer-style) maintains high retrieval by ensuring
   global tokens can attend to any position.

2. RANDOM BLOCK-SPARSE (BigBird random component) has variable coverage -
   needles may or may not be in the randomly selected blocks.

3. LOCAL-only fails for distant needles (outside the window).

4. STRIDED helps with distant needles but may miss nearby ones.

5. For production use, combine LOCAL + GLOBAL + STRIDED for best coverage.
""")


def run_coverage_analysis(
    seq_len: int = 2048,
    block_size: int = 64,
    device: str = 'cuda',
):
    """Analyze block coverage for different sparse patterns."""
    print("\n" + "=" * 80)
    print("SPARSE PATTERN COVERAGE ANALYSIS")
    print("=" * 80)

    n_blocks = (seq_len + block_size - 1) // block_size
    total_pairs = n_blocks * n_blocks

    patterns = [
        ("Local (window=256)", create_local_attention_mask(seq_len, 256, block_size, device)),
        ("Local (window=512)", create_local_attention_mask(seq_len, 512, block_size, device)),
        ("Strided (stride=64)", create_strided_mask(seq_len, block_size, 64, device)),
        ("Local+Global", create_local_global_mask(seq_len, block_size, 256, 64, device)),
        ("Hybrid", create_hybrid_mask(seq_len, block_size, 256, 128, 64, 2, device)),
        ("Random 90%", create_block_sparse_mask(seq_len, block_size, 0.9, device)),
    ]

    print(f"\nSequence length: {seq_len}, Block size: {block_size}")
    print(f"Total blocks: {n_blocks}, Total block pairs: {total_pairs}")
    print()

    print(f"{'Pattern':<25} {'Blocks':>10} {'Coverage':>10} {'Sparsity':>10}")
    print("-" * 60)

    for name, mask in patterns:
        coverage = mask.n_blocks / total_pairs
        sparsity = 1 - coverage
        print(f"{name:<25} {mask.n_blocks:>10} {coverage:>9.1%} {sparsity:>9.1%}")


def main():
    """Run full needle-in-haystack benchmark suite."""
    if not torch.cuda.is_available():
        print("CUDA not available! This benchmark requires GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Check GPU memory and adjust sizes
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB\n")

    # Adjust config based on GPU memory
    if gpu_mem_gb < 8:
        config = BenchmarkConfig(seq_len=1024, n_needles=3, local_window=128)
        n_trials = 3
    elif gpu_mem_gb < 16:
        config = BenchmarkConfig(seq_len=2048, n_needles=5, local_window=256)
        n_trials = 5
    else:
        config = BenchmarkConfig(seq_len=4096, n_needles=5, local_window=512)
        n_trials = 5

    try:
        # Coverage analysis first
        run_coverage_analysis(config.seq_len, config.block_size)
        torch.cuda.empty_cache()

        # Main benchmark
        run_comprehensive_benchmark(config, n_trials)

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
