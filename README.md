<img width="3168" height="1344" alt="VOID Banner" src="https://github.com/user-attachments/assets/758001dc-62c7-49b7-9e36-d20c7d3c3676" />

# VOID (FlashSparse)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-134%20passed-brightgreen.svg)]()

**High-performance block-sparse matrix operations for GPUs**

VOID achieves **3-15x speedups** over cuSPARSE and dense baselines by combining:

- **Block tiling** — Fixed-size dense tiles (32x32) for Tensor Core utilization
- **Morton ordering** — Z-curve spatial locality for cache efficiency
- **Triton kernels** — Custom fused kernels with FP32 accumulation
- **Software pipelining** — Async memory prefetching for latency hiding

## Performance Highlights

| Use Case | Speedup | Comparison |
|----------|---------|------------|
| Sparse Attention (Dilated) | **15x** | vs Dense |
| SpMM (Block-sparse) | **3.2x** | vs cuSPARSE BSR |
| 2:4 Structured Sparsity | **7x** | vs cuSPARSE |
| Fused SpMM+GELU | **1.5x** | vs Unfused |

## Features

### Core Operations
- **SpMM/SpMV** — Sparse-dense matrix multiplication with autotuning
- **Sparse Attention** — FlashAttention-style block-sparse patterns
- **Autograd Support** — Full backward pass for training
- **Stream-K Load Balancing** — Handles power-law row distributions

### Advanced Features
- **2:4 Structured Sparsity** — NVIDIA Sparse Tensor Core support
- **INT8/INT4 Quantization** — Per-block quantized inference
- **Dynamic Sparsity** — Runtime-adaptive patterns (top-k attention)
- **Multi-GPU Distribution** — Row/column/block sharding strategies
- **FP8 Support** — E4M3/E5M2 formats with automatic scaling
- **Operation Fusion** — SpMM + activation in single kernel
- **Pipelined Kernels** — 3-5 stage async memory prefetching

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .

# Development
uv pip install -e ".[dev]"
```

## Quick Start

### Basic SpMM

```python
import torch
import scipy.sparse as sp
from void import csr_to_void, void_spmm

# Convert scipy sparse matrix to VOID format
sparse_np = sp.random(512, 512, density=0.1, format='csr')
void_tensor = csr_to_void(sparse_np, tile_size=32).cuda()

# Sparse-dense matrix multiplication
B = torch.randn(512, 128, device='cuda')
C = void_spmm(void_tensor, B)
```

### Sparse Attention

```python
from void import local_attention

q = torch.randn(2, 8, 2048, 64, device='cuda')
k = torch.randn(2, 8, 2048, 64, device='cuda')
v = torch.randn(2, 8, 2048, 64, device='cuda')

# Sliding window attention (14x faster than dense at seq_len=2048)
out = local_attention(q, k, v, window_size=256)
```

### 2:4 Structured Sparsity

```python
from void import prune_to_2_4, void_to_structured

# Prune weights to 2:4 pattern (50% sparsity)
pruned = prune_to_2_4(weights)

# Convert for Tensor Core acceleration
structured = void_to_structured(void_tensor)
```

### INT8 Quantization

```python
from void import quantize_void_tensor, void_spmm_int8, IntQuantConfig, IntFormat

# Quantize sparse matrix to INT8
config = IntQuantConfig(format=IntFormat.INT8, symmetric=True)
int8_tensor = quantize_void_tensor(void_tensor, config)

# INT8 SpMM (2-4x faster for inference)
C = void_spmm_int8(int8_tensor, B_int8, B_scale)
```

### Dynamic Sparsity

```python
from void import dynamic_topk_attention

# Top-k sparse attention (keeps only k highest scores per query)
out = dynamic_topk_attention(q, k, v, top_k=256)
```

### Multi-GPU

```python
from void import shard_void_tensor, distributed_void_spmm, ShardingStrategy

# Shard across GPUs
dist_tensor = shard_void_tensor(void_tensor, ShardingStrategy.ROW_WISE)

# Distributed SpMM with automatic all-reduce
C = distributed_void_spmm(dist_tensor, B)
```

### Fused Operations

```python
from void import void_spmm_gelu, FusedSparseLinear

# Single kernel: C = GELU(A @ B)
C = void_spmm_gelu(void_tensor, B)

# Drop-in replacement for nn.Linear
layer = FusedSparseLinear(512, 256, void_tensor, activation="gelu")
```

## Benchmarks

### SpMM: VOID vs cuSPARSE BSR

| Block Sparsity | VOID | BSR | Speedup |
|----------------|------|-----|---------|
| 70% | 0.28ms | 0.91ms | **3.3x** |
| 80% | 0.19ms | 0.61ms | **3.3x** |
| 90% | 0.12ms | 0.32ms | **2.6x** |
| 95% | 0.06ms | 0.18ms | **2.9x** |
| 98% | 0.04ms | 0.16ms | **3.8x** |

*4096x4096 matrix, N=512, RTX GPU*

### Sparse Attention vs Dense

| Pattern | Seq=1024 | Seq=2048 |
|---------|----------|----------|
| Dilated | 3.9x | **15.0x** |
| Sliding-256 | 4.4x | **9.7x** |
| Causal (GPT) | 2.9x | **4.6x** |
| BigBird | 1.9x | **2.9x** |

### Memory Bandwidth Efficiency

| Access Pattern | Bandwidth | vs CSR |
|---------------|-----------|--------|
| Scalar CSR | 54 GB/s | 1x |
| Block 32x32 | 850 GB/s | **16x** |
| Block 64x64 | 1759 GB/s | **32x** |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│   Input     │    │ VOID Format  │    │ Kernel Select  │
│ CSR/Dense   │ -> │ Morton Order │ -> │ Autotuning     │
│ Attention   │    │ TC Alignment │    │ Stream-K       │
└─────────────┘    └──────────────┘    └────────────────┘
       |                  |                    |
       v                  v                    v
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  Core Ops   │    │  Features    │    │    Output      │
│ void_spmm() │ <- │ 2:4 / INT8   │ -> │ Dense Result   │
│ attention() │    │ Dynamic/Dist │    │ Gradients      │
└─────────────┘    └──────────────┘    └────────────────┘
```

VOID stores sparse matrices as collections of dense tiles with Morton (Z-curve) ordering for optimal cache locality.

## Running Benchmarks

```bash
# Core validation
uv run python benchmarks/validation.py

# BSR comparison (primary benchmark)
uv run python benchmarks/bsr_comparison.py

# Attention patterns
uv run python benchmarks/attention_benchmark.py

# Generate visualizations
uv run python benchmarks/visualize_results.py
```

## Tests

```bash
# All tests (requires GPU)
uv run pytest tests/ -v

# Quick check
uv run pytest tests/ -v --tb=short
```

## API Reference

<details>
<summary><b>Core Format</b></summary>

```python
from void import csr_to_void, dense_to_void, VOIDTensor

# From scipy CSR
void_tensor = csr_to_void(csr_matrix, tile_size=32)

# From dense with threshold
void_tensor = dense_to_void(dense, tile_size=32, threshold=0.01)

# Dtype conversion
void_fp16 = void_tensor.half()
void_bf16 = void_tensor.bfloat16()
```
</details>

<details>
<summary><b>SpMM Operations</b></summary>

```python
from void import void_spmm, void_spmm_autotuned, void_spmm_pipelined

C = void_spmm(A, B)                    # Basic SpMM
C = void_spmm_autotuned(A, B)          # With autotuning
C = void_spmm_pipelined(A, B)          # Async pipelining
```
</details>

<details>
<summary><b>Attention</b></summary>

```python
from void import (
    sparse_attention, local_attention, block_sparse_attention,
    create_local_attention_mask, create_causal_local_mask
)

out = sparse_attention(q, k, v, mask)
out = local_attention(q, k, v, window_size=256)
out = block_sparse_attention(q, k, v, sparsity=0.9)
```
</details>

<details>
<summary><b>Quantization</b></summary>

```python
from void import (
    quantize_void_tensor, void_spmm_int8,
    void_tensor_to_fp8, void_spmm_fp8
)

# INT8
int8_tensor = quantize_void_tensor(void_tensor, config)
C = void_spmm_int8(int8_tensor, B, scale)

# FP8
fp8_tensor = void_tensor_to_fp8(void_tensor)
C = void_spmm_fp8(fp8_tensor, B)
```
</details>

<details>
<summary><b>Neural Networks</b></summary>

```python
from void import SparseLinear, FusedSparseLinear, VOIDSpMM

# Drop-in nn.Linear replacement
layer = SparseLinear(512, 256, void_tensor, bias=True)

# With fused activation
layer = FusedSparseLinear(512, 256, void_tensor, activation="gelu")
```
</details>

## Citation

```bibtex
@software{void2025,
  title = {VOID: High-Performance Block-Sparse Matrix Operations for GPUs},
  author = {Khushiyant},
  year = {2025},
  url = {https://github.com/khushiyant/void}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Triton](https://github.com/openai/triton) — GPU programming framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Tiled attention inspiration
- [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) — Baseline comparison
