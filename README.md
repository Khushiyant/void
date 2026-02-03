<img width="3168" height="1344" alt="Gemini_Generated_Image_435v3s435v3s435v" src="https://github.com/user-attachments/assets/758001dc-62c7-49b7-9e36-d20c7d3c3676" />

# VOID (FlashSparse)


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cache-aware block tiling for sparse matrix operations on GPUs**

VOID is a high-performance sparse matrix library that combines FlashAttention-style tiling with structured sparsity for efficient GPU computation. It achieves significant speedups over cuSPARSE by leveraging:

- **Block tiling** - Fixed-size dense tiles (32x32) for Tensor Core utilization
- **Morton ordering** - Z-curve spatial locality for cache efficiency
- **Triton kernels** - Custom fused kernels with FP32 accumulation for numerical stability
- **Mixed precision** - Full FP16/BF16 support with automatic precision handling

## Features

### Core Operations
- **SpMM/SpMV Operations** - Sparse-dense matrix multiplication with autotuning
- **Sparse Attention** - FlashAttention-style block-sparse attention patterns
- **Autograd Support** - Full backward pass with fused Triton kernels
- **Stream-K Load Balancing** - Handles power-law row distributions
- **Mixed Precision** - FP32, FP16, and BF16 with FP32 accumulation

## Installation

### Using uv (recommended)

```bash
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

### Development installation

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
import torch
import scipy.sparse as sp
from void import csr_to_void, void_spmm, SparseLinear

# Convert scipy sparse matrix to VOID format
sparse_np = sp.random(512, 512, density=0.1, format='csr')
void_tensor = csr_to_void(sparse_np, tile_size=32, dtype=torch.float16).cuda()

# Sparse-dense matrix multiplication
B = torch.randn(512, 128, device='cuda', dtype=torch.float16)
C = void_spmm(void_tensor, B)

# Use in neural networks with autograd
layer = SparseLinear(512, 256, void_tensor, bias=True).cuda()
output = layer(torch.randn(32, 512, device='cuda', dtype=torch.float16))
```

### Sparse Attention

```python
from void import local_attention, create_local_attention_mask

# Local (sliding window) attention
q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)

# Window size of 128 tokens
out = local_attention(q, k, v, window_size=128, block_size=64)
```

## API Reference

### Core Format

```python
# Convert CSR to VOID format
void_tensor = csr_to_void(
    matrix,           # scipy CSR sparse matrix
    tile_size=32,     # Block size (default: 32)
    dtype=torch.float32,  # Output dtype
    device='cpu',     # Output device
)

# Convert dense to VOID (with sparsity threshold)
void_tensor = dense_to_void(
    matrix,           # Dense 2D tensor
    tile_size=32,     # Block size
    threshold=0.0,    # Minimum L1 norm to keep tile
)

# Dtype conversion
void_fp16 = void_tensor.half()
void_bf16 = void_tensor.bfloat16()
void_fp32 = void_tensor.float()
```

### Sparse Operations

```python
# SpMM: C = A @ B
C = void_spmm(void_tensor, B)

# SpMV: y = A @ x
y = void_spmv(void_tensor, x)

# Autotuned SpMM (finds optimal tile size)
C = void_spmm_autotuned(void_tensor, B)

# Stream-K load-balanced SpMM (for power-law distributions)
C = void_spmm_stream_k(void_tensor, B)
```

### Neural Network Modules

```python
# Sparse matrix multiplication module with autograd
module = VOIDSpMM(void_tensor, requires_grad=True)
output = module(input)

# Sparse linear layer (drop-in replacement for nn.Linear)
layer = SparseLinear(
    in_features=512,
    out_features=256,
    void_tensor=void_tensor,
    bias=True,
    dtype=torch.float16,  # Optional: match weight dtype
)
```

### Sparse Attention

```python
# Generic sparse attention with custom mask
from void import sparse_attention, SparseAttentionMask

mask = create_local_attention_mask(seq_len=1024, window_size=128, block_size=64)
out = sparse_attention(q, k, v, mask)

# Convenience functions
out = local_attention(q, k, v, window_size=128, causal=False)
out = block_sparse_attention(q, k, v, sparsity=0.9)
```

### Fused Operations

```python
from void import void_spmm_gelu, void_spmm_relu, fused_sparse_mlp, FusedSparseLinear

# Fused SpMM + activation (single kernel)
C = void_spmm_gelu(void_tensor, B)  # C = GELU(A @ B)
C = void_spmm_relu(void_tensor, B, bias=bias)  # C = ReLU(A @ B + bias)

# Fused sparse MLP
y = fused_sparse_mlp(x, W1, W2, activation="gelu", bias1=b1, bias2=b2)

# Drop-in replacement for nn.Linear with fused activation
layer = FusedSparseLinear(in_features=512, out_features=256,
                          void_tensor=void_tensor, activation="gelu")
```

### FP8 Quantization

```python
from void import void_tensor_to_fp8, void_spmm_fp8, FP8Config

# Convert to FP8 (E4M3 format)
fp8_tensor = void_tensor_to_fp8(void_tensor, format="e4m3")

# FP8 SpMM with automatic scaling
C = void_spmm_fp8(fp8_tensor, B, output_dtype=torch.float16)
```

### Dynamic Dispatch

```python
from void import void_spmm_auto, get_recommended_kernel

# Automatic kernel selection based on sparsity pattern
C = void_spmm_auto(void_tensor, B, activation="gelu")

# Get recommendation without executing
variant, reason = get_recommended_kernel(void_tensor, B.shape)
print(f"Recommended: {variant}, Reason: {reason}")
```

## Benchmarks

### SpMM Performance (RTX 5070 Ti, 4096x4096 matrix, N=512)

**VOID vs cuSPARSE BSR (Block Sparse Row)** - the fair comparison since both use blocks:

| Block Sparsity | VOID | BSR | CSR | vs BSR | vs CSR |
|----------------|------|-----|-----|--------|--------|
| 70% | 0.28ms | 0.91ms | 4.57ms | **3.29x** | 16.6x |
| 80% | 0.19ms | 0.61ms | 2.87ms | **3.26x** | 15.4x |
| 90% | 0.12ms | 0.32ms | 2.10ms | **2.61x** | 17.3x |
| 95% | 0.06ms | 0.18ms | 0.68ms | **2.91x** | 10.8x |
| 98% | 0.04ms | 0.16ms | 0.29ms | **3.77x** | 7.0x |

**Average: 3.17x faster than BSR, 13.4x faster than CSR**

### Sparse Attention Performance

| Sequence Length | Dense | Block-Sparse (90%) | Speedup |
|-----------------|-------|-------------------|---------|
| 1024 | 1.03ms | 0.22ms | **4.7x** |
| 2048 | 4.47ms | 0.31ms | **14.3x** |

### Kernel Fusion Performance

| Operation | Unfused | Fused | Speedup |
|-----------|---------|-------|---------|
| SpMM + ReLU | 0.53ms | 0.36ms | **1.46x** |
| SpMM + GELU | 0.40ms | 0.35ms | **1.13x** |
| FusedSparseLinear | 0.24ms | 0.20ms | **1.24x** |

*Benchmarks run with FP32, averaged over 100 iterations. Block sparsity = fraction of empty 32x32 tiles.*

> **Note**: VOID is designed for **block-sparse** patterns where entire tiles are zero. Random element sparsity (e.g., 90% zeros scattered randomly) does not create block sparsity and won't benefit from VOID.

## Architecture

### VOID Format

VOID stores sparse matrices as a collection of dense tiles:

```
Original Sparse Matrix          VOID Representation
┌─────────────────────┐         ┌────────────────────────────┐
│ ░░██░░░░░░██░░░░░░  │         │ values: [n_blocks, 32, 32] |
│ ░░██░░░░░░██░░░░░░  │   →     │ block_rows: [n_blocks]     |
│ ░░░░░░██░░░░░░░░░░  │         │ block_cols: [n_blocks]     |
│ ░░░░░░██░░░░░░░░░░  │         │ morton_codes: [n_blocks]   |
└─────────────────────┘         └────────────────────────────┘
```

Key properties:
- **Block decomposition**: Fixed 32x32 tiles for Tensor Core alignment
- **Morton ordering**: Z-curve ordering for spatial cache locality
- **Compressed metadata**: CSR-style block pointers for efficient traversal

### Triton Kernels

All kernels use:
- **FP32 accumulation**: Prevents numerical instability in mixed precision
- **Block pointers**: Efficient memory access patterns
- **Autotuning**: Runtime optimization of tile sizes and warps

## Development

### Running Tests

```bash
# All tests (requires GPU)
pytest tests/ -v

# CPU-only tests
pytest tests/ -v -k "not cuda"

# Specific test file
pytest tests/test_dtype.py -v
```

### Linting

```bash
ruff check void/ tests/
ruff format void/ tests/
```

## Running Benchmarks

```bash
# BSR comparison (critical benchmark)
python benchmarks/bsr_comparison.py

# Attention benchmark
python benchmarks/attention_benchmark.py

# Fusion benchmark
python benchmarks/fusion_benchmark.py

# Full validation
python benchmarks/validation.py
```

## Citation

If you use VOID in your research, please cite:

```bibtex
@software{void2025,
  title = {VOID: Cache-aware Block Tiling for Sparse Matrix Operations},
  author = {Khushiyant},
  year = {2025},
  url = {https://github.com/khushiyant/void}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Triton](https://github.com/openai/triton) - GPU programming framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Inspiration for tiled attention
- [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) - Baseline comparison
