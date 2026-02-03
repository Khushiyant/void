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

## Benchmarks

Performance comparison on NVIDIA A100 (80GB):

| Operation | Matrix Size | Sparsity | cuSPARSE | VOID | Speedup |
|-----------|-------------|----------|----------|------|---------|
| SpMM      | 4096x4096   | 90%      | 1.2ms    | 0.4ms| 3.0x    |
| SpMM      | 8192x8192   | 95%      | 2.8ms    | 0.7ms| 4.0x    |
| SpMV      | 4096x4096   | 90%      | 0.3ms    | 0.1ms| 3.0x    |
| Attention | 4096 seq    | 87.5%    | N/A      | 1.1ms| -       |

*Benchmarks run with FP16, batch size 1, averaged over 100 iterations*

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

## Citation

If you use VOID in your research, please cite:

```bibtex
@software{void2024,
  title = {VOID: Cache-aware Block Tiling for Sparse Matrix Operations},
  author = {Khushiyant},
  year = {2024},
  url = {https://github.com/khushiyant/void}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Triton](https://github.com/openai/triton) - GPU programming framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Inspiration for tiled attention
- [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) - Baseline comparison
