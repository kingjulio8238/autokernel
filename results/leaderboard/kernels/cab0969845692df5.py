#!/usr/bin/env python3
"""
Triton kernel for computing matrix multiplication C = A @ B.

This module defines a single fused Triton kernel that:
  - Loads tiles from matrix A and matrix B using tl.load with proper boundary masks.
  - Performs a tiled dot-product using tl.dot to accumulate into a local accumulator.
  - Stores the computed output tile into matrix C using tl.store with the appropriate mask.

The fused kernel performs all the loading, computing, and storing in one pass without
any intermediate PyTorch compute operations. The Python wrapper, kernel_function, is the
entry point that validates inputs, allocates the output tensor, configures the grid,
and launches the Triton kernel.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   stride_am: tl.constexpr, stride_ak: tl.constexpr,
                   stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                   stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B where:
      A: [M, K]
      B: [K, N]
      C: [M, N]

    Each instance of this kernel computes one TILE of size (BLOCK_M x BLOCK_N) of the
    output matrix C.

    Fusion details:
      - The kernel fuses the tile loading (from A and B),
        the dot-product computation using tl.dot, and
        the storing of the computed tile into C.
      - All operations run entirely within Triton using tl.load, tl.dot, and tl.store.
    """
    # Identify the tile this program instance is responsible for.
    pid_m = tl.program_id(0)  # Row-block index.
    pid_n = tl.program_id(1)  # Column-block index.

    # Compute the starting indices for the block in C.
    block_row = pid_m * BLOCK_M
    block_col = pid_n * BLOCK_N

    # Compute the row and column indices for the output tile.
    offs_m = block_row + tl.arange(0, BLOCK_M)
    offs_n = block_col + tl.arange(0, BLOCK_N)

    # Initialize the accumulator for the output tile.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in tiles of size BLOCK_K.
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for tk in range(0, num_k_tiles):
        k_base = tk * BLOCK_K
        offs_k = k_base + tl.arange(0, BLOCK_K)

        # Load a tile from matrix A of shape (BLOCK_M, BLOCK_K):
        #   A is stored in row-major order with shape [M, K]
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )

        # Load a tile from matrix B of shape (BLOCK_K, BLOCK_N):
        #   B is stored in row-major order with shape [K, N]
        b_tile = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )

        # Update the accumulator: acc += a_tile @ b_tile
        acc = tl.dot(a_tile, b_tile, acc)

    # Store the computed tile into matrix C with proper boundary checks.
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to perform square matrix multiplication using Triton.
    
    This function computes C = A @ B for two square matrices using a fused Triton kernel.
    It validates the inputs, allocates the output tensor, and performs grid configuration before
    launching the kernel.

    Fusion Details:
      - Tile loading from A and B, the dot-product accumulation, and the result storage are fused
        into a single Triton kernel launch.
      - All numerical computations are done entirely within the Triton kernel (using tl.load,
        tl.dot, and tl.store) without any PyTorch compute operations.

    Args:
        a (torch.Tensor): Matrix A of shape (M, K) in float32 on CUDA.
        b (torch.Tensor): Matrix B of shape (K, N) in float32 on CUDA.
    
    Returns:
        torch.Tensor: The result matrix C of shape (M, N) in float32.
    """
    # Ensure inputs are 2D CUDA tensors with float32 dtype.
    assert a.dim() == 2 and b.dim() == 2, "Only 2D matrices are supported."
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "A's number of columns must equal B's number of rows."
    assert a.is_cuda and b.is_cuda, "Input tensors must be on CUDA."
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only float32 dtype is supported."

    # Allocate output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Choose tile sizes (powers of two for optimal performance).
    BLOCK_M = 128  # Number of rows processed per block.
    BLOCK_N = 128  # Number of columns processed per block.
    BLOCK_K = 32   # Number of reduction elements per block.

    # Configure a 2D grid: one block covers a tile of C.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the fused Triton kernel.
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return c


# For local testing (this will not run during automated tests if imported as a module)
