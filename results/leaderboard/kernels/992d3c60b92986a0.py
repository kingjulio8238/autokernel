#!/usr/bin/env python
"""
Square Matrix Multiplication Kernel using Triton

This module implements a fused Triton kernel for computing the square
matrix multiplication:

    C = A @ B

where A, B, and C are square matrices of shape (N, N) with dtype torch.float32.

All numerical computations (tile loads, dot-products, and accumulation)
are performed inside the Triton kernel using triton.language operations,
with proper masking for boundary conditions. The Python wrapper
(kernel_function) is responsible only for argument validation, tensor
allocation, and grid configuration.

Note:
  – No PyTorch tensor-tensor math (e.g. torch.matmul/mm/bmm/einsum) is used
    in the core kernel or its wrapper.
  – Only allocation, shape/dtype checks, and grid configuration use PyTorch.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_SIZE_M: tl.constexpr, 
                   BLOCK_SIZE_N: tl.constexpr, 
                   BLOCK_SIZE_K: tl.constexpr):
    # Each kernel instance computes a tile of C.
    # Determine the row and column tile indices from the program ID.
    pid_m = tl.program_id(0)  # tile row index
    pid_n = tl.program_id(1)  # tile column index

    # Compute global row indices for A and C for this tile.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Compute global column indices for B and C for this tile.
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize the accumulator with zeros.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the reduction dimension in chunks of BLOCK_SIZE_K.
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        # Load a tile of A of shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # Load a tile of B of shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_tile = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        # Perform the dot-product and accumulate the result.
        acc += tl.dot(a_tile, b_tile)

    # Store the computed tile to the output matrix C with proper masking.
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

def kernel_function(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Fused matrix multiplication using a Triton kernel.

    Computes the square matrix multiplication
      C = A @ B
    entirely in the Triton kernel.

    Args:
        tensor_a (torch.Tensor): Input matrix A of shape (N, N) with dtype torch.float32.
        tensor_b (torch.Tensor): Input matrix B of shape (N, N) with dtype torch.float32.

    Returns:
        torch.Tensor: Output matrix C of shape (N, N).
    """
    # Validate that inputs are 2D, CUDA tensors with dtype float32.
    if not (tensor_a.is_cuda and tensor_b.is_cuda):
        raise ValueError("Input tensors must be CUDA tensors.")
    if tensor_a.dtype != torch.float32 or tensor_b.dtype != torch.float32:
        raise ValueError("Input tensors must have dtype torch.float32.")
    if tensor_a.ndim != 2 or tensor_b.ndim != 2:
        raise ValueError("Input tensors must be 2D matrices.")

    M, K = tensor_a.shape
    K2, N = tensor_b.shape
    if K != K2:
        raise ValueError("Inner dimensions of A and B must match.")

    # Allocate the output tensor.
    output = torch.empty((M, N), device=tensor_a.device, dtype=tensor_a.dtype)

    # Choose compile-time block sizes. These can be tuned for performance.
    BLOCK_SIZE_M = 128  # Tile height.
    BLOCK_SIZE_N = 128  # Tile width.
    BLOCK_SIZE_K = 32   # Tile size for the reduction dimension.

    # Configure a 2D grid:
    #   - The first dimension covers the row tiles.
    #   - The second dimension covers the column tiles.
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the Triton kernel.
    _matmul_kernel[grid](
        tensor_a, tensor_b, output,
        M, N, K,
        tensor_a.stride(0), tensor_a.stride(1),
        tensor_b.stride(0), tensor_b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return output

# Optional minimal test when executed as a script (avoiding any PyTorch matmul calls).
