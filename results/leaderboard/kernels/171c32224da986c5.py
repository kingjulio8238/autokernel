#!/usr/bin/env python3
"""
Triton kernel implementation for computing matrix multiplication C = A @ B.

This file defines a single fused Triton kernel that performs a tiled GEMM
on two input matrices. The kernel loads tiles from A and B with proper masking,
performs a dot product accumulation (all in fp32), and stores out the resulting tile.

IMPORTANT: All compute is performed with Triton operations (tl.load, tl.dot, tl.store, etc.).
No PyTorch tensor‐tensor operations (e.g. torch.matmul, torch.mm, etc.) are used in the compute path.
Only tensor allocation, dtype/device checks, and launch configuration use PyTorch.

The only function to be imported externally is "kernel_function".
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   stride_am: tl.constexpr, stride_ak: tl.constexpr,
                   stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                   stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                   BLOCK_SIZE_K: tl.constexpr):
    # Identify the tile indices for the M (rows) and N (columns) dimensions.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row and column indices for this output tile.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize the accumulator in fp32.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K in chunks of BLOCK_SIZE_K.
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_k_tiles):
        # Determine the offset indices in the K dimension.
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Compute pointers into A and B for the current tile.
        a_tile_ptr = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_tile_ptr = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Establish masks to handle boundaries.
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load data from global memory using the computed pointers and masks.
        a_tile = tl.load(a_tile_ptr, mask=mask_a, other=0.0)
        b_tile = tl.load(b_tile_ptr, mask=mask_b, other=0.0)
        
        # Accumulate the partial dot product result.
        acc += tl.dot(a_tile, b_tile)

    # Compute the pointer for storing the resulting tile in C.
    c_tile_ptr = c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_tile_ptr, acc, mask=mask_c)

def kernel_function(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix multiplication C = A @ B using a fused Triton kernel.

    Args:
        tensor_a (torch.Tensor): Input matrix A with shape (M, K) and dtype torch.float32.
        tensor_b (torch.Tensor): Input matrix B with shape (K, N) and dtype torch.float32.

    Returns:
        torch.Tensor: Output matrix C with shape (M, N) and dtype torch.float32.
        
    Notes:
        - Input tensors must reside on CUDA.
        - All computation (load → dot accumulate → store) happens in the Triton kernel.
          No PyTorch tensor-tensor operations (such as torch.matmul) are used.
    """
    # Validate that inputs are on CUDA and have compatible shapes and dtypes.
    assert tensor_a.is_cuda and tensor_b.is_cuda, "Input tensors must be on CUDA."
    M, K1 = tensor_a.shape
    K2, N = tensor_b.shape
    assert K1 == K2, f"Incompatible matrix dimensions: {K1} vs {K2}"
    assert tensor_a.dtype == torch.float32 and tensor_b.dtype == torch.float32, \
           "Only torch.float32 dtype is supported for this kernel."
    K = K1

    # Allocate the output tensor.
    output = torch.empty((M, N), device=tensor_a.device, dtype=tensor_a.dtype)

    # Tile sizes (tunable for performance).
    BLOCK_SIZE_M = 128  # Number of rows processed per kernel block.
    BLOCK_SIZE_N = 128  # Number of columns processed per kernel block.
    BLOCK_SIZE_K = 32   # Tile size in the reduction (K) dimension.

    # Configure the grid: one instance per tile of C.
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

# When executed as a script, perform a minimal test that only compiles and launches the kernel.
