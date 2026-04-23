#!/usr/bin/env python3
"""
Triton GEMM Kernel Implementation

This module implements a fused GEMM (matrix multiplication) kernel using Triton.
It computes C = A * B for matrices A (shape MxK) and B (shape KxN), where all tensors
are of type float32. The computation is tiled and fused in a single kernel launch,
including:
  • Loading A and B tiles from global memory with proper masking for boundary conditions.
  • Computing the dot-product per tile using tl.dot.
  • Accumulating partial results.
  • Storing the resulting C tile back to global memory.

The wrapper function 'kernel_function' performs argument validation, sets up the grid,
and allocates the output tensor. All numerical work is performed inside the Triton kernel
using tl.load, tl.dot, and tl.store.
  
This implementation strictly adheres to the following guidelines:
  - Only Triton operations are used for numerical computation.
  - No PyTorch compute functions (such as torch.matmul) are used inside the kernel.
  - All memory accesses are coalesced and masked to handle boundary cases.
  
Usage:
    from kernel import kernel_function
    C = kernel_function(A, B)
    
The test code will call kernel_function with two float32 matrices.
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
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Triton kernel for computing C = A * B.
    
    Each kernel instance computes one tile of the output matrix C of shape (BLOCK_M, BLOCK_N).
    
    Parameters:
      a_ptr      : Pointer to matrix A (shape M x K)
      b_ptr      : Pointer to matrix B (shape K x N)
      c_ptr      : Pointer to output matrix C (shape M x N)
      M, N, K    : Matrix dimensions
      stride_am  : Row stride for A
      stride_ak  : Column stride for A
      stride_bk  : Row stride for B
      stride_bn  : Column stride for B
      stride_cm  : Row stride for C
      stride_cn  : Column stride for C
      BLOCK_M    : Block size in M dimension (compile-time constant)
      BLOCK_N    : Block size in N dimension (compile-time constant)
      BLOCK_K    : Block size in K dimension (compile-time constant)
    """
    # Get the block indices in the 2D output grid.
    pid_m = tl.program_id(0)  # Block row index
    pid_n = tl.program_id(1)  # Block column index

    # Compute the starting indices for this block.
    a_row = pid_m * BLOCK_M      # Starting row index in A and C
    b_col = pid_n * BLOCK_N      # Starting column index in B and C

    # Offsets within the tile.
    offs_m = tl.arange(0, BLOCK_M)  # Row offsets [0, 1, ..., BLOCK_M-1]
    offs_n = tl.arange(0, BLOCK_N)  # Column offsets [0, 1, ..., BLOCK_N-1]

    # Initialize the accumulator for the output tile.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension tiles.
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_tiles):
        a_k = k * BLOCK_K
        k_range = tl.arange(0, BLOCK_K)

        # Compute pointers for the current A tile (shape BLOCK_M x BLOCK_K).
        a_ptrs = a_ptr + ((a_row + offs_m)[:, None] * stride_am +
                          (a_k + k_range)[None, :] * stride_ak)
        # Compute pointers for the current B tile (shape BLOCK_K x BLOCK_N).
        b_ptrs = b_ptr + ((a_k + k_range)[:, None] * stride_bk +
                          (b_col + offs_n)[None, :] * stride_bn)

        # Create masks for boundary elements to prevent out-of-bound accesses.
        mask_a = ((a_row + offs_m)[:, None] < M) & ((a_k + k_range)[None, :] < K)
        mask_b = ((a_k + k_range)[:, None] < K) & ((b_col + offs_n)[None, :] < N)

        # Load tiles from global memory. Out-of-bound entries are set to 0.0.
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate the product of the loaded tiles.
        acc += tl.dot(a_tile, b_tile)

    # Compute the pointer for storing the C tile.
    c_ptrs = c_ptr + ((a_row + offs_m)[:, None] * stride_cm +
                      (b_col + offs_n)[None, :] * stride_cn)
    # Create the mask for the output tile.
    mask_c = ((a_row + offs_m)[:, None] < M) & ((b_col + offs_n)[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def kernel_function(tensor_a, tensor_b):
    """
    Wrapper for the Triton GEMM kernel that computes C = A * B.
    
    This function performs the following steps:
      1. Validates input tensor dtypes and shapes.
      2. Allocates the output tensor.
      3. Retrieves the matrix strides for pointer arithmetic.
      4. Configures the grid based on the matrix dimensions and chosen tile sizes.
      5. Launches the Triton kernel that fuses the loading, dot-product computation,
         and storing of each output tile.
    
    Parameters:
      tensor_a (torch.Tensor): Input matrix A of shape (M, K) in float32.
      tensor_b (torch.Tensor): Input matrix B of shape (K, N) in float32.
      
    Returns:
      torch.Tensor: Output matrix C of shape (M, N) in float32.
    """
    # Ensure that only float32 data type is used.
    if tensor_a.dtype != torch.float32 or tensor_b.dtype != torch.float32:
        raise ValueError("Only float32 data type is supported for both inputs.")
    
    M, K = tensor_a.shape
    K2, N = tensor_b.shape
    if K != K2:
        raise ValueError(f"Incompatible shapes: tensor_a.shape={tensor_a.shape}, tensor_b.shape={tensor_b.shape}.")

    # Allocate the output tensor.
    output = torch.empty((M, N), device=tensor_a.device, dtype=torch.float32)

    # Retrieve strides (in number of elements) for pointer arithmetic.
    stride_am, stride_ak = tensor_a.stride(0), tensor_a.stride(1)
    stride_bk, stride_bn = tensor_b.stride(0), tensor_b.stride(1)
    stride_cm, stride_cn = output.stride(0), output.stride(1)

    # Define tile sizes as compile-time constants.
    BLOCK_M = 128   # Number of rows computed per kernel instance.
    BLOCK_N = 128   # Number of columns computed per kernel instance.
    BLOCK_K = 32    # Tile size in the reduction dimension.

    # Configure a 2D grid: one program instance per output tile.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the Triton kernel.
    _matmul_kernel[grid](
        tensor_a, tensor_b, output,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return output

# End of kernel.py
