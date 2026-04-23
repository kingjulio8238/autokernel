#!/usr/bin/env python3
"""
Kernel implementation of square matrix multiplication (GEMM) using Triton.

This module implements a Triton kernel that computes:

    C = A @ B

for two square matrices (A and B) of shape (N, N) where N = 2048*2 as specified.
The kernel uses a tiled (blocked) GEMM approach. For each output tile, it:
  1. Loads a BLOCK_SIZE_M x BLOCK_SIZE_K sub-tile of A (with proper boundary masks)
  2. Loads a BLOCK_SIZE_K x BLOCK_SIZE_N sub-tile of B (with proper masking)
  3. Computes the multiplication using tl.dot and accumulates into an accumulator tile.
  4. Stores the computed tile to C (again using tl.store with boundary masks).

The wrapper function `kernel_function` handles the input validation, output 
allocation, and grid configuration, and then launches the Triton kernel.

Fusion Rationale:
  - The entire matrix multiplication is fused in one kernel: loading of tiles,
    computing the dot product, accumulation and storing of the result are all done
    in a single pass. This minimizes memory traffic and kernel launch overhead.

NOTE: All numerical math is performed using Triton operations (tl.load, tl.store, 
tl.dot, and tl.arange) inside the kernel; no PyTorch numerical compute is used here.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    ptr_a, ptr_b, ptr_c, N,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for computing a tile of the matrix multiplication C = A @ B.
    
    Each kernel instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile.
    It iterates over the k dimension in blocks of BLOCK_SIZE_K.
    
    Parameters:
      ptr_a        : pointer to matrix A
      ptr_b        : pointer to matrix B
      ptr_c        : pointer to output matrix C
      N            : size of the square matrices (N x N)
      stride_a0    : stride for rows of A
      stride_a1    : stride for columns of A
      stride_b0    : stride for rows of B
      stride_b1    : stride for columns of B
      stride_c0    : stride for rows of C
      stride_c1    : stride for columns of C
      BLOCK_SIZE_M : block size along M dimension (number of rows per tile)
      BLOCK_SIZE_N : block size along N dimension (number of columns per tile)
      BLOCK_SIZE_K : block size along reduction dimension K
    """
    # Determine the tile indices using a 2D grid.
    pid_m = tl.program_id(0)  # tile row index
    pid_n = tl.program_id(1)  # tile column index

    # Compute the starting indices for the C tile.
    a_row_start = pid_m * BLOCK_SIZE_M
    b_col_start = pid_n * BLOCK_SIZE_N

    # Create a (BLOCK_SIZE_M x BLOCK_SIZE_N) accumulator.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Create row and column offsets for the C tile.
    offs_m = a_row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = b_col_start + tl.arange(0, BLOCK_SIZE_N)

    # Loop over the k dimension in blocks.
    num_tiles_k = tl.cdiv(N, BLOCK_SIZE_K)
    for k in range(0, num_tiles_k):
        k_offset = k * BLOCK_SIZE_K
        # create offsets for k dimension.
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

        # Load A tile: shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = ptr_a + (offs_m[:, None] * stride_a0 + offs_k[None, :] * stride_a1)
        # Use boundaries to avoid out-of-bound memory access.
        mask_a = (offs_m[:, None] < N) & (offs_k[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B tile: shape [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_ptrs = ptr_b + (offs_k[:, None] * stride_b0 + offs_n[None, :] * stride_b1)
        mask_b = (offs_k[:, None] < N) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Compute the dot product and accumulate.
        acc += tl.dot(a_tile, b_tile)
    
    # Write the accumulator to the output matrix C using proper boundary checking.
    c_ptrs = ptr_c + (offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1)
    mask_c = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)

def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton kernel performing square matrix multiplication.

    This function assumes A and B are square matrices of shape (N, N) with N = 2048*2
    and of the same dtype (e.g. torch.float32). The computation performed is:

        C = A @ B

    The function:
      1. Validates the inputs.
      2. Allocates an output tensor.
      3. Computes the grid dimensions.
      4. Launches the Triton kernel.
      5. Returns the computed output.
      
    Fusion Note: The entire GEMM computation is fused into one pass in the Triton kernel:
      loading of A and B tiles, computing tl.dot, and accumulation are done in a single kernel.
    """
    # Validate input shapes and types.
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Inputs must be 2D matrices.")
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions of A and B must match for multiplication.")
    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
        raise ValueError("Both matrices must be square.")
    
    # Extract matrix size.
    N = A.shape[0]
    
    # Ensure that the devices and dtypes match.
    if A.device != B.device:
        raise ValueError("A and B must be on the same device.")
    dtype = A.dtype

    # Allocate the output tensor C in the same device and dtype.
    C = torch.empty((N, N), device=A.device, dtype=dtype)

    # Choose block sizes. These are chosen as powers of 2 for efficiency.
    # They can be tuned further if needed.
    BLOCK_SIZE_M = 128  # tile height
    BLOCK_SIZE_N = 128  # tile width
    BLOCK_SIZE_K = 32   # reduction tile size

    # Calculate grid dimensions in 2D:
    grid_x = triton.cdiv(N, BLOCK_SIZE_M)  # number of row tiles
    grid_y = triton.cdiv(N, BLOCK_SIZE_N)  # number of column tiles
    grid = (grid_x, grid_y)

    # Launch the Triton kernel.
    _matmul_kernel[grid](
        A, B, C, N,
        A.stride(0), A.stride(1),   # strides for A
        B.stride(0), B.stride(1),   # strides for B
        C.stride(0), C.stride(1),   # strides for C
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return C

# The following main-section is only for internal testing and debugging.
