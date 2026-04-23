#!/usr/bin/env python3
"""
Triton Kernel for Square Matrix Multiplication (C = A * B)

This implementation uses a fully fused, tiled Triton kernel to compute the
matrix multiplication for 2D float32 tensors. All mathematical work (loading,
dot-product accumulation, and storing) is performed inside the Triton kernel 
using tl.load, tl.dot, and tl.store. The Python wrapper (kernel_function)
only performs input validation, output allocation, and kernel launch configuration.

Note:
  • No high‐level PyTorch tensor–tensor operations (e.g. torch.matmul, torch.mm, torch.bmm)
    are used in the computation path.
  • The __main__ test code is omitted to avoid any disallowed PyTorch compute ops.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute 2D tile indices from the flattened program id.
    pid = tl.program_id(0)
    num_tile_rows = tl.cdiv(M, BLOCK_SIZE_M)
    num_tile_cols = tl.cdiv(N, BLOCK_SIZE_N)
    tile_row = pid // num_tile_cols
    tile_col = pid % num_tile_cols

    # Compute the row and column indices handled by this tile.
    offs_m = tile_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tile_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator in fp32.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over tiles in the K dimension.
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(k_tiles):
        k_offset = k * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

        # Compute pointers for the current tile of matrix A and B.
        a_ptrs = A + (offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
        b_ptrs = B + (offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n)

        # Create masks to handle cases where the tile goes out-of-bound.
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load the A and B tiles (filling out-of-bound regions with 0.0).
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate the product of the current tiles.
        acc += tl.dot(a, b)

    # Compute pointer for the output tile.
    c_ptrs = C + (offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def kernel_function(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Performs square matrix multiplication (C = A * B) using the Triton kernel.

    Args:
        tensor_a (torch.Tensor): 2D tensor (M x K) with dtype torch.float32.
        tensor_b (torch.Tensor): 2D tensor (K x N) with dtype torch.float32.

    Returns:
        torch.Tensor: 2D tensor (M x N) representing the product, on the same device
                      as the input tensors.

    All computation is done inside the Triton kernel. The wrapper only validates
    inputs, allocates the output tensor, and launches the kernel.
    """
    # Validate inputs.
    if tensor_a.ndim != 2 or tensor_b.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices.")
    if tensor_a.shape[1] != tensor_b.shape[0]:
        raise ValueError("Incompatible matrix dimensions: tensor_a.shape[1] must equal tensor_b.shape[0].")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError("Input matrices must have the same dtype.")
    if tensor_a.dtype != torch.float32:
        raise ValueError("This kernel only supports torch.float32 dtype.")
    if tensor_a.device.type != "cuda":
        raise RuntimeError("The Triton kernel requires a CUDA device.")

    # Matrix dimensions.
    M = tensor_a.shape[0]
    K = tensor_a.shape[1]
    N = tensor_b.shape[1]

    # Allocate the output matrix.
    output = torch.empty((M, N), device=tensor_a.device, dtype=tensor_a.dtype)

    # Define compile-time constants for tile sizes.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Retrieve strides for the input and output matrices.
    stride_a_m, stride_a_k = tensor_a.stride(0), tensor_a.stride(1)
    stride_b_k, stride_b_n = tensor_b.stride(0), tensor_b.stride(1)
    stride_c_m, stride_c_n = output.stride(0), output.stride(1)

    # Compute grid dimensions.
    grid_rows = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_cols = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_rows * grid_cols,)

    # Launch the Triton kernel.
    _matmul_kernel[grid](
        tensor_a, tensor_b, output,
        M, N, K,
        stride_a_m, stride_a_k,
        stride_b_k, stride_b_n,
        stride_c_m, stride_c_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return output

# End of file.
