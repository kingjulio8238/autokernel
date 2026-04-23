#!/usr/bin/env python3
"""
Triton Histogram Kernel Implementation

This file implements a fused histogram kernel using Triton.
The kernel computes a 256-bin histogram for a 1D uint8 input tensor.
It fuses the following operations in a single kernel pass:
  • Loading a block of input data with proper masking.
  • For every bin from 0 to 255, performing a vectorized
    comparison (data == bin) and reducing (tl.sum) the count.
  • Atomically accumulating the per-block histogram counts
    into the global output histogram (of 256 int64 elements).

The wrapper function 'kernel_function' accepts a tuple (data, output)
where:
  - data: a 1D torch.Tensor of dtype torch.uint8.
  - output: a 1D torch.Tensor of shape (256,) and dtype torch.int64.
The output is assumed initially zeroed (or will be zeroed by the wrapper)
and is updated in place by the kernel.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _histogram_kernel(data_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the global block range.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Each kernel instance processes BLOCK_SIZE contiguous elements.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Apply boundary conditions.
    mask = offsets < n_elements
    # Load a block of input data (dtype: uint8) with masking.
    data_block = tl.load(data_ptr + offsets, mask=mask)
    
    # Loop over all 256 histogram bins.
    # The loop variable 'bin' is a compile-time constant.
    for bin in range(256):
        # Compare each element of the block with the bin value.
        # NOTE: Instead of calling tl.uint8 (which is a dtype, not a function),
        # we simply use the integer literal 'bin'. Triton will treat it as a constant.
        cmp = data_block == bin
        # Cast booleans to int64 and sum the number of True values.
        count = tl.sum(tl.cast(cmp, tl.int64))
        # Atomically add the block's count for this bin into the global histogram.
        tl.atomic_add(out_ptr + bin, count)

def kernel_function(args):
    """
    Triton Histogram Kernel Wrapper

    This function performs input validation, zeros out the output tensor,
    configures the grid, launches the Triton kernel, and returns the updated histogram.

    Args:
      args: A tuple (data, output) where:
            - data: a 1D torch.Tensor (dtype=torch.uint8) representing input data.
            - output: a 1D torch.Tensor of shape (256,) (dtype=torch.int64) for the histogram.

    Returns:
      The output tensor with the computed histogram.
    """
    # Support tuple or list input.
    if isinstance(args, (tuple, list)):
        if len(args) != 2:
            raise ValueError("Expected two arguments: (data, output)")
        data, output = args
    else:
        raise ValueError("Expected argument to be a tuple (data, output)")

    # Validate input types and shapes.
    if not torch.is_tensor(data):
        raise TypeError("data must be a torch.Tensor")
    if not torch.is_tensor(output):
        raise TypeError("output must be a torch.Tensor")
    if data.dtype != torch.uint8:
        raise ValueError("data tensor must be of dtype torch.uint8")
    if output.dtype != torch.int64:
        raise ValueError("output tensor must be of dtype torch.int64")
    if data.dim() != 1:
        raise ValueError("data tensor must be 1D")
    if output.numel() != 256:
        raise ValueError("output histogram tensor must have 256 elements")
    
    # Zero out the output histogram as the kernel accumulates into it.
    output.zero_()

    # Number of elements in the input.
    n_elements = data.numel()
    # Define block size; must be a multiple of 16. Here we use 1024.
    BLOCK_SIZE = 1024
    # Compute grid so that each kernel instance processes BLOCK_SIZE elements.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel.
    _histogram_kernel[grid](
        data, output, n_elements, BLOCK_SIZE
    )
    return output

# The file is self-contained and provides the kernel_function required for testing.
