#!/usr/bin/env python3
"""
Triton Kernel for Histogram Reduction

This module implements a fused Triton kernel that computes a histogram 
of a 1D uint8 input tensor and writes the bin counts (for values 0..255) 
into an output tensor of shape [256] with dtype int64. The kernel fuses 
the following steps in a single pass:
  • Loading a contiguous block (chunk) of the input tensor.
  • Comparing each element against every possible bin value (0–255).
  • Summing up counts for each bin in the block.
  • Atomically accumulating these counts into the global (output) histogram.

The Python wrapper 'kernel_function' is responsible for argument validation, 
output tensor initialization, grid configuration, and kernel launch.
Note: All numerical work (loads, comparisons, reductions, atomic updates) 
is done inside the Triton kernel using triton.language operations.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _histogram_kernel(data_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for histogram reduction.

    Each program instance processes a contiguous block of input data,
    computes the histogram of 8-bit values for its block, and then
    atomically accumulates the counts into a global histogram.

    Parameters:
      data_ptr    : Pointer to the input uint8 tensor.
      output_ptr  : Pointer to the output int64 tensor (size 256).
      n_elements  : Total number of elements in the input tensor.
      BLOCK_SIZE  : Number of elements processed per kernel instance.
    """
    # Compute the starting index for this program instance.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a block of the input data (as uint8 values).
    x = tl.load(data_ptr + offsets, mask=mask)

    # For each histogram bin from 0 to 255, compute the count in this block
    # and atomically add it to the global histogram.
    for bin in range(256):
        # Convert loop index to uint8 for an elementwise comparison.
        b = tl.cast(bin, x.dtype)
        # Create a boolean mask marking positions where x equals b.
        cmp = x == b
        # Sum the occurrences (cast boolean to int64 for reduction).
        count = tl.sum(tl.cast(cmp, tl.int64))
        # Atomically accumulate the count to the output histogram bin.
        tl.atomic_add(output_ptr + bin, count)

def kernel_function(inputs):
    """
    Fused Histogram Reduction Kernel Wrapper

    This function accepts a tuple (data, output) where:
      - data  : A 1D torch.uint8 tensor containing the input values.
      - output: An uninitialized torch.int64 tensor of shape [256] to hold
                the histogram counts.

    The wrapper performs the following steps:
      1. Validates the input arguments (device, shape, dtype).
      2. Zero-initializes the output tensor (to clear any garbage values).
      3. Computes the grid configuration based on the input size.
      4. Launches the fused Triton kernel which performs:
           • Loading of input data,
           • Per-block histogram computation,
           • Atomic accumulation into the global histogram.
      5. Returns the output tensor containing the histogram counts.

    Args:
      inputs: Tuple of (data, output)

    Returns:
      output: torch.int64 tensor of shape [256] with histogram counts.
    """
    if not isinstance(inputs, tuple) or len(inputs) != 2:
        raise ValueError("Expected input as a tuple: (data, output)")
    
    data, output = inputs

    # Validate device and dtype.
    if data.device.type != "cuda":
        raise ValueError("Input tensor must be on CUDA device")
    if data.dtype != torch.uint8:
        raise ValueError("Input tensor must be of type torch.uint8")
    if output.numel() != 256 or output.dtype != torch.int64:
        raise ValueError("Output tensor must have 256 elements and be of type torch.int64")

    # Zero-initialize the output tensor to remove any uninitialized values.
    output.zero_()

    n_elements = data.numel()
    # Define the number of elements each kernel instance will process.
    BLOCK_SIZE = 1024

    # Grid configuration: a 1D grid such that each block processes BLOCK_SIZE elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the Triton kernel.
    _histogram_kernel[grid](data, output, n_elements, BLOCK_SIZE)

    return output
