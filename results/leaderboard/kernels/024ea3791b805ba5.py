#!/usr/bin/env python3
"""
Triton kernel for computing a histogram of a uint8 tensor using atomic updates.

The kernel reads a 1D tensor of unsigned 8‐bit values (in [0,255])
and accumulates counts into a 256‐element output tensor (of dtype int64)
via atomic updates. All memory loads and atomic updates are fused in one pass.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _histogram_kernel(data_ptr, hist_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each kernel instance processes a block of BLOCK_SIZE elements.
    pid = tl.program_id(0)
    # Compute offsets in the data array for this block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bound accesses.
    mask = offsets < n_elements
    # Load a block of data; for out-of-bound positions load returns 0.
    data_block = tl.load(data_ptr + offsets, mask=mask, other=0)
    # Precompute the per–element increment values:
    # For valid indices (mask==True) set 1, else 0.
    inc_vector = tl.where(mask, tl.int64(1), tl.int64(0))
    # Loop over the BLOCK_SIZE elements (the loop gets unrolled as BLOCK_SIZE is a compile-time constant).
    for i in range(BLOCK_SIZE):
        # Retrieve the bin index from the data value, cast to int64.
        bin_index = tl.cast(data_block[i], tl.int64)
        inc_val = inc_vector[i]
        # Atomically update the histogram for that bin.
        tl.atomic_add(hist_ptr + bin_index, inc_val)

def kernel_function(inputs):
    """
    Wrapper that validates inputs, zeroes output, sets grid size,
    launches the Triton kernel, and returns the updated histogram.
    
    Expected inputs (either passed as a single tuple or list):
      data  : torch.Tensor with dtype=torch.uint8 and shape (N,)
      output: torch.Tensor with shape (256,) and dtype=torch.int64
              which will store the histogram counts.
    """
    # Validate that inputs is a 2-tuple (data, output).
    if isinstance(inputs, (list, tuple)):
        if len(inputs) == 2:
            data, output = inputs
        else:
            raise ValueError("Expected exactly two inputs: (data, output)")
    else:
        raise ValueError("Expected inputs to be a tuple of (data, output)")
    
    # Check that both inputs are torch.Tensor objects.
    if not isinstance(data, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("Both data and output must be torch.Tensor objects")
    if data.dtype != torch.uint8:
        raise ValueError("Input data must be of type torch.uint8")
    if output.dtype != torch.int64:
        raise ValueError("Output histogram must be of type torch.int64")
    if output.numel() != 256:
        raise ValueError("Output histogram must have 256 elements")
    
    # Ensure
