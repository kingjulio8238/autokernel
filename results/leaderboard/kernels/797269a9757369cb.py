#!/usr/bin/env python3
"""
Triton kernel for computing a 256-bin histogram over a 1D uint8 tensor.

This kernel computes, given an input tensor "data" of dtype uint8,
a histogram tensor "output" of shape [256] and dtype int64 such that:
    output = torch.bincount(data, minlength=256)

The implementation launches many kernel instances where each instance
processes a contiguous block of the input. The kernel loads BLOCK_SIZE
elements at a time, applies masking for bounds safety, computes a "safe"
bin index and an addition delta for each element (using tl.where rather than a Python if),
and then updates the global histogram using tl.atomic_add.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def _histogram_kernel(data_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each kernel instance (identified by pid) processes a strided subset of the input.
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    grid_stride = grid_size * BLOCK_SIZE
    iterations = tl.cdiv(n_elements - pid * BLOCK_SIZE, grid_stride)

    # For each iteration, compute the indices to load.
    for it in range(iterations):
        curr_index = pid * BLOCK_SIZE + it * grid_stride + tl.arange(0, BLOCK_SIZE)
        # Mask out indices that exceed the total number of elements.
        mask = curr_index < n_elements
        # Load a vector of uint8 values; if out-of-bound, use 0.
        vals = tl.load(data_ptr + curr_index, mask=mask, other=0)
        # Process each element in the loaded block.
        # Use tl.static_range to unroll the loop over the BLOCK_SIZE elements.
        for j in tl.static_range(BLOCK_SIZE):
            # Instead of using a Python if, use tl.where with the boolean value.
            valid = mask[j]
            safe_bin_idx = tl.where(valid, tl.cast(vals[j], tl.int64), 0)
            delta = tl.where(valid, 1, 0)
            tl.atomic_add(output_ptr + safe_bin_idx, delta)

def kernel_function(args):
    """
    Kernel wrapper to compute a 256-bin histogram from a 1D uint8 tensor.

    Expects a tuple (data, output) where:
      - data   : 1D torch.Tensor (dtype=torch.uint8) input.
      - output : 1D torch.Tensor of shape [256] and dtype torch.int64 (histogram result).

    The wrapper validates input properties, zeros the output histogram,
    configures the grid, and launches the Triton kernel which performs
    the fused histogram computation in a single pass.

    Returns:
      The updated output tensor.
    """
    # Unpack the arguments.
    if isinstance(args, (list, tuple)):
        if len(args) == 2:
            data, output = args
        else:
            raise ValueError("Expected two arguments: (data, output)")
    else:
        raise ValueError("Expected arguments as a tuple (data, output).")

    # Validate tensor dtypes and shapes.
    assert data.dtype == torch.uint8, "data must be of type torch.uint8"
    assert output.numel() == 256, "output tensor must have 256 elements"
    assert output.dtype == torch.int64, "output tensor must be of type torch.int64"

    n_elements = data.numel()
    data = data.contiguous()
    output = output.contiguous()
    output.zero_()

    # Set block size
