#!/usr/bin/env python
"""
Triton kernel for a 256-bin histogram over a uint8 input tensor.

This implementation computes a histogram in a fused manner by processing a
contiguous block (tile) of input elements per kernel instance, and then,
for each tile, counting the occurrences of every byte value (0–255) with a
vectorized comparison and reduction. Each kernel instance then atomically
adds its per‐tile counts for each bin into the single global 256‐element
output histogram.

The wrapper function 'kernel_function' below takes a tuple (data, output):
  • data   : 1D torch.Tensor with dtype=torch.uint8.
  • output : 1D torch.Tensor of shape (256,) with dtype=torch.int64.
Before launching the kernel, the wrapper validates the arguments and
initializes the output tensor to zero. The kernel launch configuration is
computed based on a chosen BLOCK_SIZE that determines how many input elements
each kernel instance processes.

Fusion details and rationale:
  - The kernel fuses the “load–reduce–atomic add” steps for each tile:
      1. Load a tile of input data (with proper out‐of‐bounds masking).
      2. For each histogram bin (0–255), count the occurrences in the tile
         via a vectorized comparison and reduction.
      3. Atomically add the tile’s count into the global histogram.
  - This design minimizes the number of slower global atomic-add operations
    (reducing from one per element to one per bin per tile) and helps the kernel
    tolerate severe contention (as in the case where 90% of entries have the same value).
    
Note on runtime restrictions: All the numerical work (loading, reduction, atomic add)
runs inside the Triton kernel using tl.load, tl.sum, tl.atomic_add, etc. Only
argument validation, tensor allocation, and grid configuration are handled via PyTorch.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def _histogram_kernel(data_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that computes a 256-bin histogram for a uint8 tensor.
    
    Each kernel instance:
      • Computes its starting offset as: block_start = program_id * BLOCK_SIZE.
      • Loads BLOCK_SIZE elements (with proper masking) from data_ptr.
      • For each bin value in range [0, 255]:
           – Compares the loaded block against the bin.
           – Sums the number of matches in that block.
           – Atomically adds the count to output_ptr[bin].
    
    Parameters:
      data_ptr   : Pointer to the input uint8 data.
      output_ptr : Pointer to the output histogram (256 int64 values).
      n_elements : Total number of elements in the input tensor.
      BLOCK_SIZE : Number of elements processed per kernel instance.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Compute the offsets for this block.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bound indices.
    mask = offsets < n_elements
    # Load a block of uint8 values from global memory.
    block_vals = tl.load(data_ptr + offsets, mask=mask)
    # For each histogram bin 0..255, compute the count in this block and update output.
    for bin in range(256):
        # Compare the block values to the current bin.
        is_bin = block_vals == bin  # returns a boolean vector.
        # Sum up the number of occurrences in the block.
        count = tl.sum(tl.cast(is_bin, tl.int64), axis=0)
        # Atomically update the global histogram for this bin.
        tl.atomic_add(output_ptr + bin, count)

def kernel_function(args):
    """
    Kernel wrapper for computing a 256-bin histogram.
    
    Expected input is a tuple (data, output) where:
      • data   : A 1D torch.Tensor with dtype=torch.uint8.
      • output : A 1D torch.Tensor of shape (256,) and dtype=torch.int64.
    
    The wrapper performs:
      - Argument validation (shape, dtype, device checks).
      - Initialization of the output histogram (sets all 256 bins to 0).
      - Grid configuration for the kernel launch.
      - Launch of the fused Triton kernel that computes the histogram.
    
    Returns:
      The output tensor containing the 256-bin histogram.
    """
    # Unpack inputs
    if isinstance(args, (tuple, list)):
        if len(args) != 2:
            raise ValueError("Expected arguments as a tuple (data, output)")
        data, output = args
    else:
        raise ValueError("Expected input as a tuple (data, output)")

    # Validate that both tensors are on the same device.
    if data.device != output.device:
        raise ValueError("data and output must reside on the same device.")
    
    # Validate dtypes of the tensors.
    if data.dtype != torch.uint8:
        raise ValueError("Input data must be of dtype torch.uint8")
    if output.dtype != torch.int64:
        raise ValueError("Output histogram must be of dtype torch.int64")
    
    # Validate tensor shapes.
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if output.ndim != 1 or output.numel() != 256:
        raise ValueError("Output histogram must be a 1D tensor with 256 elements")
    
    # Number of input elements.
    n_elements = data.numel()

    # Initialize the output histogram buffer to zero.
    output.zero_()

    # Choose a block size (tile size) for each kernel instance.
    BLOCK_SIZE = 1024  # compile-time constant for tile size.
    
    # Compute grid size as the ceiling division of n_elements by BLOCK_SIZE.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel.
    _histogram_kernel[grid](data, output, n_elements, BLOCK_SIZE)
    
    return output

# Self-test for basic functionality.
# End of file.
