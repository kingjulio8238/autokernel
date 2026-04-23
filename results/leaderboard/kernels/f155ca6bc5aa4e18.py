#!/usr/bin/env python3
"""
Kernel for computing a 256-bin histogram over uint8 data using Triton.

This implementation fuses the data load and atomic-add operations
in a single Triton kernel. Each kernel instance processes a contiguous
tile of the input vector (of size BLOCK_SIZE) and, for every valid input
element, it atomically increments the corresponding output histogram bin.
The wrapper function 'kernel_function' handles input validation, memory
initialization, grid configuration, and kernel launch.

The test harness expects:
  - Input: A 1D torch.Tensor of dtype torch.uint8.
  - Output: A 1D torch.Tensor with 256 elements of dtype torch.int64.
  
Note:
  - All the arithmetic and memory operations are implemented in the
    Triton kernel using tl.load and tl.atomic_add.
  - PyTorch usage is strictly limited to setup, validation, and allocation.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def _hist_kernel(data_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that processes a tile of the input and atomically updates
    the global histogram.

    Arguments:
      data_ptr   : pointer to the uint8 input tensor data.
      out_ptr    : pointer to the int64 output histogram (256 bins).
      n_elements : total number of elements in the input tensor.
      BLOCK_SIZE : compile-time constant; number of elements processed per kernel instance.
    """
    # Compute the start index for this kernel instance.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for out-of-bound elements.
    mask = offsets < n_elements

    # Load a block of input data as uint8.
    # Each lane loads one element (if within bounds).
    d = tl.load(data_ptr + offsets, mask=mask)
    # Convert the uint8 data to int32 so it can be used as an index.
    bin_idx = d.to(tl.int32)
    # Atomically add 1 to the histogram bin corresponding to each element.
    # The atomic operation is performed only on lanes where mask is True.
    tl.atomic_add(out_ptr + bin_idx, 1, mask=mask)


def kernel_function(tensors):
    """
    Wrapper function for the Triton histogram kernel.

    This function validates the input arguments, zeros out the output
    buffer, computes grid configuration, launches the Triton kernel, and
    returns the output histogram.

    Parameters:
      tensors (tuple): A tuple (data, output) where:
         - data: A 1D torch.Tensor of dtype torch.uint8 on CUDA.
         - output: A 1D torch.Tensor of shape (256,) and dtype torch.int64 on CUDA.

    Returns:
      torch.Tensor: The 256-bin histogram computed over the input data.
    
    Fusion details:
      - The kernel fuses data loading and atomic updates in one pass.
      - No separate PyTorch-level histogram or reduction is used.
    """
    # Unpack input and output tensors.
    data, output = tensors

    # Validate that the input data is a 1D uint8 tensor.
    if data.ndim != 1:
        raise ValueError("Input data tensor must be 1-dimensional")
    if data.dtype != torch.uint8:
        raise ValueError("Input data tensor must be of dtype torch.uint8")

    # Validate that the output tensor is a 1D tensor of 256 int64 elements.
    if output.ndim != 1 or output.numel() != 256:
        raise ValueError("Output histogram must be a 1D tensor with 256 elements")
    if output.dtype != torch.int64:
        raise ValueError("Output histogram tensor must be of dtype torch.int64")

    # Zero out the output histogram to ensure correct accumulation.
    output.zero_()

    # Ensure the tensors are contiguous.
    if not data.is_contiguous():
        data = data.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    # Determine the total number of elements.
    n_elements = data.numel()

    # Set a block size (number of elements processed per kernel instance).
    # BLOCK_SIZE is chosen as a power of 2 (e.g., 1024) for optimal performance.
    BLOCK_SIZE = 1024

    # Configure the grid:
    # Each kernel instance processes BLOCK_SIZE elements.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel.
    _hist_kernel[grid](
        data,         # pointer to input data (uint8)
        output,       # pointer to output histogram (int64)
        n_elements,   # total number of elements in input
        BLOCK_SIZE    # compile-time block size
    )

    # Return the output buffer; if the kernel did not return a value,
    # it was updated in-place.
    return output


# End of kernel.py implementation.
