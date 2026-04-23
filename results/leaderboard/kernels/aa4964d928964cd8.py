#!/usr/bin/env python3
"""
Triton ELU Kernel Implementation

This module implements a fused ELU activation using a single Triton kernel.
The ELU (Exponential Linear Unit) activation is defined elementwise as:

    y = x                        if x >= 0
    y = alpha * (exp(x) - 1)       if x < 0

The implementation fuses memory load, activation computation, and memory store in one pass.
All arithmetic is performed using Triton operations (tl.load, tl.store, tl.exp, tl.where, etc.),
and no PyTorch compute operations are called inside the kernel.

A Python wrapper function 'kernel_function' is provided which:
  - Validates input tensor properties (device, dtype, contiguity)
  - Allocates the output tensor
  - Configures grid dimensions based on the total number of elements
  - Launches the Triton kernel
  - Returns the result as a normal PyTorch tensor

This kernel is intended to be used with inputs such as:
    - x: tensor of shape (4096, 393216), dtype=torch.float32 on CUDA
    - alpha: scalar float parameter for ELU
which will be compared against torch.nn.functional.elu.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _elu_kernel(input_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for computing the ELU activation.

    Each program instance (kernel block) computes a block of BLOCK_SIZE elements.
    For each element x, the kernel computes:
        y = x                        if x >= 0
        y = alpha * (exp(x) - 1)       if x < 0

    Args:
       input_ptr: pointer to the input tensor data.
       output_ptr: pointer to the output tensor data.
       n_elements: total number of elements in the input tensor.
       alpha: the ELU hyperparameter (scalar).
       BLOCK_SIZE: the number of elements processed by one kernel instance.
    """
    # Compute the starting index for this program instance.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Generate offsets for the elements this instance will process.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to guard against out-of-bound accesses.
    mask = offsets < n_elements
    
    # Load input values.
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute ELU activation.
    # Fused computation: if x >= 0, then y = x; otherwise y = alpha * (exp(x) - 1)
    y = tl.where(x >= 0.0, x, alpha * (tl.exp(x) - 1.0))
    
    # Store the result back to memory.
    tl.store(output_ptr + offsets, y, mask=mask)

def kernel_function(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Fused Triton ELU Kernel Wrapper.

    Applies the ELU activation elementwise:
       y = x                        if x >= 0
       y = alpha * (exp(x) - 1)       if x < 0

    The computations are fused in one Triton kernel launch,
    ensuring optimal memory access and performance.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (4096, 393216) and dtype torch.float32 on CUDA.
        alpha (float): The ELU hyperparameter.
    
    Returns:
        torch.Tensor: Output tensor with the ELU activation applied,
                      having the same shape and dtype as input_tensor.
    """
    # Validate that the input is a CUDA tensor of the correct dtype.
    if not input_tensor.is_cuda:
        raise ValueError("input_tensor must be a CUDA tensor")
    if input_tensor.dtype != torch.float32:
        raise ValueError("input_tensor must be of dtype torch.float32")
    
    # Ensure the tensor is contiguous for proper pointer arithmetic.
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    
    # Allocate an output tensor with the same shape and dtype.
    output = torch.empty_like(input_tensor)
    
    # Compute the total number of elements.
    n_elements = input_tensor.numel()
    
    # Define the block size (must be a power of 2). 1024 is chosen for balanced parallelism.
    BLOCK_SIZE = 1024
    
    # Compute grid dimensions: number of program instances needed to cover all elements.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel.
    _elu_kernel[grid](
        input_tensor,  # pointer to input data
        output,        # pointer to output data
        n_elements,    # total number of elements
        alpha,         # ELU alpha parameter
        BLOCK_SIZE     # compile-time constant block size
    )
    
    return output

# Optional self-test when running this module directly.
