#!/usr/bin/env python3
"""
Triton kernel implementation for ELU activation.

This module defines a fused Triton kernel that implements the ELU activation function:
    y = x                          if x >= 0
    y = alpha * (exp(x) - 1)         if x < 0

The kernel fuses the complete operator pipeline (load, compute, store) into a single pass.
It uses tl.load, tl.store, and tl.where for efficient memory access and computation.
The Python wrapper function 'kernel_function' handles argument validation, output allocation,
and kernel launch. In addition, a ModelNew class is provided so that the runtime can
instantiate a model using an alpha parameter. All compute runs in the Triton kernels.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _elu_kernel(x_ptr, y_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    # Calculate the global element indices for this kernel instance.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements with masking.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute ELU: if x >= 0 then x, else alpha*(exp(x)-1).
    y_val = tl.where(x >= 0.0, x, alpha * (tl.exp(x) - 1.0))

    # Store the result back to global memory.
    tl.store(y_ptr + offsets, y_val, mask=mask)

def kernel_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Wrapper for the Triton ELU kernel.

    Args:
        x (torch.Tensor): The input tensor (must be on CUDA).
        alpha (float): The ELU alpha parameter.

    Returns:
        torch.Tensor: Output tensor with ELU applied (same shape, dtype, and device as input).
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA.")

    # Allocate output tensor with the same shape and properties as the input.
    y = torch.empty_like(x)
    n_elements = x.numel()

    # Define a compile-time block size (power-of-two value for optimal performance).
    BLOCK_SIZE = 1024

    # Define the grid: one kernel instance covers BLOCK_SIZE elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the Triton kernel.
    _elu_kernel[grid](
        x,          # pointer to input tensor data
        y,          # pointer to output tensor data
        n_elements, # total number of elements
        alpha,      # ELU alpha parameter
        BLOCK_SIZE  # compile-time block size
    )

    return y

#----------------------------------------------------------------------------
# Additional ModelNew class to support model instantiation with an alpha parameter.
# This enables fusion with frameworks that expect a model class (e.g., KernelBench).
#----------------------------------------------------------------------------
import torch.nn as nn

class ModelNew(nn.Module):
    """
    A simple model that applies the ELU activation using the Triton kernel.

    The __init__ accepts an 'alpha' parameter so that instantiation with an extra
    argument (e.g., ModelNew(1.0)) is valid.
    """
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernel_function(x, self.alpha)

#----------------------------------------------------------------------------
# Self-test: When this file is run directly, perform a self-test.
#----------------------------------------------------------------------------
