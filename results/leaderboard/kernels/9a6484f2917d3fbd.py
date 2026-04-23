#!/usr/bin/env python3
"""
Triton Kernel for ELU Activation

This file implements a fused Triton kernel to apply the ELU activation:
    y = x   if x > 0
    y = alpha * (exp(x) - 1)   otherwise

The kernel fuses the memory load, ELU computation, and memory store into a single
pass, following the guidelines for optimal coalesced memory accesses, proper
masking for boundary conditions, and minimal kernel launch overhead.

The exposed function "kernel_function" is a Python wrapper that:
  - Validates input dtype and contiguity.
  - Allocates the output tensor.
  - Computes the total number of elements and grid configuration.
  - Launches the Triton kernel to perform the ELU elementwise.

All mathematical computation is performed in the Triton kernel using
triton.language operations (tl.load, tl.store, tl.where, tl.exp, etc.).
No PyTorch compute operations are used inside the kernel.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def _elu_kernel(x_ptr, y_ptr, n_elements, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Fused ELU kernel.

    For each index i, computes:
          y[i] = x[i]            if x[i] > 0,
          y[i] = alpha*(exp(x[i]) - 1)   if x[i] <= 0.
          
    The kernel uses a 1D grid. Each program instance processes a contiguous
    block of BLOCK_SIZE elements. Boundary conditions are handled via masking.
    
    Parameters:
      x_ptr: Pointer to the input tensor.
      y_ptr: Pointer to the output tensor.
      n_elements: Total number of elements in the tensor.
      alpha: ELU activation parameter.
      BLOCK_SIZE: Number of elements processed per kernel instance.
    """
    # Calculate global indices for this block.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Offsets for each element in the block.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bound accesses.
    mask = offsets < n_elements

    # Load input elements.
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute ELU: if x > 0 then keep x; otherwise compute alpha*(exp(x)-1)
    y = tl.where(x > 0.0, x, alpha * (tl.exp(x) - 1.0))
    # Store the results back to global memory.
    tl.store(y_ptr + offsets, y, mask=mask)


def kernel_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Wrapper for the Triton ELU kernel.

    This function applies the ELU activation to the input tensor using a single fused
    Triton kernel. The computation performed is:

         output[i] = x[i] if x[i] > 0 else alpha*(exp(x[i]) - 1)

    The implementation fuses the following stages:
      1. Memory Load: Coalesced load of data from the input tensor.
      2. Compute: Elementwise ELU activation using tl.where and tl.exp.
      3. Memory Store: Coalesced store of results in the output tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (...) and dtype torch.float32.
        alpha (float): ELU activation parameter.

    Returns:
        torch.Tensor: Output tensor with the same shape, device, and dtype as x.
    """
    # Validate input dtype.
    if x.dtype != torch.float32:
        raise ValueError("kernel_function only supports float32 data type.")
    
    # Ensure the input tensor is contiguous.
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Allocate the output tensor.
    output = torch.empty_like(x)
    # Flatten the tensor: treat as 1D for kernel launch.
    n_elements = x.numel()

    # Set BLOCK_SIZE as a compile-time constant.
    BLOCK_SIZE = 1024  # You can tune this for your hardware.

    # Compute grid such that each kernel instance handles BLOCK_SIZE elements.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the fused Triton kernel.
    _elu_kernel[grid](x, output, n_elements, alpha, BLOCK_SIZE)

    return output


# Optional: simple test when run as standalone.
