#!/usr/bin/env python3
"""
Optimized Triton Kernel Implementation for ELU Activation

This module implements a fused Triton kernel that computes the elementwise
ELU activation function:

    ELU(x) = x                   if x > 0
             alpha * (exp(x)-1)   otherwise

for an input tensor x with shape (4096, 393216) and dtype torch.float32.
All compute is done inside the Triton kernel using triton.language operations.
The Python wrapper (kernel_function) only validates arguments, allocates output,
and launches the kernel. No torch.nn or torch.nn.functional modules are imported.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _elu_kernel(x_ptr, alpha, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute the global program id.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Generate indices for elements in this block.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to protect against out-of-bound accesses.
    mask = offsets < n_elements

    # Load input data from global memory.
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute elementwise ELU:
    # if x > 0: return x; otherwise: return alpha * (exp(x)-1)
    result = tl.where(x > 0, x, alpha * (tl.exp(x) - 1.0))
    # Store computed results to global memory.
    tl.store(out_ptr + offsets, result, mask=mask)

def kernel_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the ELU activation using a fused Triton kernel.

    Args:
        x (torch.Tensor): Input tensor. Must be on a CUDA device, contiguous,
                          and of dtype torch.float32.
        alpha (float): The ELU alpha parameter.
    
    Returns:
        torch.Tensor: Output tensor (same shape and dtype as x).
    """
    # Validate input tensor.
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device")
    if x.dtype != torch.float32:
        raise ValueError("Input tensor must be of dtype torch.float32")
    # Ensure x has a contiguous memory layout.
    x = x.contiguous()

    # Allocate output tensor.
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Set BLOCK_SIZE as a compile-time constant.
    BLOCK_SIZE = 1024

    # Compute grid size with ceiling division.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel.
    _elu_kernel[grid](x, alpha, out, n_elements, BLOCK_SIZE)
    return out

class ModelNew:
    """
    Lightweight model wrapper for the ELU activation Triton kernel.

    This class mimics a torch.nn.Module-like API without importing any torch.nn
    modules, and its __init__ accepts an alpha parameter. It provides to(),
    eval(), and __call__() methods so that external frameworks can instantiate
    and use it as expected.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.device = None

    def to(self, device):
        # Record the target device.
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return kernel_function(x, self.alpha)
