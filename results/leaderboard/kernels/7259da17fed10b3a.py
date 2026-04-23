#!/usr/bin/env python3
"""
Triton Kernel for Elementwise ELU Activation

This module implements a fused Triton kernel that applies the ELU (Exponential Linear Unit)
activation function elementwise to an input tensor. For each element x it computes:

    ELU(x) = x                       if x >= 0
             alpha * (exp(x) - 1)    if x < 0

The kernel fuses the load, compute, and store steps into a single pass.
It processes an input tensor (flattened as 1D) and correctly handles out‐of‐bound accesses via masking.

The module provides the following:
  • A Triton kernel function (_elu_kernel) that implements the ELU activation using Triton operations.
  • A Python wrapper function (kernel_function) that performs necessary validation,
    allocates the output tensor, and launches the Triton kernel.
  • A torch.nn.Module wrapper class (ModelNew) that accepts an initialization parameter (alpha)
    and forwards input tensors to the kernel_function.
    
Note: All elementwise math within the activation is implemented solely using Triton operations.
PyTorch is only used for tensor allocation, device/dtype checks, and in the ModelNew wrapper.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _elu_kernel(input_ptr, output_ptr, n_elements: tl.constexpr, alpha, BLOCK_SIZE: tl.constexpr):
    """
    Fused Triton kernel to perform ELU activation on a 1D-flattened view of the input.

    For every element x:
       if x >= 0:  out = x
       else:       out = alpha * (exp(x) - 1)

    Parameters:
      input_ptr  : Pointer to the input tensor data.
      output_ptr : Pointer to the output tensor data.
      n_elements : Total number of elements to process (a compile-time constant).
      alpha      : The negative slope coefficient in the ELU activation.
      BLOCK_SIZE : The number of elements processed per kernel block.
    """
    # Calculate the global indices for this block.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure indexes are within bounds.
    mask = offsets < n_elements
    
    # Load the input values at the computed offsets.
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute ELU activation: x if x >= 0 else alpha * (exp(x)-1)
    out = tl.where(x >= 0, x, alpha * (tl.exp(x) - 1))
    
    # Store the output values.
    tl.store(output_ptr + offsets, out, mask=mask)

def kernel_function(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Wrapper function to launch the fused Triton ELU kernel.

    This function performs the following:
      - Validates that the input tensor is on a CUDA device.
      - Ensures the tensor is contiguous.
      - Allocates an output tensor with the same shape and dtype as the input.
      - Computes the total number of elements.
      - Sets up the grid configuration and launches the Triton kernel.
      - Returns the output tensor containing the ELU activation result.

    Parameters:
      input_tensor : A PyTorch tensor on a CUDA device.
      alpha        : The alpha value for the ELU activation.

    Returns:
      A PyTorch tensor with the same shape and dtype as input_tensor.
    """
    # Verify that the input tensor is on a CUDA device.
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device")
    
    # Ensure tensor is contiguous for correct memory access.
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    
    # Allocate the output tensor.
    output = torch.empty_like(input_tensor)
    
    # Calculate the total number of elements.
    n_elements = input_tensor.numel()
    
    # Define the block size (a power-of-two for optimal performance).
    BLOCK_SIZE = 1024
    
    # Compute grid dimensions (1D launch).
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel.
    _elu_kernel[grid](input_tensor, output, n_elements, alpha, BLOCK_SIZE)
    
    return output

# ---------------------------------------------------------------------------
# ModelNew: A torch.nn.Module wrapper for the Triton ELU kernel.
# ---------------------------------------------------------------------------
# This allows the kernel to be integrated into model pipelines.
# The __init__ method is defined to take an alpha parameter so that when the model
# is instantiated (e.g., ModelNew(alpha_value)), the alpha is properly captured.
#
# Note: The model's forward pass simply delegates to the kernel_function.
#
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes the ModelNew module with a given alpha for the ELU activation.

        Parameters:
          alpha (float): The alpha constant for the ELU activation. Default is 1.0.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the ELU activation to the input tensor using the Triton kernel.

        Parameters:
          input_tensor (torch.Tensor): Input tensor on a CUDA device.
          
        Returns:
          torch.Tensor: Output tensor after applying ELU activation.
        """
        return kernel_function(input_tensor, self.alpha)

# ---------------------------------------------------------------------------
# Testing block (for standalone testing only)
# ---------------------------------------------------------------------------
