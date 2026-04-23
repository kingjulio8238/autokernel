#!/usr/bin/env python
"""
Triton Histogram Kernel

This module implements a fused Triton kernel that computes a histogram of a 1D tensor of 8‐bit (uint8)
values. The kernel iterates over the input in blocks and for each value performs an atomic update to
the corresponding output bin (there are 256 bins). This atomic update design guarantees correct
results even under high contention (i.e. many threads updating the same bin), as tested by our reference.

The kernel uses tl.load() to read input data, tl.atomic_add() for the atomic update on the
global histogram, and proper boundary checking via an in‐loop if statement. All of the numerical work
is confined to the Triton kernel, while the Python wrapper (kernel_function) performs argument
validation, tensor allocation, and grid configuration.

Fused operations:
  - Load (each element)
  - Boundary check (if index < n_elements)
  - Atomic update (increment histogram bin corresponding to the element value)

This module complies with the following requirements:
  * The actual work is done in a single Triton kernel decorated with @triton.jit.
  * The kernel implements all numerical operations using Triton language operations (tl.load, tl.atomic_add, etc.).
  * The Python wrapper function 'kernel_function' sets up the kernel launch (grid configuration, validation) and returns
    the computed histogram as a PyTorch tensor.
  
RUNTIME RESTRICTIONS:
  * No PyTorch compute operations (e.g. torch.add, torch.sum, etc.) are used inside the kernel.
  * All memory access is performed via tl.load and tl.store.
  * All computation (and atomic updates) occur within the kernel.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _histogram_kernel(data_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel to compute a histogram over a 1D uint8 tensor.

    For each element in the block (of size BLOCK_SIZE), this kernel:
      - Computes the global index.
      - Checks whether the index is in bounds.
      - Loads the uint8 value.
      - Increments the corresponding bin in the output histogram using an atomic add.
      
    Parameters:
      data_ptr   : Pointer to the input uint8 data.
      output_ptr : Pointer to the output histogram (256 bins, int64).
      n_elements : Total number of elements in the input tensor.
      BLOCK_SIZE : Number of elements processed by each kernel instance (compile-time constant).
    """
    # Get the unique program id along the 0-axis.
    pid = tl.program_id(0)
    # Each program instance processes a contiguous block of BLOCK_SIZE elements.
    # Loop over the block elements.
    for i in range(BLOCK_SIZE):
        # Compute the global index for the current element.
        idx = pid * BLOCK_SIZE + i
        # Only process valid indices.
        if idx < n_elements:
            # Load the input value (uint8).
            val = tl.load(data_ptr + idx)
            # Atomically add 1 to the histogram bin corresponding to the value.
            # Cast 'val' to int64 since the output histogram is int64.
            tl.atomic_add(output_ptr + tl.cast(val, tl.int64), 1)

def kernel_function(data: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """
    Fused histogram implementation using a single Triton kernel.

    This function validates inputs and launches the Triton kernel that:
      - Processes the input tensor 'data' (1D, contiguous, dtype uint8) in blocks.
      - For each valid element, atomically increments the corresponding bin in
        the output histogram tensor 'output' (shape: (256,), dtype int64).
      
    The entire operation (load, boundary check, atomic update) is fused inside the Triton kernel.

    Parameters:
        data   : A 1D, contiguous torch.Tensor of dtype torch.uint8.
        output : A torch.Tensor with shape (256,) and dtype torch.int64 that will store the histogram.
    
    Returns:
        output : The histogram tensor with 256 bins.
    """
    # Input validations.
    assert data.dtype == torch.uint8, "Input tensor 'data' must be of dtype torch.uint8"
    assert data.is_contiguous(), "Input tensor 'data' must be contiguous"
    assert data.ndim == 1, "Input tensor 'data' must be 1D"
    
    # Output tensor validations.
    assert output.dtype == torch.int64, "Output tensor must be of dtype torch.int64"
    assert output.numel() == 256, "Output tensor must have 256 elements (bins)"
    assert output.is_contiguous(), "Output tensor must be contiguous"
    
    # Total number of elements in the input tensor.
    n_elements = data.numel()
    
    # Set a compile-time constant BLOCK_SIZE.
    # This value may be tuned for performance; here, 1024 is chosen as a reasonable default.
    BLOCK_SIZE = 1024
    
    # Zero-initialize the output histogram.
    output.zero_()
    
    # Calculate the grid size (number of kernel instances) based on BLOCK_SIZE.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel.
    # The kernel will iterate over the input tensor in blocks and perform atomic histogram updates.
    _histogram_kernel[grid](data, output, n_elements, BLOCK_SIZE)
    
    # Return the computed histogram.
    return output

# For local testing, you can uncomment the code below.
# if __name__ == "__main__":
#     # Simple test example to run the histogram kernel.
#     n = 4096  # Must be a multiple of 16.
#     torch.manual_seed(42)
#     # Create a random input tensor of uint8.
#     data = torch.randint(0, 256, (n,), device="cuda", dtype=torch.uint8).contiguous()
#     # Introduce high contention by forcing some entries to a random "evil" value.
#     evil_value = torch.randint(0, 256, (), device="cuda", dtype=torch.uint8)
#     contention = 0.5  # 50% chance for a value to be overwritten.
#     mask = torch.rand(n, device="cuda") < contention
#     data[mask] = evil_value
#     # Allocate output histogram tensor.
#     output = torch.empty(256, device="cuda", dtype=torch.int64).contiguous()
#     # Execute the kernel.
#     hist = kernel_function(data, output)
#     # Compare with torch.bincount.
#     ref_hist = torch.bincount(data, minlength=256)
#     print("Reference Histogram:", ref_hist[:10])
#     print("Kernel Histogram   :", hist[:10])
#     print("Test Passes:", torch.all(hist == ref_hist))
    
# End of kernel module
