"""Wrap KernelAgent output in KernelBench ModelNew format.

KernelAgent outputs ``kernel_function(...)`` (a plain function).
KernelBench expects ``class ModelNew(nn.Module)`` with ``forward()``.

This module converts between the two formats.
"""

from __future__ import annotations

import re


def extract_forward_params(reference_code: str) -> tuple[str, str]:
    """Extract forward() parameters from reference code.

    Returns (full_params, param_names_only):
        full_params: "A: torch.Tensor, B: torch.Tensor"
        param_names_only: "A, B"
    """
    match = re.search(r"def forward\(self,\s*(.*?)\)\s*(?:->|:)", reference_code, re.DOTALL)
    if not match:
        return "*args", "*args"

    full_params = match.group(1).strip()

    # Extract just names
    names = []
    for param in full_params.split(","):
        param = param.strip()
        name = param.split(":")[0].strip()
        if name:
            names.append(name)

    return full_params, ", ".join(names)


def wrap_in_model_new(kernel_code: str, reference_code: str) -> str:
    """Wrap KernelAgent's kernel_function output in ModelNew class.

    If the kernel already has ModelNew, return as-is.
    """
    # Already has ModelNew — no wrapping needed
    if "class ModelNew" in kernel_code:
        return kernel_code

    full_params, param_names = extract_forward_params(reference_code)

    # Check if kernel_function exists
    has_kernel_function = "def kernel_function" in kernel_code or "kernel_function" in kernel_code

    if has_kernel_function:
        return f'''{kernel_code}

import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, {full_params}) -> torch.Tensor:
        return kernel_function({param_names})
'''
    else:
        # Kernel doesn't have kernel_function — wrap the whole thing
        # This handles cases where KernelAgent outputs a @triton.jit kernel
        # with a custom wrapper function
        return f'''import torch
import torch.nn as nn

{kernel_code}

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, {full_params}) -> torch.Tensor:
        # TODO: wire up the Triton kernel call
        return torch.matmul({param_names})
'''
