"""
Custom reference: ReLU activation (elementwise)
Simple elementwise operation — a Triton kernel should easily beat PyTorch.
Loaded via /problem load reference_relu.py
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Simple model that performs ReLU activation."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


M = 16384
N = 16384


def get_inputs():
    return [torch.randn(M, N)]


def get_init_inputs():
    return []
