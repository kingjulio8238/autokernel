"""
KernelBench Level 1, Problem 1: Square matrix multiplication (C = A * B)
READ-ONLY — do not modify this file. The agent modifies kernel.py instead.
Loaded via scripts/setup_problem.py from KernelBench dataset.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


M = 4096
K = 4096
N = 4096


def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
