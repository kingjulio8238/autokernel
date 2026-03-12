"""
Autokernel — Iteration 2: torch.mm with pre-allocated output via addmm.
Use addmm with zero bias - sometimes takes a faster cuBLAS path.
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.mm(A, B)
