"""
Autokernel — agent-modifiable kernel file.
Iteration 1: torch.compile with max-autotune for optimized GEMM.
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    @torch.compile(mode="max-autotune")
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
