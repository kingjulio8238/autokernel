"""
Autokernel — agent-modifiable kernel file.
Must define ModelNew with the same forward() signature as Model in reference.py.
Evaluated by prepare.py via KernelBench's eval harness.
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
