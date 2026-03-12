"""
Autokernel — L2 #12: Gemm + Multiply + LeakyReLU
Baseline: same ops as reference, separate PyTorch calls.
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x
