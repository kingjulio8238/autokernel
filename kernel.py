"""
Autokernel — L2 #12: Gemm + Multiply + LeakyReLU
Iter 1: Fuse multiply + LeakyReLU into single CUDA kernel (eliminate one memory pass).
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_leaky_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float multiplier,
    float negative_slope,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] * multiplier;
        output[idx] = val > 0.0f ? val : val * negative_slope;
    }
}

torch::Tensor fused_multiply_leaky_relu(torch::Tensor input, float multiplier, float negative_slope) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_multiply_leaky_relu_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), input.data_ptr<float>(),
        multiplier, negative_slope, n);
    return output;
}
"""

cpp_src = "torch::Tensor fused_multiply_leaky_relu(torch::Tensor input, float multiplier, float negative_slope);"

fused_ext = load_inline(
    name="fused_mul_leakyrelu",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_multiply_leaky_relu"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.gemm(x)
        x = fused_ext.fused_multiply_leaky_relu(x, self.multiplier, self.negative_slope)
        return x
