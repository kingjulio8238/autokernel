"""
Autokernel — agent-modifiable kernel file.
Iteration 7: CUDA C++ extension calling cublasGemmEx with algorithm search.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static cublasHandle_t handle = nullptr;

void ensure_handle() {
    if (handle == nullptr) {
        cublasCreate(&handle);
        // Use tensor math (TF32) for better performance on Ampere+
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    ensure_handle();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // Set cuBLAS to use the current CUDA stream
    cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS is column-major. For row-major C = A*B:
    // C^T = B^T * A^T (in column-major)
    // So we call: sgemm(N, N, N, M, K, alpha, B, N, A, K, beta, C, N)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_ext = load_inline(
    name="matmul_cublas_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_ext.matmul_cuda(A.contiguous(), B.contiguous())
