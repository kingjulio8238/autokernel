"""
Autokernel — agent-modifiable kernel file.
Iteration 8: cuBLASLt with algorithm heuristic search for optimal GEMM.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

static cublasLtHandle_t ltHandle = nullptr;
static void* workspace = nullptr;
static const size_t workspaceSize = 32 * 1024 * 1024; // 32 MB workspace

void ensure_handle() {
    if (ltHandle == nullptr) {
        cublasLtCreate(&ltHandle);
        cudaMalloc(&workspace, workspaceSize);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    ensure_handle();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create matrix multiplication descriptor
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Create matrix layouts (column-major interpretation of row-major data)
    // For row-major C = A*B, we compute C^T = B^T * A^T in column-major
    cublasLtMatrixLayout_t Bdesc, Adesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N);  // B^T is N x K
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K);  // A^T is K x M
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N);  // C^T is N x M

    // Create preference descriptor with workspace
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspaceSize, sizeof(workspaceSize));

    // Get the best algorithm
    cublasLtMatmulHeuristicResult_t heurResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Bdesc, Adesc, Cdesc, Cdesc,
                                    pref, 1, &heurResult, &returnedResults);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (returnedResults > 0) {
        cublasLtMatmul(ltHandle, matmulDesc,
                       &alpha,
                       B.data_ptr<float>(), Bdesc,
                       A.data_ptr<float>(), Adesc,
                       &beta,
                       C.data_ptr<float>(), Cdesc,
                       C.data_ptr<float>(), Cdesc,
                       &heurResult.algo,
                       workspace, workspaceSize,
                       stream);
    } else {
        // Fallback to basic cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta,
                    C.data_ptr<float>(), N);
        cublasDestroy(handle);
    }

    // Cleanup descriptors
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_ext = load_inline(
    name="matmul_cublaslt_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublas", "-lcublasLt"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_ext.matmul_cuda(A.contiguous(), B.contiguous())
