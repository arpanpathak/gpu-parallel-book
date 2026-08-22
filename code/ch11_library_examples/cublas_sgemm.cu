// Chapter 11, 11.4 - cuBLAS SGEMM with row-major data, runnable.
//
// Build: nvcc -arch=compute_60 cublas_sgemm.cu -o cublas_sgemm -lcublas
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./cublas_sgemm

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <stdexcept>

#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// The book's function (11.4), with status checks on the handle calls.
void gemmViaCublas(const float* dA, const float* dB, float* dC,
                   int m, int n, int k, float alpha, float beta)
{
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasCreate failed");

    // cuBLAS is column-major; for row-major C = A * B we compute C^T = B^T * A^T.
    const int ldb = n;   // B is k x n row-major -> B^T is n x k col-major
    const int lda = k;   // A is m x k row-major -> A^T is k x m col-major
    const int ldc = n;   // C is m x n row-major -> C^T is n x m col-major

    const cublasStatus_t status =
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    dB, ldb,
                    dA, lda,
                    &beta,
                    dC, ldc);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        cublasDestroy(handle);
        throw std::runtime_error("cublasSgemm failed");
    }
    cublasDestroy(handle);
}

int main()
{
    const int m = 32, n = 32, k = 32;

    std::vector<float> A(m * k), B(k * n), ref(m * n);
    unsigned int seed = 7u;
    for (int i = 0; i < m * k; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        A[i] = ((seed >> 24) / 255.0f) * 2.0f - 1.0f;
    }
    for (int i = 0; i < k * n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        B[i] = ((seed >> 24) / 255.0f) * 2.0f - 1.0f;
    }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t)
                sum += A[i * k + t] * B[t * n + j];
            ref[i * n + j] = sum;
        }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CHECK(cudaMalloc(&dC, ref.size() * sizeof(float)));
    CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float),
                     cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    gemmViaCublas(dA, dB, dC, m, n, k, alpha, beta);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got(ref.size());
    CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int i = 0; i < m * n; ++i)
        maxErr = std::max(maxErr, std::abs((double)got[i] - ref[i]));
    std::printf("cublas: max error = %g\n", maxErr);
    if (maxErr > 1e-3)
    {
        std::fprintf(stderr, "CHAPTER 11 CUBLAS TEST FAILED\n");
        return 1;
    }
    std::printf("Chapter 11 cuBLAS test PASSED (row-major GEMM via C^T)\n");
    return 0;
}
