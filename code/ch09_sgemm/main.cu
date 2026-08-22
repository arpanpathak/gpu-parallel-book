// Chapter 9 - runnable test harness for all four SGEMM kernels from the book.
// The kernels live in sgemm.cu, verbatim from the book.
//
// Build: nvcc -arch=compute_60 sgemm.cu main.cu -o sgemm
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./sgemm

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Kernels defined in sgemm.cu.
__global__ void sgemmNaive(const float* A, const float* B, float* C, int N);
__global__ void sgemmCoalesced(const float* A, const float* B, float* C, int N);
__global__ void sgemmTiled(const float* A, const float* B, float* C, int N);
__global__ void sgemmRegisterTiled(const float* A, const float* B, float* C,
                                   int N);

#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

static void cpuSgemm(const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

template <typename Launch>
static int verifyKernel(const char* name, Launch launch,
                        const std::vector<float>& A,
                        const std::vector<float>& B,
                        const std::vector<float>& ref,
                        float* dA, float* dB, float* dC, int N)
{
    CHECK(cudaMemset(dC, 0, (size_t)N * N * sizeof(float)));
    launch();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got(N * N);
    CHECK(cudaMemcpy(got.data(), dC, (size_t)N * N * sizeof(float),
                     cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int i = 0; i < N * N; ++i)
        maxErr = std::max(maxErr, std::abs((double)got[i] - ref[i]));
    std::printf("%-18s max error = %g\n", name, maxErr);
    return maxErr > 1e-3 ? 1 : 0;
}

int main()
{
    const int N = 32;              // small, multiple of T=16

    // Deterministic matrices in [-1, 1].
    std::vector<float> A(N * N), B(N * N), ref(N * N);
    unsigned int seed = 42u;
    for (int i = 0; i < N * N; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        A[i] = ((seed >> 24) / 255.0f) * 2.0f - 1.0f;
        seed = seed * 1664525u + 1013904223u;
        B[i] = ((seed >> 24) / 255.0f) * 2.0f - 1.0f;
    }
    cpuSgemm(A, B, ref, N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK(cudaMalloc(&dA, (size_t)N * N * sizeof(float)));
    CHECK(cudaMalloc(&dB, (size_t)N * N * sizeof(float)));
    CHECK(cudaMalloc(&dC, (size_t)N * N * sizeof(float)));
    CHECK(cudaMemcpy(dA, A.data(), (size_t)N * N * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), (size_t)N * N * sizeof(float),
                     cudaMemcpyHostToDevice));

    const dim3 block16(16, 16);
    const dim3 grid2((N + 15) / 16, (N + 15) / 16);

    int failures = 0;
    failures += verifyKernel("sgemmNaive", [&] {
        sgemmNaive<<<grid2, block16>>>(dA, dB, dC, N);
    }, A, B, ref, dA, dB, dC, N);

    failures += verifyKernel("sgemmCoalesced", [&] {
        sgemmCoalesced<<<grid2, block16>>>(dA, dB, dC, N);
    }, A, B, ref, dA, dB, dC, N);

    failures += verifyKernel("sgemmTiled", [&] {
        sgemmTiled<<<grid2, block16>>>(dA, dB, dC, N);
    }, A, B, ref, dA, dB, dC, N);

    // Register tiling: T/RN x T/RM = 8 x 8 threads per block.
    const dim3 blockReg(8, 8);
    failures += verifyKernel("sgemmRegisterTiled", [&] {
        sgemmRegisterTiled<<<grid2, blockReg>>>(dA, dB, dC, N);
    }, A, B, ref, dA, dB, dC, N);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));

    if (failures != 0)
    {
        std::fprintf(stderr, "CHAPTER 9 TEST FAILED (%d kernels)\n", failures);
        return 1;
    }
    std::printf("Chapter 9 test PASSED (all four SGEMM kernels match CPU)\n");
    return 0;
}
