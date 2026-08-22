// Chapter 5 - runnable test harness for the two book kernels (5.5.1, 5.5.2).
// The kernels themselves live in histogram.cu, verbatim from the book.
//
// Build: nvcc -arch=compute_60 histogram.cu main.cu -o histogram
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./histogram

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Kernels defined in histogram.cu (the book's code, unchanged).
__global__ void histogram(const unsigned char* data, int* hist, int n);
__global__ void lockedCriticalSection(float* g_data, int n);

#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

int main()
{
    const int n = 1 << 20;   // 1,048,576 elements

    // ------------------------------------------------------------------
    // 5.5.1: privatised/atomic histogram
    // ------------------------------------------------------------------
    std::vector<unsigned char> h_data(n);
    std::vector<int> h_ref(256, 0);
    // Deterministic pseudo-random bytes: a 32-bit LCG's high byte.
    unsigned int seed = 12345u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        h_data[i] = static_cast<unsigned char>(seed >> 24);
        ++h_ref[h_data[i]];
    }

    unsigned char* d_data = nullptr;
    int* d_hist = nullptr;
    CHECK(cudaMalloc(&d_data, n));
    CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    CHECK(cudaMemcpy(d_data, h_data.data(), n, cudaMemcpyHostToDevice));

    histogram<<<(n + 255) / 256, 256>>>(d_data, d_hist, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int> g_hist(256);
    CHECK(cudaMemcpy(g_hist.data(), d_hist, 256 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    int histErrors = 0;
    for (int b = 0; b < 256; ++b)
        if (g_hist[b] != h_ref[b]) ++histErrors;
    std::printf("histogram: %d bin mismatches\n", histErrors);

    // ------------------------------------------------------------------
    // 5.5.2: shared-memory spinlock
    // ------------------------------------------------------------------
    std::vector<float> h_lock(n), h_lockRef(n);
    for (int i = 0; i < n; ++i)
    {
        h_lock[i]    = static_cast<float>(i);
        h_lockRef[i] = h_lock[i] * 2.0f + 1.0f;
    }

    float* d_lock = nullptr;
    CHECK(cudaMalloc(&d_lock, n * sizeof(float)));
    CHECK(cudaMemcpy(d_lock, h_lock.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice));

    // 8 blocks x 256 threads; the kernel's grid-stride loop now uses
    // blockIdx.x so every element is processed by exactly one block.
    lockedCriticalSection<<<8, 256>>>(d_lock, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_lock.data(), d_lock, n * sizeof(float),
                     cudaMemcpyDeviceToHost));

    int lockErrors = 0;
    for (int i = 0; i < n; ++i)
        if (h_lock[i] != h_lockRef[i]) ++lockErrors;
    std::printf("spinlock:  %d mismatches\n", lockErrors);

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_lock));

    if (histErrors != 0 || lockErrors != 0)
    {
        std::fprintf(stderr, "CHAPTER 5 TEST FAILED\n");
        return 1;
    }
    std::printf("Chapter 5 test PASSED (histogram + spinlock)\n");
    return 0;
}
