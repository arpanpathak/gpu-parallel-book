// Chapter 10 - runnable test for the RAII DeviceBuffer<T> (10.2).
//
// Build: nvcc -std=c++20 -arch=compute_60 test.cu -o test
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./test

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "device_buffer.hpp"

// The Chapter 3 kernel, unchanged; DeviceBuffer::data() plugs straight in.
__global__ void addVectors(const float* a, const float* b, float* c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

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
    const int n = 1 << 20;

    // --- RAII allocation + typed copies --------------------------------
    DeviceBuffer<float> d_a(n), d_b(n), d_c(n);

    std::vector<float> h_a(n), h_b(n);
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
    }
    d_a.copyToDevice(h_a.data());
    d_b.copyToDevice(h_b.data());

    // --- Launch with raw pointers from data() --------------------------
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    addVectors<<<blocks, threads>>>(d_a.data(), d_b.data(), d_c.data(), n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_c(n);
    d_c.copyToHost(h_c.data());

    double maxErr = 0.0;
    for (int i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)h_c[i] - 3.0 * i));
    std::printf("DeviceBuffer vector-add: max error = %g\n", maxErr);

    // --- Move semantics: ownership transfers, no double-free ------------
    DeviceBuffer<float> d_moved(std::move(d_a));
    if (d_a.data() != nullptr || d_a.size() != 0 ||
        d_moved.size() != static_cast<std::size_t>(n))
    {
        std::fprintf(stderr, "move semantics broken\n");
        return 1;
    }
    std::printf("DeviceBuffer move: source emptied, target size = %zu\n",
                d_moved.size());

    // --- Destructors free everything automatically ----------------------
    if (maxErr > 0.0)
    {
        std::fprintf(stderr, "CHAPTER 10 TEST FAILED\n");
        return 1;
    }
    std::printf("Chapter 10 test PASSED (RAII DeviceBuffer)\n");
    return 0;
}
