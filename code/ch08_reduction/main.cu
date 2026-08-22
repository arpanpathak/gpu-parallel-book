// Chapter 8 - runnable test harness for the book's reduction/scan/histogram
// kernels. The kernels live in reduction.cu, verbatim from the book.
//
// Build: nvcc -rdc=true -arch=compute_60 reduction.cu main.cu -o reduction
//        (-rdc=true lets main.cu call the __device__ scanBlock across TUs)
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./reduction

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Kernels/device functions defined in reduction.cu.
__global__ void reduceFull(const float* in, float* out, int n);
__device__ void scanBlock(float* s);
__global__ void histogramPrivatised(const unsigned char* data, int* g_hist,
                                    int n);

#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// Wrapper that lets the host test scanBlock (a __device__ function).
__global__ void scanBlockLauncher(const float* in, float* out, int blockSize)
{
    __shared__ float s[1024];          // enough for blockSize <= 1024
    const int i = threadIdx.x;
    if (i < blockSize) s[i] = in[i];
    __syncthreads();
    scanBlock(s);                      // uses blockDim.x as n internally
    if (i < blockSize) out[i] = s[i];
}

int main()
{
    const int n = 1 << 20;   // 1,048,576 elements
    int failures = 0;

    // ------------------------------------------------------------------
    // reduceFull (8.6): compare the total against a CPU double sum.
    // ------------------------------------------------------------------
    std::vector<float> h_in(n);
    double cpuTotal = 0.0;
    for (int i = 0; i < n; ++i)
    {
        // Alternating values keep the sum interesting but well-conditioned.
        h_in[i] = (i % 2 == 0) ? 1.0f : -0.5f;
        cpuTotal += h_in[i];
    }

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, 64 * sizeof(float)));   // one slot per block
    CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice));

    const int blocks = 64, threads = 256;
    reduceFull<<<blocks, threads>>>(d_in, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partials(blocks);
    CHECK(cudaMemcpy(h_partials.data(), d_out, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    double gpuTotal = 0.0;
    for (float v : h_partials) gpuTotal += v;
    const double reduceErr = std::abs(gpuTotal - cpuTotal);
    std::printf("reduceFull: gpu=%.6f cpu=%.6f err=%.3g\n",
                gpuTotal, cpuTotal, reduceErr);
    if (reduceErr > 1e-3 * std::max(1.0, std::abs(cpuTotal))) ++failures;

    // ------------------------------------------------------------------
    // scanBlock (8.7): exclusive prefix of the first 256 elements.
    // ------------------------------------------------------------------
    const int scanN = 256;
    std::vector<float> h_scanIn(scanN), h_scanRef(scanN);
    float run = 0.0f;
    for (int i = 0; i < scanN; ++i)
    {
        h_scanIn[i] = static_cast<float>(i + 1);
        h_scanRef[i] = run;            // exclusive: sum of elements before i
        run += h_scanIn[i];
    }

    float* d_scanIn  = nullptr;
    float* d_scanOut = nullptr;
    CHECK(cudaMalloc(&d_scanIn,  scanN * sizeof(float)));
    CHECK(cudaMalloc(&d_scanOut, scanN * sizeof(float)));
    CHECK(cudaMemcpy(d_scanIn, h_scanIn.data(), scanN * sizeof(float),
                     cudaMemcpyHostToDevice));

    scanBlockLauncher<<<1, scanN>>>(d_scanIn, d_scanOut, scanN);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_scanOut(scanN);
    CHECK(cudaMemcpy(h_scanOut.data(), d_scanOut, scanN * sizeof(float),
                     cudaMemcpyDeviceToHost));
    int scanErrors = 0;
    for (int i = 0; i < scanN; ++i)
        if (std::abs(h_scanOut[i] - h_scanRef[i]) > 1e-4f) ++scanErrors;
    std::printf("scanBlock:  %d mismatches (exclusive prefix of 256)\n",
                scanErrors);
    if (scanErrors != 0) ++failures;

    // ------------------------------------------------------------------
    // histogramPrivatised (8.8)
    // ------------------------------------------------------------------
    std::vector<unsigned char> h_histData(n);
    std::vector<int> h_histRef(256, 0);
    unsigned int seed = 987654u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        h_histData[i] = static_cast<unsigned char>(seed >> 24);
        ++h_histRef[h_histData[i]];
    }

    unsigned char* d_histData = nullptr;
    int* d_hist = nullptr;
    CHECK(cudaMalloc(&d_histData, n));
    CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    CHECK(cudaMemcpy(d_histData, h_histData.data(), n, cudaMemcpyHostToDevice));

    histogramPrivatised<<<(n + threads - 1) / threads, threads>>>(
        d_histData, d_hist, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int> h_hist(256);
    CHECK(cudaMemcpy(h_hist.data(), d_hist, 256 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    int histErrors = 0;
    for (int b = 0; b < 256; ++b)
        if (h_hist[b] != h_histRef[b]) ++histErrors;
    std::printf("histogramPrivatised: %d bin mismatches\n", histErrors);
    if (histErrors != 0) ++failures;

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_scanIn));
    CHECK(cudaFree(d_scanOut));
    CHECK(cudaFree(d_histData));
    CHECK(cudaFree(d_hist));

    if (failures != 0)
    {
        std::fprintf(stderr, "CHAPTER 8 TEST FAILED (%d checks)\n", failures);
        return 1;
    }
    std::printf("Chapter 8 test PASSED (reduction + scan + histogram)\n");
    return 0;
}
