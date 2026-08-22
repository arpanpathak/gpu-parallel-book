// Chapter 11, 11.3 - CUB block reduction inside a custom kernel, runnable.
//
// Build: nvcc -arch=compute_60 cub_reduce.cu -o cub_reduce
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./cub_reduce

#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// The book's kernel (11.3): coarsened accumulation, then CUB BlockReduce.
template <int BLOCK_THREADS>
__global__ void reduceWithCub(const float* in, float* out, int n)
{
    // CUB block-reduction scratch space (compile-time sized).
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    // Coarsened accumulation (Chapter 8, 8.4):
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         i < n; i += gridDim.x * BLOCK_THREADS)
        sum += in[i];

    // Block-wide reduction with CUB.
    const float blockSum = BlockReduceT(temp_storage).Sum(sum);

    // Thread 0 of each block writes the block total:
    if (threadIdx.x == 0) out[blockIdx.x] = blockSum;
}

int main()
{
    const int n = 1 << 20;
    const int threads = 256;
    const int blocks  = 64;

    std::vector<float> h_in(n);
    double cpuTotal = 0.0;
    for (int i = 0; i < n; ++i)
    {
        h_in[i] = (i % 2 == 0) ? 1.0f : -0.5f;
        cpuTotal += h_in[i];
    }

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice));

    reduceWithCub<threads><<<blocks, threads>>>(d_in, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partials(blocks);
    CHECK(cudaMemcpy(h_partials.data(), d_out, blocks * sizeof(float),
                     cudaMemcpyDeviceToHost));
    double gpuTotal = 0.0;
    for (float v : h_partials) gpuTotal += v;

    const double err = std::abs(gpuTotal - cpuTotal);
    std::printf("cub: gpu = %.6f, cpu = %.6f, err = %.3g\n",
                gpuTotal, cpuTotal, err);
    if (err > 1e-3 * std::max(1.0, std::abs(cpuTotal)))
    {
        std::fprintf(stderr, "CHAPTER 11 CUB TEST FAILED\n");
        return 1;
    }
    std::printf("Chapter 11 CUB test PASSED\n");
    return 0;
}
