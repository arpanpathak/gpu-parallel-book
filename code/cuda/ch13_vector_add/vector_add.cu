// Chapter 13: vector addition on the GPU in CUDA C++, c[i] = a[i] + b[i].
// Build: nvcc -O3 -arch=native vector_add.cu -o vector_add

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Threads per block. The kernel's launch bounds name the same number.
const int BLOCK = 256;

// The number of elements. It is not a multiple of BLOCK, so the last
// block has threads with no element to add.
const int N = 1000003;

// Stops the program if a CUDA call failed.
void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// One thread adds one pair of elements.
__global__ void __launch_bounds__(256)
vectorAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// The same computation on one CPU core, used as the reference answer.
std::vector<float> vectorAddCpu(const std::vector<float>& a, const std::vector<float>& b)
{
    std::vector<float> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
    return c;
}

// Milliseconds since `start`.
double msSince(std::chrono::steady_clock::time_point start)
{
    auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

int main()
{
    // 1. Inputs on the host.
    std::vector<float> a(N), b(N);
    for (int i = 0; i < N; ++i) {
        a[i] = float(i);
        b[i] = 2.0f * float(i);
    }

    // 2. GPU 0, and a stream that orders our work on it.
    check(cudaSetDevice(0));
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    // 3. Copy the inputs to device memory and allocate the output.
    size_t bytes = N * sizeof(float);
    float *a_dev, *b_dev, *c_dev;
    check(cudaMalloc(&a_dev, bytes));
    check(cudaMalloc(&b_dev, bytes));
    check(cudaMalloc(&c_dev, bytes));
    check(cudaMemcpyAsync(a_dev, a.data(), bytes, cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(b_dev, b.data(), bytes, cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(c_dev, 0, bytes, stream));

    // 4. Enough blocks to cover N: ceil(N / BLOCK).
    int blocks = (N + BLOCK - 1) / BLOCK;

    // 5. Launch, then copy the result back. The copy waits for the kernel.
    std::vector<float> c(N);
    auto start = std::chrono::steady_clock::now();
    vectorAdd<<<blocks, BLOCK, 0, stream>>>(a_dev, b_dev, c_dev, N);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(c.data(), c_dev, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    double gpuMs = msSince(start);

    // 6. Check every element against the CPU.
    start = std::chrono::steady_clock::now();
    std::vector<float> expected = vectorAddCpu(a, b);
    double cpuMs = msSince(start);

    int wrong = 0;
    for (int i = 0; i < N; ++i) {
        wrong += c[i] != expected[i];
    }

    std::printf("elements      %d\n", N);
    std::printf("blocks        %d x %d threads", blocks, BLOCK);
    std::printf(" = %d threads\n", blocks * BLOCK);
    std::printf("idle threads  %d\n", blocks * BLOCK - N);
    std::printf("c[0..4]       [%.1f, %.1f, %.1f, %.1f]\n", c[0], c[1], c[2], c[3]);
    std::printf("c[N-1]        %.0f\n", c[N - 1]);
    std::printf("gpu + copy    %.6fms\n", gpuMs);
    std::printf("cpu           %.6fms\n", cpuMs);

    check(cudaFree(a_dev));
    check(cudaFree(b_dev));
    check(cudaFree(c_dev));
    check(cudaStreamDestroy(stream));

    if (wrong == 0) {
        std::printf("PASSED: all %d elements match the CPU\n", N);
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d elements differ from the CPU\n", wrong);
    return 1;
}
