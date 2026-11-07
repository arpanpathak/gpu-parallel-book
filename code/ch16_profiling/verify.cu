// Chapter 16: GPU results against the CPU, with a tolerance worked out in advance.
// Build: nvcc -arch=native verify.cu -o verify

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void expAccurate(const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = expf(x[i]);
    }
}

__global__ void expFast(const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __expf(x[i]);
    }
}

// Each thread sums a strided share of x; the block adds the 256 sums as a tree.
__global__ void sum(const float* x, int n, float* blockSums)
{
    __shared__ float s[256];
    int t = threadIdx.x;

    float v = 0.0f;
    for (int i = blockIdx.x * 256 + t; i < n; i += gridDim.x * 256) {
        v += x[i];
    }
    s[t] = v;
    __syncthreads();

    for (int stride = 128; stride > 0; stride /= 2) {
        if (t < stride) {
            s[t] += s[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        blockSums[blockIdx.x] = s[0];
    }
}

// The number of floats between a and b, for 2 floats of the same sign.
int64_t ulps(float a, float b)
{
    int32_t ia, ib;
    std::memcpy(&ia, &a, sizeof(float));
    std::memcpy(&ib, &b, sizeof(float));
    return std::llabs(int64_t(ia) - int64_t(ib));
}

// The largest distance in ULPs between y and the nearest float to exp(x).
int64_t expError(const std::vector<float>& x, const std::vector<float>& y)
{
    int64_t worst = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        float exact = float(std::exp(double(x[i])));
        worst = std::max(worst, ulps(y[i], exact));
    }
    return worst;
}

// 2 launches: 1,024 blocks write 1,024 partial sums, then 1 block adds them.
float gpuSum(const float* dx, int n, float* partial)
{
    sum<<<1024, 256>>>(dx, n, partial);
    sum<<<1, 256>>>(partial, 1024, partial + 1024);
    check(cudaGetLastError());

    float total;
    check(cudaMemcpy(&total, partial + 1024, sizeof(float), cudaMemcpyDeviceToHost));
    return total;
}

void expTest()
{
    const int n = 1 << 20;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = -10.0f + 20.0f * i / n;
    }

    float *dx, *dy;
    check(cudaMalloc(&dx, n * sizeof(float)));
    check(cudaMalloc(&dy, n * sizeof(float)));
    check(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int blocks = (n + 255) / 256;

    std::printf("y = exp(x) for 1,048,576 values of x in [-10, 10)\n");

    expAccurate<<<blocks, 256>>>(dx, dy, n);
    check(cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  expf    largest error %3lld ULPs\n", (long long)expError(x, y));

    expFast<<<blocks, 256>>>(dx, dy, n);
    check(cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  __expf  largest error %3lld ULPs\n", (long long)expError(x, y));

    check(cudaFree(dx));
    check(cudaFree(dy));
}

const char* verdict(double error, double tolerance)
{
    return error <= tolerance ? "PASSED" : "FAILED";
}

// Sum x on the GPU twice: correctly, and as a caller passing 1 element too few.
void sumTest(const char* title, const std::vector<float>& x, double tolerance)
{
    int n = x.size();
    float *dx, *partial;
    check(cudaMalloc(&dx, n * sizeof(float)));
    check(cudaMalloc(&partial, 1025 * sizeof(float)));
    check(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    double exact = 0.0;
    for (float v : x) {
        exact += v;
    }

    double good = std::fabs(gpuSum(dx, n, partial) - exact);
    double bad = std::fabs(gpuSum(dx, n - 1, partial) - exact);

    std::printf("%s\n", title);
    std::printf("  exact sum       %12.2f\n", exact);
    std::printf("  tolerance       %12.2f\n", tolerance);
    std::printf("  n elements      error %6.2f  %s\n", good, verdict(good, tolerance));
    std::printf("  n - 1 elements  error %6.2f  %s\n", bad, verdict(bad, tolerance));

    check(cudaFree(dx));
    check(cudaFree(partial));
}

int main()
{
    expTest();

    const int n = 1 << 24;
    std::vector<float> random(n), ones(n, 1.0f);
    uint32_t state = 1;
    double magnitude = 0.0;
    for (float& v : random) {
        state = state * 1664525u + 1013904223u;
        v = (state >> 8) / 16777216.0f;
        magnitude += v;
    }

    // Each element passes through at most 84 additions on the way to the total:
    // 64 + 8 in launch 1, 4 + 8 in launch 2. Each rounds by 2^-24 of its sum at most.
    double bound = 84 * std::ldexp(1.0, -24) * magnitude;
    sumTest("sum of 16,777,216 random floats in [0, 1)", random, bound);

    // Partial sums of ones are whole numbers up to 2^24, which floats hold exactly.
    sumTest("sum of 16,777,216 ones", ones, 0.0);
    return 0;
}
