// Chapter 7: coalescing, vector loads, and a matrix transpose in shared memory, measured.
// Build: nvcc -arch=native memory.cu -o memory

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const int TILE = 32;

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 1. Thread i reads element i * stride, so neighbouring threads read `stride` floats apart.
__global__ void strided(const float* in, float* out, int n, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[(size_t)i * stride];
    }
}

__global__ void fill(float* x, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = float(i % 1000);
    }
}

// 2. A copy, 1 float or 4 floats per thread.
__global__ void copy1(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

__global__ void copy4(const float4* in, float4* out, int n4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        out[i] = in[i];
    }
}

// 3. Transposes of a width x width matrix, 1 thread per element, blocks of 32 x 32.
__global__ void transposeNaive(const float* in, float* out, int width)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    out[x * width + y] = in[y * width + x];
}

template <int PAD>
__global__ void transposeTiled(const float* in, float* out, int width)
{
    __shared__ float tile[TILE][TILE + PAD];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    out[y * width + x] = tile[threadIdx.x][threadIdx.y];
}

// Average time of 20 runs of `launch`, in milliseconds, after 1 warm-up run.
template <typename F>
double timeKernel(F launch)
{
    launch();
    check(cudaDeviceSynchronize());

    double t0 = nowMs();
    for (int r = 0; r < 20; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    return (nowMs() - t0) / 20;
}

double gbps(double bytes, double ms)
{
    return bytes / 1e9 / (ms / 1e3);
}

void coalescing(const float* in, float* out)
{
    const int n = 16 << 20;                   // 16,777,216 threads, 64 MiB read and 64 MiB written
    std::printf("each thread reads 1 float, neighbours `stride` floats apart\n");

    for (int stride = 1; stride <= 32; stride *= 2) {
        double ms = timeKernel([&] { strided<<<n / 256, 256>>>(in, out, n, stride); });
        double useful = 2.0 * n * sizeof(float);
        std::printf("  stride %2d   %7.3f ms   %6.1f GB/s useful\n", stride, ms, gbps(useful, ms));
    }
}

void vectors(const float* in, float* out, int n)
{
    double ms1 = timeKernel([&] { copy1<<<n / 256, 256>>>(in, out, n); });
    double ms4 = timeKernel([&] {
        copy4<<<n / 4 / 256, 256>>>((const float4*)in, (float4*)out, n / 4);
    });

    std::printf("copy of %d MiB\n", int(n * sizeof(float) >> 20));
    std::printf("  1 float per thread    %6.2f ms   %6.1f GB/s\n", ms1, gbps(2.0 * n * 4, ms1));
    std::printf("  4 floats per thread   %6.2f ms   %6.1f GB/s\n", ms4, gbps(2.0 * n * 4, ms4));
}

void transposes(const float* in, float* out, int width)
{
    dim3 grid(width / TILE, width / TILE), block(TILE, TILE);
    double bytes = 2.0 * width * width * sizeof(float);

    double copy = timeKernel([&] { copy1<<<width * width / 256, 256>>>(in, out, width * width); });
    double naive = timeKernel([&] { transposeNaive<<<grid, block>>>(in, out, width); });
    double tiled = timeKernel([&] { transposeTiled<0><<<grid, block>>>(in, out, width); });
    double padded = timeKernel([&] { transposeTiled<1><<<grid, block>>>(in, out, width); });

    std::printf("transpose of a %d x %d matrix\n", width, width);
    std::printf("  copy, for reference   %6.2f ms   %6.1f GB/s\n", copy, gbps(bytes, copy));
    std::printf("  naive                 %6.2f ms   %6.1f GB/s\n", naive, gbps(bytes, naive));
    std::printf("  tiled                 %6.2f ms   %6.1f GB/s\n", tiled, gbps(bytes, tiled));
    std::printf("  tiled and padded      %6.2f ms   %6.1f GB/s\n", padded, gbps(bytes, padded));
}

bool isTranspose(const float* in, float* out, int width)
{
    transposeTiled<1><<<dim3(width / TILE, width / TILE), dim3(TILE, TILE)>>>(in, out, width);

    std::vector<float> a(width * width), b(width * width);
    check(cudaMemcpy(a.data(), in, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), out, b.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int y = 0; y < width; ++y) {
        for (int x = 0; x < width; ++x) {
            if (b[x * width + y] != a[y * width + x]) return false;
        }
    }
    return true;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs, L2 cache %d MiB\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20);

    const int width = 4096;
    const int n = width * width;              // 16,777,216 floats, 64 MiB

    const size_t inLen = (size_t)32 * n;      // 2 GiB: room for stride 32, too big for any L2

    float *in, *out;
    check(cudaMalloc(&in, inLen * sizeof(float)));
    check(cudaMalloc(&out, n * sizeof(float)));
    fill<<<(inLen + 255) / 256, 256>>>(in, inLen);

    for (int k = 0; k < 10; ++k) {            // warm up: the clocks rise under sustained load
        timeKernel([&] { copy1<<<n / 256, 256>>>(in, out, n); });
    }
    coalescing(in, out);
    vectors(in, out, n);
    transposes(in, out, width);
    std::printf("tiled transpose correct: %s\n", isTranspose(in, out, width) ? "yes" : "no");

    check(cudaFree(in));
    check(cudaFree(out));
}
