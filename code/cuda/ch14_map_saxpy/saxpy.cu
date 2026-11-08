// Chapter 14: SAXPY, y = a x + y, with 1 thread per element, as a generic map that
// takes a lambda, and with a grid-stride loop that uses fewer threads than elements.
// Build: nvcc -O3 -arch=native --extended-lambda saxpy.cu -o saxpy

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

const int BLOCK = 256;
const int N = 1 << 24;
const float A = 2.5f;
const int RUNS = 50;

// The worked example of the chapter: 12 elements, a = 2.
const float X12[12] = {2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5};
const float Y12[12] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// y[index] = a * x[index] + y[index], one thread per element.
__global__ void saxpy(float a, const float* x, float* y, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        y[index] = a * x[index] + y[index];
    }
}

// y[index] = f(x[index], y[index]), one thread per element, for any function f.
template <typename F>
__global__ void map(F f, const float* x, float* y, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        y[index] = f(x[index], y[index]);
    }
}

// y[index] = f(x[index], y[index]) for every index that starts at this thread's global
// index and steps by gridSize, the number of threads in the grid.
template <typename F>
__global__ void mapStride(F f, const float* x, float* y, int n)
{
    int gridSize = gridDim.x * blockDim.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += gridSize) {
        y[index] = f(x[index], y[index]);
    }
}

// Milliseconds per launch: 500 ms of warm-up, then RUNS launches timed with events.
template <typename L>
float timeMs(L launch)
{
    auto warm = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - warm < std::chrono::milliseconds(500)) {
        launch();
        check(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start));
    for (int r = 0; r < RUNS; ++r) {
        launch();
    }
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));

    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return ms / RUNS;
}

// Resets y, runs `launch` once, and counts the elements that differ from a x + y.
template <typename L>
int wrong(float* y, const std::vector<float>& x0, const std::vector<float>& y0, L launch)
{
    size_t bytes = N * sizeof(float);
    check(cudaMemcpy(y, y0.data(), bytes, cudaMemcpyHostToDevice));
    launch();
    check(cudaGetLastError());

    std::vector<float> got(N);
    check(cudaMemcpy(got.data(), y, bytes, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        bad += got[i] != A * x0[i] + y0[i];
    }
    return bad;
}

void report(const char* name, long threads, float ms, int bad)
{
    double gbs = 12.0 * N / ms / 1e6;
    double each = double(N) / threads;
    std::printf("  %-20s %9ld %7.1f", name, threads, each);
    std::printf(" %7.3f ms %6.1f GB/s %5d\n", ms, gbs, bad);
}

void show(const char* name, const float* values)
{
    std::printf("  %-20s", name);
    for (int i = 0; i < 12; ++i) {
        std::printf("%3.0f", values[i]);
    }
    std::printf("\n");
}

// Runs SAXPY with a = 2 on the 12 elements of the worked example, first with 2 blocks
// of 8 threads, then with a grid-stride loop on 2 blocks of 2 threads. Returns the
// number of wrong elements.
int workedExample()
{
    size_t bytes = 12 * sizeof(float);
    float *x, *y;
    check(cudaMalloc(&x, bytes));
    check(cudaMalloc(&y, bytes));
    check(cudaMemcpy(x, X12, bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(y, Y12, bytes, cudaMemcpyHostToDevice));

    std::printf("worked example: n = 12, a = 2\n");
    show("x", X12);
    show("y", Y12);

    float first[12], second[12];
    saxpy<<<2, 8>>>(2.0f, x, y, 12);
    check(cudaMemcpy(first, y, bytes, cudaMemcpyDeviceToHost));
    show("2 blocks of 8", first);

    float a = 2.0f;
    auto axpy = [a] __device__ (float x, float y) { return a * x + y; };
    check(cudaMemcpy(y, Y12, bytes, cudaMemcpyHostToDevice));
    mapStride<<<2, 2>>>(axpy, x, y, 12);
    check(cudaMemcpy(second, y, bytes, cudaMemcpyDeviceToHost));
    show("2 blocks of 2, loop", second);

    int bad = 0;
    for (int i = 0; i < 12; ++i) {
        float expect = 2.0f * X12[i] + Y12[i];
        bad += (first[i] != expect) + (second[i] != expect);
    }

    check(cudaFree(x));
    check(cudaFree(y));
    return bad;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;
    std::printf("GPU: %s, %d SMs, ", prop.name, sms);
    std::printf("at most %d threads per SM\n", prop.maxThreadsPerMultiProcessor);

    int failed = workedExample();

    std::printf("SAXPY on %d floats, a = %.1f: 12 bytes and 2 FLOP per element\n", N, A);
    std::vector<float> x0(N), y0(N);
    for (int i = 0; i < N; ++i) {
        x0[i] = (i % 1024) * 0.5f;
        y0[i] = i % 7;
    }

    size_t bytes = N * sizeof(float);
    float *x, *y, *copy;
    check(cudaMalloc(&x, bytes));
    check(cudaMalloc(&y, bytes));
    check(cudaMalloc(&copy, bytes));
    check(cudaMemcpy(x, x0.data(), bytes, cudaMemcpyHostToDevice));

    float a = A;
    auto axpy = [a] __device__ (float x, float y) { return a * x + y; };
    int blocks = (N + BLOCK - 1) / BLOCK;
    long all = long(blocks) * BLOCK;

    std::printf("  kernel                 threads per thr     time        rate wrong\n");

    auto plain = [&] { saxpy<<<blocks, BLOCK>>>(A, x, y, N); };
    int bad = wrong(y, x0, y0, plain);
    report("saxpy", all, timeMs(plain), bad);
    failed += bad;

    auto generic = [&] { map<<<blocks, BLOCK>>>(axpy, x, y, N); };
    bad = wrong(y, x0, y0, generic);
    report("map, lambda axpy", all, timeMs(generic), bad);
    failed += bad;

    auto larger = [] __device__ (float x, float y) { return fmaxf(x, y); };
    check(cudaMemcpy(y, y0.data(), bytes, cudaMemcpyHostToDevice));
    map<<<blocks, BLOCK>>>(larger, x, y, N);

    std::vector<float> got(N);
    check(cudaMemcpy(got.data(), y, bytes, cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < N; ++i) {
        bad += got[i] != std::max(x0[i], y0[i]);
    }
    std::printf("  map, lambda max: %d wrong\n", bad);
    failed += bad;

    for (int k : {1, 2, 3, 6, 12, 48}) {
        int grid = sms * k;
        auto loop = [&] { mapStride<<<grid, BLOCK>>>(axpy, x, y, N); };
        bad = wrong(y, x0, y0, loop);

        std::string name = "loop, " + std::to_string(k) + " per SM";
        report(name.c_str(), long(grid) * BLOCK, timeMs(loop), bad);
        failed += bad;
    }

    auto copyAll = [&] { cudaMemcpyAsync(copy, x, bytes, cudaMemcpyDeviceToDevice); };
    float ms = timeMs(copyAll);
    double gbs = 8.0 * N / ms / 1e6;
    std::printf("  device-to-device copy, 8 bytes per element ");
    std::printf(" %7.3f ms %6.1f GB/s\n", ms, gbs);

    check(cudaFree(x));
    check(cudaFree(y));
    check(cudaFree(copy));

    if (failed == 0) {
        std::printf("PASSED\n");
    } else {
        std::printf("FAILED: %d wrong elements\n", failed);
    }
    return 0;
}
