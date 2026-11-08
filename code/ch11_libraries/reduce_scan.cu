// Chapter 11: Thrust and CUB against the hand-written sum and scan of Chapter 8.
// Build: nvcc -arch=native -O2 -std=c++17 reduce_scan.cu -o reduce_scan

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

const int BLOCK = 256;                        // threads per block
const int GRID = 1024;                        // blocks of the hand-written sums

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

// Average time of `runs` runs of `run`, in milliseconds, after 1 warm-up run.
template <typename F>
double timeMs(F run, int runs = 20)
{
    run();
    check(cudaDeviceSynchronize());

    double t0 = nowMs();
    for (int r = 0; r < runs; ++r) {
        run();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    return (nowMs() - t0) / runs;
}

unsigned nextRandom(unsigned& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

// Chapter 8's fastest sum: float4 loads, warp shuffles, 1 barrier per block.
__device__ float warpSum(float v)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ float blockSum(float v)
{
    __shared__ float warps[BLOCK / 32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    v = warpSum(v);
    if (lane == 0) {
        warps[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        v = warpSum(lane < BLOCK / 32 ? warps[lane] : 0.0f);
    }
    return v;                                 // the block's total, in thread 0
}

__global__ void sumHand(const float4* in, float* partial, int n4)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n4; i += gridDim.x * BLOCK) {
        float4 v = in[i];
        sum += v.x + v.y + v.z + v.w;
    }

    sum = blockSum(sum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}

// The same kernel, with CUB's block reduction in place of warpSum and blockSum.
__global__ void sumCub(const float4* in, float* partial, int n4)
{
    using BlockReduce = cub::BlockReduce<float, BLOCK>;
    __shared__ BlockReduce::TempStorage temp;

    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n4; i += gridDim.x * BLOCK) {
        float4 v = in[i];
        sum += v.x + v.y + v.z + v.w;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}

struct Square {
    __host__ __device__ float operator()(float v) const { return v * v; }
};

// Prints 1 line of results; true if `sum` is within 1 part in 100,000 of `exact`.
bool report(const char* name, double ms, double bytes, float sum, double exact)
{
    double gb = bytes / ms / 1e6;
    std::printf("  %-28s %7.2f ms  %6.1f GB/s  sum %.1f\n", name, ms, gb, sum);
    return std::abs(sum - exact) < 1e-5 * exact;
}

float readBack(const float* total)
{
    float t;
    check(cudaMemcpy(&t, total, sizeof(float), cudaMemcpyDeviceToHost));
    return t;
}

bool sums(const float* in, float* partial, float* total, int n, double exact)
{
    double bytes = n * 4.0;
    const float4* in4 = (const float4*)in;
    const float4* partial4 = (const float4*)partial;
    std::printf("sum of %d floats (%d MiB), average of 20 runs\n", n, n >> 18);

    auto hand = [&] {
        sumHand<<<GRID, BLOCK>>>(in4, partial, n / 4);
        sumHand<<<1, BLOCK>>>(partial4, total, GRID / 4);
    };
    double ms = timeMs(hand);
    bool ok = report("Chapter 8, float4 + shuffles", ms, bytes, readBack(total), exact);

    auto cubBlock = [&] {
        sumCub<<<GRID, BLOCK>>>(in4, partial, n / 4);
        sumCub<<<1, BLOCK>>>(partial4, total, GRID / 4);
    };
    ms = timeMs(cubBlock);
    ok &= report("same, cub::BlockReduce", ms, bytes, readBack(total), exact);

    size_t tempBytes = 0;
    cub::DeviceReduce::Sum(nullptr, tempBytes, in, total, n);    // asks for the size only
    void* temp;
    check(cudaMalloc(&temp, tempBytes));
    auto cubDevice = [&] { cub::DeviceReduce::Sum(temp, tempBytes, in, total, n); };
    ms = timeMs(cubDevice);
    ok &= report("cub::DeviceReduce::Sum", ms, bytes, readBack(total), exact);
    std::printf("    (CUB asked for %zu bytes of temporary storage)\n", tempBytes);

    float t = 0.0f;
    auto thrustSum = [&] { t = thrust::reduce(thrust::device, in, in + n, 0.0f); };
    ms = timeMs(thrustSum);
    ok &= report("thrust::reduce", ms, bytes, t, exact);

    check(cudaFree(temp));
    return ok;
}

bool sumsOfSquares(const float* in, float* squares, int n, double exact)
{
    double bytes = n * 4.0;
    Square square;
    cuda::std::plus<float> plus;
    std::printf("sum of squares, GB/s of input read\n");

    float t = 0.0f;
    auto twoPasses = [&] {
        thrust::transform(thrust::device, in, in + n, squares, square);
        t = thrust::reduce(thrust::device, squares, squares + n, 0.0f);
    };
    double ms = timeMs(twoPasses);
    bool ok = report("transform, then reduce", ms, bytes, t, exact);

    auto fused = [&] {
        t = thrust::transform_reduce(thrust::device, in, in + n, square, 0.0f, plus);
    };
    ms = timeMs(fused);
    ok &= report("transform_reduce", ms, bytes, t, exact);
    return ok;
}

void smallSums(const float* in, float* total)
{
    const int n = 4096;
    float t = 0.0f;
    std::printf("sum of %d floats, microseconds per sum\n", n);

    auto thrustSum = [&] { t = thrust::reduce(thrust::device, in, in + n, 0.0f); };
    std::printf("  thrust::reduce                   %6.1f us\n", timeMs(thrustSum) * 1e3);

    size_t tempBytes = 0;
    cub::DeviceReduce::Sum(nullptr, tempBytes, in, total, n);
    void* temp;
    check(cudaMalloc(&temp, tempBytes));

    auto cubAndCopy = [&] {
        cub::DeviceReduce::Sum(temp, tempBytes, in, total, n);
        check(cudaMemcpy(&t, total, sizeof(float), cudaMemcpyDeviceToHost));
    };
    double usCopy = timeMs(cubAndCopy) * 1e3;
    std::printf("  cub::DeviceReduce, result copied %6.1f us\n", usCopy);

    auto cubOnly = [&] { cub::DeviceReduce::Sum(temp, tempBytes, in, total, n); };
    double usOnly = timeMs(cubOnly) * 1e3;
    std::printf("  cub::DeviceReduce, result kept   %6.1f us\n", usOnly);

    check(cudaFree(temp));
}

bool scans(int n)
{
    std::vector<unsigned> values(n), inclusive(n), result(n);
    unsigned state = 2;
    unsigned running = 0;
    for (int i = 0; i < n; ++i) {
        values[i] = nextRandom(state) >> 30;  // 0, 1, 2 or 3
        running += values[i];
        inclusive[i] = running;
    }

    unsigned *in, *out;
    check(cudaMalloc(&in, n * sizeof(unsigned)));
    check(cudaMalloc(&out, n * sizeof(unsigned)));
    check(cudaMemcpy(in, values.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    double bytes = 2.0 * n * sizeof(unsigned);    // each value read once, written once
    std::printf("scan of %d values 0 to 3, average of 20 runs\n", n);

    auto thrustScan = [&] { thrust::inclusive_scan(thrust::device, in, in + n, out); };
    double ms = timeMs(thrustScan);
    check(cudaMemcpy(result.data(), out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool ok = result == inclusive;
    std::printf("  thrust::inclusive_scan         %6.2f ms", ms);
    std::printf("  %6.1f GB/s", bytes / ms / 1e6);
    std::printf("  last %u  correct: %s\n", result[n - 1], ok ? "yes" : "no");

    size_t tempBytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, in, out, n);
    void* temp;
    check(cudaMalloc(&temp, tempBytes));

    auto cubScan = [&] { cub::DeviceScan::ExclusiveSum(temp, tempBytes, in, out, n); };
    ms = timeMs(cubScan);
    check(cudaMemcpy(result.data(), out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool okEx = true;
    for (int i = 0; i < n; ++i) {
        okEx = okEx && result[i] == inclusive[i] - values[i];
    }
    std::printf("  cub::DeviceScan::ExclusiveSum  %6.2f ms", ms);
    std::printf("  %6.1f GB/s", bytes / ms / 1e6);
    std::printf("  last %u  correct: %s\n", result[n - 1], okEx ? "yes" : "no");

    check(cudaFree(temp));
    check(cudaFree(in));
    check(cudaFree(out));
    return ok && okEx;
}

// An inclusive scan of each block of 256 values, 1 value per thread, with CUB.
__global__ void scanBlocks(const unsigned* in, unsigned* out, int n)
{
    using BlockScan = cub::BlockScan<unsigned, BLOCK>;
    __shared__ BlockScan::TempStorage temp;
    int i = blockIdx.x * BLOCK + threadIdx.x;

    unsigned v = i < n ? in[i] : 0;
    BlockScan(temp).InclusiveSum(v, v);
    if (i < n) {
        out[i] = v;
    }
}

bool blockScans(int n)
{
    std::vector<unsigned> values(n), result(n);
    unsigned state = 2;
    for (int i = 0; i < n; ++i) {
        values[i] = nextRandom(state) >> 30;  // 0, 1, 2 or 3, as in scans()
    }

    unsigned *in, *out;
    check(cudaMalloc(&in, n * sizeof(unsigned)));
    check(cudaMalloc(&out, n * sizeof(unsigned)));
    check(cudaMemcpy(in, values.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    scanBlocks<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(in, out, n);
    check(cudaMemcpy(result.data(), out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));

    bool ok = true;
    unsigned running = 0;
    for (int i = 0; i < n; ++i) {
        running = (i % BLOCK == 0 ? 0 : running) + values[i];
        ok = ok && result[i] == running;
    }
    std::printf("cub::BlockScan of %d values, 256 per block\n", n);
    std::printf("  first block:");
    for (int i = 0; i < 8; ++i) {
        std::printf(" %u", result[i]);
    }
    std::printf(" ... %u  correct: %s\n", result[BLOCK - 1], ok ? "yes" : "no");

    check(cudaFree(in));
    check(cudaFree(out));
    return ok;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs", prop.name, prop.multiProcessorCount);
    std::printf(", L2 cache %d MiB\n", prop.l2CacheSize >> 20);

    const int n = 64 << 20;                   // 67,108,864 floats, 256 MiB
    std::vector<float> values(n);
    unsigned state = 1;
    double exact = 0.0;
    double exactSquares = 0.0;
    for (int i = 0; i < n; ++i) {
        values[i] = (nextRandom(state) >> 8) / 16777216.0f;  // in [0, 1), as in Chapter 8
        exact += values[i];
        exactSquares += double(values[i]) * values[i];
    }

    float *in, *partial, *total, *squares;
    check(cudaMalloc(&in, n * sizeof(float)));
    check(cudaMalloc(&partial, GRID * sizeof(float)));
    check(cudaMalloc(&total, sizeof(float)));
    check(cudaMalloc(&squares, n * sizeof(float)));
    check(cudaMemcpy(in, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    for (int k = 0; k < 10; ++k) {            // warm up: clocks rise under sustained load
        timeMs([&] { sumHand<<<GRID, BLOCK>>>((const float4*)in, partial, n / 4); });
    }

    bool ok = sums(in, partial, total, n, exact);
    ok &= sumsOfSquares(in, squares, n, exactSquares);
    std::printf("  exact (double): sum %.1f, sum of squares %.1f\n", exact, exactSquares);
    smallSums(in, total);

    check(cudaFree(in));
    check(cudaFree(partial));
    check(cudaFree(total));
    check(cudaFree(squares));

    ok &= scans(n);
    ok &= blockScans(n);
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
}
