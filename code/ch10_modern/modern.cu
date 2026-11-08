// Chapter 10: the work of raw.cu with RAII, templates, span, atomic_ref and concepts.
// Build: nvcc -std=c++20 --extended-lambda -arch=native modern.cu -o modern

#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda/atomic>
#include "device_buffer.hpp"

constexpr int BLOCK = 256;
static_assert(BLOCK % 32 == 0, "a block is a whole number of warps");
static_assert(BLOCK <= 1024, "a block holds at most 1,024 threads");

constexpr int blocksFor(std::size_t n)
{
    return static_cast<int>((n + BLOCK - 1) / BLOCK);
}

static_assert(blocksFor(1000) == 4);

const int N = 1 << 21;                        // 2,097,152 values, as in raw.cu
const int BIG = 1 << 24;                      // 16,777,216 values for timing

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <typename F, typename T>
concept UnaryOp = std::invocable<F, T> && std::same_as<std::invoke_result_t<F, T>, T>;

// out[i] = f(in[i]), for any number type T and any operation f from T to T.
template <Number T, UnaryOp<T> F>
__global__ void transform(cuda::std::span<const T> in, cuda::std::span<T> out, F f)
{
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < in.size()) {
        out[i] = f(in[i]);
    }
}

template <Number T>
struct Scale {
    T factor;

    __host__ __device__ T operator()(T x) const { return factor * x; }
};

// The C-style kernel of raw.cu, kept to compare its speed.
__global__ void scaleRaw(const float* in, float* out, int n, float factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = factor * in[i];
    }
}

// 1 + x + x^2/2 + x^3/6, the first 4 terms of e^x, by Horner's rule.
__host__ __device__ float expPoly(float x)
{
    return 1.0f + x * (1.0f + x * (0.5f + x * (1.0f / 6.0f)));
}

// The same polynomial, with every multiply-add rounded once, on both sides.
__host__ __device__ float expPolyFma(float x)
{
    float p = fmaf(x, 1.0f / 6.0f, 0.5f);
    p = fmaf(x, p, 1.0f);
    return fmaf(x, p, 1.0f);
}

// Count the values above a threshold. atomic_ref makes an ordinary int atomic.
template <Number T>
__global__ void countAbove(cuda::std::span<const T> in, T threshold, int* count)
{
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < in.size() && in[i] > threshold) {
        cuda::atomic_ref<int, cuda::thread_scope_device> total(*count);
        total.fetch_add(1, cuda::memory_order_relaxed);
    }
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Average time of `runs` runs of `launch`, in milliseconds, after 500 ms of warm-up:
// the clocks of the GPU and of its memory rise under sustained load.
template <typename F>
double timeKernel(F launch, int runs)
{
    double start = nowMs();
    while (nowMs() - start < 500.0) {
        launch();
        check(cudaDeviceSynchronize());
    }

    double t0 = nowMs();
    for (int r = 0; r < runs; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    return (nowMs() - t0) / runs;
}

// The function of raw.cu: the early return frees both buffers.
bool scaleOnGpu(const std::vector<float>& host, std::vector<float>& result, float factor)
{
    DeviceBuffer<float> in(host);
    DeviceBuffer<float> out(host.size());

    if (factor == 0.0f) {
        return false;
    }

    transform<<<blocksFor(in.size()), BLOCK>>>(in.view(), out.span(), Scale{factor});
    check(cudaGetLastError());
    result = out.toHost();
    return true;
}

void launchTooWide(const std::vector<float>& host)
{
    DeviceBuffer<float> in(host);
    DeviceBuffer<float> out(host.size());

    transform<<<N / 2048, 2048>>>(in.view(), out.span(), Scale{2.0f});
    check(cudaGetLastError());
}

// The only way to skip a destructor: a buffer created with new and never deleted.
void leakOnGpu(const std::vector<float>& host)
{
    new DeviceBuffer<float>(host);
}

// Time transform<T> on BIG values; return true if every result is right.
template <Number T>
bool timeType(const char* name, int runs)
{
    std::vector<T> host(BIG);
    for (int i = 0; i < BIG; ++i) {
        host[i] = static_cast<T>(i % 1000);
    }

    DeviceBuffer<T> in(host);
    DeviceBuffer<T> out(host.size());
    Scale<T> triple{3};

    auto launch = [&] {
        transform<<<blocksFor(BIG), BLOCK>>>(in.view(), out.span(), triple);
    };
    double ms = timeKernel(launch, runs);
    double gbs = 2.0 * BIG * sizeof(T) / ms / 1e6;
    std::printf("  %-22s %7.2f ms  %6.1f GB/s\n", name, ms, gbs);

    std::vector<T> result = out.toHost();
    for (int i = 0; i < BIG; ++i) {
        if (result[i] != triple(host[i])) {
            return false;
        }
    }
    return true;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    std::vector<float> host(N, 1.0f);
    std::vector<float> result(N, 0.0f);
    bool passed = true;

    std::printf("ownership\n");
    double before = poolMiB();
    for (int call = 0; call < 8; ++call) {
        scaleOnGpu(host, result, 0.0f);
    }
    double lost = poolMiB() - before;
    std::printf("  8 calls that return early: %.1f MiB lost\n", lost);
    passed = passed && lost == 0.0;

    try {
        launchTooWide(host);
        passed = false;
    } catch (const std::runtime_error& e) {
        std::printf("  2048 threads per block threw: %s\n", e.what());
    }
    std::printf("  after the exception: %.1f MiB in use\n", poolMiB());

    scaleOnGpu(host, result, 2.0f);

    int correct = 0;
    for (float x : result) {
        correct += (x == 2.0f);
    }
    std::printf("  copies sized by the type: %d of %d correct\n", correct, N);
    passed = passed && correct == N;

    for (int call = 0; call < 8; ++call) {
        leakOnGpu(host);
    }
    std::printf("  8 calls with new and no delete: %.1f MiB lost\n", poolMiB() - before);

    std::printf("1 template, 3 types, %d values\n", BIG);
    {
        DeviceBuffer<float> in(std::vector<float>(BIG, 1.0f));
        DeviceBuffer<float> out(BIG);
        auto launch = [&] {
            scaleRaw<<<blocksFor(BIG), BLOCK>>>(in.data(), out.data(), BIG, 3.0f);
        };
        double ms = timeKernel(launch, 20);
        double gbs = 2.0 * BIG * sizeof(float) / ms / 1e6;
        std::printf("  %-22s %7.2f ms  %6.1f GB/s\n", "raw pointers, float", ms, gbs);
    }
    passed = timeType<float>("transform<float>", 20) && passed;
    passed = timeType<double>("transform<double>", 20) && passed;
    passed = timeType<int>("transform<int>", 20) && passed;

    std::printf("1 function on both sides, %d values in [-1, 1)\n", N);
    std::vector<float> xs(N);
    for (int i = 0; i < N; ++i) {
        xs[i] = -1.0f + 2.0f * i / N;
    }

    DeviceBuffer<float> x(xs);
    DeviceBuffer<float> y(N);

    auto plain = [] __host__ __device__ (float v) { return expPoly(v); };
    auto fused = [] __host__ __device__ (float v) { return expPolyFma(v); };

    transform<<<blocksFor(N), BLOCK>>>(x.view(), y.span(), plain);
    check(cudaGetLastError());
    std::vector<float> gpuPlain = y.toHost();

    transform<<<blocksFor(N), BLOCK>>>(x.view(), y.span(), fused);
    check(cudaGetLastError());
    std::vector<float> gpuFused = y.toHost();

    int plainDiffer = 0;
    int fusedDiffer = 0;
    for (int i = 0; i < N; ++i) {
        float cpuPlain = plain(xs[i]);
        float cpuFused = fused(xs[i]);
        plainDiffer += std::memcmp(&cpuPlain, &gpuPlain[i], sizeof(float)) != 0;
        fusedDiffer += std::memcmp(&cpuFused, &gpuFused[i], sizeof(float)) != 0;
    }
    std::printf("  x * y + z: %d of %d differ from the CPU\n", plainDiffer, N);
    std::printf("  fmaf:      %d of %d differ from the CPU\n", fusedDiffer, N);
    passed = passed && fusedDiffer == 0;

    DeviceBuffer<int> count(1);
    check(cudaMemset(count.data(), 0, sizeof(int)));
    countAbove<<<blocksFor(N), BLOCK>>>(y.view(), 1.5f, count.data());
    check(cudaGetLastError());

    int gpuCount = count.toHost()[0];
    int cpuCount = 0;
    for (float v : gpuFused) {
        cpuCount += (v > 1.5f);
    }
    std::printf("span and atomic_ref: %d values above 1.5, CPU %d\n", gpuCount, cpuCount);
    passed = passed && gpuCount == cpuCount;

    std::printf(passed ? "PASSED\n" : "FAILED\n");
    return passed ? 0 : 1;
}
