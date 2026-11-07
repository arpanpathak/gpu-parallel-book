// Chapter 11: sorting 16 Mi keys with Thrust and CUB, against std::sort on the CPU.
// Build: nvcc -arch=native -O2 -std=c++17 sort.cu -o sort

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

using Keys = std::vector<unsigned>;

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

unsigned nextRandom(unsigned& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

// The same order as thrust::less, but Thrust cannot see inside it, so it cannot use a
// radix sort and falls back to a merge sort, which only compares keys.
struct Ascending {
    __host__ __device__ bool operator()(unsigned a, unsigned b) const { return a < b; }
};

// Scratch memory for Thrust, allocated once. During a call Thrust asks for temporary
// buffers and gives them all back before it returns. The arena hands out pieces of 1
// large buffer, and starts again from its beginning when every piece is back.
struct Arena {
    using value_type = char;

    char* base = nullptr;
    size_t size = 0;
    size_t used = 0;
    int live = 0;

    char* allocate(std::ptrdiff_t bytes)
    {
        size_t start = (used + 255) / 256 * 256;          // 256-byte aligned
        if (start + bytes > size) {
            std::fprintf(stderr, "arena too small for %td bytes\n", bytes);
            std::exit(1);
        }

        used = start + bytes;
        ++live;
        return base + start;
    }

    void deallocate(char*, size_t)
    {
        --live;
        if (live == 0) {
            used = 0;
        }
    }
};

// Average time of 5 sorts, in milliseconds. Each sort starts from a fresh copy of
// `original`; only the sort itself is timed.
template <typename F>
double timeSort(F sort, unsigned* keys, const unsigned* original, int n)
{
    double total = 0.0;
    for (int r = 0; r < 6; ++r) {
        check(cudaMemcpy(keys, original, n * 4, cudaMemcpyDeviceToDevice));
        check(cudaDeviceSynchronize());

        double t0 = nowMs();
        sort();
        check(cudaDeviceSynchronize());
        if (r > 0) {                          // run 0 warms up
            total += nowMs() - t0;
        }
    }
    return total / 5;
}

bool report(const char* name, double ms, int n, const unsigned* keys, const Keys& expected)
{
    Keys got(n);
    check(cudaMemcpy(got.data(), keys, n * 4, cudaMemcpyDeviceToHost));
    bool ok = got == expected;

    double rate = n / ms / 1e3;               // millions of keys per second
    std::printf("  %-34s %7.2f ms  %7.1f Mkeys/s", name, ms, rate);
    std::printf("  correct: %s\n", ok ? "yes" : "no");
    return ok;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    const int n = 16 << 20;                   // 16,777,216 keys of 32 bits, 64 MiB
    Keys random(n), small(n);
    unsigned state = 5;
    for (int i = 0; i < n; ++i) {
        random[i] = nextRandom(state);
        small[i] = random[i] >> 24;           // keys 0 to 255: only 8 bits differ
    }

    Keys ascending = random;
    double t0 = nowMs();
    std::sort(ascending.begin(), ascending.end());
    double cpuMs = nowMs() - t0;
    Keys descending(ascending.rbegin(), ascending.rend());
    Keys smallSorted = small;
    std::sort(smallSorted.begin(), smallSorted.end());

    unsigned *original, *keys, *sorted;
    check(cudaMalloc(&original, n * 4));
    check(cudaMalloc(&keys, n * 4));
    check(cudaMalloc(&sorted, n * 4));
    check(cudaMemcpy(original, random.data(), n * 4, cudaMemcpyHostToDevice));

    std::printf("sort of %d random 32-bit keys, average of 5 runs\n", n);
    double cpuRate = n / cpuMs / 1e3;
    std::printf("  %-34s %7.2f ms", "std::sort, 1 CPU core", cpuMs);
    std::printf("  %7.1f Mkeys/s\n", cpuRate);

    auto plain = [&] { thrust::sort(thrust::device, keys, keys + n); };
    double ms = timeSort(plain, keys, original, n);
    bool ok = report("thrust::sort", ms, n, keys, ascending);

    Arena arena;
    arena.size = 2 * size_t(n) * 4;
    check(cudaMalloc(&arena.base, arena.size));
    auto policy = thrust::cuda::par(arena);

    auto radix = [&] { thrust::sort(policy, keys, keys + n); };
    ms = timeSort(radix, keys, original, n);
    ok &= report("thrust::sort, scratch reused", ms, n, keys, ascending);

    cuda::std::greater<unsigned> greater;
    auto down = [&] { thrust::sort(policy, keys, keys + n, greater); };
    ms = timeSort(down, keys, original, n);
    ok &= report("  ... with cuda::std::greater", ms, n, keys, descending);

    auto merge = [&] { thrust::sort(policy, keys, keys + n, Ascending()); };
    ms = timeSort(merge, keys, original, n);
    ok &= report("  ... with own comparison", ms, n, keys, ascending);

    size_t tempBytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, tempBytes, keys, sorted, n);
    void* temp;
    check(cudaMalloc(&temp, tempBytes));

    auto all32 = [&] { cub::DeviceRadixSort::SortKeys(temp, tempBytes, keys, sorted, n); };
    ms = timeSort(all32, keys, original, n);
    ok &= report("cub::DeviceRadixSort", ms, n, sorted, ascending);

    check(cudaMemcpy(original, small.data(), n * 4, cudaMemcpyHostToDevice));
    std::printf("sort of %d keys from 0 to 255\n", n);

    ms = timeSort(all32, keys, original, n);
    ok &= report("cub::DeviceRadixSort, bits 0 to 31", ms, n, sorted, smallSorted);

    auto low8 = [&] {
        cub::DeviceRadixSort::SortKeys(temp, tempBytes, keys, sorted, n, 0, 8);
    };
    ms = timeSort(low8, keys, original, n);
    ok &= report("cub::DeviceRadixSort, bits 0 to 7", ms, n, sorted, smallSorted);

    check(cudaFree(temp));
    check(cudaFree(arena.base));
    check(cudaFree(original));
    check(cudaFree(keys));
    check(cudaFree(sorted));

    std::printf("%s\n", ok ? "PASSED" : "FAILED");
}
