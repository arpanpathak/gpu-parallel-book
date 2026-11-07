// Chapter 16: latencies in clock cycles, measured inside a kernel with clock64().
// Build: nvcc -arch=native latency.cu -o latency

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

const int STEPS = 4096;

// 1 thread, 1 chain: each multiply-add needs the result of the one before.
__global__ void dependentFma(float* x, long long* cycles)
{
    float v = x[0];

    long long t0 = clock64();
    for (int k = 0; k < STEPS; ++k) {
        v = fmaf(v, 0.999f, 1.0f);
    }
    long long t1 = clock64();

    x[0] = v;
    cycles[0] = t1 - t0;
}

// 1 thread, 4 chains: the 4 multiply-adds of a step do not wait for each other.
__global__ void independentFma(float* x, long long* cycles)
{
    float v0 = x[0], v1 = x[1], v2 = x[2], v3 = x[3];

    long long t0 = clock64();
    for (int k = 0; k < STEPS / 4; ++k) {
        v0 = fmaf(v0, 0.999f, 1.0f);
        v1 = fmaf(v1, 0.999f, 1.0f);
        v2 = fmaf(v2, 0.999f, 1.0f);
        v3 = fmaf(v3, 0.999f, 1.0f);
    }
    long long t1 = clock64();

    x[0] = v0 + v1 + v2 + v3;
    cycles[0] = t1 - t0;
}

// 1 thread follows a chain: each load needs the index the load before returned.
// The first `warm` hops bring the chain into the caches; the next `hops` are timed.
__global__ void chase(const int* next, int warm, int hops, int* out, long long* cycles)
{
    int j = 0;
    for (int k = 0; k < warm; ++k) {
        j = next[j];
    }

    long long t0 = clock64();
    for (int k = 0; k < hops; ++k) {
        j = next[j];
    }
    long long t1 = clock64();

    out[0] = j;
    cycles[0] = t1 - t0;
}

// Cycles per load along a chain of `count` ints, each pointing `stride` ints ahead.
// Counts in `wrong` a chain that does not end where the host expects.
double chaseCycles(int count, int stride, int warm, int hops, int& wrong)
{
    std::vector<int> next(count);
    for (int i = 0; i < count; ++i) {
        next[i] = (i + stride) % count;
    }

    int *dnext, *dout;
    long long* dcycles;
    check(cudaMalloc(&dnext, count * sizeof(int)));
    check(cudaMalloc(&dout, sizeof(int)));
    check(cudaMalloc(&dcycles, sizeof(long long)));
    check(cudaMemcpy(dnext, next.data(), count * sizeof(int), cudaMemcpyHostToDevice));

    chase<<<1, 1>>>(dnext, warm, hops, dout, dcycles);
    check(cudaGetLastError());

    long long cycles;
    int end;
    check(cudaMemcpy(&cycles, dcycles, sizeof(long long), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&end, dout, sizeof(int), cudaMemcpyDeviceToHost));
    wrong += end != (long long)(warm + hops) * stride % count;

    check(cudaFree(dnext));
    check(cudaFree(dout));
    check(cudaFree(dcycles));
    return double(cycles) / hops;
}

int main()
{
    float* x;
    long long* dcycles;
    check(cudaMalloc(&x, 4 * sizeof(float)));
    check(cudaMalloc(&dcycles, sizeof(long long)));
    check(cudaMemset(x, 0, 4 * sizeof(float)));

    long long dep, ind;
    size_t size = sizeof(long long);

    dependentFma<<<1, 1>>>(x, dcycles);
    check(cudaMemcpy(&dep, dcycles, size, cudaMemcpyDeviceToHost));

    independentFma<<<1, 1>>>(x, dcycles);
    check(cudaMemcpy(&ind, dcycles, size, cudaMemcpyDeviceToHost));

    std::printf("%d multiply-adds in 1 thread, cycles per multiply-add\n", STEPS);
    std::printf("  1 dependent chain      %6.2f\n", double(dep) / STEPS);
    std::printf("  4 independent chains   %6.2f\n", double(ind) / STEPS);

    // 128-byte steps (32 ints), so that every load touches a new cache line.
    // The DRAM chain jumps 1 MiB + 128 bytes per load and is not warmed first.
    int wrong = 0;
    double l1 = chaseCycles(1024, 32, 1024, 4096, wrong);            // 4 KiB, L1
    double l2 = chaseCycles(256 << 10, 32, 8192, 8192, wrong);       // 1 MiB, L2
    double dram = chaseCycles(64 << 20, (1 << 18) + 32, 0, 4096, wrong);  // 256 MiB

    std::printf("1 thread follows a chain of loads, cycles per load\n");
    std::printf("  4 KiB chain, from L1      %6.0f\n", l1);
    std::printf("  1 MiB chain, from L2      %6.0f\n", l2);
    std::printf("  256 MiB chain, from DRAM  %6.0f\n", dram);

    const char* verdict = wrong ? "FAILED" : "PASSED";
    std::printf("%s: %d of 3 chains end in the wrong place\n", verdict, wrong);

    check(cudaFree(x));
    check(cudaFree(dcycles));
    return wrong ? 1 : 0;
}
