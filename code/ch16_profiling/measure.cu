// Chapter 16: the GPU's limits, the warm-up effect, and a kernel report by hand.
// Build: nvcc -arch=native measure.cu -o measure

#include <algorithm>
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

int attribute(cudaDeviceAttr attr)
{
    int value;
    check(cudaDeviceGetAttribute(&value, attr, 0));
    return value;
}

__global__ void fill(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = i % 1000;
    }
}

// y = 2 x: reads 4 bytes and writes 4 bytes per element.
__global__ void scale(const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 2.0f * x[i];
    }
}

// The time of 1 launch of `scale`, in milliseconds, measured with 2 events.
float timeScale(const float* x, float* y, int n)
{
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start));
    scale<<<(n + 255) / 256, 256>>>(x, y, n);
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));

    float ms;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return ms;
}

// Print the limits the GPU reports, and the peaks they imply.
void limits()
{
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0));

    int smClockKHz = attribute(cudaDevAttrClockRate);
    int memClockKHz = attribute(cudaDevAttrMemoryClockRate);
    int busBits = attribute(cudaDevAttrGlobalMemoryBusWidth);
    size_t sharedKiB = p.sharedMemPerMultiprocessor / 1024;

    std::printf("%s, compute capability %d.%d\n", p.name, p.major, p.minor);
    std::printf("  SMs                       %8d\n", p.multiProcessorCount);
    std::printf("  threads per SM, at most   %8d\n", p.maxThreadsPerMultiProcessor);
    std::printf("  registers per SM          %8d\n", p.regsPerMultiprocessor);
    std::printf("  shared memory per SM      %8zu KiB\n", sharedKiB);
    std::printf("  L2 cache                  %8d KiB\n", p.l2CacheSize / 1024);
    std::printf("  SM clock                  %8d MHz\n", smClockKHz / 1000);
    std::printf("  memory clock              %8d MHz\n", memClockKHz / 1000);
    std::printf("  memory bus                %8d bits\n", busBits);

    // 128 FP32 units per SM from compute capability 8.6; 2 FLOP per multiply-add.
    double peakGflops = p.multiProcessorCount * 128 * 2.0 * smClockKHz / 1e6;

    // 2 transfers per memory clock (double data rate) of busBits / 8 bytes each.
    double peakGBs = 2.0 * memClockKHz * 1e3 * (busBits / 8) / 1e9;

    std::printf("  FP32 peak from the clock  %8.0f GFLOP/s\n", peakGflops);
    std::printf("  bandwidth from the clock  %8.1f GB/s\n", peakGBs);
}

int main()
{
    limits();

    const int n = 64 << 20;                   // 67,108,864 floats, 256 MiB per array
    const double bytes = 2.0 * n * sizeof(float);
    float *x, *y;
    check(cudaMalloc(&x, n * sizeof(float)));
    check(cudaMalloc(&y, n * sizeof(float)));
    fill<<<(n + 255) / 256, 256>>>(x, n);
    check(cudaDeviceSynchronize());

    // The first 40 runs of y = 2 x.
    std::vector<float> ms(40);
    for (float& t : ms) {
        t = timeScale(x, y, n);
    }

    std::printf("y = 2 x over 256 MiB, the first 40 runs in ms\n");
    for (int r = 0; r < 40; ++r) {
        const char* end = r % 10 == 9 ? "\n" : "";
        std::printf(" %5.2f%s", ms[r], end);
    }

    double mean = 0.0;
    for (float t : ms) {
        mean += t / ms.size();
    }

    std::vector<float> last(ms.begin() + 20, ms.end());
    std::sort(last.begin(), last.end());
    float median = (last[9] + last[10]) / 2;

    std::printf("mean of all 40 runs     %7.2f ms\n", mean);
    std::printf("median of runs 21 to 40 %7.2f ms\n", median);

    // The report by hand: bytes moved / time.
    double gbs = bytes / (median * 1e-3) / 1e9;
    std::printf("bytes moved             %7.0f MiB\n", bytes / (1 << 20));
    std::printf("achieved bandwidth      %7.1f GB/s\n", gbs);

    int blocks;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scale, 256, 0));
    std::printf("blocks of 256 per SM    %7d = %d threads\n", blocks, blocks * 256);

    std::vector<float> result(n);
    check(cudaMemcpy(result.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int i = 0; i < n; ++i) {
        wrong += result[i] != 2.0f * (i % 1000);
    }
    const char* verdict = wrong ? "FAILED" : "PASSED";
    std::printf("%s: %d of %d elements wrong\n", verdict, wrong, n);

    check(cudaFree(x));
    check(cudaFree(y));
    return wrong ? 1 : 0;
}
