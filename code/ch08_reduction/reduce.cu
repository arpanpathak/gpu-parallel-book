// Chapter 8: a sum, a scan and a histogram at memory speed, measured.
// Build: nvcc -arch=native reduce.cu -o reduce

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const int BLOCK = 256;                        // threads per block, a power of 2
const int GRID = 1024;                        // blocks of the grid-stride kernels

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

// 1. A tree in shared memory, 1 element per thread.
__global__ void sumTree(const float* in, float* partial, int n)
{
    __shared__ float s[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    s[threadIdx.x] = i < n ? in[i] : 0.0f;

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
    }
}

// 2. Each thread first sums many elements in a register, then 1 tree per block.
__global__ void sumCoarse(const float* in, float* partial, int n)
{
    __shared__ float s[BLOCK];
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
        sum += in[i];
    }
    s[threadIdx.x] = sum;

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
    }
}

// 3. Shuffles within each warp, and 1 barrier per block.
__device__ float warpSum(float v)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;                                 // the warp's total, in lane 0
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

__global__ void sumShuffle(const float* in, float* partial, int n)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
        sum += in[i];
    }

    sum = blockSum(sum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}

// 4. The same, loading 4 floats per instruction; n4 is the number of float4s.
__global__ void sumShuffle4(const float4* in, float* partial, int n4)
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

// 5. An inclusive scan, 4 elements per thread, 1,024 per block.
__device__ unsigned warpScan(unsigned v)
{
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned left = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) {
            v += left;
        }
    }
    return v;                                 // the sum of lanes 0 to `lane`
}

__global__ void scanBlocks(const uint4* in, uint4* out, unsigned* sums, int n4)
{
    __shared__ unsigned warps[BLOCK / 32];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    uint4 a = i < n4 ? in[i] : make_uint4(0, 0, 0, 0);
    a.y += a.x;                               // the thread's own 4 values, scanned
    a.z += a.y;
    a.w += a.z;

    unsigned v = warpScan(a.w);
    if (lane == 31) {
        warps[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned w = warpScan(lane < BLOCK / 32 ? warps[lane] : 0);
        if (lane < BLOCK / 32) {
            warps[lane] = w;
        }
    }
    __syncthreads();

    unsigned before = v - a.w + (warp > 0 ? warps[warp - 1] : 0);
    a = make_uint4(before + a.x, before + a.y, before + a.z, before + a.w);
    if (i < n4) {
        out[i] = a;
    }
    if (threadIdx.x == BLOCK - 1) {
        sums[blockIdx.x] = a.w;
    }
}

__global__ void addOffsets(uint4* out, const unsigned* sums, int n4)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (blockIdx.x > 0 && i < n4) {
        unsigned s = sums[blockIdx.x - 1];
        uint4 a = out[i];
        out[i] = make_uint4(a.x + s, a.y + s, a.z + s, a.w + s);
    }
}

// Scans each block, then the block totals (by the same function), then adds them.
// n is a multiple of 4. The totals are padded to a multiple of 4 as well: the padding
// comes after the last real total, so it changes none of the sums that are used.
void scan(const unsigned* in, unsigned* out, int n, unsigned* scratch)
{
    int blocks = (n / 4 + BLOCK - 1) / BLOCK;
    scanBlocks<<<blocks, BLOCK>>>((const uint4*)in, (uint4*)out, scratch, n / 4);

    if (blocks > 1) {
        int padded = (blocks + 3) / 4 * 4;
        scan(scratch, scratch, padded, scratch + padded);
        addOffsets<<<blocks, BLOCK>>>((uint4*)out, scratch, n / 4);
    }
}

// 6. Histograms of bytes in shared bins (Chapter 5): 1 byte, or 16 bytes, per load.
__global__ void histBytes(const unsigned char* data, unsigned* bins, int n)
{
    __shared__ unsigned local[256];
    local[threadIdx.x] = 0;
    __syncthreads();

    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
        atomicAdd(&local[data[i]], 1u);
    }
    __syncthreads();

    atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
}

__device__ void countWord(unsigned* local, unsigned w)
{
    for (int k = 0; k < 4; ++k) {
        atomicAdd(&local[(w >> 8 * k) & 255], 1u);
    }
}

__global__ void histWords(const uint4* data, unsigned* bins, int n16)
{
    __shared__ unsigned local[256];
    local[threadIdx.x] = 0;
    __syncthreads();

    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n16; i += gridDim.x * BLOCK) {
        uint4 q = data[i];
        countWord(local, q.x);
        countWord(local, q.y);
        countWord(local, q.z);
        countWord(local, q.w);
    }
    __syncthreads();

    atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
}

unsigned nextRandom(unsigned& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

void sums(const float* in, float* partial, float* total, int n)
{
    double bytes = n * sizeof(float);
    std::printf("sum of %d floats (%d MiB), average of 20 runs\n", n, int(bytes) >> 20);

    auto finish = [&](int blocks) { sumShuffle<<<1, BLOCK>>>(partial, total, blocks); };
    auto report = [&](const char* name, double ms) {
        float t;
        check(cudaMemcpy(&t, total, sizeof(float), cudaMemcpyDeviceToHost));
        double gb = bytes / ms / 1e6;
        std::printf("  %-22s %6.2f ms  %6.1f GB/s  sum %.1f\n", name, ms, gb, t);
    };

    report("tree, 1 per thread", timeKernel([&] {
        sumTree<<<n / BLOCK, BLOCK>>>(in, partial, n);
        finish(n / BLOCK);
    }));
    report("coarsened", timeKernel([&] {
        sumCoarse<<<GRID, BLOCK>>>(in, partial, n);
        finish(GRID);
    }));
    report("shuffles", timeKernel([&] {
        sumShuffle<<<GRID, BLOCK>>>(in, partial, n);
        finish(GRID);
    }));
    report("shuffles, float4", timeKernel([&] {
        sumShuffle4<<<GRID, BLOCK>>>((const float4*)in, partial, n / 4);
        finish(GRID);
    }));
}

void scans(int n)
{
    std::vector<unsigned> values(n), expected(n), result(n);
    unsigned state = 2;
    unsigned running = 0;
    for (int i = 0; i < n; ++i) {
        values[i] = nextRandom(state) >> 30;  // 0, 1, 2 or 3
        running += values[i];
        expected[i] = running;
    }

    unsigned *in, *out, *scratch;
    check(cudaMalloc(&in, n * sizeof(unsigned)));
    check(cudaMalloc(&out, n * sizeof(unsigned)));
    check(cudaMalloc(&scratch, (n / 512 + 64) * sizeof(unsigned)));
    check(cudaMemcpy(in, values.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));

    double ms = timeKernel([&] { scan(in, out, n, scratch); });
    check(cudaMemcpy(result.data(), out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));

    double gb = 2.0 * n * sizeof(unsigned) / ms / 1e6;
    std::printf("inclusive scan of %d values\n", n);
    std::printf("  %6.2f ms  %6.1f GB/s  last %u  correct: %s\n", ms, gb, result[n - 1],
                result == expected ? "yes" : "no");

    check(cudaFree(in));
    check(cudaFree(out));
    check(cudaFree(scratch));
}

void histograms(int n)
{
    std::vector<unsigned char> random(n), sevens(n, 7);
    unsigned state = 3;
    for (int i = 0; i < n; ++i) {
        random[i] = nextRandom(state) >> 24;
    }

    unsigned char* data;
    unsigned* bins;
    check(cudaMalloc(&data, n));
    check(cudaMalloc(&bins, 256 * sizeof(unsigned)));

    std::printf("histogram of %d MiB, average of 20 runs\n", n >> 20);
    for (auto* bytes : {&random, &sevens}) {
        check(cudaMemcpy(data, bytes->data(), n, cudaMemcpyHostToDevice));
        const char* name = bytes == &random ? "random bytes" : "all 7       ";

        double ms1 = timeKernel([&] {
            check(cudaMemsetAsync(bins, 0, 256 * sizeof(unsigned)));
            histBytes<<<GRID, BLOCK>>>(data, bins, n);
        });
        double ms16 = timeKernel([&] {
            check(cudaMemsetAsync(bins, 0, 256 * sizeof(unsigned)));
            histWords<<<GRID, BLOCK>>>((const uint4*)data, bins, n / 16);
        });

        std::vector<unsigned> counts(256);
        check(cudaMemcpy(counts.data(), bins, 1024, cudaMemcpyDeviceToHost));
        long long total = 0;
        for (unsigned c : counts) {
            total += c;
        }
        std::printf("  %s  1 byte per load   %6.2f ms  %6.1f GB/s\n", name, ms1,
                    n / ms1 / 1e6);
        std::printf("  %s  16 bytes per load %6.2f ms  %6.1f GB/s  total %lld\n", name,
                    ms16, n / ms16 / 1e6, total);
    }

    check(cudaFree(data));
    check(cudaFree(bins));
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs, L2 cache %d MiB\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20);

    const int n = 64 << 20;                   // 67,108,864 values, 256 MiB of floats
    std::vector<float> values(n);
    unsigned state = 1;
    for (int i = 0; i < n; ++i) {
        values[i] = (nextRandom(state) >> 8) / 16777216.0f;  // in [0, 1)
    }

    float *in, *partial, *total;
    check(cudaMalloc(&in, n * sizeof(float)));
    check(cudaMalloc(&partial, n / BLOCK * sizeof(float)));
    check(cudaMalloc(&total, sizeof(float)));
    check(cudaMemcpy(in, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    for (int k = 0; k < 10; ++k) {            // warm up: clocks rise under sustained load
        timeKernel([&] { sumShuffle<<<GRID, BLOCK>>>(in, partial, n); });
    }
    sums(in, partial, total, n);

    double exact = 0.0;
    float serial = 0.0f;
    for (float v : values) {
        exact += v;
        serial += v;
    }
    std::printf("exact sum (double)         %.1f\n", exact);
    std::printf("CPU, float, 1 by 1         %.1f\n", serial);

    std::printf("GPU, shuffles, 5 runs     ");
    for (int r = 0; r < 5; ++r) {
        sumShuffle<<<GRID, BLOCK>>>(in, partial, n);
        sumShuffle<<<1, BLOCK>>>(partial, total, GRID);
        float t;
        check(cudaMemcpy(&t, total, sizeof(float), cudaMemcpyDeviceToHost));
        std::printf(" %.1f", t);
    }
    std::printf("\n");

    check(cudaFree(in));
    check(cudaFree(partial));
    check(cudaFree(total));

    scans(n);
    histograms(n);
}
