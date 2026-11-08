// Chapter 15: the dot product of 2 vectors as a reduction, 4 ways: map then reduce,
// a fused kernel with a tree in shared memory, a fused kernel with warp shuffles, and
// a fused kernel that combines its block results with atomic additions.
// Build: nvcc -O3 -arch=native dot.cu -o dot

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
#include <cuda_runtime.h>

const int BLOCK = 256;
const int GRID = 1024;
const int N = 1 << 24;
const int RUNS = 20;

// The worked example of the chapter: 16 pairs whose products add up to 63.
const float A16[16] = {1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 2, 3, 1, 3, 2, 1};
const float B16[16] = {2, 3, 1, 4, 1, 1, 4, 1, 1, 2, 2, 2, 1, 3, 1, 4};

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// The sum of value over the 32 lanes of a warp, left in lane 0.
__device__ float warpSum(float value)
{
    for (int distance = 16; distance > 0; distance /= 2) {
        value += __shfl_down_sync(0xffffffff, value, distance);
    }
    return value;
}

// The sum of value over the 256 threads of a block, returned to thread 0.
__device__ float blockSum(float value)
{
    __shared__ float warpSums[8];
    int lane = threadIdx.x % 32;
    int warpIndex = threadIdx.x / 32;

    value = warpSum(value);
    if (lane == 0) {
        warpSums[warpIndex] = value;
    }
    __syncthreads();

    float warpTotal = lane < 8 ? warpSums[lane] : 0.0f;
    return warpIndex == 0 ? warpSum(warpTotal) : 0.0f;
}

// The sum of a[index] * b[index] over this thread's elements, a grid apart.
__device__ float products(const float* a, const float* b, int n)
{
    int gridSize = gridDim.x * blockDim.x;
    float sum = 0.0f;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += gridSize) {
        sum += a[index] * b[index];
    }
    return sum;
}

// Map: elementProducts[index] = a[index] * b[index], one element per thread.
__global__ void multiply(const float* a, const float* b, float* elementProducts, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        elementProducts[index] = a[index] * b[index];
    }
}

// Reduce: 1 partial sum of input per block.
__global__ void sum(const float* input, float* partial, int n)
{
    int gridSize = gridDim.x * blockDim.x;
    float mySum = 0.0f;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += gridSize) {
        mySum += input[index];
    }

    float total = blockSum(mySum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = total;
    }
}

// Fused: products and a tree in shared memory, 1 partial sum per block.
__global__ void dotTree(const float* a, const float* b, float* partial, int n)
{
    __shared__ float cells[BLOCK];
    int threadIndex = threadIdx.x;

    cells[threadIndex] = products(a, b, n);
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (threadIndex < stride) {
            cells[threadIndex] += cells[threadIndex + stride];
        }
        __syncthreads();
    }

    if (threadIndex == 0) {
        partial[blockIdx.x] = cells[0];
    }
}

// Fused: products and warp shuffles, 1 partial sum per block.
__global__ void dotShuffle(const float* a, const float* b, float* partial, int n)
{
    float total = blockSum(products(a, b, n));

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = total;
    }
}

// Fused: products, warp shuffles, and 1 atomic addition per block.
__global__ void dotAtomic(const float* a, const float* b, float* result, int n)
{
    float total = blockSum(products(a, b, n));

    if (threadIdx.x == 0) {
        atomicAdd(result, total);
    }
}

enum Method { MAP_REDUCE, TREE, SHUFFLE, ATOMIC };

struct MethodInfo {
    const char* name;
    Method method;
    int bytesPerPair;
};

const MethodInfo METHODS[4] = {
    {"map, then reduce", MAP_REDUCE, 16},
    {"fused, shared tree", TREE, 8},
    {"fused, shuffles", SHUFFLE, 8},
    {"fused, atomic combine", ATOMIC, 8},
};

// The device memory of 1 problem.
struct Buffers {
    int n;
    float *a, *b, *products, *partial, *total;
};

Buffers upload(const std::vector<float>& a, const std::vector<float>& b)
{
    Buffers buf;
    buf.n = a.size();
    size_t bytes = buf.n * sizeof(float);

    check(cudaMalloc(&buf.a, bytes));
    check(cudaMalloc(&buf.b, bytes));
    check(cudaMalloc(&buf.products, bytes));
    check(cudaMalloc(&buf.partial, GRID * sizeof(float)));
    check(cudaMalloc(&buf.total, sizeof(float)));

    check(cudaMemcpy(buf.a, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(buf.b, b.data(), bytes, cudaMemcpyHostToDevice));
    return buf;
}

void release(Buffers& buf)
{
    check(cudaFree(buf.a));
    check(cudaFree(buf.b));
    check(cudaFree(buf.products));
    check(cudaFree(buf.partial));
    check(cudaFree(buf.total));
}

// Launches 1 dot product; the result ends in buf.total.
void run(Method method, const Buffers& buf)
{
    int n = buf.n;
    int mapBlocks = (n + BLOCK - 1) / BLOCK;

    switch (method) {
    case MAP_REDUCE:
        multiply<<<mapBlocks, BLOCK>>>(buf.a, buf.b, buf.products, n);
        sum<<<GRID, BLOCK>>>(buf.products, buf.partial, n);
        sum<<<1, BLOCK>>>(buf.partial, buf.total, GRID);
        break;
    case TREE:
        dotTree<<<GRID, BLOCK>>>(buf.a, buf.b, buf.partial, n);
        sum<<<1, BLOCK>>>(buf.partial, buf.total, GRID);
        break;
    case SHUFFLE:
        dotShuffle<<<GRID, BLOCK>>>(buf.a, buf.b, buf.partial, n);
        sum<<<1, BLOCK>>>(buf.partial, buf.total, GRID);
        break;
    case ATOMIC:
        check(cudaMemsetAsync(buf.total, 0, sizeof(float)));
        dotAtomic<<<GRID, BLOCK>>>(buf.a, buf.b, buf.total, n);
        break;
    }
    check(cudaGetLastError());
}

// Runs method once and returns its result.
float result(Method method, const Buffers& buf)
{
    float value;
    run(method, buf);
    check(cudaMemcpy(&value, buf.total, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

// Milliseconds per run of method: 10 warm-up runs, then RUNS runs timed with events.
float timeMs(Method method, const Buffers& buf)
{
    for (int warmup = 0; warmup < 10; ++warmup) {
        run(method, buf);
    }

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start));
    for (int timed = 0; timed < RUNS; ++timed) {
        run(method, buf);
    }
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));

    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return ms / RUNS;
}

// Floats in [0, 1) from a small linear congruential generator.
std::vector<float> random(int n, unsigned seed)
{
    std::vector<float> v(n);
    unsigned state = seed;

    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i] = (state >> 8) / 16777216.0f;
    }
    return v;
}

// Every method on 1 input whose dot product is an integer that a float holds exactly.
// Returns the number of methods whose result differs from expect.
int exactTest(const char* label, const std::vector<float>& a, const std::vector<float>& b)
{
    float expect = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        expect += a[i] * b[i];
    }

    Buffers buf = upload(a, b);
    int bad = 0;

    std::printf("  %-26s", label);
    for (const MethodInfo& m : METHODS) {
        float got = result(m.method, buf);
        std::printf(" %9.0f", got);
        bad += got != expect;
    }
    std::printf("   expected %.0f\n", expect);

    release(buf);
    return bad;
}

unsigned bits(float f)
{
    unsigned u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float fromBits(unsigned u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    std::printf("exact tests                   map+red      tree   shuffle    atomic\n");
    std::vector<float> a16(A16, A16 + 16), b16(B16, B16 + 16);
    int failed = exactTest("worked example, 16 pairs", a16, b16);

    std::vector<float> onesA(N), onesB(N);
    for (int i = 0; i < N; ++i) {
        onesA[i] = i % 3 == 0;
        onesB[i] = i % 2 == 0;
    }
    failed += exactTest("0s and 1s, 16777216 pairs", onesA, onesB);

    std::vector<float> a = random(N, 1);
    std::vector<float> b = random(N, 2);
    double exact = 0;
    for (int i = 0; i < N; ++i) {
        exact += double(a[i]) * b[i];
    }
    Buffers buf = upload(a, b);
    double worst = 0;

    std::printf("%d pairs of floats in [0, 1), %d runs\n", N, RUNS);
    std::printf("  method                    time   bytes/pair      rate      result\n");
    for (const MethodInfo& m : METHODS) {
        float ms = timeMs(m.method, buf);
        float got = result(m.method, buf);
        worst = std::fmax(worst, std::fabs((got - exact) / exact));

        double gbs = double(N) * m.bytesPerPair / ms / 1e6;
        std::printf("  %-22s %6.3f ms %8d B", m.name, ms, m.bytesPerPair);
        std::printf("  %6.1f GB/s  %.1f\n", gbs, got);
    }
    std::printf("  %-54s %.1f\n", "exact, in double precision", exact);

    std::printf("distinct results in 100 runs\n");
    const MethodInfo combine[2] = {{"second launch", SHUFFLE, 8}, {"atomic", ATOMIC, 8}};
    for (const MethodInfo& m : combine) {
        std::set<unsigned> seen;
        for (int trial = 0; trial < 100; ++trial) {
            seen.insert(bits(result(m.method, buf)));
        }

        float low = fromBits(*seen.begin());
        float high = fromBits(*seen.rbegin());
        std::printf("  %-14s %3zu  from %.1f to %.1f\n", m.name, seen.size(), low, high);
        worst = std::fmax(worst, std::fabs((low - exact) / exact));
        worst = std::fmax(worst, std::fabs((high - exact) / exact));
    }
    release(buf);

    std::printf("largest relative error %.2e\n", worst);
    if (failed == 0 && worst < 1e-5) {
        std::printf("PASSED\n");
    } else {
        std::printf("FAILED\n");
    }
    return 0;
}
