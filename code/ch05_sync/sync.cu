// Chapter 5: races, atomics, barriers and divergence, measured on this GPU.
// Build: nvcc -arch=native sync.cu -o sync

#include <chrono>
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

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 1. Every thread adds 1 to the same counter.
__global__ void countRacy(unsigned* counter)
{
    *counter = *counter + 1;
}

__global__ void countAtomic(unsigned* counter)
{
    atomicAdd(counter, 1u);
}

// 2. A histogram of bytes: 256 bins, 1 per byte value.
__global__ void histGlobal(const unsigned char* data, int n, unsigned* bins)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&bins[data[i]], 1u);
    }
}

__global__ void histShared(const unsigned char* data, int n, unsigned* bins)
{
    __shared__ unsigned local[256];
    local[threadIdx.x] = 0;
    __syncthreads();

    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&local[data[i]], 1u);
    }
    __syncthreads();

    atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
}

// 3. Each block sums its 256 values in shared memory, with or without barriers.
__global__ void blockSum(const float* in, float* out, bool barrier)
{
    __shared__ float s[256];
    s[threadIdx.x] = in[blockIdx.x * 256 + threadIdx.x];
    if (barrier) __syncthreads();

    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        if (barrier) __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = s[0];
    }
}

// 4. Half the threads take each branch: alternate lanes, or alternate warps.
__global__ void branches(float* x, int n, bool byWarp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bool odd = byWarp ? (threadIdx.x / 32) % 2 : threadIdx.x % 2;
    float v = x[i];

    if (odd) {
        for (int k = 0; k < 64; ++k) v = sinf(v) + 1.0f;
    } else {
        for (int k = 0; k < 64; ++k) v = sqrtf(v) * 0.5f + 0.25f;
    }

    x[i] = v;
}

// 5. A float sum by atomicAdd, in whatever order the threads arrive.
__global__ void sumAtomic(const float* in, int n, float* total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(total, in[i]);
    }
}

void race()
{
    unsigned* counter;
    check(cudaMalloc(&counter, sizeof(unsigned)));
    const int blocks = 4096, threads = 256;

    std::printf("1,048,576 threads each add 1 to 1 counter\n");
    for (int atomic = 0; atomic < 2; ++atomic) {
        unsigned result;
        check(cudaMemset(counter, 0, sizeof(unsigned)));

        if (atomic) {
            countAtomic<<<blocks, threads>>>(counter);
        } else {
            countRacy<<<blocks, threads>>>(counter);
        }

        check(cudaMemcpy(&result, counter, sizeof(unsigned), cudaMemcpyDeviceToHost));
        std::printf("  %-12s counter = %u\n", atomic ? "atomicAdd" : "plain add", result);
    }

    check(cudaFree(counter));
}

void histograms()
{
    const int n = 64 << 20;                      // 64 MiB of bytes
    std::vector<unsigned char> random(n), same(n, 7);
    for (int i = 0; i < n; ++i) {
        random[i] = std::rand() & 255;
    }

    unsigned char* data;
    unsigned* bins;
    check(cudaMalloc(&data, n));
    check(cudaMalloc(&bins, 256 * sizeof(unsigned)));

    std::printf("histogram of 64 MiB, average of 10 runs\n");
    for (const auto* input : {&random, &same}) {
        check(cudaMemcpy(data, input->data(), n, cudaMemcpyHostToDevice));

        for (int shared = 0; shared < 2; ++shared) {
            histGlobal<<<1024, 256>>>(data, n, bins);   // warm up
            check(cudaDeviceSynchronize());

            double t0 = nowMs();
            for (int r = 0; r < 10; ++r) {
                check(cudaMemset(bins, 0, 256 * sizeof(unsigned)));
                if (shared) {
                    histShared<<<1024, 256>>>(data, n, bins);
                } else {
                    histGlobal<<<1024, 256>>>(data, n, bins);
                }
            }
            check(cudaDeviceSynchronize());
            double ms = (nowMs() - t0) / 10;

            unsigned h[256];
            check(cudaMemcpy(h, bins, sizeof(h), cudaMemcpyDeviceToHost));
            unsigned long long total = 0;
            for (unsigned c : h) total += c;

            const char* name = input == &random ? "random bytes" : "all 7";
            const char* kind = shared ? "shared bins" : "global bins";
            std::printf("  %-12s %-13s %7.2f ms  total %llu\n", name, kind, ms, total);
        }
    }

    check(cudaFree(data));
    check(cudaFree(bins));
}

void barriers()
{
    const int blocks = 4096;
    std::vector<float> ones(blocks * 256, 1.0f), sums(blocks);

    float *in, *out;
    check(cudaMalloc(&in, ones.size() * sizeof(float)));
    check(cudaMalloc(&out, blocks * sizeof(float)));
    check(cudaMemcpy(in, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("4,096 blocks each sum 256 ones; the right answer is 256\n");
    for (int barrier = 1; barrier >= 0; --barrier) {
        blockSum<<<blocks, 256>>>(in, out, barrier);
        check(cudaMemcpy(sums.data(), out, blocks * sizeof(float), cudaMemcpyDeviceToHost));

        int wrong = 0;
        for (float s : sums) wrong += (s != 256.0f);
        std::printf("  %-18s wrong sums: %4d of %d   (block 0: %g)\n",
                    barrier ? "with barriers" : "without barriers", wrong, blocks, sums[0]);
    }

    check(cudaFree(in));
    check(cudaFree(out));
}

void divergence()
{
    const int n = 16 << 20;
    float* x;
    check(cudaMalloc(&x, n * sizeof(float)));
    check(cudaMemset(x, 0, n * sizeof(float)));

    std::printf("branch on every other thread, %d threads, average of 10 runs\n", n);
    for (int byWarp = 1; byWarp >= 0; --byWarp) {
        branches<<<n / 256, 256>>>(x, n, byWarp);   // warm up
        check(cudaDeviceSynchronize());

        double t0 = nowMs();
        for (int r = 0; r < 10; ++r) {
            branches<<<n / 256, 256>>>(x, n, byWarp);
        }
        check(cudaDeviceSynchronize());
        const char* name = byWarp ? "whole warps agree" : "lanes alternate";
        std::printf("  %-22s %6.2f ms\n", name, (nowMs() - t0) / 10);
    }

    check(cudaFree(x));
}

void floatSums()
{
    const int n = 1 << 20;
    std::vector<float> values(n);
    double exact = 0;
    for (int i = 0; i < n; ++i) {
        values[i] = 1.0f / (1 + i % 1000);
        exact += values[i];
    }

    float *in, *total;
    check(cudaMalloc(&in, n * sizeof(float)));
    check(cudaMalloc(&total, sizeof(float)));
    check(cudaMemcpy(in, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("sum of 1,048,576 floats by atomicAdd (exact: %.4f)\n", exact);
    for (int run = 1; run <= 4; ++run) {
        float result;
        check(cudaMemset(total, 0, sizeof(float)));
        sumAtomic<<<n / 256, 256>>>(in, n, total);
        check(cudaMemcpy(&result, total, sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("  run %d   %.4f\n", run, result);
    }

    check(cudaFree(in));
    check(cudaFree(total));
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    race();
    histograms();
    barriers();
    divergence();
    floatSums();
}
