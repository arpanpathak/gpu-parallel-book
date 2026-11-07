// Chapter 18: which GPUs can reach each other's memory, and how fast bytes move.
// Build: nvcc -arch=native peer.cu -o peer
//
// With 2 or more GPUs that can reach each other, the copies go from GPU 0 to GPU 1.
// With 1 GPU, they stay inside GPU 0 and measure its own memory instead.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const size_t N = size_t(32) << 20;   // 33,554,432 floats, 128 MiB
const size_t BYTES = N * sizeof(float);
const int REPS = 50;

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void copyKernel(const float* src, float* dst, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// The peer matrix: can the GPU of each row read and write the memory of each column?
void printPeerMatrix(int count)
{
    std::printf("peer access (row can reach column):\n      ");
    for (int j = 0; j < count; ++j) {
        std::printf("  GPU%d", j);
    }
    std::printf("\n");

    for (int i = 0; i < count; ++i) {
        std::printf("  GPU%d", i);

        for (int j = 0; j < count; ++j) {
            int ok = 0;
            if (i != j) check(cudaDeviceCanAccessPeer(&ok, i, j));

            const char* cell = i == j ? "X" : ok ? "yes" : "no";
            std::printf("  %4s", cell);
        }
        std::printf("\n");
    }
}

// Average time of 1 run of `run`, in milliseconds, after 0.4 s of warm-up runs.
template <typename F>
float averageMs(F run, cudaStream_t stream)
{
    auto begin = std::chrono::steady_clock::now();
    auto warm = std::chrono::milliseconds(400);
    while (std::chrono::steady_clock::now() - begin < warm) {
        run();
        check(cudaStreamSynchronize(stream));
    }

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start, stream));
    for (int r = 0; r < REPS; ++r) run();
    check(cudaEventRecord(stop, stream));
    check(cudaEventSynchronize(stop));

    float ms;
    check(cudaEventElapsedTime(&ms, start, stop));

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return ms / REPS;
}

bool matches(const float* dst, const std::vector<float>& expected)
{
    std::vector<float> got(N);
    check(cudaMemcpy(got.data(), dst, BYTES, cudaMemcpyDefault));

    for (size_t i = 0; i < N; ++i) {
        if (got[i] != expected[i]) return false;
    }
    return true;
}

void clear(float* p, int device)
{
    check(cudaSetDevice(device));
    check(cudaMemset(p, 0, BYTES));
    check(cudaDeviceSynchronize());
    check(cudaSetDevice(0));
}

void report(const char* how, float ms, bool ok)
{
    double gbPerS = BYTES / 1e9 / (ms / 1e3);
    const char* verdict = ok ? "PASSED" : "FAILED";
    std::printf("  %-26s %7.3f ms %7.1f GB/s  %s\n", how, ms, gbPerS, verdict);
}

int main()
{
    int count = 0;
    check(cudaGetDeviceCount(&count));
    std::printf("GPUs: %d\n", count);

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, d));

        const char* uva = prop.unifiedAddressing ? "yes" : "no";
        std::printf("  GPU%d  %s, %d SMs, ", d, prop.name, prop.multiProcessorCount);
        std::printf("unified addressing %s\n", uva);
    }
    printPeerMatrix(count);

    // Copy from GPU 0 to GPU 1 if GPU 0 can reach it, otherwise within GPU 0.
    int src = 0;
    int dst = 0;
    int ok = 0;
    if (count > 1) check(cudaDeviceCanAccessPeer(&ok, 0, 1));

    if (ok) {
        dst = 1;
        check(cudaSetDevice(src));
        check(cudaDeviceEnablePeerAccess(dst, 0));
    } else {
        std::printf("no peer to reach: copying within GPU0\n");
    }

    std::vector<float> host(N);
    for (size_t i = 0; i < N; ++i) {
        host[i] = float(i % 1000);
    }

    float* a;
    float* b;
    check(cudaSetDevice(src));
    check(cudaMalloc(&a, BYTES));
    check(cudaSetDevice(dst));
    check(cudaMalloc(&b, BYTES));
    check(cudaMemcpy(a, host.data(), BYTES, cudaMemcpyDefault));

    // Unified addressing: the pointer alone tells the runtime which GPU holds it.
    cudaPointerAttributes where;
    check(cudaPointerGetAttributes(&where, b));
    std::printf("buffer b lives on GPU%d\n", where.device);

    check(cudaSetDevice(src));
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    std::printf("%zu MiB from GPU%d to GPU%d, ", BYTES >> 20, src, dst);
    std::printf("average of %d copies\n", REPS);

    auto peerCopy = [&] {
        check(cudaMemcpyPeerAsync(b, dst, a, src, BYTES, stream));
    };

    unsigned blocks = unsigned((N + 255) / 256);
    auto kernelCopy = [&] {
        copyKernel<<<blocks, 256, 0, stream>>>(a, b, N);
    };

    clear(b, dst);
    float msPeer = averageMs(peerCopy, stream);
    bool okPeer = matches(b, host);
    report("cudaMemcpyPeerAsync", msPeer, okPeer);

    clear(b, dst);
    float msKernel = averageMs(kernelCopy, stream);
    check(cudaGetLastError());
    bool okKernel = matches(b, host);
    report("copy kernel on GPU0", msKernel, okKernel);

    check(cudaStreamDestroy(stream));
    check(cudaFree(a));
    check(cudaFree(b));
    return okPeer && okKernel ? 0 : 1;
}
