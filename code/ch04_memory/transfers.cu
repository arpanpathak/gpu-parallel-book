// Chapter 4: how fast data moves between host and device, measured on this GPU.
// Build: nvcc -arch=native transfers.cu -o transfers

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Average time of `reps` copies of `bytes`, in milliseconds, after 1 warm-up copy.
double timeCopy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps)
{
    check(cudaMemcpy(dst, src, bytes, kind));

    double t0 = nowMs();
    for (int r = 0; r < reps; ++r) {
        check(cudaMemcpy(dst, src, bytes, kind));
    }

    return (nowMs() - t0) / reps;
}

__global__ void scale(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 2.0f * x[i];
    }
}

// Average time of `reps` launches of `scale` over `x`, in milliseconds.
double timeScale(float* x, int n, int reps)
{
    double t0 = nowMs();
    for (int r = 0; r < reps; ++r) {
        scale<<<(n + 255) / 256, 256>>>(x, n);
    }

    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    return (nowMs() - t0) / reps;
}

// 1. Copy speed from pageable and from pinned host memory, both directions.
void copySpeeds(float* pageable, float* pinned, float* device, size_t bytes)
{
    struct Case { const char* name; void* dst; const void* src; cudaMemcpyKind kind; };
    const Case cases[] = {
        {"pageable host -> device", device, pageable, cudaMemcpyHostToDevice},
        {"pinned   host -> device", device, pinned, cudaMemcpyHostToDevice},
        {"device -> pageable host", pageable, device, cudaMemcpyDeviceToHost},
        {"device -> pinned   host", pinned, device, cudaMemcpyDeviceToHost},
    };

    std::printf("copies of %zu MiB\n", bytes >> 20);
    for (const Case& c : cases) {
        double ms = timeCopy(c.dst, c.src, bytes, c.kind, 20);
        std::printf("  %s  %6.2f ms  %5.1f GB/s\n", c.name, ms, bytes / 1e9 / (ms / 1e3));
    }
}

// 2. Copy time against copy size, pinned host -> device.
void sizeSweep(float* pinned, float* device)
{
    std::printf("pinned host -> device, by size\n");

    for (size_t bytes = 4 << 10; bytes <= (64 << 20); bytes *= 4) {
        double ms = timeCopy(device, pinned, bytes, cudaMemcpyHostToDevice, 200);
        std::printf("  %8zu KiB  %8.1f us  %5.1f GB/s\n",
                    bytes >> 10, ms * 1e3, bytes / 1e9 / (ms / 1e3));
    }
}

// 3. Unified memory: 1 pointer for both sides. The host writes it first.
void unifiedMemory(int n)
{
    float* managed;
    check(cudaMallocManaged(&managed, n * sizeof(float)));

    for (int i = 0; i < n; ++i) {
        managed[i] = 1.0f;
    }

    std::printf("unified memory, kernel over %d MiB\n", int(n * sizeof(float) >> 20));
    std::printf("  1st launch, after the host wrote it  %6.2f ms\n", timeScale(managed, n, 1));
    std::printf("  next 10 launches, average            %6.2f ms\n", timeScale(managed, n, 10));
    std::printf("  host reads managed[0] = %g\n", managed[0]);

    check(cudaFree(managed));
}

// 4. Zero-copy: pinned host memory that a kernel reads and writes in place.
void zeroCopy(float* device, int n)
{
    float* mapped;
    float* mappedOnDevice;
    check(cudaHostAlloc(&mapped, n * sizeof(float), cudaHostAllocMapped));
    check(cudaHostGetDevicePointer(&mappedOnDevice, mapped, 0));
    std::memset(mapped, 0, n * sizeof(float));

    timeScale(device, n, 1);                  // warm up the kernel itself
    std::printf("kernel over %d MiB, average of 10 launches\n", int(n * sizeof(float) >> 20));
    std::printf("  in device memory      %6.2f ms\n", timeScale(device, n, 10));
    std::printf("  in zero-copy memory   %6.2f ms\n", timeScale(mappedOnDevice, n, 10));

    check(cudaFreeHost(mapped));
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %s, %d copy engine(s)\n", prop.name,
                prop.integrated ? "integrated" : "discrete", prop.asyncEngineCount);

    const int n = 16 << 20;                   // 16,777,216 floats
    const size_t bytes = n * sizeof(float);   // 64 MiB

    float* pageable = static_cast<float*>(std::malloc(bytes));
    float* pinned;
    float* device;
    check(cudaMallocHost(&pinned, bytes));
    check(cudaMalloc(&device, bytes));
    std::memset(pageable, 0, bytes);
    std::memset(pinned, 0, bytes);

    copySpeeds(pageable, pinned, device, bytes);
    sizeSweep(pinned, device);
    unifiedMemory(n);
    zeroCopy(device, n);

    std::free(pageable);
    check(cudaFreeHost(pinned));
    check(cudaFree(device));
}
