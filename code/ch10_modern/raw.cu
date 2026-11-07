// Chapter 10: 3 mistakes that the CUDA C API lets through. Each one runs and does damage.
// Build: nvcc -arch=native raw.cu -o raw

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

const int N = 1 << 21;                        // 2,097,152 floats, 8 MiB per buffer
const int CALLS = 8;

__global__ void scale(const float* in, float* out, int n, float factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = factor * in[i];
    }
}

// Bytes in use from the pool that cudaMallocAsync draws on, in MiB.
double poolMiB()
{
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    unsigned long long used = 0;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    return used / 1048576.0;
}

// Mistake 1: the early return skips both frees.
bool scaleOnGpu(const std::vector<float>& host, std::vector<float>& result, float factor)
{
    int n = static_cast<int>(host.size());
    size_t bytes = n * sizeof(float);

    float* in = nullptr;
    float* out = nullptr;
    cudaMallocAsync(&in, bytes, 0);
    cudaMallocAsync(&out, bytes, 0);

    if (factor == 0.0f) {
        return false;                         // nothing to scale
    }

    cudaMemcpy(in, host.data(), bytes, cudaMemcpyHostToDevice);
    scale<<<(n + 255) / 256, 256>>>(in, out, n, factor);
    cudaMemcpy(result.data(), out, bytes, cudaMemcpyDeviceToHost);

    cudaFreeAsync(in, 0);
    cudaFreeAsync(out, 0);
    return true;
}

int main()
{
    std::vector<float> host(N, 1.0f);
    std::vector<float> result(N, 0.0f);
    size_t bytes = N * sizeof(float);

    double before = poolMiB();
    for (int call = 0; call < CALLS; ++call) {
        scaleOnGpu(host, result, 0.0f);
    }
    double lost = poolMiB() - before;
    std::printf("1. %d calls that return early: %.1f MiB lost\n", CALLS, lost);

    float* in = nullptr;
    float* out = nullptr;
    cudaMallocAsync(&in, bytes, 0);
    cudaMallocAsync(&out, bytes, 0);
    cudaMemcpy(in, host.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(out, 0, bytes);

    // Mistake 2: 2,048 threads per block is over the limit of 1,024. Nothing reports it.
    scale<<<N / 2048, 2048>>>(in, out, N, 2.0f);
    cudaMemcpy(result.data(), out, bytes, cudaMemcpyDeviceToHost);
    std::printf("2. 2048 threads per block: out[0] = %.1f, expected 2.0\n", result[0]);

    cudaError_t err = cudaGetLastError();
    std::printf("   the error, found only by asking: %s\n", cudaGetErrorString(err));

    // Mistake 3: N counts floats, but cudaMemcpy counts bytes.
    cudaMemset(in, 0, bytes);
    cudaMemcpy(in, host.data(), N, cudaMemcpyHostToDevice);
    scale<<<N / 256, 256>>>(in, out, N, 2.0f);
    cudaMemcpy(result.data(), out, bytes, cudaMemcpyDeviceToHost);

    int correct = 0;
    for (float x : result) {
        correct += (x == 2.0f);
    }
    std::printf("3. a copy of N bytes, not N floats: %d of %d correct\n", correct, N);

    cudaFreeAsync(in, 0);
    cudaFreeAsync(out, 0);
}
