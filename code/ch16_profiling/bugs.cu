// Chapter 16: 4 bugs that print no error, for Compute Sanitizer to find.
// Build: nvcc -arch=native -lineinfo bugs.cu -o bugs
// Run:   ./bugs bounds | race | barrier | uninit

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// 1. Chapter 3's vector addition without its guard `if (i < n)`.
__global__ void addNoGuard(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// 2. Chapter 5's block sum with its barriers left out.
__global__ void sumNoBarrier(const float* in, float* out)
{
    __shared__ float s[256];
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * 256 + t];

    for (int stride = 128; stride > 0; stride /= 2) {
        if (t < stride) {
            s[t] += s[t + stride];
        }
    }

    if (t == 0) {
        out[blockIdx.x] = s[0];
    }
}

// 3. Even and odd threads of each warp wait at 2 different barriers.
__global__ void barrierInBranch(float* out)
{
    __shared__ float s[256];
    int t = threadIdx.x;
    s[t] = t;

    if (t % 2 == 0) {
        __syncthreads();
        out[t] = s[255 - t];
    } else {
        __syncthreads();
        out[t] = s[255 - t];
    }
}

// 4. A sum over memory that nothing has written.
__global__ void sumUnwritten(const float* in, float* out, int n)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += 32) {
        sum += in[i];
    }
    atomicAdd(out, sum);
}

void bounds()
{
    const int n = 1000;
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    size_t bytes = n * sizeof(float);

    float *da, *db, *dc;
    check(cudaMalloc(&da, bytes));
    check(cudaMalloc(&db, bytes));
    check(cudaMalloc(&dc, bytes));
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    addNoGuard<<<4, 256>>>(da, db, dc, n);
    check(cudaGetLastError());
    check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int i = 0; i < n; ++i) {
        wrong += c[i] != 3.0f;
    }
    std::printf("1,024 threads for 1,000 elements: %d wrong elements\n", wrong);

    check(cudaFree(da));
    check(cudaFree(db));
    check(cudaFree(dc));
}

void race()
{
    const int blocks = 64;
    std::vector<float> in(blocks * 256, 1.0f), out(blocks);

    float *din, *dout;
    check(cudaMalloc(&din, in.size() * sizeof(float)));
    check(cudaMalloc(&dout, blocks * sizeof(float)));
    check(cudaMemcpy(din, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));

    sumNoBarrier<<<blocks, 256>>>(din, dout);
    check(cudaGetLastError());
    check(cudaMemcpy(out.data(), dout, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (float sum : out) {
        wrong += sum != 256.0f;
    }
    std::printf("%d blocks of 256 ones: %d wrong block sums\n", blocks, wrong);

    check(cudaFree(din));
    check(cudaFree(dout));
}

void barrier()
{
    std::vector<float> out(256);
    float* dout;
    check(cudaMalloc(&dout, 256 * sizeof(float)));

    barrierInBranch<<<1, 256>>>(dout);
    check(cudaGetLastError());
    check(cudaMemcpy(out.data(), dout, 256 * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int t = 0; t < 256; ++t) {
        wrong += out[t] != 255 - t;
    }
    std::printf("2 barriers, 1 per branch: %d wrong elements of 256\n", wrong);

    check(cudaFree(dout));
}

void uninit()
{
    const int n = 1024;
    float *din, *dout;
    check(cudaMalloc(&din, n * sizeof(float)));
    check(cudaMalloc(&dout, sizeof(float)));
    check(cudaMemset(dout, 0, sizeof(float)));

    sumUnwritten<<<1, 32>>>(din, dout, n);
    check(cudaGetLastError());

    float sum;
    check(cudaMemcpy(&sum, dout, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("sum of 1,024 floats nobody wrote: %g\n", sum);

    check(cudaFree(din));
    check(cudaFree(dout));
}

int main(int argc, char** argv)
{
    const char* bug = argc > 1 ? argv[1] : "";

    if (std::strcmp(bug, "bounds") == 0) {
        bounds();
    } else if (std::strcmp(bug, "race") == 0) {
        race();
    } else if (std::strcmp(bug, "barrier") == 0) {
        barrier();
    } else if (std::strcmp(bug, "uninit") == 0) {
        uninit();
    } else {
        std::fprintf(stderr, "usage: bugs bounds | race | barrier | uninit\n");
        return 1;
    }

    check(cudaDeviceSynchronize());
    return 0;
}
