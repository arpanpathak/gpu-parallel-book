// Chapter 3: vector addition, c = a + b, on the GPU.
// Build: nvcc -arch=native vector_add.cu -o vector_add

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Stops the program if a CUDA call failed (section 3.8).
void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void addVectors(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = 2.0f * i;
    }

    float *d_a, *d_b, *d_c;
    check(cudaMalloc(&d_a, bytes));
    check(cudaMalloc(&d_b, bytes));
    check(cudaMalloc(&d_c, bytes));
    check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    addVectors<<<blocks, threads>>>(d_a, d_b, d_c, n);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    double maxErr = 0;
    for (int i = 0; i < n; ++i) {
        maxErr = std::fmax(maxErr, std::fabs(h_c[i] - 3.0 * i));
    }
    std::printf("n = %d, %d blocks of %d threads\n", n, blocks, threads);
    std::printf("c[0] = %.0f, c[1] = %.0f, c[%d] = %.0f\n", h_c[0], h_c[1], n - 1, h_c[n - 1]);
    std::printf("max error = %g\n", maxErr);

    check(cudaFree(d_a));
    check(cudaFree(d_b));
    check(cudaFree(d_c));
}
