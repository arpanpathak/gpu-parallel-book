// Chapter 9: matrix multiplication from 1 thread per output to register tiles, measured.
// Build: nvcc -arch=native sgemm.cu -o sgemm -lcublas

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

const int N = 4096;                           // C = A B, all N x N, row-major

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

// Average time of `runs` runs of `launch`, in milliseconds, after 1 warm-up run.
template <typename F>
double timeKernel(F launch, int runs)
{
    launch();
    check(cudaDeviceSynchronize());

    double t0 = nowMs();
    for (int r = 0; r < runs; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    return (nowMs() - t0) / runs;
}

// 1. 1 thread per element of C, with threadIdx.x along a column of C.
__global__ void sgemmNaive(const float* A, const float* B, float* C, int n)
{
    int row = blockIdx.x * 16 + threadIdx.x;
    int col = blockIdx.y * 16 + threadIdx.y;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

// 2. The same, with threadIdx.x along a row of C.
__global__ void sgemmCoalesced(const float* A, const float* B, float* C, int n)
{
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

// 3. Tiles of A and B in shared memory: each value loaded is used T times.
template <int T>
__global__ void sgemmTiled(const float* A, const float* B, float* C, int n)
{
    __shared__ float tileA[T][T];
    __shared__ float tileB[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    float sum = 0.0f;
    for (int k0 = 0; k0 < n; k0 += T) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + k0 + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < T; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}

// 4. A block of 256 threads computes 64 x 64 elements of C, each thread 4 x 4 of them.
template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
sgemmRegisters(const float* A, const float* B, float* C, int n)
{
    __shared__ float tileA[64][16 + 1];       // 1 column of padding (Chapter 7)
    __shared__ float tileB[16][64];
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int row0 = blockIdx.y * 64;
    int col0 = blockIdx.x * 64;

    float acc[4][4] = {};
    for (int k0 = 0; k0 < n; k0 += 16) {
        for (int e = threadIdx.x; e < 64 * 16; e += 256) {
            tileA[e / 16][e % 16] = A[(row0 + e / 16) * n + k0 + e % 16];
            tileB[e / 64][e % 64] = B[(k0 + e / 64) * n + col0 + e % 64];
        }
        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = tileA[ty * 4 + i][k];
                b[i] = tileB[k][tx * 4 + i];
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[(row0 + ty * 4 + i) * n + col0 + tx * 4 + j] = acc[i][j];
        }
    }
}

// Registers per thread and resident blocks per SM, as the compiler and the GPU report them.
template <typename K>
void resources(const char* name, K kernel, int threads)
{
    cudaFuncAttributes attr;
    check(cudaFuncGetAttributes(&attr, kernel));
    int blocks;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
    std::printf("  %-26s %3d registers  %2d blocks per SM  %2zu bytes spilled\n", name,
                attr.numRegs, blocks, attr.localSizeBytes);
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    std::vector<float> a(N * N), b(N * N), c(N * N);
    unsigned state = 1;
    for (auto* m : {&a, &b}) {
        for (float& v : *m) {
            state = state * 1664525u + 1013904223u;
            v = (state >> 8) / 16777216.0f - 0.5f;          // in [-0.5, 0.5)
        }
    }

    size_t bytes = size_t(N) * N * sizeof(float);
    float *A, *B, *C;
    check(cudaMalloc(&A, bytes));
    check(cudaMalloc(&B, bytes));
    check(cudaMalloc(&C, bytes));
    check(cudaMemcpy(A, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(B, b.data(), bytes, cudaMemcpyHostToDevice));

    // 64 elements of C, recomputed on the CPU in double precision.
    auto maxError = [&] {
        check(cudaMemcpy(c.data(), C, bytes, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (int s = 0; s < 64; ++s) {
            int i = s * 977 % N;
            int j = (s * 1231 + 7) % N;
            double exact = 0.0;
            for (int k = 0; k < N; ++k) {
                exact += double(a[i * N + k]) * b[k * N + j];
            }
            worst = std::fmax(worst, std::fabs(exact - c[i * N + j]));
        }
        return worst;
    };
    auto report = [&](const char* name, double ms) {
        double gflops = 2.0 * N * N * N / ms / 1e6;
        std::printf("  %-26s %8.2f ms  %7.1f GFLOP/s  max error %.1e\n", name, ms, gflops,
                    maxError());
    };

    cublasHandle_t handle;
    cublasCreate(&handle);
    float one = 1.0f, zero = 0.0f;
    // cuBLAS is column-major: computing B^T A^T in its terms gives C = A B in ours.
    auto cublas = [&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B, N, A, N, &zero,
                    C, N);
    };
    for (int k = 0; k < 10; ++k) {            // warm up: clocks rise under sustained load
        timeKernel(cublas, 5);
    }

    dim3 block(16, 16), grid(N / 16, N / 16);
    dim3 grid64(N / 64, N / 64);
    std::printf("C = A B, %d x %d floats\n", N, N);
    report("1 per thread, by column", timeKernel([&] {
        sgemmNaive<<<grid, block>>>(A, B, C, N);
    }, 2));
    report("1 per thread, by row", timeKernel([&] {
        sgemmCoalesced<<<grid, block>>>(A, B, C, N);
    }, 3));
    report("tiles of 16 x 16", timeKernel([&] {
        sgemmTiled<16><<<grid, block>>>(A, B, C, N);
    }, 5));
    report("tiles of 32 x 32", timeKernel([&] {
        sgemmTiled<32><<<dim3(N / 32, N / 32), dim3(32, 32)>>>(A, B, C, N);
    }, 5));
    report("4 x 4 per thread", timeKernel([&] {
        sgemmRegisters<1><<<grid64, 256>>>(A, B, C, N);
    }, 10));
    report("4 x 4, at least 4 blocks", timeKernel([&] {
        sgemmRegisters<4><<<grid64, 256>>>(A, B, C, N);
    }, 10));
    report("4 x 4, at least 6 blocks", timeKernel([&] {
        sgemmRegisters<6><<<grid64, 256>>>(A, B, C, N);
    }, 10));
    report("cuBLAS", timeKernel(cublas, 10));

    std::printf("resources\n");
    resources("tiles of 16 x 16", sgemmTiled<16>, 256);
    resources("tiles of 32 x 32", sgemmTiled<32>, 1024);
    resources("4 x 4 per thread", sgemmRegisters<1>, 256);
    resources("4 x 4, at least 4 blocks", sgemmRegisters<4>, 256);
    resources("4 x 4, at least 6 blocks", sgemmRegisters<6>, 256);

    cublasDestroy(handle);
    check(cudaFree(A));
    check(cudaFree(B));
    check(cudaFree(C));
}
