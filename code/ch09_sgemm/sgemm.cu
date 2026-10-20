// Chapter 9: matrix multiplication, from 1 thread per element to a 4 x 4 patch per thread.
// Build: nvcc -arch=native sgemm.cu -o sgemm -lcublas

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using Matrix = std::vector<float>;

const int N = 4096;                           // C = A B, all N x N, stored row by row

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

// 1. 1 thread per element of C, threadIdx.x along a column of C.
__global__ void byColumn(const float* A, const float* B, float* C)
{
    int row = blockIdx.x * 16 + threadIdx.x;
    int col = blockIdx.y * 16 + threadIdx.y;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// 2. The same, threadIdx.x along a row of C.
__global__ void byRow(const float* A, const float* B, float* C)
{
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// 3. A block of T x T threads computes a T x T tile of C, from tiles in shared memory.
template <int T>
__global__ void tiled(const float* A, const float* B, float* C)
{
    __shared__ float tileA[T][T];
    __shared__ float tileB[T][T];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int row = blockIdx.y * T + y;
    int col = blockIdx.x * T + x;

    float sum = 0.0f;
    for (int k0 = 0; k0 < N; k0 += T) {
        tileA[y][x] = A[row * N + k0 + x];
        tileB[y][x] = B[(k0 + y) * N + col];
        __syncthreads();

        for (int k = 0; k < T; ++k) {
            sum += tileA[y][k] * tileB[k][x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// 4. A block of 256 threads computes a 64 x 64 tile of C, each thread a 4 x 4 patch of it.
template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
patches(const float* A, const float* B, float* C)
{
    __shared__ float tileA[64][16 + 1];       // 1 column of padding (Chapter 7)
    __shared__ float tileB[16][64];
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int row0 = blockIdx.y * 64;
    int col0 = blockIdx.x * 64;

    float acc[4][4] = {};
    for (int k0 = 0; k0 < N; k0 += 16) {
        for (int e = threadIdx.x; e < 64 * 16; e += 256) {
            tileA[e / 16][e % 16] = A[(row0 + e / 16) * N + k0 + e % 16];
            tileB[e / 64][e % 64] = B[(k0 + e / 64) * N + col0 + e % 64];
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
            C[(row0 + ty * 4 + i) * N + col0 + tx * 4 + j] = acc[i][j];
        }
    }
}

// Registers per thread, blocks per SM and spilled bytes, as the GPU reports them.
template <typename K>
void resources(const char* name, K kernel, int threads)
{
    cudaFuncAttributes attr;
    check(cudaFuncGetAttributes(&attr, kernel));

    int blocks;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));

    std::printf("  %-26s %3d registers", name, attr.numRegs);
    std::printf("  %d blocks per SM", blocks);
    std::printf("  %2zu bytes spilled\n", attr.localSizeBytes);
}

// Values in [-0.5, 0.5) from a small linear congruential generator.
void fillRandom(Matrix& m, unsigned& state)
{
    for (float& v : m) {
        state = state * 1664525u + 1013904223u;
        v = (state >> 8) / 16777216.0f - 0.5f;
    }
}

// The largest difference from a double-precision product, over 64 elements of C.
double maxError(const Matrix& a, const Matrix& b, const Matrix& c)
{
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
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    Matrix a(N * N), b(N * N), c(N * N);
    unsigned state = 1;
    fillRandom(a, state);
    fillRandom(b, state);

    size_t bytes = size_t(N) * N * sizeof(float);
    float *A, *B, *C;
    check(cudaMalloc(&A, bytes));
    check(cudaMalloc(&B, bytes));
    check(cudaMalloc(&C, bytes));
    check(cudaMemcpy(A, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(B, b.data(), bytes, cudaMemcpyHostToDevice));

    auto report = [&](const char* name, double ms) {
        check(cudaMemcpy(c.data(), C, bytes, cudaMemcpyDeviceToHost));
        double gflops = 2.0 * N * N * N / ms / 1e6;
        std::printf("  %-26s %8.2f ms  %7.1f GFLOP/s", name, ms, gflops);
        std::printf("  max error %.1e\n", maxError(a, b, c));
    };

    // cuBLAS stores matrices by column: B^T A^T in its terms is A B in ours.
    cublasHandle_t handle;
    cublasCreate(&handle);
    float one = 1.0f, zero = 0.0f;
    auto op = CUBLAS_OP_N;
    auto cublas = [&] {
        cublasSgemm(handle, op, op, N, N, N, &one, B, N, A, N, &zero, C, N);
    };

    dim3 block16(16, 16), grid16(N / 16, N / 16);
    dim3 block32(32, 32), grid32(N / 32, N / 32);
    dim3 grid64(N / 64, N / 64);
    auto column = [&] { byColumn<<<grid16, block16>>>(A, B, C); };
    auto row = [&] { byRow<<<grid16, block16>>>(A, B, C); };
    auto tiles16 = [&] { tiled<16><<<grid16, block16>>>(A, B, C); };
    auto tiles32 = [&] { tiled<32><<<grid32, block32>>>(A, B, C); };
    auto patches1 = [&] { patches<1><<<grid64, 256>>>(A, B, C); };
    auto patches4 = [&] { patches<4><<<grid64, 256>>>(A, B, C); };
    auto patches6 = [&] { patches<6><<<grid64, 256>>>(A, B, C); };

    for (int k = 0; k < 10; ++k) {            // warm up: clocks rise under sustained load
        timeKernel(cublas, 5);
    }

    std::printf("C = A B, %d x %d floats\n", N, N);
    report("1 per thread, by column", timeKernel(column, 2));
    report("1 per thread, by row", timeKernel(row, 3));
    report("tiles of 16 x 16", timeKernel(tiles16, 5));
    report("tiles of 32 x 32", timeKernel(tiles32, 5));
    report("4 x 4 per thread", timeKernel(patches1, 10));
    report("4 x 4, at least 4 blocks", timeKernel(patches4, 10));
    report("4 x 4, at least 6 blocks", timeKernel(patches6, 10));
    report("cuBLAS", timeKernel(cublas, 10));

    std::printf("resources\n");
    resources("tiles of 16 x 16", tiled<16>, 256);
    resources("tiles of 32 x 32", tiled<32>, 1024);
    resources("4 x 4 per thread", patches<1>, 256);
    resources("4 x 4, at least 4 blocks", patches<4>, 256);
    resources("4 x 4, at least 6 blocks", patches<6>, 256);

    cublasDestroy(handle);
    check(cudaFree(A));
    check(cudaFree(B));
    check(cudaFree(C));
}
