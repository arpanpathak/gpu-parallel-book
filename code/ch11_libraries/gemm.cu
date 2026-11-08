// Chapter 11: cuBLAS with row-major matrices, then SGEMM in FP32 and on tensor cores.
// Build: nvcc -arch=native -O2 gemm.cu -o gemm -lcublas

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using Matrix = std::vector<float>;

const int N = 4096;                           // the size of Chapter 9

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

void check(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %s\n", cublasGetStatusString(status));
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
double timeMs(F launch, int runs)
{
    launch();
    check(cudaDeviceSynchronize());

    double t0 = nowMs();
    for (int r = 0; r < runs; ++r) {
        launch();
    }
    check(cudaDeviceSynchronize());

    return (nowMs() - t0) / runs;
}

void print(const char* name, const Matrix& m, int rows, int cols)
{
    std::printf("  %s =", name);
    for (int i = 0; i < rows; ++i) {
        std::printf(i == 0 ? " [" : "; ");
        for (int j = 0; j < cols; ++j) {
            std::printf(j == 0 ? "%g" : " %g", m[i * cols + j]);
        }
    }
    std::printf("]\n");
}

// C = A B for a 2 x 3 matrix A and a 3 x 2 matrix B, all stored row by row.
bool smallProduct(cublasHandle_t handle)
{
    Matrix a = {1, 2, 3,
                4, 5, 6};
    Matrix b = {7, 8,
                9, 10,
                11, 12};
    Matrix expected = {58, 64,
                       139, 154};
    Matrix c(4);

    float *A, *B, *C;
    check(cudaMalloc(&A, 6 * sizeof(float)));
    check(cudaMalloc(&B, 6 * sizeof(float)));
    check(cudaMalloc(&C, 4 * sizeof(float)));
    check(cudaMemcpy(A, a.data(), 6 * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(B, b.data(), 6 * sizeof(float), cudaMemcpyHostToDevice));

    float one = 1.0f, zero = 0.0f;
    auto op = CUBLAS_OP_N;
    std::printf("C = A B with A 2 x 3 and B 3 x 2, stored row by row\n");
    print("A", a, 2, 3);
    print("B", b, 3, 2);

    // Wrong: cuBLAS reads each array column by column.
    check(cublasSgemm(handle, op, op, 2, 2, 3, &one, A, 2, B, 3, &zero, C, 2));
    check(cudaMemcpy(c.data(), C, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    print("C, operands as given  ", c, 2, 2);

    // Right: in column-major terms our arrays hold A^T and B^T. Ask for B^T A^T = C^T,
    // whose column-major storage is C row by row.
    check(cublasSgemm(handle, op, op, 2, 2, 3, &one, B, 2, A, 3, &zero, C, 2));
    check(cudaMemcpy(c.data(), C, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    print("C, operands swapped   ", c, 2, 2);

    check(cudaFree(A));
    check(cudaFree(B));
    check(cudaFree(C));
    return c == expected;
}

// Values in [-0.5, 0.5) from a small linear congruential generator, as in Chapter 9.
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

    check(cudaFree(nullptr));                 // create the CUDA context before timing
    double t0 = nowMs();
    cublasHandle_t handle;
    check(cublasCreate(&handle));
    std::printf("cublasCreate took %.1f ms\n", nowMs() - t0);

    bool ok = smallProduct(handle);

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

    float one = 1.0f, zero = 0.0f;
    auto op = CUBLAS_OP_N;
    auto gemm = [&] {
        check(cublasSgemm(handle, op, op, N, N, N, &one, B, N, A, N, &zero, C, N));
    };

    auto report = [&](const char* name, double ms) {
        check(cudaMemcpy(c.data(), C, bytes, cudaMemcpyDeviceToHost));
        double gflops = 2.0 * N * N * N / ms / 1e6;
        double error = maxError(a, b, c);
        std::printf("  %-22s %8.2f ms  %7.1f GFLOP/s", name, ms, gflops);
        std::printf("  max error %.1e\n", error);
        return error;
    };

    for (int k = 0; k < 10; ++k) {            // warm up: clocks rise under sustained load
        timeMs(gemm, 5);
    }

    std::printf("C = A B, %d x %d floats, average of 10 runs\n", N, N);
    double fp32 = report("cuBLAS, FP32", timeMs(gemm, 10));

    check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    double tf32 = report("cuBLAS, TF32 tensor", timeMs(gemm, 10));

    ok = ok && fp32 < 1e-3 && tf32 < 1e-2;
    std::printf("%s\n", ok ? "PASSED" : "FAILED");

    check(cublasDestroy(handle));
    check(cudaFree(A));
    check(cudaFree(B));
    check(cudaFree(C));
}
