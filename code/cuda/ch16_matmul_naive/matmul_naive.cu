// Chapter 16: matrix multiplication, 1 thread per entry of C.
// Build: nvcc -O3 -arch=native matmul_naive.cu -o matmul_naive

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

// C (m x n) = A (m x k) B (k x n), all row-major. 1 thread computes 1 entry of C.
template <typename T>
__global__ void matmul(int m, int n, int k, const T* a, const T* b, T* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    T sum = 0;
    for (int s = 0; s < k; ++s) {
        sum += a[row * k + s] * b[s * n + col];
    }
    c[row * n + col] = sum;
}

// The same product for W x W matrices, with the width W fixed at compile time.
template <int W>
__global__ void matmulFixed(const float* a, const float* b, float* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= W || col >= W) {
        return;
    }

    float sum = 0.0f;
    for (int s = 0; s < W; ++s) {
        sum += a[row * W + s] * b[s * W + col];
    }
    c[row * W + col] = sum;
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Average time of `runs` launches in milliseconds, after 1 launch that is not timed.
template <typename F>
double timeMs(F launch, int runs)
{
    launch();
    check(cudaDeviceSynchronize());

    double start = nowMs();
    for (int r = 0; r < runs; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    return (nowMs() - start) / runs;
}

// Integers in [-5, 5] from a small linear congruential generator: every partial sum
// of a product with k <= 4096 is an integer below 2^24, so float results are exact.
template <typename T>
std::vector<T> integers(size_t len, unsigned* state)
{
    std::vector<T> values(len);
    for (size_t i = 0; i < len; ++i) {
        *state = *state * 1664525u + 1013904223u;
        values[i] = T(int((*state >> 8) % 11) - 5);
    }
    return values;
}

// The number of 64 sampled entries of C that equal the exact product.
template <typename T>
int exactEntries(const std::vector<T>& a, const std::vector<T>& b,
                 const std::vector<T>& c, int n)
{
    int exact = 0;
    for (int s = 0; s < 64; ++s) {
        int i = s * 977 % n;
        int j = (s * 1231 + 7) % n;

        double sum = 0.0;
        for (int p = 0; p < n; ++p) {
            sum += double(a[i * n + p]) * double(b[p * n + j]);
        }
        exact += double(c[i * n + j]) == sum;
    }
    return exact;
}

// Launches that keep n^3 x runs near 2^33, so each size runs for a similar time.
int runsFor(int n)
{
    double runs = double(1ull << 33) / (double(n) * n * n);
    return runs < 1 ? 1 : (runs > 50 ? 50 : int(runs));
}

// A grid of side x side blocks that covers an m x n matrix C.
dim3 gridFor(int m, int n, int side)
{
    return dim3((n + side - 1) / side, (m + side - 1) / side);
}

// The 4 x 4 example with blocks of 2 x 2 threads, compared with the hand result.
bool runExample()
{
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> b{1, 0, 2, 1, 0, 1, 1, 2, 2, 1, 0, 1, 1, 2, 1, 0};
    std::vector<float> want{11, 13, 8, 8, 27, 29, 24, 24, 43, 45, 40, 40, 59, 61, 56, 56};
    std::vector<float> c(16);

    float *a_gpu, *b_gpu, *c_gpu;
    check(cudaMalloc(&a_gpu, 16 * sizeof(float)));
    check(cudaMalloc(&b_gpu, 16 * sizeof(float)));
    check(cudaMalloc(&c_gpu, 16 * sizeof(float)));
    check(cudaMemcpy(a_gpu, a.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b_gpu, b.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));

    matmul<float><<<gridFor(4, 4, 2), dim3(2, 2)>>>(4, 4, 4, a_gpu, b_gpu, c_gpu);
    check(cudaGetLastError());
    check(cudaMemcpy(c.data(), c_gpu, 16 * sizeof(float), cudaMemcpyDeviceToHost));

    check(cudaFree(a_gpu));
    check(cudaFree(b_gpu));
    check(cudaFree(c_gpu));

    std::printf("C = A B on the 4 x 4 example, a grid of 2 x 2 blocks of 2 x 2 threads\n");
    for (int i = 0; i < 4; ++i) {
        const float* row = &c[i * 4];
        std::printf("  %4.0f %4.0f %4.0f %4.0f\n", row[0], row[1], row[2], row[3]);
    }
    bool same = c == want;
    std::printf("equal to the product computed by hand: %s\n\n", same ? "yes" : "no");
    return same;
}

// Time 1 kernel on n x n matrices, print 1 line of the table, and return true if exact.
template <typename T, typename F>
bool runSize(const char* name, int n, F launch)
{
    unsigned state = 1;
    std::vector<T> a = integers<T>(size_t(n) * n, &state);
    std::vector<T> b = integers<T>(size_t(n) * n, &state);
    std::vector<T> c(size_t(n) * n);
    size_t bytes = size_t(n) * n * sizeof(T);

    T *a_gpu, *b_gpu, *c_gpu;
    check(cudaMalloc(&a_gpu, bytes));
    check(cudaMalloc(&b_gpu, bytes));
    check(cudaMalloc(&c_gpu, bytes));
    check(cudaMemcpy(a_gpu, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b_gpu, b.data(), bytes, cudaMemcpyHostToDevice));

    double ms = timeMs([&] { launch(a_gpu, b_gpu, c_gpu); }, runsFor(n));
    check(cudaMemcpy(c.data(), c_gpu, bytes, cudaMemcpyDeviceToHost));

    check(cudaFree(a_gpu));
    check(cudaFree(b_gpu));
    check(cudaFree(c_gpu));

    double flop = 2.0 * n * n * double(n);
    double gflops = flop / ms / 1e6;
    double fetched = flop * sizeof(T) / ms / 1e6;
    int exact = exactEntries(a, b, c, n);

    std::printf("  %5d  %-10s %9.3f %9.1f", n, name, ms, gflops);
    std::printf(" %10.1f %6d of 64\n", fetched, exact);
    return exact == 64;
}

template <typename T>
bool runGeneric(const char* name, int n)
{
    dim3 grid = gridFor(n, n, 16);
    dim3 block(16, 16);
    auto launch = [=](const T* a, const T* b, T* c) {
        matmul<T><<<grid, block>>>(n, n, n, a, b, c);
    };
    return runSize<T>(name, n, launch);
}

bool runFixed()
{
    const int n = 1024;
    dim3 grid = gridFor(n, n, 16);
    dim3 block(16, 16);
    auto launch = [=](const float* a, const float* b, float* c) {
        matmulFixed<n><<<grid, block>>>(a, b, c);
    };
    return runSize<float>("f32 fixed", n, launch);
}

// About 0.5 s of 1024 x 1024 products: the GPU raises its clocks under sustained load.
void warmUp()
{
    const int n = 1024;
    size_t bytes = size_t(n) * n * sizeof(float);
    float *a_gpu, *b_gpu, *c_gpu;
    check(cudaMalloc(&a_gpu, bytes));
    check(cudaMalloc(&b_gpu, bytes));
    check(cudaMalloc(&c_gpu, bytes));
    check(cudaMemset(a_gpu, 0, bytes));
    check(cudaMemset(b_gpu, 0, bytes));

    double start = nowMs();
    while (nowMs() - start < 500.0) {
        matmul<float><<<gridFor(n, n, 16), dim3(16, 16)>>>(n, n, n, a_gpu, b_gpu, c_gpu);
        check(cudaDeviceSynchronize());
    }

    check(cudaFree(a_gpu));
    check(cudaFree(b_gpu));
    check(cudaFree(c_gpu));
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);

    bool passed = runExample();
    warmUp();

    std::printf("C = A B, n x n, 1 thread per entry of C, blocks of 16 x 16 threads\n");
    std::printf("      n  type              ms   GFLOP/s  fetched GB/s   exact\n");
    for (int n : {512, 1024, 2048, 4096}) {
        passed &= runGeneric<float>("f32", n);
    }
    passed &= runFixed();
    for (int n : {512, 1024, 2048}) {
        passed &= runGeneric<double>("f64", n);
    }

    std::printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
