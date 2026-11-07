// Chapter 18: register tiling, a TM x TN block of C per thread.
// Build: nvcc -O3 -arch=native matmul_register_tiled.cu -o matmul_register_tiled

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

using Matrix = std::vector<float>;

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// C (m x n) = A (m x k) B (k x n), all row-major. A block computes a BM x BN tile of C
// in phases of BK; thread (ty, tx) computes the TM x TN block of that tile that starts
// at row ty * TM and column tx * TN, and keeps its TM * TN sums in registers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmulRegister(int m, int n, int k, const float* a, const float* b, float* c)
{
    const int threads = (BM / TM) * (BN / TN);
    __shared__ float tile_a[BM][BK + 1];      // 1 column of padding against bank conflicts
    __shared__ float tile_b[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int id = ty * (BN / TN) + tx;
    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    int phases = (k + BK - 1) / BK;
    for (int t = 0; t < phases; ++t) {
        #pragma unroll
        for (int q = 0; q < BM * BK / threads; ++q) {
            int e = q * threads + id;
            int r = row0 + e / BK;
            int s = t * BK + e % BK;
            tile_a[e / BK][e % BK] = r < m && s < k ? a[r * k + s] : 0.0f;
        }
        #pragma unroll
        for (int q = 0; q < BK * BN / threads; ++q) {
            int e = q * threads + id;
            int s = t * BK + e / BN;
            int col = col0 + e % BN;
            tile_b[e / BN][e % BN] = s < k && col < n ? b[s * n + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int u = 0; u < BK; ++u) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = tile_a[ty * TM + i][u];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = tile_b[u][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = row0 + ty * TM + i;
            int col = col0 + tx * TN + j;
            if (row < m && col < n) {
                c[row * n + col] = acc[i][j];
            }
        }
    }
}

// The tiled kernel of Chapter 17, for comparison: a block of T x T threads computes
// a T x T square of C in ceil(k / T) phases, with T x T tiles in shared memory.
template <int T>
__global__ void matmulTiled(int m, int n, int k, const float* a, const float* b, float* c)
{
    __shared__ float tile_a[T][T];
    __shared__ float tile_b[T][T];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;

    float sum = 0.0f;
    int phases = (k + T - 1) / T;
    for (int t = 0; t < phases; ++t) {
        int a_col = t * T + tx;
        int b_row = t * T + ty;
        tile_a[ty][tx] = row < m && a_col < k ? a[row * k + a_col] : 0.0f;
        tile_b[ty][tx] = b_row < k && col < n ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int u = 0; u < T; ++u) {
            sum += tile_a[ty][u] * tile_b[u][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// A product on the device: the 3 buffers and their sizes.
struct Product {
    int m, n, k;
    float *a, *b, *c;
};

template <int BM, int BN, int BK, int TM, int TN>
void launch(const Product& p)
{
    dim3 grid((p.n + BN - 1) / BN, (p.m + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    matmulRegister<BM, BN, BK, TM, TN><<<grid, block>>>(p.m, p.n, p.k, p.a, p.b, p.c);
}

void launchTiled16(const Product& p)
{
    dim3 grid((p.n + 15) / 16, (p.m + 15) / 16);
    matmulTiled<16><<<grid, dim3(16, 16)>>>(p.m, p.n, p.k, p.a, p.b, p.c);
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
Matrix integers(size_t len, unsigned* state)
{
    Matrix values(len);
    for (size_t i = 0; i < len; ++i) {
        *state = *state * 1664525u + 1013904223u;
        values[i] = float(int((*state >> 8) % 11) - 5);
    }
    return values;
}

// The number of `samples` entries of C (all of them if C is smaller) that are exact.
int exactEntries(const Matrix& a, const Matrix& b, const Matrix& c,
                 int m, int n, int k, int samples)
{
    int count = m * n < samples ? m * n : samples;
    int exact = 0;
    for (int s = 0; s < count; ++s) {
        int i = count == m * n ? s / n : s * 977 % m;
        int j = count == m * n ? s % n : (s * 1231 + 7) % n;

        double sum = 0.0;
        for (int p = 0; p < k; ++p) {
            sum += double(a[i * k + p]) * double(b[p * n + j]);
        }
        exact += double(c[i * n + j]) == sum;
    }
    return exact;
}

// Host and device copies of A, B and C for an m x k by k x n product.
struct Problem {
    Matrix a, b, c;
    Product gpu;
};

Problem makeProblem(int m, int n, int k, unsigned seed)
{
    Problem p;
    p.a = integers(size_t(m) * k, &seed);
    p.b = integers(size_t(k) * n, &seed);
    p.c.resize(size_t(m) * n);
    p.gpu = {m, n, k, nullptr, nullptr, nullptr};

    check(cudaMalloc(&p.gpu.a, p.a.size() * sizeof(float)));
    check(cudaMalloc(&p.gpu.b, p.b.size() * sizeof(float)));
    check(cudaMalloc(&p.gpu.c, p.c.size() * sizeof(float)));
    check(cudaMemcpy(p.gpu.a, p.a.data(), p.a.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.gpu.b, p.b.data(), p.b.size() * 4, cudaMemcpyHostToDevice));
    return p;
}

// Copy C back, free the device buffers, and count the exact entries.
int finish(Problem& p, int samples)
{
    check(cudaMemcpy(p.c.data(), p.gpu.c, p.c.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaFree(p.gpu.a));
    check(cudaFree(p.gpu.b));
    check(cudaFree(p.gpu.c));
    return exactEntries(p.a, p.b, p.c, p.gpu.m, p.gpu.n, p.gpu.k, samples);
}

// The 4 x 4 example: 1 block of 2 x 2 threads, each thread a 2 x 2 block of C.
bool runExample()
{
    Matrix a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Matrix b{1, 0, 2, 1, 0, 1, 1, 2, 2, 1, 0, 1, 1, 2, 1, 0};
    Matrix want{11, 13, 8, 8, 27, 29, 24, 24, 43, 45, 40, 40, 59, 61, 56, 56};
    Matrix c(16);

    Product p = {4, 4, 4, nullptr, nullptr, nullptr};
    check(cudaMalloc(&p.a, 64));
    check(cudaMalloc(&p.b, 64));
    check(cudaMalloc(&p.c, 64));
    check(cudaMemcpy(p.a, a.data(), 64, cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.b, b.data(), 64, cudaMemcpyHostToDevice));

    launch<4, 4, 2, 2, 2>(p);
    check(cudaGetLastError());
    check(cudaMemcpy(c.data(), p.c, 64, cudaMemcpyDeviceToHost));

    check(cudaFree(p.a));
    check(cudaFree(p.b));
    check(cudaFree(p.c));

    std::printf("C = A B on the 4 x 4 example, 1 block of 2 x 2 threads,");
    std::printf(" 2 x 2 entries per thread\n");
    for (int i = 0; i < 4; ++i) {
        const float* row = &c[i * 4];
        std::printf("  %4.0f %4.0f %4.0f %4.0f\n", row[0], row[1], row[2], row[3]);
    }
    bool same = c == want;
    std::printf("equal to the product computed by hand: %s\n\n", same ? "yes" : "no");
    return same;
}

// A shape that is not a multiple of any tile: every entry of C is checked.
bool runShape(int m, int n, int k)
{
    Problem p = makeProblem(m, n, k, 7);
    launch<64, 64, 16, 4, 4>(p.gpu);
    check(cudaGetLastError());

    int exact = finish(p, m * n);
    std::printf("  m = %4d, k = %4d, n = %4d, 4 x 4 per thread:", m, k, n);
    std::printf(" %7d of %7d entries exact\n\n", exact, m * n);
    return exact == m * n;
}

// Time 1 kernel on n x n matrices and print 1 line of the table. The kernel computes
// bm x bn tiles of C, tm x tn entries per thread.
template <typename F>
bool runTimed(int n, const char* name, int bm, int bn, int tm, int tn, F launch)
{
    Problem p = makeProblem(n, n, n, 1);
    double cube = double(n) * n * n;
    int runs = cube * 4 > 1e11 ? 2 : int(1e11 / (cube * 4)) + 2;
    double ms = timeMs([&] { launch(p.gpu); }, runs < 50 ? runs : 50);
    int exact = finish(p, 64);

    double gflops = 2.0 * cube / ms / 1e6;
    double reads = double(tm + tn) / (tm * tn);
    double fetched = cube * (1.0 / bm + 1.0 / bn) * 4.0 / (1 << 20);

    std::printf("  %5d  %-9s %3d x %3d %9.3f %9.1f", n, name, bm, bn, ms, gflops);
    std::printf(" %10.2f %12.1f %6d of 64\n", reads, fetched, exact);
    return exact == 64;
}

template <int BM, int BN, int BK, int TM, int TN>
bool runConfig(int n, const char* name)
{
    return runTimed(n, name, BM, BN, TM, TN, launch<BM, BN, BK, TM, TN>);
}

// About 0.5 s of 1024 x 1024 products: the GPU raises its clocks under sustained load.
void warmUp()
{
    Problem p = makeProblem(1024, 1024, 1024, 1);
    double start = nowMs();
    while (nowMs() - start < 500.0) {
        launch<64, 64, 16, 4, 4>(p.gpu);
        check(cudaDeviceSynchronize());
    }
    finish(p, 1);
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);

    bool passed = runExample();
    passed &= runShape(1000, 1000, 1000);
    warmUp();

    std::printf("C = A B, n x n, 256 threads per block\n");
    std::printf("      n  thread        block        ms   GFLOP/s");
    std::printf("  reads/FMA  fetched MiB   exact\n");
    for (int n : {1024, 2048, 4096}) {
        passed &= runTimed(n, "Ch. 17", 16, 16, 1, 1, launchTiled16);
        passed &= runConfig<16, 16, 16, 1, 1>(n, "1 x 1");
        passed &= runConfig<32, 16, 16, 2, 1>(n, "2 x 1");
        passed &= runConfig<32, 32, 16, 2, 2>(n, "2 x 2");
        passed &= runConfig<64, 64, 16, 4, 4>(n, "4 x 4");
        passed &= runConfig<128, 128, 8, 8, 8>(n, "8 x 8");
    }

    std::printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
