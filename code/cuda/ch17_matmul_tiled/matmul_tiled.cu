// Chapter 17: tiled matrix multiplication in shared memory, against the direct kernel.
// Build: nvcc -O3 -arch=native matmul_tiled.cu -o matmul_tiled

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

// C (m x n) = A (m x k) B (k x n), all row-major. 1 thread computes 1 entry of C and
// fetches all its operands from global memory.
__global__ void matmulDirect(int m, int n, int k, const float* a, const float* b, float* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = sum;
}

// The same product with T x T tiles in shared memory. A block of T x T threads computes
// a T x T square of C in ceil(k / T) phases.
template <int T>
__global__ void matmulTiled(int m, int n, int k, const float* a, const float* b, float* c)
{
    __shared__ float tile_a[T][T];
    __shared__ float tile_b[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    float sum = 0.0f;
    int phases = (k + T - 1) / T;
    for (int phase = 0; phase < phases; ++phase) {
        int a_col = phase * T + threadIdx.x;
        int b_row = phase * T + threadIdx.y;
        float a_value = row < m && a_col < k ? a[row * k + a_col] : 0.0f;
        float b_value = b_row < k && col < n ? b[b_row * n + col] : 0.0f;
        tile_a[threadIdx.y][threadIdx.x] = a_value;
        tile_b[threadIdx.y][threadIdx.x] = b_value;
        __syncthreads();

        for (int offset = 0; offset < T; ++offset) {
            sum += tile_a[threadIdx.y][offset] * tile_b[offset][threadIdx.x];
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

template <int T>
void launchTiled(const Product& p)
{
    dim3 grid((p.n + T - 1) / T, (p.m + T - 1) / T);
    matmulTiled<T><<<grid, dim3(T, T)>>>(p.m, p.n, p.k, p.a, p.b, p.c);
}

void launchDirect(const Product& p)
{
    dim3 grid((p.n + 15) / 16, (p.m + 15) / 16);
    matmulDirect<<<grid, dim3(16, 16)>>>(p.m, p.n, p.k, p.a, p.b, p.c);
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
    for (int sample = 0; sample < count; ++sample) {
        int row = count == m * n ? sample / n : sample * 977 % m;
        int col = count == m * n ? sample % n : (sample * 1231 + 7) % n;

        double sum = 0.0;
        for (int inner = 0; inner < k; ++inner) {
            sum += double(a[row * k + inner]) * double(b[inner * n + col]);
        }
        exact += double(c[row * n + col]) == sum;
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

// The 4 x 4 example with T = 2: 4 blocks of 2 x 2 threads, 2 phases each.
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

    launchTiled<2>(p);
    check(cudaGetLastError());
    check(cudaMemcpy(c.data(), p.c, 64, cudaMemcpyDeviceToHost));

    check(cudaFree(p.a));
    check(cudaFree(p.b));
    check(cudaFree(p.c));

    std::printf("C = A B on the 4 x 4 example, tiles of 2 x 2, 2 phases\n");
    for (int i = 0; i < 4; ++i) {
        const float* row = &c[i * 4];
        std::printf("  %4.0f %4.0f %4.0f %4.0f\n", row[0], row[1], row[2], row[3]);
    }
    bool same = c == want;
    std::printf("equal to the product computed by hand: %s\n\n", same ? "yes" : "no");
    return same;
}

// Sizes that are not multiples of T: every entry of C is checked.
template <int T>
bool runShape(int m, int n, int k)
{
    Problem p = makeProblem(m, n, k, 7);
    launchTiled<T>(p.gpu);
    check(cudaGetLastError());

    int exact = finish(p, m * n);
    std::printf("  m = %4d, k = %4d, n = %4d, T = %2d:", m, k, n, T);
    std::printf(" %7d of %7d entries exact\n", exact, m * n);
    return exact == m * n;
}

// Time 1 kernel on n x n matrices and print 1 line of the table.
template <typename F>
bool runTimed(const char* name, int n, int tile, F launch)
{
    Problem p = makeProblem(n, n, n, 1);
    double cube = double(n) * n * n;
    int runs = cube * 4 > 1e11 ? 2 : int(1e11 / (cube * 4)) + 2;
    double ms = timeMs([&] { launch(p.gpu); }, runs < 50 ? runs : 50);
    int exact = finish(p, 64);

    double gflops = 2.0 * cube / ms / 1e6;
    double fetched = 2.0 * cube / tile * 4.0;
    double mib = fetched / (1 << 20);
    double rate = fetched / ms / 1e6;

    std::printf("  %5d  %-10s %9.3f %9.1f %12.1f %11.1f", n, name, ms, gflops, mib, rate);
    std::printf(" %6d of 64\n", exact);
    return exact == 64;
}

// About 0.5 s of 1024 x 1024 products: the GPU raises its clocks under sustained load.
void warmUp()
{
    Problem p = makeProblem(1024, 1024, 1024, 1);
    double start = nowMs();
    while (nowMs() - start < 500.0) {
        launchTiled<16>(p.gpu);
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

    std::printf("Sizes that are not multiples of the tile width\n");
    passed &= runShape<2>(3, 3, 3);
    passed &= runShape<16>(128, 96, 64);
    passed &= runShape<16>(1000, 1000, 1000);
    std::printf("\n");

    warmUp();

    std::printf("C = A B, n x n; fetched: the modelled reads from global memory\n");
    std::printf("      n  kernel            ms   GFLOP/s");
    std::printf("  fetched MiB  fetch GB/s   exact\n");
    for (int n : {512, 1024, 2048, 4096}) {
        passed &= runTimed("direct", n, 1, launchDirect);
        passed &= runTimed("T = 8", n, 8, launchTiled<8>);
        passed &= runTimed("T = 16", n, 16, launchTiled<16>);
        passed &= runTimed("T = 32", n, 32, launchTiled<32>);
    }

    std::printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
