// Chapter 19: Jacobi iteration for the Laplace equation, 1 thread per cell.
// Build: nvcc -O3 -arch=native stencil.cu -o stencil

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

using Grid = std::vector<float>;

const int BX = 32;                            // block width: 1 warp along a row of the grid
const int BY = 8;                             // block height: 8 warps, 256 threads
const float HOT = 100.0f;                     // the top edge; the others are held at 0

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// 1 Jacobi step: each interior cell of `next` gets the mean of its 4 neighbours in `old`.
__global__ void jacobi(const float* __restrict__ old, float* __restrict__ next,
                       int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    if (interior) {
        int c = row * width + col;
        float sum = old[c - width] + old[c + width] + old[c - 1] + old[c + 1];
        next[c] = 0.25f * sum;
    }
}

// A copy: 1 read and 1 write per cell, the least traffic a step can have.
__global__ void copy(const float* __restrict__ in, float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

// A plate of width x height cells: the top row at HOT, every other cell at 0.
Grid initialGrid(int width, int height)
{
    Grid grid(size_t(width) * height, 0.0f);
    for (int col = 0; col < width; ++col) {
        grid[col] = HOT;
    }
    return grid;
}

// 1 Jacobi step on the CPU, the reference for the GPU kernel.
void cpuStep(const Grid& old, Grid& next, int width, int height)
{
    for (int row = 1; row + 1 < height; ++row) {
        for (int col = 1; col + 1 < width; ++col) {
            int c = row * width + col;
            float sum = old[c - width] + old[c + width] + old[c - 1] + old[c + 1];
            next[c] = 0.25f * sum;
        }
    }
}

// The root-mean-square difference between 2 grids, over all cells.
double rmsChange(const Grid& next, const Grid& old)
{
    double squares = 0.0;
    for (size_t i = 0; i < next.size(); ++i) {
        double d = next[i] - old[i];
        squares += d * d;
    }
    return std::sqrt(squares / next.size());
}

void printGrid(const char* label, const Grid& grid, int width)
{
    std::printf("  %s\n", label);
    for (size_t i = 0; i < grid.size(); ++i) {
        const char* gap = i % width == 0 ? "    " : " ";
        const char* end = i % width == size_t(width - 1) ? "\n" : "";
        std::printf("%s%7.2f%s", gap, grid[i], end);
    }
}

Grid toHost(const float* device, int width, int height)
{
    Grid grid(size_t(width) * height);
    size_t bytes = grid.size() * sizeof(float);
    check(cudaMemcpy(grid.data(), device, bytes, cudaMemcpyDeviceToHost));
    return grid;
}

// 2 device grids, both filled with the initial plate (boundary included).
std::pair<float*, float*> deviceGrids(const Grid& start)
{
    size_t bytes = start.size() * sizeof(float);
    float *a, *b;
    check(cudaMalloc(&a, bytes));
    check(cudaMalloc(&b, bytes));
    check(cudaMemcpy(a, start.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, start.data(), bytes, cudaMemcpyHostToDevice));
    return {a, b};
}

// Enough 32 x 8 blocks to cover a grid of width x height cells, 1 thread per cell.
dim3 blocksFor(int width, int height)
{
    return dim3((width + BX - 1) / BX, (height + BY - 1) / BY);
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Average time of `runs` calls of `launch`, in milliseconds, after 0.4 s of warm-up.
template <typename F>
double timeMs(F launch, int runs)
{
    double warm = nowMs();
    while (nowMs() - warm < 400.0) {
        launch();
        check(cudaDeviceSynchronize());
    }

    double t0 = nowMs();
    for (int r = 0; r < runs; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    return (nowMs() - t0) / runs;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    dim3 block(BX, BY);

    // 1. The 6 x 6 plate of the text, small enough to check by hand.
    int w = 6;
    int h = 6;
    Grid start = initialGrid(w, h);
    float* a;
    float* b;
    std::tie(a, b) = deviceGrids(start);

    std::printf("1. 6 x 6 plate, top edge at %.0f\n", HOT);
    std::vector<double> changes;
    for (int step = 1; step <= 100; ++step) {
        jacobi<<<blocksFor(w, h), block>>>(a, b, w, h);
        std::swap(a, b);

        Grid next = toHost(a, w, h);
        Grid old = toHost(b, w, h);
        changes.push_back(rmsChange(next, old));
        if (step <= 2 || step == 100) {
            char label[32];
            std::snprintf(label, sizeof label, "after step %d", step);
            printGrid(label, next, w);
        }
    }

    std::printf("  RMS change per step, and its ratio to the step before\n");
    std::printf("    step %3d   %9.2e\n", 1, changes[0]);
    for (int step : {2, 3, 4, 10, 20, 40}) {
        double ratio = changes[step - 1] / changes[step - 2];
        std::printf("    step %3d   %9.2e   ratio %.4f\n", step, changes[step - 1], ratio);
    }

    Grid cpuA = start;
    Grid cpuB = start;
    for (int step = 0; step < 100; ++step) {
        cpuStep(cpuA, cpuB, w, h);
        std::swap(cpuA, cpuB);
    }
    bool smallSame = toHost(a, w, h) == cpuA;
    const char* answer = smallSame ? "true" : "false";
    std::printf("  same bits as the CPU after 100 steps: %s\n", answer);
    check(cudaFree(a));
    check(cudaFree(b));

    // 2. A 64 x 64 plate, iterated until the RMS change per step is below 1e-6.
    w = 64;
    h = 64;
    start = initialGrid(w, h);
    std::tie(a, b) = deviceGrids(start);

    std::printf("2. 64 x 64 plate until the RMS change per step is below 1e-6\n");
    int steps = 0;
    while (true) {
        jacobi<<<blocksFor(w, h), block>>>(a, b, w, h);
        std::swap(a, b);
        steps += 1;

        if (steps % 1000 == 0) {
            double rms = rmsChange(toHost(a, w, h), toHost(b, w, h));
            std::printf("  step %5d   RMS change %.2e\n", steps, rms);
            if (rms < 1e-6) {
                break;
            }
        }
    }

    Grid field = toHost(a, w, h);
    cpuA = start;
    cpuB = start;
    for (int step = 0; step < steps; ++step) {
        cpuStep(cpuA, cpuB, w, h);
        std::swap(cpuA, cpuB);
    }
    float diff = 0.0f;
    for (size_t i = 0; i < field.size(); ++i) {
        diff = std::fmax(diff, std::fabs(field[i] - cpuA[i]));
    }
    std::printf("  largest difference from the CPU after %d steps: %.2e\n", steps, diff);
    check(cudaFree(a));
    check(cudaFree(b));

    // 3. A 4096 x 4096 plate: time per step and effective bandwidth.
    w = 4096;
    h = 4096;
    int n = w * h;
    start = initialGrid(w, h);
    std::tie(a, b) = deviceGrids(start);

    auto copyStep = [&] { copy<<<n / 256, 256>>>(a, b, n); };
    auto twoSteps = [&] {
        jacobi<<<blocksFor(w, h), block>>>(a, b, w, h);
        jacobi<<<blocksFor(w, h), block>>>(b, a, w, h);
    };
    double msCopy = timeMs(copyStep, 50);
    double msJacobi = timeMs(twoSteps, 50) / 2.0;

    auto gb = [&](double ms) { return 2.0 * n * 4 / 1e9 / (ms / 1e3); };
    std::printf("3. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)\n");
    std::printf("  copy kernel        %6.3f ms   %6.1f GB/s\n", msCopy, gb(msCopy));
    std::printf("  jacobi             %6.3f ms   %6.1f GB/s\n", msJacobi, gb(msJacobi));
    check(cudaFree(a));
    check(cudaFree(b));

    bool passed = smallSame && diff == 0.0f;
    std::printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
