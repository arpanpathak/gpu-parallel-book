// Chapter 20: tiled Jacobi steps with halo cells in shared memory.
// Build: nvcc -O3 -arch=native stencil_tiled.cu -o stencil_tiled

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

using Grid = std::vector<float>;

const int BX = 32;                            // cells computed by a block, across
const int BY = 8;                             // cells computed by a block, down
const int TW = BX + 2;                        // tile width: the block's cells and the halo
const int TH = BY + 2;                        // tile height
const int TT = 32;                            // temporal blocking: a 32 x 32 tile
const int STEPS = 4;                          // temporal blocking: Jacobi steps per load
const int OUT = TT - 2 * STEPS;               // temporal blocking: cells written, across
const float HOT = 100.0f;                     // the top edge; the others are held at 0

enum Kind { PLAIN, TILED, HALO_THREADS, TEMPORAL };

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// 1 Jacobi step, 1 thread per cell, every neighbor read from global memory.
__global__ void jacobi(const float* __restrict__ old, float* __restrict__ next,
                       int width, int height)
{
    int row = blockIdx.y * BY + threadIdx.y;
    int col = blockIdx.x * BX + threadIdx.x;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    if (interior) {
        int c = row * width + col;
        float sum = old[c - width] + old[c + width] + old[c - 1] + old[c + 1];
        next[c] = 0.25f * sum;
    }
}

// 1 Jacobi step from a 34 x 10 tile; 32 x 8 threads load it, some 2 or 4 cells each.
__global__ void jacobiTiled(const float* __restrict__ old, float* __restrict__ next,
                            int width, int height)
{
    __shared__ float tile[TH][TW];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int top = blockIdx.y * BY;
    int left = blockIdx.x * BX;

    for (int i = ty; i < TH; i += BY) {
        for (int j = tx; j < TW; j += BX) {
            int r = top + i - 1;
            int c = left + j - 1;
            bool inside = r >= 0 && r < height && c >= 0 && c < width;
            tile[i][j] = inside ? old[r * width + c] : 0.0f;
        }
    }
    __syncthreads();

    int row = top + ty;
    int col = left + tx;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    if (interior) {
        float up = tile[ty][tx + 1];
        float down = tile[ty + 2][tx + 1];
        float sum = up + down + tile[ty + 1][tx] + tile[ty + 1][tx + 2];
        next[row * width + col] = 0.25f * sum;
    }
}

// 1 Jacobi step from a 34 x 10 tile; 34 x 10 threads load 1 cell each, 32 x 8 compute.
__global__ void jacobiHaloThreads(const float* __restrict__ old, float* __restrict__ next,
                                  int width, int height)
{
    __shared__ float tile[TH][TW];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int r = blockIdx.y * BY + ty - 1;
    int c = blockIdx.x * BX + tx - 1;

    bool inside = r >= 0 && r < height && c >= 0 && c < width;
    tile[ty][tx] = inside ? old[r * width + c] : 0.0f;
    __syncthreads();

    bool inner = tx >= 1 && tx <= BX && ty >= 1 && ty <= BY;
    bool interior = r >= 1 && r + 1 < height && c >= 1 && c + 1 < width;

    if (inner && interior) {
        float up = tile[ty - 1][tx];
        float down = tile[ty + 1][tx];
        float sum = up + down + tile[ty][tx - 1] + tile[ty][tx + 1];
        next[r * width + c] = 0.25f * sum;
    }
}

// STEPS Jacobi steps from 1 load: a 32 x 32 tile, ping-pong in shared memory.
__global__ void jacobiTemporal(const float* __restrict__ old, float* __restrict__ next,
                               int width, int height)
{
    __shared__ float tiles[2][TT][TT];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int r = blockIdx.y * OUT + ty - STEPS;
    int c = blockIdx.x * OUT + tx - STEPS;
    bool inside = r >= 0 && r < height && c >= 0 && c < width;
    bool interior = r >= 1 && r + 1 < height && c >= 1 && c + 1 < width;

    tiles[0][ty][tx] = inside ? old[r * width + c] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int step = 1; step <= STEPS; ++step) {
        int src = (step - 1) % 2;
        int dst = step % 2;
        bool valid = tx >= step && tx < TT - step && ty >= step && ty < TT - step;

        if (valid) {
            float up = tiles[src][ty - 1][tx];
            float down = tiles[src][ty + 1][tx];
            float sum = up + down + tiles[src][ty][tx - 1] + tiles[src][ty][tx + 1];
            tiles[dst][ty][tx] = interior ? 0.25f * sum : tiles[src][ty][tx];
        }
        __syncthreads();
    }

    bool owned = tx >= STEPS && tx < TT - STEPS && ty >= STEPS && ty < TT - STEPS;
    if (owned && interior) {
        next[r * width + c] = tiles[STEPS % 2][ty][tx];
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

int stepsPerLaunch(Kind kind)
{
    return kind == TEMPORAL ? STEPS : 1;
}

// 1 launch of `kind` on a plate of width x height cells, reading `old`, writing `next`.
void launch(Kind kind, const float* old, float* next, int width, int height)
{
    dim3 blocks((width + BX - 1) / BX, (height + BY - 1) / BY);
    dim3 squares((width + OUT - 1) / OUT, (height + OUT - 1) / OUT);

    switch (kind) {
    case PLAIN:
        jacobi<<<blocks, dim3(BX, BY)>>>(old, next, width, height);
        break;
    case TILED:
        jacobiTiled<<<blocks, dim3(BX, BY)>>>(old, next, width, height);
        break;
    case HALO_THREADS:
        jacobiHaloThreads<<<blocks, dim3(TW, TH)>>>(old, next, width, height);
        break;
    case TEMPORAL:
        jacobiTemporal<<<squares, dim3(TT, TT)>>>(old, next, width, height);
        break;
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

// The plate after `steps` Jacobi steps of `kind`, starting from `start`.
Grid run(Kind kind, const Grid& start, int width, int height, int steps)
{
    size_t bytes = start.size() * sizeof(float);
    float *a, *b;
    check(cudaMalloc(&a, bytes));
    check(cudaMalloc(&b, bytes));
    check(cudaMemcpy(a, start.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, start.data(), bytes, cudaMemcpyHostToDevice));

    for (int k = 0; k < steps / stepsPerLaunch(kind); ++k) {
        launch(kind, a, b, width, height);
        std::swap(a, b);
    }
    check(cudaGetLastError());

    Grid result(start.size());
    check(cudaMemcpy(result.data(), a, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(a));
    check(cudaFree(b));
    return result;
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

    Kind kinds[] = {TILED, HALO_THREADS, TEMPORAL};
    const char* names[] = {"loop loader", "extra threads", "4 steps per load"};

    // 1. The 6 x 6 plate of the text: every tiled kernel against the plain kernel.
    int w = 6;
    int h = 6;
    Grid start = initialGrid(w, h);

    std::printf("1. 6 x 6 plate, top edge at %.0f\n", HOT);
    printGrid("after step 2, tiled", run(TILED, start, w, h, 2), w);

    bool smallSame = true;
    Grid reference = run(PLAIN, start, w, h, 4);
    for (int k = 0; k < 3; ++k) {
        bool same = run(kinds[k], start, w, h, 4) == reference;
        std::printf("  %-24s same bits as the plain kernel", names[k]);
        std::printf(" after 4 steps: %s\n", same ? "true" : "false");
        smallSame = smallSame && same;
    }

    // 2. A 4096 x 4096 plate: time per step and effective bandwidth.
    w = 4096;
    h = 4096;
    int n = w * h;
    start = initialGrid(w, h);
    size_t bytes = start.size() * sizeof(float);
    float *a, *b;
    check(cudaMalloc(&a, bytes));
    check(cudaMalloc(&b, bytes));
    check(cudaMemcpy(a, start.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, start.data(), bytes, cudaMemcpyHostToDevice));

    auto gb = [&](double ms) { return 2.0 * n * 4 / 1e9 / (ms / 1e3); };
    std::printf("2. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)\n");

    double ms = timeMs([&] { copy<<<n / 256, 256>>>(a, b, n); }, 50);
    std::printf("  %-24s %6.3f ms   %6.1f GB/s\n", "copy kernel", ms, gb(ms));

    Kind allKinds[] = {PLAIN, TILED, HALO_THREADS, TEMPORAL};
    const char* allNames[] = {"plain", names[0], names[1], names[2]};
    for (int k = 0; k < 4; ++k) {
        Kind kind = allKinds[k];
        auto twoLaunches = [&] {
            launch(kind, a, b, w, h);
            launch(kind, b, a, w, h);
        };
        ms = timeMs(twoLaunches, 50) / (2 * stepsPerLaunch(kind));
        std::printf("  %-24s %6.3f ms   %6.1f GB/s\n", allNames[k], ms, gb(ms));
    }
    check(cudaFree(a));
    check(cudaFree(b));

    bool bigSame = true;
    reference = run(PLAIN, start, w, h, 8);
    for (int k = 0; k < 3; ++k) {
        bool same = run(kinds[k], start, w, h, 8) == reference;
        std::printf("  %-24s same bits as the plain kernel", names[k]);
        std::printf(" after 8 steps: %s\n", same ? "true" : "false");
        bigSame = bigSame && same;
    }

    bool passed = smallSame && bigSame;
    std::printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
