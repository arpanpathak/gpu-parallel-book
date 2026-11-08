// Chapter 20: tiled Jacobi steps with halo cells in shared memory.
// Build: nvcc -O3 -arch=native stencil_tiled.cu -o stencil_tiled

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

using Grid = std::vector<float>;

const int BLOCK_WIDTH = 32;                   // cells computed by a block, across
const int BLOCK_HEIGHT = 8;                   // cells computed by a block, down
const int TILE_WIDTH = BLOCK_WIDTH + 2;       // the block's cells and the halo
const int TILE_HEIGHT = BLOCK_HEIGHT + 2;
const int TEMPORAL_TILE = 32;                 // temporal blocking: a 32 x 32 tile
const int STEPS = 4;                          // temporal blocking: Jacobi steps per load
const int OWN_WIDTH = TEMPORAL_TILE - 2 * STEPS;  // temporal blocking: cells written
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
    int row = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;
    int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    if (interior) {
        int center = row * width + col;
        float up = old[center - width];
        float down = old[center + width];
        float sum = up + down + old[center - 1] + old[center + 1];
        next[center] = 0.25f * sum;
    }
}

// 1 Jacobi step from a 34 x 10 tile; 32 x 8 threads load it, some 2 or 4 cells each.
__global__ void jacobiTiled(const float* __restrict__ old, float* __restrict__ next,
                            int width, int height)
{
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];
    int top = blockIdx.y * BLOCK_HEIGHT;
    int left = blockIdx.x * BLOCK_WIDTH;

    for (int slot_row = threadIdx.y; slot_row < TILE_HEIGHT; slot_row += BLOCK_HEIGHT) {
        for (int slot_col = threadIdx.x; slot_col < TILE_WIDTH; slot_col += BLOCK_WIDTH) {
            int cell_row = top + slot_row - 1;
            int cell_col = left + slot_col - 1;
            bool row_inside = cell_row >= 0 && cell_row < height;
            bool col_inside = cell_col >= 0 && cell_col < width;
            int cell = cell_row * width + cell_col;
            tile[slot_row][slot_col] = row_inside && col_inside ? old[cell] : 0.0f;
        }
    }
    __syncthreads();

    int row = top + threadIdx.y;
    int col = left + threadIdx.x;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    if (interior) {
        int center_row = threadIdx.y + 1;
        int center_col = threadIdx.x + 1;
        float up = tile[center_row - 1][center_col];
        float down = tile[center_row + 1][center_col];
        float to_left = tile[center_row][center_col - 1];
        float to_right = tile[center_row][center_col + 1];
        next[row * width + col] = 0.25f * (up + down + to_left + to_right);
    }
}

// 1 Jacobi step from a 34 x 10 tile; 34 x 10 threads load 1 cell each, 32 x 8 compute.
__global__ void jacobiHaloThreads(const float* __restrict__ old, float* __restrict__ next,
                                  int width, int height)
{
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];
    int slot_row = threadIdx.y;
    int slot_col = threadIdx.x;
    int row = blockIdx.y * BLOCK_HEIGHT + slot_row - 1;
    int col = blockIdx.x * BLOCK_WIDTH + slot_col - 1;

    bool inside = row >= 0 && row < height && col >= 0 && col < width;
    tile[slot_row][slot_col] = inside ? old[row * width + col] : 0.0f;
    __syncthreads();

    bool inner_row = slot_row >= 1 && slot_row <= BLOCK_HEIGHT;
    bool inner_col = slot_col >= 1 && slot_col <= BLOCK_WIDTH;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    if (inner_row && inner_col && interior) {
        float up = tile[slot_row - 1][slot_col];
        float down = tile[slot_row + 1][slot_col];
        float to_left = tile[slot_row][slot_col - 1];
        float to_right = tile[slot_row][slot_col + 1];
        next[row * width + col] = 0.25f * (up + down + to_left + to_right);
    }
}

// STEPS Jacobi steps from 1 load: a 32 x 32 tile, ping-pong in shared memory.
__global__ void jacobiTemporal(const float* __restrict__ old, float* __restrict__ next,
                               int width, int height)
{
    __shared__ float tiles[2][TEMPORAL_TILE][TEMPORAL_TILE];
    int slot_row = threadIdx.y;
    int slot_col = threadIdx.x;
    int row = blockIdx.y * OWN_WIDTH + slot_row - STEPS;
    int col = blockIdx.x * OWN_WIDTH + slot_col - STEPS;
    bool inside = row >= 0 && row < height && col >= 0 && col < width;
    bool interior = row >= 1 && row + 1 < height && col >= 1 && col + 1 < width;

    tiles[0][slot_row][slot_col] = inside ? old[row * width + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int step = 1; step <= STEPS; ++step) {
        int src = (step - 1) % 2;
        int dst = step % 2;
        int last = TEMPORAL_TILE - step;
        bool valid_row = slot_row >= step && slot_row < last;
        bool valid_col = slot_col >= step && slot_col < last;

        if (valid_row && valid_col) {
            float up = tiles[src][slot_row - 1][slot_col];
            float down = tiles[src][slot_row + 1][slot_col];
            float to_left = tiles[src][slot_row][slot_col - 1];
            float to_right = tiles[src][slot_row][slot_col + 1];
            float mean = 0.25f * (up + down + to_left + to_right);
            float same = tiles[src][slot_row][slot_col];
            tiles[dst][slot_row][slot_col] = interior ? mean : same;
        }
        __syncthreads();
    }

    int end = TEMPORAL_TILE - STEPS;
    bool owned = slot_col >= STEPS && slot_col < end && slot_row >= STEPS && slot_row < end;
    if (owned && interior) {
        next[row * width + col] = tiles[STEPS % 2][slot_row][slot_col];
    }
}

// A copy: 1 read and 1 write per cell, the least traffic a step can have.
__global__ void copy(const float* __restrict__ in, float* __restrict__ out, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index];
    }
}

int stepsPerLaunch(Kind kind)
{
    return kind == TEMPORAL ? STEPS : 1;
}

// 1 launch of `kind` on a plate of width x height cells, reading `old`, writing `next`.
void launch(Kind kind, const float* old, float* next, int width, int height)
{
    int block_columns = (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    int block_rows = (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    dim3 blocks(block_columns, block_rows);
    dim3 squares((width + OWN_WIDTH - 1) / OWN_WIDTH, (height + OWN_WIDTH - 1) / OWN_WIDTH);
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 tile_block(TILE_WIDTH, TILE_HEIGHT);
    dim3 square(TEMPORAL_TILE, TEMPORAL_TILE);

    switch (kind) {
    case PLAIN:
        jacobi<<<blocks, block>>>(old, next, width, height);
        break;
    case TILED:
        jacobiTiled<<<blocks, block>>>(old, next, width, height);
        break;
    case HALO_THREADS:
        jacobiHaloThreads<<<blocks, tile_block>>>(old, next, width, height);
        break;
    case TEMPORAL:
        jacobiTemporal<<<squares, square>>>(old, next, width, height);
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
    float *current, *next;
    check(cudaMalloc(&current, bytes));
    check(cudaMalloc(&next, bytes));
    check(cudaMemcpy(current, start.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(next, start.data(), bytes, cudaMemcpyHostToDevice));

    for (int launches = 0; launches < steps / stepsPerLaunch(kind); ++launches) {
        launch(kind, current, next, width, height);
        std::swap(current, next);
    }
    check(cudaGetLastError());

    Grid result(start.size());
    check(cudaMemcpy(result.data(), current, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(current));
    check(cudaFree(next));
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
    for (int timed = 0; timed < runs; ++timed) {
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
    int width = 6;
    int height = 6;
    Grid start = initialGrid(width, height);

    std::printf("1. 6 x 6 plate, top edge at %.0f\n", HOT);
    printGrid("after step 2, tiled", run(TILED, start, width, height, 2), width);

    bool smallSame = true;
    Grid reference = run(PLAIN, start, width, height, 4);
    for (int index = 0; index < 3; ++index) {
        bool same = run(kinds[index], start, width, height, 4) == reference;
        std::printf("  %-24s same bits as the plain kernel", names[index]);
        std::printf(" after 4 steps: %s\n", same ? "true" : "false");
        smallSame = smallSame && same;
    }

    // 2. A 4096 x 4096 plate: time per step and effective bandwidth.
    width = 4096;
    height = 4096;
    int cells = width * height;
    start = initialGrid(width, height);
    size_t bytes = start.size() * sizeof(float);
    float *current, *next;
    check(cudaMalloc(&current, bytes));
    check(cudaMalloc(&next, bytes));
    check(cudaMemcpy(current, start.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(next, start.data(), bytes, cudaMemcpyHostToDevice));

    auto gb = [&](double ms) { return 2.0 * cells * 4 / 1e9 / (ms / 1e3); };
    std::printf("2. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)\n");

    double ms = timeMs([&] { copy<<<cells / 256, 256>>>(current, next, cells); }, 50);
    std::printf("  %-24s %6.3f ms   %6.1f GB/s\n", "copy kernel", ms, gb(ms));

    Kind allKinds[] = {PLAIN, TILED, HALO_THREADS, TEMPORAL};
    const char* allNames[] = {"plain", names[0], names[1], names[2]};
    for (int index = 0; index < 4; ++index) {
        Kind kind = allKinds[index];
        auto twoLaunches = [&] {
            launch(kind, current, next, width, height);
            launch(kind, next, current, width, height);
        };
        ms = timeMs(twoLaunches, 50) / (2 * stepsPerLaunch(kind));
        std::printf("  %-24s %6.3f ms   %6.1f GB/s\n", allNames[index], ms, gb(ms));
    }
    check(cudaFree(current));
    check(cudaFree(next));

    bool bigSame = true;
    reference = run(PLAIN, start, width, height, 8);
    for (int index = 0; index < 3; ++index) {
        bool same = run(kinds[index], start, width, height, 8) == reference;
        std::printf("  %-24s same bits as the plain kernel", names[index]);
        std::printf(" after 8 steps: %s\n", same ? "true" : "false");
        bigSame = bigSame && same;
    }

    bool passed = smallSame && bigSame;
    std::printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
