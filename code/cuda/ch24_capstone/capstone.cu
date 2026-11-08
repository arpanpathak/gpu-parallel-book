// Chapter 24: an image pipeline. Gray conversion, histogram equalization, a separable
// Gaussian blur and Sobel edges, checked pixel by pixel against the CPU, then a batch of
// images on 1, 2 and 3 streams.
// Build: nvcc -O3 -arch=native -Xcompiler -ffp-contract=off capstone.cu -o capstone

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

const int W = 1920;
const int H = 1080;
const int PIXELS = W * H;                     // 2,073,600
const int BATCH = 128;

const int ROW_T = 128;                        // outputs per block of the row pass
const int COL_W = 32;                         // columns per block of the column pass
const int COL_H = 8;                          // rows per block of the column pass

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

// Position slot - 2 clamped to [0, size): slot `slot` of a window that starts 2 before.
__host__ __device__ int tap(int slot, int size)
{
    int position = slot < 2 ? 0 : slot - 2;
    return position < size ? position : size - 1;
}

__device__ unsigned warpScan(unsigned value)
{
    int lane = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned left = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset) {
            value += left;
        }
    }
    return value;
}

// 1 thread per pixel. BT.601 weights in 8-bit fixed point: 77 + 150 + 29 = 256.
__global__ void grayscale(const uint8_t* rgb, uint8_t* gray)
{
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < PIXELS) {
        unsigned red = rgb[3 * pixel];
        unsigned green = rgb[3 * pixel + 1];
        unsigned blue = rgb[3 * pixel + 2];
        gray[pixel] = (77 * red + 150 * green + 29 * blue + 128) >> 8;
    }
}

// Counts gray levels in shared bins, then adds each block's bins to the global ones.
__global__ void histogram(const uint8_t* gray, unsigned* bins)
{
    __shared__ unsigned local[256];

    local[threadIdx.x] = 0;
    __syncthreads();

    int stride = gridDim.x * 256;
    for (int pixel = blockIdx.x * 256 + threadIdx.x; pixel < PIXELS; pixel += stride) {
        atomicAdd(&local[gray[pixel]], 1u);
    }
    __syncthreads();

    if (local[threadIdx.x] > 0) {
        atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
    }
}

// 1 block of 256 threads, 1 thread per gray level: the cumulative count of the level,
// then its new level. Each bin is read and reset to 0 for the next image.
__global__ void equaliseTable(unsigned* bins, uint8_t* table)
{
    __shared__ unsigned warps[8];
    __shared__ unsigned first;
    int level = threadIdx.x;
    int lane = level % 32;
    int warp = level / 32;

    unsigned count = atomicExch(&bins[level], 0u);
    unsigned inclusive = warpScan(count);
    if (lane == 31) {
        warps[warp] = inclusive;
    }
    if (level == 0) {
        first = 0xffffffffu;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned scannedTotal = warpScan(lane < 8 ? warps[lane] : 0);
        if (lane < 8) {
            warps[lane] = scannedTotal;
        }
    }
    __syncthreads();

    unsigned cdf = inclusive + (warp > 0 ? warps[warp - 1] : 0);
    if (count > 0) {
        atomicMin(&first, cdf);
    }
    __syncthreads();

    uint64_t span = warps[7] - first;
    uint64_t above = cdf > first ? cdf - first : 0;
    uint64_t newLevel = span == 0 ? level : (above * 255 + span / 2) / span;
    table[level] = newLevel;
}

// 1 thread per pixel: the new level of each pixel, looked up in the table.
__global__ void applyTable(const uint8_t* gray, const uint8_t* table, uint8_t* out)
{
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < PIXELS) {
        out[pixel] = table[gray[pixel]];
    }
}

// Row pass, weights 1 4 6 4 1. Slot `slot` of the tile holds pixel left + slot - 2,
// clamped: 2 halo slots on the left, 128 own pixels, 2 halo slots on the right.
__global__ void blurRows(const uint8_t* src, uint16_t* out)
{
    __shared__ unsigned tile[ROW_T + 4];
    int left = blockIdx.x * ROW_T;
    const uint8_t* row = src + blockIdx.y * W;

    tile[threadIdx.x + 2] = row[tap(left + threadIdx.x + 2, W)];
    if (threadIdx.x < 2) {
        tile[threadIdx.x] = row[tap(left + threadIdx.x, W)];
        tile[ROW_T + 2 + threadIdx.x] = row[tap(left + ROW_T + 2 + threadIdx.x, W)];
    }
    __syncthreads();

    const unsigned* window = tile + threadIdx.x;
    unsigned sum = window[0] + 4 * window[1] + 6 * window[2] + 4 * window[3] + window[4];
    out[blockIdx.y * W + left + threadIdx.x] = sum;
}

// Column pass on a tile of 32 columns and 8 + 4 rows. The 2 passes weigh by
// 16 x 16 = 256, so the result is the rounded sum shifted right by 8.
__global__ void blurCols(const uint16_t* src, uint8_t* out)
{
    __shared__ unsigned tile[COL_H + 4][COL_W];
    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;
    int x = blockIdx.x * COL_W + tileCol;
    int top = blockIdx.y * COL_H;

    tile[tileRow + 2][tileCol] = src[tap(top + tileRow + 2, H) * W + x];
    if (tileRow < 2) {
        tile[tileRow][tileCol] = src[tap(top + tileRow, H) * W + x];
        tile[COL_H + 2 + tileRow][tileCol] = src[tap(top + COL_H + 2 + tileRow, H) * W + x];
    }
    __syncthreads();

    unsigned sum = tile[tileRow][tileCol] + 4 * tile[tileRow + 1][tileCol];
    sum += 6 * tile[tileRow + 2][tileCol] + 4 * tile[tileRow + 3][tileCol];
    sum += tile[tileRow + 4][tileCol];
    out[(top + tileRow) * W + x] = (sum + 128) >> 8;
}


// Sobel gradient magnitude divided by 4, rounded and clamped to 255. The 8 neighbors
// come from global memory; neighboring threads share them through the L1 cache.
__global__ void sobel(const uint8_t* src, uint8_t* out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, W - 1);
    int yAbove = max(y - 1, 0);
    int yBelow = min(y + 1, H - 1);

    const uint8_t* upper = src + yAbove * W;
    const uint8_t* middle = src + y * W;
    const uint8_t* lower = src + yBelow * W;

    int right = upper[xRight] + 2 * middle[xRight] + lower[xRight];
    int left = upper[xLeft] + 2 * middle[xLeft] + lower[xLeft];
    int below = lower[xLeft] + 2 * lower[x] + lower[xRight];
    int above = upper[xLeft] + 2 * upper[x] + upper[xRight];
    int gx = right - left;
    int gy = below - above;

    float magnitude = sqrtf(float(gx * gx + gy * gy));
    out[y * W + x] = fminf(magnitude * 0.25f + 0.5f, 255.0f);
}

// The device buffers of 1 image in flight.
struct Buffers {
    uint8_t* rgb;
    uint8_t* gray;
    unsigned* bins;
    uint8_t* table;
    uint8_t* equal;
    uint16_t* rows;
    uint8_t* blurred;
    uint8_t* edges;
};

Buffers allocBuffers()
{
    Buffers b;
    check(cudaMalloc(&b.rgb, 3 * PIXELS));
    check(cudaMalloc(&b.gray, PIXELS));
    check(cudaMalloc(&b.bins, 256 * sizeof(unsigned)));
    check(cudaMalloc(&b.table, 256));
    check(cudaMalloc(&b.equal, PIXELS));
    check(cudaMalloc(&b.rows, PIXELS * sizeof(uint16_t)));
    check(cudaMalloc(&b.blurred, PIXELS));
    check(cudaMalloc(&b.edges, PIXELS));
    check(cudaMemset(b.bins, 0, 256 * sizeof(unsigned)));
    return b;
}

void freeBuffers(Buffers& b)
{
    cudaFree(b.rgb);
    cudaFree(b.gray);
    cudaFree(b.bins);
    cudaFree(b.table);
    cudaFree(b.equal);
    cudaFree(b.rows);
    cudaFree(b.blurred);
    cudaFree(b.edges);
}

// Name of each stage, and the bytes it must read and write per pixel.
struct Stage {
    const char* name;
    int bytes;
};

const Stage STAGES[7] = {
    {"grayscale", 4},
    {"histogram", 1},
    {"table", 0},
    {"apply table", 2},
    {"blur rows", 3},
    {"blur columns", 3},
    {"sobel", 2},
};

// Launches stage `index` of the pipeline on `stream`.
void stage(cudaStream_t stream, Buffers& buffers, int index)
{
    int line = PIXELS / 256;
    dim3 rows(W / ROW_T, H);
    dim3 tiles(W / COL_W, H / COL_H);
    dim3 tile(COL_W, COL_H);

    switch (index) {
    case 0:
        grayscale<<<line, 256, 0, stream>>>(buffers.rgb, buffers.gray);
        break;
    case 1:
        histogram<<<64, 256, 0, stream>>>(buffers.gray, buffers.bins);
        break;
    case 2:
        equaliseTable<<<1, 256, 0, stream>>>(buffers.bins, buffers.table);
        break;
    case 3:
        applyTable<<<line, 256, 0, stream>>>(buffers.gray, buffers.table, buffers.equal);
        break;
    case 4:
        blurRows<<<rows, ROW_T, 0, stream>>>(buffers.equal, buffers.rows);
        break;
    case 5:
        blurCols<<<tiles, tile, 0, stream>>>(buffers.rows, buffers.blurred);
        break;
    default:
        sobel<<<tiles, tile, 0, stream>>>(buffers.blurred, buffers.edges);
    }
}

// All 7 stages of 1 image, in order, on `stream`.
void process(cudaStream_t stream, Buffers& buffers)
{
    for (int index = 0; index < 7; ++index) {
        stage(stream, buffers, index);
    }
}

// Average time of 100 runs of `launch`, in milliseconds, after 1 warm-up run.
template <typename F>
double timeMs(cudaStream_t stream, F launch)
{
    launch();
    check(cudaStreamSynchronize(stream));

    double t0 = nowMs();
    for (int timed = 0; timed < 100; ++timed) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    return (nowMs() - t0) / 100;
}

// Uploads, processes and downloads `count` images, image `image` on stream image % n.
// Returns images per second.
double batch(cudaStream_t* streams, Buffers* lanes, int n, uint8_t** inputs,
             uint8_t** outputs, int count)
{
    double t0 = nowMs();

    for (int image = 0; image < count; ++image) {
        int slot = image % n;
        cudaStream_t stream = streams[slot];
        Buffers& buffers = lanes[slot];

        const uint8_t* input = inputs[image % 4];
        uint8_t* output = outputs[slot];

        size_t bytes = 3 * PIXELS;
        check(cudaMemcpyAsync(buffers.rgb, input, bytes, cudaMemcpyHostToDevice, stream));
        process(stream, buffers);
        uint8_t* edges = buffers.edges;
        check(cudaMemcpyAsync(output, edges, PIXELS, cudaMemcpyDeviceToHost, stream));
    }

    for (int slot = 0; slot < n; ++slot) {
        check(cudaStreamSynchronize(streams[slot]));
    }
    return count / ((nowMs() - t0) / 1e3);
}

// The 4 images of the pipeline that are kept: gray, equalized, blurred and edges.
struct Images {
    std::vector<uint8_t> gray, equal, blurred, edges;
};

// Runs f(y, row) for every row y, the rows split into 1 band per CPU core.
template <typename F>
void parRows(F f)
{
    int cores = std::max(1u, std::thread::hardware_concurrency());
    int band = (H + cores - 1) / cores;
    std::vector<std::thread> threads;

    for (int k = 0; k < cores; ++k) {
        threads.emplace_back([=] {
            for (int y = k * band; y < std::min(H, (k + 1) * band); ++y) {
                f(y);
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
}

// The same 7 stages on the CPU, with the same integer arithmetic, rows on every core.
Images cpuPipeline(const std::vector<uint8_t>& rgb)
{
    Images img;
    img.gray.resize(PIXELS);
    parRows([&](int y) {
        for (int x = 0; x < W; ++x) {
            int byte = 3 * (y * W + x);
            unsigned red = rgb[byte];
            unsigned green = rgb[byte + 1];
            unsigned blue = rgb[byte + 2];
            img.gray[y * W + x] = (77 * red + 150 * green + 29 * blue + 128) >> 8;
        }
    });

    unsigned bins[256] = {};
    for (uint8_t level : img.gray) {
        bins[level] += 1;
    }

    unsigned first = 0;
    unsigned cdf = 0;
    for (int level = 0; level < 256 && first == 0; ++level) {
        first = bins[level];
    }
    uint64_t span = PIXELS - first;
    uint8_t table[256];
    for (int level = 0; level < 256; ++level) {
        cdf += bins[level];
        uint64_t above = cdf > first ? cdf - first : 0;
        table[level] = span == 0 ? level : (above * 255 + span / 2) / span;
    }

    img.equal.resize(PIXELS);
    parRows([&](int y) {
        for (int x = 0; x < W; ++x) {
            img.equal[y * W + x] = table[img.gray[y * W + x]];
        }
    });

    std::vector<unsigned> rows(PIXELS);
    parRows([&](int y) {
        const uint8_t* equal = &img.equal[y * W];
        for (int x = 0; x < W; ++x) {
            unsigned sum = equal[tap(x, W)] + 4 * equal[tap(x + 1, W)];
            sum += 6 * equal[tap(x + 2, W)] + 4 * equal[tap(x + 3, W)];
            rows[y * W + x] = sum + equal[tap(x + 4, W)];
        }
    });

    img.blurred.resize(PIXELS);
    parRows([&](int y) {
        for (int x = 0; x < W; ++x) {
            unsigned sum = rows[tap(y, H) * W + x] + 4 * rows[tap(y + 1, H) * W + x];
            sum += 6 * rows[tap(y + 2, H) * W + x] + 4 * rows[tap(y + 3, H) * W + x];
            sum += rows[tap(y + 4, H) * W + x];
            img.blurred[y * W + x] = (sum + 128) >> 8;
        }
    });

    img.edges.resize(PIXELS);
    parRows([&](int y) {
        const uint8_t* src = img.blurred.data();
        const uint8_t* upper = src + std::max(y - 1, 0) * W;
        const uint8_t* middle = src + y * W;
        const uint8_t* lower = src + std::min(y + 1, H - 1) * W;
        for (int x = 0; x < W; ++x) {
            int xLeft = std::max(x - 1, 0);
            int xRight = std::min(x + 1, W - 1);
            int right = upper[xRight] + 2 * middle[xRight] + lower[xRight];
            int left = upper[xLeft] + 2 * middle[xLeft] + lower[xLeft];
            int below = lower[xLeft] + 2 * lower[x] + lower[xRight];
            int above = upper[xLeft] + 2 * upper[x] + upper[xRight];
            int gx = right - left;
            int gy = below - above;

            float magnitude = std::sqrt(float(gx * gx + gy * gy));
            img.edges[y * W + x] = std::min(magnitude * 0.25f + 0.5f, 255.0f);
        }
    });
    return img;
}

// A synthetic 1920 x 1080 scene with low contrast: sky, sun, hills, a house, grain.
std::vector<uint8_t> scene(unsigned seed)
{
    unsigned state = seed;
    std::vector<uint8_t> rgb(3 * PIXELS);
    float shift = seed * 0.05f;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = float(x) / W;
            float v = float(y) / H;
            float hill = 0.62f + 0.07f * std::sin(u * 9.0f + shift * 20.0f);
            float sunX = float(x) - (0.72f + shift) * W;
            float sun = std::hypot(sunX, float(y) - 260.0f);
            bool house = u >= 0.18f && u < 0.34f && v >= 0.48f && v < 0.80f;
            bool window = u >= 0.22f && u < 0.27f && v >= 0.56f && v < 0.64f;
            bool roof = v < 0.48f && v > 0.36f;
            roof = roof && std::fabs(u - 0.26f) < (v - 0.36f) * 0.75f;

            int base[3];
            if (window) {
                base[0] = 92, base[1] = 96, base[2] = 88;
            } else if (house) {
                base[0] = 150, base[1] = 118, base[2] = 104;
            } else if (roof) {
                base[0] = 128, base[1] = 92, base[2] = 84;
            } else if (sun < 72.0f) {
                base[0] = 176, base[1] = 168, base[2] = 120;
            } else if (v > hill) {
                base[0] = 84, base[1] = 112, base[2] = 78;
            } else {
                int sky = int(v * 40.0f);
                base[0] = 110 + sky, base[1] = 126 + sky, base[2] = 150;
            }

            state = state * 1664525u + 1013904223u;
            int grain = int(state >> 28) - 8;
            int p = 3 * (y * W + x);
            for (int c = 0; c < 3; ++c) {
                rgb[p + c] = std::clamp(base[c] + grain, 0, 255);
            }
        }
    }
    return rgb;
}

// Prints how many pixels of 2 images differ, and the first one that does.
using Image = std::vector<uint8_t>;

int report(const char* label, const Image& gpu, const Image& cpu)
{
    int wrong = 0;
    int firstWrong = -1;
    for (int i = 0; i < PIXELS; ++i) {
        if (gpu[i] != cpu[i]) {
            wrong += 1;
            firstWrong = firstWrong < 0 ? i : firstWrong;
        }
    }

    std::printf("  %-10s %8d", label, wrong);
    if (firstWrong >= 0) {
        int x = firstWrong % W;
        int y = firstWrong / W;
        int g = gpu[firstWrong];
        int c = cpu[firstWrong];
        std::printf("   first at (%d, %d): GPU %d, CPU %d", x, y, g, c);
    }
    std::printf("\n");
    return wrong;
}

// FNV-1a hash of an image, to compare the outputs of 2 programs.
unsigned checksum(const std::vector<uint8_t>& data)
{
    unsigned h = 2166136261u;
    for (uint8_t byte : data) {
        h = (h ^ byte) * 16777619u;
    }
    return h;
}

std::vector<uint8_t> download(const uint8_t* device)
{
    std::vector<uint8_t> host(PIXELS);
    check(cudaMemcpy(host.data(), device, PIXELS, cudaMemcpyDeviceToHost));
    return host;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;
    std::printf("GPU: %s, %d SMs   ", prop.name, sms);
    std::printf("image %d x %d = %d pixels\n", W, H, PIXELS);

    cudaStream_t streams[3];
    for (cudaStream_t& s : streams) {
        check(cudaStreamCreate(&s));
    }
    cudaStream_t s = streams[0];

    std::vector<std::vector<uint8_t>> scenes;
    std::vector<Images> reference;
    for (unsigned seed = 0; seed < 4; ++seed) {
        scenes.push_back(scene(seed));
        reference.push_back(cpuPipeline(scenes.back()));
    }

    // 1. Each stage of 1 image, timed alone.
    Buffers lanes[3] = {allocBuffers(), allocBuffers(), allocBuffers()};
    Buffers& b = lanes[0];
    check(cudaMemcpy(b.rgb, scenes[0].data(), 3 * PIXELS, cudaMemcpyHostToDevice));
    double warm = nowMs();
    while (nowMs() - warm < 400) {
        timeMs(s, [&] { process(s, b); });             // raise the GPU clock
    }

    std::printf("stage              ms     GB/s\n");
    double total = 0;
    for (int k = 0; k < 7; ++k) {
        double ms = timeMs(s, [&] { stage(s, b, k); });
        double gb = double(STAGES[k].bytes) * PIXELS / 1e9 / (ms / 1e3);
        std::printf("  %-13s %7.3f  %7.1f\n", STAGES[k].name, ms, gb);
        total += ms;
    }
    double all = timeMs(s, [&] { process(s, b); });
    std::printf("  sum of stages %7.3f\n", total);
    std::printf("  all 7 in turn %7.3f\n", all);

    // 2. The GPU's images against the CPU's, pixel by pixel.
    Images gpu;
    gpu.gray = download(b.gray);
    gpu.equal = download(b.equal);
    gpu.blurred = download(b.blurred);
    gpu.edges = download(b.edges);
    const Images& cpu = reference[0];
    std::printf("image      pixels that differ from the CPU\n");
    int wrong = report("gray", gpu.gray, cpu.gray);
    wrong += report("equalised", gpu.equal, cpu.equal);
    wrong += report("blurred", gpu.blurred, cpu.blurred);
    wrong += report("edges", gpu.edges, cpu.edges);
    std::printf("edge image checksum: %08x\n", checksum(gpu.edges));

    // 3. The CPU on every core, 4 images.
    double t0 = nowMs();
    for (const std::vector<uint8_t>& rgb : scenes) {
        cpuPipeline(rgb);
    }
    double cpuRate = scenes.size() / ((nowMs() - t0) / 1e3);
    unsigned cores = std::thread::hardware_concurrency();

    // 4. The copies alone, then a batch: copy in, 7 stages, copy out, on 1 to 3 streams.
    uint8_t* inputs[4];
    uint8_t* outputs[3];
    for (int k = 0; k < 4; ++k) {
        check(cudaMallocHost(&inputs[k], 3 * PIXELS));
        std::memcpy(inputs[k], scenes[k].data(), 3 * PIXELS);
    }
    for (int k = 0; k < 3; ++k) {
        check(cudaMallocHost(&outputs[k], PIXELS));
    }

    auto copyIn = [&] {
        check(cudaMemcpyAsync(b.rgb, inputs[0], 3 * PIXELS, cudaMemcpyHostToDevice, s));
    };
    auto copyOut = [&] {
        check(cudaMemcpyAsync(outputs[0], b.edges, PIXELS, cudaMemcpyDeviceToHost, s));
    };
    double inMs = timeMs(s, copyIn);
    double outMs = timeMs(s, copyOut);
    double inRate = 3.0 * PIXELS / 1e9 / (inMs / 1e3);
    double outRate = 1.0 * PIXELS / 1e9 / (outMs / 1e3);
    std::printf("copy in  %6.3f ms  %5.1f GB/s\n", inMs, inRate);
    std::printf("copy out %6.3f ms  %5.1f GB/s\n", outMs, outRate);

    batch(streams, lanes, 3, inputs, outputs, BATCH);
    std::printf("batch of %d images: copy in, 7 stages, copy out\n", BATCH);
    bool correct = true;
    for (int n = 1; n <= 3; ++n) {
        double rate = batch(streams, lanes, n, inputs, outputs, BATCH);
        std::printf("  %d stream(s)          %6.1f images/s\n", n, rate);

        for (int k = 0; k < n; ++k) {
            int last = (BATCH - 1 - (BATCH - 1 - k) % n) % 4;
            const uint8_t* expect = reference[last].edges.data();
            correct = correct && std::memcmp(outputs[k], expect, PIXELS) == 0;
        }
    }
    std::printf("  CPU, %u threads       %6.1f images/s\n", cores, cpuRate);

    bool passed = wrong == 0 && correct;
    std::printf("batch outputs match the CPU: %s\n", correct ? "yes" : "no");
    std::printf("%s\n", passed ? "PASSED" : "FAILED");

    for (int k = 0; k < 4; ++k) {
        cudaFreeHost(inputs[k]);
    }
    for (int k = 0; k < 3; ++k) {
        cudaFreeHost(outputs[k]);
        freeBuffers(lanes[k]);
    }
    for (cudaStream_t& st : streams) {
        cudaStreamDestroy(st);
    }
    return passed ? 0 : 1;
}
