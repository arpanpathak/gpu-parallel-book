// Chapter 22: histograms of 4-bit digits with atomics, a radix sort by 1-bit splits,
// and a radix sort by 4-bit digits whose offsets come from a histogram and a scan.
// Build: nvcc -O3 -arch=native radix_sort_histogram.cu -o radix_sort_histogram

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const int TILE = 1024;                        // keys per block in the scan and the sort
const int N = 16 << 20;                       // keys in the measured runs
const int RADIX = 16;                         // digit values of a 4-bit pass
const int GRID = 1024;                        // blocks of the histogram kernels
const unsigned FULL = 0xffffffff;

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

// Runs `launch` for 0.4 s so that the clocks rise, then averages 20 runs, in ms.
template <typename F>
double timeMs(F launch)
{
    double warm = nowMs();
    while (nowMs() - warm < 400) {
        launch();
        check(cudaDeviceSynchronize());
    }

    double t0 = nowMs();
    for (int r = 0; r < 20; ++r) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    return (nowMs() - t0) / 20;
}

__device__ unsigned digitOf(unsigned key, int shift)
{
    return (key >> shift) & (RADIX - 1);
}

// 1. Every count is an atomic addition on 1 of 16 bins in global memory.
__global__ void histGlobal(const unsigned* keys, int shift, unsigned* bins, int n)
{
    int stride = gridDim.x * blockDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += stride) {
        atomicAdd(&bins[digitOf(keys[index], shift)], 1u);
    }
}

// 2. Each block counts into private bins in shared memory and merges them once.
__global__ void histShared(const unsigned* keys, int shift, unsigned* bins, int n)
{
    __shared__ unsigned local[RADIX];
    if (threadIdx.x < RADIX) {
        local[threadIdx.x] = 0;
    }
    __syncthreads();

    int stride = gridDim.x * blockDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += stride) {
        atomicAdd(&local[digitOf(keys[index], shift)], 1u);
    }
    __syncthreads();

    if (threadIdx.x < RADIX) {
        atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
    }
}

// 3. As 2, but the lanes of a warp with equal digits add their count with 1 atomic.
__global__ void histAggregated(const unsigned* keys, int shift, unsigned* bins, int n)
{
    __shared__ unsigned local[RADIX];
    if (threadIdx.x < RADIX) {
        local[threadIdx.x] = 0;
    }
    __syncthreads();

    int lane = threadIdx.x % 32;
    int stride = gridDim.x * blockDim.x;
    for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
        int index = base + threadIdx.x;
        unsigned digit = index < n ? digitOf(keys[index], shift) : RADIX;
        unsigned peers = __match_any_sync(FULL, digit);
        int leader = __ffs(peers) - 1;
        if (lane == leader && digit < RADIX) {
            atomicAdd(&local[digit], __popc(peers));
        }
    }
    __syncthreads();

    if (threadIdx.x < RADIX) {
        atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
    }
}

// The scan of Chapter 21: 256 threads, 4 elements each, warp shuffles and 2 barriers.
__device__ unsigned warpScan(unsigned value)
{
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned left = __shfl_up_sync(FULL, value, offset);
        if (lane >= offset) {
            value += left;
        }
    }
    return value;
}

__global__ void scanWarps(const unsigned* x, unsigned* y, unsigned* sums, int n)
{
    __shared__ unsigned warps[8];
    int first = blockIdx.x * TILE + 4 * threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    unsigned prefix[4];
    unsigned own = 0;
    for (int element = 0; element < 4; ++element) {
        own += first + element < n ? x[first + element] : 0;
        prefix[element] = own;
    }

    unsigned warpPrefix = warpScan(own);
    if (lane == 31) {
        warps[warp] = warpPrefix;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned scannedTotal = warpScan(lane < 8 ? warps[lane] : 0);
        if (lane < 8) {
            warps[lane] = scannedTotal;
        }
    }
    __syncthreads();

    unsigned before = warpPrefix - own + (warp > 0 ? warps[warp - 1] : 0);
    for (int element = 0; element < 4; ++element) {
        if (first + element < n) {
            y[first + element] = before + prefix[element];
        }
    }
    if (threadIdx.x == 255) {
        sums[blockIdx.x] = before + prefix[3];
    }
}

__global__ void addOffsets(unsigned* y, const unsigned* scanned, int n)
{
    if (blockIdx.x == 0) {
        return;
    }
    unsigned offset = scanned[blockIdx.x - 1];
    for (int step = 0; step < 4; ++step) {
        int index = blockIdx.x * TILE + step * 256 + threadIdx.x;
        if (index < n) {
            y[index] += offset;
        }
    }
}

// Inclusive scan of n elements: the tiles, then the tile totals (in place), then offsets.
void scan(const unsigned* x, unsigned* y, int n, unsigned* scratch)
{
    int tiles = (n + TILE - 1) / TILE;
    scanWarps<<<tiles, 256>>>(x, y, scratch, n);

    if (tiles > 1) {
        scan(scratch, scratch, tiles, scratch + tiles);
        addOffsets<<<tiles, 256>>>(y, scratch, n);
    }
}

// The split on 1 bit: flags[index] = 1 if the bit of keys[index] is 0.
__global__ void zeroFlags(const unsigned* keys, int bit, unsigned* flags, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        flags[index] = 1 - ((keys[index] >> bit) & 1);
    }
}

// Zero-bit keys go to their exclusive scan, one-bit keys after all zero-bit keys.
__global__ void splitScatter(const unsigned* keys, const unsigned* flags,
                             const unsigned* scanned, unsigned* out, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        unsigned zeros = scanned[n - 1];
        unsigned upto = scanned[index];       // zero-bit keys among 0 to index
        unsigned at = flags[index] == 1 ? upto - 1 : zeros + index - upto;
        out[at] = keys[index];
    }
}

// The 4-bit pass, step 1: the digit histogram of every tile, stored digit by digit:
// counts[digit * tiles + tile] is the number of keys with that digit in that tile.
__global__ void digitCounts(const unsigned* keys, int shift, unsigned* counts, int n)
{
    __shared__ unsigned local[RADIX];
    if (threadIdx.x < RADIX) {
        local[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int step = 0; step < 4; ++step) {
        int index = blockIdx.x * TILE + step * 256 + threadIdx.x;
        if (index < n) {
            atomicAdd(&local[digitOf(keys[index], shift)], 1u);
        }
    }
    __syncthreads();

    if (threadIdx.x < RADIX) {
        counts[threadIdx.x * gridDim.x + blockIdx.x] = local[threadIdx.x];
    }
}

// The 4-bit pass, step 3: every key goes to the start of its digit in its tile, plus
// the keys with the same digit before it in the tile. 4 steps of 256 keys, in order.
__global__ void scatterDigits(const unsigned* keys, int shift, const unsigned* counts,
                              const unsigned* scanned, unsigned* out, int n)
{
    __shared__ unsigned start[RADIX];
    __shared__ unsigned warpCounts[8 * RADIX];         // [warp][digit]
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    unsigned below = (1u << lane) - 1;

    if (threadIdx.x < RADIX) {
        int entry = threadIdx.x * gridDim.x + blockIdx.x;
        start[threadIdx.x] = scanned[entry] - counts[entry];  // the exclusive scan
    }
    for (int step = 0; step < 4; ++step) {
        if (threadIdx.x < 8 * RADIX) {
            warpCounts[threadIdx.x] = 0;
        }
        __syncthreads();

        int index = blockIdx.x * TILE + step * 256 + threadIdx.x;
        unsigned key = index < n ? keys[index] : 0;
        unsigned digit = index < n ? digitOf(key, shift) : RADIX;
        unsigned peers = __match_any_sync(FULL, digit);
        unsigned rank = __popc(peers & below);
        if (rank == 0 && digit < RADIX) {
            warpCounts[warp * RADIX + digit] = __popc(peers);
        }
        __syncthreads();

        if (digit < RADIX) {
            unsigned before = 0;
            for (int other = 0; other < warp; ++other) {
                before += warpCounts[other * RADIX + digit];
            }
            out[start[digit] + before + rank] = key;
        }
        __syncthreads();

        if (threadIdx.x < RADIX) {
            for (int other = 0; other < 8; ++other) {
                start[threadIdx.x] += warpCounts[other * RADIX + threadIdx.x];
            }
        }
        __syncthreads();
    }
}

struct Buffers {
    unsigned* spare;                          // the other half of each pass
    unsigned* flags;                          // 1-bit sort: flags, or 4-bit: counts
    unsigned* scanned;
    unsigned* scratch;                        // the tile totals of the scan
};

// 32 stable splits of 1 bit, bit 0 first. After an even number of passes the sorted
// keys are back in `keys`.
void sortBits(unsigned*& keys, Buffers& buffers, int n)
{
    int blocks = (n + 255) / 256;
    unsigned* flags = buffers.flags;
    unsigned* scanned = buffers.scanned;

    for (int bit = 0; bit < 32; ++bit) {
        zeroFlags<<<blocks, 256>>>(keys, bit, flags, n);
        scan(flags, scanned, n, buffers.scratch);
        splitScatter<<<blocks, 256>>>(keys, flags, scanned, buffers.spare, n);
        std::swap(keys, buffers.spare);
    }
}

// 8 passes of 4 bits: digit histograms per tile, 1 scan of the table, a stable scatter.
void sortDigits(unsigned*& keys, Buffers& buffers, int n)
{
    int tiles = (n + TILE - 1) / TILE;
    unsigned* counts = buffers.flags;
    unsigned* scanned = buffers.scanned;

    for (int shift = 0; shift < 32; shift += 4) {
        digitCounts<<<tiles, 256>>>(keys, shift, counts, n);
        scan(counts, scanned, RADIX * tiles, buffers.scratch);
        scatterDigits<<<tiles, 256>>>(keys, shift, counts, scanned, buffers.spare, n);
        std::swap(keys, buffers.spare);
    }
}

std::vector<unsigned> toHost(const unsigned* device, int n)
{
    std::vector<unsigned> host(n);
    check(cudaMemcpy(host.data(), device, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    return host;
}

void print(const char* name, const std::vector<unsigned>& v)
{
    std::printf("  %-13s", name);
    for (unsigned x : v) {
        std::printf(" %u", x);
    }
    std::printf("\n");
}

// The worked example: 8 keys of 3 bits, 3 splits, then 1 pass on a 4-bit digit.
bool smallExample(Buffers& buffers)
{
    std::vector<unsigned> host = {5, 2, 7, 2, 0, 6, 3, 5};
    unsigned *keys, *other, *bins;
    check(cudaMalloc(&keys, 8 * sizeof(unsigned)));
    check(cudaMalloc(&other, 8 * sizeof(unsigned)));
    check(cudaMalloc(&bins, RADIX * sizeof(unsigned)));
    check(cudaMemcpy(keys, host.data(), 8 * sizeof(unsigned), cudaMemcpyHostToDevice));
    std::printf("radix sort of 5 2 7 2 0 6 3 5\n");

    unsigned* from = keys;
    unsigned* to = other;
    for (int bit = 0; bit < 3; ++bit) {
        zeroFlags<<<1, 256>>>(from, bit, buffers.flags, 8);
        scan(buffers.flags, buffers.scanned, 8, buffers.scratch);
        splitScatter<<<1, 256>>>(from, buffers.flags, buffers.scanned, to, 8);
        std::swap(from, to);
        char name[16];
        std::snprintf(name, sizeof(name), "split bit %d", bit);
        print(name, toHost(from, 8));
    }
    std::vector<unsigned> bySplits = toHost(from, 8);

    check(cudaMemcpy(keys, host.data(), 8 * sizeof(unsigned), cudaMemcpyHostToDevice));
    check(cudaMemset(bins, 0, RADIX * sizeof(unsigned)));
    histShared<<<1, 256>>>(keys, 0, bins, 8);
    std::vector<unsigned> hist = toHost(bins, 8);
    print("histogram", hist);

    digitCounts<<<1, 256>>>(keys, 0, buffers.flags, 8);
    scan(buffers.flags, buffers.scanned, RADIX, buffers.scratch);
    scatterDigits<<<1, 256>>>(keys, 0, buffers.flags, buffers.scanned, other, 8);
    std::vector<unsigned> inclusive = toHost(buffers.scanned, 8);
    std::vector<unsigned> starts(8);
    for (int digit = 0; digit < 8; ++digit) {
        starts[digit] = inclusive[digit] - hist[digit];
    }
    print("digit starts", starts);
    std::vector<unsigned> byDigits = toHost(other, 8);
    print("4-bit pass", byDigits);

    check(cudaFree(keys));
    check(cudaFree(other));
    check(cudaFree(bins));
    std::vector<unsigned> sorted = {0, 2, 2, 3, 5, 5, 6, 7};
    return bySplits == sorted && byDigits == sorted;
}

// The 3 histograms of the low 4 bits of N keys, on random keys and on keys all equal.
bool histograms(unsigned* keys)
{
    unsigned* bins;
    check(cudaMalloc(&bins, RADIX * sizeof(unsigned)));
    double gb = 4.0 * N / 1e9;

    bool ok = true;
    std::printf("histograms of the low 4 bits of %d keys, average of 20 runs\n", N);
    std::printf("  keys        global ms   GB/s   shared ms   GB/s");
    std::printf("   warp-aggregated ms   GB/s\n");
    for (int input = 0; input < 2; ++input) {
        std::vector<unsigned> host(N);
        unsigned state = 7;
        for (int i = 0; i < N; ++i) {
            state = state * 1664525u + 1013904223u;
            host[i] = input == 0 ? state : 7;
        }
        check(cudaMemcpy(keys, host.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice));
        std::vector<unsigned> expected(RADIX, 0);
        for (unsigned k : host) {
            expected[k & 15] += 1;
        }

        double ms[3];
        for (int kernel = 0; kernel < 3; ++kernel) {
            auto run = [&] {
                if (kernel == 0) {
                    histGlobal<<<GRID, 256>>>(keys, 0, bins, N);
                } else if (kernel == 1) {
                    histShared<<<GRID, 256>>>(keys, 0, bins, N);
                } else {
                    histAggregated<<<GRID, 256>>>(keys, 0, bins, N);
                }
            };
            ms[kernel] = timeMs(run);

            check(cudaMemset(bins, 0, RADIX * sizeof(unsigned)));
            run();
            ok = ok && toHost(bins, RADIX) == expected;
        }
        std::printf("  %-10s %10.2f %6.1f %11.2f %6.1f %20.2f %6.1f\n",
                    input == 0 ? "random" : "all 7", ms[0], gb / (ms[0] / 1e3),
                    ms[1], gb / (ms[1] / 1e3), ms[2], gb / (ms[2] / 1e3));
    }

    check(cudaFree(bins));
    return ok;
}

// Both radix sorts of N random keys, best of 3 runs each, checked against std::sort.
bool sorts(unsigned* keys, Buffers& buffers)
{
    std::vector<unsigned> host(N);
    unsigned state = 11;
    for (int i = 0; i < N; ++i) {
        state = state * 1664525u + 1013904223u;
        host[i] = state;
    }
    std::vector<unsigned> expected = host;
    std::sort(expected.begin(), expected.end());

    bool ok = true;
    double best[2] = {1e30, 1e30};
    for (int method = 0; method < 2; ++method) {
        for (int run = 0; run < 4; ++run) {
            size_t bytes = N * sizeof(unsigned);
            check(cudaMemcpy(keys, host.data(), bytes, cudaMemcpyHostToDevice));
            double start = nowMs();
            if (method == 0) {
                sortBits(keys, buffers, N);
            } else {
                sortDigits(keys, buffers, N);
            }
            check(cudaDeviceSynchronize());
            double ms = nowMs() - start;
            if (run > 0) {                    // run 0 warms up
                best[method] = std::min(best[method], ms);
            }
            ok = ok && toHost(keys, N) == expected;
        }
    }

    std::printf("radix sorts of %d random 32-bit keys, best of 3 runs\n", N);
    std::printf("  32 passes of 1 bit  %8.2f ms %7.1f Mkeys/s\n", best[0],
                N / 1e6 / (best[0] / 1e3));
    std::printf("  8 passes of 4 bits  %8.2f ms %7.1f Mkeys/s\n", best[1],
                N / 1e6 / (best[1] / 1e3));

    // The 3 parts of 1 pass of 4 bits on the random keys, averaged like the histograms.
    check(cudaMemcpy(keys, host.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice));
    int tiles = (N + TILE - 1) / TILE;
    unsigned* counts = buffers.flags;
    unsigned* starts = buffers.scanned;
    unsigned* spare = buffers.spare;
    unsigned* scratch = buffers.scratch;
    double countMs = timeMs([&] { digitCounts<<<tiles, 256>>>(keys, 0, counts, N); });
    double scanMs = timeMs([&] { scan(counts, starts, RADIX * tiles, scratch); });
    auto move = [&] { scatterDigits<<<tiles, 256>>>(keys, 0, counts, starts, spare, N); };
    double scatterMs = timeMs(move);
    std::printf("  1 pass of 4 bits: counts %.2f ms, scan %.2f ms, scatter %.2f ms\n",
                countMs, scanMs, scatterMs);
    return ok;
}

int main()
{
    unsigned* keys;
    Buffers buffers;
    check(cudaMalloc(&keys, N * sizeof(unsigned)));
    check(cudaMalloc(&buffers.spare, N * sizeof(unsigned)));
    check(cudaMalloc(&buffers.flags, N * sizeof(unsigned)));
    check(cudaMalloc(&buffers.scanned, N * sizeof(unsigned)));
    check(cudaMalloc(&buffers.scratch, N / 512 * sizeof(unsigned)));

    bool ok = smallExample(buffers);
    ok = histograms(keys) && ok;
    ok = sorts(keys, buffers) && ok;

    check(cudaFree(keys));
    check(cudaFree(buffers.spare));
    check(cudaFree(buffers.flags));
    check(cudaFree(buffers.scanned));
    check(cudaFree(buffers.scratch));
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
