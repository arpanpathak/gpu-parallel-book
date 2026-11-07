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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&bins[digitOf(keys[i], shift)], 1u);
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&local[digitOf(keys[i], shift)], 1u);
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
        int i = base + threadIdx.x;
        unsigned d = i < n ? digitOf(keys[i], shift) : RADIX;
        unsigned peers = __match_any_sync(FULL, d);
        int leader = __ffs(peers) - 1;
        if (lane == leader && d < RADIX) {
            atomicAdd(&local[d], __popc(peers));
        }
    }
    __syncthreads();

    if (threadIdx.x < RADIX) {
        atomicAdd(&bins[threadIdx.x], local[threadIdx.x]);
    }
}

// The scan of Chapter 21: 256 threads, 4 elements each, warp shuffles and 2 barriers.
__device__ unsigned warpScan(unsigned v)
{
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned left = __shfl_up_sync(FULL, v, offset);
        if (lane >= offset) {
            v += left;
        }
    }
    return v;
}

__global__ void scanWarps(const unsigned* x, unsigned* y, unsigned* sums, int n)
{
    __shared__ unsigned warps[8];
    int first = blockIdx.x * TILE + 4 * threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    unsigned a[4];
    unsigned own = 0;
    for (int k = 0; k < 4; ++k) {
        own += first + k < n ? x[first + k] : 0;
        a[k] = own;
    }

    unsigned v = warpScan(own);
    if (lane == 31) {
        warps[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned w = warpScan(lane < 8 ? warps[lane] : 0);
        if (lane < 8) {
            warps[lane] = w;
        }
    }
    __syncthreads();

    unsigned before = v - own + (warp > 0 ? warps[warp - 1] : 0);
    for (int k = 0; k < 4; ++k) {
        if (first + k < n) {
            y[first + k] = before + a[k];
        }
    }
    if (threadIdx.x == 255) {
        sums[blockIdx.x] = before + a[3];
    }
}

__global__ void addOffsets(unsigned* y, const unsigned* scanned, int n)
{
    if (blockIdx.x == 0) {
        return;
    }
    unsigned offset = scanned[blockIdx.x - 1];
    for (int k = 0; k < 4; ++k) {
        int g = blockIdx.x * TILE + k * 256 + threadIdx.x;
        if (g < n) {
            y[g] += offset;
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

// The split on 1 bit: flags[i] = 1 if the bit of keys[i] is 0.
__global__ void zeroFlags(const unsigned* keys, int bit, unsigned* flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        flags[i] = 1 - ((keys[i] >> bit) & 1);
    }
}

// Zero-bit keys go to their exclusive scan, one-bit keys after all zero-bit keys.
__global__ void splitScatter(const unsigned* keys, const unsigned* flags,
                             const unsigned* scanned, unsigned* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned zeros = scanned[n - 1];
        unsigned upto = scanned[i];           // zero-bit keys among 0 to i
        unsigned at = flags[i] == 1 ? upto - 1 : zeros + i - upto;
        out[at] = keys[i];
    }
}

// The 4-bit pass, step 1: the digit histogram of every tile, stored digit by digit:
// counts[d * tiles + b] is the number of keys with digit d in tile b.
__global__ void digitCounts(const unsigned* keys, int shift, unsigned* counts, int n)
{
    __shared__ unsigned local[RADIX];
    if (threadIdx.x < RADIX) {
        local[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int k = 0; k < 4; ++k) {
        int i = blockIdx.x * TILE + k * 256 + threadIdx.x;
        if (i < n) {
            atomicAdd(&local[digitOf(keys[i], shift)], 1u);
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
    int t = threadIdx.x;
    int lane = t % 32;
    int warp = t / 32;
    unsigned below = (1u << lane) - 1;

    if (t < RADIX) {
        int j = t * gridDim.x + blockIdx.x;
        start[t] = scanned[j] - counts[j];             // the exclusive scan
    }
    for (int k = 0; k < 4; ++k) {
        if (t < 8 * RADIX) {
            warpCounts[t] = 0;
        }
        __syncthreads();

        int i = blockIdx.x * TILE + k * 256 + t;
        unsigned key = i < n ? keys[i] : 0;
        unsigned d = i < n ? digitOf(key, shift) : RADIX;
        unsigned peers = __match_any_sync(FULL, d);
        unsigned rank = __popc(peers & below);
        if (rank == 0 && d < RADIX) {
            warpCounts[warp * RADIX + d] = __popc(peers);
        }
        __syncthreads();

        if (d < RADIX) {
            unsigned before = 0;
            for (int w = 0; w < warp; ++w) {
                before += warpCounts[w * RADIX + d];
            }
            out[start[d] + before + rank] = key;
        }
        __syncthreads();

        if (t < RADIX) {
            for (int w = 0; w < 8; ++w) {
                start[t] += warpCounts[w * RADIX + t];
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
void sortBits(unsigned*& keys, Buffers& b, int n)
{
    int blocks = (n + 255) / 256;
    for (int bit = 0; bit < 32; ++bit) {
        zeroFlags<<<blocks, 256>>>(keys, bit, b.flags, n);
        scan(b.flags, b.scanned, n, b.scratch);
        splitScatter<<<blocks, 256>>>(keys, b.flags, b.scanned, b.spare, n);
        std::swap(keys, b.spare);
    }
}

// 8 passes of 4 bits: digit histograms per tile, 1 scan of the table, a stable scatter.
void sortDigits(unsigned*& keys, Buffers& b, int n)
{
    int tiles = (n + TILE - 1) / TILE;
    for (int shift = 0; shift < 32; shift += 4) {
        digitCounts<<<tiles, 256>>>(keys, shift, b.flags, n);
        scan(b.flags, b.scanned, RADIX * tiles, b.scratch);
        scatterDigits<<<tiles, 256>>>(keys, shift, b.flags, b.scanned, b.spare, n);
        std::swap(keys, b.spare);
    }
}

std::vector<unsigned> toHost(const unsigned* d, int n)
{
    std::vector<unsigned> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    return h;
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
bool smallExample(Buffers& b)
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
        zeroFlags<<<1, 256>>>(from, bit, b.flags, 8);
        scan(b.flags, b.scanned, 8, b.scratch);
        splitScatter<<<1, 256>>>(from, b.flags, b.scanned, to, 8);
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

    digitCounts<<<1, 256>>>(keys, 0, b.flags, 8);
    scan(b.flags, b.scanned, RADIX, b.scratch);
    scatterDigits<<<1, 256>>>(keys, 0, b.flags, b.scanned, other, 8);
    std::vector<unsigned> inclusive = toHost(b.scanned, 8);
    std::vector<unsigned> starts(8);
    for (int d = 0; d < 8; ++d) {
        starts[d] = inclusive[d] - hist[d];
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
bool sorts(unsigned* keys, Buffers& b)
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
                sortBits(keys, b, N);
            } else {
                sortDigits(keys, b, N);
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
    unsigned* counts = b.flags;
    double countMs = timeMs([&] { digitCounts<<<tiles, 256>>>(keys, 0, counts, N); });
    double scanMs = timeMs([&] { scan(counts, b.scanned, RADIX * tiles, b.scratch); });
    unsigned* starts = b.scanned;
    auto move = [&] { scatterDigits<<<tiles, 256>>>(keys, 0, counts, starts, b.spare, N); };
    double scatterMs = timeMs(move);
    std::printf("  1 pass of 4 bits: counts %.2f ms, scan %.2f ms, scatter %.2f ms\n",
                countMs, scanMs, scatterMs);
    return ok;
}

int main()
{
    unsigned* keys;
    Buffers b;
    check(cudaMalloc(&keys, N * sizeof(unsigned)));
    check(cudaMalloc(&b.spare, N * sizeof(unsigned)));
    check(cudaMalloc(&b.flags, N * sizeof(unsigned)));
    check(cudaMalloc(&b.scanned, N * sizeof(unsigned)));
    check(cudaMalloc(&b.scratch, N / 512 * sizeof(unsigned)));

    bool ok = smallExample(b);
    ok = histograms(keys) && ok;
    ok = sorts(keys, b) && ok;

    check(cudaFree(keys));
    check(cudaFree(b.spare));
    check(cudaFree(b.flags));
    check(cudaFree(b.scanned));
    check(cudaFree(b.scratch));
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
