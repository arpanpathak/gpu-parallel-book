// Chapter 21: 3 inclusive scans of a tile (Hillis-Steele, Blelloch, warp shuffles),
// the scan of a long array, and stream compaction by a scan and by warp votes.
// Build: nvcc -O3 -arch=native scan_compaction.cu -o scan_compaction

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const int TILE = 1024;                        // elements scanned by 1 block
const int N = 16 << 20;                       // elements in the measured runs

enum Method { HILLIS_STEELE, BLELLOCH, WARPS };
const char* NAMES[] = {"Hillis-Steele", "Blelloch", "warp shuffles"};

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
    for (int timed = 0; timed < 20; ++timed) {
        launch();
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    return (nowMs() - t0) / 20;
}

// Hillis-Steele: 1,024 threads, 1 element each, 2 buffers in shared memory.
__global__ void scanHillisSteele(const unsigned* x, unsigned* y, unsigned* sums, int n)
{
    __shared__ unsigned buf[2 * TILE];
    int index = blockIdx.x * TILE + threadIdx.x;

    buf[threadIdx.x] = index < n ? x[index] : 0;
    __syncthreads();

    int src = 0;
    for (int distance = 1; distance < TILE; distance *= 2) {
        int dst = TILE - src;
        unsigned value = buf[src + threadIdx.x];
        if (threadIdx.x >= distance) {
            value += buf[src + threadIdx.x - distance];
        }
        buf[dst + threadIdx.x] = value;
        __syncthreads();                      // the step is complete before any read
        src = dst;
    }

    if (index < n) {
        y[index] = buf[src + threadIdx.x];
    }
    if (threadIdx.x == TILE - 1) {
        sums[blockIdx.x] = buf[src + threadIdx.x];
    }
}

// Blelloch: 512 threads, 2 elements each, 1 buffer; an up-sweep, then a down-sweep.
__global__ void scanBlelloch(const unsigned* x, unsigned* y, unsigned* sums, int n)
{
    __shared__ unsigned cells[TILE];
    int index = blockIdx.x * TILE + 2 * threadIdx.x;

    unsigned first = index < n ? x[index] : 0;
    unsigned second = index + 1 < n ? x[index + 1] : 0;
    cells[2 * threadIdx.x] = first;
    cells[2 * threadIdx.x + 1] = second;

    for (int distance = 1; distance < TILE; distance *= 2) {
        __syncthreads();
        if (threadIdx.x < TILE / (2 * distance)) {
            int right = (2 * threadIdx.x + 2) * distance - 1;
            cells[right] += cells[right - distance];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sums[blockIdx.x] = cells[TILE - 1];
        cells[TILE - 1] = 0;
    }

    for (int distance = TILE / 2; distance >= 1; distance /= 2) {
        __syncthreads();
        if (threadIdx.x < TILE / (2 * distance)) {
            int right = (2 * threadIdx.x + 2) * distance - 1;
            unsigned left = cells[right - distance];
            cells[right - distance] = cells[right];
            cells[right] += left;
        }
    }
    __syncthreads();

    if (index < n) {
        y[index] = cells[2 * threadIdx.x] + first;    // exclusive plus own: inclusive
    }
    if (index + 1 < n) {
        y[index + 1] = cells[2 * threadIdx.x + 1] + second;
    }
}

// The scan of Chapter 8: 256 threads, 4 elements each, warp shuffles and 2 barriers.
__device__ unsigned warpScan(unsigned value)
{
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned left = __shfl_up_sync(0xffffffff, value, offset);
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

// Adds to every element of a tile the total of the tiles before it.
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

// Scans every tile of 1,024 elements on its own and writes the tile totals.
void scanTiles(Method method, const unsigned* x, unsigned* y, unsigned* sums, int n)
{
    int tiles = (n + TILE - 1) / TILE;
    if (method == HILLIS_STEELE) {
        scanHillisSteele<<<tiles, TILE>>>(x, y, sums, n);
    } else if (method == BLELLOCH) {
        scanBlelloch<<<tiles, TILE / 2>>>(x, y, sums, n);
    } else {
        scanWarps<<<tiles, 256>>>(x, y, sums, n);
    }
}

// Inclusive scan of n elements: the tiles, then the tile totals (in place), then offsets.
void scan(Method method, const unsigned* x, unsigned* y, int n, unsigned* scratch)
{
    int tiles = (n + TILE - 1) / TILE;
    scanTiles(method, x, y, scratch, n);

    if (tiles > 1) {
        scan(method, scratch, scratch, tiles, scratch + tiles);
        addOffsets<<<tiles, 256>>>(y, scratch, n);
    }
}

// flags[index] = 1 if x[index] is kept.
__global__ void keepFlags(const unsigned* x, unsigned threshold, unsigned* flags, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        flags[index] = x[index] >= threshold;
    }
}

// Writes each kept element at its exclusive scan, the inclusive scan minus its flag of 1.
__global__ void scatterKept(const unsigned* x, const unsigned* flags,
                            const unsigned* scanned, unsigned* out, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && flags[index] == 1) {
        out[scanned[index] - 1] = x[index];
    }
}

// The number of kept elements in each tile of 1,024, counted with warp votes.
__global__ void countKept(const unsigned* x, unsigned threshold, unsigned* counts, int n)
{
    __shared__ unsigned warps[8];
    int base = blockIdx.x * TILE;

    unsigned count = 0;
    for (int step = 0; step < 4; ++step) {
        int index = base + step * 256 + threadIdx.x;
        bool keep = index < n && x[index] >= threshold;
        count += __popc(__ballot_sync(0xffffffff, keep));
    }
    if (threadIdx.x % 32 == 0) {
        warps[threadIdx.x / 32] = count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned total = 0;
        for (int other = 0; other < 8; ++other) {
            total += warps[other];
        }
        counts[blockIdx.x] = total;
    }
}

// Writes the kept elements of each tile in order, after those of the earlier tiles.
__global__ void compactVotes(const unsigned* x, unsigned threshold,
                             const unsigned* scanned, unsigned* out, int n)
{
    __shared__ unsigned warps[8];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    unsigned below = (1u << lane) - 1;        // the lanes to the left of this one

    unsigned next = blockIdx.x > 0 ? scanned[blockIdx.x - 1] : 0;
    for (int step = 0; step < 4; ++step) {
        int index = blockIdx.x * TILE + step * 256 + threadIdx.x;
        bool keep = index < n && x[index] >= threshold;
        unsigned votes = __ballot_sync(0xffffffff, keep);
        if (lane == 0) {
            warps[warp] = __popc(votes);
        }
        __syncthreads();

        unsigned before = 0;
        unsigned total = 0;
        for (int other = 0; other < 8; ++other) {
            before += other < warp ? warps[other] : 0;
            total += warps[other];
        }

        if (keep) {
            out[next + before + __popc(votes & below)] = x[index];
        }
        next += total;
        __syncthreads();
    }
}

// Numbers from a small linear congruential generator.
std::vector<unsigned> randomBelow(int n, unsigned limit, unsigned seed)
{
    std::vector<unsigned> v(n);
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i] = (state >> 8) % limit;
    }
    return v;
}

void print(const char* name, const std::vector<unsigned>& v, int count)
{
    std::printf("  %-13s", name);
    for (int i = 0; i < count; ++i) {
        std::printf(" %u", v[i]);
    }
    std::printf("\n");
}

std::vector<unsigned> toHost(const unsigned* d, int n)
{
    std::vector<unsigned> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    return h;
}

// The worked example of the chapter: 8 values, 3 scans and 1 compaction.
bool smallExample(unsigned* scratch)
{
    std::vector<unsigned> x = {3, 1, 7, 0, 4, 1, 6, 3};
    std::vector<unsigned> inclusive = {3, 4, 11, 11, 15, 16, 22, 25};
    unsigned *dx, *dy, *flags, *out;
    check(cudaMalloc(&dx, 8 * sizeof(unsigned)));
    check(cudaMalloc(&dy, 8 * sizeof(unsigned)));
    check(cudaMalloc(&flags, 8 * sizeof(unsigned)));
    check(cudaMalloc(&out, 8 * sizeof(unsigned)));
    check(cudaMemcpy(dx, x.data(), 8 * sizeof(unsigned), cudaMemcpyHostToDevice));

    bool ok = true;
    std::printf("scans of 3 1 7 0 4 1 6 3\n");
    for (Method m : {HILLIS_STEELE, BLELLOCH, WARPS}) {
        scan(m, dx, dy, 8, scratch);
        std::vector<unsigned> y = toHost(dy, 8);
        print(NAMES[m], y, 8);
        ok = ok && y == inclusive;
    }

    keepFlags<<<1, 256>>>(dx, 3, flags, 8);
    scan(HILLIS_STEELE, flags, dy, 8, scratch);
    scatterKept<<<1, 256>>>(dx, flags, dy, out, 8);
    std::vector<unsigned> f = toHost(flags, 8);
    std::vector<unsigned> s = toHost(dy, 8);
    std::vector<unsigned> e(8);
    for (int i = 0; i < 8; ++i) {
        e[i] = s[i] - f[i];
    }
    std::vector<unsigned> kept = toHost(out, s[7]);

    std::printf("compaction of 3 1 7 0 4 1 6 3, keep x >= 3\n");
    print("flags", f, 8);
    print("exclusive", e, 8);
    print("output", kept, s[7]);
    ok = ok && kept == std::vector<unsigned>({3, 7, 4, 6, 3});

    check(cudaFree(dx));
    check(cudaFree(dy));
    check(cudaFree(flags));
    check(cudaFree(out));
    return ok;
}

// The 3 scans of N values: tiles alone, and the whole array, checked against the CPU.
bool scans(const unsigned* dx, unsigned* dy, unsigned* scratch,
           const std::vector<unsigned>& x)
{
    std::vector<unsigned> expected(N);
    unsigned run = 0;
    for (int i = 0; i < N; ++i) {
        run += x[i];
        expected[i] = run;
    }
    double gb = 2.0 * N * sizeof(unsigned) / 1e9;

    int perSm[3];                             // blocks that fit on 1 SM at once
    auto fit = cudaOccupancyMaxActiveBlocksPerMultiprocessor<decltype(&scanWarps)>;
    check(fit(&perSm[HILLIS_STEELE], scanHillisSteele, TILE, 0));
    check(fit(&perSm[BLELLOCH], scanBlelloch, TILE / 2, 0));
    check(fit(&perSm[WARPS], scanWarps, 256, 0));

    bool ok = true;
    std::printf("inclusive scans of %d values, average of 20 runs\n", N);
    std::printf("  method          blocks/SM   tiles ms   GB/s   whole ms   GB/s\n");
    for (Method m : {HILLIS_STEELE, BLELLOCH, WARPS}) {
        double tiles = timeMs([&] { scanTiles(m, dx, dy, scratch, N); });
        double whole = timeMs([&] { scan(m, dx, dy, N, scratch); });
        ok = ok && toHost(dy, N) == expected;
        std::printf("  %-14s %10d %10.3f %6.1f %10.3f %6.1f\n", NAMES[m], perSm[m],
                    tiles, gb / (tiles / 1e3), whole, gb / (whole / 1e3));
    }
    return ok;
}

// Compaction of N values in [0, 100) at 3 thresholds, by 3 passes and by warp votes.
bool compaction(const unsigned* dx, unsigned* scratch, const std::vector<unsigned>& x)
{
    int tiles = (N + TILE - 1) / TILE;
    unsigned *flags, *scanned, *out, *counts;
    check(cudaMalloc(&flags, N * sizeof(unsigned)));
    check(cudaMalloc(&scanned, N * sizeof(unsigned)));
    check(cudaMalloc(&out, N * sizeof(unsigned)));
    check(cudaMalloc(&counts, tiles * sizeof(unsigned)));
    int blocks = (N + 255) / 256;

    bool ok = true;
    std::printf("compaction of %d values in [0, 100), average of 20 runs\n", N);
    std::printf("  keep x >=      kept   3 passes ms   GB/s   warp votes ms   GB/s\n");
    for (unsigned threshold : {90u, 50u, 10u}) {
        std::vector<unsigned> expected;
        for (unsigned v : x) {
            if (v >= threshold) {
                expected.push_back(v);
            }
        }
        int kept = expected.size();
        double gb = (4.0 * N + 4.0 * kept) / 1e9;

        auto three = [&] {
            keepFlags<<<blocks, 256>>>(dx, threshold, flags, N);
            scan(WARPS, flags, scanned, N, scratch);
            scatterKept<<<blocks, 256>>>(dx, flags, scanned, out, N);
        };
        check(cudaMemset(out, 0, N * sizeof(unsigned)));
        double threeMs = timeMs(three);
        ok = ok && toHost(out, kept) == expected;

        auto votes = [&] {
            countKept<<<tiles, 256>>>(dx, threshold, counts, N);
            scan(WARPS, counts, counts, tiles, scratch);
            compactVotes<<<tiles, 256>>>(dx, threshold, counts, out, N);
        };
        check(cudaMemset(out, 0, N * sizeof(unsigned)));
        double votesMs = timeMs(votes);
        ok = ok && toHost(out, kept) == expected;

        std::printf("  %9u %9d %13.2f %6.1f %15.2f %6.1f\n", threshold, kept,
                    threeMs, gb / (threeMs / 1e3), votesMs, gb / (votesMs / 1e3));
    }

    check(cudaFree(flags));
    check(cudaFree(scanned));
    check(cudaFree(out));
    check(cudaFree(counts));
    return ok;
}

int main()
{
    unsigned *dx, *dy, *scratch;
    check(cudaMalloc(&dx, N * sizeof(unsigned)));
    check(cudaMalloc(&dy, N * sizeof(unsigned)));
    check(cudaMalloc(&scratch, N / 512 * sizeof(unsigned)));

    bool ok = smallExample(scratch);

    std::vector<unsigned> x = randomBelow(N, 4, 1);
    check(cudaMemcpy(dx, x.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice));
    ok = scans(dx, dy, scratch, x) && ok;

    x = randomBelow(N, 100, 2);
    check(cudaMemcpy(dx, x.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice));
    ok = compaction(dx, scratch, x) && ok;

    check(cudaFree(dx));
    check(cudaFree(dy));
    check(cudaFree(scratch));
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
