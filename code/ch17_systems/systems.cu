// Chapter 17: the GPU as a device. What CUDA reports about it, how fast the CPU
// reaches each kind of memory mapping, and a work queue with a doorbell.
// Build: nvcc -arch=native -O2 systems.cu -o systems

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda_runtime.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

double nowUs()
{
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

void spinUs(double us)
{
    double end = nowUs() + us;
    while (nowUs() < end) {
    }
}

// 1. What the CUDA runtime knows about the device and how it is attached.
void identity()
{
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0));

    char busId[32];
    check(cudaDeviceGetPCIBusId(busId, sizeof(busId), 0));

    double gib = p.totalGlobalMem / 1073741824.0;
    int l2Mib = p.l2CacheSize >> 20;
    int viaHostTables = p.pageableMemoryAccessUsesHostPageTables;

    std::printf("device            %s, sm_%d%d\n", p.name, p.major, p.minor);
    std::printf("PCI address       %s\n", busId);
    std::printf("integrated        %d\n", p.integrated);
    std::printf("memory            %.1f GiB, L2 %d MiB\n", gib, l2Mib);

    std::printf("copy engines      %d\n", p.asyncEngineCount);
    std::printf("canMapHostMemory  %d\n", p.canMapHostMemory);
    std::printf("unifiedAddressing %d\n", p.unifiedAddressing);
    std::printf("pageableAccess    %d\n", p.pageableMemoryAccess);
    std::printf("viaHostPageTables %d\n", viaHostTables);
    std::printf("hostNativeAtomic  %d\n\n", p.hostNativeAtomicSupported);
}

// CPU reads every word of `x` and returns the sum, so the reads cannot be skipped.
uint32_t cpuRead(const uint32_t* x, size_t n)
{
    uint32_t s = 0;
    for (size_t i = 0; i < n; ++i) {
        s += x[i];
    }

    return s;
}

void cpuWrite(uint32_t* x, size_t n, uint32_t v)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = v;
    }
}

// CPU write and read bandwidth over `bytes` of `x`, in GB/s, best of `reps` passes.
void timeCpu(const char* name, uint32_t* x, size_t bytes, int reps)
{
    size_t n = bytes / sizeof(uint32_t);
    cpuWrite(x, n, 1);

    double bestWrite = 1e30;
    double bestRead = 1e30;
    uint32_t sum = 0;

    for (int r = 0; r < reps; ++r) {
        double t0 = nowUs();
        cpuWrite(x, n, 1);
        double t1 = nowUs();
        sum = cpuRead(x, n);
        double t2 = nowUs();

        bestWrite = std::min(bestWrite, t1 - t0);
        bestRead = std::min(bestRead, t2 - t1);
    }

    double writeGBs = bytes / bestWrite / 1e3;
    double readGBs = bytes / bestRead / 1e3;
    const char* verdict = sum == n ? "PASSED" : "FAILED";
    std::printf("%-24s write %6.2f GB/s   ", name, writeGBs);
    std::printf("read %6.2f GB/s   %s\n", readGBs, verdict);
}

// 2. The same CPU loops over 4 kinds of memory.
void memoryTypes()
{
    const size_t bytes = 64u << 20;
    const int reps = 5;

    uint32_t* pageable = static_cast<uint32_t*>(std::malloc(bytes));
    uint32_t* pinned = nullptr;
    uint32_t* combined = nullptr;
    uint32_t* managed = nullptr;

    check(cudaMallocHost(&pinned, bytes));
    check(cudaHostAlloc(&combined, bytes, cudaHostAllocWriteCombined));
    check(cudaMallocManaged(&managed, bytes));

    std::printf("CPU access to %zu MiB through each mapping:\n", bytes >> 20);
    timeCpu("malloc (pageable)", pageable, bytes, reps);
    timeCpu("cudaMallocHost (pinned)", pinned, bytes, reps);
    timeCpu("write-combined pinned", combined, bytes, reps);
    timeCpu("cudaMallocManaged", managed, bytes, reps);
    std::printf("\n");

    check(cudaFree(managed));
    check(cudaFreeHost(combined));
    check(cudaFreeHost(pinned));
    std::free(pageable);
}

// 3. A ring of work entries and a doorbell, between a host thread and a device thread.
struct Entry {
    uint32_t opcode;
    uint32_t arg;
};

struct Channel {
    static const uint32_t capacity = 64;

    Entry ring[capacity];

    // Each counter on its own 64-byte cache line, as each register has its own address.
    alignas(64) std::atomic<uint32_t> doorbell{0};   // PUT: entries written so far
    alignas(64) std::atomic<uint32_t> get{0};        // GET: entries the device finished
    alignas(64) std::atomic<bool> stop{false};
    uint64_t sum = 0;                                // the device's result
};

// The device: wait for the doorbell, run every entry up to PUT, then report GET.
void device(Channel* ch, double workUs)
{
    uint32_t get = 0;
    while (!ch->stop.load(std::memory_order_acquire)) {
        uint32_t put = ch->doorbell.load(std::memory_order_acquire);

        while (get != put) {
            Entry e = ch->ring[get % Channel::capacity];
            spinUs(workUs);
            ch->sum += e.opcode == 1 ? e.arg : 0;

            get += 1;
            ch->get.store(get, std::memory_order_release);
        }
    }
}

// The host: write an entry, then ring the doorbell. Waits only when the ring is full.
void submit(Channel* ch, uint32_t depth, uint32_t& put, uint32_t arg, bool doorbellFirst)
{
    while (put - ch->get.load(std::memory_order_acquire) >= depth) {
    }

    Entry& slot = ch->ring[put % Channel::capacity];
    if (doorbellFirst) {
        ch->doorbell.store(put + 1, std::memory_order_release);
        slot = Entry{1, arg};
    } else {
        slot = Entry{1, arg};
        ch->doorbell.store(put + 1, std::memory_order_release);
    }

    put += 1;
}

// `bursts` times: submit `burst` entries, then do the host's own work.
void runQueue(uint32_t depth, bool doorbellFirst)
{
    const int bursts = 200;
    const int burst = 32;
    const double workUs = 2.0;
    const double hostUs = 64.0;

    Channel ch;
    std::thread dev(device, &ch, workUs);

    uint32_t put = 0;
    uint64_t expected = 0;

    double t0 = nowUs();
    for (int b = 0; b < bursts; ++b) {
        for (int k = 0; k < burst; ++k) {
            uint32_t arg = put + 1;
            expected += arg;
            submit(&ch, depth, put, arg, doorbellFirst);
        }

        spinUs(hostUs);
    }

    while (ch.get.load(std::memory_order_acquire) != put) {
    }

    double ms = (nowUs() - t0) / 1e3;
    ch.stop.store(true, std::memory_order_release);
    dev.join();

    const char* order = doorbellFirst ? "doorbell first" : "entry first";
    const char* verdict = ch.sum == expected ? "PASSED" : "FAILED";
    unsigned long long got = ch.sum;
    unsigned long long want = expected;

    std::printf("depth %2u, %-14s %7.2f ms   ", depth, order, ms);
    std::printf("sum %llu of %llu   %s\n", got, want, verdict);
}

int main(int argc, char** argv)
{
    bool broken = argc > 1 && std::strcmp(argv[1], "--doorbell-first") == 0;

    identity();
    memoryTypes();

    std::printf("Work queue: 200 bursts of 32 entries, 2 us each, 64 us of host work:\n");
    runQueue(1, broken);
    runQueue(4, broken);
    runQueue(32, broken);
    runQueue(64, broken);
    return 0;
}
