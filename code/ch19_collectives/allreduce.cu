// Chapter 19: ring and tree all-reduce, simulated on 1 GPU.
// Each rank is a slice of 1 buffer in the memory of 1 GPU. No network is measured.
// Build: nvcc -arch=native allreduce.cu -o allreduce

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// 1 send: `len` floats from offset `from` of the buffer to offset `to`.
struct Transfer {
    int from;
    int to;
};

// 1 step of a collective: transfers that run at the same time, on different links.
struct Step {
    int first;      // index of the step's first transfer in the plan
    int count;      // number of transfers
    int len;        // floats per transfer
    bool add;       // add into the destination, or overwrite it
};

struct Plan {
    std::vector<Transfer> transfers;
    std::vector<Step> steps;
};

void addStep(Plan& plan, const std::vector<Transfer>& sends, int len, bool add)
{
    int first = int(plan.transfers.size());
    plan.transfers.insert(plan.transfers.end(), sends.begin(), sends.end());
    plan.steps.push_back({first, int(sends.size()), len, add});
}

// Ring: reduce-scatter in k - 1 steps, then all-gather in k - 1 steps.
// Rank r owns elements [r n, r n + n) of the buffer; chunk c is n / k floats.
Plan ring(int k, int n)
{
    Plan plan;
    int chunk = n / k;

    for (int s = 0; s < k - 1; ++s) {
        std::vector<Transfer> sends;
        for (int r = 0; r < k; ++r) {
            int c = (r - s + k) % k;
            int next = (r + 1) % k;
            sends.push_back({r * n + c * chunk, next * n + c * chunk});
        }
        addStep(plan, sends, chunk, true);
    }

    for (int s = 0; s < k - 1; ++s) {
        std::vector<Transfer> sends;
        for (int r = 0; r < k; ++r) {
            int c = (r + 1 - s + k) % k;
            int next = (r + 1) % k;
            sends.push_back({r * n + c * chunk, next * n + c * chunk});
        }
        addStep(plan, sends, chunk, false);
    }

    return plan;
}

// Binomial tree, k a power of 2: sums climb to rank 0, then the total comes back down.
Plan tree(int k, int n)
{
    Plan plan;

    for (int d = 1; d < k; d *= 2) {
        std::vector<Transfer> sends;
        for (int r = d; r < k; r += 2 * d) {
            sends.push_back({r * n, (r - d) * n});
        }
        addStep(plan, sends, n, true);
    }

    for (int d = k / 2; d >= 1; d /= 2) {
        std::vector<Transfer> sends;
        for (int r = 0; r < k; r += 2 * d) {
            sends.push_back({r * n, (r + d) * n});
        }
        addStep(plan, sends, n, false);
    }

    return plan;
}

// Bytes sent by the rank that sends the most.
double busiestRank(const Plan& plan, int k, int n)
{
    std::vector<double> sent(k, 0.0);

    for (const Step& st : plan.steps) {
        for (int t = st.first; t < st.first + st.count; ++t) {
            sent[plan.transfers[t].from / n] += st.len * sizeof(float);
        }
    }

    return *std::max_element(sent.begin(), sent.end());
}

// `per` blocks for each transfer of the step; block b works on transfer b / per.
// Within 1 step no element is both read and written, so the transfers cannot race.
__global__ void runStep(float* all, const Transfer* sends, int len, int per, bool add)
{
    Transfer t = sends[blockIdx.x / per];
    int stride = per * blockDim.x;

    for (int i = (blockIdx.x % per) * blockDim.x + threadIdx.x; i < len; i += stride) {
        float v = all[t.from + i];
        all[t.to + i] = add ? all[t.to + i] + v : v;
    }
}

// Rank r, element i starts as value(r, i); the sum over ranks is exact in floats.
__host__ __device__ float value(int r, int i, bool demo)
{
    return demo ? float(r + 1) : float(i % 16 + r);
}

__global__ void fill(float* all, int k, int n, bool demo)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < k * n) all[g] = value(g / n, g % n, demo);
}

struct Gpu {
    float* all;
    Transfer* sends;
};

void run(const Gpu& gpu, const Step& st, cudaStream_t s)
{
    int per = std::min((st.len + 255) / 256, std::max(1024 / st.count, 1));
    const Transfer* sends = gpu.sends + st.first;
    runStep<<<st.count * per, 256, 0, s>>>(gpu.all, sends, st.len, per, st.add);
}

void upload(const Gpu& gpu, const Plan& plan)
{
    size_t bytes = plan.transfers.size() * sizeof(Transfer);
    check(cudaMemcpy(gpu.sends, plan.transfers.data(), bytes, cudaMemcpyHostToDevice));
}

void printRanks(const Gpu& gpu, int k, int n, const char* when)
{
    std::vector<float> h(k * n);
    check(cudaMemcpy(h.data(), gpu.all, h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("%s\n", when);
    for (int r = 0; r < k; ++r) {
        std::printf("  rank %d  ", r);
        for (int i = 0; i < n; ++i) std::printf("%3.0f", h[r * n + i]);
        std::printf("\n");
    }
}

// 4 ranks of 8 floats, printed after every step of the ring.
void demo(const Gpu& gpu)
{
    const int k = 4;
    const int n = 8;
    Plan plan = ring(k, n);
    upload(gpu, plan);

    fill<<<1, k * n>>>(gpu.all, k, n, true);
    printRanks(gpu, k, n, "ring all-reduce, 4 ranks of 8 floats: rank r starts with r + 1");

    for (size_t s = 0; s < plan.steps.size(); ++s) {
        run(gpu, plan.steps[s], 0);

        char when[64];
        const char* phase = s < k - 1 ? "reduce-scatter" : "all-gather";
        std::snprintf(when, sizeof when, "after step %zu (%s)", s + 1, phase);
        printRanks(gpu, k, n, when);
    }
}

// True if every rank holds the exact sum of all ranks' starting values.
bool correct(const Gpu& gpu, int k, int n)
{
    std::vector<float> h(size_t(k) * n);
    check(cudaMemcpy(h.data(), gpu.all, h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int r = 0; r < k; ++r) sum += value(r, i, false);

        for (int r = 0; r < k; ++r) {
            if (h[size_t(r) * n + i] != sum) return false;
        }
    }

    return true;
}

// Average time of 1 all-reduce in milliseconds, over `reps` runs from fresh data.
float timeAllReduce(const Gpu& gpu, const Plan& plan, int k, int n, int reps, bool* ok)
{
    upload(gpu, plan);

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    float total = 0.0f;
    int blocks = (k * n + 255) / 256;

    for (int rep = 0; rep < reps; ++rep) {
        fill<<<blocks, 256>>>(gpu.all, k, n, false);

        check(cudaEventRecord(start));
        for (const Step& st : plan.steps) run(gpu, st, 0);
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));

        float ms;
        check(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
    }

    *ok = correct(gpu, k, n);
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return total / reps;
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    const int k = 8;
    const int maxN = 4 << 20;                         // 4 Mi floats, 16 MiB per rank
    Gpu gpu;
    check(cudaMalloc(&gpu.all, size_t(k) * maxN * sizeof(float)));
    check(cudaMalloc(&gpu.sends, 256 * sizeof(Transfer)));

    demo(gpu);

    bool ok = true;
    bool all = true;
    for (int warm = 0; warm < 40; ++warm) {           // about 0.4 s: let the clocks rise
        timeAllReduce(gpu, ring(k, maxN), k, maxN, 1, &ok);
    }

    Plan r = ring(k, maxN);
    Plan t = tree(k, maxN);
    double message = maxN * sizeof(float);
    double ringShare = busiestRank(r, k, maxN) / message;
    double treeShare = busiestRank(t, k, maxN) / message;

    std::printf("\nall-reduce on %d simulated ranks\n", k);
    std::printf("          steps   busiest rank sends\n");
    std::printf("  ring    %4zu     %.2f x the message\n", r.steps.size(), ringShare);
    std::printf("  tree    %4zu     %.2f x the message\n", t.steps.size(), treeShare);

    std::printf("\n  message     ring        tree\n");
    for (int n = 1024; n <= maxN; n *= 16) {
        float ringMs = timeAllReduce(gpu, ring(k, n), k, n, 50, &ok);
        all = all && ok;
        float treeMs = timeAllReduce(gpu, tree(k, n), k, n, 50, &ok);
        all = all && ok;

        double kib = n * sizeof(float) / 1024.0;
        std::printf("  %6.0f KiB  %7.3f ms  %7.3f ms\n", kib, ringMs, treeMs);
    }

    std::printf("every rank holds the exact sum: %s\n", all ? "PASSED" : "FAILED");

    check(cudaFree(gpu.all));
    check(cudaFree(gpu.sends));
    return all ? 0 : 1;
}
