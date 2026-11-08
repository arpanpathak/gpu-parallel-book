// Chapter 23: softmax and layer normalization of the rows of a matrix, in CUDA C++.
// Build: nvcc -O3 -arch=native softmax_layernorm.cu -o softmax_layernorm

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

const int BLOCK = 256;
const int WARPS = BLOCK / 32;
const int ROWS = 4096;
const int COLS = 4096;
const int PER_THREAD = COLS / BLOCK;
const float EPS = 1e-5f;

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

__device__ float warpMax(float value)
{
    for (int mask = 16; mask > 0; mask /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
    }
    return value;
}

__device__ float warpSum(float value)
{
    for (int mask = 16; mask > 0; mask /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

// The maximum over the block, returned to every thread. `slots` holds 8 floats.
__device__ float blockMax(float value, float* slots)
{
    value = warpMax(value);
    int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        slots[warp] = value;
    }
    __syncthreads();

    float largest = slots[0];
    for (int other = 1; other < WARPS; ++other) {
        largest = fmaxf(largest, slots[other]);
    }
    __syncthreads();
    return largest;
}

// The sum over the block, returned to every thread. `slots` holds 8 floats.
__device__ float blockSum(float value, float* slots)
{
    value = warpSum(value);
    int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        slots[warp] = value;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int other = 0; other < WARPS; ++other) {
        sum += slots[other];
    }
    __syncthreads();
    return sum;
}

// A softmax state: the maximum m and the sum s of exp(x - m). (-inf, 0) is the empty state.
struct SoftmaxState {
    float largest;  // m
    float sum;      // s
};

__device__ SoftmaxState mergeSoftmax(SoftmaxState first, SoftmaxState second)
{
    float top = fmaxf(first.largest, second.largest);
    if (top == -INFINITY) {
        return {top, 0.0f};
    }
    float firstPart = first.sum * expf(first.largest - top);
    float secondPart = second.sum * expf(second.largest - top);
    return {top, firstPart + secondPart};
}

// A Welford state: the count n, the mean, and the sum q of squared deviations.
struct WelfordState {
    float count;    // n
    float mean;
    float squares;  // q
};

__device__ WelfordState mergeWelford(WelfordState first, WelfordState second)
{
    float count = first.count + second.count;
    if (count == 0.0f) {
        return first;
    }
    float delta = second.mean - first.mean;
    float mean = first.mean + delta * second.count / count;
    float between = delta * delta * first.count * second.count / count;
    return {count, mean, first.squares + second.squares + between};
}

// Softmax without the maximum: exp(x) / sum of exp(x). Overflows for large x.
__global__ void softmaxNaive(const float* in, float* out, int cols)
{
    __shared__ float slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * cols;
    float* result = out + (size_t)blockIdx.x * cols;

    float sum = 0.0f;
    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        sum += expf(row[column]);
    }
    float total = blockSum(sum, slots);

    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        result[column] = expf(row[column]) / total;
    }
}

// 3 passes over the row: the maximum, the sum of exp(x - max), the outputs.
__global__ void softmaxThreePass(const float* in, float* out, int cols)
{
    __shared__ float slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * cols;
    float* result = out + (size_t)blockIdx.x * cols;

    float largest = -INFINITY;
    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        largest = fmaxf(largest, row[column]);
    }
    float top = blockMax(largest, slots);

    float sum = 0.0f;
    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        sum += expf(row[column] - top);
    }
    float total = blockSum(sum, slots);

    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        result[column] = expf(row[column] - top) / total;
    }
}

// 2 passes: a running maximum and a rescaled running sum, then the outputs.
__global__ void softmaxOnline(const float* in, float* out, int cols)
{
    __shared__ SoftmaxState slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * cols;
    float* result = out + (size_t)blockIdx.x * cols;

    SoftmaxState state = {-INFINITY, 0.0f};
    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        float value = row[column];
        if (value > state.largest) {
            state.sum = state.sum * expf(state.largest - value) + 1.0f;
            state.largest = value;
        } else {
            state.sum += expf(value - state.largest);
        }
    }

    for (int mask = 16; mask > 0; mask /= 2) {
        SoftmaxState other;
        other.largest = __shfl_xor_sync(0xffffffff, state.largest, mask);
        other.sum = __shfl_xor_sync(0xffffffff, state.sum, mask);
        state = mergeSoftmax(state, other);
    }
    if (threadIdx.x % 32 == 0) {
        slots[threadIdx.x / 32] = state;
    }
    __syncthreads();

    SoftmaxState all = slots[0];
    for (int other = 1; other < WARPS; ++other) {
        all = mergeSoftmax(all, slots[other]);
    }

    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        result[column] = expf(row[column] - all.largest) / all.sum;
    }
}

// 1 read: each thread keeps its 16 values of the row in registers.
__global__ void softmaxFused(const float* in, float* out)
{
    __shared__ float slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * COLS;
    float* result = out + (size_t)blockIdx.x * COLS;

    float kept[PER_THREAD];
    float largest = -INFINITY;
#pragma unroll
    for (int index = 0; index < PER_THREAD; ++index) {
        kept[index] = row[index * BLOCK + threadIdx.x];
        largest = fmaxf(largest, kept[index]);
    }
    float top = blockMax(largest, slots);

    float sum = 0.0f;
#pragma unroll
    for (int index = 0; index < PER_THREAD; ++index) {
        kept[index] = expf(kept[index] - top);
        sum += kept[index];
    }
    float scale = 1.0f / blockSum(sum, slots);

#pragma unroll
    for (int index = 0; index < PER_THREAD; ++index) {
        result[index * BLOCK + threadIdx.x] = kept[index] * scale;
    }
}

// Layer norm with variance = mean of squares - square of mean, in 1 pass of sums.
__global__ void layernormNaive(const float* in, const float* gamma, const float* beta,
                               float* out, int cols)
{
    __shared__ float slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * cols;
    float* result = out + (size_t)blockIdx.x * cols;

    float sum = 0.0f;
    float squares = 0.0f;
    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        float value = row[column];
        sum += value;
        squares += value * value;
    }
    float mean = blockSum(sum, slots) / cols;
    float variance = blockSum(squares, slots) / cols - mean * mean;
    float inverseStd = 1.0f / sqrtf(variance + EPS);

    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        result[column] = (row[column] - mean) * inverseStd * gamma[column] + beta[column];
    }
}

// Layer norm with Welford's running mean and squared deviations, merged per block.
__global__ void layernormWelford(const float* in, const float* gamma, const float* beta,
                                 float* out, int cols)
{
    __shared__ WelfordState slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * cols;
    float* result = out + (size_t)blockIdx.x * cols;

    WelfordState state = {0.0f, 0.0f, 0.0f};
    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        float value = row[column];
        state.count += 1.0f;
        float delta = value - state.mean;
        state.mean += delta / state.count;
        state.squares += delta * (value - state.mean);
    }

    for (int mask = 16; mask > 0; mask /= 2) {
        WelfordState other;
        other.count = __shfl_xor_sync(0xffffffff, state.count, mask);
        other.mean = __shfl_xor_sync(0xffffffff, state.mean, mask);
        other.squares = __shfl_xor_sync(0xffffffff, state.squares, mask);
        state = mergeWelford(state, other);
    }
    if (threadIdx.x % 32 == 0) {
        slots[threadIdx.x / 32] = state;
    }
    __syncthreads();

    WelfordState all = slots[0];
    for (int other = 1; other < WARPS; ++other) {
        all = mergeWelford(all, slots[other]);
    }
    float inverseStd = 1.0f / sqrtf(all.squares / cols + EPS);

    for (int column = threadIdx.x; column < cols; column += BLOCK) {
        float normalized = (row[column] - all.mean) * inverseStd;
        result[column] = normalized * gamma[column] + beta[column];
    }
}

// 1 read: the row stays in registers; the mean first, then the squared deviations.
__global__ void layernormFused(const float* in, const float* gamma, const float* beta,
                               float* out)
{
    __shared__ float slots[WARPS];
    const float* row = in + (size_t)blockIdx.x * COLS;
    float* result = out + (size_t)blockIdx.x * COLS;

    float kept[PER_THREAD];
    float sum = 0.0f;
#pragma unroll
    for (int index = 0; index < PER_THREAD; ++index) {
        kept[index] = row[index * BLOCK + threadIdx.x];
        sum += kept[index];
    }
    float mean = blockSum(sum, slots) / COLS;

    float squares = 0.0f;
#pragma unroll
    for (int index = 0; index < PER_THREAD; ++index) {
        kept[index] -= mean;
        squares += kept[index] * kept[index];
    }
    float inverseStd = 1.0f / sqrtf(blockSum(squares, slots) / COLS + EPS);

#pragma unroll
    for (int index = 0; index < PER_THREAD; ++index) {
        int column = index * BLOCK + threadIdx.x;
        result[column] = kept[index] * inverseStd * gamma[column] + beta[column];
    }
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Time of 1 run of `launch`, in milliseconds: the median of 7 batches of 20 runs,
// after 100 warm-up runs.
template <typename F>
double timeMs(F launch)
{
    for (int warmup = 0; warmup < 100; ++warmup) {
        launch();
    }
    check(cudaDeviceSynchronize());

    std::vector<double> batches;
    for (int batch = 0; batch < 7; ++batch) {
        double start = nowMs();
        for (int timed = 0; timed < 20; ++timed) {
            launch();
        }
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        batches.push_back((nowMs() - start) / 20);
    }

    std::sort(batches.begin(), batches.end());
    return batches[3];
}

// Uniform numbers in [low, high) from a small linear congruential generator.
std::vector<float> uniform(size_t n, float low, float high, unsigned seed)
{
    std::vector<float> v(n);
    unsigned state = seed;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i] = low + (high - low) * ((state >> 8) / 16777216.0f);
    }
    return v;
}

// The softmax of each row, computed in double.
std::vector<double> softmaxReference(const std::vector<float>& in, int cols)
{
    std::vector<double> out(in.size());
    for (size_t rowIndex = 0; rowIndex < in.size() / cols; ++rowIndex) {
        const float* row = in.data() + rowIndex * cols;
        double top = -INFINITY;
        for (int column = 0; column < cols; ++column) {
            top = std::max(top, (double)row[column]);
        }
        double total = 0.0;
        for (int column = 0; column < cols; ++column) {
            total += std::exp(row[column] - top);
        }
        for (int column = 0; column < cols; ++column) {
            out[rowIndex * cols + column] = std::exp(row[column] - top) / total;
        }
    }
    return out;
}

// The layer norm of each row, computed in double.
std::vector<double> layernormReference(const std::vector<float>& in,
                                       const std::vector<float>& gamma,
                                       const std::vector<float>& beta)
{
    int cols = gamma.size();
    std::vector<double> out(in.size());
    for (size_t rowIndex = 0; rowIndex < in.size() / cols; ++rowIndex) {
        const float* row = in.data() + rowIndex * cols;
        double mean = 0.0;
        for (int column = 0; column < cols; ++column) {
            mean += row[column];
        }
        mean /= cols;

        double variance = 0.0;
        for (int column = 0; column < cols; ++column) {
            variance += (row[column] - mean) * (row[column] - mean);
        }
        double inverseStd = 1.0 / std::sqrt(variance / cols + EPS);

        for (int column = 0; column < cols; ++column) {
            double normalized = (row[column] - mean) * inverseStd;
            out[rowIndex * cols + column] = normalized * gamma[column] + beta[column];
        }
    }
    return out;
}

// The largest difference between `gpu` and `exact`; NaN if any GPU value is NaN.
double maxError(const std::vector<float>& gpu, const std::vector<double>& exact)
{
    double worst = 0.0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        double diff = std::fabs(gpu[i] - exact[i]);
        if (std::isnan(diff)) {
            return NAN;
        }
        worst = std::max(worst, diff);
    }
    return worst;
}

bool allNan(const std::vector<float>& v)
{
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isnan(x); });
}

std::vector<float> toHost(const float* d, size_t n)
{
    std::vector<float> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

float* toDevice(const std::vector<float>& h)
{
    float* d;
    check(cudaMalloc(&d, h.size() * sizeof(float)));
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// Prints a matrix of `cols` columns, 1 row per line.
void printMatrix(const char* name, const std::vector<float>& v, int cols)
{
    std::printf("  %s\n", name);
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf(i % cols == 0 ? "    " : " ");
        if (std::isnan(v[i])) {
            std::printf("%8s", "NaN");
        } else {
            std::printf("%8.4f", v[i]);
        }
        if (i % cols == cols - 1) {
            std::printf("\n");
        }
    }
}

// Prints 1 line of a results table: bytes per element, time, bandwidth and error.
void report(const char* name, int bytes, double ms, double error)
{
    double gb = (double)ROWS * COLS * bytes / 1e9 / (ms / 1e3);
    std::printf("  %-11s %10d %7.3f %7.1f %11.2e\n", name, bytes, ms, gb, error);
}

int main()
{
    const char* header = "  kernel      bytes/elem      ms    GB/s   max error";

    // 1. The 2 x 4 example X, and X shifted by 100 and by 10,000.
    std::vector<float> x = {1, 2, 3, 4, 2, 4, 4, 5};
    std::vector<float> x100(8);
    std::vector<float> x10k(8);
    for (int i = 0; i < 8; ++i) {
        x100[i] = x[i] + 100.0f;
        x10k[i] = x[i] + 10000.0f;
    }
    float* ones = toDevice(std::vector<float>(4, 1.0f));
    float* zeros = toDevice(std::vector<float>(4, 0.0f));
    float* small = toDevice(x);
    float* shifted = toDevice(x100);
    float* far = toDevice(x10k);
    float* o;
    check(cudaMalloc(&o, 8 * sizeof(float)));

    std::printf("the 2 x 4 example X = [1 2 3 4; 2 4 4 5]\n");
    softmaxNaive<<<2, BLOCK>>>(small, o, 4);
    printMatrix("softmax without the maximum, X", toHost(o, 8), 4);
    softmaxNaive<<<2, BLOCK>>>(shifted, o, 4);
    std::vector<float> naiveBig = toHost(o, 8);
    printMatrix("softmax without the maximum, X + 100", naiveBig, 4);

    softmaxThreePass<<<2, BLOCK>>>(small, o, 4);
    std::vector<float> threeX = toHost(o, 8);
    printMatrix("softmax, 3 passes, X", threeX, 4);
    softmaxThreePass<<<2, BLOCK>>>(shifted, o, 4);
    std::vector<float> threeBig = toHost(o, 8);
    printMatrix("softmax, 3 passes, X + 100", threeBig, 4);
    softmaxOnline<<<2, BLOCK>>>(small, o, 4);
    std::vector<float> onlineX = toHost(o, 8);
    printMatrix("softmax, online, X", onlineX, 4);

    layernormNaive<<<2, BLOCK>>>(small, ones, zeros, o, 4);
    std::vector<float> sqX = toHost(o, 8);
    printMatrix("layer norm, sum of squares, X", sqX, 4);
    layernormNaive<<<2, BLOCK>>>(far, ones, zeros, o, 4);
    std::vector<float> sqFar = toHost(o, 8);
    printMatrix("layer norm, sum of squares, X + 10000", sqFar, 4);
    layernormWelford<<<2, BLOCK>>>(far, ones, zeros, o, 4);
    std::vector<float> welfordFar = toHost(o, 8);
    printMatrix("layer norm, Welford, X + 10000", welfordFar, 4);

    std::vector<double> p = softmaxReference(x, 4);
    std::vector<double> y = layernormReference(x, std::vector<float>(4, 1.0f),
                                               std::vector<float>(4, 0.0f));
    bool exampleOk = maxError(threeX, p) < 1e-6 && maxError(threeBig, p) < 1e-6;
    exampleOk = exampleOk && maxError(onlineX, p) < 1e-6 && maxError(sqX, y) < 1e-4;
    exampleOk = exampleOk && maxError(welfordFar, y) < 1e-4;
    exampleOk = exampleOk && allNan(naiveBig) && allNan(sqFar);

    // 2. Softmax of 4,096 rows of 4,096 logits in [-10, 10): 16 KiB per row.
    size_t n = (size_t)ROWS * COLS;
    std::vector<float> logits = uniform(n, -10.0f, 10.0f, 1);
    float* in = toDevice(logits);
    float* out;
    check(cudaMalloc(&out, n * sizeof(float)));
    std::vector<double> exact = softmaxReference(logits, COLS);

    std::printf("softmax of %d rows x %d columns, median of 7 x 20 runs\n", ROWS, COLS);
    std::printf("%s\n", header);

    double ms = timeMs([&] { softmaxThreePass<<<ROWS, BLOCK>>>(in, out, COLS); });
    double threeError = maxError(toHost(out, n), exact);
    report("3 passes", 16, ms, threeError);

    ms = timeMs([&] { softmaxOnline<<<ROWS, BLOCK>>>(in, out, COLS); });
    double onlineError = maxError(toHost(out, n), exact);
    report("online", 12, ms, onlineError);

    ms = timeMs([&] { softmaxFused<<<ROWS, BLOCK>>>(in, out); });
    double fusedError = maxError(toHost(out, n), exact);
    report("fused", 8, ms, fusedError);

    // 3. The same logits as 256 rows of 65,536: 256 KiB per row.
    int wide = 65536;
    int longRows = n / wide;
    exact = softmaxReference(logits, wide);

    std::printf("softmax of %d rows x %d columns, median of 7 x 20 runs\n", longRows, wide);
    std::printf("%s\n", header);

    ms = timeMs([&] { softmaxThreePass<<<longRows, BLOCK>>>(in, out, wide); });
    double threeLongError = maxError(toHost(out, n), exact);
    report("3 passes", 16, ms, threeLongError);

    ms = timeMs([&] { softmaxOnline<<<longRows, BLOCK>>>(in, out, wide); });
    double onlineLongError = maxError(toHost(out, n), exact);
    report("online", 12, ms, onlineLongError);

    // 4. Layer norm of 4,096 rows of values 1,000 + [-1, 1).
    std::vector<float> values = uniform(n, 999.0f, 1001.0f, 2);
    std::vector<float> gamma(COLS);
    std::vector<float> beta(COLS);
    for (int j = 0; j < COLS; ++j) {
        gamma[j] = 0.5f + 0.25f * (j % 3);
        beta[j] = 0.1f * (j % 5) - 0.2f;
    }
    check(cudaMemcpy(in, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    float* g = toDevice(gamma);
    float* b = toDevice(beta);
    exact = layernormReference(values, gamma, beta);

    std::printf("layer norm of %d rows x %d columns, values 1,000 + [-1, 1)\n", ROWS, COLS);
    std::printf("%s\n", header);

    ms = timeMs([&] { layernormNaive<<<ROWS, BLOCK>>>(in, g, b, out, COLS); });
    double naiveError = maxError(toHost(out, n), exact);
    report("sum, sum x2", 12, ms, naiveError);

    ms = timeMs([&] { layernormWelford<<<ROWS, BLOCK>>>(in, g, b, out, COLS); });
    double welfordError = maxError(toHost(out, n), exact);
    report("Welford", 12, ms, welfordError);

    ms = timeMs([&] { layernormFused<<<ROWS, BLOCK>>>(in, g, b, out); });
    double normError = maxError(toHost(out, n), exact);
    report("fused", 8, ms, normError);

    // 5. Softmax within 1e-6, layer norm within 1e-3; the sum of squares must be off.
    bool softmaxOk = threeError < 1e-6 && onlineError < 1e-6 && fusedError < 1e-6;
    softmaxOk = softmaxOk && threeLongError < 1e-6 && onlineLongError < 1e-6;
    bool layernormOk = welfordError < 1e-3 && normError < 1e-3 && naiveError > 0.1;
    bool ok = exampleOk && softmaxOk && layernormOk;
    std::printf("%s\n", ok ? "PASSED" : "FAILED");

    cudaFree(ones);
    cudaFree(zeros);
    cudaFree(small);
    cudaFree(shifted);
    cudaFree(far);
    cudaFree(o);
    cudaFree(in);
    cudaFree(out);
    cudaFree(g);
    cudaFree(b);
    return ok ? 0 : 1;
}
