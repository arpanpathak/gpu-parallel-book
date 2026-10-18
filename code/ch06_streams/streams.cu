// Chapter 6: asynchronous launches, events, a pipeline on 2 streams, and a CUDA graph.
// Build: nvcc -arch=native streams.cu -o streams

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

// Some arithmetic per element, so that a kernel takes about as long as its copies.
__global__ void work(float* x, int n, int steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = x[i];
    for (int k = 0; k < steps; ++k) {
        v = fmaf(v, 0.999f, 0.5f);
    }
    x[i] = v;
}

__global__ void empty() {}

// 1. A launch returns before the kernel finishes; events time the kernel on the GPU.
void asyncLaunch(float* d, int n, int steps)
{
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    work<<<(n + 255) / 256, 256>>>(d, n, steps);       // warm up
    check(cudaDeviceSynchronize());

    double t0 = nowMs();
    check(cudaEventRecord(start));
    work<<<(n + 255) / 256, 256>>>(d, n, steps);
    check(cudaEventRecord(stop));
    double launched = nowMs() - t0;

    check(cudaEventSynchronize(stop));
    double finished = nowMs() - t0;

    float gpuMs;
    check(cudaEventElapsedTime(&gpuMs, start, stop));

    std::printf("a launch of a kernel over %d MiB\n", int(n * sizeof(float) >> 20));
    std::printf("  launch call returned after   %8.3f ms\n", launched);
    std::printf("  kernel finished after        %8.3f ms\n", finished);
    std::printf("  kernel time by events        %8.3f ms\n", gpuMs);

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
}

// 2. Copy in, compute, copy out: all at once on 1 stream, or in chunks on 2 streams.
void enqueue(float* host, float* d, int n, int steps, int chunks, cudaStream_t* streams)
{
    int part = n / chunks;
    size_t bytes = part * sizeof(float);

    for (int c = 0; c < chunks; ++c) {
        cudaStream_t s = streams[c % 2];
        float* h = host + c * part;
        float* g = d + c * part;

        check(cudaMemcpyAsync(g, h, bytes, cudaMemcpyHostToDevice, s));
        work<<<(part + 255) / 256, 256, 0, s>>>(g, part, steps);
        check(cudaMemcpyAsync(h, g, bytes, cudaMemcpyDeviceToHost, s));
    }
}

// Average time of 5 pipelines, in milliseconds.
double pipeline(float* host, float* d, int n, int steps, int chunks, cudaStream_t* streams)
{
    check(cudaDeviceSynchronize());
    double t0 = nowMs();

    for (int r = 0; r < 5; ++r) {
        enqueue(host, d, n, steps, chunks, streams);
    }

    check(cudaDeviceSynchronize());
    return (nowMs() - t0) / 5;
}

// The 3 stages of the 1-stream run, timed separately with events.
void stages(float* host, float* d, int n, int steps, cudaStream_t s)
{
    cudaEvent_t e[4];
    for (auto& ev : e) check(cudaEventCreate(&ev));
    size_t bytes = n * sizeof(float);

    check(cudaEventRecord(e[0], s));
    check(cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, s));
    check(cudaEventRecord(e[1], s));
    work<<<(n + 255) / 256, 256, 0, s>>>(d, n, steps);
    check(cudaEventRecord(e[2], s));
    check(cudaMemcpyAsync(host, d, bytes, cudaMemcpyDeviceToHost, s));
    check(cudaEventRecord(e[3], s));
    check(cudaEventSynchronize(e[3]));

    const char* names[] = {"copy in", "kernel", "copy out"};
    for (int k = 0; k < 3; ++k) {
        float ms;
        check(cudaEventElapsedTime(&ms, e[k], e[k + 1]));
        std::printf("  %-9s %7.2f ms\n", names[k], ms);
    }

    for (auto& ev : e) check(cudaEventDestroy(ev));
}

// 3. 1,000 launches of an empty kernel: one by one, and replayed from a graph.
void graphs(cudaStream_t s)
{
    const int launches = 1000;
    empty<<<1, 32, 0, s>>>();
    check(cudaStreamSynchronize(s));

    double t0 = nowMs();
    for (int k = 0; k < launches; ++k) {
        empty<<<1, 32, 0, s>>>();
    }
    check(cudaStreamSynchronize(s));
    double direct = (nowMs() - t0) / launches;

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < launches; ++k) {
        empty<<<1, 32, 0, s>>>();
    }
    check(cudaStreamEndCapture(s, &graph));
    check(cudaGraphInstantiate(&exec, graph, 0));

    check(cudaGraphLaunch(exec, s));                    // warm up
    check(cudaStreamSynchronize(s));

    t0 = nowMs();
    check(cudaGraphLaunch(exec, s));
    check(cudaStreamSynchronize(s));
    double replay = (nowMs() - t0) / launches;

    std::printf("1,000 launches of an empty kernel, time per kernel\n");
    std::printf("  launched one by one   %6.2f us\n", direct * 1e3);
    std::printf("  replayed from a graph %6.2f us\n", replay * 1e3);

    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
}

int main()
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d copy engine(s)\n", prop.name, prop.asyncEngineCount);

    const int n = 32 << 20;                   // 33,554,432 floats, 128 MiB
    const int steps = 256;

    float* host;
    float* d;
    check(cudaMallocHost(&host, n * sizeof(float)));
    check(cudaMalloc(&d, n * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        host[i] = 1.0f;
    }

    cudaStream_t streams[2];
    check(cudaStreamCreate(&streams[0]));
    check(cudaStreamCreate(&streams[1]));

    pipeline(host, d, n, steps, 8, streams);            // warm up: the clocks rise under load
    asyncLaunch(d, n, steps);

    std::printf("copy in, kernel, copy out over 128 MiB\n");
    stages(host, d, n, steps, streams[0]);
    std::printf("  1 stream, 1 chunk        %7.2f ms\n", pipeline(host, d, n, steps, 1, streams));
    std::printf("  2 streams, 8 chunks      %7.2f ms\n", pipeline(host, d, n, steps, 8, streams));
    std::printf("  2 streams, 32 chunks     %7.2f ms\n", pipeline(host, d, n, steps, 32, streams));

    graphs(streams[0]);

    check(cudaStreamDestroy(streams[0]));
    check(cudaStreamDestroy(streams[1]));
    check(cudaFreeHost(host));
    check(cudaFree(d));
}
