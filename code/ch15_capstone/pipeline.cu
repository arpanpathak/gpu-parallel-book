// Chapter 15 - the capstone GPU image-processing pipeline.
// Kernels verbatim from the book (15.2-15.4); scaleEdges from 15.4.1;
// histogram from 8.8. The differential test implements 15.8.
// Build: nvcc -arch=sm_90 pipeline.cu -o pipeline
// Run:   ./pipeline <width> <height> <frames>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// rgbToGray: 3 bytes/pixel RGB (uchar3) -> 1 float/pixel greyscale.
// Consecutive threads -> consecutive pixels -> coalesced reads and writes.
// The weights are the standard BT.601 luma coefficients; they sum to 1.0,
// so no scaling is needed and a constant-grey input maps to itself.
// ---------------------------------------------------------------------------
__global__ void rgbToGray(const uchar3* rgb, float* gray, int numPixels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPixels)
    {
        const uchar3 px = rgb[i];                 // 12-byte read, coalesced
        // uchar3 components are 0..255; multiply in float to avoid
        // integer truncation. The 0.114f/0.587f/0.299f order mirrors the
        // canonical definition.
        gray[i] = 0.299f * static_cast<float>(px.x)
                + 0.587f * static_cast<float>(px.y)
                + 0.114f * static_cast<float>(px.z);
    }
}

__global__ void blurH(const float* in, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        const float* row = in + y * width;
        // Weights, left to right:  [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]
        const float w[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};
        float acc = 0.0f;
        // Each tap clamps its index to the row bounds independently:
        // interior pixels use the exact stencil; edge pixels replicate.
        #pragma unroll
        for (int t = -2; t <= 2; ++t)
        {
            const int sx = min(max(x + t, 0), width - 1);
            acc += w[t + 2] * row[sx];
        }
        out[y * width + x] = acc;
    }
}

// ---------------------------------------------------------------------------
// blurV: vertical 5-tap Gaussian - the twin of blurH with x/y swapped.
// Each thread owns output pixel (y, x) and reads (y-2 .. y+2, x); the
// stencil walks down the column, clamping each tap at the image borders
// independently (replicate-edge policy), exactly as blurH does.
// ---------------------------------------------------------------------------
__global__ void blurV(const float* in, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        // Weights, top to bottom: [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]
        const float w[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};
        float acc = 0.0f;
        #pragma unroll
        for (int t = -2; t <= 2; ++t)
        {
            const int sy = min(max(y + t, 0), height - 1);
            acc += w[t + 2] * in[sy * width + x];
        }
        out[y * width + x] = acc;
    }
}

// ---------------------------------------------------------------------------
// sobel: magnitude of the gradient. Gx = (row -1 + 2*row0 + row1) convolved
// with [-1, 0, 1]; Gy = the transpose. We compute both 3x3 convolutions and
// the magnitude sqrt(Gx^2 + Gy^2) per pixel.
// ---------------------------------------------------------------------------
__global__ void sobel(const float* in, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        // Clamp the 3x3 neighbourhood at the borders (replicate-edge).
        const int xm = max(x - 1, 0), xp = min(x + 1, width  - 1);
        const int ym = max(y - 1, 0), yp = min(y + 1, height - 1);

        const float* r0 = in + ym * width;
        const float* r1 = in + y  * width;
        const float* r2 = in + yp * width;

        // Horizontal derivative Gx: [-1 0 1] across each of the three rows,
        // weighted [1 2 1] down the columns.
        const float gx = (r2[xp] + 2.0f * r1[xp] + r0[xp])
                       - (r2[xm] + 2.0f * r1[xm] + r0[xm]);
        // Vertical derivative Gy: [-1 0 1] down the columns,
        // weighted [1 2 1] across the rows.
        const float gy = (r2[xm] + 2.0f * r2[x] + r2[xp])
                       - (r0[xm] + 2.0f * r0[x] + r0[xp]);

        // Magnitude. sqrtf is the device-side square root; the SFU
        // approximation is acceptable here (we scale to uchar output).
        out[y * width + x] = sqrtf(gx * gx + gy * gy);
    }
}

// ---------------------------------------------------------------------------
// scaleEdges (15.4.1): clamp the float edge magnitude to [0, 255] and store
// as uchar. The histogram of Chapter 8 counts uchar bins, so the scaling is
// what makes the histogram a histogram of EDGE INTENSITIES rather than of
// the bytes of a float.
// ---------------------------------------------------------------------------
__global__ void scaleEdges(const float* in, unsigned char* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int v = static_cast<int>(in[i] + 0.5f);   // round, don't truncate
        out[i] = static_cast<unsigned char>(min(max(v, 0), 255));
    }
}

#define BINS 256
#define TILE 256

__global__ void histogramPrivatised(const unsigned char* data, int* g_hist,
                                    int n)
{
    // Private histogram for THIS block, in shared memory.
    __shared__ int s_hist[BINS];

    // Initialise the private histogram (all threads help; one barrier).
    for (int b = threadIdx.x; b < BINS; b += TILE) s_hist[b] = 0;
    __syncthreads();          // all bins zeroed before any thread counts

    // Accumulate. Each thread counts its elements into shared memory.
    // Shared-memory atomics are fast; contention is spread across BINS.
    for (int i = blockIdx.x * TILE + threadIdx.x; i < n; i += gridDim.x * TILE)
    {
        const int bin = data[i];
        atomicAdd(&s_hist[bin], 1);
    }

    __syncthreads();          // all counts complete before folding

    // Fold: one thread per bin adds this block's count to global memory.
    for (int b = threadIdx.x; b < BINS; b += TILE)
        if (s_hist[b] != 0)             // skip empty bins: less global traffic
            atomicAdd(&g_hist[b], s_hist[b]);
}

// ---------------------------------------------------------------------------
// CPU reference (Chapter 15, 15.8): the same five stages as plain loops,
// trivially correct and deliberately slow. clampI is the host-side clamp.
// ---------------------------------------------------------------------------
static int clampI(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void cpuRgbToGray(const unsigned char* rgb, float* gray, int n)
{
    for (int i = 0; i < n; ++i)
        gray[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1]
                + 0.114f * rgb[3 * i + 2];
}

static void cpuBlurH(const float* in, float* out, int width, int height)
{
    const float w[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            float acc = 0.0f;
            for (int t = -2; t <= 2; ++t)
                acc += w[t + 2] * in[y * width + clampI(x + t, 0, width - 1)];
            out[y * width + x] = acc;
        }
}

static void cpuBlurV(const float* in, float* out, int width, int height)
{
    const float w[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            float acc = 0.0f;
            for (int t = -2; t <= 2; ++t)
                acc += w[t + 2] * in[clampI(y + t, 0, height - 1) * width + x];
            out[y * width + x] = acc;
        }
}

static void cpuSobel(const float* in, float* out, int width, int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const int xm = clampI(x - 1, 0, width - 1), xp = clampI(x + 1, 0, width - 1);
            const int ym = clampI(y - 1, 0, height - 1), yp = clampI(y + 1, 0, height - 1);
            const float* r0 = in + ym * width;
            const float* r1 = in + y  * width;
            const float* r2 = in + yp * width;
            const float gx = (r2[xp] + 2.0f * r1[xp] + r0[xp])
                           - (r2[xm] + 2.0f * r1[xm] + r0[xm]);
            const float gy = (r2[xm] + 2.0f * r2[x] + r2[xp])
                           - (r0[xm] + 2.0f * r0[x] + r0[xp]);
            out[y * width + x] = std::sqrt(gx * gx + gy * gy);
        }
}

// ---------------------------------------------------------------------------
// Differential test (15.8): run the device pipeline and the CPU reference on
// the same small image and compare every stage. Returns 0 on success.
// ---------------------------------------------------------------------------
static int verifyAgainstCpu()
{
    const int w = 96, h = 64, n = w * h;

    // A deterministic test pattern (period 251 avoids byte-rollover edges
    // aligning with the kernel's index arithmetic).
    std::vector<unsigned char> rgb(n * 3);
    for (int i = 0; i < n; ++i)
    {
        rgb[3 * i + 0] = static_cast<unsigned char>((i * 7 + 1) % 251);
        rgb[3 * i + 1] = static_cast<unsigned char>((i * 13 + 5) % 251);
        rgb[3 * i + 2] = static_cast<unsigned char>((i * 29 + 9) % 251);
    }

    // CPU reference:
    std::vector<float> cGray(n), cBlur(n), cBlur2(n), cEdge(n);
    cpuRgbToGray(rgb.data(), cGray.data(), n);
    cpuBlurH(cGray.data(), cBlur.data(), w, h);
    cpuBlurV(cBlur.data(), cBlur2.data(), w, h);
    cpuSobel(cBlur2.data(), cEdge.data(), w, h);

    // GPU pipeline (single frame, default stream):
    unsigned char* d_rgb;  unsigned char* d_edge8;
    float *d_gray, *d_blur, *d_blur2, *d_edge;
    CHECK(cudaMalloc(&d_rgb,   (size_t)n * 3));
    CHECK(cudaMalloc(&d_gray,  (size_t)n * 4));
    CHECK(cudaMalloc(&d_blur,  (size_t)n * 4));
    CHECK(cudaMalloc(&d_blur2, (size_t)n * 4));
    CHECK(cudaMalloc(&d_edge,  (size_t)n * 4));
    CHECK(cudaMalloc(&d_edge8, (size_t)n));
    CHECK(cudaMemcpy(d_rgb, rgb.data(), (size_t)n * 3, cudaMemcpyHostToDevice));

    const dim3 b1(256); const dim3 g1((n + 255) / 256);
    const dim3 b2(32, 8); const dim3 g2((w + 31) / 32, (h + 7) / 8);
    rgbToGray<<<g1, b1>>>(reinterpret_cast<const uchar3*>(d_rgb), d_gray, n);
    blurH    <<<g2, b2>>>(d_gray,  d_blur,  w, h);
    blurV    <<<g2, b2>>>(d_blur,  d_blur2, w, h);
    sobel    <<<g2, b2>>>(d_blur2, d_edge,  w, h);
    scaleEdges<<<g1, b1>>>(d_edge, d_edge8, n);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> gGray(n), gBlur(n), gBlur2(n), gEdge(n);
    std::vector<unsigned char> gEdge8(n);
    CHECK(cudaMemcpy(gGray.data(),  d_gray,  (size_t)n * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gBlur.data(),  d_blur,  (size_t)n * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gBlur2.data(), d_blur2, (size_t)n * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gEdge.data(),  d_edge,  (size_t)n * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gEdge8.data(), d_edge8, (size_t)n,     cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_rgb));   CHECK(cudaFree(d_gray));  CHECK(cudaFree(d_blur));
    CHECK(cudaFree(d_blur2)); CHECK(cudaFree(d_edge));  CHECK(cudaFree(d_edge8));

    // Compare stage by stage. Float stages tolerate 1e-3 (FMA contraction
    // and the SFU sqrtf differ from the CPU in the last bits). The uchar
    // edge map must match within ±1: the device's approximate sqrtf can
    // round differently at a .5 boundary, but anything beyond one count is
    // a real error.
    double maxErr = 0.0;
    for (int i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)gGray[i]  - cGray[i]));
    for (int i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)gBlur[i]  - cBlur[i]));
    for (int i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)gBlur2[i] - cBlur2[i]));
    for (int i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)gEdge[i]  - cEdge[i]));
    int byteErr = 0;
    for (int i = 0; i < n; ++i)
    {
        const int cpuVal = clampI((int)(cEdge[i] + 0.5f), 0, 255);
        const int diff   = (int)gEdge8[i] - cpuVal;
        if (diff > 1 || diff < -1) ++byteErr;   // allow ±1 at rounding boundaries
    }

    std::printf("differential test: max float error = %g, %d uchar mismatches\n",
                maxErr, byteErr);
    if (maxErr > 1e-3 || byteErr > 0)
    {
        std::fprintf(stderr, "DIFFERENTIAL TEST FAILED\n");
        return 1;
    }
    std::printf("differential test PASSED (GPU matches CPU reference)\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Host side: streamed double-buffered pipeline (Chapter 6, 6.5).
// Each frame: H2D copy of the next frame overlaps the kernels of this one.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    const int width  = (argc > 1) ? std::atoi(argv[1]) : 1920;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 1080;
    const int frames = (argc > 3) ? std::atoi(argv[3]) : 60;
    const int numPixels = width * height;
    const size_t rgbBytes = numPixels * 3 * sizeof(unsigned char);
    const size_t grayBytes = numPixels * sizeof(float);
    const size_t edgeBytes = numPixels * sizeof(unsigned char);   // scaled edges

    // Verify the pipeline against the CPU reference before running frames:
    if (verifyAgainstCpu() != 0) return 1;

    // Pinned host buffers (Chapter 4): required for asynchronous copies.
    unsigned char* h_rgb[2];
    unsigned char* h_edges[2];
    for (int b = 0; b < 2; ++b)
    {
        CHECK(cudaMallocHost(&h_rgb[b],   rgbBytes));
        CHECK(cudaMallocHost(&h_edges[b], edgeBytes));
        for (int i = 0; i < numPixels * 3; ++i)
            h_rgb[b][i] = static_cast<unsigned char>(i % 251);  // test pattern
    }

    unsigned char* d_rgb[2];
    unsigned char* d_edges8;
    float* d_gray; float* d_blurred; float* d_blurred2; float* d_edges;
    int* d_hist;
    for (int b = 0; b < 2; ++b)
        CHECK(cudaMalloc(&d_rgb[b], rgbBytes));
    CHECK(cudaMalloc(&d_gray,     grayBytes));
    CHECK(cudaMalloc(&d_blurred,  grayBytes));
    CHECK(cudaMalloc(&d_blurred2, grayBytes));
    CHECK(cudaMalloc(&d_edges,    grayBytes));
    CHECK(cudaMalloc(&d_edges8,   edgeBytes));
    CHECK(cudaMalloc(&d_hist,     256 * sizeof(int)));
    // cudaMalloc does NOT zero: the histogram accumulates across frames with
    // atomicAdd, so it must start at zero (one memset before the loop).
    CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    cudaStream_t sCopy, sCompute;
    cudaEvent_t copyDone[2];
    CHECK(cudaStreamCreate(&sCopy));
    CHECK(cudaStreamCreate(&sCompute));
    for (int b = 0; b < 2; ++b) CHECK(cudaEventCreate(&copyDone[b]));

    // Launch configuration (Chapter 3).
    const dim3 block1(256);
    const dim3 grid1((numPixels + 255) / 256);
    const dim3 block2(32, 8);
    const dim3 grid2((width + 31) / 32, (height + 7) / 8);

    // Prime the pipeline: upload frame 0.
    CHECK(cudaMemcpyAsync(d_rgb[0], h_rgb[0], rgbBytes,
                          cudaMemcpyHostToDevice, sCopy));
    CHECK(cudaEventRecord(copyDone[0], sCopy));

    for (int frame = 0; frame < frames; ++frame)
    {
        const int cur = frame % 2;
        const int nxt = (frame + 1) % 2;

        if (frame + 1 < frames)
        {
            CHECK(cudaMemcpyAsync(d_rgb[nxt], h_rgb[nxt], rgbBytes,
                                  cudaMemcpyHostToDevice, sCopy));
            CHECK(cudaEventRecord(copyDone[nxt], sCopy));
        }

        // Dependency: compute waits for THIS frame's copy (its event was
        // recorded last iteration, or in the prime above).
        CHECK(cudaStreamWaitEvent(sCompute, copyDone[cur], 0));

        rgbToGray<<<grid1, block1, 0, sCompute>>>(
            reinterpret_cast<const uchar3*>(d_rgb[cur]), d_gray, numPixels);
        blurH    <<<grid2, block2, 0, sCompute>>>(d_gray, d_blurred,  width, height);
        blurV    <<<grid2, block2, 0, sCompute>>>(d_blurred, d_blurred2, width, height);
        sobel    <<<grid2, block2, 0, sCompute>>>(d_blurred2, d_edges,  width, height);
        scaleEdges<<<grid1, block1, 0, sCompute>>>(d_edges, d_edges8, numPixels);
        histogramPrivatised<<<grid2, block2, 0, sCompute>>>(d_edges8, d_hist, numPixels);

        CHECK(cudaMemcpyAsync(h_edges[cur], d_edges8, edgeBytes,
                              cudaMemcpyDeviceToHost, sCompute));
    }
    CHECK(cudaStreamSynchronize(sCompute));

    // Report a sanity statistic (Chapter 8: the histogram of edge
    // intensities - meaningful now that edges are uchar, 15.4.1).
    int h_hist[256];
    CHECK(cudaMemcpy(h_hist, d_hist, sizeof(h_hist), cudaMemcpyDeviceToHost));
    long total = 0;
    for (int b = 0; b < 256; ++b) total += h_hist[b];
    std::printf("frames=%d, histogram total=%ld (expected %d)\n",
                frames, total, frames * numPixels);

    for (int b = 0; b < 2; ++b)
    {
        CHECK(cudaFreeHost(h_rgb[b])); CHECK(cudaFreeHost(h_edges[b]));
        CHECK(cudaFree(d_rgb[b]));     CHECK(cudaEventDestroy(copyDone[b]));
    }
    CHECK(cudaFree(d_gray)); CHECK(cudaFree(d_blurred)); CHECK(cudaFree(d_blurred2));
    CHECK(cudaFree(d_edges)); CHECK(cudaFree(d_edges8)); CHECK(cudaFree(d_hist));
    CHECK(cudaStreamDestroy(sCopy)); CHECK(cudaStreamDestroy(sCompute));
    return 0;
}
