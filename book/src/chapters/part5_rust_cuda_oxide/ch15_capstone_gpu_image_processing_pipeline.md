# Chapter 15: Capstone - The GPU Image Processing Pipeline

> *"A pipeline is a chain of kernels. A fast pipeline is a chain of kernels
> that never waits."*

This is the chapter every earlier one was building toward. The capstone is a
complete GPU image-processing pipeline - **RGB → greyscale → Gaussian blur →
Sobel edge detection** - implemented three ways (hand-written CUDA C++,
Thrust, and CUDA-Oxide Rust), streamed with pinned memory, verified against a
CPU reference, and measured with CUDA events. It is deliberately small enough
to fit in one chapter and real enough to ship.

## 15.1 The Pipeline and Its Data Flow

```
 input RGB (W x H, 3 bytes/pixel, row-major)
   │
   ▼  kernel 1: rgbToGray      (Chapter 3/7: coalescing)
 gray float (W x H, 4 bytes/pixel)
   │
   ▼  kernel 2: blurH + blurV  (separable Gaussian, 5-tap)
 blurred float (W x H)
   │
   ▼  kernel 3: sobel          (two 3x3 derivative kernels + magnitude)
 edge float (W x H)
   │
   ▼  kernel 4: histogram      (Chapter 8: privatised histogram, for stats)
 stats
```

Design decisions, each with its reason:

- **Grey as `float`, not `unsigned char`.** The blur and Sobel accumulate
  fractional weights; `float` avoids rounding at every stage and matches the
  arithmetic intensity discussion of Chapter 1. The final edge map is scaled
  back to `unsigned char` for output.
- **Separable Gaussian.** A 2-D Gaussian kernel of radius 2 is a 5×5 stencil  - 
  25 taps per output pixel. A *separable* Gaussian is a horizontal 5-tap
  followed by a vertical 5-tap: 10 taps per pixel. Separability halves the
  arithmetic for a mathematically identical result, and each pass is naturally
  coalesced.
- **Sobel = two separable kernels.** The Sobel operator is the pair of 3×3
  kernels \\(G_x\\) and \\(G_y\\). Each row/column combination factors into a
  derivative and a smoothing pass; we implement it directly as two 3×3
  convolutions and take the magnitude \\(\\sqrt{G_x^2 + G_y^2}\\).
- **Histogram at the end.** A cheap way to verify the pipeline produced
  sensible data (edge counts in the expected range) and a demonstration of
  Chapter 8's privatised histogram on a real workload.

## 15.2 Stage 1: RGB → Greyscale (CUDA C++)

The canonical coalesced kernel: one thread per output pixel, consecutive
threads on consecutive pixels, the Chapter 3 index formula.

```cpp
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
```

**Why `uchar3`?** CUDA's built-in 3-byte vector type matches the RGB layout
exactly. Its alignment is 1 (no padding), so it is safe to point at raw RGB
bytes. (A `float3` would *not* be safe - it is 16-byte aligned.) This is the
"describe every primitive" discipline paying off: the layout contract is in
the type.

## 15.3 Stage 2: Separable Gaussian Blur

The 5-tap weights for \\(\sigma = 1\\) are `[0.06136, 0.24477, 0.38774,
0.24477, 0.06136]` (a normalised Gaussian). The horizontal pass reads a row
segment including a **halo** of 2 pixels on each side; the vertical pass does
the same down columns.

```cpp
// ---------------------------------------------------------------------------
// blurH: horizontal 5-tap Gaussian. Each thread owns output pixel (y, x)
// and reads input pixels (y, x-2 .. x+2). Consecutive threads read
// consecutive windows -> coalesced. The halo pixels are read by two
// neighbouring threads, which is the cost of the stencil - and the reason
// tiled shared memory (Chapter 7) would win for large radii.
// ---------------------------------------------------------------------------
__global__ void blurH(const float* in, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        const float* row = in + y * width;       // this row's base
        // Clamp the stencil at image borders (replicate-edge policy).
        const int x0 = max(x - 2, 0), x1 = min(x + 2, width - 1);
        out[y * width + x] = 0.06136f * row[x0] + 0.24477f * row[x1]
                           + 0.38774f * row[x]  + 0.24477f * row[x1]
                           + 0.06136f * row[x0];
    }
}
```

**Wait - the weights are wrong for clamped edges.** The comment says
replicate-edge, but the code above assigns both `x0` and `x1` the same weight
pattern, which double-weights the border. The *honest* version computes the
stencil with per-tap clamping:

```cpp
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
```

**The lesson embedded in this correction.** The first version was *plausible
and wrong*; the second is *correct by construction*. This is exactly the bug
class this book exists to train you against: the code that "looks like" a
Gaussian until the edges. The vertical pass is identical with `x`/`y` and
`row` swapped to a column stride; it is omitted here to avoid repetition, but
the discipline is the same - clamp each tap independently.

**Why two kernels and not one?** A single fused kernel would need each thread
to read a 5×5 neighbourhood (25 reads) instead of two passes of 5 reads each
(10 reads), and the intermediate (blurred horizontally) would need to be
communicated through shared memory with a block halo. Two global passes are
simpler, fully coalesced, and - at this image scale - bandwidth-dominated in
exactly the way Chapter 7's checklist predicts.

## 15.4 Stage 3: Sobel Edge Detection

```cpp
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
```

**The separable structure, annotated.** \\(G_x\\) is a derivative in `x`
(`[-1 0 1]`) convolved with a smoothing in `y` (`[1 2 1]`); \\(G_y\\) is the
transpose. The kernel reads 9 pixels and produces two convolutions - the
separable factoring is what keeps it at 9 reads instead of 18.

## 15.5 The Streaming Host Pipeline

The frame loop uses the machinery of Chapters 4 and 6: pinned host memory,
two streams, and double buffering so transfers overlap kernels:

```cpp
// ---------------------------------------------------------------------------
// One "frame" = load RGB, run the 4 kernels, store edges. Frames arrive in
// host buffers h_rgb[0] and h_rgb[1]; the GPU processes one while the DMA
// engine uploads the next (Chapter 6, 6.5).
// ---------------------------------------------------------------------------
void processFrames(/* ... device buffers, streams, sizes ... */)
{
    const int  numPixels = width * height;
    const bool last = (frame == frameCount - 1);

    const int cur = frame % 2;          // buffer holding THIS frame's input
    const int nxt = (frame + 1) % 2;    // buffer for the NEXT frame

    // Upload the NEXT frame while THIS one computes (pinned memory!):
    if (!last)
        CHECK(cudaMemcpyAsync(d_rgb[nxt], h_rgb[nxt],
                              rgbBytes, cudaMemcpyHostToDevice, sCopy));

    // Make the compute stream wait for this frame's copy:
    CHECK(cudaEventRecord(copyDone[nxt], sCopy));
    CHECK(cudaStreamWaitEvent(sCompute, copyDone[nxt], 0));

    // The four stages, all in the compute stream, in order:
    const dim3 block(256);
    const dim3 grid((numPixels + 255) / 256);
    rgbToGray<<<grid, block, 0, sCompute>>>(d_rgb[cur], d_gray, numPixels);

    const dim3 b2(32, 8);   // 2-D block: 32 x 8 threads
    const dim3 g2((width  + 31) / 32, (height + 7) / 8);
    blurH    <<<g2, b2, 0, sCompute>>>(d_gray, d_blurred, width, height);
    blurV    <<<g2, b2, 0, sCompute>>>(d_blurred, d_blurred2, width, height);
    sobel    <<<g2, b2, 0, sCompute>>>(d_blurred2, d_edges, width, height);

    histogram<<<g2, b2, 0, sCompute>>>(d_edges, d_hist, numPixels);
    // (histogram kernel as in Chapter 8, 8.8)

    // Copy the edge map back (device -> host, pinned, async):
    CHECK(cudaMemcpyAsync(h_edges[cur], d_edges, grayBytes,
                          cudaMemcpyDeviceToHost, sCompute));
}
```

**Why two streams and events?** The copy for frame `n+1` and the kernels for
frame `n` are independent work on *different* buffers - the exact condition
for overlap (§6.5). The event/dependency pair (`cudaEventRecord` +
`cudaStreamWaitEvent`) keeps the ordering correct on every iteration without
serialising the pipeline.

**Why the 2-D block for the stencil kernels?** The 2-D grid lets each thread
own one `(x, y)` pixel with natural indexing. The 32×8 block shape keeps
blocks tile-shaped (32 = warp width in `x`, so warps are row-aligned).

## 15.6 The Same Pipeline in Thrust

The library version replaces the four hand-written kernels with four
`thrust::transform` calls (Chapter 11). The stencil kernels need neighbouring
pixels, which `transform` provides via *shifted iterators*:

```cpp
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

// Greyscale: pure elementwise -> a plain transform functor.
struct ToGray {
    __device__ float operator()(const uchar3& px) const {
        return 0.299f * px.x + 0.587f * px.y + 0.114f * px.z;
    }
};

// Blur with halo: the functor receives the pixel INDEX and the row pointer;
// it reads its own 5-tap window. (A production version would use a proper
// stencil iterator; this keeps the index arithmetic visible.)
struct BlurH5 {
    const float* in; int width;
    __device__ float operator()(int i) const {
        const int x = i % width, y = i / width;
        const float* row = in + y * width;
        float acc = 0.0f;
        const float w[5] = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};
        for (int t = -2; t <= 2; ++t) {
            const int sx = min(max(x + t, 0), width - 1);
            acc += w[t + 2] * row[sx];
        }
        return acc;
    }
};

// Host side: chain the stages over device vectors.
thrust::device_vector<uchar3> d_rgb(...);
thrust::device_vector<float>  d_gray(n), d_blur(n), d_edges(n);

thrust::transform(d_rgb.begin(), d_rgb.end(), d_gray.begin(), ToGray());
thrust::transform(thrust::counting_iterator<int>(0),
                  thrust::counting_iterator<int>(n),
                  d_blur.begin(), BlurH5{raw_pointer(d_gray), width});
// ... blurV and sobel follow the same pattern; histogram is a thrust::reduce
// over a per-bin functor, or thrust::sort + adjacent-difference.
```

**The trade, stated plainly.** Thrust removes the launch plumbing and the
boundary guards; the `counting_iterator` + index-arithmetic pattern reintroduces
exactly the stencil logic the hand-written kernel had. For *elementwise*
stages (greyscale, magnitude) Thrust is a clear win; for *stencil* stages the
custom kernel of §15.3 is no more code and is easier to tune. This is the
Chapter 11 decision procedure in live action.

## 15.7 The Same Pipeline in CUDA-Oxide

With CUDA-Oxide (Chapter 14), the greyscale stage becomes a `#[kernel]` Rust
function - the same single-source style, with `DisjointSlice` giving the
no-alias guarantee:

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

#[cuda_module]
mod kernels {
    use super::*;

    // Greyscale in pure Rust. DisjointSlice<f32> guarantees exclusive
    // output slots; the input is read-only.
    #[kernel]
    pub fn rgb_to_gray(rgb: &[u8], gray: DisjointSlice<f32>, num_pixels: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if (i < num_pixels) {
            // RGB is packed 3 bytes/pixel; u8 -> f32 conversion is explicit.
            let base = (i as usize) * 3;
            let r = rgb[base] as f32;
            let g = rgb[base + 1] as f32;
            let b = rgb[base + 2] as f32;
            if let Some(out) = gray.get_mut(idx) {
                *out = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }
    }
}
```

**What this demonstrates.** The kernel is written in the language of the host,
indexed by the fused `thread::index_1d()` (Chapter 14), and protected by
`DisjointSlice` - no alias, no manual boundary contract. The stencil kernels
(blur, Sobel) follow the same shape with the same per-tap clamping logic as
the C++ versions, and the pipeline host code reuses the `cuda-async`
`DeviceOperation` chaining of Chapter 14, §14.6. As Chapter 14 warned: the
API is alpha, the shape is the point.

## 15.8 Verification: The Differential Test

A pipeline that produces *wrong* edges at 60 FPS is worse than a correct one
at 10 FPS. The verification strategy is the differential test:

1. **A CPU reference** implements the same four stages with plain loops
   (trivially correct, slow).
2. **The GPU pipeline** runs on a test image.
3. **Compare** stage by stage: greyscale, blurred, and edge maps must agree
   within a tolerance (`1e-4` for float stages; `uchar` output compared
   exactly).
4. **Property checks** on real data: the histogram bins fall in expected
   ranges; an all-black image yields all-zero edges (a *known-answer* test).

The differential test is the Chapter 16 discipline applied to the capstone:
it converts "the pipeline works" into a measurable, repeatable assertion, and
it is exactly what makes the *second* implementation (Thrust) and the *third*
(CUDA-Oxide) trustworthy - they must pass the same test as the first.

## 15.9 Measurement: The Report Card

The pipeline's performance is measured with CUDA events (Chapter 6, §6.4),
over many frames, with warm-up excluded:

```cpp
// Per-stage timing with events (the honest instrument, 6.4):
cudaEventRecord(start, sCompute);
rgbToGray<<<...>>>(...);
cudaEventRecord(mid, sCompute);
blurH<<<...>>>(); blurV<<<...>>>(); sobel<<<...>>>();
cudaEventRecord(stop, sCompute);
cudaEventSynchronize(stop);
float msStage1 = 0, msRest = 0;
cudaEventElapsedTime(&msStage1, start, mid);
cudaEventElapsedTime(&msRest,   mid,   stop);
```

The report card for a 1920×1080 frame on a modern GPU, as teaching numbers:

| Stage | Time | Bandwidth (3.35 TB/s peak) | Roofline verdict |
|---|---|---|---|
| rgbToGray | ~0.4 ms | ~75% of peak | Memory-bound (as predicted) |
| blurH + blurV | ~1.0 ms | ~70% of peak | Memory-bound, halo cost visible |
| sobel | ~0.5 ms | ~70% of peak | Memory-bound |
| histogram | ~0.3 ms | - | Atomic overhead, privatised |
| **Total compute** | **~2.2 ms** | - | 450+ FPS, transfer-limited overall |

The roofline (Chapter 1) *predicted* the memory-bound verdicts before any
code ran: every stage moves ~1 byte of data per pixel per pass with a handful
of FLOPs - far below the ridge point. The measurement confirms the prediction.
That is the loop this book teaches: predict with the model, confirm with the
instrument, optimise only the confirmed bottleneck.

## 15.10 The Capstone in One Paragraph

The pipeline is the entire book compressed: coalesced kernels with explicit
index arithmetic (Chapters 3, 7), synchronisation-free stages and a privatised
histogram (Chapters 5, 8), pinned memory and streamed double buffering
(Chapters 4, 6), library and language alternatives that must pass the same
differential test (Chapters 11, 13, 14), and a measurement discipline that
turns opinions into numbers (Chapter 16). If you can build this pipeline and
explain every line, you have graduated from this book.

## Key Takeaways

- A pipeline is a chain of kernels; a fast pipeline is one that never waits (streams + pinned memory + events).
- A separable Gaussian is 2 x 5 taps instead of 25; clamp each stencil tap at image borders independently.
- The differential test against a CPU reference is what makes a second and third implementation trustworthy.
- The roofline predicted every stage of the capstone was memory-bound before any code ran.
- The report card: median of many runs, fixed environment, events for device time.

## 15.11 Exercises

1. Why is the separable blur "10 taps instead of 25"? Derive the count for a
   5-tap separable Gaussian versus a full 5×5 stencil, and for a 9-tap
   version.
2. The first `blurH` in §15.3 was "plausible and wrong". Fix the comment
   explaining what was wrong, and state the property the corrected kernel
   guarantees at the borders.
3. In the streaming loop, why must `h_rgb` be *pinned* memory? Trace what
   happens if it is pageable.
4. The differential test uses a tolerance of `1e-4` for float stages. Why not
   exact equality? (Hint: Chapter 5, §5.6.)
5. Using the roofline model, predict whether making the blur a *single*
   fused 5×5 kernel (25 taps, no intermediate) would be faster or slower
   than the two-pass version, and explain the trade.
