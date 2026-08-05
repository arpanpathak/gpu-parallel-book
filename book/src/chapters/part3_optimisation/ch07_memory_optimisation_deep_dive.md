# Chapter 7: Memory Optimisation Deep Dive

> *"Most kernels are not too slow to compute; they are too slow to fetch."*

Chapter 2 promised that this chapter would show the accounting. Here it is.
We examine the three tools that dominate GPU memory performance - **coalescing
and the grid-stride loop**, **shared memory and bank conflicts**, and
**vectorised and specialised memory paths** - and we end with a complete,
optimised matrix transpose, the canonical exercise in memory optimisation.

## 7.1 The Accounting: Where Time Goes

From Chapter 2's latency table, a global load costs roughly 400-800 cycles
and a shared-memory access 20-30. The arithmetic: a warp of 32 threads doing
one global load each spends ~500 cycles waiting; the SM could have executed
~16 shared-memory accesses per lane in that time. Memory-bound kernels (the
roofline test of Chapter 1) spend most of their time in this wait.

Two levers reduce the wait:

1. **Fewer transactions** - coalescing (use each fetched byte).
2. **Fewer round trips** - reuse fetched data in shared memory or registers.

Everything in this chapter is one of those two levers.

## 7.2 Coalescing, Quantified

The hardware fetches global memory in **128-byte cache lines** and services
requests at **32-byte sector** granularity. A warp's load is coalesced when
its 32 addresses fall within as few lines as possible.

- 32 consecutive `float`s (128 bytes) → 1 line, 1 transaction. **Perfect.**
- 32 `float`s with stride 1 but misaligned start → 2 lines. **Good.**
- 32 `float`s with stride 32 → 32 lines. **Catastrophic**: 32× traffic.

Pictured, for a warp of 32 threads each loading one `float` (the grid is
128-byte cache lines; `T0..T31` are the threads; `[ ]` marks one fetched
line):

```
COALESCED (stride 1):                 UNCOALESCED (stride 32):
                                          
[ T0 T1 T2 ... T31 ]                 [T0]           [T1]           [T2]
 └── one 128-byte line ──┘             └─line─┘      └─line─┘      └─line─┘
  32 floats, 128 bytes                32 threads, 32 lines = 4 KB fetched
  1 transaction                        for 128 bytes actually used:
                                       a 32x waste of bandwidth

  T0..T31 read 0,4,8,...,124          T0 reads 0, T1 reads 128,
                                       T2 reads 256, ... (stride 32 floats)
```

The left picture is why the global-index formula exists (Chapter 3, §3.5):
it hands consecutive threads consecutive addresses *by construction*. The
right picture is what happens when the formula is inverted - the classic
column-access bug in row-major data.

The rule, restated with the hardware in mind: **consecutive thread IDs should
map to consecutive addresses**. The global-index formula from Chapter 3
satisfies this by construction for 1-D arrays and row-major 2-D data.

```cpp
// GOOD (coalesced): thread t reads element t of each row.
// for a row-major matrix in[row][col], col varies fastest:
float v = in[row * width + col];        // col == threadIdx.x mapped to col

// BAD (uncoalesced): thread t reads element t*width - stride of width floats.
// 32 threads span 32 different cache lines for a large width:
float v = in[col * width + row];        // col varies slowest
```

## 7.3 The Grid-Stride Loop

A kernel launched with more threads than the array has elements is
wasteful; a kernel with fewer threads than elements under-utilises the GPU.
The **grid-stride loop** decouples the launch size from the problem size:

```cpp
// One grid covers the array in "rounds": each thread strides forward by the
// total number of threads (gridDim.x * blockDim.x) each iteration.
__global__ void saxpyGridStride(float alpha, const float* x, float* y, int n)
{
    // Total threads in the grid:
    const int stride = gridDim.x * blockDim.x;

    // First element this thread owns:
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // March forward by 'stride' until past the end.
    for (; i < n; i += stride)
    {
        y[i] = alpha * x[i] + y[i];     // SAXPY: single-precision A·X + Y
    }
}
```

**Why this shape?** The launch size is now a *tuning parameter* (often set to
saturate the device, e.g. enough threads to fill all SMs), independent of `n`.
Each thread processes multiple elements, amortising index arithmetic and
allowing per-thread data reuse. Coalescing is preserved: within one iteration
of the loop, consecutive threads still read consecutive addresses.

## 7.4 Shared Memory: The Explicit Cache

Shared memory is the programmer-managed cache of Chapter 2. The pattern is
always a **tile**:

1. Cooperatively load a tile of global data into shared memory (coalesced
   reads);
2. `__syncthreads()` to make the tile visible;
3. Compute from shared memory, reusing each loaded value many times;
4. `__syncthreads()` before the next tile overwrites this one.

The classic example is the matrix transpose. A naive transpose kernel reads
rows (coalesced) and writes columns (uncoalesced), or vice versa - one side
always pays. The shared-memory version fixes both sides:

```cpp
// ---------------------------------------------------------------------------
// Shared-memory tiled transpose. In-place variant for a WIDTH x WIDTH matrix
// of floats, WIDTH a multiple of the tile size TILE (32 here).
//
// Stage 1 (coalesced read):  each thread reads a[iy][ix] from global memory;
//   consecutive threads map to consecutive ix → consecutive addresses.
// Stage 2 (shared memory):   the tile is stored in shared as tile[ty][tx].
// Stage 3 (coalesced write): each thread writes tile[tx][ty] to a[jx][jy];
//   this time consecutive threads (consecutive tx) map to consecutive jy →
//   consecutive addresses in the OUTPUT row. The transpose happened in
//   shared memory, so BOTH global accesses are coalesced.
// ---------------------------------------------------------------------------
#define TILE 32

__global__ void transposeTiled(const float* in, float* out, int width)
{
    // Shared tile with a padding column (see 7.5 for why +1 exists).
    __shared__ float tile[TILE][TILE + 1];

    // Global coordinates of this thread's element:
    const int ix = blockIdx.x * TILE + threadIdx.x;   // column
    const int iy = blockIdx.y * TILE + threadIdx.y;   // row

    // Stage 1: coalesced read from global memory.
    if (ix < width && iy < width)
        tile[threadIdx.y][threadIdx.x] = in[iy * width + ix];

    __syncthreads();   // everyone's tile element must be visible before reads

    // Transposed output coordinates:
    const int jx = blockIdx.y * TILE + threadIdx.x;   // column of output
    const int jy = blockIdx.x * TILE + threadIdx.y;   // row of output

    // Stage 3: write the transposed element. Consecutive threads (x) map to
    // consecutive jy rows at the same jx - i.e., consecutive addresses in
    // the row-major output. Coalesced.
    if (jx < width && jy < width)
        out[jy * width + jx] = tile[threadIdx.x][threadIdx.y];
}
```

**The reasoning in full.** A naive kernel doing `out[j][i] = in[i][j]` would
have threads with consecutive ids reading consecutive `i` (coalesced reads)
but writing `j`-major addresses (uncoalesced writes). The tile decouples the
two: the *data layout* is transposed inside shared memory, so both the global
read and the global write are coalesced. Shared memory pays for its freedom.

## 7.5 Bank Conflicts and the One-Column Padding

From Chapter 2, shared memory is 32 banks of 4 bytes. The bank of an address
is `(address / 4) mod 32`. Consider the un-padded tile `float tile[32][32]`:

- Row `r` starts at byte `r * 128`, so row `r` occupies banks
  `(r * 32) mod 32 = 0` - **every row starts on bank 0**.
- When the kernel reads `tile[threadIdx.y][threadIdx.x]` with consecutive
  `threadIdx.x`, all 32 threads of a warp hit banks `0..31` - fine.

But a *column* read `tile[threadIdx.x][threadIdx.y]` (as the transpose does
in stage 3) has consecutive threads reading addresses `r * 32` words apart  - 
all 32 threads hit **bank 0**: a 32-way bank conflict, 32 cycles instead of 1.

**The fix is padding by one column** (`float tile[32][33]`): row `r` now
starts at byte `r * 132`, so row `r` starts at bank `(r * 33) mod 32 = r`.
A column read `tile[t][r]` for consecutive `t` now hits banks `0..31` exactly
once each. One float of padding per row converts a 32-cycle stall into a
1-cycle access. Padding is the cheapest performance win in CUDA.

```cpp
// Padding rule of thumb:
//   float tile[TILE][TILE];        // 32-way conflicts on column access
//   float tile[TILE][TILE + 1];    // conflict-free column access
```

## 7.6 Vectorised Loads: `float4` and Alignment

A warp loading 32 `float`s fetches 128 bytes in one transaction - but each
*instruction* moves 4 bytes per lane. The memory subsystem is happiest with
128-bit accesses. **Vectorised loads** let one instruction move 16 bytes per
lane: a warp then moves 512 bytes per instruction.

```cpp
// Load four floats per thread, one instruction per four floats.
__global__ void saxpyVec4(float alpha, const float4* x, float4* y, int n4)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4)
    {
        float4 xv = x[i];             // 16-byte aligned load
        float4 yv = y[i];
        yv.x = alpha * xv.x + yv.x;   // four lanes of arithmetic per thread
        yv.y = alpha * xv.y + yv.y;
        yv.z = alpha * xv.z + yv.z;
        yv.w = alpha * xv.w + yv.w;
        y[i] = yv;                    // 16-byte aligned store
    }
}
```

**The alignment contract.** `float4` requires **16-byte alignment**. A buffer
allocated with `cudaMalloc` is 256-byte aligned, so `reinterpret_cast`ing it
to `float4*` is safe; a pointer offset by an odd number of floats is not. The
general contract: a vectorised load must be aligned to the vector's size. If
your data does not satisfy that, either pad the allocation or handle the
tail elements scalar-wise.

**Why vectorisation helps beyond coalescing.** Fewer instructions (one load
vs four), fewer memory requests, and the 128-bit path uses the bus more
efficiently. On memory-bound kernels, float4-style access routinely adds
20-40% throughput. The `int4`, `double2`, and `uint4` types follow the same
rules.

## 7.7 Constant Memory

> **Primitive - constant memory.** A 64 KB read-only memory space, cached in
> a dedicated per-SM cache, optimised for **broadcasts**: when all threads of
> a warp read the *same* address, the hardware serves all 32 lanes in one
> access. When threads read *different* addresses, it serialises - the exact
> inverse of shared memory's behaviour.

```cpp
// Declared at file scope, device-side:
__constant__ float g_coeffs[16];

// Host fills it with cudaMemcpyToSymbol (note: symbol, not pointer):
float h_coeffs[16] = { /* ... */ };
CHECK(cudaMemcpyToSymbol(g_coeffs, h_coeffs, sizeof(h_coeffs)));

// Kernel reads: all threads read the SAME coefficient per call  - 
// a broadcast, served in one access from the constant cache.
__global__ void applyCoeffs(const float* in, float* out, int n, int c)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * g_coeffs[c];   // same address for all threads
}
```

**When to use constant memory.** Kernel parameters (all threads read the same
value), lookup tables indexed by a *uniform* value, coefficients. When to
avoid it: data indexed differently per thread - a *divergent* constant access
serialises and is slower than global memory.

## 7.8 Texture and Surface Memory (Briefly)

Texture memory is a cached read-only path with two special powers: **spatial
locality** (2-D caches for images - neighbouring pixels in any direction) and
**hardware interpolation** (bilinear filtering, used by graphics). On modern
GPUs, the L1/L2 caches largely subsume its raw speed advantage; its remaining
value is the interpolation hardware and the `cudaTextureObject` API for
image-like data. If your kernel reads images with 2-D locality, a texture
object is a legitimate optimisation; for linear 1-D access, global memory with
good coalescing is equal or better. The capstone (Chapter 15) uses plain
global memory for its image pipeline and stays within 5% of peak bandwidth  - 
a useful baseline that says: *coalescing first, specialised paths second*.

## 7.9 The Optimisation Checklist

When a kernel is memory-bound, walk this list in order:

1. **Is it coalesced?** Consecutive threads → consecutive addresses?
2. **Is the launch a grid-stride loop** sized to the device, or a
   one-thread-per-element launch?
3. **Is data reused?** If yes, tile it in shared memory (§7.4); check bank
   conflicts and pad (§7.5).
4. **Can accesses be vectorised?** `float4`/`double2` with correct alignment
   (§7.6).
5. **Are uniform values in constant memory?** (§7.7)
6. **Is the transfer pipeline streamed?** Pinned memory + streams (Chapters
   4, 6) - memory-bound kernels are no faster than their copies.

Each step is cheap to try and easy to measure (Chapter 16 shows the
measurement discipline). Never apply step 6 without measuring the result of
step 1.

## Key Takeaways

- Coalescing means touching the fewest 128-byte cache lines per warp access: consecutive threads, consecutive addresses.
- The grid-stride loop decouples launch size from problem size and preserves coalescing.
- Shared memory is the explicit cache; padding by one column eliminates bank conflicts.
- float4-style vectorised loads move 16 bytes per thread and require 16-byte alignment.
- Constant memory broadcasts uniform reads for free; per-thread divergent reads are slower than global memory.

## 7.10 Exercises

1. A warp loads 32 `float`s starting at byte offset 4 (misaligned by one
   float). How many 128-byte lines are touched? How many would be touched if
   the start were byte-aligned to 128?
2. Explain why the padding `[TILE][TILE + 1]` fixes column-access bank
   conflicts, using the formula `bank = (byte_address / 4) mod 32`.
3. You are transposing a `1024 × 1024` float matrix with `TILE = 32`.
   Count the shared-memory traffic per tile for the padded and un-padded
   versions (assume one column-read per thread).
4. When would you *not* use constant memory for a lookup table? Give a
   concrete access pattern that makes it slower than global memory.
