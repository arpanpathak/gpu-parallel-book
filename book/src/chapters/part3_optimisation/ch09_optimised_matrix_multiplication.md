# Chapter 9: Optimised Matrix Multiplication

> *"Matrix multiplication is the one kernel every GPU vendor gets right. You
> should be able to explain why."*
> 📦 **Code companion:** the complete, buildable code for this chapter lives in [`code/ch09_sgemm/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch09_sgemm) in the repository.

Matrix multiplication (\\(C = A \times B\\), all \\(N \times N\\)) is the
canonical GPU workload - the inner loop of deep learning, linear algebra, and
scientific computing. It is also the perfect teaching kernel: it is
compute-bound (so the optimisations are about *arithmetic reuse*, not just
memory), it exercises every idea from Chapters 2, 7 and 8, and its optimised
form is the one shipped in cuBLAS (Chapter 11). We develop it in five stages,
measuring the reasoning at each step.

## 9.1 Why SGEMM Is Compute-Bound

Single-precision GEMM (\\(N^3\\) multiply-adds) has \\(2N^3\\) FLOPs. The
inputs are \\(N^2\\) elements of A and \\(N^2\\) of B; the output \\(N^2\\) of
C. With perfect caching, the minimum traffic is \\(3N^2\\) elements = \\(12N^2\\)
bytes. The arithmetic intensity (Chapter 1):

\\[ I = \frac{2N^3}{12N^2} = \frac{N}{6}\ \text{FLOP/byte} \\]

For \\(N = 4096\\), \\(I \\approx 683\\) FLOP/byte - two orders of magnitude
above the ~18 FLOP/byte ridge point of a modern GPU (Chapter 2). SGEMM is
**compute-bound**: the memory system is not the constraint; *keeping the
arithmetic units fed* is. Every optimisation below is about reuse: each value
loaded from memory must feed as many FLOPs as possible.

## 9.2 Stage 0: The Naive Kernel

```cpp
// One thread per output element C[i][j]. Each thread loops over k.
// Rows of C and A are contiguous (row-major); columns of B are NOT.
__global__ void sgemmNaive(const float* A, const float* B, float* C,
                           int N)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // row of C
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // col of C

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        // A[i][k] : consecutive threads read CONSECUTIVE i? No - they read
        //   A[i*N + k]; consecutive threads differ in j, so the SAME k and
        //   DIFFERENT i → stride-N addresses. Uncoalesced!
        // B[k][j] : consecutive threads read B[k*N + j] - consecutive j →
        //   coalesced. Half the traffic is good.
        sum += A[i * N + k] * B[k * N + j];
    C[i * N + j] = sum;
}
```

**Why it is slow.** The read of A is stride-`N` (uncoalesced, Chapter 7), and
every output element re-reads a full row of A and column of B from global
memory: \\(2N^3\\) bytes moved for \\(2N^3\\) FLOPs - intensity \\(1\\), far
below the ridge. The kernel is effectively memory-bound *because of its own
access pattern*. The shared-memory tiling of §9.4 fixes exactly this.

## 9.3 Stage 1: Make the Block Shape Match the Memory

First, a cheap fix: swap the thread-to-element mapping so that *both* operand
reads are coalesced. A block covering a *tile of output* with `threadIdx.x`
mapping to `j` and `threadIdx.y` to `i` gives:

- `B[k][j]`: consecutive threads → consecutive `j` → coalesced ✓
- `A[i][k]`: consecutive threads → same `i`, consecutive... no: `i` varies
  with `threadIdx.y`, so within a row of threads `i` is constant and `j`
  varies - `A[i][k]` is *uniform per thread-row* (broadcast), not strided.

The broadcast is served efficiently (all threads of a warp read the same
address in the same load). So a block-oriented launch with `threadIdx.x → j`
gives coalesced B reads and broadcast A reads:

```cpp
__global__ void sgemmCoalesced(const float* A, const float* B, float* C,
                               int N)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // row
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // col

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[i * N + k] * B[k * N + j];   // B coalesced, A broadcast
    C[i * N + j] = sum;
}
```

Better, but the *reuse* is still zero: every output re-reads \\(2N\\) floats
from global memory. The intensity is still ~1. The fix for reuse is tiling.

## 9.4 Stage 2: Shared-Memory Tiling

The classic formulation. A block of \\(T \times T\\) threads computes a
\\(T \times T\\) tile of C. It loads a \\(T \times T\\) tile of A and a
\\(T \times T\\) tile of B into shared memory, advances `k` in steps of
`T`, and each loaded value feeds \\(T\\) threads. The reuse factor is \\(T\\):
one global load of a tile serves \\(T^2\\) multiply-adds instead of
\\(T\\) (naive).

```cpp
#define T 16          // tile size: 16x16 threads per block

__global__ void sgemmTiled(const float* A, const float* B, float* C, int N)
{
    // Tiles in shared memory. The +1 padding (Chapter 7, 7.5) avoids bank
    // conflicts on column access; A-tile is read by column in the k-loop.
    __shared__ float sA[T][T + 1];
    __shared__ float sB[T][T + 1];

    // Output coordinates of this thread:
    const int row = blockIdx.y * T + threadIdx.y;   // global row of C
    const int col = blockIdx.x * T + threadIdx.x;   // global col of C

    float acc = 0.0f;    // this thread's partial C[row][col]

    // Sweep k in tiles of T. The block needs the A-tile column [k0..k0+T)
    // and the B-tile row [k0..k0+T) for each k0 step.
    for (int k0 = 0; k0 < N; k0 += T)
    {
        // Coalesced global loads into shared memory.
        // sA[ty][tx] = A[row][k0+tx]  (row segment, coalesced)
        // sB[ty][tx] = B[k0+ty][col]  (column segment, coalesced)
        sA[threadIdx.y][threadIdx.x] = A[row * N + k0 + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];

        __syncthreads();   // tile complete before any thread reads it

        // Inner product over the tile. Each thread reads:
        //   sA[ty][k]  - row of the A-tile   (bank-conflict-free due to pad)
        //   sB[k][tx]  - column of the B-tile (broadcast along the row)
        #pragma unroll
        for (int k = 0; k < T; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();   // tile done: no thread may reuse it yet
    }

    C[row * N + col] = acc;
}
```

**Why `#pragma unroll`?** The inner `k` loop is a compile-time-fixed trip
count (16). Unrolling emits 16 straight-line FMAs with no loop bookkeeping and
lets the compiler schedule the shared-memory loads ahead of the arithmetic  - 
hiding shared-memory latency behind FMA work. This single pragma is worth
10-20% on this kernel.

**The bank-conflict analysis.** `sA[ty][k]` for fixed `ty`, varying `k` over
a warp's threads: threads differ in `ty` (since `threadIdx.x = tx` varies
fastest and `k` is the same for all threads). A warp is 32 threads = 2 rows
of 16. Within a row of 16 threads, `ty` is constant and `k` constant → all
read the *same* address → broadcast, free. Across the two rows, addresses
differ by row stride 17 words → banks differ by 17 → no conflict. The padding
keeps the row stride (17) coprime with the bank count (32), which is exactly
the §7.5 rule.

**Why `T = 16`?** A 16×16 tile uses 2 × 16 × 17 × 4 = 2,176 bytes of shared
memory and 256 threads per block. It is the classic size because it balances
reuse (16×) with occupancy (many blocks per SM). Larger tiles (32×32) give
more reuse but fewer resident blocks; §9.6 shows the occupancy trade-off.

**The reuse accounting.** Each element of A loaded into shared memory is used
by \\(T = 16\\) threads (the column of the tile). Each B element by 16
threads. Global traffic drops by a factor of 16 versus the naive kernel; the
kernel is now compute-bound, which is where it belongs.

## 9.5 Stage 3: Register Tiling

The tiled kernel still reads shared memory for every FMA: one shared load per
multiply-add. Shared-memory bandwidth is finite - with 32 banks × 4 bytes, an
SM can supply at most 128 bytes/cycle, and the FP32 units can consume 128
FLOPs/cycle (128 cores × FMA). The FMA-to-load ratio is already at the limit.
The fix: **each thread computes more than one output element**, reusing each
shared-memory value across *registers*.

```cpp
#define T 16
#define RM 2          // rows of output per thread
#define RN 2          // cols of output per thread

__global__ void sgemmRegisterTiled(const float* A, const float* B, float* C,
                                   int N)
{
    __shared__ float sA[T][T + 1];
    __shared__ float sB[T][T + 1];

    // Each thread now owns an RM x RN micro-tile of C.
    // The block covers a T x T output tile with T*T/(RM*RN) threads.
    const int tx = threadIdx.x;                 // 0..T/RN-1
    const int ty = threadIdx.y;                 // 0..T/RM-1

    // Global coordinates of this thread's micro-tile (top-left corner):
    const int row0 = blockIdx.y * T + ty * RM;
    const int col0 = blockIdx.x * T + tx * RN;

    // Accumulators live in registers, one per micro-tile element:
    float acc[RM][RN];
    #pragma unroll
    for (int r = 0; r < RM; ++r)
        for (int c = 0; c < RN; ++c) acc[r][c] = 0.0f;

    for (int k0 = 0; k0 < N; k0 += T)
    {
        // Tile load: T*T elements spread over T*T/(RM*RN) threads, so each
        // thread loads an RM x RN patch of each tile. Every element of the
        // A-tile and B-tile is loaded exactly once: thread (tx, ty) covers
        // rows ty*RM..ty*RM+RM-1 and columns tx*RN..tx*RN+RN-1. Consecutive
        // tx cover consecutive columns -> coalesced row segments.
        #pragma unroll
        for (int r = 0; r < RM; ++r)
            for (int c = 0; c < RN; ++c)
                sA[ty * RM + r][tx * RN + c] =
                    A[(row0 + r) * N + k0 + tx * RN + c];
        #pragma unroll
        for (int r = 0; r < RM; ++r)
            for (int c = 0; c < RN; ++c)
                sB[ty * RM + r][tx * RN + c] =
                    B[(k0 + ty * RM + r) * N + col0 + tx * RN + c];

        __syncthreads();

        // Micro-tile FMA loop: for each k, each shared value feeds RM*RN
        // FMAs, all from registers. Shared loads drop by a factor RM*RN.
        #pragma unroll
        for (int k = 0; k < T; ++k)
        {
            // Load A-row segment and B-col segment ONCE into registers:
            float a_reg[RM], b_reg[RN];
            #pragma unroll
            for (int r = 0; r < RM; ++r)
                a_reg[r] = sA[ty * RM + r][k];
            #pragma unroll
            for (int c = 0; c < RN; ++c)
                b_reg[c] = sB[k][tx * RN + c];

            // RM*RN FMAs, zero shared-memory traffic in the inner product:
            #pragma unroll
            for (int r = 0; r < RM; ++r)
                for (int c = 0; c < RN; ++c)
                    acc[r][c] += a_reg[r] * b_reg[c];
        }

        __syncthreads();
    }

    // Write the micro-tile back:
    #pragma unroll
    for (int r = 0; r < RM; ++r)
        for (int c = 0; c < RN; ++c)
            C[(row0 + r) * N + col0 + c] = acc[r][c];
}
```

**Why RM × RN = 4 (or 8, 16)?** Each shared-memory value now feeds `RM * RN`
FMAs from registers. With 2×2, shared traffic drops 4×; with 4×4, 16×.
The limit is registers: each accumulator, plus the `a_reg`/`b_reg` arrays,
consumes registers per thread, and register pressure caps occupancy (§2.9).
A 4×4 micro-tile uses ~32+ registers; 8×8 would spill on most GPUs. The
production kernels shipped in cuBLAS use micro-tiles of 8×8 with special
register-allocation tricks; our 2×2 and 4×4 versions capture the idea.

**The correctness note on the tile-load.** Each thread loads `RM` elements of
the A-tile row at `sA[ty*RM + r][tx*RN]` - note that `RN` elements of the row
are loaded by `RN` different threads *in the same row* (`tx` ranges over
`T/RN`), so the row segment `[tx*RN, tx*RN+RN)` is covered by `RN` threads.
The loads are coalesced because consecutive `tx` cover consecutive columns.

## 9.6 Occupancy and Launch Bounds

The register-tiled kernel consumes more registers per thread. If the compiler
uses, say, 40 registers, occupancy falls to `64K / (40 × 256)` ≈ 6 blocks/SM.
That may be fine - the kernel is compute-bound and register tiling gives it
enough ILP - but the compiler should be *told* the budget so it does not
spill:

```cpp
// Tell ptxas: this kernel must fit in at most 256 threads/block, and I
// want at least 4 blocks/SM resident. ptxas will trade registers for
// occupancy within the budget rather than silently spilling.
__global__ void __launch_bounds__(256, 4)
sgemmRegisterTiled(const float* A, const float* B, float* C, int N) { /* ... */ }
```

**Why `__launch_bounds__`?** Without it, `ptxas` minimises register use for
correctness but may choose more registers than the target occupancy allows.
`__launch_bounds__(maxThreads, minBlocksPerSM)` gives the compiler a hard
constraint: use at most `maxThreads` per block and keep at least
`minBlocksPerSM` blocks resident. It converts the occupancy reasoning of
Chapter 2 into a compiler directive.

**The tuning loop.** For a given micro-tile, measure the kernel at
`minBlocksPerSM` = 2, 4, 6, 8 (Chapter 16's benchmarking discipline). The
sweet spot is where register tiling's ILP and occupancy's latency hiding
balance. There is no universal answer - which is why the measurements, not
the folklore, decide.

## 9.7 What the Optimised Kernel Achieves

With `T = 16`, 2×2 register tiling, padding, and `__launch_bounds__`, the
kernel of §9.5 typically reaches 60-75% of peak FP32 on a modern GPU
(measured on the same hardware as the roofline numbers of Chapter 2). The
remaining gap is the shared-memory FMA supply rate and the k-loop's tile
overhead. Closing it further requires:

- **Warp-level tiling** (each warp computes a 32×8 tile with `ldmatrix`,
  the layout descriptor instruction) - the territory of cuBLAS;
- **Tensor cores** (`mma` instructions), which multiply 16×16×16 tiles per
  instruction - a completely different pipeline covered in Chapter 11.

Both are beyond this chapter's scope, but the journey from naive (5% of peak)
to register-tiled (70%) is the same journey every optimisation chapter in this
book teaches: *find the bottleneck, remove it, measure, repeat*.

## Key Takeaways

- SGEMM is compute-bound (intensity N/6 FLOP/byte): the goal is arithmetic reuse, not just coalescing.
- The naive kernel has zero reuse and runs at a few percent of peak.
- Shared-memory tiling makes each loaded element feed T threads; global traffic drops by T.
- Register tiling makes each shared-memory value feed RM x RN FMAs from registers.
- __launch_bounds__ trades registers for occupancy; the right balance is found by measuring.

## 9.8 Exercises

1. Verify the arithmetic-intensity claim: show that for
   \\(N = 4096\\), \\(I = N/6 \\approx 683\\) FLOP/byte, and compare it with
   the ridge point of Chapter 2.
2. In §9.4, count the shared-memory bytes loaded per block per `k0` step.
   How many FLOPs do they feed? Show that the ratio is \\(T\\) FLOPs per
   shared byte (hint: \\(T^2\\) FMAs over \\(2T^2\\) loads... where does the
   padding change the byte count?).
3. Explain why `__launch_bounds__(256, 4)` can *decrease* performance even
   though it increases occupancy.
4. The §9.5 tile-load stores the B-tile row segment with `RN` threads per
   row. Trace which thread loads `sB[3][7]` for `T=16, RM=RN=2`.
