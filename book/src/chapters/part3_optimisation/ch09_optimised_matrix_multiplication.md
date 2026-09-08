# Chapter 9: Optimised Matrix Multiplication

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch09_sgemm/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch09_sgemm)
> in the repository.

Matrix multiplication (\\(C = A \times B\\), all \\(N \times N\\)) is the
canonical GPU workload: it is the inner loop of deep learning, linear algebra,
and scientific computing. It is also a useful teaching kernel because it is
compute-bound at large \\(N\\), exercises ideas from Chapters 2, 7, and 8, and
has an optimised form similar in structure to the kernels shipped in cuBLAS
(Chapter 11). This chapter develops it in stages and gives the reasoning at
each step.

## 9.1 SGEMM Is Compute-Bound

Single-precision GEMM performs \\(N^3\\) multiply-adds, or \\(2N^3\\) FLOPs.
The inputs are \\(N^2\\) elements of A and \\(N^2\\) of B; the output is
\\(N^2\\) elements of C. With perfect caching, the minimum traffic is
\\(3N^2\\) elements, or \\(12N^2\\) bytes. The arithmetic intensity (Chapter 1)
is:

\\[ I = \frac{2N^3}{12N^2} = \frac{N}{6}\ \text{FLOP/byte} \\]

For \\(N = 4096\\), \\(I \approx 683\\) FLOP/byte, two orders of magnitude above
the roughly 18 FLOP/byte ridge point of a modern GPU (Chapter 2). At this size
SGEMM is **compute-bound**: the memory system is not the constraint. Keeping
the arithmetic units fed is the constraint. Each optimisation below increases
reuse so that every value loaded from memory feeds as many FLOPs as possible.

## 9.2 Stage 0: The Naive Kernel

The simplest kernel assigns one thread to one output element and loops over the
shared dimension \\(k\\):

```cpp
// One thread per output element C[i][j]. Each thread loops over k.
// This mapping makes threadIdx.x the ROW index, which is the wrong choice
// for row-major memory.
__global__ void sgemmNaive(const float* A, const float* B, float* C,
                           int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // row of C
    const int j = blockIdx.y * blockDim.y + threadIdx.y;   // col of C

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        // A[i][k] : consecutive threads (consecutive i) read A[i*N + k] with
        //   stride N between lanes. Uncoalesced.
        // B[k][j] : consecutive threads have the same j within a warp, so this
        //   address is a broadcast, not a stream.
        // C[i][j] : consecutive threads write C[i*N + j] with stride N.
        //   Uncoalesced.
        sum += A[i * N + k] * B[k * N + j];
    C[i * N + j] = sum;
}
```

The read of A is stride-\\(N\\) and the write of C is stride-\\(N\\). Every
output element also re-reads a full row of A and a full column of B from global
memory: \\(2N^3\\) bytes moved for \\(2N^3\\) FLOPs, intensity near 1, far below
the ridge. The kernel is memory-bound because of its access pattern, not because
matrix multiplication is inherently memory-bound.

## 9.3 Stage 1: Make the Block Shape Match Memory

A cheap fix changes the thread-to-element mapping so that `threadIdx.x` indexes
the column and `threadIdx.y` indexes the row. For a warp whose `x` lanes run
across a row of the output tile:

- `B[k][j]`: consecutive threads read consecutive `j`, so the read is
  coalesced.
- `A[i][k]`: all threads in a warp row have the same `i`, so they read the same
  address. The hardware broadcasts it.
- `C[i][j]`: consecutive threads write consecutive `j`, so the write is
  coalesced.

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

This removes the uncoalesced traffic, but reuse is still zero: each output
element re-reads \\(2N\\) floats from global memory, and intensity remains near
1. The fix for reuse is tiling.

## 9.4 Stage 2: Shared-Memory Tiling

A block of \\(T \times T\\) threads computes a \\(T \times T\\) tile of C. It
loads a \\(T \times T\\) tile of A and a \\(T \times T\\) tile of B into shared
memory, advances \\(k\\) in steps of \\(T\\), and each loaded value feeds
\\(T\\) threads. The reuse factor is \\(T\\): one global load serves \\(T^2\\)
multiply-adds instead of \\(T\\).

```cpp
#define T 16          // tile size: 16x16 threads per block

__global__ void sgemmTiled(const float* A, const float* B, float* C, int N)
{
    // Tiles in shared memory. The +1 padding (Chapter 7, 7.5) avoids bank
    // conflicts on column access; the A-tile is read by column in the k-loop.
    __shared__ float sA[T][T + 1];
    __shared__ float sB[T][T + 1];

    // Output coordinates of this thread:
    const int row = blockIdx.y * T + threadIdx.y;   // global row of C
    const int col = blockIdx.x * T + threadIdx.x;   // global col of C

    float acc = 0.0f;    // this thread's partial C[row][col]

    // Sweep k in tiles of T. The block needs the A-tile columns [k0..k0+T)
    // and the B-tile rows [k0..k0+T) for each k0 step.
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

        __syncthreads();   // tile done before the next tile overwrites it
    }

    C[row * N + col] = acc;
}
```

**Role of `#pragma unroll`.** The inner \\(k\\) loop has a compile-time trip
count of 16. Unrolling emits straight-line FMAs, removes loop bookkeeping, and
lets the compiler schedule shared-memory loads ahead of the arithmetic, hiding
shared-memory latency behind FMA work. On this kernel the pragma is typically
worth 10-20%.

**Bank-conflict analysis.** For `sA[threadIdx.y][k]`, a warp of 32 threads with
\\(T=16\\) spans two block rows. Within one row, all 16 lanes read the same
address (broadcast). Across the two rows, addresses differ by the padded row
stride of 17 words, hence by 17 banks. No conflict occurs because the padding
keeps the row stride coprime with the 32-bank layout. For
`sB[k][threadIdx.x]`, consecutive `threadIdx.x` read consecutive columns of one
padded row; lanes in the second block row repeat the same addresses, which the
hardware serves as broadcasts.

**Choice of \\(T = 16\\).** A 16x16 tile uses \\(2 \times 16 \times 17 \times
4 = 2,176\\) bytes of shared memory and 256 threads per block. The size balances
reuse (16x) against occupancy. Larger tiles such as 32x32 give more reuse but
fewer resident blocks; §9.6 shows the occupancy trade.

**Reuse accounting.** Each element of A loaded into shared memory is used by
\\(T = 16\\) threads, and each element of B by 16 threads. Global traffic drops
by a factor of 16 relative to the naive kernel. The kernel is now
compute-bound.

## 9.5 Stage 3: Register Tiling

The tiled kernel still reads shared memory for every FMA: one shared load per
multiply-add. Shared-memory bandwidth is finite. With 32 banks of 4 bytes, an
SM can supply at most 128 bytes per cycle, and modern FP32 units can consume
128 FLOPs per cycle. The next step gives each thread more than one output
element so that each shared-memory value is reused from registers:

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
                    B[(k0 + ty * RM + r) * N + col0 + c];

        __syncthreads();

        // Micro-tile FMA loop: for each k, each shared value feeds RM*RN
        // FMAs, all from registers. Shared loads drop by a factor RM*RN.
        #pragma unroll
        for (int k = 0; k < T; ++k)
        {
            // Load the A-row segment and B-col segment once into registers:
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

**Micro-tile size and registers.** Each shared-memory value now feeds
\\(RM \times RN\\) FMAs from registers. With a 2x2 micro-tile, shared-memory
traffic drops by 4x; with 4x4, by 16x. The limit is register pressure. Each
accumulator, plus the `a_reg` and `b_reg` arrays, consumes registers per
thread, and register pressure caps occupancy (§2.9). A 4x4 micro-tile uses
roughly 32 or more registers; 8x8 would spill on most GPUs. Production kernels
in cuBLAS use larger micro-tiles with specialised register allocation; the 2x2
and 4x4 versions capture the mechanism.

**Tile-load indexing.** In the A-tile load, each thread writes an \\(RM\\)-row
patch at `sA[ty*RM + r][tx*RN + c]`. The `RN` elements of one row are covered
by `RN` different threads because `tx` ranges over \\(T/RN\\) values. The loads
are coalesced because consecutive `tx` cover consecutive columns. In the B-tile
load, `col0` already contains the thread's \\(tx \times RN\\) column offset, so
the global column is `col0 + c`. Adding `tx*RN` a second time would read the
wrong B elements.

## 9.6 Occupancy and Launch Bounds

Register tiling increases registers per thread. If the compiler uses, for
example, 40 registers, occupancy falls to \\(64K / (40 \times 256) \approx 6\\)
blocks per SM. That may be acceptable because the kernel is compute-bound and
register tiling provides instruction-level parallelism. The compiler should be
told the budget so it does not silently spill:

```cpp
// Tell ptxas: this kernel must fit in at most 256 threads/block, and at least
// 4 blocks/SM must be resident. ptxas trades registers for occupancy within
// the budget rather than spilling.
__global__ void __launch_bounds__(256, 4)
sgemmRegisterTiled(const float* A, const float* B, float* C, int N) { /* ... */ }
```

The body is omitted here; apply the attribute to the full kernel of §9.5. The
companion `code/ch09_sgemm/sgemm.cu` ships the combined final definition.

Without `__launch_bounds__`, `ptxas` minimises register use for correctness but
may use more registers than the target occupancy allows.
`__launch_bounds__(maxThreads, minBlocksPerSM)` turns the occupancy reasoning of
Chapter 2 into a compiler constraint.

The right setting is found by measurement. For a given micro-tile, run the
kernel with `minBlocksPerSM` equal to 2, 4, 6, and 8 and compare (Chapter 16).
The balance between register-tiling ILP and occupancy latency hiding has no
universal answer.

## 9.7 What the Optimised Kernel Achieves

With \\(T=16\\), a 2x2 register tile, padding, and `__launch_bounds__`, the
kernel of §9.5 typically reaches 60-75% of peak FP32 on a modern GPU. The
remaining gap comes from shared-memory FMA supply and tile-loop overhead.
Closing it further requires techniques beyond this chapter:

- **Warp-level tiling**, in which each warp computes a wide micro-tile with
  `ldmatrix`, the layout-descriptor load instruction used by cuBLAS;
- **Tensor cores**, whose `mma` instructions multiply 16x16x16 tiles per
  instruction on a separate pipeline (Chapter 11).

The progression from naive to register-tiled follows the method used throughout
the book: find the bottleneck, remove it, measure, and repeat.

## What Each Optimisation Stage Does

SGEMM contains, in one program, every optimisation idea introduced earlier in
the book.

The naive kernel's problem is memory, not arithmetic. With `threadIdx.x`
mapped to the row, consecutive threads in a warp access A with stride \\(N\\)
and write C with stride \\(N\\). Half of the traffic is uncoalesced, so the
kernel becomes memory-bound despite performing useful arithmetic.

The coalesced kernel maps `threadIdx.x` to the column. B reads are coalesced, A
reads are broadcast, and C writes are coalesced. Both kernels still re-read
operands from global memory for every output element, so intensity remains low.

Shared-memory tiling introduces reuse. A block of 16x16 threads loads a 16x16
tile of A and a 16x16 tile of B into shared memory and computes the output
tile. Each global load is used by 16 threads instead of one, reducing global
traffic by a factor of 16. The new bottleneck is shared-memory bandwidth,
because the inner product reads shared memory for every FMA.

Register tiling removes that bottleneck. With a 2x2 micro-tile, each thread
loads row and column segments into registers once per \\(k\\) and performs four
FMAs with no shared-memory traffic in the inner loop. The FMA-to-shared-load
ratio improves by a factor of four at the cost of registers, which is where
`__launch_bounds__` enters. The compiler chooses a register count, and that
choice trades occupancy (Chapter 2) against spills.

The same progression applies to any optimised kernel: identify the current
bottleneck (uncoalesced access, no reuse, shared-memory bandwidth, or
registers), remove it, and measure. The roofline model explains why each stage
matters: each stage raises arithmetic intensity by moving data closer to the
arithmetic units, from global memory to shared memory to registers.

## Common Pitfalls

- Getting tile-load indexing wrong by double-counting an offset, such as
  writing `col0 + tx*RN + c` when `col0` already contains the thread's column
  offset. Trace one thread's load by hand before launching.
- Using `__launch_bounds__` without measuring. Forcing occupancy can increase
  register spills and slow the kernel.
- Forgetting the second `__syncthreads()` after the inner-product loop. The
  next tile load would overwrite shared memory while some threads still read it.
- Assuming the naive kernel is fast because it is simple. With the wrong
  thread-to-data mapping, uncoalesced A and C traffic can be 20-30x larger than
  necessary.

## Check Your Understanding

<details>
<summary>Why is SGEMM compute-bound at N=4096?</summary>

Its intensity is \\(N/6 \\approx 683\\) FLOP/byte, far above typical ridge
points of 20-40 FLOP/byte. The memory system can feed the arithmetic units; the
limit is keeping the FMAs fed with data reused from shared memory and
registers.
</details>

<details>
<summary>What does the +1 padding in sA[T][T+1] do?</summary>

It makes the row stride 17 words instead of 16. Because 17 is coprime with the
32-bank shared-memory layout, column accesses land on distinct banks rather
than forming a 32-way bank conflict.
</details>

<details>
<summary>Why can __launch_bounds__ decrease performance?</summary>

It constrains the compiler to a register budget. If the kernel needs more
registers than the budget allows, ptxas spills to local memory, and the spill
traffic can cost more than the occupancy gain.
</details>

## Key Takeaways

- SGEMM is compute-bound (intensity \\(N/6\\) FLOP/byte); the goal is arithmetic reuse, not just coalescing.
- The naive kernel with the wrong thread-to-element mapping has zero reuse and uncoalesced A and C traffic.
- Shared-memory tiling makes each loaded element feed \\(T\\) threads; global traffic drops by \\(T\\).
- Register tiling makes each shared-memory value feed \\(RM \times RN\\) FMAs from registers.
- `__launch_bounds__` trades registers for occupancy; the right balance is found by measuring.

## 9.8 Exercises

1. Verify the arithmetic-intensity claim: show that for \\(N = 4096\\),
   \\(I = N/6 \approx 683\\) FLOP/byte and compare it with the ridge point of
   Chapter 2.
2. In §9.4, count the shared-memory bytes loaded per block per \\(k0\\)
   step. How many FLOPs do they feed? Show that the ratio is \\(T\\) FLOPs per
   shared byte.
3. Explain why `__launch_bounds__(256, 4)` can decrease performance even
   though it increases occupancy.
4. In the §9.5 tile load, trace which thread loads `sB[3][7]` for
   \\(T=16, RM=RN=2\\).
