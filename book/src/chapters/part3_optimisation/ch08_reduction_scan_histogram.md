# Chapter 8: Reduction, Scan & Histogram

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch08_reduction/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch08_reduction)
> in the repository.

This chapter covers three canonical data-parallel algorithms that appear, in
disguise, in almost every real GPU application: **reduction** (sum, max, or
min of an array), **scan** (prefix sums, an array of partial results), and
**histogram** (counting occurrences). Each is developed from a naive version to
an optimised version, with the reasoning for every transformation. These
patterns are the vocabulary of Chapter 9's matrix multiplication and the
capstone's image pipeline.

## 8.1 The Reduction Problem

Given an array \\(a\\) of \\(n\\) elements, compute
\\(\sum_{i=0}^{n-1} a_i\\). On a CPU this is a loop of \\(n\\) additions. On a
GPU the challenge is different: additions are cheap, but combining partial
results across threads requires communication, and communication is expensive.
A good reduction minimises the number of communication rounds and keeps every
thread busy between them.

## 8.2 Stage 0: One Thread Does It All

```cpp
__global__ void reduceNaive(const float* in, float* out, int n)
{
    if (threadIdx.x == 0)              // ONE thread
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += in[i];   // serial, n additions
        out[0] = sum;
    }
}
```

This kernel is correct but uses one thread. It is useful as a reference
implementation whose result the optimised kernels are checked against.

## 8.3 Stage 1: Tree Reduction in Shared Memory

Addition is associative, so the additions can be reordered into a tree. In
round 1, threads 0..\\(n/2-1\\) each add two elements. In round 2, threads
0..\\(n/4-1\\) each add two partial sums, and so on. The tree has \\(\log_2 n\\)
levels, so \\(n/2\\) additions complete in \\(\log_2 n\\) parallel rounds.

```cpp
// Reduce one block's worth of data (one element per thread) using shared memory.
__global__ void reduceTree(const float* in, float* out, int n)
{
    __shared__ float s[TILE];          // TILE = blockDim.x, a power of two

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load with a boundary guard; out-of-range elements contribute zero.
    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;

    // Tree: each level halves the number of active threads.
    // Level 0: threads 0..TILE/2-1 add s[t] and s[t + TILE/2].
    // Level k: active threads < TILE >> (k+1).
    for (int stride = TILE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();               // make all writes visible
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
    }

    // Thread 0 owns the block's total.
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
```

The barrier inside the loop is required because, at each level, thread
\\(t\\) reads a partial sum written by thread \\(t + stride\\) at the previous
level. The barrier guarantees the previous level's writes are complete before
the next level reads them.

The halving scheme assumes `TILE` is a power of two, so every level halves
evenly and the active set `threadIdx.x < stride` is contiguous.
Non-power-of-two block sizes complicate the active-set arithmetic without
benefit; 128, 256, and 512 dominate practice. For unsigned sizes,
`stride >>= 1` and `stride /= 2` are equivalent; the shift form documents the
halving intent.

**Cost.** The tree has \\(\log_2 TILE\\) levels and one barrier per level. This
kernel pays \\(\log_2 256 = 8\\) barriers per block for a 256-thread block.
Stage 3 removes all but one.

## 8.4 Stage 2: Thread Coarsening

One element per thread leaves most threads idle after the initial load: each
thread loads one value and then participates in \\(\log_2 TILE\\) additions.
**Thread coarsening** makes each thread process many elements in a grid-stride
loop (Chapter 7, §7.3) before entering the tree:

```cpp
// Each thread accumulates ELEMS_PER_THREAD elements first (coalesced
// grid-stride loop), then ONE tree reduction over the block.
#define ELEMS_PER_THREAD 4

__global__ void reduceCoarsened(const float* in, float* out, int n)
{
    __shared__ float s[TILE];

    const int stride = gridDim.x * blockDim.x;    // total threads in grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride accumulation: each thread sums its share of the array.
    float sum = 0.0f;
    for (; i < n; i += stride)
        sum += in[i];                              // coalesced within each pass

    s[threadIdx.x] = sum;

    for (int stride2 = TILE / 2; stride2 > 0; stride2 >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride2)
            s[threadIdx.x] += s[threadIdx.x + stride2];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
```

Coarsening helps for three reasons: each thread issues several independent
loads before communicating, increasing memory-level parallelism; the serial
addition into a local register uses no barrier or shared memory; and fewer
blocks means fewer partial results to combine later. The grid-stride loop also
makes the kernel correct for any \\(n\\), not only multiples of the grid size.

## 8.5 Stage 3: Warp Shuffles

The tree's barriers are its main cost. A warp's 32 threads, however, can
exchange data through registers without touching memory or using a barrier. The
instruction is the **warp shuffle**:

> **Primitive - warp shuffle.** `__shfl_down_sync(mask, value, delta)` moves
> `value` from lane `lane + delta` to lane `lane` within one warp, through the
> register file. No memory access or barrier is required. `mask` is the set of
> participating lanes (all 32: `0xffffffff`). All 32 lanes must execute the
> shuffle or the behaviour is undefined.

```cpp
// Reduce a warp's 32 lanes to lane 0 using only shuffles: 5 steps.
__device__ float warpReduce(float val)
{
    // mask: all 32 lanes participate. Steps move data rightward by
    // 16, 8, 4, 2, 1 - the warp-size equivalents of the tree's strides.
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;      // lane 0 now holds the warp total
}
```

A warp has 32 lanes. The first shuffle moves the partial sums of lanes 16..31
into lanes 0..15, the second folds lanes 8..15 into 0..7, and so on. Five
steps reduce a warp, matching the \\(\log_2 32\\) levels of the shared-memory
tree, but at register speed with no barrier and no shared memory.

## 8.6 The Complete Block Reduction

The production form combines coarsening (§8.4), warp shuffles (§8.5), and one
shared-memory round per block. Each warp writes one value to shared memory,
one barrier makes those values visible, and the first warp combines them with
shuffles:

```cpp
#define TILE 256            // block size, a multiple of the warp size 32
#define WARPS (TILE / 32)   // 8 warps per block

// Warp-level reduction (from 8.5), returns the warp's total in lane 0.
__device__ float warpReduce(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

__global__ void reduceFull(const float* in, float* out, int n)
{
    __shared__ float s[WARPS];          // one slot per warp

    // --- Coarsened accumulation over the grid -----------------------------
    const int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        sum += in[i];

    // --- Warp-level reduction: 8 warps * 5 shuffle steps, no barriers -----
    // Lane 0 of each warp now holds that warp's partial total.
    sum = warpReduce(sum);

    // --- One shared-memory round ------------------------------------------
    const int lane   = threadIdx.x % 32;     // position within my warp
    const int warpId = threadIdx.x / 32;     // which warp am I

    if (lane == 0) s[warpId] = sum;          // each warp writes ONE value
    __syncthreads();                         // the ONLY barrier per block

    // --- First warp combines the 8 warp totals -----------------------------
    if (warpId == 0)
    {
        // Lane < WARPS loads a warp total; others load identity (0.0f).
        const float v = (lane < WARPS) ? s[lane] : 0.0f;
        sum = warpReduce(v);                 // shuffle again over 8 values
        if (lane == 0) out[blockIdx.x] = sum;   // block total
    }
}
```

**Cost.** One barrier per block, down from eight; one shared-memory round trip
per block; and five shuffle steps per warp. The only global traffic is the
coalesced read. This kernel is the standard against which reductions are
judged and routinely achieves over 90% of peak bandwidth.

**Determinism.** The tree order is fixed by the code, so the summation order is
fixed and the result is bit-reproducible across runs. An atomic-based reduction
(Chapter 5, §5.6) is not.

## 8.7 Scan (Prefix Sum)

> **Primitive - inclusive scan.** Given \\(a\\), compute
> \\(y_i = a_0 + a_1 + \cdots + a_i\\).
> **Primitive - exclusive scan.** Given \\(a\\), compute
> \\(y_i = a_0 + \cdots + a_{i-1}\\), with \\(y_0 = 0\\).
> An exclusive scan of an array is an inclusive scan shifted right by one.

Scans appear in stream compaction (filtering), radix sort, and workload
equalisation: algorithms that need to know where data lands. The work-efficient
**Blelloch scan** is the canonical GPU formulation. It has two phases:

1. **Upsweep** - a tree reduction that computes partial sums, as in §8.3, but
   stores the internal nodes instead of discarding them.
2. **Downsweep** - a second tree that propagates totals to produce exclusive
   prefix sums.

```cpp
// Exclusive scan of a BLOCK's data, in place in shared memory.
// Assumes blockDim.x is a power of two. After the call,
//   s[0] = 0, s[i] = a_0 + ... + a_{i-1}  (exclusive prefix sums)
// The block total ends in s[blockDim.x - 1].
__device__ void scanBlock(float* s)
{
    const int n = blockDim.x;

    // --- Phase 1: upsweep (tree reduction, storing internal nodes) --------
    // After level k, s[i] holds the sum of the 2^(k+1) elements ending at i.
    for (int stride = 1; stride < n; stride <<= 1)
    {
        __syncthreads();
        const int t = (threadIdx.x + 1) * 2 * stride - 1;   // right child
        if (t < n)
            s[t] += s[t - stride];                          // parent = sum
    }

    // --- Phase 2: downsweep (propagate carries back down) -----------------
    // First, the root's carry is the identity for addition: the sum of the
    // (empty) sequence before the whole array.
    if (threadIdx.x == 0) s[n - 1] = 0.0f;

    // Walk the tree top-down. At each level, the pair rooted at right child
    // t (left child t - stride) receives its CARRY - the sum of everything
    // before its subtree, which the parent level stored in s[t]:
    //   - the left child inherits the carry unchanged;
    //   - the right child gets carry + (its old value, the left subtree sum).
    // Inductively every slot ends up holding the sum of the elements before
    // it: the exclusive prefix.
    for (int stride = n / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        const int t = (threadIdx.x + 1) * 2 * stride - 1;   // right child
        if (t < n)
        {
            const float carry = s[t];            // sum before this pair
            s[t] = carry + s[t - stride];        // right child: carry + left sum
            s[t - stride] = carry;               // left child: just the carry
        }
    }
    __syncthreads();
}
```

The index arithmetic `(threadIdx.x + 1) * 2 * stride - 1` identifies the right
child of each subtree: within a group of size `2*stride`, these are exactly the
odd indices. The important property is that each level performs disjoint
writes; no two threads write the same slot, so no atomicity is needed.

The downsweep produces exclusive sums through the invariant of the carry. At
each level, `s[t]` holds the sum of everything before the pair `(t - stride,
t)`. The root's carry is set to zero before the loop. Each step passes the
carry unchanged to the left child and passes `carry + left_subtree_sum` to the
right child. Inductively, slot \\(i\\) ends with the sum of elements
\\(0..i-1\\). A hand trace on `[1, 2, 3, 4]` is Exercise 3. Note that `s[t]`
must be read into `carry` before `s[t - stride]` is overwritten.

**Cost.** The scan has two passes, each with \\(\log_2 n\\) barrier levels.
Here the barrier count is inherent to the algorithm rather than an
implementation defect. A single block can scan at most 1,024 elements. Larger
arrays require a two-level scheme of block scans plus a scan of block totals,
which the CUB library provides (Chapter 11).

## 8.8 Histogram: Privatisation

The naive histogram of Chapter 5 performs one global `atomicAdd` per element
and serialises on contended bins. The standard fix is **privatisation**: each
block accumulates into its own shared-memory histogram, where atomics are
cheaper, and one thread per block folds the private histogram into global
memory at the end.

```cpp
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
```

Contention now occurs in shared memory, where an atomic is roughly an order of
magnitude cheaper than a global atomic, and it is spread across 256 bins rather
than concentrated at one global address. The global fold touches each bin once
per block. For highly skewed data, all elements fall in one bin and shared
atomics still contend; the next-level fix is per-warp histograms. Privatisation
handles the common case.

The kernel uses two barriers, one after zeroing and one before the fold. Both
are uniformly reachable because the loops have compile-time trip counts.

## 8.9 Summary Table

| Algorithm | Naive cost | Optimised cost | Key tool |
|---|---|---|---|
| Reduction | \\(O(n)\\) serial or \\(\log_2 n\\) barriers | 1 barrier, \\(O(\log_2 32)\\) shuffles | Warp shuffle + coarsening |
| Scan | \\(O(n)\\) serial | \\(2 \log_2 n\\) barriers | Blelloch upsweep/downsweep |
| Histogram | global atomic per element | shared privatisation + fold | Per-block private bins |

## Where Partial Results Live

A reduction's arithmetic is simple; its performance is determined by how
partial results travel between threads, how often they touch memory, and how
often threads wait for one another.

A single-threaded reduction keeps a running sum in one register. There is no
communication, but only one thread works. A GPU has thousands of execution
units, so the problem is not how to add but how to divide the array among
threads, combine their results, and minimise the cost of combining.

Each reduction version answers that question differently:

1. **The naive kernel** keeps everything in one thread. It is correct but uses
   one thread out of thousands.
2. **The shared-memory tree** divides work among all threads of a block and
   combines partial sums through shared memory. Its cost is a barrier at every
   tree level.
3. **The coarsened kernel** accumulates several elements per thread in a
   register before entering the tree, reducing the frequency of communication.
4. **The warp-shuffle kernel** moves data between lanes through registers.
   Shuffles need no barrier because the lanes of a warp execute together.
5. **The full kernel** combines all of these: coarsen, shuffle within warps,
   write one value per warp to shared memory, one barrier, and a final shuffle
   over warp totals.

Scan is a reduction with a harder requirement: every output position needs the
sum of everything before it. The Blelloch scan solves it with an upsweep that
stores internal tree nodes and a downsweep that propagates carries down the
tree. The arithmetic is still addition; the work is in the index arithmetic and
in ordering writes so that a thread reads a value before it is overwritten.

Histogram privatisation applies the same principle to atomics. Global atomics
are expensive when many threads target the same bin because the hardware
serialises read-modify-write cycles. A privatised histogram accumulates in
shared memory and folds into global memory once. Communicate as little as
possible, and when communication is unavoidable, do it in bulk.

Reduction, scan, and histogram are three views of one problem: combining
distributed data with minimal communication. The patterns recur in matrix
multiplication (Chapter 9), library primitives (Chapter 11), and multi-GPU
collectives (Chapter 19).

## Common Pitfalls

- Using non-power-of-two block sizes with tree algorithms. The halving scheme
  assumes `TILE` is a power of two.
- Calling `warpReduce` from only some lanes. All 32 lanes must execute
  `__shfl_down_sync` with the same mask, or the behaviour is undefined.
- Forgetting the final `__syncthreads()` after a shared-memory scan. The last
  read must not begin before the last write is visible.
- Using global atomics for the whole histogram instead of privatising. Global
  contention serialises; shared privatisation plus one fold per block is the
  standard fix.

## Check Your Understanding

<details>
<summary>Why is one barrier per block enough in reduceFull?</summary>

Each warp reduces its 32 lanes with shuffles, using no memory or barrier. Only
one value per warp is written to shared memory, so one barrier makes those
values visible to warp 0, which then combines them with shuffles.
</details>

<details>
<summary>What does warpReduce return for lanes other than lane 0?</summary>

The shuffle loop leaves partial values in every lane. The documented result,
and the value used by the kernel, is the total in lane 0. Other lanes contain
intermediate sums and should not be used.
</details>

<details>
<summary>Why is a fixed tree reduction bit-reproducible but an atomic reduction is not?</summary>

A fixed tree always sums in the same order, so floating-point rounding is
identical on every run. Atomics allow the hardware to choose an order that can
differ between runs, changing the last bits.
</details>

## Key Takeaways

- The optimised reduction: coarsen (grid-stride), reduce within each warp by shuffle, then one shared-memory round and one barrier per block.
- Warp shuffles exchange values through registers, with no memory or barrier.
- The Blelloch scan is an upsweep that stores internal nodes and a downsweep that propagates carries to build exclusive prefixes.
- Histograms should be privatised per block in shared memory and folded into global memory once.
- A fixed tree order makes reductions bit-reproducible across runs.

## 8.10 Exercises

1. Derive the number of barriers in the Stage-1 tree reduction for
   `TILE = 512`, and compare it with the full kernel of §8.6.
2. In `reduceFull`, why does lane 0 of each warp write `s[warpId]` and not
   every lane? What would happen if every lane wrote its own `sum`?
3. Trace the Blelloch downsweep on `[1, 2, 3, 4]` by hand, showing the state
   of `s` after each level. Verify the exclusive prefix `[0, 1, 3, 6]`.
4. The histogram fold skips empty bins with `if (s_hist[b] != 0)`. Is this
   correct? Is it always faster? Consider a case where every bin is non-empty.
