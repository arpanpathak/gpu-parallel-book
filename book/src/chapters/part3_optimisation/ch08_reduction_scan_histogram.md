# Chapter 8: Reduction, Scan & Histogram

> *"Every parallel algorithm is a reduction, a scan, or a lie."*

This chapter covers the three canonical data-parallel algorithms that appear,
in disguise, in almost every real GPU application: **reduction** (sum,
max, min — fold an array to one value), **scan** (prefix sums — fold an array
to an array), and **histogram** (count occurrences). Each is presented in
stages, from the naive version to the optimised one, with the reasoning for
every transformation. These three patterns are the vocabulary of Chapter 9's
matrix multiplication and the capstone's image pipeline.

## 8.1 The Reduction Problem

Given an array \\(a\\) of \\(n\\) elements, compute
\\(\Sigma_{i=0}^{n-1} a_i\\). On a CPU this is a loop of \\(n\\) additions. On
a GPU the challenge is different: additions are cheap, but *combining results
across threads* requires communication, and communication is the expensive
thing. A good reduction minimises the number of communication rounds and keeps
every thread busy between them.

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

This "works" and is useless: one thread, thousands of cycles, 0.003% of the
GPU used. Its only value is as the reference implementation whose answer we
check the real kernels against.

## 8.3 Stage 1: Tree Reduction in Shared Memory

The insight: addition is associative, so we may *reorder* the additions into a
tree. In round 1, threads 0..n/2-1 each add two elements; in round 2,
threads 0..n/4-1 each add two partials; and so on. The tree has
\\(\log_2 n\\) levels, so \\(n/2\\) additions complete in \\(\log_2 n\\)
parallel rounds.

```cpp
// Reduce one block's worth of data (blockDim.x elements per thread is
// handled in Stage 2; here: one element per thread) using shared memory.
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
        __syncthreads();               // everyone's writes visible
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
    }

    // Thread 0 owns the block's total.
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
```

**Why `__syncthreads()` inside the loop?** At each level, thread `t` reads a
partial sum written by thread `t + stride` *at the previous level*. The
barrier guarantees the previous level's writes are complete and visible before
the next level reads them.

**Why a power-of-two block size?** The halving scheme assumes `TILE` is a
power of two, so every level halves evenly and the active set
`threadIdx.x < stride` is contiguous. Non-power-of-two block sizes make the
active-set arithmetic messy for no benefit; 128, 256 and 512 dominate practice.

**Why `stride >>= 1` and not `stride /= 2`?** For unsigned sizes both are
identical; `>> 1` documents the halving intent and avoids any doubt about
integer division semantics. Either is correct.

**Cost accounting.** The tree has \\(\log_2 TILE\\) levels; each level is one
barrier. This kernel therefore pays \\(\log_2 256 = 8\\) barriers per block
for 256 threads. Stage 3 removes all but one.

## 8.4 Stage 2: Thread Coarsening

One element per thread leaves most of each thread idle: each thread loads one
value and then participates in \\(\log_2 TILE\\) additions. The fix is
**thread coarsening** — each thread processes *many* elements in a grid-stride
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

**Why is this faster?** Three reasons: (1) each thread loads 4+ values before
any communication, so memory-level parallelism is higher; (2) the serial
addition into a local register is free (no barrier, no shared memory); (3)
fewer blocks means fewer block-level reductions to combine later. The
grid-stride loop also makes the kernel *correct for any n*, not just
multiples of the grid size.

## 8.5 Stage 3: Warp Shuffles

The tree's barriers are the cost. But a warp's 32 threads share an execution
unit — they can exchange data through **registers** without touching memory or
barriers at all. The instruction is the **warp shuffle**:

> **Primitive — warp shuffle.** `__shfl_down_sync(mask, value, delta)` moves
> `value` from lane `lane + delta` to lane `lane`, within one warp, through
> the register file. No memory, no barrier. `mask` is the set of participating
> lanes (all 32: `0xffffffff`). All 32 lanes must execute the shuffle or the
> behaviour is undefined.

```cpp
// Reduce a warp's 32 lanes to lane 0 using only shuffles: 5 steps.
__device__ float warpReduce(float val)
{
    // mask: all 32 lanes participate. Steps move data rightward by
    // 16, 8, 4, 2, 1 — the warp-size equivalents of the tree's strides.
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;      // lane 0 now holds the warp total
}
```

**Why 16, 8, 4, 2, 1?** A warp is 32 lanes. The first shuffle moves the sum
of lanes 16..31 into lanes 0..15; the second folds 8..15 into 0..7; and so on.
Five steps halve a warp — the same \\(\log_2 32\\) levels as the shared-memory
tree, but at register speed with **no barrier and no shared memory**.

## 8.6 The Complete Block Reduction

The production form combines everything: coarsening (§8.4), warp shuffle
(§8.5), and exactly **one** shared-memory transaction + one barrier per block
(one value per warp written to shared, then warp 0's shuffle over those
values):

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

**The accounting.** One barrier per block (down from 8), one shared-memory
round trip per block, five shuffle steps per warp. The shuffle steps are the
only "communication" the bulk of the work ever performs. This kernel is the
standard against which reductions are judged; it routinely achieves 90%+ of
peak bandwidth because its only global traffic is the coalesced read.

**Determinism.** The tree order is fixed by the code, so the summation order
is fixed — the result is *bit-reproducible* across runs, unlike an
atomic-based reduction (Chapter 5, §5.6). This is a feature, not an accident.

## 8.7 Scan (Prefix Sum)

> **Primitive — inclusive scan.** Given \\(a\\), compute
> \\(y_i = a_0 + a_1 + \cdots + a_i\\).
> **Primitive — exclusive scan.** Given \\(a\\), compute
> \\(y_i = a_0 + \cdots + a_{i-1}\\), with \\(y_0 = 0\\).
> An exclusive scan of an array is an inclusive scan shifted right by one.

Scans appear in stream compaction (filtering), radix sort, and the
equalisation of workloads — any algorithm that needs *where* the data went.
The work-efficient **Blelloch scan** is the canonical GPU formulation. It has
two phases:

1. **Upsweep** — a tree reduction that computes partial sums (exactly the
   tree of §8.3, but *storing* the internal nodes instead of discarding them).
2. **Downsweep** — a second tree that *propagates* the totals to produce
   exclusive prefix sums.

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
    // t (left child t - stride) receives its CARRY — the sum of everything
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

**Why the index arithmetic `(threadIdx.x + 1) * 2 * stride - 1`?** In the
upsweep at level `stride`, the element at index `t` is the right child of the
subtree whose left child ends at `t - stride`. Writing `t` as
`(threadIdx.x + 1) * 2 * stride - 1` gives the *odd* indices within each
`2*stride` group: exactly the right children. The formula is fiddly; the
*property* that matters is that each level is a disjoint set of writes — no
two threads write the same slot, so no atomicity is needed.

**Why does the downsweep produce *exclusive* sums?** The invariant is the
*carry*: at each level, `s[t]` holds the sum of everything before the pair
`(t - stride, t)`. The root's carry is set to `0` (the identity) before the
loop. Each step hands the carry to the left child unchanged, and passes
`carry + (left child's old value)` — the sum of everything before the *right*
child — down the right side. Inductively, slot `i` ends up holding the sum of
elements `0..i-1` — the exclusive prefix. A trace on `[1, 2, 3, 4]` is
Exercise 3 below; note the order of the writes: `s[t]` must be read into
`carry` *before* `s[t - stride]` is overwritten.

**The cost.** Two passes, each with \\(\log_2 n\\) barrier levels: the scan
is the rare case where the number of barriers is *inherent* to the algorithm,
not an implementation defect. A single block can scan at most 1,024 elements;
scanning more requires a two-level scheme (block scans + a scan of block
totals), which the library CUB provides ready-made (Chapter 11).

## 8.8 Histogram: Privatisation

The naive histogram of Chapter 5 (`atomicAdd` per element) serialises on
contended bins. The production fix is **privatisation**: every block
accumulates into its *own* shared-memory histogram (shared-memory atomics are
much cheaper than global), and one thread per block folds the private
histograms into global memory once at the end.

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

**Why is this fast?** Contention now happens in shared memory, where an
atomic is roughly an order of magnitude cheaper than a global atomic, and it
is spread across 256 bins instead of funneled into one address per warp. The
global fold touches each bin once per block. For skewed data (all elements in
one bin), the shared atomics still contend — the next-level fix is *per-warp*
histograms — but privatisation handles the common case.

**The barrier count.** Two barriers: one after zeroing, one before the fold.
Both are uniformly reachable — the loops are compile-time shaped, so no thread
can skip a barrier.

## 8.9 Summary Table

| Algorithm | Naive cost | Optimised cost | Key tool |
|---|---|---|---|
| Reduction | \\(O(n)\\) serial or \\(\log_2 n\\) barriers | 1 barrier, \\(O(\log_2 32)\\) shuffles | Warp shuffle + coarsening |
| Scan | \\(O(n)\\) serial | \\(2 \log_2 n\\) barriers | Blelloch upsweep/downsweep |
| Histogram | global atomic per element | shared privatisation + fold | Per-block private bins |

## 8.10 Exercises

1. Derive the number of barriers in the Stage-1 tree reduction for
   `TILE = 512`, and compare it with the full kernel of §8.6.
2. In `reduceFull`, why does lane 0 of each warp *write* `s[warpId]` and
   not every lane? What would happen if every lane wrote its own `sum`?
3. Trace the Blelloch downsweep on `[1, 2, 3, 4]` by hand, showing the state
   of `s` after each level. Verify the exclusive prefix `[0, 1, 3, 6]`.
4. The histogram fold skips empty bins with `if (s_hist[b] != 0)`. Is this
   correct? Is it always faster? (Consider a case where every bin is
   non-empty.)
