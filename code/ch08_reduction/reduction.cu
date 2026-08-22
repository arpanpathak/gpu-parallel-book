// Chapter 8 - reduction, scan and histogram kernels, all verbatim from the
// book (8.2-8.8). The runnable test harness lives in main.cu.
//
// Build: nvcc -rdc=true -arch=compute_60 reduction.cu main.cu -o reduction
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./reduction

// ===========================================================================
// 8.2 Stage 0: one thread does it all (reference implementation)
// ===========================================================================
__global__ void reduceNaive(const float* in, float* out, int n)
{
    if (threadIdx.x == 0)              // ONE thread
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += in[i];   // serial, n additions
        out[0] = sum;
    }
}

// ===========================================================================
// 8.3 Stage 1: tree reduction in shared memory
// ===========================================================================
#ifndef TILE
#define TILE 256
#endif

__global__ void reduceTree(const float* in, float* out, int n)
{
    __shared__ float s[TILE];          // TILE = blockDim.x, a power of two

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load with a boundary guard; out-of-range elements contribute zero.
    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;

    // Tree: each level halves the number of active threads.
    for (int stride = TILE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();               // everyone's writes visible
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
    }

    // Thread 0 owns the block's total.
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// ===========================================================================
// 8.4 Stage 2: thread coarsening
// ===========================================================================
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

// ===========================================================================
// 8.5 + 8.6 warp shuffle reduction and the complete block reduction
// ===========================================================================
#define WARPS (TILE / 32)   // 8 warps per 256-thread block

// Reduce a warp's 32 lanes to lane 0 using only shuffles: 5 steps.
__device__ float warpReduce(float val)
{
    // mask: all 32 lanes participate. Steps move data rightward by
    // 16, 8, 4, 2, 1 - the warp-size equivalents of the tree's strides.
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;      // lane 0 now holds the warp total
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

// ===========================================================================
// 8.7 Blelloch exclusive scan (block-level, in shared memory)
// ===========================================================================
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

// ===========================================================================
// 8.8 Privatised histogram
// ===========================================================================
#define BINS 256

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
