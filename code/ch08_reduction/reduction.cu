// Chapter 8 - warp-shuffle reduction, Blelloch scan, privatised histogram.
// All code verbatim from the book.

// Reduce a warp's 32 lanes to lane 0 using only shuffles: 5 steps.
__device__ float warpReduce(float val)
{
    // mask: all 32 lanes participate. Steps move data rightward by
    // 16, 8, 4, 2, 1 - the warp-size equivalents of the tree's strides.
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;      // lane 0 now holds the warp total
}

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
