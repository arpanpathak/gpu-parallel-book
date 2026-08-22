// Chapter 5, 5.5.1 - privatised histogram and spinlock, verbatim from the book.

// Count how many elements fall into each of 256 bins.
// data   : device array of unsigned char (0..255), n elements
// hist   : device array of 256 ints, zero-initialised on the host
// bins   : the bin count, 256
__global__ void histogram(const unsigned char* data, int* hist, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int bin = data[i];               // value is the bin index
        // The atomic serialises concurrent increments to the same bin.
        // Without it, two threads could both read old, both add 1, and
        // both write old+1 - losing one count (the classic lost update).
        atomicAdd(&hist[bin], 1);
    }
}

// Chapter 5, 5.5.2 - a shared-memory spinlock built from atomicCAS.

// A simple lock in shared memory. One mutex per block.
__global__ void lockedCriticalSection(float* g_data, int n)
{
    __shared__ int s_lock;    // 0 = unlocked, 1 = locked

    // Thread 0 initialises the lock. (Alternatively use a static initialiser.)
    if (threadIdx.x == 0) s_lock = 0;
    __syncthreads();          // everyone must see the lock before using it

    // Each block owns a disjoint strided partition of g_data. Without the
    // blockIdx.x offset, every block would process the SAME indices and the
    // per-block locks could not protect against cross-block races.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        // Acquire: atomically swap 1 into the lock; if the old value was 0,
        // we won the lock. If it was 1, someone else holds it - retry.
        while (atomicCAS(&s_lock, 0, 1) != 0) { /* spin */ }

        // Critical section: safe because we exclusively hold the lock.
        g_data[i] = g_data[i] * 2.0f + 1.0f;

        // Release: plain store is fine here (see the fence discussion below),
        // but a __threadfence_block() before it is the textbook-safe form.
        __threadfence_block();
        s_lock = 0;
    }
}
