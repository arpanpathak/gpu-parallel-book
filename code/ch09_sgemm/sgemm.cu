// Chapter 9 - optimised matrix multiplication kernels, verbatim from the book.

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

// Tell ptxas: this kernel must fit in at most 256 threads/block, and I
// want at least 4 blocks/SM resident. ptxas will trade registers for
// occupancy within the budget rather than silently spilling.
__global__ void __launch_bounds__(256, 4)
sgemmRegisterTiled(const float* A, const float* B, float* C, int N) { /* ... */ }
