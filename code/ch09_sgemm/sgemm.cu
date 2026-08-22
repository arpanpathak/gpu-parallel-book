// Chapter 9 - optimised matrix multiplication kernels, all verbatim from the
// book (9.2-9.6). The runnable test harness lives in main.cu.
//
// Build: nvcc -arch=compute_60 sgemm.cu main.cu -o sgemm
//        (Jetson Orin: -arch=sm_87; A100: -arch=sm_80; H100: -arch=sm_90)
// Run:   ./sgemm

// ===========================================================================
// 9.2 Stage 0: the naive kernel
// ===========================================================================
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

// ===========================================================================
// 9.3 Stage 1: make the block shape match the memory
// ===========================================================================
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

// ===========================================================================
// 9.4 Stage 2: shared-memory tiling
// ===========================================================================
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

// ===========================================================================
// 9.5 Stage 3: register tiling (final form with the __launch_bounds__ from
// 9.6 applied directly to the definition)
// ===========================================================================
#define RM 2          // rows of output per thread
#define RN 2          // cols of output per thread

// Tell ptxas: this kernel must fit in at most 256 threads/block, and I
// want at least 4 blocks/SM resident. ptxas will trade registers for
// occupancy within the budget rather than silently spilling.
__global__ void __launch_bounds__(256, 4)
sgemmRegisterTiled(const float* A, const float* B, float* C, int N)
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
