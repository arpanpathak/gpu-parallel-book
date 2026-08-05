// Chapter 3, 3.6 - the first complete CUDA program, verbatim from the book.
// Build: nvcc -arch=sm_90 vector_add.cu -o vector_add

// kernel.cu
// ---------------------------------------------------------------------------
// __global__ : this function runs on the DEVICE, launched from the host.
// Return type must be void. The argument is a device pointer to n floats.
// ---------------------------------------------------------------------------
__global__ void addVectors(const float* a, const float* b, float* c, int n)
{
    // --- Who am I? -----------------------------------------------------
    // blockIdx.x : index of my block within the grid (0-based).
    // blockDim.x : number of threads in my block (set at launch).
    // threadIdx.x: index of my thread within my block (0-based).
    // The product blockIdx.x * blockDim.x is the first global thread index
    // covered by my block; adding threadIdx.x gives my global index.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Boundary guard -------------------------------------------------
    // The grid may cover more threads than n (we launch a rounded-up grid,
    // see the host code). Threads whose index is >= n must do nothing.
    // Without this guard we would read and write past the end of the arrays
    // - an out-of-bounds memory error on the device.
    if (i < n)
    {
        // Element-wise add. Each thread owns exactly one output element,
        // so no two threads ever write the same address: no races here.
        c[i] = a[i] + b[i];
    }
}

#include <cstdio>
#include <cuda_runtime.h>   // All CUDA runtime API declarations live here.

// ---------------------------------------------------------------------------
// Error-checking helper (see 3.8). Every CUDA call that can fail is routed
// through CHECK, which prints the file and line on failure and aborts.
// ---------------------------------------------------------------------------
#define CHECK(call)                                                       \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

int main()
{
    // --- Problem size -----------------------------------------------------
    // n is the number of elements; nBytes is the byte size of each array.
    // We use size_t because array sizes can exceed the range of int.
    const int    n      = 1 << 20;      // 1,048,576 elements (a power of two)
    const size_t nBytes = n * sizeof(float);

    // --- Host allocations (pageable memory, see Chapter 4) ---------------
    float* h_a = new float[n];   // host input A
    float* h_b = new float[n];   // host input B
    float* h_c = new float[n];   // host output C

    // Fill the inputs with a deterministic pattern so we can verify results.
    for (int i = 0; i < n; ++i) { h_a[i] = 1.0f * i;  h_b[i] = 2.0f * i; }

    // --- Device allocations ----------------------------------------------
    // cudaMalloc allocates in GLOBAL MEMORY on the device. The returned
    // pointers are valid only on the device (see 3.1).
    float* d_a = nullptr;   // device input A
    float* d_b = nullptr;   // device input B
    float* d_c = nullptr;   // device output C
    CHECK(cudaMalloc((void**)&d_a, nBytes));
    CHECK(cudaMalloc((void**)&d_b, nBytes));
    CHECK(cudaMalloc((void**)&d_c, nBytes));

    // --- Host -> device copy ----------------------------------------------
    // cudaMemcpy(dst, src, bytes, kind). The kind cudaMemcpyHostToDevice
    // tells the runtime the direction of the copy (see 3.7).
    CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));

    // --- Launch configuration --------------------------------------------
    // Block size: 256 threads per block. Why 256? A multiple of the warp
    // size (32) so every warp is full, and small enough that many blocks
    // fit per SM (see occupancy, 2.9). Values of 128-512 are typical.
    const int threadsPerBlock = 256;
    // Grid size: ceil(n / threadsPerBlock). The + (threadsPerBlock - 1)
    // trick rounds UP so that the grid covers every element. Some threads
    // will therefore exceed n and hit the boundary guard in the kernel.
    const int blocksPerGrid  = (n + threadsPerBlock - 1) / threadsPerBlock;

    // --- Launch ------------------------------------------------------------
    // addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    // The launch is asynchronous: the host does NOT wait for the kernel;
    // control returns to the host immediately (see Chapter 6).
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Kernel launches do not report errors synchronously. Check the last
    // error now; if the launch itself failed (bad config, bad pointer),
    // this catches it.
    CHECK(cudaGetLastError());

    // --- Synchronise --------------------------------------------------------
    // cudaDeviceSynchronize blocks the host until ALL device work issued
    // so far has completed. Required before we copy the results back.
    CHECK(cudaDeviceSynchronize());

    // --- Device -> host copy ------------------------------------------------
    CHECK(cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost));

    // --- Verify -------------------------------------------------------------
    // We know the correct answer: h_c[i] should equal 3*i. Verify a few
    // samples and report the worst error.
    double maxErr = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double err = std::abs(static_cast<double>(h_c[i]) - 3.0 * i);
        if (err > maxErr) maxErr = err;
    }
    std::printf("max error = %g\n", maxErr);

    // --- Cleanup ------------------------------------------------------------
    delete[] h_a;  delete[] h_b;  delete[] h_c;
    CHECK(cudaFree(d_a));  CHECK(cudaFree(d_b));  CHECK(cudaFree(d_c));
    return 0;
}
