# Chapter 11: The Library Ecosystem - Thrust, CUB & cuBLAS

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch11_library_examples/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch11_library_examples)
> in the repository.

Chapters 3-10 covered writing kernels. This chapter covers when not to. The
CUDA ecosystem ships three libraries that implement, in production-tuned form,
most of the algorithms of Chapters 8 and 9: **Thrust** (high-level algorithms),
**CUB** (block-level primitives), and **cuBLAS** (dense linear algebra). A
professional GPU engineer uses them first and writes custom kernels only where
the libraries cannot express the problem. The earlier chapters are what you
need to understand what the libraries do underneath.

## 11.1 The Case for Libraries

NVIDIA's libraries are tuned for every generation of GPU by engineers with
access to the hardware design:

- They dispatch to **architecture-specific kernels**, including tensor cores
  for GEMM and custom shuffle reductions for scans.
- They are **hand-optimised** beyond what a compiler alone achieves.
- They are **tested** against known-good references and profiled on every
  release.

A hand-written SGEMM that reaches 70% of peak (Chapter 9) is good; cuBLAS often
reaches about 95% of peak with tensor cores. The engineering decision is not
"libraries or custom kernels". It is "does my problem fit a library?"

## 11.2 Thrust: STL for the GPU

Thrust is the closest CUDA library to the C++ standard library. It provides
containers such as `thrust::device_vector` and algorithms such as `transform`,
`reduce`, `sort`, and `exclusive_scan` that operate on device memory with
`std::`-like syntax:

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// Thrust version of a Chapter 8 reduction and a Chapter 7 SAXPY. The
// algorithms dispatch to tuned kernels internally; the host code describes
// WHAT, not HOW.
// ---------------------------------------------------------------------------
void thrustExample(int n)
{
    // device_vector: an RAII device array (like Chapter 10's DeviceBuffer).
    thrust::device_vector<float> x(n), y(n);

    // thrust::sequence fills x with 0..n-1 (parallel, device-side).
    thrust::sequence(x.begin(), x.end());

    // thrust::transform applies a functor elementwise. thrust::device is the
    // execution policy that says "run on the GPU"; it must be the FIRST
    // argument (policy, first, last, result, op).
    thrust::transform(thrust::device,
                      x.begin(), x.end(), y.begin(),
                      [] __device__ (float v) { return v * 2.0f + 1.0f; });

    // thrust::reduce folds the array (the tuned reduction).
    const float total = thrust::reduce(y.begin(), y.end(), 0.0f,
                                       thrust::plus<float>());

    // thrust::sort, thrust::exclusive_scan, etc. follow the same shape.
    thrust::sort(y.begin(), y.end());
}
```

The lambda is marked `__device__` because Thrust must compile the functor for
the device. Modern Thrust can infer `__host__ __device__` for plain lambdas,
but the explicit qualifier makes the intent unambiguous and is the documented
style.

`thrust::device_vector` and algorithm dispatch carry their own allocation and
launch logic. For a one-off reduction of a large array, the overhead is noise.
For a per-frame micro-pipeline in a tight loop, it is not. Measure (Chapter 16)
before assuming.

## 11.3 CUB: Block-Level Primitives

Thrust works at the container level. **CUB** works at the block level: it
provides `cub::BlockReduce`, `cub::BlockScan`, `cub::BlockHistogram`, and
`cub::WarpReduce` for embedding in custom kernels. Use CUB when a kernel needs
a block-sized reduction, scan, or histogram inside custom logic:

```cpp
#include <cub/cub.cuh>

// A kernel that reduces its block's partial sums using CUB. CUB's BlockReduce
// is the tuned version of Chapter 8's reduceFull.
template <int BLOCK_THREADS>
__global__ void reduceWithCub(const float* in, float* out, int n)
{
    // CUB block-reduction scratch space (compile-time sized).
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    // Coarsened accumulation (Chapter 8, 8.4):
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         i < n; i += gridDim.x * BLOCK_THREADS)
        sum += in[i];

    // Block-wide reduction with CUB. Sum() is the "combine" operation;
    // cub::Sum is a device-side functor wrapping fadd.
    const float blockSum = BlockReduceT(temp_storage).Sum(sum);

    // Thread 0 of each block writes the block total:
    if (threadIdx.x == 0) out[blockIdx.x] = blockSum;
}
```

CUB's `BlockReduce` handles the edge cases a hand-written kernel would need to
debug: non-power-of-two block sizes, the choice between shuffle-only and
shuffle-plus-shared strategies, and architecture-specific tuning. The Chapter 8
kernel is the explanation; CUB is the implementation to ship.

CUB is header-only and template-heavy, so compile times grow and error messages
can be intimidating. The runtime cost is zero because the templates compile to
the same SASS as a hand-written kernel.

## 11.4 cuBLAS: Dense Linear Algebra

cuBLAS implements the BLAS (Basic Linear Algebra Subprograms) interface for
GPUs: `sgemm`, `saxpy`, `sdot`, `sgemv`, and batched variants used by deep
learning. Its API uses the classic handle-based C-library style:

```cpp
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// C = alpha * A * B + beta * C   (the GEMM of Chapter 9, via cuBLAS)
// ---------------------------------------------------------------------------
void gemmViaCublas(const float* dA, const float* dB, float* dC,
                   int m, int n, int k, float alpha, float beta)
{
    // cuBLAS calls are not thread-safe by default; each context needs its own
    // handle. Create one handle per context and reuse it for all calls.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS, like Fortran BLAS, is COLUMN-major: matrix columns are
    // contiguous. Row-major C (m x n) = A (m x k) * B (k x n) has the same
    // flat memory layout as column-major C^T = B^T * A^T, so the operands and
    // sizes are swapped and the transposed product is computed:
    //   cublasSgemm(..., n, m, k, ..., dB, ldb, dA, lda, ..., dC, ldc)
    // Each leading dimension is the number of rows in the column-major view:
    const int ldb = n;   // B is k x n row-major -> B^T is n x k col-major, ld = n
    const int lda = k;   // A is m x k row-major -> A^T is k x m col-major, ld = k
    const int ldc = n;   // C is m x n row-major -> C^T is n x m col-major, ld = n

    // cuBLAS returns a status, not an exception. Check it:
    const cublasStatus_t status =
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,   // no transposes; the swap does the work
                    n, m, k,                    // SWAPPED sizes: compute C^T = B^T * A^T
                    &alpha,
                    dB, ldb,                    // B first (it plays the "A" role)
                    dA, lda,                    // A second (it plays the "B" role)
                    &beta,
                    dC, ldc);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm failed");

    cublasDestroy(handle);
}
```

The handle carries per-context state: stream association, workspace, and
heuristics. It lets the library keep state without global variables, which
would break multi-context and multi-threaded programs. The handle's stream can
be set with `cublasSetStream(handle, stream)` so cuBLAS calls participate in
the pipeline of Chapter 6.

The column-major convention is the classic source of bugs. Row-major data must
either be transposed (with `CUBLAS_OP_T`) or have its dimensions and operands
swapped as shown above. When in doubt, verify with a 2x2 example before scaling
up.

For large matrices, cuBLAS uses tensor cores and reaches 90%+ of peak with no
optimisation effort, compared with roughly 70% for the hand-written Chapter 9
kernel. The Chapter 9 kernel's value is understanding; the cuBLAS call's value
is shipping.

## 11.5 cuFFT and cuRAND

Two more libraries complete the common toolkit:

- **cuFFT** computes Fast Fourier Transforms in 1-D, 2-D, and 3-D, batched,
  complex, and real. Hand-written GPU FFTs are a research project; cuFFT is a
  product. Its API follows the FFTW planner model: create a plan describing the
  transform, then execute it on different data.
- **cuRAND** generates random numbers on the device with several generators
  (XORWOW, MRG32k3a, Philox, and others) and distributions (uniform, normal,
  Poisson). It can generate device-side, so kernels can draw random numbers
  internally for Monte Carlo work.

Both follow the handle/plan pattern: create once, configure, execute
repeatedly.

## 11.6 Library or Custom Kernel?

Given a GPU problem, evaluate in order:

1. **Is it an algorithm in Thrust, CUB, cuBLAS, cuFFT, or cuRAND?** Use the
   library. The tuned, tested version beats a first custom kernel, and the
   saved time goes into profiling the parts that matter.
2. **Is the library call the bottleneck?** Profile it (Chapter 16). If it is,
   determine whether the data layout fits the library's assumptions (leading
   dimensions, transposes). That is usually fixable without a custom kernel.
3. **Does the problem require custom per-element logic?** Write a custom
   kernel, but use CUB block primitives inside it. Custom does not mean
   from-scratch.
4. **Is the custom kernel measured as the hot path?** Only then hand-roll the
   full optimisation progression (Chapter 9).

This procedure prevents the reverse failure: rewriting `thrust::sort` because
"it might be faster" while the real bottleneck sits in a poorly coalesced custom
kernel nearby. Measure first; the library is the default.

**Worked decision: normalise a 100M-float array.** Normalisation is
elementwise, so `thrust::transform` with a functor is the library answer. There
is no custom logic to isolate, and no measured hot path yet. The correct move is
one `thrust::transform` call, which reaches roughly 95% of bandwidth. A
hand-tuned kernel would gain nothing because the transform is memory-bound and
Thrust's dispatcher already coalesces the access (Chapter 1).

**Worked decision: a 7-tap separable blur per frame.** No library call matches
a stencil with a halo. The halo logic is custom, so write a custom kernel with
coalesced row reads and clamped indices; Chapter 15's `blurH` is exactly this
shape. Only if the profiler shows that kernel as the pipeline bottleneck should
shared-memory tiling (Chapter 7) be added.

In both cases the decision is driven by the algorithm's shape and the
profiler's numbers, not by preference.

## Libraries and Understanding

Using Thrust, CUB, or cuBLAS does not remove the need to understand reductions,
scans, or GEMM. These libraries are tuned implementations of the algorithms
studied in Chapters 8 and 9, and their value depends on understanding what they
do underneath.

When `thrust::reduce` is called, the library solves the same reduction problem
with the same strategies: tree reductions, warp shuffles, shared-memory tiles,
and architecture-specific tuning. The difference is that the library version is
tested and tuned. Understanding the algorithm makes it possible to predict when
the library will be fast (large arrays, standard types) and when it may not be
(tiny arrays, unusual layouts, per-frame allocation overhead).

The same applies to CUB. `cub::BlockReduce` is the production version of
Chapter 8's `reduceFull` with edge cases already handled. Without the
hand-written version, CUB's template errors and tuning knobs are difficult to
read. With it, CUB is a tool that can be deployed confidently.

cuBLAS is the clearest example of why understanding matters. It is column-major
because it inherits Fortran BLAS conventions. Feeding row-major data with
`CUBLAS_OP_N` silently computes a transposed product. This is not a library
bug; it is a mismatch between the caller's layout assumption and the library's
definition. The dimension-swap trick in §11.4 works because transposition is
understood mathematically.

The professional workflow is: know the algorithm, reach for the library,
profile it, and hand-roll only when the profiler proves the library is the
bottleneck.

## Common Pitfalls

- Passing row-major data to cuBLAS with `CUBLAS_OP_N` and receiving the
  transposed answer. Transpose the operands or use the dimension-swap trick
  from §11.4.
- Using `thrust::device_vector` in a per-frame hot loop. Its convenience
  carries allocation and dispatch overhead; measure before assuming it is free.
- Reaching for CUB without understanding its template syntax. The errors are
  intimidating, but the pattern of temp storage plus `Sum()` is small once
  seen.
- Rewriting a library call "because it might be faster". The decision procedure
  requires a profiler, not a hunch.

## Check Your Understanding

<details>
<summary>Why does cuBLAS need leading dimensions?</summary>

Matrices can be sub-matrices (tiles) of larger buffers. `lda` tells cuBLAS how
many elements separate the start of one row or column from the next, so it can
walk a sub-matrix correctly instead of assuming full density.
</details>

<details>
<summary>What does CUB's BlockReduce provide beyond Chapter 8's reduceFull?</summary>

CUB handles non-power-of-two block sizes, architecture-specific tuning, and
edge cases the hand-written kernel would need to debug. The Chapter 8 kernel is
the explanation; CUB is the tested implementation.
</details>

<details>
<summary>When should thrust::sort be replaced with a custom radix sort?</summary>

Only when profiling shows `thrust::sort` is a significant fraction of runtime
and the data or keys have properties a radix sort can exploit, such as
fixed-size keys or a known range. Measure sort time and end-to-end time before
and after.
</details>

## Key Takeaways

- Use tuned, tested libraries first: Thrust (algorithms), CUB (block primitives), cuBLAS (dense linear algebra).
- Thrust mirrors the STL: device_vector, transform, reduce, sort, scans.
- CUB slots into custom kernels: cub::BlockReduce, cub::BlockScan, cub::BlockHistogram.
- cuBLAS is handle-based and column-major; the transpose trap is the classic bug.
- Replace a library call only when the profiler proves it is the bottleneck.

## 11.7 Exercises

1. Rewrite the Chapter 8 privatised histogram using `cub::BlockHistogram`
   inside a custom kernel. What does CUB provide that §8.8 had to implement?
2. Why does cuBLAS need `lda`, `ldb`, and `ldc`? What would break if it
   assumed full density?
3. Explain the column-major trap with a concrete 2x2 example: what does
   `cublasSgemm` return if row-major A and B are fed with `CUBLAS_OP_N`?
4. Under what measurable condition would you replace a `thrust::sort` with a
   custom radix sort? List the two measurements you would take first.

## Sources and Further Reading

- NVIDIA, *CUDA C++ Programming Guide*, "Thrust" and library sections: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *Thrust Quick Start Guide*: <https://docs.nvidia.com/cuda/thrust/>
- NVIDIA, *CUB documentation*: <https://nvidia.github.io/cccl/>
- NVIDIA, *cuBLAS documentation*: <https://docs.nvidia.com/cuda/cublas/>
- NVIDIA, *CUDA C++ Best Practices Guide*, "Use Optimized Libraries": <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
