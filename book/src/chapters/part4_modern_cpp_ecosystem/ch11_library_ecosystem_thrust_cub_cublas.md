# Chapter 11: The Library Ecosystem - Thrust, CUB & cuBLAS

> *"The best kernel you will ever write is the one NVIDIA already shipped."*
> 📦 **Code companion:** the complete, buildable code for this chapter lives in [`code/ch11_library_examples/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch11_library_examples) in the repository.

Chapters 3-10 taught you to write kernels. This chapter teaches you when *not*
to. The CUDA ecosystem ships three libraries that implement, in battle-tested
and hardware-tuned form, most of the algorithms of Chapters 8 and 9: **Thrust**
(high-level algorithms), **CUB** (block-level primitives), and **cuBLAS**
(dense linear algebra). A professional GPU engineer uses them first and writes
custom kernels only where the libraries cannot express the problem - and
*this* book's earlier chapters are exactly what you need to understand what
the libraries are doing underneath.

## 11.1 The Value Proposition

NVIDIA's libraries are tuned by engineers with direct access to the hardware
designers, for every generation of GPU:

- They dispatch to **specialised kernels per architecture** (tensor cores for
  GEMM, custom shuffle reductions for scans);
- They are **hand-optimised** beyond what the compiler alone achieves;
- They are **tested** against known-good references and profiled on every
  release.

Your hand-written SGEMM from Chapter 9 (70% of peak) is genuinely good; cuBLAS
ships ~95% of peak with tensor cores. The engineering decision is not
"libraries or custom kernels" - it is *"does my problem fit a library?"*.

## 11.2 Thrust: STL for the GPU

Thrust is the closest thing CUDA has to the C++ standard library. It provides
containers (`thrust::device_vector`) and algorithms (`transform`, `reduce`,
`sort`, `exclusive_scan`, ...) that operate on device memory with a syntax
mirroring `std::`:

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// Thrust version of the Chapter 8 reduction + Chapter 7 SAXPY. The
// algorithms are dispatched to tuned kernels internally; the host code
// describes WHAT, not HOW.
// ---------------------------------------------------------------------------
void thrustExample(int n)
{
    // device_vector: a RAII device array (like Chapter 10's DeviceBuffer).
    thrust::device_vector<float> x(n), y(n);

    // thrust::sequence fills x with 0..n-1 (parallel, device-side).
    thrust::sequence(x.begin(), x.end());

    // thrust::transform applies a functor elementwise. thrust::device is the
    // execution policy that says "run on the GPU"; it must be the FIRST
    // argument (policy, first, last, result, op).
    thrust::transform(thrust::device,
                      x.begin(), x.end(), y.begin(),
                      [] __device__ (float v) { return v * 2.0f + 1.0f; });

    // thrust::reduce folds the array (the Chapter 8 reduction, tuned).
    const float total = thrust::reduce(y.begin(), y.end(), 0.0f,
                                       thrust::plus<float>());

    // thrust::sort, thrust::exclusive_scan, etc. follow the same shape.
    thrust::sort(y.begin(), y.end());
}
```

**Why the `__device__` on the lambda?** Thrust must compile the functor for
the device. A plain captureless lambda *can* work via automatic
`__host__ __device__` inference in modern Thrust, but the explicit
`__device__` makes the intent unambiguous and is the documented style.

**The cost of convenience.** `thrust::device_vector` and the algorithm
dispatches carry their own allocations and launch logic. For a one-off
reduction of a large array, that overhead is noise; for a per-frame
micro-pipeline in a tight loop, it is not. Measure (Chapter 16) before you
assume.

## 11.3 CUB: Block-Level Primitives

Thrust works at the *container* level. **CUB** works at the *block* level: it
gives you the building blocks - `cub::BlockReduce`, `cub::BlockScan`,
`cub::BlockHistogram`, `cub::WarpReduce` - that your own kernels can embed.
This is the library to reach for when you need a *block-sized* reduction
inside a kernel that also does something custom:

```cpp
#include <cub/cub.cuh>

// A kernel that reduces its block's partial sums using CUB, then folds
// block results with a warp shuffle. CUB's BlockReduce is the tuned version
// of Chapter 8's reduceFull.
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

**Why use CUB instead of writing the reduction again?** Because CUB's
`BlockReduce` handles the *edge cases* you would otherwise debug for hours:
non-power-of-two block sizes, the choice between shuffle-only and
shuffle+shared strategies, and per-architecture tuning. Your Chapter 8 kernel
is the *explanation*; CUB is the *implementation* you ship.

**The cost of CUB.** It is header-only and template-heavy: compile times grow,
and error messages can be intimidating. The runtime cost is zero - the
templates inline to the same SASS you would write.

## 11.4 cuBLAS: Dense Linear Algebra

cuBLAS is the BLAS (Basic Linear Algebra Subprograms) for GPUs: `sgemm`
(single-precision GEMM), `saxpy`, `sdot`, `sgemv`, and the batched variants
used by deep learning. Its API is the classic *handle-based* C library style:

```cpp
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// C = alpha * A * B + beta * C   (the GEMM of Chapter 9, via cuBLAS)
// ---------------------------------------------------------------------------
void gemmViaCublas(const float* dA, const float* dB, float* dC,
                   int m, int n, int k, float alpha, float beta)
{
    // cuBLAS functions are NOT thread-safe by default; each context gets a
    // handle. Create once per context, reuse for all calls.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS, like Fortran BLAS, is COLUMN-major: matrix columns are
    // contiguous. Row-major C (m x n) = A (m x k) * B (k x n) has the same
    // flat memory layout as column-major C^T = B^T * A^T, so we swap the
    // operands and the problem sizes and compute the transposed product:
    //   cublasSgemm(..., n, m, k, ..., dB, ldb, dA, lda, ..., dC, ldc)
    // Each leading dimension is the matrix's ROW length (the number of
    // elements per row in row-major storage):
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

**Why a *handle*?** The handle carries per-context state (stream association,
workspace, heuristics). It lets the library keep state without global
variables, which would break multi-context and multi-thread programs. The
handle's stream can be set with `cublasSetStream(handle, stream)` so cuBLAS
calls participate in your pipeline (Chapter 6).

**The column-major trap.** cuBLAS, like Fortran BLAS, is column-major: matrix
columns are contiguous. Row-major data must either be transposed (with
`CUBLAS_OP_T`) or have its dimensions swapped. The classic bug is feeding
row-major data with `CUBLAS_OP_N` and getting the transposed answer. When in
doubt, verify with a 2×2 case before scaling up.

**Why use cuBLAS for GEMM at all?** Performance: for large matrices it uses
tensor cores and achieves 90%+ of peak, versus ~70% for the hand-written
kernel of Chapter 9 - and it took zero optimisation effort. The Chapter 9
kernel's value is *understanding*; the cuBLAS call's value is *shipping*.

## 11.5 cuFFT and cuRAND: The Specialists

Two more libraries complete the common toolkit:

- **cuFFT** - Fast Fourier Transforms (1-D, 2-D, 3-D, batched, complex and
  real). Hand-written FFTs on GPUs are a research project; cuFFT is the
  product. Its API mirrors FFTW's planner model: create a *plan* describing
  the transform, execute it many times on different data.
- **cuRAND** - random number generation on the device, with many generators
  (XORWOW, MRG32k3a, Philox, ...) and distributions (uniform, normal,
  Poisson). The distinguishing feature: it can generate *device-side*,
  so kernels can draw random numbers internally (important for Monte Carlo).

Both follow the handle/plan pattern: create once, configure, execute
repeatedly. Both are worth knowing by name and purpose, even if this book does
not devote chapters to them.

## 11.6 The Decision Procedure: Library or Custom Kernel?

When faced with a GPU problem, walk this list:

1. **Is it an algorithm in Thrust/CUB/cuBLAS/cuFFT/cuRAND?** → Use the
   library. The tuned, tested version beats your first kernel, and the time
   saved goes into profiling the parts that matter.
2. **Is the library call the bottleneck?** → Profile it (Chapter 16). If it
   is, the question becomes whether your *data layout* fits the library's
   assumptions (leading dimensions, transposes) - usually fixable without a
   custom kernel.
3. **Does the problem have custom per-element logic?** → Write a custom
   *kernel*, but use CUB's block primitives inside it. Custom is not
   synonymous with from-scratch.
4. **Is the custom kernel the hot path, measured?** → Only now hand-roll the
   full optimisation (Chapter 9's journey).

The failure mode this procedure prevents is the reverse: rewriting
`thrust::sort` because "it might be faster", while the actual bottleneck sits
in a badly-coalesced custom kernel next door. Measure first; the library is
the default.

**A worked decision: "I need to normalise a 100M-float array."** Walk the
list: (1) normalisation is elementwise - `thrust::transform` with a functor
is the library answer; (2) no bottleneck to profile yet, because we have not
written anything; (3) no custom logic - the functor *is* the per-element
logic; (4) no custom kernel needed. The correct engineering move is one
`thrust::transform` call, measured at ~95% of bandwidth. Rewriting it as a
hand-tuned kernel would take an hour and gain nothing, because the roofline
(Chapter 1) already says the transform is memory-bound and Thrust's
dispatcher already coalesces.

**A second worked decision: "I need a 7-tap separable blur per frame."**
(1) No library call matches a stencil with a halo; (2) nothing to profile
yet; (3) the halo logic is custom, but the surrounding machinery is generic -
so write a custom kernel and keep it simple: coalesced row reads, per-tap
clamped indices
(Chapter 15's `blurH` is exactly this shape). (4) Only if the profiler shows
this kernel as the pipeline's bottleneck do you reach for shared-memory
tiling (Chapter 7). This is the capstone's actual path in Chapter 15.

The pattern in both cases is the same: **the decision is driven by the
algorithm's shape and the profiler's numbers, never by taste.**

## Deeper Explanation: Libraries Are Optimised Answers to Problems You Should Still Understand

There is a common misunderstanding that using Thrust/CUB/cuBLAS means you do
not need to understand reductions, scans, or GEMM. The opposite is true: the
libraries are *tuned implementations of the exact algorithms in Chapters 8-9*.
If you do not understand the algorithm, you cannot predict when the library
will be fast, when it will silently choose the wrong path, or when your data
layout conflicts with its assumptions (cuBLAS's column-major trap is the
canonical example). Libraries remove the *implementation* burden, not the
*conceptual* one. The professional workflow is: know the algorithm, reach for
the library, profile it, and only hand-roll when the profiler proves the
library is the bottleneck.

## Common Pitfalls

- Passing row-major data to cuBLAS with `CUBLAS_OP_N` and getting the
  transposed answer. Either transpose operands or use the dimension-swap trick
  from §11.4.
- Using `thrust::device_vector` in a per-frame hot loop. Its convenience
  carries allocation and dispatch overhead; measure before assuming it is
  free.
- Reaching for CUB without understanding its template syntax. The errors are
  intimidating, but the pattern (temp storage + `Sum()`) is small once you
  have seen it.
- Rewriting a library call "because it might be faster". The decision
  procedure requires a profiler, not a hunch.

## Check Your Understanding

<details>
<summary>Why does cuBLAS need leading dimensions?</summary>

Because matrices can be sub-matrices (tiles) of larger buffers. `lda` tells
cuBLAS how many elements separate the start of one row/column from the next,
so it can walk a sub-matrix correctly instead of assuming full density.
</details>

<details>
<summary>What does CUB's BlockReduce give you that Chapter 8's reduceFull did not?</summary>

CUB handles non-power-of-two block sizes, architecture-specific tuning, and
edge cases the hand-written kernel would need to debug. The Chapter 8 kernel
is the explanation; CUB is the tested implementation.
</details>

<details>
<summary>When should you replace thrust::sort with a custom radix sort?</summary>

Only when profiling shows `thrust::sort` is a significant fraction of runtime
and your data/keys have properties a radix sort can exploit (fixed-size keys,
known range). Measure sort time and end-to-end time before and after.
</details>

## Key Takeaways

- Use the tuned, tested libraries first: Thrust (algorithms), CUB (block primitives), cuBLAS (dense linear algebra).
- Thrust mirrors the STL: device_vector, transform, reduce, sort, scans.
- CUB slots into your own kernels: cub::BlockReduce, cub::BlockScan, cub::BlockHistogram.
- cuBLAS is handle-based and column-major - the transpose trap is the classic bug.
- Rewrite a library call only when the profiler proves it is the bottleneck.

## 11.7 Exercises

1. Rewrite the Chapter 8 privatised histogram using `cub::BlockHistogram`
   inside a custom kernel. What does CUB provide that §8.8 had to implement?
2. Why does cuBLAS need the leading-dimension arguments `lda`, `ldb`, `ldc`
   at all? What would break if it assumed full density?
3. Explain the column-major trap with a concrete 2×2 example: what does
   `cublasSgemm` return if you feed row-major A and B with `CUBLAS_OP_N`?
4. Under what measurable condition would you replace a `thrust::sort` with a
   custom radix sort? List the two measurements you would take first.
