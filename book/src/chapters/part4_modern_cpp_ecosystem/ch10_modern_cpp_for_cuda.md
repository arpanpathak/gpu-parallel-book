# Chapter 10: Modern C++ for CUDA

> *"Within C++, there is a much smaller and cleaner language struggling to get out."*
> — Bjarne Stroustrup, *The Design and Evolution of C++* (1994)

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch10_device_buffer/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch10_device_buffer)
> in the repository.

Chapters 3-9 wrote CUDA in a C-style dialect: raw `cudaMalloc` pointers, manual
`cudaFree` calls, and a `CHECK` macro for errors. This chapter applies modern
C++ (C++17/20) to GPU programming: **RAII** to manage allocations, **templates**
to make kernels generic, **`constexpr`** to make configuration compile-time,
and **exceptions** to make errors visible. The result is the style used by the
rest of the book and the capstone.

## 10.1 Problems with Raw CUDA C

The Chapter 3 vector-add had three structural weaknesses, all inherited from the
C API:

1. **Resource leaks.** Every `cudaMalloc` must be matched with `cudaFree`. An
   early `return` or an exception between the two leaks device memory. Device
   memory is scarce and process-scoped; a leaked allocation is unavailable
   until the process exits.
2. **Unchecked errors.** The `CHECK` macro routes calls through an error check,
   but nothing enforces the discipline. A call added without `CHECK` is
   unchecked.
3. **Weak type safety.** The API receives `void**`, so `float*` and `int*`
   buffers are not distinguished. A wrong cast can compile and corrupt memory.

The C++ remedies are RAII, templates, and exceptions.

## 10.2 RAII: The Device Buffer

> **Primitive - RAII (Resource Acquisition Is Initialisation).** A resource
> (here, a device allocation) is acquired in a constructor and released in the
> corresponding destructor. The language guarantees the destructor runs when
> the object dies, whether by scope exit, `return`, or exception, so the
> resource cannot leak.

```cpp
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

// Throw a std::runtime_error describing a failed CUDA call.
[[noreturn]] inline void throwCudaError(cudaError_t err, const char* what)
{
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
}

// RAII wrapper for a device allocation of T.
template <typename T>
class DeviceBuffer
{
public:
    // --- Construction: allocate on the device -----------------------------
    // static_assert: only trivially-copyable types may live in device memory
    // without custom copy semantics. This converts a runtime confusion into
    // a compile-time error.
    static_assert(std::is_trivially_copyable_v<T>,
                  "DeviceBuffer<T> requires a trivially copyable T");

    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        const cudaError_t err = cudaMalloc((void**)&ptr_, count_ * sizeof(T));
        if (err != cudaSuccess) throwCudaError(err, "cudaMalloc");
    }

    // --- No copying (a device buffer is a unique resource) ----------------
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // --- Move semantics: transfer ownership, never copy the bytes ---------
    // After a move, the source is empty (nullptr). The destructor must
    // handle nullptr gracefully - hence the check in ~DeviceBuffer.
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_   = nullptr;    // source relinquishes the allocation
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();               // release what we held
            ptr_   = other.ptr_;   // take ownership
            count_ = other.count_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // --- Destruction: release the device allocation -----------------------
    ~DeviceBuffer() { reset(); }

    // --- Accessors ---------------------------------------------------------
    T*       data()       noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

    // Host <-> device transfer helpers (explicit and checked).
    void copyToDevice(const T* hostSrc)
    {
        const cudaError_t err = cudaMemcpy(ptr_, hostSrc,
                                           count_ * sizeof(T),
                                           cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throwCudaError(err, "cudaMemcpy H2D");
    }

    void copyToHost(T* hostDst) const
    {
        const cudaError_t err = cudaMemcpy(hostDst, ptr_,
                                           count_ * sizeof(T),
                                           cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throwCudaError(err, "cudaMemcpy D2H");
    }

private:
    void reset() noexcept
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);        // best-effort: destructors must not throw
            ptr_   = nullptr;
            count_ = 0;
        }
    }

    T*         ptr_ = nullptr;   // device pointer
    std::size_t count_ = 0;      // number of elements
};
```

The `static_assert` requires trivially copyable types because device memory is
raw storage. Copying an object with internal pointers or virtual tables through
device memory would corrupt it. The constraint moves a runtime hazard into a
compile-time error.

The destructor is `noexcept` because destructors must not throw; throwing during
stack unwinding would terminate the program. `cudaFree` in a destructor is
best-effort. If a program must know about free failures, it should use an
explicit `release()` method instead.

Copying a `DeviceBuffer` would duplicate the pointer, leaving two objects that
both free the same allocation. Deleting the copy operations and retaining move
semantics gives ownership like that of `std::unique_ptr`.

### 10.2.1 Usage

```cpp
// Allocate 1M floats on the device:
DeviceBuffer<float> d_in(1 << 20), d_out(1 << 20);

// Copy from a host array:
std::vector<float> h_in(1 << 20, 1.0f);
d_in.copyToDevice(h_in.data());

// Launch a kernel (indexing unchanged from Chapter 3):
const int threads = 256;
const int blocks  = (static_cast<int>(d_in.size()) + threads - 1) / threads;
addVectors<<<blocks, threads>>>(d_in.data(), d_out.data(), d_in.size());
CHECK(cudaGetLastError());

// Copy back:
std::vector<float> h_out(d_out.size());
d_out.copyToHost(h_out.data());
// d_in and d_out are freed automatically at scope exit - no cudaFree calls.
```

The Chapter 3 program becomes shorter and its failure modes disappear. The
kernel itself is unchanged because `DeviceBuffer::data()` returns the raw
device pointer the kernel expects.

## 10.3 Templates: One Kernel, Many Types

CUDA supports C++ templates in device code. A kernel can be generic over its
element type and its operation. The compiler instantiates the specialisations
that the program uses:

```cpp
// Generic elementwise transform. F is any callable (function object,
// lambda, function pointer) invocable as F(T) -> T. The compiler
// instantiates a separate device function for each (T, F) pair.
template <typename T, typename F>
__global__ void transformKernel(const T* in, T* out, int n, F f)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(in[i]);
}

// Host side: launch with a lambda. The lambda must be __device__-compatible;
// a captureless lambda is (it has no state to copy to the device).
void runTransform(const DeviceBuffer<float>& d_in, DeviceBuffer<float>& d_out,
                  int n)
{
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    transformKernel<<<blocks, threads>>>(d_in.data(), d_out.data(), n,
                                         [](float x) { return x * 2.0f + 1.0f; });
}
```

A captureless lambda is a legal kernel argument because the CUDA compiler
lowers it to an empty struct with an `operator()`: a function object with no
state. Passing it costs zero bytes, and the compiler inlines the call. A
capturing lambda carries state that must be copied to the device as kernel
arguments. Modern CUDA permits small, trivially copyable captures by value;
references are not copied.

Templates have no runtime cost because the compiler emits the instantiations.
The cost is compile time and binary size: each `(T, F)` pair is a separate
kernel.

## 10.4 `__host__ __device__` Functions

A function qualified with both `__host__` and `__device__` is compiled twice
from a single source, once for each side. This is how shared algorithm code is
written:

```cpp
// One definition, two compilations. On the host it is ordinary C++;
// on the device it becomes SASS. This function can be called from kernels
// and from host code, and the two sides produce identical results for
// identical inputs - a foundation for differential testing (Chapter 16).
__host__ __device__ inline float clampf(float v, float lo, float hi)
{
    return fminf(fmaxf(v, lo), hi);   // fminf/fmaxf exist on both sides
}
```

A `__device__` compilation cannot call host functions. A `__host__ __device__`
function may therefore use only facilities available on both sides: the CUDA
math library (`fminf`, `sqrtf`, `sinf`, ...), `constexpr` arithmetic, and plain
C++. It cannot use `std::vector`, `new`, or I/O unless the calls are guarded by
`#ifdef __CUDA_ARCH__`, which is defined only during device compilation:

```cpp
__host__ __device__ float maybeLog(float x)
{
#ifdef __CUDA_ARCH__
    return logf(x);        // device path: CUDA math library
#else
    return std::log(x);    // host path: standard library
#endif
}
```

This dual compilation is the basis of CUDA's single-source style and of
testable kernels: the same function can be exercised on the CPU and the GPU,
and a discrepancy indicates a device-side bug (Chapter 16).

## 10.5 `constexpr` and `static_assert`

Kernel configuration such as tile sizes and unroll factors should be
compile-time constants:

```cpp
// Compile-time kernel configuration. These are typed values, not macros.
constexpr int kBlockSize = 256;
constexpr int kUnroll    = 4;
constexpr int kMaxDim    = 1 << 16;

// Compile-time sanity checks: configuration is validated when the file is
// compiled, not when the kernel runs.
static_assert(kBlockSize % 32 == 0,          "block size must be a warp multiple");
static_assert(kUnroll >= 1 && kUnroll <= 8,  "unroll factor out of range");
static_assert(kMaxDim <= (1 << 20),          "dimension bound too large");
```

`constexpr` is preferable to `#define` because `constexpr` variables are typed,
scoped, and can be used in `static_assert`. Macros are textual and untyped; a
typo can become a confusing error at a distant use site.

## 10.6 CUDA 12 and the `cuda::` Namespace

CUDA 12.x continues to modernise the API. The `cuda::` C++ namespace, in
headers such as `<cuda/atomic>`, provides safer alternatives
(`cuda::stream_ref`, `cuda::event`, `cuda::memcpy_async`, `cuda::barrier`,
`cuda::atomic`) with standard-library-compatible names and semantics:

```cpp
#include <cuda/atomic>
// A CUDA-aware atomic that composes with std::atomic's memory-order model:
__device__ cuda::atomic<int, cuda::thread_scope_device> g_counter{0};
```

`cuda::atomic<T, cuda::thread_scope_device>` provides the `std::atomic`
interface for device memory with scoped ordering. It is the preferred
replacement for raw `atomicAdd` when acquire/release semantics are needed
rather than relaxed increments.

The old C API remains fully supported and is the stable foundation taught in
Chapters 3-6. New code should prefer the modern idioms where they exist.

## 10.7 C++20 Concepts

Templates are powerful; concepts make their errors legible. A constrained
version of `transformKernel`:

```cpp
#include <concepts>

// The transform operation must be an invocable mapping T to T.
template <typename T, std::invocable<T> F>
    requires std::same_as<std::invoke_result_t<F, T>, T>
__global__ void transformKernel(const T* in, T* out, int n, F f)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(in[i]);
}
```

Without the constraint, a wrong functor produces a deep error inside the kernel
body. With the constraint, the compiler reports at the call site that `F` is
not invocable as required. The cost is compile time; the benefit is that kernel
templates scale to real codebases.

## 10.8 Style Summary

| Old C-style habit | Modern replacement | Property gained |
|---|---|---|
| `cudaMalloc`/`cudaFree` by hand | `DeviceBuffer<T>` RAII | No leaks, no double-free |
| `CHECK` macro discipline | Exceptions from a checked helper | Errors cannot be ignored |
| One kernel per type | Template kernels + lambdas | Reuse without copies |
| `#define TILE 16` | `constexpr int kTile = 16` | Typed, scoped, checkable |
| `atomicAdd` everywhere | `cuda::atomic` where ordering matters | Memory-model clarity |
| Untested device math | `__host__ __device__` + differential tests | Same code, both sides |

## Making the Compiler Enforce the Rules

The early chapters ask the programmer to follow rules by hand: check every CUDA
call, match every allocation with a free, and never copy a device buffer. These
rules are easy to state and easy to violate under refactoring or exception
handling. Modern C++ moves the rules into the type system so that violations
become compile errors instead of runtime bugs.

"Never leak or double-free a device allocation" becomes `DeviceBuffer<T>`: the
constructor allocates, the destructor frees, and deleted copy operations ensure
that two objects cannot own the same pointer. "Only copy trivially copyable
types through device memory" becomes a `static_assert` in the class. "Write
generic kernels without duplicating code" becomes templates and lambdas. "Make
configuration checkable" becomes `constexpr` constants used in `static_assert`.

This is the same philosophy used by safety-critical software: a type system
makes illegal states unrepresentable. If a program cannot be written, it cannot
fail at runtime. The parts that remain unprovable in C++ - index arithmetic,
launch geometry - are exactly the parts later chapters isolate into explicit
`unsafe` blocks (Chapter 13) or type-level constructs such as `DisjointSlice`
(Chapter 14). The trajectory from raw `cudaMalloc` to Rust kernels is the same:
move obligations from "remember to do this" to "the compiler will not let you
do otherwise."

## Common Pitfalls

- Copying a `DeviceBuffer` by accident. Copy operations are deleted on purpose;
  use `std::move` to transfer ownership.
- Passing a capturing lambda with non-trivially-copyable state to a kernel.
  Captured state travels through the launch; keep it small and trivially
  copyable.
- Calling host-only facilities such as `std::vector` or I/O inside a
  `__device__` function. Use `#ifdef __CUDA_ARCH__` to separate paths.
- Letting exceptions cross the CUDA launch boundary without cleanup. RAII
  handles device memory, but other host state must also be exception-safe.

## Check Your Understanding

<details>
<summary>Why must DeviceBuffer delete its copy constructor?</summary>

A copy would duplicate the pointer, producing two objects that both believe
they own the same device allocation. Both destructors would call `cudaFree`,
causing a double-free. Move semantics transfer the pointer and null the source,
so only one owner remains.
</details>

<details>
<summary>What type would fail static_assert(is_trivially_copyable_v&lt;T&gt;)?</summary>

A type with a user-defined copy constructor, virtual functions, or internal
pointers that need deep copying, such as `std::string`. Byte-copying it through
device memory would duplicate or corrupt its internal state.
</details>

<details>
<summary>Why is a captureless lambda a legal kernel argument?</summary>

It lowers to an empty struct with an `operator()`: a function object with no
state. Passing it costs zero bytes and the compiler inlines the call on the
device. A capturing lambda is legal when its state is trivially copyable and
small enough to travel through the launch.
</details>

## Key Takeaways

- RAII (`DeviceBuffer<T>`) makes device-allocation leaks and double-frees impossible.
- static_assert moves invariants (trivially copyable, block sizes) into the type system.
- Templates and captureless lambdas give zero-cost generic kernels.
- __host__ __device__ compiles one function for both sides, the foundation of differential testing.
- Use constexpr for configuration and cuda::atomic for modern memory ordering.

## 10.9 Exercises

1. Why must `DeviceBuffer` delete its copy constructor? Trace the double-free
   that a copy would allow.
2. Give a concrete type that would fail
   `static_assert(std::is_trivially_copyable_v<T>, ...)` and explain what
   copying it through device memory would corrupt.
3. Write a `__host__ __device__` function `lerp(a, b, t)` and explain what it
   lets you test on the host that you could not test on the device alone.
4. When is a capturing lambda a legal kernel argument, and what constraint
   applies to the captured state?

## Sources and Further Reading

- NVIDIA, *CUDA C++ Programming Guide*, "Programming Model" and "Memory Hierarchy": <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *CUDA C++ Best Practices Guide*, for resource management and performance guidance: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- Bjarne Stroustrup, *The C++ Programming Language* and *The Design and Evolution of C++*, for RAII, value semantics, move semantics, and templates.
