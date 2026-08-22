# Chapter 10: Modern C++ for CUDA

> *"CUDA is C++ with a different address space. The rules of good C++ still
> apply - more so, because the stakes are higher."*
> 📦 **Code companion:** the complete, buildable code for this chapter lives in [`code/ch10_device_buffer/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch10_device_buffer) in the repository.

Chapters 3-9 wrote CUDA in a C-style dialect: raw `cudaMalloc` pointers,
manual `cudaFree` calls, unchecked errors in the interest of brevity. This
chapter is the reckoning. We apply the full strength of modern C++ (C++17/20)
to GPU programming: **RAII** to make leaks impossible, **templates** to make
kernels generic, **`constexpr`** to make configuration compile-time, and
**exceptions** to make errors impossible to ignore. The result is the style
that the rest of the book (and the capstone) uses.

## 10.1 The Problem with Raw CUDA C

The Chapter 3 vector-add had three structural weaknesses, all of which are
features of the C API:

1. **Leaks.** `cudaMalloc` must be matched with `cudaFree`. An early `return`
   or an exception between the two leaks device memory - and device memory is
   *scarce* (8-80 GB) and *per-process*: a leaked allocation is gone until the
   process exits.
2. **Unchecked errors.** Every call that can fail was routed through `CHECK`,
   but nothing *enforced* that discipline.
3. **No type safety.** `float*` and `int*` are both `void*` to the API; a
   wrong cast compiles and corrupts.

The C++ answer to all three is RAII, templates, and exceptions - the tools
below.

## 10.2 RAII: The Device Buffer

> **Primitive - RAII (Resource Acquisition Is Initialisation).** A resource
> (here: a device allocation) is acquired in a constructor and released in the
> corresponding destructor. The language guarantees the destructor runs when
> the object dies - whether by scope exit, `return`, or exception - so the
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

    // Host <-> device transfer helpers (keep them explicit and checked).
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

**Why `static_assert`?** Device memory is raw storage; copying an object with
internal pointers or virtual tables through it would silently corrupt the
object. Requiring *trivially copyable* types makes the misuse a compile error
instead of a runtime mystery. This is the modern-C++ habit in miniature:
*express the invariant in the type system*.

**Why is the destructor `noexcept`?** Destructors must not throw (throwing in
a destructor during stack unwinding terminates the program). `cudaFree` is
best-effort in a destructor; the error is logged by the runtime, and
`reset()` swallows it. If you *need* to know about free failures, provide an
explicit `release()` that throws.

**Why move and not copy?** Copying a `DeviceBuffer` would mean copying the
*pointer* - two objects owning the same allocation, both freeing it →
double-free. Deleting the copy operations and keeping only moves gives the
ownership semantics of `std::unique_ptr`, which is exactly right.

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

The whole Chapter 3 program shrinks, and its failure modes disappear. The
kernel is untouched: `DeviceBuffer::data()` returns the raw device pointer the
kernel expects, so the RAII layer costs nothing at launch time.

## 10.3 Templates: One Kernel, Many Types

CUDA supports C++ templates in device code. A kernel can be generic over its
element type and its operation - the compiler instantiates exactly the
specialisations you use:

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

**Why does a lambda work as a kernel argument?** The CUDA compiler lowers a
*captureless* lambda to an empty struct with an `operator()` - a function
object with no state. Passing it as a kernel argument is free (zero bytes),
and the compiler inlines the call. A *capturing* lambda has state (the
captured values) that must be copied to the device as kernel arguments - legal
in modern CUDA (values, not references), but the state travels through the
launch, so keep it small and trivially copyable.

**The cost of templates.** None at runtime - the instantiations are compiled,
not interpreted. The cost is compile time and binary size: each `(T, F)` pair
is a separate kernel. This is the standard trade: type safety and reuse for
compile time.

## 10.4 `__host__ __device__` Functions: One Definition, Two Worlds

A function qualified with both `__host__` and `__device__` is compiled twice
 - once for each side - from a single source. This is how shared *algorithm*
code is written:

```cpp
// One definition, two compilations. On the host it is ordinary C++;
// on the device it becomes SASS. This function can be called from kernels
// AND from host code, and the two sides produce identical results (for
// identical inputs) - a boon for testing (Chapter 16).
__host__ __device__ inline float clampf(float v, float lo, float hi)
{
    return fminf(fmaxf(v, lo), hi);   // fminf/fmaxf exist on both sides
}
```

**The catch.** A `__device__` compilation cannot call host functions, so a
`__host__ __device__` function may only use *both-side* facilities: the CUDA
math library (`fminf`, `sqrtf`, `sinf`, ...), `constexpr` arithmetic, and
plain C++. No `std::vector`, no `new`, no I/O - unless you guard the calls
with `#ifdef __CUDA_ARCH__`, which is defined only during device compilation:

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

This dual-compilation trick is the backbone of **CUDA C++ "single-source"**
style and of testable kernels: the same function is exercised on the CPU and
the GPU, and any discrepancy is a device-side bug (Chapter 16's differential
testing).

## 10.5 `constexpr` and `static_assert`: Configuration at Compile Time

Kernel configuration (tile sizes, unroll factors) should be compile-time
constants. `constexpr` makes that the default:

```cpp
// Compile-time kernel configuration. These are real values, not macros:
// they have types, they participate in overload resolution, and they can
// be used in static_assert.
constexpr int kBlockSize = 256;
constexpr int kUnroll    = 4;
constexpr int kMaxDim    = 1 << 16;

// Compile-time sanity checks: the configuration is validated when the file
// is compiled, not when the kernel runs.
static_assert(kBlockSize % 32 == 0,          "block size must be a warp multiple");
static_assert(kUnroll >= 1 && kUnroll <= 8,  "unroll factor out of range");
static_assert(kMaxDim <= (1 << 20),          "dimension bound too large");
```

**Why `constexpr` over `#define`?** Macros are textual and have no type; a
typo becomes a confusing error at a distant use site. `constexpr` variables
are typed, scoped, and checkable. Every magic number this book's standards
ban (§CODING_STANDARDS) becomes a `constexpr` constant.

## 10.6 CUDA 12 and the `cuda::` Namespace

Modern CUDA (12.x) continues to modernise the API surface: the
`cuda::` C++ namespace (header `<cuda/...>`) provides safer alternatives
(`cuda::stream_ref`, `cuda::event`, `cuda::memcpy_async`, `cuda::barrier`,
`cuda::atomic`) that integrate with the standard library's naming and
semantics. Two worth knowing now:

```cpp
#include <cuda/atomic>
// A CUDA-aware atomic that composes with std::atomic's memory-order model:
__device__ cuda::atomic<int, cuda::thread_scope_device> g_counter{0};
```

`cuda::atomic<T, cuda::thread_scope_device>` is the C++20 `std::atomic`
interface for device memory, with scoped ordering - the *modern* replacement
for the raw `atomicAdd` of Chapter 5 when you need acquire/release semantics
rather than relaxed increments.

The ecosystem is moving toward a *standard-C++-flavoured* CUDA: RAII,
atomics with memory orders, and structured barriers. The old C API remains
fully supported - the runtime API you learned in Chapters 3-6 *is* the stable
foundation - but new code should prefer the modern idioms where they exist.

## 10.7 C++20 Concepts: Constraining the Templates

Templates are powerful; concepts make their *errors* legible. A constrained
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

**Why constrain?** Without the concept, passing `f` that returns the wrong
type produces a deep error inside the kernel body. With the constraint, the
compiler says at the *call site*: "F is not invocable as required." The cost
is compile time; the benefit is that kernel templates scale to real codebases
without becoming debugging labyrinths.

## 10.8 A Style Summary

| Old C-style habit | Modern replacement | Property gained |
|---|---|---|
| `cudaMalloc`/`cudaFree` by hand | `DeviceBuffer<T>` RAII | No leaks, no double-free |
| `CHECK` macro discipline | Exceptions from a checked helper | Errors cannot be ignored |
| One kernel per type | Template kernels + lambdas | Reuse without copies |
| `#define TILE 16` | `constexpr int kTile = 16` | Typed, scoped, checkable |
| `atomicAdd` everywhere | `cuda::atomic` where ordering matters | Memory-model clarity |
| Untested device math | `__host__ __device__` + differential tests | Same code, both sides |

## Deeper Explanation: Modern C++ Is How You Make the Compiler Enforce the Book's Rules

Chapters 3-9 asked you to *remember* rules: check every CUDA call, match every
free, never copy a device buffer. Chapter 10 converts those rules into
language mechanisms the compiler enforces. `DeviceBuffer`'s deleted copy
constructor makes double-frees a compile error; `static_assert` makes
non-trivially-copyable types a compile error; templates and concepts make
generic kernels type-safe; `constexpr` makes configuration checkable. The
deeper lesson is that every good API is a *contract expressed in types*: if a
program is invalid, it should not compile, not fail mysteriously at runtime.
This is the same philosophy that later chapters apply to Rust (Chapter 13) and
CUDA-Oxide (Chapter 14): move as many obligations as possible from the
programmer's memory into the compiler's checks.

## Common Pitfalls

- Copying a `DeviceBuffer` by accident. The copy is deleted on purpose; use
  `std::move` to transfer ownership.
- Passing a capturing lambda with non-trivially-copyable state to a kernel.
  Captured state travels through the launch; keep it small and trivially
  copyable.
- Calling host-only facilities (`std::vector`, I/O) inside a `__device__`
  function. Use `#ifdef __CUDA_ARCH__` to separate host/device paths.
- Letting exceptions cross the CUDA launch boundary without cleanup. RAII
  handles device memory, but make sure the rest of the host state is
  exception-safe too.

## Check Your Understanding

<details>
<summary>Why must DeviceBuffer delete its copy constructor?</summary>

A copy would duplicate the pointer, producing two objects that both believe
they own the same device allocation. Both destructors would call
`cudaFree`, a double-free. Move semantics transfer the pointer and null the
source, so only one owner remains.
</details>

<details>
<summary>What type would fail static_assert(is_trivially_copyable_v&lt;T&gt;)?</summary>

A type with a user-defined copy constructor, virtual functions, or internal
pointers that need deep copying - e.g., `std::string`. Raw byte-copying it
through device memory would duplicate or corrupt its internal state.
</details>

<details>
<summary>Why is a captureless lambda a legal kernel argument?</summary>

It lowers to an empty struct with an `operator()` - a function object with no
state. Passing it is zero bytes, and the compiler inlines the call on the
device. A capturing lambda is legal too, but its captured state must be
trivially copyable and small enough to travel through the launch.
</details>

## Key Takeaways

- RAII (`DeviceBuffer<T>`) makes device-allocation leaks and double-frees impossible.
- static_assert moves invariants (trivially copyable, block sizes) into the type system.
- Templates and captureless lambdas give zero-cost generic kernels.
- __host__ __device__ compiles one function for both sides - the foundation of differential testing.
- constexpr for configuration, cuda::atomic for modern memory ordering.

## 10.9 Exercises

1. Why must `DeviceBuffer` delete its copy constructor? Trace the
   double-free that a copy would allow.
2. `static_assert(std::is_trivially_copyable_v<T>, ...)` - give a concrete
   type that would fail this assertion, and explain what copying it through
   device memory would corrupt.
3. Write a `__host__ __device__` function `lerp(a, b, t)` and a short
   explanation of what it lets you test on the host that you could not test
   on the device alone.
4. When is a capturing lambda a legal kernel argument, and what is the
   constraint on the captured state?
