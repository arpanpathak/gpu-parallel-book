<img class="plate" src="../../art/ch10.svg" alt="Hoot, an owl, sits on a tree stump at dusk and watches a rope of lanterns. The first 4 lanterns glow; a brace beneath them reads in scope. The last 2 hang dark; a brace beneath them reads scope ended. Nobody walked along the rope to put them out: each lantern switched itself off when its scope ended, as a C++ destructor releases what a constructor took.">

# Chapter 10: Modern C++ for CUDA

Chapters 3 to 9 wrote their **host** code, the part of a CUDA program that runs on the CPU, in
the style of the CUDA C API: a raw pointer from `cudaMalloc`, a matching `cudaFree` at the end,
and an error code returned by every call. That style keeps every step visible, which suits a
first program. A **kernel**, the function that runs on the GPU, is launched from that host code,
and the host code also allocates, copies and frees the kernel's memory. The C style leaves 3
duties to the programmer: free every allocation on every path out of a function, look at every
error code, and pass every size in the unit the function expects. The compiler checks none of
them.

In this chapter we hand those duties to the language. We first run 3 small programs in which
each duty is forgotten, and measure the damage: 128 MiB of memory lost, a kernel that never
ran, and 3 in 4 results wrong, all without a single error message. Then we rebuild the same
programs with the tools of modern C++, the C++ standardised in 2011 and after: a class that
frees its memory by itself, templates, functions compiled for both processors, compile-time
checks, the `cuda::` library and C++20 concepts. Each tool is measured, so we can see that it
costs nothing at run time. Each one has a counterpart in Rust, shown beside it, where the same
rules are part of the language itself.

<div class="covers" markdown="1">

This chapter covers

- 3 mistakes that the CUDA C API lets through, run and measured
- RAII: a constructor that allocates and a destructor that frees, on every path out of a scope
- 1 owner per allocation: why a copy is forbidden and a move is allowed
- errors as exceptions, with the file and line of the failed call and no macro
- template kernels, and the kernels the compiler writes from them
- `__host__ __device__` functions, and why the same expression can round differently on each side
- `constexpr` and `static_assert`: checks that run while the program compiles
- `cuda::std::span`, `cuda::atomic_ref` and thread scopes
- C++20 concepts, and a mistake that compiles without them
- the same rules in Rust: `Drop`, moves, `?`, generic kernels and trait bounds

</div>

The code of the chapter is in
[`code/ch10_modern/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch10_modern)
for C++ and in
[`code/oxide/ch10_modern/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch10_modern)
for Rust. The outputs in this chapter come from 2 integrated GPUs: a Jetson Orin NX (8 SMs,
102 GB/s, CUDA 12.6), whose runs the listings show, and a Jetson AGX Thor (20 SMs, 273 GB/s,
CUDA 13.2), which gives the same counts.

## 10.1 Problems with raw CUDA C

A **resource** is something a program acquires from the system and must give back: device
memory, a stream, an event, an open file. The CUDA C API hands out resources and takes them
back through pairs of calls, such as `cudaMalloc` and `cudaFree`, and it reports every failure
as a return value of type `cudaError_t`. Both conventions are sound. Both depend on the caller
remembering something at every call site, and this section measures what happens when the
caller forgets.

The program `raw.cu` makes 3 such mistakes, each in a few lines that look reasonable.

### 10.1.1 Counting the memory a program holds

To measure a leak we need an exact count of the device memory the program holds. The programs of
this chapter allocate with `cudaMallocAsync(&pointer, bytes, stream)`, the **stream-ordered
allocator** of CUDA 11.2 and later. It behaves like `cudaMalloc`, except that the allocation and
the matching `cudaFreeAsync` take their place in a **stream**, the queue of GPU work that runs
in order (Chapter 6), like a kernel, and the memory comes from a **pool**: a reserve of device
memory that the driver keeps for each GPU and reuses between allocations. The pool counts the
bytes handed out and not yet given back, in the attribute `cudaMemPoolAttrUsedMemCurrent`.

<p class="listing"><b>Listing 10.1</b> The bytes in use from the pool, in MiB. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch10_modern/raw.cu">code/ch10_modern/raw.cu</a></p>

```cpp
{{#include ../../../../code/ch10_modern/raw.cu:19:28}}
```

`cudaMemGetInfo`, the older way to ask, reports the free memory of the whole device, and other
programs change it. On an integrated GPU, which shares its memory with the operating system, an
allocation of 128 MiB moved the free memory that `cudaMemGetInfo` reported by only 0.5 MiB on
the first GPU and by 129 MiB on the second. The pool's count does not depend on anything
outside the program. 1 MiB is 2<sup>20</sup> bytes, 1,048,576 bytes, which is the number the
listing divides by.

### 10.1.2 A leak on an early return

A **leak** is a resource that a program acquired and can no longer give back, because no part of
the program still holds its address. The first function of `raw.cu` allocates 2 buffers, then
decides that it has nothing to do and returns before it reaches the 2 frees.

<p class="listing"><b>Listing 10.2</b> An early return before the frees.</p>

```cpp
{{#include ../../../../code/ch10_modern/raw.cu:30:52}}
```

The `return false` is correct by itself: a factor of 0 needs no GPU. The mistake is what it
skips. The pointers `in` and `out` are local variables, so they disappear with the return, and
with them the only record of where the 2 buffers are.

<div class="example" markdown="1">

Worked example: 8 calls with a factor of 0

1. Each buffer holds N = 2,097,152 floats of 4 bytes: 8,388,608 bytes, 8 MiB.
2. Each call allocates 2 buffers and frees neither: 16 MiB per call.
3. 8 calls keep 8 × 16 MiB = 128 MiB until the program ends.

</div>

<figure>
<img src="../../figures/ch10-leak.svg" alt="Two columns of steps. Raw pointers: 2 calls of cudaMallocAsync of 8 MiB, then if factor is 0 return false; the 2 calls of cudaFreeAsync below are dashed, skipped on every early return. After 8 calls the pool holds 8 segments of 16 MiB, 128 MiB. DeviceBuffer: the same 2 allocations and the same return, followed by the destructors of out and in, which free both buffers; after 8 calls the pool is empty.">
<figcaption><b>Figure 10.1</b> The early return skips the 2 frees. A destructor, section 10.2, runs on the return itself.</figcaption>
</figure>

```console
$ nvcc -arch=native raw.cu -o raw && ./raw
1. 8 calls that return early: 128.0 MiB lost
```

The count matches the hand calculation to the byte. Nothing reported the leak, and nothing
would: the memory belongs to the program until it exits. A long-running program, a server or a
training job that calls such a function millions of times, runs out of device memory hours
later, in code that has nothing to do with the leak.

### 10.1.3 An error that nobody asked about

A **block** is a group of threads launched together, all on 1 SM and able to share memory and
wait for each other. A block holds at most 1,024 threads on every GPU since compute capability
2.0, the architecture version of 2010. The second mistake launches blocks of 2,048.

<p class="listing"><b>Listing 10.3</b> A launch over the limit, and the error that only a question reveals.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch10_modern/raw.cu:74:80}}
    // ...
}
```

```console
2. 2048 threads per block: out[0] = 0.0, expected 2.0
   the error, found only by asking: invalid configuration argument
```

The launch syntax `<<<...>>>` returns nothing, so a launch that fails cannot hand back an error
code. CUDA records the failure instead, and the next call to `cudaGetLastError` returns it. The
copy that follows succeeds; it copies an array that no kernel wrote, and `out[0]` is still the
0.0 of the `cudaMemset`. If the program never calls `cudaGetLastError`, it continues with wrong
data and no message. On the second GPU, CUDA 13.2 reports the same launch as `invalid argument`; the
wording varies between versions, the silence does not.

### 10.1.4 Bytes where elements were meant

`cudaMemcpy(dst, src, count, kind)` copies `count` **bytes**. Its pointers have type `void*`, a
pointer to memory of no particular type, so the function cannot know that they point at floats,
and cannot check a count of floats against them. The third mistake passes `N`, the number of
floats, where the number of bytes, `N * sizeof(float)`, was meant.

<p class="listing"><b>Listing 10.4</b> A count of floats passed as a count of bytes.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch10_modern/raw.cu:82:92}}
    // ...
}
```

<div class="example" markdown="1">

Worked example: N = 16 floats

1. The host array holds 16 floats of 4 bytes: 64 bytes.
2. `cudaMemcpy(in, host, 16, ...)` copies 16 bytes: 16 ÷ 4 = 4 floats.
3. The other 12 floats of `in` keep the 0.0 written by `cudaMemset`.
4. The kernel doubles every element: 4 results are 2.0, 12 are 0.0. 4 of 16 are right, 1 in 4.

</div>

<figure>
<img src="../../figures/ch10-bytes.svg" alt="A row of 16 host floats, all 1.0. A bracket over the first 4 marks 16 bytes: cudaMemcpy with a count of 16 copies only those. On the device, the first 4 floats of in are 1.0 and the other 12 are 0.0; after the kernel doubles them, out holds 4 values of 2.0 and 12 of 0.0. 4 of 16 are correct, because 16 bytes are 4 floats. In the program, 524,288 of 2,097,152.">
<figcaption><b>Figure 10.2</b> A count of 16 copies 16 bytes, which are 4 floats.</figcaption>
</figure>

```console
3. a copy of N bytes, not N floats: 524288 of 2097152 correct
```

With N = 2,097,152 the copy moved 2,097,152 bytes, 524,288 floats: exactly 1 in 4, as for 16.
Every call returned `cudaSuccess`. The API did what its arguments asked for.

<div class="bee bee-shocked" markdown="1">

The C API frees what it is told to free and reports what it is asked about. In 3 short programs
it lost 128 MiB, skipped a kernel and copied 1/4 of an array, and printed no error.

</div>

## 10.2 RAII: The Device Buffer

The 3 mistakes share a cause: a rule that holds at every call site is written out again, by
hand, at every call site. C++ lets us write such a rule once, inside a type, and have the
compiler apply it wherever the type is used. We start with the first rule, "every allocation is
freed".

### 10.2.1 Constructors, destructors and scope

A few definitions come first.

- A **scope** is a region of a program, delimited by braces `{ }`, in which a name is visible.
  The body of a function is a scope; so is the body of a loop.
- An **object** of a class type is created by a **constructor**, a member function of the class
  that runs when the object is defined.
- The object is destroyed by its **destructor**, `~ClassName()`, a member function that the
  compiler calls when the object's lifetime ends. For a local variable, the lifetime ends when
  control leaves its scope, by any route: the closing brace, a `return`, a `break`, or an
  exception passing through.

**RAII**, short for *resource acquisition is initialisation*, is the practice of acquiring a
resource in a constructor and releasing it in the destructor of the same object. The resource
then lives exactly as long as the object, and the release happens on every route out of the
scope, because the compiler, not the programmer, places the call to the destructor.

```text
{                                   # a scope opens
    buffer ← DeviceBuffer(n)        # constructor: allocate n elements
    ...                             # use buffer
    if done early: return           # compiler inserts: destroy buffer, then return
    ...
}                                   # compiler inserts: destroy buffer
                                    # destroy = destructor: free the allocation
```

<figure>
<img src="../../figures/ch10-lifetime.svg" alt="A column of steps inside a zone labelled 1 scope. DeviceBuffer<float> in(host) calls the constructor, which allocates with cudaMallocAsync and copies; kernels then read in.view(). 3 ways out branch from there: return false, check() throws, and the closing brace. All 3 join into 1 box, the destructor, which calls cudaFreeAsync and gives the 8 MiB back to the pool.">
<figcaption><b>Figure 10.3</b> 3 ways out of a scope lead to 1 destructor.</figcaption>
</figure>

The C++ standard library already works this way. A `std::vector` frees its array in its
destructor, and a `std::unique_ptr` deletes the object it points to. Device memory needs a
class of its own only because `new` and `delete` manage host memory; device memory comes from
CUDA calls.

### 10.2.2 The class

`DeviceBuffer<T>` holds `count` values of type `T` in device memory. Its first constructor
allocates; its second allocates and copies a host vector in; its destructor frees.

<p class="listing"><b>Listing 10.5</b> <code>DeviceBuffer</code>: construction and destruction. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch10_modern/device_buffer.hpp">code/ch10_modern/device_buffer.hpp</a></p>

```cpp
{{#include ../../../../code/ch10_modern/device_buffer.hpp:38:57}}
    // ...
};
```

3 details carry the design.

1. **The size is computed from the type.** `bytes()` returns `count_ * sizeof(T)`. No caller
   ever writes a byte count, so the mistake of section 10.1.4 has no place to happen.
2. **The second constructor delegates to the first.** `: DeviceBuffer(host.size())` allocates
   before the body copies. If the copy throws (section 10.2.4), the object is already
   constructed, so its destructor runs and the allocation is freed.
3. **The `static_assert` rejects types that cannot live in device memory.** A type is
   **trivially copyable** when copying its bytes produces a valid copy of the object: `float`,
   `int`, a struct of floats. A `std::string` is not; it holds a pointer to host memory that a
   byte copy would duplicate, and the GPU could not follow. `DeviceBuffer<std::string>` is a
   compile error. Section 10.5 explains `static_assert`.

The `explicit` keyword stops the compiler from turning a number into a buffer by itself: without
it, a function taking a `DeviceBuffer<float>` would accept the argument `5` and allocate 5
floats.

### 10.2.3 1 owner per allocation

A class that frees in its destructor must decide what a copy of an object means. Suppose
`b = a` copied the pointer, as a copy of a struct of raw pointers does. `a` and `b` would both
hold address 0x7f00, and both destructors would free it. Freeing the same allocation twice, a
**double free**, corrupts the allocator's records; a later allocation can then return memory
that another buffer still uses.

The class therefore has exactly 1 **owner** per allocation: the 1 object whose destructor frees
it. A copy is forbidden; the 2 lines `= delete` remove the copy constructor and the copy
assignment, so `b = a` does not compile. A **move** is allowed: `b = std::move(a)` transfers
the pointer to `b` and leaves `a` holding `nullptr`. The destructor of `a` then has nothing to
free. A move copies 16 bytes, the pointer and the count, never the 8 MiB they describe.

<figure>
<img src="../../figures/ch10-ownership.svg" alt="Two panels. Copy, deleted: objects a and b both hold data_ = 0x7f00 and both point at the same 8 MiB allocation; ~b frees it and ~a frees it again, a double free, so the copy is made a compile error. Move, allowed: after b = std::move(a), b holds 0x7f00 and a holds nullptr; only b points at the allocation, ~b frees it once and ~a has nothing to free.">
<figcaption><b>Figure 10.4</b> 2 owners free twice; a move leaves 1 owner.</figcaption>
</figure>

<p class="listing"><b>Listing 10.6</b> Copies deleted, moves allowed.</p>

```cpp
// `count` values of T in device memory, owned by exactly 1 object.
template <typename T>
class DeviceBuffer {
    // ...
{{#include ../../../../code/ch10_modern/device_buffer.hpp:59:75}}
    // ...
};
```

`std::exchange(other.data_, nullptr)` returns the old value of `other.data_` and stores
`nullptr` in its place, in 1 step. The move assignment frees what `*this` held before taking
`other`'s pointer; the test `this != &other` keeps `a = std::move(a)` from freeing the buffer it
is about to keep. Both moves are `noexcept`, a promise that they never throw; they only exchange
2 numbers, and nothing in them can fail.

C++ programmers know this set of members as the **rule of 5**: a class that defines any of the
destructor, copy constructor, copy assignment, move constructor and move assignment usually
needs to define, or delete, all 5. The table lists our 5 choices.

| Member | `DeviceBuffer` | Reason |
|---|---|---|
| destructor | frees `data_` | the object owns the allocation |
| copy constructor | deleted | 2 owners would free twice |
| copy assignment | deleted | same |
| move constructor | takes the pointer, leaves `nullptr` | 1 owner after the move |
| move assignment | frees its own, then takes the pointer | same, without leaking the old buffer |

A copy written by mistake now fails to compile. We added the line
`DeviceBuffer<float> copy = in;` to the program and compiled it:

```console
copy.cu(113): error: function "DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer<T> &) [with T=float]" (declared at line 59 of device_buffer.hpp) cannot be referenced -- it is a deleted function
      DeviceBuffer<float> copy = in;
                                 ^
```

The rest of the class gives access to the memory and frees it once.

<p class="listing"><b>Listing 10.7</b> Access, and the 1 place that frees.</p>

```cpp
// `count` values of T in device memory, owned by exactly 1 object.
template <typename T>
class DeviceBuffer {
    // ...
{{#include ../../../../code/ch10_modern/device_buffer.hpp:77:100}}
};
```

`view()` returns a read-only window on the buffer and `span()` a writable one; section 10.6.1
introduces the type. `release()` tests for `nullptr` because a moved-from buffer owns nothing.

### 10.2.4 Errors without a macro

Chapters 3 to 9 checked each call with a small function that printed the error and ended the
program. Many CUDA programs use a macro instead, `CHECK(call)`, for 1 reason: a macro can paste
`__FILE__` and `__LINE__` into the message, so the message names the call that failed. A
function could not see where it was called from, until C++20 added `std::source_location`.

A **default argument** is a value that a function parameter takes when the caller leaves it out.
`std::source_location::current()` used as a default argument is evaluated at the call site, so
it records the caller's file and line, not the line inside `check`.

<p class="listing"><b>Listing 10.8</b> <code>check</code>, with the caller's file and line.</p>

```cpp
{{#include ../../../../code/ch10_modern/device_buffer.hpp:14:25}}
```

`check` reports a failure with an **exception**: an object thrown with `throw` that travels up
the chain of calls until a `catch` takes it. On its way, the exception leaves every scope it
passes through, and every local object in those scopes is destroyed. Figure 10.3 draws
the route as "check() throws": the buffers of a function that fails are freed before the error
reaches whoever handles it.

The destructor itself does not call `check`. A destructor may run while an exception is
already travelling, and a second exception thrown at that moment ends the program
(`std::terminate`). The destructor therefore frees and ignores the result; a program that needs
to know whether a free succeeded can call `cudaDeviceSynchronize` through `check` before the
buffer goes out of scope.

### 10.2.5 The mistakes of section 10.1, again

<p class="listing"><b>Listing 10.9</b> The early return and the launch over the limit, with <code>DeviceBuffer</code> and <code>check</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch10_modern/modern.cu">code/ch10_modern/modern.cu</a></p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:112:135}}
```

`main` makes the same 8 calls as `raw.cu`, launches the 2,048 threads inside a `try`, and copies
2,097,152 floats through the typed constructor.

<p class="listing"><b>Listing 10.10</b> The measurements of section 10.1, repeated.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch10_modern/modern.cu:176:200}}
    // ...
}
```

```console
$ nvcc -std=c++20 --extended-lambda -arch=native modern.cu -o modern && ./modern
GPU: Orin, 8 SMs
ownership
  8 calls that return early: 0.0 MiB lost
  2048 threads per block threw: modern.cu:134: invalid configuration argument
  after the exception: 0.0 MiB in use
  copies sized by the type: 2097152 of 2097152 correct
```

All 3 mistakes are gone. The early returns lost 0.0 MiB. The launch error became an exception
that names line 134, the `check(cudaGetLastError())` after the launch, and the 2 buffers of
`launchTooWide`, 16 MiB, were freed while the exception travelled to `main`. Every element was
copied, because no caller states a byte count.

<div class="bee bee-happy" markdown="1">

Put the release of a resource in a destructor, and the compiler runs it on every way out of the
scope: a return, an exception, or the closing brace.

</div>

### 10.2.6 The same rules in Rust

Rust builds RAII into the language. Its destructor is the method `drop` of the trait `Drop`.
The compiler calls it when a value's owner goes out of scope. cuda-core's `DeviceBuffer<T>`
implements `Drop` with a call to free its allocation, so a Rust program has no free to forget.
Ownership is also the default: `let b = a;` **moves** the buffer, and any later use of `a` is a
compile error. A copy must be asked for explicitly, and `DeviceBuffer` does not offer one.

Errors travel as values of type `Result<T, E>`, either `Ok(value)` or `Err(error)`. The `?`
operator after a call returns the `Err` from the enclosing function at once, and the return,
like any other, drops every local buffer.

`pool_copy` allocates from the same pool as the C++ program, so its count can be compared. Safe
cuda-core functions such as `DeviceBuffer::from_host` allocate with the synchronous `cuMemAlloc`,
which the pool does not count; `uninitialized_async` takes the memory from the pool, and is
`unsafe` only because the buffer's contents are undefined until written. The copy on the next
line writes every element.

<p class="listing"><b>Listing 10.11</b> The early return in Rust. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch10_modern/src/main.rs">code/oxide/ch10_modern/src/main.rs</a></p>

```rust,ignore
{{#include ../../../../code/oxide/ch10_modern/src/main.rs:95:124}}
```

The Rust launch carries a **launch contract** (Chapter 13): the kernel declares that it runs in
blocks of 256 threads. `prepare_scale` checks the requested configuration against it on the
host, before any call to the driver.

<p class="listing"><b>Listing 10.12</b> The launch over the limit, and the only way to leak.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch10_modern/src/main.rs:126:146}}
```

`std::mem::forget` takes ownership of a value and never drops it. It is safe Rust, because a
leak cannot make a program read or write wrong memory, but it has to be written out by name.

```console
$ cargo oxide run
ownership
  8 calls that return early: 0.0 MiB lost
  2048 threads per block returned: scale: block (2048, 1, 1) does not match required block (256, 1, 1)
  after the error: 0.0 MiB in use
  copies sized by the type: 2097152 of 2097152 correct
  8 calls with mem::forget: 64.0 MiB lost
```

The 8 calls of `forget_on_gpu` each forgot 1 buffer of 8 MiB: 64.0 MiB, as counted. The rest
matches the C++ program, with 1 difference: the wrong block size never reached the driver.

<details class="check">
<summary>A function creates a DeviceBuffer, then calls a function that throws. Who frees the buffer, and when?</summary>

The destructor of the buffer, while the exception passes through the function's scope on its
way to a `catch`. The function's own code after the throwing call never runs, but the compiler
destroys every local object of every scope the exception leaves.

</details>

## 10.3 Templates: One Kernel, Many Types

### 10.3.1 A template and its instantiations

The kernel `scale` of `raw.cu` multiplies floats. A program that also multiplies doubles and
ints would need 3 copies of it that differ only in the type. A **template** is a function or
class written with a **template parameter**, a placeholder for a type, here `T`. The template
is not code yet. When the program uses it with a particular type, the compiler substitutes the
type and compiles the result, an **instantiation**. `transform<float, Scale<float>>` and
`transform<double, Scale<double>>` are 2 different kernels in the binary, as if each had been
typed by hand.

Our kernel is generic in 2 ways: in the element type `T`, and in the operation `F` applied to
each element. Its pseudocode is that of the C-style `scale`, with the multiplication replaced by
a call to `f`.

```text
KERNEL transform<T, F>(in, out, f):         # T: element type, F: operation from T to T
    i ← blockIdx.x × blockDim.x + threadIdx.x  # this thread's index (Chapter 3)
    if i < length of in:
        out[i] ← f(in[i])
```

<p class="listing"><b>Listing 10.13</b> A template kernel and an operation to pass to it.</p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:34:49}}
```

The line `template <Number T, UnaryOp<T> F>` says that `T` must be a number and `F` an
operation from `T` to `T`; section 10.7 defines both conditions. The arguments `in` and `out` are
spans, pointers that carry their length, so the guard compares `i` with `in.size()` rather than
with a separate `n` that could disagree (section 10.6.1). The index `i` is a `std::size_t`,
64 bits, so the kernel works on arrays of more than 2<sup>31</sup> elements.

`Scale<T>` is a **function object**: a struct with an `operator()`, so that `triple(x)` calls it
like a function. It carries its state, the factor, as a member, and travels to the GPU as a
kernel argument of `sizeof(T)` bytes. Its `operator()` is marked `__host__ __device__` so that
the kernel can call it; section 10.4 explains the 2 qualifiers.

<figure>
<img src="../../figures/ch10-instantiate.svg" alt="A template, transform with type T and operation F, fans out into 3 kernels the compiler writes: transform<float>, 4 bytes in and 4 out, 74.6 GB/s; transform<double>, 8 bytes in and 8 out, 90.2 GB/s; transform<int>, 4 bytes in and 4 out, 77.0 GB/s. Below, the hand-written float kernel scaleRaw ran at 74.5 GB/s.">
<figcaption><b>Figure 10.5</b> 3 sets of types, 3 kernels, each at the speed of a hand-written one.</figcaption>
</figure>

<p class="listing"><b>Listing 10.14</b> 1 timing function, instantiated for 3 types.</p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:137:164}}
```

`timeType` is itself a template. `timeType<double>` creates `Scale<double>` and launches
`transform<double, Scale<double>>`; the compiler instantiates the whole chain from the 1 line
that calls it.

### 10.3.2 Lambdas as operations

A **lambda** is a function object written in place: `[](float v) { return 2.0f * v; }` defines an
unnamed struct with that `operator()` and creates 1 object of it. Passing one to a kernel looks
natural, and does not compile:

```console
hostlambda.cu(227): error: The closure type for a lambda ("lambda [](float)->float", defined at hostlambda.cu:224) cannot be used in the template argument type of a __global__ function template instantiation, unless the lambda is defined within a __device__ or __global__ function, or the flag '-extended-lambda' is specified and the lambda is an extended lambda (a __device__ or __host__ __device__ lambda defined within a __host__ or __host__ __device__ function)
```

A lambda written in host code is a host function, and the GPU cannot run host code. The message
names the 2 ways out. An **extended lambda** carries `__device__` or `__host__ __device__` after
its capture list, and the compiler builds a device version of it; the program must then be
compiled with `--extended-lambda`. Section 10.4.2 uses 2 of them.

A lambda can also **capture** variables, copy them into its object: `[factor](float v) {...}`.
The copies travel to the GPU with the kernel's arguments, so they must be trivially copyable.
A capture by reference, `[&factor]`, would send the address of a host variable, which the GPU
cannot use, and nvcc refuses it with the error
`An extended __device__ lambda cannot capture variables by reference`.

### 10.3.3 What the abstraction costs

A template, a function object and a span are abstractions: names for things the hand-written
kernel did directly. The measurement compares `transform<float>` with `scaleRaw`, the C-style
kernel of `raw.cu`, on the same 16,777,216 floats, and runs the other 2 instantiations beside
them. Each timing follows 500 ms of warm-up, because a GPU raises the clocks of its cores
and of its memory only under sustained load.

```console
1 template, 3 types, 16777216 values
  raw pointers, float       1.80 ms    74.5 GB/s
  transform<float>          1.80 ms    74.6 GB/s
  transform<double>         2.98 ms    90.2 GB/s
  transform<int>            1.74 ms    77.0 GB/s
```

The rate counts each element read once and written once: 2 × 16,777,216 × 4 bytes = 134.2 MB for
float, divided by the time. `transform<float>` and `scaleRaw` both took 1.80 ms; the abstraction
cost nothing measurable. Repeated runs drift by 10 to 20% in both directions, and the 2 kernels
moved together. `transform<double>` moved twice the bytes, 268.4 MB, in 1.66 times the time. At
8 bytes per element, each thread's load and store carry twice the data, and the kernel came
closer to the GPU's 102 GB/s: 90.2 GB/s is 88% of it.

Each kernel's speed is set by memory, so on any GPU the least time follows from the bytes moved
and the memory bandwidth, the most bytes the memory can transfer per second.

| GPU | Memory bandwidth | Least time for `transform<float>`, 134.2 MB |
|---|---|---|
| Jetson Orin NX 16 GB | 102 GB/s | 1.32 ms |
| Jetson AGX Thor | 273 GB/s | 0.49 ms |
| DGX Spark (GB10) | 273 GB/s | 0.49 ms |
| GeForce RTX 4090 | 1,008 GB/s | 0.133 ms |
| GeForce RTX 5090 | 1,792 GB/s | 0.075 ms |
| A100 SXM 80 GB | 2,039 GB/s | 0.066 ms |
| H100 SXM | 3,350 GB/s | 0.040 ms |

The measured 1.80 ms is 74.6 GB/s, 73% of the GPU's 102 GB/s. A template does not move that limit;
only fewer bytes would.

### 10.3.4 Generic kernels in Rust

Rust's templates are **generics**: a function with a type parameter `T` and a list of conditions
on it. The compiler instantiates a generic function for each type it is used with, a step Rust
calls **monomorphisation**. cuda-oxide applies it to kernels: `scale::<f32>` and `scale::<f64>`
become 2 kernels.

<p class="listing"><b>Listing 10.15</b> A generic kernel.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch10_modern/src/main.rs:41:54}}
    // ...
}
```

`T: Copy + Mul<Output = T>` lists what the body needs: values of `T` can be copied, and 2 of
them multiply into a `T`. The host side names the type when it prepares the launch,
`prepare_scale::<T>`, and a generic host function can do so for its own `T`.

<p class="listing"><b>Listing 10.16</b> 1 generic timing function for 3 types.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch10_modern/src/main.rs:169:193}}
```

```console
1 generic kernel, 3 types, 16777216 values
  scale::<f32>      1.92 ms    69.9 GB/s
  scale::<f64>      2.88 ms    93.3 GB/s
  scale::<i32>      1.77 ms    75.7 GB/s
```

The Rust instantiations ran within 7% of the C++ ones, inside the drift between runs, and in
the same order: double fastest per byte.

<div class="bee bee-happy" markdown="1">

A template is a recipe for kernels. Each set of types gives a separate kernel, compiled as if
written by hand, and it runs at the same speed.

</div>

## 10.4 `__host__ __device__` Functions

### 10.4.1 1 definition, 2 compilations

CUDA marks every function with the processor it is compiled for. A `__global__` function is a
kernel: launched from the host, run on the device. A `__device__` function runs on the device and
is called from device code. A function with no mark, or marked `__host__`, runs on the host. A
function marked both `__host__ __device__` is compiled twice from 1 definition, once for each
processor, and can be called from either side.

A test can then compare the 2 compilations: run the function on the CPU, run it in a kernel, and
check that the results agree. Testing a GPU result against a CPU result computed by the same
code is called **differential testing**.

```text
for each x in the test values:
    cpu[x] ← f(x)                    # host compilation
gpu ← run f on every x in a kernel   # device compilation
count the x where the bits of cpu[x] and gpu[x] differ
```

A function compiled for both sides may use only what exists on both: arithmetic, the CUDA maths
functions such as `fmaf`, `sqrtf` and `expf`, and plain C++. A part that must differ is guarded
by the macro `__CUDA_ARCH__`, which the compiler defines only while it compiles device code:

```cpp
__host__ __device__ float logOf(float x)
{
#ifdef __CUDA_ARCH__
    return __logf(x);           // device: the fast hardware approximation
#else
    return std::log(x);         // host: the standard library
#endif
}
```

### 10.4.2 The same source, different bits

The polynomial 1 + x + x²/2 + x³/6 is the start of the series of e<sup>x</sup>. Horner's rule
writes it as 1 + x(1 + x(1/2 + x/6)): 3 multiplications, each followed by an addition.

<p class="listing"><b>Listing 10.17</b> 1 polynomial, written 2 ways, for both processors.</p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:60:72}}
```

`main` wraps each function in an extended `__host__ __device__` lambda, runs it in `transform` on
2,097,152 values from −1 to 1, calls the same lambda on the CPU, and compares the bits of the
results with `memcmp`.

<p class="listing"><b>Listing 10.18</b> The differential test.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch10_modern/modern.cu:226:247}}
    // ...
}
```

```console
1 function on both sides, 2097152 values in [-1, 1)
  x * y + z: 585903 of 2097152 differ from the CPU
  fmaf:      0 of 2097152 differ from the CPU
```

The plain version differs in 585,903 values, 28%. The cause is an instruction. A **fused
multiply-add**, FMA, computes a × b + c with a single rounding at the end: the product is kept
exact inside the addition. A multiplication followed by an addition rounds twice, once after
each operation. The GPU compiler fuses each `x * (...) + ...` of `expPoly` into an FMA by default;
the host compiler, here, did not.

<div class="example" markdown="1">

Worked example: 1.23 × 4.56 − 5.60 with 3 significant digits

1. 2 roundings: 1.23 × 4.56 = 5.6088, rounded to 5.61. Then 5.61 − 5.60 = 0.0100.
2. 1 rounding: 1.23 × 4.56 − 5.60 = 0.0088 exactly, rounded to 0.00880.
3. The results differ by 14%, because the first rounding threw away 0.0012 that the subtraction
   then exposed.

</div>

<figure>
<img src="../../figures/ch10-fma.svg" alt="Two columns worked with 3 significant decimal digits. Left, 2 roundings: 1.23 times 4.56 = 5.6088, rounded to 5.61, minus 5.60 gives 0.0100. Right, 1 rounding: 1.23 times 4.56 minus 5.60 is exactly 0.0088, rounded to 0.00880. In the program, the polynomial written as x * y + z differs between CPU and GPU in 585,903 of 2,097,152 values; written with fmaf on both sides, in 0.">
<figcaption><b>Figure 10.6</b> A fused multiply-add rounds once; a multiply and an add round twice.</figcaption>
</figure>

A float keeps 24 binary digits, so in `expPoly` the 2 results differ in their last binary
digits, about 1 part in 10<sup>7</sup>. Neither result is wrong: each follows the rounding rules
for its own sequence of operations, and the fused one rounds fewer times. But a differential
test that expects identical bits fails on 28% of its inputs. `expPolyFma` writes
each step as `fmaf(a, b, c)`, which rounds once on both processors because the host's `fmaf`
follows the same IEEE 754 rule. Its 0 differences show that the device compilation is correct.

`nvcc --fmad=false` would turn the fusing off for the whole file, and make every kernel slower,
because each fused operation becomes 2 instructions. Writing `fmaf` where the bits must agree
states the intent in the code, at the 1 place it applies.

### 10.4.3 In Rust

In cuda-oxide, an ordinary Rust function called from a kernel is compiled for the device as well
as for the host; it needs no qualifier.

<p class="listing"><b>Listing 10.19</b> The polynomial in Rust, called by a kernel and by the CPU.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch10_modern/src/main.rs:25:35}}
```

`x.mul_add(a, b)` is Rust's FMA. The kernels `poly` and `poly_fma` call the 2 functions, and
`main` compares their results with the same functions run on the CPU.

```console
1 function on both sides, 2097152 values in [-1, 1)
  x * y + z: 585903 of 2097152 differ from the CPU
  mul_add:   0 of 2097152 differ from the CPU
```

The counts are the same as in C++ to the value: the device compiler fuses, the host compiler does
not (Rust never fuses on its own), and `mul_add` rounds once on both.

<div class="bee bee-shocked" markdown="1">

The same source can give different bits on the CPU and the GPU, because the GPU fuses a multiply
and an add into 1 rounding. Write `fmaf` (or `mul_add`) where 2 sides must agree to the bit.

</div>

## 10.5 `constexpr` and `static_assert`

A **constant expression** is an expression whose value the compiler can compute while it compiles
the program. A variable declared `constexpr` must be initialised with one, and a function
declared `constexpr` can be evaluated by the compiler when its arguments are constant
expressions. A `static_assert(condition, message)` evaluates its condition while compiling and
stops the compilation, printing the message, when the condition is false.

Together they move checks from run time to compile time. A **warp** is a group of 32 threads of
a block that the hardware runs together, 1 instruction for all 32. A block size that is not a
multiple of 32 leaves the last warp of every block partly idle; a size above 1,024 threads
cannot launch at all (section 10.1.3).

<p class="listing"><b>Listing 10.20</b> The block size, checked while compiling.</p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:14:23}}
```

<div class="example" markdown="1">

Worked example: `blocksFor(1000)` with blocks of 256

1. (1,000 + 256 − 1) / 256 = 1,255 / 256 = 4 in integer division.
2. 4 blocks hold 1,024 threads, enough for 1,000 elements; 3 blocks, 768 threads, would not be.
3. The compiler performs this division itself, and `static_assert(blocksFor(1000) == 4)` holds.

</div>

Changing the first line to `constexpr int BLOCK = 100;` makes the file fail to compile:

```console
block100.cu(15): error: static assertion failed with "a block is a whole number of warps"
  static_assert(BLOCK % 32 == 0, "a block is a whole number of warps");
  ^

block100.cu(23): error: static assertion failed
  static_assert(blocksFor(1000) == 4);
  ^
```

With blocks of 100, 1,000 elements need 10 blocks, so the second assertion fails too. A
`constexpr` constant is preferred to `#define BLOCK 256` because it has a type and a scope, and
can appear in `static_assert`; a macro is text pasted wherever its name appears.

Rust writes the same checks as constant assertions: `assert!` inside a `const` item is evaluated
by the compiler, and a false condition is a compile error.

<p class="listing"><b>Listing 10.21</b> The checks in Rust.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch10_modern/src/main.rs:16:18}}
```

## 10.6 CUDA 12 and the `cuda::` Namespace

CUDA ships a C++ library for device code called **libcu++**. Its names live in 2 namespaces:
`cuda::std::`, which holds versions of standard library types that work in device code, and
`cuda::`, which holds extensions for the GPU. Both headers used here, `<cuda/std/span>` and
`<cuda/atomic>`, compiled and ran with CUDA 12.6 and with CUDA 13.2.

### 10.6.1 `cuda::std::span`

A **span** is a pointer and a length kept together: a view of a contiguous array that it does
not own. `cuda::std::span<const T>` allows reading; `cuda::std::span<T>` allows writing too.
`DeviceBuffer::view()` and `span()` return one of each, and `transform` receives them:

```cpp
transform<<<blocksFor(in.size()), BLOCK>>>(in.view(), out.span(), Scale{factor});
```

The C-style kernel received a pointer and a separate `n`, and a caller could pass an `n` that
belonged to another array. A span's length comes from the buffer it views, so the guard
`i < in.size()` always uses the right one. A span is 16 bytes, a pointer and a length, and
passes to the kernel by value like any other argument. A Rust slice, `&[T]`, has been a pointer
and a length from the start, which is why the Rust kernels of this book never take a separate
count.

### 10.6.2 `cuda::atomic_ref` and thread scopes

An **atomic** operation on a memory location completes as 1 indivisible step: no other thread
can see or change the location halfway through it (Chapter 5). `cuda::atomic_ref<int, scope>`
applies atomic operations to an ordinary `int` that already exists, here a counter in a
`DeviceBuffer<int>`; the `int` itself stays a plain `int`.

The **memory order** of an atomic operation says what else it orders: whether writes made before
it become visible to a thread that sees its result. `cuda::memory_order_relaxed` orders nothing
but the counter itself, which is all a count needs, and costs the least.

The **thread scope** names the set of threads that must agree on the value. The narrower the set,
the less the hardware has to do to keep them in agreement.

| Scope | The threads that must agree | Example |
|---|---|---|
| `cuda::thread_scope_block` | the threads of 1 block | a counter in shared memory (Chapter 5) |
| `cuda::thread_scope_device` | every thread of 1 GPU | a counter in global memory |
| `cuda::thread_scope_system` | the CPU and every GPU | a flag in memory both processors write |

<figure>
<img src="../../figures/ch10-scopes.svg" alt="Three nested zones. The outer zone, system, holds the CPU and every GPU and goes with thread_scope_system, for memory that the CPU and the GPU both update, such as a flag. Inside it, device holds every thread of 1 GPU and goes with thread_scope_device: countAbove runs 8,192 blocks of 256 threads that add to 1 int in global memory, and counted 622,552 of 2,097,152 values. Inside that, block holds the 256 threads of 1 block and goes with thread_scope_block, as for the shared-memory counters of Chapter 5.">
<figcaption><b>Figure 10.7</b> The 3 scopes, each the set of threads that must agree on a value.</figcaption>
</figure>

<p class="listing"><b>Listing 10.22</b> A count with <code>cuda::atomic_ref</code>.</p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:74:83}}
```

The kernel counts the results of `expPolyFma` above 1.5. Its 2,097,152 threads in 8,192 blocks
of 256 add to 1 `int` in global memory, so the scope must be the device.

```console
span and atomic_ref: 622552 values above 1.5, CPU 622552
PASSED
```

The count can be checked by hand. The polynomial reaches 1.5 at x = 0.40629, a little to the
right of ln 1.5 = 0.40547, because it is smaller than e<sup>x</sup> for x > 0. The inputs are
spaced evenly from −1 to 1, so the share above 0.40629 is (1 − 0.40629) ÷ 2 = 0.29686, and
0.29686 × 2,097,152 = 622,552 values: the number both processors counted.

The Rust programs of Chapter 5 use cuda-device's `DeviceAtomicU32`, which plays the same part.

## 10.7 C++20 Concepts

A template accepts any type for which its body compiles. A **concept** is a named condition on
template parameters, checked when the template is used. Our 2 concepts say what `transform`
needs.

<p class="listing"><b>Listing 10.23</b> 2 concepts.</p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu:28:32}}
```

`Number<T>` holds when `T` is an integer or floating-point type. `UnaryOp<F, T>` holds when an
`F` can be called with a `T` and returns exactly a `T`. Writing `UnaryOp<T> F` in the template's
parameter list applies the concept to `F`, with `T` as its second argument.

The condition catches a real mistake. We put an operation that returns a `bool`,
`[] __device__ (float v) { return v > 0.0f; }`, in place of `Scale{factor}` in `scaleOnGpu`, and
compiled it first against a version of `transform` without the concepts. It compiled with no warning, because C++ converts a `bool` to a `float` silently,
and it ran:

```console
  copies sized by the type: 0 of 2097152 correct
```

Every result was 1.0, the float value of `true`, and none was the 2.0 expected. With the
concepts, the same call does not compile:

```console
concept.cu(118): error: no instance of function template "transform" matches the argument list
            argument types are: (cuda::std::__4::span<const float, 18446744073709551615UL>, cuda::std::__4::span<float, 18446744073709551615UL>, lambda [](float)->bool)
concept.cu(36): note #3327-D: candidate function template "transform" failed deduction
```

The message points at the call and shows the lambda's type, `(float)->bool`, which is the
mistake. A Rust generic states its conditions the same way, as **trait bounds**. Rust checks
them when the function is used, and also when it is defined: the body of `scale<T>` may only do
what `Copy + Mul<Output = T>` allows.

## 10.8 Style Summary

Figure 10.8 collects the mistakes run in this chapter and the point at which each is caught.

<figure>
<img src="../../figures/ch10-caught.svg" alt="A table of 6 mistakes against 3 ways of writing the program. Leak on early return: C API 128 MiB lost; modern C++ and Rust cannot happen. 2,048 threads per block: C API silent, out = 0; C++ throws with file and line; Rust returns an Err from prepare. Bytes for elements: C API 1 in 4 copied; C++ and Rust cannot happen. Copy of an owner: C API double free; C++ compile error; Rust moved, not copied. Block not a multiple of 32: C API not checked; C++ static_assert; Rust const assert. An operation that returns bool: C API has no generics; C++ concept error; Rust trait bound error.">
<figcaption><b>Figure 10.8</b> 6 mistakes of this chapter, written 3 ways, and when each is caught.</figcaption>
</figure>

The pattern behind the figure is to move a rule from "remember to do this" to "the program cannot
be written otherwise". Some rules become impossible to break, because no call site writes the
step any more: a destructor frees, a type computes the byte count. Some become compile errors:
deleted copies, `static_assert`, concepts. The rest become run-time errors that cannot be
ignored: an exception, or an `Err` that `?` passes up. The parts that remain unprovable in C++,
such as index arithmetic, are the ones Rust kernels keep inside `DisjointSlice` and `unsafe`
blocks with a written reason (Chapter 13).

| C-style habit | Modern replacement | What the compiler now does |
|---|---|---|
| `cudaMalloc` and `cudaFree` by hand | `DeviceBuffer<T>` | frees on every path, forbids 2 owners |
| byte counts at each call | sizes from `sizeof(T)` inside the class | no call site writes a byte count |
| `CHECK` macro | `check()` with `std::source_location`, exceptions | names the failed line; frees while unwinding |
| 1 kernel per type | template kernels and function objects | writes 1 kernel per type |
| `#define TILE 16` | `constexpr int TILE = 16` and `static_assert` | checks configuration while compiling |
| pointer and separate `n` | `cuda::std::span` | keeps the length with the pointer |
| `atomicAdd` on a raw pointer | `cuda::atomic_ref` with a scope | states which threads must agree |
| unconstrained templates | concepts | rejects a wrong operation at the call |

All of it costs nothing at run time: `transform<float>` ran at 74.6 GB/s against 74.5 GB/s for the
hand-written kernel.

## 10.9 The complete programs

<p class="listing"><b>Listing 10.24</b> The complete <code>device_buffer.hpp</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch10_modern/device_buffer.hpp">code/ch10_modern/device_buffer.hpp</a></p>

```cpp
{{#include ../../../../code/ch10_modern/device_buffer.hpp}}
```

<p class="listing"><b>Listing 10.25</b> The complete <code>raw.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch10_modern/raw.cu">code/ch10_modern/raw.cu</a></p>

```cpp
{{#include ../../../../code/ch10_modern/raw.cu}}
```

<p class="listing"><b>Listing 10.26</b> The complete <code>modern.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch10_modern/modern.cu">code/ch10_modern/modern.cu</a></p>

```cpp
{{#include ../../../../code/ch10_modern/modern.cu}}
```

<p class="listing"><b>Listing 10.27</b> The complete Rust program. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch10_modern/src/main.rs">code/oxide/ch10_modern/src/main.rs</a></p>

```rust,ignore
{{#include ../../../../code/oxide/ch10_modern/src/main.rs}}
```

## Common mistakes

- **Copying an owner by hand.** A struct that holds a raw device pointer and is copied leads to
  2 frees. Delete the copy operations and move with `std::move`.
- **Throwing from a destructor.** A destructor can run while an exception travels; a second
  exception ends the program. Free and ignore the result there.
- **A host lambda passed to a kernel.** Mark it `__device__` or `__host__ __device__` and compile
  with `--extended-lambda`.
- **A capture by reference in a device lambda.** It sends a host address to the GPU. Capture by
  value, and only trivially copyable values.
- **Comparing CPU and GPU results bit for bit with plain `a * b + c`.** The GPU fuses it into an
  FMA. Write `fmaf` where the bits must agree, or compare with a tolerance.
- **A thread scope wider than needed.** `thread_scope_system` costs more than
  `thread_scope_device`; use the narrowest scope that contains every thread that touches the
  value.
- **Measuring without a warm-up.** With 0.3 s of warm-up, the kernels of this chapter ran at 33 to
  48 GB/s; with 500 ms before each measurement, at 70 to 90 GB/s. The clocks of a GPU and of its
  memory rise only under sustained load.

<div class="summary" markdown="1">

## Summary

- The C API leaves 3 duties at every call site. Forgotten, they lost 128 MiB in 8 early returns,
  skipped a kernel with no message, and copied 1 in 4 floats.
- RAII acquires in a constructor and releases in the destructor; the compiler runs the destructor
  on every way out of a scope. `DeviceBuffer<T>` lost 0.0 MiB.
- 1 owner per allocation: copies are deleted, moves hand the pointer over and leave `nullptr`.
- `check()` with `std::source_location` names the failed line without a macro, and its exception
  frees every buffer it passes.
- Template kernels give 1 kernel per type at hand-written speed: 74.6 GB/s against 74.5 GB/s.
- `__host__ __device__` functions compile for both processors; the GPU's FMA made 585,903 of
  2,097,152 results differ in the last bit until `fmaf` made both sides round once.
- `constexpr`, `static_assert`, spans, `cuda::atomic_ref` with a scope, and concepts each turn a
  rule into something the compiler checks.
- Rust has each of these in the language: `Drop`, moves, `Result` and `?`, generics with trait
  bounds, constant assertions; leaking takes an explicit `mem::forget`.

</div>

Chapter 11 puts these tools to use in NVIDIA's libraries: Thrust's `device_vector` is a
`DeviceBuffer` with algorithms, CUB is a library of template building blocks, and cuBLAS returns
to the matrix product of Chapter 9.

## Exercises

1. Remove the `= delete` from the copy constructor of `DeviceBuffer`, write `b = a`, and run the
   program. What happens at the end of the scope, and why?
2. Add a `release()` that frees early and leaves the object empty, and use it in a loop that
   allocates a large buffer each iteration. Show with `poolMiB()` that the peak stays at 1 buffer.
3. Write `DeviceBuffer<std::string> s(4);` and read the error. Which property of `std::string`
   does it object to?
4. Write a `__host__ __device__` function `lerp(a, b, t) = a + t × (b − a)`, test it
   differentially on 1,048,576 values, and count the differences. Rewrite it with `fmaf` so that
   none remain.
5. Instantiate `transform` for `unsigned char` and measure it. Explain the rate from the bytes
   each thread moves.
6. Change `countAbove` to `thread_scope_block` and run it. Is the count still right on your GPU?
   Is it guaranteed to be?
7. Write a concept `Floating` that accepts only `float` and `double`, and use it on a kernel that
   calls `sqrt`. What error does `transform<int>` give now?
8. In Rust, replace `pool_copy` with `DeviceBuffer::from_host` in `scale_on_gpu`. Does the pool
   still count the buffers? Why?

## Sources and further reading

- NVIDIA, *CUDA C++ Programming Guide*, "Function Execution Space Specifiers", "Extended
  Lambdas" and "Stream Ordered Memory Allocator":
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *libcu++ documentation*, `cuda::std::span`, `cuda::atomic_ref` and thread scopes:
  <https://nvidia.github.io/cccl/libcudacxx/>
- NVIDIA, *CUDA Floating Point and IEEE 754*, on fused multiply-add and differences between CPU
  and GPU results: <https://docs.nvidia.com/cuda/floating-point/>
- B. Stroustrup, *The C++ Programming Language*, 4th ed., Addison-Wesley, 2013, chapters on
  resource management, copy and move, and templates.
- cppreference, "RAII", "The rule of three/five/zero", "std::source_location", "Constraints and
  concepts": <https://en.cppreference.com/>
- S. Klabnik and C. Nichols, *The Rust Programming Language*, chapters "Understanding Ownership"
  and "Generic Types, Traits, and Lifetimes": <https://doc.rust-lang.org/book/>
