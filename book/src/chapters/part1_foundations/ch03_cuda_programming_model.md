# Chapter 3: The CUDA Programming Model

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch03_vector_add/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch03_vector_add)
> in the repository.

This chapter introduces the CUDA programming model: how a function becomes a
kernel, how a launch describes a grid of work, and how data moves between the
CPU (the *host*) and the GPU (the *device*). The first complete program, a
vector addition, is presented with line-by-line commentary.

## 3.1 Host and Device

CUDA programs are divided into two worlds:

- **Host** - the CPU and its memory. The host launches kernels and moves data.
- **Device** - the GPU and its memory (global memory, §2.5). The device
  executes kernels.

The two worlds do not share an address space. A pointer returned by
`cudaMalloc` is a device pointer: dereferencing it on the host is undefined
behaviour and, in practice, a crash. Data crosses the boundary explicitly with
`cudaMemcpy`. This separation is the most common source of confusion for new
CUDA programmers, and it remains the mental model even after Chapter 4
introduces pinned memory and unified memory.

> **Primitive - host.** The CPU side of a CUDA program.
> **Primitive - device.** The GPU side of a CUDA program.
> **Primitive - kernel.** A function that runs on the device, launched by the
> host, and executed by many threads.

The separation corresponds to physical reality. The host is a CPU across a bus;
the device is the GPU die with its GPCs of SMs, chip-wide L2, and DRAM. Every
`cudaMemcpy` crosses that bus; every kernel launch delivers work to the SMs.
The two address spaces are separate because the hardware is physically
separate: different DRAM, different caches, different execution units. The
programming model does not hide that boundary, and most of the CUDA API's
"ceremony" follows from it.

## 3.2 Function Qualifiers

CUDA extends C++ with three function qualifiers:

- `__global__` - the kernel qualifier. The function runs on the device and is
  called from the host (or from the device in cooperative launch on later CUDA
  generations). A `__global__` function must return `void`. Its arguments are
  copied from host memory to the device before launch.
- `__device__` - the function runs on the device and can be called only from
  device code (from a kernel or from another `__device__` function).
- `__host__` - the default: a normal host function. It can be combined as
  `__host__ __device__` to compile one function for both sides, a common
  pattern in modern CUDA (Chapter 10).

A `__device__` function cannot call a `__host__` function because the device
has no host runtime. A `__global__` function cannot be called recursively on
most architectures and cannot take a variable number of arguments.

## 3.3 The Launch Configuration: Grids and Blocks

A kernel launch has the form:

```cpp
myKernel<<<gridDim, blockDim>>>(args...);
```

The double-angle-bracket expression is the **execution configuration**. It
describes the shape of the work. Both arguments have type `dim3`, a
three-component vector type with fields `x`, `y`, and `z`, each an unsigned
integer.

- **blockDim** is the number of threads per block, with one to three
  dimensions. The total number of threads per block is
  `blockDim.x * blockDim.y * blockDim.z` and must not exceed 1,024 on modern
  hardware.
- **gridDim** is the number of blocks in the grid, with one to three
  dimensions. The total number of threads in the kernel is the product of
  `gridDim` and `blockDim` over all dimensions.

The CUDA Programming Guide states the following launch limits for current
architectures:

| Dimension | Limit |
|---|---|
| Threads per block | 1,024 |
| Block size `x` | 1,024 |
| Block size `y` | 1,024 |
| Block size `z` | 64 |
| Grid size `x` | 2³¹ − 1 |
| Grid size `y` | 65,535 |
| Grid size `z` | 65,535 |

These are architectural limits, not suggestions. A launch that violates them
fails on the host before any kernel runs, so the error appears in a
`cudaGetLastError()` check rather than in device code.

Two 3-D vectors therefore describe the entire launch. `gridDim` specifies the
number of blocks along each axis; `blockDim` specifies the number of threads
along each axis inside every block. Together they describe a 3-D grid of 3-D
blocks.

![The launch hierarchy in 3-D: the grid is a 3-D array of blocks, each block a 3-D array of threads; global position = blockIdx * blockDim + threadIdx](../../assets/ch03_grid_block_3d.svg)

The diagram has three parts:

1. **The grid** is a 3-D array of blocks. `gridDim = (3, 2, 2)` means 3 blocks
   along x, 2 along y, and 2 along z: 12 blocks. Each cube is a block with
   coordinates `blockIdx = (x, y, z)`.
2. **Each block** is itself a 3-D array of threads. `blockDim = (4, 4, 2)`
   means 4 threads along x, 4 along y, and 2 along z: 4 x 4 x 2 = 32 threads,
   which is exactly one warp (§2.3). Each thread has coordinates `threadIdx =
   (x, y, z)` inside its block.
3. **The global position of a thread** is the block's position times the
   block's size plus the thread's position inside the block:

\\[ \text{gx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x} \\]
\\[ \text{gy} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y} \\]
\\[ \text{gz} = \text{blockIdx.z} \times \text{blockDim.z} + \text{threadIdx.z} \\]

The 1-D formula used in §3.5, `blockIdx.x * blockDim.x + threadIdx.x`, is the
x-component of this vector identity. The total number of threads is:

\\[ \text{gridDim.x} \cdot \text{gridDim.y} \cdot \text{gridDim.z} \cdot
\text{blockDim.x} \cdot \text{blockDim.y} \cdot \text{blockDim.z} \\]

Three dimensions exist because real data is often two- or three-dimensional
(images, volumes, grids). A 2-D launch lets a kernel index an image as `(x, y)`
instead of flattening it by hand:

![Why 2-D launches: an 8 x 8 image with a 2 x 2 grid of blocks, each a 4 x 4 tile of threads; global position = blockIdx * blockDim + threadIdx per axis](../../assets/ch03_image_tiles.svg)

The image example gives the working model: **blocks tile the data; threads
fill each tile.** The hardware linearises thread IDs (x fastest, then y, then
z), but the kernel thinks in the data's own shape.

The hardware view (from Chapter 2) is: blocks are assigned to SMs, each block
is partitioned into warps of 32 consecutive threads, and warps execute in
lockstep.

The launch is a declaration, not a loop. `<<<grid, block>>>` does not describe
an order of execution. It states how much work exists and how it is shaped; the
hardware decides which SM runs which block and when. The launch is therefore a
contract with the machine: enough blocks to fill every SM, blocks small enough
to fit SM resources (registers, shared memory, the 1,024-thread cap), and a
shape that maps to the data's natural dimensions.

The following figure shows a small launch, `kernel<<<3, 8>>>` (3 blocks of 8
threads):

![Launch hierarchy: a grid of three blocks of eight threads, and the global index formula](../../assets/ch03_launch_hierarchy.svg)

## 3.4 Built-in Variables

Inside a kernel, four read-only built-in variables describe the launch:

| Variable | Type | Meaning |
|---|---|---|
| `threadIdx` | `dim3` | The thread's position within its block: `threadIdx.x`, `.y`, `.z` |
| `blockIdx` | `dim3` | The block's position within the grid: `blockIdx.x`, `.y`, `.z` |
| `blockDim` | `dim3` | Threads per block (the `blockDim` passed at launch) |
| `gridDim` | `dim3` | Blocks per grid (the `gridDim` passed at launch) |

These are provided by the hardware, not declared by the program. `blockIdx`
identifies the block, `threadIdx` identifies the thread within the block, and
`blockDim`/`gridDim` give the sizes of the two containers. The global formula
`blockIdx * blockDim + threadIdx` translates thread identity into a position in
the data.

## 3.5 The Global Index Formula

The core indexing expression for a 1-D problem is:

```cpp
// Global linear index of this thread, assuming a 1-D grid and 1-D blocks.
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
```

The derivation is direct: thread `threadIdx.x` lives in block `blockIdx.x`.
Each block contains `blockDim.x` threads, so block number `blockIdx.x` starts
at `blockIdx.x * blockDim.x`. Adding the position inside the block gives the
global position. A 1-D launch is the general 3-D launch with the y and z
components equal to one. For a 2-D problem the formula composes per axis:

```cpp
// 2-D indexing: x and y are independent linear indices in their dimension.
unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;  // column
unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;  // row
// Row-major flattening of a width x height image:
unsigned int idx = iy * width + ix;                        // linear memory index
```

In row-major layout, consecutive `ix` values are consecutive in memory. Because
consecutive threads have consecutive `threadIdx.x`, and therefore consecutive
`ix`, this indexing scheme is coalesced by construction (§2.7).

**Worked example.** Launch `kernel<<<4, 256>>>` (4 blocks of 256 threads,
covering 1,024 global indices). For the thread with `blockIdx.x = 2` and
`threadIdx.x = 137`:

```
global index = blockIdx.x * blockDim.x + threadIdx.x
             = 2 * 256 + 137
             = 512 + 137
             = 649
```

That thread owns element 649. Block `blockIdx.x = 2` covers global indices
512..767. Its warp 0 contains threads 0..31, or global indices 512..543: 32
consecutive addresses, coalesced by construction. If the array has 900 elements
rather than a multiple of 1,024, the threads owning indices 900..1,023 are
masked by the `if (i < n)` guard in the kernel.

## 3.6 The First Kernel: Vector Addition

The first complete program adds two `float` arrays element-wise: `c = a + b`
for arrays of length `n`.

### 3.6.1 The Kernel

```cpp
// kernel.cu
// ---------------------------------------------------------------------------
// __global__ : this function runs on the DEVICE, launched from the host.
// Return type must be void. The argument is a device pointer to n floats.
// ---------------------------------------------------------------------------
__global__ void addVectors(const float* a, const float* b, float* c, int n)
{
    // --- Thread identity ---------------------------------------------------
    // blockIdx.x : index of this block within the grid (0-based).
    // blockDim.x : number of threads in this block (set at launch).
    // threadIdx.x: index of this thread within its block (0-based).
    // The product blockIdx.x * blockDim.x is the first global thread index
    // covered by this block; adding threadIdx.x gives the global index.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Boundary guard ----------------------------------------------------
    // The grid may cover more threads than n because the host rounds the grid
    // size up. Threads whose index is >= n must do nothing. Without this guard
    // the kernel would read and write past the end of the arrays.
    if (i < n)
    {
        // Each thread owns exactly one output element, so no two threads write
        // the same address and no race is possible.
        c[i] = a[i] + b[i];
    }
}
```

### 3.6.2 The Host Code

```cpp
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
    // size_t is used because array sizes can exceed the range of int.
    const int    n      = 1 << 20;      // 1,048,576 elements (a power of two)
    const size_t nBytes = n * sizeof(float);

    // --- Host allocations (pageable memory, see Chapter 4) ---------------
    float* h_a = new float[n];   // host input A
    float* h_b = new float[n];   // host input B
    float* h_c = new float[n];   // host output C

    // Fill the inputs with a deterministic pattern so the result is verifiable.
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
    // Block size: 256 threads per block. This is a multiple of the warp size
    // (32), so every warp is full, and small enough that several blocks fit
    // per SM (see occupancy, 2.9). Values of 128-512 are typical.
    const int threadsPerBlock = 256;
    // Grid size: ceil(n / threadsPerBlock). The + (threadsPerBlock - 1)
    // rounds up so that the grid covers every element. Some threads will
    // therefore have i >= n and hit the boundary guard in the kernel.
    const int blocksPerGrid  = (n + threadsPerBlock - 1) / threadsPerBlock;

    // --- Launch ------------------------------------------------------------
    // Kernel launches are asynchronous: the host does not wait for the kernel
    // to finish and control returns to the host immediately (Chapter 6).
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Kernel launches do not report errors synchronously. Check the last
    // error now; if the launch itself failed (bad configuration or pointer),
    // this call catches it.
    CHECK(cudaGetLastError());

    // --- Synchronise --------------------------------------------------------
    // cudaDeviceSynchronize blocks the host until all device work issued so
    // far has completed. This is required before copying the results back.
    CHECK(cudaDeviceSynchronize());

    // --- Device -> host copy ------------------------------------------------
    CHECK(cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost));

    // --- Verify -------------------------------------------------------------
    // The expected value is h_c[i] == 3*i. Check all elements and report the
    // largest error.
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
```

### 3.6.3 Design Rationale

- **One thread per element.** The work is perfectly partitioned, no thread
  depends on another, and coalescing is automatic because the global index
  increases with `threadIdx.x`.
- **Rounded-up grid with a boundary guard.** Rounding the grid up to a multiple
  of the block size means the guard `if (i < n)` is required. The guard costs
  one comparison per thread and makes any problem size safe; computing an exact
  grid adds complexity without a performance benefit.
- **Power-of-two size in the example.** `n = 1 << 20` is used for simplicity.
  Production code uses arbitrary sizes and relies on the boundary guard.

## 3.7 Memory API Primitives

The runtime API functions used above appear throughout the book:

| Function | Behaviour |
|---|---|
| `cudaMalloc(void** p, size_t bytes)` | Allocate `bytes` of device global memory and store the device pointer in `*p`. Returns `cudaSuccess` or an error code. |
| `cudaFree(void* p)` | Free a device allocation made by `cudaMalloc`. |
| `cudaMemcpy(dst, src, bytes, kind)` | Copy `bytes` between host and device. `kind` is one of `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, or `cudaMemcpyHostToHost`. The copy is synchronous and completes before the call returns. |
| `cudaGetLastError()` | Return and clear the last asynchronous error recorded for the calling thread. |
| `cudaGetErrorString(err)` | Return human-readable text for a `cudaError_t`. |
| `cudaDeviceSynchronize()` | Block the host until all preceding device work completes. |

`cudaMalloc` takes `void**` because it is a C-style output-parameter function:
it must write a pointer into the caller's variable. In C++, an allocation
function would return a pointer; CUDA's C heritage writes through a
pointer-to-pointer. The explicit `(void**)` cast is required because C++ does
not allow an implicit conversion from `float**` to `void**`; only `T*` to
`void*` is implicit.

`cudaMemcpy` needs a direction argument because a host pointer and a device
pointer cannot be distinguished by address alone. On some platforms the two
address ranges overlap numerically. The direction flag removes the ambiguity.

## 3.8 Error Handling

CUDA runtime functions return a `cudaError_t`, an enum in which `cudaSuccess`
is 0 and every other value is an error code. There are two failure modes:

1. **Synchronous errors** are detected by the call itself, such as an invalid
   argument or an illegal `cudaMemcpy` kind. The call returns the error code.
2. **Asynchronous errors** are detected after the call, such as an invalid
   kernel launch or an illegal memory access inside a kernel. The launch
   returns `cudaSuccess`; the error surfaces on the next CUDA API call from the
   same thread. This is why `cudaGetLastError()` is called immediately after
   the launch.

The `CHECK` macro reports the offending source line through
`cudaGetErrorString`. Production code should do something more graceful than
`std::exit`, but the discipline of checking every call is not optional. An
unchecked CUDA error can produce a silently wrong answer or a corrupt image.

## 3.9 Compilation and the Build Pipeline

CUDA source files use the `.cu` extension and are compiled by `nvcc`, NVIDIA's
compiler driver. The pipeline has two phases:

1. **Host pass.** `nvcc` separates the host code, compiles it with the host C++
   compiler (`g++` or `clang++`), and replaces each kernel launch
   (`kernel<<<...>>>`) with runtime-API calls that package the arguments and
   launch the kernel.
2. **Device pass.** `nvcc` compiles the `__global__` and `__device__`
   functions to **PTX** (Parallel Thread Execution), NVIDIA's portable virtual
   instruction set, and then to **SASS**, the machine code of the target GPU,
   via `ptxas`.

```bash
# Compile for a specific architecture. This book's portable default is
# compute_60 (Pascal-class PTX): the driver JIT-compiles it to any CUDA 12.x
# GPU, from T4 and P100 to A100, Jetson Orin, and H100.
nvcc -arch=compute_60 kernel.cu -o kernel
# Native SASS alternatives (faster startup, less portable):
#   Jetson Orin : nvcc -arch=sm_87 kernel.cu -o kernel
#   A100        : nvcc -arch=sm_80 kernel.cu -o kernel
#   H100        : nvcc -arch=sm_90 kernel.cu -o kernel
```

> **Primitive - PTX.** The intermediate virtual ISA (Chapter 12 covers it in
> detail). PTX is portable across GPU generations and is translated to SASS by
> the driver at load time when no SASS is embedded.
> **Primitive - SASS.** The GPU's machine code, tied to a specific compute
> capability.

`nvcc` can compile `.cu` files on a machine without an NVIDIA GPU; the resulting
binary will not run there. Every `.cu` file in this book can be compiled with
`nvcc -arch=compute_60 -o bin src.cu` and run on any CUDA 12.x GPU (Pascal or
newer) through the driver's JIT.

## 3.10 Summary

- The launch configuration is a declaration of parallelism, not a loop. The
  hardware schedules the grid onto SMs and the blocks onto warp slots.
- `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` are hardware-provided
  primitives. The global index formula `blockIdx.x * blockDim.x + threadIdx.x`
  maps thread identity to data address.
- Host and device have separate address spaces. Transfers are explicit
  (`cudaMemcpy`) and allocations are explicit (`cudaMalloc`/`cudaFree`).
- Check every CUDA call. Kernel launches are asynchronous, so
  `cudaGetLastError()` after the launch and `cudaDeviceSynchronize()` before
  reading results are both required.

## Launch Semantics: Declaration, Not Order

The conceptual shift in CUDA is to read launch syntax as a declaration of work
rather than a loop. A CPU `for` loop specifies an order of execution. A CUDA
launch specifies how much work exists, shaped in a particular way, and leaves
the mapping to blocks, SMs, and issue order to the hardware.

This has practical consequences. Because the hardware may schedule blocks in
any order, a kernel must not depend on block order for correctness. Two blocks
that need to exchange data must do so through explicit mechanisms: atomics,
separate kernel launches, or cooperative groups. Because the same launch can
run on a GPU with 10 or 100 SMs, a kernel must not assume a particular SM
count. Well-written CUDA kernels are portable across the NVIDIA product line
without source changes for this reason.

The boundary guard is a second consequence of the same design. Rounded-up grids
are the standard way to handle problem sizes that are not multiples of the
block size. The guard `if (i < n)` turns extra threads into no-ops. Its
performance cost is small: at most one warp in the last block diverges. It
prevents out-of-bounds accesses that could otherwise corrupt adjacent memory
and produce silently wrong results.

## Common Pitfalls

- Using `int` for sizes that can exceed 2 billion bytes or indices. `nBytes`
  should be `size_t`; grid and block dimensions are unsigned and have hardware
  limits.
- Launching a grid that does not cover all elements and omitting the boundary
  guard. Rounding up plus `if (i < n)` is the safe pattern.
- Forgetting that the launch is asynchronous. Checking errors immediately after
  the launch is necessary but not sufficient; the host must synchronise before
  reading device results.
- Assuming blocks execute in order. Block scheduling order is not defined and
  must never affect correctness.

## Check Your Understanding

<details>
<summary>Why must a __global__ function return void?</summary>

A kernel is launched for thousands of threads and has no single caller to
receive a return value. Its output consists of the memory writes it performs.
Results are communicated through output buffers rather than return values.
</details>

<details>
<summary>For n = 1,000,000 and 256 threads per block, how many blocks are launched and how many threads are masked?</summary>

blocks = ceil(1,000,000 / 256) = 3,907. Total threads = 3,907 x 256 =
1,000,192, so 192 threads in the last block are masked by `if (i < n)`.
</details>

<details>
<summary>What does cudaGetLastError() catch that cudaDeviceSynchronize() does not?</summary>

`cudaGetLastError()` returns launch-configuration errors recorded
asynchronously (for example, an invalid block size or kernel pointer) before
the host synchronises. `cudaDeviceSynchronize()` waits for completion and
surfaces execution errors such as illegal memory accesses. Both are needed.
</details>

## Key Takeaways

- Host and device have separate address spaces; every transfer is an explicit cudaMemcpy.
- Function qualifiers: __global__ (kernel called from host), __device__ (device only), __host__ (host).
- The launch configuration `<<<grid, block>>>` is a declaration of parallelism, not a loop.
- threadIdx, blockIdx, blockDim, and gridDim are hardware-provided primitives; the global index is blockIdx.x * blockDim.x + threadIdx.x.
- Boundary guards make rounded-up grids safe; they diverge only in the last partial block.
- Check every CUDA call: cudaGetLastError() after the launch, cudaDeviceSynchronize() before copying results back.

## 3.11 Exercises

1. Write the 2-D global index formula for a `width x height` image and show
   that consecutive threads in a warp read consecutive memory addresses when
   the block covers a contiguous row segment.
2. Why must a `__global__` function return `void`? What would a return value
   mean for 1,000,000 threads?
3. Compute `blocksPerGrid` for `n = 1,000,000` and `threadsPerBlock = 256`.
   How many threads in the last block are masked by the boundary guard?
4. What happens if `cudaMemcpy` is called with `cudaMemcpyHostToDevice` but a
   device pointer is passed as the source? Do not try it on a machine you care
   about.

## Sources and Further Reading

- NVIDIA, *CUDA C++ Programming Guide*, "Programming Model" chapter: thread hierarchy, memory hierarchy, heterogeneous programming, and kernel launch syntax: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *CUDA C++ Best Practices Guide*, for host-device transfer and launch-configuration guidance: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVIDIA, `deviceQuery` CUDA sample, for reading a GPU's compute capability and resource limits.
