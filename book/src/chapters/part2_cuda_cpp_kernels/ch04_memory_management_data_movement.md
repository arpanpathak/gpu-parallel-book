# Chapter 4: Memory Management & Data Movement

Chapter 3 moved data with `cudaMalloc` and `cudaMemcpy` without describing how
the transfer happens. This chapter covers the mechanism. The GPU's data path is
a pipeline with three distinct actors: host DRAM, the transfer bus (PCIe or
NVLink), and device DRAM. The performance of a real application is often
dominated by the slowest stage of that pipeline. The chapter covers the memory
kinds a CUDA program can allocate, the copy primitives that move data, and the
unified memory model in which the host and device share a virtual address
space.

## 4.1 The Transfer Pipeline

A `cudaMemcpy` from host to device executes in stages:

1. The CPU reads the data from **pageable host memory**, the ordinary
   `malloc`/`new` kind used in Chapter 3.
2. The runtime copies it through a staging area. The PCIe/NVLink controller
   cannot DMA directly from a pageable page, because the operating system may
   swap that page out at any moment. The runtime therefore copies host data to
   a **pinned staging buffer**, then to the device. This adds an extra copy and
   an extra latency.
3. The device writes the data into device global memory over the transfer bus.

Each stage has a bandwidth. Total transfer time is bounded by the slowest
stage, and identifying that stage is the central problem of host-device data
movement.

## 4.2 Pageable vs Pinned Host Memory

> **Primitive - pageable memory.** Ordinary host memory allocated with
> `malloc`/`new`. The OS may move or swap the underlying pages at any time, so
> the GPU's DMA engine cannot access them directly.
> **Primitive - pinned (page-locked) memory.** Host memory whose pages the OS
> has agreed not to swap. The DMA engine can access it directly. Pinned memory
> is allocated with `cudaMallocHost` or `cudaHostAlloc`.

Pinned memory provides two properties:

1. **Direct DMA.** The runtime skips the staging copy, so a host-device copy is
   one transfer rather than two.
2. **Asynchronous transfers.** `cudaMemcpyAsync` (Chapter 6) requires pinned
   host memory. A pageable pointer cannot be used asynchronously because the
   DMA engine would need to chase OS page tables.

Pinned memory is not swappable. A large pinned allocation reduces the OS's
freedom and can degrade system performance. The usual policy is to pin buffers
on the streaming path and leave everything else pageable.

```cpp
// ---------------------------------------------------------------------------
// Pinned allocation. cudaMallocHost allocates host memory that the CUDA
// runtime has pinned. It must be freed with cudaFreeHost, not free/delete.
// ---------------------------------------------------------------------------
float* h_pinned = nullptr;
CHECK(cudaMallocHost((void**)&h_pinned, nBytes));   // pinned, DMA-able, non-swappable

float* h_pageable = new float[n];           // ordinary heap memory, swappable

// ... use ...

CHECK(cudaFreeHost(h_pinned));              // correct deallocation for pinned
delete[] h_pageable;                        // correct deallocation for heap
```

The pinned pages carry bookkeeping in the CUDA runtime. Freeing them with
`free()` or `delete` would corrupt the runtime's view of the allocation.
`cudaMallocHost` must be paired with `cudaFreeHost`; `cudaHostAlloc` also uses
`cudaFreeHost`. `cudaMalloc` is paired with `cudaFree`.

## 4.3 Transfer Bandwidth: The Numbers

As teaching figures for a PCIe Gen4 x16 link (about 25-32 GB/s effective) and a
modern GPU (about 1 TB/s HBM for the RTX family and 3.35 TB/s for the H100):

| Copy | Approximate effective bandwidth |
|---|---|
| Host pageable → device | 6-8 GB/s (staging hop dominates) |
| Host pinned → device | 20-25 GB/s (PCIe-limited) |
| Device → host pinned | 20-25 GB/s |
| Device → device | 1-3 TB/s (HBM) |

For \\(N\\) bytes and effective bandwidth \\(B\\), a transfer takes:

\\[ T = \frac{N}{B} \\]

Copying 1 GB host-to-device takes roughly 40 ms from pinned memory and 140 ms
from pageable memory at the bandwidths above. A kernel that processes that 1 GB
may run for 1 ms. Transfer time can exceed computation time by two orders of
magnitude, which is why streaming (Chapter 6) overlaps transfers with
computation instead of serialising them.

## 4.4 Measuring Your Own Transfer Time

A transfer should be measured as an average over repeated copies so that
one-time overheads amortise. The following host-only snippet (no CUDA kernel
required) compares pinned and pageable copies. It uses `std::chrono` because
CUDA events have not yet been introduced (Chapter 6).

```cpp
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

// Time (in milliseconds) one cudaMemcpy of 'bytes' bytes from host to device.
double timeHostToDeviceCopy(void* dst, const void* src, size_t bytes)
{
    const int reps = 100;   // repeat to amortise launch overhead
    // Warm up once so page tables and caches are not part of the timing.
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();          // ensure the last copy actually finished
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms / reps;                 // average per-copy time
}
```

The first copy touches the pages, populates TLB entries, and performs the
pin-on-demand work for pageable memory. Excluding it gives a steady-state
number, which describes sustained behaviour.

`cudaMemcpy` is synchronous in the sense that the data copy completes before the
call returns, so the last timed copy has already finished. The
`cudaDeviceSynchronize()` at the end is defensive: it also waits for any
asynchronous kernel work issued earlier, keeping the measurement clean.

## 4.5 Unified Memory: `cudaMallocManaged`

> **Primitive - unified memory (UM).** A single virtual address space shared by
> host and device. A pointer allocated with `cudaMallocManaged` can be
> dereferenced on the host and in kernels without explicit copies. The driver
> migrates pages on demand.

Unified memory changes the programming model: explicit `cudaMemcpy` calls are
not required. The driver decides when data moves, and its decisions can carry
hidden per-page costs.

```cpp
// One allocation, usable on both sides:
float* m = nullptr;
CHECK(cudaMallocManaged((void**)&m, nBytes));

// Host fills it like an ordinary array:
for (int i = 0; i < n; ++i) m[i] = static_cast<float>(i);

// Kernel reads it directly - no copy issued:
addVectors<<<blocksPerGrid, threadsPerBlock>>>(m, m, m, n);  // m = m + m
CHECK(cudaDeviceSynchronize());   // page faults migrate data on demand
```

The driver splits the allocation into pages. When the host touches a page, the
page is placed in host memory. When a kernel touches a page that is not on the
device, a device-side page fault migrates it from host memory over the bus.
The first touch of each page pays this migration cost; later accesses are
local.

Two functions make migration explicit:

```cpp
// Hint the driver to migrate the range [m, m + nBytes) to the device now,
// so the kernel does not pay page faults during execution:
CHECK(cudaMemPrefetchAsync(m, nBytes, 0 /* device 0 */));

// When the host needs the results back, prefetch to the host (cudaCpuDeviceId):
CHECK(cudaMemPrefetchAsync(m, nBytes, cudaCpuDeviceId));
```

`cudaMemPrefetchAsync` is the explicit form of what the driver otherwise does
lazily. A kernel that page-faults through 1 GB of unified memory pays a fault
per page, which can stall the kernel for milliseconds. Prefetching before the
kernel moves the cost out of kernel time.

Unified memory is useful when explicit copies are inconvenient, such as data
structures with complex pointer graphs, or when an allocation is too large for
device memory and can be streamed through with `cudaMemAdvise`. Its cost is
reduced deterministic control over data movement. A working rule is to
understand `cudaMemcpy` first, use unified memory where it fits the access
pattern, and measure both options.

## 4.6 Zero-Copy Host Memory

> **Primitive - zero-copy.** Host memory mapped into the device address space.
> Kernels access it directly over the bus; no explicit copy occurs. Each access
> pays bus latency, so zero-copy is useful only for small or rarely re-read
> data.

```cpp
float* h_mapped = nullptr;
// cudaHostAllocMapped: allocate pinned host memory AND map it into the device
// address space (zero-copy).
CHECK(cudaHostAlloc((void**)&h_mapped, nBytes, cudaHostAllocMapped));

// Obtain the device-side pointer for the same memory:
float* d_mapped = nullptr;
CHECK(cudaHostGetDevicePointer((void**)&d_mapped, h_mapped, 0 /* flags, must be 0 */));

// d_mapped can now be passed to kernels; the kernel's reads and writes go
// directly over PCIe to host memory.
```

Zero-copy is useful in two cases: data so small that a copy costs more than the
kernel, and data produced by a kernel that the host must observe immediately
without a copy-back. It is not useful for large data that is re-read, because
every access crosses the bus at full latency.

## 4.7 Choosing a Memory Kind

| Need | Tool | Rationale |
|---|---|---|
| One-time setup copy | `cudaMemcpy` (pageable) | Simplicity; the staging hop occurs once. |
| Streaming or repeated copies | `cudaMallocHost` pinned + `cudaMemcpy` (or async, Ch. 6) | Direct DMA, no staging hop. |
| Pointer-heavy structures | `cudaMallocManaged` + `cudaMemPrefetchAsync` | Driver migrates whole graphs. |
| Tiny, frequently read host data | Zero-copy `cudaHostAllocMapped` | No explicit copy; bus latency is acceptable for small data. |
| Same-GPU scratch space | `cudaMalloc` device memory | Full HBM bandwidth, no bus. |

A common failure mode is using `cudaMallocManaged` for every allocation because
it is convenient, then observing low bandwidth because lazy migration converts a
streaming copy into per-page faults. The memory kind is part of the algorithm
design, not a setup detail.

## 4.8 Transfer Overlap: A Preview

A transfer and a kernel that operate on different data can run concurrently if
the runtime can see that they are independent. The canonical shape is
**double buffering**:

![Double buffering: copies of the next chunk overlap kernels on the current chunk](../../assets/ch04_double_buffer.svg)

With two buffers, the copy for the next chunk overlaps computation on the
current chunk. Transfer cost disappears from the critical path, provided the
transfers are pinned and asynchronous. Chapter 6 builds this pipeline in full;
the capstone (Chapter 15) uses it for images.

## Memory Kinds and Their Cost Models

Memory allocation is not plumbing. Each memory kind embodies a different cost
model, and the cost model determines whether a program runs at memory speed or
at latency speed.

Pageable memory lives in ordinary OS pages that the kernel can swap or move at
any time. The GPU's DMA engine cannot safely touch those pages because their
physical addresses may change mid-transfer. The runtime therefore copies the
data into a pinned staging buffer and then DMA's from there. That extra copy
adds latency and consumes host memory bandwidth. Pinned memory
(`cudaMallocHost`) locks the physical pages in place so the DMA engine can
access them directly. This is not a micro-optimisation: it is the difference
between one transfer and two.

Unified memory (`cudaMallocManaged`) presents one virtual address usable by
both the CPU and GPU, and the driver migrates pages on demand. The cost hidden
by the API is the page fault: every migration crosses the bus, and a kernel
that touches a gigabyte of unified memory for the first time may pay a fault
per page. `cudaMemPrefetchAsync` moves those migrations out of the kernel's
critical path.

Zero-copy memory (`cudaHostAllocMapped`) maps host memory into the device
address space. A kernel can read or write it directly over the bus with no
explicit copy. Every access crosses the bus at PCIe latency, so zero-copy wins
only for data that is small, accessed rarely, or produced by the kernel for
immediate host consumption. For large, repeatedly read data, per-access bus
latency exceeds the cost of one bulk copy.

The unifying principle is that data movement is not free, and memory kinds move
data at different times: eagerly for copies, on demand for unified memory, and
on every access for zero-copy. Choosing the wrong kind for an access pattern
can change a kernel from bandwidth-bound to latency-bound.

## Common Pitfalls

- Using pageable memory for frequent transfers. Pin what is streamed.
- Using unified memory for everything because it is convenient. Lazy migration
  turns a streaming copy into per-page faults; prefetch explicitly.
- Mismatching allocation and free: `cudaMalloc` -> `cudaFree`,
  `cudaMallocHost` -> `cudaFreeHost`, `cudaHostAlloc` -> `cudaFreeHost`.
  Mismatched frees corrupt runtime bookkeeping.
- Using zero-copy for large, repeatedly read data. Every access crosses the
  bus; zero-copy is fast only for small, rarely re-read data.

## Check Your Understanding

<details>
<summary>Why can't cudaMemcpyAsync use pageable memory?</summary>

Asynchronous copies are performed by the DMA engine, which needs physical
pages that will not move. Pageable pages can be swapped, so the runtime would
have to stage through a pinned buffer synchronously, destroying the
asynchrony. Pinned memory guarantees stable physical pages.
</details>

<details>
<summary>What cost does unified memory hide?</summary>

Page faults and migrations. When the GPU touches a page resident on the host,
or the host touches a page resident on the device, the driver migrates the page
over the bus. The first touch of each page pays this cost.
`cudaMemPrefetchAsync` makes migrations explicit and removes them from kernel
time.
</details>

<details>
<summary>When is zero-copy the right choice?</summary>

When the data is small enough that a copy costs more than direct bus access, or
when the kernel produces data the host must see immediately. It is the wrong
choice for large or repeatedly read data, because every access crosses the bus
at full latency.
</details>

## Key Takeaways

- Transfers can cost 100x more than the kernel that uses the data; data movement is part of the computation.
- Pinned memory (`cudaMallocHost`) enables direct DMA and asynchronous transfers; pageable memory goes through a staging copy.
- Unified memory (`cudaMallocManaged`) hides the copy but migrates pages lazily; prefetch explicitly with `cudaMemPrefetchAsync`.
- Zero-copy (`cudaHostAllocMapped`) suits small or rarely re-read data; device memory suits hot data.
- Match every allocation with its paired free (`cudaFree`, `cudaFreeHost`) and measure bandwidth before optimising.

## 4.9 Exercises

1. A 4 GB dataset is copied host-to-device from pageable and from pinned
   memory. Using the bandwidths in §4.3, by how many milliseconds is the pinned
   copy faster?
2. Why can `cudaMemcpyAsync` not be used with pageable host memory? Trace the
   sequence of events the DMA engine would need.
3. A kernel reads each element of a 1 GB unified-memory array exactly once.
   Where would you place `cudaMemPrefetchAsync`, and why?
4. Zero-copy memory is described as fast for small, rarely re-read data. Using
   the latency numbers of Chapter 2, explain what happens if a kernel re-reads
   the same 1 MB zero-copy region 1,000 times.
