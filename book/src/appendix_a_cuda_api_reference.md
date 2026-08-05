# Appendix A — CUDA API Reference

> *"Every primitive of the CUDA programming model, gathered in one place. When
> a chapter says 'the primitive', this is the definition it means."*

This appendix is the book's vocabulary list: every type, built-in variable,
function and flag used in the main text, with its meaning and where it is
discussed. It is intended as a lookup table, not as a tutorial — the tutorials
are the chapters.

## A.1 Execution Configuration

| Syntax | Meaning | Chapter |
|---|---|---|
| `kernel<<<gridDim, blockDim>>>(args...)` | Launch `kernel` with a grid of `gridDim` blocks of `blockDim` threads each | 3 |
| `kernel<<<gridDim, blockDim, sharedBytes, stream>>>` | As above, with dynamic shared memory (bytes) and an explicit stream | 6, 7 |
| `dim3` | Three-`unsigned` vector type; fields `.x`, `.y`, `.z` | 3 |
| `threadIdx` | The thread's position within its block (a `dim3`) | 3 |
| `blockIdx` | The block's position within the grid (a `dim3`) | 3 |
| `blockDim` | Threads per block, as launched (a `dim3`) | 3 |
| `gridDim` | Blocks per grid, as launched (a `dim3`) | 3 |
| `__launch_bounds__(maxThreads, minBlocks)` | Compiler directive: register budget for occupancy | 9 |

The universal 1-D global index: `blockIdx.x * blockDim.x + threadIdx.x`.
The 2-D composition is in Chapter 3, §3.5.

## A.2 Function Qualifiers

| Qualifier | Runs on | Called from | Chapter |
|---|---|---|---|
| `__global__` | Device | Host | 3 |
| `__device__` | Device | Device only | 3 |
| `__host__` (default) | Host | Host | 3 |
| `__host__ __device__` | Both | Both | 10 |

`extern "C" __global__` keeps the kernel symbol unmangled for driver-API /
NVRTC lookup (Chapters 12, 13).

## A.3 Vector Types

| Type | Bytes | Alignment | Notes |
|---|---|---|---|
| `uchar3` | 3 | 1 | RGB pixels; no padding (Chapter 15) |
| `float2`, `float4` | 8, 16 | 4, 16 | Vectorised loads (Chapter 7, §7.6) |
| `int2`, `int4`, `double2`, `uint4` | 8/16/16/16 | as size | Same vectorisation rules |
| `size_t` | platform | — | Byte sizes; use instead of `int` (Chapter 3) |

Vectorised accesses require alignment to the vector size.

## A.4 Memory Management

| Function | Behaviour | Chapter |
|---|---|---|
| `cudaMalloc(void** p, size_t n)` | Allocate `n` bytes in device global memory | 3 |
| `cudaFree(void* p)` | Free a device allocation | 3 |
| `cudaMemcpy(dst, src, n, kind)` | Synchronous copy; `kind` = `HostToDevice`, `DeviceToHost`, `DeviceToDevice`, `HostToHost` | 3 |
| `cudaMemcpyAsync(dst, src, n, kind, stream)` | Asynchronous copy, queued on `stream`; **requires pinned host memory** | 4, 6 |
| `cudaMallocHost(void** p, size_t n)` | Allocate pinned (page-locked) host memory | 4 |
| `cudaHostAlloc(void** p, size_t n, flags)` | Pinned host memory; `cudaHostAllocMapped` adds zero-copy mapping | 4 |
| `cudaHostGetDevicePointer(void** dp, void* hp, 0)` | Device pointer for zero-copy mapped host memory | 4 |
| `cudaFreeHost(void* p)` | Free pinned host memory | 4 |
| `cudaMallocManaged(void** p, size_t n)` | Unified memory (host + device address space) | 4 |
| `cudaMemPrefetchAsync(p, n, device, stream)` | Migrate unified-memory pages now | 4 |
| `cudaMemcpyToSymbol(sym, src, n)` | Copy into `__constant__` memory | 7 |

Memory kinds and when to use each: Chapter 4, §4.7.

## A.5 Synchronisation and Memory Ordering

| Primitive | Meaning | Chapter |
|---|---|---|
| `__syncthreads()` | Block-wide barrier; **must be uniformly reachable** | 5 |
| `__threadfence()` | Order my device-scope global accesses | 5 |
| `__threadfence_block()` | Order my block-scope accesses | 5 |
| `__threadfence_system()` | Order host+device accesses | 5 |
| `volatile` | Disable register caching of a location | 5 |
| `atomicAdd/Sub/Exch/CAS/Min/Max/And/Or/Xor` | Hardware read-modify-write; return old value | 5 |
| `cuda::atomic<T, scope>` | C++20-style atomic with memory orders (CUDA 12) | 10 |

The atomics table with semantics: Chapter 5, §5.5.

## A.6 Streams and Events

| Function | Behaviour | Chapter |
|---|---|---|
| `cudaStreamCreate(&s)` | Create a stream | 6 |
| `cudaStreamDestroy(s)` | Destroy a stream | 6 |
| `cudaStreamCreateWithFlags(&s, flag)` | `cudaStreamNonBlocking` disables default-stream sync | 6 |
| `cudaStreamCreateWithPriority(&s, flag, prio)` | Priority stream; range from `cudaDeviceGetStreamPriorityRange` | 6 |
| `cudaStreamSynchronize(s)` | Wait for all work in `s` | 6 |
| `cudaEventCreate(&e)` / `cudaEventDestroy(e)` | Create/destroy an event | 6 |
| `cudaEventRecord(e, s)` | Mark the stream position | 6 |
| `cudaEventSynchronize(e)` | Wait until the device reaches `e` | 6 |
| `cudaEventElapsedTime(&ms, e0, e1)` | Time between two events | 6, 16 |
| `cudaStreamWaitEvent(s, e)` | Make `s` wait for `e` (cross-stream dependency) | 6 |
| `cudaGraphCreate/Instantiate/Launch/Destroy` | Capture and replay device work | 6 |
| `cudaDeviceSynchronize()` | Wait for all device work | 3, 6 |

The synchronisation cheat sheet: Chapter 6, §6.8.

## A.7 Error Handling

| Primitive | Behaviour | Chapter |
|---|---|---|
| `cudaError_t` | Enum; `cudaSuccess == 0`, everything else is an error | 3 |
| `cudaGetLastError()` | Return and clear the last asynchronous error | 3 |
| `cudaGetErrorString(e)` | Human-readable error text | 3 |
| `cudaDeviceSynchronize()` | Also surfaces async kernel errors | 3, 6 |
| `cublasStatus_t`, `nvrtcResult`, `CUresult` | Library error types (cuBLAS, NVRTC, driver API) | 11, 12 |

Two error modes (synchronous vs asynchronous) and the `CHECK` discipline:
Chapter 3, §3.8.

## A.8 Device Math Library (selected)

`fminf`, `fmaxf`, `sqrtf`, `fabsf`, `sinf`, `cosf`, `expf`, `logf`,
`powf`, `fmaf` (fused multiply-add). The `__host__ __device__` versions work
on both sides (Chapter 10, §10.4). Fast approximations live in the `__`
prefixed forms (`__expf`, `__sinf`); enable globally with
`--use_fast_math` — but measure before trusting (Chapter 16, §16.7).

## A.9 Warp-Level Primitives

| Primitive | Meaning | Chapter |
|---|---|---|
| `__shfl_down_sync(mask, val, delta)` | Move `val` from lane `lane+delta` to `lane` | 8 |
| `__shfl_sync`, `__shfl_up_sync`, `__shfl_xor_sync` | Other shuffle directions | 8 |
| `0xffffffffu` | The 32-lane mask for `_sync` primitives | 8 |
| `__activemask()` | Mask of currently active lanes (use with care) | 8 |

All `_sync` primitives require all named lanes to execute them.

## A.10 Driver API and NVRTC (Chapter 12)

| Primitive | Meaning |
|---|---|
| `cuInit`, `cuDeviceGet`, `cuCtxCreate`, `cuCtxDestroy` | Context lifecycle |
| `cuModuleLoadData`, `cuModuleGetFunction`, `cuModuleUnload` | Load PTX/cubin, fetch kernel by name |
| `cuLaunchKernel(f, gx,gy,gz, bx,by,bz, smem, stream, args, extra)` | Raw launch with `void*` argument array |
| `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTX`, `nvrtcGetProgramLog` | Runtime compilation of CUDA source |
| PTX / cubin / fatbin | Portable ISA / SASS binary / multi-arch container |

## A.11 Tools and Environment (Chapter 16)

| Tool | Purpose |
|---|---|
| `nvcc` | Offline compiler (`-arch=sm_90`, `-ptx`) |
| `nsys profile` | System-level timeline |
| `ncu --set full` | Kernel-level counters |
| `compute-sanitizer --tool memcheck/racecheck/initcheck/synccheck` | Runtime error detection |
| `cuda-gdb` | Interactive device debugger |
| `clock64()` | In-kernel cycle counter |
| `CUDA_CACHE_MAXSIZE` | JIT cache size (Chapter 12) |
