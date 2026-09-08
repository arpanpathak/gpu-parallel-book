# Chapter 12: NVRTC, Runtime Compilation & the Driver API

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch12_nvrtc/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch12_nvrtc)
> in the repository.

Everything so far has used the **CUDA runtime API** - `cudaMalloc`,
`cudaMemcpy`, and `cudaStreamCreate` - and the offline toolchain (`nvcc` to PTX
to SASS, Chapter 3). This chapter introduces the **driver API**, which exposes
lower-level objects (contexts, modules, and kernels), and **NVRTC** (NVIDIA
Runtime Compilation), which compiles CUDA source at run time inside a program.
Together they enable JIT compilation, user-supplied kernels, and code generated
from runtime parameters.

## 12.1 The Two APIs

> **Primitive - runtime API.** The high-level `cuda*` functions used in
> Chapters 3-6. It initialises a context implicitly, manages device memory and
> streams, and launches kernels by name at compile time.
> **Primitive - driver API.** The low-level `cu*` functions. The program
> creates contexts explicitly, loads modules of compiled kernels, extracts
> kernel handles, and launches them with a raw parameter array.

The runtime API is implemented on top of the driver API. Every `cudaMalloc`
has a `cuMemAlloc` equivalent, and every `kernel<<<>>>` launch corresponds to a
`cuLaunchKernel`. The runtime is more convenient. The driver is more explicit,
and it is the only API that can launch kernels that did not exist when the
program was compiled.

## 12.2 PTX, cubin, and fatbin

The offline pipeline produces PTX and SASS (Chapter 3). The artefacts have
names:

> **Primitive - PTX.** The portable virtual ISA. It is architecture-independent
> within CUDA's versioning and is compiled to SASS by the driver at load time.
> **Primitive - cubin.** A CUDA binary: SASS for one specific compute
> capability, produced by `ptxas`.
> **Primitive - fatbin.** A container that bundles multiple cubins and PTX for
> different architectures so that one executable runs on many GPUs. When code
> is compiled with `nvcc -arch=sm_90`, the host binary embeds a fatbin.

`nvcc` produces all of these; `cuobjdump` and `nvdisasm` inspect them. The
important property for this chapter is that **PTX is text**: it can be
generated, examined, and even written by hand. NVRTC produces PTX at run time;
the driver loads it.

## 12.3 NVRTC: Compiling CUDA Source in a Program

NVRTC compiles a CUDA source string to PTX at run time:

```cpp
#include <nvrtc.h>
#include <cuda.h>            // the driver API
#include <string>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Compile the given CUDA source to PTX using NVRTC.
// Returns the PTX as a string. Throws std::runtime_error on failure with
// the compiler's log (which is where the kernel's errors appear).
// ---------------------------------------------------------------------------
std::string compileToPtx(const char* source, const char* name)
{
    // 1. Create an NVRTC program from the source text.
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, name, 0, nullptr,
                                         nullptr);
    if (res != NVRTC_SUCCESS) throw std::runtime_error("nvrtcCreateProgram");

    // 2. Compile. Options are passed as strings, exactly like nvcc flags.
    const char* options[] = {"-arch=compute_60", "-std=c++17"};
    res = nvrtcCompileProgram(prog, 2, options);

    // 3. On failure, fetch the compilation log and report it.
    if (res != NVRTC_SUCCESS)
    {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error(std::string("NVRTC compile failed:\n") +
                                 log.data());
    }

    // 4. Fetch the PTX text.
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    return std::string(ptx.data(), ptxSize);
}
```

The `-arch=compute_60` option makes NVRTC compile for a virtual architecture.
The driver later JIT-compiles the PTX to the actual SASS of the installed GPU.
Pascal-class `compute_60` PTX runs on every CUDA 12.x GPU through the driver's
forward-compatible JIT. A higher virtual architecture such as `compute_90`
targets Hopper-class GPUs and may produce better SASS there, at the cost of
portability.

Runtime compilation is useful for three reasons:

1. **User-supplied code.** The program accepts kernels as strings, the pattern
   behind JIT-based DSLs and kernel playground tools.
2. **Runtime-specialised code generation.** A solver can generate a kernel
   with unrolling and constants specialised to the runtime problem size, which
   a generic precompiled kernel cannot do.
3. **Deployment simplicity.** PTX can be shipped instead of per-architecture
   fatbins; the driver JIT-compiles at first use.

The cost is compile time at run time, typically hundreds of milliseconds, plus
the complexity of the two-stage load.

## 12.4 The Driver API: Loading and Launching

Given PTX, the driver API turns it into an executable kernel:

```cpp
// ---------------------------------------------------------------------------
// Load PTX text into the current driver context and launch a kernel
// "addVectors" with the given grid/block shape and raw parameters.
// ---------------------------------------------------------------------------
void launchFromPtx(const std::string& ptx,
                   const float* d_a, const float* d_b, float* d_c, int n)
{
    // 1. Initialise the driver API (idempotent).
    cuInit(0);

    // 2. Create a context on device 0. The driver API has no implicit context;
    //    this is the "explicit" part of the driver API. Production code that
    //    uses the runtime API should use the primary/current context instead of
    //    creating a second one (see 12.6).
    CUdevice  device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // 3. Load the PTX into a MODULE: a collection of compiled kernels.
    CUmodule module;
    CUresult res = cuModuleLoadData(&module, ptx.c_str());
    if (res != CUDA_SUCCESS) throw std::runtime_error("cuModuleLoadData");

    // 4. Get a handle to the kernel by NAME. The kernel must exist in the
    //    PTX with that exact name. A plain __global__ function named
    //    addVectors is stored as "addVectors" (C++ name mangling applies only
    //    to non-C-linkage functions).
    CUfunction kernel;
    res = cuModuleGetFunction(&kernel, module, "addVectors");
    if (res != CUDA_SUCCESS) throw std::runtime_error("cuModuleGetFunction");

    // 5. Package the kernel arguments. The driver API takes a raw array of
    //    POINTERS TO the arguments - hence the address-of dance below.
    void* args[] = { &d_a, &d_b, &d_c, &n };

    // 6. Launch. Grid and block dimensions, sharedMemBytes, stream, kernel,
    //    and args. The grid must cover all n elements.
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    res = cuLaunchKernel(kernel,
                         blocks, 1, 1,        // grid
                         threads, 1, 1,       // block
                         0, nullptr,          // no dynamic shared, default stream
                         args, nullptr);      // arguments, no extra options
    if (res != CUDA_SUCCESS) throw std::runtime_error("cuLaunchKernel");

    cuCtxSynchronize();

    // 7. Destroy the module and context in a long-running process. For a
    //    kernel compiled once and replayed many times, keep both alive.
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
```

The `args` array is a `void**` where each element points to the storage of one
kernel argument. For a `float*` parameter `d_a`, the storage is the pointer
variable, so the code passes `&d_a`. Passing `d_a` directly is the classic
driver-API crash: the driver reads `args[i]`, interprets the pointer value as
the argument's bytes, and loads the wrong data.

The driver API requires explicit context management because the runtime's
implicit context is hidden. Contexts control resource lifetime and allow
interoperation with libraries; the price is the ceremony above.

## 12.5 The JIT Cache

The first use of a module pays the PTX-to-SASS JIT compile. Subsequent loads of
the same PTX in the same process reuse the driver's in-memory cache. Across
processes, the driver persists compiled binaries in `~/.nv/ComputeCache`. The
cache can be controlled with environment variables:

```bash
export CUDA_CACHE_MAXSIZE=1073741824   # 1 GB on-disk cache
export CUDA_CACHE_DISABLE=0            # 0 = enabled
```

NVRTC compiles source to PTX in the process. The driver compiles PTX to SASS
and caches that result. A long-running server should compile once at startup,
keep the module alive for the process lifetime, and never recompile per
request.

## 12.6 Runtime API + Driver API: The Hybrid

The two APIs can coexist. The runtime manages memory and streams; the driver
launches JIT-compiled kernels. The bridge is the **current context**: the
runtime's implicit context is also the driver's current context, so device
pointers obtained from `cudaMalloc` remain valid for `cuLaunchKernel` in the
same thread. This hybrid - `cudaMalloc` for memory, NVRTC plus driver for the
kernel - is the pattern used by the capstone in Chapter 15.

## What the Runtime API Automates

The runtime API is convenient because it makes decisions implicitly: it creates
a context lazily, loads modules, manages memory, and hides launch plumbing. The
driver API exposes those decisions. Contexts are explicit objects. Modules are
containers of compiled kernels loaded from PTX or cubin data. Kernel arguments
are not type-checked by the compiler; they are packaged into a raw array of
pointers and handed to `cuLaunchKernel`.

This exposure is necessary for code that did not exist at compile time. NVRTC
takes CUDA source as a string, compiles it to PTX, and returns the text. The
driver loads that PTX, JIT-compiles it to SASS for the installed GPU, and
returns a function handle. PTX is architecture-independent, so the same text
can become SASS for a T4, an A100, or an H100.

The most common driver-API bug also illustrates the model. `cuLaunchKernel`
receives `void** args`, an array where each element is a pointer to the storage
of one kernel argument. For a `float*` parameter `d_a`, the storage is the
local pointer variable, so the code passes `&d_a`. If `d_a` is passed instead,
the driver interprets the pointer value itself as the argument's bytes and
typically crashes. The driver API does not know the kernel signature, so the
program must describe where every argument lives. The type safety provided by
the runtime API is replaced here by a precise convention.

The hybrid pattern is the pragmatic synthesis: runtime API for memory and
streams, driver API for the kernel. Because both share the current context,
`cudaMalloc` pointers remain valid when passed to `cuLaunchKernel`.

## Common Pitfalls

- Passing `d_a` instead of `&d_a` in the `args` array. The driver treats each
  `args[i]` as a pointer to the argument's storage; passing the pointer value
  directly makes it read the pointer's bytes as the argument.
- Ignoring the NVRTC compile log. A failed compile reports the problem, but
  only if the log is fetched and printed.
- Creating a second context when the runtime already has one. Use the current
  or primary context; a separate context can break pointer validity.
- Forgetting that PTX for `compute_90` does not run on older GPUs. Choose a
  virtual architecture that matches the deployment range.

## Check Your Understanding

<details>
<summary>Why must args[i] be the address of the argument?</summary>

`cuLaunchKernel` receives a `void**` in which each element is a pointer to the
argument's storage. For a `float*` parameter `d_a`, the storage is the local
pointer variable, so the driver needs `&d_a`. Passing `d_a` makes the driver
read the pointer value as the argument's bytes, which loads wrong data and
typically crashes.
</details>

<details>
<summary>What is the difference between PTX and cubin?</summary>

PTX is portable virtual ISA text. It is architecture-independent and can be
JIT-compiled by the driver for the installed GPU. A cubin is SASS for one
specific compute capability and cannot run on another architecture without
recompilation.
</details>

<details>
<summary>Why is the hybrid (runtime memory + driver kernel) pattern useful?</summary>

It keeps the convenient runtime API for allocations and copies while using the
driver API for the one thing the runtime cannot do: launching a kernel compiled
at run time. Both share the same current context, so device pointers remain
valid across the boundary.
</details>

## Key Takeaways

- The runtime API (`cuda*`) is built on the driver API (`cu*`); the driver is the only way to launch kernels that did not exist at compile time.
- PTX is portable text; cubin is SASS for one architecture; fatbin bundles several.
- NVRTC compiles a source string to PTX at run time; errors appear in the compilation log.
- cuLaunchKernel takes a raw array of pointers-to-arguments; passing the pointer instead of its address is the classic crash.
- The JIT cache (`CUDA_CACHE_MAXSIZE`) and module reuse make runtime compilation production-viable.

## 12.7 Exercises

1. List the three artefacts produced by the offline toolchain and where each is
   compiled: host toolchain, `ptxas`, or the driver at load time.
2. In `launchFromPtx`, why is the argument for a `float*` parameter `&d_a` and
   not `d_a`? What exactly does the driver read from `args[i]`?
3. A server receives kernel source from users. Argue for or against caching
   compiled PTX keyed by a hash of the source, and name the two caches
   involved.
4. When would you choose `-arch=compute_60` over `-arch=compute_90` for NVRTC,
   and what does each choice cost?

## Sources and Further Reading

- NVIDIA, *CUDA Runtime API* and *Driver API* reference: <https://docs.nvidia.com/cuda/cuda-runtime-api/> and <https://docs.nvidia.com/cuda/cuda-driver-api/>
- NVIDIA, *NVRTC User Guide*: <https://docs.nvidia.com/cuda/nvrtc/>
- NVIDIA, *CUDA C++ Programming Guide*, "Just-in-Time Compilation" and "Compute Capabilities": <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
