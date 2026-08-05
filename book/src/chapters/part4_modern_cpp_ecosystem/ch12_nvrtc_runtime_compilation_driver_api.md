# Chapter 12: NVRTC, Runtime Compilation & the Driver API

> *"The compiler is not always your toolchain's secret. Sometimes it is your
> program's input."*

Everything so far has used the **CUDA runtime API** — the `cudaMalloc`,
`cudaMemcpy`, `cudaStreamCreate` functions — and the offline toolchain
(`nvcc` → PTX → SASS, Chapter 3). This chapter opens the second door: the
**driver API**, which exposes the lower-level objects (contexts, modules,
kernels), and **NVRTC** (NVIDIA Runtime Compilation), which compiles CUDA
source *at run time*, inside your program. Together they enable JIT
compilation, user-supplied kernels, and code generated from runtime
parameters. The capstone uses exactly this machinery.

## 12.1 The Two APIs

> **Primitive — runtime API.** The high-level `cuda*` functions (Chapters
> 3–6). It initialises a context implicitly, manages device memory, streams,
> and launches kernels by *name* at compile time.
> **Primitive — driver API.** The low-level `cu*` functions. You create
> contexts explicitly, load *modules* (compiled kernels), extract kernel
> handles, and launch them with a raw parameter array.

The runtime API is implemented *on top of* the driver API. Everything you did
with `cudaMalloc` has a `cuMemAlloc` equivalent; every `kernel<<<>>>` is a
`cuLaunchKernel`. The runtime is more convenient; the driver is more explicit
and is the *only* API that can launch kernels that did not exist when your
program was compiled — the defining feature of this chapter.

## 12.2 PTX, cubin, and fatbin

The offline pipeline produced PTX and SASS (Chapter 3). The artefacts have
names:

> **Primitive — PTX.** The portable virtual ISA. Architecture-independent
> (within CUDA's versioning), compiled to SASS by the driver at load time.
> **Primitive — cubin.** A CUDA *binary*: SASS for one specific compute
> capability, produced by `ptxas`.
> **Primitive — fatbin.** A container bundling multiple cubins (and PTX) for
> different architectures, so one executable runs on many GPUs. When you
> compile with `nvcc -arch=sm_90`, the `.so`/`.exe` embeds a fatbin.

`nvcc` produces all of these; `cuobjdump` and `nvdisasm` inspect them. For
this chapter the important fact is that **PTX is text**: it can be generated,
examined, and even written by hand. NVRTC produces PTX at run time; the
driver loads it.

## 12.3 NVRTC: Compiling CUDA Source in Your Program

NVRTC compiles a CUDA source *string* to PTX at run time. The flow:

```cpp
#include <nvrtc.h>
#include <cuda.h>            // the driver API
#include <string>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Compile the given CUDA source to PTX using NVRTC.
// Returns the PTX as a string. Throws std::runtime_error on failure with
// the compiler's log (which is where your kernel's errors appear).
// ---------------------------------------------------------------------------
std::string compileToPtx(const char* source, const char* name)
{
    // 1. Create an NVRTC program from the source text.
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, name, 0, nullptr,
                                         nullptr);
    if (res != NVRTC_SUCCESS) throw std::runtime_error("nvrtcCreateProgram");

    // 2. Compile. Options are passed as strings, exactly like nvcc flags.
    const char* options[] = {"-arch=compute_90", "-std=c++17"};
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

**Why `-arch=compute_90`?** NVRTC compiles to PTX for a *virtual*
architecture. The driver later JIT-compiles that PTX to the *actual* SASS of
whatever GPU is present. Choosing `compute_90` targets Hopper-class GPUs; a
lower virtual arch (e.g. `compute_80`) produces more portable PTX at the cost
of potentially less optimal SASS.

**Why compile at run time at all?**

1. **User-supplied code.** The program can accept kernels as strings (the
   pattern behind JIT-based DSLs and "kernel playground" tools).
2. **Runtime-tuned code generation.** A solver can generate a kernel with
   loop unrolling and constants *specialised to the runtime problem size*,
   which a generic precompiled kernel cannot do.
3. **Deployment simplicity.** Ship PTX (portable) instead of fatbins per
   architecture; the driver JITs at first use.

The cost: compile time at run time (hundreds of milliseconds) and the
complexity of the two-stage load. That is why NVRTC belongs in *this* chapter
and not Chapter 3.

## 12.4 The Driver API: Loading and Launching

With PTX in hand, the driver API turns it into an executable kernel:

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

    // 2. Create a context on device 0 (the driver API has no implicit
    //    context — this is the "explicit" part of the driver API).
    CUdevice  device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // 3. Load the PTX into a MODULE: a collection of compiled kernels.
    CUmodule module;
    CUresult res = cuModuleLoadData(&module, ptx.c_str());
    if (res != CUDA_SUCCESS) throw std::runtime_error("cuModuleLoadData");

    // 4. Get a handle to the kernel by NAME. The kernel must exist in the
    //    PTX with that exact name (nvcc mangles C++ names; a plain
    //    __global__ function named addVectors is stored as "addVectors").
    CUfunction kernel;
    res = cuModuleGetFunction(&kernel, module, "addVectors");
    if (res != CUDA_SUCCESS) throw std::runtime_error("cuModuleGetFunction");

    // 5. Package the kernel arguments. The driver API takes a raw array of
    //    POINTERS TO the arguments — hence the address-of dance below.
    void* args[] = { &d_a, &d_b, &d_c, &n };

    // 6. Launch. gridDimX/Y/Z, blockDimX/Y/Z, sharedMemBytes, stream,
    //    kernel, args.
    res = cuLaunchKernel(kernel,
                         1024, 1, 1,        // grid: 1024 blocks
                         256,  1, 1,        // block: 256 threads
                         0, nullptr,        // no dynamic shared, default stream
                         args, nullptr);    // arguments, no extra options
    if (res != CUDA_SUCCESS) throw std::runtime_error("cuLaunchKernel");

    cuCtxSynchronize();

    // 6. The context and module live until the program exits (or until we
    //    destroy them). In a long-running process, destroy them explicitly.
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
```

**Why the `args` array of `void*`?** The driver API cannot know the kernel's
signature (the kernel did not exist at compile time). It must be told where
each argument lives: `args[i]` is a pointer to the *i-th argument's storage*.
For a pointer argument `d_a`, the storage is the pointer variable, hence
`&d_a`. Getting this wrong (passing `d_a` instead of `&d_a`) is the classic
driver-API crash; the driver dereferences `args[i]` and reads the wrong bytes
as the pointer value.

**Why the explicit context?** The runtime API creates a context lazily and
manages it for you. The driver API exposes the context so you can control
resource lifetime, host multiple contexts (rarely wise), and interoperate
with libraries. The price is the ceremony above.

## 12.5 The JIT Cache: Making Runtime Compilation Cheap

First use of a module pays the JIT compile (PTX → SASS). Subsequent loads of
the *same* PTX in the same process reuse the driver's in-memory cache, and
across processes the driver persists compiled binaries in
`~/.nv/ComputeCache`. You control the cache with environment variables:

```bash
export CUDA_CACHE_MAXSIZE=1073741824   # 1 GB on-disk cache
export CUDA_CACHE_DISABLE=0            # 0 = enabled
```

**The reasoning.** NVRTC compiles source → PTX (your cost, in-process);
the driver compiles PTX → SASS (cached). For a long-running server, the
correct pattern is: compile once at startup, keep the module alive for the
process lifetime, never recompile per request.

## 12.6 Runtime API + Driver API: The Hybrid

The two APIs can coexist in one program: the runtime manages memory and
streams; the driver launches the JIT-compiled kernel. The bridge is the
**current context**: the runtime's implicit context is also the driver's
current context, so device pointers obtained from `cudaMalloc` are valid for
`cuLaunchKernel` in the same thread. This hybrid — `cudaMalloc` for memory,
NVRTC+driver for the kernel — is the pragmatic sweet spot, and it is the shape
the capstone uses in Chapter 15.

## 12.7 Exercises

1. List the three artefacts produced by the offline toolchain and where each
   is compiled (host toolchain, `ptxas`, or the driver at load time).
2. In `launchFromPtx`, why is the argument for a `float*` parameter
   `&d_a` and not `d_a`? What exactly does the driver read from `args[i]`?
3. A server receives kernel source from users. Argue for or against caching
   compiled PTX keyed by a hash of the source, and name the two caches
   involved.
4. When would you choose `-arch=compute_80` over `-arch=compute_90` for
   NVRTC, and what does each choice cost?
