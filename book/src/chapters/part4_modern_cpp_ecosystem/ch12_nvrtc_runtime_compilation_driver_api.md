<img class="plate" src="../../art/ch12.svg" alt="Hue, a chameleon, grips a branch at dusk. On the left hangs a sheet of source code that the program has just written: float p, then lines of p = p * v plus a constant. Hue's body changes colour from tail to head, blue like the paper's ink at the tail, yellow at the head. Its tongue reaches out to a chip marked sm_87, the GPU the code is compiled for. A tag says written, compiled and loaded while the program runs.">

# Chapter 12: NVRTC, Runtime Compilation & the Driver API

Every kernel in the book so far existed before its program started. `nvcc` compiled it together
with the host code, placed the machine code inside the executable, and the CUDA runtime found it
there when the program called it with `<<<...>>>`. The values a kernel worked on arrived at run
time, but the instructions that worked on them were fixed when the program was built.

Some programs cannot work that way. A deep-learning framework fuses several operations into 1
kernel only after it has seen the shapes of the tensors. A simulation reads its coefficients
from a file. A tool lets its users type a kernel. In each case the best kernel, or the only
possible one, is known only while the program runs. This chapter shows how a program writes the
source of a kernel as text, compiles it with **NVRTC**, the CUDA runtime compilation library, and
loads and launches the result through the **driver API**, the lower layer on which the CUDA
runtime itself is built.

We follow 1 example from start to finish: a polynomial of degree 32 evaluated at 16,777,216
points. A generic kernel reads its 33 coefficients from memory. A specialised kernel, written by
the program after it knows the coefficients, contains them as constants. Both give bit-identical
results; on a Jetson Orin NX the specialised one runs about 1.4 times as fast, and compiling it
costs about 50 ms. The chapter measures both sides of that trade.

<div class="covers" markdown="1">

This chapter covers

- what the runtime API does for us, and the driver API objects underneath it: context, module,
  function
- PTX, cubin and fatbin, and which code runs on which GPU
- Horner's rule, and a kernel specialised to its coefficients
- NVRTC: compiling CUDA C++ from a string, for the GPU the program finds, with its error log
- loading a module, finding a kernel, and launching it with an array of argument addresses
- the same steps in Rust, where cuda-core is a driver-API program
- what runtime compilation costs, what it buys, and the JIT cache that saves the cost next time
- sharing 1 context between the runtime API and the driver API

</div>

The code of the chapter is in
[`code/ch12_nvrtc/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch12_nvrtc)
for C++ and in
[`code/oxide/ch12_driver/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch12_driver)
for Rust.

## 12.1 Two ways to talk to the GPU

### 12.1.1 What the runtime API does for us

A **kernel** is a function that runs on the GPU, once per thread of a launch (Chapter 3). The
functions we have used to allocate memory, copy data and launch kernels, `cudaMalloc`,
`cudaMemcpy` and `<<<...>>>`, belong to the **CUDA runtime API**, a library called `libcudart`
whose functions begin with `cuda`. It does 3 jobs for us without being asked.

1. **It creates a context.** A **context** is the GPU's equivalent of a process: it owns the
   memory a program has allocated on 1 GPU, the kernels it has loaded, and its streams. The
   runtime creates 1 context per GPU the first time a program needs it, and every runtime call
   then uses it.
2. **It registers our kernels.** `nvcc` adds code to the executable that runs before `main`. It
   hands the runtime the GPU machine code of every kernel in the file, together with the host
   function that stands for each kernel. When the program launches `polyGeneric`, the runtime
   already knows which machine code that name means.
3. **It packs the arguments.** In `polyGeneric<<<blocks, threads>>>(X, Y, n, C, degree)` the
   compiler knows the kernel's parameter list. It checks the types of the 5 arguments, and
   generates the code that copies their values into the launch.

All 3 jobs depend on the kernel being known when the program is compiled. A kernel written while
the program runs has no registration and no parameter list the compiler could check, so the
runtime API has no way to launch it.

### 12.1.2 The driver API

The runtime API is built on a lower library, the **CUDA driver API**, `libcuda`, whose functions
begin with `cu`. `libcuda` is part of the GPU driver, not of the CUDA toolkit, which is why every
machine with an NVIDIA driver has it. The driver API exposes the objects that the runtime API
manages for us:

| Object | What it is | Created by |
|---|---|---|
| device (`CUdevice`) | 1 GPU, by number | `cuDeviceGet` |
| context (`CUcontext`) | the memory, kernels and streams of 1 program on 1 GPU | `cuDevicePrimaryCtxRetain` |
| module (`CUmodule`) | 1 loaded piece of GPU code, holding 1 or more kernels | `cuModuleLoadData` |
| function (`CUfunction`) | 1 kernel inside a module, found by its name | `cuModuleGetFunction` |

A program that holds a function can launch it with `cuLaunchKernel`, whichever way its code was
produced. Figure 12.1 shows where each library sits. The runtime API, NVRTC and the Rust crate
cuda-core all arrive at the same driver API.

<figure>
<img src="../../figures/ch12-layers.svg" alt="4 rows of cards. Top: 3 kinds of program, C++ with triple angle brackets, C++ built at run time, and Rust with cuda-oxide. Below them: the runtime API, which creates the context on the first call, registers kernels and packs arguments; NVRTC, which turns source text into PTX or a cubin; and cuda-core. All 3 sit on the driver API in libcuda, which reaches the kernel-mode driver and the GPU.">
<figcaption><b>Figure 12.1</b> 3 ways into the GPU, and the driver API under all of them.</figcaption>
</figure>

Most runtime calls have a driver call underneath them:

| Runtime API | Driver API |
|---|---|
| first call creates the context | `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain` |
| `cudaMalloc`, `cudaFree` | `cuMemAlloc`, `cuMemFree` |
| `cudaMemcpy` | `cuMemcpyHtoD`, `cuMemcpyDtoH` |
| kernels registered before `main` | `cuModuleLoadData`, `cuModuleGetFunction` |
| `kernel<<<grid, block>>>(args...)` | `cuLaunchKernel(function, grid, block, ..., args)` |

### 12.1.3 cuda-core is a driver-API program

The Rust chapters have used the driver API from the start. cuda-oxide's host crate, cuda-core,
calls `libcuda` directly; it has no runtime API underneath. `CudaContext::new(0)` calls `cuInit`,
`cuDeviceGet` and `cuDevicePrimaryCtxRetain`, and `kernels::load(&ctx)` passes the GPU code that
cargo-oxide embedded in the executable to `cuModuleLoadData`. A program that compiles a kernel at
run time therefore needs no new crate in Rust: it hands its own code to the same functions.

<div class="bee bee-happy" markdown="1">

The runtime API is a convenience layer over the driver API. A kernel that did not exist when the
program was compiled can only be launched through the driver API.

</div>

## 12.2 PTX, cubin and fatbin

### 12.2.1 Virtual and real architectures

Every generation of NVIDIA GPU has its own machine instructions. The **compute capability** names
a generation and its features with 2 numbers: 8.7 for the Jetson Orin NX, 9.0 for the H100. A
compiler that targets 1 compute capability writes it as `sm_87` or `sm_90`. Code compiled for
`sm_90` does not run on an `sm_87` GPU, because some of its instructions do not exist there.

NVIDIA therefore defines a second, portable form. **PTX** is the instruction set of a *virtual*
GPU, written as text. It names a virtual architecture, such as `compute_87`, which fixes the
features the code may use, but not the machine instructions. A program carrying PTX can run on a
GPU that did not exist when it was compiled: the GPU driver translates the PTX into that GPU's
machine instructions when the program loads it. Translating code at the moment it is needed is
called **just-in-time (JIT) compilation**. This book never prints PTX; we only measure its size.

The real machine code comes in 2 containers:

- A **cubin**, CUDA binary, holds machine code for 1 real architecture, such as `sm_87`.
- A **fatbin**, fat binary, bundles several cubins and PTX, so that 1 executable runs on many
  GPUs. `nvcc` embeds a fatbin in every executable it builds, and the driver picks the cubin that
  matches the GPU, or JIT-compiles the PTX when none matches.

Figure 12.2 shows the 3 forms twice: made by `nvcc` before the program runs, and made by NVRTC
while it runs. The sizes are those of the chapter's polynomial on the Orin NX.

<figure>
<img src="../../figures/ch12-pipeline.svg" alt="Two zones. At build time, nvcc turns a .cu source file into PTX for a virtual GPU, compute_87, then into a cubin of machine code for sm_87 only, and bundles cubins and PTX into a fatbin inside the executable. While the program runs, NVRTC turns a source string of 1,315 bytes into PTX of 2,237 bytes in 44 to 88 ms or a cubin of 3,624 bytes in 49 to 74 ms. Loading the PTX takes 0.7 to 45 ms, depending on the driver's cache; loading the cubin takes 0.05 ms.">
<figcaption><b>Figure 12.2</b> Source, PTX and cubin, made before the program starts or while it runs.</figcaption>
</figure>

### 12.2.2 Which code runs on which GPU

2 rules from NVIDIA's programming guide decide whether a piece of code can run on a GPU.

- A cubin built for compute capability X.y runs on a GPU of capability X.z only when z ≥ y: the
  same major generation, and the same or a later minor version.
- PTX for `compute_X.y` can be JIT-compiled for any GPU of capability X.y or higher, including
  later generations.

| GPU | Compute capability | Real architecture | Runs a cubin for `sm_80`? | Runs PTX for `compute_80`? |
|---|---|---|---|---|
| A100 | 8.0 | `sm_80` | yes | yes |
| Jetson Orin NX | 8.7 | `sm_87` | yes, 8.7 ≥ 8.0 | yes |
| GeForce RTX 4090 | 8.9 | `sm_89` | yes, 8.9 ≥ 8.0 | yes |
| H100 | 9.0 | `sm_90` | no, major 9 ≠ 8 | yes |
| Jetson AGX Thor | 11.0 | `sm_110` | no | yes |
| GeForce RTX 5090 | 12.0 | `sm_120` | no | yes |
| DGX Spark (GB10) | 12.1 | `sm_121` | no | yes |

A program that compiles at run time avoids the question. It asks the driver which GPU it is on
and compiles for exactly that GPU.

<div class="bee bee-happy" markdown="1">

PTX is portable and must be JIT-compiled; a cubin is ready to run, but only on its own generation.

</div>

## 12.3 A kernel that depends on run-time values

### 12.3.1 Horner's rule

A polynomial of degree \\(d\\) has \\(d + 1\\) coefficients:

\\[ p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_d x^d \\]

Computing every power of \\(x\\) separately wastes multiplications. **Horner's rule** factors the
polynomial so that each coefficient costs 1 multiplication and 1 addition:

\\[ p(x) = c_0 + x\,(c_1 + x\,(c_2 + \dots + x\,c_d)) \\]

It starts with the highest coefficient, and at each step multiplies the running value by \\(x\\)
and adds the next coefficient down. Figure 12.3 works a polynomial of degree 2 by hand.

<div class="example" markdown="1">

Worked example: \\(p(x) = 1 + x/2 + x^2/3\\) at \\(x = 0.5\\)

1. The coefficients are \\(c_0 = 1\\), \\(c_1 = 1/2\\), \\(c_2 = 1/3\\). Start with
   \\(p = c_2 = 0.3333\\).
2. \\(p = 0.3333 \times 0.5 + 0.5 = 0.6667\\).
3. \\(p = 0.6667 \times 0.5 + 1 = 1.3333\\).
4. Check term by term: \\(1 + 0.5/2 + 0.25/3 = 1 + 0.25 + 0.0833 = 1.3333\\).

</div>

<figure>
<img src="../../figures/ch12-horner.svg" alt="Horner's rule for p of x = 1 + x over 2 + x squared over 3 at x = 0.5, in 3 rows. Start: p = c[2] = 1/3 = 0.3333. k = 1: p = 0.3333 times 0.5 plus 0.5 = 0.6667. k = 0: p = 0.6667 times 0.5 plus 1 = 1.3333, which matches 1 + 0.25 + 0.0833 computed term by term.">
<figcaption><b>Figure 12.3</b> Horner's rule for degree 2: 2 multiply-adds, from the highest coefficient down.</figcaption>
</figure>

A multiplication followed by an addition is 1 instruction on a GPU, the **fused multiply-add**,
which rounds once instead of twice. A polynomial of degree \\(d\\) therefore costs \\(d\\)
multiply-adds, \\(2d\\) floating-point operations (FLOP).

```text
HORNER(c[0..d], x):
    p ← c[d]
    for k ← d − 1 down to 0:
        p ← p × x + c[k]            # 1 fused multiply-add
    return p
```

The chapter's example has degree \\(d\\) = 32 and coefficients \\(c_k = 1/(k + 1)\\), evaluated
at \\(N\\) = 2<sup>24</sup> = 16,777,216 points spread evenly over \\([-1, 1)\\).

| Symbol | Meaning | In the example |
|---|---|---|
| \\(d\\) | degree | 32 |
| \\(c_k\\) | coefficient \\(k\\) | \\(1/(k + 1)\\): 1, 0.5, 0.3333, …, 0.0303 |
| \\(N\\) | number of points | 16,777,216 |
| \\(2dN\\) | operations | 2 × 32 × 16,777,216 = 1.07 GFLOP |
| \\(8N\\) | bytes moved: 4 read and 4 written per point | 134.2 MB |

### 12.3.2 The generic kernel

The **generic** kernel takes the degree and the coefficients as arguments. 1 thread evaluates
the polynomial at 1 point. `nvcc` compiles it with the rest of the program, so it must work for
any degree and any coefficients.

<p class="listing"><b>Listing 12.1</b> The generic kernel, in C++. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch12_nvrtc/nvrtc.cu">code/ch12_nvrtc/nvrtc.cu</a></p>

```cpp
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu:66:80}}
```

At each of the 32 steps, the loop reads `c[k]` from memory, because the compiler cannot know its
value, counts `k` down and tests it, and then performs 1 multiply-add. All the threads of a warp
read the same `c[k]`, so the read comes from the cache, but it is still an instruction the GPU
must issue.

The Rust kernel is the same loop. It takes the degree from the length of the slice `c`, so it too
works for any degree. `mul_add` asks for the fused multiply-add by name: Rust, unlike C++, does
not let the compiler fuse `p * v + c[k]` on its own, because fusing changes the rounding.

<p class="listing"><b>Listing 12.2</b> The generic kernel, in Rust. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch12_driver/src/main.rs">code/oxide/ch12_driver/src/main.rs</a></p>

```rust,ignore
{{#include ../../../../code/oxide/ch12_driver/src/main.rs:19:41}}
```

### 12.3.3 Writing the specialised kernel as text

Once the program knows the coefficients, it can write a kernel that contains them. **Specialising**
a kernel means compiling it for particular values that a generic kernel receives as arguments.
The specialised polynomial has no loop and no reads of `c`: 32 lines, each 1 multiply-add with a
constant.

```text
SPECIALISED SOURCE(c[0..d]):
    text ← kernel header, i, bounds test, v ← x[i]
    append "p = c[d];"                       written as a number
    for k ← d − 1 down to 0:
        append "p = p * v + c[k];"           written as a number
    append "y[i] = p;"
    return text
```

<p class="listing"><b>Listing 12.3</b> The program writes the source of the specialised kernel.</p>

```cpp
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu:82:104}}
```

3 details of the text matter.

- **`extern "C"`** keeps the kernel's name `poly` as it is. Without it, C++ encodes the parameter
  types into the name the compiler emits, here `_Z4polyPKfPfi`, so that overloaded functions get
  different names; this encoding is called **name mangling**. Section 12.5.2 looks the kernel up
  by name, so the name must be predictable.
- **`%.9e`** prints each coefficient with 9 significant digits, for example `3.125000000e-02` for
  1/32. 9 decimal digits are enough to identify every `float` exactly, so the constant in the
  text is the same `float` the generic kernel reads from memory.
- **The `f` suffix** makes each constant a `float`. Without it, `3.125000000e-02` is a `double`
  in C++, and the whole multiply-add would be done in double precision.

Figure 12.4 places the 2 kernels side by side, with the times measured in section 12.6.

<figure>
<img src="../../figures/ch12-specialise.svg" alt="Two code sheets. The generic kernel, built by nvcc, loops from c[degree] down, and each step reads c[k] from memory, counts k and does 1 multiply-add. The specialised kernel, built by NVRTC, is 32 straight lines such as p = p times v plus 3.13e-2f: each step is 1 multiply-add with the constant inside it. Below, bars: generic 2.61 ms, specialised 1.84 ms, and the time to move 134 MB at the rated 102 GB/s, 1.32 ms.">
<figcaption><b>Figure 12.4</b> The same polynomial: 1 kernel reads its coefficients, the other contains them.</figcaption>
</figure>

<div class="bee bee-happy" markdown="1">

A program can write a kernel as a string after it knows the values that a generic kernel would
have to read at run time.

</div>

## 12.4 NVRTC: compiling source inside the program

### 12.4.1 The 5 calls

**NVRTC**, NVIDIA Runtime Compilation, is a library, `libnvrtc`, that compiles CUDA C++ source
held in a string. It contains the same compiler front end as `nvcc`, but it needs no files, no
`nvcc` on the machine, and no GPU: it produces PTX or a cubin as bytes in memory. A compilation
takes 5 steps.

```text
COMPILE(source, target):
    prog ← nvrtcCreateProgram(source, name for messages)
    if nvrtcCompileProgram(prog, ["--gpu-architecture=" + target]) fails:
        print nvrtcGetProgramLog(prog)          # the compiler's error messages
        return nothing
    image ← nvrtcGetCUBIN(prog)  if target is real, such as sm_87
            nvrtcGetPTX(prog)    if target is virtual, such as compute_87
    nvrtcDestroyProgram(prog)
    return image
```

<p class="listing"><b>Listing 12.4</b> Compiling a string with NVRTC, in C++.</p>

```cpp
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu:106:142}}
```

The option `--gpu-architecture` decides what NVRTC produces. A real architecture, `sm_87`, gives
a cubin, which `nvrtcGetCUBIN` returns; a virtual one, `compute_87`, gives only PTX. The other
options are those of `nvcc`, written as strings: `-std=c++17`, `-default-device`, `-I` and so on.

### 12.4.2 Choosing the target from the GPU

The program asks the driver for the GPU's compute capability and builds the architecture name
from it: major 8 and minor 7 give `sm_87`. The same lines create the context of section 12.5.1.

<p class="listing"><b>Listing 12.5</b> A context, and the target for the GPU the program runs on.</p>

```cpp
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu:154:173}}
    // ...
}
```

```console
GPU: Orin, compute capability 8.7, target sm_87                   # Jetson Orin NX
GPU: NVIDIA Thor, compute capability 11.0, target sm_110          # Jetson AGX Thor
```

The same source gives a cubin of 3,624 bytes for `sm_87` and of 6,368 bytes for `sm_110`. The
PTX hardly changes, 2,237 bytes and 2,250 bytes: it describes the same virtual machine for both.

### 12.4.3 The compile log

A compiler finds mistakes in source code, and a program that writes source can make them. NVRTC
does not print its messages. It keeps them in the program's **log**, a string the program must ask
for with `nvrtcGetProgramLogSize` and `nvrtcGetProgramLog`. The chapter's program compiles a
second copy of its source in which `y[i] = p` has become `y[i] = q`, and prints the log:

```console
a source with a mistake
  compile failed, log:
poly.cu(42): error: identifier "q" is undefined
      y[i] = q;
             ^

1 error detected in the compilation of "poly.cu".
```

`poly.cu` is the name the program gave in `nvrtcCreateProgram`; no such file exists. Line 42 is
the last line of the generated text: 8 lines of header, 1 for `float p`, 32 multiply-adds, then
`y[i] = q`. Without the log, the program would know only that compilation failed.

### 12.4.4 NVRTC from Rust

cuda-oxide has no NVRTC wrapper at the revision this book uses, and the Rust program needs only 7
NVRTC functions, so it declares them itself. A declaration of functions written in another
language is a **foreign function interface (FFI)**. The `#[link(name = "nvrtc")]` attribute asks
the linker for `libnvrtc`; cuda-oxide's `cuda-bindings` crate already adds the toolkit's `lib64`
directory to the linker's search path, so `Cargo.toml` is the same as in Chapter 13 and gains no
new dependency. Rust 2024 writes the block as `unsafe extern "C"`: the compiler cannot check that
the declarations match the library, so the person who writes them takes responsibility for that.

<p class="listing"><b>Listing 12.6</b> The 7 NVRTC functions, declared in Rust.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch12_driver/src/main.rs:43:66}}
```

`Program` stands for NVRTC's `nvrtcProgram`, a pointer to an object only NVRTC understands. Each
function returns an `nvrtcResult`, where 0 means success, so `c_int` is the right Rust type for
it. The `compile` function follows the same 5 steps as in C++. `CString` adds the NUL byte that C
expects at the end of a string, and the small function `ok` turns a nonzero result into a Rust
error, so that `?` can pass it up.

<p class="listing"><b>Listing 12.7</b> Compiling a string with NVRTC, in Rust.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch12_driver/src/main.rs:92:133}}
```

The Rust program's source text is the same kernel. Rust prints the exponent of a float without a
leading zero, `3.125000000e-2`, so its source is 1,281 bytes instead of 1,315; C++ reads both
spellings as the same number.

<details class="check">
<summary>Why does the program compile for sm_87 rather than for compute_60, which would run on every GPU since 2016?</summary>

Portable code is for machines the program has never seen. A program compiling at run time
already knows its GPU. For `sm_87` NVRTC produces a cubin that loads in about 0.05 ms. PTX for any
virtual architecture must still be JIT-compiled by the driver when it is loaded, at 13 to 45 ms on
the Orin NX (section 12.7). PTX for an older virtual architecture also keeps the compiler from
using any feature added since.

</details>

## 12.5 The driver API: load, look up, launch

### 12.5.1 A context

Listing 12.5 began with 4 driver calls. `cuInit(0)` initialises the driver API; it must come
before any other `cu` call, and its argument must be 0. `cuDeviceGet` turns the number 0 into a
`CUdevice`. `cuDevicePrimaryCtxRetain` returns the device's **primary context**, the 1 context
per GPU that the runtime API uses, and counts 1 more user of it. `cuCtxSetCurrent` makes it the
context that later driver calls on this host thread act on.

The driver API can also create a separate context with `cuCtxCreate`. A device pointer belongs to
the context that allocated it and is not valid in another. A program that mixed the runtime API
with a context of its own would hold 2 sets of memory that do not mix. Retaining the primary context
keeps 1 context for the whole program (section 12.8). At the end, `cuDevicePrimaryCtxRelease`
counts 1 user fewer.

### 12.5.2 Modules and functions

`cuModuleLoadData` loads 1 image, PTX, cubin or fatbin, into the current context and returns a
module. `cuModuleGetFunction` finds a kernel in the module by its name.

<p class="listing"><b>Listing 12.8</b> Load an image and find the kernel in it.</p>

```cpp
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu:144:152}}
```

The name is the one in the machine code. Section 12.3.3 kept it as `poly` with `extern "C"`. A
copy of the program whose source left out `extern "C"` compiled without a complaint and then
stopped at the lookup:

```console
GPU: Orin, compute capability 8.7, target sm_87
NVRTC, a polynomial of degree 32, 1304 bytes of source
  compile to PTX        50.01 ms    2309 bytes
  compile to cubin      51.30 ms    3752 bytes
CUDA driver error: CUDA_ERROR_NOT_FOUND
```

The module held a kernel named `_Z4polyPKfPfi`, and no kernel named `poly`. A second copy, which
compiled its cubin for `sm_90` and loaded it on the Orin NX, stopped one step earlier, in
`cuModuleLoadData`, with `CUDA_ERROR_INVALID_SOURCE`: the driver does not load machine code for
another generation.

### 12.5.3 The argument array

`cuLaunchKernel` takes the function, the grid and block shape, the bytes of dynamic shared memory,
a stream, and the kernel's arguments. The driver does not know the kernel's parameter list. It
learns the size of each parameter from the module, so it needs only 1 thing from the program: for
each parameter, the **address** where the argument's value is stored. The program passes these
addresses as an array of `void*`.

<div class="example" markdown="1">

Worked example: the arguments of `poly(const float* x, float* y, int n)`

1. The host variables are `X` and `Y`, which hold device addresses, and `n`, which holds
   16,777,216.
2. `args[0] = &X`, `args[1] = &Y`, `args[2] = &n`: the addresses of the 3 variables.
3. The driver reads 8 bytes at `args[0]`, 8 at `args[1]` and 4 at `args[2]`, 8 + 8 + 4 = 20 bytes,
   and copies them into the kernel's parameters `x`, `y` and `n`.

</div>

<p class="listing"><b>Listing 12.9</b> The argument array and the 2 launches.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu:213:222}}
    // ...
}
```

The lambda `special` launches 65,536 blocks of 256 threads, 1 thread per point, on the default
stream (`nullptr`), with no dynamic shared memory (0). The last argument, `extra`, is a second
way to pass arguments, as 1 packed buffer; we leave it empty.

The common mistake is to write `X` where `&X` belongs. Figure 12.5 shows why it fails. A copy of
the program with `void* args[] = {X, Y, &n};` ran the generic kernel and died at the first
`cuLaunchKernel`:

```console
$ ./nvrtc                                         # with void* args[] = {X, Y, &n};
GPU: Orin, compute capability 8.7, target sm_87
NVRTC, a polynomial of degree 32, 1315 bytes of source
  ...
  generic, nvcc, <<<...>>>          2.897 ms   370.6 GFLOP/s
Segmentation fault
$ echo $?
139
```

The driver took the device address in `X` for the address of the argument on the host, and the
CPU's read of that address failed. Exit code 139 is 128 + 11, signal 11, `SIGSEGV`. On another
system the read might succeed and pass a meaningless value to the kernel instead.

<figure>
<img src="../../figures/ch12-args.svg" alt="Top zone, the correct form: args[0] = &X points to the host variable X, which holds the device address 0x203a00000, 8 bytes; args[1] = &Y points to Y, 8 bytes; args[2] = &n points to n = 16,777,216, 4 bytes. The driver follows each pointer and copies 20 bytes of parameters into the kernel's x, y and n. Bottom zone, the mistake: args[0] = X, so the driver reads 8 bytes at the device address 0x203a00000 on the CPU, and the program stops with a segmentation fault, exit code 139.">
<figcaption><b>Figure 12.5</b> Each entry of the array is the address of 1 argument, not the argument.</figcaption>
</figure>

### 12.5.4 The same steps in Rust

cuda-core wraps the same 3 driver steps. `CudaContext::load_module_from_image` calls `cuModuleLoadData` and returns a `CudaModule`, which
unloads itself with `cuModuleUnload` when it is dropped. `load_function` calls
`cuModuleGetFunction`. `launch_kernel_on_stream` calls `cuLaunchKernel` with a slice of argument
addresses, and is `unsafe`: like the C++ driver, it cannot check that the addresses match the
kernel's parameters.

<p class="listing"><b>Listing 12.10</b> Loading the NVRTC cubin, and the raw launch beside the typed one, in Rust.</p>

```rust,ignore
fn main() -> Result<(), Error> {
    // ...
{{#include ../../../../code/oxide/ch12_driver/src/main.rs:171:206}}
    // ...
}
```

`x.cu_deviceptr()` returns the device address held by a `DeviceBuffer`, a `u64`, so `x_ptr` plays
the part of `X` in C++. `&raw mut x_ptr` takes its address without creating a Rust reference, and
`.cast()` turns it into the untyped `*mut c_void` the driver expects.

The typed launch, `module.poly_generic(...)`, does the same work behind a checked signature. The
`#[cuda_module]` attribute reads the kernel's parameter list when the crate is compiled and
generates a method with matching Rust types, so a misplaced argument is a compile error rather
than a crash. A copy of the crate that swapped `&mut y` and `&c` did not build:

```console
error[E0308]: mismatched types
   --> src/main.rs:188:62
    |
188 |         Ok(module.poly_generic(&stream, &config, &x, &mut y, &c)?)
    |                   ------------                               ^^ types differ in mutability
    |                   |
    |                   arguments to this method are incorrect
    |
    = note: expected mutable reference `&mut DeviceBuffer<_>`
                       found reference `&DeviceBuffer<_>`
```

A kernel compiled at run time has no such signature at build time, in either language. That is
the price of building it at run time: the program, not the compiler, vouches for its arguments.

<div class="bee bee-shocked" markdown="1">

`cuLaunchKernel` reads each argument through the address the program gives it. Pass `&X`, never
`X`.

</div>

## 12.6 What it costs and what it buys

The C++ program compiles the specialised kernel to PTX and to a cubin, loads both, warms the GPU
up for about 0.4 s with the generic kernel, and then times 20 runs of each kernel. It compares
every one of the 16,777,216 results of the 2 kernels, and checks every 4,099th result against
Horner's rule in double precision on the CPU.

```console
$ ./nvrtc                                         # on a Jetson Orin NX
GPU: Orin, compute capability 8.7, target sm_87
NVRTC, a polynomial of degree 32, 1315 bytes of source
  compile to PTX        45.74 ms    2237 bytes
  compile to cubin      51.99 ms    3624 bytes
  load PTX               0.68 ms
  load cubin             0.05 ms
p(x) for 16777216 values of x
  generic, nvcc, <<<...>>>          2.605 ms   412.1 GFLOP/s
  specialised, NVRTC, cuLaunch      1.840 ms   583.4 GFLOP/s
  max error 5.1e-07, generic vs specialised 0.0e+00
a source with a mistake
  ...
PASSED
```

The lines read as follows.

- **Compile, 45.74 ms and 51.99 ms.** NVRTC ran the whole compiler on the CPU, once for each
  output. Across our runs the time ranged from 44 ms to 88 ms, with the load other programs put on
  the CPU.
- **Load PTX, 0.68 ms.** This run found the driver's compiled copy in its cache; section 12.7
  explains the cache, and the 13 to 45 ms a load costs without it.
- **Load cubin, 0.05 ms.** A cubin is ready to run, so loading it only copies it into the context.
- **The kernels.** The specialised kernel ran 2.605 / 1.840 = 1.42 times as fast as the generic
  one. Across the quiet runs the ratio was 1.3 to 1.46; it drifts with the GPU clock, as Chapter 6
  measured.
- **The results.** The 2 kernels agree to the last bit, a difference of 0.0, because both perform
  the same fused multiply-adds on the same `float` values in the same order. Both are within
  5.1 × 10<sup>−7</sup> of the double-precision value.

Why the specialised kernel wins follows from the arithmetic intensity of Chapter 1, the FLOP done
per byte moved.

<div class="example" markdown="1">

Worked example: is the polynomial limited by memory or by arithmetic?

1. Per point: 32 multiply-adds, 64 FLOP, for 4 bytes read and 4 written: 64 / 8 = 8 FLOP per byte.
2. The Orin NX does 2,400 GFLOP/s and its memory delivers 102 GB/s: the ridge point is
   2,400 / 102 = 23.5 FLOP per byte.
3. 8 < 23.5, so the work is **memory-bound**: at best, the time is that of moving
   134.2 MB at 102 GB/s, 1.32 ms.
4. The specialised kernel took 1.840 ms: 134.2 MB / 1.840 ms = 72.9 GB/s, 71% of the rated
   bandwidth, and about what a plain copy reaches on this board (68 to 78 GB/s in Chapter 7).
5. The generic kernel took 2.605 ms: 51.5 GB/s.

</div>

The specialised kernel runs at the speed of memory, so its 32 multiply-adds per point cost
nothing extra. The generic kernel issues more instructions per point: a read of `c[k]` and the
counting of `k` at each step. At 8 FLOP per byte the work sits close enough to the ridge point
that these additional instructions make the SMs, not memory, the limit. A polynomial of lower
degree would leave more room, and specialising it would gain less; a polynomial of higher
degree would gain more. Exercise 3 measures that.

The Rust program gives the same answer:

```console
$ cargo oxide run                                 # on a Jetson Orin NX
GPU: Orin, compute capability 8.7, target sm_87
NVRTC, 1281 bytes of source
  compile to cubin      57.66 ms    3624 bytes
  load cubin             0.56 ms
p(x) for 16777216 values of x
  generic, typed launch        2.857 ms   375.9 GFLOP/s
  specialised, raw launch      1.955 ms   549.3 GFLOP/s
  max error 5.1e-7, generic vs specialised 0.0e0
a source with a mistake
  ...
PASSED
```

The specialised kernel is the same NVRTC cubin, and took 1.96 ms against 1.84 ms in C++,
within the 10% that runs drift on this board. The generic Rust kernel, built by cuda-oxide, ran
within 10% of the generic C++ one, and its results match the specialised kernel's bit for bit,
because `mul_add` performs the same fused operation. Both programs also pass on a Jetson AGX
Thor. On your own GPU, the ratio of the 2 kernels is the number to compare.

Figure 12.6 puts every cost of the run on 1 logarithmic axis, and works out when compiling pays
for itself.

<figure>
<img src="../../figures/ch12-costs.svg" alt="Horizontal bars on a logarithmic time axis from 0.01 ms to 100 ms, measured on a Jetson Orin NX: compile to cubin 52 ms; load PTX with a cache miss 32 ms; 1 run of the generic kernel 2.61 ms; 1 run of the specialised kernel 1.84 ms; load PTX with a cache hit 0.77 ms; load cubin 0.05 ms. Below, a card: each run saves 0.77 ms, compiling costs 52 ms, so it pays back after 68 runs.">
<figcaption><b>Figure 12.6</b> Compiling costs about 20 to 30 runs of the kernel; it pays back after 68 runs.</figcaption>
</figure>

<div class="example" markdown="1">

Worked example: when does compiling at run time pay?

1. Each run of the specialised kernel saves 2.61 − 1.84 = 0.77 ms.
2. Compiling to a cubin and loading it costs 51.99 + 0.05 = 52 ms.
3. 52 / 0.77 = 68 runs to break even. A program that evaluates the polynomial 68 times or more
   gains; one that evaluates it once loses about 51 ms.

</div>

| Symbol | Meaning | In the example |
|---|---|---|
| \\(T_c\\) | time to compile and load | 52 ms |
| \\(t_g\\), \\(t_s\\) | 1 run of the generic and the specialised kernel | 2.61 ms, 1.84 ms |
| \\(T_c / (t_g - t_s)\\) | runs needed to break even | 52 / 0.77 = 68 |

<div class="bee bee-happy" markdown="1">

Runtime compilation costs tens of milliseconds once. It pays when the kernel it produces runs
many times.

</div>

## 12.7 The JIT cache

Loading PTX makes the driver compile it, and that compilation is not free. To avoid paying it in
every process, the driver keeps a **JIT cache**: a directory of machine code it has compiled
before, looked up by a key made from the PTX, the GPU and the driver version. The first load of a
given PTX compiles it and stores the result; later loads, in this process or any later one, copy
the stored machine code (figure 12.7).

<figure>
<img src="../../figures/ch12-cache.svg" alt="A flow. cuModuleLoadData receives 2,237 bytes of PTX and looks it up in the cache, keyed by the PTX, the GPU and the driver version. The first time, a miss: the driver compiles the PTX, 13 to 45 ms on the Orin NX, and stores the result for the next process, in CUDA_CACHE_PATH, by default ~/.nv/ComputeCache. Later, a hit: the machine code is copied from the file in 0.7 ms. Either way the module in the context holds the same code.">
<figcaption><b>Figure 12.7</b> 1 process pays to compile the PTX; later processes read the result.</figcaption>
</figure>

4 environment variables control the cache:

| Variable | Effect |
|---|---|
| `CUDA_CACHE_PATH` | the cache directory; `~/.nv/ComputeCache` when unset |
| `CUDA_CACHE_MAXSIZE` | the most the cache may hold, in bytes |
| `CUDA_CACHE_DISABLE` | `1` turns the cache off |
| `CUDA_FORCE_PTX_JIT` | `1` makes the driver ignore cubins and JIT-compile the PTX of every fatbin |

To measure the cache, we pointed `CUDA_CACHE_PATH` at an empty directory and ran the program
twice, then once more with the cache turned off.

```console
$ export CUDA_CACHE_PATH=$PWD/jitcache            # an empty directory
$ ./nvrtc | grep 'load PTX'
  load PTX              31.98 ms                  # miss: compiled, then stored
$ ./nvrtc | grep 'load PTX'
  load PTX               0.77 ms                  # hit
$ CUDA_CACHE_DISABLE=1 ./nvrtc | grep 'load PTX'
  load PTX              14.44 ms                  # compiled every time
```

In this sequence the hit cost 31.98 / 0.77 = 42 times less than the miss. Across our runs a
load without the cache took 13 to 45 ms and a load from it 0.68 to 0.97 ms. The cache directory
held 24 KB after the 3 runs. The driver discards the cache when it is upgraded,
because a new driver may compile the same PTX differently.

The cache stores the driver's work, not NVRTC's. A program that compiles source at run time pays
NVRTC every time, 44 to 88 ms here, unless it keeps the result itself. A long-running program
compiles each distinct source once and keeps the module loaded. A program that starts often can
save the cubin to a file, named by a hash of the source, the target and the NVRTC version, and
load the file next time.

<details class="check">
<summary>A data centre upgrades its GPU driver overnight. What happens to the first run of each program the next morning?</summary>

The driver discards the JIT cache when it is upgraded, so every PTX the programs load is compiled
again, once, at 13 to 45 ms each on a board like the Orin NX. Programs that load cubins, from a
fatbin or from NVRTC, do not notice: a cubin needs no JIT compilation.

</details>

## 12.8 Runtime API and driver API together

The chapter's C++ program uses both APIs in 1 process. The runtime API allocates memory, copies
it and launches `polyGeneric` with `<<<...>>>`; the driver API loads NVRTC's modules and launches
`poly`. The pointer `X` from `cudaMalloc` goes straight into the argument array of
`cuLaunchKernel`. That works because both APIs use the same context: the primary context, which
the runtime API uses for every call and which `cuDevicePrimaryCtxRetain` returns (figure 12.8).

<figure>
<img src="../../figures/ch12-context.svg" alt="On the left, 3 callers: the runtime API with cudaMalloc, cudaMemcpy and triple angle brackets; the driver API with cuModuleLoadData, cuLaunchKernel and cuModuleUnload; and cuda-core with CudaContext::new, load_module_from_image and launch_kernel_on_stream. On the right, 1 primary context of GPU 0 holds the memory X, Y and C from cudaMalloc, the module from the fatbin with polyGeneric, and the module from NVRTC with poly. All 3 callers reach it.">
<figcaption><b>Figure 12.8</b> The runtime API, the driver API and cuda-core share the primary context.</figcaption>
</figure>

Since CUDA 12.2 the driver also loads a module's kernels lazily by default
(`CUDA_MODULE_LOADING=LAZY`): part of the loading moves from `cuModuleLoadData` to the first
launch of each kernel. The warm-up runs of section 12.6 absorb that cost, so the times measure
the kernels alone.

<div class="bee bee-happy" markdown="1">

Retain the primary context, and pointers from the runtime API are valid in the driver API.

</div>

## 12.9 When to compile at run time

Runtime compilation earns its cost in 3 situations.

1. **The kernel depends on values known only at run time**, and runs often enough to repay the
   compilation, as in section 12.6. Sizes, shapes, coefficients, and which operations to fuse
   are typical values. PyTorch's compiler, for example, generates fused kernels after it has
   seen a model's tensor shapes.
2. **The code comes from outside the program.** A tool that lets its users write kernels, or a
   Python library such as CuPy that accepts CUDA C++ source as a string, compiles that source
   with NVRTC.
3. **The program must run on GPUs it was not built for.** Compiling at start-up for the GPU found
   produces the best machine code for that GPU, including ones released after the program.

It is the wrong tool for a kernel whose source never changes: `nvcc` compiles that once, on the
developer's machine, with the compiler warnings in front of the developer. A fatbin with cubins
for the GPUs we know and PTX for the newest virtual architecture covers future GPUs as well:

```console
$ nvcc app.cu -o app -gencode arch=compute_87,code=[sm_87,sm_90,compute_87]
```

`arch=compute_87` names the virtual architecture the source is compiled for. `code=` lists what
goes into the fatbin: cubins for `sm_87`, the Orin NX, and `sm_90`, the H100, and the
`compute_87` PTX itself, which the driver can JIT-compile on any later GPU.

| Situation | Choose |
|---|---|
| fixed kernel, known GPUs | `nvcc`, cubins in the fatbin |
| fixed kernel, future GPUs too | `nvcc`, cubins + PTX for the newest virtual architecture |
| kernel depends on run-time values, runs many times | NVRTC to a cubin for the GPU found |
| user-supplied kernel | NVRTC, with the log shown to the user |
| kernel depends on run-time values, runs once | the generic kernel |

## 12.10 The complete programs

<p class="listing"><b>Listing 12.11</b> The complete <code>nvrtc.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch12_nvrtc/nvrtc.cu">code/ch12_nvrtc/nvrtc.cu</a></p>

```cpp
{{#include ../../../../code/ch12_nvrtc/nvrtc.cu}}
```

<p class="listing"><b>Listing 12.12</b> The complete Rust program. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch12_driver/src/main.rs">code/oxide/ch12_driver/src/main.rs</a></p>

```rust,ignore
{{#include ../../../../code/oxide/ch12_driver/src/main.rs}}
```

## Common mistakes

- **`X` instead of `&X` in the argument array.** The driver reads the argument through each
  address; the program crashed with a segmentation fault.
- **A kernel name without `extern "C"`.** The name is mangled, and `cuModuleGetFunction` returns
  `CUDA_ERROR_NOT_FOUND`.
- **Ignoring the log.** `nvrtcCompileProgram` reports only that compilation failed; the reason is
  in the log.
- **A constant without the `f` suffix.** `0.5` is a `double` in C++, and the arithmetic around it
  becomes double precision.
- **A cubin for another GPU.** A cubin for `sm_90` does not load on an `sm_87` GPU; ask the
  driver for the compute capability and compile for it.
- **A second context.** Memory from `cudaMalloc` belongs to the primary context; retain that
  context instead of creating one with `cuCtxCreate`.
- **Compiling in a loop.** Compile each distinct source once, keep the module, and reuse it.

<div class="summary" markdown="1">

## Summary

- The runtime API creates a context, registers kernels and packs arguments for us, all of which
  needs the kernel to exist at build time. The driver API exposes the objects underneath: device,
  context, module, function. cuda-core is built directly on it.
- PTX is portable text for a virtual GPU and is JIT-compiled by the driver; a cubin is machine code
  for 1 GPU generation; a fatbin bundles both.
- NVRTC compiles CUDA C++ from a string in 5 calls, to PTX or to a cubin for the GPU the program
  asks the driver about. Its errors are in the log.
- `cuModuleLoadData`, `cuModuleGetFunction` and `cuLaunchKernel` load, find and launch the
  result. The argument array holds the address of each argument.
- A polynomial of degree 32 specialised to its coefficients ran 1.3 to 1.46 times as fast as the
  generic kernel on the Orin NX, at 71% of the rated memory bandwidth, with bit-identical results.
  Compiling cost about 52 ms, repaid after 68 runs.
- The JIT cache turned a 13 to 45 ms PTX load into a 0.7 ms one for every later process.
- Retaining the primary context lets the runtime API and the driver API share memory.

</div>

Chapter 13 starts Part V, where every kernel is written in Rust with cuda-oxide, and the
`kernels::load` call of every program is the driver-API load of this chapter.

## Exercises

1. Compile the specialised source to a cubin for `compute_87` instead of `sm_87`. What does
   `nvrtcGetCUBINSize` return, and why?
2. Change the program to load only the PTX, with `CUDA_CACHE_DISABLE=1`, and time 10 loads in 1
   process. Does the driver compile the PTX again each time?
3. Run the program for degrees 4, 8, 16, 32, 64 and 128. Plot the time of both kernels against
   the degree, mark the degree where the work stops being memory-bound on your GPU, and explain
   where the ratio of the 2 kernels grows.
4. Write the coefficients with `%.3e` instead of `%.9e`. Does the result still match the generic
   kernel bit for bit? How large is the difference?
5. Remove `extern "C"` from the generated source, and look the kernel up by its mangled name,
   `_Z4polyPKfPfi`. Then find the name without guessing: NVRTC's `nvrtcAddNameExpression` and
   `nvrtcGetLoweredName` return the mangled name of a C++ function.
6. Save the cubin to a file named by a hash of the source and the target, and load it with
   `cuModuleLoadData` on the next run. How long does the program take to reach the first launch?
7. In the Rust program, change `args` to hold `x_ptr as *mut c_void` instead of its address. Run
   it, and explain the result with figure 12.5.
8. Using the break-even formula of section 12.6, how many runs would repay compilation on a GPU
   whose specialised and generic kernels take 0.08 ms and 0.11 ms?

## Sources and further reading

- NVIDIA, *NVRTC User Guide*: <https://docs.nvidia.com/cuda/nvrtc/>
- NVIDIA, *CUDA Driver API*, modules and execution control:
  <https://docs.nvidia.com/cuda/cuda-driver-api/>
- NVIDIA, *CUDA C++ Programming Guide*, "Compilation with NVCC", "Binary Compatibility", "PTX
  Compatibility", "Just-in-Time Compilation" and "Lazy Loading":
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *CUDA Environment Variables*: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables>
- NVIDIA, *NVCC*, "GPU Compilation" and `-gencode`: <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>
- NVIDIA, cuda-oxide, `cuda-core` crate (`CudaContext`, `CudaModule`, `launch_kernel_on_stream`),
  rev b7fe543: <https://github.com/NVlabs/cuda-oxide>
