# Chapter 13: Rust Meets the GPU

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch13_rust_vector_add/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch13_rust_vector_add)
> in the repository.

This part of the book changes language but not hardware. The GPU remains the
machine of Chapter 2, and the kernels of Chapters 3-12 still run on it. What
changes is the host: Rust replaces C++ for allocating device memory, moving
data, and launching kernels. This chapter covers the Rust CUDA ecosystem
(`rustacuda` and `cudarc`) and a complete Rust host program, with the safety
properties Rust provides at each step.

## 13.1 Rust on the Host

The GPU failure modes discussed in earlier chapters are often host-side
failures first: a leak is a missing `cudaFree`, a use-after-free is a dangling
device pointer, and a race is usually launched from the host. Rust's ownership
system addresses these directly:

- **Ownership and lifetimes.** A `CudaSlice<T>` owns its device allocation.
  When it is dropped, the allocation is freed. Double-frees and leaks become
  type errors rather than runtime incidents.
- **No data races by construction.** The borrow checker prevents two mutable
  references to the same buffer from existing simultaneously, a guarantee the
  C++ compiler does not offer.
- **`Result`-based errors.** CUDA's `cudaError_t` becomes a typed
  `Result<T, CudaError>`. Ignoring an error is a compile-time warning through
  `must_use`, not silent misbehaviour.

The cost is the one Rust charges everywhere: the borrow checker rejects code it
cannot prove safe, and the FFI boundary, where `unsafe` lives, must be drawn
precisely. This chapter is about drawing that boundary well.

## 13.2 The Ecosystem: `rustacuda` and `cudarc`

Two host libraries dominate:

- **`rustacuda`** - an older wrapper over the CUDA driver API. It provides
  safe modules for contexts, modules, functions, streams, and memory. It is
  historically important and now largely superseded for new work.
- **`cudarc`** - the actively maintained wrapper around the driver API
  (`cudarc::driver`), NVRTC (`cudarc::nvrtc`), and the CUDA libraries (cuBLAS,
  cuDNN, cuFFT, cuRAND, NCCL). It exposes three layers per wrapper: `safe`
  (high-level, checked), `result` (thin, returns error codes), and `sys` (raw
  FFI). This book uses `cudarc`.

**CUDA-Oxide** (Chapter 14) is different in kind: not a wrapper around CUDA
C++, but a compiler that turns Rust kernels into PTX. This chapter uses
`cudarc` to drive existing kernels compiled from C++.

## 13.3 The Kernel, Compiled Ahead of Time

The Chapter 3 vector-add kernel is reused and compiled to PTX with `nvcc`:

```cuda
// kernels/vector_add.cu
// Compiled once, ahead of time, to PTX:
//   nvcc -arch=compute_60 -ptx kernels/vector_add.cu -o vector_add.ptx
// compute_60 PTX runs on any CUDA 12.x GPU (Pascal or newer) via the
// driver's JIT; use compute_87 on Jetson Orin, compute_80 on A100, or
// compute_90 on H100 for native SASS.
extern "C" __global__ void vector_add(const float* a, const float* b,
                                      float* c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

CUDA C++ mangles kernel names like any C++ symbol. The driver API loads
kernels by name (Chapter 12), so `extern "C"` guarantees the module symbol is
literally `vector_add`. The Rust side can then look it up without demangling.

## 13.4 The Rust Host Program

```rust
// main.rs - Rust host driving the vector_add kernel via cudarc.
// Requires: CUDA toolkit installed (for the driver and nvrtc), and the
// vector_add.ptx file in the working directory (or embedded; see 13.5).

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;   // wraps PTX source: Ptx::from_file or Ptx::from_src

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Device handle -----------------------------------------------------
    // CudaDevice::new(0) opens the first GPU, initialising the driver and
    // creating the CUDA context. It returns Result: no GPU -> Err here,
    // reported as a typed error instead of a crash.
    let dev = CudaDevice::new(0)?;

    // --- Problem size ------------------------------------------------------
    const N: usize = 1 << 20;              // 1,048,576 elements
    let n32: i32 = N as i32;               // kernel expects a 32-bit int

    // --- Host data ----------------------------------------------------------
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 2.0 * i as f32).collect();

    // --- Device allocation ---------------------------------------------------
    // alloc_zeros allocates device memory and zero-initialises it. The
    // returned CudaSlice<f32> OWNS the allocation: dropping it frees it.
    // No cudaFree call appears in this program.
    let mut d_a: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let mut d_b: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let mut d_c: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;

    // --- Host -> device copies ----------------------------------------------
    // htod_copy_into takes ownership of the host Vec because the copy is
    // queued on the device's stream; no other code can mutate the buffer while
    // it is in flight. We pass clones so the originals remain usable for
    // verification below. The &mut destination expresses that the copy mutates
    // the device buffer, and Rust requires exclusive access to do so.
    dev.htod_copy_into(a.clone(), &mut d_a)?;
    dev.htod_copy_into(b.clone(), &mut d_b)?;

    // --- Load the PTX and fetch the kernel handle ---------------------------
    // load_ptx loads the module and registers the named kernel. The PTX is
    // wrapped in a Ptx: Ptx::from_file for a path, Ptx::from_src for an
    // embedded string (13.5). Missing files and missing symbols surface as
    // Results.
    let module = "vector_add";
    dev.load_ptx(Ptx::from_file("vector_add.ptx"), module, &["vector_add"])?;
    let f = dev.get_func(module, "vector_add")
        .ok_or("kernel 'vector_add' not found in the loaded module")?;

    // --- Launch --------------------------------------------------------------
    // The launch is marked unsafe: the configuration (grid/block shape) and
    // the argument tuple must match the kernel's real signature and index
    // space. cudarc checks argument arity and types at the type level but
    // cannot verify the kernel's internal assumptions - hence the SAFETY
    // comment.
    //
    // SAFETY: the grid covers N threads (LaunchConfig::for_num_elems rounds up
    // to whole warps), the kernel guards with `if (i < n)`, and the argument
    // tuple (&d_a, &d_b, &mut d_c, n32) matches the extern "C" signature.
    unsafe {
        f.launch(LaunchConfig::for_num_elems(N as u32),
                 (&d_a, &d_b, &mut d_c, n32))
    }?;

    // --- Device -> host copy -------------------------------------------------
    // dtoh_sync_copy blocks until the stream's work completes and copies the
    // result back. The ? propagates any device error encountered.
    let c: Vec<f32> = dev.dtoh_sync_copy(&d_c)?;

    // --- Verify ---------------------------------------------------------------
    let max_err = c.iter().zip(a.iter().zip(b.iter()))
        .map(|(c, (a, b))| (c - (a + b)).abs())
        .fold(0.0f32, f32::max);
    println!("max error = {max_err}");

    // d_a, d_b, d_c are dropped here; their destructors free the allocations.
    // The device handle's context is cleaned up on drop.
    Ok(())
}
```

**Safety ledger.** The only `unsafe` block is the launch. The type system
checks the arity and types of the argument tuple `(&d_a, &d_b, &mut d_c,
n32)`, but not the semantics: that the kernel's index arithmetic matches the
launch configuration, or that `n32` matches the kernel's `int n`. The `SAFETY`
comment states the invariants a reviewer must check, the same contract Chapter
3 expressed as comments in C++.

Everything else, allocation, copies, and module loading, is a safe API:
ownership guarantees lifetimes, and `Result` guarantees error handling.

**What is not solved.** The kernel is still C++ and still unsafe by
construction: an out-of-bounds write inside `vector_add` corrupts whatever it
corrupts, and Rust cannot observe it. Rust secures the host, not the device.
CUDA-Oxide (Chapter 14) addresses the device side.

## 13.5 Embedding the PTX

A filesystem dependency is fragile in production. The PTX can be embedded in
the binary at compile time:

```rust
// build.rs (or a const in the crate) embeds the PTX text.
const VECTOR_ADD_PTX: &str = include_str!("vector_add.ptx");

// Load directly from the embedded string instead of the filesystem.
// Ptx::from_src wraps the string; Ptx::from_file wraps a path (13.4).
dev.load_ptx(Ptx::from_src(VECTOR_ADD_PTX), module, &["vector_add"])?;
```

`include_str!` inlines the file at compile time. The binary is self-contained,
and the kernel cannot go missing during deployment. The capstone uses this
pattern.

## 13.6 Comparing with the C++ Host

| Concern | C++ (Chapters 3-6) | Rust + cudarc |
|---|---|---|
| Allocation lifetime | Manual `cudaMalloc`/`cudaFree` | `CudaSlice` RAII on drop |
| Copy direction | `cudaMemcpy` with direction enum | Typed `htod_copy_into` / `dtoh_sync_copy` |
| Error handling | `CHECK` macro discipline | Typed `Result` with `?` |
| Kernel launch | `<<<>>>`, unchecked args | `unsafe` launch with typed args + SAFETY comment |
| Data races | Compiler silent | Borrow checker rejects at compile time |
| Device-side safety | No help | No help (until CUDA-Oxide) |

Each right-hand entry removes a bug class that the left-hand column required a
discipline or a runtime tool to catch. The price, the `unsafe` block and its
`SAFETY` comment, is explicit and small.

## 13.7 Lifetimes in Action

The claims in §13.1 have concrete demonstrations. The borrow checker rejects
whole classes of GPU programs at compile time:

```rust
// What the borrow checker prevents (none of these compile):

// 1. Two mutable borrows of the same device buffer. The launch tuple below
//    needs two &mut to d_c, which Rust forbids:
//      f.launch(config, (&mut d_c, &mut d_c))   // ERROR: cannot borrow twice
//    In C++ this compiles and produces a race inside the kernel.

// 2. Using a buffer after moving it. The CudaSlice is gone after the move, so
//    any use is a compile error rather than a use-after-free:
//      let stolen = d_a;               // d_a is moved into stolen
//      dev.htod_copy_into(&a, &mut d_a);   // ERROR: borrow of moved value
//    In C++, the equivalent sequence (copying a raw pointer, then freeing it)
//    is a use-after-free that requires Compute Sanitizer to catch at run time.

// 3. Passing a host Vec where a device slice is expected. The types do not
//    match, so the error is at the call site:
//      dev.htod_copy_into(&host_vec, &mut d_c);  // ERROR: type mismatch
```

Each rejected program is a bug class that the C++ chapters required a runtime
tool (Compute Sanitizer, Chapter 16) or a discipline (the `CHECK` macro,
Chapter 3) to catch. Rust moves detection to the compiler, which runs earlier
and cannot be forgotten. The borrow checker is the `CHECK` macro promoted to a
compile-time guarantee.

## The Unsafe Boundary Is an Explicit Contract

The design decision in this chapter is where the unsafe boundary is drawn.
Rust's safety guarantees apply only to code within the rules of ownership and
borrowing. A CUDA kernel launch sits at the edge of those rules because the
host cannot see inside the kernel and therefore cannot prove that the kernel's
index arithmetic matches the launch configuration. That unprovable obligation
is what the `unsafe` block marks.

Before the launch, the guarantees are compile-time. `CudaDevice::new(0)`
returns a `Result`, so a missing GPU must be handled. `alloc_zeros` returns an
owned `CudaSlice`, so the allocation is freed when the slice is dropped.
`htod_copy_into` takes ownership of the host vector, so the data cannot be
mutated while the copy is in flight. The launch requires `&mut d_c` for the
output buffer, so two kernels cannot hold mutable references to the same buffer
simultaneously.

The launch is where the type system runs out. The tuple `(&d_a, &d_b, &mut
d_c, n32)` has the right arity and types, but nothing in the types proves that
the grid covers exactly `N` elements, that the kernel's `if (i < n)` guard
matches, or that `n32` is the correct interpretation of the kernel's `int n`.
The `SAFETY` comment is the contract that fills the gap: a reviewer must be
able to verify those facts from the comment and the surrounding context. This
is the same contract Chapter 3 expressed as C++ comments. Rust makes everything
around the launch enforceable by the compiler, leaving one small, documented,
auditable seam instead of a program-wide discipline.

## Common Pitfalls

- Forgetting `extern "C"` on the kernel. The driver looks up kernels by
  unmangled name; without it, `get_func(module, "vector_add")` fails.
- Copying a `CudaSlice` instead of moving it. Device buffers are unique
  resources; use `&mut` and moves, not copies.
- Putting `unsafe` around the whole program instead of just the launch. The
  purpose is to isolate the unprovable part, not to annotate everything.
- Ignoring the `SAFETY` comment. If the kernel-side assumptions cannot be
  stated, the launch is not known to be correct.

## Check Your Understanding

<details>
<summary>Why does extern "C" matter?</summary>

C++ mangles function names, for example `_Z11vector_add...`. The CUDA driver
loads kernels by exact string name, so `extern "C"` guarantees the symbol is
literally `vector_add`, letting the Rust host look it up without demangling.
</details>

<details>
<summary>Why is &mut d_c required in the launch tuple?</summary>

The kernel writes through the `c` pointer. Rust requires exclusive access for
mutation, so the launch needs `&mut d_c`; passing `&d_c` would be a compile
error because shared references cannot be used for mutation. The borrow checker
also prevents two kernels from holding `&mut` to the same buffer at once.
</details>

<details>
<summary>What does include_str! provide over Ptx::from_file?</summary>

`include_str!` embeds the PTX text into the binary at compile time. The binary
is self-contained and cannot lose the kernel file during deployment.
`Ptx::from_file` reads the filesystem at run time, which is simpler for
experimentation but fragile in production.
</details>

## Key Takeaways

- Rust secures the host: ownership prevents leaks and double-frees, the borrow checker prevents data races, and Result prevents ignored errors.
- cudarc provides RAII device memory (CudaSlice), typed copies, and module loading from PTX.
- extern "C" keeps kernel symbols unmangled so the driver can find them by name.
- The unsafe launch is explicit; the SAFETY comment states the kernel-side obligations the type system cannot check.
- include_str! embeds PTX in the binary, removing the filesystem dependency.

## 13.8 Exercises

1. Explain why `extern "C"` on the kernel matters for
   `dev.get_func(module, "vector_add")`. What would happen without it?
2. The launch is wrapped in `unsafe`. List three kernel-side assumptions that
   the `SAFETY` comment must document.
3. Trace the lifetimes: why is `&mut d_c` (not `&d_c`) required in the launch
   tuple, and what does the borrow checker prevent?
4. Compare `include_str!` with a runtime filesystem read. When is the
   filesystem version the right choice?

## Sources and Further Reading

- NVIDIA, *CUDA C++ Programming Guide*: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- The Rust Book, "Ownership" and "Unsafe Rust": <https://doc.rust-lang.org/book/>
- `rustacuda` crate documentation: <https://docs.rs/rustacuda/>
- `cudarc` crate documentation: <https://docs.rs/cudarc/>
