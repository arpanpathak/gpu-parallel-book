# Chapter 13: Rust Meets the GPU

> *"Rust does not make the GPU safer. It makes the *host* safer, which is where
> the crashes were."*

This part of the book changes language but not hardware. The GPU is still the
machine of Chapter 2; the kernels of Chapters 3–12 still run on it. What
changes is the *host*: instead of C++ calling `cudaMalloc` and launching
kernels, we use Rust. This chapter covers why that matters, the ecosystem
(`rustacuda` and `cudarc`), and a complete Rust host program that allocates
device memory, moves data, and launches a CUDA kernel — with the safety
properties Rust brings to each step.

## 13.1 Why Rust on the Host

The GPU's failure modes (Chapter 5: races; Chapter 10: leaks) are host-side
failures first: a leak is a missing `cudaFree`, a use-after-free is a dangling
device pointer, a race is often *launched* from the host. Rust's ownership
system attacks exactly these:

- **Ownership and lifetimes.** A `CudaSlice<T>` owns its device allocation;
  when it is dropped, the allocation is freed. Double-frees and leaks become
  type errors, not runtime incidents.
- **No data races by construction.** The borrow checker prevents two mutable
  references to the same buffer from existing simultaneously — a guarantee
  the C++ compiler never offers.
- **`Result`-based errors.** CUDA's `cudaError_t` becomes a typed `Result<T,
  CudaError>`; ignoring an error is a compile-time warning (the `must_use`
  attribute), not a silent misbehaviour.

The cost is what Rust always costs: the borrow checker sometimes fights you,
and the FFI boundary (where unsafe lives) must be drawn honestly. This chapter
is about drawing that boundary well.

## 13.2 The Ecosystem: `rustacuda` and `cudarc`

Two host libraries dominate:

- **`rustacuda`** — the older wrapper over the CUDA driver API. Safe-ish
  modules for contexts, modules, functions, streams, and memory. Historically
  important; now largely superseded for new work.
- **`cudarc`** — the actively maintained wrapper ("CUDA in Rust"). It wraps
  the driver API (`cudarc::driver`), NVRTC (`cudarc::nvrtc`), and the
  libraries (cuBLAS, cuDNN, cuFFT, cuRAND, NCCL) with three layers per
  wrapper: `safe` (high-level, checked), `result` (thin, returns error codes),
  and `sys` (raw FFI). This book uses `cudarc`.

The *third* member of the ecosystem, **CUDA-Oxide** (Chapter 14), is
different in kind: not a wrapper around CUDA C++ but a compiler that turns
Rust kernels into PTX. Chapter 14 is devoted to it. This chapter stays with
`cudarc`, which drives *existing* (C++-compiled) kernels.

## 13.3 The Kernel, Compiled Ahead of Time

We reuse the Chapter 3 vector-add kernel, compiled to PTX with `nvcc` (not
NVRTC — we keep the toolchain classic for this chapter):

```cuda
// kernels/vector_add.cu
// Compiled once, ahead of time, to PTX:
//   nvcc -arch=compute_90 -ptx kernels/vector_add.cu -o vector_add.ptx
extern "C" __global__ void vector_add(const float* a, const float* b,
                                      float* c, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**Why `extern "C"`?** CUDA C++ mangles kernel names like any C++ symbol. The
driver API loads kernels *by name* (Chapter 12); `extern "C"` guarantees the
module symbol is literally `vector_add`, so the Rust side can look it up
without demangling. This is the same convention NVRTC examples use.

## 13.4 The Rust Host Program

```rust
// main.rs — Rust host driving the vector_add kernel via cudarc.
// Requires: CUDA toolkit installed (for the driver and nvrtc), and the
// vector_add.ptx file next to the binary (or embedded; see 13.5).

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

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
    // No cudaFree call anywhere in this program.
    let mut d_a: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let mut d_b: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let mut d_c: CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;

    // --- Host -> device copies ----------------------------------------------
    // htod_copy_into copies a &[f32] into a device slice. The '&mut' on the
    // destination is the ownership language: the copy mutates the device
    // buffer, and Rust requires exclusive access to do so.
    dev.htod_copy_into(&a, &mut d_a)?;
    dev.htod_copy_into(&b, &mut d_b)?;

    // --- Load the PTX and fetch the kernel handle ---------------------------
    // load_ptx loads the module from the filesystem and registers the named
    // kernel. Errors (missing file, missing symbol) surface as Results.
    let module = "vector_add";
    dev.load_ptx("vector_add.ptx", module, &["vector_add"])?;
    let f = dev.get_func(module, "vector_add")?;

    // --- Launch --------------------------------------------------------------
    // The launch is marked unsafe: the configuration (grid/block shape) and
    // the argument tuple must match the kernel's real signature and index
    // space. cudarc checks argument arity and types at the type level but
    // cannot verify the kernel's internal assumptions — hence SAFETY.
    //
    // SAFETY: the grid covers exactly N threads (LaunchConfig::for_num_elems
    // rounds up to whole warps), the kernel guards with `if (i < n)`, and the
    // argument tuple (a, b, &mut c, n) matches the extern "C" signature.
    unsafe {
        f.launch(LaunchConfig::for_num_elems(N),
                 (&d_a, &d_b, &mut d_c, n32))
    }?;

    // --- Device -> host copy -------------------------------------------------
    // dtoh_sync_copy blocks until the stream's work completes and copies the
    // result back. The trailing ? propagates any device error encountered.
    let c: Vec<f32> = dev.dtoh_sync_copy(&d_c)?;

    // --- Verify ---------------------------------------------------------------
    let max_err = c.iter().zip(a.iter().zip(b.iter()))
        .map(|(c, (a, b))| (c - (a + b)).abs())
        .fold(0.0f32, f32::max);
    println!("max error = {max_err}");

    // d_a, d_b, d_c are dropped here; the allocations are freed by their
    // destructors. The device handle's context is cleaned up on drop.
    Ok(())
}
```

**The safety ledger.** What is unsafe in this program, and why?

1. The `unsafe { f.launch(...) }` block — the raw launch. The type system
   checks the *arity* and *types* of the arguments (the tuple `(&d_a, &d_b,
   &mut d_c, n32)`), but not the *semantics*: that the kernel's index
   arithmetic matches `LaunchConfig`, that `n32` matches the kernel's `int n`.
   The `SAFETY` comment states the invariants a reviewer must check — the same
   contract Chapter 3 expressed as comments in C++.
2. Everything else — allocation, copies, module loading — is safe API:
   ownership guarantees the lifetimes, `Result` guarantees the errors.

**What is *not* solved.** The kernel itself is still C++ and still
unsafe-by-construction: an out-of-bounds write inside `vector_add` corrupts
whatever it corrupts, and Rust cannot see it. This is the honest boundary:
Rust secures the host, not the device. CUDA-Oxide (Chapter 14) attacks the
device side.

## 13.5 Embedding the PTX

A filesystem dependency is fragile in production. The canonical fix embeds the
PTX in the binary at compile time:

```rust
// build.rs (or a const in the crate) embeds the PTX text.
const VECTOR_ADD_PTX: &str = include_str!("vector_add.ptx");

// Load directly from the embedded string instead of the filesystem:
dev.load_ptx(VECTOR_ADD_PTX, module, &["vector_add"])?;
```

`include_str!` inlines the file at compile time: the binary is self-contained
and the kernel cannot go missing in deployment. This is the pattern the
capstone uses.

## 13.6 Comparing with the C++ Host

| Concern | C++ (Chapters 3–6) | Rust + cudarc |
|---|---|---|
| Allocation lifetime | Manual `cudaMalloc`/`cudaFree` | `CudaSlice` RAII on drop |
| Copy direction | `cudaMemcpy` with direction enum | Typed `htod_copy_into` / `dtoh_sync_copy` |
| Error handling | `CHECK` macro discipline | Typed `Result` with `?` |
| Kernel launch | `<<<>>>`, unchecked args | `unsafe` launch with typed args + SAFETY comment |
| Data races | Compiler silent | Borrow checker rejects at compile time |
| Device-side safety | No help | No help (until CUDA-Oxide) |

The table is the pitch: every column on the left was a class of bug; every
entry on the right removes a class. The price — the `unsafe` block and its
SAFETY comment — is honest and small.

## 13.7 Exercises

1. Explain why `extern "C"` on the kernel matters for `dev.get_func(module,
   "vector_add")`. What would happen without it?
2. The launch is wrapped in `unsafe`. List three kernel-side assumptions
   that the `SAFETY` comment must document.
3. Trace the lifetimes: why is `&mut d_c` (not `&d_c`) required in the
   launch tuple, and what does the borrow checker prevent?
4. Compare `include_str!` with a runtime filesystem read. When is the
   filesystem version the right choice?
