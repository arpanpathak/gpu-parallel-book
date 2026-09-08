# Chapter 14: CUDA-Oxide - Kernels in Pure Rust

> **Code companion:** the complete, buildable code for this chapter lives in
> [`code/ch14_cuda_oxide/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch14_cuda_oxide)
> in the repository.

Chapter 13 secured the host with Rust, but the kernel remained C++ compiled by
`nvcc` and invoked through an `unsafe` boundary. **CUDA-Oxide** is NVIDIA Labs'
experimental answer to the remaining gap: a `rustc` codegen backend that
compiles idiomatic Rust kernels directly to PTX. There is no DSL, no
foreign-language binding, and no `nvcc`; host and device code can live in the
same file. This chapter describes the project as documented by its repository,
including its pipeline and the parts that remain experimental.

## 14.1 What CUDA-Oxide Is

CUDA-Oxide (repository `NVlabs/cuda-oxide`, announced May 2026) is described by
its authors as:

> *"An experimental Rust-to-CUDA compiler that lets you write SIMT GPU kernels
> in safe(ish), idiomatic Rust. It compiles standard Rust code directly to
> PTX - no DSLs, no foreign language bindings, just Rust."*

Its design goals, from the project documentation:

- **Single-source compilation.** Host and device code live in the same file and
  are built with one command, `cargo oxide build`.
- **A rustc codegen backend** that compiles `#[kernel]` functions to PTX.
- **Device-side abstractions**: type-safe indexing, shared memory, scoped
  atomics, barriers, TMA, and warp or cluster operations.
- **Compile-time kernel policies** for separate tuned specialisations without
  runtime policy arguments.
- **A host-side runtime** (`cuda-core`, `cuda-async`) for memory management,
  pinned host transfers, and kernel launching.

The project's own term "safe(ish)" matters. CUDA-Oxide keeps Rust's type system
and ownership on the device, but SIMT programming includes operations, such as
raw launch configuration and memory ordering, that cannot yet be fully proven
safe. The project is explicit about the boundary.

## 14.2 The Compilation Pipeline

CUDA-Oxide does not translate Rust to CUDA C. It reuses rustc's internal
representations and replaces only the code generation stage:

![CUDA-Oxide compilation pipeline: Rust to MIR to Pliron to LLVM IR to PTX](../../assets/ch14_rust_to_ptx.svg)

Because the front end is real rustc, ownership, borrowing, pattern matching,
and traits are checked before GPU code is generated. A kernel that violates the
borrow checker never becomes PTX. The experimental part is the back end:
Pliron is a young framework, and the project warns that lowering to LLVM IR may
contain bugs and missing features.

## 14.3 Installation

CUDA-Oxide is Linux-only at the time of writing (tested on Ubuntu 24.04) and
requires:

- **`cargo-oxide`**, the cargo subcommand that drives the build (`cargo oxide
  build/run/inspect/...`);
- **Rust nightly** with the `rust-src`, `rustc-dev`, and `llvm-tools`
  components, pinned in the project's `rust-toolchain.toml`;
- **CUDA Toolkit 12.x or newer**;
- **Clang and libclang** development headers, required by `bindgen` when
  building the host `cuda-bindings` crate.

```bash
# Install the cargo subcommand with the pinned nightly toolchain:
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide

# On first run, cargo-oxide fetches and builds the codegen backend.
# Verify CUDA is on the path:
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

The workflow is cargo-shaped:

```bash
cargo oxide run    host_closure      # build and run an example
cargo oxide inspect vecadd           # build and print the generated PTX
cargo oxide pipeline vecadd          # show the full pipeline (MIR -> Pliron -> LLVM -> PTX)
cargo oxide sanitize vecadd --tool memcheck   # CUDA correctness checks
cargo oxide debug vecadd --tui       # debug with cuda-gdb
```

## 14.4 A First Kernel: The Generic `map`

The project's documented example is a generic elementwise `map`, the Rust
equivalent of Chapter 10's `transformKernel`, with the kernel written in Rust:

```rust
// Single source file: device AND host code together.
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::{cuda_module, load_kernel_module};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// ---------------------------------------------------------------------------
// Device side: a generic kernel that applies any function to each element.
// F can be a closure with captures - rustc monomorphises it to a concrete
// type at compile time, exactly like a C++ template instantiation.
// ---------------------------------------------------------------------------
#[cuda_module]
mod kernels {
    use super::*;

    // The #[kernel] attribute tells the backend to compile this function
    // to PTX. It is the Rust equivalent of __global__.
    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T],
                                              mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();        // threadIdx/blockIdx, fused
        let i = idx.get();                    // the global linear index
        // DisjointSlice guarantees this thread's slot is exclusive:
        // two threads can never get_mut the same element.
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host side: allocate, load the module, launch.
// ---------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;           // open GPU 0
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input  = DeviceBuffer::from_host(&stream, &data)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1024)?;

    // Load the module: the codegen backend writes host_closure.ptx next to
    // Cargo.toml; load_kernel_module reads that file and returns a CUDA
    // module. from_module binds it to the typed launch API generated by
    // #[cuda_module].
    let module = load_kernel_module(&ctx, "host_closure")?;
    let typed = kernels::from_module(module)?;

    // Launch with a closure. `factor` is captured and passed to the GPU
    // automatically (scalarised into a kernel parameter).
    let factor = 2.5f32;
    // SAFETY: this raw configuration is fully 1-D, matches index_1d(), and
    // launches one thread per output element. A launch contract can move
    // this proof into the generated safe API.
    unsafe {
        typed.map::<f32, _>(
            stream.as_ref(),
            LaunchConfig::for_num_elems(1024),
            move |x: f32| x * factor,
            &input,
            &mut output,
        )
    }?;

    let result = output.to_host_vec(&stream)?;
    assert!((result[1] - 2.5).abs() < 1e-5);
    println!("PASSED: CUDA-Oxide map closure produced expected results");
    Ok(())
}
```

The device-side pieces are:

- `#[cuda_module] mod kernels { ... }` makes the backend compile the
  `#[kernel]` functions in the module to PTX and generate the host loading and
  launch glue.
- `#[kernel] pub fn map<T: Copy, F: Fn(T) -> T + Copy>(...)` declares a
  generic kernel. The compiler instantiates one PTX function per `(T, F)`
  combination, the same monomorphisation used by C++ templates in Chapter 10.
- `thread::index_1d()` is the fused equivalent of
  `blockIdx.x * blockDim.x + threadIdx.x` from Chapter 3, returned as a typed
  index.
- `DisjointSlice<T>` is a guaranteed-disjoint view of the output. Its
  `get_mut(idx)` method returns a mutable reference to this thread's exclusive
  element. Two threads cannot obtain mutable access to the same slot, making
  the one-thread-per-output pattern a type-level guarantee.
- `if let Some(out_elem) = out.get_mut(idx)` expresses the boundary guard:
  `None` is the out-of-range case.

The host-side pieces are:

- `CudaContext::new(0)` opens the GPU.
- `DeviceBuffer::from_host(&stream, &data)` allocates and copies in one call on
  the given stream.
- `load_kernel_module` reads the PTX file produced by the backend, and
  `kernels::from_module` binds it to the generated typed launch API. The
  `typed.map::<f32, _>(...)` method is generated from the kernel signature, so
  arguments are type-checked against the kernel parameter list. In the
  standalone generic-kernel build, the supported path loads the PTX file
  because a PTX bundle is not yet embedded in the executable. Non-generic
  kernels can use the embedded `kernels::load`.
- `unsafe { ... }` marks the raw launch because `LaunchConfig` is raw data:
  nothing in its type proves that the grid shape matches the kernel's indexing
  assumptions. The `SAFETY` comment states the proof obligation, as in Chapter
  13.

## 14.5 The Safety Progression: `#[launch_contract]`

CUDA-Oxide's plan for the raw `unsafe` launch is the **launch contract**: a
`#[launch_contract(...)]` attribute that moves configuration validation into
generated code. A kernel annotated with a contract receives a checked
`PreparedLaunch` through a safe generated method. Launch dimensions and
resources are validated against the declared contract instead of being an
unverifiable `unsafe` obligation.

This is the project's roadmap in miniature: each `unsafe` block is a known gap
with a planned replacement. The `unsafe` in this chapter is not a licence to
ignore safety; it is documented debt that the project is paying down.

## 14.6 Async: `cuda-async` and `DeviceOperation`

The `cuda-async` crate changes the launch shape for composable asynchronous
work. The explicit `stream:` argument disappears and the launch returns a lazy
`DeviceOperation` that executes when `.sync()` or `.await` is called:

```rust
use cuda_async::device_operation::DeviceOperation;

// Assuming module, input, output come from the cuda-async setup:
let factor = 2.5f32;
let launch = unsafe {
    // SAFETY: the raw launch is 1-D and matches this kernel's index space.
    module.map_async::<f32, _>(
        LaunchConfig::for_num_elems(1024),
        move |x: f32| x * factor,
        &input,
        &mut output,
    )?
};
launch.sync()?;      // or: .await?;
```

The lazy operation lets a program build a graph of GPU work without executing
it, the same idea as CUDA Graphs (Chapter 6, §6.7), expressed as composable
Rust values. `.sync()` blocks; `.await` composes with `async/await` host code.
The capstone uses this shape.

## 14.7 Device-Side Abstractions Beyond `map`

The project documents device-side facilities beyond simple indexing:

- **Shared memory** - typed, scoped allocation within a block;
- **Scoped atomics** - atomic operations with explicit thread scopes, as in
  Chapter 5, with Rust scoping;
- **Barriers** - block and cluster synchronisation;
- **TMA** (Tensor Memory Accelerator) - Hopper's bulk asynchronous copies;
- **Warp/cluster operations** - shuffle-like primitives such as Chapter 8's
  `__shfl_down_sync`, with type-safe masks.

These are in active development. The API may change between revisions; this is
the nature of alpha software.

## 14.8 CUDA-Oxide vs the Alternatives

| Approach | Kernel language | Device safety | Maturity |
|---|---|---|---|
| CUDA C++ (Chapters 3-12) | C++ | None (by hand) | Production |
| Rust host + C++ kernel (Ch. 13) | C++ | Host only | Production |
| CUDA-Oxide (this chapter) | Rust | Type-checked, safe(ish) | Alpha, Linux, nightly |
| `cudarc` nvrtc JIT | C++ string | Host only | Production |

CUDA-Oxide is not yet a production tool for most teams. It is an architecture
preview: a demonstration that Rust can target the GPU without giving up its
guarantees, and a first draft of the safety story SIMT programming needs. Its
pipeline (MIR to Pliron to LLVM to PTX) and abstractions (`DisjointSlice`,
launch contracts, async operations) point toward the shape of CUDA's Rust
future. The underlying principles, type-safe indexing, explicit safety
obligations, and single-source compilation, are the same principles this book
has used since Chapter 3.

## Safe SIMT and Hardware Assumptions

In CUDA C++, the "one thread per output element" rule is enforced by a comment.
Nothing in the language stops a thread from writing `out[i + 1]` or `out[2*i]`;
the compiler accepts it, and the bug appears later as corrupt data or an
illegal memory access. CUDA-Oxide expresses such conventions in the type system
so violations become compile errors.

`DisjointSlice<T>` is the clearest example. Its `get_mut(idx)` returns a
mutable reference to the element owned by this thread and guarantees that no
other thread can obtain a mutable reference to the same element. The "one
thread per output, no races" property that Chapter 3 stated in a comment is now
an API property. `thread::index_1d()` fuses the index formula into one typed
operation, so the formula cannot be mistyped as a hand-written expression can.
Because the kernel is Rust, the borrow checker runs on the device code before
any PTX is generated: a kernel that would create two mutable references to the
same slot never compiles.

`unsafe` remains because launch geometry is still raw data. `LaunchConfig`
describes the number of threads, but nothing in its type proves that this count
matches the kernel's indexing assumptions. The `SAFETY` comment documents the
obligation. The `#[launch_contract]` roadmap shows how this obligation can
become a checked precondition: a kernel declares its contract, and the generated
launch path validates the configuration against it.

The architectural point is that a system is safest when invalid programs cannot
be written, not when it has the most runtime checks. CUDA-Oxide is an early,
incomplete version of that idea: it moves some conventions into types, leaves
others as documented unsafe obligations, and is explicit about the difference.

## Common Pitfalls

- Assuming CUDA-Oxide is production-ready. It is alpha; APIs change between
  revisions. The companion code tracks CUDA-Oxide 0.2.1.
- Treating `DisjointSlice` as a licence to ignore bounds. `get_mut` returns
  `None` for out-of-range indices; forgetting the `if let` guard still skips
  work silently.
- Believing `unsafe` means unchecked. It means "checked by a human through the
  SAFETY comment"; write the comment before writing the launch.
- Porting CUDA C++ idioms verbatim, such as pointer arithmetic, instead of
  using the Rust-native abstractions demonstrated in this chapter.

## Check Your Understanding

<details>
<summary>What does DisjointSlice prevent that a comment in CUDA C++ does not?</summary>

It makes the one-thread-per-output property a type guarantee: two threads
cannot obtain `get_mut` for the same element. In CUDA C++, that invariant is
only a comment; nothing stops a thread from writing any index.
</details>

<details>
<summary>Why is the launch still unsafe if the kernel is safe?</summary>

The kernel's internal indexing may be safe, but the launch configuration is raw
data. Nothing in `LaunchConfig` proves the grid covers the output exactly as
the kernel assumes, so the launch remains an unsafe obligation documented by a
SAFETY comment.
</details>

<details>
<summary>What does #[launch_contract] change about the obligation?</summary>

It moves the geometry proof into generated code. Launch dimensions and
resources are validated against the kernel's declared contract, so the unsafe
obligation becomes a checked precondition instead of a manual comment.
</details>

## Key Takeaways

- CUDA-Oxide is NVIDIA Labs' rustc backend: #[kernel] Rust functions compile to PTX, with no nvcc or DSL.
- The pipeline Rust -> MIR -> Pliron -> LLVM -> PTX keeps rustc's front-end guarantees (ownership, borrow checking) before GPU code is generated.
- DisjointSlice provides the one-thread-per-output, no-races guarantee at the type level.
- LaunchConfig is raw data; launching is unsafe until a #[launch_contract] moves the proof into generated code.
- CUDA-Oxide is alpha, Linux-only, and nightly-only: treat it as an architecture preview, not a production dependency.

## 14.9 Exercises

1. Compare `thread::index_1d()` with the Chapter 3 formula
   `blockIdx.x * blockDim.x + threadIdx.x`. What does the fused abstraction
   prevent?
2. Why is `DisjointSlice<T>`'s `get_mut` the type-level version of the Chapter
   3 "one thread per output, no races" comment?
3. The raw launch is `unsafe` with a `SAFETY` comment; `#[launch_contract]`
   moves the proof into generated code. Explain the difference in terms of the
   obligation, not the syntax.
4. Using the pipeline diagram in §14.2, identify which phases run on the host
   toolchain and which produce device code. Why does borrow checking happen
   before any PTX generation?

## Sources and Further Reading

- NVIDIA Labs, CUDA-Oxide repository: <https://github.com/NVlabs/cuda-oxide>
- NVIDIA, *CUDA C++ Programming Guide*, "Compute Capabilities" for PTX/SASS details: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- Rust Reference, "Inline assembly" and "Unsafe" sections: <https://doc.rust-lang.org/reference/>
