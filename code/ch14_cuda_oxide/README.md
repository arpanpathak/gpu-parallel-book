# Chapter 14 - CUDA-Oxide: Kernels in Pure Rust

The generic `#[kernel]` `map` example from the book (14.4), updated to the
current CUDA-Oxide 0.2.1 API.

## Building

CUDA-Oxide is an experimental project with its own build driver:

```bash
# From inside this directory (a standalone CUDA-Oxide project):
cargo oxide run host_closure
```

The command compiles the kernel to PTX, writes `host_closure.ptx` next to
`Cargo.toml`, builds the host binary, and runs it. Expected output:

```text
PASSED: CUDA-Oxide map closure produced expected results
```

> **Why `load_kernel_module` instead of `kernels::load()`?** For a module
> that contains only *generic* kernels, CUDA-Oxide's standalone build does
> not embed a PTX bundle into the executable. `load_kernel_module` reads the
> generated `host_closure.ptx` from the manifest directory instead, which is
> the supported path for standalone generic-kernel projects.

## Requirements

- Linux (tested on Ubuntu 24.04 and Jetson Orin)
- CUDA Toolkit 12.x+
- Rust nightly with `rust-src`, `rustc-dev`, `llvm-tools`
- `cargo-oxide` (installed from the repository)
- Clang + libclang dev headers (for `bindgen`)

> The project is alpha: expect bugs, incomplete features, and API breakage.
> The book (14.8) treats it as an architecture preview, not a production
> dependency.
