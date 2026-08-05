# Chapter 14 - CUDA-Oxide: Kernels in Pure Rust

The generic `#[kernel]` `map` example from the book (14.4), verbatim.

## Building

CUDA-Oxide is an experimental project with its own build driver:

```bash
# From inside the NVlabs/cuda-oxide checkout, the example runs as:
cargo oxide run host_closure

# Or, for your own project bootstrapped with:
nix run github:NVlabs/cuda-oxide#new my-project
```

See the [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) repository
for installation (nightly Rust, CUDA 12.x+, clang/libclang, Linux).

## Requirements

- Linux (tested on Ubuntu 24.04)
- CUDA Toolkit 12.x+
- Rust nightly with `rust-src`, `rustc-dev`, `llvm-tools`
- `cargo-oxide` (installed from the repository)
- Clang + libclang dev headers (for `bindgen`)

> The project is alpha: expect bugs, incomplete features, and API breakage.
> The book (14.8) treats it as an architecture preview, not a production
> dependency.
