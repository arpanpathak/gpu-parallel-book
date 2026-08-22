# Chapter 13 - Rust Meets the GPU

Rust host program driving the Chapter 3 vector-add kernel via `cudarc`.

## Building

```bash
# 1. Compile the CUDA kernel to PTX (needs the CUDA toolkit).
#    compute_60 PTX runs on any CUDA 12.x GPU (Pascal or newer) via JIT;
#    use compute_87 on Jetson Orin, compute_80 on A100, compute_90 on H100.
nvcc -arch=compute_60 -ptx vector_add.cu -o vector_add.ptx

# 2. Build and run the Rust host:
cargo build --release
./target/release/vector-add
```

`vector_add.cu` is the kernel from Chapter 13 (extern "C" so the driver can
look it up by name). The PTX is committed so `cargo run` works immediately;
re-generate it with the `nvcc` command above if you change the kernel. The PTX
must be in the working directory from which you launch the binary (or embed it
with `include_str!` as described in the book, 13.5).

## Requirements

- CUDA toolkit (for `nvcc` and the driver)
- Rust toolchain
- The `cudarc` crate, configured with the feature matching your CUDA version
  (see Cargo.toml)
