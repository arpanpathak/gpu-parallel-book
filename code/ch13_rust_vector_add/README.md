# Chapter 13 - Rust Meets the GPU

Rust host program driving the Chapter 3 vector-add kernel via `cudarc`.

## Building

```bash
# 1. Compile the CUDA kernel to PTX (needs the CUDA toolkit):
nvcc -arch=compute_90 -ptx vector_add.cu -o vector_add.ptx

# 2. Build and run the Rust host:
cargo build --release
./target/release/vector-add
```

`vector_add.cu` is the kernel from Chapter 3 (see
`../ch03_vector_add/vector_add.cu`). The PTX must sit next to the binary, or
embed it with `include_str!` as described in the book (13.5).

## Requirements

- CUDA toolkit (for `nvcc` and the driver)
- Rust toolchain
- The `cudarc` crate, configured with the feature matching your CUDA version
  (see Cargo.toml)
