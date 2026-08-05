# Code Companion for "CUDA Kernels: GPU & Parallel Programming from First Principles"

Every runnable code block in the book lives here, organised by chapter. The
files are the exact code printed in the book, including the line-by-line
comments; each directory builds standalone.

## Chapter map

| Directory | Chapter | What it contains | Build with |
|---|---|---|---|
| [`ch03_vector_add/`](ch03_vector_add/) | Ch 3 - The CUDA Programming Model | The first complete program: vector addition | `nvcc -arch=sm_90 vector_add.cu -o vector_add` |
| [`ch05_histogram/`](ch05_histogram/) | Ch 5 - Synchronisation, Atomics & Races | Privatised histogram + a shared-memory spinlock | `nvcc -arch=sm_90 histogram.cu -o histogram` |
| [`ch08_reduction/`](ch08_reduction/) | Ch 8 - Reduction, Scan & Histogram | Warp-shuffle reduction, Blelloch scan, privatised histogram | `nvcc -arch=sm_90 reduction.cu -o reduction` |
| [`ch09_sgemm/`](ch09_sgemm/) | Ch 9 - Optimised Matrix Multiplication | Shared-memory tiled and register-tiled SGEMM kernels | `nvcc -arch=sm_90 sgemm.cu -o sgemm` |
| [`ch10_device_buffer/`](ch10_device_buffer/) | Ch 10 - Modern C++ for CUDA | RAII `DeviceBuffer<T>` and error-handling helpers | `nvcc -std=c++20 -arch=sm_90 test.cpp -o test` |
| [`ch13_rust_vector_add/`](ch13_rust_vector_add/) | Ch 13 - Rust Meets the GPU | Rust host driving a CUDA kernel via `cudarc` | `cargo build --release` (needs CUDA toolkit) |
| [`ch14_cuda_oxide/`](ch14_cuda_oxide/) | Ch 14 - CUDA-Oxide | The `#[kernel]` map example in pure Rust | `cargo oxide run host_closure` |
| [`ch15_capstone/`](ch15_capstone/) | Ch 15 - Capstone Pipeline | The full RGB -> blur -> Sobel -> histogram pipeline | `nvcc -arch=sm_90 pipeline.cu -o pipeline` |

## Requirements

- **CUDA Toolkit 12.x** for all C/CUDA targets (`nvcc`).
- **Rust nightly + `cargo-oxide`** for the CUDA-Oxide example (Ch 14) - see
  the CUDA-Oxide repository for setup.
- **`cudarc`** for the Rust host example (Ch 13) - it links against the CUDA
  driver at build time.

No NVIDIA GPU on your machine? Use a free cloud GPU (see the book's foreword
and the repository README).

## How the code relates to the book

Each file is the book's code verbatim. The comments in each file are the
reasoning audit trail: index arithmetic, memory accesses, and synchronisation
rationale are all stated in the comments, exactly as CODING_STANDARDS.md
requires.

If a file fails to compile or produces wrong results, open an issue - that is
a bug in the book.
