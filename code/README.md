# Code Companion for "CUDA Kernels: GPU & Parallel Programming from First Principles"

Every runnable code block in the book lives here, organised by chapter. The
files are the exact code printed in the book, including the line-by-line
comments; each directory builds standalone.

## Chapter map

| Directory | Chapter | What it contains | Build with |
|---|---|---|---|
| [`ch03_vector_add/`](ch03_vector_add/) | Ch 3 - The CUDA Programming Model | The first complete program: vector addition | `nvcc -arch=compute_60 vector_add.cu -o vector_add` |
| [`ch05_histogram/`](ch05_histogram/) | Ch 5 - Synchronisation, Atomics & Races | Privatised histogram + a shared-memory spinlock | `nvcc -arch=compute_60 histogram.cu main.cu -o histogram` |
| [`ch08_reduction/`](ch08_reduction/) | Ch 8 - Reduction, Scan & Histogram | Tree/shuffle reduction, Blelloch scan, privatised histogram | `nvcc -rdc=true -arch=compute_60 reduction.cu main.cu -o reduction` |
| [`ch09_sgemm/`](ch09_sgemm/) | Ch 9 - Optimised Matrix Multiplication | Naive, coalesced, tiled and register-tiled SGEMM kernels | `nvcc -arch=compute_60 sgemm.cu main.cu -o sgemm` |
| [`ch10_device_buffer/`](ch10_device_buffer/) | Ch 10 - Modern C++ for CUDA | RAII `DeviceBuffer<T>` and error-handling helpers | `nvcc -std=c++20 -arch=compute_60 test.cu -o test` |
| [`ch11_library_examples/`](ch11_library_examples/) | Ch 11 - Thrust, CUB & cuBLAS | Thrust transform/reduce, CUB `BlockReduce`, row-major cuBLAS SGEMM | See that directory's per-file headers |
| [`ch12_nvrtc/`](ch12_nvrtc/) | Ch 12 - NVRTC & the Driver API | Runtime PTX compilation and driver-API launch | `nvcc -arch=compute_60 main.cu -o nvrtc_example -lnvrtc -lcuda` |
| [`ch13_rust_vector_add/`](ch13_rust_vector_add/) | Ch 13 - Rust Meets the GPU | Rust host driving a CUDA kernel via `cudarc` | `cargo build --release` (needs CUDA toolkit) |
| [`ch14_cuda_oxide/`](ch14_cuda_oxide/) | Ch 14 - CUDA-Oxide | The `#[kernel]` map example in pure Rust | `cargo oxide run host_closure` |
| [`ch15_capstone/`](ch15_capstone/) | Ch 15 - Capstone Pipeline | The full RGB -> blur -> Sobel -> histogram pipeline | `nvcc -arch=compute_60 pipeline.cu -o pipeline` |

**Architecture flags.** `-arch=compute_60` produces portable PTX that the
driver JIT-compiles on any CUDA 12.x GPU (Pascal or newer), including T4,
P100, A100, Jetson Orin and H100. For native SASS use `-arch=sm_87` on Jetson
Orin, `-arch=sm_80` on A100, or `-arch=sm_90` on H100.

## Requirements

- **CUDA Toolkit 12.x** for all C/CUDA targets (`nvcc`). Every example has
  been validated on Jetson Orin (sm_87) with CUDA 12.6; the portable
  `-arch=compute_60` builds also run on T4, P100, A100 and H100.
- **Rust nightly + `cargo-oxide`** for the CUDA-Oxide example (Ch 14) - see
  the CUDA-Oxide repository for setup.
- **`cudarc`** for the Rust host example (Ch 13) - it links against the CUDA
  driver at build time.

No NVIDIA GPU on your machine? Use a free cloud GPU (see the book's foreword
and the repository README).

## Validating everything

Run [`./validate.sh`](validate.sh) to build and run every example:

```bash
./validate.sh            # portable compute_60 PTX
ARCH=sm_87 ./validate.sh # native SASS on Jetson Orin
```

## How the code relates to the book

Each file is the book's code verbatim. The comments in each file are the
reasoning audit trail: index arithmetic, memory accesses, and synchronisation
rationale are all stated in the comments, exactly as CODING_STANDARDS.md
requires.

If a file fails to compile or produces wrong results, open an issue - that is
a bug in the book.
