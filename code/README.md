# Code Companion for "GPU Architecture & Parallel Programming from First Principles"

Every runnable code block in the book lives here, organised by chapter. The
files are the exact code printed in the book, including the line-by-line
comments; each directory builds standalone.

## Chapter map

| Directory | Chapter | What it contains | Build with |
|---|---|---|---|
| [`ch03_vector_add/`](ch03_vector_add/) | Ch 3 - The CUDA Programming Model | Vector addition, and `whoami` (each thread prints its index) | `nvcc -arch=native vector_add.cu -o vector_add` and `nvcc -arch=native whoami.cu -o whoami` |
| [`ch04_memory/`](ch04_memory/) | Ch 4 - Memory Management & Data Movement | Pageable vs pinned copy speed, unified memory, zero-copy (Rust: `oxide/ch04_transfers`) | `nvcc -arch=native transfers.cu -o transfers` |
| [`ch05_sync/`](ch05_sync/) | Ch 5 - Synchronisation, Atomics & Races | A race on 1 counter, global vs privatised histogram, missing barriers, divergence, float atomic sums (Rust: `oxide/ch05_sync`) | `nvcc -arch=native sync.cu -o sync` |
| [`ch06_streams/`](ch06_streams/) | Ch 6 - Streams, Events & Asynchrony | Asynchronous launch timed with events, a pipeline of chunks on 2 streams, a CUDA graph of 1,000 launches (Rust: `oxide/ch06_streams`) | `nvcc -arch=native streams.cu -o streams` |
| [`ch07_memory/`](ch07_memory/) | Ch 7 - Memory Optimisation | Strided reads and 32-byte sectors, `float4` copies, naive, tiled and padded transposes (Rust: `oxide/ch07_memory`) | `nvcc -arch=native memory.cu -o memory` |
| [`ch08_reduction/`](ch08_reduction/) | Ch 8 - Reduction, Scan & Histogram | A sum in 4 measured stages (tree, coarsened, shuffles, `float4`), a scan of many blocks, a histogram with 16-byte loads (Rust: `oxide/ch08_reduce`) | `nvcc -arch=native reduce.cu -o reduce` |
| [`ch09_sgemm/`](ch09_sgemm/) | Ch 9 - Optimised Matrix Multiplication | 1 thread per element (by column, by row), shared tiles of 16 and 32, 4 x 4 register tiles with `__launch_bounds__`, against cuBLAS (Rust: `oxide/ch09_sgemm`) | `nvcc -arch=native sgemm.cu -o sgemm -lcublas` |
| [`ch10_modern/`](ch10_modern/) | Ch 10 - Modern C++ for CUDA | 3 mistakes the C API lets through (`raw.cu`); a RAII `DeviceBuffer<T>`, a `check()` without a macro, template kernels, `__host__ __device__` functions, `cuda::std::span`, `cuda::atomic_ref` and concepts (`modern.cu`) (Rust: `oxide/ch10_modern`) | `nvcc -arch=native raw.cu -o raw` and `nvcc -std=c++20 --extended-lambda -arch=native modern.cu -o modern` |
| [`ch11_libraries/`](ch11_libraries/) | Ch 11 - Thrust, CUB & cuBLAS | Thrust and CUB sums and scans against Chapter 8, sorting, row-major cuBLAS SGEMM in FP32 and TF32, cuFFT and cuRAND (Rust: `oxide/ch11_blocks`) | `nvcc -arch=native -O2 reduce_scan.cu -o reduce_scan`, likewise `sort.cu`, `gemm.cu -lcublas`, `fft_rand.cu -lcurand -lcufft` |
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
