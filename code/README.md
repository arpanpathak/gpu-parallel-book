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
| [`ch12_nvrtc/`](ch12_nvrtc/) | Ch 12 - NVRTC & the Driver API | A polynomial kernel written as text at run time, compiled with NVRTC to PTX and cubin, loaded and launched with the driver API, timed against the generic kernel; the JIT cache (Rust: `oxide/ch12_driver`) | `nvcc -arch=native nvrtc.cu -o nvrtc -lnvrtc -lcuda` |
| [`ch13_rust_vector_add/`](ch13_rust_vector_add/) | Ch 13 - Rust Meets the GPU | Rust host driving a CUDA kernel via `cudarc` | `cargo build --release` (needs CUDA toolkit) |
| [`ch14_cuda_oxide/`](ch14_cuda_oxide/) | Earlier draft of Part V | The `#[kernel]` map example in pure Rust | `cargo oxide run host_closure` |
| [`ch15_capstone/`](ch15_capstone/) | Earlier draft of the capstone | The full RGB -> blur -> Sobel -> histogram pipeline | `nvcc -arch=compute_60 pipeline.cu -o pipeline` |
| [`cuda/ch14_map_saxpy/`](cuda/ch14_map_saxpy/) and [`oxide/ch14_map_saxpy/`](oxide/ch14_map_saxpy/) | Ch 14 - Map and SAXPY | SAXPY 1 thread per element, a generic map with a lambda or closure, grid-stride loops over a sweep of grid sizes | build line at the top of `saxpy.cu`; `cargo oxide run` in the crate |
| [`cuda/ch15_dot_product/`](cuda/ch15_dot_product/) and [`oxide/ch15_dot_product/`](oxide/ch15_dot_product/) | Ch 15 - Reduction and the Dot Product | Map then reduce, a fused tree in shared memory, warp shuffles, atomic combining, exact tests | build line at the top of `dot.cu`; `cargo oxide run` in the crate |
| [`cuda/ch16_matmul_naive/`](cuda/ch16_matmul_naive/) and [`oxide/ch16_matmul_naive/`](oxide/ch16_matmul_naive/) | Ch 16 - Matrix Multiplication, One Thread per Entry | The 4 x 4 worked example, the direct kernel in f32 and f64, a width fixed at compile time | build line at the top of `matmul_naive.cu`; `cargo oxide run` in the crate |
| [`cuda/ch17_matmul_tiled/`](cuda/ch17_matmul_tiled/) and [`oxide/ch17_matmul_tiled/`](oxide/ch17_matmul_tiled/) | Ch 17 - Tiled Matrix Multiplication | Tiles of 8, 16 and 32 in shared memory, exact checks at sizes that are not multiples of the tile | build line at the top of `matmul_tiled.cu`; `cargo oxide run` in the crate |
| [`cuda/ch18_matmul_register_tiled/`](cuda/ch18_matmul_register_tiled/) and [`oxide/ch18_matmul_register_tiled/`](oxide/ch18_matmul_register_tiled/) | Ch 18 - Register Tiling | Thread tiles of 1 x 1 to 8 x 8 against the Chapter 17 kernel | build line at the top of `matmul_register_tiled.cu`; `cargo oxide run` in the crate |
| [`cuda/ch19_stencil/`](cuda/ch19_stencil/) and [`oxide/ch19_stencil/`](oxide/ch19_stencil/) | Ch 19 - Stencils and Jacobi Iteration | The 6 x 6 plate by hand, Jacobi steps on 2 buffers, convergence, a copy kernel for comparison | build line at the top of `stencil.cu`; `cargo oxide run` in the crate |
| [`cuda/ch20_stencil_tiled/`](cuda/ch20_stencil_tiled/) and [`oxide/ch20_stencil_tiled/`](oxide/ch20_stencil_tiled/) | Ch 20 - Tiled Stencils with Halo Cells | 2 ways of loading a tile with its halo, temporal blocking with several steps per load | build line at the top of `stencil_tiled.cu`; `cargo oxide run` in the crate |
| [`cuda/ch21_scan_compaction/`](cuda/ch21_scan_compaction/) and [`oxide/ch21_scan_compaction/`](oxide/ch21_scan_compaction/) | Ch 21 - Scan and Stream Compaction | Hillis-Steele, Blelloch and warp-shuffle scans of many blocks, compaction in 3 passes and with warp votes | build line at the top of `scan_compaction.cu`; `cargo oxide run` in the crate |
| [`cuda/ch22_radix_sort_histogram/`](cuda/ch22_radix_sort_histogram/) and [`oxide/ch22_radix_sort_histogram/`](oxide/ch22_radix_sort_histogram/) | Ch 22 - Radix Sort and Histograms | Histograms with global, private and warp-aggregated bins, radix sorts by 1 bit and by 4 bits | build line at the top of `radix_sort_histogram.cu`; `cargo oxide run` in the crate |
| [`cuda/ch23_softmax_layernorm/`](cuda/ch23_softmax_layernorm/) and [`oxide/ch23_softmax_layernorm/`](oxide/ch23_softmax_layernorm/) | Ch 23 - Softmax and Layer Normalization | 3-pass, online and fused softmax, layer normalization with sums of squares and with Welford's method | build line at the top of `softmax_layernorm.cu`; `cargo oxide run` in the crate |
| [`cuda/ch24_capstone/`](cuda/ch24_capstone/) and [`oxide/ch24_capstone/`](oxide/ch24_capstone/) | Ch 24 - Capstone: An Image Pipeline | Gray, histogram equalisation, separable blur and Sobel, checked against the CPU, a batch on 1 to 3 streams | build line at the top of `capstone.cu`; `cargo oxide run` in the crate |
| [`ch16_profiling/`](ch16_profiling/) | Ch 25 - Profiling, Debugging & Performance Engineering | 4 bugs for Compute Sanitizer, the GPU's limits and a warm-up curve, latencies with `clock64()`, tolerances in ULPs (Rust: `oxide/ch16_latency`) | `nvcc -arch=native -lineinfo <name>.cu -o <name>` for `bugs`, `measure`, `latency`, `verify` |
| [`ch17_systems/`](ch17_systems/) | Ch 26 - GPU Systems & MMIO | What the runtime reports about the GPU, CPU speed over 4 memory types, a work ring with a doorbell between 2 threads (Rust: `oxide/ch17_systems`) | `nvcc -arch=native -O3 systems.cu -o systems` |
| [`ch18_peer/`](ch18_peer/) | Ch 27 - NVLink & NVSwitch | GPU count, the peer-access matrix, UVA, and a copy and a copy kernel from GPU 0 to GPU 1, or within GPU 0 on 1 GPU (Rust: `oxide/ch18_peer`) | `nvcc -arch=native peer.cu -o peer` |
| [`ch19_collectives/`](ch19_collectives/) | Ch 28 - NCCL & Collective Communication | Ring and tree all-reduce simulated on 1 GPU, k ranks as slices of 1 buffer, checked exact and timed (Rust: `oxide/ch19_collectives`) | `nvcc -arch=native allreduce.cu -o allreduce` |

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
