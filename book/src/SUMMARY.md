# Summary

[Cover](cover.md)
[About this book](about-this-book.md)
[About the cover](about-the-cover.md)

---

# Part I  -  The Foundations of GPU Computing

- [Chapter 1: The Mathematics of Parallelism](chapters/part1_foundations/ch01_mathematics_of_parallelism.md)
- [Chapter 2: GPU Hardware from First Principles](chapters/part1_foundations/ch02_gpu_hardware_first_principles.md)
- [Chapter 3: The CUDA Programming Model](chapters/part1_foundations/ch03_cuda_programming_model.md)

# Part II  -  Writing CUDA C++ Kernels

- [Chapter 4: Memory Management & Data Movement](chapters/part2_cuda_cpp_kernels/ch04_memory_management_data_movement.md)
- [Chapter 5: Synchronization, Atomics & Race Conditions](chapters/part2_cuda_cpp_kernels/ch05_synchronisation_atomics_races.md)
- [Chapter 6: Streams, Events & Asynchronous Execution](chapters/part2_cuda_cpp_kernels/ch06_streams_events_asynchrony.md)

# Part III  -  Optimization & Advanced Patterns

- [Chapter 7: Memory Optimization Deep Dive](chapters/part3_optimisation/ch07_memory_optimisation_deep_dive.md)
- [Chapter 8: Reduction, Scan & Histogram](chapters/part3_optimisation/ch08_reduction_scan_histogram.md)
- [Chapter 9: Optimized Matrix Multiplication](chapters/part3_optimisation/ch09_optimised_matrix_multiplication.md)

# Part IV  -  Modern C++ & The CUDA Ecosystem

- [Chapter 10: Modern C++ for CUDA](chapters/part4_modern_cpp_ecosystem/ch10_modern_cpp_for_cuda.md)
- [Chapter 11: The Library Ecosystem  -  Thrust, CUB & cuBLAS](chapters/part4_modern_cpp_ecosystem/ch11_library_ecosystem_thrust_cub_cublas.md)
- [Chapter 12: NVRTC, Runtime Compilation & the Driver API](chapters/part4_modern_cpp_ecosystem/ch12_nvrtc_runtime_compilation_driver_api.md)

# Part V  -  Parallel Algorithms in Rust and CUDA C++

- [Chapter 13: Your First Rust Kernel](chapters/part5_oxide/ch13_your_first_rust_kernel.md)
- [Elementwise Operations and Reductions](chapters/part5_oxide/elementwise_reductions.md)
    - [Chapter 14: Map and SAXPY](chapters/part5_oxide/ch14_map_saxpy.md)
    - [Chapter 15: Reduction and the Dot Product](chapters/part5_oxide/ch15_dot_product.md)
- [Matrix Multiplication](chapters/part5_oxide/matrix_multiplication.md)
    - [Chapter 16: Matrix Multiplication, One Thread per Entry](chapters/part5_oxide/ch16_matmul_naive.md)
    - [Chapter 17: Tiled Matrix Multiplication](chapters/part5_oxide/ch17_matmul_tiled.md)
    - [Chapter 18: Register Tiling, Several Entries per Thread](chapters/part5_oxide/ch18_matmul_register_tiled.md)
- [Stencils](chapters/part5_oxide/stencils.md)
    - [Chapter 19: Stencils and Jacobi Iteration](chapters/part5_oxide/ch19_stencil.md)
    - [Chapter 20: Tiled Stencils with Halo Cells](chapters/part5_oxide/ch20_stencil_tiled.md)
- [Scan-Based Algorithms](chapters/part5_oxide/scan_algorithms.md)
    - [Chapter 21: Scan and Stream Compaction](chapters/part5_oxide/ch21_scan_compaction.md)
    - [Chapter 22: Radix Sort and Histograms](chapters/part5_oxide/ch22_radix_sort_histogram.md)
- [Normalization](chapters/part5_oxide/normalization.md)
    - [Chapter 23: Softmax and Layer Normalization](chapters/part5_oxide/ch23_softmax_layernorm.md)
- [Chapter 24: Capstone: An Image Pipeline](chapters/part5_oxide/ch24_capstone.md)

# Part VI  -  The Engineering Mindset

- [Chapter 25: Profiling, Debugging & Performance Engineering](chapters/part6_engineering/ch25_profiling_debugging_performance.md)

# Part VII  -  Systems & Multi-GPU Programming

- [Chapter 26: GPU Systems Programming & Memory-Mapped I/O](chapters/part7_systems_multi_gpu/ch26_gpu_systems_mmio.md)
- [Chapter 27: NVLink & NVSwitch - The Multi-GPU Interconnect](chapters/part7_systems_multi_gpu/ch27_nvlink_nvswitch.md)
- [Chapter 28: NCCL & Multi-GPU Collective Communication](chapters/part7_systems_multi_gpu/ch28_nccl_collectives.md)

# Earlier drafts (being replaced by Part V)

- [Chapter 13: Rust Meets the GPU](chapters/part5_rust_cuda_oxide/ch13_rust_meets_the_gpu.md)
- [Chapter 14: CUDA-Oxide  -  Kernels in Pure Rust](chapters/part5_rust_cuda_oxide/ch14_cuda_oxide_kernels_in_pure_rust.md)
- [Chapter 15: Capstone  -  The GPU Image Processing Pipeline](chapters/part5_rust_cuda_oxide/ch15_capstone_gpu_image_processing_pipeline.md)
- [Chapter 15b: Benchmarking CUDA-Oxide on Jetson Orin](chapters/part5_rust_cuda_oxide/ch15b_benchmarking_cuda_oxide_jetson.md)

---

[Epilogue  -  The Road Ahead](epilogue.md)
[Appendix A  -  CUDA API Reference](appendix_a_cuda_api_reference.md)
[Appendix B  -  Mathematical Notation Reference](appendix_b_math_notation.md)
[Appendix C  -  Recommended Reading & Tools](appendix_c_reading.md)
