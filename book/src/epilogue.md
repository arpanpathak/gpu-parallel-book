# Epilogue - The Road Ahead

You have worked from the mathematics of parallelism to a streamed, profiled,
and verified image pipeline implemented three ways. This epilogue briefly
describes where the field is heading.

## Hardware

Each GPU generation moves the ridge point of Chapter 1: more FLOPs, more
bandwidth, and increasingly specialised arithmetic. Tensor cores, introduced
with Volta and central to AI workloads since, execute dense matrix
multiplication in hardware instead of looping over CUDA cores. Hopper added the
Tensor Memory Accelerator (TMA) for bulk asynchronous copies and thread-block
clusters. Blackwell continues the trend. The memory hierarchy, the warp, and
the SM from Chapter 2 still define the machine; the arithmetic units are
becoming more specialised.

The skills in this book are foundational. Tensor-core programming is still
thread-block programming with a different instruction set. TMA is still
coalescing expressed in bulk. When the next specialised unit appears, it will
fit the same model.

## Software

Three currents are visible:

- **The SIMT model is cross-vendor.** NVIDIA's ecosystem (CUDA, cuBLAS,
  Nsight) remains the reference, but SYCL (Khronos's single-source C++ model),
  HIP (AMD's CUDA-compatible API), and wgpu/WebGPU (browser and native Rust)
  expose the same underlying execution model. Grids, warps, coalescing, and
  shared memory translate directly to these APIs.
- **Rust is becoming viable for GPU work.** `cudarc` provides a production-grade
  Rust host (Chapter 13). CUDA-Oxide (Chapter 14) is an early attempt to bring
  Rust's guarantees to the kernel itself. Both are young. They point in the
  same direction: many GPU failure modes originate in the host language, and
  languages that remove those failure modes are likely to become more common
  for GPU hosts.
- **Libraries continue to absorb implementation work.** Every year more of the
  hard work moves into tuned libraries (Chapter 11). The engineers who use
  those libraries effectively are not the ones who memorised the APIs; they are
  the ones who can read a profiler, explain why a kernel is memory-bound, and
  decide when a custom kernel is worth writing.

## The Discipline

The most durable idea in this book is the loop of Chapter 16: measure, profile,
hypothesise, change one thing, re-measure, and verify. Hardware changes,
languages change, and libraries change. The loop does not.

## The Invitation

This book is published on GitHub Pages and is open to pull requests. If a
kernel is unclear, a claim is unmeasured, or a chapter misses a concept that
confused you, open an issue.

Write kernels that are fast, correct, and understood.

- *Arpan Pathak*
