# Epilogue - The Road Ahead


You have worked from the mathematics of parallelism to a streamed, profiled,
verified image pipeline, written three ways. What follows is a short look at
where the field is heading.

## The Hardware

Every generation of GPU has moved the ridge point of Chapter 1: more FLOPs,
more bandwidth, and - most consequentially - *specialised arithmetic*.
Tensor cores, introduced with Volta and central to every AI workload since,
are dense matrix-multiply units that execute in one instruction what a
CUDA-core loop would take hundreds of cycles to do. Hopper added the TMA
(bulk asynchronous copies) and thread-block clusters; Blackwell continues the
trend. The memory hierarchy, the warp, and the SM from Chapter 2 still define
the machine, but the arithmetic units are becoming more specialised.

The skills in this book are foundational rather than obsolete. Tensor-core
programming is still thread-block programming with a different instruction
set; TMA is still coalescing, expressed in bulk. When the next specialised
unit appears, you will recognise its shape.

## The Software

Three currents are visible today:

- **CUDA is here to stay, and so is its competition.** NVIDIA's ecosystem
  (CUDA, cuBLAS, Nsight) remains the reference, but the cross-vendor world is
  real: **SYCL** (Khronos's single-source C++ model), **HIP** (AMD's CUDA-
  compatible API), and **wgpu/WebGPU** (browser and native Rust). The
  programming model you learned - grids, warps, coalescing, shared memory  - 
  translates directly to all of them, because they all expose the same SIMT
  execution model.
- **Rust is becoming viable.** `cudarc` gives Rust a production-grade host
  (Chapter 13). **CUDA-Oxide** (Chapter 14) is the first credible attempt to
  bring Rust's guarantees to the kernel itself. Both are young; both point in
  the same direction: GPU failure modes are mostly host-language failure modes,
  and languages that remove those failure modes are likely to become the
  default host choice for GPU work.
- **The libraries are absorbing more of the hard work, which is fine.** Every
  year, more of the hard work moves into tuned libraries (Chapter 11). The
  engineers who *use* those libraries effectively are not the ones who
  memorised the API - they are the ones who can read the profiler, explain why
  a kernel is memory-bound, and know when a custom kernel is actually worth
  writing. That is what this book trained you to do.

## The Discipline

The most durable idea in this book is the loop of Chapter 16: measure,
profile, hypothesise, change one thing, re-measure, verify. Hardware changes,
languages change, libraries change - the loop does not.

## The Invitation

This book is a living document, published on GitHub Pages and open to pull
requests. If a kernel is unclear, if a claim is unmeasured, if a chapter is
missing the concept that confused you - open an issue.

Write kernels that are fast, correct, and understood.

- *Arpan Pathak*
