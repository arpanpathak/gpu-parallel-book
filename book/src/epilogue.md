# Epilogue - The Road Ahead

> *"You now understand a machine that did not exist twenty years ago and will
> not exist, in its current form, twenty years from now. That is the nature of
> the field. It is also the point of this book."*

You have travelled from the mathematics of parallelism to a streamed,
profiled, verified image pipeline, written three ways. Before you close the
book, it is worth looking at where the road goes - not as prophecy, but as a
map of the territory you are now equipped to navigate.

## The Hardware

Every generation of GPU has moved the ridge point of Chapter 1: more FLOPs,
more bandwidth, and - most consequentially - *specialised arithmetic*.
Tensor cores, introduced with Volta and central to every AI workload since,
are a different kind of computer inside the GPU: dense matrix-multiply units
that execute in one instruction what a CUDA-core loop would take hundreds of
cycles to do. Hopper added the TMA (bulk asynchronous copies) and
thread-block clusters; Blackwell continues the trend. The lesson of Chapter 2
still holds - the memory hierarchy, the warp, the SM - but the *arithmetic*
landscape keeps splitting into specialised lanes.

The consequence for you: the skills in this book are not obsolete, they are
*foundational*. Tensor-core programming is still thread-block programming with
a different instruction set; TMA is still coalescing, expressed in bulk. When
the next specialised unit appears, you will recognise its shape.

## The Software

Three currents are visible today:

- **CUDA is here to stay, and so is its competition.** NVIDIA's ecosystem
  (CUDA, cuBLAS, Nsight) remains the reference, but the cross-vendor world is
  real: **SYCL** (Khronos's single-source C++ model), **HIP** (AMD's CUDA-
  compatible API), and **wgpu/WebGPU** (browser and native Rust). The
  programming model you learned - grids, warps, coalescing, shared memory  - 
  translates directly to all of them, because they are all SIMT machines
  wearing different clothes.
- **Rust is arriving.** `cudarc` gives Rust a production-grade host (Chapter
  13). **CUDA-Oxide** (Chapter 14) is the first credible attempt to bring
  Rust's guarantees to the kernel itself. Both are young; both point in the
  same direction - that the GPU's failure modes are, at bottom, *host-language*
  failure modes, and the languages that eliminate those failure modes will
  win the mindshare of the next generation of GPU engineers.
- **The libraries are winning, and that is fine.** Every year, more of the
  hard work moves into tuned libraries (Chapter 11). The engineers who *use*
  those libraries effectively are not the ones who memorised the API - they
  are the ones who can read the profiler, explain why a kernel is
  memory-bound, and know when a custom kernel is actually worth writing. That
  is exactly what this book trained you to do.

## The Discipline

The most durable thing in this book is not a kernel. It is the loop of
Chapter 16: measure, profile, hypothesise, change one thing, re-measure,
verify. Hardware changes, languages change, libraries change - the loop does
not. It is the same discipline a surgeon brings to an operation, a pilot to an
approach, and an engineer to a machine they respect. The GPU is a machine
worth respecting: it is fast, it is precise, and it will do exactly what you
told it, including the wrong things.

## The Invitation

This book is a living document, published on GitHub Pages and open to pull
requests. If a kernel is unclear, if a claim is unmeasured, if a chapter is
missing the concept that confused you - open an issue. The road ahead is
paved by the people who walk it, and you are now walking it with the lights
on.

The GPU is waiting. It has been waiting since Chapter 1. Go make it do
something that is fast, correct, and understood.

- *Arpan Pathak*
