# Foreword

This book is a practical introduction to GPU and parallel programming with
CUDA. It starts with the hardware model, develops the CUDA C++ programming
model, covers the modern C++ and Rust ecosystems, and finishes with profiling,
multi-GPU systems, and a complete image-processing pipeline implemented three
ways.

## GPUs and Parallel Scaling

For decades, the default way to make a program faster was a faster serial
processor: a higher clock, a deeper pipeline, or a larger cache. That path
stopped scaling in the mid-2000s, when clock speeds plateaued and power limits
made further serial scaling impractical. The industry response was
parallelism: many cores, then many threads per core, and then specialised
accelerators. A modern GPU executes trillions of floating-point operations per
second, moves memory at terabytes per second, and keeps far more threads in
flight than a CPU core count. The relevant question is no longer how fast a
single core can go but how much work can be kept in flight at once.

The GPU is a useful teaching target because its performance model is compact.
A small set of principles - the warp, the memory hierarchy, and the roofline -
explains most of the difference between a kernel that runs at a few percent of
peak and one that runs near it.

## The Failure Modes

CUDA hides the hardware behind an API, and the API does not report every
problem. A matrix multiply can run at a fraction of peak when threads in a warp
read columns instead of rows. A reduction can lose counts when threads update a
shared histogram without atomics. A transfer can be twice as slow when host
memory is pageable instead of pinned. None of these failures produces an error
message. They produce a slow benchmark or a subtly wrong result.

This book covers the hardware model, the CUDA C++ programming model, the C++
and Rust ecosystem, and the profiling tools, so these failures are diagnosed
rather than mysterious.

## What You Will Build

The later chapters build toward one project: a complete GPU image-processing
pipeline. The pipeline reads an image, converts it to greyscale, applies a
separable Gaussian blur, runs a Sobel edge detector, and writes the result. It
is implemented three times:

1. in CUDA C++ with documented kernels;
2. with the Thrust/CUB/cuBLAS library ecosystem;
3. in Rust with CUDA-Oxide, NVIDIA's experimental compiler that turns Rust
   kernels into PTX.

The pipeline includes pinned-memory transfers, streamed double buffering, an
occupancy-aware kernel configuration, and reproducible benchmarks.

## Who This Book Is For

This book is for readers who:

- can write C++ or Rust but have used GPUs only through library calls they did
  not fully understand;
- have launched a kernel, seen incorrect output, and did not know whether the
  bug was in index arithmetic, memory layout, or synchronisation;
- want to understand warps, streaming multiprocessors, and the memory hierarchy
  before writing CUDA;
- write Rust and want to know what CUDA-Oxide changes and what it does not;
- do not own a GPU and want to learn on free cloud instances;
- work on software with tight performance budgets and zero tolerance for silent
  correctness failures.

No prior GPU experience is required. C++ or Rust experience and access to a
CUDA-capable machine are sufficient. Terms are defined when they first appear,
and hardware numbers are accompanied by their reasoning.

## Hardware Requirements

The examples target CUDA 12.x and modern NVIDIA compute capabilities, while
remaining portable through PTX. The repository README lists cloud options;
free tiers are sufficient for every example. Where a feature is
architecture-specific, the text says so.

CUDA-Oxide is an experimental, alpha-stage compiler. Its API is evolving and
its examples may change. The chapters that cover it describe the project as it
exists at the time of writing, with code in the style of its documented
examples. Treat those chapters as a map of the territory rather than a
specification.

## How the Book Is Organised

**Part I - Foundations of GPU Computing** (Chapters 1-3) covers the mathematics
of parallelism, GPU hardware, and the CUDA programming model. Later chapters
assume the terms defined here.

**Part II - Writing CUDA C++ Kernels** (Chapters 4-6) covers memory management,
synchronisation and atomics, and streams and events for asynchronous execution.

**Part III - Optimisation & Advanced Patterns** (Chapters 7-9) covers memory
optimisation, reductions and scans, and a step-by-step matrix multiplication.

**Part IV - Modern C++ & The CUDA Ecosystem** (Chapters 10-12) covers RAII,
templates, and C++ idioms; the Thrust, CUB, and cuBLAS libraries; and NVRTC
runtime compilation.

**Part V - Rust, CUDA-Oxide & Safe GPU Programming** (Chapters 13-15b) covers
Rust host code, CUDA-Oxide kernels, the image-processing capstone, and a Jetson
Orin benchmark study.

**Part VI - The Engineering Mindset** (Chapter 16) covers Nsight profiling,
Compute Sanitizer, and reproducible performance engineering.

**Part VII - Systems & Multi-GPU Programming** (Chapters 17-19) covers GPU
systems programming, NVLink and NVSwitch, and NCCL collectives.

## Corrections

If you find a bug in the prose or the code, open an issue or submit a pull
request. The GPU ecosystem changes; the book is intended to change with it.

- *Arpan Pathak*
