# Foreword

This is not a tutorial. This is a confession.

Every kernel in this book was written after a performance bug. Every memory
diagram was drawn after a bank conflict, a misaligned access, or a silent race
condition produced a wrong result at two in the morning. This book is the scar
tissue of those battles, organised into a curriculum so that you do not have to
earn every scar yourself.

## Why This Book Exists

Modern software engineering has developed a strange relationship with the GPU.
We treat it as a magic box: a library call here, a framework call there, and
suddenly our training loop is twenty times faster. The library does the hard
part, we tell ourselves, and we never look inside. And that is true — until it
is not.

The abstraction leaks at the worst possible moments. Your matrix multiplication
runs at 3% of peak because the threads in a warp read columns instead of rows.
Your reduction silently drops half of the data because threads in the same warp
diverged across a `__syncthreads()`. Your latency doubles because the memory
allocation was pageable instead of pinned. None of these failures produce an
error message. They produce a benchmark that is embarrassingly slow, or a
result that is subtly, catastrophically wrong.

This book is my attempt to build the bridge between "the library works" and "I
understand why it works". The bridge has three lanes: the **hardware** model,
the **CUDA C++** programming model, and the modern ecosystem of **C++ idioms,
Rust and CUDA-Oxide** that now surrounds the GPU.

## What You Will Build

Every chapter builds toward one project: **a complete GPU image-processing
pipeline** — read an image, convert it to greyscale, apply a separable Gaussian
blur, run a Sobel edge detector, and write the result — implemented three
times:

1. in **CUDA C++** with hand-written, fully commented kernels;
2. with the **Thrust/CUB/cuBLAS** library ecosystem;
3. in **pure Rust** with NVIDIA's experimental **CUDA-Oxide** compiler, which
   turns idiomatic Rust into PTX.

You will not build a toy. You will build the same pipeline a camera vendor
would ship, complete with pinned-memory transfers, streamed double buffering,
an occupancy-tuned kernel configuration, and reproducible benchmarks. When you
have finished, you will be able to look at any CUDA kernel — including the ones
inside the libraries you already use — and explain, line by line, what it does
and why it is fast.

## Who This Book Is For

You should read this book if:

- You can write C++ or Rust, but every GPU program you have written so far was
  a library call you did not fully understand.
- You have launched a kernel, seen it produce garbage, and had no idea whether
  the bug was in your index arithmetic, your memory layout, or your
  synchronisation.
- You suspect that most GPU tutorials skip the hardware model and want to
  understand the primitives — the warp, the streaming multiprocessor, the
  memory hierarchy — before touching a single CUDA API.
- You write Rust and want to know what CUDA-Oxide changes, and what it does not.
- You ship software whose performance budget is measured in microseconds and
  whose correctness budget is zero.

You do not need prior GPU experience. You need to be willing to sit with the
hardware model. I will not hand-wave the memory hierarchy. Every term is
defined when it first appears; every primitive — every type, every built-in
variable, every API call — is described before it is used. Where the book
refers to a number (register counts, memory bandwidths, transaction sizes), it
gives you the reasoning behind it, not just the number.

## The Structure

The book is organised into six parts:

**Part I — Foundations of GPU Computing** (Chapters 1–3) covers the mathematics
of parallelism, the GPU hardware model, and the CUDA programming model. Read
this part carefully; every later chapter assumes the primitives defined here.

**Part II — Writing CUDA C++ Kernels** (Chapters 4–6) covers memory management,
synchronisation, atomics, and asynchronous execution with streams and events.

**Part III — Optimisation & Advanced Patterns** (Chapters 7–9) covers memory
optimisation, the canonical parallel algorithms (reduction, scan, histogram),
and a complete, step-by-step optimisation of matrix multiplication.

**Part IV — Modern C++ & The CUDA Ecosystem** (Chapters 10–12) covers RAII
wrappers, templates and modern C++ idioms, the Thrust/CUB/cuBLAS libraries, and
runtime compilation with NVRTC.

**Part V — Rust, CUDA-Oxide & Safe GPU Programming** (Chapters 13–15) covers
Rust host code driving CUDA kernels, NVIDIA's experimental CUDA-Oxide compiler
for writing kernels in pure Rust, and the image-processing capstone.

**Part VI — The Engineering Mindset** (Chapter 16) covers profiling with Nsight
Compute, debugging with Compute Sanitizer, and reproducible performance
engineering.

## A Note on the Coding Standards

This project has a constitution. You will find it as `CODING_STANDARDS.md` in
the repository root. It is not a suggestion. Every code block in this book
follows it:

- Every primitive is explained before it is used.
- No magic numbers — if it is not `0`, `1`, `-1`, or a power of two required by
  the CUDA API, it gets a named constant.
- Every kernel is commented line by line.
- Every CUDA API call that can fail is checked.
- Every `__syncthreads()` and every atomic carries a comment stating which data
  it protects and why.

## A Note on Hardware and Honesty

The examples in this book target the CUDA 12.x toolkit and are written against
the compute capability of modern NVIDIA GPUs (Ada and Hopper architectures,
compute capability 8.x and 9.0). Where a feature is architecture-specific, the
book says so explicitly.

I must also be honest about the tooling. NVIDIA's CUDA-Oxide is an
experimental, alpha-stage compiler; its API is evolving and its syntax may
change. The chapters that cover it describe the project as it exists today,
with code written in the style of its documented examples. Treat those chapters
as a map of the territory, not a surveyor's certificate.

If you find a bug in the book — in the prose or in the code — open an issue or
submit a pull request. This is a living document. The GPU does not stop
changing, and neither should the book.

Let us build something that is fast, and that we understand.

— *Arpan Pathak*
