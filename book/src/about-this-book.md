# About this book

Computing is entering its parallel age.

For more than thirty years, from the first microprocessors to the middle of the 2000s,
most programs became faster without being changed. Each new generation of processor ran
at a higher clock rate, and the same code finished sooner. Around 2005 that stopped. A
faster clock made a chip too hot to cool, so clock rates settled at a few gigahertz, and
chip designers spent each new generation's extra transistors on more processors instead
of faster ones. Since then, a program becomes faster mainly when it is written to do many
things at once.

The GPU is the clearest example of that change. A CPU has a few large cores, each designed
to finish one sequence of instructions as quickly as possible. A GPU has thousands of small
arithmetic units and keeps tens of thousands of threads ready to run, and it is designed to
finish a great many independent pieces of work in the same time. GPUs were first built to
colour the pixels of a screen, a task made of millions of independent pieces. The same
design now runs scientific simulations, trains and serves machine-learning models, and
processes images and signals.

Programming a GPU well does not depend on a large body of special knowledge. A small number
of ideas explain most of what makes a GPU program fast or slow: how threads are grouped and
scheduled, how data moves through the levels of memory, and how much arithmetic a program
performs for each byte it reads. This book builds those ideas from the beginning, and then
uses them to write and measure real programs.

## The Hardware Gap

A CUDA library call can hide the hardware behind it. A matrix multiply runs at 3% of peak when threads in a warp read columns instead of rows. A reduction can silently drop half of its data when threads diverge across a `__syncthreads()`. A host allocation that is pageable instead of pinned doubles transfer latency. None of these failures print an error. They produce a slow benchmark or a subtly wrong result.

This book covers the hardware model, the CUDA C++ programming model, and the C++ and Rust ecosystem around them, so those failures are diagnosable instead of mysterious.

## What You Will Build

Every chapter builds toward one project: **a complete GPU image-processing pipeline** - read an image, convert it to greyscale, apply a separable Gaussian blur, run a Sobel edge detector, and write the result - implemented three times:

1. in **CUDA C++** with hand-written, fully commented kernels;
2. with the **Thrust/CUB/cuBLAS** library ecosystem;
3. in **pure Rust** with NVIDIA's experimental **CUDA-Oxide** compiler, which turns idiomatic Rust into PTX.

You will build the same kind of pipeline a camera vendor would ship: pinned-memory transfers, streamed double buffering, an occupancy-tuned kernel configuration, and reproducible benchmarks. When you have finished, you will be able to look at any CUDA kernel - including the ones inside the libraries you already use - and explain, line by line, what it does and why it is fast.

## Who This Book Is For

You should read this book if:

- You can write C++ or Rust, but every GPU program you have written so far was a library call you did not fully understand.
- You have launched a kernel, seen it produce garbage, and had no idea whether the bug was in your index arithmetic, your memory layout, or your synchronisation.
- You suspect that most GPU tutorials skip the hardware model and want to understand the primitives - the warp, the streaming multiprocessor, the memory hierarchy - before touching a single CUDA API.
- You write Rust and want to know what CUDA-Oxide changes, and what it does not.
- You do not own a GPU and want to learn on the free compute that the cloud gives away (see the next section).
- You ship software whose performance budget is measured in microseconds and whose correctness budget is zero.

You do not need prior GPU experience. You need to be willing to sit with the hardware model. This book does not hand-wave the memory hierarchy. Every term is defined when it first appears; every primitive - every type, every built-in variable, every API call - is described before it is used. Where the book refers to a number (register counts, memory bandwidths, transaction sizes), it gives you the reasoning behind it, not just the number.

## You Do Not Need to Own a GPU

Every line of code in this book runs on any NVIDIA GPU with CUDA 12.x - including the free ones. If you do not own a GPU, the cloud has you covered, and most providers give away enough free compute to finish this entire book:

- **Google Colab** - free T4 GPUs in browser notebooks, zero setup; the fastest way to run your first kernel.
- **Kaggle Notebooks** - free GPU hours (roughly 30 per week, refreshed weekly), data-science friendly.
- **Google Cloud** - a new-account trial credit (about USD 300 at the time of writing) covers serious L4 and A100 sessions.
- **Microsoft Azure** - a new-account credit (about USD 200) for NC/ND-series GPU virtual machines.
- **AWS** - GPU instances (g4dn, g5, p4); the free tier is CPU-only, but the Activate and Educate programmes grant credits to startups and students.
- **Paperspace / Gradient** - GPU notebooks and cloud workstations with free and low-cost tiers.
- **Lambda, RunPod, Vast.ai** - cheap on-demand GPUs (RTX 4090 up to H100) when you outgrow the free tiers.
- **NVIDIA LaunchPad** - free, time-boxed hands-on labs on real NVIDIA hardware.
- **Modal** - serverless GPU code with recurring free compute credits (about USD 30 per month at the time of writing).

Credit amounts and session limits change frequently; check the current terms before you sign up. The repository README contains a fuller comparison table to help you choose.

## The Structure

The book is organised into six parts:

**Part I - Foundations of GPU Computing** (Chapters 1-3) covers the mathematics of parallelism, the GPU hardware model, and the CUDA programming model. Read this part carefully; every later chapter assumes the primitives defined here.

**Part II - Writing CUDA C++ Kernels** (Chapters 4-6) covers memory management, synchronisation, atomics, and asynchronous execution with streams and events.

**Part III - Optimisation & Advanced Patterns** (Chapters 7-9) covers memory optimisation, the canonical parallel algorithms (reduction, scan, histogram), and a complete, step-by-step optimisation of matrix multiplication.

**Part IV - Modern C++ & The CUDA Ecosystem** (Chapters 10-12) covers RAII wrappers, templates and modern C++ idioms, the Thrust/CUB/cuBLAS libraries, and runtime compilation with NVRTC.

**Part V - Rust, CUDA-Oxide & Safe GPU Programming** (Chapters 13-15) covers Rust host code driving CUDA kernels, NVIDIA's experimental CUDA-Oxide compiler for writing kernels in pure Rust, and the image-processing capstone.

**Part VI - The Engineering Mindset** (Chapter 16) covers profiling with Nsight Compute, debugging with Compute Sanitizer, and reproducible performance engineering.

## A Note on the Coding Standards

This project has a constitution. You will find it as `CODING_STANDARDS.md` in the repository root. It is not a suggestion. Every code block in this book follows it:

- Every primitive is explained before it is used.
- No magic numbers - if it is not `0`, `1`, `-1`, or a power of two required by the CUDA API, it gets a named constant.
- Every kernel is commented line by line.
- Every CUDA API call that can fail is checked.
- Every `__syncthreads()` and every atomic carries a comment stating which data it protects and why.

## A Note on Hardware and Honesty

The examples in this book target the CUDA 12.x toolkit and are written against the compute capability of modern NVIDIA GPUs (Ada and Hopper architectures, compute capability 8.x and 9.0). You do not need to own this hardware: the section "You Do Not Need to Own a GPU" lists the cloud options, and the free tiers alone are enough for everything in this book. Where a feature is architecture-specific, the book says so explicitly.

This book is also direct about the tooling. NVIDIA's CUDA-Oxide is an experimental, alpha-stage compiler; its API is evolving and its syntax may change. The chapters that cover it describe the project as it exists today, with code written in the style of its documented examples. Treat those chapters as a map of the territory, not a surveyor's certificate.

If you find a bug in the book - in the prose or in the code - open an issue or submit a pull request. This is a living document. The GPU does not stop changing, and neither should the book.

Chapter 1 starts from the mathematics and builds the programming model from
the hardware up. No GPU knowledge is assumed; the requirements are the C++ or
Rust listed above and a machine that can run CUDA 12.x examples.

- *Arpan Pathak*
