# CUDA Kernels: GPU & Parallel Programming from First Principles

A comprehensive, no-fluff guide to GPU and parallel programming with CUDA.
Understand the hardware model, write fully documented CUDA C++ kernels, apply
modern C++ idioms, and write GPU kernels in pure Rust with CUDA-Oxide — all
wrapped around a complete image-processing capstone project.

Read it here: **https://arpanpathak.github.io/gpu-parallel-book/**

## Structure

- **Part I — Foundations of GPU Computing** (Chapters 1–3): the mathematics of
  parallelism, the GPU hardware model, and the CUDA programming model.
- **Part II — Writing CUDA C++ Kernels** (Chapters 4–6): memory management,
  synchronisation and atomics, and streams/events for asynchronous execution.
- **Part III — Optimisation & Advanced Patterns** (Chapters 7–9): memory
  optimisation, reduction/scan/histogram, and optimised matrix multiplication.
- **Part IV — Modern C++ & The CUDA Ecosystem** (Chapters 10–12): modern C++
  idioms, the Thrust/CUB/cuBLAS libraries, and NVRTC runtime compilation.
- **Part V — Rust, CUDA-Oxide & Safe GPU Programming** (Chapters 13–15):
  Rust host code with `cudarc`, NVIDIA's CUDA-Oxide Rust-to-CUDA compiler, and
  the image-processing capstone.
- **Part VI — The Engineering Mindset** (Chapter 16): profiling, debugging and
  performance engineering.

## Building locally

Requires [mdBook](https://rust-lang.github.io/mdBook/) (v0.5.x):

```bash
mdbook build book
```

## Repository layout

```
gpu-parallel-book/
├── .github/workflows/deploy-pages.yml   # GitHub Actions → GitHub Pages
├── book/
│   ├── book.toml                        # mdBook configuration
│   └── src/                             # Markdown source of the book
└── CODING_STANDARDS.md                  # The project's constitution
```
