# CUDA Kernels: GPU & Parallel Programming from First Principles

> A comprehensive, no-fluff guide to GPU and parallel programming with CUDA:
> the hardware model, fully documented CUDA C++ kernels, modern C++ idioms,
> and GPU kernels in pure Rust with CUDA-Oxide - all wrapped around a complete
> image-processing capstone project.

<div align="center">

[![NVIDIA CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white&style=for-the-badge)](https://developer.nvidia.com/cuda-toolkit)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-GPU-76B900?logo=nvidia&logoColor=white&style=for-the-badge)](https://www.nvidia.com/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white&style=for-the-badge)](https://en.cppreference.com/w/cpp/20)
[![Rust](https://img.shields.io/badge/Rust-2021-B7410E?logo=rust&logoColor=white&style=for-the-badge)](https://www.rust-lang.org/)
[![CUDA-Oxide](https://img.shields.io/badge/CUDA--Oxide-alpha-82aaff?logo=nvidia&logoColor=white&style=for-the-badge)](https://github.com/NVlabs/cuda-oxide)

[![mdBook](https://img.shields.io/badge/mdBook-v0.5.x-3B7DD8?logo=rust&logoColor=white)](https://rust-lang.github.io/mdBook/)
[![GitHub Pages](https://img.shields.io/github/deployments/arpanpathak/gpu-parallel-book/github-pages?label=GitHub%20Pages&logo=githubpages&logoColor=white)](https://arpanpathak.github.io/gpu-parallel-book/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>

## Read the book

**https://arpanpathak.github.io/gpu-parallel-book/**

## What is inside

- **Part I - Foundations of GPU Computing** (Chapters 1-3): the mathematics of
  parallelism (Amdahl, Gustafson, the roofline model), the GPU hardware model
  (warps, streaming multiprocessors, the memory hierarchy), and the CUDA
  programming model.
- **Part II - Writing CUDA C++ Kernels** (Chapters 4-6): memory management and
  data movement, synchronisation/atomics/race conditions, and streams/events
  for asynchronous execution.
- **Part III - Optimisation & Advanced Patterns** (Chapters 7-9): memory
  optimisation, reduction/scan/histogram, and optimised matrix
  multiplication.
- **Part IV - Modern C++ & The CUDA Ecosystem** (Chapters 10-12): RAII and
  modern C++ idioms, the Thrust/CUB/cuBLAS libraries, and NVRTC runtime
  compilation with the driver API.
- **Part V - Rust, CUDA-Oxide & Safe GPU Programming** (Chapters 13-15): Rust
  host code with `cudarc`, NVIDIA's experimental CUDA-Oxide Rust-to-CUDA
  compiler, and the image-processing capstone.
- **Part VI - The Engineering Mindset** (Chapter 16): profiling with Nsight
  Compute, debugging with Compute Sanitizer, and reproducible performance
  engineering.

Plus a foreword, an epilogue, and three appendices (CUDA API reference,
mathematical notation, recommended reading).

## Highlights

- Every concept explained from first principles; every primitive (type,
  built-in variable, API call) defined before use.
- Every code block fully commented, with the reasoning for each design
  decision.
- A complete capstone: an RGB -> greyscale -> Gaussian blur -> Sobel pipeline,
  implemented three ways (hand-written CUDA C++, Thrust, CUDA-Oxide Rust),
  streamed with pinned memory and verified against a CPU reference.
- A Night Owl inspired dark theme (bluish-black, easy on the eyes) with full
  support for printing.

## Cloud GPUs: learn without owning hardware

Every kernel in this book runs on any NVIDIA GPU with CUDA 12.x - including
the free ones. If you do not own a GPU, start in the cloud; most providers
give away enough free compute to finish the entire book.

| Provider | What you get | Free credits (approx.; check current terms) | Best for |
|---|---|---|---|
| [Google Colab](https://colab.research.google.com/) | Free T4 GPU in browser notebooks | Free tier with dynamic session limits | First kernels, zero setup |
| [Kaggle Notebooks](https://www.kaggle.com/) | Free GPU hours (T4/P100) | Roughly 30 GPU hours per week | Notebook-based learning |
| [Google Cloud](https://cloud.google.com/free) | Full GPU/TPU VMs (L4, A100, H100) | Roughly USD 300 new-account trial | Serious projects |
| [Microsoft Azure](https://azure.microsoft.com/) | NC/ND-series GPU VMs | Roughly USD 200 new-account credit | Enterprise stack |
| [AWS](https://aws.amazon.com/) | g4dn/g5/p4 GPU instances | Free tier is CPU-only; Activate/Educate credits for startups and students | Enterprise stack |
| [Paperspace / Gradient](https://www.paperspace.com/) | GPU notebooks + cloud workstations | Small signup credit; free notebook tiers | One-click ML |
| [Lambda](https://lambdalabs.com/gpu-cloud) | RTX 4090 / A100 / H100 on demand | None typically | High-end training |
| [RunPod](https://www.runpod.io/) | Pay-as-you-go + serverless GPU | Occasional promos | Flexible jobs |
| [Vast.ai](https://vast.ai/) | Community GPU marketplace | None | Cheapest spot GPUs |
| [NVIDIA LaunchPad](https://www.nvidia.com/en-us/launchpad/) | Time-boxed hands-on labs on NVIDIA hardware | Free labs | Trying NVIDIA tech hands-on |
| [Modal](https://modal.com/) | Serverless GPU Python | Roughly USD 30/month recurring credits | Code-first workloads |

Notes:

- Free-credit amounts and session limits change frequently. Always check the
  provider's current terms before signing up.
- Everything this book teaches fits in a free tier: the heaviest capstone
  pipeline runs in minutes on a T4.
- When you outgrow the free tiers, pay-as-you-go GPU providers (Lambda,
  RunPod, Vast.ai) are usually cheaper than the big three clouds for
  GPU-only work.

## Repository layout

```
gpu-parallel-book/
├── LICENSE                      # MIT license
├── CODE_OF_CONDUCT.md           # Contributor Covenant 2.1
├── CONTRIBUTING.md              # How to contribute (read this first)
├── SECURITY.md                  # How to report vulnerabilities
├── CODING_STANDARDS.md          # The book's constitution
├── .github/workflows/
│   └── deploy-pages.yml         # GitHub Actions -> GitHub Pages
└── book/
    ├── book.toml                # mdBook configuration
    ├── custom.css               # Night Owl dark theme
    └── src/                     # Markdown source of the book
```

## Building locally

Requires [mdBook](https://rust-lang.github.io/mdBook/) v0.5.x:

```bash
mdbook build book       # compile to book/book/
mdbook serve book       # live preview at http://localhost:3000
```

## Contributing

Corrections, clarifications and missing concepts are all welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first. If you find a bug in a
kernel, a formula, or a number, open an issue - the book is a living document.

## License

MIT. See [LICENSE](LICENSE) for details.
