# Appendix C - Recommended Reading & Tools

> *"A book is a beginning, not a destination. Here is where the road
> continues."*

## C.1 Books

- **Programming Massively Parallel Processors: A Hands-on Approach** - David
  B. Kirk and Wen-mei W. Hwu. The standard academic text on GPU programming;
  the source of many of the patterns this book presents from first principles
  (reduction, tiling, coalescing).
- **CUDA C++ Best Practices Guide** - NVIDIA. The official optimisation
  handbook; the §7.9 checklist in this book is a compressed version of its
  structure.
- **Parallel Programming with C++** - Richard Vuduc et al. (course notes).
  The mathematics of Chapter 1 in more depth (Amdahl, Gustafson, roofline).
- **The Rust Book** - Steve Klabnik and Carol Nichols. The language reference
  for Part V, free online.
- **Performance Analysis of the CUDA Programming Model** - for the warp-level
  and SIMT details behind Chapter 2.

## C.2 Official Documentation (Primary Sources)

- **CUDA C++ Programming Guide** - the authoritative language reference.
- **PTX ISA Reference** - the virtual ISA of Chapters 3 and 12, in detail.
- **CUDA Toolkit Documentation** - API references for the runtime, driver,
  cuBLAS, Thrust, CUB, cuFFT, cuRAND, NVRTC.
- **Nsight Compute / Nsight Systems User Guides** - the profilers of Chapter
  16.
- **Compute Sanitizer User Guide** - the debugger of Chapter 16.
- **CUDA-Oxide (NVlabs/cuda-oxide)** - the README and examples are the
  primary source for Chapter 14; the project moves quickly, so read the
  repository, not just this book.
- **NCCL Documentation** - the official `nccl` user guide and API reference
  for Chapter 19 (collectives, algorithms, environment variables).
- **NVLink & NVSwitch** - NVIDIA's interconnect architecture papers and DGX
  system guides for Chapter 18.
- **Linux kernel PCI/IOMMU documentation** - `Documentation/PCI/` and
  `Documentation/IOMMU.txt` for the systems layer of Chapter 17.

## C.3 Tools

| Tool | Purpose | Chapter |
|---|---|---|
| `nvcc` | CUDA compiler driver | 3 |
| `nsys` | System-level profiling | 16 |
| `ncu` | Kernel-level profiling | 16 |
| `compute-sanitizer` | Memory/race/init/sync checking | 16 |
| `cuda-gdb` | Device debugging | 16 |
| `cuobjdump`, `nvdisasm` | Inspect cubins, SASS | 12 |
| `deviceQuery` (CUDA sample) | Hardware capabilities | 2 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Occupancy computation | 2, 16 |
| `cargo-oxide` | CUDA-Oxide build driver | 14 |
| `cudarc` (crates.io) | Rust host wrapper | 13 |
| `lspci` | PCIe devices, BARs, drivers | 17 |
| `nvidia-smi topo -m` | GPU-to-GPU topology (NVLink vs PCIe) | 18 |
| `nvidia-smi nvlink -s` | NVLink link status | 18 |
| `nccl-tests` | Collective bandwidth/latency benchmarks | 19 |

## C.4 Online Resources

- **NVIDIA Developer Blog** - architecture deep dives (tensor cores, TMA,
  CUDA Graphs).
- **NVIDIA GPU Computing Samples** (`cuda-samples` repository) - reference
  kernels for every pattern in this book.
- **NVIDIA's official CUDA-Oxide repository and Discord** - the community
  around the Rust compiler of Chapter 14.
- **crates.io/cudarc** - the Rust host library of Chapter 13, with examples.

## C.5 Cloud GPU Services

If you do not own an NVIDIA GPU, the cloud gives you enough free compute to
finish this book. The quickest starts: **Google Colab** (free T4 in a
browser, zero setup) and **Kaggle Notebooks** (roughly 30 free GPU hours per
week). For serious sessions, the big clouds offer new-account credits
(Google Cloud roughly USD 300, Microsoft Azure roughly USD 200); for cheap
on-demand GPUs, try Lambda, RunPod or Vast.ai; for serverless Python with
recurring credits, Modal. **NVIDIA LaunchPad** offers free, time-boxed
hands-on labs on real NVIDIA hardware.

The repository README keeps a fuller comparison table with links; credit
amounts change frequently, so always check the provider's current terms.

## C.6 How to Continue

1. **Re-implement the capstone** (Chapter 15) from memory, without looking.
   The gaps in your memory are the gaps in your understanding.
2. **Profile your own machine.** Run `deviceQuery`, compute your ridge point
   (Chapter 1), and take one kernel from this book to 90% of measured peak
   bandwidth using Chapter 7's checklist.
3. **Read the CUDA-Oxide repository** and track its releases. The alpha
   project of Chapter 14 is the fastest-moving part of this book.
4. **Port the pipeline** to a cross-vendor API (SYCL, HIP, or wgpu) and
   observe which ideas transfer unchanged. The answer - almost all of them  - 
   is the epilogue's point made measurable.
