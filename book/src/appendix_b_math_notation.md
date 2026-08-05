# Appendix B - Mathematical Notation Reference

> *"The notation used in this book, defined once and used everywhere."*

This appendix gathers the mathematics that appears throughout the text. It is
not a mathematics course; it is a dictionary. Every symbol below appears at
least once in the main chapters.

## B.1 Sets and Scalars

| Symbol | Meaning | First use |
|---|---|---|
| \\(\mathbb{R}\\) | The real numbers; \\(\mathbb{R}^n\\) is \\(n\\)-dimensional real space | Ch. 1 |
| \\(n, N, k, i, j\\) | Indices and sizes (integers) | Ch. 1-15 |
| \\(p\\) | The number of processing units (threads, cores) | Ch. 1 |
| \\(f\\) | The serial fraction of a workload (Amdahl) | Ch. 1 |
| \\(s\\) | The serial fraction of parallel time (Gustafson) | Ch. 1 |

## B.2 Performance Quantities

| Symbol | Meaning | Definition | First use |
|---|---|---|---|
| \\(T_1\\) | Serial execution time | - | Ch. 1 |
| \\(T_p\\) | Execution time on \\(p\\) units | - | Ch. 1 |
| \\(S(p)\\) | Speedup | \\(T_1 / T_p\\) | Ch. 1 |
| \\(E(p)\\) | Efficiency | \\(S(p) / p\\) | Ch. 1 |
| \\(P_{\text{peak}}\\) | Peak FLOP rate (FLOP/s) | - | Ch. 1 |
| \\(B\\) | Peak memory bandwidth (bytes/s) | - | Ch. 1 |
| \\(I\\) | Arithmetic intensity | FLOPs ÷ bytes | Ch. 1 |
| \\(I_{\text{ridge}}\\) | Ridge point | \\(P_{\text{peak}} / B\\) | Ch. 1 |

The roofline inequality: \\(P \\le \\min(P_{\\text{peak}},\\; I \\cdot B)\\),
with the two regimes *compute-bound* (\\(I > I_{\\text{ridge}}\\)) and
*memory-bound* (\\(I < I_{\\text{ridge}}\\)). Chapter 1, §1.8.

## B.3 Sums and Sequences

| Symbol | Meaning | First use |
|---|---|---|
| \\(\sum_{i=0}^{n-1} a_i\\) | The sum of the sequence \\(a_0, \\ldots, a_{n-1}\\) | Ch. 8 |
| \\(a_i\\) | The \\(i\\)-th element of a sequence | Ch. 8 |
| \\(\\log_2 n\\) | The base-2 logarithm of \\(n\\) (the tree height) | Ch. 8 |
| \\(\\lfloor x \\rfloor\\) | The floor of \\(x\\) (largest integer ≤ \\(x\\)) | Ch. 2 |

Tree reduction performs \\(n/2\\) additions per level over \\(\\log_2 n\\)
levels; the bank index is \\(\\lfloor \\text{addr}/4 \\rfloor \\bmod 32\\).
Chapter 2, §2.8; Chapter 8.

## B.4 Matrices and Vectors

| Symbol | Meaning | First use |
|---|---|---|
| \\(A, B, C\\) | Matrices (bold or capital letters) | Ch. 9 |
| \\(A[i][j]\\) or \\(a_{ij}\\) | The element at row \\(i\\), column \\(j\\) | Ch. 9 |
| \\(N\\) | Matrix dimension (\\(N \\times N\\)) | Ch. 9 |
| \\(C = A \\times B\\) | Matrix product: \\(c_{ij} = \\sum_k a_{ik} b_{kj}\\) | Ch. 9 |
| \\(G_x, G_y\\) | Sobel derivative kernels | Ch. 15 |

The FLOP count of \\(N \\times N\\) multiplication: \\(2N^3\\). The arithmetic
intensity: \\(N/6\\) FLOP/byte. Chapter 9, §9.1.

## B.5 Statistics and Error

| Symbol | Meaning | First use |
|---|---|---|
| \\(\\sigma\\) | Standard deviation (Gaussian blur radius) | Ch. 15 |
| \\(\\max |x - y|\\) | Maximum absolute difference (verification) | Ch. 3 |
| \\(1 \\times 10^{-4}\\) | The float-stage verification tolerance | Ch. 15 |

The Gaussian weights of the 5-tap blur, \\(\\sigma = 1\\):
\\([0.06136, 0.24477, 0.38774, 0.24477, 0.06136]\\). Chapter 15, §15.3.

## B.6 Greek Letters Used

| Letter | Role in this book |
|---|---|
| \\(\\alpha\\) | SAXPY scaling factor (\\(y = \\alpha x + y\\)) |
| \\(\\beta\\) | cuBLAS GEMM scaling factor (\\(C = \\alpha AB + \\beta C\\)) |
| \\(\\sigma\\) | Gaussian standard deviation |
| \\(\\Sigma\\) | Summation operator |

## B.7 Conventions

- **Row-major** storage is assumed everywhere unless stated otherwise:
  element \\((i, j)\\) of a \\(W\\)-wide matrix lives at linear index
  \\(i \\cdot W + j\\). Consecutive threads map to consecutive \\(j\\) - the
  coalescing convention of Chapter 7.
- **Zero-based** indexing, matching the hardware (`threadIdx.x` starts at 0).
- **FLOPs** counts a fused multiply-add as two operations (the convention
  behind Chapter 2's peak rates).
- **Powers of two** dominate block sizes and tile sizes; when a formula
  requires one, the text says so (Chapter 8, §8.3).
