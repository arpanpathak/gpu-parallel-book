<img class="plate" src="../../art/ch11.svg" alt="Folio, an owl in reading glasses, stands in front of a tall bookcase. The shelves hold rows of books, and 6 of them face outwards with their titles: reduce, scan, sort, gemm, fft and rand, the ready-made kernels of the libraries. Folio holds 1 more book, marked BlockReduce, to lend to a kernel. A tag says tuned once, borrowed by every program.">

# Chapter 11: The Library Ecosystem - Thrust, CUB & cuBLAS

Chapters 8 and 9 built a sum, a scan, a histogram and a matrix product by hand, as
**kernels**, functions that run on the GPU in many threads at once, and measured each step.
The CUDA toolkit already contains tuned versions of all of them. A **library** is
code written, tested and tuned by someone else, which a program calls instead of writing its
own. NVIDIA ships several with every toolkit, and each one is tuned for every GPU generation
by engineers who see the hardware before it is released.

This chapter puts each library next to the kernel it replaces and measures both on the same
GPU, a Jetson Orin NX (8 SMs, 102 GB/s of memory bandwidth, 2.4 TFLOP/s FP32), on the same day.
The numbers answer 2 practical questions. When is the library as fast as the hand-written
kernel, or faster? And what does a library call cost that a kernel launch does not? For small
inputs the second answer is dominated by overhead: on a sum of 4,096 floats, most of the time
of a convenient library call goes to memory allocation and waiting, not to adding.

<div class="covers" markdown="1">

This chapter covers

- what Thrust, CUB, cuBLAS, cuFFT and cuRAND do, and which earlier kernel each replaces
- Thrust's algorithms on device memory, and fusing 2 passes into 1 with `transform_reduce`
- a single-pass scan that moves half the bytes of Chapter 8's scan
- radix sort, worked by hand, built from a histogram and a scan
- the scratch memory a library call needs, and what allocating it on every call costs
- CUB's block-level building blocks inside your own kernel, and their Rust counterpart
- cuBLAS with row-major matrices, worked with a 2 × 3 example
- TF32 tensor cores: 4 times the speed, 200 times the error
- deciding between a library and a custom kernel, with measured numbers

</div>

The code of the chapter is in
[`code/ch11_libraries/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/ch11_libraries)
for C++ and in
[`code/oxide/ch11_blocks/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch11_blocks)
for Rust.

## 11.1 The case for libraries

### 11.1.1 What the toolkit ships

The 5 libraries of this chapter cover most of the algorithms of Part III and a few that this
book does not build by hand. They come with the CUDA toolkit; no separate download is needed.
2 terms recur in the table. **SGEMM** is the single-precision matrix product \\(C = AB\\) of
Chapter 9. **Scratch memory** is temporary device memory that an algorithm needs only while
it runs, such as the partial sums of the blocks of a reduction.

| Library | What it provides | Replaces | Header, link flag |
|---|---|---|---|
| **Thrust** | whole-array algorithms called from the host: `reduce`, `transform`, `scan`, `sort` | Chapter 8's sum and scan | `<thrust/...>`, none |
| **CUB** | the same algorithms with scratch memory managed by the caller, and building blocks for use inside a kernel | Chapter 8's block sum and block scan | `<cub/cub.cuh>`, none |
| **cuBLAS** | dense linear algebra: SGEMM, matrix-vector products, dot products | Chapter 9's SGEMM | `<cublas_v2.h>`, `-lcublas` |
| **cuFFT** | fast Fourier transforms | nothing in this book | `<cufft.h>`, `-lcufft` |
| **cuRAND** | random numbers on the GPU | nothing in this book | `<curand.h>`, `-lcurand` |

Thrust and CUB consist only of C++ templates in header files, so they are compiled into the
program like any other code and need no link flag. They belong to the **CUDA C++ Core
Libraries**, CCCL, together with `libcu++`, which provides the `cuda::std::` versions of the
C++ standard library used below. The version follows the toolkit: CUDA 12.6
ships CCCL 2.5, and CUDA 13.2 ships CCCL 3.2. cuBLAS, cuFFT and cuRAND are
compiled libraries, linked with a flag.

Figure 11.1 places them by where they are called from and who manages the memory.

<figure>
<img src="../../figures/ch11-levels.svg" alt="3 levels in a column. At the top, called from the host, Thrust: thrust::reduce, thrust::sort; it allocates its own scratch memory and launches its own kernels. In the middle, also from the host, CUB device-wide calls: cub::DeviceReduce, cub::DeviceScan, cub::DeviceRadixSort; the caller passes in the scratch memory. At the bottom, inside the caller's own kernel, CUB block and warp building blocks: cub::BlockReduce, cub::BlockScan, cub::WarpReduce. Beside the column, 3 separate libraries that work through a handle or plan: cuBLAS, cuFFT and cuRAND.">
<figcaption><b>Figure 11.1</b> Thrust on top of CUB, from the host down to the inside of a kernel; cuBLAS, cuFFT and cuRAND beside them.</figcaption>
</figure>

The levels trade convenience for control. A Thrust call does everything: it finds scratch
memory, launches kernels and, for a reduction, returns the answer to the host. A CUB
device-wide call does the same work but leaves the scratch memory, and the moment of
waiting, to the caller. A CUB block-level call does only 1 step, inside a kernel the
programmer writes. Thrust is built on CUB, so the kernels at the top and middle levels are the
same; sections 11.2 and 11.3 measure what the difference in convenience costs.

### 11.1.2 Why a library can beat a careful kernel

A library has 3 advantages over a kernel written for 1 program. Its authors choose a
different kernel for each GPU generation, with block sizes, loads and algorithms tuned on
that hardware. It uses algorithms that take more effort than 1 program can justify: the scan
of section 11.2.3 needs a careful protocol between blocks that Chapter 8 did not attempt. And
it is tested on every release against reference results and on many shapes of input.

A library also has 1 disadvantage, which this chapter measures: it cannot know the program's
context. It does not know that the same scratch memory could serve 1,000 calls, or that the
result is needed on the GPU and not on the host. The programmer who knows these things can
pass them in, through the lower levels of figure 11.1.

<div class="callout note" markdown="1">

**NOTE:** cuda-oxide, the Rust compiler of this book, has no bindings to Thrust, CUB, cuBLAS,
cuFFT or cuRAND at revision `b7fe543`. A Rust program can call those C libraries through
Rust's foreign function interface, which is beyond this chapter. cuda-oxide does ship its own
building blocks for use inside a kernel, the counterparts of CUB's block level, and section
11.3.2 measures them.

</div>

## 11.2 Thrust: algorithms on device memory

### 11.2.1 A sum in 1 line

Thrust's algorithms have the names and arguments of the C++ standard library's: a range
given by 2 positions, `first` and `last`, then the rest. The first argument is an
**execution policy**, which says where the algorithm runs: `thrust::device` runs it on the
GPU, on data in device memory. With raw device pointers the range is `in` to `in + n`.

```text
thrust::reduce(policy, first, last, initial value)
    → initial value + every element in [first, last), combined in any order
```

`thrust::reduce` adds the elements in a tree, as Chapter 8 did, so the order of the additions
differs from a loop's; the result is rounded differently but no less accurately. Listing
11.1 times it on the 67,108,864 floats of Chapter 8, 256 MiB, beside Chapter 8's fastest
kernel.

<p class="listing"><b>Listing 11.1</b> A sum with <code>thrust::reduce</code>, timed. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/reduce_scan.cu">code/ch11_libraries/reduce_scan.cu</a></p>

```cpp
bool sums(const float* in, float* partial, float* total, int n, double exact)
{
    // ...
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:166:169}}
    // ...
}
```

`thrust::reduce` returns the sum to the host as a `float`, so the call also waits for the
GPU to finish and copies 4 bytes back. The program checks every sum against a sum in double
precision and prints PASSED when all agree to 1 part in 100,000.

```console
$ ./reduce_scan
GPU: Orin, 8 SMs, L2 cache 2 MiB
sum of 67108864 floats (256 MiB), average of 20 runs
  Chapter 8, float4 + shuffles    2.87 ms    93.4 GB/s  sum 33557628.0
  same, cub::BlockReduce          2.90 ms    92.5 GB/s  sum 33557624.0
  cub::DeviceReduce::Sum          2.92 ms    91.8 GB/s  sum 33557628.0
    (CUB asked for 1279 bytes of temporary storage)
  thrust::reduce                  2.89 ms    93.0 GB/s  sum 33557628.0
```

All 4 ran within 2% of each other, 91.8 to 93.4 GB/s, which is 90% to 92% of the GPU's
102 GB/s. Chapter 8 showed that this is the ceiling: a sum reads each byte once and does 1
addition per 4 bytes, so the memory sets the speed. Work whose speed the memory sets is
called **memory-bound**, and no library can read faster than the memory delivers. The sums differ in the last digits because each method adds in a different
order; all are within 7 of the exact 33,557,630.3, a relative error of 2 × 10<sup>−7</sup>.
Across 3 runs the times drifted by up to 20%, so differences of a few percent between the 4
lines mean nothing.

<div class="bee bee-happy" markdown="1">

On a large array, a memory-bound library call runs as fast as the best hand-written kernel,
because both run at the speed of the memory.

</div>

### 11.2.2 Fusing 2 passes into 1

A **pass** is 1 trip through an array in memory. The sum of squares
\\(\sum_i x_i^2\\) can be written with 2 Thrust calls: `transform` squares every element into
a second array, and `reduce` adds the second array. Each call is a pass. `transform_reduce`
does both in 1 pass: it squares each element as it reads it and adds the square at once. The
square lives only in a register. Combining 2 passes into 1 is called **fusion**.

<div class="example" markdown="1">

Worked example: the bytes moved by 2 passes and by 1, for 256 MiB of floats

1. `transform` reads 256 MiB and writes 256 MiB of squares: 512 MiB.
2. `reduce` reads the 256 MiB of squares again: 256 MiB.
3. Together: 768 MiB, 3 times the input.
4. `transform_reduce` reads the input once: 256 MiB.
5. At the 92 GB/s of section 11.2.1, 768 MiB take 805 MB / 92 GB/s = 8.8 ms and 256 MiB take
   268 MB / 92 GB/s = 2.9 ms.

</div>

<figure>
<img src="../../figures/ch11-fusion.svg" alt="2 rows of bars drawn to scale. Top, transform then reduce: read 256 MiB, write 256 MiB of squares, read the squares again, 768 MiB in all, measured 10.51 ms. Bottom, transform_reduce: read 256 MiB once, each square kept in a register, measured 3.15 ms. A note says 3 times the bytes, 3.3 times the time.">
<figcaption><b>Figure 11.2</b> 2 passes move 3 times the bytes of 1 fused pass.</figcaption>
</figure>

A **functor** is an object that can be called like a function; Thrust takes the operation as
one. `Square` is marked `__host__ __device__` so that it compiles for the GPU, and
`cuda::std::plus<float>` is the standard library's addition, in the `cuda::std` version that
runs on both sides.

<p class="listing"><b>Listing 11.2</b> The sum of squares in 2 passes and in 1.</p>

```cpp
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:117:119}}
```

```cpp
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:175:196}}
```

```console
sum of squares, GB/s of input read
  transform, then reduce         10.51 ms    25.5 GB/s  sum 22372548.0
  transform_reduce                3.15 ms    85.3 GB/s  sum 22372548.0
```

The fused pass ran 3.3 times as fast, close to the 3 times predicted from the bytes; the rest
is the second launch and the extra allocation of the 2-pass version. The GB/s column counts
only the 256 MiB of input, so the 2-pass version shows a third of the memory's speed,
although its memory was as busy as in the fused pass.

<div class="bee bee-shocked" markdown="1">

A chain of library calls writes every intermediate result to memory and reads it back. On
memory-bound work, fusing the chain into 1 call divides the time by the number of passes
saved.

</div>

### 11.2.3 Scan in 1 pass

A **scan**, or prefix sum, replaces each element by a running total. An **inclusive** scan
includes the element itself; an **exclusive** scan stops before it, so it starts at 0.
For the values 3, 1, 2, 0:

| i | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| value | 3 | 1 | 2 | 0 |
| inclusive scan | 3 | 4 | 6 | 6 |
| exclusive scan | 0 | 3 | 4 | 6 |

The exclusive scan is the inclusive one shifted right by 1, and element \\(i\\) of it is the
number of items before position \\(i\\). Section 11.2.4 needs exactly that number.

Chapter 8's scan made 3 trips: each block scanned its 1,024 values and wrote them out, the
block totals were scanned, and a third kernel read every value again to add its block's
offset. CUB, which Thrust calls, scans in 1 trip with a method called **decoupled
look-back**. Each block scans its own values in registers and publishes its total in a flag
in memory. It then looks at the flags of the blocks to its left: once a neighbour has
published the running total of everything before it, the block adds that number, publishes
its own running total, and writes its values out once, finished. The blocks wait for each
other only over a few bytes of flags, never over the data.

<div class="example" markdown="1">

Worked example: the bytes moved by the 2 scans, for 64 Mi values of 4 bytes, 256 MiB

1. Chapter 8: the first kernel reads 256 MiB and writes 256 MiB; the third reads and writes
   them again. The block totals are 1,024 times smaller and add little. In all, 1,024 MiB.
2. Decoupled look-back: read 256 MiB, write 256 MiB, 512 MiB.
3. Chapter 8's scan today: 12.49 ms for 1,024 MiB, 1,074 MB / 12.49 ms = 86 GB/s of real
   traffic, although its program reports 43.0 GB/s because it counts only 512 MiB.

</div>

<p class="listing"><b>Listing 11.3</b> An inclusive scan with Thrust, an exclusive scan with CUB.</p>

```cpp
bool scans(int n)
{
    // ...
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:244:258}}
    // ...
}
```

```console
scan of 67108864 values 0 to 3, average of 20 runs
  thrust::inclusive_scan           7.88 ms    68.2 GB/s  last 100652862  correct: yes
  cub::DeviceScan::ExclusiveSum    7.23 ms    74.3 GB/s  last 100652860  correct: yes
```

Both columns count 512 MiB, the bytes a scan must move. The library scans ran at 68.2 and
74.3 GB/s against Chapter 8's 43.0 GB/s measured on the same board the same day: 1.6 to 1.7
times as fast, almost entirely because they move half the bytes. The 2 last values differ by
the last element, 2, which the exclusive scan leaves out.

### 11.2.4 Sorting

A sort is the clearest case for a library, because the fast GPU algorithm is long to write.
It is a **radix sort**: instead of comparing keys, it sorts them 1 digit at a time, starting
with the lowest digit. Each pass must be **stable**: keys with the same digit keep the order
they had, so the order set by the lower digits survives.

Each pass is built from the 2 algorithms of Chapter 8. A histogram counts how many keys have
each digit. An exclusive scan of the counts gives the first position of each digit's group in
the output. Then every key is written to its group's next free position.

```text
radix sort, digits of b bits, keys of B bits:
    for shift ← 0, b, 2b, … , B − b:
        digit(k) ← (k >> shift) mod 2^b
        count[d] ← number of keys with digit d          # a histogram
        start[d] ← count[0] + … + count[d − 1]          # an exclusive scan
        for each key k, in order:                       # a stable scatter
            out[start[digit(k)]] ← k
            start[digit(k)] ← start[digit(k)] + 1
        keys ← out
```

Figure 11.3 sorts 6 keys of 2 decimal digits by hand, so the digits can be read at a glance.

<div class="example" markdown="1">

Worked example: 53, 21, 37, 28, 15, 31

1. Pass 1, the ones digit: 3, 1, 7, 8, 5, 1. Counts: digit 1 twice, digits 3, 5, 7 and 8
   once each.
2. Exclusive scan of the counts: digit 1 starts at 0, digit 3 at 2, digit 5 at 3, digit 7 at
   4, digit 8 at 5.
3. Scatter, in input order: 21 → 0, 31 → 1, 53 → 2, 15 → 3, 37 → 4, 28 → 5. Result: 21, 31,
   53, 15, 37, 28.
4. Pass 2, the tens digit: 2, 3, 5, 1, 3, 2. Starts: digit 1 at 0, digit 2 at 1, digit 3 at
   3, digit 5 at 5.
5. Scatter: 15, 21, 28, 31, 37, 53. Sorted, after 2 passes for 2 digits.

</div>

<figure>
<img src="../../figures/ch11-radix.svg" alt="2 passes of a radix sort on the keys 53, 21, 37, 28, 15, 31. Each pass shows 3 rows. Pass 1, by the ones digit: the keys coloured by their digit; a histogram row of counts for digits 0 to 9, 0 2 0 1 0 1 0 1 1 0; the exclusive scan of the counts, 0 0 2 2 3 3 4 4 5 6; and the output 21 31 53 15 37 28 in groups of colour. Pass 2, by the tens digit: counts 0 1 2 2 0 1 and starts 0 0 1 3 5 5, and the sorted output 15 21 28 31 37 53.">
<figcaption><b>Figure 11.3</b> A radix sort: each pass is a histogram, an exclusive scan and a stable scatter.</figcaption>
</figure>

On a GPU the digits are binary. CUB sorts 8 bits per pass, 256 possible digits, so a 32-bit
key takes 4 passes, and each pass reads and writes every key. Its work grows with the number
of keys times the number of passes, not with \\(n \log n\\) as a comparison sort's does.

The radix sort has 1 limit: it must know what the key's bits mean. Thrust uses it when the
keys are numbers and the order is a standard one, `less` or `greater`. Given any other
comparison, such as the functor `Ascending` in listing 11.4, Thrust cannot look inside it and
uses a **merge sort**, which only asks the comparison which of 2 keys comes first. `Ascending`
gives the same order as `less`, so the 2 results are identical; only the algorithm differs.

Thrust needs scratch memory to sort: a second array of keys, plus space for the counts. By
default it allocates the memory with `cudaMalloc` at the start of each call and frees it with
`cudaFree` at the end. An **allocator** is an object that hands out memory; Thrust accepts one
in its execution policy, `thrust::cuda::par(allocator)`, and then takes its scratch memory
from there. `Arena` keeps 1 large buffer and hands out pieces of it.

<p class="listing"><b>Listing 11.4</b> A comparison Thrust cannot see inside, and an allocator that reuses 1 buffer. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/sort.cu">code/ch11_libraries/sort.cu</a></p>

```cpp
{{#include ../../../../code/ch11_libraries/sort.cu:37:74}}
```

<p class="listing"><b>Listing 11.5</b> 4 ways to sort with Thrust.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch11_libraries/sort.cu:141:161}}
    // ...
}
```

The program sorts 16,777,216 random 32-bit keys, 64 MiB, and checks each result against
`std::sort` on the CPU. CUB's own radix sort, `cub::DeviceRadixSort`, runs with scratch memory
allocated once, as section 11.3.3 explains.

```console
$ ./sort
GPU: Orin, 8 SMs
sort of 16777216 random 32-bit keys, average of 5 runs
  std::sort, 1 CPU core              1626.26 ms     10.3 Mkeys/s
  thrust::sort                         24.79 ms    676.7 Mkeys/s  correct: yes
  thrust::sort, scratch reused         13.05 ms   1285.5 Mkeys/s  correct: yes
    ... with cuda::std::greater        30.92 ms    542.6 Mkeys/s  correct: yes
    ... with own comparison            24.43 ms    686.8 Mkeys/s  correct: yes
  cub::DeviceRadixSort                  8.96 ms   1872.8 Mkeys/s  correct: yes
sort of 16777216 keys from 0 to 255
  cub::DeviceRadixSort, bits 0 to 31    8.12 ms   2067.2 Mkeys/s  correct: yes
  cub::DeviceRadixSort, bits 0 to 7     3.13 ms   5356.5 Mkeys/s  correct: yes
PASSED
```

Mkeys/s is millions of keys sorted per second. 5 lessons are in the 9 lines.

1. **The GPU sorts 66 to 180 times as fast as 1 CPU core**: 24.79 ms or less against 1,626
   ms.
2. **Allocation took almost half of the plain `thrust::sort`.** With the same scratch buffer
   reused, the sort took 13.05 ms instead of 24.79 ms: about 12 ms of each call went to
   `cudaMalloc` and `cudaFree`. A separate test on the same board timed 1 `cudaMalloc` and
   `cudaFree` of 64 MiB at 10 ms, and of 128 MiB at 20 ms. The cost depends on the state of
   the system: across 9 runs the difference ranged from 1 ms to 18 ms, on a board whose
   GPU memory is the system memory that the CPU's programs were using too. Measure it on
   yours.
3. **Radix against merge.** With its comparison hidden, the same order took 24.43 ms
   instead of 13.05 ms, 1.9 times as long.
4. **A fast path depends on what the library recognises.** In CCCL 2.5, Thrust's sort takes
   the radix path only if the comparison's type is exactly `thrust::less<Key>` or
   `thrust::greater<Key>`; `cuda::std::greater` is a different type there, so the descending
   sort fell back to the merge sort, 30.92 ms. CCCL 3.2 makes `thrust::greater` a
   name for `cuda::std::greater`, and the same source took the radix path, as fast as the
   ascending sort. Only a measurement shows which path a call took.
5. **Fewer bits, fewer passes.** For keys from 0 to 255, only bits 0 to 7 differ. Told so
   with `begin_bit` = 0 and `end_bit` = 8, CUB made 1 pass instead of 4 and finished 2.6 times
   as fast, 3.13 ms against 8.12 ms. The program, not the library, knows the range of its keys.

The 2 merge sorts differ by 26%, more than their code does: on a board shared with other
programs, sorts that take 20 to 30 ms drifted by 20% to 30% between runs.

<figure>
<img src="../../figures/ch11-sort.svg" alt="Horizontal bars of sorting speed in millions of keys per second for 16,777,216 32-bit keys. std::sort on 1 CPU core 10.3. thrust::sort 676.7. thrust::sort with scratch reused 1,285.5. With cuda::std::greater 542.6 and with its own comparison 686.8, both merge sorts. cub::DeviceRadixSort 1,872.8. Keys from 0 to 255 sorted on bits 0 to 7 only: 5,356.5.">
<figcaption><b>Figure 11.4</b> Sorting 16 Mi keys, in millions of keys per second.</figcaption>
</figure>

<details class="check"><summary>Why does a radix sort need every pass to be stable?</summary>

After pass 1 the keys are in order of their lowest digit. Pass 2 groups them by the next
digit; within a group, the keys must stay in the order pass 1 left them, or the lower digit's
order is lost. In figure 11.3, 21 and 28 share the tens digit 2, and 21 stays ahead of 28
only because pass 2 keeps the order of pass 1.

</details>

## 11.3 CUB: the building blocks

### 11.3.1 A block reduction inside your kernel

CUB's lowest level works inside a kernel the programmer writes. A **block** is a group of
threads of 1 kernel that run on the same SM, 1 of the GPU's streaming multiprocessors, and
share its fast **shared memory**; a **warp** is 32 threads of a block that execute each
instruction together. `cub::BlockReduce` combines 1 value from each thread of a block into 1
value, the job of Chapter 8's `blockSum`. That function used **shuffles**, instructions that
pass a value from 1 thread of a warp to another, within each warp; then 1 slot per warp in
shared memory; then a last shuffle in warp 0. Listing 11.6
is Chapter 8's fastest kernel with `blockSum` replaced by CUB.

<p class="listing"><b>Listing 11.6</b> Chapter 8's sum with <code>cub::BlockReduce</code>.</p>

```cpp
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:99:115}}
```

`cub::BlockReduce<float, BLOCK>` is a class template: the element type and the number of
threads are fixed when the program is compiled, and CUB picks its algorithm from them.
`TempStorage` is a type defined inside it, a block of shared memory sized for the algorithm
that was picked. Shared memory must be declared with a size known at compile time, and only
CUB knows how many bytes its algorithm needs for 256 floats; the type carries that size to
the `__shared__` declaration. `BlockReduce(temp)` builds the object on that storage, and
`.Sum(sum)` returns the total of the block in thread 0.

The sum of section 11.2.1 ran this kernel at 92.5 GB/s against 93.4 GB/s for Chapter 8's,
within the noise. CUB does not make the kernel faster; it removes 25 lines that had to be
written, and checked, by hand. It also handles cases Chapter 8's code did not: a block size
that is not a multiple of 32, and types other than `float`. The same header provides
`cub::BlockScan`, `cub::BlockHistogram`, `cub::BlockRadixSort`, `cub::WarpReduce` and
`cub::WarpScan`, all used the same way.

<div class="bee bee-cute" markdown="1">

`TempStorage` is a type so that CUB can tell the compiler how much shared memory to set
aside, before the kernel runs.

</div>

### 11.3.2 The same blocks in Rust

cuda-oxide's `cooperative_groups` module provides the Rust counterparts: `warp_reduce`,
`warp_scan`, `block_reduce` and `block_scan`, for `u32`, `i32` and `f32`, with `Sum`, `Min` and
`Max`. They work in 3 steps, as `cub::BlockReduce` does: each warp reduces its 32 values with
shuffles, lane 0 of each warp writes its total to a slot in shared memory, and warp 0 reduces
the slots. The slots are a `SharedArray` with 1 element per warp, 256 / 32 = 8 for a block of
256 threads; the kernel declares it and passes `&raw mut` to it, the Rust 2024 way to point at
a `static mut` without a reference.

<p class="listing"><b>Listing 11.7</b> Chapter 8's Rust sum with <code>block_reduce</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch11_blocks/src/main.rs">code/oxide/ch11_blocks/src/main.rs</a></p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch11_blocks/src/main.rs:64:84}}
    // ...
}
```

The `8` in `block_reduce::<f32, Sum, 8>` is the number of warps, and the call checks it at
compile time: a block of more than 32 warps fails to build. `block_scan` has the same shape.
Listing 11.8 scans each block of 256 values, 1 per thread; `out.get_mut(i)` returns `None` for
the threads past the end, so the kernel writes without `unsafe`.

<p class="listing"><b>Listing 11.8</b> A scan of each block with <code>block_scan</code>.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch11_blocks/src/main.rs:86:99}}
}
```

```console
$ cargo oxide run
sum of 67108864 floats, average of 20 runs
  Chapter 8, by hand    3.95 ms    68.0 GB/s  sum 33557632.0
  block_reduce          4.10 ms    65.5 GB/s  sum 33557632.0
  exact (f64)                               sum 33557630.3
block_scan of 67108864 values, 256 per block
  first block: [0, 1, 1, 3, 5, 5, 5, 6] ... 403
PASSED
```

The 2 Rust sums ran within 4% of each other and gave the same total. Both read 1 float per
load, so they reach 68 GB/s rather than the 93 GB/s of the C++ kernels with `float4` loads,
the difference Chapter 8 measured. The first 8 scanned values rise by 0 to 3 at each step, as
the inputs 0, 1, 0, 2, 2, 0, 0, 1 require, and the program checked all 67,108,864.

### 11.3.3 Device-wide calls and their scratch memory

CUB's middle level runs a whole algorithm on a whole array, as Thrust does, but takes its
scratch memory from the caller. A CUB device-wide call is made twice:

```text
1. call with no scratch memory (nullptr)   → CUB writes the number of bytes it needs
2. allocate that many bytes, once
3. call with the scratch memory            → CUB launches its kernels and returns at once
   (repeat step 3 as often as needed, with the same memory)
```

<p class="listing"><b>Listing 11.9</b> <code>cub::DeviceReduce::Sum</code>, called twice.</p>

```cpp
bool sums(const float* in, float* partial, float* total, int n, double exact)
{
    // ...
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:157:164}}
    // ...
}
```

For the sum of 64 Mi floats CUB asked for 1,279 bytes, room for the partial sums of its
blocks. The result `total` stays in device memory, and the call does not wait for the GPU.
Both differences from `thrust::reduce` cost nothing on 256 MiB of data, where the work takes
2.9 ms. On a small array they decide the time.

<div class="example" markdown="1">

Worked example: a sum of 4,096 floats, 16 KiB

1. At 92 GB/s, reading 16 KiB takes 16,384 B / 92 GB/s = 0.18 µs.
2. Measured per sum: `thrust::reduce` 68.1 µs; CUB with the result copied to the host 29.9
   µs; CUB with the result left on the GPU 8.8 µs.
3. Allocating and freeing the scratch memory: 68.1 − 29.9 = 38.2 µs.
4. Copying 4 bytes back and waiting for the GPU to finish: 29.9 − 8.8 = 21.1 µs.
5. The 8.8 µs left is the launch of CUB's kernel and its run; even that is 49 times the time
   the memory needs.

</div>

<p class="listing"><b>Listing 11.10</b> 3 ways to sum 4,096 floats.</p>

```cpp
{{#include ../../../../code/ch11_libraries/reduce_scan.cu:198:224}}
```

```console
sum of 4096 floats, microseconds per sum
  thrust::reduce                     68.1 us
  cub::DeviceReduce, result copied   29.9 us
  cub::DeviceReduce, result kept      8.8 us
```

Across 6 runs `thrust::reduce` took 44 to 68 µs and CUB with the result kept 4.7 to 8.8 µs.
In 5 of the 6 runs the 3 lines came in the order shown; in 1, the copy back was slow, and CUB
with the result copied took 78.6 µs. Anything that waits for the host varies the most.

<figure>
<img src="../../figures/ch11-scratch.svg" alt="Left: CUB's 2 calls. Call 1 with nullptr returns 1,279 bytes; the program allocates them once; call 2 with the memory runs the sum and can be repeated. Right: 3 bars for a sum of 4,096 floats, drawn to scale. thrust::reduce, 68.1 microseconds: 38.2 allocating and freeing, 21.1 copying the result back and waiting, 8.8 launching and summing. CUB with the result copied, 29.9 microseconds: 21.1 copy and wait, 8.8 sum. CUB with the result kept, 8.8 microseconds. The data itself needs 0.18 microseconds.">
<figcaption><b>Figure 11.5</b> CUB's 2-call pattern, and where 68.1 µs of a small Thrust sum go.</figcaption>
</figure>

A program that sums 1,000 small arrays per second spends 68 ms of each second in
`thrust::reduce`, and under 9 ms with CUB and a result kept on the GPU for the next kernel.
The library call is the same algorithm in both cases; the difference is what the program
told it.

<div class="bee bee-shocked" markdown="1">

On small inputs a library call's cost is allocation and waiting, not arithmetic. Allocate
scratch memory once, and keep results on the GPU when the next step runs there.

</div>

## 11.4 cuBLAS: dense linear algebra

### 11.4.1 Handles

cuBLAS implements the **BLAS**, Basic Linear Algebra Subprograms, a standard set of routines
first published for Fortran in 1979: dot products and vector updates (level 1), matrix-vector
products (level 2) and matrix-matrix products (level 3), among them SGEMM. Every cuBLAS call
takes a **handle**, an object created once by `cublasCreate` that holds the library's state:
its scratch memory, the **stream**, the queue of GPU work of Chapter 6, that its kernels go
to (`cublasSetStream`), and its settings, such
as the math mode of section 11.4.3. Keeping the state in a handle instead of global variables
lets 2 threads of a program use cuBLAS at the same time, each with its own handle.

Creating a handle is slow. The program of this section measures it:

```console
cublasCreate took 13.2 ms
```

13.2 ms is 3,500 times the 3.77 µs per launch of an empty kernel measured in section 6.7.
A program creates 1 handle at the start and passes it to every call.

### 11.4.2 Row-major data in a column-major library

A matrix is stored in memory as 1 line of numbers. **Row-major** order stores it row by row,
as C and C++ arrays do and as every chapter of this book has. **Column-major** order stores it
column by column, as Fortran does. cuBLAS keeps the Fortran convention of the original BLAS.

The same 6 numbers 1, 2, 3, 4, 5, 6 in memory are 2 different matrices under the 2 orders.
Read row by row into 2 rows of 3, they are \\(A = [1\ 2\ 3;\ 4\ 5\ 6]\\). Read column by
column into 3 rows of 2, the first column is 1, 2, 3 and the second is 4, 5, 6. The matrix
is \\([1\ 4;\ 2\ 5;\ 3\ 6]\\): the **transpose** of \\(A\\), written \\(A^T\\), with rows and
columns swapped. A row-major array, handed to a column-major library, is its transpose.

<figure>
<img src="../../figures/ch11-layout.svg" alt="Top: 6 numbers in memory, 1 2 3 4 5 6. Left: read row by row as a 2 by 3 matrix, they are A, rows 1 2 3 and 4 5 6. Right: read column by column as a 3 by 2 matrix, the columns are 1 2 3 and 4 5 6, so the rows are 1 4, 2 5, 3 6: the transpose of A. Below, the rule: C = A B is the same as C transpose = B transpose A transpose, so a column-major library asked for B transpose times A transpose, with the operands swapped, writes C in row-major order.">
<figcaption><b>Figure 11.6</b> The same 6 numbers are A row by row and the transpose of A column by column.</figcaption>
</figure>

The transpose of a product reverses the order of its factors: \\((AB)^T = B^T A^T\\). The
fix follows. Our arrays for \\(A\\) and \\(B\\) are, in cuBLAS's eyes, \\(A^T\\) and
\\(B^T\\). Asking cuBLAS for the product of its first operand \\(B^T\\) and its second operand
\\(A^T\\) yields \\(B^T A^T = (AB)^T = C^T\\), which cuBLAS stores column by column. A
column-major \\(C^T\\) has exactly the memory of a row-major \\(C\\). Swapping the 2 operands,
and the 2 outer sizes, is all it takes; nothing is copied or transposed.

Each matrix also has a **leading dimension**, the distance in elements from the start of 1
column to the start of the next in column-major storage. For a whole matrix it is the number
of rows cuBLAS sees; it can be larger when the matrix is a piece of a bigger one.

<div class="example" markdown="1">

Worked example: \\(A\\) is 2 × 3, \\(B = [7\ 8;\ 9\ 10;\ 11\ 12]\\) is 3 × 2

1. The right answer: row 0 of \\(C\\) is 1 × 7 + 2 × 9 + 3 × 11 = 58 and 1 × 8 + 2 × 10 + 3 ×
   12 = 64; row 1 is 139 and 154.
2. The wrong call passes \\(A\\) and \\(B\\) as given. cuBLAS reads \\(A\\)'s memory as 2 rows
   with leading dimension 2, columns (1, 2), (3, 4), (5, 6): the matrix \\([1\ 3\ 5;\ 2\ 4\
   6]\\). It reads \\(B\\) as \\([7\ 10;\ 8\ 11;\ 9\ 12]\\).
3. Their product is \\([76\ 103;\ 100\ 136]\\), since 1 × 7 + 3 × 8 + 5 × 9 = 76. cuBLAS
   stores it column by column: 76, 100, 103, 136. The program reads that row by row, as
   \\([76\ 100;\ 103\ 136]\\). No error is reported, and every number is wrong.
4. The right call passes \\(B\\) first, with leading dimension 2, and \\(A\\) second, with
   leading dimension 3.

</div>

<p class="listing"><b>Listing 11.11</b> The wrong call and the right one. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/gemm.cu">code/ch11_libraries/gemm.cu</a></p>

```cpp
bool smallProduct(cublasHandle_t handle)
{
    // ...
{{#include ../../../../code/ch11_libraries/gemm.cu:85:100}}
    // ...
}
```

The 2 calls differ only in the order of their operands and leading dimensions. cuBLAS
computes \\(C \leftarrow \alpha \cdot \mathrm{op}(A)\,\mathrm{op}(B) + \beta C\\), and the
arguments after the handle follow that formula:

| Arguments | Meaning | In the right call |
|---|---|---|
| `op, op` | an operation on each operand; `CUBLAS_OP_N` is none | none, none |
| `2, 2, 3` | \\(m\\), \\(n\\), \\(k\\): an \\(m \times k\\) operand times a \\(k \times n\\) one | 2, 2, 3 |
| `&one` | the factor \\(\alpha\\) | 1 |
| `B, 2` | the first operand and its leading dimension | \\(B^T\\), 2 rows |
| `A, 3` | the second operand and its leading dimension | \\(A^T\\), 3 rows |
| `&zero` | the factor \\(\beta\\); 0 ignores what \\(C\\) held before | 0 |
| `C, 2` | the result and its leading dimension | \\(C^T\\), 2 rows |

```console
C = A B with A 2 x 3 and B 3 x 2, stored row by row
  A = [1 2 3; 4 5 6]
  B = [7 8; 9 10; 11 12]
  C, operands as given   = [76 100; 103 136]
  C, operands swapped    = [58 64; 139 154]
```

Both lines match the hand-worked numbers. Chapter 9 called cuBLAS the same way for its square
matrices, where the 2 sizes are equal and only the operand order shows.

<div class="bee bee-angry" markdown="1">

A row-major array is a column-major library's transpose. Swap the operands, never trust a
square test alone, and check a small product by hand.

</div>

### 11.4.3 Speed, and tensor cores

Listing 11.12 runs the \\(N\\) = 4,096 product of Chapter 9 twice. The first call uses cuBLAS's
default math, in FP32 on the ordinary arithmetic units. Before the second, `cublasSetMathMode`
allows **TF32** on the **tensor cores**, units that multiply small matrices in 1 instruction.

<p class="listing"><b>Listing 11.12</b> SGEMM in FP32, then with TF32 tensor cores allowed.</p>

```cpp
int main()
{
    // ...
{{#include ../../../../code/ch11_libraries/gemm.cu:160:183}}
    // ...
}
```

```console
$ ./gemm
C = A B, 4096 x 4096 floats, average of 10 runs
  cuBLAS, FP32              91.34 ms   1504.6 GFLOP/s  max error 1.9e-05
  cuBLAS, TF32 tensor       21.05 ms   6530.4 GFLOP/s  max error 4.0e-03
PASSED
```

In FP32, cuBLAS reached 1,504.6 GFLOP/s, 63% of the GPU's 2.4 TFLOP/s FP32 peak; Chapter
9's best kernel reached 931.9 GFLOP/s, 62% of cuBLAS. The `max error` column compares 64
elements with a product computed in double precision, as in Chapter 9.

With tensor cores allowed, the same call ran 4.3 times as fast, 6,530.4 GFLOP/s, 2.7 times the
FP32 peak of the ordinary units, and its error grew 210 times, from 1.9 × 10<sup>−5</sup> to
4.0 × 10<sup>−3</sup>. The reason is the format. TF32, TensorFloat-32, keeps the 8-bit
exponent of FP32, so it covers the same range of sizes, but only 10 of its 23 bits of
**mantissa**, the digits of the number. Each input is rounded to 10 bits before it is
multiplied; the products are added in FP32.

<div class="example" markdown="1">

Worked example: 1/3 in FP32 and in TF32

1. In binary, 1/3 = 1.010101… × 2<sup>−2</sup>; the mantissa is the pattern 01 repeated.
2. FP32 keeps 23 bits of it: 0.33333334.
3. TF32 keeps 10 bits, 1.0101010101 in binary = 1 + 341/1,024 = 1.3330078, times
   2<sup>−2</sup>: 0.33325195.
4. The error is 0.33333333 − 0.33325195 = 8.1 × 10<sup>−5</sup>, 2.4 × 10<sup>−4</sup> of the
   value, against 3 × 10<sup>−8</sup> of the value in FP32. A sum of 4,096 such products
   carries the error into the result.

</div>

<figure>
<img src="../../figures/ch11-tf32.svg" alt="The bits of 2 formats as rows of boxes. FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits; 1/3 is 0.33333334. TF32: 1 sign bit, the same 8 exponent bits, 10 mantissa bits, with the other 13 greyed out; 1/3 is 0.33325195. Below, 2 measured results for N = 4,096: FP32 1,504.6 GFLOP/s with max error 1.9e-5; TF32 on tensor cores 6,530.4 GFLOP/s with max error 4.0e-3.">
<figcaption><b>Figure 11.7</b> TF32 keeps FP32's range and 10 of its 23 mantissa bits: 4.3 times the speed, 210 times the error.</figcaption>
</figure>

The gain from tensor cores differs widely between GPUs. Data-centre GPUs give them far more of
the chip than consumer GPUs do:

| GPU | FP32 peak | TF32 tensor peak, dense | Ratio |
|---|---|---|---|
| A100 SXM | 19.5 TFLOP/s | 156 TFLOP/s | 8.0 |
| H100 SXM | 67 TFLOP/s | 495 TFLOP/s | 7.4 |
| GeForce RTX 4090 | 82.6 TFLOP/s | 82.6 TFLOP/s | 1.0 |
| The measured GPU | 1.50 TFLOP/s (cuBLAS) | 6.53 TFLOP/s (cuBLAS) | 4.3 |

The peaks are from NVIDIA's A100, H100 and Ada architecture documents; "dense" excludes the
doubled figures quoted for sparse matrices. TF32 is used by default for training neural
networks, whose results tolerate errors far larger than 10<sup>−3</sup>. A simulation that
needs 7 correct digits must leave it off, and cuBLAS leaves it off unless the program asks.

## 11.5 cuFFT and cuRAND

### 11.5.1 Random numbers

cuRAND generates random numbers on the GPU, either from the host into an array or from inside a
kernel, 1 number per thread. A **pseudo-random generator** computes a sequence that looks
random from a starting number, the **seed**; the same seed gives the same sequence, which makes
a run repeatable. `CURAND_RNG_PSEUDO_PHILOX4_32_10` selects Philox, a **counter-based**
generator: number \\(i\\) of the sequence is a scrambling function of \\(i\\) and the seed
alone, so every thread can compute its own numbers without waiting for the numbers before
them.

<p class="listing"><b>Listing 11.13</b> 16 Mi normal numbers with cuRAND. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/fft_rand.cu">code/ch11_libraries/fft_rand.cu</a></p>

```cpp
bool normalNumbers()
{
    // ...
{{#include ../../../../code/ch11_libraries/fft_rand.cu:52:66}}
    // ...
}
```

The program draws 16,777,216 numbers from the **normal distribution** with mean 0 and
standard deviation 1, the bell curve, and checks both with 2 Thrust reductions: the mean of
the numbers and of their squares.

```console
$ ./fft_rand
cuRAND, 16777216 normal numbers in 1.01 ms (16688 million per second)
  mean -0.00042, standard deviation 0.99994
```

64 MiB of numbers in 1.01 ms is 66 GB/s of writes, so the generator runs close to the speed of
the memory. With 16.8 million samples, the mean of a fair sample would stray from 0 by about
1 / √16,777,216 = 0.00024 on average; −0.00042 is within 2 of those steps.

### 11.5.2 Frequencies with cuFFT

A **Fourier transform** describes a signal as a sum of waves and gives, for each frequency, how
strong its wave is. For \\(N\\) samples, the **discrete Fourier transform** gives \\(N\\)
complex numbers; computed directly, each takes \\(N\\) multiplications, \\(N^2\\) in all. The
**fast Fourier transform**, FFT, published by Cooley and Tukey in 1965, splits the problem in
halves recursively and needs about \\(N \log_2 N\\): for \\(N\\) = 1,024, about 10,240
operations instead of 1,048,576.

cuFFT works with a **plan**, made once for a size and a kind of transform, then executed on
any number of arrays of that shape. For real input it returns \\(N/2 + 1\\) frequencies; the
other half is a mirror image and carries no new information.

<div class="example" markdown="1">

Worked example: \\(\sin(50t) + 0.5 \sin(120t)\\), 1,024 samples

1. The signal is 2 waves: 50 cycles over the 1,024 samples with height 1, and 120 cycles with
   height 0.5.
2. A wave of height \\(h\\) that fits a whole number of cycles shows up as 1 frequency of
   magnitude \\(hN/2\\): 1 × 1,024 / 2 = 512 at frequency 50, and 0.5 × 512 = 256 at 120.
3. Every other frequency is 0, up to rounding.

</div>

<p class="listing"><b>Listing 11.14</b> A plan, then the transform.</p>

```cpp
bool spectrum()
{
    // ...
{{#include ../../../../code/ch11_libraries/fft_rand.cu:107:112}}
    // ...
}
```

```console
cuFFT, 1024 samples of sin(50 t) + 0.5 sin(120 t)
  frequency  50: magnitude 512.0
  frequency 120: magnitude 256.0
PASSED
```

<figure>
<img src="../../figures/ch11-spectrum.svg" alt="Left: the first 128 samples of the signal sin(50 t) + 0.5 sin(120 t), a fast wobble on a slower wave. An arrow labelled cufftExecR2C leads right to the spectrum: 513 frequencies along the bottom, all 0 except a bar of height 512 at frequency 50 and a bar of height 256 at frequency 120.">
<figcaption><b>Figure 11.8</b> 1,024 samples in, 513 frequencies out: 2 peaks, at 50 and 120.</figcaption>
</figure>

The program printed every frequency above magnitude 1: only the 2 predicted, with the
predicted heights.

## 11.6 Library or custom kernel?

Figure 11.9 collects the measurements of the chapter for the 3 algorithms that the book also
wrote by hand.

<figure>
<img src="../../figures/ch11-versus.svg" alt="3 panels of horizontal bars. Sum of 256 MiB, in GB/s: Chapter 8 kernel 93.4, cub::BlockReduce 92.5, cub::DeviceReduce 91.8, thrust::reduce 93.0. Scan of 64 Mi values, in GB/s: Chapter 8 43.0, thrust::inclusive_scan 68.2, cub::DeviceScan 74.3. SGEMM with N = 4,096, in GFLOP/s: Chapter 9 best 931.9, cuBLAS FP32 1,504.6, cuBLAS TF32 6,530.4.">
<figcaption><b>Figure 11.9</b> The book's kernels against the libraries, on the same GPU.</figcaption>
</figure>

<div class="lld" markdown="1">

| Problem | Library | Book's kernel | Measured | Choice |
|---|---|---|---|---|
| sum of a large array | `thrust::reduce` or CUB | Chapter 8 | 93.0 vs 93.4 GB/s | library: same speed, no code |
| sum of a small array, often | CUB, scratch reused, result kept | Chapter 8 | 8.8 µs vs 68.1 µs for `thrust::reduce` | CUB device-wide call |
| scan | CUB or Thrust | Chapter 8 | 74.3 vs 43.0 GB/s | library: half the bytes |
| sort | CUB radix sort | none | 1,873 Mkeys/s, 180 times 1 CPU core | library |
| SGEMM | cuBLAS | Chapter 9 | 1,505 vs 932 GFLOP/s | library; TF32 if 10 bits suffice |
| a step inside a custom kernel | CUB block level | Chapter 8 | 92.5 vs 93.4 GB/s | CUB inside the kernel |
| a stencil with a halo | none | custom | | custom kernel |

</div>

The table suggests an order of questions.

1. **Is the algorithm in a library?** Then call it, and measure the call. On large inputs the
   library matched or beat every kernel of Part III.
2. **Is the call slow because of its convenience?** On small or repeated inputs, move down a
   level of figure 11.1: CUB's device-wide call with scratch memory allocated once, results
   kept on the GPU, or an allocator for Thrust.
3. **Does the problem need custom logic per element?** Write a kernel, but use CUB's or
   cuda-oxide's block-level building blocks inside it. Custom does not mean from scratch.
4. **Is a chain of library calls memory-bound?** Each call is a pass. Fuse them, with
   `transform_reduce`, with Thrust's transform iterators, or with 1 custom kernel.

2 examples show the order at work.

**Normalising 100,000,000 floats** to the range 0 to 1 needs the smallest and largest values,
then 1 multiply-add per element. `thrust::minmax_element` finds both in 1 pass, reading 400
MB, and `thrust::transform` reads the 400 MB again and writes 400 MB of results: 1.2 GB, about
13 ms at 92 GB/s. A custom kernel cannot avoid the second pass, because no element can be
scaled before the largest one is known; it could save nothing.

**A 7-point blur of each frame of a video** reads each pixel's 3 neighbours on each side. No
library call fits a stencil with its halo of neighbours, so the blur is a custom kernel. It
reads along rows, so that neighbouring threads read neighbouring pixels (Chapter 7). Tiles in
shared memory come later, and only if the profiler of Chapter 16 shows the blur to be the
slowest part of the pipeline.

<div class="bee bee-happy" markdown="1">

Use the library by default, measure the call, and move down a level, not straight to a kernel
of your own, when the measurement disappoints.

</div>

## 11.7 The complete programs

<p class="listing"><b>Listing 11.15</b> The complete <code>reduce_scan.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/reduce_scan.cu">code/ch11_libraries/reduce_scan.cu</a></p>

```cpp
{{#include ../../../../code/ch11_libraries/reduce_scan.cu}}
```

<p class="listing"><b>Listing 11.16</b> The complete <code>sort.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/sort.cu">code/ch11_libraries/sort.cu</a></p>

```cpp
{{#include ../../../../code/ch11_libraries/sort.cu}}
```

<p class="listing"><b>Listing 11.17</b> The complete <code>gemm.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/gemm.cu">code/ch11_libraries/gemm.cu</a></p>

```cpp
{{#include ../../../../code/ch11_libraries/gemm.cu}}
```

<p class="listing"><b>Listing 11.18</b> The complete <code>fft_rand.cu</code>. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/ch11_libraries/fft_rand.cu">code/ch11_libraries/fft_rand.cu</a></p>

```cpp
{{#include ../../../../code/ch11_libraries/fft_rand.cu}}
```

<p class="listing"><b>Listing 11.19</b> The complete Rust program. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch11_blocks/src/main.rs">code/oxide/ch11_blocks/src/main.rs</a></p>

```rust,ignore
{{#include ../../../../code/oxide/ch11_blocks/src/main.rs}}
```

## Common mistakes

- **Row-major data passed to cuBLAS as it is.** The result is the product of 2 different
  matrices, with no error reported. Swap the operands and check a non-square case by hand.
- **Testing only with square matrices.** With \\(m = n = k\\) a wrong leading dimension can go
  unnoticed; the 2 × 3 example of section 11.4.2 catches it.
- **A library call in a loop over small inputs.** Each `thrust::reduce` allocated, waited and
  copied: 68.1 µs for 0.18 µs of reading. Reuse scratch memory and keep results on the GPU.
- **Creating a cuBLAS handle per call.** `cublasCreate` took 13 to 22 ms.
- **A chain of Thrust calls on memory-bound data.** Every call is a pass; fuse them.
- **Assuming the fast path.** A custom comparison, or a standard one the library version does
  not recognise, turned a radix sort into a merge sort at half the speed.
- **TF32 left on where 7 digits are needed.** It multiplies with 10-bit mantissas; the SGEMM
  error grew from 1.9 × 10<sup>−5</sup> to 4.0 × 10<sup>−3</sup>.
- **Sorting all 32 bits of keys that use 8.** `begin_bit` and `end_bit` cut the passes from 4
  to 1.

<div class="summary" markdown="1">

## Summary

- Thrust calls whole-array algorithms from the host; CUB offers the same algorithms with
  caller-managed scratch memory, and building blocks for use inside a kernel. cuBLAS, cuFFT
  and cuRAND work through a handle or a plan created once.
- On 256 MiB, `thrust::reduce`, CUB and Chapter 8's best kernel all ran at 92 to 93 GB/s:
  memory-bound work runs at the memory's speed, whoever writes it.
- `transform_reduce` fused 2 passes into 1 and ran 3.3 times as fast as `transform` followed
  by `reduce`.
- CUB's single-pass scan moves 512 MiB where Chapter 8's moved 1,024 MiB, and ran 1.7 times as
  fast.
- A radix sort is a histogram, an exclusive scan and a stable scatter per digit. On 16 Mi keys
  the GPU sorted up to 180 times as fast as 1 CPU core; reusing scratch memory nearly halved
  `thrust::sort`'s time, and a hidden comparison nearly doubled it.
- On a sum of 4,096 floats, `thrust::reduce` took 68.1 µs, 38.2 µs of it allocating and 21.1
  µs copying and waiting; CUB with its scratch memory reused and the result kept took 8.8 µs.
- `cub::BlockReduce` and cuda-oxide's `block_reduce` replace Chapter 8's hand-written block
  sum at the same speed. `TempStorage` is a type so that its size is known at compile time.
- A row-major array is its transpose to cuBLAS; asking for \\(B^T A^T\\) writes \\(C\\) in
  row-major order. cuBLAS ran SGEMM at 1,505 GFLOP/s in FP32, 1.6 times Chapter 9's best, and
  at 6,530 GFLOP/s with TF32 tensor cores, with 210 times the error.

</div>

Chapter 12 goes below the libraries: it compiles CUDA source while the program runs, with
NVRTC, and loads and launches the result through the driver API.

## Exercises

1. Change the sum of 4,096 floats in `reduce_scan.cu` to 65,536 and to 1,048,576 floats. At
   what size does `thrust::reduce` come within 10% of CUB with the result kept? Explain the
   size from the 38.2 µs and 21.1 µs of section 11.3.3.
2. Write the sum of squares as `thrust::transform` into a `thrust::device_vector`, then
   `thrust::reduce`, and as 1 custom kernel. Predict both times from the bytes moved, then
   measure.
3. Sort the 6 keys 47, 12, 40, 18, 42, 11 by hand with a radix sort of 2 decimal digits.
   Write the counts, the starts and the output of both passes.
4. Use `cub::BlockScan` in place of `warpScan` in Chapter 8's `scanBlocks`. Count the lines
   saved and measure the speed.
5. In `gemm.cu`, compute \\(C = AB\\) for a 3 × 2 matrix \\(A\\) and a 2 × 4 matrix \\(B\\),
   stored row by row. Work out all 6 arguments that change, then check the result by hand.
6. Instead of swapping the operands, call `cublasSgemm` with `CUBLAS_OP_T` for both operands.
   What does cuBLAS then compute, and in which order is the result stored?
7. Rerun `sort.cu` with `thrust::greater<unsigned>` in place of `cuda::std::greater<unsigned>`.
   On which toolkit does the time change, and why?
8. The measured GPU ran TF32 SGEMM 4.3 times as fast as FP32; the A100's peaks differ 8 times and
   the RTX 4090's not at all. Look up your GPU's FP32 and TF32 peaks and predict the ratio,
   then measure it with `gemm.cu`.
9. Generate 16 Mi uniform numbers with `curandGenerateUniform` and estimate π by counting the
   pairs \\((x, y)\\) with \\(x^2 + y^2 < 1\\), using `thrust::count_if` on a zipped pair of
   arrays. How many correct digits do 8 Mi pairs give?

## Sources and further reading

- NVIDIA, *CUDA C++ Core Libraries (CCCL)*: Thrust, CUB and libcu++: <https://nvidia.github.io/cccl/>
- NVIDIA, *cuBLAS Library*, `cublasSgemm`, leading dimensions and `cublasSetMathMode`:
  <https://docs.nvidia.com/cuda/cublas/>
- NVIDIA, *cuFFT Library User's Guide*: <https://docs.nvidia.com/cuda/cufft/>
- NVIDIA, *cuRAND Library*: <https://docs.nvidia.com/cuda/curand/>
- NVIDIA, *A100 Tensor Core GPU Architecture* (2020), TF32; *H100 Tensor Core GPU
  Architecture* (2022); *Ada GPU Architecture* (2022), the peak rates of section 11.4.3.
- D. Merrill and M. Garland, "Single-pass Parallel Prefix Scan with Decoupled Look-back",
  NVIDIA Technical Report NVR-2016-002, 2016.
- A. Adinets and D. Merrill, "Onesweep: A Faster Least Significant Digit Radix Sort for
  GPUs", 2022.
- C. Lawson, R. Hanson, D. Kincaid and F. Krogh, "Basic Linear Algebra Subprograms for Fortran
  Usage", *ACM Transactions on Mathematical Software* 5(3), 1979.
- J. Salmon, M. Moraes, R. Dror and D. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3",
  SC11, 2011 (the Philox generator).
- J. Cooley and J. Tukey, "An Algorithm for the Machine Calculation of Complex Fourier
  Series", *Mathematics of Computation* 19, 1965.
- cuda-oxide, `cuda_device::cooperative_groups` (`block_reduce`, `block_scan`), revision
  `b7fe543`: <https://github.com/NVlabs/cuda-oxide>
