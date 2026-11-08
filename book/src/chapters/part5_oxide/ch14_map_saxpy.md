<img class="plate" src="../../art/ch14.svg" alt="Shuttle, a crab, walks along a long table of numbered cells, 0 to 15, coloured in a repeating pattern of 4. Dashed arcs mark its hops: from cell 1 to 5, to 9, to 13, always 4 cells at a time, and it fills every cell it lands on. 3 more crabs further back would cover the other colours. A tag reads i, i + stride, i + 2 stride: the grid-stride loop.">

# Chapter 14: Map and SAXPY

Many computations produce an output array in which every element depends only on the elements at
the same position in the input arrays. Scaling a vector, adding 2 vectors, converting the pixels
of an image to grey and applying an activation function to every value of a tensor all have this
form. Such a computation is called a map. Because no element depends on another, a map is the
simplest computation to run in parallel, and it is the first one we derive in full on the GPU.

We study the map through SAXPY, the operation \\(y \leftarrow a x + y\\) on 2 vectors of
single-precision floats. We work it out by hand on 12 elements and write the sequential algorithm.
From it we derive 2 parallel algorithms: 1 thread per element, and a grid-stride loop in which
fewer threads than elements cover the array. We prove that the grid-stride loop updates every
element exactly once, and we break both algorithms on purpose to see what goes wrong. Counting the
bytes and operations of SAXPY then explains why its speed is set by the memory and not by the
arithmetic. The implementation is given in Rust with cuda-oxide and in CUDA C++, including a
generic map that takes a Rust closure or a C++ lambda, and we compare the measured rates of both
with the bandwidth of the GPU.

<div class="covers" markdown="1">

This chapter covers

- SAXPY and the map pattern, worked by hand on 12 elements
- the sequential algorithm and its count of operations and bytes
- 1 thread per element: blocks, the grid that covers \\(n\\) elements, and the guard \\(i < n\\)
- the grid-stride loop, derived from a grid with fewer threads than elements
- a proof that the grid-stride loop updates every element exactly once
- 2 broken variants: a missing guard and a wrong stride
- arithmetic intensity, and why SAXPY is limited by memory bandwidth
- generic maps with Rust closures and C++ lambdas
- measured bandwidth against the peak, and the number of blocks per SM a loop needs

</div>

The code of the chapter is in
[`code/oxide/ch14_map_saxpy/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch14_map_saxpy)
(Rust) and
[`code/cuda/ch14_map_saxpy/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch14_map_saxpy)
(C++). The Rust crate uses the same `Cargo.toml` and `rust-toolchain.toml` as the vector addition
of Chapter 13.

## 14.1 The problem

### 14.1.1 A worked example

We use the same 2 vectors throughout the chapter. They have 12 elements each, and the scalar is
\\(a = 2\\):

\\[
x = (2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5), \qquad y = (3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8).
\\]

Element \\(i\\) of the result is \\(2 x_i + y_i\\). For element 0 we compute
\\(2 \cdot 2 + 3 = 7\\), for element 1, \\(2 \cdot 7 + 1 = 15\\), and so on along the vectors.
Figure 14.1 shows the 12 results. Keep 1 of them in mind: element 9 is
\\(2 \cdot 8 + 3 = 19\\). We will follow this element through every algorithm of the chapter, and
in Section 14.5 we will see a faulty kernel turn element 2 into 8 in place of \\(2 \cdot 1 + 4 = 6\\).

<figure>
<img src="../../figures/ch14-example.svg" alt="The worked example. Row i holds the indices 0 to 11. Row x holds 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5; row y holds 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8. With a = 2 the result 2x + y is 7, 15, 6, 17, 9, 25, 4, 22, 9, 19, 13, 18. Column 9 is highlighted: 2 times 8 plus 3 is 19.">
<figcaption><b>Figure 14.1</b> SAXPY on the worked example with \(a = 2\). Element 9 is \(2 \cdot 8 + 3 = 19\).</figcaption>
</figure>

### 14.1.2 Definitions

<div class="definition">

<span class="box-label">Definition 14.1</span> **SAXPY.** Let \\(a\\) be a number and let \\(x\\)
and \\(y\\) be vectors of \\(n\\) single-precision floating-point numbers. SAXPY replaces \\(y\\)
by \\(a x + y\\), that is, it sets

\\[ y_i \leftarrow a\,x_i + y_i \qquad \text{for every } i \in [0, n). \\]

*Example.* With \\(a = 2\\), \\(x_9 = 8\\) and \\(y_9 = 3\\), SAXPY sets \\(y_9 = 19\\).

</div>

The name comes from the Basic Linear Algebra Subprograms (BLAS), a standard set of routines for
linear algebra published in 1979: *S* stands for single precision, and *AXPY* for "\\(a\\) times
\\(x\\) plus \\(y\\)". The result overwrites \\(y\\), so the computation is said to be **in
place**: the same array is both an input and the output. We write the range \\([a, b)\\) for the
integers \\(a, a + 1, \ldots, b - 1\\), as in the paper on which this part of the book is based,
so \\(i \in [0, 12)\\) runs over \\(0, 1, \ldots, 11\\).

SAXPY is 1 member of a larger family. In every member, output element \\(i\\) is computed by the
same function from the input elements at position \\(i\\), and from nothing else.

<div class="definition">

<span class="box-label">Definition 14.2</span> **Map.** A map applies the same function \\(f\\)
to the elements at each position of 1 or more input arrays of length \\(n\\), and produces an
output array of length \\(n\\) with

\\[ z_i = f(x_i, y_i, \ldots) \qquad \text{for every } i \in [0, n). \\]

*Example.* SAXPY is a map with 2 inputs and \\(f(x, y) = a x + y\\). Vector addition is a map with
\\(f(x, y) = x + y\\), and \\(y_i \leftarrow \max(x_i, y_i)\\) is a map with \\(f(x, y) =
\max(x, y)\\).

</div>

<div class="exercises">

Exercises

14.1 Compute elements 0 to 3 of SAXPY on the worked example with \\(a = 3\\) in place of 2.<br>
14.2 Which of these computations are maps: (a) \\(z_i = x_i^2\\); (b) \\(z_i = x_i - x_{i-1}\\);
(c) \\(z_i = x_i / s\\), where \\(s\\) is the sum of all elements of \\(x\\)? Explain.<br>
14.3 SAXPY overwrites \\(y\\). Name 1 advantage and 1 risk of computing in place.

</div>

## 14.2 The sequential algorithm

### 14.2.1 The loop

On a CPU, Definition 14.1 becomes a single loop over the positions.

<div class="algo">
<p class="algo-name">Saxpy<span style="font-variant: normal">(\(a, x, y, n\))</span></p>
<ol>
<li>for \(i \in [0, n)\) {</li>
<li style="--d:1">\(y_i = a \cdot x_i + y_i\) <span class="c">// 2 reads, 1 write, 2 FLOP</span></li>
<li>}</li>
</ol>
</div>

A **floating-point operation**, abbreviated FLOP, is 1 addition, subtraction, multiplication or
division of floating-point numbers. Line 2 performs 2 of them, a multiplication and an addition.
It also reads 2 values, \\(x_i\\) and \\(y_i\\), and writes 1 value, \\(y_i\\); each is a float of
4 bytes, so line 2 moves 12 bytes.

Line 2 executes once for every \\(i\\), that is \\(n\\) times. Our example therefore performs
\\(12 \cdot 2 = 24\\) FLOP and moves \\(12 \cdot 12 = 144\\) bytes. The arrays measured at the end
of the chapter have \\(n = 16{,}777{,}216 = 2^{24}\\) elements, for which the loop performs 33.6
million FLOP and moves 201.3 MB. The CPU executes the iterations 1 after another (Figure 14.2).

<figure>
<img src="../../figures/ch14-sequential.svg" alt="The sequential loop visits the 12 elements in order, 1 to 12 along a time axis. Visit 10 handles element 9: it reads x[9] = 8 and y[9] = 3 and writes y[9] = 19, with 2 reads, 1 write and 2 floating-point operations. In total 12 visits make 24 FLOP and move 144 bytes.">
<figcaption><b>Figure 14.2</b> The order in which <span class="sc">Saxpy</span> visits the elements. Visit 10 computes element 9 with 2 reads, 1 write and 2 FLOP.</figcaption>
</figure>

### 14.2.2 What the iterations share

Before we run the loop in parallel, we must know which iterations depend on which. Iteration
\\(i\\) reads \\(x_i\\) and \\(y_i\\) and writes \\(y_i\\). Iteration 9 reads \\(x_9 = 8\\) and
\\(y_9 = 3\\) and writes 19; no other iteration reads or writes position 9. The same holds for
every position, so the 12 iterations touch 12 disjoint pairs of memory locations.

Compare this with a computation that looks similar, the running sum
\\(s_i = s_{i-1} + y_i\\). Its element 9 needs element 8, which needs element 7, and so on back to
element 0. The top half of Figure 14.3 draws the data that each SAXPY result uses; every arrow
stays inside its column. The bottom half draws the same for the running sum, where every result
is joined to the one before it.

<figure>
<img src="../../figures/ch14-independence.svg" alt="Top: SAXPY on the example. Each result 7, 15, 6, ... 18 comes from its own column only, so all 12 can be computed in 1 step. Bottom, for contrast: the running sum of y, 3, 4, 8, 9, 14, 23, 25, 31, 36, 39, 44, 52, where each result needs the previous one, a chain of 12 steps.">
<figcaption><b>Figure 14.3</b> Top: each result of SAXPY uses only its own column. Bottom: each result of a running sum uses the result before it.</figcaption>
</figure>

<div class="key-observation">

<span class="box-label">Key observation</span> Iteration \\(i\\) of <span class="sc">Saxpy</span>
reads and writes only position \\(i\\). No iteration uses a value that another iteration
produces, so the 12 iterations may run in any order, and in particular all at the same time,
each on its own thread.

</div>

The running sum has no such property, and in Chapter 15 we show the different algorithm it needs.

<div class="exercises">

Exercises

14.4 How many FLOP and how many bytes does <span class="sc">Saxpy</span> need for
\\(n = 1{,}000\\)?<br>
14.5 The loop \\(y_i = y_{i+1} + x_i\\) for \\(i \in [0, n - 1)\\) also reads and writes \\(y\\).
Is it a map? What goes wrong if its iterations run at the same time?

</div>

## 14.3 One thread per element

### 14.3.1 Threads, blocks and grids

The key observation asks for 1 thread per position. We recall the terms of Chapter 13 that a GPU
program uses to describe its threads.

<div class="definition">

<span class="box-label">Definition 14.3</span> **Kernel and thread.** A **kernel** is a function
that the GPU executes many times at once. Each of these executions is a **thread**; every thread
runs the same code with its own position and its own variables.

*Example.* We will run SAXPY on the example as a kernel with 16 threads, 12 of which compute 1
element each.

</div>

<div class="definition">

<span class="box-label">Definition 14.4</span> **Block and grid.** A **block** is a group of
threads that run on the same streaming multiprocessor (SM) of the GPU and can share a fast memory
and wait for one another. The **grid** is the set of all blocks of 1 launch of a kernel. Each
thread can read 3 numbers: the index of its block in the grid, `blockIdx.x`; the number of
threads in a block, `blockDim.x`; and its own index within its block, `threadIdx.x`.

*Example.* A grid of 2 blocks of 8 threads has 16 threads. Thread 1 of block 1 reads
`blockIdx.x = 1`, `blockDim.x = 8` and `threadIdx.x = 1`.

</div>

Blocks 0 to \\(b - 1\\) come before block \\(b\\), and each holds `blockDim.x` threads, so block
\\(b\\) starts at position \\(b \cdot\\) `blockDim.x`. A thread adds its index within the block
and obtains its **global index**

\\[ i = \textit{blockIdx.x} \cdot \textit{blockDim.x} + \textit{threadIdx.x}. \\]

In the example, thread 1 of block 1 computes \\(i = 1 \cdot 8 + 1 = 9\\), and therefore element 9
(Figure 14.4). Every thread of the grid has a different global index, because 2 threads of the
same block differ in `threadIdx.x`, and threads of different blocks lie in different ranges of
`blockDim.x` positions.

### 14.3.2 A grid that covers \\(n\\) elements

A launch fixes the number of threads per block, \\(B\\), and the number of blocks. To give every
element a thread, the grid must hold at least \\(n\\) threads, so it needs at least \\(n / B\\)
blocks. The number of blocks is an integer, so we round up:

\\[ \text{blocks} = \lceil n / B \rceil. \\]

With \\(n = 12\\) and \\(B = 8\\), \\(\lceil 12 / 8 \rceil = 2\\) blocks hold 16 threads. Threads
0 to 11 each own 1 element, and threads 12 to 15, the last 4 threads of block 1, have no element
at all. In general the grid has \\(B \lceil n / B \rceil - n\\) such threads, fewer than \\(B\\);
for the measured arrays, \\(n = 2^{24}\\) and \\(B = 256\\) give exactly 65,536 blocks and none.

<figure>
<img src="../../figures/ch14-grid.svg" alt="The grid for n = 12 with blocks of 8 threads: 2 blocks, 16 threads, indices i = 0 to 15. Threads with i from 0 to 11 each own 1 element of y. Threads 12 to 15, the last 4 of block 1, are dashed: i is at least n and they have no element. Thread 1 of block 1 is highlighted: i = 1 times 8 plus 1 = 9.">
<figcaption><b>Figure 14.4</b> 12 elements in 2 blocks of 8 threads. Thread 1 of block 1 owns element \(1 \cdot 8 + 1 = 9\); threads 12 to 15 own nothing.</figcaption>
</figure>

A thread without an element must not touch memory, so every thread first compares its index with
\\(n\\). We obtain the parallel algorithm by deleting the loop of <span class="sc">Saxpy</span>
and letting the global index play the role of \\(i\\). Every thread executes the following
procedure.

<div class="algo">
<p class="algo-name">Saxpy-Kernel<span style="font-variant: normal">(\(a, x, y, n\))</span></p>
<ol>
<li>\(i = \textit{blockIdx.x} \cdot \textit{blockDim.x} + \textit{threadIdx.x}\)</li>
<li>if \(i \lt n\) { <span class="c">// the guard</span></li>
<li style="--d:1">\(y_i = a \cdot x_i + y_i\) <span class="c">// 2 reads, 1 write in global memory</span></li>
<li>}</li>
</ol>
</div>

The host launches <span class="sc">Saxpy-Kernel</span> on a grid of \\(\lceil n / B \rceil\\)
blocks of \\(B\\) threads. Line 1 is the formula of Section 14.3.1, line 2 is the **guard** that
sends the threads beyond the end of the array home, and line 3 is line 2 of
<span class="sc">Saxpy</span>, executed by 1 thread instead of 1 iteration.

### 14.3.3 The work of 1 thread

The vectors live in the GPU's main memory.

<div class="definition">

<span class="box-label">Definition 14.5</span> **Global memory.** Global memory is the main memory
of the GPU. Every thread of every block can read and write it, it holds gigabytes, and a read
takes several hundred clock cycles to return its value. **Registers** hold the variables of a
single thread and respond in about 1 cycle.

*Example.* \\(x\\) and \\(y\\) are in global memory; the product \\(2 \cdot 8\\) computed by thread
1 of block 1 is in a register.

</div>

Figure 14.5 follows thread 1 of block 1. It reads \\(x_9 = 8\\) and \\(y_9 = 3\\) from global
memory, computes \\(2 \cdot 8 + 3 = 19\\) in registers, and writes 19 back to \\(y_9\\): 2 reads,
1 write, 12 bytes and 2 FLOP, exactly the work of 1 iteration of the sequential loop.

<figure>
<img src="../../figures/ch14-thread.svg" alt="The work of thread 1 of block 1. It reads x[9] = 8 and y[9] = 3 from global memory, computes 2 times 8 plus 3 = 19 in registers with 2 floating-point operations, and writes 19 to y[9]: 12 bytes moved for 2 FLOP.">
<figcaption><b>Figure 14.5</b> The work of 1 thread of <span class="sc">Saxpy-Kernel</span>: 2 reads and 1 write in global memory, 2 FLOP in registers.</figcaption>
</figure>

The parallel algorithm therefore does the same work as the sequential one: 24 FLOP and 144 bytes
on the example, 12 bytes per element in general. What changes is the time. The \\(n\\) iterations
no longer form a sequence of \\(n\\) steps; with enough threads they all fit in 1 step. The cost
that remains, 12 bytes per element from global memory, cannot be reduced by any arrangement of
threads, because every value of \\(x\\) and \\(y\\) is used exactly once. In Section 14.6 we show
that this cost decides the speed of SAXPY.

<div class="exercises">

Exercises

14.6 With blocks of 8 threads, which thread computes element 5, and which element does thread 3
of block 1 compute?<br>
14.7 How many blocks of 256 threads cover \\(n = 1{,}000{,}000\\) elements, and how many threads
have no element?<br>
14.8 A program launches \\(n / B\\) blocks, rounded down, instead of \\(\lceil n / B \rceil\\).
Which elements of the example are never computed with \\(B = 8\\)?

</div>

## 14.4 Fewer threads than elements

### 14.4.1 Why a smaller grid

<span class="sc">Saxpy-Kernel</span> launches 1 thread for every element, 16,777,216 threads for
the measured arrays. A GPU does not run so many at once. A Jetson Orin NX, for example, holds at
most 1,536 threads on each of its 8 SMs, 12,288 threads in total, so the 65,536 blocks
of 256 threads run in about 1,366 rounds, each block starting when an earlier one has finished.
Nothing is wrong with this, and we measure it at full speed in Section 14.8. 3 reasons nevertheless
lead to a grid with fewer threads than elements.

1. Work that a thread does once, before or after its elements, is shared by all of its elements.
   The dot product of Chapter 15 relies on this: each thread first adds many products in a
   register, and only the few register sums are combined afterwards.
2. The grid can be sized to the GPU, from its number of SMs, and need not change with \\(n\\).
3. A single prepared launch then serves arrays of every length.

### 14.4.2 Which elements a thread takes

Suppose the grid has \\(P\\) threads and \\(P < n\\). Each thread must take several elements, and
together the threads must take each element exactly once. On the example we use \\(P = 4\\)
threads, 2 blocks of 2, for 12 elements, so each thread takes 3. 2 assignments come to mind
(Figure 14.6).

- **Chunks.** Thread \\(t\\) takes the 3 neighbouring elements \\(3t\\), \\(3t + 1\\) and
  \\(3t + 2\\). Thread 3 takes elements 9, 10 and 11.
- **Stride.** Thread \\(t\\) takes elements \\(t\\), \\(t + 4\\) and \\(t + 8\\). Thread 1 takes
  elements 1, 5 and 9.

Both assignments are correct. They differ in the addresses that the 4 threads use at the same
moment, and that difference decides how fast the memory serves them. To see why, we need 2 more
terms.

<div class="definition">

<span class="box-label">Definition 14.6</span> **Warp.** A **warp** is a group of 32 threads of a
block with consecutive indices, threads 0 to 31, 32 to 63, and so on. The SM issues each
instruction for a whole warp at once, so the 32 threads of a warp execute their loads at the same
moment. The position of a thread within its warp, 0 to 31, is called its **lane**.

*Example.* A block of 256 threads consists of 8 warps. In the example the 4 threads stand for the
lanes of 1 warp.

</div>

<div class="definition">

<span class="box-label">Definition 14.7</span> **Coalesced access.** The memory system serves
global memory in aligned segments of 32 bytes. An access by a warp is **coalesced** when its
threads use neighbouring addresses, so that the 32 loads of 4 bytes fall in 4 segments of 32
bytes, 128 bytes in all, and every byte fetched is used.

*Example.* If lanes 0 to 31 read \\(x_0\\) to \\(x_{31}\\), the warp reads 128 neighbouring
bytes: a coalesced access.

</div>

In step 0 of the chunk assignment, the 4 threads read elements 0, 3, 6 and 9: 4 values spread
over 40 bytes. With 32 threads per warp and chunks of 3, the addresses of 1 load instruction
are 12 bytes apart and touch 12 segments of 32 bytes: the memory delivers 384 bytes for 128 useful
ones. In step 0 of
the stride assignment, the threads read elements 0, 1, 2 and 3: 16 neighbouring bytes, and with
32 threads, 128 neighbouring bytes in 4 segments. The stride assignment keeps every step of every
warp coalesced, and Figure 14.6 shows it for all 3 steps.

<figure>
<img src="../../figures/ch14-assign.svg" alt="2 ways to give 12 elements to 4 threads, shown as 3 steps. Top, chunks: in step 0 the 4 threads touch elements 0, 3, 6 and 9, spread over 40 bytes. Bottom, stride 4: in step 0 they touch elements 0 to 3, then 4 to 7, then 8 to 11, each step 16 neighbouring bytes. Colours mark threads 0 to 3.">
<figcaption><b>Figure 14.6</b> 12 elements on 4 threads. Chunks spread the 4 accesses of each step over 40 bytes; a stride of 4 keeps them in 16 neighbouring bytes.</figcaption>
</figure>

| Thread \\(t\\) | Block, thread in block | Elements | Values of \\(y\\) after SAXPY |
|---|---|---|---|
| 0 | 0, 0 | 0, 4, 8 | 7, 9, 9 |
| 1 | 0, 1 | 1, 5, 9 | 15, 25, 19 |
| 2 | 1, 0 | 2, 6, 10 | 6, 4, 13 |
| 3 | 1, 1 | 3, 7, 11 | 17, 22, 18 |

<p class="listing"><b>Table 14.1</b> The stride assignment on the example: 4 threads, 12 elements, \(P = 4\).</p>

### 14.4.3 The grid-stride loop

The stride assignment gives thread \\(t\\) the elements \\(t, t + P, t + 2P, \ldots\\) below
\\(n\\), where \\(t\\) is the global index of Section 14.3.1 and \\(P\\) is the number of threads
in the grid.

<div class="definition">

<span class="box-label">Definition 14.8</span> **Grid-stride loop.** In a grid-stride loop, each
thread starts at its global index \\(t\\) and advances by the number of threads in the grid,
\\(P = \textit{gridDim.x} \cdot \textit{blockDim.x}\\), where `gridDim.x` is the number of blocks
of the grid. The distance \\(P\\) is called the **stride**.

*Example.* With 2 blocks of 2 threads, \\(P = 4\\), and thread 1 of block 0 (global index 1)
visits elements 1, 5 and 9.

</div>

<div class="algo">
<p class="algo-name">Grid-Stride-Saxpy<span style="font-variant: normal">(\(a, x, y, n\))</span></p>
<ol>
<li>\(P = \textit{gridDim.x} \cdot \textit{blockDim.x}\) <span class="c">// threads in the grid</span></li>
<li>\(i = \textit{blockIdx.x} \cdot \textit{blockDim.x} + \textit{threadIdx.x}\)</li>
<li>while \(i \lt n\) {</li>
<li style="--d:1">\(y_i = a \cdot x_i + y_i\)</li>
<li style="--d:1">\(i = i + P\)</li>
<li>}</li>
</ol>
</div>

Lines 2 and 4 are lines 1 and 3 of <span class="sc">Saxpy-Kernel</span>. The guard of line 2
there has become the condition of the loop in line 3: a thread whose index is already at least
\\(n\\) executes the loop 0 times. Line 5 advances by the stride. On the example, thread 1 runs
the loop 3 times, with \\(i = 1, 5, 9\\), computes 15, 25 and 19, and leaves when \\(i = 13\\).

Thread \\(t\\) processes the elements \\(t + kP < n\\), that is
\\(\lceil (n - t) / P \rceil\\) elements. For \\(n = 12\\) and \\(P = 4\\), each of the 4 threads
processes 3. For \\(P = 5\\), threads 0 and 1 process 3 and threads 2, 3 and 4 process 2; the
loop handles an \\(n\\) that is not a multiple of \\(P\\) with no extra code.

<div class="exercises">

Exercises

14.9 With \\(P = 5\\), list the elements of the example that each thread processes.<br>
14.10 A warp of 32 threads uses chunks of 4 elements. How many 32-byte segments does 1 load
instruction touch, and how many of the fetched bytes are used?<br>
14.11 How many times does the loop of <span class="sc">Grid-Stride-Saxpy</span> run for thread 7
of a grid of 2,048 threads and \\(n = 16{,}777{,}216\\)?

</div>

## 14.5 Correctness

The 2 kernels are short, and each depends on 1 condition that is easy to get wrong: the guard of
<span class="sc">Saxpy-Kernel</span> and the stride of <span class="sc">Grid-Stride-Saxpy</span>.
We remove each in turn and watch the result on the example.

### 14.5.1 Without the guard

Suppose we delete line 2 of <span class="sc">Saxpy-Kernel</span> and keep the launch of 2 blocks
of 8. Threads 12 to 15 compute addresses 48 to 60 bytes past the start of \\(y\\) and write
\\(2 x_i + y_i\\) there, into memory that does not belong to \\(y\\) (Figure 14.7).

<figure>
<img src="../../figures/ch14-overrun.svg" alt="Device memory as 4-byte slots. The first 12 hold y[0] to y[11]. The next 4 belong to other data. Without the test i less than n, threads 4 to 7 of block 1, global indices 12 to 15, write into those 4 slots.">
<figcaption><b>Figure 14.7</b> Without the guard, threads 12 to 15 write into the 4 slots after \(y\).</figcaption>
</figure>

We ran the C++ kernel without the guard, with \\(x\\) and \\(y\\) each followed by 4 more slots
of other data, holding 1 after \\(x\\) and 100 after \\(y\\):

```console
y[0..12):    7 15  6 17  9 25  4 22  9 19 13 18
other data:  102 102 102 102
no error
```

The 12 elements of \\(y\\) are correct, and the 4 values after \\(y\\) have changed from 100 to
\\(2 \cdot 1 + 100 = 102\\). The GPU reported no error: the 4 slots lay inside memory that the
program owned, so nothing stopped the writes. In a real program those slots hold another array,
and the damage appears later, in a computation that never touched SAXPY.

The Rust kernel cannot make this mistake. Its output is a `DisjointSlice<f32>`, and the only way
to obtain a writable element is `y.get_mut(idx)`, which returns `None` when `idx` lies at or
beyond the end of the slice. The guard of line 2 is therefore part of the type, and a kernel that
forgets it does not compile.

### 14.5.2 The stride: every element exactly once

<span class="sc">Grid-Stride-Saxpy</span> is correct only if every element is updated exactly
once. Not at all would leave an element unchanged; twice would add \\(a x_i\\) twice, because the
second update reads the result of the first.

<div class="theorem">

<span class="box-label">Theorem 14.1</span> When every thread of a grid of \\(P\\) threads has
finished <span class="sc">Grid-Stride-Saxpy</span>, every element \\(y_e\\) with
\\(e \in [0, n)\\) has been updated exactly once, by the thread with global index
\\(e \bmod P\\).

</div>

<div class="proof">

*Proof.* Fix a thread with global index \\(t \in [0, P)\\). We show that the loop of lines 3 to 6
maintains the invariant: *at the start of iteration \\(k\\), \\(i = t + kP\\), and the thread has
updated exactly the elements \\(t + jP\\) with \\(j \in [0, k)\\).*

*Initialization.* Before iteration 0, line 2 has set \\(i = t\\), and the thread has updated no
element; both parts hold with \\(k = 0\\).

*Maintenance.* Assume the invariant at the start of iteration \\(k\\), and that \\(i < n\\). Line 4
updates element \\(t + kP\\), and line 5 sets \\(i = t + (k + 1)P\\). The thread has now updated
the elements \\(t + jP\\) with \\(j \in [0, k + 1)\\), which is the invariant for \\(k + 1\\).

*Termination.* The loop ends at the first \\(k\\) with \\(t + kP \ge n\\). By the invariant the
thread has updated exactly the elements below \\(n\\) of the form \\(t + jP\\), each once,
because these values of \\(i\\) increase strictly.

Now take any element \\(e \in [0, n)\\) and write \\(e = jP + r\\) with \\(r = e \bmod P\\). The
thread with \\(t = r\\) updates \\(e\\) in its iteration \\(j\\). Every other thread \\(t \ne r\\)
updates only elements that leave the remainder \\(t\\) when divided by \\(P\\), so it never updates
\\(e\\). Hence \\(e\\) is updated exactly once.

</div>

The proof uses 2 facts about the launch: the grid has exactly \\(P\\) threads with the global
indices 0 to \\(P - 1\\), and line 1 computes the same \\(P\\). If line 1 computes a different
number, the remainders no longer separate the threads.

### 14.5.3 With the wrong stride

A common slip is to advance by the number of threads in a block, `blockDim.x`, instead of the
number in the grid. On the example, line 1 then gives \\(P = 2\\) instead of 4. Global thread 0
visits elements 0, 2, 4, 6, 8, 10, and global thread 2, thread 0 of block 1, starts at element 2
and visits 2, 4, 6, 8, 10. Elements 2 to 11 are each updated by 2 threads. Figure 14.8 shows 1
order in which this can happen: thread 0 updates \\(y_2\\) from 4 to \\(2 \cdot 1 + 4 = 6\\),
and thread 2 then reads the 6 and writes \\(2 \cdot 1 + 6 = 8\\).

<figure>
<img src="../../figures/ch14-twice.svg" alt="A grid-stride loop with the wrong stride, 2 in place of 4. Global thread 0 visits elements 0, 2, 4, 6, 8, 10 and global thread 2 visits 2, 4, 6, 8, 10. Thread 0 computes y[2] = 2 times 1 plus 4 = 6; thread 2 later reads the 6 and computes 2 times 1 plus 6 = 8. The result is 8 in place of 6.">
<figcaption><b>Figure 14.8</b> With a stride of 2 in place of 4, 2 threads update \(y_2\). If the second reads the first's result, \(y_2 = 8\) in place of 6.</figcaption>
</figure>

We ran the C++ program with `int threads = blockDim.x;` in place of
`gridDim.x * blockDim.x`. On the 12 elements of the example the output was correct: the 2 blocks
ran at the same moment on 2 SMs, both threads read \\(y_2 = 4\\) before either wrote, and both
wrote 6. On the measured arrays, with 8 blocks of 256 threads and \\(P = 256\\) in place of
2,048, the program reported

```console
  y[256] = 644, expected 324 (x = 128, y = 4)
  y[257] = 647.5, expected 326.25 (x = 128.5, y = 5)
  y[258] = 651, expected 328.5 (x = 129, y = 6)
  loop, 1 per SM: 2366 wrong elements
```

Element 256 is visited by block 0 in its second iteration and by block 1 in its first. When the
second visit read the result of the first, it added \\(2.5 \cdot 128 = 320\\) a second time:
\\(4 + 320 + 320 = 644\\). In 3 further runs the count of wrong elements was 3,197, 2,781 and 3,038; most
double visits read \\(y_i\\) before the other block's write arrived and wrote the correct value by
chance. A bug of this kind can pass a small test and fail a large one, and it can fail by a
different amount on every run.

<div class="rule-box">

<span class="box-label">Rule</span> In an in-place map, every element must be updated by exactly
1 thread, exactly once. Compute the stride from the whole grid, `gridDim.x * blockDim.x`, and the
start from the global index.

</div>

In Rust, the grid-stride kernel cannot use `get_mut`, because `get_mut` accepts only the index of
the calling thread, and the loop writes at the computed indices \\(t + kP\\). It uses
`get_unchecked_mut(i)` inside an `unsafe` block, and the comment above the block, written
`// SAFETY:` by convention, states Theorem 14.1 in 2 lines. The compiler does not check that
argument. The wrong stride compiles in Rust as well; the argument in the comment is then false.

<div class="exercises">

Exercises

14.12 In Theorem 14.1, where does the proof use that line 1 computes the number of threads of the
whole grid?<br>
14.13 With the wrong stride \\(P = 2\\) on the example, which elements can end up wrong, and what
is the wrong value of \\(y_9\\) if both updates happen 1 after the other?<br>
14.14 A kernel computes the start as `threadIdx.x` instead of the global index, with the correct
stride. Which elements of the example are updated twice, and which never?

</div>

## 14.6 Counting bytes and operations

### 14.6.1 Arithmetic intensity

In Section 14.3.3 we found that SAXPY moves 12 bytes for every 2 FLOP. To decide whether the memory or
the arithmetic limits its speed, we compare this ratio with the corresponding ratio of the GPU.

<div class="definition">

<span class="box-label">Definition 14.9</span> **Bandwidth and peak rate.** The **memory
bandwidth** of a GPU is the largest number of bytes per second that its global memory can deliver.
The **peak arithmetic rate** is the largest number of FLOP per second that its arithmetic units can
perform.

*Example.* The Jetson Orin NX has a memory bandwidth of 102 GB/s and a single-precision peak of
2.4 TFLOP/s; the Jetson AGX Thor has 273 GB/s and 8.1 TFLOP/s.

</div>

<div class="definition">

<span class="box-label">Definition 14.10</span> **Arithmetic intensity and ridge point.** The
**arithmetic intensity** of a kernel is the number of FLOP it performs per byte it moves to or
from global memory. The **ridge point** of a GPU is its peak arithmetic rate divided by its memory
bandwidth: the number of FLOP the GPU can perform in the time its memory delivers 1 byte. A kernel
whose arithmetic intensity lies below the ridge point is **memory-bound**: its speed is limited by
the memory bandwidth.

*Example.* SAXPY has an arithmetic intensity of \\(2 / 12 = 0.17\\) FLOP/B. The ridge point of the
Orin NX is \\(2{,}400 / 102 = 23.5\\) FLOP/B.

</div>

The ridge point of the Thor is \\(8{,}100 / 273 = 29.7\\) FLOP/B. SAXPY lies 141 times below the
first and 178 times below the second (Figure 14.9). Every GPU in the table of Chapter 2 has a ridge
point above 10 FLOP/B, so SAXPY is memory-bound on all of them.

<figure>
<img src="../../figures/ch14-intensity.svg" alt="Top: 1 element of SAXPY reads x[i] and y[i] and writes y[i], 4 bytes each, for 2 floating-point operations: 0.17 FLOP per byte. Middle: the ridge points, 23.5 FLOP per byte for the Jetson Orin NX and 29.7 for the Jetson AGX Thor, dwarf SAXPY's 0.17. Bottom: for 16,777,216 elements on the Orin NX the memory traffic takes 1.97 ms and the arithmetic 0.014 ms.">
<figcaption><b>Figure 14.9</b> SAXPY performs 0.17 FLOP per byte, far below the ridge points of the 2 GPUs. Its memory traffic takes 140 times as long as its arithmetic.</figcaption>
</figure>

### 14.6.2 The time the memory allows

A memory-bound kernel cannot finish before its bytes have crossed the memory. For SAXPY on
\\(n\\) elements and a GPU of bandwidth \\(W\\),

\\[ T \ge \frac{12\,n}{W}. \\]

For \\(n = 2^{24}\\), the 201.3 MB of SAXPY need at least \\(201.3 \text{ MB} / 102 \text{ GB/s}
= 1.97\\) ms on the Orin NX. Its 33.6 million FLOP would take \\(33.6 \text{ MFLOP} / 2.4
\text{ TFLOP/s} = 0.014\\) ms. A faster arithmetic unit would therefore not change the time of
SAXPY at all; only a faster memory would. Table 14.2 gives the bound for 4 GPUs.

| GPU | Memory bandwidth | Least time for \\(n = 2^{24}\\) | FLOP rate at that time |
|---|---|---|---|
| Jetson Orin NX 16 GB | 102 GB/s | 1.97 ms | 17 GFLOP/s |
| Jetson AGX Thor | 273 GB/s | 0.74 ms | 46 GFLOP/s |
| GeForce RTX 4090 | 1,008 GB/s | 0.20 ms | 168 GFLOP/s |
| H100 SXM | 3,350 GB/s | 0.060 ms | 558 GFLOP/s |

<p class="listing"><b>Table 14.2</b> The least time of SAXPY on \(2^{24}\) elements, from the memory bandwidth. Bandwidths from NVIDIA's data sheets.</p>

Even on an H100, SAXPY at its bound performs 558 GFLOP/s, less than 1% of that GPU's
single-precision peak of 67 TFLOP/s. The speed of a map is therefore measured in bytes per second,
and an implementation is judged by how close it comes to the memory bandwidth. In Section 14.8
we measure how close ours come.

<div class="exercises">

Exercises

14.15 DAXPY is SAXPY on 8-byte double-precision values. What is its arithmetic intensity, and
what is its least time for \\(n = 2^{24}\\) on the Orin NX?<br>
14.16 A map computes \\(z_i = \sqrt{x_i^2 + y_i^2}\\) into a third array. Count its bytes per
element, assuming a square root counts as 1 FLOP, and decide whether it is memory-bound on the
Orin NX.

</div>

## 14.7 Implementation

### 14.7.1 The SAXPY kernel

Listing 14.1 shows <span class="sc">Saxpy-Kernel</span> in both languages.

<p class="listing"><b>Listing 14.1</b> SAXPY, 1 thread per element. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch14_map_saxpy/src/main.rs">Rust</a> <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/cuda/ch14_map_saxpy/saxpy.cu">C++</a></p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs:29:40}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch14_map_saxpy/saxpy.cu:30:38}}
```

</div>
</div>

In C++, line 1 of the pseudocode is written out with `blockIdx.x`, `blockDim.x` and
`threadIdx.x`, and the guard is the `if (i < n)`. In Rust, `thread::index_1d()` computes the same
global index, and `y.get_mut(idx)` performs the guard: it returns `Some` with a mutable reference
to element `idx` when the index is in range, and `None` otherwise. The type of `idx` is
`ThreadIndex`, which only `thread::index_1d()` can create, so a thread can obtain a writable
reference only to its own element (Chapter 13). The scalar `a` is passed by value in both
languages: the host copies its 4 bytes into the launch, and every thread reads the same value.

The 2 attributes above the Rust kernel describe how it may be launched. `#[launch_bounds(256)]`
promises that no block has more than 256 threads, and `#[launch_contract(domain = 1)]` states
that the kernel uses a 1-dimensional grid. Together they let the host prepare a launch with any
block size up to 256 and check it once, before the first launch, so that the launch call itself
needs no `unsafe`. The worked example uses that freedom: it runs the kernel on 2 blocks of 8, the
grid of Figure 14.4.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn worked_example(
    // ...
) -> Result<usize, Error> {
    // ...
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs:147:148}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
int workedExample()
{
    // ...
{{#include ../../../../code/cuda/ch14_map_saxpy/saxpy.cu:144:144}}
    // ...
}
```

</div>
</div>

### 14.7.2 The grid-stride kernel and a generic map

The grid-stride kernel of Listing 14.2 applies any function `f` of 2 floats, not only SAXPY. In
Rust, `f` is a closure; in C++, a lambda.

<div class="definition">

<span class="box-label">Definition 14.11</span> **Closure.** A **closure** is a function without
a name, written inside another function, that can use variables of the function in which it is
written. A variable that a closure uses is said to be **captured**. C++ calls the same construct
a **lambda**.

*Example.* In Rust, `move |x: f32, y: f32| a * x + y` is a closure of 2 parameters that captures
`a`; in C++, `[a] __device__ (float x, float y) { return a * x + y; }` is the corresponding
lambda.

</div>

<p class="listing"><b>Listing 14.2</b> The grid-stride loop, generic over the function applied to each element.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs:58:77}}
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch14_map_saxpy/saxpy.cu:51:61}}
```

</div>
</div>

The variable `threads` is \\(P\\) of line 1, the start of the loop is line 2, and the loop
condition is line 3. Line 4 has become the call `f(x[i], y[i])`, and line 5 the increment by
`threads`. In C++ the loop is a `for` statement; in Rust it is a `while` loop over an ordinary
`usize`. The Rust loop writes at the computed index `i`, which is not the thread's own
`ThreadIndex`, so it calls `get_unchecked_mut(i)` in an `unsafe` block, and its `SAFETY` comment
is the argument of Theorem 14.1. The method also omits the bounds check, which the loop condition
`i < y.len()` makes unnecessary.

The host passes the function as the first argument of the launch. Listing 14.3 builds the SAXPY
closure, prepares the launch for the example's 2 blocks of 2 threads and launches it.

<p class="listing"><b>Listing 14.3</b> A closure that captures \(a\), passed to the grid-stride kernel.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn worked_example(
    // ...
) -> Result<usize, Error> {
    // ...
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs:152:156}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
int workedExample()
{
    // ...
{{#include ../../../../code/cuda/ch14_map_saxpy/saxpy.cu:148:151}}
    // ...
}
```

</div>
</div>

2 mechanisms make a function argument work on the GPU, and they are the same in both languages
(Figure 14.10). First, every closure has its own type, which the program cannot name. In Rust the
kernel is generic over a type parameter `F` with the bound `Fn(f32, f32) -> f32 + Copy`: `F` can
be called with 2 floats and returns a float, and its values can be copied byte for byte. The
compiler generates a separate copy of the kernel for every closure type it is launched with, a
process called **monomorphisation**. In C++ the kernel is a template, `template <typename F>`, and
the compiler instantiates it once for every lambda type in the same way. Inside each copy the call
`f(...)` is a call to 1 known function, which the compiler inlines. Second, a closure value
consists of its captured variables. The closure `axpy` holds the 4 bytes of `a`, and the launch
copies them into a kernel parameter, exactly as it copies the scalar `a` of Listing 14.1.

<figure>
<img src="../../figures/ch14-closure.svg" alt="2 closures on the host: move |x, y| a * x + y, which captures a = 2.5 (4 bytes), and |x, y| x.max(y), which captures nothing (0 bytes). Each has its own type, and the compiler builds 1 kernel map for each type; the captured bytes travel as 1 kernel parameter. C++ lambdas and the template map<F> work the same way.">
<figcaption><b>Figure 14.10</b> Each closure type gets its own compiled kernel, and each closure value travels as 1 kernel parameter.</figcaption>
</figure>

The 2 languages check different things. C++ requires the lambda to be marked `__device__`, and
`nvcc` accepts such a lambda in host code only with the option `--extended-lambda`. It rejects a
lambda that captures by reference, `[&]`, with the error
`An extended __device__ lambda cannot capture variables by reference`, but a lambda that captures a host pointer by value compiles. In
Rust, the bound `Copy` rejects any closure that owns memory, such as a captured `Vec<f32>`, and the
keyword `move` makes the closure copy `a` instead of borrowing it. A Rust closure that borrows a
host variable still compiles, because a reference is `Copy`. Both the C++ lambda with a host
pointer and the Rust closure with a host reference then read a host address on the GPU; the result is
shown in the Common mistakes at the end of the chapter.

The program also launches the 1-thread-per-element form of the map, `map`, with the closure
`axpy` and with a second closure, `|x, y| x.max(y)`, which captures nothing. Both are checked
element by element against the CPU.

<div class="exercises">

Exercises

14.17 Write the closure, in Rust and in C++, that turns the generic map into
\\(y_i \leftarrow x_i - y_i\\). How many bytes does its kernel parameter hold?<br>
14.18 Why must the Rust bound include `Copy`? What would a closure that captured a `Vec<f32>` mean
on the GPU?<br>
14.19 Listing 14.2 computes `threads` from `blockDim_x()` in Rust and `blockDim.x` in C++. Why is
this safer than a constant 256?

</div>

## 14.8 Measurements

### 14.8.1 Platform and method

The measurements come from an integrated GPU, a Jetson Orin NX 16 GB (8 SMs, 102 GB/s).
Each program fills \\(x\\) and \\(y\\) with \\(2^{24}\\) values, 64 MiB per vector,
more than the 2 MiB L2 cache of the measured GPU, so that every launch reads and writes global memory.
It uses \\(x_i = (i \bmod 1024) \cdot 0.5\\), \\(y_i = i \bmod 7\\) and \\(a = 2.5\\): every
product \\(2.5\,x_i\\) is a multiple of 1.25 below 1,280, which a float represents exactly, so the
results are compared with the CPU's for exact equality. Each kernel runs for 500 ms to raise the
GPU's clock, then 50 launches are timed with CUDA events; the rate is the 201.3 MB of 1 launch
divided by its time. Repeated runs differed by up to 5%.

### 14.8.2 Results

The Rust program printed the following; the C++ program prints the same lines, with "lambda" in
place of "closure".

```console
$ cargo oxide run
GPU: Orin, 8 SMs, at most 1536 threads per SM
worked example: n = 12, a = 2
  x                     2  7  1  8  2  8  1  8  2  8  4  5
  y                     3  1  4  1  5  9  2  6  5  3  5  8
  2 blocks of 8         7 15  6 17  9 25  4 22  9 19 13 18
  2 blocks of 2, loop   7 15  6 17  9 25  4 22  9 19 13 18
SAXPY on 16777216 floats, a = 2.5: 12 bytes and 2 FLOP per element
  kernel                 threads per thr     time        rate wrong
  saxpy                 16777216     1.0   2.240 ms   89.9 GB/s     0
  map, closure axpy     16777216     1.0   2.215 ms   90.9 GB/s     0
  map, closure max: 0 wrong
  loop, 1 per SM            2048  8192.0   6.122 ms   32.9 GB/s     0
  loop, 2 per SM            4096  4096.0   3.373 ms   59.7 GB/s     0
  loop, 3 per SM            6144  2730.7   2.575 ms   78.2 GB/s     0
  loop, 6 per SM           12288  1365.3   2.279 ms   88.3 GB/s     0
  loop, 12 per SM          24576   682.7   2.186 ms   92.1 GB/s     0
  loop, 48 per SM          98304   170.7   2.191 ms   91.9 GB/s     0
  device-to-device copy, 8 bytes per element    1.579 ms   85.0 GB/s
PASSED
```

The first 6 lines are the worked example. Both kernels, 2 blocks of 8 threads and the grid-stride
loop on 2 blocks of 2, produce the row of Figure 14.1, with \\(y_9 = 19\\). In the table, the
column `per thr` is the number of elements per thread, and `wrong` counts the elements that differ
from the CPU's result: 0 for every kernel. Table 14.3 collects the times of both languages.

| Kernel | Threads | Rust time | Rust rate | C++ time | C++ rate |
|---|---|---|---|---|---|
| SAXPY, 1 thread per element | 16,777,216 | 2.240 ms | 89.9 GB/s | 2.208 ms | 91.2 GB/s |
| generic map, closure or lambda | 16,777,216 | 2.215 ms | 90.9 GB/s | 2.219 ms | 90.7 GB/s |
| grid-stride loop, 1 block per SM | 2,048 | 6.122 ms | 32.9 GB/s | 5.949 ms | 33.8 GB/s |
| grid-stride loop, 2 blocks per SM | 4,096 | 3.373 ms | 59.7 GB/s | 3.292 ms | 61.1 GB/s |
| grid-stride loop, 3 blocks per SM | 6,144 | 2.575 ms | 78.2 GB/s | 2.502 ms | 80.5 GB/s |
| grid-stride loop, 6 blocks per SM | 12,288 | 2.279 ms | 88.3 GB/s | 2.163 ms | 93.1 GB/s |
| grid-stride loop, 12 blocks per SM | 24,576 | 2.186 ms | 92.1 GB/s | 2.178 ms | 92.4 GB/s |
| grid-stride loop, 48 blocks per SM | 98,304 | 2.191 ms | 91.9 GB/s | 2.232 ms | 90.2 GB/s |
| device-to-device copy (driver) | | 1.579 ms | 85.0 GB/s | 1.579 ms | 85.0 GB/s |

<p class="listing"><b>Table 14.3</b> SAXPY on \(2^{24}\) elements on the measured GPU. The rate is 201.3 MB divided by the time; the copy moves 134.2 MB.</p>

3 results stand out (Figure 14.11). The 2 languages run at the same speed, within the variation
between runs. The generic map with a closure or a lambda is as fast as the hand-written SAXPY
kernel: monomorphisation produced the same arithmetic on the same data, and the value of `a`
arrives in a kernel parameter in both cases. And no kernel exceeds 93.1 GB/s, 91% of the 102
GB/s of the memory.

<figure>
<img src="../../figures/ch14-measured.svg" alt="Measured rates on the measured GPU, Rust and C++: saxpy 89.9 and 91.2 GB/s; map with a closure or lambda 90.9 and 90.7 GB/s; grid-stride loop with 12 blocks per SM 92.1 and 92.4 GB/s; the driver's device-to-device copy 85.0 GB/s in both. A dashed line marks 102 GB/s.">
<figcaption><b>Figure 14.11</b> Measured rates of SAXPY on the measured GPU against the memory bandwidth of 102 GB/s.</figcaption>
</figure>

### 14.8.3 Why 91 GB/s, not 102 GB/s

In Section 14.6.2 we predicted at least 1.97 ms at 102 GB/s; the kernels took 2.16 ms to 2.24 ms, 10%
to 14% longer. The figure of 102 GB/s is the rate at the pins of the memory chips: the measured GPU
moves 16 bytes per transfer over a 128-bit bus at 6,400 million transfers per second, and
\\(16 \text{ B} \cdot 6.4 \cdot 10^9 \text{ /s} = 102.4 \text{ GB/s}\\). The memory cannot
transfer data on every one of those cycles. It must refresh its rows at regular intervals, it must
open a row before it reads from it, and it loses cycles each time the bus turns from reading to
writing. SAXPY writes 1 value for every 2 it reads, so its stream of requests changes direction
often.

2 measurements bound what a kernel can expect from this memory. The driver's own device-to-device
copy, which also reads and writes, reached 85.0 GB/s, less than SAXPY. The dot product of Chapter
15, which only reads, reaches 92 GB/s to 100 GB/s. SAXPY's 90 GB/s to 93 GB/s lies between the 2, at 88%
to 91% of the pin rate. Nothing in the kernel is left to remove: it reads each input once, writes
each output once, and every access is coalesced.

### 14.8.4 How many blocks a grid-stride loop needs

The grid-stride rows of Table 14.3 show a second effect. With 1 block of 256 threads per SM, the
loop reaches only 32.9 GB/s, a third of the bandwidth. The rate almost doubles with 2 blocks,
rises again with 3, and from 6 blocks on it matches the kernel with 1 thread per element. The
kernel is the same in every row; only the number of threads changes. The explanation is a law of
queues.

<div class="definition">

<span class="box-label">Definition 14.12</span> **Little's law.** In a system that serves
requests, the **throughput** \\(X\\) is the number of requests completed per unit of time, the
**latency** \\(L\\) is the time from the start of a request to its completion, and the
**concurrency** \\(C\\) is the number of requests in progress at once. In a steady state,

\\[ X = \frac{C}{L}. \\]

*Example.* With 16,384 bytes of loads in progress and a latency of 0.50 µs, memory delivers
\\(16{,}384 \text{ B} / 0.50\ \mu\text{s} = 32.8\\) GB/s.

</div>

In each iteration a thread issues its 2 loads, 8 bytes, and it cannot issue the loads of the next
iteration before it has stored the current result. Each thread therefore keeps about 8 bytes in
flight. With 1 block per SM the GPU runs \\(8 \cdot 256 = 2{,}048\\) threads, which keep
\\(2{,}048 \cdot 8 = 16{,}384\\) bytes in flight. The measured 32.9 GB/s then implies a latency of
\\(16{,}384 \text{ B} / 32.9 \text{ GB/s} = 0.50\\) µs, about 500 cycles of the GPU's clock, the
latency of global memory in Definition 14.5.

With the latency fixed, the law predicts twice the rate for twice the concurrency: 65.8 GB/s for 2
blocks per SM, against 59.7 GB/s and 61.1 GB/s measured, and 98.7 GB/s for 3 blocks, against 78.2
GB/s and 80.5 GB/s. Beyond that, the bandwidth caps the rate (Figure 14.12). The measurements fall
below the prediction at 2 and 3 blocks because the latency grows as the memory becomes busier, and
because a thread does not have 8 bytes in flight at every moment; it also computes, stores and
tests the loop condition. Read the other way, the law states how many bytes must be in flight to
reach the bandwidth: \\(C = X L = 102 \text{ GB/s} \cdot 0.50\ \mu\text{s} = 51\\) KB, or 6,375
threads with 8 bytes each, a little more than 3 blocks of 256 threads on each of the 8 SMs.

<figure>
<img src="../../figures/ch14-blocks.svg" alt="Measured rate of the grid-stride SAXPY against the number of blocks per SM on the measured GPU. Rust and C++: 1 block 32.9 and 33.8 GB/s, 2 blocks 59.7 and 61.1, 3 blocks 78.2 and 80.5, 6 blocks 88.3 and 93.1, 12 blocks 92.1 and 92.4, 48 blocks 91.9 and 90.2. A dashed line from Little's law rises in proportion to the blocks until it meets 102 GB/s.">
<figcaption><b>Figure 14.12</b> The rate of the grid-stride loop follows the bytes in flight, as Little's law predicts, until the bandwidth caps it.</figcaption>
</figure>

The measured GPU holds 1,536 threads per SM, 6 blocks of 256, so a grid of 6 blocks per SM fills the
GPU in 1 round, and the measurements show no gain beyond it. A grid-stride loop should therefore
be launched with at least as many blocks as the GPU holds at once, which the host program computes
from the number of SMs and the threads per SM that it reads from the device.

<div class="exercises">

Exercises

14.20 From Table 14.3, compute the fraction of the 102 GB/s that each language reaches with 1
thread per element.<br>
14.21 A GPU with 132 SMs, 2,048 threads per SM and 3,350 GB/s has a memory latency of about 0.6 µs.
How many bytes must be in flight to reach its bandwidth, and how many blocks of 256 threads per SM
provide them at 8 bytes per thread?

</div>

## 14.9 The complete programs

<p class="listing"><b>Listing 14.4</b> The complete programs. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch14_map_saxpy/src/main.rs">code/oxide/ch14_map_saxpy/src/main.rs</a> <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/cuda/ch14_map_saxpy/saxpy.cu">code/cuda/ch14_map_saxpy/saxpy.cu</a></p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch14_map_saxpy/saxpy.cu}}
```

</div>
</div>

## Common mistakes

**A missing guard.** Without `if (i < n)`, the threads of the last block that lie beyond \\(n\\)
write past the end of the array, and the GPU usually reports nothing (Section 14.5.1). In Rust,
`DisjointSlice::get_mut` makes the guard part of the type.

**A stride computed from the block instead of the grid.** `blockDim.x` in place of
`gridDim.x * blockDim.x` makes several blocks update the same elements. The result of an in-place
map is then wrong by a different amount on each run (Section 14.5.3).

**Using `get_mut` with a computed index.** In a grid-stride loop the index `i` is a `usize`, and
`get_mut` accepts only the `ThreadIndex` of the calling thread. Replacing `get_unchecked_mut(i)`
in Listing 14.2 by `y.get_mut(i)` stops the build:

```console
$ cargo oxide build
error[E0308]: mismatched types
   --> src/main.rs:73:41
    |
 73 |             if let Some(yi) = y.get_mut(i) {
    |                                 ------- ^ expected `ThreadIndex<'_>`, found `usize`
    |                                 |
    |                                 arguments to this method are incorrect
```

**Capturing host memory in a closure.** A closure that refers to a host variable captures its
address. The Rust closure `move |x: f32, y: f32| x + y + host[0]`, with `let host = &x0;`,
compiles, because a reference is `Copy`, and the kernel then reads a host address. On the measured
GPU the next synchronisation reports
`Error: DriverError(700, "an illegal memory access was encountered")`. A C++ lambda that captures a
host pointer by value, `[hp]`, compiles and fails with the same error, "an illegal memory access was
encountered"; `nvcc` rejects a capture by reference outright.

**A grid-stride loop with too few blocks.** With 1 block of 256 threads per SM the loop reached a
third of the bandwidth (Section 14.8.4). Size the grid from the number of SMs and keep at least 1
full round of resident blocks.

**Measuring arrays that fit in the cache.** A GPU with a large L2 cache keeps small arrays between
launches, and the measured rate then describes the cache. The arrays of this chapter, 128 MiB
together, exceed the L2 cache of every GPU in Table 14.2.

## Summary

- A map computes each output element from the input elements at the same position. SAXPY,
  \\(y \leftarrow a x + y\\), is a map; on the example it turns \\(y_9 = 3\\) into
  \\(2 \cdot 8 + 3 = 19\\).
- The sequential loop performs \\(n\\) iterations of 2 FLOP and 12 bytes. No iteration uses a value
  that another produces, so each can be given to its own thread.
- With blocks of \\(B\\) threads, \\(\lceil n / B \rceil\\) blocks cover \\(n\\) elements, and the
  guard \\(i < n\\) stops the extra threads: 2 blocks of 8 and 4 idle threads for the example.
- A grid-stride loop lets \\(P < n\\) threads cover the array: thread \\(t\\) takes
  \\(t, t + P, t + 2P, \ldots\\), which keeps each warp's accesses coalesced. Theorem 14.1 proves
  that each element is updated exactly once; with a wrong stride, 2,366 elements of the measured
  arrays were updated twice.
- SAXPY performs 0.17 FLOP per byte, far below the ridge point of every GPU, so its time is at
  least \\(12 n / W\\): 1.97 ms for \\(2^{24}\\) elements at 102 GB/s.
- A generic map takes a Rust closure or a C++ lambda; the compiler builds 1 kernel per closure
  type, and the captured values travel as 1 kernel parameter.
- On the measured GPU, SAXPY ran at 90 GB/s to 93 GB/s in both languages, about 90% of the 102 GB/s of
  the memory; a grid-stride loop needs about 3 blocks of 256 threads per SM to keep enough bytes in
  flight, and 6 to fill the GPU.

## Exercises

1. Run the program on your GPU. From its first line, compute how many blocks of 256 threads fit
   on 1 SM, and check whether the grid-stride rate stops rising at about that number of blocks per
   SM.
2. Write a generic map that computes \\(y_i \leftarrow \max(0, x_i) + y_i\\) and measure it. Does
   the extra comparison change the rate? Explain with the arithmetic intensity.
3. Change the Rust closure `axpy` so that it captures 2 values, `a` and `b`, and computes
   \\(a x + b y\\). How large is the kernel parameter that carries it?
4. Using the latency of 0.50 µs from Section 14.8.4, predict the rate of the grid-stride kernel
   with 1 block of 256 threads per SM on a GPU with 20 SMs and 273 GB/s. How many blocks per SM
   would reach the bandwidth?
5. Suppose each thread of the grid-stride loop handled 2 elements per iteration, loading 16 bytes
   before computing. By Little's law, how many blocks per SM would keep the memory of the measured GPU
   busy?
6. Prove that <span class="sc">Saxpy-Kernel</span> updates every element exactly once when
   launched with \\(\lceil n / B \rceil\\) blocks of \\(B\\) threads.
7. Write SAXPY on `f64` in both languages and measure it. Is its time twice that of the `f32`
   version, as the byte count predicts?

## Sources and further reading

- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026. The style of
  worked example, pseudocode and proof used in this part of the book.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed.,
  MIT Press, 2022. The pseudocode conventions and loop-invariant proofs.
- C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T. Krogh, "Basic Linear Algebra Subprograms for
  Fortran Usage", *ACM Transactions on Mathematical Software* 5(3), 1979. The original definition
  of SAXPY.
- M. Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", NVIDIA Technical Blog,
  2013.
- J. D. C. Little, "A Proof for the Queuing Formula: L = λW", *Operations Research* 9(3), 1961.
- NVIDIA, *CUDA C++ Programming Guide*, sections "Extended Lambdas" and "Compute Capabilities".
- *The Rust Reference*, chapters "Closure expressions" and "Closure types"; the cuda-oxide
  repository, <https://github.com/NVlabs/cuda-oxide>.
