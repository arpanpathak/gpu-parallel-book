<img class="plate" src="../../art/ch14.svg" alt="Shuttle, a crab, walks along a long table of numbered cells, 0 to 15, coloured in a repeating pattern of 4. Dashed arcs mark its hops: from cell 1 to 5, to 9, to 13, always 4 cells at a time, and it fills every cell it lands on. 3 more crabs further back would cover the other colours. A tag reads i, i + stride, i + 2 stride: the grid-stride loop.">

# Chapter 14: Map and SAXPY

Many computations produce an output array in which every element depends only on the elements at
the same position in the input arrays. Scaling a vector, adding 2 vectors, converting the pixels
of an image to grey, and applying an activation function to every value of a tensor all have this
form. Such a computation is called a map. Because no element of the output depends on any other
element of the output, a map is the simplest computation to run in parallel, and it is therefore
the first computation that we develop in full on the GPU.

In this chapter, we study the map through a single example, SAXPY, which multiplies a vector
\\(x\\) by a number \\(a\\) and adds the result to a second vector \\(y\\). We first work out
SAXPY by hand on 2 vectors of 12 elements and write the sequential loop that computes it. From
this loop, we derive 2 parallel algorithms. In the first, every element receives its own thread.
In the second, called a grid-stride loop, fewer threads than elements share the array, and each
thread processes several elements. We prove that the grid-stride loop updates every element
exactly once, and we then break both algorithms on purpose to see which wrong values appear. After
that, we count the bytes and the arithmetic operations of SAXPY, and this count explains why its
speed is set by the memory rather than by the arithmetic units. Finally, we implement both
algorithms in Rust with cuda-oxide and in CUDA C++, write a generic map that accepts a Rust
closure or a C++ lambda, and compare the measured speeds with the bandwidth of the GPU's memory.

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

We use the same 2 vectors throughout the chapter, so that every number in the chapter can be
checked by hand. Each vector has 12 elements, and the number by which we multiply \\(x\\), called
the scalar, is \\(a = 2\\):

\\[
x = (2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5), \qquad y = (3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8).
\\]

We number the positions of a vector from 0, and we write \\(x_i\\) for the element of \\(x\\) at
position \\(i\\). For example, \\(x_0 = 2\\) is the first element of \\(x\\), and \\(x_9 = 8\\) is
the tenth. Element \\(i\\) of the result is \\(2 x_i + y_i\\), that is, twice the element of \\(x\\)
plus the element of \\(y\\) at the same position. For element 0, we compute
\\(2 \cdot 2 + 3 = 7\\), and for element 1, we compute \\(2 \cdot 7 + 1 = 15\\). The remaining
elements follow in the same way, and Figure 14.1 shows all 12 results. Keep in mind 1 of them,
element 9, which is \\(2 \cdot 8 + 3 = 19\\). We follow this element through every algorithm of the
chapter. In Section 14.5, we also meet a faulty kernel that turns element 2 into 8 instead of
\\(2 \cdot 1 + 4 = 6\\).

<figure>
<img src="../../figures/ch14-example.svg" alt="The worked example. Row i holds the indices 0 to 11. Row x holds 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5; row y holds 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8. With a = 2 the result 2x + y is 7, 15, 6, 17, 9, 25, 4, 22, 9, 19, 13, 18. Column 9 is highlighted: 2 times 8 plus 3 is 19.">
<figcaption><b>Figure 14.1</b> SAXPY on the worked example with \(a = 2\). Element 9 is \(2 \cdot 8 + 3 = 19\).</figcaption>
</figure>

### 14.1.2 Definitions

Before we state the definition, we fix 1 piece of notation. We write the range \\([a, b)\\) for
the integers \\(a, a + 1, \ldots, b - 1\\), that is, from \\(a\\) up to but not including \\(b\\),
as in the paper on which this part of the book is based. For example, \\(i \in [0, 12)\\) means
that \\(i\\) runs over \\(0, 1, \ldots, 11\\), which are exactly the 12 positions of our vectors.

<div class="definition">

<span class="box-label">Definition 14.1</span> **SAXPY.** Let \\(a\\) be a number and let \\(x\\)
and \\(y\\) be vectors of \\(n\\) single-precision floating-point numbers. SAXPY replaces \\(y\\)
by \\(a x + y\\), that is, it sets

\\[ y_i \leftarrow a\,x_i + y_i \qquad \text{for every } i \in [0, n). \\]

*Example.* With \\(a = 2\\), \\(x_9 = 8\\) and \\(y_9 = 3\\), SAXPY sets \\(y_9 = 19\\).

</div>

| Symbol | Meaning | In the example |
|---|---|---|
| \\(n\\) | number of elements in each vector | 12 |
| \\(i\\) | position of an element, from 0 to \\(n - 1\\) | 9 |
| \\(a\\) | the scalar that multiplies \\(x\\) | 2 |
| \\(x_i\\) | element of \\(x\\) at position \\(i\\) | \\(x_9 = 8\\) |
| \\(y_i\\) | element of \\(y\\) at position \\(i\\), before and after the update | \\(y_9 = 3\\), then 19 |
| \\(\leftarrow\\) | "is replaced by" | |

The arrow \\(\leftarrow\\) means that the new value on the left replaces the old value of
\\(y_i\\), which is used on the right. The name SAXPY comes from the Basic Linear Algebra
Subprograms (BLAS), a standard set of routines for linear algebra published in 1979. In the name,
*S* stands for single precision, and *AXPY* stands for "\\(a\\) times \\(x\\) plus \\(y\\)". Since
the result overwrites \\(y\\), the computation is said to be **in place**, which means that the
same array is both an input and the output.

SAXPY is 1 member of a larger family of computations. In every member of the family, output
element \\(i\\) is computed by the same function from the input elements at position \\(i\\), and
from nothing else.

<div class="definition">

<span class="box-label">Definition 14.2</span> **Map.** A map applies the same function \\(f\\)
to the elements at each position of 1 or more input arrays of length \\(n\\), and produces an
output array of length \\(n\\) with

\\[ z_i = f(x_i, y_i, \ldots) \qquad \text{for every } i \in [0, n). \\]

*Example.* SAXPY is a map with 2 inputs and \\(f(x, y) = a x + y\\). Vector addition is a map with
\\(f(x, y) = x + y\\), and \\(y_i \leftarrow \max(x_i, y_i)\\) is a map with \\(f(x, y) =
\max(x, y)\\).

</div>

| Symbol | Meaning | SAXPY on the example |
|---|---|---|
| \\(f\\) | the function applied at every position | \\(f(x, y) = 2x + y\\) |
| \\(x_i, y_i, \ldots\\) | the input elements at position \\(i\\), 1 from each input array | \\(x_9 = 8\\), \\(y_9 = 3\\) |
| \\(z_i\\) | the output element at position \\(i\\) | \\(z_9 = f(8, 3) = 19\\) |

The dots in \\(f(x_i, y_i, \ldots)\\) indicate that a map may have any number of input arrays. In
SAXPY, the output array \\(z\\) is the array \\(y\\) itself, because the result overwrites \\(y\\).

<div class="exercises">

Exercises

14.1 Compute elements 0 to 3 of SAXPY on the worked example with \\(a = 3\\) in place of 2.<br>
14.2 Which of these computations are maps: (a) \\(z_i = x_i^2\\); (b) \\(z_i = x_i - x_{i-1}\\);
(c) \\(z_i = x_i / s\\), where \\(s\\) is the sum of all elements of \\(x\\)? Explain.<br>
14.3 SAXPY overwrites \\(y\\). Name 1 advantage and 1 risk of computing in place.

</div>

## 14.2 The sequential algorithm

### 14.2.1 The loop

On a CPU, Definition 14.1 becomes a single loop that visits the positions 1 after another and
updates each of them.

<div class="algo">
<p class="algo-name">Saxpy<span style="font-variant: normal">(\(a, x, y, n\))</span></p>
<ol>
<li>for \(i \in [0, n)\) {</li>
<li style="--d:1">\(y_i = a \cdot x_i + y_i\) <span class="c">// 2 reads, 1 write, 2 FLOP</span></li>
<li>}</li>
</ol>
</div>

To judge the cost of this loop later, we count 2 quantities for each iteration: the arithmetic it
performs and the bytes it moves to and from memory. A **floating-point operation**, abbreviated
FLOP, is 1 addition, subtraction, multiplication or division of floating-point numbers. Line 2
performs 2 of them, the multiplication \\(a \cdot x_i\\) and the addition of \\(y_i\\). Line 2 also
reads 2 values, \\(x_i\\) and \\(y_i\\), and writes 1 value, the new \\(y_i\\). Each of these values
is a float of 4 bytes, so line 2 moves \\(3 \cdot 4 = 12\\) bytes.

Line 2 executes once for every \\(i\\), that is, \\(n\\) times. Our example therefore performs
\\(12 \cdot 2 = 24\\) FLOP and moves \\(12 \cdot 12 = 144\\) bytes. The arrays that we measure at
the end of the chapter have \\(n = 16{,}777{,}216 = 2^{24}\\) elements, and for them the loop
performs 33.6 million FLOP and moves 201.3 MB. The CPU executes the iterations 1 after another, as
shown in Figure 14.2.

<figure>
<img src="../../figures/ch14-sequential.svg" alt="The sequential loop visits the 12 elements in order, 1 to 12 along a time axis. Visit 10 handles element 9: it reads x[9] = 8 and y[9] = 3 and writes y[9] = 19, with 2 reads, 1 write and 2 floating-point operations. In total 12 visits make 24 FLOP and move 144 bytes.">
<figcaption><b>Figure 14.2</b> The order in which <span class="sc">Saxpy</span> visits the elements. Visit 10 computes element 9 with 2 reads, 1 write and 2 FLOP.</figcaption>
</figure>

### 14.2.2 What the iterations share

Before we can run the iterations of the loop at the same time, we must know whether any iteration
depends on another. An iteration depends on another if it reads a value that the other one writes,
because it must then wait for that value. Iteration \\(i\\) reads \\(x_i\\) and \\(y_i\\), and it
writes \\(y_i\\). For example, iteration 9 reads \\(x_9 = 8\\) and \\(y_9 = 3\\) and writes 19, and
no other iteration reads or writes position 9. The same holds for every position, so the 12
iterations touch 12 separate pairs of memory locations, and no iteration ever needs a value that
another iteration produces.

To see that this property is not automatic, consider a computation that looks similar, the running
sum \\(s_i = s_{i-1} + y_i\\). Here \\(s_i\\) is the sum of the first \\(i + 1\\) elements of
\\(y\\), and it is computed from the previous sum \\(s_{i-1}\\). Element 9 of the running sum
therefore needs element 8, which in turn needs element 7, and the chain continues back to element 0.
The top half of Figure 14.3 draws the data that each result of SAXPY uses, and every arrow stays
inside its own column. The bottom half draws the same for the running sum, in which every result is
joined to the result before it.

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

The running sum does not have this property, and it needs a different algorithm. In Chapter 15, we
develop that algorithm for a closely related computation, the sum of all elements of an array.

<div class="exercises">

Exercises

14.4 How many FLOP and how many bytes does <span class="sc">Saxpy</span> need for
\\(n = 1{,}000\\)?<br>
14.5 The loop \\(y_i = y_{i+1} + x_i\\) for \\(i \in [0, n - 1)\\) also reads and writes \\(y\\).
Is it a map? What goes wrong if its iterations run at the same time?

</div>

## 14.3 One thread per element

### 14.3.1 Threads, blocks and grids

The key observation suggests that we give every position its own thread. To do this, we need the
terms with which a GPU program describes its threads. We introduced them in Chapter 13, and we
recall them here.

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

Each thread must now find the element that it is responsible for. We want to number all threads
of the grid consecutively, from 0 upwards, so that thread number \\(i\\) can take element \\(i\\).
Consider block number \\(b\\). The blocks 0 to \\(b - 1\\) come before it, and each of them holds
`blockDim.x` threads, so together they hold \\(b \cdot\\) `blockDim.x` threads. The first thread of
block \\(b\\) therefore receives the number \\(b \cdot\\) `blockDim.x`, and a thread of block
\\(b\\) adds its own index within the block to this number. The result is called the thread's
**global index**:

\\[ i = \textit{blockIdx.x} \cdot \textit{blockDim.x} + \textit{threadIdx.x}. \\]

| Symbol | Meaning | Thread 1 of block 1 |
|---|---|---|
| \\(\textit{blockIdx.x}\\) | number of the thread's block in the grid | 1 |
| \\(\textit{blockDim.x}\\) | number of threads in every block | 8 |
| \\(\textit{blockIdx.x} \cdot \textit{blockDim.x}\\) | number of threads in the blocks before this one | \\(1 \cdot 8 = 8\\) |
| \\(\textit{threadIdx.x}\\) | index of the thread within its block | 1 |
| \\(i\\) | global index: the thread's number in the whole grid | \\(8 + 1 = 9\\) |

In our example, thread 1 of block 1 computes \\(i = 1 \cdot 8 + 1 = 9\\), and it therefore computes
element 9, as shown in Figure 14.4. Every thread of the grid has a different global index. The
threads of 1 block differ in `threadIdx.x`, and threads of different blocks lie in
different ranges of `blockDim.x` consecutive numbers, so no 2 threads can arrive at the same
\\(i\\).

### 14.3.2 A grid that covers \\(n\\) elements

When the host launches a kernel, it fixes 2 numbers: the number of threads in each block, which we
call \\(B\\), and the number of blocks in the grid. To give every element its own thread, the grid
must hold at least \\(n\\) threads. A grid of \\(k\\) blocks holds \\(kB\\) threads, so we need
\\(kB \ge n\\), that is, at least \\(n / B\\) blocks. Since the number of blocks must be a whole
number, we round \\(n / B\\) up to the next integer:

\\[ \text{blocks} = \lceil n / B \rceil. \\]

| Symbol | Meaning | In the example |
|---|---|---|
| \\(n\\) | number of elements | 12 |
| \\(B\\) | number of threads in each block (`blockDim.x`) | 8 |
| \\(\lceil n / B \rceil\\) | \\(n / B\\) rounded up to the next integer | \\(\lceil 1.5 \rceil = 2\\) |

With \\(n = 12\\) and \\(B = 8\\), the quotient \\(12 / 8 = 1.5\\) is rounded up to 2, and the 2
blocks hold 16 threads. Threads 0 to 11 each own 1 element, and threads 12 to 15, which are the last
4 threads of block 1, have no element at all. In general, the grid has \\(B \lceil n / B \rceil -
n\\) such threads, which is always fewer than \\(B\\); in our example, this is \\(16 - 12 = 4\\).
For the measured arrays, \\(n = 2^{24}\\) is a multiple of \\(B = 256\\), so the grid has exactly
65,536 blocks and no idle threads.

<figure>
<img src="../../figures/ch14-grid.svg" alt="The grid for n = 12 with blocks of 8 threads: 2 blocks, 16 threads, indices i = 0 to 15. Threads with i from 0 to 11 each own 1 element of y. Threads 12 to 15, the last 4 of block 1, are dashed: i is at least n and they have no element. Thread 1 of block 1 is highlighted: i = 1 times 8 plus 1 = 9.">
<figcaption><b>Figure 14.4</b> 12 elements in 2 blocks of 8 threads. Thread 1 of block 1 owns element \(1 \cdot 8 + 1 = 9\); threads 12 to 15 own nothing.</figcaption>
</figure>

A thread without an element must not touch memory, because the positions from \\(n\\) onwards do
not belong to \\(y\\). Every thread therefore first compares its global index with \\(n\\). We now
obtain the parallel algorithm from <span class="sc">Saxpy</span> by deleting its loop and letting
the global index play the role of the loop variable \\(i\\). Every thread executes the following
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
blocks of \\(B\\) threads. Line 1 computes the global index with the formula of Section 14.3.1.
Line 2 is called the **guard**: a thread whose index lies at or beyond the end of the array skips
the update and does nothing. Line 3 is line 2 of <span class="sc">Saxpy</span>, now executed by 1
thread instead of by 1 iteration of a loop. In our example, thread 9 passes the guard, since
\\(9 < 12\\), and computes \\(y_9 = 19\\), while thread 13 fails it, since \\(13 \ge 12\\), and
does nothing.

### 14.3.3 The work of 1 thread

To understand the cost of a thread's work, we need to know where the vectors are stored. They are
stored in the main memory of the GPU.

<div class="definition">

<span class="box-label">Definition 14.5</span> **Global memory.** Global memory is the main memory
of the GPU. Every thread of every block can read and write it, it holds gigabytes, and a read
takes several hundred clock cycles to return its value. **Registers** hold the variables of a
single thread and respond in about 1 cycle.

*Example.* \\(x\\) and \\(y\\) are in global memory; the product \\(2 \cdot 8\\) computed by thread
1 of block 1 is in a register.

</div>

In Figure 14.5, we follow thread 1 of block 1. It reads \\(x_9 = 8\\) and \\(y_9 = 3\\) from
global memory, computes \\(2 \cdot 8 + 3 = 19\\) in its registers, and writes 19 back to \\(y_9\\).
In total, it makes 2 reads and 1 write, moves 12 bytes and performs 2 FLOP, which is exactly the
work of 1 iteration of the sequential loop.

<figure>
<img src="../../figures/ch14-thread.svg" alt="The work of thread 1 of block 1. It reads x[9] = 8 and y[9] = 3 from global memory, computes 2 times 8 plus 3 = 19 in registers with 2 floating-point operations, and writes 19 to y[9]: 12 bytes moved for 2 FLOP.">
<figcaption><b>Figure 14.5</b> The work of 1 thread of <span class="sc">Saxpy-Kernel</span>: 2 reads and 1 write in global memory, 2 FLOP in registers.</figcaption>
</figure>

The parallel algorithm therefore does the same total work as the sequential one, namely 24 FLOP
and 144 bytes on the example, and 12 bytes per element in general. What changes is the time. The
\\(n\\) iterations no longer form a sequence of \\(n\\) steps, and with enough threads, they can
all be performed in 1 step. Note, however, that the cost of 12 bytes per element from global memory
remains. No arrangement of threads can reduce it, because every value of \\(x\\) and \\(y\\) is
used exactly once, so there is nothing that one thread could reuse from another. In Section 14.6,
we show that this cost of 12 bytes per element decides the speed of SAXPY.

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

<span class="sc">Saxpy-Kernel</span> launches 1 thread for every element, which means 16,777,216
threads for the measured arrays. A GPU does not run so many threads at once. A Jetson Orin NX, for
example, holds at most 1,536 threads on each of its 8 SMs, which is 12,288 threads in total.
The 65,536 blocks of 256 threads therefore run in about 1,366 rounds, and each block starts when an
earlier block has finished. Nothing is wrong with this arrangement, and in Section 14.8, we measure
that it runs at full speed. There are nevertheless 3 reasons to prefer a grid with fewer threads
than elements.

1. Work that a thread does once, before or after it processes its elements, is then shared by all
   of its elements. In Chapter 15, we rely on this for the dot product: each thread first adds many
   products in a register, and only the few register sums are combined afterwards.
2. The size of the grid can be chosen to fit the GPU, from its number of SMs, and it need not
   change with \\(n\\).
3. A single prepared launch can then serve arrays of every length.

### 14.4.2 Which elements a thread takes

Suppose that the grid has \\(P\\) threads in total and that \\(P < n\\). Each thread must then take
several elements, and together the threads must take each element exactly once. In our example, we
use \\(P = 4\\) threads, in 2 blocks of 2 threads, for the 12 elements, so each thread takes
\\(12 / 4 = 3\\) elements. There are 2 natural ways to assign the elements to the threads, and
both are drawn in Figure 14.6.

- **Chunks.** Thread \\(t\\) takes the 3 neighbouring elements \\(3t\\), \\(3t + 1\\) and
  \\(3t + 2\\). Thread 3 takes elements 9, 10 and 11.
- **Stride.** Thread \\(t\\) takes elements \\(t\\), \\(t + 4\\) and \\(t + 8\\). Thread 1 takes
  elements 1, 5 and 9.

In both assignments, \\(t\\) is the global index of the thread, from 0 to 3. Both assignments are
correct, since each gives every element to exactly 1 thread. They differ, however, in the
addresses that the 4 threads use at the same moment, and this difference decides how fast the
memory can serve them. To see why, we need 2 more terms.

<div class="definition">

<span class="box-label">Definition 14.6</span> **Warp.** A **warp** is a group of 32 threads of a
block with consecutive indices, threads 0 to 31, 32 to 63, and so on. The SM issues each
instruction for a whole warp at once, so the 32 threads of a warp execute their loads at the same
moment. The position of a thread within its warp, 0 to 31, is called its **lane**.

*Example.* A block of 256 threads consists of 8 warps. In the example, the 4 threads stand for the
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

We can now compare the 2 assignments. Each thread processes its elements in steps, 1 element per
step, and in each step the threads of a warp load their elements at the same moment. In step 0 of
the chunk assignment, the 4 threads read elements 0, 3, 6 and 9, so their 4 values are spread over
40 bytes. With 32 threads per warp and chunks of 3 elements, the addresses of 1 load instruction
are \\(3 \cdot 4 = 12\\) bytes apart, and the warp touches 12 segments of 32 bytes. The memory
therefore delivers 384 bytes, of which only 128 are used. In step 0 of the stride assignment, the
threads read elements 0, 1, 2 and 3, which occupy 16 neighbouring bytes. With 32 threads, the warp
reads 128 neighbouring bytes in 4 segments, and every byte delivered is used. The stride assignment
keeps every step of every warp coalesced in this way, and Figure 14.6 shows it for all 3 steps.

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

Table 14.1 lists the stride assignment for the example. Thread 1 starts at element 1, and each
further element lies 4 places after the previous one, because 4 is the number of threads in the
grid. In general, the stride assignment gives thread \\(t\\) the elements
\\(t, t + P, t + 2P, \ldots\\) that lie below \\(n\\), where \\(t\\) is the global index of
Section 14.3.1 and \\(P\\) is the number of threads in the grid. A loop that walks through these
elements has a name.

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

Lines 2 and 4 are lines 1 and 3 of <span class="sc">Saxpy-Kernel</span>. The guard of line 2 in
<span class="sc">Saxpy-Kernel</span> has become the condition of the loop in line 3, so a thread
whose index is already at least \\(n\\) executes the loop 0 times. Line 5 advances the index by the
stride. In our example, thread 1 runs the loop 3 times, with \\(i = 1\\), 5 and 9. It computes
15, 25 and 19, and it leaves the loop when \\(i = 13\\), since \\(13 \ge 12\\).

Let us count how many elements a thread processes. We write \\(k\\) for the number of the
iteration, starting from 0, so that in iteration \\(k\\) the thread is at element \\(t + kP\\).
Thread \\(t\\) processes the elements \\(t + kP\\) that are smaller than \\(n\\), and there are
\\(\lceil (n - t) / P \rceil\\) of them. For thread 1 of our example, \\(\lceil (12 - 1) / 4
\rceil = \lceil 2.75 \rceil = 3\\), which matches its 3 iterations.

| Symbol | Meaning | Thread 1 of the example |
|---|---|---|
| \\(t\\) | global index of the thread, its first element | 1 |
| \\(P\\) | number of threads in the grid, the stride | 4 |
| \\(k\\) | number of the iteration, from 0 | 0, 1, 2 |
| \\(t + kP\\) | element processed in iteration \\(k\\) | 1, 5, 9 |
| \\(\lceil (n - t) / P \rceil\\) | number of elements the thread processes | 3 |

For \\(n = 12\\) and \\(P = 4\\), each of the 4 threads processes 3 elements. For \\(P = 5\\),
threads 0 and 1 process 3 elements, and threads 2, 3 and 4 process 2. The loop therefore handles
an \\(n\\) that is not a multiple of \\(P\\) with no extra code.

<div class="exercises">

Exercises

14.9 With \\(P = 5\\), list the elements of the example that each thread processes.<br>
14.10 A warp of 32 threads uses chunks of 4 elements. How many 32-byte segments does 1 load
instruction touch, and how many of the fetched bytes are used?<br>
14.11 How many times does the loop of <span class="sc">Grid-Stride-Saxpy</span> run for thread 7
of a grid of 2,048 threads and \\(n = 16{,}777{,}216\\)?

</div>

## 14.5 Correctness

The 2 kernels are short, but each of them depends on 1 detail that is easy to get wrong. In
<span class="sc">Saxpy-Kernel</span>, this detail is the guard, and in
<span class="sc">Grid-Stride-Saxpy</span>, it is the stride. In this section, we remove or change
each of them in turn and follow the effect on the example.

### 14.5.1 Without the guard

Suppose that we delete line 2 of <span class="sc">Saxpy-Kernel</span> and keep the launch of 2
blocks of 8 threads. Threads 12 to 15 then no longer stop. Each of them computes the address of
its element \\(y_i\\), which lies \\(4i\\) bytes after the start of \\(y\\), so the 4 threads use
the addresses 48 to 60 bytes past the start of \\(y\\). They write \\(2 x_i + y_i\\) there, into
memory that does not belong to \\(y\\), as shown in Figure 14.7.

<figure>
<img src="../../figures/ch14-overrun.svg" alt="Device memory as 4-byte slots. The first 12 hold y[0] to y[11]. The next 4 belong to other data. Without the test i less than n, threads 4 to 7 of block 1, global indices 12 to 15, write into those 4 slots.">
<figcaption><b>Figure 14.7</b> Without the guard, threads 12 to 15 write into the 4 slots after \(y\).</figcaption>
</figure>

We ran the C++ kernel without the guard. To make the damage visible, we placed 4 more slots of
other data after each vector, holding the value 1 after \\(x\\) and the value 100 after \\(y\\).
The program printed the following.

```console
y[0..12):    7 15  6 17  9 25  4 22  9 19 13 18
other data:  102 102 102 102
no error
```

The 12 elements of \\(y\\) are correct, but the 4 values after \\(y\\) have changed from 100 to
\\(2 \cdot 1 + 100 = 102\\). The GPU reported no error, because the 4 slots lay inside memory that
the program owned, and nothing therefore stopped the writes. In a real program, those slots would
hold another array, and the damage would appear later, in a computation that never touched SAXPY.

In Rust, the same kernel cannot make this mistake. Its output is a `DisjointSlice<f32>`, and the
only way to obtain a writable element of it is `y.get_mut(idx)`, which returns `None` when `idx`
lies at or beyond the end of the slice. The guard of line 2 is therefore part of the type of the
output, and a kernel that forgets the guard does not compile.

### 14.5.2 The stride: every element exactly once

<span class="sc">Grid-Stride-Saxpy</span> is correct only if every element is updated exactly
once. An element that is never updated keeps its old value. An element that is updated twice
receives \\(a x_i\\) twice, because the second update reads the result of the first. For example,
if \\(y_9\\) were updated twice, it would become \\(2 \cdot 8 + 19 = 35\\) instead of 19. We
therefore prove that the grid-stride loop updates every element exactly once.

Table 14.1 already suggests which thread updates which element. Thread 1 updates elements 1, 5 and
9, and each of these leaves the remainder 1 when divided by \\(P = 4\\). We write \\(e \bmod P\\)
for the remainder of \\(e\\) divided by \\(P\\). For example, \\(9 \bmod 4 = 1\\), since
\\(9 = 2 \cdot 4 + 1\\), and element 9 is indeed updated by thread 1. The theorem states that this
holds for every element.

<div class="theorem">

<span class="box-label">Theorem 14.1</span> When every thread of a grid of \\(P\\) threads has
finished <span class="sc">Grid-Stride-Saxpy</span>, every element \\(y_e\\) with
\\(e \in [0, n)\\) has been updated exactly once, by the thread with global index
\\(e \bmod P\\).

</div>

<div class="proof">

*Proof.* The proof has 2 parts. In the first part, we describe what 1 thread does, and in the
second part, we show that the threads together update every element once. Fix a thread with
global index \\(t \in [0, P)\\). We show that the loop of lines 3 to 6 maintains the following
invariant, a statement that is true at the start of every iteration: *at the start of iteration
\\(k\\), \\(i = t + kP\\), and the thread has updated exactly the elements \\(t + jP\\) with
\\(j \in [0, k)\\).* For thread 1 of the example, at the start of iteration 2, the invariant says
that \\(i = 1 + 2 \cdot 4 = 9\\) and that the thread has updated elements 1 and 5.

*Initialization.* Before iteration 0, line 2 has set \\(i = t\\), and the thread has not yet
updated any element. Both parts of the invariant therefore hold with \\(k = 0\\), since
\\(t + 0 \cdot P = t\\) and the range \\([0, 0)\\) is empty.

*Maintenance.* Assume that the invariant holds at the start of iteration \\(k\\), and that
\\(i < n\\), so that the iteration runs. Line 4 updates element \\(i = t + kP\\), and line 5 sets
\\(i = t + kP + P = t + (k + 1)P\\). The thread has now updated the elements \\(t + jP\\) with
\\(j \in [0, k + 1)\\), which is the invariant for iteration \\(k + 1\\).

*Termination.* The loop ends at the first \\(k\\) with \\(t + kP \ge n\\). By the invariant, the
thread has then updated exactly the elements below \\(n\\) of the form \\(t + jP\\). It has updated
each of them only once, because the values of \\(i\\) increase strictly from 1 iteration to the
next, so no value repeats.

For the second part, take any element \\(e \in [0, n)\\), and write \\(e = jP + r\\) with
\\(r = e \bmod P\\). The thread with \\(t = r\\) updates \\(e\\) in its iteration \\(j\\). Every
other thread \\(t \ne r\\) updates only elements that leave the remainder \\(t\\) when divided by
\\(P\\), so it never updates \\(e\\). Hence, \\(e\\) is updated exactly once. In the example, element
\\(e = 9 = 2 \cdot 4 + 1\\) is updated by thread 1 in its iteration 2, and by no other thread.

</div>

Note that the proof relies on 2 facts about the launch. The first is that the grid has exactly
\\(P\\) threads, with the global indices 0 to \\(P - 1\\). The second is that line 1 computes this
same \\(P\\). If line 1 computes a different number, the remainders no longer separate the
threads, and 2 threads may update the same element.

### 14.5.3 With the wrong stride

A common mistake is to advance by the number of threads in a block, `blockDim.x`, instead of the
number of threads in the grid. In our example, line 1 then computes \\(P = 2\\) instead of 4.
Global thread 0 visits elements 0, 2, 4, 6, 8 and 10. Global thread 2, which is thread 0 of
block 1, starts at element 2 and visits elements 2, 4, 6, 8 and 10. In the same way, threads 1 and 3
both visit the odd elements from 3 onwards, so each of the elements 2 to 11 is updated by 2 threads.
Figure 14.8 shows 1 order in which this can happen. Thread 0 updates \\(y_2\\) from 4 to
\\(2 \cdot 1 + 4 = 6\\), and thread 2 then reads the 6 and writes \\(2 \cdot 1 + 6 = 8\\).

<figure>
<img src="../../figures/ch14-twice.svg" alt="A grid-stride loop with the wrong stride, 2 in place of 4. Global thread 0 visits elements 0, 2, 4, 6, 8, 10 and global thread 2 visits 2, 4, 6, 8, 10. Thread 0 computes y[2] = 2 times 1 plus 4 = 6; thread 2 later reads the 6 and computes 2 times 1 plus 6 = 8. The result is 8 in place of 6.">
<figcaption><b>Figure 14.8</b> With a stride of 2 in place of 4, 2 threads update \(y_2\). If the second reads the first's result, \(y_2 = 8\) in place of 6.</figcaption>
</figure>

We ran the C++ program with `int threads = blockDim.x;` in place of
`gridDim.x * blockDim.x`. On the 12 elements of the example, the output was correct. The 2 blocks
ran at the same moment on 2 SMs, both threads read \\(y_2 = 4\\) before either of them wrote, and
both wrote 6. On the measured arrays, with 8 blocks of 256 threads and therefore \\(P = 256\\)
instead of 2,048, the program reported the following errors.

```console
  y[256] = 644, expected 324 (x = 128, y = 4)
  y[257] = 647.5, expected 326.25 (x = 128.5, y = 5)
  y[258] = 651, expected 328.5 (x = 129, y = 6)
  loop, 1 per SM: 2366 wrong elements
```

Let us check the first line. Element 256 is visited by thread 0 of block 0 in its second iteration,
since \\(0 + 1 \cdot 256 = 256\\), and by thread 0 of block 1 in its first iteration, since its
global index is 256. The measured program uses \\(a = 2.5\\) and \\(x_{256} = 128\\), so each
update adds \\(2.5 \cdot 128 = 320\\). The second visit read the result of the first and added 320
a second time, which gives \\(4 + 320 + 320 = 644\\) instead of 324. In 3 further runs, the count
of wrong elements was 3,197, 2,781 and 3,038. Most double visits read \\(y_i\\) before the other
block's write had arrived, and they therefore wrote the correct value by chance. A bug of this kind
can pass a small test and fail a large one, and it can fail by a different amount on every run.

<div class="rule-box">

<span class="box-label">Rule</span> In an in-place map, every element must be updated by exactly
1 thread, exactly once. Compute the stride from the whole grid, `gridDim.x * blockDim.x`, and the
start from the global index.

</div>

In Rust, the grid-stride kernel cannot use `get_mut`, because `get_mut` accepts only the index of
the calling thread, while the loop writes at the computed indices \\(t + kP\\). The kernel
therefore uses `get_unchecked_mut(i)` inside an `unsafe` block, and the comment above the block,
which by convention begins with `// SAFETY:`, states the argument of Theorem 14.1 in 2 lines. The
compiler does not check this argument. The wrong stride therefore compiles in Rust as well, and
the argument in the comment is then simply false.

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

Recall from Section 14.3.3 that SAXPY moves 12 bytes for every 2 FLOP that it performs. A GPU can
deliver only a limited number of bytes per second from its memory, and it can perform only a
limited number of FLOP per second. Which of the 2 limits SAXPY reaches first depends on how many
FLOP it performs per byte. To decide whether the memory or the arithmetic limits its speed, we
therefore compare this ratio with the corresponding ratio of the GPU.

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

To read the ridge point, consider the Orin NX. In the time in which its memory delivers 1 byte, its
arithmetic units can perform 23.5 FLOP. SAXPY, however, has only 0.17 FLOP to perform for every
byte, so the arithmetic units spend most of their time waiting for data. The ridge point of the
Thor is \\(8{,}100 / 273 = 29.7\\) FLOP/B. The arithmetic intensity of SAXPY lies 141 times below
the first ridge point and 178 times below the second, as shown in Figure 14.9. Every GPU in the
table of Chapter 2 has a ridge point above 10 FLOP/B, so SAXPY is memory-bound on all of them.

<figure>
<img src="../../figures/ch14-intensity.svg" alt="Top: 1 element of SAXPY reads x[i] and y[i] and writes y[i], 4 bytes each, for 2 floating-point operations: 0.17 FLOP per byte. Middle: the ridge points, 23.5 FLOP per byte for the Jetson Orin NX and 29.7 for the Jetson AGX Thor, dwarf SAXPY's 0.17. Bottom: for 16,777,216 elements on the Orin NX the memory traffic takes 1.97 ms and the arithmetic 0.014 ms.">
<figcaption><b>Figure 14.9</b> SAXPY performs 0.17 FLOP per byte, far below the ridge points of the 2 GPUs. Its memory traffic takes 140 times as long as its arithmetic.</figcaption>
</figure>

### 14.6.2 The time the memory allows

A memory-bound kernel cannot finish before all of its bytes have passed through the memory. SAXPY
on \\(n\\) elements moves \\(12n\\) bytes, and a GPU whose memory bandwidth is \\(W\\) bytes per
second needs at least \\(12n / W\\) seconds to move them. The time \\(T\\) of SAXPY therefore
satisfies

\\[ T \ge \frac{12\,n}{W}. \\]

| Symbol | Meaning | Measured arrays on the Orin NX |
|---|---|---|
| \\(T\\) | time of 1 run of SAXPY | at least 1.97 ms |
| \\(n\\) | number of elements | \\(2^{24} = 16{,}777{,}216\\) |
| \\(12\\) | bytes moved per element: 2 reads and 1 write of 4 bytes | 12 |
| \\(12n\\) | bytes moved in total | 201.3 MB |
| \\(W\\) | memory bandwidth of the GPU | 102 GB/s |

For \\(n = 2^{24}\\), the 201.3 MB of SAXPY therefore need at least
\\(201.3 \text{ MB} / 102 \text{ GB/s} = 1.97\\) ms on the Orin NX. Its 33.6 million FLOP, on the
other hand, would take only \\(33.6 \text{ MFLOP} / 2.4 \text{ TFLOP/s} = 0.014\\) ms. A faster
arithmetic unit would therefore not change the time of SAXPY at all, and only a faster memory
would. Table 14.2 gives the bound for 4 GPUs.

| GPU | Memory bandwidth | Least time for \\(n = 2^{24}\\) | FLOP rate at that time |
|---|---|---|---|
| Jetson Orin NX 16 GB | 102 GB/s | 1.97 ms | 17 GFLOP/s |
| Jetson AGX Thor | 273 GB/s | 0.74 ms | 46 GFLOP/s |
| GeForce RTX 4090 | 1,008 GB/s | 0.20 ms | 168 GFLOP/s |
| H100 SXM | 3,350 GB/s | 0.060 ms | 558 GFLOP/s |

<p class="listing"><b>Table 14.2</b> The least time of SAXPY on \(2^{24}\) elements, from the memory bandwidth. Bandwidths from NVIDIA's data sheets.</p>

The last column divides the 33.6 million FLOP by the least time. Even on an H100, SAXPY at its
bound performs 558 GFLOP/s, which is less than 1% of that GPU's single-precision peak of
67 TFLOP/s. The speed of a map is therefore measured in bytes per second, and an implementation is
judged by how close it comes to the memory bandwidth. In Section 14.8, we measure how close our
implementations come.

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

We now turn the pseudocode into code. Listing 14.1 shows <span class="sc">Saxpy-Kernel</span> in
both languages, with 1 thread per element.

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
`threadIdx.x`, and the guard of line 2 is the `if (i < n)`. In Rust, `thread::index_1d()` computes
the same global index, and `y.get_mut(idx)` performs the guard. It returns `Some` with a mutable
reference to element `idx` when the index is in range, and `None` otherwise. Recall from Chapter 13
that the type of `idx` is `ThreadIndex`, which only `thread::index_1d()` can create, so a thread
can obtain a writable reference only to its own element. In both languages, the scalar `a` is
passed by value: the host copies its 4 bytes into the launch, and every thread reads the same
value.

The 2 attributes above the Rust kernel describe how the kernel may be launched.
`#[launch_bounds(256)]` promises that no block has more than 256 threads, and
`#[launch_contract(domain = 1)]` states that the kernel uses a 1-dimensional grid. Together, they
let the host prepare a launch with any block size up to 256 and check it once, before the first
launch, so that the launch call itself needs no `unsafe`. The worked example makes use of this
freedom. The following code runs the kernel on 2 blocks of 8 threads, the grid of Figure 14.4.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn worked_example(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) -> Result<usize, Error> {
    // ...
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs:147:148}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// Runs SAXPY with a = 2 on the 12 elements of the worked example, first with 2 blocks
// of 8 threads, then with a grid-stride loop on 2 blocks of 2 threads. Returns the
// number of wrong elements.
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

We implement the grid-stride loop as a generic map, that is, as a kernel that applies any function
`f` of 2 floats, not only the function of SAXPY. The function is passed to the kernel as an
argument. In Rust, `f` is a closure, and in C++, it is a lambda.

<div class="definition">

<span class="box-label">Definition 14.11</span> **Closure.** A **closure** is a function without
a name, written inside another function, that can use variables of the function in which it is
written. A variable that a closure uses is said to be **captured**. C++ calls the same construct
a **lambda**.

*Example.* In Rust, `move |x: f32, y: f32| a * x + y` is a closure of 2 parameters that captures
`a`; in C++, `[a] __device__ (float x, float y) { return a * x + y; }` is the corresponding
lambda.

</div>

Listing 14.2 shows the grid-stride kernel, which applies the function `f` to every element that
the thread is responsible for.

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

The variable `threads` is the stride \\(P\\) of line 1, the initial value of `i` is line 2, and
the loop condition is line 3. Line 4 has become the call `f(x[i], y[i])`, and line 5 is the
increment by `threads`. In C++, the loop is a `for` statement, and in Rust, it is a `while` loop
over an ordinary `usize`. The Rust loop writes at the computed index `i`, which is not the
thread's own `ThreadIndex`, so it calls `get_unchecked_mut(i)` in an `unsafe` block, and its
`SAFETY` comment is the argument of Theorem 14.1. The method also omits the bounds check, which the
loop condition `i < y.len()` makes unnecessary.

The host passes the function as the first argument of the launch. Listing 14.3 builds the SAXPY
closure, prepares the launch for the example's 2 blocks of 2 threads, and launches the kernel.

<p class="listing"><b>Listing 14.3</b> A closure that captures \(a\), passed to the grid-stride kernel.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn worked_example(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) -> Result<usize, Error> {
    // ...
{{#include ../../../../code/oxide/ch14_map_saxpy/src/main.rs:152:156}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// Runs SAXPY with a = 2 on the 12 elements of the worked example, first with 2 blocks
// of 8 threads, then with a grid-stride loop on 2 blocks of 2 threads. Returns the
// number of wrong elements.
int workedExample()
{
    // ...
{{#include ../../../../code/cuda/ch14_map_saxpy/saxpy.cu:148:151}}
    // ...
}
```

</div>
</div>

A function argument works on the GPU through 2 mechanisms, which are the same in both languages
and are drawn in Figure 14.10. The first mechanism concerns the code of the function. Every
closure has its own type, which the program cannot name. In Rust, the kernel is therefore generic
over a type parameter `F` with the bound `Fn(f32, f32) -> f32 + Copy`. This bound states that a
value of type `F` can be called with 2 floats and returns a float, and that its values can be
copied byte for byte. The compiler generates a separate copy of the kernel for every closure type
with which the kernel is launched, a process called **monomorphisation**. In C++, the kernel is a
template, `template <typename F>`, and the compiler instantiates it once for every lambda type in
the same way. Inside each copy, the call `f(...)` is a call to 1 known function, which the compiler
can inline. The second mechanism concerns the data of the function. A closure value consists of
its captured variables. The closure `axpy` holds the 4 bytes of `a`, and the launch copies these
bytes into a kernel parameter, exactly as it copies the scalar `a` of Listing 14.1.

<figure>
<img src="../../figures/ch14-closure.svg" alt="2 closures on the host: move |x, y| a * x + y, which captures a = 2.5 (4 bytes), and |x, y| x.max(y), which captures nothing (0 bytes). Each has its own type, and the compiler builds 1 kernel map for each type; the captured bytes travel as 1 kernel parameter. C++ lambdas and the template map<F> work the same way.">
<figcaption><b>Figure 14.10</b> Each closure type gets its own compiled kernel, and each closure value travels as 1 kernel parameter.</figcaption>
</figure>

The 2 languages check different properties of the function argument. C++ requires the lambda to
be marked `__device__`, and `nvcc` accepts such a lambda in host code only with the option
`--extended-lambda`. It rejects a lambda that captures by reference, `[&]`, with the error
`An extended __device__ lambda cannot capture variables by reference`. A lambda that captures a
host pointer by value, however, compiles. In Rust, the bound `Copy` rejects any closure that owns
memory, such as a closure that captures a `Vec<f32>`, and the keyword `move` makes the closure copy
`a` instead of borrowing it. A Rust closure that borrows a host variable still compiles, because a
reference is `Copy`. Both the C++ lambda with a host pointer and the Rust closure with a host
reference then read a host address on the GPU. We show the result in the Common mistakes at the end
of the chapter.

The program also launches the 1-thread-per-element form of the generic map, the kernel `map`, with
the closure `axpy` and with a second closure, `|x, y| x.max(y)`, which captures nothing. Both
results are checked element by element against the CPU.

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

The measurements come from an integrated GPU, a Jetson Orin NX 16 GB, with 8 SMs and a memory
bandwidth of 102 GB/s. Each program fills \\(x\\) and \\(y\\) with \\(2^{24}\\) values, which is
64 MiB per vector. This is more than the 2 MiB L2 cache of the measured GPU, so every launch reads
and writes global memory rather than the cache. The values are \\(x_i = (i \bmod 1024) \cdot
0.5\\), \\(y_i = i \bmod 7\\) and \\(a = 2.5\\). With these values, every product \\(2.5\,x_i\\)
is a multiple of 1.25 below 1,280, which a float represents exactly. For example, \\(x_{256} =
128\\) gives the product 320. Since no result is rounded, the results can be compared with those of
the CPU for exact equality. Each kernel first runs for 500 ms to raise the GPU's clock, and then
50 launches are timed with CUDA events. The rate is the 201.3 MB of 1 launch divided by the time of
1 launch. Repeated runs differed by up to 5%.

### 14.8.2 Results

The Rust program printed the following lines. The C++ program prints the same lines, with "lambda"
in place of "closure".

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

The first 6 lines are the worked example. Both kernels, the kernel with 2 blocks of 8 threads and
the grid-stride loop on 2 blocks of 2 threads, produce the row of Figure 14.1, with \\(y_9 = 19\\).
In the table below the example, the column `per thr` is the number of elements per thread, and the
column `wrong` counts the elements that differ from the CPU's result, which is 0 for every kernel.
Table 14.3 collects the times of both languages.

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

We draw 3 conclusions from these measurements, which are also plotted in Figure 14.11. First, the
2 languages run at the same speed, within the variation between runs. Second, the generic map with
a closure or a lambda is as fast as the hand-written SAXPY kernel. Monomorphisation produced the
same arithmetic on the same data, and the value of `a` arrives in a kernel parameter in both cases.
Third, no kernel exceeds 93.1 GB/s, which is 91% of the 102 GB/s of the memory.

<figure>
<img src="../../figures/ch14-measured.svg" alt="Measured rates on the measured GPU, Rust and C++: saxpy 89.9 and 91.2 GB/s; map with a closure or lambda 90.9 and 90.7 GB/s; grid-stride loop with 12 blocks per SM 92.1 and 92.4 GB/s; the driver's device-to-device copy 85.0 GB/s in both. A dashed line marks 102 GB/s.">
<figcaption><b>Figure 14.11</b> Measured rates of SAXPY on the measured GPU against the memory bandwidth of 102 GB/s.</figcaption>
</figure>

### 14.8.3 Why 91 GB/s, not 102 GB/s

In Section 14.6.2, we predicted a time of at least 1.97 ms at 102 GB/s, and the kernels took
2.16 ms to 2.24 ms, which is 10% to 14% longer. The figure of 102 GB/s is the rate at the pins of
the memory chips. The measured GPU moves 16 bytes per transfer over a 128-bit bus, at 6,400 million
transfers per second, and \\(16 \text{ B} \cdot 6.4 \cdot 10^9 \text{ /s} = 102.4 \text{ GB/s}\\).
The memory, however, cannot transfer data on every one of those cycles. It must refresh its rows at
regular intervals, it must open a row before it can read from it, and it loses cycles each time the
bus turns from reading to writing. SAXPY writes 1 value for every 2 values that it reads, so its
stream of requests changes direction often.

We have 2 other measurements that bound what a kernel can expect from this memory. The driver's own
device-to-device copy, which also reads and writes, reached 85.0 GB/s, which is less than SAXPY.
The dot product of Chapter 15, which only reads, reaches 92 GB/s to 100 GB/s. The 90 GB/s to
93 GB/s of SAXPY lie between these 2 rates, at 88% to 91% of the pin rate. Nothing in the kernel
is left to remove, since it reads each input once, writes each output once, and every access is
coalesced.

### 14.8.4 How many blocks a grid-stride loop needs

The grid-stride rows of Table 14.3 show a second effect. With 1 block of 256 threads per SM, the
loop reaches only 32.9 GB/s, which is a third of the bandwidth. The rate almost doubles with 2
blocks per SM, rises again with 3, and from 6 blocks onwards, it matches the kernel with 1 thread
per element. The kernel is the same in every row, and only the number of threads changes. To
explain this effect, we use a law from the theory of queues, which relates the rate at which a
system completes requests to the number of requests that it works on at once.

<div class="definition">

<span class="box-label">Definition 14.12</span> **Little's law.** In a system that serves
requests, the **throughput** \\(X\\) is the number of requests completed per unit of time, the
**latency** \\(L\\) is the time from the start of a request to its completion, and the
**concurrency** \\(C\\) is the number of requests in progress at once. In a steady state,

\\[ X = \frac{C}{L}. \\]

*Example.* With 16,384 bytes of loads in progress and a latency of 0.50 µs, memory delivers
\\(16{,}384 \text{ B} / 0.50\ \mu\text{s} = 32.8\\) GB/s.

</div>

| Symbol | Meaning | 1 block per SM |
|---|---|---|
| \\(X\\) | throughput: bytes delivered by the memory per second | 32.9 GB/s measured |
| \\(C\\) | concurrency: bytes of loads in progress at once | 16,384 B |
| \\(L\\) | latency: time from issuing a load to receiving its data | 0.50 µs |

The reasoning behind the law is short. If the memory works on \\(C\\) bytes at a
time, and each byte takes \\(L\\) seconds to arrive, then every \\(L\\) seconds, \\(C\\) bytes are
completed, so the memory delivers \\(C / L\\) bytes per second. To raise the throughput without
changing the latency, a program must therefore keep more bytes in progress at once.

We now apply the law to the grid-stride loop. In each iteration, a thread issues its 2 loads of 4
bytes each, 8 bytes in all, and it cannot issue the loads of the next iteration before it has
stored the current result. Each thread therefore keeps about 8 bytes in flight. With 1 block per
SM, the GPU runs \\(8 \cdot 256 = 2{,}048\\) threads, which keep \\(2{,}048 \cdot 8 = 16{,}384\\)
bytes in flight. The measured 32.9 GB/s then implies a latency of
\\(L = C / X = 16{,}384 \text{ B} / 32.9 \text{ GB/s} = 0.50\\) µs. This is about 500 cycles of
the GPU's clock, which is the latency of global memory given in Definition 14.5.

If the latency stays fixed, the law predicts twice the rate for twice the concurrency. For 2 blocks
per SM, it predicts 65.8 GB/s, against 59.7 GB/s and 61.1 GB/s measured, and for 3 blocks, it
predicts 98.7 GB/s, against 78.2 GB/s and 80.5 GB/s measured. Beyond 3 blocks, the bandwidth caps
the rate, as shown in Figure 14.12. The measurements fall below the prediction at 2 and 3 blocks
for 2 reasons. The latency grows as the memory becomes busier, and a thread does not have 8 bytes
in flight at every moment, since it also computes, stores and tests the loop condition. Read the
other way, the law states how many bytes must be in flight to reach the bandwidth:
\\(C = X L = 102 \text{ GB/s} \cdot 0.50\ \mu\text{s} = 51\\) KB. At 8 bytes per thread, this is
6,375 threads, a little more than 3 blocks of 256 threads on each of the 8 SMs.

<figure>
<img src="../../figures/ch14-blocks.svg" alt="Measured rate of the grid-stride SAXPY against the number of blocks per SM on the measured GPU. Rust and C++: 1 block 32.9 and 33.8 GB/s, 2 blocks 59.7 and 61.1, 3 blocks 78.2 and 80.5, 6 blocks 88.3 and 93.1, 12 blocks 92.1 and 92.4, 48 blocks 91.9 and 90.2. A dashed line from Little's law rises in proportion to the blocks until it meets 102 GB/s.">
<figcaption><b>Figure 14.12</b> The rate of the grid-stride loop follows the bytes in flight, as Little's law predicts, until the bandwidth caps it.</figcaption>
</figure>

The measured GPU holds 1,536 threads per SM, which is 6 blocks of 256 threads. A grid of 6 blocks
per SM therefore fills the GPU in 1 round, and the measurements show no gain beyond it. In general,
a grid-stride loop should be launched with at least as many blocks as the GPU holds at once. The
host program computes this number from the number of SMs and the number of threads per SM, both of
which it reads from the device.

<div class="exercises">

Exercises

14.20 From Table 14.3, compute the fraction of the 102 GB/s that each language reaches with 1
thread per element.<br>
14.21 A GPU with 132 SMs, 2,048 threads per SM and 3,350 GB/s has a memory latency of about 0.6 µs.
How many bytes must be in flight to reach its bandwidth, and how many blocks of 256 threads per SM
provide them at 8 bytes per thread?

</div>

## 14.9 The complete programs

Listing 14.4 gives the complete programs, from which every excerpt of this chapter was taken.

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

**A missing guard.** Without `if (i < n)`, the threads of the last block whose index lies at or
beyond \\(n\\) write past the end of the array, and the GPU usually reports nothing
(Section 14.5.1). In Rust, `DisjointSlice::get_mut` makes the guard part of the type.

**A stride computed from the block instead of the grid.** `blockDim.x` in place of
`gridDim.x * blockDim.x` makes several blocks update the same elements. The result of an in-place
map is then wrong, by a different amount on each run (Section 14.5.3).

**Using `get_mut` with a computed index.** In a grid-stride loop, the index `i` is a `usize`, and
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
GPU, the next synchronisation reports
`Error: DriverError(700, "an illegal memory access was encountered")`. A C++ lambda that captures a
host pointer by value, `[hp]`, compiles and fails with the same error, "an illegal memory access was
encountered"; `nvcc` rejects a capture by reference outright.

**A grid-stride loop with too few blocks.** With 1 block of 256 threads per SM the loop reached a
third of the bandwidth (Section 14.8.4). The grid should be sized from the number of SMs, so that
it contains at least 1 full round of resident blocks.

**Measuring arrays that fit in the cache.** A GPU with a large L2 cache keeps small arrays between
launches, and the measured rate then describes the cache rather than the memory. The arrays of this
chapter, 128 MiB together, exceed the L2 cache of every GPU in Table 14.2.

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
