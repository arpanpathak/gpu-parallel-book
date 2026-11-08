<img class="plate" src="../../art/ch16.svg" alt="Abacus, a crab, stands between a row of 4 cells and a column of 4 cells. 1 claw points along the row, 1 2 3 4, the other down the column, 1 0 2 1. Beside it stands a wooden abacus with 4 rods, and the beads pushed to the left on each rod are the 4 products, 1, 0, 6 and 4. A tag reads 1 + 0 + 6 + 4 = 11: 1 entry of C, computed by 1 thread.">

# Chapter 16: Matrix Multiplication, One Thread per Entry

The product of 2 matrices is the most studied computation on the GPU. It is the core of the
dense linear algebra in scientific simulations, and the layers of a neural network spend most of
their time computing it. It is also the standard first example of a parallel algorithm, for 2
reasons. First, every entry of the result can be computed independently of every other entry,
so the work divides naturally among many threads. Second, the same input values are needed by
many entries of the result, which makes it easy to count how often each value is read from
memory, and therefore to see where the time goes.

In this chapter, we derive the simplest GPU kernel for the matrix product. We work with a single
pair of 4 × 4 matrices from the first section to the last, small enough that every number can be
checked with a pencil. We first compute the product by hand and with a sequential algorithm on
the CPU. We then introduce the parts of the GPU that the kernel needs, namely threads, blocks and
grids, and the memories of the device. After that, we write the direct kernel, in which 1 thread
computes 1 entry of the result, and we count how often it reads each input value from memory.
This count leads to the observation on which the tiled kernel of Chapter 17 is built. Finally, we
implement the kernel in Rust with cuda-oxide and in CUDA C++, and we measure it.

<div class="covers">

This chapter covers

- the matrix product, worked by hand on a 4 × 4 example
- row-major storage and the sequential algorithm, with a count of its work
- threads, blocks and grids in 1, 2 and 3 dimensions
- the memories of the device, and why global memory is the expensive one
- the direct kernel, and the number of values it fetches from global memory
- the kernel in Rust (`DisjointSlice`, `Runtime2DIndex`, launch contracts, a generic element
  type) and in CUDA C++ (a template)
- measured speeds, and why they are far below the arithmetic peak

</div>

The code of the chapter is in
[`code/oxide/ch16_matmul_naive/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch16_matmul_naive)
for Rust and in
[`code/cuda/ch16_matmul_naive/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch16_matmul_naive)
for C++.

We write algorithms in the style of Cormen, Leiserson, Rivest and Stein, with procedure names in
small capitals, numbered lines, mathematical notation, and braces that delimit blocks. The range
\\([a, b)\\) denotes the integers \\(a, a + 1, \ldots, b - 1\\), exactly like `a..b` in Rust. For
example, \\(i \in [0, 4)\\) means that \\(i\\) takes the values 0, 1, 2 and 3. Indices start at 0
throughout.

## 16.1 The matrix product

### 16.1.1 A worked example

We use the same pair of matrices throughout this chapter and the two chapters that follow it. The
2 factors, \\(A\\) and \\(B\\), and their product, \\(C\\), are

\\[
\underbrace{\begin{pmatrix} 1 & 2 & 3 & 4 \\\\ 5 & 6 & 7 & 8 \\\\ 9 & 10 & 11 & 12 \\\\ 13 & 14 & 15 & 16 \end{pmatrix}}_{A}
\times
\underbrace{\begin{pmatrix} 1 & 0 & 2 & 1 \\\\ 0 & 1 & 1 & 2 \\\\ 2 & 1 & 0 & 1 \\\\ 1 & 2 & 1 & 0 \end{pmatrix}}_{B}
=
\underbrace{\begin{pmatrix} 11 & 13 & 8 & 8 \\\\ 27 & 29 & 24 & 24 \\\\ 43 & 45 & 40 & 40 \\\\ 59 & 61 & 56 & 56 \end{pmatrix}}_{C}.
\\]

Each entry of \\(C\\) is computed from 1 row of \\(A\\) and 1 column of \\(B\\). To obtain the
top-left entry of \\(C\\), which is 11, we walk along row 0 of \\(A\\) and down column 0 of
\\(B\\) at the same time, multiply the 2 numbers we meet at each step, and add the 4 products.
Row 0 of \\(A\\) holds 1, 2, 3 and 4, and column 0 of \\(B\\) holds 1, 0, 2 and 1, so the entry is
\\(1 \cdot 1 + 2 \cdot 0 + 3 \cdot 2 + 4 \cdot 1 = 1 + 0 + 6 + 4 = 11\\), as shown in
Figure 16.1. Every other entry of \\(C\\) is computed in the same way from a different pair of a
row and a column. For example, the entry in row 0 and column 1 pairs row 0 of \\(A\\) with
column 1 of \\(B\\), which holds 0, 1, 1 and 2, and equals
\\(1 \cdot 0 + 2 \cdot 1 + 3 \cdot 1 + 4 \cdot 2 = 13\\).

<figure>
<img src="../../figures/ch16-example.svg" alt="The 4 by 4 matrices A, B and C = A B. A holds 1 to 16 row by row. Row 0 of A, 1 2 3 4, is shaded lavender, column 0 of B, 1 0 2 1, is shaded amber, and the entry c00 = 11 is shaded in C. Below, the 4 products 1 times 1, 2 times 0, 3 times 2 and 4 times 1 are added: 1 + 0 + 6 + 4 = 11.">
<figcaption><b>Figure 16.1</b> The entry c<sub>00</sub>: row 0 of A paired with column 0 of B. The 4 products sum to 11.</figcaption>
</figure>

The 2 entries we have just computed, \\(c_{00} = 11\\) and \\(c_{01} = 13\\), will appear again
in Chapter 17. In that chapter, we remove 1 line at a time from a correct kernel, and the 2
incorrect kernels that result produce the values 7 and 15 in their place.

### 16.1.2 Definition

The rule we have just used pairs a row of \\(A\\) with a column of \\(B\\), entry by entry, so a
row of \\(A\\) and a column of \\(B\\) must have the same length. We call that length \\(K\\). In
our example, every row of \\(A\\) and every column of \\(B\\) holds 4 values, so \\(K = 4\\). We
write \\(a_{is}\\) for the entry of \\(A\\) in row \\(i\\) and column \\(s\\); for example,
\\(a_{02} = 3\\) and \\(a_{21} = 10\\). In the same way, \\(b_{sj}\\) is the entry of \\(B\\) in
row \\(s\\) and column \\(j\\), and \\(c_{ij}\\) is the entry of \\(C\\) in row \\(i\\) and column
\\(j\\).

<div class="definition">

<span class="box-label">Definition 16.1</span> **Matrix product.** Let \\(A\\) be an
\\(M \times K\\) matrix and \\(B\\) a \\(K \times N\\) matrix. Their product \\(C = AB\\) is the
\\(M \times N\\) matrix with entries

\\[ c_{ij} = \sum_{s \in [0, K)} a_{is}\\, b_{sj}, \qquad i \in [0, M),\ j \in [0, N). \\]

In words, \\(c_{ij}\\) is the dot product of row \\(i\\) of \\(A\\) and column \\(j\\) of \\(B\\),
that is, the sum of the products of their entries, taken in pairs.

*Example.* In our example, \\(M = K = N = 4\\), and the definition gives
\\(c_{00} = 1 \cdot 1 + 2 \cdot 0 + 3 \cdot 2 + 4 \cdot 1 = 11\\), the sum of Figure 16.1.

</div>

| Symbol | Meaning | In the example |
|---|---|---|
| \\(M\\) | number of rows of \\(A\\) and of \\(C\\) | 4 |
| \\(N\\) | number of columns of \\(B\\) and of \\(C\\) | 4 |
| \\(K\\) | number of columns of \\(A\\) and of rows of \\(B\\), the length of each dot product | 4 |
| \\(i, j\\) | row and column of the entry of \\(C\\) being computed | 0, 0 for \\(c_{00}\\) |
| \\(s\\) | position along row \\(i\\) of \\(A\\) and down column \\(j\\) of \\(B\\) | 0, 1, 2, 3 |
| \\(a_{is}\\) | entry of \\(A\\) in row \\(i\\), column \\(s\\) | \\(a_{02} = 3\\) for \\(i = 0\\), \\(s = 2\\) |
| \\(b_{sj}\\) | entry of \\(B\\) in row \\(s\\), column \\(j\\) | \\(b_{20} = 2\\) for \\(s = 2\\), \\(j = 0\\) |

The term with \\(s = 2\\) in the sum for \\(c_{00}\\) is therefore \\(a_{02}\\, b_{20} = 3 \cdot 2
= 6\\), the third of the 4 products in Figure 16.1.

The 3 sizes play different roles. \\(M\\) is the number of rows of \\(A\\) and of \\(C\\), and
\\(N\\) is the number of columns of \\(B\\) and of \\(C\\). \\(K\\) is the length of every dot
product. Note that \\(K\\) does not appear in the shape of the result at all; it only determines
how much work each entry takes. For example, the product of a 2 × 3 matrix and a 3 × 5 matrix has
\\(M = 2\\), \\(K = 3\\) and \\(N = 5\\), and it is a 2 × 5 matrix whose 10 entries are each a
sum of 3 products. The kernels of this part accept any 3 sizes, and we measure them on square
matrices, \\(M = N = K = n\\).

### 16.1.3 Storage in row-major order

The memory of a computer is a single line of numbered locations, so a 2-dimensional matrix must be
laid out along that line. The usual choice copies the matrix out in the order in which we read a
page, that is, row 0 first, then row 1, then row 2, and so on to the last row (Figure 16.2).
Consider the entry 10 of \\(A\\), which lies in row 2 and column 1. Rows 0 and 1 come before it,
and they hold 8 values. Within row 2, 1 entry comes before it. The entry 10 is therefore preceded
by \\(8 + 1 = 9\\) values, and since locations are numbered from 0, it is stored at location
\\(2 \cdot 4 + 1 = 9\\).

<figure>
<img src="../../figures/ch16-rowmajor.svg" alt="Top: A as a 4 by 4 grid, 1 color per row; the entry 10 in row 2, column 1 is outlined. A note reads: row 2, column 1 is stored at location 2 times 4 plus 1 = 9. Bottom: the same 16 values in memory, locations 0 to 15, row 0 at 0 to 3, row 1 at 4 to 7, row 2 at 8 to 11 and row 3 at 12 to 15. The value 10 is outlined at location 9.">
<figcaption><b>Figure 16.2</b> Matrix A in row-major order, 1 color per row. Entry 10, in row 2 and column 1, lands at location 2 · 4 + 1 = 9.</figcaption>
</figure>

In general, to find the entry in row \\(r\\) and column \\(c\\), we skip the \\(r\\) complete rows
above it and then the \\(c\\) entries that precede it in its own row.

<div class="definition">

<span class="box-label">Definition 16.2</span> **Row-major order.** In row-major order, the rows
of a matrix are stored one after another. In a matrix whose rows have width \\(W\\), entry
\\((r, c)\\) is stored at location

\\[ \text{Index}(r, c, W) = r \cdot W + c. \\]

The number \\(W\\) is called the **row width** of the matrix.

*Example.* \\(A\\) has rows of width 4, so \\(a_{21} = 10\\) is stored at
\\(\text{Index}(2, 1, 4) = 9\\).

</div>

| Symbol | Meaning | For \\(a_{21} = 10\\) |
|---|---|---|
| \\(r\\) | row of the entry, which is also the number of complete rows before it | 2 |
| \\(c\\) | column of the entry, which is also the number of entries before it in its row | 1 |
| \\(W\\) | row width, the number of entries in each row | 4 |
| \\(r \cdot W + c\\) | location of the entry in memory | \\(2 \cdot 4 + 1 = 9\\) |

Row-major order has a consequence that we use often in the later chapters. Neighbors in a row
are neighbors in memory, while neighbors in a column lie \\(W\\) locations apart. For example,
\\(a_{21} = 10\\) and \\(a_{22} = 11\\) are stored at locations 9 and 10, whereas \\(a_{21} = 10\\)
and \\(a_{31} = 14\\) are stored at locations 9 and 13. In the product, matrix \\(A\\) has rows of
width \\(K\\), and matrices \\(B\\) and \\(C\\) have rows of width \\(N\\). Therefore, entry
\\(a_{is}\\) is stored at location \\(iK + s\\), entry \\(b_{sj}\\) at location \\(sN + j\\), and
entry \\(c_{ij}\\) at location \\(iN + j\\). Rust slices, C arrays, and every matrix in this book
use row-major order.

<div class="exercises">

Exercises

16.1 Compute \\(c_{23}\\) from Definition 16.1. Which row of \\(A\\) and which column of \\(B\\)
do you use?<br>
16.2 At which location is \\(b_{31}\\) stored, in row-major order with \\(W = 4\\)? Which value
does it hold?<br>
16.3 A matrix of 3 rows is stored in row-major order in 12 locations. What is its row width, and
where is entry \\((2, 0)\\)?

</div>

## 16.2 The sequential algorithm

On a CPU, Definition 16.1 becomes 3 nested loops. The outer 2 loops choose an entry of \\(C\\),
one row \\(\mathsf{row}\\) and one column \\(\mathsf{col}\\) at a time, and the inner loop adds up the \\(K\\) products
for that entry.

<div class="algo">
<p class="algo-name">Matrix-Multiply(\(A, B, C, M, K, N\))</p>
<ol>
<li><b class="kw">for</b> \(\mathsf{row} \in [0, M)\) {</li>
<li style="--d:1"><b class="kw">for</b> \(\mathsf{col} \in [0, N)\) {</li>
<li style="--d:2">\(c_{\mathsf{row}, \mathsf{col}} = 0\)</li>
<li class="gap" style="--d:2"><b class="kw">for</b> \(\mathsf{inner} \in [0, K)\) {</li>
<li style="--d:3">\(c_{\mathsf{row}, \mathsf{col}} = c_{\mathsf{row}, \mathsf{col}} + a_{\mathsf{row}, \mathsf{inner}} \cdot b_{\mathsf{inner}, \mathsf{col}}\)</li>
<li style="--d:2">}</li>
<li style="--d:1">}</li>
<li>}</li>
</ol>
</div>

Let us count the work. Line 5 executes once for every choice of \\(\mathsf{row}\\), \\(\mathsf{col}\\) and \\(\mathsf{inner}\\).
There are \\(M\\) choices of \\(\mathsf{row}\\), \\(N\\) choices of \\(\mathsf{col}\\) and \\(K\\) choices of \\(\mathsf{inner}\\), so
line 5 executes \\(MNK\\) times. Each execution is 1 **multiply-add**, that is, 1 multiplication
followed by 1 addition. We count work in **floating-point operations** (FLOP), so 1 multiply-add
is 2 FLOP, and the whole product is \\(2MNK\\) FLOP. Our example needs
\\(4 \cdot 4 \cdot 4 = 64\\) multiply-adds, or 128 FLOP. The count grows quickly with the size.
For 2 matrices of size 1,024 × 1,024, the product needs \\(1{,}024^3 = 1.07 \times 10^9\\)
multiply-adds, and for 2 matrices of size 4,096 × 4,096, it needs \\(6.9 \times 10^{10}\\)
multiply-adds, which is \\(1.37 \times 10^{11}\\) FLOP.

The CPU performs these multiply-adds one after another, and it visits the entries of \\(C\\) in
order, row by row, as shown in Figure 16.3. At visit 7, it computes \\(c_{12}\\), the entry in row
1 and column 2. Line 5 runs 4 times, for \\(\mathsf{inner} = 0, 1, 2, 3\\), and pairs row 1 of \\(A\\), which
holds 5, 6, 7 and 8, with column 2 of \\(B\\), which holds 2, 1, 0 and 1. The result is
\\(c_{12} = 5 \cdot 2 + 6 \cdot 1 + 7 \cdot 0 + 8 \cdot 1 = 24\\).

<figure>
<img src="../../figures/ch16-sequential.svg" alt="C as a 4 by 4 grid whose cells hold the order in which the sequential algorithm visits them, 1 to 16 row by row. Visit 7, entry c12, is highlighted. The text says that at visit 7 line 5 runs for inner = 0 to 3 and computes c12 = 5 times 2 + 6 times 1 + 7 times 0 + 8 times 1 = 24, and that 16 visits of 4 steps make 64 multiply-adds, performed one after another.">
<figcaption><b>Figure 16.3</b> The order in which <span style="font-variant: small-caps">Matrix-Multiply</span> fills C. Visit 7 computes c<sub>12</sub> = 24 with 4 multiply-adds.</figcaption>
</figure>

Nothing in the algorithm requires this order. Visit 7 reads row 1 of \\(A\\) and column 2 of
\\(B\\), which no visit changes, and it writes \\(c_{12}\\), which no other visit reads or writes.
The same holds for each of the 16 visits. The visits are therefore independent of one another,
and they could all take place at the same time. A GPU is built for exactly this situation, and in
the next section, we describe the parts of it that we need.

<div class="exercises">

Exercises

16.4 Show that <span style="font-variant: small-caps">Matrix-Multiply</span> reads each entry of
\\(A\\) exactly \\(N\\) times. How many times does it read each entry of \\(B\\)?<br>
16.5 How many FLOP does the product of a 300 × 500 matrix and a 500 × 700 matrix take?<br>
16.6 Exchange the loops over \\(\mathsf{col}\\) and \\(\mathsf{inner}\\) in the algorithm. Is the result still correct?
Which line must move?

</div>

## 16.3 The graphics processor

### 16.3.1 Host and device

A computer with a GPU contains 2 processors, each with its own memory. We call the CPU and its
memory the **host**, and the GPU and its memory the **device**. Data move between the 2 memories
only when the program copies them explicitly. For this reason, every program of this part follows
the same 5 steps, shown in Figure 16.4. The host creates \\(A\\) and \\(B\\), copies them to the
device, and launches the computation; when the computation has finished, the host copies \\(C\\)
back and checks it.

<figure>
<img src="../../figures/ch16-host-device.svg" alt="Top: the 5 steps of every program in a row: Host creates A and B; Copy in, A and B to the GPU; Launch, threads compute C; Copy out, C to the host; Verify, compare on the host. The first and last steps run on the host, the middle 3 on the device. Bottom: the system. The host holds a CPU running main() and the host memory with a_host and b_host. The device holds SMs 0 to p minus 1, which run blocks, and the global memory with a_gpu, b_gpu and c_gpu. A link between the 2 memories carries only explicit copies.">
<figcaption><b>Figure 16.4</b> The 5 steps of every program, and the system. Host and device each have a processor and a memory, and the link between them carries only explicit copies.</figcaption>
</figure>

On an integrated GPU, the host and the device share 1 physical memory. The program nevertheless
copies between the host's buffers and the device's buffers, and the 5 steps are the same.

### 16.3.2 Threads, blocks and grids

Recall from Section 16.2 that no entry of \\(C\\) depends on any other, so all 16 entries may be
computed at the same time. A GPU takes advantage of such independence by running 1 function many
times at once, each time on a different piece of the data.

<div class="definition">

<span class="box-label">Definition 16.3</span> **Kernel and thread.** A **kernel** is a function
that the GPU runs many times at once. Each of these runs is called a **thread**. Every thread has
its own position among the threads and its own variables.

*Example.* We will run the direct kernel of Section 16.4 with 16 threads, 1 per entry of
\\(C\\). The thread at row 1 and column 2 computes \\(c_{12} = 24\\).

</div>

The GPU organizes the threads of a kernel on 2 levels. Threads are grouped into blocks, and the
blocks together form a grid.

<div class="definition">

<span class="box-label">Definition 16.4</span> **Block and grid.** A **block** is a group of
threads that run on the same processor of the device, share a region of fast memory, and can
wait for one another. The **grid** is the set of all blocks of 1 launch.

*Example.* With blocks of 2 × 2 threads, a grid of 2 × 2 blocks covers \\(C\\). Block
\\((0, 0)\\) computes \\(c_{00}\\), \\(c_{01}\\), \\(c_{10}\\) and \\(c_{11}\\).

</div>

<figure>
<img src="../../figures/ch16-grid.svg" alt="A grid of 2 by 2 blocks over the 4 by 4 matrix C, 1 color per block: block (0, 0) lavender, (0, 1) mint, (1, 0) peach and (1, 1) pink. Each block holds 4 threads labeled (threadIdx.y, threadIdx.x) from (0, 0) to (1, 1). Thread (1, 0) of block (0, 1) is outlined. The panel on the right: the block covers rows 0 to 1 and columns 2 to 3; row = 0 times 2 + 1 = 1, col = 1 times 2 + 0 = 2, so the thread computes c12 = 24.">
<figcaption><b>Figure 16.5</b> A grid of 2 × 2 blocks of 2 × 2 threads over C, 1 color per block. Thread (1, 0) of block (0, 1) computes c<sub>12</sub>.</figcaption>
</figure>

Blocks and grids can extend along 1, 2 or 3 axes, named \\(x\\), \\(y\\) and \\(z\\), and we choose
the shape to match the data. For each thread, the GPU provides 3 values from which the thread can
work out which piece of the data belongs to it. The value *blockIdx* is the position of the
thread's block within the grid, *threadIdx* is the position of the thread within its block, and
*blockDim* is the size of a block, that is, the number of threads it holds along each axis. Each
of the 3 values has 1 component per axis, written with a suffix; for example, *threadIdx.x* is
the position of the thread along the axis \\(x\\). We first see how a thread uses these values in
1 dimension, where the computation is easiest to follow.

**1 dimension.** Suppose that a vector of 12 elements is handled by blocks of 4 threads, so that
3 blocks lie side by side, as in Figure 16.6. Consider thread 1 of block 2. Blocks 0 and 1 come
before block 2, and together they cover elements 0 to 7. Block 2 therefore starts at element 8,
and its thread 1 handles element \\(8 + 1 = 9\\). In general, a block starts at element
\\(\mathsf{blockIdx{.}x} \cdot \mathsf{blockDim{.}x}\\), because each of the \\(\mathsf{blockIdx{.}x}\\) blocks before it covers
\\(\mathsf{blockDim{.}x}\\) elements. A thread then adds its own position within the block, so it
handles element

\\[ \mathsf{index} = \mathsf{blockIdx{.}x} \cdot \mathsf{blockDim{.}x} + \mathsf{threadIdx{.}x}. \\]

| Symbol | Meaning | Thread 1 of block 2 |
|---|---|---|
| \\(\mathsf{blockIdx{.}x}\\) | position of the block in the grid | 2 |
| \\(\mathsf{blockDim{.}x}\\) | number of threads in a block | 4 |
| \\(\mathsf{threadIdx{.}x}\\) | position of the thread within its block | 1 |
| \\(\mathsf{index}\\) | element that the thread handles | \\(2 \cdot 4 + 1 = 9\\) |

<figure>
<img src="../../figures/ch16-1d.svg" alt="12 elements in 3 blocks of 4 threads, 1 color per block. Each block's threads are labeled threadIdx.x = 0 to 3, and below them the element index runs from 0 to 11. Thread 1 of block 2 is outlined, above index 9, and a line reads: block 2, thread 1: index = 2 times 4 plus 1 = 9.">
<figcaption><b>Figure 16.6</b> 1 dimension: 12 elements in 3 blocks of 4 threads. Thread 1 of block 2 handles element 2 · 4 + 1 = 9.</figcaption>
</figure>

**2 dimensions.** A matrix has rows and columns, so we arrange the threads of a block, and the
blocks of the grid, in rows and columns as well. The axis \\(x\\) counts columns, and the axis
\\(y\\) counts rows. For the matrix product, we cut \\(C\\) into squares and give each square to one block. Each
thread computes one entry of its block's square. The number of threads along a row of the block
is \\(\mathsf{blockDim{.}x}\\), and the number along a column is \\(\mathsf{blockDim{.}y}\\). We use
square blocks, so the 2 numbers are equal. In our example, \\(\mathsf{blockDim{.}x} = \mathsf{blockDim{.}y} = 2\\), so each block holds 4
threads, which compute a 2 × 2 square of \\(C\\), and 2 × 2 blocks cover the 4 × 4 matrix, as in
Figure 16.5.

We write all positions as (row, column). The position of a thread within its block is
(\\(\mathsf{threadIdx{.}y}\\), \\(\mathsf{threadIdx{.}x}\\)): \\(\mathsf{threadIdx{.}y}\\) is the row of the thread within its block,
and \\(\mathsf{threadIdx{.}x}\\) is its column. In the same way, the position of a block within the grid
is (\\(\mathsf{blockIdx{.}y}\\), \\(\mathsf{blockIdx{.}x}\\)). Thus, thread \\((1, 0)\\) of block \\((0, 1)\\),
which is outlined in Figure 16.5, is the thread in row 1 and column 0 of the block that lies in
row 0 and column 1 of the grid. For this thread, \\(\mathsf{threadIdx{.}y} = 1\\), \\(\mathsf{threadIdx{.}x} = 0\\),
\\(\mathsf{blockIdx{.}y} = 0\\) and \\(\mathsf{blockIdx{.}x} = 1\\).

Let us find the entry of \\(C\\) that this thread computes. Its block covers rows 0 and 1 and
columns 2 and 3 of \\(C\\). Each block row of the grid covers \\(\mathsf{blockDim{.}y}\\) rows of \\(C\\), and each
block column covers \\(\mathsf{blockDim{.}x}\\) columns, so the top-left corner of block \\((0, 1)\\) is at row
\\(0 \cdot 2 = 0\\) and column \\(1 \cdot 2 = 2\\). Within the block, the thread lies 1 row down
and 0 columns across from that corner. It therefore computes the entry in row \\(0 + 1 = 1\\) and
column \\(2 + 0 = 2\\), which is \\(c_{12}\\). The same reasoning applies to every thread. We call
the row and the column of the entry that a thread computes \\(\mathsf{row}\\) and
\\(\mathsf{col}\\), and they are

\\[ \mathsf{row} = \mathsf{blockIdx{.}y} \cdot \mathsf{blockDim{.}y} + \mathsf{threadIdx{.}y}, \qquad
\mathsf{col} = \mathsf{blockIdx{.}x} \cdot \mathsf{blockDim{.}x} + \mathsf{threadIdx{.}x}. \\]

| Symbol | Meaning | Thread \\((1, 0)\\) of block \\((0, 1)\\) |
|---|---|---|
| \\(\mathsf{blockIdx{.}y}\\) | row of the block within the grid | 0 |
| \\(\mathsf{blockIdx{.}x}\\) | column of the block within the grid | 1 |
| \\(\mathsf{threadIdx{.}y}\\) | row of the thread within its block | 1 |
| \\(\mathsf{threadIdx{.}x}\\) | column of the thread within its block | 0 |
| \\(\mathsf{blockDim{.}y}\\), \\(\mathsf{blockDim{.}x}\\) | number of threads along each side of a block | 2 |
| \\(\mathsf{row}\\) | row of the entry of \\(C\\) that the thread computes | \\(0 \cdot 2 + 1 = 1\\) |
| \\(\mathsf{col}\\) | column of the entry of \\(C\\) that the thread computes | \\(1 \cdot 2 + 0 = 2\\) |

As a second example, thread \\((0, 1)\\) of block \\((0, 0)\\) has \\(\mathsf{threadIdx{.}y} = 0\\) and \\(\mathsf{threadIdx{.}x} = 1\\),
and it computes the entry with \\(\mathsf{row} = 0 \cdot 2 + 0 = 0\\) and
\\(\mathsf{col} = 0 \cdot 2 + 1 = 1\\), that is, \\(c_{01} = 13\\). Within block \\((0, 0)\\),
the 4 threads \\((0, 0)\\), \\((0, 1)\\), \\((1, 0)\\) and \\((1, 1)\\) compute \\(c_{00}\\),
\\(c_{01}\\), \\(c_{10}\\) and \\(c_{11}\\), so for this block, the position of a thread and the
position of its entry coincide. In cuda-oxide, the values \\(\mathsf{row}\\) and
\\(\mathsf{col}\\) are returned by `thread::index_2d_row()` and `thread::index_2d_col()`.

The choice of \\(x\\) for the column is deliberate. The GPU executes the threads of a block in
groups of 32 threads with consecutive positions, and it counts positions along \\(x\\) first.

<div class="definition">

<span class="box-label">Definition 16.5</span> **Warp.** A **warp** is a group of 32 threads of
a block with consecutive positions, counted along \\(x\\) first, that the hardware issues
together: 1 instruction is fetched and executed for all 32 threads at once.

*Example.* In a block of 16 × 16 threads, threads \\((0, 0)\\) to \\((0, 15)\\) and \\((1, 0)\\)
to \\((1, 15)\\) form warp 0.

</div>

Since \\(x\\) runs along the columns, the threads of a warp compute neighboring entries of a row
of \\(C\\). For example, the 16 threads \\((0, 0)\\) to \\((0, 15)\\) of warp 0 compute 16
neighboring entries of 1 row of \\(C\\). At each step of the dot product, these threads read the
same entry of \\(A\\) and 16 neighboring entries of a row of \\(B\\), which lie next to each other
in memory by Definition 16.2. The memory delivers such neighboring values together, in 1
transaction. In Section 9.3, we measured the opposite choice, with \\(x\\) along the rows, and
found it to be 5.4 times slower.

**3 dimensions.** For data in a volume, such as temperatures on a 3-dimensional mesh, we add the
axis \\(z\\) and reason in the same way along each of the 3 axes. Along \\(x\\), a thread has the
position

\\[ x = \mathsf{blockIdx{.}x} \cdot \mathsf{blockDim{.}x} + \mathsf{threadIdx{.}x}, \\]

and the positions \\(y\\) and \\(z\\) are computed in the same way from the components \\(y\\) and
\\(z\\) of the 3 values. Figure 16.7 covers a 4 × 2 × 4 volume with 4 blocks of 2 × 2 × 2
threads. Here, we write positions in the order \\((x, y, z)\\). Thread \\((1, 0, 0)\\) of block
\\((1, 0, 1)\\) lies at \\(x = 1 \cdot 2 + 1 = 3\\), \\(y = 0 \cdot 2 + 0 = 0\\) and
\\(z = 1 \cdot 2 + 0 = 2\\).

| Symbol | Meaning | Thread \\((1, 0, 0)\\) of block \\((1, 0, 1)\\), along \\(x\\) |
|---|---|---|
| \\(\mathsf{blockIdx{.}x}\\) | position of the block along \\(x\\) | 1 |
| \\(\mathsf{blockDim{.}x}\\) | number of threads in a block along \\(x\\) | 2 |
| \\(\mathsf{threadIdx{.}x}\\) | position of the thread within its block along \\(x\\) | 1 |
| \\(x\\) | position of the thread in the volume along \\(x\\) | \\(1 \cdot 2 + 1 = 3\\) |

<figure>
<img src="../../figures/ch16-3d.svg" alt="A box of 4 by 2 by 4 cells drawn in perspective, cut into 4 blocks of 2 by 2 by 2 threads, 1 color per block. Axes: x to the right, y downwards, z into the page. The cell of thread (1, 0, 0) in block (1, 0, 1) is outlined on the top and right faces. The text computes its position: x = 1 times 2 + 1 = 3, y = 0 times 2 + 0 = 0, z = 1 times 2 + 0 = 2.">
<figcaption><b>Figure 16.7</b> 3 dimensions: a 4 × 2 × 4 volume in 4 blocks of 2 × 2 × 2 threads. Thread (1, 0, 0) of block (1, 0, 1) sits at x = 3, y = 0, z = 2.</figcaption>
</figure>

When we launch a kernel, we give the size of the grid and the size of a block, each as a triple
\\((x, y, z)\\). For the matrix product, we use a grid of
\\((\lceil N/\mathsf{blockDim{.}x} \rceil, \lceil M/\mathsf{blockDim{.}y} \rceil, 1)\\) blocks. The
notation \\(\lceil N/\mathsf{blockDim{.}x} \rceil\\) denotes the quotient rounded up to the next integer.
The grid needs \\(\lceil N/\mathsf{blockDim{.}x} \rceil\\) blocks along \\(x\\) to cover the \\(N\\) columns of \\(C\\),
and \\(\lceil M/\mathsf{blockDim{.}y} \rceil\\) blocks along \\(y\\) to cover its \\(M\\) rows. In our example,
\\(N = M = 4\\) and the blocks are 2 × 2, so the grid has \\(\lceil 4/2 \rceil = 2\\) blocks along each axis,
as in Figure 16.5. The rounding up matters when the block width does not divide the size of the matrix.
For example, with \\(N = 1{,}000\\) and blocks of 16 × 16 threads, the quotient is \\(1{,}000 / 16 = 62.5\\), so
the grid needs 63 blocks along \\(x\\). These 63 blocks cover \\(63 \cdot 16 = 1{,}008\\) columns,
and the last block therefore has 8 threads in each of its rows that lie past the last column of
\\(C\\).

### 16.3.3 Scheduling and scope

The device executes blocks on processors of its own, which are called streaming multiprocessors.

<div class="definition">

<span class="box-label">Definition 16.6</span> **Streaming multiprocessor.** A **streaming
multiprocessor** (SM) is 1 of the processors of a GPU. It contains arithmetic units, registers
and shared memory, and it runs whole blocks: at launch, each block is assigned to 1 SM and stays
there until it finishes.

*Example.* The Jetson Orin NX has 8 SMs, and the Jetson AGX Thor has 20.

</div>

An SM can hold several blocks at once. If the grid has more blocks than the SMs can hold, the
remaining blocks wait, and they run in later **waves**, as shown in Figure 16.8. In the figure, 2
SMs hold 2 blocks each, so 4 of the 8 blocks run in the first wave and the other 4 in the second.
The GPU runs blocks in no guaranteed order. Therefore, a correct kernel must never depend on the
order in which its blocks run.

<figure>
<img src="../../figures/ch16-waves.svg" alt="8 blocks wait on the left. 2 SMs each hold 2 blocks at a time. In the first wave SM 0 runs blocks 0 and 2 and SM 1 runs blocks 1 and 3; in the second wave SM 0 runs blocks 4 and 6 and SM 1 runs blocks 5 and 7. A time axis runs underneath.">
<figcaption><b>Figure 16.8</b> 8 blocks on 2 SMs that hold 2 blocks each. Blocks 0 to 3 run in the first wave and blocks 4 to 7 in the second.</figcaption>
</figure>

The block is also the unit of cooperation between threads. Threads in the same block share a
region of shared memory and can wait for one another at a barrier, which we define in Chapter 17.
Threads in different blocks can do neither, and they can exchange data only through global
memory, as shown in Figure 16.9. The direct kernel of this chapter needs no cooperation at all,
since every thread works alone. The tiled kernel of Chapter 17, in contrast, depends on it.

<figure>
<img src="../../figures/ch16-cooperation.svg" alt="2 blocks, each with 4 threads drawn as circles. In each block, arrows run from the 4 threads to the shared memory of that block, and a line underneath marks a barrier for the 4 threads of the block. A red dashed line with a cross between the 2 blocks shows that they share neither shared memory nor a barrier. Both blocks reach global memory below, the only memory that threads of different blocks share.">
<figcaption><b>Figure 16.9</b> Threads of 1 block share shared memory and a barrier. Threads of different blocks can exchange data only through global memory.</figcaption>
</figure>

### 16.3.4 Memories of the device

To compute \\(c_{12}\\), a thread needs row 1 of \\(A\\) and column 2 of \\(B\\). How long the
thread waits for these values depends on where they are stored, so we now describe the memories
inside the device, which are shown in Figure 16.10.

<figure>
<img src="../../figures/ch16-memories.svg" alt="3 SM cards side by side: SM 0, SM 1 and SM p minus 1. Each holds a grid of arithmetic units, registers that hold 1 thread's variables, and shared memory that holds 1 block's tiles. Blue labels show 1 possible placement: blocks (0, 0) and (1, 0) on SM 0, blocks (0, 1) and (1, 1) on SM 1. All SMs connect to an L2 cache shared by all SMs, which holds recently used values of global memory, and below it to global memory, the device DRAM, which holds A, B and C.">
<figcaption><b>Figure 16.10</b> The device. Each SM contains arithmetic units, registers and shared memory and runs whole blocks; all SMs reach global memory through a common L2 cache.</figcaption>
</figure>

<div class="definition">

<span class="box-label">Definition 16.7</span> **Registers, shared memory, global memory.**
**Registers** hold the variables of a single thread and respond in about 1 cycle. **Shared
memory** belongs to a block. Every thread of the block can read and write it, no other thread
can, it responds within a few cycles, and values enter it only when threads of the block store
them there. **Global memory** is visible to every thread, holds gigabytes, and takes several
hundred cycles to respond. Between the SMs and global memory sits the **L2 cache**, which keeps
recently used values of global memory; the program has no control over its contents. Each SM
also has a smaller **L1 cache** of its own.

*Example.* In the direct kernel, the variable `sum` lives in a register, and \\(A\\), \\(B\\) and
\\(C\\) live in global memory. The tiled kernel of Chapter 17 adds tiles in shared memory.

</div>

| Storage | Visible to | Latency | Capacity |
|---|---|---|---|
| Registers | 1 thread | about 1 cycle | 65,536 values of 4 bytes per SM |
| Shared memory | 1 block | a few cycles | 48 to 228 KB per SM |
| L2 cache | all SMs | tens to 100 cycles | a few MB to tens of MB |
| Global memory | all threads | hundreds of cycles | gigabytes |

The table suggests a strategy. Arithmetic and registers are cheap, whereas global memory is
expensive, since a read from it takes hundreds of cycles. A fast kernel therefore fetches each
value from global memory as few times as possible, and it takes its operands from storage closer
to the arithmetic units whenever it can. The direct kernel of the next section does not yet do
this, and we will count what that costs.

<div class="exercises">

Exercises

16.7 Which entry of \\(C\\) does thread \\((0, 1)\\) of block \\((1, 1)\\) compute in Figure 16.5?<br>
16.8 With blocks of 16 × 16 threads, how many blocks cover a 1,000 × 1,000 matrix \\(C\\)? How many threads
in the edge blocks have no entry to compute?<br>
16.9 A grid has 132 blocks, and the device has 66 SMs that each hold 1 block at a time. How many
waves are needed?

</div>

## 16.4 The direct kernel

### 16.4.1 The algorithm

We obtain the simplest GPU kernel from <span style="font-variant: small-caps">Matrix-Multiply</span>
by removing its 2 outer loops. In the sequential algorithm, these loops chose the entry
\\(c_{ij}\\) to compute. In the kernel, there is one thread for every entry of \\(C\\), and the
position of a thread in the grid takes the place of the 2 outer loops. The thread computes the
entry \\(c_{\mathsf{row}, \mathsf{col}}\\), whose row and column come from the formulas of
Section 16.3.2. Every thread executes the following procedure.

<div class="algo">
<p class="algo-name">Direct-Kernel(\(A, B, C, M, K, N\))</p>
<ol>
<li>\(\mathsf{row} = \mathsf{blockIdx{.}y} \cdot \mathsf{blockDim{.}y} + \mathsf{threadIdx{.}y}\)</li>
<li>\(\mathsf{col} = \mathsf{blockIdx{.}x} \cdot \mathsf{blockDim{.}x} + \mathsf{threadIdx{.}x}\)</li>
<li class="gap"><b class="kw">if</b> \(\mathsf{row} \notin [0, M)\) <b class="kw">or</b> \(\mathsf{col} \notin [0, N)\) { <span class="c">// thread lies outside C</span></li>
<li style="--d:1"><b class="kw">return</b></li>
<li>}</li>
<li class="gap">\(\mathsf{sum} = 0\)</li>
<li class="gap"><b class="kw">for</b> \(\mathsf{inner} \in [0, K)\) {</li>
<li style="--d:1">\(\mathsf{sum} = \mathsf{sum} + a_{\mathsf{row}, \mathsf{inner}} \cdot b_{\mathsf{inner}, \mathsf{col}}\) <span class="c">// 2 reads from global memory</span></li>
<li>}</li>
<li class="gap">\(c_{\mathsf{row}, \mathsf{col}} = \mathsf{sum}\)</li>
</ol>
</div>

Lines 1 and 2 compute the position of the thread's entry with the formulas of Section 16.3.2.
Lines 3 to 5 end the thread at once if its entry lies outside \\(C\\), which happens when the
size of the matrix is not a multiple of the block width. Lines 6 to 9 are the inner loop of
<span style="font-variant: small-caps">Matrix-Multiply</span>, with the partial sum kept in the
variable \\(\mathsf{sum}\\), which lives in a register. Line 10 stores the result in global
memory. We let every thread write a different entry of \\(C\\), and every thread reads only \\(A\\) and
\\(B\\), which no thread writes. Therefore, the threads never interfere with one another. In
Figure 16.11, we follow a single thread through its work.

<figure>
<img src="../../figures/ch16-one-thread.svg" alt="Matrices A, B and C. Row 1 of A, 5 6 7 8, is shaded lavender; column 2 of B, 2 1 0 1, is shaded amber; in C only the entry c12 = 24 is filled. Text below: thread (1, 0) of block (0, 1) has row 1 and column 2; sum = 5 times 2 + 6 times 1 + 7 times 0 + 8 times 1 = 24, stored in c12; 8 reads from global memory and 1 write.">
<figcaption><b>Figure 16.11</b> The work of 1 thread of the direct kernel: row 1 of A times column 2 of B gives c<sub>12</sub> = 24, with 8 fetches from global memory.</figcaption>
</figure>

The thread is thread \\((1, 0)\\) of block \\((0, 1)\\), in blocks of 2 × 2 threads. Lines 1 and 2 give
\\(\mathsf{row} = 0 \cdot 2 + 1 = 1\\) and
\\(\mathsf{col} = 1 \cdot 2 + 0 = 2\\). Both values lie inside \\(C\\), so the test of line 3
fails and the thread continues. The loop runs for \\(\mathsf{inner} = 0, 1, 2, 3\\). In the pass with
\\(\mathsf{inner} = 0\\), line 8 reads \\(a_{10} = 5\\) and \\(b_{02} = 2\\) and adds their product, 10, to
\\(\mathsf{sum}\\). The remaining 3 passes add \\(6 \cdot 1\\), \\(7 \cdot 0\\) and
\\(8 \cdot 1\\), so that \\(\mathsf{sum}\\) ends at
\\(5 \cdot 2 + 6 \cdot 1 + 7 \cdot 0 + 8 \cdot 1 = 24\\), and line 10 stores 24 in \\(c_{12}\\).
Each pass of line 8 fetches 1 value of \\(A\\) and 1 value of \\(B\\) from global memory, so the
thread makes \\(4 \cdot 2 = 8\\) fetches and 1 write.

### 16.4.2 Correctness

The direct kernel needs only a short argument. Fix a thread with \\(\mathsf{row} \in [0, M)\\) and
\\(\mathsf{col} \in [0, N)\\). We claim that before the pass of line 8 with index \\(\mathsf{inner}\\), the
variable `sum` holds the first \\(\mathsf{inner}\\) terms of the dot product, that is,

\\[ \mathsf{sum} = \sum_{j \in [0, \mathsf{inner})} a_{\mathsf{row}, j}\\, b_{j, \mathsf{col}}. \\]

Here \\(j\\) runs over the terms that have already been added, from 0 to \\(\mathsf{inner} - 1\\). For
example, for thread \\((1, 0)\\) of block \\((0, 1)\\), before the pass with \\(\mathsf{inner} = 2\\), `sum`
holds \\(a_{10}\\, b_{02} + a_{11}\\, b_{12} = 5 \cdot 2 + 6 \cdot 1 = 16\\). The claim holds for
\\(\mathsf{inner} = 0\\), where the sum is empty and `sum` is 0 by line 6, and each pass of line 8 adds the
next term, so the claim holds before every pass. When the loop ends, \\(\mathsf{inner} = K\\), and `sum` holds
all \\(K\\) terms, which is \\(c_{\mathsf{row}, \mathsf{col}}\\) by Definition 16.1. Finally, no
other thread has the same \\((\mathsf{row}, \mathsf{col})\\), because any 2 threads differ either
in their block or in their position within it. Therefore, no other thread writes that entry.

The only part of the kernel that requires care is the guard of lines 3 to 5. Without the guard, a
thread past the last column computes an index \\(\mathsf{row} \cdot N + \mathsf{col}\\) that lands
in the next row. For example, with \\(N = 1{,}000\\), the thread with \\(\mathsf{row} = 0\\) and
\\(\mathsf{col} = 1{,}000\\) would write location 1,000, which holds the entry in row 1 and column
0. A thread past the last row writes beyond the end of \\(C\\). To observe this, we removed lines
3 to 5 from the C++ kernel and ran it on matrices of 1,000 × 1,000. A grid of 63 × 63 blocks of
16 × 16 threads covers 1,008 rows and 1,008 columns, so it overhangs the matrices by 8 rows and 8
columns. The program reported the following error.

```console
kernel: an illegal memory access was encountered
```

The threads in the 8 extra rows read past the end of \\(A\\) and wrote past the end of \\(C\\),
and the GPU stopped the kernel. A smaller overhang can stay inside memory that the program owns.
In that case, the kernel finishes, and the error shows only in the values of \\(C\\). In
Section 16.5.6, we make the same change in Rust, where the outcome is different.

### 16.4.3 Counting the fetches

The kernel is correct, but it reads far more from global memory than it needs to. Consider the
value \\(a_{01} = 2\\), which lies in row 0 of \\(A\\). All 4 entries of row 0 of \\(C\\), namely
\\(c_{00}\\), \\(c_{01}\\), \\(c_{02}\\) and \\(c_{03}\\), are computed from row 0 of \\(A\\), so
4 threads need \\(a_{01}\\). In the direct kernel, each of these 4 threads fetches \\(a_{01}\\)
from global memory on its own, as shown in Figure 16.12.

<figure>
<img src="../../figures/ch16-waste.svg" alt="Matrix A on the left with the value a01 = 2 highlighted, and C on the right with the 4 entries c00, c01, c02 and c03 of row 0 shaded. 4 separate curved arrows run from a01 to the 4 entries: each of the 4 threads that compute row 0 of C fetches a01 from global memory on its own. A note below: in all, 16 threads times 8 fetches = 128 fetches from global memory, although A and B hold only 32 distinct values.">
<figcaption><b>Figure 16.12</b> The 4 threads that compute row 0 of C each fetch a<sub>01</sub> from global memory: 4 fetches of the same value. In all, 128 fetches of 32 distinct values.</figcaption>
</figure>

The same holds for every value. Each entry of \\(A\\) is fetched by the 4 threads that compute
its row of \\(C\\), and each entry of \\(B\\) is fetched by the 4 threads that compute its column
of \\(C\\). For example, \\(b_{21} = 1\\) is fetched by the threads that compute \\(c_{01}\\),
\\(c_{11}\\), \\(c_{21}\\) and \\(c_{31}\\). In total, 16 threads each fetch 4 values of \\(A\\)
and 4 values of \\(B\\), which makes 128 fetches, although the 2 matrices hold only 32 distinct
values.

In general, the kernel runs one thread for each of the \\(MN\\) entries of \\(C\\), and each thread
makes \\(2K\\) fetches, \\(K\\) from its row of \\(A\\) and \\(K\\) from its column of \\(B\\). The
direct kernel therefore performs \\(MN \cdot 2K = 2MNK\\) fetches, which is 1 fetch for every FLOP.
In our example, \\(MN = 16\\) threads each make \\(2K = 8\\) fetches, which gives the 128 fetches
counted above. Each entry of \\(A\\) is fetched \\(N\\) times, once by each of the \\(N\\) threads
of its row of \\(C\\), and each entry of \\(B\\) is fetched \\(M\\) times.

For realistic sizes, these numbers become large. For 2 matrices of 1,024 × 1,024 floats of 4
bytes each, the kernel makes \\(2 \cdot 1{,}024^3 = 2.1 \times 10^9\\) fetches, or 8.6 GB of
traffic, to read only 8.4 MB of input. Every value is fetched 1,024 times, and each fetch that
reaches global memory costs hundreds of cycles. The ratio of work to traffic is 2 FLOP for every 2
floats fetched, that is, 1 FLOP per float. Recall from Section 9.1.3 that a GPU needs about 100
FLOP per float fetched from memory to keep its arithmetic units busy, namely 94 on the Orin NX and
119 on the Thor.

<div class="key-observation">

<span class="box-label">Key observation</span> The 4 threads computing row 0 of \\(C\\) all need
the same value \\(a_{01}\\). If these threads belong to the same block, one of them can copy
\\(a_{01}\\) into shared memory, and the other 3 can take it from there.

</div>

In Chapter 17, we turn this observation into the tiled kernel. In the rest of this chapter, we
implement the direct kernel and measure how much its repeated fetches cost.

<div class="exercises">

Exercises

16.10 Which threads need \\(b_{21}\\)? How many times does the direct kernel fetch it?<br>
16.11 For \\(M = N = K = 2{,}048\\), how many times does the direct kernel fetch each entry of
\\(A\\), and how many bytes does it fetch in all?<br>
16.12 For \\(M = 3\\), \\(K = 3\\), \\(N = 3\\) and blocks of 2 × 2 threads, how many threads
does the grid hold, and how many of them return at line 4?

</div>

## 16.5 Implementation

We now turn the direct kernel into code. Both languages express the same procedure, and they
differ mainly in what the compiler checks. In each listing below, the Rust version is shown in
the first tab and the C++ version in the second.

### 16.5.1 The kernel

In cuda-oxide, a kernel is an ordinary Rust function marked `#[kernel]`, placed inside a module
marked `#[cuda_module]`. The compiler turns the module into GPU code, and for each kernel, it
generates host-side methods that prepare and launch it. In CUDA C++, a kernel is a function
marked `__global__`. The following code shows the kernel `matmul` in both languages.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch16_matmul_naive/src/main.rs:24:61}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu:18:33}}
```

</div>
</div>

Both kernels follow <span style="font-variant: small-caps">Direct-Kernel</span> almost line for
line. In Rust, `index_2d_row()` and `index_2d_col()` compute lines 1 and 2 of the pseudocode, and in C++,
the 2 expressions with `blockIdx` compute the same lines. The
test of `row` and `col` corresponds to lines 3 to 5, and the loop over `inner` corresponds to lines 7
to 9. Every index follows Definition 16.2. The expression `a[row * k + inner]` is
\\(a_{\mathsf{row}, \mathsf{inner}}\\) in a matrix of row width \\(k\\), and `b[inner * n + col]` is
\\(b_{\mathsf{inner}, \mathsf{col}}\\) in a matrix of row width \\(n\\). For example, for thread \\((1, 0)\\) of
block \\((0, 1)\\) with \\(\mathsf{inner} = 0\\), these are `a[1 * 4 + 0]`, which is `a[4]` and holds 5, and
`b[0 * 4 + 2]`, which is `b[2]` and holds 2.

The Rust kernel receives the sizes as `u32`, which is the type of a launch argument, and converts
them to `usize` once, so that all index arithmetic uses 1 type. Its loop is a `while` loop marked
`#[unroll(4)]`. Recall from Section 9.7 that cuda-oxide unrolls a `while` loop marked `#[unroll]`,
but does not unroll a `for` loop over a range.

### 16.5.2 The output slice

The two kernels differ in how they write \\(C\\). The C++ kernel receives a raw pointer `T* c` and
stores to `c[row * n + col]`. Nothing in the language prevents 2 threads from storing to the same
address, or a thread from storing past the end of the matrix. In Rust, a kernel that took
`&mut [T]` would allow every thread to write anywhere in \\(C\\), and nothing would prevent 2
threads from writing the same element at the same time.

<div class="definition">

<span class="box-label">Definition 16.8</span> **Data race.** A **data race** occurs when 2
threads access the same memory location, at least one of them writes it, and nothing orders the 2
accesses. The value that results depends on the timing of the threads.

*Example.* If thread \\((0, 0)\\) and thread \\((0, 1)\\) both stored their sums to \\(c_{00}\\),
the entry would hold 11 or 13, depending on which store came last.

</div>

For this reason, cuda-oxide gives the output the type `DisjointSlice<T, IndexSpace>`. A thread
can write an element of such a slice only through a **witness**. A witness is a value of type
`ThreadIndex` that names the element the thread owns; it is computed from the thread's own
coordinates, and only the thread itself can obtain it. The second type parameter, the **index
space**, states how coordinates are turned into an element. The kernel `matmul` uses the index
space `Runtime2DIndex`, which works in 3 steps. First, the host binds the row width \\(n\\) to the
buffer with `RowWidth::new(&mut c, n)`. Second, in the kernel, `thread::index_2d_runtime(&c)`
returns a witness that holds the thread's row and column, or `None` if the column is not less
than the bound width. Third, `c.get_mut(idx)` returns a mutable reference to element
\\(\mathsf{row} \cdot n + \mathsf{col}\\), or `None` if that element lies past the end of `c`.
For example, with \\(n = 4\\), thread \\((1, 0)\\) of block \\((0, 1)\\) receives a witness for
row 1 and column 2, and `get_mut` gives it element \\(1 \cdot 4 + 2 = 6\\) of `c`, which is
\\(c_{12}\\).

The argument of Section 16.4.2 that no 2 threads write the same entry is now carried by the
types. Any 2 threads have different coordinates. For a fixed width \\(n\\) and columns less than
\\(n\\), different coordinates (row, col) give different elements
\\(\mathsf{row} \cdot n + \mathsf{col}\\), so 2 threads never hold the same element. As a result,
the kernel needs no `unsafe` block.

A second index space, `Index2D<W>`, makes the row width part of the type. The following code shows
the kernel `matmul_fixed`, which uses it to multiply matrices of `FIXED` × `FIXED` = 1,024 × 1,024
floats, together with the C++ template `matmulFixed<W>`, which multiplies \\(W \times W\\)
matrices.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch16_matmul_naive/src/main.rs:63:82}}
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu:35:50}}
```

</div>
</div>

In the Rust kernel, `thread::index_2d::<FIXED>()` computes
\\(\mathsf{row} \cdot 1{,}024 + \mathsf{col}\\) and returns it as a witness of type
`ThreadIndex<Index2D<1024>>`. The method `get_mut` of a `DisjointSlice<f32, Index2D<1024>>`
accepts only a witness of this type. A witness made for another width has another type, and
passing it to this slice is a compile error. The C++ template `matmulFixed<W>` also has the width
as a compile-time constant, but it does not have the check, because the index expression is
written by hand, and a wrong width in it still compiles. In both languages, the price of a
compile-time width is 1 compiled kernel per matrix size.

### 16.5.3 The launch contract

The attribute `#[launch_contract]` states the conditions under which a launch of the kernel is
correct, and cuda-oxide checks them on the host before the kernel starts. The contract of `matmul`
has three parts. The setting `domain = 2` declares a 2-dimensional launch with \\(z = 1\\), which the
2-dimensional index space requires. The setting `coordinates = u32` requires every coordinate of
the launch to fit in 32 bits. The attribute `#[launch_bounds(256)]`, which accompanies the
contract, limits a block to 256 threads, so that the same kernel serves both the blocks of 2 × 2
threads of our example and the blocks of 16 × 16 threads of the measurements. Finally, the
`requires` clause adds 3 relations between the sizes and the lengths of the buffers. The relation
`a.len() >= m * k`, for example, states that the buffer `a` holds at least the \\(MK\\) values of
an \\(M \times K\\) matrix. The following code shows the contract in front of the Rust kernel, and
the C++ function `violated`, which tests the same 3 relations.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
#[kernel]
#[launch_bounds(256)]
#[launch_contract(
    domain = 2,
    coordinates = u32,
    requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
)]
pub fn matmul<T>(
    m: u32,
    n: u32,
    k: u32,
    a: &[T],
    b: &[T],
    mut c: DisjointSlice<T, Runtime2DIndex>,
) where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu:232:246}}
```

</div>
</div>

In C++, we write the 3 relations ourselves, in the host function `violated`, and call it before a
launch. In this way, the C++ program rejects the same launches as the launch contract does in Rust.

In Rust, the host prepares a launch once with `module.prepare_matmul::<T>(config)`, which checks
the grid and block shapes against the contract and against the limits of the GPU. It then
launches the kernel with `module.matmul(stream, &prepared, m, n, k, &a, &b, width)`, which
evaluates the 3 relations with the actual lengths and sizes. To show the check at work, the
program deliberately launches `matmul` once with buffers of 256 × 256 floats and \\(k = 512\\).
The buffer `a` then holds \\(256 \cdot 256 = 65{,}536\\) values, while an \\(m \times k\\) matrix
needs \\(256 \cdot 512 = 131{,}072\\). The relation \\(a.\text{len}() \ge mk\\) therefore becomes
\\(65{,}536 \ge 131{,}072\\), which is false, and no kernel runs. The program prints the
following line.

```console
k = 2 n refused before launch: matmul: size requirement `a.len() >= m * k` violated: left-hand side is 65536, right-hand side is 131072
```

The C++ launch `matmul<T><<<grid, block>>>(m, n, k, a, b, c)` has no such check. With the same
mistake, each thread would read row `row` of a matrix twice as wide as `a`, and the threads of the
lower half of the grid would read beyond the end of `a`.

The following code shows how the host computes the launch configuration and launches the kernel
for an \\(n \times n\\) product in each language.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch16_matmul_naive/src/main.rs:139:143}}

fn run_generic<T>(
    module: &Module,
    stream: &Arc<CudaStream>,
    name: &str,
    n: usize,
) -> Result<bool, Error>
where
    T: DeviceCopy + Copy + Default + Add<Output = T> + Mul<Output = T>,
    T: From<f32> + Into<f64>,
{
    // ...
{{#include ../../../../code/oxide/ch16_matmul_naive/src/main.rs:198:204}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu:114:118}}

{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu:187:196}}
```

</div>
</div>

The function `config` computes the grid of Section 16.3.2, with \\(\lceil n / \mathsf{side} \rceil\\) blocks
along \\(x\\) and \\(\lceil m / \mathsf{side} \rceil\\) blocks along \\(y\\), where \\(\mathsf{side}\\) is the number of threads along each side of a block. The method `div_ceil` performs
the division rounded up. For example, `config(1000, 1000, 16)` gives a grid of 63 × 63 blocks. The
C++ function `gridFor` computes the same grid with the expression \\((n + \mathsf{side} - 1) / \mathsf{side}\\). The Rust
closure creates a new `RowWidth` for every launch, because a `RowWidth` holds the mutable borrow
of `c` for the duration of 1 launch. The function `time_ms` launches the kernel once without
timing it, and then it averages `runs` launches between 2 synchronizations of the stream.

### 16.5.4 One kernel for f32 and f64

Both kernels are written once for any element type. In C++, `template <typename T>` accepts any
type, and the compiler reports an error only if the body of the kernel uses an operation that the
type lacks. In Rust, the kernel states in a trait bound which operations it needs from \\(T\\).
The following code shows the signature of `matmul` in each language, with the bound in Rust and
the template header in C++.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
pub fn matmul<T>(
    m: u32,
    n: u32,
    k: u32,
    a: &[T],
    b: &[T],
    mut c: DisjointSlice<T, Runtime2DIndex>,
) where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu:19:21}}
    // ...
}
```

</div>
</div>

The bound lists exactly the operations that the kernel performs, namely copying values
(`Copy`), creating the initial 0 with `T::default()` (`Default`), adding (`Add`), and multiplying
(`Mul`). A type that lacks one of them is refused at the call, and the error message names the
missing trait. cuda-oxide compiles 1 GPU function for each type with which the host prepares the
kernel, here `prepare_matmul::<f32>` and `prepare_matmul::<f64>`. The 2 versions perform the same
number of operations. However, an `f64` occupies 8 bytes instead of 4, and a GPU has separate
units for 64-bit arithmetic, whose number differs widely between GPU models.

### 16.5.5 Exact verification

Both programs fill \\(A\\) and \\(B\\) with integers in \\([-5, 5]\\) from the same linear
congruential generator. Every product of 2 such integers is then at most 25 in magnitude, and
every partial sum is at most \\(25K\\) in magnitude, which is \\(102{,}400\\) for
\\(K = 4{,}096\\). Single-precision floating point represents every integer up to
\\(2^{24} = 16{,}777{,}216\\) exactly. Therefore, each multiply-add is exact, the GPU and the CPU
must produce identical results in whatever order they add the products, and any difference
signals a bug. The programs compare 64 sampled entries of \\(C\\) with a product computed on the
CPU, and they require all 64 to be equal. They also run the 4 × 4 example with blocks of 2 × 2
threads, print \\(C\\), and compare it with the product computed by hand in Section 16.1.

### 16.5.6 Removing the guard in Rust

In Section 16.4.2, we removed lines 3 to 5 from the C++ kernel, and the kernel faulted. In the
Rust kernel, the guard is the condition `row < m && col < n` in front of the 2 `let` patterns. We
removed this condition and ran the kernel on matrices of 1,000 × 1,000, comparing all 1,000,000
entries with the CPU. The program printed the following line.

```console
no guard, n = 1000: 0 of 1000000 entries wrong
```

The result is correct. A thread past the last column receives `None` from `index_2d_runtime`, a
thread past the last row receives `None` from `get_mut`, and in both cases, the thread skips the
loop and the write. The test of the witness therefore acts as a second guard. The explicit guard
is nevertheless still needed, because the width bound to `c` and the length of `c` are chosen by
the host, and nothing forces them to equal \\(n\\) and \\(mn\\). If the host bound a larger
buffer, a thread in row \\(m\\) would receive an element of `c` and would then read row \\(m\\) of
`a`, which does not exist. That read is bounds-checked, so the kernel would stop with an error
rather than read another buffer. The guard, however, ties the reads to the sizes of `a` and `b`,
and it prevents the kernel from failing at all.

<div class="exercises">

Exercises

16.13 Change the Rust launch to blocks of 32 × 8 threads. Which lines must change, and does the
contract accept the new shape?<br>
16.14 Write `matmul::<i32>` launches for the 4 × 4 example. Is the result exact? Which trait
bound does `i32` satisfy that a type such as `bool` does not?<br>
16.15 In the C++ kernel, exchange `blockIdx.x` and `blockIdx.y` in the computation of `row` and
`col` but not in the grid. For which matrix shapes is the result still correct?

</div>

## 16.6 Measurements

### 16.6.1 Platform and method

The measurements come from an integrated GPU, a Jetson Orin NX 16 GB, with 8 SMs, a memory
bandwidth of 102 GB/s, and an FP32 peak of 2,400 GFLOP/s at 1.173 GHz. Each program first runs
the GPU for 0.5 s so that its clocks reach their sustained values. It then launches each kernel
once without timing it, and finally it averages up to 50 launches, fewer for the large sizes.
Each program ran 3 times. The tables show the quietest run, and repeated runs differed by up to
10%.

### 16.6.2 Results

The Rust program prints the following output.

```console
$ cargo oxide run
C = A B on the 4 x 4 example, a grid of 2 x 2 blocks of 2 x 2 threads
    11   13    8    8
    27   29   24   24
    43   45   40   40
    59   61   56   56
equal to the product computed by hand: yes

C = A B, n x n, 1 thread per entry of C, blocks of 16 x 16 threads
      n  type              ms   GFLOP/s  fetched GB/s   exact
    512  f32            1.469     182.7      730.8     64 of 64
   1024  f32           15.659     137.1      548.6     64 of 64
   2048  f32          124.013     138.5      554.1     64 of 64
   4096  f32          997.150     137.8      551.3     64 of 64
   1024  f32 fixed     11.071     194.0      775.9     64 of 64
    512  f64            7.641      35.1      281.0     64 of 64
   1024  f64           61.076      35.2      281.3     64 of 64
   2048  f64          488.370      35.2      281.4     64 of 64
k = 2 n refused before launch: matmul: size requirement `a.len() >= m * k` violated: left-hand side is 65536, right-hand side is 131072
PASSED
```

The first lines reproduce the product of Section 16.1, with \\(c_{00} = 11\\) and
\\(c_{01} = 13\\). The C++ program prints the same example and the same table. The following
table shows the times of the two programs side by side.

| n | type | C++ time | C++ rate | Rust time | Rust rate | requested fetches (C++) |
|---|---|---|---|---|---|---|
| 512 | f32 | 1.465 ms | 183.2 GFLOP/s | 1.469 ms | 182.7 GFLOP/s | 732.9 GB/s |
| 1,024 | f32 | 11.803 ms | 181.9 GFLOP/s | 15.659 ms | 137.1 GFLOP/s | 727.8 GB/s |
| 2,048 | f32 | 93.828 ms | 183.1 GFLOP/s | 124.013 ms | 138.5 GFLOP/s | 732.4 GB/s |
| 4,096 | f32 | 743.091 ms | 185.0 GFLOP/s | 997.150 ms | 137.8 GFLOP/s | 739.8 GB/s |
| 1,024 | f32, fixed width | 11.493 ms | 186.9 GFLOP/s | 11.071 ms | 194.0 GFLOP/s | 747.4 GB/s |
| 512 | f64 | 7.588 ms | 35.4 GFLOP/s | 7.641 ms | 35.1 GFLOP/s | 283.0 GB/s |
| 1,024 | f64 | 61.045 ms | 35.2 GFLOP/s | 61.076 ms | 35.2 GFLOP/s | 281.4 GB/s |
| 2,048 | f64 | 487.566 ms | 35.2 GFLOP/s | 488.370 ms | 35.2 GFLOP/s | 281.9 GB/s |

The rate counts \\(2n^3\\) FLOP for each product. For example, for \\(n = 4{,}096\\), the kernel
performs \\(2 \cdot 4{,}096^3 = 1.37 \times 10^{11}\\) FLOP in the time shown. The column
*fetched GB/s* divides the bytes that the kernel requests, \\(2n^3\\) values of 4 or 8 bytes each,
by the time. All 64 sampled entries were exact in every run, in single and in double precision.

<figure>
<img src="../../figures/ch16-measured.svg" alt="2 panels of horizontal bars for the direct kernel on the measured GPU. Top, the rate in GFLOP/s at n = 4,096: C++ f32 185.0, Rust f32 137.8, Rust f32 with the width fixed at compile time 194.0 (n = 1,024), and f64 35.2 at n = 2,048, against a dashed line for the FP32 peak of 2,400 and a short mark for the FP64 peak of 75. Bottom, the requested fetch rate in GB/s: C++ 739.8 and Rust 551.3, against a dashed line at 102 GB/s, the bandwidth of global memory. A note: at least 1 minus 102 / 740 = 86% of the C++ kernel's fetches were answered by the caches.">
<figcaption><b>Figure 16.13</b> The direct kernel on the measured GPU: its rate against the FP32 peak, and its requested fetch rate against the bandwidth of global memory.</figcaption>
</figure>

### 16.6.3 Why the rate is far below the peak

The C++ kernel ran at 185.0 GFLOP/s for \\(n = 4{,}096\\), which is 7.7% of the FP32 peak of
2,400 GFLOP/s. In other words, the arithmetic units were idle more than 92% of the time. The
reason is the one we found in Section 16.4.3, where we saw that the kernel fetches 1 float for
every FLOP. If every fetch went to global memory, the 102 GB/s of the measured GPU would deliver
\\(102 / 4 = 25.5 \times 10^9\\) floats per second, and the kernel could therefore perform at most
25.5 GFLOP/s.

The measured rate is 7.3 times this limit, and the fetch column of the table explains how. The
kernel requested \\(2 \cdot 4{,}096^3 \cdot 4\\) bytes, or 550 GB, in 0.743 s, which is
739.8 GB/s, 7.3 times what global memory can deliver. Since global memory delivers at most
102 GB/s, at least \\(1 - 102/740 = 86\\%\\) of the requested fetches must have been answered by
the L1 and L2 caches. The repeated fetches of Figure 16.12 happen close together in time. The 16
threads of a warp that share a row of \\(C\\) request the same value of \\(A\\) in the same
instruction, and the hardware delivers that value once to all of them. In addition, neighboring
blocks request the same rows and columns shortly after one another. The caches therefore recover
much of the reuse, but only because the requests happen to be close in time, and the kernel spends
the remaining time waiting for memory. The kernel is thus limited by how fast the memory system
delivers operands, not by the arithmetic.

The `f64` kernel ran at 35.2 GFLOP/s. The measured GPU reports a ratio of 32 between its FP32 and
FP64 rates, so its FP64 peak is \\(2{,}400 / 32 = 75\\) GFLOP/s, and the `f64` kernel reached 47%
of that peak, compared with 7.7% for `f32`. Its operands are twice as large, and it requested
282 GB/s, but on this GPU, the small number of FP64 units, not the memory, sets its speed. On a
data-center GPU with an FP32:FP64 ratio of 2, the `f64` kernel would be limited by memory, like
the `f32` kernel.

The two languages agree at 512 × 512 and in double precision. From 1,024 × 1,024 upwards, however,
the generic Rust kernel, whose sizes are known only at run time, ran at 138 GFLOP/s, compared with
182 to 185 GFLOP/s in C++. The Rust kernel with the width in its type ran at 194.0 GFLOP/s, the
fastest of all. Both Rust kernels read the same values in the same order, and they differ only in
what the compiler knows. When \\(n\\) and \\(k\\) are constants, the loop has a fixed trip count,
which is a multiple of 4, and the addresses of consecutive reads of `b` lie a constant 4,096 bytes
apart. When the sizes are known only at run time, the compiler must handle a possible remainder of
the unrolled loop and recompute every address. A width in the type costs 1 compiled kernel per
matrix size, and in Chapter 17, we recover more speed for every size by tiling.

The measured rates also show which factors do not matter. The block shape of 16 × 16 threads
keeps the reads of a warp contiguous, and the rates are almost the same at every size from 512 to
4,096. Once the matrices exceed the caches, every size waits for memory in the same proportion.
The remedy is therefore to fetch less often from global memory, which is what the key observation
of Section 16.4.3 proposes.

## 16.7 The complete programs

The following code shows the complete Rust program and the complete C++ program.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch16_matmul_naive/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch16_matmul_naive/matmul_naive.cu}}
```

</div>
</div>

To build and run the Rust program, we run `cargo oxide run` in its directory. To build and run
the C++ program, we run
`nvcc -O3 -arch=native matmul_naive.cu -o matmul_naive && ./matmul_naive`.

## Common mistakes

- **Taking the row from \\(x\\).** The result is still correct, but the 32 threads of a warp then
  walk down a column of \\(B\\), whose entries lie \\(N\\) floats apart, and the kernel runs
  several times slower (Section 9.3).
- **Rounding the grid down.** A grid of `n / 16` blocks leaves the last \\(n \bmod 16\\) rows and
  columns of \\(C\\) unwritten. For example, with \\(n = 1{,}000\\), it leaves 8 rows and 8
  columns unwritten. The size of the grid must be rounded up, with `div_ceil` or with
  \\((n + 15) / 16\\).
- **Dropping the guard in C++.** Threads of the edge blocks then read and write outside the
  matrices. With \\(n = 1{,}000\\), the kernel stopped with an illegal memory access.
- **Binding the wrong row width.** With `RowWidth::new(&mut c, k)` instead of
  `RowWidth::new(&mut c, n)`, every thread writes the element of a grid of width \\(k\\). The
  writes stay disjoint, so the program is safe, but the result is wrong unless \\(k = n\\).
- **Mixing up the 3 sizes.** \\(A\\) has row width \\(K\\), and \\(B\\) and \\(C\\) have row
  width \\(N\\). Square test matrices hide a mistake here, so a rectangular product should be
  tested as well.
- **Comparing random floats exactly.** With random real inputs, the order of the additions
  changes the rounding. Small integers, as used here, give exact results; otherwise, the results
  must be compared with a tolerance.

## Summary

<div class="summary">

- The product \\(C = AB\\) of an \\(M \times K\\) and a \\(K \times N\\) matrix has the entries
  \\(c_{ij} = \sum_s a_{is} b_{sj}\\), and it costs \\(MNK\\) multiply-adds, or \\(2MNK\\) FLOP.
  Our 4 × 4 example needs 64 multiply-adds, and \\(c_{00} = 11\\) and \\(c_{01} = 13\\).
- In row-major order, entry \\((r, c)\\) of a matrix of row width \\(W\\) is stored at location
  \\(rW + c\\).
- A kernel runs as many threads. The threads are grouped in blocks of up to 1,024 threads, each of
  which runs on 1 SM and shares shared memory, and the blocks form a grid. Along each axis, a
  thread finds its entry from \\(\mathsf{blockIdx} \cdot \mathsf{blockDim} + \mathsf{threadIdx}\\).
- Registers are fast and private to a thread, shared memory is fast and belongs to a block, and
  global memory is large and shared by all threads, but slow.
- The direct kernel gives each thread 1 entry of \\(C\\). It is correct, but it fetches every value
  of \\(A\\) \\(N\\) times and every value of \\(B\\) \\(M\\) times, which makes \\(2MNK\\)
  fetches, 1 per FLOP.
- In Rust, `DisjointSlice` with `Runtime2DIndex` or `Index2D<W>` lets each thread write only its
  own entry without `unsafe`, and the launch contract refuses buffers that are too small for the
  sizes.
- On the measured GPU, the kernel ran at about 7% of the FP32 peak. The caches answered most of
  its repeated fetches, and the rest of the time went to waiting for memory.

</div>

In Chapter 17, we let the threads of a block share the values they fetch, through shared memory,
and we derive the tiled kernel from the key observation of Section 16.4.3.

## Exercises

1. Compute \\(c_{32}\\) of the example by hand, and identify the thread of Figure 16.5 that
   computes it.
2. For \\(n = 1{,}000\\) and blocks of 16 × 16 threads, how many threads does the grid contain,
   and what fraction of them fail the guard?
3. Extend the Rust host program to rectangular matrices and check the product for
   \\(M = 300\\), \\(K = 500\\), \\(N = 700\\).
4. Explain why the `f64` results of this chapter are exact. Would they stay exact for integers in
   \\([-1{,}000, 1{,}000]\\) and \\(K = 4{,}096\\)? And the `f32` results?
5. Using the FP32:FP64 ratio of your GPU, predict the rate of `matmul::<f64>` if it were limited by
   arithmetic, and compare with your measurement.
6. Remove `#[unroll(4)]` from `matmul` and measure again. What changes?
7. Write a kernel that computes \\(C = A^{\mathsf T} B\\) without forming \\(A^{\mathsf T}\\). Which
   index expression changes, and how does the access pattern of a warp change?
8. Suppose a cache holds every value that a block fetches. How many fetches does block \\((0, 0)\\)
   of Figure 16.5 then make from global memory, and how does that compare with the 32 fetches it
   requests?

## Sources and further reading

- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026: the worked
  example, the direct kernel and its fetch count.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed.,
  MIT Press, 2022: the pseudocode conventions.
- NVIDIA, *CUDA C++ Programming Guide*, "Thread Hierarchy" and "Memory Hierarchy":
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- cuda-oxide, rev b7fe543, `cuda-device` crate: `DisjointSlice`, `Index2D`, `Runtime2DIndex` and
  the thread index functions; `cuda-macros`: `#[launch_contract]`:
  <https://github.com/NVlabs/cuda-oxide>
- Chapter 9 of this book, for the same kernel in CUDA C++ and the effect of the block shape.
