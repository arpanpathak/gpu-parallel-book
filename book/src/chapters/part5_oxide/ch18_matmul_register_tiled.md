
<img class="plate" src="../../art/ch18.svg" alt="Patch, a crab, stands on the sand beside a slate board. To the left of the board stand 2 cells of A, 1 and 5, and above it 2 cells of B, 1 and 0. On the board are their 4 products, 1 0 / 5 0. Further right, a larger board holds the 4 by 4 matrix C, 11 13 8 8 / 27 29 24 24 / 43 45 40 40 / 59 61 56 56, split into 4 coloured patches of 2 by 2, 1 per thread. A tag says read 4 values, make 4 products.">

# Chapter 18: Register Tiling, Several Entries per Thread

The tiled kernel of Chapter 17 fetches each value from global memory 16 times less often than the
direct kernel, yet on the GPU measured there it ran only about 1.5 times as fast. In Section 17.8.2,
we located the new limit in shared memory: every multiply-add of the tiled kernel reads 2 values
from shared memory, and shared memory cannot deliver them as fast as the arithmetic units consume
them. The kernel ends near 1/8 of the arithmetic peak on every GPU, whatever the tile width.

In this chapter, we remove that limit with the same idea that removed the first one, applied 1
level closer to the arithmetic units. We start from Exercise 17.12 on the 4 × 4 matrices of
Chapter 16: each thread of the tiled kernel performs 2 FLOP per 2 shared-memory reads. We let 1
thread compute 2 entries of \\(C\\), then a 2 × 2 block of entries, and find that the inner loop
becomes a sum of outer products in which every value read from shared memory is used several
times from a register. We count the reads per multiply-add in general, derive the kernel in
pseudocode, prove it correct, implement it in Rust and CUDA C++, and measure it against the tiled
kernel of Chapter 17 and the C++ kernels of Chapter 9.

<div class="covers">

This chapter covers

- the shared-memory reads of the tiled kernel, counted on the example
- 1 thread computing a 2 × 1 and then a 2 × 2 block of \\(C\\)
- the inner loop as a sum of outer products
- reads per multiply-add, \\((T_M + T_N) / (T_M T_N)\\), and registers as the third level of
  storage
- the register-tiled kernel in pseudocode, and its loop invariant
- the kernel in Rust (const generics, `#[unroll]`, `RuntimeRowMajorTiles`) and in CUDA C++
  (templates, `#pragma unroll`)
- measured speeds for thread tiles from 1 × 1 to 8 × 8, and what limits each

</div>

The code of the chapter is in
[`code/oxide/ch18_matmul_register_tiled/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch18_matmul_register_tiled)
for Rust and in
[`code/cuda/ch18_matmul_register_tiled/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch18_matmul_register_tiled)
for C++. In Chapter 9, we built a kernel with a 4 × 4 patch per thread in C++ and in Rust; here we
derive that kernel from the tiled kernel step by step and generalize its shape.

## 18.1 The reads of the tiled kernel

### 18.1.1 1 thread in 1 phase

We return to block \\((0, 0)\\) of Chapter 17 with \\(T = 2\\), in phase 0. The tiles hold
\\(\textit{tileA} = \begin{pmatrix} 1 & 2 \\\\ 5 & 6 \end{pmatrix}\\) and
\\(\textit{tileB} = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \end{pmatrix}\\). Thread \\((0, 0)\\) runs
line 10 of <span style="font-variant: small-caps">Tiled-Kernel</span> twice. For \\(u = 0\\) it
reads \\(\textit{tileA}[0][0] = 1\\) and \\(\textit{tileB}[0][0] = 1\\) and adds \\(1 \cdot 1\\);
for \\(u = 1\\) it reads 2 and 0 and adds \\(2 \cdot 0\\) (Figure 18.1). That is 4 reads from shared
memory for 2 multiply-adds, 4 FLOP.

<figure>
<img src="../../figures/ch18-tiled-reads.svg" alt="Left: tileA = 1 2 / 5 6 and tileB = 1 0 / 0 1 in shared memory during phase 0; row 0 of tileA and column 0 of tileB are highlighted. Right: thread (0, 0) of the tiled kernel of Chapter 17 at u = 0 reads 1 and 1 and adds 1 times 1; at u = 1 it reads 2 and 0 and adds 2 times 0: 2 reads from shared memory for each multiply-add. Below: 1 thread makes 4 reads for 2 multiply-adds, 4 FLOP, in a phase; the block of 4 threads makes 16 reads for 8 multiply-adds. Thread (0, 1) reads the same row 0 of tileA again.">
<figcaption><b>Figure 18.1</b> In the tiled kernel of Chapter 17, thread (0, 0) reads 2 values from shared memory for each multiply-add.</figcaption>
</figure>

The block of 4 threads makes 16 reads for 8 multiply-adds in the phase. Some of these reads are
repeated. Thread \\((0, 1)\\) reads row 0 of *tileA* again, the same 2 values that thread
\\((0, 0)\\) has read a moment earlier, and thread \\((1, 0)\\) reads column 0 of *tileB* again. In general,
every value of *tileA* is read by the \\(T\\) threads of its row of the block and every value of
*tileB* by the \\(T\\) threads of its column: the same waste that we found in global memory in
Section 16.4.3, 1 level closer to the arithmetic.

### 18.1.2 What the reads cost

In Section 17.8.2, we turned this count into a limit. An SM's shared memory serves 1 read instruction
of a warp per cycle, 32 values, and the SM's 128 FP32 units could perform 128 multiply-adds in that
cycle. At 2 reads per multiply-add, the SM completes at most 16 multiply-adds per cycle, 1/8 of its
peak: 300 GFLOP/s on the GPU of Chapter 17, against which the tiled kernel measured 272 GFLOP/s. A larger
tile does not help, because the ratio of 2 reads per multiply-add does not depend on \\(T\\). Only a
kernel that performs more multiply-adds for each value it reads can go faster.

<div class="exercises">

Exercises

18.1 In phase 1 of block \\((0, 0)\\), which values does thread \\((1, 1)\\) read from shared memory,
and which of them does another thread of the block also read?<br>
18.2 With \\(T = 16\\), how many reads from shared memory does 1 block make in 1 phase, and how many
distinct values do the 2 tiles hold?

</div>

## 18.2 Several entries per thread

### 18.2.1 A 2 × 1 block of C

Let 1 thread compute 2 entries of \\(C\\) instead of 1: \\(c_{00}\\) and \\(c_{10}\\), the top 2
entries of column 0. Both use column 0 of \\(B\\). For \\(u = 0\\) the thread needs \\(a_{00} = 1\\)
for \\(c_{00}\\), \\(a_{10} = 5\\) for \\(c_{10}\\), and \\(b_{00} = 1\\) for both. It reads the 3
values once, keeps them in registers, and performs 2 multiply-adds: \\(1 \cdot 1\\) into the sum of
\\(c_{00}\\) and \\(5 \cdot 1\\) into the sum of \\(c_{10}\\) (Figure 18.2).

<figure>
<img src="../../figures/ch18-two-by-one.svg" alt="A, B and C of the 4 by 4 example. a00 = 1 and a10 = 5 of A, b00 = 1 of B, and c00 and c10 of C are highlighted. In registers at u = 0: a_reg = 1, 5 and b_reg = 1, 3 reads from shared memory. The thread adds 1 times 1 = 1 to c00 and 5 times 1 = 5 to c10: 3 reads for 2 multiply-adds, because b00 is read once and used twice. 2 threads of Chapter 17 needed 4 reads for the same 2 multiply-adds: 1.5 reads per multiply-add instead of 2.">
<figcaption><b>Figure 18.2</b> A thread that computes c<sub>00</sub> and c<sub>10</sub> reads b<sub>00</sub> once and uses it twice: 3 reads for 2 multiply-adds.</figcaption>
</figure>

In the tiled kernel, the same 2 multiply-adds took 2 threads and 4 reads, because both threads read
\\(b_{00}\\). Now they take 1 thread and 3 reads: 1.5 reads per multiply-add instead of 2. The saving
comes from the value of \\(B\\) that the 2 entries share; it is read once and used twice from a
register.

### 18.2.2 A 2 × 2 block of C

The next step is a square. Let 1 thread compute \\(c_{00}\\), \\(c_{01}\\), \\(c_{10}\\) and
\\(c_{11}\\), the 4 entries that the whole block \\((0, 0)\\) computed in Chapter 17. For each
\\(u\\), the thread needs 2 values of column \\(u\\) of \\(A\\), \\(a_{0u}\\) and \\(a_{1u}\\), and 2
values of row \\(u\\) of \\(B\\), \\(b_{u0}\\) and \\(b_{u1}\\). Each value of \\(A\\) meets each
value of \\(B\\) in exactly 1 of the 4 entries. For \\(u = 0\\) the thread reads \\((1, 5)\\) from
\\(A\\) and \\((1, 0)\\) from \\(B\\) and forms all 4 products (Figure 18.3):

\\[ \begin{pmatrix} 1 \\\\ 5 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix}
= \begin{pmatrix} 1 \cdot 1 & 1 \cdot 0 \\\\ 5 \cdot 1 & 5 \cdot 0 \end{pmatrix}
= \begin{pmatrix} 1 & 0 \\\\ 5 & 0 \end{pmatrix}. \\]

<figure>
<img src="../../figures/ch18-two-by-two.svg" alt="For u = 0, the values 1 and 5 from column 0 of A run down the left, and the values 1 and 0 from row 0 of B run across the top. The 4 cells hold the products 1 times 1 = 1, 1 times 0 = 0, 5 times 1 = 5 and 5 times 0 = 0, which are added to c00, c01, c10 and c11. Cards: 4 reads, 2 values of A and 2 of B; 4 multiply-adds, every value read is used twice; 1 read per multiply-add.">
<figcaption><b>Figure 18.3</b> For a 2 × 2 block of C, 1 thread reads 4 values and forms their 4 products, the outer product.</figcaption>
</figure>

That is 4 reads for 4 multiply-adds, 1 read per multiply-add, half the cost in the tiled kernel.
Every value read is used twice.

<div class="definition">

<span class="box-label">Definition 18.1</span> **Outer product.** The **outer product** of a
column vector \\(x\\) of length \\(p\\) and a row vector \\(y\\) of length \\(q\\) is the
\\(p \times q\\) matrix \\(xy\\) with entries \\(x_i y_j\\). It contains every product of an entry
of \\(x\\) with an entry of \\(y\\): \\(pq\\) products from \\(p + q\\) values.

*Example.* The outer product of \\((1, 5)\\) and \\((1, 0)\\) is
\\(\begin{pmatrix} 1 & 0 \\\\ 5 & 0 \end{pmatrix}\\), the 4 products of Figure 18.3.

</div>

### 18.2.3 The outer-product form

Repeating the step for \\(u = 0, 1, 2, 3\\) adds 4 outer products, 1 for each column \\(u\\) of
\\(A\\) and row \\(u\\) of \\(B\\), and their sum is the 2 × 2 block of \\(C\\) (Figure 18.4):

\\[ \begin{pmatrix} 1 & 0 \\\\ 5 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 2 \\\\ 0 & 6 \end{pmatrix}
+ \begin{pmatrix} 6 & 3 \\\\ 14 & 7 \end{pmatrix} + \begin{pmatrix} 4 & 8 \\\\ 8 & 16 \end{pmatrix}
= \begin{pmatrix} 11 & 13 \\\\ 27 & 29 \end{pmatrix}. \\]

<figure>
<img src="../../figures/ch18-outer-sum.svg" alt="4 panels, 1 for each u from 0 to 3. In each, the 2 values of column u of A, rows 0 and 1, multiply the 2 values of row u of B, columns 0 and 1. u = 0: (1, 5) times (1, 0) gives 1 0 / 5 0. u = 1: (2, 6) times (0, 1) gives 0 2 / 0 6. u = 2: (3, 7) times (2, 1) gives 6 3 / 14 7. u = 3: (4, 8) times (1, 2) gives 4 8 / 8 16. Their sum is 11 13 / 27 29, the 2 by 2 block of C in rows 0 and 1 and columns 0 and 1: for example c00 = 1 + 0 + 6 + 4 = 11 and c11 = 0 + 6 + 7 + 16 = 29.">
<figcaption><b>Figure 18.4</b> The 2 × 2 block of C in rows 0 and 1 and columns 0 and 1 is the sum of 4 outer products, 1 for each u.</figcaption>
</figure>

The entry \\(c_{00}\\) appears as \\(1 + 0 + 6 + 4 = 11\\), the same 4 products as in Figure 16.1,
added in the same order; only the bookkeeping has changed. The tiled kernel ran the loop over
\\(s\\) inside each entry. The new form runs the loop over \\(s\\) outside, and inside it touches
every entry of the thread's block once. Written for a block of \\(T_M\\) rows and \\(T_N\\) columns
of \\(C\\) that starts at row \\(i_0\\) and column \\(j_0\\),

\\[ c_{i_0 + i,\\; j_0 + j} = \sum_{s \in [0, K)} a_{i_0 + i,\\, s}\\; b_{s,\\, j_0 + j},
\qquad i \in [0, T_M),\ j \in [0, T_N), \\]

the step for 1 value of \\(s\\) reads the \\(T_M\\) values \\(a_{i_0 + i, s}\\) and the \\(T_N\\)
values \\(b_{s, j_0 + j}\\), and adds their outer product to the \\(T_M T_N\\) sums.

### 18.2.4 Counting the reads

Each step reads \\(T_M + T_N\\) values and performs \\(T_M T_N\\) multiply-adds, so the reads per
multiply-add are

\\[ \frac{T_M + T_N}{T_M T_N}. \\]

For a 1 × 1 block, the tiled kernel, this is 2. For 2 × 1 it is \\(3/2 = 1.5\\), for 2 × 2
\\(4/4 = 1\\), for 4 × 4 \\(8/16 = 0.5\\), and for 8 × 8 \\(16/64 = 0.25\\) (Figure 18.5). For a
square block of side \\(T_M = T_N = R\\) the ratio is \\(2/R\\): it falls in proportion to the side,
as the fetches from global memory fell in proportion to \\(T\\) in Chapter 17. The price is
that the thread must keep \\(T_M T_N\\) sums alive for the whole kernel, plus the \\(T_M + T_N\\)
values of the current step.

<figure>
<img src="../../figures/ch18-reads-per-fma.svg" alt="Horizontal bars of shared-memory reads per multiply-add, (TM + TN) divided by TM times TN, for thread tiles 1 by 1: 2; 2 by 1: 1.5; 2 by 2: 1; 4 by 4: 0.5; 8 by 8: 0.25. Beside each bar, the number of sums a thread keeps in registers: 1, 2, 4, 16 and 64. The 1 by 1 row is the tiled kernel of Chapter 17.">
<figcaption><b>Figure 18.5</b> Shared-memory reads per multiply-add, (T<sub>M</sub> + T<sub>N</sub>) / (T<sub>M</sub> · T<sub>N</sub>), and the sums each thread keeps in registers.</figcaption>
</figure>

<div class="key-observation">

<span class="box-label">Key observation</span> A thread that computes a \\(T_M \times T_N\\) block
of \\(C\\) can read \\(T_M\\) values of \\(A\\) and \\(T_N\\) values of \\(B\\) once per step, keep
them in registers, and use each of them \\(T_N\\) or \\(T_M\\) times. Its reads from shared memory
per multiply-add fall from 2 to \\((T_M + T_N) / (T_M T_N)\\).

</div>

<div class="exercises">

Exercises

18.3 Compute the outer products for the 2 × 2 block of \\(C\\) in rows 2 and 3 and columns 0 and 1,
and check that they sum to \\(\begin{pmatrix} 43 & 45 \\\\ 59 & 61 \end{pmatrix}\\).<br>
18.4 For a thread tile of 4 × 2, how many reads and how many multiply-adds does 1 step take? Which
is better for a fixed number of 8 sums, 4 × 2 or 8 × 1?<br>
18.5 Show that for a fixed number \\(P = T_M T_N\\) of sums, the reads per multiply-add are smallest
when \\(T_M = T_N\\).

</div>

## 18.3 Registers as the third level

The sums of Section 18.2 live in registers. In Chapter 16, we described 3 kinds of storage: global
memory, shared memory and registers. The direct kernel took its operands from the first, the tiled
kernel from the second, and the new kernel takes most of them from the third.

<div class="definition">

<span class="box-label">Definition 18.2</span> **Register tile.** A **register tile**, or thread
tile, is a \\(T_M \times T_N\\) block of \\(C\\) that 1 thread computes, with its \\(T_M T_N\\) sums
held in registers for the whole kernel. **Register tiling** is the technique of giving each thread
a register tile and computing it as a sum of outer products.

*Example.* With \\(T_M = T_N = 2\\), thread \\((0, 0)\\) of our example holds the 4 sums of
\\(c_{00}\\), \\(c_{01}\\), \\(c_{10}\\) and \\(c_{11}\\) in 4 registers.

</div>

Registers are the fastest storage of the GPU, but they are a fixed resource of each SM: 65,536
registers of 4 bytes on both measured GPUs, divided among all the threads resident on the SM, with
at most 255 per thread. A kernel that uses more registers per thread fits fewer threads on an SM,
which lowers its occupancy (Definition 17.3). A thread with a 4 × 4 tile needs 16 registers for its
sums and 8 for the values of the current step, plus registers for addresses and loop counters.

Figure 18.6 shows the 3 levels with the counts of a kernel in which each thread computes a 4 × 4
tile, as in Chapter 9. A block of 256 threads, 16 × 16, then computes a 64 × 64 tile of \\(C\\).
In each phase, the block copies 64 × 16 values of \\(A\\) and 16 × 64 values of \\(B\\) from global
memory into shared memory, 2,048 floats, 8 per thread. In each of the 16 steps of a phase, a thread
reads 8 floats from shared memory and performs 16 multiply-adds on registers.

<figure>
<img src="../../figures/ch18-levels.svg" alt="3 cards joined by arrows. Global memory holds A, B and C, is seen by all threads and answers in hundreds of cycles. Shared memory holds tileA of 64 by 16 and tileB of 16 by 64 floats for 1 block during 1 phase. Registers hold a_reg of 4 values, b_reg of 4 values and acc of 4 by 4 = 16 sums for 1 thread. Below, the counts for 4 by 4 per thread with blocks of 64 by 64 and BK = 16: per phase a block copies 64 times 16 plus 16 times 64 = 2,048 floats, 8 per thread; per value of u a thread reads 8 floats from shared memory and does 16 multiply-adds, 0.5 reads per multiply-add.">
<figcaption><b>Figure 18.6</b> The 3 levels of memory: global memory, shared memory and registers, with their counts for 4 × 4 entries per thread.</figcaption>
</figure>

The 2 levels of tiling multiply. Let a block compute a \\(B_M \times B_N\\) tile of \\(C\\) in
phases of \\(B_K\\) values of \\(s\\). Per phase it copies \\(B_M B_K + B_K B_N\\) values and
performs \\(B_M B_N B_K\\) multiply-adds, so over the whole product it fetches

\\[ \frac{M}{B_M} \cdot \frac{N}{B_N} \cdot \frac{K}{B_K} \cdot (B_M B_K + B_K B_N)
= MNK \left( \frac{1}{B_N} + \frac{1}{B_M} \right) \\]

values from global memory, \\(2MNK/64\\) for \\(B_M = B_N = 64\\): 4 times fewer than the tiled
kernel with \\(T = 16\\). Independently, each thread reads \\((T_M + T_N)/(T_M T_N)\\) values from
shared memory per multiply-add. Global memory sees the block tile, shared memory sees the thread
tile, and the arithmetic units see registers.

<div class="exercises">

Exercises

18.6 For \\(B_M = B_N = 128\\), \\(B_K = 8\\) and 8 × 8 per thread, how many threads does a block
have, how many floats does each copy per phase, and how many FLOP per float does the kernel perform
on values fetched from global memory?<br>
18.7 At 110 registers per thread, how many blocks of 256 threads fit in the 65,536 registers of an
SM? What occupancy does that give on an SM that holds 1,536 threads?

</div>

## 18.4 The algorithm

### 18.4.1 Who computes what

A block now has \\((B_M / T_M) \cdot (B_N / T_N)\\) threads, arranged as \\(B_M/T_M\\) rows and
\\(B_N/T_N\\) columns. Thread \\((ty, tx)\\) of block \\((by, bx)\\) owns the \\(T_M \times T_N\\)
block of \\(C\\) whose top-left entry is at row \\(by \cdot B_M + ty \cdot T_M\\) and column
\\(bx \cdot B_N + tx \cdot T_N\\). In our example, we take \\(B_M = B_N = 4\\), \\(B_K = 2\\) and
\\(T_M = T_N = 2\\): a single block of 2 × 2 threads covers all of \\(C\\), and each thread owns a
2 × 2 block of it (Figure 18.7). Thread \\((0, 0)\\) alone now computes what the 4 threads of block
\\((0, 0)\\) computed in Chapter 17.

<figure>
<img src="../../figures/ch18-ownership.svg" alt="Left: C of the example, 11 13 8 8 / 27 29 24 24 / 43 45 40 40 / 59 61 56 56, coloured in 4 blocks of 2 by 2, 1 colour per thread of the single block of 2 by 2 threads: thread (ty, tx) owns rows 2ty and 2ty + 1 and columns 2tx and 2tx + 1. Right: the tiles of phase 0, tileA of 4 by 2 = columns 0 and 1 of A, 1 2 / 5 6 / 9 10 / 13 14, and tileB of 2 by 4 = rows 0 and 1 of B, 1 0 2 1 / 0 1 1 2, coloured by the thread that copies each value: thread id = 2ty + tx copies elements id and id + 4 of each tile in row-major order. A card recalls that in Chapter 17 with T = 2 the 4 threads of block (0, 0) computed c00, c01, c10 and c11; here thread (0, 0) alone computes them.">
<figcaption><b>Figure 18.7</b> In the example, each thread of the single 2 × 2 block owns a 2 × 2 block of C and copies 2 values of each tile.</figcaption>
</figure>

The tiles are no longer square. In phase \\(t\\), the block needs the \\(B_M \times B_K\\) piece of
\\(A\\) in its rows and in columns \\([tB_K, tB_K + B_K)\\), and the \\(B_K \times B_N\\) piece of
\\(B\\) in rows \\([tB_K, tB_K + B_K)\\) and its columns. In the example, both tiles hold 8 values and
the block has 4 threads, so each thread copies 2 values of each. In general, we number the threads
of a block \\(\textit{id} = ty \cdot (B_N/T_N) + tx\\), number the slots of each tile in row-major
order, and let thread \\(\textit{id}\\) copy slots \\(\textit{id}\\),
\\(\textit{id} + P\\), \\(\textit{id} + 2P\\), and so on, where \\(P\\) is the number of threads.
Consecutive threads then copy consecutive values of a row, which lie next to each other in global
memory. For this to cover each tile exactly once, \\(P\\) must divide \\(B_M B_K\\) and
\\(B_K B_N\\); all our configurations satisfy this.

### 18.4.2 The kernel

<div class="algo">
<p class="algo-name">Register-Tiled-Kernel(\(A, B, C, M, K, N\))</p>
<ol>
<li>\(P = (B_M/T_M)(B_N/T_N)\), \(\textit{id} = ty \cdot (B_N/T_N) + tx\) <span class="c">// threads, my number</span></li>
<li>\(r_0 = \textit{blockIdx.y} \cdot B_M\), \(c_0 = \textit{blockIdx.x} \cdot B_N\)</li>
<li>\(\textit{acc}[i][j] = 0\) for all \(i \in [0, T_M)\), \(j \in [0, T_N)\) <span class="c">// in registers</span></li>
<li>for \(t \in [0, \lceil K/B_K \rceil)\) {</li>
<li style="--d:1">for \(e \in \{\textit{id}, \textit{id} + P, \ldots\} \cap [0, B_M B_K)\) { <span class="c">// copy tile A</span></li>
<li style="--d:2">\(\textit{tileA}[\lfloor e/B_K \rfloor][e \bmod B_K] = a_{r_0 + \lfloor e/B_K \rfloor,\, tB_K + e \bmod B_K}\), or 0 outside \(A\)</li>
<li style="--d:1">}</li>
<li style="--d:1">for \(e \in \{\textit{id}, \textit{id} + P, \ldots\} \cap [0, B_K B_N)\) { <span class="c">// copy tile B</span></li>
<li style="--d:2">\(\textit{tileB}[\lfloor e/B_N \rfloor][e \bmod B_N] = b_{tB_K + \lfloor e/B_N \rfloor,\, c_0 + e \bmod B_N}\), or 0 outside \(B\)</li>
<li style="--d:1">}</li>
<li style="--d:1"><span class="sc">Barrier</span>()</li>
<li style="--d:1">for \(u \in [0, B_K)\) {</li>
<li style="--d:2">\(\textit{aReg}[i] = \textit{tileA}[ty \cdot T_M + i][u]\) for all \(i \in [0, T_M)\) <span class="c">// \(T_M\) reads</span></li>
<li style="--d:2">\(\textit{bReg}[j] = \textit{tileB}[u][tx \cdot T_N + j]\) for all \(j \in [0, T_N)\) <span class="c">// \(T_N\) reads</span></li>
<li style="--d:2">\(\textit{acc}[i][j] = \textit{acc}[i][j] + \textit{aReg}[i] \cdot \textit{bReg}[j]\) for all \(i, j\) <span class="c">// outer product</span></li>
<li style="--d:1">}</li>
<li style="--d:1"><span class="sc">Barrier</span>()</li>
<li>}</li>
<li>for \(i \in [0, T_M)\), \(j \in [0, T_N)\) {</li>
<li style="--d:1">\(\textit{row} = r_0 + ty \cdot T_M + i\), \(\textit{col} = c_0 + tx \cdot T_N + j\)</li>
<li style="--d:1">if \(\textit{row} \in [0, M)\) and \(\textit{col} \in [0, N)\) { \(c_{\textit{row}, \textit{col}} = \textit{acc}[i][j]\) }</li>
<li>}</li>
</ol>
</div>

The procedure has the shape of <span style="font-variant: small-caps">Tiled-Kernel</span>. Lines 5
to 10 are the joint copy, now several values per thread; line 11 and line 17 are its 2 barriers.
Lines 12 to 16 replace the inner loop: lines 13 and 14 read \\(T_M + T_N\\) values from shared memory
into registers, and line 15 adds their outer product, \\(T_M T_N\\) multiply-adds on registers only.
Lines 19 to 22 store the \\(T_M T_N\\) results. As in Chapter 17, every thread stays in the loop to
the end and reaches every barrier, and only the stores are guarded.

In Figure 18.8, we run the procedure for thread \\((0, 0)\\) of the example. In phase 0, *tileA*
holds columns 0 and 1 of \\(A\\) and *tileB* rows 0 and 1 of \\(B\\). For \\(u = 0\\) the thread reads
*aReg* = (1, 5) and *bReg* = (1, 0) and adds their outer product; for \\(u = 1\\) it reads (2, 6)
and (0, 1). After phase 0 its sums are \\(\begin{pmatrix} 1 & 2 \\\\ 5 & 6 \end{pmatrix}\\), the
same 4 values that the 4 threads of Chapter 17 held after phase 0 (Figure 17.3). Phase 1 adds
\\(\begin{pmatrix} 10 & 11 \\\\ 22 & 23 \end{pmatrix}\\), and the thread ends with
\\(c_{00} = 11\\), \\(c_{01} = 13\\), \\(c_{10} = 27\\) and \\(c_{11} = 29\\).

<figure>
<img src="../../figures/ch18-phases.svg" alt="2 rows, 1 per phase. Phase 0 uses columns 0 and 1 of A and rows 0 and 1 of B, u = 0 and 1; thread (0, 0) reads rows 0 and 1 of that part of A and columns 0 and 1 of that part of B. Its 4 sums go from 0 0 / 0 0 to 1 2 / 5 6. Phase 1 uses columns 2 and 3 of A and rows 2 and 3 of B, u = 2 and 3, and adds 10 11 / 22 23, so the sums become 11 13 / 27 29: c00, c01, c10 and c11.">
<figcaption><b>Figure 18.8</b> The 4 sums of thread (0, 0) over the 2 phases: 1 2 / 5 6 after phase 0, then 11 13 / 27 29.</figcaption>
</figure>

<div class="exercises">

Exercises

18.8 Run phase 0 of the example for thread \\((1, 1)\\). Which values does it copy into each tile,
which values does it read for \\(u = 0\\), and what are its 4 sums after the phase?<br>
18.9 With \\(B_M = B_N = 64\\), \\(B_K = 16\\) and \\(T_M = T_N = 4\\), which slots of *tileA* does
thread \\(\textit{id} = 37\\) copy, and from which rows and columns of \\(A\\) in phase \\(t\\)?

</div>

## 18.5 Correctness

The kernel changes who computes what, but not the order of the terms in any entry, and its proof is
that of Theorem 17.1 with the invariant written for every sum of a thread.

<div class="theorem">

<span class="box-label">Theorem 18.1</span> When
<span style="font-variant: small-caps">Register-Tiled-Kernel</span> finishes, every entry of \\(C\\)
equals the corresponding entry of \\(AB\\), provided the block has \\(P\\) threads and \\(P\\) divides
\\(B_M B_K\\) and \\(B_K B_N\\).

</div>

<div class="proof">

*Proof.* Fix a thread and a pair \\(i \in [0, T_M)\\), \\(j \in [0, T_N)\\), let
\\(\textit{row} = r_0 + ty \cdot T_M + i\\) and \\(\textit{col} = c_0 + tx \cdot T_N + j\\), and treat entries beyond the
edge of a matrix as 0. We show that the loop of lines 4 to 18 maintains

\\[ \text{at the start of phase } t, \qquad \textit{acc}[i][j] = \sum_{s \in [0,\\, tB_K)} a_{\textit{row}, s}\\;
b_{s, \textit{col}}. \\]

*Initialization.* Line 3 sets every sum to 0, the empty sum.

*Maintenance.* In lines 5 to 10, thread \\(\textit{id}\\) writes slots \\(\textit{id}\\),
\\(\textit{id} + P, \ldots\\) of each tile. Because \\(P\\) divides the size of each tile, the \\(P\\)
threads write every slot exactly once, and the barrier of line 11 makes every write visible before
any thread reaches line 13. Hence \\(\textit{tileA}[ty \cdot T_M + i][u] = a_{\textit{row},\\, tB_K + u}\\) and
\\(\textit{tileB}[u][tx \cdot T_N + j] = b_{tB_K + u,\\, \textit{col}}\\) for every \\(u \in [0, B_K)\\), and over
the \\(B_K\\) passes of lines 12 to 16, line 15 adds exactly the terms with
\\(s \in [tB_K, tB_K + B_K)\\) to \\(\textit{acc}[i][j]\\). Because of the barrier of line 17, no
thread overwrites a tile for phase \\(t + 1\\) while another thread still reads it in phase \\(t\\).
So the invariant holds at the start of phase \\(t + 1\\).

*Termination.* The loop ends with \\(t = \lceil K/B_K \rceil\\), so \\(tB_K \ge K\\), and
\\(\textit{acc}[i][j] = c_{\textit{row}, \textit{col}}\\). Line 21 stores it if \\((\textit{row}, \textit{col})\\) lies in \\(C\\), and distinct
threads and distinct \\((i, j)\\) give distinct \\((\textit{row}, \textit{col})\\), so every entry of \\(C\\) is stored
exactly once.

</div>

The 2 barriers play the roles they played in Chapter 17, and removing either produces the same
errors as Sections 17.6.1 and 17.6.2. The new condition is the 1 on \\(P\\). If the block had fewer
threads than \\(P\\), some slots of each tile would never be written, and the sums would add values
left over from an earlier phase; if \\(P\\) did not divide the size of a tile, the last pass of the
copy loop would run past it. The Rust host function that builds the launch derives the block shape
from the 5 constants, so that the condition cannot be broken by the caller, and checks a second
condition of the Rust implementation, described in Section 18.6.4.

<div class="rule-box">

<span class="box-label">Rule</span> The number of threads in a block must be the number the kernel
was derived for. Compute the block shape from the tile sizes, in 1 place.

</div>

<div class="exercises">

Exercises

18.10 Which line of the proof fails if the block has \\(P/2\\) threads? Which entries of the example
would then be wrong?<br>
18.11 Write the invariant of Theorem 18.1 for thread \\((1, 0)\\) of the example at the start of
phase 1, with numbers.

</div>

## 18.6 Implementation

### 18.6.1 The tile sizes as compile-time constants

The kernel has 5 sizes, \\(B_M\\), \\(B_N\\), \\(B_K\\), \\(T_M\\) and \\(T_N\\), and all 5 must be
known when the kernel is compiled: the sums of a thread can stay in registers only if the compiler
knows how many there are and can name each one. A register has a fixed name, chosen at compile time;
no instruction selects a register by an index computed while the kernel runs. In C++ the 5 sizes are
template parameters, and in Rust they are const generic parameters. Both produce 1 compiled kernel
for each combination the program uses.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch18_matmul_register_tiled/src/main.rs:83:119}}
        // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch18_matmul_register_tiled/matmul_register_tiled.cu:20:39}}
    // ...
}
```

</div>
</div>

The register tile is the array `acc` of `TM` × `TN` floats: `[[0.0f32; TN]; TM]` in Rust and
`float acc[TM][TN] = {}` in C++. A Rust array type may use a const generic parameter as its length,
as here, but not an expression such as `TM * TN`; the 2-dimensional array needs none. The tiles in
shared memory follow Chapter 17: C++ declares 2 `__shared__` arrays sized by the template parameters,
and Rust, which cannot size a `static` by a generic parameter, takes both from 1 region of dynamic
shared memory. *tileA* has 1 column of padding, \\(B_K + 1\\) values per row. Without it, the 2 rows
of threads in a warp of a 16 × 16 block read rows of *tileA* that are \\(T_M B_K\\) values apart,
64 for a 4 × 4 tile, which fall in the same bank of shared memory, and the 2 reads would be served 1
after the other (Section 9.5.3).

### 18.6.2 The copy and the outer product

The copy of lines 5 to 11:

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
    pub fn matmul_register<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        const TM: usize,
        const TN: usize,
    >(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<TM, TN>>,
    ) {
        // ...
{{#include ../../../../code/oxide/ch18_matmul_register_tiled/src/main.rs:120:144}}
            // ...
    }
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// C (m x n) = A (m x k) B (k x n), all row-major. A block computes a BM x BN tile of C
// in phases of BK; thread (ty, tx) computes the TM x TN block of that tile that starts
// at row ty * TM and column tx * TN, and keeps its TM * TN sums in registers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmulRegister(int m, int n, int k, const float* a, const float* b, float* c)
{
    // ...
{{#include ../../../../code/cuda/ch18_matmul_register_tiled/matmul_register_tiled.cu:41:57}}
        // ...
}
```

</div>
</div>

The loops over `q` are the loops of lines 5 and 8, written with a fixed count, \\(B_M B_K / P\\)
passes for *tileA* and \\(B_K B_N / P\\) for *tileB*, 4 each for the 4 × 4 configuration. With a fixed
count the compiler can unroll the loop and issue all the loads from global memory of a phase before
the first store to shared memory, so their waiting times overlap. An earlier version that ran
`e` from `id` in steps of \\(P\\) while \\(e < B_M B_K\\), the literal form of line 5, does the same
work but its count is unknown to the compiler; on the measured GPU it ran the 2 × 2 configuration at
380 GFLOP/s, against 580 GFLOP/s with the fixed count.

The steps of lines 12 to 16:

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
    pub fn matmul_register<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        const TM: usize,
        const TN: usize,
    >(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<TM, TN>>,
    ) {
        // ...
        while t < k.div_ceil(BK) {
            // ...
{{#include ../../../../code/oxide/ch18_matmul_register_tiled/src/main.rs:146:182}}
            // ...
        }
        // ...
    }
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// C (m x n) = A (m x k) B (k x n), all row-major. A block computes a BM x BN tile of C
// in phases of BK; thread (ty, tx) computes the TM x TN block of that tile that starts
// at row ty * TM and column tx * TN, and keeps its TM * TN sums in registers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmulRegister(int m, int n, int k, const float* a, const float* b, float* c)
{
    // ...
    for (int t = 0; t < phases; ++t) {
        // ...
{{#include ../../../../code/cuda/ch18_matmul_register_tiled/matmul_register_tiled.cu:59:77}}
    }
    // ...
}
```

</div>
</div>

Every loop here has a count fixed by the constants, and every one must be unrolled. Unrolled, the
loops over `i` and `j` become \\(T_M T_N\\) separate multiply-adds with constant indices, and each
`acc[i][j]`, `a_reg[i]` and `b_reg[j]` names 1 register. A loop that is not unrolled must keep `acc`
in local memory, a per-thread area of global memory, because its index is computed at run time. In
C++, `#pragma unroll` asks the compiler to unroll the next loop; nvcc unrolls loops with constant
bounds of this size on its own, and the pragma states the intent. In cuda-oxide, `#[unroll]` on a
`while` loop does the same, and it is required: cuda-oxide at the revision used here does not unroll
a `for` loop over a range, and in Section 9.7 we measured the 4 × 4 kernel 33% slower with `for`
loops.

### 18.6.3 Writing the register tile

Each thread writes \\(T_M T_N\\) entries of \\(C\\), not 1. In C++ the store of lines 19 to 22 is a
pair of loops with a guard. In Rust, `DisjointSlice` with `Runtime2DIndex` gives a thread 1 element;
in Chapter 9 we wrote the 4 × 4 patch through `get_unchecked_mut`, an `unsafe` call whose `SAFETY`
comment argued that the patches do not overlap. cuda-oxide has an index space for this case,
`RuntimeRowMajorTiles<TM, TN>`, in which thread \\((y, x)\\) of the grid owns rows
\\([y T_M, y T_M + T_M)\\) and columns \\([x T_N, x T_N + T_N)\\) of a matrix whose row width the host
binds with `RowWidth`, exactly as in Chapter 16.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
    pub fn matmul_register<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        const TM: usize,
        const TN: usize,
    >(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<TM, TN>>,
    ) {
        // ...
{{#include ../../../../code/oxide/ch18_matmul_register_tiled/src/main.rs:187:203}}
    }
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// C (m x n) = A (m x k) B (k x n), all row-major. A block computes a BM x BN tile of C
// in phases of BK; thread (ty, tx) computes the TM x TN block of that tile that starts
// at row ty * TM and column tx * TN, and keeps its TM * TN sums in registers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmulRegister(int m, int n, int k, const float* a, const float* b, float* c)
{
    // ...
{{#include ../../../../code/cuda/ch18_matmul_register_tiled/matmul_register_tiled.cu:80:90}}
}
```

</div>
</div>

`thread::coord_2d_u32(lc)` returns the thread's grid coordinates, row \\(by \cdot (B_M/T_M) + ty\\)
and column \\(bx \cdot (B_N/T_N) + tx\\), and is available only in a kernel whose contract declares
`domain = 2, coordinates = u32` and whose `#[kernel(launch_context = lc)]` names the launch context.
`c.tile_2d32_rt(...)` checks, before it creates any pointer, that the whole \\(T_M \times T_N\\)
rectangle of this thread lies inside a single row band of the slice, and returns a tile view, or
`None`. `LocalIndex32::new(i)` turns `i` into an index that is known to be below `TM`, and
`tile.at(r, q).write(...)` stores without a further check. Distinct grid coordinates give disjoint
rectangles, so the write needs no `unsafe`, and the multiplication of thread coordinates by the tile
sizes reproduces exactly the ownership of Section 18.4.1.

### 18.6.4 The launch

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch18_matmul_register_tiled/src/main.rs:207:229}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch18_matmul_register_tiled/matmul_register_tiled.cu:132:138}}
```

</div>
</div>

Both functions derive the grid and the block from the constants, as the Rule asks: \\(B_N/T_N\\)
threads along \\(x\\), \\(B_M/T_M\\) along \\(y\\), and enough blocks to cover \\(C\\). The Rust
function also computes the dynamic shared memory, \\((B_M (B_K + 1) + B_K B_N) \cdot 4\\) bytes,
8,448 bytes for the 4 × 4 configuration, inside the range of 64 to 16,384 bytes that the contract
accepts. It returns an error unless \\(T_M\\) divides \\(m\\) and \\(T_N\\) divides \\(n\\).
`tile_2d32_rt` returns a tile only when the whole rectangle fits, so a thread whose tile crossed the
edge of \\(C\\) would write nothing, and its entries inside \\(C\\) would stay 0; the check turns that
silent error into a refused launch. The C++ store has a guard per entry and handles partial tiles,
at the cost of a comparison per store. Both programs check every entry of a 1,000 × 1,000 product
with the 4 × 4 configuration, whose 64 × 64 blocks overhang \\(C\\) by 24 rows and 24 columns.

<div class="exercises">

Exercises

18.12 Add the configuration \\(B_M = 64\\), \\(B_N = 32\\), \\(B_K = 16\\), \\(T_M = 4\\),
\\(T_N = 2\\) to both programs. How many threads does a block have, and how many values does each
copy per tile per phase?<br>
18.13 Remove `#[unroll]` from the loops over `i` and `j` in the Rust kernel, run the 4 × 4
configuration, and explain the change in speed.<br>
18.14 Launch the Rust kernel for a 1,002 × 1,002 product with the 4 × 4 configuration. What does
`regtile_config` return, and why?

</div>

## 18.7 Measurements

### 18.7.1 Results

The measurements come from the GPU of Chapters 16 and 17 (8 SMs, 102 GB/s, 2,400 GFLOP/s), with
the same method. Both programs run the 4 × 4 example with 1 block of 2 × 2 threads, check the
1,000 × 1,000 product, and then time 6 kernels on square matrices of 1,024, 2,048 and 4,096: the
tiled kernel of Chapter 17 with \\(T = 16\\), and the register-tiled kernel with thread tiles of
1 × 1, 2 × 1, 2 × 2, 4 × 4 and 8 × 8, all with 256 threads per block.

The Rust program prints:

```console
$ cargo oxide run
C = A B on the 4 x 4 example, 1 block of 2 x 2 threads, 2 x 2 entries per thread
    11   13    8    8
    27   29   24   24
    43   45   40   40
    59   61   56   56
equal to the product computed by hand: yes

  m = 1000, k = 1000, n = 1000, 4 x 4 per thread: 1000000 of 1000000 entries exact

C = A B, n x n, 256 threads per block
      n  thread        block        ms   GFLOP/s  reads/FMA  fetched MiB   exact
   1024  Ch. 17     16 x  16     7.888     272.2       2.00        512.0     64 of 64
   1024  1 x 1      16 x  16     9.657     222.4       2.00        512.0     64 of 64
   1024  2 x 1      32 x  16     5.559     386.3       1.50        384.0     64 of 64
   1024  2 x 2      32 x  32     3.792     566.2       1.00        256.0     64 of 64
   1024  4 x 4      64 x  64     2.541     845.1       0.50        128.0     64 of 64
   1024  8 x 8     128 x 128     2.821     761.2       0.25         64.0     64 of 64
   2048  Ch. 17     16 x  16    62.312     275.7       2.00       4096.0     64 of 64
   2048  1 x 1      16 x  16    78.145     219.8       2.00       4096.0     64 of 64
   2048  2 x 1      32 x  16    43.445     395.4       1.50       3072.0     64 of 64
   2048  2 x 2      32 x  32    30.245     568.0       1.00       2048.0     64 of 64
   2048  4 x 4      64 x  64    19.861     865.0       0.50       1024.0     64 of 64
   2048  8 x 8     128 x 128    21.330     805.4       0.25        512.0     64 of 64
   4096  Ch. 17     16 x  16   500.956     274.4       2.00      32768.0     64 of 64
   4096  1 x 1      16 x  16   626.368     219.4       2.00      32768.0     64 of 64
   4096  2 x 1      32 x  16   352.620     389.8       1.50      24576.0     64 of 64
   4096  2 x 2      32 x  32   241.749     568.5       1.00      16384.0     64 of 64
   4096  4 x 4      64 x  64   156.975     875.5       0.50       8192.0     64 of 64
   4096  8 x 8     128 x 128   166.083     827.5       0.25       4096.0     64 of 64
PASSED
```

The column *reads/FMA* is \\((T_M + T_N)/(T_M T_N)\\), and *fetched MiB* the model of
Section 18.3, \\(n^3 (1/B_M + 1/B_N)\\) values of 4 bytes. The C++ program prints the same lines;
the table compares the 2 at \\(n = 4{,}096\\), with the shared-memory limit of Section 18.1.2,
\\(300 \cdot 2 / \text{reads}\\) GFLOP/s, and the registers per thread that nvcc reports for each
C++ kernel.

| Kernel | Reads per multiply-add | Limit | C++ rate | Rust rate | C++ registers |
|---|---|---|---|---|---|
| Chapter 17, \\(T = 16\\) | 2 | 300 GFLOP/s | 271.9 GFLOP/s | 274.4 GFLOP/s | 38 |
| 1 × 1, 16 × 16 block | 2 | 300 GFLOP/s | 220.4 GFLOP/s | 219.4 GFLOP/s | 37 |
| 2 × 1, 32 × 16 block | 1.5 | 400 GFLOP/s | 391.5 GFLOP/s | 389.8 GFLOP/s | 40 |
| 2 × 2, 32 × 32 block | 1 | 600 GFLOP/s | 583.5 GFLOP/s | 568.5 GFLOP/s | 40 |
| 4 × 4, 64 × 64 block | 0.5 | 1,200 GFLOP/s | 1,009.9 GFLOP/s | 875.5 GFLOP/s | 72 |
| 8 × 8, 128 × 128 block | 0.25 | 2,400 GFLOP/s | 773.1 GFLOP/s | 827.5 GFLOP/s | 138 |

<figure>
<img src="../../figures/ch18-measured.svg" alt="Horizontal bars of the rate in GFLOP/s at n = 4,096 on the measured GPU, C++ and Rust for each kernel, with a red mark at the shared-memory limit of 300 times 2 divided by the reads per multiply-add. Chapter 17's kernel with T = 16: 271.9 and 274.4, limit 300. 1 by 1: 220.4 and 219.4, limit 300. 2 by 1: 391.5 and 389.8, limit 400. 2 by 2: 583.5 and 568.5, limit 600. 4 by 4: 1,009.9 and 875.5, limit 1,200. 8 by 8: 773.1 and 827.5, limit 2,400, the FP32 peak.">
<figcaption><b>Figure 18.9</b> Register tiles on the measured GPU at n = 4,096, against the shared-memory limit of each thread tile.</figcaption>
</figure>

### 18.7.2 Comparison with the model

Up to the 2 × 2 tile, the model of Section 18.1.2 predicts the measurements. Halving the reads per
multiply-add from 2 to 1 doubled the rate, and each configuration ran within 3% to 5% of its limit:
391.5 of 400 GFLOP/s for 2 × 1 and 583.5 of 600 for 2 × 2 in C++, with Rust within 3% of C++. The
1 × 1 configuration performs the same arithmetic as the tiled kernel of Chapter 17 and fetches the
same values, but ran at 220 GFLOP/s instead of 272; the 2 kernels differ only in their copy code,
which in the general kernel computes each slot from the flat number \\(e\\).

The 4 × 4 tile reached 1,009.9 GFLOP/s in C++, 42% of the FP32 peak and 3.7 times the tiled kernel
of Chapter 17, but only 84% of its limit of 1,200. Two other limits now come into view. Its
operands from global memory, 64 FLOP per float, allow at most \\(25.5 \cdot 64 = 1{,}632\\) GFLOP/s
on its 102 GB/s. And its 72 registers per thread leave room for 3 blocks of 256 threads on
an SM, \\(65{,}536 / (72 \cdot 256) = 3.6\\), 768 threads, an occupancy of 50%. With half as many
warps to switch between, the SM hides less of the time each warp spends waiting for shared memory
and for the barriers.

The 8 × 8 tile reads shared memory least, 0.25 times per multiply-add, but ran slower than the 4 × 4
tile in C++. Its 64 sums and 16 operands took 138 registers per thread, so only 1 block of 256
threads fits on an SM, \\(65{,}536 / (138 \cdot 256) = 1.9\\), an occupancy of 17%. When that block
waits at a barrier, the SM has nothing else to run. The Rust compiler allocated registers
differently, and its 8 × 8 kernel ran at 827.5 GFLOP/s, faster than the C++ one; its 4 × 4 kernel
at 875.5 GFLOP/s was slower. Larger register tiles help only while the registers they need leave
enough threads on each SM.

These results agree with those of Chapter 9, where we measured a 4 × 4 patch kernel of the same
shape on the same GPU at 901.4 GFLOP/s in C++, 931.9 GFLOP/s with `__launch_bounds__` asking for
4 blocks per SM, and 844.9 GFLOP/s in Rust. Our C++ kernel is 8% faster than the best of those, in
part because of the fixed-count copy loops of Section 18.6.2. cuBLAS ran the same product at
1,495.3 GFLOP/s, 62% of the peak; the techniques it adds are listed in Section 9.8, among them wider
loads from shared memory and copying the next tiles while computing on the current ones.

From the direct kernel of Chapter 16 to the 4 × 4 register tile, the 4,096 × 4,096 product went from
743 ms to 136 ms in C++, 5.5 times as fast, and each step removed 1 measured limit: tiling removed
the dependence on global memory, and register tiling the dependence on shared memory.

## 18.8 The complete programs

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch18_matmul_register_tiled/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch18_matmul_register_tiled/matmul_register_tiled.cu}}
```

</div>
</div>

Build and run the Rust program with `cargo oxide run` in its directory, and the C++ program with
`nvcc -O3 -arch=native matmul_register_tiled.cu -o matmul_register_tiled && ./matmul_register_tiled`.

## Common mistakes

- **Loops that are not unrolled.** If the loops over the register tile are not unrolled, `acc`
  moves to local memory and every multiply-add becomes a load and a store. In cuda-oxide write them
  as `while` loops with `#[unroll]`.
- **Too large a register tile.** 64 sums per thread need more than 100 registers, and fewer blocks
  fit on an SM; the 8 × 8 tile ran slower than the 4 × 4 tile.
- **A block shape that does not match the constants.** The copy loop assumes exactly
  \\((B_M/T_M)(B_N/T_N)\\) threads. Derive the block from the constants in 1 function.
- **Copy loops with a run-time count.** A copy loop whose count the compiler cannot see is not
  unrolled, and the loads of a phase wait 1 after another.
- **Forgetting the padding of tileA.** The threads of a warp then read *tileA* in the same bank and
  the reads are served 1 after another.
- **Partial thread tiles in Rust.** `tile_2d32_rt` writes only whole tiles; a matrix size that is
  not a multiple of the thread tile must be refused or padded.

## Summary

<div class="summary">

- Each multiply-add of the tiled kernel reads 2 values from shared memory, which limits it to 1/8 of
  the arithmetic peak.
- A thread that computes a \\(T_M \times T_N\\) block of \\(C\\) reads \\(T_M\\) values of \\(A\\)
  and \\(T_N\\) values of \\(B\\) per step and adds their outer product to its sums: on the example,
  4 outer products give \\(\begin{pmatrix} 11 & 13 \\\\ 27 & 29 \end{pmatrix}\\).
- Reads per multiply-add fall to \\((T_M + T_N)/(T_M T_N)\\): 1.5, 1, 0.5 and 0.25 for 2 × 1, 2 × 2,
  4 × 4 and 8 × 8. The sums live in registers, the third level of storage.
- A block computes a \\(B_M \times B_N\\) tile in phases of \\(B_K\\) and fetches
  \\(MNK(1/B_M + 1/B_N)\\) values from global memory.
- The proof of Chapter 17 carries over with the invariant written for every sum; the block must have
  exactly the number of threads the copy assumes.
- The 5 sizes are template parameters in C++ and const generic parameters in Rust, and every loop
  over the register tile is unrolled. In Rust, `RuntimeRowMajorTiles` lets each thread write its
  whole tile without `unsafe`.
- On the measured GPU the rate followed the shared-memory limit up to the 2 × 2 tile (583.5 of
  600 GFLOP/s). The 4 × 4 tile reached 1,009.9 GFLOP/s in C++ and 875.5 in Rust, 3.7 and 3.2 times
  the tiled kernel; the 8 × 8 tile was slower in both languages, because its registers leave few blocks on each
  SM (1 in C++).

</div>

In Chapter 19, we turn to stencil computations, in which each output depends on a small
neighbourhood of the input, and apply the same staging in shared memory to a grid with borders.

## Exercises

1. Compute, for the example with \\(T_M = T_N = 2\\), the number of reads from shared memory and the
   number of multiply-adds of the whole kernel, and compare with the tiled kernel of Chapter 17.
2. For \\(B_M = B_N = 64\\), \\(B_K = 16\\), \\(T_M = T_N = 4\\) and \\(n = 4{,}096\\), compute the
   bytes fetched from global memory and the FLOP per float fetched.
3. Extend both programs with a 4 × 8 thread tile (\\(B_M = 64\\), \\(B_N = 128\\), \\(B_K = 8\\)).
   Predict its reads per multiply-add and its shared-memory limit, and measure it.
4. Store *tileA* transposed, as \\(B_K \times B_M\\), so that line 13 reads \\(T_M\\) consecutive
   values. Which loads into shared memory now conflict, and is the kernel faster?
5. Replace the Rust store through `RuntimeRowMajorTiles` by `get_unchecked_mut` as in Chapter 9, and
   write the `SAFETY` comment that the new code needs.
6. With the register counts of Section 18.7.2, compute the occupancy of each configuration on an SM
   that holds 1,536 threads and 65,536 registers.
7. The 8 × 8 configuration uses \\(B_K = 8\\). Predict the effect of \\(B_K = 16\\) on shared memory per
   block and on the number of barriers, and measure it.

## Sources and further reading

- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026: the tiled kernel and
  its Exercise 7.2, from which we start in this chapter.
- V. Volkov and J. Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra", SC08, 2008: register
  blocking of matrix products on GPUs.
- S. Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance", 2022.
- NVIDIA, *CUDA C++ Programming Guide*, "Shared Memory" (banks) and `#pragma unroll`:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- cuda-oxide, rev b7fe543, `crates/cuda-device/src/view.rs`: `RuntimeRowMajorTiles`,
  `tile_2d32_rt` and `LocalIndex32`; `crates/cuda-device/src/thread.rs`: `coord_2d_u32`.
- Chapter 9 of this book, for the 4 × 4 patch kernel in C++ and Rust, occupancy and
  `__launch_bounds__`, and the comparison with cuBLAS.
