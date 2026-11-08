<img class="plate" src="../../art/ch17.svg" alt="Tiler, a hermit crab, walks along a beach under a shell built of square tiles. Behind it on the sand lie 2 tall stacks of tiles, marked A and B: the matrices in global memory. Ahead lies a small tide pool marked shared memory, where 2 tiles already rest, 1 from each stack. A tag says copy a tile once, read it 16 times: the idea of a tiled matrix multiplication.">

# Chapter 17: Tiled Matrix Multiplication

A GPU can perform arithmetic far faster than it can read operands from its global memory. In the
direct kernel of Chapter 16, every thread read both of its operands from global memory for every
multiply-add it performed, so the kernel spent most of its time waiting for memory rather than
computing. Tiling is the standard technique for reducing this traffic. The threads of a block
first copy a small piece of each input matrix into the fast on-chip shared memory, and then every
thread of the block takes its operands from that copy. A value that would otherwise be read from
global memory by many threads is then read from global memory only once per block.

In this chapter, we develop the tiled kernel step by step, using the same pair of 4 × 4 matrices
as in Chapter 16, so that every number can be checked by hand. We first determine which values
one block needs and how often the direct kernel reads them. We then split each dot product into
phases, let the threads of a block load the pieces they need into shared memory together, and
count the reads from global memory again. After that we remove each of the kernel's 2 barriers
in turn and follow the wrong values that appear, and we prove that the complete kernel is correct.
Finally, we implement the kernel in Rust and in CUDA C++, measure it, and explain why the measured
speedup is much smaller than the reduction in reads.

<div class="covers">

This chapter covers

- the values 1 block of the direct kernel needs, and how often it fetches them
- splitting a dot product into phases, and tiles in shared memory
- the barrier, and the tiled algorithm with its 2 barriers
- the fetches of the tiled kernel: \\(2MNK/T\\) instead of \\(2MNK\\)
- each barrier removed, with the wrong values 7 and 15 on the example
- a proof of correctness by a loop invariant, and matrices of any size
- shared memory in Rust (`SharedArray`, `DynamicSharedArray`, `unsafe` with `// SAFETY:`) and in
  CUDA C++ (`__shared__` arrays sized by a template parameter)
- measured speeds, and why the speedup is not \\(T\\)

</div>

The code of the chapter is in
[`code/oxide/ch17_matmul_tiled/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch17_matmul_tiled)
for Rust and in
[`code/cuda/ch17_matmul_tiled/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch17_matmul_tiled)
for C++. We use the notation of Chapter 16: \\(A\\) is \\(M \times K\\), \\(B\\) is
\\(K \times N\\), \\(C = AB\\), all stored in row-major order, and \\([a, b)\\) is the range of
integers from \\(a\\) to \\(b - 1\\).

## 17.1 Values needed by 1 block

Recall from Section 16.4.3 that the 4 threads computing row 0 of \\(C\\) all read the same value
\\(a_{01}\\) from global memory, each on its own. Our goal is to let such threads share a single
copy of the value. Since only threads of the same block can share data, we start by asking which
values one block needs. We take block \\((0, 0)\\) of the grid in Figure 16.5. Its 4 threads
compute the top-left 2 × 2 square of \\(C\\), that is, \\(c_{00} = 11\\), \\(c_{01} = 13\\),
\\(c_{10} = 27\\) and \\(c_{11} = 29\\). An entry \\(c_{ij}\\) needs row \\(i\\) of \\(A\\) and
column \\(j\\) of \\(B\\), so together the 4 threads need rows 0 and 1 of \\(A\\) and columns 0 and
1 of \\(B\\). These are 8 values from each matrix, 16 values in all, as shown in Figure 17.1.

<figure>
<img src="../../figures/ch17-block-needs.svg" alt="Matrices A, B and C of the 4 by 4 example. Rows 0 and 1 of A (1 2 3 4 and 5 6 7 8) are outlined in blue, columns 0 and 1 of B (1 0 2 1 and 0 1 1 2) in amber, and the other cells are faded. In C, the top-left 2 by 2 square is block (0, 0), its 4 cells coloured for threads (0,0), (0,1), (1,0) and (1,1). The note: block (0, 0) needs 16 values, and the direct kernel fetches them 32 times.">
<figcaption><b>Figure 17.1</b> The values needed by block (0, 0): 2 rows of A and 2 columns of B.</figcaption>
</figure>

Let us count how often the direct kernel reads these 16 values. Thread \\((0, 0)\\) reads the 4
values of row 0 of \\(A\\) and the 4 values of column 0 of \\(B\\). Thread \\((0, 1)\\) reads the
same row 0 of \\(A\\) again, together with column 1 of \\(B\\). In the same way, each of the 4
threads makes 8 reads, so the block makes 32 reads in total, and every one of the 16 values is
read twice. If the block instead copied each of the 16 values into shared memory once, and its
threads then took their operands from that copy, the block would read global memory only 16
times, half as often.

This plan has one difficulty. Shared memory is small: a block can use at most 48 KB of it by
default, and a streaming multiprocessor (SM) has between 48 KB and 228 KB in total, shared by all
the blocks that it runs. In our example, the rows are only 4 values long, but in a real problem a
row of \\(A\\) holds thousands of values. For example, with \\(K = 4{,}096\\), a block of 16 × 16
threads needs 16 rows of \\(A\\) and 16 columns of \\(B\\), which hold
\\(2 \cdot 16 \cdot 4{,}096 = 131{,}072\\) floats, or 512 KB. Such a strip of rows and columns
does not fit into shared memory, so the block cannot copy everything it needs at once. In the next
section we see that it does not have to.

## 17.2 Splitting the sum

The key to the solution is that a thread does not need its whole row and column at the same time.
The entry \\(c_{00}\\) is a sum of 4 products, and the terms of a sum may be added in any grouping.
Figure 17.2 groups the 4 terms of \\(c_{00}\\) in 2 pairs. The first pair uses the first 2 values
of row 0 of \\(A\\) and the first 2 values of column 0 of \\(B\\), and adds up to
\\(1 \cdot 1 + 2 \cdot 0 = 1\\). The second pair uses the last 2 values of each, and adds up to
\\(3 \cdot 2 + 4 \cdot 1 = 10\\). Together the 2 pairs give \\(1 + 10 = 11\\), which is
\\(c_{00}\\).

<figure>
<img src="../../figures/ch17-split.svg" alt="Matrices A and B of the 4 by 4 example. Row 0 of A and column 0 of B are coloured in 2 groups: columns 0 and 1 of A with rows 0 and 1 of B in blue, columns 2 and 3 of A with rows 2 and 3 of B in amber. Below, the blue group gives 1 times 1 plus 2 times 0 = 1 (phase 0), the amber group 3 times 2 plus 4 times 1 = 10 (phase 1), and 1 + 10 = 11, the entry c00.">
<figcaption><b>Figure 17.2</b> c<sub>00</sub> in 2 groups: phase 0 (blue) uses columns 0 and 1 of A and rows 0 and 1 of B, phase 1 (amber) the rest.</figcaption>
</figure>

Now consider all 4 threads of the block. For its first pair of terms, every thread needs only
columns 0 and 1 of rows 0 and 1 of \\(A\\), which form a 2 × 2 piece of \\(A\\), and rows 0 and 1
of columns 0 and 1 of \\(B\\), which form a 2 × 2 piece of \\(B\\). For its second pair, every
thread needs the 2 × 2 pieces next to these: columns 2 and 3 of \\(A\\), and rows 2 and 3 of
\\(B\\). The block can therefore work in 2 rounds. In each round, it copies 1 small piece of
\\(A\\) and 1 small piece of \\(B\\) into shared memory, and every thread adds the matching pair of
terms to its sum. At any moment the block holds only 2 pieces of 4 values each in shared memory,
however long the rows are.

<div class="definition">

<span class="box-label">Definition 17.1</span> **Tile and phase.** A **tile** is a square piece of
\\(A\\) or \\(B\\), of \\(T\\) rows and \\(T\\) columns, that a block holds in shared memory. The
number \\(T\\) is called the **tile width**. A **phase** is 1 round of the tiled kernel: the block
copies 1 tile of \\(A\\) and 1 tile of \\(B\\) into shared memory, and every thread adds \\(T\\)
products to its sum.

*Example.* With \\(T = 2\\), block \\((0, 0)\\) runs 2 phases. In phase 0, it uses the tile of
\\(A\\) in rows 0 and 1 and columns 0 and 1; in phase 1, the tile of \\(A\\) in rows 0 and 1 and
columns 2 and 3.

</div>

We can now state the grouping in general. We number the phases \\(t = 0, 1, 2, \ldots\\). Phase
\\(t\\) handles \\(T\\) consecutive terms of each sum, so phase 0 handles the terms with
\\(s = 0, \ldots, T - 1\\), phase 1 the terms with \\(s = T, \ldots, 2T - 1\\), and phase \\(t\\)
the \\(T\\) terms that begin at \\(s = tT\\). The product \\(tT\\) is therefore the position at
which phase \\(t\\) begins. In our example, with \\(T = 2\\), phase 0 begins at \\(s = 0\\) and
phase 1 at \\(s = 1 \cdot 2 = 2\\). We write the position of a term within its phase as \\(u\\),
which runs from 0 to \\(T - 1\\), so that every index \\(s\\) can be written as \\(s = tT + u\\).
For example, the term with \\(s = 3\\) is the term with \\(u = 1\\) in phase \\(t = 1\\), since
\\(3 = 1 \cdot 2 + 1\\). With this notation, the definition of the matrix product in
Definition 16.1 becomes a sum over the phases of a sum within each phase:

\\[ c_{ij} = \sum_{t \in [0, \lceil K/T \rceil)} \\;
\underbrace{\sum_{u \in [0, T)} a_{i,\\, tT + u}\\; b_{tT + u,\\, j}}_{\text{phase } t}. \\]

| Symbol | Meaning | In the example |
|---|---|---|
| \\(T\\) | tile width: rows and columns of a tile, terms added per phase | 2 |
| \\(t\\) | number of the phase | 0 or 1 |
| \\(tT\\) | index \\(s\\) at which phase \\(t\\) begins | 0 in phase 0, 2 in phase 1 |
| \\(u\\) | position of a term inside its phase | 0 or 1 |
| \\(\lceil K/T \rceil\\) | number of phases, \\(K/T\\) rounded up | \\(4/2 = 2\\) |
| \\(a_{i,\\, tT+u}\\) | value of \\(A\\) in row \\(i\\), column \\(tT + u\\) | \\(a_{0,3} = 4\\) for \\(i = 0\\), \\(t = 1\\), \\(u = 1\\) |

The number of phases is \\(\lceil K/T \rceil\\), the quotient \\(K/T\\) rounded up, because when
\\(K\\) is not a multiple of \\(T\\), the last phase is only partly filled. In that case, we treat the
entries beyond the edge of a matrix as 0, so that the missing terms add nothing to the sum. Note
also how the tiles move. From 1 phase to the next, the tile of \\(A\\) moves \\(T\\) columns to the
right along the rows of the block, and the tile of \\(B\\) moves \\(T\\) rows down along its
columns. Each thread still uses its entire row of \\(A\\) and column of \\(B\\); it simply receives
them 1 tile at a time.

## 17.3 Loading a tile together

Next, we decide which thread copies which value of a tile. A tile holds \\(T \times T = T^2\\)
values, and a block of the tiled kernel has \\(T \times T = T^2\\) threads, so the work divides
evenly: each thread copies exactly 1 value of the tile of \\(A\\) and 1 value of the tile of
\\(B\\). The natural choice is to let every thread copy the value that belongs at its own position
in the tile. Recall from Chapter 16 that thread \\((ty, tx)\\) is the thread in row \\(ty\\) and
column \\(tx\\) of its block, and that it computes the entry of \\(C\\) in row \\(\textit{row}\\)
and column \\(\textit{col}\\). We call the 2 tiles in shared memory *tileA* and *tileB*. In phase
\\(t\\), thread \\((ty, tx)\\) performs the 2 copies

\\[ \textit{tileA}[ty][tx] = a_{\textit{row},\\; tT + tx}, \qquad
\textit{tileB}[ty][tx] = b_{tT + ty,\\; \textit{col}}. \\]

The first copy reads from the thread's own row of \\(A\\), at column \\(tT + tx\\): the tile of
phase \\(t\\) begins at column \\(tT\\), and the thread takes the value \\(tx\\) places further
along. The second copy reads from the thread's own column of \\(B\\), at row \\(tT + ty\\), that is,
\\(ty\\) places below the first row of the tile. For example, in phase 1 thread \\((0, 1)\\) of
block \\((0, 0)\\) has \\(\textit{row} = 0\\) and \\(\textit{col} = 1\\), and copies
\\(a_{0,\\, 2 + 1} = a_{03} = 4\\) into *tileA*[0][1] and \\(b_{2 + 0,\\, 1} = b_{21} = 1\\) into
*tileB*[0][1].

| Symbol | Meaning | Thread \\((0, 1)\\), phase 1 |
|---|---|---|
| \\(ty, tx\\) | row and column of the thread within its block | 0, 1 |
| \\(\textit{row}, \textit{col}\\) | row and column of the entry of \\(C\\) the thread computes | 0, 1 |
| \\(tT + tx\\) | column of \\(A\\) the thread copies | \\(1 \cdot 2 + 1 = 3\\) |
| \\(tT + ty\\) | row of \\(B\\) the thread copies | \\(1 \cdot 2 + 0 = 2\\) |

As \\((ty, tx)\\) runs over all threads of the block, these copies fill every position of both
tiles exactly once, so no value is copied twice and none is missed. Note also that the threads
with the same \\(ty\\) copy \\(T\\) neighbouring values of the same row of \\(A\\), which lie next
to each other in memory, so their reads can be combined into few memory transactions. In
Figure 17.3 we follow block \\((0, 0)\\) through both phases. Each colour belongs to 1 thread and
marks the values that the thread copies as well as the sum that it accumulates.

<figure>
<img src="../../figures/ch17-phases.svg" alt="Block (0, 0) in 2 phases, 1 colour per thread. Phase 0: the tile of A at columns 0 and 1 (1 2, 5 6) and the tile of B at rows 0 and 1 (1 0, 0 1) are outlined in the global matrices; each value is coloured by the thread that copies it into the same slot of tileA or tileB in shared memory; the 4 sums are then 1, 2, 5 and 6. Phase 1: the tile of A moves to columns 2 and 3 (3 4, 7 8) and the tile of B to rows 2 and 3 (2 1, 1 2); the sums become 11, 13, 27 and 29, the top-left square of C.">
<figcaption><b>Figure 17.3</b> Block (0, 0) in 2 phases. Each colour marks 1 thread: the values it copies, the shared-memory slots it fills, and the sum it accumulates.</figcaption>
</figure>

Let us follow thread \\((0, 0)\\). In phase 0, it copies \\(a_{00} = 1\\) and \\(b_{00} = 1\\),
while the other 3 threads of the block copy the remaining 6 values of the 2 tiles. Once both tiles
are complete, the thread multiplies row 0 of *tileA* by column 0 of *tileB* and obtains
\\(1 \cdot 1 + 2 \cdot 0 = 1\\). In phase 1, the tiles hold the next 2 pieces of \\(A\\) and
\\(B\\), and the thread adds \\(3 \cdot 2 + 4 \cdot 1 = 10\\), which brings its sum to
\\(c_{00} = 11\\). Thread \\((0, 1)\\) performs the same steps with column 1 of each tile of
\\(B\\). It obtains \\(1 \cdot 0 + 2 \cdot 1 = 2\\) in phase 0, and \\(2 + 3 \cdot 1 + 4 \cdot 2 =
13\\) after phase 1, which is \\(c_{01}\\).

We can now see where the saving comes from. Consider thread \\((0, 1)\\) in phase 0, shown in
Figure 17.4. It uses 4 values from shared memory, but it copied only 2 of them, \\(a_{01}\\) and
\\(b_{01}\\); the other 2 were copied by its neighbours. In general, each value of *tileA* is used
by the \\(T\\) threads in its row of the block, and each value of *tileB* by the \\(T\\) threads
in its column. Every value that the block reads from global memory is therefore used \\(T\\) times.

<figure>
<img src="../../figures/ch17-operands.svg" alt="The shared tiles of block (0, 0) in phase 0: tileA holds 1 2 and 5 6, tileB holds 1 0 and 0 1. Row 0 of tileA and column 1 of tileB are coloured by the threads that wrote them: 1 by thread (0,0), 2 by thread (0,1), 0 by thread (0,1) and 1 by thread (1,1). Thread (0, 1) computes 1 times 0 plus 2 times 1 = 2. It copied 2 of its 4 operands; its neighbours copied the other 2.">
<figcaption><b>Figure 17.4</b> The operands of thread (0, 1) in phase 0. Other threads of the block copied 2 of the 4 operands.</figcaption>
</figure>

<div class="exercises">

Exercises

17.1 Carry out both phases of block \\((0, 1)\\), which computes \\(c_{02}\\), \\(c_{03}\\),
\\(c_{12}\\) and \\(c_{13}\\). Which values does thread \\((1, 0)\\) copy in each phase?<br>
17.2 With \\(T = 4\\), the whole example is a single block. How many phases does it run, and which
values does thread \\((2, 3)\\) copy?<br>
17.3 In phase \\(t\\), which thread of block \\((by, bx)\\) copies \\(a_{i, s}\\)? Express \\(ty\\)
and \\(tx\\) in terms of \\(i\\), \\(s\\), \\(by\\), \\(t\\) and \\(T\\).

</div>

## 17.4 The algorithm

### 17.4.1 The barrier

In Figure 17.4, thread \\((0, 1)\\) reads values that other threads wrote. The threads of a block
do not move in lockstep: 1 thread may run ahead while another falls behind, because their warps
are scheduled independently and their memory reads finish at different times. When threads
exchange data through shared memory, they need a meeting point where all of them wait for one
another.

<div class="definition">

<span class="box-label">Definition 17.2</span> **Barrier.** A **barrier** is a point in a kernel
that no thread of a block may pass until every thread of the block has reached it. The writes to
shared memory that a thread made before the barrier are visible to every thread of the block after
it. In cuda-oxide a barrier is `thread::sync_threads()`, and in CUDA C++ `__syncthreads()`.

*Example.* In Figure 17.5, thread 3 arrives first and waits; thread 1 arrives last, and then all 4
threads continue together.

</div>

<figure>
<img src="../../figures/ch17-barrier.svg" alt="4 threads of a block as horizontal bars on a time axis, with a vertical line for the barrier. Thread 3 arrives first, then thread 0, thread 2 and finally thread 1; each early thread waits on a dotted line until the barrier. After the barrier, all 4 bars start at the same time.">
<figcaption><b>Figure 17.5</b> A barrier. The 4 threads arrive at different times, wait (dotted), and leave together.</figcaption>
</figure>

### 17.4.2 The tiled kernel

Each phase repeats the same 4 steps: copy the tiles, wait, compute, wait (Figure 17.6). The 2 waits
are barriers, and in Section 17.6 we explain why each of them is needed.

<figure>
<img src="../../figures/ch17-cycle.svg" alt="4 boxes in a row: load tiles, global to shared; barrier, all loads complete; compute, T multiply-adds; barrier, all reads complete. An arrow runs from the last box back to the first, labelled next phase: t = t + 1, while t is less than the ceiling of K over T.">
<figcaption><b>Figure 17.6</b> 1 phase of the tiled kernel. The cycle repeats ⌈K/T⌉ times.</figcaption>
</figure>

Every thread executes the following procedure. The arrays *tileA* and *tileB* live in shared
memory, 1 pair per block.

<div class="algo">
<p class="algo-name">Tiled-Kernel(\(A, B, C, M, K, N\))</p>
<ol>
<li>\(tx = \textit{threadIdx.x}\), \(ty = \textit{threadIdx.y}\)</li>
<li>\(\textit{row} = \textit{blockIdx.y} \cdot T + ty\), \(\textit{col} = \textit{blockIdx.x} \cdot T + tx\)</li>
<li>\(\textit{sum} = 0\)</li>
<li>for \(t \in [0, \lceil K/T \rceil)\) {</li>
<li style="--d:1"><span class="c" style="float:none;margin-left:0">// copy 1 value of each tile; 0 outside the matrix</span></li>
<li style="--d:1">\(\textit{tileA}[ty][tx] = a_{\textit{row},\, tT + tx}\) if \(\textit{row} \in [0, M)\) and \(tT + tx \in [0, K)\), else 0</li>
<li style="--d:1">\(\textit{tileB}[ty][tx] = b_{tT + ty,\, \textit{col}}\) if \(tT + ty \in [0, K)\) and \(\textit{col} \in [0, N)\), else 0</li>
<li style="--d:1"><span class="sc">Barrier</span>() <span class="c">// wait until both tiles are full</span></li>
<li style="--d:1">for \(u \in [0, T)\) {</li>
<li style="--d:2">\(\textit{sum} = \textit{sum} + \textit{tileA}[ty][u] \cdot \textit{tileB}[u][tx]\)</li>
<li style="--d:1">}</li>
<li style="--d:1"><span class="sc">Barrier</span>() <span class="c">// wait until every thread has read the tiles</span></li>
<li>}</li>
<li>if \(\textit{row} \in [0, M)\) and \(\textit{col} \in [0, N)\) {</li>
<li style="--d:1">\(c_{\textit{row}, \textit{col}} = \textit{sum}\)</li>
<li>}</li>
</ol>
</div>

Lines 1 and 2 are the same as in
<span style="font-variant: small-caps">Direct-Kernel</span>. Lines 6 and 7 are the joint copy of
Section 17.3, and lines 9 to 11 add 1 phase of the sum, the inner sum over \\(u\\) of the formula
in Section 17.2. Lines 8 and 12 are the 2 barriers, and the zeros in lines 6 and 7 handle matrices
whose size is not a multiple of \\(T\\); we treat both in Section 17.6. Lines 14 to 16 store the
result, and they replace the early return of the direct kernel, for a reason given in Section 17.6.5.

## 17.5 Counting the fetches again

In each phase, every thread fetches 2 values from global memory, 1 for each tile. Block \\((0, 0)\\)
has 4 threads and runs 2 phases, so it makes \\(4 \cdot 2 \cdot 2 = 16\\) fetches, half of the 32 in
the direct kernel. Over all 4 blocks we count 64 fetches instead of 128 (Figure 17.7).

<figure>
<img src="../../figures/ch17-fetches.svg" alt="2 horizontal bars. The direct kernel makes 128 fetches from global memory in the 4 by 4 example; the tiled kernel with 2 by 2 tiles makes 64. A dashed line at 32 marks the number of different values in A and B.">
<figcaption><b>Figure 17.7</b> Fetches from global memory in the 4 × 4 example.</figcaption>
</figure>

In general, a block makes \\(2T^2\\) fetches per phase. When \\(T\\) divides every dimension, the
grid has \\((M/T)(N/T)\\) blocks and each block runs \\(K/T\\) phases, so the total is

\\[ \frac{M}{T} \cdot \frac{N}{T} \cdot \frac{K}{T} \cdot 2T^2 = \frac{2MNK}{T}. \\]

Tiling thus divides the traffic to global memory by exactly \\(T\\): each value of \\(A\\) is
fetched \\(N/T\\) times instead of \\(N\\) times, and each value of \\(B\\) \\(M/T\\) times. With
\\(T = 16\\), the 1,024 × 1,024 product moves 0.54 GB instead of 8.6 GB. In the units of
Chapter 9, the kernel performs \\(T\\) FLOP per float fetched from global memory, 16 instead of 1.

What limits \\(T\\)? A block holds at most 1,024 threads, so \\(T \le 32\\). The 2 tiles take
\\(2T^2\\) values of 4 bytes, which for \\(T = 16\\) is 2 KB of shared memory and for \\(T = 32\\)
8 KB, well within the 48 KB a block may use. In Section 17.8, we will see that the largest \\(T\\) is
not the fastest.

<div class="exercises">

Exercises

17.4 With \\(T = 4\\), the example is 1 block. How many fetches does it make, and how often is each
value fetched?<br>
17.5 Let \\(M = N = K = 100\\) and \\(T = 16\\). How many phases does each block run, and what
fraction of the values copied in the last phase are zeros?<br>
17.6 For \\(M = N = K = 4{,}096\\) and \\(T = 32\\), how many bytes does the tiled kernel fetch
from global memory?

</div>

## 17.6 Correctness

The threads of a block run at different speeds, yet they read and write the same shared-memory
slots. Without coordination, 2 things can go wrong: a thread can read a slot before it has been
filled, or overwrite a slot that another thread has not finished reading. We now remove the
barriers of <span style="font-variant: small-caps">Tiled-Kernel</span> one at a time and follow each
error on our example.

### 17.6.1 The first barrier: copy before computing

Suppose we delete line 8. In phase 1, thread \\((0, 0)\\) finishes its copying early and starts
computing. It needs \\(\textit{tileB}[1][0]\\), which thread \\((1, 0)\\) is about to fill with
\\(b_{30} = 1\\), but thread \\((1, 0)\\) is running late (Figure 17.8).

<figure>
<img src="../../figures/ch17-no-barrier1.svg" alt="A timeline of phase 1 without the first barrier. Thread (0,0) loads, then computes 3 times 2 plus 4 times tileB[1][0]. Thread (1,0) is late: its load of tileB[1][0] = 1 is still running when thread (0,0) reads the slot, marked by a red line. Notes: thread (0,0) reads tileB[1][0] before it is refilled; the stale value 0 gives 3 times 2 plus 4 times 0 = 6 in place of 10; result c00 = 7 in place of 11.">
<figcaption><b>Figure 17.8</b> Without line 8. Thread (0, 0) reads a slot before thread (1, 0) refills it, and picks up the value left over from phase 0.</figcaption>
</figure>

The slot still holds \\(b_{10} = 0\\) from phase 0. Thread \\((0, 0)\\) adds
\\(3 \cdot 2 + 4 \cdot 0 = 6\\) in place of 10, and we get \\(c_{00} = 1 + 6 = 7\\) instead of 11.
Line 8 prevents this: no thread may start computing until every thread has finished copying.

### 17.6.2 The second barrier: compute before the next copy

Now we restore line 8 and delete line 12 instead. In phase 0, thread \\((1, 1)\\) finishes its
multiply-adds early, moves on to phase 1, and stores \\(b_{31} = 2\\) in
\\(\textit{tileB}[1][1]\\). But thread \\((0, 1)\\) is still in phase 0 and still needs the old
value, \\(b_{11} = 1\\) (Figure 17.9).

<figure>
<img src="../../figures/ch17-no-barrier2.svg" alt="A timeline without the second barrier. Thread (0,1) is still computing phase 0, 1 times 0 plus 2 times tileB[1][1]. Thread (1,1) has finished its phase-0 load and compute and starts the phase-1 load, which stores tileB[1][1] = 2; a red line marks the moment thread (0,1) reads the slot. Notes: thread (1,1) overwrites tileB[1][1] from 1 to 2 during phase 0; thread (0,1) then computes 1 times 0 plus 2 times 2 = 4 in place of 2; result c01 = 15 in place of 13.">
<figcaption><b>Figure 17.9</b> Without line 12. Thread (1, 1) overwrites a slot while thread (0, 1) still needs the old value.</figcaption>
</figure>

Thread \\((0, 1)\\) computes \\(1 \cdot 0 + 2 \cdot 2 = 4\\) in place of 2 in phase 0, and phase 1
adds the correct 11, so this time we get \\(c_{01} = 15\\) instead of 13. Line 12 prevents this: no
thread may overwrite a tile until every thread has finished reading it. With both barriers in
place, every phase follows the same orderly schedule (Figure 17.10).

<figure>
<img src="../../figures/ch17-schedule.svg" alt="A timeline of 4 threads over 1 phase with both barriers. The loads of phase t end at different times, and each thread waits on a dotted line until barrier 1. All 4 threads then compute phase t, meet at barrier 2, and start the loads of phase t + 1.">
<figcaption><b>Figure 17.10</b> 1 phase with both barriers. The dotted segments show threads waiting for the slowest thread.</figcaption>
</figure>

The 2 errors are of different kinds. The first barrier orders a write before a read of the same
slot: the reading thread needs the value the writing thread produces, a **read-after-write**
dependence. The second barrier orders a read before the next write of the same slot: the writer
needs nothing from the reader, and the conflict exists only because both use the same slot in
consecutive phases, a **write-after-read** dependence. Both are data races in the sense of
Definition 16.8.

### 17.6.3 The broken kernels on the GPU

Figures 17.8 and 17.9 show what can happen; whether it happens depends on the timing of the
threads. We built the Rust kernel twice more, once without each barrier, and ran each 3 times on
the 4 × 4 example with \\(T = 2\\) and 5 times on matrices of 512 × 512 with \\(T = 16\\), comparing
every entry with the CPU. On the example, all 6 runs printed `equal to the product computed by
hand: yes`. At 512 × 512 the 2 kernels printed:

```console
without barrier 1, n = 512, T = 16: 260406 of 262144 entries wrong
without barrier 1, n = 512, T = 16: 260433 of 262144 entries wrong
without barrier 1, n = 512, T = 16: 260452 of 262144 entries wrong
without barrier 1, n = 512, T = 16: 260531 of 262144 entries wrong
without barrier 1, n = 512, T = 16: 260351 of 262144 entries wrong
```

```console
without barrier 2, n = 512, T = 16: 24255 of 262144 entries wrong
without barrier 2, n = 512, T = 16: 24704 of 262144 entries wrong
without barrier 2, n = 512, T = 16: 27347 of 262144 entries wrong
without barrier 2, n = 512, T = 16: 25287 of 262144 entries wrong
without barrier 2, n = 512, T = 16: 23913 of 262144 entries wrong
```

On the 4 × 4 example both broken kernels happened to give the right product, in every run. A
block of 2 × 2 threads is 4 threads of a single warp, and the hardware issued their loads and their
multiply-adds together, so no thread ran ahead of another. A missing barrier does not guarantee a
wrong result; it allows one. At 512 × 512, with blocks of 256 threads in 8 warps that the SM
schedules independently, the errors appeared at once. Without the first barrier, 99.3% of the
entries were wrong: almost every dot product met at least 1 slot that its writer had not filled
yet. Without the second, about 9.5% were wrong, because the error needs 1 warp to finish a whole
phase while another is still in it. The count changed from run to run, since it follows the
order in which the warps happen to run. The CUDA C++ kernels behaved in the same way, with 260,014
to 260,438 and 23,424 to 25,694 wrong entries. Neither program reported an error, and both compiled
without a warning: a missing barrier breaks no rule that a compiler checks.

### 17.6.4 The proof

We can now prove that the kernel is correct. The idea is to watch the variable *sum* of 1 thread
grow, phase by phase, as in Figure 17.11: before phase \\(t\\), it holds exactly the first \\(tT\\)
terms of its entry of \\(C\\).

<figure>
<img src="../../figures/ch17-invariant.svg" alt="Row 0 of A, 1 2 3 4, and column 0 of B, 1 0 2 1, drawn as 2 rows of cells: the first 2 cells of each in blue (phase 0), the last 2 in amber (phase 1). Below, 3 bars show how much of the dot product sum covers: at the start of t = 0 no terms, sum = 0; at the start of t = 1 the terms s in [0, 2), sum = 1 times 1 plus 2 times 0 = 1; at the end, t = 2, the terms s in [0, 4), sum = 1 + 3 times 2 + 4 times 1 = 11 = c00.">
<figcaption><b>Figure 17.11</b> The loop invariant for thread (0, 0) with T = 2. Before phase t, sum covers the terms s ∈ [0, tT).</figcaption>
</figure>

<div class="theorem">

<span class="box-label">Theorem 17.1</span> When <span style="font-variant: small-caps">Tiled-Kernel</span>
finishes, every entry of \\(C\\) equals the corresponding entry of \\(AB\\).

</div>

<div class="proof">

*Proof.* Fix a thread with \\(\textit{row} \in [0, M)\\) and \\(\textit{col} \in [0, N)\\), and
treat entries beyond the edge of a matrix as 0. We show that the loop of lines 4 to 13 maintains
the invariant

\\[ \text{at the start of phase } t, \qquad \textit{sum} = \sum_{s \in [0,\\, tT)}
a_{\textit{row}, s}\\; b_{s, \textit{col}}. \\]

*Initialization.* Before phase 0, \\(\textit{sum} = 0\\), which is the empty sum.

*Maintenance.* Assume the invariant holds at the start of phase \\(t\\). In lines 6 and 7, the
threads of the block fill both tiles completely (Section 17.3). Because of the barrier in line 8,
every one of these stores is complete and visible before any thread reaches line 10; this excludes
the error of Section 17.6.1. Hence \\(\textit{tileA}[ty][u] = a_{\textit{row}, tT + u}\\) and
\\(\textit{tileB}[u][tx] = b_{tT + u, \textit{col}}\\) for every \\(u \in [0, T)\\), and lines 9
to 11 add exactly the terms with \\(s \in [tT, tT + T)\\). Because of the barrier in line 12, no
thread overwrites a tile for phase \\(t + 1\\) while another thread is still reading it in phase
\\(t\\); this excludes the error of Section 17.6.2. So the invariant holds at the start of phase
\\(t + 1\\).

*Termination.* The loop ends with \\(t = \lceil K/T \rceil\\), so \\(tT \ge K\\). All terms with
\\(s \ge K\\) are 0, hence \\(\textit{sum} = c_{\textit{row}, \textit{col}}\\), and line 15
stores it.

</div>

The proof uses each barrier exactly once, in the maintenance step, and each use excludes 1 of the
2 errors. A thread outside \\(C\\) is not covered by the proof, because it stores nothing; but it
still takes part in lines 6 to 12, and in the next section we show why it must.

### 17.6.5 Matrices of any size

Nothing in the algorithm requires the dimensions to be multiples of \\(T\\). Take \\(C\\) of size
3 × 3 with \\(T = 2\\). We still need 4 blocks, they stick out past the edge of \\(C\\), and 7 of
their 16 threads have no entry to compute. If also \\(K = 3\\), the second tile of \\(A\\) reaches
for a column that does not exist (Figure 17.12).

<figure>
<img src="../../figures/ch17-edges.svg" alt="Left: a 4 by 4 grid of threads from 4 blocks of 2 by 2 laid over a 3 by 3 matrix C, outlined in black. The 9 cells inside C are coloured by block; the 7 cells in row 3 and column 3 are dashed: threads with no entry. Right: tileA in phase 1 when K = 3. Its left column holds A[0][2] and A[1][2]; its right column would hold column 3 of A, which does not exist, so both slots hold 0.">
<figcaption><b>Figure 17.12</b> Left: 2 × 2 blocks over a 3 × 3 matrix; dashed cells are threads with no entry. Right: a tile that extends past the last column of A is filled with zeros.</figcaption>
</figure>

For a missing value, lines 6 and 7 store 0, and a product with 0 adds nothing. Threads with no
entry need more care, and they explain the only structural difference between the 2 kernels. In
the direct kernel such a thread returns at once. In the tiled kernel, it must not: a barrier
releases the block only when every thread has arrived, and a thread that has returned never
arrives, so the rest of the block would wait forever, or, on some GPUs, continue with an undefined
result. Instead, these threads stay in the loop, copy zeros, wait at every barrier with everyone
else, and skip only the final store in lines 14 to 16. Both programs check the 3 × 3 product with
\\(T = 2\\), a 128 × 64 by 64 × 96 product and a 1,000 × 1,000 product with \\(T = 16\\), every
entry of each.

<div class="rule-box">

<span class="box-label">Rule</span> Every thread of a block must reach every barrier. Never place a
barrier where some threads of the block can skip it.

</div>

<div class="exercises">

Exercises

17.7 Without line 8, thread \\((1, 1)\\) might read \\(\textit{tileA}[1][0]\\) in phase 1 before
the slot is refilled. Which value would it use, and what would \\(c_{11}\\) become?<br>
17.8 Explain why the proof of Theorem 17.1 breaks when a thread returns early, even though that
thread writes no entry of \\(C\\).<br>
17.9 Without line 12, could \\(c_{00}\\) become wrong in our example? Which thread would have to
run ahead, and which slot would it overwrite?

</div>

## 17.7 Implementation

### 17.7.1 Shared arrays

In CUDA C++ an array in shared memory is a local array declared `__shared__`. Its length must be
known at compile time, and a template parameter qualifies, so `matmulTiled<T>` declares 2 tiles of
`T` × `T` floats and the compiler builds 1 kernel for each `T` the program uses.

In cuda-oxide a fixed-size array in shared memory is a `static mut` of type `SharedArray<f32, N>`
declared inside the kernel, with `N` known at compile time; we use 1 in Chapter 18 for comparison. A
`static` declared inside a function is a separate item of the program, with exactly 1 instance,
and Rust does not allow its type to mention a generic parameter of the function that contains it:
`SharedArray<f32, { T * T }>` inside `fn matmul_tiled<const T: usize>` is refused with error
E0401. A kernel generic over the tile width therefore takes its tiles from **dynamic shared
memory**, a single region per block whose size in bytes the host passes in the launch
configuration. `DynamicSharedArray::<f32>::get()` returns a raw pointer `*mut f32` to the start of
the region, and `DynamicSharedArray::<f32>::offset(bytes)` a pointer `bytes` bytes further on.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
    pub fn matmul_tiled<const T: usize>(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs:76:77}}
        // ...
    }
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// The same product with T x T tiles in shared memory. A block of T x T threads computes
// a T x T square of C in ceil(k / T) phases.
template <int T>
__global__ void matmulTiled(int m, int n, int k, const float* a, const float* b, float* c)
{
{{#include ../../../../code/cuda/ch17_matmul_tiled/matmul_tiled.cu:42:43}}
    // ...
}
```

</div>
</div>

*tileA* starts at byte 0 of the region and *tileB* at byte \\(4T^2\\), right after the \\(T^2\\)
floats of *tileA*. Both programs store a tile as a flat array of \\(T^2\\) values in row-major order
(Figure 17.13): the C++ array `tile_a[ty][tx]` is laid out in this way by the compiler, and the
Rust kernel writes the index \\(ty \cdot T + tx\\) of Definition 16.2 itself.

<figure>
<img src="../../figures/ch17-flat-tile.svg" alt="Left: a 2 by 2 tile with slots (0,0), (0,1), (1,0) and (1,1), 1 colour per thread. Right: the same 4 slots in a flat array at indices 0 to 3, in row-major order; slot (1,0) at index 2 is outlined. The rule: index(ty, tx, T) = ty times T plus tx, so (1, 0) is at 1 times 2 plus 0 = 2. With T = 16 the array holds 256 values.">
<figcaption><b>Figure 17.13</b> A 2 × 2 tile stored as a flat array of 4 slots. Slot (1, 0) is at 1 · 2 + 0 = 2.</figcaption>
</figure>

### 17.7.2 The tiled kernel

Before reading the code, we trace where the data of 1 thread travel (Figure 17.14): each phase, 2
values go from global memory into shared memory; from there, \\(2T\\) values feed the
multiply-adds; at the end, a single store writes *sum* into \\(C\\).

<figure>
<img src="../../figures/ch17-path.svg" alt="4 boxes joined by arrows: global memory holding A and B, shared memory holding tileA and tileB, registers holding sum, and global memory holding C. Under the arrows: load, 2 values per phase; read, 2T values per phase; store, 1 value.">
<figcaption><b>Figure 17.14</b> The path of data through 1 thread of the tiled kernel.</figcaption>
</figure>

The copy of lines 6 to 8:

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
    pub fn matmul_tiled<const T: usize>(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        // ...
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs:85:99}}
            // ...
    }
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// The same product with T x T tiles in shared memory. A block of T x T threads computes
// a T x T square of C in ceil(k / T) phases.
template <int T>
__global__ void matmulTiled(int m, int n, int k, const float* a, const float* b, float* c)
{
    // ...
{{#include ../../../../code/cuda/ch17_matmul_tiled/matmul_tiled.cu:50:57}}
        // ...
}
```

</div>
</div>

The loop over `t` is line 4, and `a_col` and `b_row` are the indices \\(tT + tx\\) and
\\(tT + ty\\) of lines 6 and 7. A value past the edge of a matrix becomes 0. The call to the barrier
sits outside every `if`, so every thread reaches it, as the Rule requires.

The Rust copy is split in 2 halves. The reads `a[row * k + a_col]` and `b[b_row * n + col]` are
ordinary slice reads, bounds-checked like every slice access in safe Rust. The stores through the
raw pointers `tile_a` and `tile_b` are not: the compiler cannot check how many threads write a slot
of shared memory, or when, so every access to it must stand in an `unsafe` block. Each `unsafe`
block carries a `// SAFETY:` comment, the Rust convention for stating why the block is sound, and
the 2 comments of the kernel summarize Section 17.6: every thread writes only its own slot, which
lies inside the tile because the block has \\(T \times T\\) threads, and the barriers keep writes and
reads of the tiles apart. In C++ the same stores need no marking; the argument is the same, but the
language does not ask for it.

The phase sum of lines 9 to 12 and the store of lines 14 to 16:

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
    pub fn matmul_tiled<const T: usize>(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        // ...
        while t < k.div_ceil(T) {
            // ...
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs:101:108}}
            // ...
        }
        // ...
    }
}

mod kernels {
    // ...
    pub fn matmul_tiled<const T: usize>(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, Runtime2DIndex>,
    ) {
        // ...
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs:113:119}}
    }
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
// The same product with T x T tiles in shared memory. A block of T x T threads computes
// a T x T square of C in ceil(k / T) phases.
template <int T>
__global__ void matmulTiled(int m, int n, int k, const float* a, const float* b, float* c)
{
    // ...
    for (int t = 0; t < phases; ++t) {
        // ...
{{#include ../../../../code/cuda/ch17_matmul_tiled/matmul_tiled.cu:59:62}}
    }
    // ...
}

// The same product with T x T tiles in shared memory. A block of T x T threads computes
// a T x T square of C in ceil(k / T) phases.
template <int T>
__global__ void matmulTiled(int m, int n, int k, const float* a, const float* b, float* c)
{
    // ...
{{#include ../../../../code/cuda/ch17_matmul_tiled/matmul_tiled.cu:65:67}}
}
```

</div>
</div>

`tile_a.add(ty * T + u)` walks along row `ty` of *tileA*, and `tile_b.add(u * T + tx)` walks down
column `tx` of *tileB*. The loop is a `while` loop with `#[unroll]`: inside each specialization of
the generic kernel, `T` is a constant, so the loop has a fixed count and the compiler replaces it
by \\(T\\) copies of its body. The C++ compiler unrolls the `for` loop over a template constant by
itself. The final store goes through `DisjointSlice` exactly as in the direct kernel, guarded by
`row < m && col < n`, and without any early return.

### 17.7.3 The launch

The launch configuration must match `T` in 3 respects: the block must have \\(T \times T\\) threads,
the grid must cover \\(C\\), and in Rust the dynamic shared memory must hold 2 tiles,
\\(2T^2 \cdot 4\\) bytes.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs:123:130}}

impl Problem {
    // ...
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs:198:209}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch17_matmul_tiled/matmul_tiled.cu:76:81}}
```

</div>
</div>

For \\(T = 16\\) and \\(M = N = 1{,}000\\), `tiled_config` returns a grid of 63 × 63 blocks of
16 × 16 threads and 2,048 bytes of dynamic shared memory. The launch contract of the Rust kernel
checks what it can. `dynamic_shared_range = (32, 8192)` accepts the 32 bytes of \\(T = 2\\) up to
the 8,192 bytes of \\(T = 32\\), and `#[launch_bounds(1024)]` limits a block to 1,024 threads; the
`requires` clause checks the buffer lengths as in Chapter 16. An exact block shape such as
`block = (16, 16, 1)` must be written as literal numbers, so it cannot depend on `T`, and the
contract cannot check that the block is \\(T \times T\\). That condition moves into the program:
`tiled_config::<T>` derives the block, the grid and the shared memory from `T` in 1 place, and the
`SAFETY` comment of the copy names it.

<div class="exercises">

Exercises

17.10 Set `T` to 32. Predict the effect on fetches from global memory and on shared memory per
block.<br>
17.11 Launch `matmul_tiled::<16>` with blocks of 8 × 8 threads and 2,048 bytes of dynamic shared
memory. Does the contract accept the launch? Which part of the `SAFETY` argument fails, and which
entries of \\(C\\) are wrong?<br>
17.12 Each thread of the tiled kernel performs 2 arithmetic operations for every 2 values it reads
from shared memory. Design a variant in which each thread computes several entries of \\(C\\), so
that every value read from shared memory is used more than once.

</div>

## 17.8 Measurements

### 17.8.1 Results

The measurements come from the GPU of Chapter 16 (8 SMs, 102 GB/s, an FP32 peak of
2,400 GFLOP/s), with the method of Section 16.6.1. Both programs time the direct kernel and the
tiled kernel with \\(T = 8\\), 16 and 32 on square matrices from 512 × 512 to 4,096 × 4,096.

The Rust program prints:

```console
$ cargo oxide run
C = A B on the 4 x 4 example, tiles of 2 x 2, 2 phases
    11   13    8    8
    27   29   24   24
    43   45   40   40
    59   61   56   56
equal to the product computed by hand: yes

Sizes that are not multiples of the tile width
  m =    3, k =    3, n =    3, T =  2:       9 of       9 entries exact
  m =  128, k =   64, n =   96, T = 16:   12288 of   12288 entries exact
  m = 1000, k = 1000, n = 1000, T = 16: 1000000 of 1000000 entries exact

C = A B, n x n; fetched: the modelled reads from global memory
      n  kernel            ms   GFLOP/s  fetched MiB  fetch GB/s   exact
    512  direct         1.462     183.6       1024.0       734.3     64 of 64
    512  T = 8          1.918     140.0        128.0        70.0     64 of 64
    512  T = 16         0.946     283.6         64.0        70.9     64 of 64
    512  T = 32         1.133     237.0         32.0        29.6     64 of 64
   1024  direct        15.561     138.0       8192.0       552.0     64 of 64
   1024  T = 8         15.368     139.7       1024.0        69.9     64 of 64
   1024  T = 16         8.192     262.1        512.0        65.5     64 of 64
   1024  T = 32         9.816     218.8        256.0        27.3     64 of 64
   2048  direct       125.589     136.8      65536.0       547.2     64 of 64
   2048  T = 8        123.308     139.3       8192.0        69.7     64 of 64
   2048  T = 16        66.321     259.0       4096.0        64.8     64 of 64
   2048  T = 32        80.277     214.0       2048.0        26.8     64 of 64
   4096  direct       996.270     138.0     524288.0       551.8     64 of 64
   4096  T = 8       1004.860     136.8      65536.0        68.4     64 of 64
   4096  T = 16       531.679     258.5      32768.0        64.6     64 of 64
   4096  T = 32       668.652     205.5      16384.0        25.7     64 of 64
PASSED
```

The column *fetched MiB* is the model of Section 17.5, \\(2n^3/T\\) values of 4 bytes, with
\\(T = 1\\) for the direct kernel; *fetch GB/s* divides it by the time. The C++ program prints the
same lines. The table compares the 2 languages at the smallest and the largest size.

| n | kernel | modelled fetches | C++ time | C++ rate | Rust time | Rust rate |
|---|---|---|---|---|---|---|
| 512 | direct | 1,024 MiB | 1.495 ms | 179.5 GFLOP/s | 1.462 ms | 183.6 GFLOP/s |
| 512 | \\(T = 8\\) | 128 MiB | 1.809 ms | 148.4 GFLOP/s | 1.918 ms | 140.0 GFLOP/s |
| 512 | \\(T = 16\\) | 64 MiB | 0.976 ms | 275.2 GFLOP/s | 0.946 ms | 283.6 GFLOP/s |
| 512 | \\(T = 32\\) | 32 MiB | 1.079 ms | 248.7 GFLOP/s | 1.133 ms | 237.0 GFLOP/s |
| 4,096 | direct | 524,288 MiB | 744.953 ms | 184.5 GFLOP/s | 996.270 ms | 138.0 GFLOP/s |
| 4,096 | \\(T = 8\\) | 65,536 MiB | 982.887 ms | 139.8 GFLOP/s | 1,004.860 ms | 136.8 GFLOP/s |
| 4,096 | \\(T = 16\\) | 32,768 MiB | 505.159 ms | 272.1 GFLOP/s | 531.679 ms | 258.5 GFLOP/s |
| 4,096 | \\(T = 32\\) | 16,384 MiB | 637.641 ms | 215.5 GFLOP/s | 668.652 ms | 205.5 GFLOP/s |

With \\(T = 16\\), the tiled kernel made 16 times fewer fetches and ran 1.53 times as fast as the
direct kernel in C++ at 512 × 512 (1.495 ms against 0.976 ms) and 1.47 times as fast at
4,096 × 4,096. In Rust the factors are 1.55 and 1.87; the second is larger because the generic
Rust direct kernel is slower at large sizes (Section 16.6.3). The Rust and C++ tiled kernels agree
within 6% at every size. All checked entries were exact, including the 3 shapes that are not
multiples of the tile width.

<figure>
<img src="../../figures/ch17-measured.svg" alt="2 panels for n = 4,096 on the measured GPU. Left, the rate in GFLOP/s of each kernel in C++ and in Rust: direct 184.5 and 138.0, T = 8 139.8 and 136.8, T = 16 272.1 and 258.5, T = 32 215.5 and 205.5, with a dashed line at 300 GFLOP/s, the limit set by 2 shared-memory reads per multiply-add. Right, the requested fetch rate of the C++ kernels in GB/s: direct 738.0, T = 8 69.9, T = 16 68.0, T = 32 26.9, with a dashed line at 102 GB/s, the bandwidth of global memory.">
<figcaption><b>Figure 17.15</b> The kernels at n = 4,096 on the measured GPU: rates against the shared-memory limit, and requested fetch rates against the bandwidth of global memory.</figcaption>
</figure>

### 17.8.2 Why the speedup is 1.5, not 16

Our model in Section 17.5 counted fetches as if every fetch went all the way to global memory.
In Section 16.6.3, we saw that this is not true for the direct kernel: it requested 738 GB/s, 7.2
times what global memory can deliver, so at least 86% of its fetches were answered by the L1 and L2
caches. Tiling removes 15 of every 16 requested fetches, but most of the fetches it removes were
already cheap. The hardware caches had recovered part of the reuse that tiling makes explicit.

The difference between the 2 approaches is control. A value stays in a cache only by chance, and
the chance shrinks as the matrices grow beyond the cache. A value in shared memory stays there for
exactly 1 phase, by construction, at every matrix size. The tiled kernel with \\(T = 16\\)
requested 68 GB/s from global memory, two-thirds of the 102 GB/s available, so global memory is no
longer what limits it.

What limits it now is shared memory. Each pass of line 10 performs 1 multiply-add and reads 2
values from shared memory, \\(\textit{tileA}[ty][u]\\) and \\(\textit{tileB}[u][tx]\\). The shared
memory of an SM serves 1 read instruction of a warp per cycle: 32 values of 4 bytes, 128 bytes per
cycle, when the 32 addresses fall in different banks or coincide. The 128 FP32 units of the SM can
perform 128 multiply-adds per cycle. A warp's 32 multiply-adds need 2 read instructions, 2 cycles
of shared memory, so the SM completes at most 16 multiply-adds per cycle, 1/8 of what its units
could do. On the measured GPU that limit is \\(2{,}400 / 8 = 300\\) GFLOP/s. The C++ kernel with
\\(T = 16\\) reached 272 to 275 GFLOP/s at every size, 91% of the limit, and the Rust kernel 259 to
284 GFLOP/s. A larger tile reduces the fetches from global memory but not this ratio: every
multiply-add of the tiled kernel still costs 2 reads of shared memory.

The other 2 tile widths are slower for reasons of their own. A block of 32 × 32 = 1,024 threads
leaves no room for a second block on an SM of the measured GPU, which holds at most 1,536 threads.

<div class="definition">

<span class="box-label">Definition 17.3</span> **Occupancy.** The **occupancy** of a kernel is the
number of threads resident on an SM divided by the largest number the SM can hold.

*Example.* With \\(T = 32\\) each SM of the measured GPU runs 1 block of 1,024 threads out of 1,536, an
occupancy of 67%; with \\(T = 16\\) it runs 6 blocks of 256 threads, 100%.

</div>

With \\(T = 32\\), when the threads of the single block wait at a barrier, the SM has no other
block to run, and it idles; the kernel ran at 216 GFLOP/s despite half the fetches of \\(T = 16\\).
With \\(T = 8\\), a block of 64 threads performs only 8 multiply-adds per thread between 2 barriers,
the copy and the barriers take a larger share of each phase, and an SM holds at most 16 blocks,
1,024 threads; it ran at 140 GFLOP/s, slower than the direct kernel in C++.

The speedup differs between GPUs, because it is set by how well the caches serve the direct kernel.
On a Jetson AGX Thor, Rust kernels for the same 2 algorithms with \\(T = 16\\) took 1.158 ms and
0.285 ms for 512 × 512 matrices, a speedup of 4.06 (Pathak, 2026). There the direct kernel
requested 927 GB/s, 3.4 times the 273 GB/s of the Thor's memory, so the caches answered at least
70% of its fetches, against 86% on the measured GPU, and tiling had more to gain. The tiled kernel ran at
941.5 GFLOP/s, 93% of the Thor's shared-memory limit of \\(8{,}100 / 8 = 1{,}012\\) GFLOP/s. On both
GPUs the tiled kernel ends near the same limit, 1/8 of the arithmetic peak, which we remove in
Chapter 18.

## 17.9 The complete programs

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch17_matmul_tiled/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch17_matmul_tiled/matmul_tiled.cu}}
```

</div>
</div>

Build and run the Rust program with `cargo oxide run` in its directory, and the C++ program with
`nvcc -O3 -arch=native matmul_tiled.cu -o matmul_tiled && ./matmul_tiled`.

## Common mistakes

- **Omitting a barrier.** Without the first barrier threads read slots before they are filled;
  without the second, a fast thread overwrites a tile that a slow one still reads. Neither mistake
  produces an error or a compiler warning (Section 17.6.3).
- **Returning early from a tiled kernel.** A thread outside \\(C\\) must still copy zeros and reach
  every barrier; only its final store is skipped.
- **Forgetting the zeros.** Without the `else 0` of lines 6 and 7, a tile at the edge holds values
  of the previous phase or of another matrix, and the last phase adds them.
- **A block shape that does not match the tile.** With fewer than \\(T \times T\\) threads, part of
  each tile is never filled; with more, threads write past the end of the tile. In Rust, derive the
  shape from `T` in 1 function.
- **Assuming shared-memory indices are checked.** An index through the pointers of
  `DynamicSharedArray`, or into a C++ `__shared__` array, is not compared with any length.
- **Choosing the largest tile.** \\(T = 32\\) fetches half as much as \\(T = 16\\) but ran slower,
  because a block of 1,024 threads fills an SM on its own (Section 17.8.2).

## Summary

<div class="summary">

- A block of \\(T \times T\\) threads needs \\(T\\) rows of \\(A\\) and \\(T\\) columns of \\(B\\);
  the direct kernel fetches each of these values \\(T\\) times.
- Splitting each dot product into phases of \\(T\\) terms lets the block copy 1 tile of \\(A\\) and
  1 tile of \\(B\\) into shared memory per phase, 1 value per thread, and every thread adds \\(T\\)
  terms from the copy.
- The tiled kernel fetches \\(2MNK/T\\) values instead of \\(2MNK\\): \\(T\\) FLOP per float
  instead of 1.
- The first barrier orders the copy before the reads (without it, \\(c_{00} = 7\\)); the second
  orders the reads before the next copy (without it, \\(c_{01} = 15\\)). A loop invariant proves the
  kernel correct, and the proof uses each barrier once.
- Zeros outside the matrices and threads that stay in the loop handle any size. Every thread of a
  block must reach every barrier.
- In Rust, shared memory is `unsafe` to access, and each access states its argument in a
  `// SAFETY:` comment; a tile sized by a generic parameter lives in dynamic shared memory. In C++
  a `__shared__` array can be sized by a template parameter.
- On the measured GPU tiles of 16 × 16 ran about 1.5 times as fast as the direct kernel, not 16 times:
  the caches had already absorbed most of the direct kernel's repeated fetches, and the tiled kernel
  is limited by its 2 shared-memory reads per multiply-add.

</div>

In Chapter 18, we remove the second limit by letting each thread compute several entries of \\(C\\)
and keep their sums in registers, so that each value read from shared memory is used several times.

## Exercises

1. For \\(T = 16\\) and \\(M = N = K = 4{,}096\\), how many floats does 1 block copy from global
   memory over the whole kernel, and how many FLOP does it perform? Check that the ratio is 16.
2. Thread \\((7, 12)\\) of block \\((3, 5)\\) with \\(T = 16\\): which slot of each tile does it write,
   and which entries of \\(A\\) and \\(B\\) does it copy in phase 2?
3. Remove the second barrier from the Rust kernel and run it 5 times on your GPU at
   512 × 512. Record the number of wrong entries each time, and explain why the number varies.
4. Change the phase sum to read `tile_a.add(u * T + ty)` instead of `tile_a.add(ty * T + u)`,
   without changing the copy. Which entries of the 4 × 4 example are still correct, and why?
5. Add \\(T = 4\\) to the program. Predict its fetches and its shared-memory limit, and compare with
   the measured rate.
6. Draw the timeline of Figure 17.8 for thread \\((1, 1)\\) instead of \\((0, 0)\\). Which entry
   becomes wrong, and what value does it take?
7. Rewrite `tiled_config::<T>` to return an error when `T * T` exceeds 1,024 threads, and call it
   for \\(T = 64\\).

## Sources and further reading

- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026: the derivation of
  the tiled kernel on the 4 × 4 example, the 2 broken kernels, the proof, and the measurements on a
  Jetson AGX Thor quoted in Section 17.8.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed.,
  MIT Press, 2022: loop invariants.
- NVIDIA, *CUDA C++ Programming Guide*, "Shared Memory" and `__syncthreads()`:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- cuda-oxide, rev b7fe543, `crates/cuda-device/src/shared.rs`: `SharedArray` and
  `DynamicSharedArray`; `cuda-macros`: `dynamic_shared_range` and `#[launch_bounds]`.
- The Rust Reference, "Static items", and `rustc --explain E0401`.
- V. Volkov and J. Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra", SC08, 2008.
- Chapter 9 of this book, for the same algorithm in CUDA C++ and the comparison with cuBLAS.
