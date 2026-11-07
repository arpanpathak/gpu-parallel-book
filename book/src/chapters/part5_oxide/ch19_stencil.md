<img class="plate" src="../../art/ch19.svg" alt="Ripple, a crab, stands on the sand beside a grid of square cells, 6 across and 6 down. The top row is hot, 100, and coloured red; the other edges are cold, 0, and blue. The inner cells shade from orange near the top to blue near the bottom, the steady temperatures of a heated plate. 1 inner cell and its 4 neighbours are outlined in white. A tag says each cell becomes the mean of its 4 neighbours.">

# Chapter 19: Stencils and Jacobi Iteration

The temperature inside a heated plate, the electric potential between charged conductors and the
pressure of a fluid at rest are all described by the same partial differential equation, the
Laplace equation. A computer solves it on a grid of points: the value at each point is computed
again and again from the values of its 4 neighbours, until the values stop changing. A computation
in which every point of a grid is updated from a fixed pattern of neighbours is called a stencil
computation. Stencils are among the most common workloads of a GPU, because all points of a grid
can be updated at the same time.

In this chapter we derive the parallel algorithm for 1 stencil, the 5-point stencil of the Laplace
equation, and its standard solver, Jacobi iteration. We work on a single 6 × 6 plate from the
first section to the last. We compute its steps by hand and show why an update in place produces a
different answer. We then prove that 2 buffers used in turn compute Jacobi iteration exactly, and
count the memory traffic that sets the speed of the kernel. Both implementations, in Rust and in CUDA
C++, check themselves against the hand-worked values.

<div class="covers" markdown="1">

This chapter covers

- the Laplace equation on a grid, and the 5-point stencil derived from it
- Jacobi iteration, worked by hand on a 6 × 6 plate
- why an update in place computes a different method, and the exact wrong value it produces
- 2 buffers used in turn, and a proof that they compute Jacobi iteration
- 1 thread per cell, in Rust with `DisjointSlice` and in C++ with `__restrict__`
- how fast the iteration converges, measured and predicted
- arithmetic intensity, and why a stencil is limited by memory bandwidth

</div>

The code of the chapter is in
[`code/oxide/ch19_stencil/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch19_stencil)
for Rust and in
[`code/cuda/ch19_stencil/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch19_stencil)
for C++.

## 19.1 The problem

### 19.1.1 A heated plate

We use the same plate throughout the chapter: a square plate whose 4 edges are held at fixed
temperatures: the top edge at 100 degrees and the other 3 edges at 0 degrees. Heat flows in
through the hot edge and out through the cold ones. After enough time the temperature at every
point stops changing, and we want to know that final temperature at every point.

We replace the plate by a grid of 6 × 6 cells, numbered by row and column from 0 to 5 (Figure
19.1). The cells on the edge of the grid have known temperatures; the 16 cells inside are the
unknowns.

<figure>
<img src="../../figures/ch19-plate.svg" alt="A 6 by 6 grid of cells. The 6 cells of row 0 hold 100 and are coloured red. The other edge cells, in column 0, column 5 and row 5, hold 0 and are coloured blue. The 16 cells of rows 1 to 4 and columns 1 to 4 hold question marks. A legend explains that the 20 edge cells are boundary cells with fixed values and the 16 inner cells are interior cells, the unknowns.">
<figcaption><b>Figure 19.1</b> The 6 × 6 plate: 20 boundary cells with fixed temperatures and 16 interior cells to compute.</figcaption>
</figure>

<div class="definition">

<span class="box-label">Definition 19.1</span> **Grid, boundary cell, interior cell.** A **grid** of
\\(W \times H\\) cells holds 1 value per cell; cell \\((r, c)\\) lies in row \\(r \in [0, H)\\) and
column \\(c \in [0, W)\\). A **boundary cell** lies in the first or last row or column; its value is
given by the problem and never changes. Every other cell is an **interior cell**, and its value is
unknown.

*Example.* Our plate has \\(W = H = 6\\): 36 cells, of which 20 are boundary cells and \\(4 \times
4 = 16\\) are interior cells, in rows 1 to 4 and columns 1 to 4.

</div>

The grid is stored in **row-major order**, row 0 first, then row 1, and so on, so that cell
\\((r, c)\\) is element \\(rW + c\\) of an array. Cell (2, 3) of our plate is element
\\(2 \cdot 6 + 3 = 15\\). The 2 neighbours of a cell in the same row are 1 element apart; the 2
neighbours in the same column are \\(W\\) elements apart.

### 19.1.2 The rule for the final temperature

Physics gives the rule that the final temperatures obey: at every interior cell, the temperature is
the mean of the temperatures of its 4 neighbours,

\\[ u_{r,c} = \frac{1}{4} \left( u_{r-1,c} + u_{r+1,c} + u_{r,c-1} + u_{r,c+1} \right). \\]

A cell warmer than all its neighbours would lose heat to them, and a cell colder than all of them
would gain heat, so in the final state neither can exist. We can check the rule on the answer.
Section 19.2 computes the final temperatures of our plate; in it, cell (1, 1) holds 45.45 and its
neighbours hold 100 above, 22.35 below, 0 to the left and 59.47 to the right. Their mean is
\\((100 + 22.35 + 0 + 59.47) / 4 = 181.82 / 4 = 45.45\\), the value of the cell itself (Figure
19.2).

<figure>
<img src="../../figures/ch19-stencil.svg" alt="Left: a cross of 5 cells, the centre (r, c) and its neighbours (r-1, c) above, (r+1, c) below, (r, c-1) to the left and (r, c+1) to the right, with weight 1/4 on each neighbour and 0 on the centre. Right: cell (1, 1) of the final plate: up 100, down 22.35, left 0, right 59.47, sum 181.82, times 1/4 gives 45.45, the value of the cell.">
<figcaption><b>Figure 19.2</b> The 5-point stencil, checked on cell (1, 1) of the final plate.</figcaption>
</figure>

The rule is the discrete form of the **Laplace equation**, which states that the sum of the second
derivatives of the temperature \\(u(x, y)\\) along \\(x\\) and along \\(y\\) is 0:

\\[ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0. \\]

On a grid with spacing \\(h\\) between cells, the second derivative along \\(x\\) is approximated by
the **finite difference** \\((u_{r,c+1} - 2u_{r,c} + u_{r,c-1}) / h^2\\), and the same along
\\(y\\). For \\(f(x) = x^2\\), whose second derivative is 2, the difference at \\(x = 1\\) with
\\(h = 1\\) gives \\((4 - 2 + 0) / 1 = 2\\), exactly. Substituting both differences into the
equation, multiplying by \\(h^2\\) and solving for \\(u_{r,c}\\) gives the rule above; the
spacing \\(h\\) cancels.

<div class="definition">

<span class="box-label">Definition 19.2</span> **Stencil.** A **stencil** is a fixed pattern of
neighbours, with a weight for each, from which a new value of a cell is computed. A **stencil
computation** applies the same stencil to every interior cell of a grid. The **5-point stencil**
uses a cell and its 4 neighbours in the same row and column, with weight 1/4 on each neighbour and
0 on the cell itself.

*Example.* Applied to cell (1, 1) of the final plate, the 5-point stencil gives 45.45 (Figure
19.2).

</div>

The rule gives 1 equation for each of the 16 interior cells, and each equation contains up to 4
unknowns. For cell (1, 1), whose upper and left neighbours are boundary cells, it reads
\\(u_{1,1} = (100 + u_{2,1} + 0 + u_{1,2}) / 4\\). A system of 16 equations can be solved by
elimination, but a grid of 4096 × 4096 cells has 16.7 million interior cells, and elimination on a
system of that size needs far more memory and time than the iterative method of the next
section.

<div class="exercises">

Exercises

19.1 Write the equation of the rule for cell (2, 3) of the plate. Which of its neighbours are
boundary cells?<br>
19.2 At which element of the array is cell (4, 1) stored, in row-major order with \\(W = 6\\)?<br>
19.3 Apply the finite difference to \\(f(x) = x^3\\) at \\(x = 1\\) with \\(h = 1\\) and with
\\(h = 0.5\\), and compare both with the exact second derivative, 6.

</div>

## 19.2 The sequential algorithm

### 19.2.1 Jacobi iteration

We do not know the final temperatures, but we know a property they have: each interior value is
the mean of its neighbours. Jacobi iteration turns that property into a procedure. It starts from
a guess and replaces every interior value by the mean of its neighbours, again and again.

<div class="definition">

<span class="box-label">Definition 19.3</span> **Jacobi iteration.** Let \\(u^{(0)}\\) be a grid
whose boundary cells hold the given values and whose interior cells hold a guess. **Jacobi
iteration** computes the grids \\(u^{(1)}, u^{(2)}, \ldots\\), where each step keeps the boundary
cells and sets every interior cell to

\\[ u^{(k+1)}_{r,c} = \frac{1}{4} \left( u^{(k)}_{r-1,c} + u^{(k)}_{r+1,c} + u^{(k)}_{r,c-1} +
u^{(k)}_{r,c+1} \right). \\]

The right-hand side uses only values of step \\(k\\).

*Example.* We start from 0 in every interior cell. After step 1, the 4 cells of row 1 hold 25; all
other interior cells still hold 0.

</div>

The method is named after Carl Gustav Jacob Jacobi. Let us carry out its first 2 steps by hand.
Before step 1 every interior cell is 0. Each of the 4 cells of row 1 has the hot boundary cell
above it and 3 neighbours at 0, so it becomes \\((100 + 0 + 0 + 0) / 4 = 25\\). Each cell of rows
2 to 4 has only neighbours at 0 and stays at 0.

In step 2, cell (1, 1) has 100 above, 0 below, 0 to its left and 25 to its right, so it becomes
\\(125 / 4 = 31.25\\). Cell (1, 2) has 25 on both sides and becomes \\((100 + 0 + 25 + 25) / 4 =
37.5\\). Each cell of row 2 has 25 above it and 0 elsewhere, so it becomes \\(25 / 4 = 6.25\\).
After 2 steps, heat has moved 2 rows into the plate. Figure 19.3 shows the 2 steps and the plate
after 100 steps, when the values agree with the final temperatures to 2 decimal places.

<figure>
<img src="../../figures/ch19-steps.svg" alt="4 copies of the 6 by 6 plate coloured by temperature. Step 0: top row 100, all else 0. Step 1: the 4 cells of row 1 are 25. Step 2: row 1 reads 31.25, 37.5, 37.5, 31.25 and row 2 reads 6.25 in each of its 4 interior cells. Step 100: rows 1 to 4 read 45.45, 59.47, 59.47, 45.45 / 22.35, 32.95, 32.95, 22.35 / 10.98, 17.05, 17.05, 10.98 / 4.55, 7.20, 7.20, 4.55.">
<figcaption><b>Figure 19.3</b> Jacobi iteration on the plate: steps 0, 1, 2 and 100.</figcaption>
</figure>

### 19.2.2 The algorithm

We write 1 step as a procedure that reads the grid \\(u\\) and writes the grid \\(v\\), and the
iteration as a loop around it. We write algorithms in the style of Cormen, Leiserson, Rivest and
Stein: procedure names in small capitals, numbered lines, and the half-open range \\([a, b)\\) for
the integers \\(a, a + 1, \ldots, b - 1\\).

<div class="algo">
<p class="algo-name">Jacobi-Step(\(u, v, W, H\))</p>
<ol>
<li>for \(r \in [1, H - 1)\) {</li>
<li style="--d:1">for \(c \in [1, W - 1)\) {</li>
<li style="--d:2">\(v_{r,c} = (u_{r-1,c} + u_{r+1,c} + u_{r,c-1} + u_{r,c+1}) / 4\)</li>
<li style="--d:1">}</li>
<li>}</li>
</ol>
</div>

<div class="algo">
<p class="algo-name">Jacobi(\(u, W, H, K\))</p>
<ol>
<li>\(v = u\) <span class="c">// both grids hold the boundary</span></li>
<li>for \(k \in [0, K)\) {</li>
<li style="--d:1"><span class="sc">Jacobi-Step</span>(\(u, v, W, H\))</li>
<li style="--d:1">exchange the names \(u\) and \(v\) <span class="c">// u is the newest grid</span></li>
<li>}</li>
<li>return \(u\)</li>
</ol>
</div>

Lines 1 and 2 of Jacobi-Step visit the interior cells; the boundary cells of \\(v\\) are never
written. Line 1 of Jacobi therefore copies the whole starting grid into \\(v\\), so that \\(v\\)
holds the boundary values too. Line 4 exchanges the names after each step, so that the grid
written last becomes the grid read by the next step.

### 19.2.3 Counting the work

Line 3 of Jacobi-Step executes once for each interior cell, \\((W - 2)(H - 2)\\) times per step.
Each execution reads 4 values, performs 3 additions and 1 multiplication by 1/4, 4 floating-point
operations, and writes 1 value. On our plate a step performs 16 updates: 64 reads, 64 operations
and 16 writes. On a grid of 4096 × 4096 cells a step performs 16.7 million updates, and Section
19.7 shows that such a grid needs millions of steps. A CPU performs the updates one after another;
the next section asks which of them may run at the same time.

<div class="exercises">

Exercises

19.4 Carry out step 3 of the plate by hand, starting from the values of step 2 in Figure 19.3.<br>
19.5 How many reads, writes and floating-point operations does Jacobi perform for \\(K = 100\\)
steps on the plate?<br>
19.6 Why does line 1 of Jacobi copy the whole grid and not only the boundary?

</div>

## 19.3 Updating in place

### 19.3.1 1 grid in place of 2

Jacobi keeps 2 grids. A natural question is whether 1 grid would do: line 3 could write the new
value of a cell into the same grid it reads from. Let us run step 1 that way on our plate, visiting
the cells row by row as the loops of Jacobi-Step do.

Cell (1, 1) is visited first and becomes \\((100 + 0 + 0 + 0) / 4 = 25\\), as before. Cell (1, 2)
is visited next. Its left neighbour is cell (1, 1), which now holds 25, not 0, so it becomes
\\((100 + 0 + 25 + 0) / 4 = 31.25\\) in place of 25. The error spreads: cell (1, 3) becomes
32.81, and cell (2, 1), whose upper neighbour already holds 25, becomes 6.25 in place of 0 (Figure
19.4).

<figure>
<img src="../../figures/ch19-inplace.svg" alt="Left: cell (1, 2) in step 1 computed from the old grid, (100 + 0 + 0 + 0) / 4 = 25. Right: the same cell computed in place after cell (1, 1) was already overwritten with 25: (100 + 0 + 25 + 0) / 4 = 31.25. Below, a timeline of 2 threads: the thread of cell (1, 1) writes 25; the thread of cell (1, 2) then reads that 25 as its left neighbour.">
<figcaption><b>Figure 19.4</b> In place, cell (1, 2) reads a neighbour that step 1 has already overwritten, and becomes 31.25 in place of 25.</figcaption>
</figure>

On a CPU the update in place gives another well-defined method, called Gauss-Seidel iteration,
which converges as well. On a GPU it gives no well-defined method at all. The cells are updated by
different threads. The thread of cell (1, 2) may read its left neighbour before or after the
thread of cell (1, 1) writes it, and the order is decided by how the hardware schedules the 2
threads. The
result can be 25, 31.25, or a mixture across the grid, and it can change from run to run. Section
19.5 runs this variant.

### 19.3.2 Which updates are independent

Look again at Definition 19.3. The right-hand side uses only values of step \\(k\\), and the
left-hand side writes only values of step \\(k + 1\\). No update of a step uses the result of
another update of the same step.

<div class="key-observation">

<span class="box-label">Key observation</span> The 16 updates of a Jacobi step read only the grid
of the previous step, so no update depends on another and all 16 can run at the same time,
provided that no update overwrites a value of the previous step that another update still has to
read.

</div>

The condition at the end of the observation is met by keeping the 2 grids of Jacobi apart: every
update reads from one and writes into the other, and the next step exchanges their roles.

<div class="definition">

<span class="box-label">Definition 19.4</span> **Ping-pong buffers.** **Ping-pong buffering** keeps
2 buffers of the same size. Each step reads 1 buffer and writes the other, and after each step the
2 exchange roles, so that the buffer written last is the one read next.

*Example.* Step 1 of the plate reads buffer \\(a\\) and writes buffer \\(b\\); step 2 reads \\(b\\)
and writes \\(a\\); step 3 reads \\(a\\) again.

</div>

<div class="exercises">

Exercises

19.7 Finish step 1 in place for row 1 and row 2 of the plate, visiting the cells row by row. Which
of the 16 values differ from Jacobi's?<br>
19.8 Suppose the cells of a step in place were visited column by column instead. Which value would
cell (1, 2) receive in step 1?

</div>

## 19.4 The parallel algorithm

### 19.4.1 Threads and blocks

A GPU runs 1 function many times at once. The terms below were introduced in Chapters 3 and 13; we
restate them because the rest of the chapter uses them.

<div class="definition">

<span class="box-label">Definition 19.5</span> **Kernel, thread, block, grid of blocks, warp.** A
**kernel** is a function that the GPU runs many times at once; each run is a **thread**, with its
own position and its own variables. Threads are grouped into **blocks** of up to 1,024 threads that
run on the same streaming multiprocessor (SM), share a fast on-chip memory and can wait for one
another. The blocks of 1 launch form a **grid of blocks**. The SM issues instructions for 32
threads of a block at a time; such a group is a **warp**.

*Example.* For the plate, the program launches 1 block of 32 × 8 threads. Its 256 threads form 8
warps, 1 per row of the block.

</div>

The word "grid" now has 2 meanings, the grid of cells and the grid of blocks. We say "grid" for
the cells and "the blocks of a launch" for the other.

We give each cell 1 thread. A thread finds its cell from 3 values that the GPU supplies: the
position of its block among the blocks (blockIdx), its position inside the block (threadIdx) and the
size of a block (blockDim). With blocks of \\(B_x \times B_y = 32 \times 8\\) threads, the thread
at position (threadIdx.x, threadIdx.y) of block (blockIdx.x, blockIdx.y) owns the cell

\\[ row = \text{blockIdx.y} \cdot B_y + \text{threadIdx.y}, \qquad col = \text{blockIdx.x} \cdot
B_x + \text{threadIdx.x}. \\]

A grid of \\(W \times H\\) cells needs \\(\lceil W / B_x \rceil \times \lceil H / B_y \rceil\\)
blocks. For the plate this is 1 block; its threads with \\(col \ge 6\\) or \\(row \ge 6\\) have no
cell, and of the 36 threads that do, the 20 on the boundary have nothing to compute.

Why 32 × 8? The 32 threads of a warp lie along a row of the grid, so they read 32 neighbouring
values, 128 bytes, at once.

<div class="definition">

<span class="box-label">Definition 19.6</span> **Coalesced access.** Global memory moves data in
**sectors** of 32 bytes (Chapter 7). A warp's access is **coalesced** when its 32 threads read
neighbouring addresses, so that the 32 values arrive in as few sectors as possible.

*Example.* A warp reading 32 neighbouring floats of a row, 128 bytes, needs 4 sectors when the row
segment starts at a multiple of 32 bytes. A warp reading 32 floats of a column needs 32 sectors.

</div>

### 19.4.2 The kernel

Every thread executes the following procedure. The 2 grids \\(u\\) and \\(v\\) are 2 different
buffers in global memory, the large memory of the GPU that every thread can read and write.

<div class="algo">
<p class="algo-name">Jacobi-Kernel(\(u, v, W, H\))</p>
<ol>
<li>\(row = \text{blockIdx.y} \cdot B_y + \text{threadIdx.y}\), \(col = \text{blockIdx.x} \cdot B_x + \text{threadIdx.x}\)</li>
<li>if \(row \in [1, H - 1)\) and \(col \in [1, W - 1)\) { <span class="c">// interior cells only</span></li>
<li style="--d:1">\(v_{row,col} = (u_{row-1,col} + u_{row+1,col} + u_{row,col-1} + u_{row,col+1}) / 4\)</li>
<li>}</li>
</ol>
</div>

<div class="algo">
<p class="algo-name">Jacobi-GPU(\(u, W, H, K\))</p>
<ol>
<li>copy \(u\) into 2 device buffers \(a\) and \(b\) <span class="c">// both hold the boundary</span></li>
<li>for \(k \in [0, K)\) {</li>
<li style="--d:1">launch <span class="sc">Jacobi-Kernel</span>(\(a, b, W, H\)) on \(\lceil W/B_x \rceil \times \lceil H/B_y \rceil\) blocks</li>
<li style="--d:1">exchange the names \(a\) and \(b\)</li>
<li>}</li>
<li>copy \(a\) to the host</li>
</ol>
</div>

Jacobi-Kernel is the body of Jacobi-Step with its 2 loops removed: the position of the thread plays
the role of \\(r\\) and \\(c\\). Line 2 sends threads outside the interior home, and line 3 is line
3 of Jacobi-Step. Jacobi-GPU is Jacobi with the step replaced by a launch. Launches issued one after
another on the same stream run in that order, so the launch of step \\(k + 1\\) starts only after
every thread of step \\(k\\) has written its value.

### 19.4.3 The work of 1 thread

Figure 19.5 follows the thread of cell (2, 2) through step 2. It reads 4 values from buffer \\(a\\),
which holds step 1: 25 above, 0 below, 0 to the left and 0 to the right. It writes
\\(25 / 4 = 6.25\\) into cell (2, 2) of buffer \\(b\\). The thread makes 4 reads from global
memory and 1 write, and none of the values it reads was written in the same step.

<figure>
<img src="../../figures/ch19-thread.svg" alt="Left: buffer a, the plate after step 1, with row 1 at 25. The 4 neighbours of cell (2, 2) are outlined: 25 above, 0 below, 0 left, 0 right. An arrow leads to buffer b on the right, where cell (2, 2) receives 6.25. A note counts 4 reads from global memory and 1 write.">
<figcaption><b>Figure 19.5</b> The work of 1 thread in step 2: 4 reads from buffer a, 1 write into buffer b.</figcaption>
</figure>

Over the whole step, the 16 interior threads of the plate read 64 values and write 16. A grid of
4096 × 4096 cells has 16.7 million interior threads per step, which read 67 million values,
268 MB, and write 16.7 million, 67 MB. Section 19.8 shows that most of those reads never reach the
memory chips.

<div class="exercises">

Exercises

19.9 Which cell does thread (5, 3) of block (2, 1) own with blocks of 32 × 8 threads?<br>
19.10 How many blocks of 32 × 8 threads cover a grid of 1000 × 1000 cells, and how many of their
threads have no cell?<br>
19.11 Follow the thread of cell (1, 2) through step 2, as Figure 19.5 does for cell (2, 2).

</div>

## 19.5 Correctness

The kernel is correct only if the buffers are used as Jacobi-GPU prescribes. We now break the
prescription in 2 ways, run each broken variant, and then prove that the correct program computes
Jacobi iteration exactly.

### 19.5.1 1 buffer for both arguments

Suppose we launch the kernel with the same buffer as \\(u\\) and \\(v\\), which is the update in
place of Section 19.3. In C++ the launch `jacobi<<<blocks, block>>>(a, a, w, h)` compiles without
a warning. We ran it 10 steps on the 4096 × 4096 plate, 3 times, and compared each result with 10
correct steps:

```console
in place, 4096 x 4096, run 1: 28420 of 16777216 cells differ from Jacobi after 10 steps
in place, 4096 x 4096, run 2: 28330 of 16777216 cells differ from Jacobi after 10 steps
in place, 4096 x 4096, run 3: 28464 of 16777216 cells differ from Jacobi after 10 steps
```

The number of wrong cells changes from run to run, as Section 19.3 predicted. On the 6 × 6 plate
the same launch printed the correct values of Figure 19.3. There all 36 cells belong to 1 block,
and in our runs every thread of the block read its neighbours before any of them wrote; nothing
in the hardware guarantees that order. A test on a small grid therefore
cannot show the error. It appears where 1 block reads cells that another block, running earlier or
later, writes.

In Rust the mistake does not compile. The kernel takes the old grid as `&[f32]` and the new one as
a `DisjointSlice`, which the launch builds from `&mut` borrow of a buffer. Passing the same buffer
to both borrows it as shared and as mutable at once:

```console
error[E0502]: cannot borrow `a` as mutable because it is also borrowed as immutable
   --> src/main.rs:146:58
    |
146 |         module.jacobi(&stream, &small, &a, RowWidth::new(&mut a, w as u32))?;
    |                ------                  --                ^^^^^^ mutable borrow occurs here
    |                |                       |
    |                |                       immutable borrow occurs here
    |                immutable borrow later used by call
```

### 19.5.2 A buffer without the boundary

The compiler cannot check what a buffer holds. Suppose that line 1 of Jacobi-GPU copies the plate
into \\(a\\) only, and \\(b\\) is created filled with 0. The kernel never writes a boundary cell, so
the top row of \\(b\\) stays at 0. Step 1 is unaffected: it reads \\(a\\). Step 2 reads \\(b\\),
whose top edge is cold, and cell (1, 1) becomes \\((0 + 0 + 0 + 25) / 4 = 6.25\\) in place of
31.25 (Figure 19.6). We ran this variant:

```console
  zeroed second buffer, after step 2
     100.00  100.00  100.00  100.00  100.00  100.00
       0.00    6.25   12.50   12.50    6.25    0.00
       0.00    6.25    6.25    6.25    6.25    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
  zeroed second buffer, after step 200
     100.00  100.00  100.00  100.00  100.00  100.00
       0.00   11.55   21.76   21.76   11.55    0.00
       0.00   13.88   17.55   17.55   13.88    0.00
       0.00    4.79    8.45    8.45    4.79    0.00
       0.00    2.45    3.58    3.58    2.45    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
```

<figure>
<img src="../../figures/ch19-boundary-bug.svg" alt="Left: buffer b after step 1 when it was created filled with 0: row 1 holds 25 but the top row holds 0 instead of 100. Right: step 2 reads that buffer; cell (1, 1) gets (0 + 0 + 0 + 25) / 4 = 6.25 instead of (100 + 0 + 0 + 25) / 4 = 31.25.">
<figcaption><b>Figure 19.6</b> A second buffer without the boundary: step 2 reads a cold top edge, and cell (1, 1) becomes 6.25 in place of 31.25.</figcaption>
</figure>

The iteration still converges, but to the answer of a different problem, in which the top edge is
hot only in every second step. After 200 steps cell (2, 2) holds 17.55 where the correct value is
32.95. No error is reported, and only a comparison with known values exposes the mistake.

### 19.5.3 A proof of correctness

We can now prove that the correct program computes Jacobi iteration. We watch the 2 buffers at the
start of each pass through the loop of Jacobi-GPU (Figure 19.7).

<figure>
<img src="../../figures/ch19-pingpong.svg" alt="3 rows, one per step. Before step 1, buffer a holds u0 and buffer b holds u0. The kernel reads a and writes the interior of b, which then holds u1; the names are exchanged, so a holds u1 and b holds u0. Before step 2, a holds u1; after step 2 and the exchange, a holds u2 and b holds u1. A note states the invariant: at the start of pass k, a holds u(k) and b holds the boundary values.">
<figcaption><b>Figure 19.7</b> The 2 buffers at the start of each pass: a holds the newest grid, b holds the boundary.</figcaption>
</figure>

<div class="theorem">

<span class="box-label">Theorem 19.1</span> When Jacobi-GPU finishes, buffer \\(a\\) holds the grid
\\(u^{(K)}\\) of Definition 19.3, in every cell.

</div>

<div class="proof">

*Proof.* We show that the loop of lines 2 to 5 maintains the following invariant: at the start of
pass \\(k\\), (i) buffer \\(a\\) holds \\(u^{(k)}\\) in every cell, and (ii) every boundary cell of
\\(b\\) holds its given value.

*Initialization.* Before pass 0, line 1 has copied the starting grid \\(u = u^{(0)}\\) into both
buffers. Both parts hold.

*Maintenance.* Assume the invariant at the start of pass \\(k\\). The launch in line 3 reads only
\\(a\\) and writes only interior cells of \\(b\\), and \\(a\\) and \\(b\\) are different buffers, so
no thread reads a value that another thread of the launch writes. Every thread therefore reads
values of \\(u^{(k)}\\), and line 3 of Jacobi-Kernel stores \\(u^{(k+1)}_{r,c}\\) into each interior
cell of \\(b\\). The boundary cells of \\(b\\) are not written and still hold the given values,
which are also the boundary values of \\(u^{(k+1)}\\). Hence \\(b\\) holds \\(u^{(k+1)}\\) in every
cell. Buffer \\(a\\) was not written, so its boundary cells still hold the given values. Line 4
exchanges the names, after which (i) and (ii) hold for pass \\(k + 1\\). The next launch starts
only after this one has finished, because launches on 1 stream run in order.

*Termination.* The loop ends after \\(K\\) passes, so \\(a\\) holds \\(u^{(K)}\\), and line 6
copies it to the host.

</div>

Each broken variant violates 1 step of the proof. The launch with 1 buffer violates "\\(a\\) and
\\(b\\) are different buffers" in the maintenance step. The zeroed buffer violates part (ii) of the
invariant at initialization, and the proof then fails at the first exchange.

<div class="rule-box">

<span class="box-label">Rule</span> A Jacobi step reads 1 buffer and writes the other, never the
same one, and both buffers hold the boundary values before the first step.

</div>

<div class="exercises">

Exercises

19.12 In the zeroed-buffer variant, compute cell (1, 2) after step 2 by hand and compare it with
the output above.<br>
19.13 Suppose the kernel also wrote the boundary cells, copying them from \\(u\\) to \\(v\\). Would
the zeroed second buffer still give a wrong answer? Which part of the invariant would the proof no
longer need?<br>
19.14 The output above shows the in-place launch with 3 different results. Why can a block of the
4096 × 4096 grid read a value that another block has already overwritten, while the single block of
the 6 × 6 plate did not?

</div>

## 19.6 Implementation

### 19.6.1 The kernel

Listing 19.1 shows the kernel in both languages. Each thread computes its position (line 1 of
Jacobi-Kernel), keeps only interior cells (line 2), and stores the mean of its 4 neighbours (line
3). The neighbours above and below are \\(W\\) elements away, at `c - width` and `c + width`; those
to the left and right are at `c - 1` and `c + 1`. Both versions add the 4 values in the same order
and multiply by 0.25 last, as the CPU reference of the programs does, so that GPU and CPU round
identically.

<p class="listing"><b>Listing 19.1</b> 1 Jacobi step, 1 thread per cell.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch19_stencil/src/main.rs:23:43}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch19_stencil/stencil.cu:27:40}}
```

</div>
</div>

The 2 versions differ in how they state that the 2 grids do not overlap. The C++ kernel receives 2
raw pointers and marks them `__restrict__`, a promise to the compiler that no element is reached
through both. The compiler does not check the promise. The Rust kernel receives the old grid as a
shared slice, `&[f32]`, and the new grid as a `DisjointSlice<f32, Runtime2DIndex>`, a slice that
each thread may write only at its own position. The host attaches the row width to the buffer with
`RowWidth::new`; on the device, `row_width()` returns it and `thread::index_2d_runtime(&new)`
returns the thread's (row, column) position as an index that only this thread holds. `get_mut`
turns that index into element `row * width + col`, checks it against the length, and returns a
mutable reference to it. No 2 threads hold the same index, so the write needs no `unsafe` block.
The aliasing rule of Section 19.5.1 is the same guarantee, checked by the compiler on the host.

### 19.6.2 The host loop

Listing 19.2 is Jacobi-GPU for the plate: lines 1 to 5 of the pseudocode, with the grid printed
after steps 1, 2 and 100.

<p class="listing"><b>Listing 19.2</b> 100 steps on the 6 × 6 plate, with the 2 buffers exchanged after each step.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn main() -> Result<(), Error> {
    // ...
{{#include ../../../../code/oxide/ch19_stencil/src/main.rs:136:154}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
int main()
{
    // ...
{{#include ../../../../code/cuda/ch19_stencil/stencil.cu:153:175}}
    // ...
}
```

</div>
</div>

Both buffers are filled from the same host grid `start`, which is line 1 of Jacobi-GPU and part
(ii) of the invariant. The exchange of line 4 moves no grid data. In C++, `std::swap(a, b)`
exchanges 2 pointers of 8 bytes. In Rust, `std::mem::swap(&mut a, &mut b)` exchanges 2
`DeviceBuffer` values, small handles of 40 bytes that hold the device address, the length and a
reference to the context; the grids stay where they are in device memory. A step on a 4096 × 4096
grid moves at least 134 MB, while the exchange moves 80 bytes of host memory.

### 19.6.3 Running the example

Both programs print the same values; the Rust program on the Orin NX prints:

```console
$ cargo oxide run
GPU: Orin, 8 SMs
1. 6 x 6 plate, top edge at 100
  after step 1
     100.00  100.00  100.00  100.00  100.00  100.00
       0.00   25.00   25.00   25.00   25.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
  after step 2
     100.00  100.00  100.00  100.00  100.00  100.00
       0.00   31.25   37.50   37.50   31.25    0.00
       0.00    6.25    6.25    6.25    6.25    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
  after step 100
     100.00  100.00  100.00  100.00  100.00  100.00
       0.00   45.45   59.47   59.47   45.45    0.00
       0.00   22.35   32.95   32.95   22.35    0.00
       0.00   10.98   17.05   17.05   10.98    0.00
       0.00    4.55    7.20    7.20    4.55    0.00
       0.00    0.00    0.00    0.00    0.00    0.00
```

Every value of steps 1 and 2 agrees with Section 19.2.1. The program also runs the same 100 steps
on the CPU with the function `cpu_step` and reports `same bits as the CPU after 100 steps: true`.
The 2 computations agree in every bit of every cell. Each cell is computed by the same 3 additions
in the same order followed by the same multiplication, and IEEE 754 single-precision arithmetic
rounds such a sequence identically on both processors.

<div class="exercises">

Exercises

19.15 Change the programs to print the grid after step 3 and compare it with your answer to
Exercise 19.4.<br>
19.16 The Rust launch passes `&a` and `RowWidth::new(&mut b, ...)`. Explain why the borrow checker
accepts this call but rejects the one of Section 19.5.1.

</div>

## 19.7 Convergence

### 19.7.1 Measuring the change

After 100 steps the plate agrees with the final temperatures to 2 decimal places. A solver must
decide by itself when to stop, and it does so by measuring how much a step still changes the grid.

<div class="definition">

<span class="box-label">Definition 19.7</span> **RMS change.** The **root-mean-square (RMS) change**
of a step from grid \\(u\\) to grid \\(v\\) of \\(N\\) cells is

\\[ \Delta = \sqrt{\frac{1}{N} \sum_{i \in [0, N)} (v_i - u_i)^2 }. \\]

*Example.* Step 1 of the plate changes 4 of the 36 cells by 25 each:
\\(\Delta = \sqrt{4 \cdot 625 / 36} = 8.33\\) degrees.

</div>

Both programs copy the 2 buffers to the host after every step of the plate and compute the RMS
change there; a 6 × 6 grid is 144 bytes, so the copy costs microseconds. They print the change and
its ratio to the change of the step before:

```console
  RMS change per step, and its ratio to the step before
    step   1    8.33e+00
    step   2    3.90e+00   ratio 0.4677
    step   3    2.52e+00   ratio 0.6478
    step   4    1.82e+00   ratio 0.7192
    step  10    4.48e-01   ratio 0.8075
    step  20    5.38e-02   ratio 0.8090
    step  40    7.75e-04   ratio 0.8086
```

After the first few steps, the change shrinks by the same factor every step, 0.809.

### 19.7.2 The rate of convergence

The factor can be predicted. Let the **error** of a grid be its difference from the final
temperatures. A Jacobi step multiplies the error by the same averaging that it applies to the
grid, because the final temperatures are left unchanged by a step. An error pattern that varies
quickly from cell to cell is averaged away in a few steps. The pattern that survives longest is the
smoothest one that is 0 on the boundary: a single bump, largest in the middle. On a grid whose
cells are numbered 0 to \\(n\\) along each side, that pattern is

\\[ e_{r,c} = \sin\frac{\pi r}{n} \, \sin\frac{\pi c}{n}. \\]

The 2 neighbours of cell \\(r\\) in its column hold \\(\sin(\pi(r - 1)/n)\\) and \\(\sin(\pi(r +
1)/n)\\) times the same column factor. By the identity \\(\sin(x - y) + \sin(x + y) = 2 \sin x \cos
y\\), their sum is \\(2\cos(\pi/n)\\) times the value of the cell, and the same holds along the
row. The mean of the 4 neighbours is therefore \\(\cos(\pi/n)\\) times the value of the cell, and
1 step multiplies this pattern by

\\[ \rho = \cos\frac{\pi}{n}. \\]

Our plate has \\(n = 5\\), and \\(\cos(\pi/5) = 0.8090\\), the ratio that the programs measured from
step 10 on. The early ratios are smaller because the rougher error patterns, which shrink faster,
have not yet died out.

The same programs then iterate a 64 × 64 plate, with \\(n = 63\\), until the RMS change of a step
falls below \\(10^{-6}\\) degrees. They measure the change every 1,000 steps:

```console
2. 64 x 64 plate until the RMS change per step is below 1e-6
  step  1000   RMS change 7.49e-03
  step  2000   RMS change 2.06e-03
  step  3000   RMS change 5.95e-04
  step  4000   RMS change 1.71e-04
  step  5000   RMS change 4.95e-05
  step  6000   RMS change 1.44e-05
  step  7000   RMS change 4.57e-06
  step  8000   RMS change 1.79e-06
  step  9000   RMS change 5.59e-07
  largest difference from the CPU after 9000 steps: 0.00e+00
```

The prediction is \\(\rho = \cos(\pi/63) = 0.998757\\) per step, so 1,000 steps multiply the change
by \\(0.998757^{1000} = 0.288\\). The measured ratios are \\(2.06 / 7.49 = 0.275\\),
\\(0.595 / 2.06 = 0.289\\), \\(0.171 / 0.595 = 0.287\\) and \\(0.0495 / 0.171 = 0.289\\) (Figure
19.8). After step 7,000 the ratio drifts upward, to 0.39 and 0.31, because the changes have become
smaller than the rounding error of single-precision values near 50, about \\(4 \times 10^{-6}\\)
degrees.

<figure>
<img src="../../figures/ch19-convergence.svg" alt="2 charts with a logarithmic vertical axis of RMS change per step. Left, the 6 by 6 plate, steps 1 to 40: the measured points fall on a straight line of slope 0.809 per step, cos(pi / 5). Right, the 64 by 64 plate, steps 1,000 to 9,000: 9 measured points from 7.49e-3 to 5.59e-7 follow a dashed line of 0.288 per 1,000 steps, cos(pi / 63) to the power 1,000.">
<figcaption><b>Figure 19.8</b> The RMS change per step falls by \(\cos(\pi/n)\) per step: 0.809 on the 6 × 6 plate and 0.288 per 1,000 steps on the 64 × 64 plate.</figcaption>
</figure>

The prediction also gives the cost of the method on a large grid. Reducing the error by a factor of
\\(e = 2.718\\) takes about \\(1 / (1 - \rho) \approx 2n^2/\pi^2\\) steps: 5 for our plate, about
800 for \\(n = 63\\), and about 3.4 million for a grid of 4096 × 4096 cells. The number of steps
grows with the square of the width, so large solvers use Jacobi steps as one part of faster
methods, such as multigrid, which apply a few Jacobi steps on a series of coarser grids. A fast
Jacobi step is the building block of those methods, and the rest of this chapter and the next
measure how fast a step can be.

<div class="exercises">

Exercises

19.17 Predict \\(\rho\\) and the factor per 1,000 steps for a 128 × 128 plate. Change the programs
and compare.<br>
19.18 About how many steps reduce the error of the 64 × 64 plate by a factor of 1,000?

</div>

## 19.8 Memory traffic and measurements

### 19.8.1 Arithmetic intensity

Each update performs 4 floating-point operations (FLOP). How many bytes must it move? The least
any step can move is 1 read and 1 write of each cell, 8 bytes, since every old value must be read
at least once and every new value written once.

<div class="definition">

<span class="box-label">Definition 19.8</span> **Arithmetic intensity, memory-bound.** The
**arithmetic intensity** of a computation is the number of floating-point operations it performs per
byte it moves to or from memory. A kernel is **memory-bound** on a GPU when its arithmetic intensity
lies below the GPU's **ridge point**, its peak arithmetic rate divided by its memory bandwidth
(Chapter 2); its speed is then set by the bytes it moves, not by the operations it performs.

*Example.* A Jacobi step performs 4 FLOP per cell and moves at least 8 bytes per cell, so its
arithmetic intensity is at most \\(4 / 8 = 0.5\\) FLOP per byte.

</div>

The ridge points of the GPUs compared in Chapter 2 range from about 10 FLOP per byte (A100) to
about 114 FLOP per byte (DGX Spark); the Jetson Orin NX lies at about 24 and the Jetson AGX Thor at
about 30 (Figure 19.9). A Jacobi step lies below every one of them by a factor of 20 or more, so it
is memory-bound on every GPU of the table. Its speed is measured best by its **effective
bandwidth**: the 8 bytes per cell that a step must move, divided by the time of the step. A step
can be no faster than a copy of the grid, which moves the same 8 bytes per cell.

<figure>
<img src="../../figures/ch19-intensity.svg" alt="Horizontal bars of the ridge points of 7 GPUs on a logarithmic axis of FLOP per byte from 0.1 to 200: A100 10, H100 20, Orin NX 24, AGX Thor 30, RTX 5090 58, RTX 4090 82, DGX Spark 114. A dashed red line at 0.5 FLOP per byte marks the Jacobi step, far left of the end of every bar.">
<figcaption><b>Figure 19.9</b> A Jacobi step, 0.5 FLOP per byte, lies far below the ridge point of every GPU.</figcaption>
</figure>

### 19.8.2 Measurements

The measurements come from a Jetson Orin NX, an integrated GPU with 8 SMs and 102 GB/s of memory
bandwidth; both programs also give the same results, bit for bit, on a Jetson AGX Thor (20 SMs,
273 GB/s). Both programs time a step on a 4096 × 4096 grid, 64 MiB per buffer, far larger than the
2 MiB L2 cache of the Orin NX, so that the grid must come from memory.
Each program first runs the kernel for 0.4 s, because these GPUs raise their clock frequency only
under sustained load, and then averages 50 calls; repeated runs differ by 10 to 20%. The C++ program
printed, on the Orin NX:

```console
3. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)
  copy kernel         1.756 ms     76.4 GB/s
  jacobi              1.966 ms     68.3 GB/s
PASSED
```

and the Rust program:

```console
3. 4096 x 4096 plate, 8 bytes per cell per step (1 read, 1 write)
  copy kernel         1.784 ms     75.2 GB/s
  jacobi              2.322 ms     57.8 GB/s
PASSED
```

| Kernel | Time per step | Effective bandwidth | Share of the copy |
|---|---|---|---|
| copy, C++ | 1.756 ms | 76.4 GB/s | 100% |
| Jacobi step, C++ | 1.966 ms | 68.3 GB/s | 89% |
| Jacobi step, Rust | 2.322 ms | 57.8 GB/s | 76% |

<figure>
<img src="../../figures/ch19-bandwidth.svg" alt="Horizontal bars of effective bandwidth on a 4096 by 4096 grid, Orin NX: copy kernel 76.4 GB/s, Jacobi step in C++ 68.3 GB/s, Jacobi step in Rust 57.8 GB/s. A dashed line marks the rated 102 GB/s. A second dashed line marks 67.9 GB/s, the rate predicted for 9 bytes per cell.">
<figcaption><b>Figure 19.10</b> Effective bandwidth of a copy and of the Jacobi step on the Orin NX.</figcaption>
</figure>

### 19.8.3 Why the step runs at 89% of a copy

The copy reaches 76.4 GB/s, 75% of the rated 102 GB/s; Chapter 7 showed that no kernel reaches the
rated figure. The copy is the ceiling for a step. The C++ step reaches 89% of it. If every one of
its 4 reads per cell went to memory, a step would move 16 bytes of reads and 4 of writes, 20 bytes
per cell, and take at least \\(20 / 8 = 2.5\\) times as long as the copy, 4.4 ms. It took 1.97 ms.
Requesting 20 bytes per cell in 1.966 ms is a rate of \\(20 \cdot 16.7 \times 10^6 / 1.966 \times
10^{-3}\\) = 171 GB/s, 1.7 times the 102 GB/s that the memory can deliver at all. At least
\\(1 - 102/171 = 40\\%\\) of the requested bytes must therefore have come from the L1 and L2
caches, on-chip memories that keep recently read values.

The caches find those values because neighbouring threads read neighbouring rows (Figure 19.11).
The 32 threads of a warp read 3 row segments of the old grid: the row above, their own row with 1
extra cell at each end, and the row below. The 8 warps of a block cover 8 consecutive rows, so the
row above warp 1 is the row of warp 0, and the row below is the row of warp 2. A block of 32 × 8
threads needs only 10 different row segments of the old grid, its own 8 rows and 1 on each side.
If each segment is read from memory once, a block reads \\(10 / 8 = 1.25\\) old values per cell and
writes 1: 9 bytes per cell instead of 8. At the copy's rate, 9 bytes per cell take
\\(1.756 \cdot 9 / 8 = 1.98\\) ms, against the measured 1.966 ms.

<figure>
<img src="../../figures/ch19-warp-reads.svg" alt="A block of 32 by 8 threads drawn as 8 rows of a grid. For warp 3, its 3 row segments are highlighted: row 2 above, row 3 itself with 1 extra cell at each end, row 4 below. Rows 2 and 4 are the rows of warps 2 and 4. Brackets show that the block as a whole touches 10 rows of the old grid, its 8 rows and 1 on each side, so each old value is fetched from memory about 1.25 times per cell written.">
<figcaption><b>Figure 19.11</b> The rows read by 1 warp are the rows of its neighbours: a block of 8 rows needs 10 rows of the old grid.</figcaption>
</figure>

The step is thus close to the best it can be. It moves about 9 bytes per cell where 8 are needed,
and its speed follows from that count. Chapter 20 asks whether a block can do better by copying its
10 rows into shared memory itself.

### 19.8.4 Why the Rust step is slower

The Rust kernel runs at 57.8 GB/s, 85% of the C++ kernel, although it reads and writes the same
cells. Its memory traffic is the same; its integer work is not. The C++ kernel computes positions
in 32-bit `int`. The Rust kernel computes them in 64-bit `usize`, derives the height with a 64-bit
division, `new.len() / width`, and checks each index against the length of its slice. To separate
these costs we changed only the C++ kernel, to 64-bit positions and the same division of the
length by the width; it then ran at 58.9 GB/s, the speed of the Rust kernel. A memory-bound kernel
is not free of arithmetic. With only 4 floating-point operations per cell, the integer
instructions that compute positions and check bounds make up most of the kernel, and a 64-bit
division, which the GPU performs as a sequence of many instructions, delays the loads.

<div class="exercises">

Exercises

19.19 Rewrite the Rust kernel so that it computes positions in `u32` and receives the height as an
argument, then measure it. How close does it come to the C++ kernel?<br>
19.20 Change the block shape of both programs to 16 × 16 and predict, with the row count of
Section 19.8.3, how many old values a block reads per cell. Then measure.

</div>

## 19.9 The complete programs

<p class="listing"><b>Listing 19.3</b> The complete Rust program. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch19_stencil/src/main.rs">code/oxide/ch19_stencil/src/main.rs</a></p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch19_stencil/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch19_stencil/stencil.cu}}
```

</div>
</div>

## Common mistakes

- **Updating the grid in place.** Threads then read neighbours that other threads have already
  replaced, and the result depends on their timing. On a small grid inside 1 block the error may
  not appear at all (Section 19.5.1). Use 2 buffers; in Rust the safe launch path rejects the same
  buffer as both arguments.
- **Leaving the boundary out of 1 buffer.** The kernel never writes boundary cells, so both
  buffers must hold them before the first step. The mistake converges to a wrong answer without
  any error (Section 19.5.2).
- **Copying the new grid back into the old one after every step.** Exchanging the 2 names moves 80
  bytes of host memory; a copy moves the whole grid.
- **Checking convergence after every step on a large grid.** Each check waits for the GPU and moves
  data to the host. Check every few hundred or few thousand steps.
- **Expecting Jacobi iteration to converge quickly on a large grid.** The steps needed grow with the
  square of the width; use Jacobi steps inside a faster method such as multigrid.
- **Judging a stencil kernel by its FLOP rate.** At 0.5 FLOP per byte the kernel is memory-bound;
  compare its effective bandwidth with a copy.

## Summary

- The Laplace equation, written with finite differences, says that at the final state every
  interior cell equals the mean of its 4 neighbours: the 5-point stencil.
- Jacobi iteration replaces every interior value by the mean of its old neighbours. On the 6 × 6
  plate, step 1 gives 25 in row 1, step 2 gives 31.25, 37.5 and 6.25, and 100 steps reach the final
  temperatures to 2 decimal places.
- An update in place computes a different value, 31.25 in place of 25 at cell (1, 2) in step 1. On
  a GPU it gives results that change from run to run: 28,420, 28,330 and 28,464 wrong cells in 3
  runs on a 4096 × 4096 grid.
- Ping-pong buffers keep the steps apart. Theorem 19.1 proves that they compute Jacobi iteration
  exactly, provided both buffers hold the boundary and the 2 arguments of a launch differ.
- Rust states the condition in types: the old grid is `&[f32]`, the new one a `DisjointSlice`, and
  the borrow checker rejects 1 buffer as both. C++ states it with `__restrict__`, unchecked.
- The RMS change falls by \\(\cos(\pi/n)\\) per step: 0.809 on the 6 × 6 plate and 0.288 per 1,000
  steps on a 64 × 64 plate, as measured.
- A step performs 0.5 FLOP per byte and is memory-bound on every GPU. The C++ step reached
  68.3 GB/s, 89% of a copy, because the caches already reduce its reads to about 1.25 per cell.

## Exercises

1. Change the boundary of the plate so that the top edge is at 100 and the left edge at 50. Predict
   the mean of the 4 centre cells at the final state from the symmetry of the problem, and check it
   with the programs.
2. A 9-point stencil also uses the 4 diagonal neighbours. How many reads per cell does it request,
   and how many old rows does a block of 32 × 8 threads touch? Predict its effective bandwidth from
   the measurements of Section 19.8.
3. Write a kernel that computes the RMS change on the GPU with a block reduction (Chapter 15) and
   use it for the 64 × 64 plate. Measure how often the loop can afford to check.
4. Modify the in-place variant so that it runs on the 6 × 6 plate with blocks of 1 × 1 threads.
   Does the result now differ from Jacobi's? Explain from the order in which blocks run.
5. Explain why Theorem 19.1 needs the sentence "launches on 1 stream run in order". Construct a
   program with 2 streams in which the proof fails.

## Sources and further reading

- R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM,
  2007. Chapters 2 and 3 derive the 5-point Laplacian and its error; chapter 4 analyses Jacobi
  iteration and its convergence factor.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd edition, SIAM, 2003. Jacobi and
  Gauss-Seidel iteration and their convergence.
- W. L. Briggs, V. E. Henson and S. F. McCormick, *A Multigrid Tutorial*, 2nd edition, SIAM, 2000.
  How Jacobi steps on coarser grids remove smooth errors quickly.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th
  edition, MIT Press, 2022. The pseudocode conventions and loop invariants used here.
- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026. The same method of
  presentation applied to the matrix product.
- NVlabs, *cuda-oxide*, rev `b7fe543`: `cuda-device/src/disjoint.rs` documents `Runtime2DIndex`,
  `index_2d_runtime` and why `get_mut` on a slice of run-time width is safe.
