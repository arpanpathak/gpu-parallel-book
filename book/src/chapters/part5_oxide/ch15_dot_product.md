<img class="plate" src="../../art/ch15.svg" alt="A pyramid of crabs on a sandy shore at dusk. 4 crabs on the sand hold up the products 2, 6, 3 and 4. On a rock above them, 2 crabs hold the sums of pairs, 8 and 7. On the highest rock, 1 crab holds the total, 15. A tag reads: pairs, then pairs of pairs.">

# Chapter 15: Reduction and the Dot Product

The dot product of 2 vectors multiplies them element by element and adds the products. It is the
inner loop of a matrix-vector product, it gives the length of a vector and the angle between 2
vectors, and every iterative solver computes several of them in each step to decide how far it is
from convergence. The multiplications are a map, the pattern of Chapter 14, and run in parallel in
the same way. The additions are not: all products must flow into 1 number, and the obvious loop
adds them 1 after another. A computation that combines many values into 1 in this way is called a
reduction, and it is the second basic pattern of parallel programming.

We derive the parallel reduction on a dot product of 2 vectors of 16 small integers. We write the
sequential loop and its chain of dependent additions, and use the associativity of addition to
regroup the additions into a tree. The tree becomes a kernel in which the threads of a block add
their values in shared memory, step by step, separated by barriers. We remove a barrier to see the
wrong sum it produces and prove the tree correct with a loop invariant. We then replace most of
the barriers by warp shuffles, and compare 2 ways of combining the sums of different blocks: a
second launch and atomic additions. The implementation is given in Rust with cuda-oxide and in
CUDA C++. From the measurements we explain why the fused kernel runs more than twice as fast as a
map followed by a separate reduction, and why atomic additions change the last bits of the result
from run to run.

<div class="covers" markdown="1">

This chapter covers

- the dot product as a map followed by a reduction, worked by hand on 16 elements
- the sequential sum and its chain of dependent additions
- associativity, commutativity and the reduction tree
- the tree in shared memory: strides, active threads and barriers, step by step
- a missing barrier and the wrong sum it produces
- a loop-invariant proof that the tree computes the sum
- warp shuffles and a block sum with 1 barrier
- combining the block sums with a second launch or with atomic additions
- why the order of float additions changes the result
- measured rates in Rust and C++ against the memory bandwidth

</div>

The code of the chapter is in
[`code/oxide/ch15_dot_product/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch15_dot_product)
(Rust) and
[`code/cuda/ch15_dot_product/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch15_dot_product)
(C++). The Rust crate uses the same `Cargo.toml` and `rust-toolchain.toml` as Chapter 13.

## 15.1 The problem

### 15.1.1 A worked example

We use the same pair of vectors throughout the chapter. Each has 16 small integers:

\\[
\begin{aligned}
a &= (1, 2, 3, 1, 5, 1, 2, 3,\ 3, 1, 2, 3, 1, 3, 2, 1),\\\\
b &= (2, 3, 1, 4, 1, 1, 4, 1,\ 1, 2, 2, 2, 1, 3, 1, 4).
\end{aligned}
\\]

We multiply them position by position. The first 8 products are \\(1 \cdot 2 = 2\\),
\\(2 \cdot 3 = 6\\), 3, 4, 5, 1, 8 and 3, and they add up to 32. The last 8 are 3, 2, 4, 6, 1, 9,
2 and 4, and they add up to 31. The dot product is \\(32 + 31 = 63\\) (Figure 15.1).

<figure>
<img src="../../figures/ch15-example.svg" alt="The worked example: vectors a = 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 2, 3, 1, 3, 2, 1 and b = 2, 3, 1, 4, 1, 1, 4, 1, 1, 2, 2, 2, 1, 3, 1, 4. Their products are 2, 6, 3, 4, 5, 1, 8, 3 (sum 32) and 3, 2, 4, 6, 1, 9, 2, 4 (sum 31). The dot product is 63.">
<figcaption><b>Figure 15.1</b> The worked example. The 16 products add up to \(32 + 31 = 63\).</figcaption>
</figure>

Keep 3 numbers in mind: the first 8 products add up to 32, the last 8 to 31, and all 16 to 63.
Later in the chapter the GPU splits the work exactly along this line, 1 block for each half, and
the faulty kernels of Sections 15.4 and 15.6 produce 24 in place of 32 and 31 in place of 63.

### 15.1.2 Definitions

<div class="definition">

<span class="box-label">Definition 15.1</span> **Dot product.** The dot product of 2 vectors
\\(a\\) and \\(b\\) of \\(n\\) elements is

\\[ a \cdot b = \sum_{i \in [0, n)} a_i\, b_i . \\]

*Example.* For the vectors above, \\(a \cdot b = 2 + 6 + 3 + \cdots + 4 = 63\\).

</div>

The dot product has 2 parts. The products \\(p_i = a_i b_i\\) form a map in the sense of
Definition 14.2: each depends only on position \\(i\\). The sum then combines all \\(n\\) products
into 1 number.

<div class="definition">

<span class="box-label">Definition 15.2</span> **Reduction.** A reduction combines the \\(n\\)
elements of an array into 1 value with a binary operator \\(\oplus\\), as
\\(p_0 \oplus p_1 \oplus \cdots \oplus p_{n-1}\\). The **identity** of the operator is the value
\\(e\\) with \\(e \oplus v = v\\) for every \\(v\\).

*Example.* The sum is a reduction with \\(\oplus = +\\) and identity 0. The maximum is a reduction
with \\(\oplus = \max\\) and identity \\(-\infty\\).

</div>

<div class="exercises">

Exercises

15.1 Compute the dot product of \\((1, 0, 2, 3)\\) and \\((4, 5, 1, 2)\\).<br>
15.2 What are the operator and the identity of a reduction that counts the elements of an array
that are larger than 10?<br>
15.3 In the worked example, which position contributes the largest product, and how much does it
contribute?

</div>

## 15.2 The sequential algorithm

### 15.2.1 The loop and its chain

On a CPU, Definition 15.1 becomes 1 loop with a running sum.

<div class="algo">
<p class="algo-name">Dot<span style="font-variant: normal">(\(a, b, n\))</span></p>
<ol>
<li>\(s = 0\)</li>
<li>for \(i \in [0, n)\) {</li>
<li style="--d:1">\(s = s + a_i \cdot b_i\) <span class="c">// 2 reads, 2 FLOP</span></li>
<li>}</li>
<li>return \(s\)</li>
</ol>
</div>

Line 3 executes \\(n\\) times. Each time it reads 2 floats, 8 bytes, and performs 2 FLOP, 1
multiplication and 1 addition. Our example therefore performs 32 FLOP on 128 bytes; the vectors
measured at the end of the chapter have \\(n = 2^{24}\\) elements, 33.6 million FLOP on 134.2 MB.

The loop of SAXPY in Chapter 14 had iterations that shared nothing. This loop is different.
Iteration \\(i\\) reads \\(s\\), which iteration \\(i - 1\\) wrote, so it cannot start before
iteration \\(i - 1\\) has finished. On the example, the running sum takes the values 2, 8, 11, 15,
20, 21, 29, 32, 35, 37, 41, 47, 48, 57, 59 and 63, each obtained from the one before (Figure
15.2). The 16 multiplications are independent; the 16 additions form a chain of 16 steps, and no
number of threads can shorten that chain while the additions are performed in this order.

<figure>
<img src="../../figures/ch15-chain.svg" alt="The sequential sum of the 16 products. The running sums are 2, 8, 11, 15, 20, 21, 29, 32, 35, 37, 41, 47, 48, 57, 59, 63, each joined to the next: 16 additions in a chain of 16 steps.">
<figcaption><b>Figure 15.2</b> The running sum of <span class="sc">Dot</span>: each addition needs the result of the one before, a chain of 16 steps.</figcaption>
</figure>

### 15.2.2 Regrouping the additions

The order of the chain is a choice that the loop makes, not a property of the sum. Addition has 2
properties that allow any other order.

<div class="definition">

<span class="box-label">Definition 15.3</span> **Associative and commutative operators.** An
operator \\(\oplus\\) is **associative** if \\((u \oplus v) \oplus w = u \oplus (v \oplus w)\\)
for all values, so that the parentheses of a long expression may be placed anywhere. It is
**commutative** if \\(u \oplus v = v \oplus u\\), so that the operands may be reordered.

*Example.* Integer addition is both: \\((2 + 6) + 3 = 2 + (6 + 3) = 11\\) and \\(2 + 6 = 6 + 2\\).
Subtraction is neither: \\((2 - 6) - 3 = -7\\), but \\(2 - (6 - 3) = -1\\).

</div>

Associativity lets us add the 16 products in pairs, then the pairs in pairs, and so on (Figure
15.3). The first step adds 8 independent pairs, \\(2 + 6 = 8\\), \\(3 + 4 = 7\\), and so on to
\\(2 + 4 = 6\\). The second step adds 4 pairs of these sums, the third 2, and the fourth adds
\\(32 + 31 = 63\\). The tree performs \\(8 + 4 + 2 + 1 = 15\\) additions, as many as the chain
needs after its first addition to 0, but in 4 steps instead of 16, because the additions of each
step are independent of each other.

<figure>
<img src="../../figures/ch15-regroup.svg" alt="The 16 products regrouped in pairs. Step 1 makes 8, 7, 6, 11, 5, 10, 10, 6; step 2 makes 15, 17, 15, 16; step 3 makes 32 and 31; step 4 makes 63. 15 additions in 4 steps.">
<figcaption><b>Figure 15.3</b> The 16 products added in pairs: 15 additions in 4 steps.</figcaption>
</figure>

In general, a tree reduces \\(n\\) values with \\(n - 1\\) operations in \\(\lceil \log_2 n \rceil\\)
steps, because every step halves the number of values. For \\(n = 2^{24}\\) the chain is 16,777,216
steps long, and the tree 24.

<div class="key-observation">

<span class="box-label">Key observation</span> Because addition is associative and commutative,
the 16 products may be added in any grouping. Added in pairs, then in pairs of pairs, they need 4
steps instead of 16, and the additions within each step are independent, so a different thread can
perform each of them.

</div>

Floating-point addition is associative only approximately, because every addition rounds its
result. In Section 15.6.3, we show what that changes; until then, we work with the exact integers
of the example.

<div class="exercises">

Exercises

15.4 Draw the tree for the 8 values \\(4, 1, 3, 5, 2, 2, 6, 1\\). How many additions and how many
steps does it take?<br>
15.5 How many steps does a tree need for \\(n = 1{,}000\\) values? Explain the rounding up.<br>
15.6 The maximum is associative and commutative. Run the tree of Figure 15.3 with \\(\max\\) in
place of \\(+\\).

</div>

## 15.3 A tree in shared memory

### 15.3.1 What the threads of a block need

Each addition in the tree combines 2 values that 2 different threads produced in the step before.
The threads must therefore exchange values, and they must know when a value is ready. A block
provides both (Definition 14.4): a memory that all its threads can read and write, and a way to
wait for one another.

<div class="definition">

<span class="box-label">Definition 15.4</span> **Shared memory.** Shared memory is a small memory
on each SM, typically 48 KB to 228 KB, that the threads of 1 block can all read and write and no
other thread can access. It responds within a few cycles. Each block receives its own copy of
every shared array, which exists from the start of the block to its end and starts with undefined
contents.

*Example.* Block 0 of our example holds its 8 products in a shared array `cells` of 8 floats;
block 1 has its own `cells`.

</div>

<div class="definition">

<span class="box-label">Definition 15.5</span> **Barrier.** A barrier is a point in a kernel that
no thread of a block may pass until every thread of the block has reached it. All writes to shared
memory made before the barrier are visible to every thread of the block after it. In CUDA C++ the
barrier is `__syncthreads()`, in cuda-oxide `thread::sync_threads()`.

*Example.* After a barrier, thread 0 of block 0 can read the product that thread 4 stored before
the barrier.

</div>

### 15.3.2 Which cells a thread adds

The tree of Figure 15.3 adds neighbours: cells 0 and 1, cells 2 and 3, and so on. On a GPU we
prefer another pairing. In the step with **stride** \\(s\\), thread \\(t\\) adds cell \\(t + s\\)
into cell \\(t\\), for every \\(t < s\\). With 8 cells the strides are 4, 2 and 1. In the first
step threads 0 to 3 add cells 4 to 7 into cells 0 to 3; in the second, threads 0 and 1 add cells 2
and 3 into cells 0 and 1; in the third, thread 0 adds cell 1 into cell 0.

The stride pairing adds \\(p_0 + p_4\\) where Figure 15.3 added \\(p_0 + p_1\\), so it relies on
commutativity as well as associativity. In return, the threads that work in each step are always
the lowest \\(s\\) threads, 0 to \\(s - 1\\), and they read the neighbouring cells \\(s\\) to
\\(2s - 1\\). In a block of 256 threads the first steps keep whole warps busy and whole warps idle,
instead of leaving half the threads of every warp idle, and the cells that a warp reads at once
are neighbours.

Figure 15.4 runs this pairing in block 0 of the example. The 8 cells start with the products 2,
6, 3, 4, 5, 1, 8 and 3. With \\(s = 4\\), threads 0 to 3 compute \\(2 + 5 = 7\\),
\\(6 + 1 = 7\\), \\(3 + 8 = 11\\) and \\(4 + 3 = 7\\). With \\(s = 2\\), threads 0 and 1 compute
\\(7 + 11 = 18\\) and \\(7 + 7 = 14\\). With \\(s = 1\\), thread 0 computes \\(18 + 14 = 32\\).
Block 1 computes 4, 11, 6, 10, then 10, 21, then 31 in the same way.

<figure>
<img src="../../figures/ch15-steps.svg" alt="The tree in the shared memory of block 0. The 8 cells start as 2, 6, 3, 4, 5, 1, 8, 3. With s = 4, threads 0 to 3 add cell t + 4 into cell t, giving 7, 7, 11, 7. With s = 2, threads 0 and 1 give 18 and 14. With s = 1, thread 0 gives 32. A barrier separates the steps; cells no longer read are grey.">
<figcaption><b>Figure 15.4</b> The tree in the shared memory of block 0. Each colour is 1 thread; a barrier separates the steps.</figcaption>
</figure>

### 15.3.3 The algorithm

A real block has \\(B = 256\\) threads, and a real vector has millions of elements, many more than
the threads of the grid. Each thread therefore starts with the grid-stride loop of Chapter 14: it
multiplies and adds the elements \\(g, g + P, g + 2P, \ldots\\) in a register, where \\(g\\) is its
global index and \\(P\\) the number of threads in the grid. The products are never stored in
memory. The tree then combines the \\(B\\) register sums of the block, and thread 0 writes the
block's sum to a **partial sum** array, 1 element per block. Every thread executes the following
procedure; `cells` is a shared array of \\(B\\) floats.

<div class="algo">
<p class="algo-name">Dot-Block<span style="font-variant: normal">(\(a, b, \mathit{partial}, n\))</span></p>
<ol>
<li>\(t = \textit{threadIdx.x}\), \(\quad i = \textit{blockIdx.x} \cdot B + t\), \(\quad P = \textit{gridDim.x} \cdot B\)</li>
<li>\(v = 0\)</li>
<li>while \(i \lt n\) { <span class="c">// the grid-stride loop of Chapter 14</span></li>
<li style="--d:1">\(v = v + a_i \cdot b_i\)</li>
<li style="--d:1">\(i = i + P\)</li>
<li>}</li>
<li>\(\mathit{cells}[t] = v\)</li>
<li><span class="sc">Barrier</span>() <span class="c">// every cell holds its value</span></li>
<li>for \(s = B/2, B/4, \ldots, 1\) {</li>
<li style="--d:1">if \(t \lt s\) {</li>
<li style="--d:2">\(\mathit{cells}[t] = \mathit{cells}[t] + \mathit{cells}[t + s]\)</li>
<li style="--d:1">}</li>
<li style="--d:1"><span class="sc">Barrier</span>() <span class="c">// step \(s\) is complete</span></li>
<li>}</li>
<li>if \(t = 0\) {</li>
<li style="--d:1">\(\mathit{partial}[\textit{blockIdx.x}] = \mathit{cells}[0]\)</li>
<li>}</li>
</ol>
</div>

Lines 1 to 6 compute the thread's register sum; on the example, launched with 2 blocks of 8
threads, \\(P = 16 = n\\), so the loop runs once and \\(v\\) is the thread's single product. Line 7
stores the sum in the thread's own cell, and the barrier of line 8 waits until all \\(B\\) cells
are filled. Lines 9 to 14 are the tree of Figure 15.4, with a barrier after every step, and lines
15 to 17 write the block's sum. On the example, block 0 writes 32 into `partial[0]` and block 1
writes 31 into `partial[1]`. We add the partial sums in Section 15.6.

Lines 8 and 13 lie outside every `if`. The barrier of line 13 is executed by all \\(B\\) threads in
every step, including the threads with \\(t \ge s\\) that have nothing to add.

### 15.3.4 Counting

A block of \\(B\\) threads performs \\(B - 1\\) additions in its tree, in \\(\log_2 B\\) steps,
with \\(\log_2 B + 1\\) barriers. For \\(B = 256\\) that is 255 additions in 8 steps and 9
barriers. The number of working threads halves in every step, 128, 64, and so on to 1, so in the
last 5 steps only warp 0 works, and the other 7 warps wait at the barrier.

The tree's cost is paid once per block, not once per element. With a grid of 1,024 blocks of 256
threads, \\(P = 262{,}144\\), and each thread adds \\(2^{24} / 262{,}144 = 64\\) products in lines
3 to 6 before the tree begins. The kernel reads every element of \\(a\\) and \\(b\\) exactly once,
8 bytes per element, and performs 2 FLOP per element: 0.25 FLOP/B, far below the ridge point of
any GPU (Section 14.6). Like SAXPY, the dot product is memory-bound.

<div class="exercises">

Exercises

15.7 Run Figure 15.4 for block 1, whose cells start with 3, 2, 4, 6, 1, 9, 2, 4. Which threads
work in each step, and what does each compute?<br>
15.8 For \\(B = 512\\), how many steps, additions and barriers does the tree take?<br>
15.9 With 1,024 blocks of 256 threads and \\(n = 10^6\\), how many products does thread 0 add in
lines 3 to 6, and how many does thread 262,143 add?

</div>

## 15.4 Correctness

The threads of a block run at different speeds, and they read cells that other threads write. 2
things can go wrong: a thread can read a cell before the thread responsible for it has written
the value of the current step, and a thread can overwrite a cell that another thread has not yet
read. The 2 barriers prevent both. We remove the barrier of line 13 and watch the example.

### 15.4.1 Without the barrier between the steps

Suppose line 13 is deleted, and in block 0 thread 2 runs late. Thread 0 finishes its addition for
\\(s = 4\\), \\(2 + 5 = 7\\), and moves on to \\(s = 2\\), where it needs cell 2. Thread 2 should
have stored \\(3 + 8 = 11\\) there, but it has not yet done so, and cell 2 still holds the product
3 (Figure 15.5). Thread 0 computes \\(7 + 3 = 10\\) in place of 18, and in the last step
\\(10 + 14 = 24\\) in place of 32.

<figure>
<img src="../../figures/ch15-broken.svg" alt="Without the barrier after step s = 4. Thread 0 finishes s = 4 early and in step s = 2 reads cell 2 while it still holds 3, because thread 2 has not yet stored 3 + 8 = 11. Cell 0 becomes 7 + 3 = 10 in place of 18, and the block sum 10 + 14 = 24 in place of 32.">
<figcaption><b>Figure 15.5</b> Without the barrier of line 13, thread 0 reads cell 2 before thread 2 has updated it. Block 0 sums to 24 in place of 32.</figcaption>
</figure>

We deleted the barrier from both programs and ran them. The C++ build also contained the faulty
plain addition of Section 15.6.2, which affects only the last column. It printed

```console
exact tests                   map+red      tree   shuffle    atomic
  worked example, 16 pairs          63        26        63        63   expected 63
  0s and 1s, 16777216 pairs    2796203    760424   2796203    338598   expected 2796203
```

and the Rust program 36 for the worked example and 778,180 for the second test. On \\(2^{24}\\)
floats in \\([0, 1)\\) the tree returned 920,497.0 (C++) and 1,159,452.9 (Rust) in place of
4,195,680.0, and a second run gave different wrong values, 756,503 and 778,042 for the second test.
The program runs the example on the real block of 256 threads, so the precise wrong value differs
from the 24 of Figure 15.5. Without the barrier, nothing orders the store of 1 step before the
loads of the next: the warps of a block reach each step at different times, and the compiler may
also load a cell earlier than the program text suggests, because it assumes that no other thread
changes the cell in between. Every wrong value has the same cause, the one of Figure 15.5.

The barrier of line 8 protects the first step in the same way: without it, thread 0 could add cell
4 before thread 4 has stored its sum there, and would add whatever the undefined shared memory
held.

### 15.4.2 A proof that the tree is correct

With both barriers in place, the tree computes the block's sum. The idea of the proof is to watch
which products each cell holds at the start of every step (Figure 15.6). At the start of the step
with stride \\(s = 4\\), cell 0 holds \\(p_0\\); at the start of \\(s = 2\\), it holds
\\(p_0 + p_4\\); at the start of \\(s = 1\\), \\(p_0 + p_2 + p_4 + p_6\\); and after the last step,
all 8 products. The indices that cell \\(k\\) has collected are those that leave the remainder
\\(k\\) when divided by \\(2s\\).

<figure>
<img src="../../figures/ch15-invariant.svg" alt="The loop invariant for cell 0 of block 0. At the start of step s = 4 it holds p0 = 2; of s = 2, p0 + p4 = 7; of s = 1, p0 + p2 + p4 + p6 = 18; after s = 1, all 8 products, 32.">
<figcaption><b>Figure 15.6</b> The loop invariant for cell 0 of block 0: at the start of the step with stride \(s\), it holds the products with index \(j \equiv 0 \pmod{2s}\).</figcaption>
</figure>

<div class="theorem">

<span class="box-label">Theorem 15.1</span> Let \\(B\\) be a power of 2 and let \\(v_j\\) be the
value that thread \\(j\\) of a block stores in line 7 of <span class="sc">Dot-Block</span>. When
the loop of lines 9 to 14 finishes, \\(\mathit{cells}[0] = \sum_{j \in [0, B)} v_j\\).

</div>

<div class="proof">

*Proof.* We show that the loop maintains the invariant: *at the start of the iteration with stride
\\(s\\), for every \\(k \in [0, 2s)\\),*

\\[ \mathit{cells}[k] = \sum_{j \in [0, B),\ j \equiv k \ (\mathrm{mod}\ 2s)} v_j . \\]

*Initialization.* The first iteration has \\(s = B/2\\), so \\(2s = B\\), and the only
\\(j \in [0, B)\\) with \\(j \equiv k \pmod{B}\\) is \\(k\\) itself. Line 7 stored \\(v_k\\) in
\\(\mathit{cells}[k]\\), and the barrier of line 8 makes every one of these stores complete before
any thread reads a cell in line 11. The invariant holds.

*Maintenance.* Assume the invariant at the start of the iteration with stride \\(s\\). In line 11,
each thread \\(t < s\\) reads cells \\(t\\) and \\(t + s\\) and writes cell \\(t\\). The cells
written, \\([0, s)\\), and the cells only read, \\([s, 2s)\\), are disjoint, and each cell of
\\([0, s)\\) is written by exactly 1 thread, so no read in this iteration sees a write of this
iteration except the thread's own. After line 11, for \\(t < s\\), cell \\(t\\) holds the sum over
\\(j \equiv t \pmod{2s}\\) plus the sum over \\(j \equiv t + s \pmod{2s}\\), which is the sum over
\\(j \equiv t \pmod{s}\\). The barrier of line 13 makes these writes complete before any thread
starts the next iteration, whose stride is \\(s/2\\), so \\(2 \cdot (s/2) = s\\): the invariant
holds for the next iteration.

*Termination.* After the iteration with \\(s = 1\\), the maintenance step shows that cell 0 holds
the sum over \\(j \equiv 0 \pmod{1}\\), that is, over all \\(j \in [0, B)\\).

</div>

The proof uses each barrier once: line 8 in the initialization and line 13 in the maintenance.
Removing either breaks exactly the step of the proof that uses it, and in Section 15.4.1 we saw
the wrong value that follows.

<div class="rule-box">

<span class="box-label">Rule</span> Every thread of a block must reach every barrier. Never place
a barrier inside a branch that only some threads of the block take, such as the `if` of line 10.

</div>

A barrier inside `if t < s` would be reached by \\(s\\) threads and awaited by \\(B\\); the
behaviour of such a kernel is undefined, and in practice it hangs or returns wrong values.

### 15.4.3 Shared memory in Rust

In cuda-oxide a shared array is a `static mut` of type `SharedArray<f32, 256>` declared inside the
kernel. Rust requires an `unsafe` block for every access to a `static mut`, because the compiler
cannot prove that 2 threads do not access it at the same time. Writing the store of line 7 without
the block stops the build:

```console
$ cargo oxide build
error[E0133]: use of mutable static is unsafe and requires unsafe block
   --> src/main.rs:117:9
    |
117 |         CELLS[t] = products(a, b);
    |         ^^^^^ use of mutable static
    |
    = note: mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior
```

The program's `// SAFETY:` comments state the 2 facts that the proof of Theorem 15.1 uses: each
thread writes only its own cell before the barrier of line 8, and in each step thread \\(t\\)
alone writes cell \\(t\\) and reads cell \\(t + s\\), which no thread writes in that step. The
compiler accepts the program without the barrier as well, because the `unsafe` blocks are still
there; only the argument in their comments is then false.

<div class="exercises">

Exercises

15.10 Without the barrier of line 13, suppose thread 1 of block 1 runs late in step \\(s = 4\\).
Which cell does thread 0 read too early in step \\(s = 2\\), and what block sum results?<br>
15.11 In the proof of Theorem 15.1, where is it used that \\(B\\) is a power of 2? What goes wrong
for \\(B = 6\\)?<br>
15.12 Explain why moving the barrier of line 13 inside the `if` of line 10 can make the kernel
hang.

</div>

## 15.5 Warp shuffles

### 15.5.1 Exchanging registers within a warp

The last 5 steps of the tree in a block of 256 threads run with 16, 8, 4, 2 and 1 working
threads, all in warp 0, and each still ends with a barrier for the whole block. The threads of 1
warp, however, can exchange values without shared memory and without a barrier.

<div class="definition">

<span class="box-label">Definition 15.6</span> **Warp shuffle.** A warp shuffle is an instruction
with which every lane of a warp (Definition 14.6) reads a register of another lane of the same
warp. In the form *shuffle down* with offset \\(d\\), lane \\(k\\) receives the value that lane
\\(k + d\\) holds; a lane with \\(k + d \ge 32\\) receives its own value. All lanes named in the
instruction's mask execute it together, so the values are exchanged without a barrier.

*Example.* With offset 4, lane 0 receives the value of lane 4, and lane 3 the value of lane 7.

</div>

A shuffle-down with offsets 16, 8, 4, 2 and 1, each followed by an addition, is the tree of
Section 15.3 inside 1 warp, with registers in place of cells. Figure 15.7 draws it on 8 lanes
with the products of block 0, which makes it the last 3 of the 5 steps: the registers take the
same values as the cells of Figure 15.4, 7, 7, 11, 7, then 18, 14, then 32.

<figure>
<img src="../../figures/ch15-shuffle.svg" alt="A warp sum by shuffles, drawn on 8 lanes. The registers start as 2, 6, 3, 4, 5, 1, 8, 3. With offset 4, lanes 0 to 3 add the values of lanes 4 to 7: 7, 7, 11, 7. With offset 2: 18, 14. With offset 1: lane 0 holds 32. The same numbers as the tree in shared memory, with no shared memory and no barrier.">
<figcaption><b>Figure 15.7</b> A warp sum by shuffles, drawn on 8 lanes with the products of block 0. The values are those of Figure 15.4.</figcaption>
</figure>

<div class="algo">
<p class="algo-name">Warp-Sum<span style="font-variant: normal">(\(v\))</span></p>
<ol>
<li>for \(d = 16, 8, 4, 2, 1\) {</li>
<li style="--d:1">\(v = v + \) <span class="sc">Shuffle-Down</span>\((v, d)\) <span class="c">// lane \(k\) adds lane \(k + d\)</span></li>
<li>}</li>
<li>return \(v\) <span class="c">// the sum of the warp, in lane 0</span></li>
</ol>
</div>

### 15.5.2 A block sum with 1 barrier

A block of 256 threads is 8 warps. Each warp first reduces its 32 values with
<span class="sc">Warp-Sum</span>, and lane 0 of each warp stores the warp's sum in 1 of 8 shared
slots. 1 barrier makes the 8 sums visible, and warp 0 reduces them with a second
<span class="sc">Warp-Sum</span> (Figure 15.8).

<div class="algo">
<p class="algo-name">Block-Sum<span style="font-variant: normal">(\(v\))</span></p>
<ol>
<li>\(\mathit{lane} = \textit{threadIdx.x} \bmod 32\), \(\quad w = \lfloor \textit{threadIdx.x} / 32 \rfloor\)</li>
<li>\(v = \) <span class="sc">Warp-Sum</span>\((v)\)</li>
<li>if \(\mathit{lane} = 0\) { \(\mathit{warps}[w] = v\) }</li>
<li><span class="sc">Barrier</span>()</li>
<li>\(u = \mathit{warps}[\mathit{lane}]\) if \(\mathit{lane} \lt 8\), else \(0\)</li>
<li>return <span class="sc">Warp-Sum</span>\((u)\) if \(w = 0\), else \(0\) <span class="c">// the block sum, in thread 0</span></li>
</ol>
</div>

<figure>
<img src="../../figures/ch15-blocksum.svg" alt="The block sum with shuffles. Each of the 8 warps of a 256-thread block sums its 32 values with 5 shuffles, and lane 0 stores the warp sum in w[0] to w[7] in shared memory. After 1 barrier, warp 0 reads the 8 values and sums them with 5 more shuffles; thread 0 holds the block sum.">
<figcaption><b>Figure 15.8</b> The block sum: 5 shuffles in every warp, 1 barrier, 5 shuffles in warp 0.</figcaption>
</figure>

In line 5, lanes 8 to 31 of warp 0 contribute 0, the identity of addition, so the second
<span class="sc">Warp-Sum</span> adds exactly the 8 warp sums. Only warp 0 calls it the second
time, and all 32 of its lanes do, as the shuffle requires. Compared with the tree of Section 15.3,
<span class="sc">Block-Sum</span> uses 1 barrier instead of 9 and 8 floats of shared memory
instead of 256. A fused kernel with shuffles computes the register sum \\(v\\) with lines 1 to 6
of <span class="sc">Dot-Block</span> and replaces lines 7 to 14 by \\(v = \\)
<span class="sc">Block-Sum</span>\\((v)\\).

<div class="exercises">

Exercises

15.13 In <span class="sc">Warp-Sum</span> on 32 lanes holding the values 0 to 31, what does lane
0 hold after each of the 5 steps?<br>
15.14 Why may lines 5 and 6 of <span class="sc">Block-Sum</span> not read `warps` before the
barrier of line 4?

</div>

## 15.6 Combining the block results

### 15.6.1 A second launch

After <span class="sc">Dot-Block</span>, `partial[0] = 32` and `partial[1] = 31`. The 2 numbers
lie in different blocks, and the threads of different blocks can share neither a barrier nor
shared memory: the blocks of a large grid do not even run at the same time. The end of a kernel,
however, is a point that every block has passed. A second kernel launched on the same stream
starts only after the first has finished, so it sees every partial sum. We launch the sum kernel
again with 1 block, which adds the partial sums with the same grid-stride loop and block sum. On
the example it computes \\(32 + 31 = 63\\) (Figure 15.9, left).

### 15.6.2 Atomic additions

The second launch can be avoided if every block adds its sum directly into the result. That
requires an addition that no other thread can interrupt.

<div class="definition">

<span class="box-label">Definition 15.7</span> **Atomic operation.** An atomic operation is a
read-modify-write of 1 memory location that executes as 1 indivisible step: no other thread can
access the location between the read and the write. `atomicAdd(p, v)` in C++ and
`fetch_add(v, ...)` on a `DeviceAtomicF32` in Rust add \\(v\\) to the value at \\(p\\) atomically.

*Example.* If block 1 finishes first, it atomically changes the result from 0 to 31, and block 0
then changes it from 31 to 63.

</div>

Figure 15.9 compares the 2 methods. With the second launch, the 2 partial sums are added in a
fixed order by the program. With atomic additions, they are added in the order in which the
blocks finish, which changes from run to run.

<figure>
<img src="../../figures/ch15-combine.svg" alt="2 ways to combine the block sums 32 and 31. Left: each block writes its partial sum, and a second kernel, which starts after every block of the first is done, adds them in a fixed order: 63. Right: each block adds its sum into result with an atomic addition, in the order the blocks finish, here block 1 then block 0: 0, 31, 63.">
<figcaption><b>Figure 15.9</b> Left: a second launch adds the partial sums in a fixed order. Right: atomic additions add them in the order the blocks finish.</figcaption>
</figure>

The addition must be atomic. With an ordinary `*result += total`, a block reads the result, adds
its sum and writes it back, in 3 separate steps. If block 0 and block 1 both read 0 before either
writes, block 0 writes 32 and block 1 writes 31, and the later write wins: the result is 31 in
place of 63 (Figure 15.10).

<figure>
<img src="../../figures/ch15-race.svg" alt="A plain addition in place of an atomic one. Block 0 reads result = 0, then block 1 reads 0. Block 0 writes 32, and block 1 later writes 0 + 31. The result is 31 in place of 63.">
<figcaption><b>Figure 15.10</b> With a plain addition, both blocks read 0 and the later write wins: 31 in place of 63.</figcaption>
</figure>

We replaced `atomicAdd(result, total)` by `*result += total` in the C++ program. The 0-and-1 test
returned 338,598 in place of 2,796,203, and the dot product of the random floats returned 100
different values in 100 runs, from 479,931.1 to 528,424.6, about 12% of the correct 4,195,680.
The worked example, launched with 1,024 blocks of which only block 0 has a nonzero sum, returned
the correct 63 in 1 run and 0 in another: a block with the sum 0 read the result before block 0
wrote 63 and wrote its 0 afterwards.

### 15.6.3 The order of float additions

The 2 correct methods add the same numbers in different orders. For integers the order does not
matter. For floats it does, because every addition rounds its result to the nearest float.

A float has 24 significant bits. Below \\(2^{24} = 16{,}777{,}216\\) every integer is a float;
above it, the floats are 2 apart. Figure 15.11 adds 16,777,216, 1 and 1 in 2 orders. From left to
right, \\(16{,}777{,}216 + 1 = 16{,}777{,}217\\) is not a float and rounds back to 16,777,216, and
the second \\(+ 1\\) is lost in the same way. With the small values first, \\(1 + 1 = 2\\) and
\\(16{,}777{,}216 + 2 = 16{,}777{,}218\\), exactly.

<figure>
<img src="../../figures/ch15-order.svg" alt="The same 3 floats, 16,777,216, 1 and 1, added in 2 orders. Left to right, each + 1 gives 16,777,217, which is not a float and rounds back to 16,777,216. With the small values first, 1 + 1 = 2 and 16,777,216 + 2 = 16,777,218 exactly.">
<figcaption><b>Figure 15.11</b> The same 3 floats added in 2 orders give 2 different sums.</figcaption>
</figure>

A second launch fixes the order of every addition: each thread adds its elements in the order of
its loop, the shuffles combine fixed pairs of lanes, and the last block adds the partial sums in a
fixed pattern. The result is the same, bit for bit, in every run with the same grid. With atomic
additions, the 1,024 block sums arrive in an order that depends on timing, and the last bits of
the result can change from run to run. Both results are correct to the precision of a float.

The program uses this to verify itself exactly, as is done with small integers in the paper on
tiled matrix multiplication. Its second test fills \\(a\\) and \\(b\\) with 0s and 1s, so that
every product, every partial sum and the total are integers no larger than \\(2^{24}\\), all of
which are floats. No addition then rounds, every order gives the same result, and every method must
return exactly 2,796,203, the number of positions divisible by 6.

<div class="exercises">

Exercises

15.15 With 4 blocks whose sums are 10, 20, 30 and 40, list 2 orders of atomic additions and the
intermediate values of the result for each.<br>
15.16 Why does the 0-and-1 test stay exact with \\(n = 2^{24}\\), but not necessarily with
\\(n = 2^{25}\\)?<br>
15.17 In the faulty plain addition, what is the smallest wrong result the worked example can
produce, and what is the largest?

</div>

## 15.7 Implementation

### 15.7.1 The register sum

The grid-stride loop of lines 1 to 6 is a device function, a function that runs on the GPU and is
called from kernels. C++ marks it `__device__`, Rust `#[device]`.

<p class="listing"><b>Listing 15.1</b> The register sum of 1 thread: lines 1 to 6 of <span class="sc">Dot-Block</span>.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch15_dot_product/src/main.rs:63:75}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch15_dot_product/dot.cu:57:67}}
```

</div>
</div>

The loop reads `a[i]` and `b[i]` and adds the product to `sum` without storing it. In Rust the
slices are indexed with an ordinary `usize`; the loop condition `i < a.len()` lets the compiler
drop the check on `a`, and the check on `b` costs 1 comparison per 8 bytes loaded, which does not
change the speed of a memory-bound loop. The Rust function computes \\(P\\) from the constant
`BLOCK`, which the launch contract of every kernel that calls it fixes at 256.

### 15.7.2 The tree kernel

<p class="listing"><b>Listing 15.2</b> The fused kernel with the tree in shared memory: <span class="sc">Dot-Block</span>.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch15_dot_product/src/main.rs:109:137}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch15_dot_product/dot.cu:95:114}}
```

</div>
</div>

The call `products(a, b)` is lines 1 to 6, the store into `cells[t]` or `CELLS[t]` is line 7, and
the first barrier is line 8. The loop is lines 9 to 14, with the barrier of line 13 after the `if`,
and the final `if t == 0` is lines 15 to 17.

The 2 languages differ in 3 places. First, C++ declares the shared array with `__shared__`; Rust
declares a `static mut SharedArray<f32, 256>`, whose length is part of its type, which is why the
Rust kernel's launch contract fixes the block at exactly 256 threads. Every access is in an
`unsafe` block with its argument (Section 15.4.3). Second, the C++ loop halves `s` directly. The
Rust loop counts `step` from 0 to 7 and computes `s = 128 >> step`, because cuda-oxide unrolls a
loop marked `#[unroll]` only when it has an integer counter that is compared with a constant and
increased by a constant; with `s /= 2` the compiler reports that the loop was not unrolled.
Third, thread 0 writes `partial[blockIdx.x]`, an index that is not its own global index, so Rust
uses `get_unchecked_mut` with the argument that only thread 0 of each block writes, at its own
block number.

### 15.7.3 The warp and block sums

<p class="listing"><b>Listing 15.3</b> <span class="sc">Warp-Sum</span> and <span class="sc">Block-Sum</span>.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch15_dot_product/src/main.rs:31:61}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch15_dot_product/dot.cu:31:55}}
```

</div>
</div>

`__shfl_down_sync(0xffffffff, v, offset)` and `warp::shuffle_down_f32(v, offset)` are the
shuffle of Definition 15.6; the mask `0xffffffff` names all 32 lanes, and the Rust function
applies the same mask. The Rust counter is declared `u32` and the shifted constant written `16u32`:
with both types left to inference, the GPU backend of cuda-oxide stopped with an internal error
about a shift of a 64-bit value by a 32-bit amount. The shared array `WARPS` is declared inside the
device function; every block that calls `block_sum` still receives its own 8 slots.

### 15.7.4 Combining the block sums

<p class="listing"><b>Listing 15.4</b> The fused kernel with an atomic addition per block.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch15_dot_product/src/main.rs:169:178}}
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch15_dot_product/dot.cu:126:134}}
```

</div>
</div>

In Rust the result is a slice of `DeviceAtomicF32`, received as a shared reference `&[...]`, not a
mutable one. The atomic type permits modification through a shared reference, like the atomic
types of the Rust standard library, because each modification is 1 indivisible operation; the
kernel therefore needs no `unsafe`. `AtomicOrdering::Relaxed` asks for atomicity and nothing more,
which suffices, because the host reads the result only after the kernel has finished. A device
buffer of `DeviceAtomicF32` cannot be copied with `to_host_vec`, so the Rust program uses 2 small
kernels, `clear` and `publish`, to set it to 0 before the launch and to copy it into an ordinary
float afterwards. C++ sets the result to 0 with `cudaMemsetAsync`.

The host prepares every launch once and runs the 4 designs with 1 closure in Rust and 1 function
in C++. Listing 15.5 shows the launches: 3 for the map followed by a reduction, 2 for the fused
kernels with a second launch, and 3 (Rust) or 2 (C++) for the atomic design.

<p class="listing"><b>Listing 15.5</b> The launches of the 4 designs.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn main() -> Result<(), Error> {
    // ...
{{#include ../../../../code/oxide/ch15_dot_product/src/main.rs:306:343}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch15_dot_product/dot.cu:183:209}}
```

</div>
</div>

<div class="exercises">

Exercises

15.18 Write a fused kernel that combines 1 atomic addition per warp instead of per block. How many
atomic operations does 1 run perform with 1,024 blocks of 256 threads?<br>
15.19 Reuse `block_sum` to compute \\(\sum a_i^2\\), the squared length of a vector. How many bytes
per element does it read?

</div>

## 15.8 Measurements

### 15.8.1 Platform and method

The measurements come from an integrated GPU, a Jetson Orin NX 16 GB (8 SMs, 102 GB/s).
Each program first runs the 2 exact tests of Section 15.6.3, then computes the dot product
of \\(2^{24}\\) pairs of floats in \\([0, 1)\\) from a fixed generator, 64 MiB per vector, with
each of the 4 designs. Every design runs 10 times as a warm-up and 20 times under CUDA events. The
rate is the number of bytes the design must move, 16 or 8 per pair, divided by its time. The
program then runs the second-launch and atomic designs 100 times each and counts the distinct
results. Repeated runs differed by up to 9%.

### 15.8.2 Results

```console
$ cargo oxide run
GPU: Orin, 8 SMs
exact tests                   map+red      tree   shuffle    atomic
  worked example, 16 pairs          63        63        63        63   expected 63
  0s and 1s, 16777216 pairs    2796203   2796203   2796203   2796203   expected 2796203
16777216 pairs of floats in [0, 1), 20 runs
  method                    time   bytes/pair      rate      result
  map, then reduce        3.334 ms       16 B    80.5 GB/s  4195680.0
  fused, shared tree      1.409 ms        8 B    95.2 GB/s  4195680.0
  fused, shuffles         1.405 ms        8 B    95.5 GB/s  4195680.0
  fused, atomic combine   1.405 ms        8 B    95.5 GB/s  4195681.0
  exact, in double precision                             4195680.2
distinct results in 100 runs
  second launch    1  from 4195680.0 to 4195680.0
  atomic           3  from 4195680.5 to 4195681.5
largest relative error 3.20e-7
PASSED
```

Every design returns 63 for the worked example and exactly 2,796,203 for the 0-and-1 test. Table
15.1 collects the times of both languages.

| Design | Bytes per pair | Rust time | Rust rate | C++ time | C++ rate |
|---|---|---|---|---|---|
| map, then reduce | 16 | 3.334 ms | 80.5 GB/s | 3.096 ms | 86.7 GB/s |
| fused, tree in shared memory | 8 | 1.409 ms | 95.2 GB/s | 1.412 ms | 95.1 GB/s |
| fused, warp shuffles | 8 | 1.405 ms | 95.5 GB/s | 1.456 ms | 92.2 GB/s |
| fused, atomic combine | 8 | 1.405 ms | 95.5 GB/s | 1.389 ms | 96.6 GB/s |

<p class="listing"><b>Table 15.1</b> The dot product of \(2^{24}\) pairs on the measured GPU. Least times at 102 GB/s: 2.63 ms for 16 bytes per pair, 1.32 ms for 8.</p>

<figure>
<img src="../../figures/ch15-measured.svg" alt="Measured rates of the 4 designs on the measured GPU, Rust and C++, against 102 GB/s.">
<figcaption><b>Figure 15.12</b> Measured rates of the 4 designs against the memory bandwidth of 102 GB/s.</figcaption>
</figure>

### 15.8.3 Why the fused kernel is more than twice as fast

The design that stores the products reads \\(a_i\\) and \\(b_i\\), writes \\(p_i\\), and reads
\\(p_i\\) back in the reduction: 16 bytes per pair, 268 MB for \\(2^{24}\\) pairs. The fused
kernels read only \\(a_i\\) and \\(b_i\\): 8 bytes per pair, 134 MB (Figure 15.13). At 102 GB/s
these amounts take at least 2.63 ms and 1.32 ms, a ratio of 2.

<figure>
<img src="../../figures/ch15-traffic.svg" alt="Bytes moved per pair. Map then reduce reads a[i] and b[i], writes p[i] and reads it back: 16 bytes. The fused kernel reads a[i] and b[i] and adds the product to a register: 8 bytes. For 16,777,216 pairs, 268 MB against 134 MB, 2.63 ms against 1.32 ms at 102 GB/s.">
<figcaption><b>Figure 15.13</b> Bytes moved per pair: 16 when the products are stored, 8 when they stay in registers.</figcaption>
</figure>

The measured ratio is \\(3.334 / 1.405 = 2.4\\) in Rust and \\(3.096 / 1.412 = 2.2\\) in C++.
Halving the bytes accounts for a factor of 2. The rest comes from the lower rate of the first
design, 80.5 GB/s and 86.7 GB/s against 95 GB/s: its multiplication kernel writes 1 value for
every 2 it reads, like SAXPY, and SAXPY reached at most 93 GB/s in Chapter 14, for the reasons
given in Section 14.8.3. The fused kernels only read, and reach 92 GB/s to 97 GB/s, 90% to 95% of
the pin rate of the memory. The Rust map-then-reduce design ran 8% to 11% slower than the C++ one in
every run, while the fused kernels ran at the same speed in both languages.

### 15.8.4 Barriers, shuffles and atomics

The tree with 9 barriers and the warp shuffles with 1 run at the same speed, 1.409 ms and 1.405 ms
in Rust. Each thread adds 64 products in its register before the block sum begins, so a block pays
for its 9 barriers once per \\(64 \cdot 256 = 16{,}384\\) products, a small part of its time.
In Chapter 8, we found the same for the plain sum: after coarsening, its tree and its shuffle sum
ran at 64.1 GB/s and 64.4 GB/s. Shuffles matter where a combination is frequent, as in the many short
reductions of a softmax over the rows of a matrix (Chapter 23).

The atomic design is as fast as the others: its 1,024 atomic additions are few, and its extra
launches, `clear` and `publish`, each run 1 block. Its results differ. The exact dot product,
computed in double precision on the CPU, is 4,195,680.2. Between \\(2^{22} = 4{,}194{,}304\\) and
\\(2^{23}\\) the floats are 0.5 apart, so the nearest float is 4,195,680.0. The second-launch
designs return exactly that value, and the 100 runs of the shuffle design returned the same bits
every time. The atomic design returned 3 different values in 100 runs, 4,195,680.5, 4,195,681.0
and 4,195,681.5, 1 to 3 float steps above the nearest float, and in other runs also 4,195,680.0.
All are correct to a relative error of \\(3.2 \cdot 10^{-7}\\), about the precision of a float. A
program that must reproduce its results bit for bit, for example to compare 2 versions of a
simulation, needs the second launch.

<div class="exercises">

Exercises

15.20 From Table 15.1, compute the fraction of the 102 GB/s that each fused design reaches in each
language.<br>
15.21 Predict the time of the fused dot product of \\(2^{26}\\) pairs on the measured GPU, and on a GPU
with 3,350 GB/s.

</div>

## 15.9 The complete programs

<p class="listing"><b>Listing 15.6</b> The complete programs. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch15_dot_product/src/main.rs">code/oxide/ch15_dot_product/src/main.rs</a> <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/cuda/ch15_dot_product/dot.cu">code/cuda/ch15_dot_product/dot.cu</a></p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch15_dot_product/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch15_dot_product/dot.cu}}
```

</div>
</div>

## Common mistakes

**Storing the products of a map that a reduction consumes.** The array of products doubles the
memory traffic of a memory-bound computation. Fuse the map into the reduction and keep the
products in a register.

**A missing barrier between the steps of the tree.** The kernel compiles in both languages and
returns a wrong sum that changes from run to run (Section 15.4.1).

**A barrier inside a branch that only some threads take.** Every thread of the block must reach
every `__syncthreads()` or `thread::sync_threads()`.

**A plain addition where an atomic one is needed.** `*result += total` from many blocks loses most
of the additions (Section 15.6.2).

**Expecting identical bits from atomic float additions.** The order of the additions changes from
run to run, and so can the last bits of the result. Use a second launch when results must repeat
exactly.

**An `unsafe` block without its argument.** Each access to a shared array needs a stated reason
why no other thread writes the same element at the same time, and the reason usually names a
barrier.

## Summary

- The dot product is a map, \\(p_i = a_i b_i\\), followed by a reduction, the sum. On the example
  the 16 products add up to \\(32 + 31 = 63\\).
- The sequential sum is a chain of \\(n\\) dependent additions. Because addition is associative
  and commutative, the additions can be regrouped into a tree of \\(\log_2 n\\) steps, 4 for the
  example.
- In shared memory, the step with stride \\(s\\) lets threads \\(t < s\\) add cell \\(t + s\\) into
  cell \\(t\\); block 0 of the example computes 7, 7, 11, 7, then 18, 14, then 32. A barrier must
  follow every step: without it the example's block summed to 24 in place of 32, and the real
  programs returned 26 and 36 in place of 63.
- Theorem 15.1 proves the tree correct with the invariant that, at stride \\(s\\), cell \\(k\\)
  holds the values with index \\(j \equiv k \pmod{2s}\\).
- Warp shuffles exchange registers within a warp without a barrier; a block sum with shuffles
  needs 1 barrier and 8 shared floats instead of 9 and 256.
- Block sums are combined by a second launch, in a fixed order, or by atomic additions, in the
  order the blocks finish. A plain addition loses sums. Atomic float additions gave 3 different
  last bits in 100 runs; the second launch gave the same bits every time.
- The fused kernels read 8 bytes per pair and ran at 92 GB/s to 97 GB/s on the measured GPU, 2.2 to
  2.4 times as fast as a map followed by a reduction, which moves 16.

## Exercises

1. Compute the least time of the fused dot product of \\(2^{26}\\) pairs on your GPU from its
   memory bandwidth. Change `N` in both programs and compare.
2. Change both programs to `f64`. The fused kernel then reads 16 bytes per pair. Is its time twice
   that of the `f32` version?
3. Change `GRID` from 1,024 to 64 and to 8,192. How many products does each thread add in each
   case, and how do the times of the tree and shuffle kernels change?
4. In Listing 15.2, which cells does thread 5 read and write in each of the 8 steps? In which
   steps is it idle?
5. Write the maximum of an array as a reduction, reusing the tree of
   <span class="sc">Dot-Block</span> with \\(\max\\) in place of \\(+\\). What must the threads
   with no element store in line 7?
6. Prove that <span class="sc">Block-Sum</span> returns the sum of the 256 values to thread 0,
   using Theorem 15.1 for each warp.
7. The exact dot product of the measured vectors is 4,195,680.2. Explain why no float program can
   return it, and which float is the best possible answer.

## Sources and further reading

- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026. The style of
  worked example, broken variants and loop-invariant proof used here, and the exact verification
  with small integers.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed.,
  MIT Press, 2022. Pseudocode conventions and loop invariants.
- M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA, 2007.
- J. Luitjens, "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014.
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", *ACM
  Computing Surveys* 23(1), 1991.
- NVIDIA, *CUDA C++ Programming Guide*, sections "Warp Shuffle Functions", "Atomic Functions" and
  "Synchronization Functions".
- *The Rust Reference*, "Static items" and "Unsafety"; the cuda-oxide repository,
  <https://github.com/NVlabs/cuda-oxide>.
