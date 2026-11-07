<img class="plate" src="../../art/ch23.svg" alt="Lumen, a crab, stands on a beach at dusk beside a rope of 4 paper lanterns marked 1, 2, 3 and 4. Each lantern shines in proportion to the exponential of its number: the first glows faintly, and the last outshines the other 3 together. Tags under the lanterns give their shares of the light, 3%, 9%, 24% and 64%, which add up to 100%: the softmax of 1, 2, 3 and 4.">

# Chapter 23: Softmax and Layer Normalization

Every layer of a transformer model applies 2 operations to the rows of a matrix. Softmax turns
a row of scores into probabilities, and layer normalization rescales a row to a mean of 0 and a
variance of 1. Neither operation can compute an output element from the input element
alone. Each output depends on a statistic of the whole row, the sum of the exponentials in the
case of softmax, and the mean and the variance in the case of layer normalization. Each
operation is therefore a reduction over the row, followed by an element-by-element computation
that uses the result of the reduction.

In this chapter, we implement both operations as cuda-oxide kernels and measure them. Both
perform only a few arithmetic operations per byte, so their speed is set by the number of times
each kernel reads the row from memory. We develop 3 versions of each operation that read the row
3, 2 and 1 times. Both operations also contain a numerical trap: the exponentials of softmax
overflow for large inputs, and the obvious formula for the variance loses all of its significant
digits when the values are large compared with their spread. We demonstrate both failures on the
GPU and derive the standard remedies, the subtraction of the row maximum and Welford's method.

<div class="covers" markdown="1">

This chapter covers

- the softmax function, and why the row maximum is subtracted before the exponential
- layer normalization, and 3 ways to compute a variance
- cancellation in floating-point arithmetic, measured
- Welford's method for a running mean and variance, and how 2 partial results are merged
- 1 block of threads per row, and block-wide maximum and sum in cuda-oxide
- the online softmax, which finds the maximum and the sum in 1 pass
- fused kernels that keep a row in registers and read it once
- the bytes per element of each version, and its measured time

</div>

The code of the chapter is in
[`code/oxide/ch23_softmax_layernorm/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch23_softmax_layernorm).
Its `Cargo.toml` and `rust-toolchain.toml` are those of Chapter 13 with a different package name.

## 23.1 The softmax function

### 23.1.1 Definition

The **softmax** of a vector \\(x_0, \dots, x_{n-1}\\) is the vector \\(p_0, \dots, p_{n-1}\\) with

\\[ p_i = \frac{e^{x_i}}{\sum_{j=0}^{n-1} e^{x_j}} \\]

The inputs \\(x_i\\) are called **logits**. Each output is positive, because an exponential is
positive, and the outputs add up to 1, because every output has the same denominator, which is
the sum of all the numerators. A softmax therefore turns any row of real numbers into a
probability distribution. It also preserves order: a larger logit receives a larger probability,
and a logit that is larger by 1 receives a probability that is larger by a factor of
\\(e \approx 2.718\\).

| Symbol | Meaning | In the example |
|---|---|---|
| \\(x_i\\) | logit \\(i\\) of the row | 1, 2, 3, 4 |
| \\(e^{x_i}\\) | its exponential | 2.718, 7.389, 20.086, 54.598 |
| \\(\sum_j e^{x_j}\\) | the sum of the exponentials of the row | 84.791 |
| \\(p_i\\) | probability \\(i\\) | 0.0321, 0.0871, 0.2369, 0.6439 |

Figure 23.1 works the softmax of the 4 logits 1, 2, 3 and 4 by hand. The exponentials are 2.718,
7.389, 20.086 and 54.598, and their sum is 84.791. Dividing each exponential by the sum gives
0.0321, 0.0871, 0.2369 and 0.6439. The largest logit receives almost 2 thirds of the total,
although it is only 4 times as large as the smallest, because the exponential magnifies
differences.

<figure>
<img src="../../figures/ch23-softmax.svg" alt="A table for the logits 1, 2, 3 and 4. Their exponentials are 2.718, 7.389, 20.086 and 54.598, which sum to 84.791. Subtracting the maximum 4 gives -3, -2, -1 and 0, whose exponentials 0.0498, 0.1353, 0.3679 and 1 sum to 1.5530. Both routes give the same probabilities 0.0321, 0.0871, 0.2369 and 0.6439, which sum to 1.">
<figcaption><b>Figure 23.1</b> The softmax of 1, 2, 3 and 4, computed with and without the maximum subtracted.</figcaption>
</figure>

Softmax appears in 2 places in a transformer. In an attention layer, it is applied to every row
of the matrix of scores between tokens, so the row length is the length of the sequence, often
1,024 to 32,768 elements. At the output of the model, it is applied to a row with 1 logit per
word of the vocabulary, often 32,000 to 128,000 elements. A GPU implementation must therefore
handle many short rows as well as a few long ones, and we measure both cases in section 20.6.

### 23.1.2 Overflow and the row maximum

The largest finite value of type `f32` is about \\(3.4 \times 10^{38}\\), which is
\\(e^{88.7}\\). An exponential of any argument above 88.7 is therefore rounded to infinity. Logits
of 100 are not unusual in practice. For the row 100, 101, 102 and 103, every exponential is
infinite, and every probability becomes \\(\infty / \infty\\), which IEEE arithmetic defines as
NaN, "not a number". The left half of Figure 23.2 follows this computation.

The remedy rests on a property of the formula. Multiplying the numerator and the denominator of
a fraction by the same nonzero number does not change its value. If we multiply both by
\\(e^{-m}\\), where \\(m\\) is any constant, we obtain

\\[ p_i = \frac{e^{x_i} \cdot e^{-m}}{\sum_j e^{x_j} \cdot e^{-m}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}} \\]

The result is the same for every \\(m\\), so we are free to choose it. We choose the row maximum,
\\(m = \max_j x_j\\). Every argument \\(x_i - m\\) is then 0 or negative, so every exponential lies
between 0 and 1 and cannot overflow. The element that equals the maximum contributes
\\(e^0 = 1\\) to the sum, so the denominator is at least 1 and the division cannot be by 0. For
the row 1, 2, 3, 4, the shifted arguments are \\(-3, -2, -1, 0\\). Their exponentials 0.0498,
0.1353, 0.3679 and 1 add up to 1.5530, and the probabilities are the same as before, as the right
half of Figure 23.1 shows. For the row 100, 101, 102, 103, the shifted arguments are the same
\\(-3\\) to 0, and the probabilities are again 0.0321 to 0.6439 (Figure 23.2).

<figure>
<img src="../../figures/ch23-overflow.svg" alt="Left: the logits 100, 101, 102 and 103 have exponentials from 2.7e43 to 5.4e44, all above the largest float, 3.4e38, so each becomes infinity and the division gives NaN. Right: after subtracting the maximum 103 the arguments are -3 to 0, the exponentials 0.0498 to 1 sum to 1.5530, and the probabilities are 0.0321, 0.0871, 0.2369 and 0.6439.">
<figcaption><b>Figure 23.2</b> Without the maximum, every exponential of the row 100 to 103 overflows; with it, none does.</figcaption>
</figure>

The subtraction has a cost. The maximum must be known before any exponential can be computed,
so the straightforward algorithm passes over the row 3 times:

```text
softmax(row x of n elements):
    m ← −∞
    for j ← 0 to n − 1:   m ← max(m, x[j])          # pass 1: the maximum
    s ← 0
    for j ← 0 to n − 1:   s ← s + exp(x[j] − m)     # pass 2: the sum
    for j ← 0 to n − 1:   p[j] ← exp(x[j] − m) / s  # pass 3: the outputs
```

Each pass reads the whole row. For a row that is too large to stay in a cache, each pass reads it
from memory, and section 20.4 reduces the number of passes.

## 23.2 Layer normalization

### 23.2.1 Definition

**Layer normalization** rescales each row of a matrix so that its elements have a mean of 0 and a
variance of 1, and then applies a scale and a shift that are learned for each column. For a row
\\(x_0, \dots, x_{n-1}\\), the output is

\\[ y_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}} \, \gamma_j + \beta_j,
\qquad \mu = \frac{1}{n} \sum_{j} x_j, \qquad \sigma^2 = \frac{1}{n} \sum_{j} (x_j - \mu)^2 \\]

| Symbol | Meaning | In the example |
|---|---|---|
| \\(\mu\\) | the mean of the row | 3.75 |
| \\(\sigma^2\\) | the variance of the row, the mean squared deviation from \\(\mu\\) | 1.1875 |
| \\(\varepsilon\\) | a small constant that keeps the division finite for a constant row | \\(10^{-5}\\) |
| \\(\gamma_j, \beta_j\\) | the scale and shift of column \\(j\\) | 1 and 0 |
| \\(y_j\\) | output \\(j\\) | −1.6059, 0.2294, 0.2294, 1.1471 |

For the row 2, 4, 4, 5, the mean is \\(15 / 4 = 3.75\\). The deviations from the mean are
\\(-1.75, 0.25, 0.25, 1.25\\), their squares are 3.0625, 0.0625, 0.0625 and 1.5625, and the
variance is their mean, \\(4.75 / 4 = 1.1875\\). With \\(\varepsilon = 10^{-5}\\), the factor
\\(1 / \sqrt{\sigma^2 + \varepsilon}\\) is 0.91766, and with \\(\gamma_j = 1\\) and
\\(\beta_j = 0\\) the outputs are \\(-1.75 \times 0.91766 = -1.6059\\), then 0.2294, 0.2294 and
1.1471. Without \\(\varepsilon\\), a row whose elements are all equal would have a variance of 0,
and the formula would divide 0 by 0.

### 23.2.2 Computing the variance

The definition of \\(\sigma^2\\) needs the mean \\(\mu\\) before the deviations can be summed, so
it suggests 2 passes over the row: 1 for the mean and 1 for the squared deviations. A third
pass computes the outputs. The variance can also be written as

\\[ \sigma^2 = \frac{1}{n} \sum_j x_j^2 - \mu^2 \\]

which needs only the sum of the values and the sum of their squares, and both can be accumulated
in the same pass. For the row 2, 4, 4, 5, the mean of the squares is
\\((4 + 16 + 16 + 25) / 4 = 15.25\\) and the square of the mean is \\(3.75^2 = 14.0625\\). Their
difference is 1.1875, as before. In exact arithmetic the 2 formulas are equal. In floating-point
arithmetic they are not.

A number of type `f32` has a 24-bit significand, so it is exact to about 1 part in
\\(1.7 \times 10^{7}\\). The gap between 2 neighbouring floats therefore grows with their size:
it is \\(6.1 \times 10^{-5}\\) for numbers between 512 and 1,024, 0.0625 near 1,000,000, and 256
near \\(4 \times 10^{9}\\). Consider a row of 4,096 values uniformly distributed between 999 and
1,001. Its mean is close to 1,000 and its variance is about 0.33. The mean of the squares is then
close to 1,000,000.33, and the square of the mean close to 1,000,000.00. Each of these 2 numbers
can be held only to the nearest 0.0625, and the sum of 4,096 squares, about \\(4.1 \times 10^9\\),
only to the nearest 256, which is again 0.0625 per element after the division by 4,096. The
variance, 0.33, is the difference of 2 numbers that each carry rounding errors of several times
0.0625, so most of its digits are rounding error (Figure 23.3). The loss of significant digits
when 2 nearly equal numbers are subtracted is called **cancellation**. We measure its effect on
the GPU in section 20.6.

<figure>
<img src="../../figures/ch23-cancel.svg" alt="A table. The mean of the squares of values near 1,000 is about 1,000,000.33 and the square of the mean 1,000,000.00; floats near 1,000,000 are spaced 0.0625 apart. The sum of 4,096 squares, about 4,096,001,365, is held to the nearest 256. The true variance, 0.33, is a small difference between 2 large uncertain numbers. Welford's method accumulates squared deviations of about 0.33 instead.">
<figcaption><b>Figure 23.3</b> The variance of values near 1,000 as a difference of 2 numbers near 1,000,000.</figcaption>
</figure>

### 23.2.3 Welford's method

**Welford's method** computes the mean and the variance in 1 pass without forming the squares of
the values. It keeps 3 numbers while it reads the values: the count \\(n\\), the running mean,
and the running sum of squared deviations from the mean, which we call \\(q\\). When a new value
\\(x\\) arrives, it updates them as follows:

```text
welford_update(state (n, mean, q), value x):
    n     ← n + 1
    delta ← x − mean                 # deviation from the old mean
    mean  ← mean + delta / n
    q     ← q + delta × (x − mean)   # old deviation times new deviation
```

After the last value, the variance is \\(q / n\\). The left half of Figure 23.4 applies the update
to 2, 4, 4 and 5. After the first value the mean is 2 and \\(q\\) is 0. The second value, 4,
deviates by 2 from the old mean; the mean moves to 3, and \\(q\\) grows by
\\(2 \times (4 - 3) = 2\\). The third value moves the mean to 3.3333 and \\(q\\) to 2.6667, and the
fourth moves them to 3.75 and 4.75, which gives the variance \\(4.75 / 4 = 1.1875\\), the same as
the definition. Every quantity in the computation is a deviation from the mean or a product of
2 deviations. For values near 1,000 with a variance of 0.33, the numbers that are added are
therefore of the size of the variance itself, and no cancellation occurs.

<figure>
<img src="../../figures/ch23-welford.svg" alt="Left: Welford's method on 2, 4, 4 and 5. The mean goes 2, 3, 3.3333, 3.75 and the sum of squared deviations q goes 0, 2, 2.6667, 4.75, so the variance is 4.75 / 4 = 1.1875. Right: the halves 2, 4 (mean 3, q 2) and 4, 5 (mean 4.5, q 0.5) merge to mean 3.75 and q = 2 + 0.5 + 1.5 squared times 2 times 2 / 4 = 4.75.">
<figcaption><b>Figure 23.4</b> Welford's update on 2, 4, 4 and 5, and the merge of the 2 halves.</figcaption>
</figure>

A GPU computes a row with many threads, and each thread sees only part of the row. Each thread
therefore runs Welford's method on its own values, and the partial results must then be combined.
Two states \\((n_a, \mu_a, q_a)\\) and \\((n_b, \mu_b, q_b)\\) merge into 1 by the following rule,
due to Chan, Golub and LeVeque:

\\[ n = n_a + n_b, \qquad \delta = \mu_b - \mu_a, \qquad \mu = \mu_a + \delta \frac{n_b}{n},
\qquad q = q_a + q_b + \delta^2 \frac{n_a n_b}{n} \\]

The last term accounts for the distance between the 2 means: each half measured its deviations
from its own mean, and the merged state needs them from the common mean. The right half of
Figure 23.4 merges the halves 2, 4 (mean 3, \\(q = 2\\)) and 4, 5 (mean 4.5, \\(q = 0.5\\)). The
difference of the means is 1.5, and the merged mean is \\(3 + 1.5 \times 2 / 4 = 3.75\\). The
merged \\(q\\) is \\(2 + 0.5 + 1.5^2 \times 2 \times 2 / 4 = 4.75\\), the same as the sequential
computation.

## 23.3 One block per row

### 23.3.1 Assigning rows to blocks

Our matrices have 4,096 rows of 4,096 `f32` values, 64 MiB in all, which is larger than the L2
cache of an H100, 50 MB, so a pass over the matrix reads it from memory. We assign 1 block of 256 threads to each row, so the grid has
4,096 blocks. Thread \\(t\\) of a block reads the columns \\(t, t + 256, t + 512, \dots\\), up to
\\(t + 3{,}840\\), which is \\(4{,}096 / 256 = 16\\) values (Figure 23.5). At each step the 32
threads of a warp read 32 consecutive columns, 128 consecutive bytes, so every access is
coalesced in the sense of Chapter 7.

<figure>
<img src="../../figures/ch23-row-block.svg" alt="A row of 4,096 columns drawn as 16 groups of 256. Thread 5 reads column 5 of every group, 16 values in all. The 256 partial results become 8 warp results by shuffles, the 8 are written to shared memory by lane 0 of each warp, and every thread then reads the 8 slots to obtain the row result.">
<figcaption><b>Figure 23.5</b> 1 block per row: each thread reduces 16 values, then the block combines 256 partial results.</figcaption>
</figure>

A block per row is a natural choice when the row has thousands of elements, because each thread
then has several values to process and the block-wide reduction is a small part of the work. For
rows of a few hundred elements, a common alternative assigns 1 warp per row, so that the
reduction needs no shared memory and no barrier. We keep 1 block per row throughout the chapter.

### 23.3.2 Block-wide maximum and sum

Each thread first reduces its 16 values to 1, its partial maximum or partial sum. The block must
then combine 256 partial results into 1 and make the result available to all 256 threads,
because every thread needs the maximum and the sum to compute its outputs.

The first step combines the 32 values of a warp with shuffles. In section 8.5 we used
`shuffle_down`, which leaves the complete sum in lane 0 only. Here we use `shuffle_xor`. In the
step with mask 16, lane \\(i\\) exchanges its value with lane \\(i \oplus 16\\): lanes 0 and 16
swap values, as do 1 and 17, and so on. Both lanes of each pair add the value they receive. After
the steps with masks 16, 8, 4, 2 and 1, every lane holds the sum of all 32 values. This pattern
is called a **butterfly** reduction.

<p class="listing"><b>Listing 23.1</b> Warp-wide maximum and sum with butterfly shuffles. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch23_softmax_layernorm/src/main.rs">code/oxide/ch23_softmax_layernorm/src/main.rs</a></p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:24:42}}
}
```

The `while` loop runs for the masks 16, 8, 4, 2 and 1, 5 steps for 32 lanes. The device function
`f32::max` returns the larger of its 2 arguments, and the backend compiles it, like `exp` and
`sqrt` later in the chapter, to the corresponding GPU math function.

The second step combines the 8 warp results of the block. Each warp writes its result into a
slot of shared memory, and after a barrier every thread reads all 8 slots and combines them
itself. Reading 8 values is cheaper than a second round of shuffles followed by another barrier,
and every thread ends with the result.

<p class="listing"><b>Listing 23.2</b> The block-wide maximum. <code>block_sum</code> differs only in the operation.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:44:64}}
}
```

The function receives the 8 slots as a raw pointer, `slots`, because a `#[device]` helper cannot
declare shared memory of its own; each kernel declares a `static mut SharedArray<f32, 8>` and
passes its address. Lane 0 of each warp writes its slot in the first `if`. The first barrier
makes the 8 writes visible to every thread before any thread reads them. The second barrier is
easy to overlook. A kernel calls `block_max` and then `block_sum` with the same slots, and
without the second barrier, a fast warp could leave `block_max`, enter `block_sum` and overwrite
its slot while a slow warp is still reading the maxima. Each `unsafe` block carries a `SAFETY`
comment that states which threads touch the slots between which barriers, which is the argument
that the accesses do not race.

## 23.4 Softmax kernels

### 23.4.1 Three passes

The first kernel follows the pseudocode of section 20.1.2 directly.

<p class="listing"><b>Listing 23.3</b> Softmax in 3 passes over the row.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:130:163}}
}
```

Block \\(b\\) processes row \\(b\\), which starts at element `base = b × cols`. Each of the 3
`while` loops starts at column `tid` and advances by 256 columns, the access pattern of
Figure 23.5. The first loop finds the thread's partial maximum, and `block_max` turns the 256
partial maxima into the row maximum `top`. The second loop sums the shifted exponentials, and
`block_sum` produces the row total. The third loop reads each element once more and writes its
probability.

The output is a `DisjointSlice<f32>`, which in safe code lets each thread write only the element
at its own global index (Chapter 13). Here thread \\(t\\) of block \\(b\\) writes the 16 elements
\\(b \times 4{,}096 + t + 256k\\), which no other thread writes, so the kernel uses
`get_unchecked_mut` and states that argument in the `SAFETY` comment.

Each element is read 3 times and written once. With 4 bytes per value, the kernel moves
\\(4 \times 4 = 16\\) bytes per element if every read goes to memory.

### 23.4.2 The online softmax

The first 2 passes of the 3-pass kernel can be combined. Suppose we have processed part of a row
and know its maximum so far, \\(m\\), and the sum of the exponentials of the part,
\\(s = \sum e^{x_j - m}\\), measured relative to that maximum. When the next value \\(x\\) arrives,
there are 2 cases. If \\(x \le m\\), the maximum does not change and we add \\(e^{x - m}\\) to
\\(s\\). If \\(x > m\\), the new maximum is \\(x\\), and every term of \\(s\\) was computed relative
to the old maximum. Multiplying \\(s\\) by \\(e^{m - x}\\) converts all of them at once, because
\\(e^{x_j - m} \cdot e^{m - x} = e^{x_j - x}\\). The new value contributes \\(e^{x - x} = 1\\):

```text
online_update(state (m, s), value x):
    if x > m:   s ← s × exp(m − x) + 1;   m ← x
    else:       s ← s + exp(x − m)
```

Figure 23.6 applies the update to 1, 2, 3, 4. Each value is a new maximum, so each step rescales
\\(s\\) by \\(e^{-1} = 0.3679\\) and adds 1: \\(s\\) goes from 1 to 1.3679, 1.5032 and 1.5530,
which is the sum of Figure 23.1. This formulation was published by Milakov and Gimelshein in 2018
and is called the **online softmax**.

<figure>
<img src="../../figures/ch23-online.svg" alt="A trace over the logits 1, 2, 3, 4. Start: m = 1, s = 1. At x = 2: m = 2, s = 1 times exp(-1) plus 1 = 1.3679. At x = 3: m = 3, s = 1.5032. At x = 4: m = 4, s = 1.5530, the same sum as the 2-pass method gives.">
<figcaption><b>Figure 23.6</b> The running maximum and the rescaled sum over 1, 2, 3, 4.</figcaption>
</figure>

The same reasoning merges 2 partial states \\((m_a, s_a)\\) and \\((m_b, s_b)\\). The merged
maximum is \\(m = \max(m_a, m_b)\\), and each sum is converted to it before they are added,
\\(s = s_a e^{m_a - m} + s_b e^{m_b - m}\\). The kernel uses this merge to combine the states of
the threads.

<p class="listing"><b>Listing 23.4</b> The merge of 2 softmax states, and the online softmax kernel.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:88:93}}

    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:165:220}}
}
```

Each thread starts from its first value, with \\(m = x\\) and \\(s = 1\\), and applies the update to
its other 15 values. Starting from the first value instead of from \\(m = -\infty\\) and
\\(s = 0\\) avoids the expression \\(-\infty - (-\infty)\\), which is NaN. The butterfly then
merges the states within each warp, shuffling both \\(m\\) and \\(s\\) in each step, and every
thread merges the 8 warp states from shared memory. The second loop computes the outputs as in
the 3-pass kernel. The row is read twice and written once, 12 bytes per element.

### 23.4.3 A fused kernel

A row of 4,096 values occupies 16 KiB, and a block of 256 threads holds 16 of its values per
thread. A thread can keep 16 values in 16 registers. It can therefore read its values once and
compute the maximum, the exponentials and the outputs from the registers.

<p class="listing"><b>Listing 23.5</b> Softmax with the row in registers.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:222:261}}
}
```

The row length must be a constant, `COLS = 4096`, because the array `x` of
`PER_THREAD = COLS / 256 = 16` values must have a size known when the kernel is compiled. Each
loop is a `while` loop with `#[unroll]` and a constant trip count, the form that section 9.7
found necessary for the backend to place a small array in registers instead of local memory.
The second loop overwrites each value with its exponential, so the third loop only multiplies by
the reciprocal of the sum. The kernel reads each element once and writes it once, 8 bytes per
element.

The fused kernel cannot handle every row. A row of 65,536 values would need 256 values per
thread, more registers than a thread can hold, and a row length that is known only at run time
cannot size the array. Libraries handle this by compiling several fused variants for common
lengths and falling back to a 2-pass kernel for the others.

### 23.4.4 Bytes per element

Figure 23.7 compares the memory traffic of the 3 kernels. A memory-bound kernel takes a time
roughly proportional to the bytes it moves, so if every read went to memory, the 3 kernels would
take time in the ratio 16 : 12 : 8. For our matrix of 4,096 × 4,096 elements, the fused kernel
moves \\(16{,}777{,}216 \times 8\\) bytes, 134 MB, which takes 1.32 ms at 102 GB/s.

<figure>
<img src="../../figures/ch23-passes.svg" alt="3 rows of bars. The 3-pass kernel reads each element 3 times, for the maximum, the sum and the output, and writes it once: 16 bytes. The online kernel reads it twice and writes once: 12 bytes. The fused kernel reads once and writes once: 8 bytes, because its values stay in registers.">
<figcaption><b>Figure 23.7</b> Memory traffic per element of the 3 softmax kernels.</figcaption>
</figure>

The count assumes that every read reaches memory. A block reads its row of 16 KiB 3 times within
a short interval, and the second and third reads can be served by the L1 or L2 cache. We
measure how much the cache saves in section 20.6.

## 23.5 Layer normalization kernels

The 3 layer normalization kernels have the same structure as the softmax kernels: 1 block per
row, a block-wide reduction, and an output loop. They differ in how they compute the variance.

### 23.5.1 Sum and sum of squares

The first kernel accumulates \\(\sum x\\) and \\(\sum x^2\\) in 1 loop and computes the variance
as the mean of the squares minus the square of the mean, the formula of section 20.2.2.

<p class="listing"><b>Listing 23.6</b> Layer normalization from the sum and the sum of squares.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:263:299}}
}
```

The kernel reads each element twice, once for the sums and once for the outputs, and writes it
once, 12 bytes per element. The arrays `gamma` and `beta` hold 4,096 values each, 32 KiB in
total, and every block reads the same values, so after the first blocks they come from the
cache; we do not count them.

### 23.5.2 Welford's method

The second kernel replaces the 2 sums with a Welford state per thread and merges the states with
the rule of section 20.2.3.

<p class="listing"><b>Listing 23.7</b> The merge of 2 Welford states, and layer normalization with Welford's method.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:95:102}}

    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:301:365}}
}
```

The first loop is the update of section 20.2.3, applied to the 16 values of the thread. The state
is a tuple of 3 `f32` values, and the butterfly shuffles each of the 3 in every step. After the
warp merge, lane 0 of each warp writes its state into 3 consecutive slots of `STATE`, which
therefore holds \\(8 \times 3 = 24\\) floats, and every thread merges the 8 states. The variance
is \\(q / n\\) with \\(n = 4{,}096\\). The kernel moves 12 bytes per element, as the previous one
does.

### 23.5.3 A fused kernel

With the row in registers, the variance can be computed by its definition, in 2 passes over the
registers: the first pass for the mean, the second for the squared deviations. The passes cost
no memory traffic, and the subtraction of the mean comes before the squaring, so no cancellation
occurs.

<p class="listing"><b>Listing 23.8</b> Layer normalization with the row in registers.</p>

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:367:411}}
}
```

The second loop subtracts the mean from each register in place, so the output loop multiplies
the centred values by the factor `rstd`, the reciprocal of the standard deviation. The kernel
reads each element once and writes it once, 8 bytes per element.

## 23.6 Running the program

### 23.6.1 The host program

The host program creates the context and loads the module as in Chapter 13. It generates the
inputs on the CPU with the linear congruential generator of Chapter 8 and launches each kernel
with 1 block per row. It then compares every output with a softmax or a layer normalization
computed in `f64` on the CPU. The comparison reports the largest absolute difference over all
16,777,216 elements.

Each kernel takes about 2 ms, so a single launch is too short to time reliably. The function
`time_ms` first runs the kernel 100 times so that the GPU raises its clocks, then times 7 batches
of 20 launches and returns the median batch time divided by 20. The median ignores a batch that
was slowed by other work on the GPU.

<p class="listing"><b>Listing 23.9</b> Timing a kernel by the median of 7 batches.</p>

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:414:437}}
```

The main function measures the kernels in 4 groups. The first runs the 2 rows of 4 logits of
Figure 23.2, and the second the 3 softmax kernels on rows of 4,096. The third runs the 3-pass and
online kernels on the same logits arranged as 256 rows of 65,536, and the fourth the 3 layer
normalization kernels. Each group follows the same steps.

<p class="listing"><b>Listing 23.10</b> The softmax group: prepare the launches, then time and check each kernel.</p>

```rust,ignore
fn main() -> Result<(), Error> {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:520:550}}
    // ...
}
```

### 23.6.2 The measurements

The measurements come from a Jetson Orin NX, an integrated GPU with 8 SMs and 102 GB/s of memory
bandwidth. Repeated runs differ by 10% to 25%, and we show the run with the lowest times of 3.

```console
$ cargo oxide run
softmax of 1 row of 4 logits
  logits      [1.0, 2.0, 3.0, 4.0]
  exp(x)      0.0321  0.0871  0.2369  0.6439
  exp(x - m)  0.0321  0.0871  0.2369  0.6439
  logits      [100.0, 101.0, 102.0, 103.0]
  exp(x)      NaN  NaN  NaN  NaN
  exp(x - m)  0.0321  0.0871  0.2369  0.6439
softmax of 4096 rows x 4096 columns, median of 7 x 20 runs
  kernel      bytes/elem      ms    GB/s   max error
  3 passes            16   2.028   132.4     1.37e-9
  online              12   2.245    89.7     1.36e-9
  fused                8   1.446    92.8     1.37e-9
softmax of 256 rows x 65536 columns, median of 7 x 20 runs
  kernel      bytes/elem      ms    GB/s   max error
  3 passes            16   3.984    67.4    1.22e-10
  online              12   3.009    66.9    1.18e-10
layer norm of 4096 rows x 4096 columns, values 1,000 + [-1, 1)
  kernel      bytes/elem      ms    GB/s   max error
  sum, sum x2         12   2.288    88.0      2.97e0
  Welford             12   2.168    92.9     1.88e-4
  fused                8   1.794    74.8     1.81e-4
PASSED
```

The first group reproduces the hand calculations. For the logits 1 to 4, the kernel without the
maximum and the kernel with it both print 0.0321, 0.0871, 0.2369 and 0.6439, the values of
Figure 23.1. For the logits 100 to 103, the kernel without the maximum prints NaN for every
element, as Figure 23.2 predicts. The 3-pass kernel prints the same 4 probabilities as for 1 to
4, because the 2 rows differ only by a constant.

The GB/s column divides the bytes per element of the model in Figure 23.7 by the measured time.
For the fused softmax, the model is exact: every element is read once and written once, and the
kernel reaches 92.8 GB/s, 91% of the 102 GB/s of the memory. For the 3-pass kernel on rows of
4,096, the same calculation gives 132.4 GB/s, more than the memory can deliver. The model
therefore overcounts: some of the second and third reads of each 16 KiB row were served by the
cache. The kernel is still 1.40 times as slow as the fused kernel (2.028 / 1.446), because its
cache hits are not free and its 3 loops and 2 block reductions add work. On rows of 4,096, the
online kernel saves nothing over 3 passes, since the read it removes was mostly a cache hit.

<figure>
<img src="../../figures/ch23-measured.svg" alt="Horizontal bars of time per matrix of 4,096 by 4,096 floats. Softmax with rows of 4,096: 3 passes 2.028 ms, online 2.245 ms, fused 1.446 ms. With rows of 65,536: 3 passes 3.984 ms, online 3.009 ms. Layer norm: sum of squares 2.288 ms but wrong, Welford 2.168 ms, fused 1.794 ms. A dashed line marks 1.32 ms, the time to read and write every element once at 102 GB/s.">
<figcaption><b>Figure 23.8</b> Measured time per 64 MiB matrix for each kernel, against the time to read and write it once.</figcaption>
</figure>

The rows of 65,536 values occupy 256 KiB each, and a block can no longer keep its row in the cache
between passes. The 3-pass kernel now takes 3.984 ms, nearly twice as long as on short rows, and
the online kernel 3.009 ms. Their ratio, 1.32, is close to the ratio of their memory traffic,
\\(16 / 12 = 1.33\\). For long rows, such as the output layer of a language model, the online kernel
is the faster of the 2; for short rows, a fused kernel is faster than either.

The layer normalization group shows the effect of cancellation. The kernel that computes the
variance from the sum of squares is about as fast as the others, but its largest error is 2.97
on outputs whose typical size is about 1. The Welford and fused kernels differ from the `f64`
reference by at most \\(1.9 \times 10^{-4}\\). That remaining error has a definite source. The
values lie near 1,000, where neighbouring floats are \\(6.1 \times 10^{-5}\\) apart, so the mean
itself can be held only to a few times that spacing. Dividing by the standard deviation,
\\(\sqrt{0.33} = 0.58\\), multiplies the error by 1.7. The program therefore accepts layer
normalization errors below \\(10^{-3}\\) and softmax errors below \\(10^{-6}\\), and requires the
sum-of-squares kernel to exceed \\(0.1\\), so that it reports PASSED only if the kernels behave as
the analysis predicts.

<div class="bee bee-shocked" markdown="1">

The formula \\(\sigma^2 = \overline{x^2} - \mu^2\\) is exact in algebra and useless in `f32` when
the mean is large compared with the spread. Compute a variance from deviations, by 2 passes or by
Welford's method.

</div>

## 23.7 The complete program

<p class="listing"><b>Listing 23.11</b> The complete program. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch23_softmax_layernorm/src/main.rs">code/oxide/ch23_softmax_layernorm/src/main.rs</a></p>

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs}}
```

## Common mistakes

- **Computing the exponential before subtracting the maximum.** Logits above 88.7 overflow `f32`,
  and the softmax of the row becomes NaN.
- **Initialising an online state with \\(m = -\infty\\) and \\(s = 0\\) and merging it.** The merge
  computes \\(e^{-\infty - (-\infty)}\\), which is NaN. Start each thread from its first value, or
  test for an empty state before merging.
- **Computing the variance as the mean of the squares minus the square of the mean.** The result
  loses its significant digits when the mean is large compared with the spread, and can even be
  negative, which makes the square root NaN.
- **Leaving out the second barrier of a block reduction.** A warp that finishes early can
  overwrite a shared slot that another warp is still reading, when 2 reductions use the same
  slots one after the other.
- **Counting bytes without asking where they come from.** A model that assumes every read goes to
  memory overestimates the cost of repeated reads of a short row, which the cache serves.
- **Leaving out \\(\varepsilon\\).** A row of equal values has a variance of 0, and the division
  gives NaN.

## Summary

- Softmax divides the exponential of each logit by the sum of the exponentials of its row.
  Subtracting the row maximum first leaves every result unchanged and keeps every exponential in
  \\((0, 1]\\), so nothing overflows.
- Layer normalization subtracts the mean of the row and divides by the standard deviation, with a
  small \\(\varepsilon\\) under the square root, then applies a scale and a shift per column.
- The variance computed as the mean of the squares minus the square of the mean suffers from
  cancellation. On rows near 1,000 its error reached 2.97. Welford's method, or 2 passes over
  values held in registers, kept the error below \\(1.9 \times 10^{-4}\\).
- 1 block per row, with butterfly shuffles in each warp and 8 shared slots per block, gives every
  thread the maximum, the sum or the Welford state of its row.
- The online softmax finds the maximum and the sum in 1 pass by rescaling the running sum when
  the maximum grows. It reads a row twice instead of 3 times, which saved 25% on rows of 65,536.
- A fused kernel keeps its share of the row in registers and reads it once. On rows of 4,096 it
  reached 91% of the memory bandwidth and was the fastest version of both operations.

## Exercises

1. Compute by hand the softmax of the logits 0, 0, 0, 10. What fraction of the total does the
   last logit receive, and what does the result say about the effect of 1 large logit?
2. Show that subtracting any constant \\(c\\) from every logit leaves the softmax unchanged, but
   that only \\(c \ge \max_j x_j\\) guarantees that no exponential overflows. Which choice of
   \\(c\\) also guarantees that the denominator is at least 1?
3. Apply Welford's update by hand to 1,000.5, 999.5, 1,000.5, 999.5. Then compute the mean of the
   squares minus the square of the mean, rounding each square to the nearest 0.0625. Compare the
   2 variances with the exact value, 0.25.
4. Merge the Welford states of 2, 4, 4 and of 5 with the rule of section 20.2.3, and check the
   result against Figure 23.4.
5. Change `ROWS` and `COLS` to 16,384 rows of 1,024 columns. Which constant must also change for
   the fused kernels to remain correct, and how do you expect the 3-pass kernel to compare with the
   fused kernel? Run the program and compare.
6. The online kernel computes 1 exponential per element in its first pass and 1 in its second.
   Count the exponentials of the 3-pass kernel. Why does the online kernel not save arithmetic,
   and what does it save?
7. Write a softmax kernel that assigns 1 warp of 32 threads to each row, with no shared memory
   and no barrier. Measure it against the fused kernel on rows of 1,024.
8. The output layer of a language model applies softmax to rows of 128,000 logits. Estimate the
   time of the 3-pass and online kernels for 64 such rows on a GPU with 3.35 TB/s of memory
   bandwidth, assuming no cache hits.

## Sources and further reading

- M. Milakov and N. Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867,
  2018. The online softmax of section 20.4.2.
- B. P. Welford, *Note on a method for calculating corrected sums of squares and products*,
  Technometrics 4(3), 1962. The update of section 20.2.3.
- T. F. Chan, G. H. Golub and R. J. LeVeque, *Updating formulae and a pairwise algorithm for
  computing sample variances*, Stanford technical report STAN-CS-79-773, 1979. The merge rule.
- J. L. Ba, J. R. Kiros and G. E. Hinton, *Layer normalization*, arXiv:1607.06450, 2016.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002. Chapter 1
  on cancellation and chapter 4 on the accuracy of summation.
- T. Dao et al., *FlashAttention: fast and memory-efficient exact attention with IO-awareness*,
  NeurIPS 2022. The online softmax applied inside attention.
