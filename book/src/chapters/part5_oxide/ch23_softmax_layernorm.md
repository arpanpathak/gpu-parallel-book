<img class="plate" src="../../art/ch23.svg" alt="Lumen, a crab, stands on a beach at dusk beside a rope of 4 paper lanterns marked 1, 2, 3 and 4. Each lantern shines in proportion to the exponential of its number: the first glows faintly, and the last outshines the other 3 together. Tags under the lanterns give their shares of the light, 3%, 9%, 24% and 64%, which add up to 100%: the softmax of 1, 2, 3 and 4.">

# Chapter 23: Softmax and Layer Normalization

Every layer of a transformer model applies 2 operations to the rows of a matrix. Softmax turns a
row of scores into probabilities that add up to 1, and layer normalization shifts and scales a
row so that its values have mean 0 and variance 1. In both, every output value depends on a
statistic of its whole row: the largest value and the sum of exponentials for softmax, the mean
and the variance for layer normalization. Each operation is therefore a reduction over the row
followed by an element-by-element computation that uses the result of the reduction.

In this chapter we derive both operations on a matrix of 2 rows of 4 values, small enough to check
with a pencil. We first find 2 numerical failures in the direct formulas, an overflow in softmax
and a cancellation in the variance, and the standard remedies for each. We then remove the extra
passes over the row that the remedies seem to require: the online softmax finds the maximum and
the sum in 1 pass, and Welford's method finds the mean and the variance in 1 pass. Both rest on
the same idea, a partial result that can be corrected when more of the row arrives, and both
combine partial results from many threads. We prove each algorithm correct, map it onto 1 block
of threads per row, implement it in Rust and in CUDA C++, and measure it against the memory
bandwidth of the GPU.

<div class="covers" markdown="1">

This chapter covers

- softmax and layer normalization of the rows of a matrix, computed by hand
- overflow of the exponential in `f32`, and why subtracting the row maximum leaves softmax unchanged
- cancellation in the variance formula \\(\overline{x^2} - \mu^2\\), with a negative variance in `f32`
- the online softmax and Welford's method, each with a loop invariant and a proof
- merging 2 partial results, and the empty state
- 1 block per row: warp shuffles, block-wide reductions and their 2 barriers
- fused kernels that read each value once, and the bytes per element of every version
- measured times in Rust and C++, and the reasons for each gap to the memory bandwidth

</div>

The code of the chapter is in
[`code/oxide/ch23_softmax_layernorm/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch23_softmax_layernorm)
for Rust and in
[`code/cuda/ch23_softmax_layernorm/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch23_softmax_layernorm)
for C++. The Rust crate's `Cargo.toml` and `rust-toolchain.toml` are those of Chapter 13 with a
different package name.

## 23.1 The problem

### 23.1.1 The worked example

We use the same matrix throughout the chapter. It has 2 rows of 4 values:

\\[ X = \begin{pmatrix} 1 & 2 & 3 & 4 \\\\ 2 & 4 & 4 & 5 \end{pmatrix}. \\]

We write \\(x_{rj}\\) for the value in row \\(r\\) and column \\(j\\), and \\(n\\) for the length
of a row, here 4. Both operations of the chapter work on 1 row at a time: they read a row of
\\(X\\) and write the row with the same number of the result. Figure 23.1 shows \\(X\\) and the 2
results we compute by hand in this section, the softmax \\(P\\) and the layer normalization
\\(Y\\).

<figure>
<img src="../../figures/ch23-example.svg" alt="The 2 by 4 matrix X with rows 1, 2, 3, 4 and 2, 4, 4, 5. Its softmax P has rows 0.0321, 0.0871, 0.2369, 0.6439 and 0.0279, 0.2060, 0.2060, 0.5601. Its layer normalization Y has rows -1.3416, -0.4472, 0.4472, 1.3416 and -1.6059, 0.2294, 0.2294, 1.1471.">
<figcaption><b>Figure 23.1</b> The worked example \(X\), its softmax \(P\) and its layer normalization \(Y\). Row 0 of each result depends only on row 0 of \(X\).</figcaption>
</figure>

Keep 2 entries in mind: \\(p_{03} = 0.6439\\) and \\(y_{00} = -1.3416\\). In Section 23.2 the
direct formulas turn both into NaN for rows that differ from row 0 only by a constant.

The matrix is stored in **row-major order**: row 0 first, then row 1, so that value \\(x_{rj}\\)
sits at position \\(r \cdot n + j\\) of a single array of \\(2n = 8\\) floats. A whole row is
therefore a contiguous piece of memory, a fact the GPU algorithm of Section 23.4 relies on.

### 23.1.2 Softmax

<div class="definition">

<span class="box-label">Definition 23.1</span> **Softmax.** The softmax of a row
\\(x_0, \dots, x_{n-1}\\) is the row \\(p_0, \dots, p_{n-1}\\) with

\\[ p_j = \frac{e^{x_j}}{\sum_{i \in [0, n)} e^{x_i}}, \qquad j \in [0, n). \\]

The inputs \\(x_j\\) are often called *logits*, and the outputs are positive and add up to 1.

*Example.* For row 0 of \\(X\\), \\(p_3 = e^4 / (e^1 + e^2 + e^3 + e^4) = 54.598 / 84.791 = 0.6439\\).

</div>

Every output is positive because an exponential is positive, and the outputs add up to 1 because
they share 1 denominator, the sum of all the numerators. Softmax therefore turns any row of real
numbers into a probability distribution. It preserves order: a larger input receives a larger
output, and an input larger by 1 receives an output larger by a factor of \\(e = 2.718\\).

Figure 23.2 computes row 0 by hand. The 4 exponentials are 2.718, 7.389, 20.086 and 54.598, and
their sum is 84.791. Dividing each by the sum gives 0.0321, 0.0871, 0.2369 and 0.6439. The value 4
receives 64% of the total, although it is only 4 times the value 1, because the exponential
magnifies differences.

<figure>
<img src="../../figures/ch23-softmax.svg" alt="Row 0 holds 1, 2, 3, 4. Their exponentials are 2.718, 7.389, 20.086 and 54.598, which sum to 84.791. Dividing each by the sum gives 0.0321, 0.0871, 0.2369 and 0.6439, which add up to 1.">
<figcaption><b>Figure 23.2</b> Softmax of row 0 by hand. Column 3 receives \(54.598 / 84.791 = 0.6439\).</figcaption>
</figure>

Row 1 works the same way: \\(e^2 + e^4 + e^4 + e^5 = 7.389 + 54.598 + 54.598 + 148.413 = 264.998\\),
and the outputs are 0.0279, 0.2060, 0.2060 and 0.5601. The 2 equal inputs receive equal outputs.

Softmax appears in 2 places in a transformer. In an attention layer it is applied to every row of
a matrix of scores between tokens, so a row holds 1 value per token of the sequence, often 1,024
to 32,768 values. At the output of the model it is applied to a row with 1 value per word of the
vocabulary, often 32,000 to 128,000 values. A GPU implementation must handle many short rows as
well as a few long ones, and Section 23.7 measures both.

### 23.1.3 Layer normalization

<div class="definition">

<span class="box-label">Definition 23.2</span> **Mean, variance and layer normalization.** The
mean of a row is \\(\mu = \frac{1}{n} \sum_{j} x_j\\), and its variance is the mean squared
deviation from the mean, \\(\sigma^2 = \frac{1}{n} \sum_j (x_j - \mu)^2\\). The square root
\\(\sigma\\) is the standard deviation. The layer normalization of the row is

\\[ y_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}} \, \gamma_j + \beta_j, \qquad j \in [0, n), \\]

where \\(\varepsilon\\) is a small constant, and \\(\gamma_j\\) and \\(\beta_j\\) are a scale and a
shift for column \\(j\\) that the model learns.

*Example.* For row 0 of \\(X\\), \\(\mu = 2.5\\), \\(\sigma^2 = 1.25\\), and with \\(\gamma_j = 1\\),
\\(\beta_j = 0\\) and \\(\varepsilon = 10^{-5}\\), \\(y_0 = -1.5 / \sqrt{1.25001} = -1.3416\\).

</div>

Figure 23.3 carries out the definition on row 0. The mean is \\(10 / 4 = 2.5\\). The deviations
from the mean are \\(-1.5, -0.5, 0.5, 1.5\\), their squares are 2.25, 0.25, 0.25 and 2.25, and the
variance is their mean, \\(5 / 4 = 1.25\\). Each deviation is then multiplied by
\\(1 / \sqrt{1.25 + 10^{-5}} = 0.8944\\), which gives \\(-1.3416, -0.4472, 0.4472\\) and
\\(1.3416\\). The outputs have mean 0 and variance \\(1.25 / 1.25001\\), which is 1 up to
\\(\varepsilon\\).

<figure>
<img src="../../figures/ch23-layernorm.svg" alt="Row 0 holds 1, 2, 3, 4. The mean is 2.5. The deviations are -1.5, -0.5, 0.5, 1.5, their squares 2.25, 0.25, 0.25, 2.25, and the variance 1.25. Multiplying each deviation by 1 / sqrt(1.25 + 0.00001) = 0.8944 gives -1.3416, -0.4472, 0.4472 and 1.3416.">
<figcaption><b>Figure 23.3</b> Layer normalization of row 0 by hand, with \(\gamma_j = 1\) and \(\beta_j = 0\).</figcaption>
</figure>

For row 1 the mean is \\(15 / 4 = 3.75\\), the deviations are \\(-1.75, 0.25, 0.25, 1.25\\), the
variance is \\(4.75 / 4 = 1.1875\\), and the outputs are \\(-1.6059, 0.2294, 0.2294\\) and
\\(1.1471\\). The constant \\(\varepsilon\\) has an effect only for a row whose values are all equal: its
variance is 0, and without \\(\varepsilon\\) every output would be \\(0 / 0\\).

Both definitions have the same shape. Each needs 1 or 2 numbers computed from the whole row, the
sum of exponentials or the mean and the variance, before it can produce its first output. The
rest of the chapter is about computing those numbers quickly and accurately.

<div class="exercises">

Exercises

23.1 Compute the softmax of the row 0, 0, 0, 10. Which fraction of the total does the last value
receive?<br>
23.2 Show that the outputs of softmax add up to 1, and that the outputs of layer normalization
with \\(\gamma_j = 1\\) and \\(\beta_j = 0\\) add up to 0.<br>
23.3 Compute the layer normalization of the row 4, 4, 4, 4 with and without \\(\varepsilon\\).

</div>

## 23.2 The sequential algorithms

### 23.2.1 Overflow in `f32`

The kernels of the chapter compute in **single precision**, the 32-bit floating-point type `f32`
in Rust and `float` in C++. We recall what that type can hold.

<div class="definition">

<span class="box-label">Definition 23.3</span> **The `f32` type.** An `f32` number holds a sign,
an 8-bit exponent and a 24-bit significand. Its largest finite value is about
\\(3.40 \times 10^{38}\\). A result larger than that is rounded to the special value
\\(\infty\\), and an expression without a defined value, such as \\(\infty / \infty\\) or
\\(0 \cdot \infty\\), gives the special value **NaN**, "not a number". Every arithmetic operation
with a NaN operand gives NaN.

*Example.* \\(e^{88} = 1.65 \times 10^{38}\\) fits in an `f32`; \\(e^{89} = 4.49 \times 10^{38}\\)
does not and becomes \\(\infty\\).

</div>

The largest `f32` equals \\(e^{88.72}\\), so the exponential of any input above 88.72 overflows.
Inputs of that size are not unusual in a model. Add 100 to every value of \\(X\\). Row 0 becomes
101, 102, 103 and 104, and the exponentials are \\(7.31 \times 10^{43}\\) to
\\(1.47 \times 10^{45}\\), all beyond the largest `f32`. Each becomes \\(\infty\\), their sum is
\\(\infty\\), and every output is \\(\infty / \infty\\), which is NaN (left half of Figure 23.4).
The entry \\(p_{03} = 0.6439\\) is gone, and so is every other entry of the row.

<figure>
<img src="../../figures/ch23-overflow.svg" alt="Left: row 0 of X + 100 holds 101 to 104, whose exponentials 7.31e43 to 1.47e45 exceed the largest float32, 3.40e38, so each becomes infinity and every output is infinity divided by infinity, NaN. Right: after subtracting the maximum 104 the arguments are -3 to 0, the exponentials sum to 1.5530, and the outputs are 0.0321, 0.0871, 0.2369 and 0.6439, the same as for row 0 of X.">
<figcaption><b>Figure 23.4</b> Row 0 of \(X + 100\). Left: the direct formula overflows. Right: with the maximum subtracted, no exponential exceeds 1.</figcaption>
</figure>

### 23.2.2 Subtracting the maximum

The remedy is a property of the formula: multiplying the numerator and the denominator of a
fraction by the same nonzero number does not change its value.

<div class="theorem">

<span class="box-label">Theorem 23.1</span> For every real number \\(c\\), the softmax of
\\(x_0 - c, \dots, x_{n-1} - c\\) equals the softmax of \\(x_0, \dots, x_{n-1}\\). If
\\(c = m = \max_j x_j\\), then every exponential \\(e^{x_j - m}\\) lies in \\((0, 1]\\), and the
denominator \\(\sum_j e^{x_j - m}\\) lies in \\([1, n]\\).

</div>

<div class="proof">

*Proof.* Since \\(e^{x_j - c} = e^{x_j} e^{-c}\\),

\\[ \frac{e^{x_j - c}}{\sum_i e^{x_i - c}} = \frac{e^{x_j} e^{-c}}{e^{-c} \sum_i e^{x_i}} = \frac{e^{x_j}}{\sum_i e^{x_i}}, \\]

because the positive factor \\(e^{-c}\\) cancels. With \\(c = m\\), every \\(x_j - m \le 0\\), so
\\(0 < e^{x_j - m} \le 1\\). The sum has \\(n\\) such terms, so it is at most \\(n\\), and the term
of a column where \\(x_j = m\\) equals \\(e^0 = 1\\), so the sum is at least 1.

</div>

The theorem gives more than freedom from overflow. A denominator of at least 1 cannot be 0, so the
division is always defined. And shifting a whole row by a constant does not change its softmax, so
\\(X + 100\\) has exactly the softmax \\(P\\) of \\(X\\). For row 0 of \\(X + 100\\), the shifted
inputs are \\(-3, -2, -1, 0\\), the exponentials 0.0498, 0.1353, 0.3679 and 1 add up to 1.5530,
and the outputs are 0.0321, 0.0871, 0.2369 and 0.6439 again (right half of Figure 23.4).

### 23.2.3 Softmax in 3 passes

The subtraction has a cost. The maximum must be known before the first exponential is computed, so
the algorithm reads the row 3 times.

<div class="definition">

<span class="box-label">Definition 23.4</span> **Pass.** A pass is 1 traversal of a row in which
every value of the row is read once. The **memory traffic** of an algorithm is the number of bytes
it reads and writes in main memory.

*Example.* An algorithm that reads a row of 4 `f32` values 3 times and writes 4 results has 3
passes and moves \\(12 \cdot 4 + 4 \cdot 4 = 64\\) bytes.

</div>

<div class="algo">
<p class="algo-name">Softmax-3-Pass(\(x, p, n\))</p>
<ol>
<li>\(m = -\infty\)</li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">\(m = \max(m, x_j)\) <span class="c">// pass 1: the maximum</span></li>
<li>}</li>
<li>\(s = 0\)</li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">\(s = s + e^{x_j - m}\) <span class="c">// pass 2: the sum</span></li>
<li>}</li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">\(p_j = e^{x_j - m} / s\) <span class="c">// pass 3: the outputs</span></li>
<li>}</li>
</ol>
</div>

Lines 1 to 4 find the maximum, lines 5 to 8 add the shifted exponentials, and lines 9 to 11 write
the outputs. Each of lines 3, 7 and 10 executes \\(n\\) times, so the algorithm makes
\\(n\\) comparisons, \\(2n\\) exponentials and \\(n\\) divisions. Figure 23.5 follows the 3 passes
on row 1. Pass 1 finds \\(m = 5\\). Pass 2 adds \\(e^{-3} + e^{-1} + e^{-1} + e^{0} =
0.0498 + 0.3679 + 0.3679 + 1 = 1.7855\\). Pass 3 writes 0.0279, 0.2060, 0.2060 and 0.5601.

<figure>
<img src="../../figures/ch23-three-pass.svg" alt="Row 1 holds 2, 4, 4, 5. Pass 1 reads it and finds the maximum 5. Pass 2 reads it again and adds exp(x - 5): 0.0498, 0.3679, 0.3679 and 1, sum 1.7855. Pass 3 reads it a third time and writes 0.0279, 0.2060, 0.2060 and 0.5601. 12 reads and 4 writes of 4 bytes, 16 bytes per element.">
<figcaption><b>Figure 23.5</b> The 3 passes of <span class="sc">Softmax-3-Pass</span> on row 1: 12 reads and 4 writes, 16 bytes per element.</figcaption>
</figure>

Each pass reads all 4 values, so the row of 16 bytes is read 3 times, and 16 bytes of output are
written: 64 bytes, or 16 bytes per element. In general a matrix of \\(R\\) rows of \\(C\\) values
costs \\(16RC\\) bytes. For \\(R = C = 4{,}096\\) that is 268 MB, which takes 2.63 ms on a GPU
whose memory delivers 102 GB/s. The arithmetic, 3 operations per element, is small beside the
traffic, so the passes decide the time.

### 23.2.4 Layer normalization in 2 or 3 passes

The definition of the variance needs the mean before the deviations can be summed, so the direct
algorithm also reads the row 3 times: once for the mean, once for the variance, once for the
outputs. A well-known identity removes 1 of the passes.

<div class="theorem">

<span class="box-label">Theorem 23.2</span> The variance of a row equals the mean of the squares
minus the square of the mean: \\(\sigma^2 = \frac{1}{n} \sum_j x_j^2 - \mu^2\\).

</div>

<div class="proof">

*Proof.* Expand the square: \\(\frac{1}{n} \sum_j (x_j - \mu)^2 = \frac{1}{n} \sum_j x_j^2 -
2 \mu \frac{1}{n} \sum_j x_j + \mu^2 = \frac{1}{n} \sum_j x_j^2 - 2\mu^2 + \mu^2\\).

</div>

The sum of the values and the sum of their squares can be accumulated in the same pass, so the
variance needs 1 pass instead of 2:

<div class="algo">
<p class="algo-name">LayerNorm-Sum-Of-Squares(\(x, y, n, \gamma, \beta\))</p>
<ol>
<li>\(a = 0\), \(b = 0\)</li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">\(a = a + x_j\), \(b = b + x_j^2\) <span class="c">// pass 1: 2 sums</span></li>
<li>}</li>
<li>\(\mu = a / n\), \(\sigma^2 = b / n - \mu^2\)</li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">\(y_j = (x_j - \mu) / \sqrt{\sigma^2 + \varepsilon} \cdot \gamma_j + \beta_j\) <span class="c">// pass 2: the outputs</span></li>
<li>}</li>
</ol>
</div>

On row 0, line 3 accumulates \\(a = 10\\) and \\(b = 1 + 4 + 9 + 16 = 30\\), and line 5 gives
\\(\mu = 2.5\\) and \\(\sigma^2 = 7.5 - 6.25 = 1.25\\), the variance of Figure 23.3. The algorithm
reads each value twice and writes it once, 12 bytes per element. In exact arithmetic it is
correct. In `f32` it is not.

### 23.2.5 Cancellation

<div class="definition">

<span class="box-label">Definition 23.5</span> **Spacing and cancellation.** The `f32` numbers
between \\(2^k\\) and \\(2^{k+1}\\) are spaced \\(2^{k-23}\\) apart, so every result in that range
is rounded to a multiple of \\(2^{k-23}\\). **Cancellation** is the loss of correct digits when 2
nearly equal numbers, each carrying a rounding error, are subtracted: the difference is small,
but the errors are not.

*Example.* Between \\(2^{26} = 67{,}108{,}864\\) and \\(2^{27}\\), `f32` numbers are
\\(2^3 = 8\\) apart, so \\(10{,}001^2 = 100{,}020{,}001\\) is stored as 100,020,000.

</div>

Add 10,000 to every value of \\(X\\). Row 0 becomes 10,001, 10,002, 10,003 and 10,004. Its
deviations from the mean are those of row 0 of \\(X\\), so its variance is still 1.25 and its
layer normalization is still \\(-1.3416, -0.4472, 0.4472, 1.3416\\). Figure 23.6 runs
<span class="sc">LayerNorm-Sum-Of-Squares</span> on it in `f32`, 1 rounding at a time.

<figure>
<img src="../../figures/ch23-cancel.svg" alt="Row 0 of X + 10,000 holds 10,001 to 10,004. Their squares round in float32 to 100,020,000, 100,040,000, 100,060,008 and 100,080,016, and the running sum ends at 400,200,000. The mean of the squares is 100,050,000. The mean is 10,002.5, whose square 100,050,006.25 rounds to 100,050,008. The difference is -8, a negative variance, where the exact variance is 1.25.">
<figcaption><b>Figure 23.6</b> The variance of row 0 of \(X + 10{,}000\) by the sum of squares, in <code>f32</code>: \(-8\) instead of 1.25.</figcaption>
</figure>

The squares lie near \\(10^8\\), where `f32` numbers are 8 apart, and 3 of the 4 are rounded. The
running sum passes \\(2.68 \times 10^8\\), where the spacing becomes 32, and the last addition,
\\(300{,}120{,}000 + 100{,}080{,}016 = 400{,}200{,}016\\), lies exactly halfway between 2
neighbours and is rounded to 400,200,000. Dividing by 4 gives the mean of the squares,
100,050,000. The mean is 10,002.5, exact in `f32`, but its square, 100,050,006.25, is rounded to
100,050,008. Line 5 subtracts:

\\[ \sigma^2 = 100{,}050{,}000 - 100{,}050{,}008 = -8. \\]

The true difference, 1.25, is smaller than the spacing 8 of the numbers being subtracted, so not
a single correct digit survives. The result is not even positive, \\(\sqrt{-8 + 10^{-5}}\\) is NaN,
and every output of the row is NaN. The entry \\(y_{00} = -1.3416\\) is gone. The GPU kernel of
Section 23.6 prints the same 8 NaNs for \\(X + 10{,}000\\).

The failure needs no extreme values. Rows of activations with a large common offset and a small
spread occur in practice, and Section 23.7 measures an error of 2.97 on outputs of size 1 for
values near 1,000. The remedy is to never form the squares of the values, only squares of
deviations, which are small whenever the variance is small.

<div class="exercises">

Exercises

23.4 Which is the largest integer \\(k\\) for which \\(e^k\\) is finite in `f32`? And in `f64`,
whose largest value is about \\(1.8 \times 10^{308}\\)?<br>
23.5 Show that Theorem 23.1 holds for any \\(c\\), but that only \\(c \ge \max_j x_j\\) guarantees
that no exponential overflows. Which choice of \\(c\\) also guarantees a denominator of at least
1?<br>
23.6 Repeat Figure 23.6 for row 0 of \\(X + 1{,}000\\), where `f32` numbers near \\(10^6\\) are
0.0625 apart. Is the variance exact?

</div>

## 23.3 One pass over the row

### 23.3.1 The dependence on the maximum

<span class="sc">Softmax-3-Pass</span> reads the row 3 times because of 2 dependences: line 7 needs
\\(m\\), which is known only after the last iteration of line 3, and line 10 needs \\(s\\), which
is known only after the last iteration of line 7. The second dependence is real: no output can be
written before the whole sum is known. The first one deserves a closer look.

Suppose we process row 1, 2, 4, 4, 5, from the left and pretend that the largest value seen so far
is the maximum. After the first value, the maximum so far is 2 and the sum so far is
\\(e^{2-2} = 1\\). The second value, 4, is larger, so the sum so far was taken relative to the
wrong maximum. It is not lost, however. Its only term, \\(e^{2-2}\\), becomes the term we want,
\\(e^{2-4}\\), when we multiply it by \\(e^{2-4}\\):

\\[ e^{2-2} \cdot e^{2-4} = e^{2-4}. \\]

The same multiplication corrects any number of terms at once, because every term of the sum was
taken relative to the same maximum.

<div class="key-observation">

<span class="box-label">Key observation</span> A sum of exponentials taken relative to 1 maximum
\\(m\\) becomes the sum relative to a larger maximum \\(m'\\) by 1 multiplication, since
\\(e^{x - m} \cdot e^{m - m'} = e^{x - m'}\\) for every term. The maximum therefore does not have
to be known before the sum begins: the sum can be corrected each time the maximum grows.

</div>

### 23.3.2 The online softmax

We now carry out the observation on row 1, 1 value at a time, keeping 2 numbers: \\(m\\), the
largest value seen so far, and \\(s\\), the sum of \\(e^{x - m}\\) over the values seen so far. We
start from \\(m = -\infty\\) and \\(s = 0\\), which describe a row with no values (Figure 23.7).

1. Value 2. It is larger than \\(m = -\infty\\). The old sum is multiplied by
   \\(e^{-\infty - 2} = 0\\), and the new value contributes \\(e^{2-2} = 1\\): \\(m = 2\\), \\(s = 1\\).
2. Value 4. It is larger than \\(m = 2\\). The old sum is moved to the new maximum, and the new
   value contributes \\(e^0 = 1\\): \\(s = 1 \cdot e^{2-4} + 1 = 0.1353 + 1 = 1.1353\\), \\(m = 4\\).
3. Value 4. It is not larger than \\(m = 4\\), so the maximum stays, and the value adds
   \\(e^{4-4} = 1\\): \\(s = 2.1353\\).
4. Value 5. It is larger than \\(m = 4\\): \\(s = 2.1353 \cdot e^{4-5} + 1 = 0.7855 + 1 = 1.7855\\),
   \\(m = 5\\).

The final pair, \\(m = 5\\) and \\(s = 1.7855\\), is exactly what passes 1 and 2 of Figure 23.5
computed, after 1 pass instead of 2.

<figure>
<img src="../../figures/ch23-online.svg" alt="The online softmax over row 1, 2, 4, 4, 5. It starts from m = -inf and s = 0. After 2: m = 2, s = 1. After 4: m = 4, s = 1 times exp(-2) plus 1 = 1.1353. After the second 4: m stays 4 and s = 2.1353. After 5: m = 5, s = 2.1353 times exp(-1) plus 1 = 1.7855, the same sum as 3 passes give.">
<figcaption><b>Figure 23.7</b> The online softmax on row 1. Steps 1, 2 and 4 find a new maximum and rescale the sum; step 3 adds to it.</figcaption>
</figure>

<div class="definition">

<span class="box-label">Definition 23.6</span> **Softmax state.** A softmax state \\((m, s)\\)
describes a set \\(S\\) of values when \\(m = \max S\\) and \\(s = \sum_{x \in S} e^{x - m}\\). The
**empty state** \\((-\infty, 0)\\) describes the empty set.

*Example.* \\((4, 1.1353)\\) describes the values 2, 4: \\(4 = \max\{2, 4\}\\) and
\\(e^{2-4} + e^{4-4} = 1.1353\\).

</div>

<div class="algo">
<p class="algo-name">Online-Softmax(\(x, p, n\))</p>
<ol>
<li>\(m = -\infty\), \(s = 0\) <span class="c">// the empty state</span></li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">if \(x_j > m\) {</li>
<li style="--d:2">\(s = s \cdot e^{m - x_j} + 1\) <span class="c">// move the sum to the new maximum</span></li>
<li style="--d:2">\(m = x_j\)</li>
<li style="--d:1">} else {</li>
<li style="--d:2">\(s = s + e^{x_j - m}\)</li>
<li style="--d:1">}</li>
<li>}</li>
<li>for \(j \in [0, n)\) {</li>
<li style="--d:1">\(p_j = e^{x_j - m} / s\)</li>
<li>}</li>
</ol>
</div>

Lines 1 to 9 are the first pass and replace lines 1 to 8 of <span class="sc">Softmax-3-Pass</span>;
lines 10 to 12 are its third pass unchanged. Line 4 handles the first value without a special case:
when \\(m = -\infty\\), the factor \\(e^{m - x_j}\\) is 0 and \\(s\\) becomes 1. The loop computes
1 exponential per value, either on line 4 or on line 7, so the work is the same as before: \\(2n\\)
exponentials in all. What changes is the traffic: 2 reads and 1 write per element, 12 bytes
instead of 16.

The algorithm is short, but each step relies on the previous state being exactly right, so we
prove it.

<div class="theorem">

<span class="box-label">Theorem 23.3</span> When the loop of lines 2 to 9 of
<span class="sc">Online-Softmax</span> ends, \\(m = \max_j x_j\\) and
\\(s = \sum_{j \in [0,n)} e^{x_j - m}\\). The outputs of line 11 are therefore the softmax of
\\(x\\).

</div>

<div class="proof">

*Proof.* We show that the loop maintains the invariant

\\[ \text{at the start of iteration } j, \quad (m, s) \text{ describes } \{x_0, \dots, x_{j-1}\}, \\]

that is, \\(m = \max_{i < j} x_i\\) and \\(s = \sum_{i < j} e^{x_i - m}\\) (Figure 23.8).

*Initialization.* Before iteration 0, the set is empty and line 1 sets the empty state.

*Maintenance.* Assume the invariant at the start of iteration \\(j\\). If \\(x_j > m\\), the new
maximum is \\(x_j\\). Line 4 multiplies every term \\(e^{x_i - m}\\) of \\(s\\) by
\\(e^{m - x_j}\\), which turns it into \\(e^{x_i - x_j}\\) by the key observation, and adds
\\(e^{x_j - x_j} = 1\\), the term of the new value. Line 5 sets \\(m = x_j\\). If
\\(x_j \le m\\), the maximum does not change, and line 7 adds the term \\(e^{x_j - m}\\) of the new
value. In both cases \\((m, s)\\) describes \\(\{x_0, \dots, x_j\}\\) at the start of iteration
\\(j + 1\\).

*Termination.* The loop ends with \\(j = n\\), when \\((m, s)\\) describes the whole row. Line 11
then computes \\(e^{x_j - m} / s\\), which is the softmax by Theorem 23.1.

</div>

<figure>
<img src="../../figures/ch23-invariant.svg" alt="Row 1, 2, 4, 4, 5, with a bar over the first k values for k = 0 to 4. After k values, m is the maximum of those k values and s is the sum of exp(x - m) over them: m = 2, s = 1; m = 4, s = 1.1353; m = 4, s = 2.1353; m = 5, s = 1.7855.">
<figcaption><b>Figure 23.8</b> The loop invariant on row 1. After \(k\) values, \((m, s)\) describes exactly the first \(k\) values.</figcaption>
</figure>

The online softmax was published by Milakov and Gimelshein in 2018. It is also the core of
attention kernels that never store the full matrix of scores, because they too see a row of
scores in pieces.

### 23.3.3 Merging 2 states

A GPU does not traverse a row from the left. Many threads each traverse a part of it, and the
parts must then be combined. The key observation combines 2 states as easily as it adds 1 value.

<div class="algo">
<p class="algo-name">Merge-Softmax(\((m_a, s_a), (m_b, s_b)\))</p>
<ol>
<li>\(m = \max(m_a, m_b)\)</li>
<li>if \(m = -\infty\) {</li>
<li style="--d:1">return \((-\infty, 0)\) <span class="c">// both states are empty</span></li>
<li>}</li>
<li>return \((m,\; s_a e^{m_a - m} + s_b e^{m_b - m})\)</li>
</ol>
</div>

<div class="theorem">

<span class="box-label">Theorem 23.4</span> If \\((m_a, s_a)\\) describes a set \\(A\\) and
\\((m_b, s_b)\\) describes a set \\(B\\) disjoint from \\(A\\), then
<span class="sc">Merge-Softmax</span> returns the state that describes \\(A \cup B\\).

</div>

<div class="proof">

*Proof.* The maximum of \\(A \cup B\\) is the larger of the 2 maxima, line 1. If both sets are
empty, so is the union, and line 3 returns the empty state. Otherwise \\(m\\) is finite, and line 5
moves each sum to \\(m\\) by the key observation: \\(s_a e^{m_a - m} = \sum_{x \in A} e^{x - m}\\),
and likewise for \\(B\\). If 1 of the sets is empty, its factor is \\(e^{-\infty} = 0\\) and its
sum 0 contributes nothing. Because \\(A\\) and \\(B\\) are disjoint, the 2 sums together contain
each term of \\(A \cup B\\) once.

</div>

The left half of Figure 23.9 merges the 2 halves of row 1. The values 2, 4 have the state
\\((4, 1.1353)\\), and the values 4, 5 have \\((5, e^{-1} + 1) = (5, 1.3679)\\). The merge gives
\\(m = 5\\) and \\(s = 1.1353 \cdot e^{-1} + 1.3679 \cdot e^{0} = 1.7855\\), the sum of Figure
23.7.

<figure>
<img src="../../figures/ch23-merge.svg" alt="Left: the softmax states of 2, 4 (m = 4, s = 1.1353) and of 4, 5 (m = 5, s = 1.3679) merge into m = 5 and s = 1.1353 times exp(-1) plus 1.3679 = 1.7855. Right: the Welford states of 1, 2 (n = 2, mean 1.5, q = 0.5) and of 3, 4 (n = 2, mean 3.5, q = 0.5) merge into n = 4, mean 2.5 and q = 0.5 + 0.5 + 4 times 2 times 2 / 4 = 5.">
<figcaption><b>Figure 23.9</b> Merging 2 partial states. Left: softmax states of the halves of row 1. Right: Welford states of the halves of row 0 (Section 23.3.5).</figcaption>
</figure>

2 consequences matter for the GPU. First, the result depends only on the union of the sets, so
merges can be grouped in any order: merging \\(A\\) with \\(B\\) and then with \\(C\\) gives the
same state as merging \\(A\\) with the merge of \\(B\\) and \\(C\\). An operation with this
property is called **associative**, and associativity is what allows a tree of merges instead of a
chain. Second, the test on line 2 is necessary. Without it, 2 empty states give
\\(0 \cdot e^{-\infty - (-\infty)}\\), and \\(-\infty - (-\infty)\\) is NaN in `f32`. On a GPU
many threads can hold empty states, and Section 23.5 shows what happens when line 2 is missing.

### 23.3.4 Welford's method

The same idea removes the extra pass of the variance and the cancellation at the same time. We keep
3 numbers while reading the row: the count \\(n\\) of values seen, their mean \\(\mu\\), and the
sum \\(q\\) of their squared deviations from \\(\mu\\). The variance is then \\(q / n\\). When a new
value \\(x\\) arrives, the mean moves, and every old deviation was measured from the old mean. As
with the maximum, the old sum can be corrected without visiting the old values again.

Let \\(\mu\\) be the mean of the first \\(k - 1\\) values and \\(\delta = x - \mu\\) the deviation
of the new value from it. The new mean is

\\[ \mu' = \frac{(k - 1)\mu + x}{k} = \mu + \frac{\delta}{k}. \\]

The old values now deviate from \\(\mu'\\) instead of \\(\mu\\). Shifting the centre of
\\(k - 1\\) values by \\(\mu' - \mu = \delta / k\\) adds \\((k - 1)(\delta / k)^2\\) to the sum of
their squared deviations, because the deviations from \\(\mu\\) add up to 0 and the cross terms
vanish. The new value adds \\((x - \mu')^2 = (\delta (k-1)/k)^2\\). Together,

\\[ q' = q + (k-1)\frac{\delta^2}{k^2} + \frac{(k-1)^2 \delta^2}{k^2} = q + \delta^2 \frac{k-1}{k}
= q + \delta \, (x - \mu'), \\]

where the last step uses \\(x - \mu' = \delta (k-1)/k\\). The update needs only the old mean and
the new mean.

<div class="algo">
<p class="algo-name">Welford-Update(\((n, \mu, q), x\))</p>
<ol>
<li>\(n = n + 1\)</li>
<li>\(\delta = x - \mu\) <span class="c">// deviation from the old mean</span></li>
<li>\(\mu = \mu + \delta / n\)</li>
<li>\(q = q + \delta \cdot (x - \mu)\) <span class="c">// old deviation times new deviation</span></li>
<li>return \((n, \mu, q)\)</li>
</ol>
</div>

Figure 23.10 applies the update to row 0, starting from \\((0, 0, 0)\\). The value 1 gives
\\(\delta = 1\\), \\(\mu = 1\\) and \\(q = 1 \cdot 0 = 0\\). The value 2 gives \\(\delta = 1\\),
\\(\mu = 1.5\\) and \\(q = 0 + 1 \cdot 0.5 = 0.5\\). The value 3 gives \\(\delta = 1.5\\),
\\(\mu = 2\\) and \\(q = 0.5 + 1.5 \cdot 1 = 2\\). The value 4 gives \\(\delta = 2\\),
\\(\mu = 2.5\\) and \\(q = 2 + 2 \cdot 1.5 = 5\\). The variance is \\(5 / 4 = 1.25\\), as in Figure
23.3.

<figure>
<img src="../../figures/ch23-welford.svg" alt="Welford's update on 1, 2, 3, 4. The deviations from the old mean are 1, 1, 1.5 and 2. The mean goes 1, 1.5, 2, 2.5 and q goes 0, 0.5, 2, 5, so the variance is 5 / 4 = 1.25. On 10,001 to 10,004 in float32 the same steps give q = 5.">
<figcaption><b>Figure 23.10</b> Welford's update on row 0. Every quantity added to \(q\) is a product of 2 deviations.</figcaption>
</figure>

Now run the same steps on 10,001 to 10,004 in `f32`. The means 10,001, 10,001.5, 10,002 and
10,002.5 are exact in `f32`, and the deviations are 1, 1, 1.5 and 2 as before, so \\(q\\) is again
exactly 5 and the variance 1.25. No square of a value near 10,000 is ever formed. Every number added
to \\(q\\) is of the size of the deviations, and those are small whenever the variance is small.
The sum of squares lacked exactly this property.

<div class="theorem">

<span class="box-label">Theorem 23.5</span> After <span class="sc">Welford-Update</span> has been
applied to \\(x_0, \dots, x_{k-1}\\), starting from \\((0, 0, 0)\\), the state is
\\(\big(k, \mu_k, \sum_{i<k} (x_i - \mu_k)^2\big)\\), where \\(\mu_k\\) is the mean of the \\(k\\)
values.

</div>

<div class="proof">

*Proof.* We use the invariant that after \\(k\\) updates the state is
\\((k, \mu_k, q_k)\\) with \\(q_k = \sum_{i<k} (x_i - \mu_k)^2\\).

*Initialization.* For \\(k = 0\\) the state \\((0, 0, 0)\\) has count 0 and \\(q_0 = 0\\), the empty
sum. The mean 0 of the empty state does not enter any later result: line 3 of the first update
gives \\(\mu = 0 + (x_0 - 0)/1 = x_0\\).

*Maintenance.* Assume the invariant after \\(k - 1\\) updates and apply the update to \\(x_{k-1}\\).
Line 1 makes the count \\(k\\), and line 3 computes \\(\mu_k = \mu_{k-1} + \delta / k\\), the new
mean derived above. For the sum, write each old deviation as
\\(x_i - \mu_k = (x_i - \mu_{k-1}) - \delta/k\\). Squaring and summing over \\(i < k - 1\\) gives
\\(q_{k-1} + (k-1)\delta^2/k^2\\), since \\(\sum_{i<k-1} (x_i - \mu_{k-1}) = 0\\). Adding the new
term \\((x_{k-1} - \mu_k)^2 = \delta^2 (k-1)^2/k^2\\) gives \\(q_{k-1} + \delta^2 (k-1)/k\\), which
is what line 4 adds, \\(\delta (x_{k-1} - \mu_k)\\).

*Termination.* After the last value the state is \\((n, \mu, q)\\) with \\(q / n = \sigma^2\\).

</div>

Welford's method costs 1 division per value, on line 3, where the sum of squares costs 1
multiplication. Section 23.7 measures what the division costs on a GPU.

### 23.3.5 Merging 2 Welford states

As with softmax, the threads of a GPU each hold the state of a part of the row, and the states must
be merged. The rule is due to Chan, Golub and LeVeque.

<div class="algo">
<p class="algo-name">Merge-Welford(\((n_a, \mu_a, q_a), (n_b, \mu_b, q_b)\))</p>
<ol>
<li>\(n = n_a + n_b\)</li>
<li>if \(n = 0\) {</li>
<li style="--d:1">return \((0, 0, 0)\) <span class="c">// both states are empty</span></li>
<li>}</li>
<li>\(\delta = \mu_b - \mu_a\)</li>
<li>\(\mu = \mu_a + \delta \, n_b / n\)</li>
<li>\(q = q_a + q_b + \delta^2 \, n_a n_b / n\)</li>
<li>return \((n, \mu, q)\)</li>
</ol>
</div>

<div class="theorem">

<span class="box-label">Theorem 23.6</span> If \\((n_a, \mu_a, q_a)\\) and \\((n_b, \mu_b, q_b)\\)
are the states of 2 disjoint sets \\(A\\) and \\(B\\), then <span class="sc">Merge-Welford</span>
returns the state of \\(A \cup B\\).

</div>

<div class="proof">

*Proof.* If both sets are empty, line 3 returns the empty state. Otherwise the mean of the union is
the weighted mean \\((n_a \mu_a + n_b \mu_b)/n = \mu_a + \delta n_b / n\\), line 6. Each value of
\\(A\\) deviates from the new mean \\(\mu\\) by its deviation from \\(\mu_a\\) minus
\\(\mu - \mu_a = \delta n_b / n\\). As in the proof of Theorem 23.5, the cross terms vanish, so
\\(A\\) contributes \\(q_a + n_a (\delta n_b / n)^2\\). Likewise \\(B\\) contributes
\\(q_b + n_b (\delta n_a / n)^2\\). The sum of the 2 extra terms is
\\(\delta^2 n_a n_b (n_b + n_a) / n^2 = \delta^2 n_a n_b / n\\), line 7.

</div>

The right half of Figure 23.9 merges the halves of row 0. The values 1, 2 have the state
\\((2, 1.5, 0.5)\\), and the values 3, 4 have \\((2, 3.5, 0.5)\\). Then \\(\delta = 2\\),
\\(\mu = 1.5 + 2 \cdot 2 / 4 = 2.5\\), and \\(q = 0.5 + 0.5 + 4 \cdot 2 \cdot 2 / 4 = 5\\), the
result of Figure 23.10. The last term of line 7 accounts for the distance between the 2 means: each
half measured its deviations from its own mean, and the union needs them from the common one.

<div class="exercises">

Exercises

23.7 Run <span class="sc">Online-Softmax</span> on row 0, 1, 2, 3, 4. How many times does line 4
execute, and how many times line 7?<br>
23.8 Merge the softmax states of the values 2 and of 4, 4, 5 (first compute each state), and check
the result against Figure 23.7.<br>
23.9 Apply <span class="sc">Welford-Update</span> to row 1 and check that \\(q / 4 = 1.1875\\).
Then merge the states of 2, 4, 4 and of 5 with <span class="sc">Merge-Welford</span>.

</div>

## 23.4 The parallel algorithm

### 23.4.1 1 block per row

We now distribute the work over the threads of a GPU. We recall the terms we need.

<div class="definition">

<span class="box-label">Definition 23.7</span> **Kernel, thread, block, grid.** A **kernel** is a
function that the GPU runs many times at once; each run is a **thread** with its own variables. A
**block** is a group of threads that run on the same streaming multiprocessor (SM), share a region
of fast memory, and can wait for one another. The **grid** is the set of all blocks of 1 launch.
Inside a block, a thread knows its number \\(t\\), and its block's number \\(b\\).

*Example.* For the 4,096 rows of Section 23.7, the grid has 4,096 blocks of 256 threads; for
\\(X\\), it has 2 blocks of 256 threads.

</div>

The rows are independent, and every row needs a reduction whose result all its outputs share. We
therefore give each row to 1 block: block \\(b\\) computes row \\(b\\). The threads of the block
divide the row between them in a fixed pattern. With \\(B = 256\\) threads per block, thread
\\(t\\) takes the columns \\(t, t + 256, t + 512, \dots\\) of its row. A row of 4,096 values gives
each thread \\(4{,}096 / 256 = 16\\) values; thread 5 takes columns 5, 261, 517, ..., 3,845
(Figure 23.11).

<figure>
<img src="../../figures/ch23-row-block.svg" alt="A row of 4,096 columns drawn as 16 groups of 256. Thread 5 reads column 5 of every group, 16 values in all. Below, block 1 of the example: threads 0 to 3 hold 2, 4, 4 and 5, and threads 4 to 255 hold no value.">
<figcaption><b>Figure 23.11</b> 1 block per row. Top: the 16 columns of thread 5 in a row of 4,096. Bottom: block 1 of the example, where threads 4 to 255 hold no value.</figcaption>
</figure>

The pattern is chosen for the memory. The threads of a block run in groups of 32, and the hardware
serves the reads of a group together.

<div class="definition">

<span class="box-label">Definition 23.8</span> **Warp, lane, coalesced access.** A **warp** is a
group of 32 consecutive threads of a block, threads \\(32w\\) to \\(32w + 31\\) for warp \\(w\\),
that execute each instruction together. The position \\(t \bmod 32\\) of a thread in its warp is
its **lane**. A memory access of a warp is **coalesced** when its 32 threads read 32 consecutive
4-byte values, 128 contiguous bytes, which the memory system serves together.

*Example.* A block of 256 threads has 8 warps. In the first step, warp 0 of a block reads columns
0 to 31 of its row, 128 contiguous bytes.

</div>

In every step of the pattern, the 32 threads of a warp read 32 consecutive columns, so every
access is coalesced. The alternative, giving thread \\(t\\) the 16 contiguous columns
\\(16t\\) to \\(16t + 15\\), would make the 32 threads of a warp read addresses 64 bytes apart.

On \\(X\\), block 1 has 256 threads for 4 values. Threads 0 to 3 receive 2, 4, 4 and 5, and
threads 4 to 255 receive nothing. Each thread runs <span class="sc">Online-Softmax</span> or
<span class="sc">Welford-Update</span> over its own columns, and a thread without columns keeps the
empty state. Every thread then holds the state of a different set of columns, the sets are
disjoint, and their union is the row. By Theorems 23.4 and 23.6, merging all 256 states gives the
state of the row. It remains to merge them quickly, and to give the result to all 256 threads,
because every thread writes outputs.

### 23.4.2 Merging inside a warp

<div class="definition">

<span class="box-label">Definition 23.9</span> **Warp shuffle.** A warp shuffle lets every lane of
a warp read a register of another lane of the same warp in 1 instruction, without shared memory and
without a barrier. In the **xor shuffle** with mask \\(k\\), lane \\(i\\) receives the value of lane
\\(i \oplus k\\), where \\(\oplus\\) is the bitwise exclusive or of the 2 lane numbers.

*Example.* With mask 16, lane 3 receives the value of lane \\(3 \oplus 16 = 19\\), and lane 19 the
value of lane 3: the 2 lanes exchange values.

</div>

With mask \\(k\\), the lanes form pairs \\(\{i, i \oplus k\}\\) that exchange values. If both lanes
of each pair merge the value they receive into their own, both hold the merge of the pair. Repeating
with masks 16, 8, 4, 2 and 1 leaves every lane holding the merge of all 32. The pattern is called a
**butterfly**, after the crossing lines of its diagram. Figure 23.12 shows it on 8 lanes, the
maximum part of block 1 of the example: 4 lanes hold 2, 4, 4, 5 and 4 lanes hold \\(-\infty\\).
After masks 4, 2 and 1, every lane holds 5.

<figure>
<img src="../../figures/ch23-butterfly.svg" alt="8 lanes hold 2, 4, 4, 5 and 4 times minus infinity. With mask 4 each lane takes the maximum of itself and the lane 4 away: 2, 4, 4, 5, 2, 4, 4, 5. With mask 2: 4, 5, 4, 5, 4, 5, 4, 5. With mask 1 every lane holds 5.">
<figcaption><b>Figure 23.12</b> A butterfly maximum on 8 lanes. A warp of 32 lanes needs 5 steps, with masks 16, 8, 4, 2 and 1.</figcaption>
</figure>

The butterfly is correct for the same reason as the merges. Before the first step, lane \\(i\\)
holds the state of its own set of columns. After the steps with masks 16 down to \\(k\\), lane
\\(i\\) holds the merge over all lanes whose numbers differ from \\(i\\) only in the bits of value
\\(k\\) or more. The step with mask \\(k / 2\\) merges this group with the group of lane
\\(i \oplus k/2\\), which is disjoint from it and together with it forms the group for
\\(k / 2\\). After the step with mask 1, the group is the whole warp. Chapter 8 used a different
shuffle, `shfl_down`, which leaves the result in lane 0 only; the xor shuffle leaves it in every
lane.

### 23.4.3 Merging across the warps of a block

Shuffles do not cross warps. The 8 warps of a block exchange their results through memory that the
whole block shares.

<div class="definition">

<span class="box-label">Definition 23.10</span> **Shared memory and barrier.** **Shared memory** is
a small, fast memory on each SM that the threads of 1 block can read and write, and no other block
can. A **barrier** is a point in a kernel that no thread of a block may pass until every thread of
the block has reached it.

*Example.* A block of 8 warps keeps 8 slots of shared memory, 1 per warp, 32 bytes in all.

</div>

Lane 0 of each warp writes its warp's result to its slot. A barrier makes sure all 8 slots are
written. Then every thread reads all 8 slots and merges them itself, so every thread ends with the
row's result. Reading 8 slots costs less than a second round of shuffles, and it needs no
broadcast afterwards (Figure 23.13).

<div class="algo">
<p class="algo-name">Block-Reduce(\(v, \otimes, \mathit{slots}\))</p>
<ol>
<li>for \(k \in (16, 8, 4, 2, 1)\) {</li>
<li style="--d:1">\(v = v \otimes \text{Shuffle-Xor}(v, k)\) <span class="c">// the butterfly of Section 23.4.2</span></li>
<li>}</li>
<li>if \(t \bmod 32 = 0\) {</li>
<li style="--d:1">\(\mathit{slots}[\lfloor t / 32 \rfloor] = v\) <span class="c">// lane 0 writes its warp's slot</span></li>
<li>}</li>
<li><span class="sc">Barrier</span>() <span class="c">// barrier 1: all slots written</span></li>
<li>\(r = \mathit{slots}[0] \otimes \mathit{slots}[1] \otimes \dots \otimes \mathit{slots}[7]\)</li>
<li><span class="sc">Barrier</span>() <span class="c">// barrier 2: all slots read</span></li>
<li>return \(r\)</li>
</ol>
</div>

Here \\(\otimes\\) is the merge: \\(\max\\) for the maximum, \\(+\\) for a sum,
<span class="sc">Merge-Softmax</span> or <span class="sc">Merge-Welford</span> for a state. Line 7
is needed because line 8 reads slots that other warps write on line 5. Line 9 is needed when the
same slots are used again by a second reduction: no warp may write a slot of the second reduction
while another warp still reads the first. Section 23.5 removes line 9 and shows the result.

<figure>
<img src="../../figures/ch23-block-reduce.svg" alt="8 warps each reduce their 32 values by shuffles, and lane 0 of each writes the result into its slot in shared memory. After barrier 1 every thread adds the 8 slots. Barrier 2 waits until every thread has read them, so that a later reduction may write the slots again.">
<figcaption><b>Figure 23.13</b> <span class="sc">Block-Reduce</span>: shuffles within each warp, 8 slots in shared memory, 2 barriers.</figcaption>
</figure>

### 23.4.4 The kernels

With these pieces, every thread of block \\(b\\) executes the following procedure for the online
softmax. We write \\(B = 256\\) for the block size and \\(C\\) for the row length.

<div class="algo">
<p class="algo-name">Softmax-Online-Kernel(\(X, P, C\))</p>
<ol>
<li>\((m, s) = (-\infty, 0)\)</li>
<li>for \(j = t, t + B, t + 2B, \dots\) while \(j \lt C\) {</li>
<li style="--d:1">update \((m, s)\) with \(x_{bj}\) <span class="c">// lines 3 to 8 of Online-Softmax</span></li>
<li>}</li>
<li>\((m, s) = \) <span class="sc">Block-Reduce</span>\(((m, s),\) <span class="sc">Merge-Softmax</span>\()\)</li>
<li>for \(j = t, t + B, t + 2B, \dots\) while \(j \lt C\) {</li>
<li style="--d:1">\(p_{bj} = e^{x_{bj} - m} / s\)</li>
<li>}</li>
</ol>
</div>

Lines 1 to 4 run the first pass of <span class="sc">Online-Softmax</span> on the thread's own
columns, line 5 merges the 256 states into the state of the row and gives it to every thread, and
lines 6 to 8 write the thread's outputs. Because the kernel calls <span class="sc">Block-Reduce</span>
once, barrier 2 is not needed and the code leaves it out. The layer-normalization kernel has the
same shape, with <span class="sc">Welford-Update</span> on line 3,
<span class="sc">Merge-Welford</span> on line 5, and \\(y_{bj} = (x_{bj} - \mu)\, \gamma_j /
\sqrt{q / C + \varepsilon} + \beta_j\\) on line 7. The 3-pass softmax calls
<span class="sc">Block-Reduce</span> twice, with \\(\max\\) after pass 1 and with \\(+\\) after
pass 2.

Each version still reads the row at least twice, once to reduce and once to write. A thread of a
row of 4,096 holds only 16 values, however, and a thread can keep 16 values in its registers.

<div class="definition">

<span class="box-label">Definition 23.11</span> **Register, fused kernel.** **Registers** are the
fastest storage of a GPU; each thread has its own, and a value in a register costs no memory
traffic to reuse. A **fused kernel** keeps its share of the row in registers between the reduction
and the output, so it reads each value from memory once.

*Example.* For rows of 4,096 values and 256 threads, a fused kernel holds 16 values per thread in
16 registers, 64 bytes per thread.

</div>

<div class="algo">
<p class="algo-name">Softmax-Fused-Kernel(\(X, P\))</p>
<ol>
<li>for \(k \in [0, 16)\) {</li>
<li style="--d:1">\(r_k = x_{b,\, kB + t}\) <span class="c">// the only read of the row</span></li>
<li>}</li>
<li>\(m = \) <span class="sc">Block-Reduce</span>\((\max_k r_k, \max)\)</li>
<li>for \(k \in [0, 16)\) {</li>
<li style="--d:1">\(r_k = e^{r_k - m}\)</li>
<li>}</li>
<li>\(s = \) <span class="sc">Block-Reduce</span>\((\sum_k r_k, +)\)</li>
<li>for \(k \in [0, 16)\) {</li>
<li style="--d:1">\(p_{b,\, kB + t} = r_k / s\)</li>
<li>}</li>
</ol>
</div>

The fused kernel needs no online softmax: once the values are in registers, a second and a third
pass over them cost no traffic, and the plain 3-pass order is the simplest. Line 6 overwrites each
value with its exponential, so line 10 only divides. The fused layer normalization, likewise, uses
the definition directly: 1 reduction for the mean, then 1 over the squared deviations
\\((r_k - \mu)^2\\), both from registers, which avoids cancellation without Welford's division.
The price is generality. The number of values per thread must be a constant when the kernel is
compiled, because registers cannot be indexed by a value known only at run time. A row of
65,536 values would also need 256 registers per thread for the row alone. Libraries compile fused
kernels for common row lengths and fall back to a 2-pass kernel for the others.

### 23.4.5 Counting the bytes

Figure 23.14 counts the traffic of each kernel per element, with 4 bytes per value.

<figure>
<img src="../../figures/ch23-passes.svg" alt="3 rows of bars. The 3-pass kernel reads each element 3 times, for the maximum, the sum and the output, and writes it once: 16 bytes. The online kernel reads it twice and writes once: 12 bytes. The fused kernel reads once and writes once: 8 bytes, because its values stay in registers.">
<figcaption><b>Figure 23.14</b> Memory traffic per element of the softmax kernels. The layer-normalization kernels move 12 bytes (sum of squares, Welford) and 8 bytes (fused).</figcaption>
</figure>

For a matrix of \\(R\\) rows of \\(C\\) values, a kernel with \\(b\\) bytes per element moves
\\(bRC\\) bytes, and needs at least \\(bRC / W\\) seconds on a memory of bandwidth \\(W\\). With
\\(R = C = 4{,}096\\) and \\(W = 102\\) GB/s:

| Kernel | Bytes per element | Bytes per matrix | Lower bound at 102 GB/s |
|---|---|---|---|
| softmax, 3 passes | 16 | 268 MB | 2.63 ms |
| softmax, online; layer norm, sum of squares or Welford | 12 | 201 MB | 1.97 ms |
| softmax or layer norm, fused | 8 | 134 MB | 1.32 ms |

Each kernel performs a few arithmetic operations per 4 bytes it reads, far fewer than a GPU can
perform in the time the memory needs to deliver the bytes, so the bytes set the time. The model
assumes that every read reaches main memory. A block reads its row of 16 KiB repeatedly within a
short time, and some of the repeated reads may be served by a cache instead; Section 23.7 measures
how much.

<div class="exercises">

Exercises

23.10 Which columns does thread 200 read in a row of 4,096? Which warp does it belong to, and
which lane?<br>
23.11 Carry out Figure 23.12 with the sum instead of the maximum, on the values 0.0498, 0.3679,
0.3679, 1 and 4 zeros.<br>
23.12 A row has 65,536 values and a block has 256 threads. How many values does each thread read,
and how many bytes does the 3-pass kernel move for 256 such rows?

</div>

## 23.5 Correctness

The threads of a block run at different speeds, and 1 warp can be several instructions ahead of
another. We now remove 1 safeguard at a time from the kernels and watch what goes wrong.

### 23.5.1 The second barrier

Remove line 9 of <span class="sc">Block-Reduce</span> and run the layer normalization by the sum
of squares, which calls <span class="sc">Block-Reduce</span> twice in a row, once for
\\(\sum x\\) and once for \\(\sum x^2\\), with the same 8 slots. Between the 2 calls, a warp does
almost nothing: 1 division and 5 shuffles. A fast warp can therefore leave the first call, finish
the shuffles of the second, and write its sum of squares into its slot while a slow warp is still
reading the sums of the first call.

Figure 23.15 follows this on row 94 of the test matrix of Section 23.7, whose 4,096 values lie
between 999 and 1,001. Each warp holds 512 values, so the sums in the slots are near 512,000 and
the sums of squares near \\(5.12 \times 10^8\\). Suppose warp 5 finishes reading the 8 sums early
and writes its sum of squares, 511,965,734, into slot 5, while warp 0 has read only slots 0 to 4.
Warp 0 then adds 511,965,734 instead of 511,982.8, and obtains the sum 515,549,697 instead of
4,095,945. Its mean is \\(515{,}549{,}697 / 4{,}096 = 125{,}866.6\\) instead of 999.99, its variance
\\(999{,}973.7 - 125{,}866.6^2\\) is negative, and its 512 outputs are NaN.

<figure>
<img src="../../figures/ch23-race.svg" alt="A timeline for row 94 without the second barrier. Warp 5 finishes reading the 8 sums early, shuffles its sum of squares and writes it, 511,965,734, into slot 5. Warp 0 is still reading the slots, reads the new value and computes the mean 125,866.6 instead of 999.99, a negative variance, and NaN outputs.">
<figcaption><b>Figure 23.15</b> Without barrier 2, warp 5 overwrites slot 5 while warp 0 still reads it.</figcaption>
</figure>

A failure of this kind is called a **race**: the result depends on the relative speed of the warps. On most
launches the slow warp finishes reading in time and the result is correct. We ran the kernel without
barrier 2 on the 4,096 rows of the test, 100 times, and compared every row with a reference:

```console
first bad row 94 error NaN
layernorm sum of squares: 11 of 100 runs wrong, 12 rows
```

11 of 100 launches produced at least 1 row of NaN, 12 rows in all, and the first wrong row of the
first failing launch was row 94. The other 89 launches were correct. A race is therefore hard to find:
a test that runs the kernel once passes 9 times in 10. On the example \\(X\\) the race
cannot change the output at all: only warp 0 of each block holds values, and the other warps
write sums of 0, which do not change any total (Exercise 23.13).

### 23.5.2 The empty state

Restore barrier 2, and remove instead lines 2 to 4 of <span class="sc">Merge-Softmax</span>, the
test for 2 empty states. On rows of 4,096, every thread holds 16 values and no state is empty, so
every result is still correct. On \\(X\\), threads 4 to 255 of each block hold the empty state
\\((-\infty, 0)\\). In the first step of the butterfly, lane 4 merges with lane 20, both empty:
\\(m = -\infty\\) and \\(s = 0 \cdot e^{-\infty - (-\infty)} + 0 \cdot e^{-\infty - (-\infty)}\\).
The difference \\(-\infty - (-\infty)\\) is NaN, so \\(s\\) is NaN. Every later merge that
includes lane 4 inherits the NaN, and after the 5 steps every lane of warp 0 holds
\\(s = \text{NaN}\\). The program, run with the test removed, prints:

```console
  softmax, online, X
         NaN      NaN      NaN      NaN
         NaN      NaN      NaN      NaN
```

Theorem 23.4 assumed that the merge returns the empty state for 2 empty states; without line 2 the
merge does not, and the proof no longer applies. The same holds for
<span class="sc">Merge-Welford</span>, where 2 empty states would compute \\(0 \cdot 0 / 0\\).

### 23.5.3 The kernels are correct

<div class="theorem">

<span class="box-label">Theorem 23.7</span> When <span class="sc">Softmax-Online-Kernel</span>
finishes, \\(p_{bj}\\) is the softmax of row \\(b\\) of \\(X\\) for every \\(b\\) and every
\\(j \in [0, C)\\).

</div>

<div class="proof">

*Proof.* Fix a block \\(b\\). By Theorem 23.3 applied to the columns of thread \\(t\\), after line 4
the state of thread \\(t\\) describes the set \\(S_t = \{x_{bj} : j \equiv t \pmod{B}\}\\), which is
empty when \\(t \ge C\\). The sets \\(S_0, \dots, S_{255}\\) are disjoint, and their union is row
\\(b\\). In the butterfly of line 5, each step merges 2 disjoint groups of lanes (Section 23.4.2),
so by Theorem 23.4 lane 0 of warp \\(w\\) holds the state of \\(\bigcup_{t \in \text{warp } w}
S_t\\) after the 5 steps. The merge is defined even for 2 empty states, because of line 2 of
<span class="sc">Merge-Softmax</span>. Lane 0 writes it to slot \\(w\\). Because of barrier 1,
every slot is written before any thread reads it, so every thread merges the 8 states of the 8
warps, which by Theorem 23.4 is the state \\((m, s)\\) of the whole row. The slots are not written
again in this kernel, so no barrier 2 is needed. Lines 6 to 8 then compute \\(e^{x_{bj} - m} / s\\)
for the thread's columns, which is the softmax by Theorem 23.1, and the columns of the 256 threads
cover the row. Different blocks touch different rows, so they do not interfere.

</div>

The same argument, with Theorems 23.5 and 23.6, proves the Welford kernel. For a kernel that calls
<span class="sc">Block-Reduce</span> twice with the same slots, the proof needs 1 more fact: no slot
is written by the second call before every thread has read it in the first. Barrier 2 provides
that fact, and Section 23.5.1 removed it.

<div class="rule-box">

<span class="box-label">Rule</span> Between the last read of a shared slot and the next write to
it, place a barrier that every thread of the block reaches.

</div>

<div class="exercises">

Exercises

23.13 Explain why removing barrier 2 can never change the outputs for \\(X\\), where only warp 0
of each block holds values.<br>
23.14 In the fused softmax, <span class="sc">Block-Reduce</span> is called with \\(\max\\) and then
with \\(+\\). Describe an interleaving without barrier 2 that gives a wrong maximum, and the effect
on the outputs.<br>
23.15 Why does the online kernel of Section 23.4.4 not need barrier 2, while the 3-pass kernel
does?

</div>

## 23.6 Implementation

### 23.6.1 Warp and block reductions

The warp maximum and the warp sum are lines 1 to 3 of <span class="sc">Block-Reduce</span>, with
\\(\max\\) and \\(+\\) as the merge.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:25:43}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:27:41}}
```

</div>
</div>

The loop runs for the masks 16, 8, 4, 2 and 1. In Rust it is a `while` loop, the form cuda-oxide
compiles for device loops (Section 9.7), and `warp::shuffle_xor_f32` is the xor shuffle. In C++,
`__shfl_xor_sync` takes a first argument, `0xffffffff`, that names the 32 lanes taking part; all of
them must execute the call. Both `f32::max` and `fmaxf` return the larger of 2 floats.

The block maximum adds lines 4 to 10:

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:45:65}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:43:59}}
```

</div>
</div>

Lane 0 writes the slot of its warp, the first call to `sync_threads` is barrier 1, the loop over
`w` is line 8, and the second call is barrier 2. Both barriers sit outside every `if`, so every
thread of the block reaches them. `block_sum` differs only in the merge. The 2 languages differ in
where the slots live. A cuda-oxide `#[device]` function cannot declare shared memory of its own, so
each kernel declares `static mut SLOTS: SharedArray<f32, 8>` and passes a raw pointer to the
helper. Every access through the pointer is `unsafe`, with a `// SAFETY:` comment that names which
threads touch the slots between which barriers. The C++ helper receives the same pointer from a
`__shared__` array of its kernel and needs no such annotation.

### 23.6.2 The 3-pass kernel

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:137:170}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:130:152}}
```

</div>
</div>

Block \\(b\\) starts at element `base = b × cols` of the matrix, the row-major position of row
\\(b\\). Each of the 3 loops starts at column `tid` and advances by `BLOCK`, the pattern of Figure
23.11; they are the 3 passes of <span class="sc">Softmax-3-Pass</span>, with `block_max` and
`block_sum` between them. The kernel works for any row length, including the 4 of \\(X\\).

The Rust kernel writes through a `DisjointSlice<f32>`, the output type of Chapter 13, whose safe
method lets each thread write only the element at its own global index. Here thread \\(t\\) of
block \\(b\\) writes the elements \\(b \cdot C + t + 256k\\), which no other thread writes, but
which are not its global index, so the kernel calls `get_unchecked_mut` and states that argument
in the `SAFETY` comment. The C++ kernel writes through a plain pointer. The attribute
`#[launch_contract(domain = 1, block = (256, 1, 1))]` records that the kernel expects 1-dimensional
blocks of 256 threads; the host's `prepare_softmax_three_pass` checks a launch configuration against
it before the first launch. The Rust kernel also reads through a slice, `input[base + j]`, and every
such read is checked against the slice length; Section 23.7.5 measures what the checks cost.

### 23.6.3 The online kernel

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:89:97}}

{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:172:227}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:79:92}}

{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:154:191}}
```

</div>
</div>

`merge_softmax` is <span class="sc">Merge-Softmax</span>, with the test for 2 empty states that
Section 23.5.2 removed. The first loop of the kernel is lines 1 to 4 of
<span class="sc">Softmax-Online-Kernel</span>: the `if` and the `else` are lines 3 to 8 of
<span class="sc">Online-Softmax</span>. The butterfly shuffles both numbers of the state in each
step, lane 0 of each warp writes its state to 2 slots, and after the barrier every thread merges
the 8 states. The final loop is lines 6 to 8. In Rust the state is 2 local variables updated by a
destructuring assignment, `(m, s) = merge_softmax(m, s, m2, s2)`; in C++ it is a 2-member
`struct`, and the slots are an array of that `struct`.

### 23.6.4 Welford's kernel

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:99:109}}

{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:308:372}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:94:110}}

{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:247:285}}
```

</div>
</div>

The first loop applies <span class="sc">Welford-Update</span> to the thread's columns; its 4 lines
are the 4 lines of the procedure. The state holds the count as a float, so that all 3 numbers can
be shuffled with the same instruction; counts up to \\(2^{24}\\) are exact in `f32`. After the warp
merge, lane 0 writes 3 numbers, which in Rust occupy 3 consecutive slots of a
`SharedArray<f32, 24>` and in C++ 1 slot of an array of `WelfordState`. The variance is
\\(q / C\\), and `rstd` is \\(1 / \sqrt{q / C + \varepsilon}\\).

### 23.6.5 The fused kernels

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:229:268}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:193:221}}
```

</div>
</div>

The 3 loops are lines 1 to 3, 5 to 7 and 9 to 11 of <span class="sc">Softmax-Fused-Kernel</span>.
The array `x` of `PER_THREAD = COLS / 256 = 16` values has a size fixed at compile time, which is
why the row length is the constant `COLS = 4096` and not an argument. Each loop has a constant trip
count and is fully unrolled, so that every `x[k]` names a fixed register. In Rust the loop is a
`while` loop marked `#[unroll]`, which Section 9.7 found necessary for cuda-oxide to keep a small
array in registers; in C++ it is a `for` loop under `#pragma unroll`. The fused layer normalization in the complete programs
has the same 3 loops, with the mean, the squared deviations and the outputs.

### 23.6.6 The host program

The host program first runs the kernels on \\(X\\), \\(X + 100\\) and \\(X + 10{,}000\\):

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
fn main() -> Result<(), Error> {
    // ...
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:535:557}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
int main()
{
    // ...
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:489:504}}
    // ...
}
```

</div>
</div>

A launch configuration of 2 blocks of 256 threads covers the 2 rows. In Rust, each
`prepare_<kernel>` call checks the configuration against the kernel's launch contract once, and the
launches that follow use the prepared configuration; in C++ the launch is written
`softmaxThreePass<<<2, BLOCK>>>(...)`. The output for the example:

```console
$ cargo oxide run
the 2 x 4 example X = [1 2 3 4; 2 4 4 5]
  softmax without the maximum, X
      0.0321   0.0871   0.2369   0.6439
      0.0279   0.2060   0.2060   0.5601
  softmax without the maximum, X + 100
         NaN      NaN      NaN      NaN
         NaN      NaN      NaN      NaN
  softmax, 3 passes, X
      0.0321   0.0871   0.2369   0.6439
      0.0279   0.2060   0.2060   0.5601
  softmax, 3 passes, X + 100
      0.0321   0.0871   0.2369   0.6439
      0.0279   0.2060   0.2060   0.5601
  softmax, online, X
      0.0321   0.0871   0.2369   0.6439
      0.0279   0.2060   0.2060   0.5601
  layer norm, sum of squares, X
     -1.3416  -0.4472   0.4472   1.3416
     -1.6059   0.2294   0.2294   1.1471
  layer norm, sum of squares, X + 10000
         NaN      NaN      NaN      NaN
         NaN      NaN      NaN      NaN
  layer norm, Welford, X + 10000
     -1.3416  -0.4472   0.4472   1.3416
     -1.6059   0.2294   0.2294   1.1471
```

Every line matches a hand computation of this chapter. Without the maximum, \\(X\\) gives \\(P\\)
of Figure 23.1, and \\(X + 100\\) gives NaN, as in Figure 23.4. With the maximum, \\(X\\) and
\\(X + 100\\) give the same \\(P\\), as Theorem 23.1 states. The sum of squares gives \\(Y\\) for
\\(X\\) and NaN for \\(X + 10{,}000\\), as in Figure 23.6, and Welford's method gives \\(Y\\) for
both. The program compares each of these with a softmax and a layer normalization computed in
`f64` on the CPU, and requires the NaNs where the analysis predicts them.

For the timed runs, a kernel takes about 2 ms, too short to time 1 launch reliably. The function
`time_ms` runs the kernel 100 times first, so that the GPU raises its clock, then times 7 batches
of 20 launches and returns the median batch time divided by 20.

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs:421:444}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu:325:348}}
```

</div>
</div>

<div class="exercises">

Exercises

23.16 `block_sum` in Rust starts its loop over the slots at `w = 0` with `s = 0.0`, while
`block_max` starts at `w = 1` with the value of slot 0. Why can `block_max` not start from
`m = 0.0`?<br>
23.17 Change `ROWS` and `COLS` to 16,384 rows of 1,024 columns. Which constant must change for the
fused kernels to stay correct?

</div>

## 23.7 Measurements

### 23.7.1 Platform and method

The measurements come from a Jetson Orin NX 16 GB, an integrated GPU with 8 SMs, 102 GB/s of
memory bandwidth and 2 MiB of L2 cache. The test matrices have 4,096 rows of 4,096 `f32` values, 64 MiB, far larger than the L2
cache. The softmax inputs are uniform in \\([-10, 10)\\); the layer-normalization inputs are uniform
in \\([999, 1001)\\), so that their mean is large compared with their spread, with per-column
\\(\gamma_j\\) and \\(\beta_j\\). The same softmax inputs are also arranged as 256 rows of 65,536
values. Each time is the median of 7 batches of 20 launches. Repeated runs differ by up to 10%,
and we quote 1 run of 3 in which neither program shared the GPU or the memory with other work.

### 23.7.2 Results

The Rust program printed, after the example:

```console
softmax of 4096 rows x 4096 columns, median of 7 x 20 runs
  kernel      bytes/elem      ms    GB/s   max error
  3 passes            16   2.025   132.6     1.37e-9
  online              12   2.255    89.3     1.36e-9
  fused                8   1.435    93.6     1.37e-9
softmax of 256 rows x 65536 columns, median of 7 x 20 runs
  kernel      bytes/elem      ms    GB/s   max error
  3 passes            16   3.985    67.4    1.22e-10
  online              12   3.015    66.8    1.18e-10
layer norm of 4096 rows x 4096 columns, values 1,000 + [-1, 1)
  kernel      bytes/elem      ms    GB/s   max error
  sum, sum x2         12   1.808   111.3      2.97e0
  Welford             12   2.112    95.3     1.88e-4
  fused                8   1.424    94.3     1.81e-4
PASSED
```

The GB/s column divides the modelled bytes of Section 23.4.5 by the measured time. The table
collects the times of both programs, and of the Rust program built with
`cargo oxide run --unchecked-indexing`, which Section 23.7.5 explains:

| Kernel | Bytes per element | Rust (ms) | Rust, unchecked (ms) | C++ (ms) | C++, modelled (GB/s) |
|---|---|---|---|---|---|
| softmax, 3 passes | 16 | 2.025 | 1.684 | 1.434 | 187.2 |
| softmax, online | 12 | 2.255 | 1.545 | 1.480 | 136.1 |
| softmax, fused | 8 | 1.435 | 1.446 | 1.417 | 94.8 |
| softmax, 3 passes, rows of 65,536 | 16 | 3.985 | 3.088 | 3.129 | 85.8 |
| softmax, online, rows of 65,536 | 12 | 3.015 | 2.453 | 2.373 | 84.8 |
| layer norm, sum of squares | 12 | 1.808 | 1.433 | 1.464 | 137.5 |
| layer norm, Welford | 12 | 2.112 | 1.902 | 1.920 | 104.9 |
| layer norm, fused | 8 | 1.424 | 1.424 | 1.445 | 92.9 |

<figure>
<img src="../../figures/ch23-measured.svg" alt="Paired bars of Rust and C++ times per 4,096 by 4,096 matrix on the measured GPU. Softmax rows of 4,096: 3 passes 2.025 and 1.434 ms, online 2.255 and 1.480 ms, fused 1.435 and 1.417 ms. Rows of 65,536: 3 passes 3.985 and 3.129 ms, online 3.015 and 2.373 ms. Layer norm: sum of squares 1.808 and 1.464 ms, Welford 2.112 and 1.920 ms, fused 1.424 and 1.445 ms. Dashed lines mark the modelled byte count at 102 GB/s.">
<figcaption><b>Figure 23.16</b> Measured time per 64 MiB matrix, Rust (upper bar) and C++ (lower bar). The dashed line of each kernel is its modelled bytes at 102 GB/s.</figcaption>
</figure>

The fused kernels match the model. They move 134 MB in 1.42 to 1.45 ms, 93 to 95 GB/s, which is
91 to 93% of the 102 GB/s of the memory, the fraction a well-behaved streaming kernel reaches on
this GPU. The other numbers need explanations, which the next 4 sections give.

### 23.7.3 Why 3 passes cost no more than 1 on rows of 4,096

The C++ 3-pass softmax takes 1.434 ms, the same as the fused kernel, although it reads each value 3
times. Its modelled rate, 187.2 GB/s, is 1.8 times the bandwidth of the memory, which is
impossible if every read reached the memory. Most of its second and third reads were therefore
served by the L2 cache.

The arithmetic agrees. An SM of this GPU holds at most 1,536 threads, 6 blocks of 256, so at most
\\(8 \cdot 6 = 48\\) rows are being processed at any moment. Each row occupies 16 KiB, so all rows
in progress fit in \\(48 \cdot 16 = 768\\) KiB, less than the 2 MiB of the L2 cache. A row read in
pass 1 is still in the cache when passes 2 and 3 read it, and the traffic to the memory is 8 bytes
per element, as for the fused kernel. The online kernel saves nothing here, 1.480 ms against
1.434 ms in C++, because the read it removes was a cache hit.

### 23.7.4 Rows of 65,536

With rows of 65,536 values, 256 KiB each, the 48 rows in progress would occupy
\\(48 \cdot 256\\) KiB = 12 MiB, 6 times the L2 cache. A row is no longer in the cache when the
next pass reads it, and every pass reaches the memory. The model of Section 23.4.5 now holds: the
C++ 3-pass kernel takes 3.129 ms for 268 MB, 85.8 GB/s, and the online kernel 2.373 ms for 201 MB,
84.8 GB/s. The ratio of the 2 times, \\(3.129 / 2.373 = 1.32\\), is the ratio of their traffic,
\\(16 / 12 = 1.33\\). For long rows, such as the output layer of a language model, the online
softmax saves a quarter of the time; for short rows the cache had already removed the pass.

### 23.7.5 Bounds checks in the Rust loops

The Rust 3-pass and online kernels are 41% and 52% slower than the C++ kernels on rows of 4,096
(2.025 against 1.434 ms, 2.255 against 1.480 ms), while the fused kernels run at the same speed.
The difference is in how the kernels read the row. A Rust kernel reads `input[base + j]` from a
slice, and every such read compares the index with the slice length and branches to a trap when it
is out of range. In the fused kernels the 16 reads are unrolled and independent, and the checks cost
nothing measurable. In the other kernels the reads sit in loops whose trip count is known only at
run time, and the checks are not free there.

cuda-oxide can remove the checks. Built with `cargo oxide run --unchecked-indexing`, the same
program runs the online kernel in 1.545 ms, the long-row kernels in 3.088 and 2.453 ms, and the sum
of squares in 1.433 ms, all within 5% of C++. The 3-pass kernel on short rows improves from 2.025
to 1.684 ms but remains 17% slower than C++. The flag makes every out-of-range index undefined
behaviour, exactly like `get_unchecked`, in every kernel of the program. The kernels of this
chapter read only indices \\(b \cdot C + j\\) with \\(b < R\\) and \\(j < C\\), so they stay in
range, but the compiler no longer checks that claim. A per-kernel form,
`#[kernel(unchecked_indexing)]`, limits the trust to the kernels that need the speed.

### 23.7.6 Welford's division, and the accuracy

The Welford kernel moves the same 12 bytes per element as the sum of squares but takes
\\(1.920 - 1.464 = 0.456\\) ms longer in C++. The difference is computation. Per value, the sum of
squares performs 2 additions, 1 of them with a multiplication; Welford's update performs 3
subtractions or additions, 1 multiplication-addition, and a division, which a GPU computes with a
sequence of several instructions. The kernel is close enough to the memory limit that this extra
arithmetic shows in the time. The fused layer normalization avoids both costs: it moves 8 bytes and
computes the variance from registers in 2 inexpensive passes.

The error column shows the cancellation of Section 23.2.5 on the GPU. The sum of squares reaches an
error of 2.97 on outputs whose typical size is 1, so its results are useless, at the same speed as
the correct kernels. Welford's method and the fused kernel stay within \\(1.9 \times 10^{-4}\\) of
the `f64` reference. The remaining error has a definite source. The values lie near 1,000, where
`f32` numbers are \\(6.1 \times 10^{-5}\\) apart, so the mean is exact only to a few times that
spacing. Dividing by the standard deviation, \\(\sqrt{0.33} = 0.58\\), multiplies the error by
1.7. The softmax errors, about \\(10^{-9}\\) on probabilities near \\(2.4 \times 10^{-4}\\), are at
the level of `f32` rounding.

<div class="exercises">

Exercises

23.18 With 4 bytes per value and 102 GB/s, how long must the fused layer normalization take at
least for 16,384 rows of 1,024 values? Would the 3-pass softmax be served by the L2 cache at that
row length?<br>
23.19 Estimate the times of the 3-pass and online softmax for 64 rows of 128,000 values on a GPU
with 3.35 TB/s of memory bandwidth and 50 MB of L2 cache.

</div>

## 23.8 The complete programs

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch23_softmax_layernorm/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch23_softmax_layernorm/softmax_layernorm.cu}}
```

</div>
</div>

## Common mistakes

- **Computing the exponential before subtracting the maximum.** Inputs above 88.72 overflow `f32`,
  and the softmax of the row becomes NaN.
- **Computing the variance as the mean of the squares minus the square of the mean.** The result
  loses all its digits when the mean is large compared with the spread, and can be negative, which
  makes the square root NaN.
- **Merging 2 empty states without a test.** \\(-\infty - (-\infty)\\) is NaN for the softmax
  merge, and \\(0 / 0\\) is NaN for the Welford merge. Threads without values hold empty states
  whenever a row is shorter than the block.
- **Leaving out the second barrier of a block reduction.** A warp that finishes early overwrites a
  slot that another warp still reads when 2 reductions use the same slots. The result is wrong on a
  few launches in a hundred.
- **Counting bytes without asking where they come from.** A model that sends every read to main
  memory overestimates repeated reads of a short row, which the L2 cache serves.
- **Leaving out \\(\varepsilon\\).** A row of equal values has variance 0, and the division gives
  NaN.

## Summary

- Softmax divides the exponential of each value by the sum of the exponentials of its row. Layer
  normalization subtracts the row mean and divides by the standard deviation, with a small
  \\(\varepsilon\\) under the square root, then applies a scale and a shift per column.
- Subtracting the row maximum leaves softmax unchanged (Theorem 23.1) and keeps every exponential in
  \\((0, 1]\\). The variance computed as \\(\overline{x^2} - \mu^2\\) suffers from cancellation:
  \\(-8\\) instead of 1.25 for 10,001 to 10,004 in `f32`.
- A sum of exponentials relative to 1 maximum is moved to a larger one by 1 multiplication. The
  online softmax uses this to find the maximum and the sum in 1 pass, and Welford's method applies
  the same idea to the mean and the sum of squared deviations.
- Partial states of disjoint parts of a row merge into the state of their union. The merges are
  associative, which allows a tree of merges, and must treat 2 empty states separately.
- 1 block per row reads coalesced columns, merges within each warp by an xor butterfly and across
  warps through 8 shared slots and 2 barriers. Removing the second barrier produced NaN rows in 11
  of 100 launches.
- Fused kernels keep the row in registers, read it once, and reached 91 to 93% of the 102 GB/s of
  the measured GPU. On rows of 4,096 the L2 cache served the extra passes of the other kernels; on
  rows of 65,536 it could not, and the online softmax was 1.32 times as fast as 3 passes, the ratio
  of their traffic.

## Exercises

1. Compute by hand the softmax of the row 0, 0, 0, 10, with and without subtracting the maximum.
   What fraction of the total does the last value receive?
2. Apply Welford's update by hand to 1,000.5, 999.5, 1,000.5, 999.5. Then compute the mean of the
   squares minus the square of the mean, rounding each square to the nearest 0.0625. Compare both
   variances with the exact value, 0.25.
3. Prove that <span class="sc">Merge-Softmax</span> is commutative: merging \\(a\\) with \\(b\\)
   gives the same state as merging \\(b\\) with \\(a\\).
4. Write <span class="sc">Online-Softmax</span> for a row that arrives in pieces of 4 values, using
   <span class="sc">Merge-Softmax</span> once per piece. How many exponentials does it compute per
   value?
5. Write a softmax kernel that assigns 1 warp of 32 threads to each row, with no shared memory and
   no barrier, and measure it against the fused kernel on rows of 1,024.
6. The online kernel computes 1 exponential per value in its first loop and 1 in its second. Count
   the exponentials of the 3-pass kernel. What does the online kernel save, if not arithmetic?
7. Replace the division in <span class="sc">Welford-Update</span> by a multiplication with a
   reciprocal computed once per count. Is the result still exact on Figure 23.10? Measure the
   effect on the kernel.
8. Build the Rust program with `--unchecked-indexing` and run it on your GPU. Which kernels change,
   and by how much?

## Sources and further reading

- M. Milakov and N. Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867,
  2018. The online softmax of Section 23.3.2.
- B. P. Welford, *Note on a method for calculating corrected sums of squares and products*,
  Technometrics 4(3), 1962. The update of Section 23.3.4.
- T. F. Chan, G. H. Golub and R. J. LeVeque, *Updating formulae and a pairwise algorithm for
  computing sample variances*, Stanford technical report STAN-CS-79-773, 1979. The merge of Section
  23.3.5.
- J. L. Ba, J. R. Kiros and G. E. Hinton, *Layer normalization*, arXiv:1607.06450, 2016.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002. Chapter 1 on
  cancellation, and chapter 4 on the accuracy of summation.
- T. Dao et al., *FlashAttention: fast and memory-efficient exact attention with IO-awareness*,
  NeurIPS 2022. The online softmax applied inside attention.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed.,
  MIT Press, 2022. Loop invariants and the pseudocode conventions of this chapter.
- A. Pathak, *An Introduction to Tiled Parallel Programming, Simplified*, 2026. The worked-example
  method of this chapter, applied to the matrix product.
