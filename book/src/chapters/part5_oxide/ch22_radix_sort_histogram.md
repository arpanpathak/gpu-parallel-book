<img class="plate" src="../../art/ch22.svg" alt="Bins, a crab, stands on the sand beside 4 crates labeled with the 2-bit digits 00, 01, 10 and 11. Chalked on each crate is the number of keys with that digit, 1, 2, 3 and 2, and above each crate the place where its first key goes, 0, 1, 3 and 6. Bins holds up a pebble marked 5, binary 101, on its way to crate 01. A tag reads count, scan, then place.">

# Chapter 22: Radix Sort and Histograms

Sorting puts a list of keys in increasing order. Most sorting algorithms compare keys 2 at a time,
and the order of the comparisons depends on the keys, which makes these algorithms awkward to run on
thousands of threads. A **radix sort** compares nothing. It sorts integer keys by their digits, 1
digit per pass, and in each pass, every key computes its output position from 2 counts: the number
of keys that have a smaller digit, and the number of keys before it that have the same digit. Both
counts come from the scan of Chapter 21, and the first one also comes from a histogram, the count of
the keys with each digit value.

In this chapter, we derive a radix sort on 8 keys of 3 bits, small enough that every pass can be
checked by hand. We start with passes on 1 bit, each of which consists of 2 stream compactions, and
we show why every pass must keep keys with equal digits in their input order. We then count keys
with atomic operations, measure what happens when many threads count into the same bin, and remove
most of that contention with private bins in shared memory. Finally, we use a histogram and an
exclusive scan to compute the positions of a pass on a digit of 4 bits, and we measure a sort that
runs 6.5 times as fast as the sort by single bits.

<div class="covers" markdown="1">

This chapter covers

- radix sort, least significant digit first, worked by hand on 8 keys of 3 bits
- a split on 1 bit as 2 compactions, and why each pass must be stable
- a proof that a sort by stable passes is correct
- histograms with atomic operations, contention, private bins and warp aggregation
- a pass on a digit of several bits from a histogram, an exclusive scan and ranks
- measured histograms and sorts of 16,777,216 keys

</div>

The code of the chapter is in
[`code/oxide/ch22_radix_sort_histogram/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/oxide/ch22_radix_sort_histogram)
for Rust and in
[`code/cuda/ch22_radix_sort_histogram/`](https://github.com/arpanpathak/gpu-parallel-book/tree/main/code/cuda/ch22_radix_sort_histogram)
for C++. The programs use the warp-shuffle scan of Chapter 21 (Section 21.10.3) without change. In
Chapters 5 and 8, we measured histograms of bytes in C++ and in Rust, and we cite those results
where we reuse them.

## 22.1 The problem

### 22.1.1 A worked example

Throughout the chapter, we sort the same 8 keys, which we call \\(K\\):

\\[ K = (5,\ 2,\ 7,\ 2,\ 0,\ 6,\ 3,\ 5). \\]

As in Chapter 21, we number the positions from 0, so that \\(K_0 = 5\\) is the first key and
\\(K_7 = 5\\) the last. Every key is below 8, so it fits in 3 bits. For example, 5 is 101 in
binary, 2 is 010, and 7 is 111. The sorted list is \\((0, 2, 2, 3, 5, 5, 6, 7)\\). Figure 22.1
shows the keys with their bits, and the sorted keys with the position from which each one came.
The key 2 occurs twice, at positions 1 and 3, and so does the key 5, at positions 0 and 7. We
follow both pairs through the chapter.

<figure>
<img src="../../figures/ch22-keys.svg" alt="The keys 5, 2, 7, 2, 0, 6, 3, 5 with their binary digits 101, 010, 111, 010, 000, 110, 011, 101, and the sorted keys 0, 2, 2, 3, 5, 5, 6, 7, labeled with the positions they came from: 4, 1, 3, 6, 0, 7, 5, 2.">
<figcaption><b>Figure 22.1</b> The 8 keys, their bits, and the sorted order.</figcaption>
</figure>

### 22.1.2 Definitions

A radix sort works on the bits of a key, so we first state precisely how a bit, and a group of
bits, is read from a key.

<div class="definition">

<span class="box-label">Definition 22.1</span> **Key, bit and digit.** A key is an unsigned
integer of \\(B\\) bits. Bit \\(b\\) of key \\(k\\) is \\(\lfloor k / 2^b \rfloor \bmod 2\\), and
bit 0 is the least significant. A digit of \\(r\\) bits at shift \\(s\\) is
\\(\lfloor k / 2^s \rfloor \bmod 2^r\\), the bits \\(s\\) to \\(s + r - 1\\) read as a number.

*Example.* \\(B = 3\\). The key 6 = 110 has bit 0 equal to 0 and bit 2 equal to 1; its digit of 2
bits at shift 0 is 10, that is 2.

</div>

Dividing a key by \\(2^b\\) and rounding down removes its lowest \\(b\\) bits, and taking the
remainder modulo 2 then keeps only the lowest of the remaining bits, which is bit \\(b\\) of the
original key. For the key 6 and \\(b = 2\\), we get \\(\lfloor 6 / 4 \rfloor = 1\\) and
\\(1 \bmod 2 = 1\\). A digit is read in the same way, except that the remainder modulo \\(2^r\\)
keeps the lowest \\(r\\) bits instead of 1.

| Symbol | Meaning | Key 6 = 110 |
|---|---|---|
| \\(B\\) | number of bits of every key | 3 |
| \\(b\\) | number of a bit, 0 for the least significant | 2 |
| \\(\lfloor k / 2^b \rfloor \bmod 2\\) | bit \\(b\\) of key \\(k\\) | \\(\lfloor 6/4 \rfloor \bmod 2 = 1\\) |
| \\(r\\) | number of bits of a digit | 2 |
| \\(s\\) | shift: the number of the lowest bit of the digit | 0 |
| \\(\lfloor k / 2^s \rfloor \bmod 2^r\\) | the digit of \\(r\\) bits at shift \\(s\\) | \\(6 \bmod 4 = 2\\) |

The second definition names the property on which the whole chapter depends.

<div class="definition">

<span class="box-label">Definition 22.2</span> **Stable.** A rearrangement of a list by some
property of its keys is stable if any 2 keys with equal property keep the relative order they had
in the input.

*Example.* Sorting \\(K\\) by value is stable if the 2 from position 1 comes before the 2 from
position 3, as in Figure 22.1.

</div>

Stability is invisible when the keys are plain numbers, because 2 equal numbers look the same. It
becomes visible when each key carries a payload, such as the index of a record. As we show in
Section 22.4, stability is also required inside the radix sort itself.

<div class="exercises">

Exercises

22.1 Write the keys \\((6, 1, 4, 4, 7, 0, 3, 1)\\) in binary, and give their digits of 2 bits at
shift 0.<br>
22.2 How many bits do keys below 1,000 need? How many digits of 4 bits?

</div>

## 22.2 Sorting by 1 bit at a time

### 22.2.1 The sequential algorithm

A radix sort processes the bits from the least significant to the most significant. In each pass,
it reads the keys in their current order and splits them into 2 lists: first the keys whose
current bit is 0, and then the keys whose current bit is 1, each list in the order in which the
keys were read. In the following procedure, \\(Z\\) is the list of keys whose bit is 0, and \\(O\\)
is the list of keys whose bit is 1.

<div class="algo">
<p class="algo-name">Radix-Sort-Bits(\(K, n, B\))</p>
<ol>
<li><b class="kw">for</b> \(b \in [0, B)\) { <span class="c">// 1 pass per bit, bit 0 first</span></li>
<li style="--d:1">\(Z\) = empty list, \(O\) = empty list</li>
<li style="--d:1"><b class="kw">for</b> \(i \in [0, n)\) {</li>
<li style="--d:2"><b class="kw">if</b> bit \(b\) of \(K_i\) is 0: append \(K_i\) to \(Z\)</li>
<li style="--d:2"><b class="kw">else</b>: append \(K_i\) to \(O\)</li>
<li style="--d:1">}</li>
<li style="--d:1">\(K\) = \(Z\) followed by \(O\) <span class="c">// the split on bit \(b\)</span></li>
<li>}</li>
</ol>
</div>

Lines 3 to 7 form 1 pass, which we call the **split** on bit \\(b\\). Figure 22.2 shows the 3
passes on our keys. After the split on bit 0, the even keys \\((2, 2, 0, 6)\\) come before the odd
keys \\((5, 7, 3, 5)\\). After the split on bit 1, the keys are ordered by their 2 lowest bits.
First comes 0, whose 2 lowest bits are 00, then 5 and 5 (01), then 2, 2 and 6 (10), and finally 7
and 3 (11). After the split on bit 2, the keys are sorted.

<figure>
<img src="../../figures/ch22-passes.svg" alt="Keys 5, 2, 7, 2, 0, 6, 3, 5. After a split on bit 0: 2, 2, 0, 6, 5, 7, 3, 5. After bit 1: 0, 5, 5, 2, 2, 6, 7, 3. After bit 2: 0, 2, 2, 3, 5, 5, 6, 7, sorted. Binary digits are written under every key.">
<figcaption><b>Figure 22.2</b> A radix sort of the 8 keys in 3 passes, 1 bit per pass. The color shows the bit of the next pass.</figcaption>
</figure>

Line 4 or line 5 executes once per key and pass, that is, \\(nB\\) times: \\(8 \cdot 3 = 24\\)
times for our example, and \\(32n\\) times for keys of 32 bits. The work therefore grows linearly
with \\(n\\), and no 2 keys are ever compared.

### 22.2.2 Why the least significant bit comes first

The last pass decides the order by the most significant bit, and the order established by every
earlier pass must survive it. It survives because each split keeps the order of the keys within
each of its 2 lists. Consider the 2 keys 2 and 3 after the pass on bit 1 in Figure 22.2. The key
2 = 010 comes before 3 = 011, because the pass on bit 0 put the even keys first. The pass on bit 2
sees the same bit, 0, in both keys, and so it keeps 2 before 3. The following theorem states this
argument in general.

<div class="theorem">

<span class="box-label">Theorem 22.1</span> If every pass of Radix-Sort-Bits is a stable split,
then after the pass on bit \\(b\\), the keys are in increasing order of \\(k \bmod 2^{b+1}\\),
their bits 0 to \\(b\\), and keys with equal bits 0 to \\(b\\) are in input order. In particular,
after the pass on bit \\(B - 1\\) the keys are sorted, and the sort is stable.

</div>

The quantity \\(k \bmod 2^{b+1}\\) is the number formed by the lowest \\(b + 1\\) bits of the key,
bits 0 to \\(b\\), which are exactly the bits that the passes so far have looked at. After the pass
on bit 1, for example, it is \\(k \bmod 4\\), the 2 lowest bits. In the order of Figure 22.2 after
that pass, \\((0, 5, 5, 2, 2, 6, 7, 3)\\), these values are \\((0, 1, 1, 2, 2, 2, 3, 3)\\), which
are indeed in increasing order.

<div class="proof">

*Proof.* We show that the loop of lines 1 to 8 maintains the statement as an invariant, writing
\\(v_b(k) = k \bmod 2^{b+1}\\).

*Initialization.* Before the first pass, take \\(b = -1\\): \\(v_{-1}(k) = 0\\) for every key, and
the keys are in input order.

*Maintenance.* Assume the statement holds after the pass on bit \\(b - 1\\), and take 2 keys
\\(k\\) and \\(k'\\). If their bits \\(b\\) differ, the split on bit \\(b\\) puts the key with bit
0 first. Bit \\(b\\) is the most significant of bits 0 to \\(b\\), so that key has the smaller
\\(v_b\\), and the order is correct. If their bits \\(b\\) are equal, the split is stable and keeps
the order of the previous pass, which is the order of \\(v_{b-1}\\), with input order among equal
values. Since \\(v_b(k) - v_b(k') = v_{b-1}(k) - v_{b-1}(k')\\) when bit \\(b\\) is equal, that
order is the order of \\(v_b\\), with input order among equal values.

*Termination.* After the pass on bit \\(B - 1\\), \\(v_{B-1}(k) = k\\) for every key of \\(B\\)
bits, so the keys are sorted, and equal keys are in input order.

</div>

| Symbol | Meaning | Keys 2 and 3, \\(b = 2\\) |
|---|---|---|
| \\(v_b(k)\\) | the lowest \\(b + 1\\) bits of \\(k\\), read as a number | \\(v_2(2) = 2\\), \\(v_2(3) = 3\\) |
| \\(v_{b-1}(k)\\) | the same before the pass on bit \\(b\\) | \\(v_1(2) = 2\\), \\(v_1(3) = 3\\) |

The case of equal bits can be checked on the keys 2 and 3 in the pass on bit 2. Both have bit 2
equal to 0, so their difference does not change: \\(v_2(3) - v_2(2) = 3 - 2 = 1\\), which equals
\\(v_1(3) - v_1(2)\\). The order of the previous pass, 2 before 3, is therefore also the correct
order after this pass. The proof uses stability in exactly this 1 place, the case of equal bits.
In Section 22.4, we show what happens to our keys when that case fails.

<div class="exercises">

Exercises

22.3 Sort \\((6, 1, 4, 4, 7, 0, 3, 1)\\) with Radix-Sort-Bits, writing the keys after each pass.<br>
22.4 Sort \\(K\\) with the most significant bit first, by the same splits. Where does the result
differ from the sorted list, and why?

</div>

## 22.3 A split is 2 compactions

### 22.3.1 Positions from a scan

A split on the GPU cannot append keys to lists, because the threads would have to agree on the
order of the appends. Instead, every key computes its own output position, as in the stream
compaction of Section 21.8. The keys whose bit \\(b\\) is 0 form 1 compaction. Their flag is
\\(z_i = 1\\) when bit \\(b\\) of key \\(i\\) is 0, and key \\(i\\) goes to position \\(e_i\\), the
exclusive scan of these flags. The keys whose bit is 1 form a second compaction, written after the
keys of the first one. We write \\(Z\\) for the number of keys whose bit is 0.

The second compaction needs no second scan. Of the \\(i\\) keys before position \\(i\\), \\(e_i\\)
have bit 0, so the remaining \\(i - e_i\\) have bit 1. A key with bit 1 must therefore come after
all \\(Z\\) keys with bit 0 and after the \\(i - e_i\\) keys with bit 1 that precede it, and so it
goes to position

\\[ Z + i - e_i. \\]

| Symbol | Meaning | Key 7 at position 2, bit 0 |
|---|---|---|
| \\(i\\) | input position of the key | 2 |
| \\(z_i\\) | 1 if bit \\(b\\) of the key is 0, else 0 | 0 |
| \\(e_i\\) | keys with bit 0 before position \\(i\\): the exclusive scan of \\(z\\) | 1 |
| \\(i - e_i\\) | keys with bit 1 before position \\(i\\) | 1 |
| \\(Z\\) | number of keys with bit 0 in the whole list | 4 |

Figure 22.3 works out the split on bit 0 of our keys. The flags of the keys with bit 0 are
\\((0, 1, 0, 1, 1, 1, 0, 0)\\), their exclusive scan is \\((0, 0, 1, 1, 2, 3, 4, 4)\\), and
\\(Z = 4\\). The key 5 at position 0 has bit 1 and goes to \\(4 + 0 - 0 = 4\\). The key 7 at
position 2 also has bit 1 and goes to \\(4 + 2 - 1 = 5\\). The key 2 at position 3 has bit 0 and
goes to \\(e_3 = 1\\).

<figure>
<img src="../../figures/ch22-split.svg" alt="A split of 5, 2, 7, 2, 0, 6, 3, 5 on bit 0 as 2 compactions. Flags for bit 0 equal to 0: 0, 1, 0, 1, 1, 1, 0, 0, exclusive scan 0, 0, 1, 1, 2, 3, 4, 4. Flags for bit 0 equal to 1: 1, 0, 1, 0, 0, 0, 1, 1, with 4 plus their exclusive scan 4, 5, 5, 6, 6, 6, 6, 7. Positions 4, 0, 5, 1, 2, 3, 6, 7; output 2, 2, 0, 6, 5, 7, 3, 5.">
<figcaption><b>Figure 22.3</b> The split on bit 0 as 2 compactions. The second row of positions is \(4 + i - e_i\).</figcaption>
</figure>

<div class="key-observation">

<span class="box-label">Key observation</span> 1 exclusive scan of the zero flags places both
groups of a split: a key with bit 0 goes to \\(e_i\\), and a key with bit 1 goes to
\\(Z + i - e_i\\), where \\(Z\\) is the number of keys with bit 0. Each group is a stream
compaction, so the split inherits the stability of compaction.

</div>

### 22.3.2 The algorithm

In the following procedure, \\(s\\) is the inclusive scan of the flags, from which line 5 recovers
the exclusive scan \\(e_i\\), and line 3 reads \\(Z\\) as the last inclusive value.

<div class="algo">
<p class="algo-name">Parallel-Split(\(K, out, n, b\))</p>
<ol>
<li><b class="kw">for</b> every \(i \in [0, n)\) in parallel: \(z_i = 1\) <b class="kw">if</b> bit \(b\) of \(K_i\) is 0, <b class="kw">else</b> 0</li>
<li>\(s\) = the inclusive scan of \(z\) <span class="c">// Scan of Section 21.7</span></li>
<li>\(Z = s_{n-1}\) <span class="c">// the keys with bit 0</span></li>
<li><b class="kw">for</b> every \(i \in [0, n)\) in parallel {</li>
<li style="--d:1">\(e_i = s_i - z_i\) <span class="c">// the exclusive scan</span></li>
<li style="--d:1"><b class="kw">if</b> \(z_i = 1\): \(out_{e_i} = K_i\)</li>
<li style="--d:1"><b class="kw">else</b>: \(out_{Z + i - e_i} = K_i\)</li>
<li>}</li>
</ol>
</div>

<div class="theorem">

<span class="box-label">Theorem 22.2</span> Parallel-Split writes every key exactly once, the
positions form a permutation of \\([0, n)\\), and the split is stable.

</div>

<div class="proof">

*Proof.* By Theorem 21.2, the keys with bit 0 receive the distinct positions \\([0, Z)\\) in input
order. A key with bit 1 at position \\(i\\) receives \\(Z\\) plus the number of keys with bit 1
before it. For 2 such keys at \\(i < j\\), that number grows by at least 1. These positions are
therefore distinct, increase with the input position, and run from \\(Z\\) to \\(Z + (n - Z) - 1 = n - 1\\).
The 2 ranges \\([0, Z)\\) and \\([Z, n)\\) do not overlap and together hold \\(n\\) positions.

</div>

In our example, the 4 keys with bit 1 receive the positions 4, 5, 6 and 7, in the order of their
input positions 0, 2, 6 and 7, and the 4 keys with bit 0 receive the positions 0 to 3. A pass of
Parallel-Split moves every key once, as the sequential split does. In addition, the parallel split
writes the flags, scans them and reads them back, and we count the cost of this extra traffic in
Section 22.5.

<div class="exercises">

Exercises

22.5 Carry out Parallel-Split on bit 1 of \\((2, 2, 0, 6, 5, 7, 3, 5)\\), the output of Figure
22.3, and check the result against Figure 22.2.<br>
22.6 Why does line 7 not need the inclusive scan of the flags of the bit-1 keys?

</div>

## 22.4 Stability

### 22.4.1 An unstable split

Suppose that a split puts the keys with bit 0 first, as it must, but writes the keys with bit 1 in
some other order, for example in reverse. On its own bit, this is still a correct split. We apply
it to bit 1 of the keys after the first pass, \\((2, 2, 0, 6, 5, 7, 3, 5)\\). The keys whose bit 1
is 0 are \\((0, 5, 5)\\), and they stay in order. The keys whose bit 1 is 1 are
\\((2, 2, 6, 7, 3)\\), and they come out reversed, as \\((3, 7, 6, 2, 2)\\). The pass on bit 2,
which is stable again, then gives \\((0, 3, 2, 2, 5, 5, 7, 6)\\), as Figure 22.4 shows.

<figure>
<img src="../../figures/ch22-unstable.svg" alt="From 2, 2, 0, 6, 5, 7, 3, 5, a stable split on bit 1 gives 0, 5, 5, 2, 2, 6, 7, 3 and then 0, 2, 2, 3, 5, 5, 6, 7 after bit 2. An unstable split that reverses the one-bit keys gives 0, 5, 5, 3, 7, 6, 2, 2 and then 0, 3, 2, 2, 5, 5, 7, 6 after bit 2: positions 1, 3, 6 and 7 are wrong.">
<figcaption><b>Figure 22.4</b> A stable and an unstable split on bit 1, each followed by the split on bit 2.</figcaption>
</figure>

The keys 3 = 011 and 2 = 010 have the same bits 1 and 2, and only bit 0 tells them apart. The first
pass had put 2 before 3. The unstable pass reversed them, and the last pass, which saw equal bits,
kept the reversed order. As a result, 3 now precedes 2, and 7 precedes 6, which differ in the same
way. This is the case of equal bits in the proof of Theorem 22.1, and it is the only place where
the proof needs stability.

A split that hands out output positions in the order in which the threads arrive, for example by
incrementing a shared counter, produces such an order. We run a split of this kind on the GPU in
Section 22.8.

<div class="rule-box">

<span class="box-label">Rule</span> Every pass of a least-significant-digit radix sort must be
stable. A pass that places equal digits in any other order than the input order destroys the work
of the passes before it.

</div>

<div class="exercises">

Exercises

22.7 Which positions of the final result are wrong if the first pass, on bit 0, is the unstable
one and the other 2 are stable?<br>
22.8 Does the last pass of a radix sort need to be stable when the keys carry no payload? Give a
reason.

</div>

## 22.5 The cost of 1-bit passes

A sort of 32-bit keys by Parallel-Split takes 32 passes. Each pass is memory-bound. It performs a
few operations per key and moves several bytes per key, so its time is set by the bytes it moves.
We count these bytes per key of 4 bytes. Line 1 reads the key and writes the flag, which is 8
bytes. The scan of Section 21.7 reads and writes the flags twice, which is 16 bytes. Lines 4 to 8
read the key, the flag and the scan, and write the key, which is another 16 bytes. A pass therefore
moves \\(8 + 16 + 16 = 40\\) bytes per key, and the sort moves \\(32 \times 40 = 1{,}280\\) bytes
per key, or 21.5 GB to sort 16,777,216 keys of 64 MiB in all.

The way to reduce this traffic is to reduce the number of passes. With a digit of \\(r\\) bits, a
sort of 32-bit keys takes \\(32 / r\\) passes, so that with \\(r = 4\\), it needs 8 passes instead
of 32. A pass on a digit of \\(r\\) bits, however, splits the keys into \\(2^r\\) groups rather
than 2, and 1 scan of 1 flag no longer gives every group its start. Every key needs to know how
many keys have each smaller digit, so the keys must be counted by digit. That count is a
histogram, the subject of the next section.

## 22.6 Histograms

### 22.6.1 Counting by value

<div class="definition">

<span class="box-label">Definition 22.3</span> **Histogram.** The histogram of a list of keys
over \\(R\\) bins is the array \\(h\\) of \\(R\\) counts in which \\(h_v\\) is the number of keys
whose digit, or value, is \\(v\\).

*Example.* The histogram of \\(K\\) by value, over 8 bins, is \\(h = (1, 0, 2, 1, 0, 2, 1, 1)\\):
1 key 0, no key 1, 2 keys 2, and so on.

</div>

Each count of a histogram is called a **bin**. In the example, bin 2 holds \\(h_2 = 2\\), because
the key 2 occurs twice, at positions 1 and 3. The sequential algorithm reads every key once and
adds 1 to its bin.

<div class="algo">
<p class="algo-name">Histogram(\(K, h, n, R\))</p>
<ol>
<li><b class="kw">for</b> \(v \in [0, R)\): \(h_v = 0\)</li>
<li><b class="kw">for</b> \(i \in [0, n)\) {</li>
<li style="--d:1">\(h_{K_i} = h_{K_i} + 1\) <span class="c">// a read, an addition and a write</span></li>
<li>}</li>
</ol>
</div>

In line 3, the key \\(K_i\\) itself serves as the index of the bin. For \\(i = 1\\), for example,
\\(K_1 = 2\\), and line 3 adds 1 to \\(h_2\\).

### 22.6.2 1 thread per key, and atomic additions

The direct parallel version gives every key its own thread, and each thread executes line 3 for its
key. Line 3, however, consists of 3 operations. It reads the bin, adds 1, and writes the bin back.
In our example, threads 1 and 3 hold the same key, 2. If both read \\(h_2 = 0\\) before either of
them writes, both write 1, and 1 of the 2 counts is lost. The GPU offers an operation that cannot be
split in this way.

<div class="definition">

<span class="box-label">Definition 22.4</span> **Atomic operation.** An atomic operation reads a
memory location, modifies the value and writes it back as 1 indivisible step. No other thread can
access the location between the read and the write. When several threads apply atomic operations
to the same location, the hardware performs them one after the other, in an order it chooses, and
every one of them takes effect. In CUDA C++, the atomic addition is `atomicAdd(&h[v], 1)`; in
cuda-oxide, it is the method `fetch_add` of an atomic type such as `DeviceAtomicU32`.

*Example.* Threads 1 and 3 each apply an atomic addition of 1 to \\(h_2\\). Whatever their order,
\\(h_2\\) ends at 2.

</div>

Atomic operations make the histogram correct, but they introduce a new cost when many threads
update the same bin.

<div class="definition">

<span class="box-label">Definition 22.5</span> **Contention.** Contention is the waiting that
occurs when several threads apply atomic operations to the same location at about the same time:
since the operations are performed one after the other, each waits for those before it.

*Example.* Bins 2 and 5 each receive 2 requests from our 8 threads, and 1 of the 2 requests waits
for the other (Figure 22.5).

</div>

<figure>
<img src="../../figures/ch22-atomics.svg" alt="8 threads, 1 per key 5, 2, 7, 2, 0, 6, 3, 5, each sending an atomic addition to the bin of its key. Bin 2 receives the requests of threads 1 and 3, bin 5 those of threads 0 and 7, and each other occupied bin 1 request. The bins end with the counts 1, 0, 2, 1, 0, 2, 1, 1.">
<figcaption><b>Figure 22.5</b> A histogram of the 8 keys with atomic additions. The chips above each bin are the requests it receives.</figcaption>
</figure>

Atomic operations on global memory are performed in the L2 cache, which all SMs share, and the
requests to 1 address are queued there. In Section 5.5.2, we measured this queue. A histogram of
64 MiB of bytes into 256 global bins took 33.40 ms for random bytes and 134.65 ms for bytes that
were all 7, 4 times as long, because every request went to the same address. A radix pass with
digits of 4 bits has only 16 bins, so even random keys send 1/16 of all requests to each address.

### 22.6.3 Private bins

The solution is to let fewer threads share a bin. Each block counts its keys into its own copy of
the bins in shared memory, where only the threads of that block compete, and where an atomic
operation completes within the SM. When the block has counted all of its keys, it adds each of its
bins to the corresponding global bin once, as Figure 22.6 shows.

<div class="definition">

<span class="box-label">Definition 22.6</span> **Privatization.** Privatization gives each block,
or each warp, a private copy of a shared result, lets it update only its copy, and merges the
copies into the shared result at the end.

*Example.* Block 0 counts the keys 5, 2, 7, 2 into the private bins
\\((0, 0, 2, 0, 0, 1, 0, 1)\\), block 1 counts 0, 6, 3, 5 into \\((1, 0, 0, 1, 0, 1, 1, 0)\\), and
their sum is the histogram.

</div>

<figure>
<img src="../../figures/ch22-privatize.svg" alt="2 blocks of 4 keys. Block 0, keys 5, 2, 7, 2, counts into private bins 0, 0, 2, 0, 0, 1, 0, 1; block 1, keys 0, 6, 3, 5, into 1, 0, 0, 1, 0, 1, 1, 0. Both add their bins to the global bins, which end at 1, 0, 2, 1, 0, 2, 1, 1.">
<figcaption><b>Figure 22.6</b> Private bins in shared memory, merged once per block.</figcaption>
</figure>

In the following procedure, the grid has \\(G\\) blocks of \\(T\\) threads each, \\(b\\) is the
number of the block, \\(t\\) is the index of the thread in its block, and \\(p\\) is the array of
\\(R\\) private bins in shared memory.

<div class="algo">
<p class="algo-name">Histogram-Private(\(K, h, n, R\)) <span class="c">// block \(b\) of \(G\), thread \(t\) of \(T\)</span></p>
<ol>
<li><b class="kw">if</b> \(t &lt; R\): \(p[t] = 0\) <span class="c">// \(p\) in shared memory</span></li>
<li><span class="sc">Barrier</span>() <span class="c">// every private bin is 0</span></li>
<li><b class="kw">for</b> \(i = bT + t\); \(i &lt; n\); \(i = i + GT\) {</li>
<li style="--d:1"><span class="sc">Atomic-Add</span>(\(p[\mathsf{digit}(K_i)]\), 1)</li>
<li>}</li>
<li><span class="sc">Barrier</span>() <span class="c">// every key of the block is counted</span></li>
<li><b class="kw">if</b> \(t &lt; R\): <span class="sc">Atomic-Add</span>(\(h_t\), \(p[t]\)) <span class="c">// the merge</span></li>
</ol>
</div>

| Symbol | Meaning | Example: \\(G = 2\\), \\(T = 2\\), \\(n = 8\\) |
|---|---|---|
| \\(G\\), \\(T\\) | number of blocks, and threads per block | 2 and 2 |
| \\(GT\\) | number of threads in the grid, the stride of the loop | 4 |
| \\(bT + t\\) | the first key of thread \\(t\\) of block \\(b\\) | 3 for \\(b = 1\\), \\(t = 1\\) |
| \\(R\\) | number of bins | 8 |

Line 3 is a **grid-stride loop**. The \\(GT\\) threads of the grid take the keys in rounds, so that
in each round, neighboring threads read neighboring keys. In the small grid of the table, thread 1
of block 1 starts at key \\(1 \cdot 2 + 1 = 3\\) and then moves on to key \\(3 + 4 = 7\\), while
the 4 threads together read keys 0 to 3 in the first round and keys 4 to 7 in the second. The 2
barriers separate 3 phases. Without the first barrier, a thread could count into a bin that another
thread has not yet set to 0. Without the second barrier, thread \\(t\\) could merge bin \\(t\\)
before the other threads have finished counting into it.

With private bins, the number of atomic operations on global memory falls from 1 per key to
\\(R\\) per block. With \\(G = 1{,}024\\) blocks and \\(R = 16\\) bins, a histogram of 16,777,216
keys makes \\(1{,}024 \cdot 16 = 16{,}384\\) atomic additions on global memory instead of
16,777,216, which is 1,024 times fewer.

### 22.6.4 Warp aggregation

Inside a block, some contention remains. When the 32 lanes of a warp hold the same digit, 32
atomic operations on 1 bin in shared memory are performed one after the other. The lanes can,
however, combine their counts first, with the following instruction.

<div class="definition">

<span class="box-label">Definition 22.7</span> **Match.** A match is an instruction that all 32
lanes of a warp execute together, each with a value; it returns to every lane a 32-bit word
whose bit \\(k\\) is 1 if lane \\(k\\) holds the same value. In C++, it is
`__match_any_sync(mask, v)`; in cuda-oxide, `warp::match_any_sync(mask, v)`.

*Example.* For 8 lanes holding the digits \\((1, 2, 3, 2, 0, 2, 3, 1)\\), lane 5 receives the
bits of lanes 1, 3 and 5, the lanes with digit 2.

</div>

With a match, the lowest lane of each group of lanes with equal digits adds the popcount of its
word, which is the size of the group, with 1 atomic operation, and the other lanes of the group add
nothing. In the example of Definition 22.7, lane 1 adds 3 to the bin of digit 2, and lanes 3 and 5
add nothing. A warp then makes at most 1 atomic operation per distinct digit, instead of 32. This
technique is called **warp aggregation**. It costs a match instruction for every key, and in
Section 22.10, we measure whether it pays for itself.

<div class="exercises">

Exercises

22.9 Compute the histogram of \\((6, 1, 4, 4, 7, 0, 3, 1)\\) over 8 bins. Which bins receive more
than 1 request?<br>
22.10 For 16,777,216 keys, 1,024 blocks and 256 bins, how many atomic operations does
Histogram-Private make on global memory, and how many on shared memory?<br>
22.11 What does the merge of line 7 compute if the second barrier is removed and 1 thread of the
block is delayed?

</div>

## 22.7 A pass on a digit of several bits

### 22.7.1 The positions from a histogram

We now sort by digits of 2 bits. Our keys have 3 bits, so a first pass on the digit at shift 0,
the 2 low bits, followed by a second pass on bit 2, sorts them. The digits of the first pass are
\\((1, 2, 3, 2, 0, 2, 3, 1)\\). For example, the key 5 = 101 has the 2 low bits 01, so its digit
is 1. The histogram of these digits over 4 bins is \\((1, 2, 3, 2)\\): 1 key with digit 0, 2 with
digit 1, 3 with digit 2 and 2 with digit 3.

In the output of the pass, the key with digit 0 comes first, at position 0. The 2 keys with digit 1
follow at positions 1 and 2, the 3 keys with digit 2 at positions 3, 4 and 5, and the 2 keys with
digit 3 at positions 6 and 7. The start of each digit, \\((0, 1, 3, 6)\\), is therefore the
exclusive scan of the histogram. Within a digit, the keys keep their input order. The first key
with digit 2, the 2 at position 1, goes to position 3, the second, the 2 at position 3, goes to
position 4, and the third, the 6 at position 5, goes to position 5. The number of keys with the
same digit that come before a key has its own name.

<div class="definition">

<span class="box-label">Definition 22.8</span> **Rank.** The rank of a key in a pass is the
number of keys before it, in input order, that have the same digit.

*Example.* In the pass on the 2 low bits, the 6 at position 5 has digit 2, and 2 keys with digit
2 come before it, at positions 1 and 3: its rank is 2.

</div>

<div class="key-observation">

<span class="box-label">Key observation</span> The exclusive scan of the digit histogram gives
the start of every digit in the output, and a key goes to the start of its digit plus its rank.
The split of Section 22.3 is the case of 1 bit: 2 bins, a start of 0 for digit 0 and \\(Z\\) for
digit 1, and ranks \\(e_i\\) and \\(i - e_i\\).

</div>

For the 6 at position 5, the start of digit 2 is 3 and its rank is 2, so it goes to position
\\(3 + 2 = 5\\). Figure 22.7 carries out the whole pass. The output, \\((0, 5, 5, 2, 2, 6, 7, 3)\\),
is the output of the 2 splits on bits 0 and 1 in Figure 22.2, as it must be, since 1 stable pass on
2 bits sorts by those 2 bits, as 2 stable passes on 1 bit do.

<figure>
<img src="../../figures/ch22-digit-pass.svg" alt="A pass on the 2-bit digit of 5, 2, 7, 2, 0, 6, 3, 5. Digits 1, 2, 3, 2, 0, 2, 3, 1. Histogram 1, 2, 3, 2; exclusive scan 0, 1, 3, 6. Ranks among equal digits 0, 0, 0, 1, 0, 2, 1, 1. Positions 1, 3, 6, 4, 0, 5, 7, 2. Output 0, 5, 5, 2, 2, 6, 7, 3.">
<figcaption><b>Figure 22.7</b> A pass on the 2 low bits: histogram, exclusive scan, rank, position.</figcaption>
</figure>

On a CPU, this pass is the counting sort, which consists of 3 loops over the keys and the bins. In
the following procedure, \\(s\\) is the shift and \\(r\\) the number of bits of the digit, as in
Definition 22.1, and \\(start[d]\\) is the next free output position for a key with digit \\(d\\).

<div class="algo">
<p class="algo-name">Counting-Pass(\(K, out, n, s, r\))</p>
<ol>
<li>\(h\) = the histogram of the digits \(\lfloor K_i / 2^s \rfloor \bmod 2^r\) over \(2^r\) bins</li>
<li>\(start\) = the exclusive scan of \(h\)</li>
<li><b class="kw">for</b> \(i \in [0, n)\) { <span class="c">// in input order</span></li>
<li style="--d:1">\(d\) = the digit of \(K_i\)</li>
<li style="--d:1">\(out_{start[d]} = K_i\)</li>
<li style="--d:1">\(start[d] = start[d] + 1\) <span class="c">// the next key with digit \(d\)</span></li>
<li>}</li>
</ol>
</div>

In line 5, \\(start[d]\\) has grown by 1 for every earlier key with digit \\(d\\), so it equals the
start of \\(d\\) plus the rank of the key. For the 6 at position 5, \\(start[2]\\) began at 3 and
grew by 1 for each of the 2 keys at positions 1 and 3, so it equals 5. Line 6 forms a chain like
the counter of Sequential-Compact, because the rank of each key depends on all the keys before it.

### 22.7.2 Many tiles

On the GPU, the keys are cut into tiles of 1,024 keys, 1 block per tile, as in Section 21.7. A
block can count the digits of its own tile and rank its own keys. The position of a key, however,
also depends on the keys with smaller digits in every tile, and on the keys with the same digit in
earlier tiles. Both of these numbers come from 1 scan, provided that the counts of all tiles are
laid out digit by digit. The table holds first the count of digit 0 in tile 0, in tile 1, and so
on, then the counts of digit 1 in every tile, and so on. With \\(m\\) tiles, the count of digit
\\(d\\) in tile \\(b\\) is entry \\(dm + b\\) of the table. The exclusive scan of this table at
entry \\((d, b)\\) is the number of keys with a smaller digit anywhere, plus the number of keys
with digit \\(d\\) in the tiles before \\(b\\), which is exactly the start of digit \\(d\\) in
tile \\(b\\).

Figure 22.8 shows the table for our keys, cut into 2 tiles of 4. Tile 0 holds the digits
\\((1, 2, 3, 2)\\), and tile 1 holds \\((0, 2, 3, 1)\\). With \\(m = 2\\), the count of digit 1 in
tile 1 is entry \\(1 \cdot 2 + 1 = 3\\) of the table. The scan of the table places digit 1 of tile
1 at position 2, after the key 0 of tile 1 and the key 5 of tile 0. The keys of tile 1 then land at
positions 0, 5, 7 and 2, the same positions as in Figure 22.7.

<figure>
<img src="../../figures/ch22-tiles.svg" alt="The digit counts of 2 tiles stored digit by digit: 0, 1, 1, 1, 2, 1, 1, 1 for digit 0 of tiles 0 and 1, digit 1 of tiles 0 and 1, and so on. Their exclusive scan 0, 0, 1, 2, 3, 5, 6, 7 is the start of each digit in each tile.">
<figcaption><b>Figure 22.8</b> The table of digit counts, digit by digit, and its exclusive scan.</figcaption>
</figure>

Inside a tile, the rank of a key is the number of keys with the same digit before it in the tile.
A block processes its tile in 4 steps of 256 keys. Within a step, the match of Definition 22.7
gives each lane the lanes of its warp that hold the same digit. Keeping only the bits of the lanes
below it, and counting them, gives the rank of the key within the warp, just as the ballot gave the
position within the warp in Section 21.9 (Figure 22.9). The counts of the earlier warps of the
step, and of the earlier steps of the tile, complete the rank.

<figure>
<img src="../../figures/ch22-match.svg" alt="8 lanes with digits 1, 2, 3, 2, 0, 2, 3, 1. For lane 5, whose digit is 2, match_any returns lanes 1, 3 and 5. The mask of lanes below 5 keeps lanes 1 and 3, so the rank of lane 5 among the lanes with digit 2 is 2, and the warp holds 3 keys with digit 2.">
<figcaption><b>Figure 22.9</b> The rank of lane 5 among the lanes with the same digit, from 1 match.</figcaption>
</figure>

### 22.7.3 The algorithm

A pass on the digit of \\(r = 4\\) bits at shift \\(s\\) has three parts. The first kernel counts the
digits of every tile with private bins, as in lines 1 to 6 of Histogram-Private but without a
grid-stride loop, and writes its 16 counts into column \\(b\\) of the table \\(c\\). Scan then
computes the inclusive scan of the table, which has \\(16m\\) entries for \\(m\\) tiles. The second
kernel moves the keys.

In the following procedure, \\(c\\) is the table of counts and \\(s\\) its inclusive scan, so that
\\(s_j - c_j\\) is the exclusive scan at entry \\(j\\). In line 1, thread \\(t < 16\\) stands for
digit \\(t\\), and entry \\(tm + b\\) is the count of digit \\(t\\) in tile \\(b\\). The shared
array \\(start\\) holds, for each digit, the output position of the next key with that digit, and
the shared array \\(wc\\) holds, for each warp \\(w\\) and digit \\(d\\), the number of keys with
digit \\(d\\) in warp \\(w\\) in the current step.

<div class="algo">
<p class="algo-name">Scatter-Digits(\(K, out, c, s\)) <span class="c">// tile \(b\) of \(m\), thread \(t\), lane \(\ell\) of warp \(w\)</span></p>
<ol>
<li><b class="kw">if</b> \(t &lt; 16\): \(start[t] = s_{tm + b} - c_{tm + b}\) <span class="c">// the exclusive scan of the table</span></li>
<li><b class="kw">for</b> \(k \in [0, 4)\) { <span class="c">// 4 steps of 256 keys, in order</span></li>
<li style="--d:1">set \(wc[\cdot][\cdot]\) to 0; <span class="sc">Barrier</span>() <span class="c">// \(wc[w][d]\): keys of digit \(d\) in warp \(w\)</span></li>
<li style="--d:1">\(i = 1024b + 256k + t\), \(d\) = the digit of \(K_i\)</li>
<li style="--d:1">\(peers\) = <span class="sc">Match</span>(\(d\)), \(rank\) = <span class="sc">Popc</span>(\(peers \wedge (2^{\ell} - 1)\))</li>
<li style="--d:1"><b class="kw">if</b> \(rank = 0\): \(wc[w][d]\) = <span class="sc">Popc</span>(\(peers\)) <span class="c">// 1 lane per digit and warp</span></li>
<li style="--d:1"><span class="sc">Barrier</span>()</li>
<li style="--d:1">\(out[start[d] + wc[0][d] + \cdots + wc[w-1][d] + rank] = K_i\)</li>
<li style="--d:1"><span class="sc">Barrier</span>()</li>
<li style="--d:1"><b class="kw">if</b> \(t &lt; 16\): \(start[t] = start[t] + wc[0][t] + \cdots + wc[7][t]\)</li>
<li style="--d:1"><span class="sc">Barrier</span>()</li>
<li>}</li>
</ol>
</div>

| Symbol | Meaning | Lane 5 in the 8-lane example |
|---|---|---|
| \\(d\\) | digit of the key of this thread | 2 |
| \\(peers\\) | lanes of the warp with the same digit, from the match | lanes 1, 3 and 5 |
| \\(2^{\ell} - 1\\) | mask of the lanes below lane \\(\ell\\) | lanes 0 to 4 |
| \\(rank\\) | lanes below this one with the same digit | 2 (lanes 1 and 3) |
| \\(wc[w][d]\\) | keys with digit \\(d\\) in warp \\(w\\), written by the lane with rank 0 | 3, written by lane 1 |
| \\(start[d]\\) | output position of the next key with digit \\(d\\) in this tile | 3 in the 1-tile example |

Line 8 corresponds to line 5 of Counting-Pass, with the rank assembled from three parts. The first
part is the number of keys with the same digit in the earlier steps of the tile, which line 10 has
added to \\(start[d]\\). The second part is the number of such keys in the earlier warps of the
step, and the third part is the number in the earlier lanes of the warp. In the 8-lane example,
taken as a single tile and a single warp, lane 5 writes its key 6 to position
\\(start[2] + rank = 3 + 2 = 5\\).

Each of the 4 barriers separates a write to the shared arrays from the reads that need it. The
barrier of line 3 completes the clearing of \\(wc\\) before any lane writes a count, and line 7
completes the counts before any thread sums them. Line 9 completes every read of \\(start\\)
before line 10 changes it, and line 11 completes the change before the next step reads
\\(start\\).

### 22.7.4 Counting the bytes

A pass reads every key twice, once in the counting kernel and once in the scatter, and writes it
once, which is 12 bytes per key. The table holds 16 counts per 1,024 keys, so its scan moves less
than 1 byte per key. A sort of 32-bit keys by 8 passes of 4 bits therefore moves about
\\(8 \times 12 = 96\\) bytes per key, against 1,280 bytes for 32 passes of 1 bit, which is 13 times
less.

<div class="exercises">

Exercises

22.12 Carry out Counting-Pass on the digit of 2 bits at shift 1 of \\(K\\). What are the
histogram, the starts and the output?<br>
22.13 Build the table of Figure 22.8 for 4 tiles of 2 keys, and give its exclusive scan.<br>
22.14 How many bytes per key does a sort by 4 passes of 8 bits move, if the scan of the table is
counted too? (A tile has 256 counts.)

</div>

## 22.8 Correctness

<div class="theorem">

<span class="box-label">Theorem 22.3</span> A pass of Scatter-Digits writes every key of the
input exactly once, the positions form a permutation of \\([0, n)\\), and keys with equal digits
keep their input order.

</div>

<div class="proof">

*Proof.* Let \\(c(d, b)\\) be the number of keys with digit \\(d\\) in tile \\(b\\), and
\\(S(d, b)\\) the exclusive scan of the table at entry \\((d, b)\\): the number of keys with a
digit below \\(d\\), plus the keys with digit \\(d\\) in tiles before \\(b\\). We show that the
keys with digit \\(d\\) of tile \\(b\\) receive the positions \\(S(d, b)\\) to
\\(S(d, b) + c(d, b) - 1\\), in input order. The step \\(k\\), the warp \\(w\\) and the lane
\\(\ell\\) of a key order the keys of a tile as their indices do, since the index is
\\(1024b + 256k + 32w + \ell\\). In line 8, the position of a key is \\(S(d, b)\\) plus 3 counts of
keys with digit \\(d\\): those in the earlier steps (added to \\(start[d]\\) by line 10 of each
earlier step), those in the earlier warps of the step, and those in the earlier lanes of the warp.
Together they are exactly the keys with digit \\(d\\) before it in the tile. These numbers are
\\(0, 1, \ldots, c(d, b) - 1\\) in input order. The ranges for different \\((d, b)\\) follow each
other in the order of the table, without gap or overlap, and the last ends at \\(n - 1\\).

</div>

| Symbol | Meaning | Digit 1, tile 1 of Figure 22.8 |
|---|---|---|
| \\(c(d, b)\\) | keys with digit \\(d\\) in tile \\(b\\) | 1 (the key 5) |
| \\(S(d, b)\\) | start of digit \\(d\\) in tile \\(b\\): the exclusive scan of the table | 2 |
| \\(S(d, b)\\) to \\(S(d, b) + c(d, b) - 1\\) | the positions of these keys | 2 to 2 |

Rank and stability are the same thing here. The rank counts the keys that come before a key in
input order, so a pass that places keys by rank is stable by construction, and a pass that places
them by any other count is not. To see the difference on the GPU, we replaced the position of
line 8 by an atomic increment of \\(start[d]\\), and removed line 10. The increment hands every key
the next free position of its digit, in the order in which the threads arrive. We ran the C++
program with this change, and it printed the following lines.

```console
$ ./unstable                                    # out[atomicAdd(&start[d], 1)] = key
radix sort of 5 2 7 2 0 6 3 5
  ...
  4-bit pass    0 2 2 3 5 5 6 7
...
  method 1 run 1: 5618124 neighbours out of order, 15278248 keys misplaced
  method 1 run 2: 5627284 neighbours out of order, 15294346 keys misplaced
  method 1 run 3: 5638523 neighbours out of order, 15291387 keys misplaced
FAILED
```

On the 8 keys, the result was correct, because the 8 lanes of 1 warp happened to obtain their
positions in lane order. On 16,777,216 keys, 15.3 million keys ended in the wrong place, 5.6
million neighbors were out of order, and the counts differed from run to run. Every pass still
grouped the keys correctly by its own digit. Only the order inside the groups was lost, and with it
the work of the earlier passes, as in Figure 22.4.

<div class="exercises">

Exercises

22.15 In Scatter-Digits, which wrong position can a key receive if the barrier of line 9 is
removed? Describe a schedule.<br>
22.16 Prove that \\(S(d, b) + c(d, b) = S(d, b + 1)\\) for \\(b < m - 1\\), and
\\(S(d, m - 1) + c(d, m - 1) = S(d + 1, 0)\\).

</div>

## 22.9 Implementation

### 22.9.1 The histograms

The following listing implements Histogram-Private for digits of 4 bits. Each block clears its 16
private bins, counts its keys into them with a grid-stride loop, and merges them into the global
bins.

<p class="listing"><b>Listing 22.1</b> A histogram of 4-bit digits with private bins.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:54:83}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu:65:83}}
```

</div>
</div>

The bins in shared memory are a `SharedArray<u32, RADIX>` in Rust and a `__shared__` array in
C++. CUDA C++ applies `atomicAdd` to any address. cuda-oxide, on the other hand, gives atomics their
own types, with the scope of the threads that may share them in the name. The type
`DeviceAtomicU32` is used for global memory, which every thread of the grid can reach, and
`BlockAtomicU32` for shared memory, which only the threads of the block can reach. The global bins
arrive as a slice `&[DeviceAtomicU32]`, which the host creates by reinterpreting a buffer of `u32`
with `cast_elem`. A shared bin is viewed as a `BlockAtomicU32` with `BlockAtomicU32::from_ptr`,
which the helper `shared_bin` wraps. The pointer comes from the `SharedArray`, so the call is
`unsafe`, and its safety comment states that the digit is below 16.

The next listing implements the same histogram with warp aggregation. The lanes of a warp that
hold the same digit find each other with a match, and only the lowest of them adds the count of the
group to the private bin.

<p class="listing"><b>Listing 22.2</b> The same histogram with warp aggregation.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:85:121}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu:85:110}}
```

</div>
</div>

The loop runs over `base`, the first key of the round of the block, rather than over the key
itself, so that all 32 lanes of every warp execute the match together, including the lanes past
the end of the array. Those lanes match on the value 16, which no digit has, and add nothing. The
variable `leader` is the lowest lane of the group, the position of the lowest 1 bit of `peers`.

### 22.9.2 The split

The following listing implements lines 4 to 8 of Parallel-Split, in which every key computes its
position from the scan of the zero flags and writes itself there. The flags and the scan are the
kernels of Chapter 21.

<p class="listing"><b>Listing 22.3</b> The scatter of a split on 1 bit.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:233:252}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu:199:210}}
```

</div>
</div>

The variable `zeros` is \\(Z\\), and `upto` is the inclusive scan \\(s_i\\), so that `upto - 1` is
\\(e_i\\) for a key with bit 0, and `zeros + i - upto` is \\(Z + i - e_i\\) for a key with bit 1,
whose flag is 0. We let every thread read `scanned[n - 1]`, the number of keys with bit 0, and the cache
serves all but the first of these reads. The Rust write uses `get_unchecked_mut`, and its safety
comment is Theorem 22.2: the positions form a permutation of \\([0, n)\\).

### 22.9.3 The pass on a 4-bit digit

A pass on a 4-bit digit takes 3 listings. The following listing implements the first kernel of the
pass, which counts the digits of each tile in private bins and writes the 16 counts into the table,
digit by digit.

<p class="listing"><b>Listing 22.4</b> The digit counts of every tile, digit by digit.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:254:289}}
    // ...
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu:212:233}}
```

</div>
</div>

The final write stores the count of digit \\(d\\) of block \\(b\\) at entry
`tid * tiles + block`, which is entry \\(dm + b\\) of the table of Section 22.7.2. The next listing
implements Scatter-Digits, which moves every key to the start of its digit in its tile plus its
rank.

<p class="listing"><b>Listing 22.5</b> The scatter of a pass on a 4-bit digit.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
mod kernels {
    // ...
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:291:365}}
}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu:235:283}}
```

</div>
</div>

The shared arrays `start` and `warpCounts` (`START` and `WARP_COUNTS` in Rust) are the arrays
\\(start\\) and \\(wc\\) of the pseudocode, and the 4 calls to the barrier are lines 3, 7, 9 and
11. The rank is `popc(peers & below)`, where `below` has 1 bits for the lanes below this one. A
lane past the end of the array gets the digit 16, matches only with other such lanes, and writes
nothing, but it still reaches every barrier. The Rust write uses `get_unchecked_mut`, and its
safety comment is Theorem 22.3.

The last listing runs the 8 passes on the host. Each pass launches the two kernels and the scan of
the table between them.

<p class="listing"><b>Listing 22.6</b> 8 passes of 4 bits on the host.</p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:474:490}}

{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs:506:518}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu:305:315}}
```

</div>
</div>

The 3 launches of a pass run on 1 stream, so the scatter reads a complete scan. Each pass writes
the keys into a second buffer, and the host then exchanges the 2 buffers, with `std::mem::swap` of
the 2 `DeviceBuffer` values in Rust and with `std::swap` of the 2 pointers in C++, without moving
any data on the GPU. After 8 passes, which is an even number, the sorted keys are back in the first
buffer.

<div class="exercises">

Exercises

22.17 Change `digit_counts` and `scatter_digits` for digits of 8 bits. Which shared arrays grow,
and how large is the table for 16,777,216 keys?<br>
22.18 Why does `hist_aggregated` compare `lane` with `leader` instead of testing `rank == 0` as
`scatter_digits` does? Are the 2 tests equivalent?

</div>

## 22.10 Measurements

### 22.10.1 Platform and method

The measurements come from a Jetson Orin NX 16 GB, an integrated GPU with 8 SMs (sm_87) and
102 GB/s of memory bandwidth. The histograms count the low 4 bits of 16,777,216 keys of 4 bytes,
64 MiB in all, with 1,024 blocks of 256 threads, both on random keys and on keys that are all 7.
Each histogram time is the average of 20 runs after 0.4 s of warm-up. The sorts take 16,777,216
random keys of 32 bits, and each sort time is the best of 3 runs after 1 run of warm-up. Every
histogram is compared bin by bin, and every sort key by key, with the result computed on the CPU.
Between runs, times varied by up to 10%.

The C++ program printed the following output.

```console
$ ./radix_sort_histogram
radix sort of 5 2 7 2 0 6 3 5
  split bit 0   2 2 0 6 5 7 3 5
  split bit 1   0 5 5 2 2 6 7 3
  split bit 2   0 2 2 3 5 5 6 7
  histogram     1 0 2 1 0 2 1 1
  digit starts  0 1 1 3 4 4 6 7
  4-bit pass    0 2 2 3 5 5 6 7
histograms of the low 4 bits of 16777216 keys, average of 20 runs
  keys        global ms   GB/s   shared ms   GB/s   warp-aggregated ms   GB/s
  random           3.23   20.8        1.02   66.1                 1.91   35.1
  all 7           15.15    4.4        1.01   66.2                 1.31   51.2
radix sorts of 16777216 random 32-bit keys, best of 3 runs
  32 passes of 1 bit    262.26 ms    64.0 Mkeys/s
  8 passes of 4 bits     40.13 ms   418.1 Mkeys/s
  1 pass of 4 bits: counts 1.15 ms, scan 0.04 ms, scatter 3.83 ms
PASSED
```

The first 4 lines repeat Figure 22.2, and the histogram line is the histogram of
Definition 22.3. Because the keys have only 3 bits, 1 pass on the digit of 4 bits at shift 0 sorts
them. The digit starts \\((0, 1, 1, 3, 4, 4, 6, 7)\\) are the exclusive scan of the histogram, and
the pass is a counting sort of the whole keys. The Rust program printed the same small example and
times within 5% of these: 3.24, 1.00 and 2.00 ms for the random keys, 15.24, 1.01 and 1.28 ms for
the keys that are all 7, and 255.32 ms and 39.89 ms for the 2 sorts.

### 22.10.2 Contention, measured

Figure 22.10 draws the histogram times. With global bins, the keys that are all 7 took 15.15 ms,
and the random keys took 3.23 ms, 4.7 times less. In Section 5.5.2, we measured a factor of 4 for
bytes in 256 global bins. With all keys equal, every one of the 16,777,216 atomic additions went to
1 address, and the L2 cache performed them one after the other, at a rate of
\\(16{,}777{,}216 / 15.15\\ \text{ms} = 1.11\\) billion per second. Spreading them over 16 addresses
raised the rate only to 5.19 billion per second.

<figure>
<img src="../../figures/ch22-hist-measured.svg" alt="Measured histogram times. random keys: global 3.23 ms, shared 1.02 ms, aggregated 1.91 ms; all 7 keys: global 15.15 ms, shared 1.01 ms, aggregated 1.31 ms.">
<figcaption><b>Figure 22.10</b> Histograms of 16,777,216 keys into 16 bins on the measured GPU.</figcaption>
</figure>

With private bins, the 2 inputs took the same time, 1.02 and 1.01 ms, and both read the keys at
66 GB/s, 65% of the memory bandwidth. The 1,024 blocks merged 16,384 counts into global memory
instead of 16,777,216, and the contention in shared memory, where every key of the second input
also went to 1 bin, no longer set the time. In Section 8.8, we found the same for bytes: once the
bins were private, the speed of the histogram was set by its loads.

Warp aggregation made the histogram slower, 1.91 ms against 1.02 ms on random keys, and 1.31 ms
against 1.01 ms on equal keys. Every key paid for the match instruction, and the atomic operations
that it saved were not the limit. On random keys, it saved few of them in any case. The 32 random
digits of a warp, drawn from 16 values, take on average \\(16 (1 - (15/16)^{32}) = 14.0\\) distinct
values, because each of the 16 digits is missing from the warp with probability \\((15/16)^{32}\\).
A warp therefore still made about 14 atomic additions instead of 32. Aggregation pays when many
lanes of a warp update the same location in global memory, and private bins had already removed
that case.

### 22.10.3 Why the 4-bit sort is 6.5 times faster, not 13

The 32 passes of 1 bit took 262.26 ms. By Section 22.5, they move 1,280 bytes per key, 21.5 GB for
16,777,216 keys, so they used the memory at 81.9 GB/s, 80% of its bandwidth. The sort by single
bits is therefore limited by its traffic. The 8 passes of 4 bits move 96 bytes per key, 13.3 times
less, but they took 40.13 ms, only 6.5 times less (Figure 22.11), and so they used the memory at
only 40.1 GB/s.

<figure>
<img src="../../figures/ch22-sort-measured.svg" alt="Measured sort times for 16,777,216 keys: 32 passes of 1 bit 262.3 ms; 8 passes of 4 bits 40.1 ms; cub::DeviceRadixSort of Chapter 11 9.0 ms.">
<figcaption><b>Figure 22.11</b> Sorting 16,777,216 random keys of 32 bits on the measured GPU. The CUB time is from Chapter 11.</figcaption>
</figure>

The last line of the output locates the difference. In 1 pass, counting the digits took 1.15 ms,
reading 67.1 MB at 58.4 GB/s, and the scan of the table of 262,144 counts took 0.04 ms. The scatter
took 3.83 ms, reading and writing 134.2 MB at only 35.0 GB/s, so the scatter is the slow part. In a
step, the 32 keys of a warp go to about 14 different places, 1 per distinct digit, as computed
above. A store instruction of the warp then becomes about 14 separate writes of a few bytes each,
whereas the copy of 32 consecutive keys is 4 writes of 32 bytes, and the memory system handles each
of those writes as a separate request. The scatter also waits at 16 barriers per tile. The splits
of the 1-bit sort scatter too, but every warp writes to at most 2 places, and their writes are
nearly as regular as a copy.

The library sort that we measured in Chapter 11, `cub::DeviceRadixSort`, sorted 16,777,216 keys in
8.96 ms on the same GPU, 4.5 times faster than our 4-bit sort. It uses digits of 8 bits, and
therefore 4 passes instead of 8 (Section 11.2.4). It ranks and reorders the keys of each tile in
shared memory, so that the writes of a warp go to few places, and it computes the digit starts with
a single-pass scan (Adinets and Merrill, 2022). Exercise 5 at the end of the chapter takes the
first of those steps.

## 22.11 The complete programs

The following listing contains the complete programs of this chapter, in Rust and in CUDA C++. Each
program runs the small example, measures the 3 histograms and the 2 sorts on 16,777,216 keys,
checks every result against the CPU, and prints the times.

<p class="listing"><b>Listing 22.7</b> The complete programs. <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/oxide/ch22_radix_sort_histogram/src/main.rs">code/oxide/ch22_radix_sort_histogram/src/main.rs</a> and <a href="https://github.com/arpanpathak/gpu-parallel-book/blob/main/code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu">code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu</a></p>

<div class="code-tabs">
<div class="code-tab" data-lang="Rust">

```rust,ignore
{{#include ../../../../code/oxide/ch22_radix_sort_histogram/src/main.rs}}
```

</div>
<div class="code-tab" data-lang="C++">

```cpp
{{#include ../../../../code/cuda/ch22_radix_sort_histogram/radix_sort_histogram.cu}}
```

</div>
</div>

## Common mistakes

- **An unstable pass.** Positions handed out by an atomic counter group the keys by digit but lose
  their order, and the sort fails on large inputs while it may pass on small ones (Section 22.8).
  Place keys by start plus rank.
- **The most significant digit first.** A least-significant-digit sort by stable passes must start
  at bit 0. Starting at the top requires sorting each group separately.
- **Counting into global bins.** With few bins, every global atomic operation queues behind others
  for the same address. Count into private bins in shared memory and merge once per block.
- **Merging before every thread has counted.** Without the barrier before the merge, a thread
  can add its private bin to the global bin while other threads still count into it.
- **A match or a ballot inside a branch.** All lanes of the warp must execute it. Give the lanes
  past the end of the array a value that no key has, and call it unconditionally.
- **The table of counts laid out tile by tile.** Only the digit-major layout makes 1 scan give the
  start of every digit in every tile.

## Summary

- A radix sort orders integer keys by their digits, least significant first. Theorem 22.1 shows
  that it is correct when every pass is stable.
- A split on 1 bit is 2 stream compactions that share 1 exclusive scan: keys with bit 0 go to
  \\(e_i\\), and keys with bit 1 go to \\(Z + i - e_i\\).
- A histogram counts keys by digit. Atomic additions make one thread per key correct, but contention
  makes them slow when many threads update 1 bin, and private bins in shared memory remove most of
  it.
- A pass on a digit of \\(r\\) bits places every key at the exclusive scan of the digit histogram
  plus its rank. Laying out the counts of the tiles digit by digit lets 1 scan serve every tile.
- Passes of 4 bits move 96 bytes per key for a 32-bit sort, against 1,280 for passes of 1 bit,
  and the measured sort ran 6.5 times as fast.

In Chapter 23, we turn to softmax and layer normalization, which combine reductions with arithmetic
on every element.

## Exercises

1. Sort \\((13, 2, 9, 2, 15, 0, 7, 13)\\), keys of 4 bits, by 2 passes of 2 bits. Give the
   histogram, the starts and the ranks of each pass.
2. Extend the radix sort to sort pairs (key, index), so that it returns the permutation that sorts
   the keys. Which kernels change?
3. Signed integers in two's complement do not sort correctly as unsigned keys. Which single bit
   must be flipped before the sort, and after it?
4. Measure the histogram of 8-bit digits (256 bins) with the three kernels of Section 22.6, on random
   keys and on keys that are all equal, and compare with the 16-bin results.
5. Replace the 3 passes of Section 22.7 (count, scan, scatter) by a version that keeps each tile's
   keys sorted by digit in shared memory before writing them, so that the writes of a warp go to
   fewer places. Measure the effect.
6. Estimate, from the byte count of Section 22.7.4, the time of a sort of 16,777,216 keys by 4
   passes of 8 bits at the measured rate of the 4-bit sort.

## Sources and further reading

- D. E. Knuth, *The Art of Computer Programming*, vol. 3, *Sorting and Searching*, 2nd ed.,
  Addison-Wesley, 1998, Section 5.2.5: radix sorting.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed.,
  MIT Press, 2022, Sections 8.2 and 8.3: counting sort and radix sort.
- G. E. Blelloch, "Prefix Sums and Their Applications", Technical Report CMU-CS-90-190, Carnegie
  Mellon University, 1990: the split and radix sort from scans.
- N. Satish, M. Harris and M. Garland, "Designing Efficient Sorting Algorithms for Manycore GPUs",
  IEEE IPDPS, 2009.
- D. Merrill and A. Grimshaw, "High Performance and Scalable Radix Sorting", *Parallel Processing
  Letters* 21(2), 2011.
- A. Adinets and D. Merrill, "Onesweep: A Faster Least Significant Digit Radix Sort for GPUs",
  arXiv:2206.01784, 2022.
- A. Pathak, "An Introduction to Tiled Parallel Programming, Simplified", 2026.
- NVIDIA, *CUDA C++ Programming Guide*, "Atomic Functions" and "Warp Match Functions".
