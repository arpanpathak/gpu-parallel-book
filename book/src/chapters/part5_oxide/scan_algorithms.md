# Scan-Based Algorithms

In the next 2 chapters, we treat algorithms whose output positions are not known in advance. A
stream compaction keeps some of the elements of an array and packs them together at the front of
another array. How far an element moves depends on how many elements before it were kept. A radix
sort moves every key to a place that depends on 2 numbers: how many keys have a smaller digit, and
how many keys before it have the same digit. In both cases, each element could compute its output
position on its own, if it knew a count of the elements before it.

Such a count is a scan. For example, the exclusive scan of an array of 0s and 1s gives every
element the number of 1s before it, and the exclusive scan of a histogram gives every bin the
number of elements in the bins before it. At first sight, a scan seems to need a strict order,
since every running total is built from the one before it. Because addition is associative,
however, the totals can be built by combining partial sums in a tree, in a number of steps that
grows only with the logarithm of the length of the array. The idea that connects the 2 chapters is
therefore to turn each data-dependent position into a count, and each count into a scan.

In Chapter 21, Scan and Stream Compaction, we derive 2 parallel scans from the sequential loop, on
1 example of 8 values. The Hillis-Steele scan doubles the length of the ranges it sums at every
step. The Blelloch scan builds a tree of partial sums and then passes the sums down the tree. We
show what each barrier and the second buffer prevent, prove the Hillis-Steele scan correct, and
extend both scans to arrays of many blocks. Finally, we build stream compaction on the scan, first
in 3 passes and then with warp votes.

In Chapter 22, Radix Sort and Histograms, we sort 8 keys of 3 bits by a sequence of stable passes.
A pass on 1 bit consists of 2 stream compactions. A pass on a digit of several bits needs the count
of every digit, which is a histogram, and the exclusive scan of that histogram gives each digit its
start in the output. We count with atomic operations, measure their contention, remove most of it
with private bins in shared memory, and prove that a sort by stable passes is correct.

In both chapters, we build on Chapter 8, where we developed a fast scan and a histogram in CUDA C++
and in Rust, and on Chapter 15, where we built the tree of a reduction. We implement every
algorithm in Rust with cuda-oxide and in CUDA C++, and we reuse the rules for barriers and shared
memory that we established for the tiled matrix multiplication.
