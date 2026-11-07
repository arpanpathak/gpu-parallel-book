# Scan-Based Algorithms

The 2 chapters of this group treat algorithms whose output positions are not known in advance.
A stream compaction keeps some of the elements of an array and packs them together; how far an
element moves depends on how many elements before it were kept. A radix sort moves every key to
a place that depends on how many keys have a smaller digit, and how many keys before it have the
same digit. In both cases each element could compute its position on its own, if it knew a
count of the elements before it.

That count is a scan. The exclusive scan of an array of 0s and 1s gives every element the number
of 1s before it, and the exclusive scan of a histogram gives every bin the number of elements in
the bins before it. A scan seems to need a strict order, since every running total is built from
the one before. Because addition is associative, however, the totals can be built by combining
partial sums in a tree, in a number of steps that grows with the logarithm of the array length. The idea
that connects the 2 chapters is therefore to turn each data-dependent position into a count, and
each count into a scan.

Chapter 21, Scan and Stream Compaction, derives 2 parallel scans from the sequential loop on 1
example of 8 values. The Hillis-Steele scan doubles the length of the ranges it sums at every
step. The Blelloch scan builds a tree of partial sums and then passes the sums down it. It shows what each barrier and the second buffer prevent, proves the Hillis-Steele scan
correct, extends both to arrays of many blocks, and builds stream compaction on the scan, first in
3 passes and then with warp votes.

Chapter 22, Radix Sort and Histograms, sorts 8 keys of 3 bits by a sequence of stable passes. A
pass on 1 bit is 2 stream compactions; a pass on a digit of several bits needs the count of every
digit, a histogram, whose exclusive scan gives each digit its start. The chapter counts with
atomic operations, measures their contention, removes most of it with private bins in shared
memory, and proves that a sort by stable passes is correct.

Both chapters build on Chapter 8, which developed a fast scan and a histogram in CUDA C++ and in
Rust, and on Chapter 15 for the tree of a reduction. They implement every algorithm in Rust with
cuda-oxide and in CUDA C++, and they reuse the barrier and shared-memory rules of the tiled
matrix multiplication.
