# Normalization

A normalization rescales the values of a row of a matrix by statistics of that row. The family
contains 1 chapter, Chapter 23, with the 2 normalizations that every layer of a transformer model
applies: softmax, which divides the exponential of each value by the sum of the exponentials of
its row, and layer normalization, which subtracts the mean of the row and divides by its standard
deviation. Both appear many times per layer, on rows of a few hundred to more than 100,000 values,
so their speed matters as much as their accuracy.

The 2 operations share 1 structure. No output can be computed from its input alone: every output
needs a number computed from the whole row first, the maximum and the sum of exponentials, or the
mean and the variance. Each operation is therefore a reduction over the row followed by an
element-by-element computation, and on a GPU 1 block of threads handles 1 row: the threads reduce
their parts of the row, combine the parts, and write the outputs.

The idea that connects the algorithms of the family is a partial result that can be corrected. A
sum of exponentials taken relative to 1 maximum can be moved to a larger maximum by 1
multiplication, and a sum of squared deviations from 1 mean can be moved to a new mean by 1
addition. With these corrections, a single pass over the row computes both statistics that the
direct formulas compute in 2 passes, and 2 partial results computed by different threads merge into
the result of their union. The online softmax and Welford's method are the 2 algorithms built on
this idea, and both are proved correct with a loop invariant.

Chapter 23 derives both operations on a matrix of 2 rows of 4 values. It shows 2 numerical failures
of the direct formulas in `f32`, an overflow and a cancellation, and the remedies, then develops the
1-pass algorithms, their merge rules, the block-wide reduction with warp shuffles and barriers, and
fused kernels that read each value once. It builds on the reductions and warp shuffles of Chapter 8
and on the cuda-oxide kernels, shared memory and launch contracts of Chapters 13 and 15, and it
measures every kernel against the memory bandwidth in the way of Chapter 7.
