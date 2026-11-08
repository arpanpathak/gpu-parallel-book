# Normalization

A normalization rescales the values of a row of a matrix by statistics of that row. This family
contains 1 chapter, Chapter 23, which treats the 2 normalizations that every layer of a transformer
model applies. The first is softmax, which divides the exponential of each value by the sum of the
exponentials of its row. The second is layer normalization, which subtracts the mean of the row
and divides by its standard deviation. Both operations appear many times per layer, on rows of a
few hundred to more than 100,000 values, so their speed matters as much as their accuracy.

The 2 operations share 1 structure. In neither of them can an output be computed from its input
alone, because every output needs a number that is computed from the whole row first: the maximum
and the sum of exponentials in softmax, and the mean and the variance in layer normalization. Each
operation is therefore a reduction over the row followed by an element-by-element computation. On
a GPU, 1 block of threads handles 1 row. The threads of the block reduce their parts of the row,
combine the partial results, and then write the outputs.

The idea that connects the algorithms of the family is a partial result that can be corrected. A
sum of exponentials taken relative to 1 maximum can be moved to a larger maximum by 1
multiplication, and a sum of squared deviations from 1 mean can be moved to a new mean by 1
addition. With these corrections, a single pass over the row computes both of the statistics that
the direct formulas compute in 2 passes. In the same way, 2 partial results computed by different
threads merge into the result of their union. The online softmax and Welford's method are the 2
algorithms built on this idea, and we prove both of them correct with a loop invariant.

In Chapter 23, we derive both operations on a matrix of 2 rows of 4 values. We show 2 numerical
failures of the direct formulas in `f32`, an overflow and a cancellation, together with their
remedies. We then develop the 1-pass algorithms, their merge rules, the block-wide reduction with
warp shuffles and barriers, and fused kernels that read each value only once. We build on
the reductions and warp shuffles of Chapter 8 and on the cuda-oxide kernels, shared memory and
launch contracts of Chapters 13 and 15. As in Chapter 7, we measure every kernel against the memory
bandwidth.
