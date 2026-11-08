# Elementwise Operations and Reductions

The 2 chapters of this group derive the 2 simplest patterns of parallel computation. In an
elementwise operation, or map, every element of the output is computed from the elements at the
same position in the inputs. In a reduction, all elements of the input are combined into 1 value,
such as a sum, a maximum or a count. Almost every GPU program contains both: the activation of a
neural network layer is a map, the norm of a vector is a reduction, and a dot product is a map
followed by a reduction.

The idea that connects the 2 patterns is the dependence between the operations of the sequential
loop. In a map, iteration \\(i\\) reads and writes only position \\(i\\), so the iterations share
nothing and each can be given to its own thread. In a reduction, every iteration reads the running
result of the one before, a chain as long as the array. The chain is broken by a property of the
operator, not of the hardware: because addition is associative and commutative, the additions can
be regrouped into a tree whose levels each consist of independent additions. Both chapters also
share 1 measurement result. Each element is used exactly once, so neither pattern can reuse data,
and the speed of both is set by the memory bandwidth of the GPU.

Chapter 14 derives the map on SAXPY, \\(y \leftarrow a x + y\\), worked by hand on 12 elements. It
covers a grid of \\(\lceil n / B \rceil\\) blocks with a guard for the threads beyond the end, then
the grid-stride loop, in which fewer threads than elements cover the array, with a proof that each
element is updated exactly once. It writes a generic map that takes a Rust closure or a C++
lambda, and measures how close SAXPY comes to the bandwidth.

Chapter 15 derives the reduction on the dot product of 2 vectors of 16 small integers. It turns
the tree of additions into a kernel that works in shared memory, step by step between barriers,
proves the tree correct with a loop invariant, replaces most barriers by warp shuffles, and
combines the sums of the blocks with a second launch or with atomic additions, whose order changes
the last bits of a floating-point result.

The group builds on the first Rust kernel of Chapter 13, from which it takes `DisjointSlice` and
the safe launch path, and on Chapters 7 and 8, which introduced coalesced access, the grid-stride
loop and the reduction in C++. Every algorithm is given in CUDA C++ and in Rust with cuda-oxide,
side by side.
