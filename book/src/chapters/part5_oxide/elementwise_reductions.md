# Elementwise Operations and Reductions

In the next 2 chapters, we develop the 2 simplest patterns of parallel computation. In an
elementwise operation, or map, every element of the output is computed from the elements at the
same position in the inputs. In a reduction, all elements of the input are combined into 1 value,
such as a sum, a maximum or a count. Almost every GPU program contains both patterns. The
activation function of a neural network layer is a map, the length of a vector is computed by a
reduction, and a dot product is a map followed by a reduction.

The idea that connects the 2 patterns is the dependence between the iterations of the sequential
loop. In a map, iteration \\(i\\) reads and writes only position \\(i\\), so the iterations share
nothing, and each iteration can be given to its own thread. In a reduction, every iteration reads
the running result of the iteration before it, which forms a chain as long as the array. This
chain is broken by a property of the operator, not of the hardware. Because addition is
associative and commutative, the additions can be regrouped into a tree, and the additions within
each level of the tree are independent of each other. The 2 chapters also share 1 result of the
measurements. Each element is used exactly once, so neither pattern can reuse data, and the speed
of both is set by the memory bandwidth of the GPU.

In Chapter 14, we develop the map on SAXPY, the operation \\(y \leftarrow a x + y\\), which
replaces each element of a vector \\(y\\) by \\(a\\) times the element of \\(x\\) at the same
position plus the old element of \\(y\\). We work it out by hand on 12 elements. We first give
every element its own thread, using just enough blocks to cover the array and a guard that stops
the threads beyond its end. We then introduce the grid-stride loop, in which fewer threads than
elements cover the array, and prove that it updates each element exactly once. Finally, we write a
generic map that takes a Rust closure or a C++ lambda, and we measure how close SAXPY comes to the
bandwidth of the memory.

In Chapter 15, we develop the reduction on the dot product of 2 vectors of 16 small integers. We
turn the tree of additions into a kernel that works in shared memory, step by step, with a barrier
between consecutive steps, and we prove the tree correct with a loop invariant. We then replace
most of the barriers by warp shuffles, and we combine the sums of the blocks either with a second
launch or with atomic additions, whose order changes the last bits of a floating-point result.

Both chapters build on the first Rust kernel of Chapter 13, from which we take `DisjointSlice` and
the safe launch path, and on Chapters 7 and 8, in which we introduced coalesced access, the
grid-stride loop and the reduction in C++. Every algorithm is given both in CUDA C++ and in Rust
with cuda-oxide.
