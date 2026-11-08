# Stencils

A stencil computation updates every cell of a grid from a fixed pattern of its neighbors. The
temperature of a heated plate, the potential between charged conductors, the blur of an image and
the step of a fluid simulation are all stencil computations. In the next 2 chapters, we study the
most common one, the 5-point stencil of the Laplace equation, iterated with Jacobi's method. We use
a single plate of 6 × 6 cells throughout, so that every value can be checked by hand.

The idea that connects the 2 chapters is the separation of old values from new ones. Every update
of a Jacobi step reads only values of the previous step. All updates of a step are therefore
independent of one another, and a GPU may run them at the same time, but only if no update
overwrites a value that another update still has to read. In Chapter 19, we enforce this
separation between steps with 2 buffers in global memory. In Chapter 20, we enforce it within a
block with a tile in shared memory and a barrier, and, for temporal blocking, with 2 tiles used in
turn.

In Chapter 19, Stencils and Jacobi Iteration, we derive the 5-point stencil from the Laplace
equation, work 2 Jacobi steps by hand, show the exact wrong value that an update in place
produces, and prove with a loop invariant that 2 buffers used in turn compute Jacobi iteration. We
then write the kernel with 1 thread per cell, in Rust with `DisjointSlice` and in C++ with
`__restrict__`, measure how fast the iteration converges, and show from its arithmetic intensity
why the step is limited by memory bandwidth.

In Chapter 20, Tiled Stencils with Halo Cells, we count how often the direct kernel reads each
value, define the tile and its halo, compare 2 ways of loading the halo, and show the wrong values
that appear without the barrier and without the halo. We then count the loads per result as a
function of the tile size, measure why the tile gains little on GPUs whose caches already capture
the reuse, and measure temporal blocking, which performs several steps on each tile.

In both chapters, we build on Chapter 7 for sectors, coalescing and shared memory, and on Chapter
5 for barriers and race conditions. We also apply the method of the tiled matrix product of
Chapters 9 and 17, which stages data in shared memory, to a computation with far less reuse per
value.
