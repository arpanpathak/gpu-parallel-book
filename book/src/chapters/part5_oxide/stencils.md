# Stencils

A stencil computation updates every cell of a grid from a fixed pattern of its neighbours. The
temperature of a heated plate, the potential between charged conductors, the blur of an image and
the step of a fluid simulation are all stencils. The 2 chapters of this group study the most
common one, the 5-point stencil of the Laplace equation, iterated with Jacobi's method, on a
single 6 × 6 plate whose values every reader can check by hand.

The idea that connects the 2 chapters is the separation of old values from new ones. Every update
of a Jacobi step reads only values of the previous step. That makes all updates of a step
independent, so a GPU may run them at the same time, but only if no update overwrites a value that
another still has to read. Chapter 19 enforces the separation between steps with 2 buffers in
global memory; Chapter 20 enforces it within a block with a tile in shared memory and a barrier,
and, for temporal blocking, with 2 tiles used in turn.

Chapter 19, Stencils and Jacobi Iteration, derives the 5-point stencil from the Laplace equation,
works 2 Jacobi steps by hand, shows the exact wrong value that an update in place produces, and
proves with a loop invariant that 2 buffers used in turn compute Jacobi iteration. It then writes
the kernel with 1 thread per cell, in Rust with `DisjointSlice` and in C++ with `__restrict__`,
measures how fast the iteration converges, and shows from its arithmetic intensity why the step is
limited by memory bandwidth.

Chapter 20, Tiled Stencils with Halo Cells, counts how often the direct kernel reads each value,
defines the tile and its halo, compares 2 ways of loading the halo, and shows the wrong values
produced without the barrier and without the halo. It counts the loads per result as a function of
the tile size, measures why the tile gains little on GPUs whose caches already capture the reuse,
and measures temporal blocking, which performs several steps on each tile.

The group builds on Chapter 7 for sectors, coalescing and shared memory, and on Chapter 5 for
barriers and race conditions. It applies the method of the tiled matrix product of Chapters 9 and
17, staging data in shared memory, to a computation with far less reuse per value.
