# Matrix Multiplication

In the next 3 chapters, we compute the same product, \\(C = AB\\), with 3 kernels, and we derive
each kernel from the one before it. The matrix product is the classic first example of parallel
programming on a GPU, for 2 reasons. First, every entry of \\(C\\) can be computed independently
of the others, which makes the product easy to parallelize. Second, every input value is needed
by many entries of \\(C\\). As a result, the cost of the product depends mainly on how often each
input value is read, and on the memory from which it is read.

The 3 chapters are built on 1 idea. When several threads need the same value, the value should be
fetched once from slow storage into fast storage, and all of these threads should then read it
from the fast storage. In Chapter 16, we count what happens without this idea. In the direct
kernel, there is 1 thread for each entry of \\(C\\), and each thread fetches every operand of its
dot product from global memory. With \\(A\\) of size \\(M \times K\\) and \\(B\\) of size
\\(K \times N\\), there are \\(MN\\) threads, and each thread fetches \\(2K\\) values, namely
\\(K\\) values from a row of \\(A\\) and \\(K\\) values from a column of \\(B\\). This is 1 fetch
for every floating-point operation.

In Chapter 17, we apply the idea to a block of threads and
its shared memory. The threads of a block copy square pieces of \\(A\\) and \\(B\\), called tiles,
into shared memory, which divides the number of fetches from global memory by the tile width
\\(T\\), the number of rows and columns in a tile. In Chapter 18, we apply the idea to a single
thread and its registers, which divides the number of reads from shared memory by the size of the
block of \\(C\\) that each thread computes.

All 3 chapters work with the same pair of 4 × 4 matrices, whose product begins with
\\(c_{00} = 11\\) and \\(c_{01} = 13\\). In Chapter 16, we compute the product by hand and with
the sequential algorithm, we introduce threads, blocks, grids and the memories of the device, and
we derive the direct kernel. In Chapter 17, we split each dot product into phases and stage tiles
in shared memory. We then show, on the example, the wrong values 7 and 15 that appear when either
of the kernel's 2 barriers is removed, and we prove the tiled kernel correct with a loop
invariant. In Chapter 18, we let each thread compute a block of entries of \\(C\\) as a sum of
outer products, and we measure register tiles from 1 × 1 to 8 × 8.

In each chapter, we implement the kernel in Rust with cuda-oxide and in CUDA C++, show the 2
versions side by side, check the results exactly against the CPU, and measure them. We build on
Chapter 9, where we developed the same sequence of kernels in CUDA C++ and compared them with
cuBLAS, and on Chapters 13 to 15, where we introduced kernels, `DisjointSlice`, launch contracts
and shared memory in Rust.
