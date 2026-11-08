# Matrix Multiplication

In the next 3 chapters, we compute the same product, \\(C = AB\\), with 3 kernels, and derive
each kernel from the one before. The product is the classic first example of parallel
programming on a GPU: every entry of \\(C\\) can be computed independently of the others, and
every input value is needed by many entries. The first property makes the product easy to
parallelize; the second makes its cost a question of how often each value is read, and from
where.

The 3 chapters share 1 idea. A value that several threads need should be fetched once from slow
storage into fast storage, and then used from there by all of them. In Chapter 16, we count what
happens without this: in the direct kernel each of the \\(MN\\) threads fetches its \\(2K\\)
operands from global memory, 1 value for every floating-point operation. In Chapter 17, we apply the
idea to a block of threads and shared memory, which divides the fetches from global memory by the
tile width \\(T\\). In Chapter 18, we apply it to a single thread and its registers, which divides
the reads from shared memory by the size of the block of \\(C\\) that each thread computes.

All 3 chapters work with the same pair of 4 × 4 matrices, whose product begins with
\\(c_{00} = 11\\) and \\(c_{01} = 13\\). In Chapter 16, we compute the product by hand and with the
sequential algorithm, introduce threads, blocks, grids and the memories of the device, and
derive the direct kernel. In Chapter 17, we split each dot product into phases, stage tiles in
shared memory, show on the example the wrong values 7 and 15 that appear when either of its 2
barriers is removed, and prove the tiled kernel correct with a loop invariant. In Chapter 18, we
let each thread compute a block of entries as a sum of outer products and measure register tiles
from 1 × 1 to 8 × 8.

In each chapter we implement the kernel in Rust with cuda-oxide and in CUDA C++, show the 2 side
by side, check the results exactly against the CPU, and measure them. We build on
Chapter 9, where we developed the same sequence of kernels in CUDA C++ and compared them with
cuBLAS, and on Chapters 13 to 15, where we introduced kernels, `DisjointSlice`, launch contracts
and shared memory in Rust.
