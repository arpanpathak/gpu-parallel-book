# Matrix Multiplication

The 3 chapters of this group compute the same product, \\(C = AB\\), with 3 kernels, and derive
each kernel from the one before. The product is the classic first example of parallel
programming on a GPU: every entry of \\(C\\) can be computed independently of the others, and
every input value is needed by many entries. The first property makes the product easy to
parallelize; the second makes its cost a question of how often each value is read, and from
where.

The 3 chapters share 1 idea. A value that several threads need should be fetched once from slow
storage into fast storage, and then used from there by all of them. Chapter 16 counts what
happens without this: in the direct kernel each of the \\(MN\\) threads fetches its \\(2K\\)
operands from global memory, 1 value for every floating-point operation. Chapter 17 applies the
idea to a block of threads and shared memory, which divides the fetches from global memory by
the tile width \\(T\\). Chapter 18 applies it to a single thread and its registers, which divides
the reads from shared memory by the size of the block of \\(C\\) that each thread computes.

All 3 chapters work with the same pair of 4 × 4 matrices, whose product begins with
\\(c_{00} = 11\\) and \\(c_{01} = 13\\). Chapter 16 computes the product by hand and with the
sequential algorithm, introduces threads, blocks, grids and the memories of the device, and
derives the direct kernel. Chapter 17 splits each dot product into phases, stages tiles in shared
memory, shows on the example the wrong values 7 and 15 that appear when either of its 2 barriers is
removed, and proves the tiled kernel correct with a loop invariant. Chapter 18 lets each thread
compute a block of entries as a sum of outer products and measures register tiles from 1 × 1 to
8 × 8.

Each chapter implements its kernel in Rust with cuda-oxide and in CUDA C++, shows the 2 side by
side, checks the results exactly against the CPU, and measures them on a Jetson Orin NX. The
chapters build on Chapter 9, which developed the same sequence of kernels in CUDA C++ and compared
them with cuBLAS, and on Chapters 13 to 15, which introduced kernels, `DisjointSlice`, launch
contracts and shared memory in Rust.
