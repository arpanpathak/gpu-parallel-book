# CODING_STANDARDS.md

This project has a constitution. It applies to every line of code in this book.

1. **Every primitive is explained.** No code block appears without a preceding
   explanation of every type, function and built-in variable it uses.
2. **No magic numbers.** Any value other than `0`, `1`, `-1` or a power of two
   required by the CUDA API is declared as a named constant.
3. **Every kernel is commented line by line.** A reviewer must be able to
   verify the index arithmetic and the memory accesses from the comments alone.
4. **No silent error handling.** Every CUDA API call that can fail is checked,
   and the failure is logged or propagated.
5. **Boundary conditions are explicit.** Kernels that guard against out-of-range
   indices say so, and kernels whose launch configuration guarantees
   in-range access prove it in the comments.
6. **Synchronisation is justified.** Every `__syncthreads()`, fence and atomic
   carries a comment explaining which data it protects and why.
7. **No assumptions about the reader.** The book never refers to a framework or
   library API without first defining the primitive it is built upon.
8. **Benchmarks are reproducible.** Any performance claim is accompanied by the
   hardware, the CUDA version, the compiler flags, and the measurement method.

These rules are not here to annoy you. Every violation listed above has been,
at some point, a production incident or a silent numerical error.
