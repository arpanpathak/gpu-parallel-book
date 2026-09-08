# Chapter 1: The Mathematics of Parallelism

> *"The purpose of computing is insight, not numbers."*
> — Richard Hamming, *Numerical Methods for Scientists and Engineers* (1962)

This chapter defines the quantitative vocabulary used throughout the book:
*speedup*, *efficiency*, *Amdahl's law*, *Gustafson-Barsis scaling*, *strong and
weak scaling*, *Flynn's taxonomy*, and the *roofline model*. These concepts are
the tools for deciding whether an optimisation is worth the effort and which
hardware resource limits a kernel.

## 1.1 Latency, Throughput, and the Meaning of "Faster"

A program described as "slow" is usually slow in one of two distinct senses. On
a GPU the distinction determines the correct optimisation strategy.

**Latency** is the elapsed time between the start of an operation and its
completion. A memory access has latency. A network round trip has latency.
Latency is measured in time units, typically nanoseconds or milliseconds.

**Throughput** is the number of operations completed per unit time. A pipeline
that processes 60 frames per second has a throughput of 60 Hz. Throughput is
measured in operations per second.

A CPU is engineered to **minimise latency**. It uses a small number of fast
cores, each with branch prediction, out-of-order execution, and large caches
that hide the latency of DRAM.

A GPU is engineered to **maximise throughput**. It contains a large number of
simpler execution units. Individually they are slower than a CPU core, but
collectively they complete millions of operations per clock. The GPU hides
latency not by predicting what happens next but by keeping many independent
threads in flight, so that execution units always have work while other threads
wait.

Throughput and concurrency are related by a well-known queueing result.
In a system that sustains a steady flow of work:

\\[ \text{throughput} = \frac{\text{concurrency}}{\text{latency}} \\]

This is Little's law. To see why it matters for GPUs, suppose a memory access
takes 500 cycles and the machine has no other work to issue during that window.
The memory system then delivers one access per 500 cycles, regardless of clock
speed. If the machine keeps 1,000 independent accesses in flight, each still
taking 500 cycles, it completes 1,000 accesses every 500 cycles: a 1,000-fold
increase in throughput with no change in latency. This is why GPUs use many
concurrent threads. **Latency is not reduced; it is amortised over
concurrency.**

This result appears throughout the book as a defining property:

> **Primitive - latency hiding.** When an execution unit must wait for a slow
> operation, such as a memory access or a division, the unit would otherwise be
> idle. The GPU switches to another ready thread. The cost of the wait is
> hidden, not eliminated.

A GPU is therefore well suited to computations that can be divided into many
independent pieces and poorly suited to a single sequential computation. The
mathematics of that division is the subject of this chapter.

## 1.2 Speedup and Efficiency

Let \\(T_1\\) be the execution time of a program on one processing unit (one
core or one thread), and \\(T_p\\) its execution time on \\(p\\) processing
units. The standard definitions are:

**Speedup** is the ratio of the serial time to the parallel time:

\\[ S(p) = \frac{T_1}{T_p} \\]

A perfect speedup of \\(p\\) means the program completes in \\(1/p\\) of its
serial time.

**Efficiency** is the speedup per processing unit:

\\[ E(p) = \frac{S(p)}{p} = \frac{T_1}{p \cdot T_p} \\]

An efficiency of 1.0 means every processing unit contributes in proportion to
the ideal. An efficiency of 0.5 means half of the added hardware's potential is
not converted into performance. Efficiency is the more diagnostic of the two:
speedup measures the improvement over the serial run, while efficiency measures
how much of the added hardware is actually used. A report of "10x speedup on 64
cores" says nothing by itself; the efficiency is 10/64 = 0.156, meaning 84% of
the hardware is idle.

Two bounds follow from the definitions. Because \\(T_p > 0\\):

\\[ S(p) = \frac{T_1}{T_p} \le p \quad \text{and} \quad 0 < E(p) \le 1 \\]

Equality holds only when \\(T_p = T_1/p\\), meaning the work divides perfectly
among processors and every processor is busy for the whole execution. Load
imbalance, communication, synchronisation, redundant work, memory contention,
and end-of-computation idle time all increase \\(T_p\\) and reduce efficiency.
The serial fraction introduced by Amdahl's law creates the same ceiling in
another form.

A GPU may have tens of thousands of threads in flight. If the achievable
efficiency is 20%, the hardware is five times larger than the useful work
requires. Most optimisations in this book raise efficiency by keeping threads
busy, keeping memory transactions full, and removing serialisation points.

## 1.3 Amdahl's Law

Gene Amdahl observed in 1967 that every program contains a serial fraction: the
part that cannot be parallelised, such as initialisation, I/O, a single
reduction step, or a dependency chain. Let \\(f\\) be the fraction of the
*serial* execution time that is strictly serial. The parallelisable fraction is
\\((1 - f)\\). If the parallel part is perfectly parallelised across \\(p\\)
units, the best possible total time is:

\\[ T_p = f \cdot T_1 + \frac{(1 - f) \cdot T_1}{p} \\]

The maximum speedup is therefore:

\\[ S(p) = \frac{T_1}{f \cdot T_1 + \frac{(1 - f) \cdot T_1}{p}}
= \frac{1}{f + \frac{1 - f}{p}} \\]

The limit as \\(p \to \infty\\) is:

\\[ \lim_{p \to \infty} S(p) = \frac{1}{f} \\]

**The serial fraction is a hard ceiling.** If 5% of a program is serial, no
amount of parallelism can produce more than a 20x speedup, because the serial
part still requires \\(0.05 \cdot T_1\\) regardless of the number of processing
units.

The serial fraction is easy to underestimate on a GPU. Host launch overhead, a
single reduction step, and dependency chains all count. As a worked example,
consider a pipeline with 10 microseconds of host overhead (serial) and a kernel
that takes 100 microseconds on one GPU and scales perfectly. Here
\\(f = 10/110 \approx 0.091\\), so the maximum speedup is \\(1/0.091 \approx
11\\). No number of GPUs can make the pipeline faster than 11x. Chapter 6
exists because hiding host overhead with streams is one way to reduce the
effective serial fraction.

Amdahl's law assumes a **fixed problem size**. When the problem grows with the
number of processing units, the conclusion changes. That is the subject of the
next section.

## 1.4 Gustafson-Barsis Law

John Gustafson and Edwin Barsis argued in 1988 that users do not normally keep
the problem size fixed when they acquire more hardware. They solve larger
problems in the same wall-clock time. Let \\(s\\) be the serial fraction of the
*parallel* execution time, measured when all \\(p\\) units are busy. The scaled
speedup is:

\\[ S(p) = p + (1 - p) \cdot s \\]

The derivation shows where the formula comes from. Let \\(T_p\\) be the
wall-clock time on \\(p\\) units. Split it into a serial part \\(s \cdot T_p\\)
and a parallel part \\((1 - s) \cdot T_p\\). If the parallel part ran on one
unit instead of \\(p\\), it would take \\(p \cdot (1 - s) \cdot T_p\\). The
serial part takes the same time in either case, so the estimated single-unit
time is:

\\[ T_1 = s \cdot T_p + p \cdot (1 - s) \cdot T_p \\]

and:

\\[ S(p) = \frac{T_1}{T_p} = s + p \cdot (1 - s) = p + (1 - p) \cdot s \\]

The term \\((1 - p) \cdot s\\) is negative for \\(p > 1\\), so the speedup is
always somewhat below \\(p\\). The gap depends on the serial fraction. If
\\(s = 0.01\\) and \\(p = 1,000\\), the scaled speedup is approximately
\\(1,000 - 9.99 \approx 990\\). The same serial fraction would cap Amdahl-style
fixed-workload speedup at 100, because in Gustafson scaling the parallel
workload also grows.

The two laws answer different questions:

- **Amdahl:** "How much faster does a fixed workload run with more units?"
- **Gustafson:** "How much larger a workload can run in the same time with more
  units?"

Increasing an image resolution or a matrix dimension is Gustafson scaling: the
workload grows and the parallel fraction grows with it. Optimising a fixed-size
kernel is Amdahl scaling. Identifying the regime determines which optimisation
is meaningful.

## 1.5 Strong Scaling and Weak Scaling

The two regimes have standard names:

- **Strong scaling** fixes the problem size and increases the number of units.
  Its limit is Amdahl's law. Strong scaling applies to latency-critical
  workloads whose size is fixed by the application, such as a 1080p frame that
  must be processed at 60 Hz.
- **Weak scaling** fixes the problem size *per unit* and increases the number
  of units, so the total problem grows with the hardware. Its limit is
  Gustafson's law. Weak scaling applies to throughput workloads such as larger
  batches or larger grids.

Kernel configuration decisions are strong- or weak-scaling decisions in
miniature. Using more threads per element increases parallel work per thread
(weak scaling); using fewer threads that each do more work holds total work
fixed (strong scaling).

## 1.6 Types of Parallelism

Parallelism is not a single idea. Each form maps to different hardware
mechanisms:

- **Task parallelism** runs different functions concurrently on different data,
  for example decoding one frame while filtering another. On a GPU, task
  parallelism is coarse: the hardware has a small number of independent
  execution contexts, exposed to CUDA as streams (Chapter 6).
- **Data parallelism** runs the same function on many data elements. This is
  the native mode of a GPU: one kernel, millions of elements.
- **Pipeline parallelism** splits a computation into stages and has each stage
  process a different element at the same time. A convolution pipeline may
  load, compute, and store in overlapping stages. On a GPU, pipelining appears
  both in hardware (instruction and memory pipelines) and in software (double
  buffering, Chapter 6).

A GPU is a **data-parallel** machine. When the term "massively parallel" is
used for GPUs, it means data parallelism. Task parallelism on a GPU is an
available but secondary technique.

## 1.7 Flynn's Taxonomy: SISD, SIMD, SIMT, MIMD

Michael Flynn's 1966 taxonomy classifies computers by the number of instruction
streams and data streams they operate on:

- **SISD** (single instruction, single data): a conventional scalar CPU core.
  One instruction stream operates on one data stream.
- **SIMD** (single instruction, multiple data): one instruction operates on a
  vector of data elements. SSE and AVX on x86 CPUs are examples. The compiler
  or programmer packs data into wide registers; a 256-bit AVX register holds
  eight 32-bit floats, and one instruction can add all eight at once.
- **MIMD** (multiple instruction, multiple data): each processing unit runs its
  own instruction stream on its own data. Multi-core CPUs and GPU streaming
  multiprocessors as a whole belong here.
- **SIMT** (single instruction, multiple threads): NVIDIA's execution model,
  combining aspects of SIMD and MIMD. The hardware fetches one instruction per
  cycle for a group of threads called a **warp** (Chapter 2). Each thread has
  its own registers and program counter. The group executes one instruction at
  a time, but each lane applies it to its own data.

SIMT differs from SIMD in an important way. In SIMD, data elements are packed
into a vector register, and all lanes always execute the same instruction. In
SIMT, threads appear to execute independently. The hardware executes them in
lockstep only when their control flow agrees. If threads in the same warp take
different branches, the hardware serialises the branches (Chapter 5). SIMT
therefore provides the programming convenience of MIMD (data-dependent control
flow per thread) while exposing the cost of SIMD when paths diverge.

## 1.8 Arithmetic Intensity and the Roofline Model

The roofline model, introduced by Williams, Waterman, and Patterson in 2009,
answers one question: for a given computation, is the limit set by the
arithmetic units or by the memory system?

### 1.8.1 Arithmetic Intensity: Work per Byte

A GPU has two limiting resources with different units:

- **Arithmetic units** (FP32 cores) perform work at a maximum rate
  \\(P_{\text{peak}}\\) FLOP/s.
- **Memory system** (DRAM, L2, buses) delivers data at a maximum rate
  \\(B\\) bytes/s.

Before the arithmetic units can operate on a value, the value must arrive from
memory. Memory bandwidth is a finite per-second budget. It is therefore useful
to describe a kernel by the ratio of its work to its data movement:

\\[ I = \frac{\text{FLOPs}}{\text{Bytes}} \\]

\\(I\\) is the **arithmetic intensity**: the number of floating-point
operations performed per byte moved. It is analogous to fuel efficiency: miles
per gallon.

The ratio decides which resource runs out first.

- A **low-intensity** kernel performs few operations per byte. It exhausts the
  memory byte budget while the arithmetic units still have capacity. Such a
  kernel is **memory-bound**. Additional arithmetic throughput does not help
  because the memory system is the bottleneck.
- A **high-intensity** kernel performs many operations per byte. The arithmetic
  units saturate before the memory system does. Such a kernel is
  **compute-bound**. Additional bandwidth does not help because the arithmetic
  units are the bottleneck.

Data reuse raises intensity. A byte loaded and used for many operations
contributes to many FLOPs. A byte loaded, used once, and discarded contributes
to one FLOP.

![Arithmetic intensity as work per byte: the same machine, four kernels, and where each sits relative to the ridge point](../../assets/ch01_intensity_scale.svg)

The diagram shows four kernels on the same machine. A vector add moves 12 bytes
(two reads and one write) for one FLOP, giving intensity \\(1/12 \\approx
0.08\\) FLOP/byte. A dense matrix multiply reuses each loaded value many times;
depending on the problem and implementation, its intensity can be hundreds of
FLOP/byte. The machine did not change; the degree of data reuse did.

### 1.8.2 The Ridge Point

Let \\(P_{\text{peak}}\\) be the peak floating-point throughput in FLOP/s and
\\(B\\) the peak memory bandwidth in bytes/s. If a kernel has intensity
\\(I\\), the achievable performance satisfies:

\\[ P \le \min(P_{\text{peak}},\; I \cdot B) \\]

The term \\(I \cdot B\\) follows from unit bookkeeping. A kernel performing
\\(I\\) FLOPs per byte must receive \\(P / I\\) bytes/s to sustain a
performance of \\(P\\) FLOP/s. Because the memory system can deliver at most
\\(B\\) bytes/s:

\\[ \frac{P}{I} \le B \quad \Longrightarrow \quad P \le I \cdot B \\]

This is the **bandwidth ceiling**. The arithmetic units impose the second
ceiling, \\(P_{\text{peak}}\\). Both constraints hold simultaneously, so the
achievable rate is their minimum.

The two ceilings meet at the **ridge point**, the intensity at which the memory
system and arithmetic units are exactly balanced:

\\[ I_{\text{ridge}} = \frac{P_{\text{peak}}}{B} \\]

![The roofline model: the bandwidth diagonal, the arithmetic roof, and the ridge point that separates memory-bound from compute-bound kernels](../../assets/ch01_roofline.svg)

Below the ridge point, performance is limited by the bandwidth diagonal
(\\(I \cdot B\\)). Above it, performance is limited by the arithmetic roof
(\\(P_{\text{peak}}\\)). The ridge point converts the machine's two raw
specifications into one number that can be compared against any kernel.
Hardware chapters in this book quote it for each generation (e.g., §2.1).

**Worked numbers.** Consider a GPU with \\(P_{\text{peak}} = 40\\) TFLOP/s of
FP32 and \\(B = 1\\) TB/s. Its ridge point is \\(40\\) FLOP/byte. A vector add
performs one addition per output element \\(c[i] = a[i] + b[i]\\). It reads two
4-byte floats and writes one 4-byte float, moving 12 bytes:

\\[ I = \frac{1\ \text{FLOP}}{(2\ \text{reads} + 1\ \text{write}) \times
4\ \text{bytes}} = \frac{1}{12} \approx 0.08\ \text{FLOP/byte} \\]

That is roughly 500x below the ridge point. Vector addition is memory-bound; no
arithmetic optimisation helps, while bandwidth optimisation does (coalescing,
§2.7; avoiding redundant reads, Chapter 7). This result motivates Chapter 7:
for many real kernels, the bytes are the problem, not the arithmetic.

### 1.8.3 CPU-Bound, Memory-Bound, and I/O-Bound

The roofline model classifies resources inside the GPU. The same reasoning
applies to whole programs on any machine, where the limiting resource may also
be outside the processor. Every program is eventually limited by one of three
resources:

- **The CPU**: instruction-issue and execution capacity.
- **The memory system**: DRAM bandwidth and latency.
- **An input/output device**: disk, network, PCIe bus, or GPU transfer.

A program is named after the resource that limits its total execution time.

> **Definition - the bound of a program.** A program is *bound by resource
> \\(R\\)* if its total execution time is approximately the time it spends
> using (or waiting on) \\(R\\). Concretely: \\(R\\)'s utilisation is near 100%
> while the other resources idle, and increasing \\(R\\)'s capacity alone
> reduces total time proportionally, while increasing any other resource's
> capacity changes nothing.

If the total time \\(T\\) is split into busy time for the CPU, memory, and I/O,
the verdicts follow from one ratio:

- **CPU-bound**: \\(T_{\text{CPU}} / T \approx 1\\). Example: SHA-256 hashing
  of ten million in-memory keys. The data is already in RAM, so memory and I/O
  idle while the CPU executes thousands of instructions per key. A faster CPU
  reduces the time; faster RAM or a faster disk does not.
- **Memory-bound**: \\(T_{\text{mem}} / T \approx 1\\). Example: the vector add
  of §1.8.2, whose intensity lies far below the ridge point. Faster memory
  helps; additional arithmetic throughput does not.
- **I/O-bound**: \\(T_{\text{io}} / T \approx 1\\). Example: streaming 10 GB
  from disk. CPU and memory idle waiting for data. A faster disk, or
  asynchronous I/O (Chapter 6), helps; a faster CPU does not.

![The bound taxonomy: three programs, three walls. Each panel shows the utilisation of CPU, memory and I/O during one run; the resource pinned at ~100% is the one the program is named after, and the only one worth optimising.](../../assets/ch01_bound_taxonomy.svg)

![Where the time goes: the same three programs as timelines. Rows are resources; coloured blocks are busy intervals, dark gaps are idle time. The resource whose row is almost solid is the bound; the composition bar below each example shows the dominant colour.](../../assets/ch01_bound_examples.svg)

A practical test identifies the bound. Double the capacity of one resource and
re-measure:

| You double... | ...and total time halves? | Then you are |
|---|---|---|
| CPU clock | ✓ | CPU-bound |
| Memory bandwidth | ✓ | Memory-bound |
| Disk / network / PCIe rate | ✓ | I/O-bound |
| All three | ✗ | Serial-bound (Amdahl, §1.3) |

Each optimisation chapter in this book is an argument that a specific resource
is the bottleneck. Coalescing (§2.7) targets memory-bound kernels. Register
tiling (Chapter 9) targets compute-bound kernels. Streams and asynchronous
transfers (Chapter 6) target I/O-bound kernels. The GPU-side term
*compute-bound* corresponds to the CPU-side term *CPU-bound*: in both cases the
processor is the limiting resource. Profiling (Chapter 16) exists to determine
which resource is saturated before optimising.

## 1.9 The Cost of Synchronisation

Parallel work must occasionally rendezvous: threads must order their access to
shared data or combine partial results. The programming primitives are
introduced in Chapter 5, but their economics belong in this chapter.

Three costs attend any synchronisation point:

1. **Idle time.** Threads waiting at a barrier cannot use their execution
   units. A barrier converts available parallelism into a serial stall.
2. **Memory visibility.** For one thread to observe another thread's write, the
   write must become visible through the memory system. On a GPU this requires
   cache flushes or atomic operations and is not free.
3. **Load imbalance.** A barrier completes only when the slowest thread
   reaches it. If one thread does ten times the work of its neighbours, every
   barrier waits for that straggler.

The engineering consequence is to minimise the number of synchronisation points
and keep the work between them balanced. The optimised reduction in Chapter 8
reduces the number of block-level barriers from \\(\log_2 N\\) to a constant
for this reason.

## 1.10 Vocabulary Summary

The terms defined in this chapter form the working vocabulary of the book:

| Term | Definition | Defined |
|---|---|---|
| Latency | Time from start to completion of one operation | §1.1 |
| Throughput | Operations completed per unit time | §1.1 |
| Speedup \\(S(p)\\) | \\(T_1 / T_p\\) | §1.2 |
| Efficiency \\(E(p)\\) | \\(S(p)/p\\) | §1.2 |
| Serial fraction \\(f\\) | The non-parallelisable fraction | §1.3 |
| Strong scaling | Fixed problem, more units | §1.5 |
| Weak scaling | Fixed per-unit problem, more units | §1.5 |
| SIMT | Single instruction, multiple threads | §1.7 |
| Arithmetic intensity \\(I\\) | FLOPs per byte moved | §1.8 |
| Ridge point | \\(P_{\text{peak}} / B\\) | §1.8 |
| Memory-bound | \\(I < I_{\text{ridge}}\\), limited by bandwidth | §1.8 |
| Compute-bound | \\(I > I_{\text{ridge}}\\), limited by FLOPs | §1.8 |
| CPU-bound | \\(T_{\text{CPU}} / T \approx 1\\), limited by the processor | §1.8 |
| I/O-bound | \\(T_{\text{IO}} / T \approx 1\\), limited by disk/network/PCIe | §1.8 |

## Amdahl and Gustafson in Practice

The two laws are not competing claims. They answer different questions under
different assumptions.

Amdahl's law fixes the problem size and the serial fraction, then asks how fast
the same problem can run with more processors. It states that the serial part
is an unremovable floor. Gustafson's law allows the workload to grow with the
hardware and asks how much work can finish in the same wall-clock time. Real
users usually follow the Gustafson pattern: when a machine grows, they run
larger models, higher-resolution images, or larger batches rather than rerunning
the same small problem faster.

Making a fixed 1080p frame meet a 60 Hz deadline is an Amdahl regime: the work
is fixed, and every microsecond of host overhead or synchronisation contributes
to the serial fraction that caps speedup. Increasing batch size after adding
GPUs is a Gustafson regime: the total work grows with the hardware, and the
parallel fraction grows with it.

A speedup number is meaningful only with context. Any claim should state the
problem size, the hardware, and whether the serial fraction was measured or
assumed. Chapter 16 therefore requires recording the environment and
methodology for every performance claim.

There is a second consequence of Amdahl's law that is often missed: the serial
fraction is a property of the chosen *decomposition*, not an immutable property
of the program. A reduction that appears serial in one formulation can become
parallel with a tree. A host-side copy that appears to be overhead can be
hidden with streams. The laws do not state what \\(f\\) is; they state what
\\(f\\) costs. Reducing \\(f\\) is one of the main activities of GPU
engineering and recurs throughout the book.

## Common Pitfalls

- Quoting speedup without efficiency. A "64x speedup on 128 cores" is 50%
  efficiency; half the machine is idle.
- Forgetting that the serial fraction includes host-side overhead. Launch
  overhead, copies, and synchronisation are part of \\(f\\), sometimes the
  dominant part.
- Treating a memory-bound kernel as compute-bound. If \\(I < I_{\text{ridge}}\\),
  adding arithmetic throughput does not help; the memory system is the limit.
- Confusing strong and weak scaling when designing experiments. State whether
  the problem size is fixed or grows with the device count.

## Check Your Understanding

<details>
<summary>Why is efficiency a more informative metric than speedup?</summary>

Efficiency divides speedup by the number of processing units. A vendor can
report "10x speedup on 64 cores", but efficiency is only 10/64 = 0.156: 84% of
the hardware contributes nothing beyond what the serial run already achieved.
Efficiency exposes the waste that raw speedup hides.
</details>

<details>
<summary>A kernel has 2% serial time. What is the Amdahl ceiling?</summary>

The maximum speedup is 1/f = 1/0.02 = 50x, no matter how many cores or GPUs
are added. The 2% serial part alone takes 2% of the original time, so total
time cannot go below that.
</details>

<details>
<summary>Vector add has intensity 0.08 FLOP/byte on a machine with ridge 40. Is it memory-bound or compute-bound?</summary>

Memory-bound. 0.08 is far below the ridge point, so the bandwidth diagonal
caps performance at \\(I \times B\\). The FP32 units are idle waiting for bytes;
additional arithmetic throughput changes nothing.
</details>

## Key Takeaways

- Parallelism buys throughput, not latency. The GPU hides latency by keeping many warps in flight.
- Speedup is \\(S(p) = T_1 / T_p\\); efficiency \\(S(p) / p\\) is the informative metric.
- Amdahl's law: a serial fraction \\(f\\) caps fixed-workload speedup at \\(1/f\\), regardless of unit count.
- Gustafson-Barsis: when the problem grows with the hardware, scaled speedup grows linearly.
- Arithmetic intensity \\(I = \\) FLOPs / bytes, compared with the ridge point \\(P_{\text{peak}} / B\\), decides memory-bound vs compute-bound.
- A program is bound by the resource whose utilisation is near 100%: CPU, memory, or I/O. If doubling one resource halves the time, that resource is the wall.
- SIMT executes one instruction per warp; divergent control flow serialises divergent paths.

## 1.11 Exercises

1. A kernel has a serial fraction of \\(f = 0.02\\). What is the maximum
   speedup Amdahl's law permits, regardless of hardware?
2. A vector add moves 4 bytes per element read and 4 bytes per element
   written, and performs 1 FLOP per element. Compute its arithmetic intensity.
3. On a machine with a ridge point of 40 FLOP/byte, is the vector add
   memory-bound or compute-bound? What is the maximum utilisation of peak
   FLOPs achievable?
4. Explain why a GPU designed for throughput executes a warp of threads in
   lockstep even though each thread has its own program counter.
5. A program downloads 10 GB from the network at 2 GB/s and compresses it on
   the CPU at 20 GB/s. Estimate the CPU utilisation. Is the program CPU-bound,
   memory-bound or I/O-bound? Which single change - a 2x faster CPU or a 2x
   faster network - speeds it up more?

## Sources and Further Reading

- Gene M. Amdahl, "Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities," AFIPS Conference Proceedings, 1967. The paper behind Amdahl's law.
- John L. Gustafson, "Reevaluating Amdahl's Law," Communications of the ACM 31(5), 1988. The paper behind Gustafson's scaled-speedup law.
- Samuel Williams, Andrew Waterman, and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Communications of the ACM 52(4), 2009. The original roofline paper.
- NVIDIA, *CUDA C++ Programming Guide*, "Compute Capabilities" appendix, for authoritative per-generation hardware limits: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *CUDA C++ Best Practices Guide*, for optimisation methodology: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
