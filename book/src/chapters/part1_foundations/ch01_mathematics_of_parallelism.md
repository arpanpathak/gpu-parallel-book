# Chapter 1: The Mathematics of Parallelism

> *"A fast program is not the same thing as a parallel program. The mathematics
> below is the difference between the two."*

Before we touch a single CUDA API, we must understand what parallelism can and
cannot buy us. This chapter establishes the mathematical vocabulary of the
entire book: *speedup*, *efficiency*, *Amdahl's law*, *scaling*, *Flynn's
taxonomy*, and the *roofline model*. None of these ideas are optional context;
they are the instruments you will use to decide, for every kernel in this book,
whether a given optimisation is worth the effort.

## 1.1 Latency, Throughput, and the Meaning of "Faster"

When we say a program is "slow", we usually mean one of two things, and the
distinction matters enormously on a GPU.

- **Latency** is the time between the start of an operation and its
  completion. A network round trip has latency. A single memory access has
  latency. Latency is measured in time units (nanoseconds, milliseconds).
- **Throughput** is the number of operations completed per unit time. A
  pipeline that processes 60 frames per second has a throughput of 60 Hz.
  Throughput is measured in operations per second.

A CPU is designed to **minimise latency**: a small number of very fast cores,
each executing one instruction stream with a branch predictor, out-of-order
execution, and large caches to hide the latency of DRAM.

A GPU is designed to **maximise throughput**: a very large number of simple
execution units, none of which is individually fast, but which together
complete millions of operations per clock. The GPU hides latency not by
predicting what happens next, but by having so many independent threads in
flight that the hardware always has something to do while others wait.

This is the first primitive of the book:

> **Primitive - latency hiding.** If an execution unit must wait for a slow
> operation (a memory access, a division), the unit is idle. The GPU avoids
> idleness by switching to another ready thread. The cost of the wait is
> hidden, not eliminated.

Consequently, a GPU is a poor tool for a single sequential computation and an
excellent tool for a computation that can be decomposed into many independent
pieces. The mathematics of that decomposition is the subject of this chapter.

## 1.2 Speedup and Efficiency

Let \\(T_1\\) be the time a program takes to solve a problem on a single
processing unit (a single core, a single thread), and let \\(T_p\\) be the time
it takes on \\(p\\) processing units. We define:

**Speedup** - the ratio of the serial time to the parallel time:

\\[ S(p) = \frac{T_1}{T_p} \\]

A perfect speedup of \\(p\\) means the \\(p\\)-fold work is done in
\\(\frac{1}{p}\\) the time. We then define:

**Efficiency** - the speedup per processing unit:

\\[ E(p) = \frac{S(p)}{p} = \frac{T_1}{p \cdot T_p} \\]

An efficiency of 1.0 (100%) is ideal: every processing unit contributes
proportionally. An efficiency of 0.5 means half of the processing units'
potential is being wasted. Efficiency is the honest measure; speedup is the
flattering one. A vendor will report "10x speedup on 64 cores" and omit that
the efficiency is 0.156.

**Why efficiency matters on a GPU.** A GPU may have tens of thousands of
threads in flight. If the achievable efficiency is 20%, you are paying for
five times more hardware than you are using. Almost every optimisation in this
book is, at heart, an attempt to raise efficiency - by keeping threads busy,
by keeping memory transactions full, and by removing serialisation points.

## 1.3 Amdahl's Law

Gene Amdahl observed in 1967 that any program has a serial fraction: the part
that cannot be parallelised (initialisation, I/O, a single reduction step, a
dependency chain). Let \\(f\\) be the fraction of the *serial* execution time
that is strictly serial. The parallelisable fraction is \\((1 - f)\\). If the
parallel part is perfectly parallelised across \\(p\\) units, the best possible
total time is:

\\[ T_p = f \cdot T_1 + \frac{(1 - f) \cdot T_1}{p} \\]

and therefore the maximum speedup is:

\\[ S(p) = \frac{T_1}{f \cdot T_1 + \frac{(1 - f) \cdot T_1}{p}}
= \frac{1}{f + \frac{1 - f}{p}} \\]

The crucial property is the limit as \\(p \to \infty\\):

\\[ \lim_{p \to \infty} S(p) = \frac{1}{f} \\]

**The serial fraction is a hard ceiling.** If 5% of your program is serial,
no amount of parallelism can exceed a 20x speedup, because the serial part
still takes \\(0.05 \cdot T_1\\) regardless of how many units you add.

**Worked example.** Consider a kernel launch pipeline: 10 microseconds of host
overhead (serial) plus a kernel that takes 100 microseconds on one GPU and
scales perfectly. Here \\(f = 10/110 \\approx 0.091\\). The maximum speedup is
\\(1/0.091 \\approx 11\\). No matter how many GPUs you buy, the pipeline cannot
be faster than 11x. This is why Chapter 6 (streams and asynchronous execution)
is dedicated to hiding host overhead: the serial fraction is the enemy.

**Why Amdahl's law is pessimistic.** Amdahl assumed the *problem size is
fixed*. If the problem grows with the number of processing units, the
conclusion changes. That is the subject of the next section.

## 1.4 Gustafson-Barsis Law

John Gustafson and Edwin Barsis argued in 1988 that in practice the problem
size is not fixed: given more hardware, users solve *larger* problems in the
same wall-clock time. Let \\(s\\) be the serial fraction of the *parallel*
execution time (the time when all \\(p\\) units are busy). The scaled speedup
is:

\\[ S(p) = p + (1 - p) \cdot s \\]

Unlike Amdahl's law, this grows *linearly* with \\(p\\) for fixed \\(s\\).
The two laws answer different questions:

- **Amdahl:** "How much faster does my *fixed* workload run with more units?"
- **Gustafson:** "How much *larger* a workload can I run in the same time with
  more units?"

**Why both matter for GPU programming.** When you increase the image resolution
or the matrix dimension, you are doing Gustafson scaling: the workload grows,
and the GPU's parallel fraction grows with it. When you optimise a fixed-size
kernel, you are fighting Amdahl's law. Knowing which regime you are in tells
you which optimisation is worthwhile.

## 1.5 Strong Scaling and Weak Scaling

These two terms name the two regimes above:

- **Strong scaling** - fixed problem size, increasing units. The limit is
  Amdahl's law. Used for latency-critical workloads where the problem size is
  dictated by the application (a 1080p frame must be processed at 60 Hz).
- **Weak scaling** - fixed problem size *per unit*, increasing units. The
  total problem grows with the units. The limit is Gustafson's law. Used for
  throughput workloads (larger batch, larger grid).

Every kernel configuration decision in this book is a strong-vs-weak scaling
decision in miniature: whether to use more threads per element (weak, more
parallel work per unit) or fewer threads doing more work each (strong, fixed
total work).

## 1.6 Types of Parallelism

Parallelism is not one idea but several, and each maps to different hardware:

- **Task parallelism** - different *functions* run concurrently on different
  data (e.g., decode one frame while filtering another). On a GPU, task
  parallelism is coarse and limited: a GPU has few independent "task" slots,
  but they correspond to *streams* (Chapter 6).
- **Data parallelism** - the *same* function runs on many data elements. This
  is the natural mode of the GPU: one kernel, millions of elements.
- **Pipeline parallelism** - a computation is split into stages; each stage
  processes a different element simultaneously. The classic example is a
  convolution pipeline: stage one loads, stage two computes, stage three
  stores. On a GPU, pipeline parallelism appears both at the hardware level
  (the memory pipeline, the instruction pipeline) and at the application level
  (double buffering, Chapter 6).

A GPU is a **data-parallel** machine. When you read "massively parallel", the
word "parallel" means "data parallel". Task parallelism on a GPU is an
afterthought; data parallelism is the design centre.

## 1.7 Flynn's Taxonomy: SISD, SIMD, SIMT, MIMD

Michael Flynn's 1966 taxonomy classifies computers by whether they operate on
one or many *instruction streams* and one or many *data streams*. The four
combinations are:

- **SISD** (single instruction, single data) - a conventional scalar CPU core.
  One instruction stream, one data stream. Your laptop's cores, in scalar mode.
- **SIMD** (single instruction, multiple data) - one instruction operates on a
  *vector* of data elements simultaneously. Examples: SSE and AVX on x86 CPUs.
  The programmer (or compiler) explicitly packs data into wide registers; a
  256-bit AVX register holds eight 32-bit floats, and one instruction adds all
  eight at once.
- **MIMD** (multiple instruction, multiple data) - each processing unit runs
  its own instruction stream on its own data. Examples: multi-core CPUs, GPU
  streaming multiprocessors as a whole.
- **SIMT** (single instruction, multiple threads) - NVIDIA's execution model,
  a hybrid of SIMD and MIMD. The hardware fetches *one* instruction per cycle
  for a *group* of threads (a **warp**, defined in Chapter 2), but each thread
  has its own registers, its own program counter, and its own data. This
  combination - one instruction, many independent thread contexts - is the
  single most important architectural idea in this book.

**Why SIMT is not SIMD.** In SIMD, the data elements are explicitly packed
into a vector register, and divergence is impossible: all lanes execute the
same instruction, always. In SIMT, threads *appear* to execute independently;
the hardware executes them in lockstep *when their control flow agrees*. If
threads in the same warp take different branches, the hardware serialises the
branches (Chapter 5). SIMT gives you the programming convenience of MIMD
(each thread can follow its own data-dependent path) with the cost of SIMD
when paths diverge.

## 1.8 Arithmetic Intensity and the Roofline Model

The roofline model, introduced by Williams, Waterman and Patterson in 2009, is
the most useful performance model in this book. It answers one question: *for
a given computation, is the limit set by the arithmetic units or by the memory
system?*

Define:

**Arithmetic intensity** - the ratio of floating-point operations to bytes
moved, \\(I = \frac{\text{FLOPs}}{\text{Bytes}}\\). A dense matrix multiply has
high intensity (many operations per byte); a vector add has low intensity
(one operation per byte).

Let \\(P_{\text{peak}}\\) be the machine's peak floating-point throughput
(FLOP/s) and \\(B\\) its peak memory bandwidth (bytes/s). The **ridge point**
is the intensity at which the two limits meet:

\\[ I_{\text{ridge}} = \frac{P_{\text{peak}}}{B} \\]

The achievable performance obeys:

\\[ P \le \min(P_{\text{peak}},\; I \cdot B) \\]

In words: if your intensity is below the ridge point, you are **memory-bound**
and performance is capped by bandwidth (\\(I \cdot B\\)); if above, you are
**compute-bound** and capped by peak FLOP rate.

**Worked numbers.** An RTX-class GPU with \\(P_{\text{peak}} = 40\\) TFLOP/s
of FP32 and \\(B = 1\\) TB/s has a ridge point of
\\(I_{\text{ridge}} = 40\\) FLOP/byte. A vector add moving 4-byte floats has
intensity \\(I = 1\\) FLOP / (2 reads + 1 write) × 4 bytes ≈ 0.08 FLOP/byte  - 
deep in memory-bound territory. No amount of arithmetic optimisation will make
a vector add faster; only bandwidth optimisation will. This single
observation explains why Chapter 7 is devoted to memory.

## 1.9 The Cost of Synchronisation

Parallel work must occasionally rendezvous: threads must agree on an order, or
share a partial result. The primitive operations are introduced in Chapter 5,
but the *economics* belong here.

Three costs attend any synchronisation point:

1. **Idle time.** While threads wait at a barrier, their execution units do
   nothing. The barrier converts available parallelism into a serial stall.
2. **Memory visibility cost.** For one thread to see another thread's write,
   the write must be flushed and made visible (caches must be coherent or
   bypassed). On a GPU this is not free.
3. **Load imbalance.** A barrier is only as fast as the *slowest* thread
   reaching it. If one thread does ten times the work of its neighbours, every
   barrier in the loop pays for the straggler.

The engineering corollary is: **minimise the number of synchronisation points,
and make the work between them as balanced as possible.** The optimised
reduction in Chapter 8 exists almost entirely to reduce the number of
block-level barriers from \\(\log_2 N\\) to a constant.

## 1.10 A Vocabulary Summary

The terms defined in this chapter are the book's working vocabulary:

| Term | Definition | First used in anger |
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

## Key Takeaways

- Parallelism buys throughput, not latency; the GPU hides latency by keeping many warps in flight.
- Speedup S(p) = T1 / Tp; efficiency S(p) / p is the honest metric.
- Amdahl's law: a serial fraction f caps speedup at 1/f, no matter how many units you add.
- Gustafson-Barsis: when the problem grows with the hardware, scaled speedup grows linearly.
- Arithmetic intensity I = FLOPs / bytes, compared with the ridge point P_peak / B, decides memory-bound vs compute-bound.
- SIMT executes one instruction per warp; divergent control flow serialises the divergent paths.

## 1.11 Exercises

1. A kernel has a serial fraction of \\(f = 0.02\\). What is the maximum
   speedup Amdahl's law permits, regardless of hardware?
2. A vector add moves 4 bytes per element read and 4 bytes per element
   written, and performs 1 FLOP per element. Compute its arithmetic intensity.
3. On a machine with a ridge point of 40 FLOP/byte, is the vector add
   memory-bound or compute-bound? What is the maximum utilisation of peak
   FLOPs achievable?
4. Explain, in your own words, why a GPU designed for throughput would
   willingly execute a warp of threads in lockstep even though each thread has
   its own program counter.
