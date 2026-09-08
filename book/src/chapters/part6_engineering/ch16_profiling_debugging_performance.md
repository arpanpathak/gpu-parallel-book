# Chapter 16: Profiling, Debugging & Performance Engineering

> *"We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil."*
> — Donald E. Knuth, "Structured Programming with go to Statements" (1974)

Every earlier chapter claimed that one implementation is faster than another.
This chapter is about proving such claims. It covers the four instruments of
GPU engineering: **Nsight Systems** and **Nsight Compute** (profilers),
**Compute Sanitizer** (debugger), and the **benchmarking discipline**. It also
covers differential and property testing, which keep optimisations grounded.
After this chapter you can take any kernel from this book and answer two
questions reproducibly: is it correct, and is it fast?

## 16.1 The Two Profilers

NVIDIA ships two profilers with distinct jobs:

> **Primitive - Nsight Systems (`nsys`).** A system-level profiler. It shows a
> timeline of the whole program: when kernels ran, when transfers ran, when the
> CPU was idle, and how streams overlapped. It answers where the time goes and,
> in particular, whether the GPU was idle waiting for the host.
> **Primitive - Nsight Compute (`ncu`).** A kernel-level profiler. It reports
> per-kernel hardware counters: achieved occupancy, memory throughput,
> shared-memory bank conflicts, warp stall reasons, and FLOP counts. It answers
> why a kernel is slow.

The workflow is `nsys` first, `ncu` second. If the GPU is idle 40% of the time,
kernel tuning does not help; the fix is streams and overlap (Chapter 6). Only
when the timeline shows the GPU busy should a kernel be examined with `ncu`.

```bash
# System-level timeline (10 seconds of the application):
nsys profile --duration=10 ./pipeline

# Kernel-level analysis of the sobel kernel (the profiler replays the kernel
# under instrumentation):
ncu --kernel-name regex:sobel --set full ./pipeline
```

`ncu` replays the kernel because collecting full counters changes timing. It
runs the kernel multiple times under instrumentation and aggregates counters,
so the numbers describe the kernel rather than the profiler's overhead. This is
also why `ncu` cannot profile every metric group at once; use `--set` presets.

## 16.2 Reading the Timeline (nsys)

A healthy streamed pipeline (Chapter 15) shows:

![Nsight Systems timeline: copies and kernels overlap, keeping the GPU busy](../../assets/ch16_nsys_timeline.svg)

The GPU bar is continuously busy: copies and kernels overlap, and the only gaps
are pipeline priming. Unhealthy timelines and their diagnoses:

| Timeline symptom | Diagnosis | Fix |
|---|---|---|
| GPU idle between kernel and next copy | Default-stream serialisation | Name streams, use `cudaStreamNonBlocking` (Ch. 6) |
| Small kernel gaps every frame | Host launch overhead | CUDA Graphs (Ch. 6, §6.7) |
| Long grey "CPU time" blocks | Host-side stall (I/O, alloc) | Pre-allocate, pin memory (Ch. 4) |
| Copy and kernel never overlap | Pageable memory | `cudaMallocHost` (Ch. 4) |

Each symptom has a one-line fix and each fix maps to a chapter already read.

## 16.3 The Kernel Report (ncu)

For a single kernel, `ncu --set full` reports the metrics this book has trained
the reader to interpret:

- **Achieved occupancy** (§2.9): warps resident versus the theoretical maximum.
  Low occupancy plus memory stalls means too few warps are available to hide
  latency.
- **Memory throughput**: percentage of peak DRAM bandwidth used. Near 100% on a
  memory-bound kernel means coalescing is working. Far below it means the
  checklist of Chapter 7 (§7.9) applies.
- **Shared-memory bank conflicts**: cycles lost to bank conflicts (§2.8, §7.5).
  Zero is achievable with padding.
- **Warp stall reasons**: why warps wait. `long_scoreboard` means waiting on a
  global load; `short_scoreboard` means shared memory; `barrier` means waiting
  at `__syncthreads`; `drain` means stores are not flushed. Each stall reason
  points at a different chapter of this book.

The discipline is: record the metric, form a hypothesis, change one thing, and
re-measure. Two simultaneous changes make a measurement uninterpretable.

## 16.4 Compute Sanitizer

> **Primitive - Compute Sanitizer (`compute-sanitizer`).** A runtime tool that
> instruments a kernel to detect memory and synchronisation errors that would
> otherwise be silent: out-of-bounds accesses, misaligned accesses, data races,
> and invalid `__syncthreads` usage.

```bash
# Memory checking: out-of-bounds, uninitialised, and misaligned accesses.
compute-sanitizer --tool memcheck ./pipeline

# Race checking: data races between threads (Chapter 5's bug class).
compute-sanitizer --tool racecheck ./pipeline

# Initialisation checking: reads of uninitialised memory.
compute-sanitizer --tool initcheck ./pipeline

# Synchronisation checking: divergent __syncthreads (Chapter 5, 5.3).
compute-sanitizer --tool synccheck ./pipeline
```

A CPU out-of-bounds write usually crashes at the instruction. A GPU
out-of-bounds write can corrupt adjacent memory in the same allocation: the
kernel reports success and the corruption appears three stages later as a wrong
image. `memcheck` identifies the write at the moment it happens, with the
thread and instruction.

Every race, bank conflict, and divergence class discussed in Chapters 5 and 7
has a detector. Run the detector before trusting reasoning.

## 16.5 cuda-gdb

For bugs that resist automatic tools, `cuda-gdb` provides an interactive
debugger for device code: breakpoints inside kernels, inspection of
`threadIdx` and `blockIdx`, register and shared-memory watches, and
warp-by-warp stepping.

```bash
cuda-gdb ./pipeline
(cuda-gdb) break sobel
(cuda-gdb) run
(cuda-gdb) set cuda break_on_launch application   # stop at every kernel
(cuda-gdb) info cuda kernels                      # list active kernels
(cuda-gdb) thread 5                               # select a specific thread
(cuda-gdb) print x                                # inspect kernel variables
```

Reach for `cuda-gdb` after Compute Sanitizer has cleared memory and trace
errors. A remaining logical bug, such as wrong index arithmetic or wrong stencil
weights, can be inspected by breaking on a kernel and checking a specific
thread's values by hand.

## 16.6 clock64(): Timing Inside the Kernel

Profiler replay can change the answer for kernels whose performance depends on
cache state. `clock64()` reads a per-SM cycle counter from inside the kernel:

```cpp
// Time a code region from INSIDE the kernel. Returns SM cycles.
// Useful when profiler replay perturbs the measurement; otherwise prefer ncu.
__device__ long long profileRegion()
{
    const long long t0 = clock64();
    // ... the region being timed ...
    const long long t1 = clock64();
    return t1 - t0;              // SM cycles (see device clock rate)
}
```

`clock64()` measures this thread's view: the scheduler may preempt the warp
mid-region, and the SM clock can vary with power state. Use it for relative
comparisons of code paths within one kernel run, not as a cross-run benchmark.
For cross-run numbers, use CUDA events (Chapter 6) and the discipline of §16.7.

## 16.7 The Benchmarking Discipline

A number from one run is not a measurement. The reproducible protocol used for
every claim in this book is:

1. **Warm up.** Run the kernel several times before measuring so that caches,
   page tables, and JIT state are steady.
2. **Repeat and report the median**, not the mean. The median is robust to
   outliers such as OS preemption and clock boost. Report the spread, for
   example P10/P90, alongside.
3. **Use events, not host timers**, for device work (Chapter 6, §6.4).
4. **Fix the environment.** Record the GPU (`nvidia-smi -L`), the CUDA version
   (`nvcc --version`), the driver, and the compiler flags. `-arch`, `-O3`, and
   `--use_fast_math` all change results.
5. **Verify the output** before trusting the timing. A fast wrong kernel is not
   a result.

```cpp
// The protocol in miniature. ncu or nsys can validate further, but this
// structure is the minimum reproducible measurement.
float benchmarkKernel(int iters)
{
    // warm-up:
    kernel<<<grid, block>>>(...);  cudaDeviceSynchronize();

    std::vector<float> times;
    for (int r = 0; r < iters; ++r)
    {
        cudaEventRecord(start);  kernel<<<grid, block>>>(...);
        cudaEventRecord(stop);   cudaEventSynchronize(stop);
        float ms;  cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];   // median
}
```

## 16.8 Verification: Differential and Property Testing

Performance engineering without correctness is not useful. Two techniques from
the capstone generalise:

- **Differential testing** compares the GPU result with a trusted CPU
  reference (Chapter 15, §15.8). Run it in CI on every change. A refactor that
  changes the last bit of a reduction (Chapter 5, §5.6) is caught rather than
  shipped.
- **Property testing** asserts invariants that hold for any input: a
  histogram's counts sum to the input length, a transpose's output is the
  input's transpose, and an edge map of a constant image is all zeros.
  Property tests find bugs that differential tests miss because both
  implementations can be wrong in the same way.

Once the differential and property suites exist, an optimisation is just a
change run through the suite. This is what lets the optimisation progression of
Chapters 7-9 proceed without fear.

## 16.9 The Engineering Loop

The chapter reduces to a loop:

1. **Measure** (`nsys` timeline; is the GPU busy?).
2. **Profile** (`ncu`; what is the kernel's bottleneck?).
3. **Hypothesise** (name the chapter that addresses the bottleneck).
4. **Change one thing**.
5. **Re-measure** (median of many runs, fixed environment).
6. **Verify** (differential and property tests still pass).

Skipping step 2 or 6 is guessing. Following all six steps is engineering.
Everything in this book, the roofline of Chapter 1, the coalescing of Chapter 7,
and the pipelines of Chapter 15, is an argument about what step 3 should say.
The loop is how the argument is checked.

## The Profiler as a Hypothesis Machine

Nsight Compute reports many metrics, and it is tempting to read them as grades:
occupancy 80%, memory throughput 90%, therefore good. The professional reading
is different: each metric is a hypothesis about why the kernel is not as fast
as it could be, and metrics only make sense when combined into an explanation.

A kernel with low achieved occupancy and high memory stall time suggests that
too few warps are resident to hide global-load latency. The fix is not
"increase occupancy"; it is "increase the pool of ready warps without spilling
registers or starving shared memory", which may mean reducing registers,
changing block size, or restructuring the kernel. A kernel with very high memory
throughput on a stage the roofline predicted to be memory-bound confirms that
memory is the wall; the move is to reduce the amount of memory traffic by
tiling, vectorisation, or an algorithmic change, not to tune the access pattern
further. A stall reason such as `long_scoreboard` points to global loads;
`barrier` points to `__syncthreads`. Each points to a different chapter.

The same mindset applies to Compute Sanitizer. A clean `memcheck` or
`racecheck` run is evidence that one class of bug was not detected on the inputs
that ran. Races are timing-dependent and memory errors depend on the addresses
touched. The tools find classes of bugs; differential and property tests verify
behaviour. Together they make an optimisation trustworthy.

The engineering loop exists because a single number does not give the answer.
It says what to change next. The kernel is the hypothesis, the profiler is the
instrument, and the median-of-many-runs measurement is the experiment.

## Common Pitfalls

- Profiling a debug build. Optimised builds have different register usage,
  inlining, and performance; always profile release builds.
- Trusting one run. Kernels are subject to clock boost, thermal state, and OS
  noise; report the median of many runs.
- Changing two variables between measurements. The result is uninterpretable;
  change one thing, re-measure, and repeat.
- Skipping warm-up. The first launch pays JIT, page-table, and cache warm-up
  costs that are not part of steady-state performance.
- Believing a fast wrong kernel is a result. Verify output before trusting
  timings.

## Check Your Understanding

<details>
<summary>Why does ncu replay the kernel under instrumentation?</summary>

Full counter collection changes timing and state. Replaying the kernel multiple
times under instrumentation lets the profiler aggregate hardware counters
without the distortion of a single heavily instrumented run.
</details>

<details>
<summary>Why report the median instead of the mean?</summary>

The median is robust to outliers such as OS preemption, clock boost, and
thermal throttling. The mean is dragged by rare large outliers and does not
represent the typical run.
</details>

<details>
<summary>What does racecheck catch that a passing test does not?</summary>

A race is timing-dependent and may not manifest on the inputs tested.
`racecheck` instruments memory accesses and detects unsynchronised read/write
pairs even when the race happens to produce the right answer on the test cases.
</details>

## Key Takeaways

- nsys answers where the time goes; ncu answers why a kernel is slow.
- Compute Sanitizer finds what kernels hide: memcheck, racecheck, initcheck, synccheck.
- cuda-gdb debugs kernels interactively, thread by thread.
- The benchmark protocol: warm up, repeat, report the median, use events, fix the environment, verify the output.
- The loop measure, profile, hypothesise, change one thing, re-measure, verify underlies every performance claim in this book.

## 16.10 Exercises

1. A kernel shows 100% memory throughput in `ncu`, but the application is slow.
   Which profiler do you run next, and what do you look for?
2. `racecheck` reports a race in a kernel that passes all tests. Explain why
   this is not a contradiction and what class of bug it represents (Chapter 5,
   §5.1).
3. Why does the benchmark report the median rather than the mean? Give a
   concrete source of outliers that the median neutralises.
4. A colleague optimises a kernel and reports a 30% speedup measured with
   `std::chrono` around a single launch. List three things wrong with that
   measurement.

## Sources and Further Reading

- NVIDIA, *Nsight Systems User Guide*: <https://docs.nvidia.com/nsight-systems/>
- NVIDIA, *Nsight Compute User Guide*: <https://docs.nvidia.com/nsight-compute/>
- NVIDIA, *Compute Sanitizer User Guide*: <https://docs.nvidia.com/compute-sanitizer/>
- NVIDIA, *CUDA C++ Best Practices Guide*, "Profiling" section: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
