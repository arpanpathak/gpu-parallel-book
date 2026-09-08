# Chapter 2: GPU Hardware from First Principles

> *"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."*
> — Edsger W. Dijkstra, "The Humble Programmer" (1972)

CUDA's programming model exposes a grid of blocks, a block of threads, and a
hierarchy of memories. The hardware underneath is less regular, and performance
depends on its details. This chapter describes the machine: the *streaming
multiprocessor*, the *warp*, the *memory hierarchy*, and the rules of
*coalescing* and *occupancy*. Every term defined here is used in later
chapters.

The running example is a modern NVIDIA GPU of the Hopper family (compute
capability 9.0, such as the H100). Numbers differ between generations, but the
structure does not.

## 2.1 The GPU at a Glance

A GPU consists of many identical compute clusters and a memory system. The H100
has, for example:

- **132 streaming multiprocessors (SMs)** - the compute clusters;
- **128 FP32 cores per SM** - the arithmetic units;
- **64 KB to 228 KB of shared memory per SM**, configurable;
- **64 K registers per SM**, partitioned among the threads;
- **50 MB of L2 cache**, shared by all SMs;
- **HBM3 DRAM** with a bandwidth of roughly **3.35 TB/s**.

![The GPU at a glance: host CPU, transfer bus, GPCs of SMs, chip-wide L2, memory controllers and HBM3 DRAM](../../assets/ch02_gpu_die.svg)

The diagram shows the whole machine. The host CPU sits across a bus. The die is
organised into *graphics processing clusters* (GPCs), each holding several SMs.
Every SM funnels into one chip-wide L2; L2 feeds the memory controllers; the
controllers drive the HBM3 stacks. This physical map underlies the reasoning in
every later chapter.

The headline figures are design consequences rather than arbitrary
specifications:

- **Many SMs.** A GPU is a throughput machine (Chapter 1, §1.1). Chip area is
  spent on many small compute clusters instead of a few large cores because
  parallelism, not single-thread speed, is the product. The 132 SMs of the H100
  are what fit when each SM is deliberately small.
- **128 FP32 cores per SM.** An SM has four warp schedulers (§2.2), and each
  scheduler can issue one warp instruction per clock, covering 32 lanes. 128 =
  4 x 32: one full warp per scheduler per clock without lane sharing. The
  number follows from the warp as the unit of execution.
- **64 K registers per SM.** Registers provide the working storage for resident
  warps. A deeper register file holds more warps and hides more latency (§2.9),
  at the cost of chip area and clock speed. The 64 K size is an engineering
  balance.
- **High DRAM bandwidth, high DRAM latency.** HBM3 stacks memory vertically
  beside the die on a silicon interposer and uses thousands of narrow channels;
  this is where 3.35 TB/s comes from. Every access still leaves the chip, which
  is why latency remains hundreds of cycles (§2.6) and why the on-chip memory
  hierarchy exists.

The FP32 throughput of such a chip is on the order of 60-70 TFLOP/s. With
memory bandwidth of 3.35 TB/s, the roofline formula from Chapter 1 gives:

\\[ I_{\text{ridge}} = \frac{60 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{B/s}} \approx 18\ \text{FLOP/byte} \\]

A kernel below roughly 18 FLOP/byte is memory-bound on this machine. Most of the
optimisation chapters explain kernels by reference to this ridge point.

## 2.2 The Streaming Multiprocessor (SM)

The SM is the GPU's unit of compute. It is a small, heavily multithreaded
processor, closer to a 128-lane vector machine than to a CPU core.

Each SM contains:

- **FP32 cores** (also called CUDA cores): single-precision floating-point
  units that can perform one FMA (fused multiply-add) per core per clock. An
  FMA computes \\(a \cdot b + c\\) in one instruction; counting it as two FLOPs
  is why peak FLOP rates are reported as large numbers.
- **INT32 cores**: integer units. In modern architectures they share dispatch
  with the FP32 units but have their own register ports.
- **Tensor cores**: specialised matrix-multiply units for AI workloads. They
  form a separate pipeline and return in Chapter 11.
- **Special function units (SFUs)**: fast approximate implementations of
  transcendental functions such as `sin`, `cos`, `exp`, `log`, `1/x`, and
  `rsqrt`. Each SFU serves a warp at one result per clock per unit.
- **A register file** of 64 K 32-bit registers.
- **A shared memory / L1 cache** unit.
- **Four warp schedulers** (on modern SMs), each able to issue one instruction
  per clock to a warp.

An SM is not a multicore CPU. It does not run one instruction stream per core.
It has a small number of warp schedulers, each feeding instructions to a warp
of threads. Threads provide the data parallelism; the scheduler provides the
control.

![Anatomy of a streaming multiprocessor: register file, shared memory, warps, schedulers and execution units](../../assets/ch02_sm_anatomy.svg)

The diagram is a data-flow picture. Warps live in the register file, reach
shared memory and arithmetic units through the schedulers, and access the rest
of the chip through the L1/L2 path. The four schedulers are the control plane;
the FP32/INT32/SFU/Tensor units are the data plane; shared memory and registers
are the on-chip storage.

## 2.3 The Warp

> **Primitive - warp.** A warp is a group of **32 consecutive threads** that
> are scheduled and executed together. The warp is the hardware's unit of
> execution, just as the thread is the programmer's unit of logic.

The hardware fetches one instruction and broadcasts it to 32 lanes. All 32
lanes execute it in the same clock, and each lane applies it to its own
registers and its own data:

![Anatomy of a warp: one instruction fetched once, broadcast to 32 lanes, each with its own registers and data](../../assets/ch02_warp_anatomy.svg)

The scheduler does not manage 32 threads as 32 independent items. It manages
them as one warp. When it issues an instruction, every active lane executes it
on whatever data that lane's thread holds.

The warp size of 32 is an architectural constant across every NVIDIA GPU to
date. It results from three engineering constraints:

- **Instruction-cost amortisation.** Fetching and decoding an instruction costs
  the same whether it serves one thread or 32. A wider warp spreads that fixed
  cost over more useful work.
- **Power-of-two addressing.** Warp boundaries fall at 32, 64, 96, and so on,
  making thread-to-warp arithmetic (division and modulo by 32) cheap in
  hardware.
- **Memory-system granularity.** 32 threads x 4 bytes = 128 bytes, the size of
  one cache line (§2.7). A warp of consecutive threads can be satisfied by one
  memory transaction.

The cost of a wider warp is coarser divergence granularity: one thread taking a
different branch makes the whole warp pay. The size 32 balances instruction
amortisation against divergence waste, and NVIDIA has retained it across
generations.

**Execution semantics.** The warp scheduler picks an instruction for a warp;
the instruction is fetched once and issued to all 32 lanes at the same time.
Each lane has its own registers, so lanes can hold different data, but they
execute the same instruction at the same time. This is SIMT (Chapter 1, §1.7).
A CPU runs one instruction stream per core; a GPU runs one instruction stream
per 32 threads.

**Divergence.** If two threads in a warp take different branches of an `if`,
the hardware cannot execute both paths simultaneously. It executes the `then`
path with the other lanes masked off, then the `else` path, then reconverges.
The two paths run serially, each consuming full-warp instruction slots. A 50/50
branch costs approximately twice the work of a uniform branch. Chapter 5
discusses this in detail.

**One instruction, many loads.** Because all 32 lanes share one instruction, a
single memory load instruction issued to a warp performs 32 loads. How those 32
loads are serviced is the subject of §2.7 (coalescing).

**Partial warps.** A kernel launched with 1,000 threads creates 31 full warps
(32 x 31 = 992 threads) plus one partial warp of 8 threads. The remaining 24
lanes of that last warp are disabled but still occupy scheduling slots. They
consume occupancy (§2.9) without doing work. Real kernels normally use block
sizes that are multiples of 32 (Chapter 3).

## 2.4 Blocks, Grids, and the Hardware's View

CUDA's programming model (Chapter 3) organises threads as a **grid** of
**thread blocks**, each block containing a group of **threads**. The hardware
mapping is:

- A **thread block** is scheduled onto one SM as a unit. All threads of a block
  run on the same SM, which makes block-level shared memory and
  `__syncthreads()` possible.
- A block is partitioned into **warps** by consecutive thread IDs. Threads 0-31
  form warp 0, threads 32-63 form warp 1, and so on. For a 2-D block, threads
  are linearised in x-major order (x varies fastest).
- An SM runs **many blocks concurrently**, time-slicing its warps. The number
  depends on occupancy (§2.9).

The block is the unit of cooperation: all threads in it can share memory and
synchronise. The warp is the unit of execution: the hardware moves whole warps.
The two levels are distinct:

- The programmer chooses the **block** size (`blockDim` in Chapter 3). Sizes
  are normally multiples of 32 so that no warp is partially empty.
- The hardware partitions blocks into **warps** invisibly. The programmer does
  not create warps and rarely addresses one directly. The warp exists so that
  the SM can schedule 32 threads at the cost of one.

![A block of 256 threads is chopped into 8 warps of 32 consecutive threads; the whole block is placed on one SM](../../assets/ch02_block_to_warps.svg)

## 2.5 The Memory Hierarchy

The GPU memory hierarchy is a hierarchy of distance and size:

![The GPU memory hierarchy: grid to SM to L2 to global and host memory](../../assets/ch02_memory_hierarchy.svg)

From the SM outward, each level is larger and slower:

**1. Registers.** Private to a single thread; 32 bits wide; up to 255 per
thread. A register has no address; instructions name it directly (`R0`, `R1`,
...). Register access is fast, but an SM has only 64 K registers shared by all
resident threads. Register pressure directly limits occupancy (§2.9).

**2. Shared memory.** Private to a block; on-chip; configurable as part of the
SM's 228 KB (H100) unified L1/shared resource. Access latency is roughly 20-30
cycles, compared with hundreds of cycles for global memory. Shared memory is
the programmer-managed cache and the central subject of Chapter 7.

**3. L1 cache.** On-chip, per-SM, unified with shared memory. Global loads that
hit L1 avoid the trip to DRAM. L1 cache lines are 128 bytes.

**4. L2 cache.** On-chip, shared by all SMs; 50 MB on the H100. It caches
global, constant, and texture accesses. L2 is the coherence point between SMs:
blocks on different SMs exchange data through L2 or through atomics (Chapter
5).

**5. Global memory.** The GPU's DRAM (HBM3) and the largest, slowest level.
`cudaMalloc` allocations live here (Chapter 4). Bandwidth is enormous (3.35
TB/s) and latency is hundreds of cycles. Most optimisation work reduces global
traffic.

**6. Constant and texture memory.** Two specialised read-only paths. Constant
memory is a 64 KB cache that broadcasts a single value to all threads in a warp
when they read the same address, which makes it suitable for kernel parameters.
Texture memory is a cached read-only path with hardware support for 2-D spatial
locality and interpolation, used for images. Chapter 7 discusses both.

**7. Local memory.** Despite the name, "local" memory is global memory
allocated per thread. It is used when a thread's register demand exceeds the
register file (a *register spill*). Local memory is slow and spills are
avoided when possible. The compiler reports spills with
`--ptxas-options=-v`.

## 2.6 The Latency Table

The following numbers are typical orders of magnitude for a modern GPU. They
are teaching figures, not datasheet values:

| Resource | Approximate latency | Notes |
|---|---|---|
| Register | ~0 cycles | Operand to instruction |
| Shared memory | ~20-30 cycles | On-chip, banked |
| L1 hit | ~30 cycles | Per-SM |
| L2 hit | ~200 cycles | Chip-wide |
| Global DRAM | ~400-800 cycles | HBM3 |
| Host memory (PCIe) | ~1,000+ cycles + transfer time | Off-chip, CPU side |

The table is a map of physical distance. Registers sit on the SM a few
millimetres from the arithmetic units. Shared memory and L1 are on the same
die. L2 spans the chip. DRAM is a separate package beside the die on an
interposer. Host memory is across a bus and an operating-system boundary. Each
step off the SM adds distance and arbitration, because more circuits compete
for the same wires. The 20x gap between shared memory and DRAM is not a tuning
detail; it is the difference between an on-chip wire and an off-chip trip, and
it motivates the optimisation chapters.

A practical consequence is that one global memory access costs roughly 30
shared-memory accesses. An algorithm that reuses data in shared memory buys
speed with engineering effort. Chapter 7 quantifies the trade.

## 2.7 Coalescing: How a Warp Reads Memory

Coalescing determines whether a GPU kernel uses its memory system efficiently
or wastes most of its available bandwidth.

A warp executing a load consists of 32 threads issuing the same load
instruction together. Each thread wants its own piece of data, for example a
`float` occupying four bytes. The memory system receives 32 separate requests.
Its cost depends on where those addresses lie.

If the 32 addresses are contiguous, the memory system can treat them as one
block: it fetches one contiguous region and returns each thread its slice. The
warp is served by one transaction, or two at most. If the addresses are
scattered, for example every 32nd `float` with no reuse, the memory system
cannot group them. Each request may become its own transaction, and the warp
pays for many trips to memory for the same amount of useful data.

This property is **coalescing**. A warp whose accesses can be grouped into few
transactions is *coalesced*; one whose accesses spread across many sectors is
*uncoalesced*.

> **Primitive - coalescing.** A warp load is serviced at sector granularity
> (32 bytes; four sectors per 128-byte cache line). The load is *coalesced*
> when the warp's addresses fall in as few sectors as possible and
> *uncoalesced* when they sprawl. To a first approximation, the cost of a warp
> load is the number of sectors it touches.

**Sectors and transactions.** The memory system delivers data in fixed-size
chunks of 32-byte **sectors**, a quarter of a 128-byte cache line. The setup
cost of a memory transaction (DRAM row activation, address decode, bus
transfer) is paid per transaction, not per byte, so a half-empty sector costs
almost as much as a full one. The relevant quantity for a warp load is the
number of sectors touched:

- Thirty-two lanes reading 32 consecutive `float`s touch exactly 128 bytes:
  one cache line, one or two transactions.
- Thirty-two lanes reading every 32nd `float` touch 32 different lines: 32
  transactions for the same number of useful bytes.

The hardware does not detect access patterns or rearrange requests. It counts
the sectors a warp touches and bills accordingly. Global memory bandwidth is
the scarcest resource on a memory-bound kernel (Chapter 1's roofline). With a
stride of 32 floats between consecutive threads, each fetched 128-byte line
delivers only one useful 4-byte word, wasting roughly 97% of the transferred
bytes.

The layout habit that follows from this model is used throughout the book:
arrange data so consecutive threads touch consecutive addresses. Row-major
matrices in Chapter 9, thread-to-pixel mappings in Chapter 15, and padded
shared-memory arrays in Chapter 7 all follow this rule. It derives from the
hardware's sector-based charging model rather than from a memorised guideline.

## 2.8 Shared Memory Banks

Shared memory is fast because it is **banked**. It is physically organised into
32 banks, each four bytes wide, that can be accessed simultaneously. A shared
memory word maps to a bank by:

\\[ \text{bank} = \left\lfloor \frac{\text{address in bytes}}{4} \right\rfloor \bmod 32 \\]

When a warp accesses shared memory, the hardware services one access per bank
per cycle. If two threads in the warp access the same bank, the hardware
serialises those accesses. This is a **bank conflict**, and it costs extra
cycles.

- Threads 0-31 reading consecutive words: all 32 banks are busy, one access,
  no conflict.
- Threads 0-31 reading words with stride 32: all 32 threads hit bank 0, a
  32-way conflict that takes 32 cycles.
- Threads 0-31 reading the same word: the hardware broadcasts the value, one
  access, no conflict.

Formally, if a warp's shared-memory accesses hit bank \\(b\\) exactly \\(n_b\\)
times, the hardware must issue \\(n_b\\) accesses to that bank. The warp's
access completes in \\(\max_b n_b\\) cycles. A conflict-free access has
\\(\max_b n_b = 1\\); the worst case is \\(\max_b n_b = 32\\), when all threads
hit one bank. Bank conflicts therefore multiply shared-memory access cost by a
factor between 1 and 32 without moving any additional data.

Bank conflicts are a shared-memory phenomenon. Global memory has sectors and
lines, not banks. Chapter 7 shows the standard fix for bank conflicts: padding.

## 2.9 Occupancy

> **Primitive - occupancy.** Occupancy is the ratio of active warps on an SM to
> the maximum number of warps the SM can hold. An SM with 64 warp slots at 100%
> occupancy has 64 warps resident.

When a warp stalls on a global load, which takes hundreds of cycles, the
scheduler switches to another resident warp (Chapter 1, §1.1). If occupancy is
high, another warp is usually ready. If it is low, the SM may idle.

Occupancy is limited by the SM's finite resources:

- **Registers:** 64 K per SM. At 32 registers per thread, the SM can host 2,048
  threads (64 K / 32). At 128 registers per thread, only 512 threads.
- **Threads per SM:** a hardware maximum, 2,048 on most modern SMs.
- **Threads per block and blocks per SM:** up to 1,024 threads per block and 32
  blocks per SM (architecture-specific).
- **Shared memory:** 228 KB per SM on the H100. A block declaring 100 KB of
  shared memory leaves room for only two such blocks.

The occupancy of a launch configuration is the minimum over these limits. The
occupancy calculator spreadsheet and the runtime function
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (Chapter 16) compute it.

Writing the arithmetic explicitly makes the trade visible. Let \\(R_{SM}\\)
be registers per SM, \\(T_{SM}\\) the hardware thread limit per SM, \\(B_{SM}\\)
the block limit per SM, and \\(S_{SM}\\) shared memory per SM. Let a block use
\\(T_B\\) threads, \\(R_T\\) registers per thread, and \\(S_B\\) bytes of shared
memory. The number of blocks that fit is:

\\[ B_{\max} = \min\left(
\left\lfloor \frac{R_{SM}}{T_B \cdot R_T} \right\rfloor,\;
\left\lfloor \frac{T_{SM}}{T_B} \right\rfloor,\;
B_{SM},\;
\left\lfloor \frac{S_{SM}}{S_B} \right\rfloor\ \text{if } S_B > 0
\right) \\]

The number of resident threads is \\(B_{\max} \cdot T_B\\), and occupancy is:

\\[ \text{occupancy} = \frac{B_{\max} \cdot T_B}{T_{SM}} \\]

The four constraints are not alternatives. All four budgets are consumed
simultaneously; the minimum is the binding one.

**Worked calculation.** Assume an SM with 64 K registers, 2,048 threads per SM,
32 blocks per SM, and 228 KB shared memory, and launch blocks of 256 threads (8
warps):

| Constraint | Equation | Blocks allowed |
|---|---|---|
| Registers (32/thread) | 64 K / (256 x 32) | 8 blocks |
| Threads per SM | 2,048 / 256 | 8 blocks |
| Blocks per SM | hardware limit | 32 blocks |
| Shared memory (0 bytes used) | no demand on the 228 KB budget | not a constraint |

The minimum is 8 blocks, or 64 warps resident. If the SM holds at most 64
warps, this is 100% occupancy. With 64 registers per thread, the register term
becomes 64 K / (256 x 64) = 4 blocks and occupancy drops to 50%. With 128
registers per thread it becomes 2 blocks, or 25% occupancy. Chapter 9's
`__launch_bounds__` controls this trade by telling the compiler how many
registers it may use per thread.

![Occupancy as warp slots: 8, 4 and 2 resident blocks of 8 warps each](../../assets/ch02_occupancy.svg)

The diagram shows the same arithmetic. Each cell is one warp slot; each row is
one block's eight warps; the dim cells are slots the scheduler cannot use
because the register file is exhausted.

High occupancy is not always the right goal. A memory-bound kernel with
long-latency global loads benefits from a large pool of warps because the pool
gives the scheduler alternatives while any one warp waits. A compute-bound
kernel whose operands already reside in registers rarely stalls, so a smaller
pool may suffice. Raising occupancy by reducing register usage can force the
compiler to spill registers to local memory, adding memory traffic and slowing
the kernel. A kernel that needs a large shared-memory tile per block may
deliberately use fewer blocks per SM, accepting lower occupancy in exchange for
less global traffic and more data reuse.

The correct objective is not to maximise occupancy but to give the scheduler
enough ready work without starving the kernel of registers or shared memory.
`__launch_bounds__` makes this trade explicit at compile time.

## 2.10 The SM in Action: Time Slicing

Suppose an SM has 64 warp slots and a kernel is launched with blocks of 256
threads (8 warps per block). If the occupancy calculation permits 8 blocks per
SM, the SM hosts 8 blocks = 64 warps = 100% occupancy.

![Time slicing: while warp 3 waits on a global load, the scheduler keeps issuing other warps](../../assets/ch02_warp_time_slicing.svg)

Each of the four warp schedulers owns 16 warps. A scheduler can issue an
instruction from one of its warps each clock. When warp 3 issues a global load,
it will not be ready for roughly 500 cycles; the scheduler issues instructions
from warps 4, 5, and others in the meantime. When warp 3's load returns, the
scheduler resumes issuing for it.

No thread, driver, or explicit scheduling call rotates the warps. The hardware
rotates among resident warps automatically. The programmer supplies enough
warps (occupancy) and enough independent work per warp (instruction-level
parallelism and coalesced accesses) to keep the rotation from stalling.

A first-order estimate makes this concrete. If a warp issues a global load and
has nothing else ready for \\(L\\) cycles, a scheduler that issues at most one
instruction per cycle needs at least \\(L\\) other ready warps to keep its
execution units busy. With \\(L \approx 500\\), that naive estimate would
require 500 warps per scheduler, far more than the hardware can host. Real
warps do not stall on every instruction, and memory pipelining allows multiple
loads to be outstanding. The estimate is therefore not a literal requirement;
it explains why occupancy matters at all.

## 2.11 Architecture Generations

The structure described in this chapter is stable across NVIDIA GPUs, but the
numbers are not. Compute capability (CC) encodes the generation: CC 7.x is
Volta, CC 8.x is Ampere, CC 9.0 is Hopper, and CC 10.x is Blackwell. Each
generation changes SM size, register file size, warp scheduling, tensor core
capabilities, and shared memory capacity. Any performance claim should be read
with the target compute capability in mind.

`deviceQuery`, a CUDA sample, reports the SM count, compute capability, register
file size, shared memory per SM, and the launch limits from §2.9 for the
installed GPU. Chapter 16 explains how to read that output.

## Common Pitfalls

- Assuming higher occupancy is always faster. Register spills and reduced
  shared memory per block can make 50% occupancy beat 100%.
- Treating the warp as the programming unit. Programs address threads; the
  hardware executes warps. Divergence, coalescing, and shuffle operations all
  depend on warp boundaries.
- Ignoring bank conflicts. A `tile[32][32]` column access can be 32x slower
  than a row access. Padding each row by one float removes the conflict at
  negligible cost.
- Assuming the numbers in this chapter apply to every GPU. Check the compute
  capability and SM limits with `deviceQuery`.

## Check Your Understanding

<details>
<summary>Must a block be resident on a single SM?</summary>

Yes. Block-scoped synchronisation (`__syncthreads`) and shared memory require
all threads of a block to be co-located so they can communicate through a
common on-chip resource. Splitting a block across SMs would make an efficient
block-wide barrier impossible.
</details>

<details>
<summary>A kernel uses 64 registers per thread. How many 256-thread blocks fit in an SM with a 64 K register file?</summary>

Each block uses 256 x 64 = 16,384 registers. The register file holds 65,536 /
16,384 = 4 blocks. If the thread and block limits allow more, registers cap
occupancy at 4 blocks.
</details>

<details>
<summary>Why do 32 consecutive floats cost one 128-byte line, but 32 floats with stride 32 cost 32 lines?</summary>

A warp's 32 consecutive floats span 128 bytes, exactly one cache line. With
stride 32, each thread's float lives in a different 128-byte region (for a
large array width), so the hardware fetches 32 separate lines. The amount of
useful data is the same; the memory traffic is roughly 32x larger.
</details>

## Key Takeaways

- The warp of 32 threads is the hardware unit of execution, not the thread.
- The physical layout is GPCs of SMs above a chip-wide L2 above HBM3 DRAM, with the host across PCIe/NVLink.
- Blocks map to SMs; warps are consecutive thread IDs inside a block.
- Memory hierarchy: registers, shared memory, L1, L2, global DRAM - each level larger and slower (roughly 20-30 cycles for shared memory, 400-800 for DRAM).
- Coalescing: consecutive threads should read consecutive addresses; the hardware fetches 128-byte lines in 32-byte sectors.
- Shared memory has 32 banks of 4 bytes; a 32-way bank conflict costs 32 cycles; padding fixes it.
- Occupancy is the ratio of resident warps to the SM maximum; registers, threads, blocks, and shared memory each cap it.

## 2.12 Exercises

1. A kernel uses 64 registers per thread. How many threads can one SM host
   before the register file is exhausted, with 64 K registers per SM?
2. The same kernel now uses 128 registers per thread. What is the maximum
   occupancy given a hardware limit of 2,048 threads per SM?
3. A warp reads 32 consecutive `int`s (4 bytes each). How many 128-byte cache
   lines does the hardware fetch? How many would it fetch if the threads read
   every 32nd `int`?
4. Why must all threads of a block be resident on the same SM? Which
   synchronisation and memory primitive does this enable?
5. Estimate the number of cycles a warp-level shared-memory access takes when
   all 32 threads read the same 4-byte word, and when all 32 threads read words
   separated by 128 bytes.

## Sources and Further Reading

- NVIDIA, *CUDA C++ Programming Guide*, "Hardware Implementation" and "Compute Capabilities": <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *H100 Tensor Core GPU Architecture* whitepaper and product page, for generation-specific numbers quoted in this chapter.
- David B. Kirk and Wen-mei W. Hwu, *Programming Massively Parallel Processors: A Hands-on Approach*, 3rd/4th ed., Morgan Kaufmann. Chapter-level treatment of GPU compute architecture, warp scheduling, memory coalescing, and occupancy.
