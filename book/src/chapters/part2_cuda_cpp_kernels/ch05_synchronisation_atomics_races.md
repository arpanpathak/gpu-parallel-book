# Chapter 5: Synchronisation, Atomics & Race Conditions

> *"Parallelism is the ability to disagree about the order of events. Most bugs
> are disagreements you did not intend."*

Chapters 3 and 4 built kernels whose threads never communicated. Real kernels
share data, and sharing data in parallel is where correctness lives. This
chapter covers the three mechanisms the CUDA programming model provides  - 
**warp divergence** (the cost of independent control flow), **barriers**
(`__syncthreads`) and **atomics** (hardware-arbitrated read-modify-write
operations) - and the failure mode that motivates all of them: the **race
condition**.

## 5.1 The Race Condition, Defined

> **Primitive - race condition.** Two or more threads access the same memory
> location, at least one access is a *write*, and the accesses are not ordered
> by any synchronisation mechanism. The result depends on the order in which
> the hardware happens to execute the threads - an order you cannot predict.

Races in CUDA are worse than races on a CPU because of two multipliers:

1. **Scale.** A kernel has thousands to millions of threads; a race between
   *any two* of them corrupts the result, and the corrupted result may only
   appear on one input in a thousand.
2. **Asynchrony.** The kernel reports success while the corruption is silently
   stored. There is no exception; there is a wrong answer.

The tools of this chapter exist to *order* accesses. Every race fix is, at
heart, the installation of an order.

## 5.2 Warp Divergence: Control Flow in SIMT

From Chapter 2: a warp executes one instruction at a time for all 32 lanes.
Consider:

```cpp
__global__ void conditionalAdd(const float* a, float* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Some threads take this path, others do not.
        out[i] = a[i] + 1.0f;
    }
}
```

If every thread in a warp has `i < n`, the warp takes the branch unanimously
and there is no cost. If *some* threads have `i >= n` while others do not,
the hardware must:

1. Execute the `then` path with the `i < n` lanes active, the others masked;
2. Execute the `else` path (empty here) with the remaining lanes active;
3. Rejoin the warp.

The two paths run **serially**. This is **warp divergence**, and it is the
price SIMT pays for the convenience of per-thread control flow.

**The practical rule.** Divergence is a *per-warp* phenomenon. If the branch
depends on `threadIdx.x % 2` (alternating threads), every warp in the grid
diverges and pays double. If the branch depends on `blockIdx.x % 2` (whole
blocks), whole warps agree and there is no cost. Structure your data-dependent
branches so that *contiguous ranges of threads* take the same path wherever
possible.

**The boundary guard is fine.** The `if (i < n)` guard in Chapter 3 diverges
only in the *last* (partial) block of the grid - at most one warp per grid.
The cost is one extra instruction path in one block. This is why the guard is
free in practice: divergence at block boundaries is negligible.

## 5.3 `__syncthreads()`: The Block Barrier

> **Primitive - barrier.** A point at which every thread in a *group* must
> arrive before any thread may proceed. `__syncthreads()` is a barrier over
> **one thread block**, provided by the hardware.

Semantics: when a thread executes `__syncthreads()`, it waits until *all*
threads of its block have reached that same `__syncthreads()`. Then all
proceed. Its purpose is memory ordering *within the block*: a write by thread
A before the barrier is visible to thread B after the barrier. Shared memory
writes (Chapter 7) rely on exactly this.

```cpp
__global__ void blockReduceDemo(const float* in, float* out, int n)
{
    __shared__ float s_partial[256];   // shared array, one slot per thread

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: every thread reads its element and stashes it in shared memory.
    // No ordering needed yet: each thread writes its own slot.
    s_partial[threadIdx.x] = (i < n) ? in[i] : 0.0f;

    // Phase 2: BEFORE any thread reads another thread's slot, all writes
    // must be complete and visible. The barrier provides both the "all have
    // arrived" condition and the visibility guarantee.
    __syncthreads();                    // <-- required, see below

    // Phase 3: now thread 0 can safely read everyone's slot.
    if (threadIdx.x == 0)
    {
        float sum = 0.0f;
        for (int k = 0; k < blockDim.x; ++k) sum += s_partial[k];
        out[blockIdx.x] = sum;
    }
}
```

**Why is the barrier *required*?** Without it, thread 0 might read
`s_partial[5]` before thread 5 has written it. On the actual hardware, thread
5's write may still be in its private pipeline; the read could return garbage.
The barrier is the contract that makes the phase structure valid.

**The two cardinal sins of `__syncthreads()`:**

1. **Divergent barriers.** If some threads of a block reach a
   `__syncthreads()` while others do not (because they took a different
   branch), the barrier waits for threads that will never arrive: **deadlock**.
   The hardware does not detect this; the kernel hangs, and only
   `cudaDeviceReset` (or a watchdog timeout) recovers.
2. **Barriers in divergent loops.** The same deadlock occurs if the loop trip
   count differs between threads. The barrier must be *uniformly reachable*:
   every thread must execute it the same number of times.

```cpp
// DEADLOCK: threads with even index skip the barrier, odd ones wait forever.
if (threadIdx.x % 2 == 0) { /* no barrier here */ }
__syncthreads();            // threads with even index never arrive
```

**Why is there no grid-wide barrier?** Blocks on different SMs cannot
synchronise cheaply (they might not even be resident at the same time).
A grid-wide barrier exists (`cooperative groups`), but it requires a
*cooperative launch* where every block is resident simultaneously, which caps
grid size. For cross-block communication, use atomics (§5.5) or split the work
into two kernel launches - the classic and honest solution.

## 5.4 Visibility: Caches, `volatile`, and Fences

A barrier orders *block-internal* accesses. Cross-block and host-device
visibility have their own rules, because modern GPUs have caches:

- L1 caches are per-SM and are *not* coherent between SMs. A thread on SM 0
  may read a stale value of a location written by SM 1 - unless the access is
  made visible via L2, the coherence point.
- The compiler may also *reorder* or *cache* loads and stores in registers
  unless told otherwise.

Two tools handle this:

> **Primitive - `volatile`.** Tells the compiler: "this memory may change
> outside your knowledge; do not cache it in registers; emit the access every
> time." Used for *device-scope* communication through global memory where the
> compiler's register caching would otherwise hide the value.

> **Primitive - memory fence.** An instruction that forces the ordering of
> memory operations at a given scope. `__threadfence()` orders global-memory
> accesses for the *device*; `__threadfence_block()` for the block;
> `__threadfence_system()` for host *and* device. A fence does not wait for
> other threads; it forces *your* prior writes to become visible before *your*
> later writes.

The canonical pattern - a device-scope flag - combines volatile with a fence:

```cpp
// Shared state: a buffer and a "ready" flag, both in global memory.
__device__ float  g_buffer[1024];
__device__ int    g_ready = 0;

// Producer kernel: writes data, then sets the flag.
__global__ void producer()
{
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        g_buffer[i] = static_cast<float>(i);

    // Make ALL preceding writes visible device-wide BEFORE the flag write.
    // Without the fence, another SM could see g_ready == 1 while some
    // g_buffer writes are still in flight in L1.
    __threadfence();

    if (threadIdx.x == 0)
        g_ready = 1;            // must be volatile; compiler cannot cache it
}

// Consumer kernel (different SM, launched after): polls the flag.
__global__ void consumer()
{
    while (volatileLoad(&g_ready) == 0) { /* spin */ }
    // Now g_buffer writes are guaranteed visible.
    float x = g_buffer[threadIdx.x];
}
```

where `volatileLoad` is a `volatile int*` read. In practice, prefer the
higher-level abstractions - atomics (§5.5) and cooperative groups - over
hand-rolled fences; the fences exist so you can *understand* what the
abstractions do, and for the rare case you must do it yourself.

## 5.5 Atomics: Hardware-Arbitrated Read-Modify-Write

> **Primitive - atomic operation.** A read-modify-write (e.g., *read, add,
> write*) that the hardware guarantees to execute *indivisibly* with respect
> to other threads. Two threads performing `atomicAdd` on the same location
> cannot interleave: the result is exactly as if the two adds happened in some
> serial order. The hardware arbitrates the order; you never see a torn value.

The runtime API provides these atomic functions for `int`, `unsigned int`,
`unsigned long long`, `float` (for add), and on modern GPUs `double`:

| Function | Operation | Returns |
|---|---|---|
| `atomicAdd(addr, v)` | `*addr += v` | the *old* value |
| `atomicSub(addr, v)` | `*addr -= v` | the old value |
| `atomicExch(addr, v)` | `*addr = v` | the old value |
| `atomicCAS(addr, cmp, v)` | if `*addr == cmp` then `*addr = v` | the old value |
| `atomicMin(addr, v)` | `*addr = min(*addr, v)` | the old value |
| `atomicMax(addr, v)` | `*addr = max(*addr, v)` | the old value |
| `atomicAnd(addr, v)` | `*addr &= v` | the old value |
| `atomicOr(addr, v)` | `*addr \|= v` | the old value |
| `atomicXor(addr, v)` | `*addr ^= v` | the old value |

They operate on global *and* shared memory. The returned *old* value is the
key to lock-free algorithms: `atomicCAS` (compare-and-swap) is the universal
primitive from which every other synchronisation structure can be built.

### 5.5.1 Worked example: a histogram

A histogram counts occurrences of values. If every thread increments the same
global counter, the increments must be atomic:

```cpp
// Count how many elements fall into each of 256 bins.
// data   : device array of unsigned char (0..255), n elements
// hist   : device array of 256 ints, zero-initialised on the host
// bins   : the bin count, 256
__global__ void histogram(const unsigned char* data, int* hist, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int bin = data[i];               // value is the bin index
        // The atomic serialises concurrent increments to the same bin.
        // Without it, two threads could both read old, both add 1, and
        // both write old+1 - losing one count (the classic lost update).
        atomicAdd(&hist[bin], 1);
    }
}
```

**The cost.** Atomics on the *same* address from many threads serialise  - 
hardware arbitration is a bottleneck. The classic fix is **privatisation**
(Chapter 8): give each *block* its own histogram in shared memory, accumulate
into shared memory (where atomics are cheaper), then one thread per block
folds the private histograms into global memory.

### 5.5.2 Worked example: a spinlock built from `atomicCAS`

The universal pattern: `atomicCAS` implements *test-and-set*, and test-and-set
implements a lock.

```cpp
// A simple lock in shared memory. One mutex per block.
__global__ void lockedCriticalSection(float* g_data, int n)
{
    __shared__ int s_lock;    // 0 = unlocked, 1 = locked

    // Thread 0 initialises the lock. (Alternatively use a static initialiser.)
    if (threadIdx.x == 0) s_lock = 0;
    __syncthreads();          // everyone must see the lock before using it

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        // Acquire: atomically swap 1 into the lock; if the old value was 0,
        // we won the lock. If it was 1, someone else holds it - retry.
        while (atomicCAS(&s_lock, 0, 1) != 0) { /* spin */ }

        // Critical section: safe because we exclusively hold the lock.
        g_data[i] = g_data[i] * 2.0f + 1.0f;

        // Release: plain store is fine here (see the fence discussion below),
        // but a __threadfence_block() before it is the textbook-safe form.
        __threadfence_block();
        s_lock = 0;
    }
}
```

**Why `atomicCAS(&s_lock, 0, 1)`?** It says: "if the lock is currently 0
(unlocked), set it to 1 and tell me what it *was*." If the returned value is
0, this thread won the lock; otherwise it retries. The spin is the price of
contention.

**Why the fence before release?** The release store must not be observed
before the critical-section writes are visible. `__threadfence_block()` orders
the block's shared-memory accesses so that a thread acquiring the lock next
sees a consistent state.

**The honest engineering note.** Locks in GPU kernels are almost always a
design smell: they serialise work on a machine built for parallelism. The
patterns that *avoid* locks (privatisation, partitioning, lock-free atomics)
are uniformly faster. This example exists so you understand what the
abstractions protect you from - not as a recommendation.

## 5.6 Floating-Point Non-Determinism: The Silent Race

There is a race that passes every test and then changes your answer anyway:
floating-point addition is not associative.

```cpp
// (a + b) + c  may differ from  a + (b + c)  in the last bits.
```

If two threads reduce partial sums in different orders across runs (because
atomics or scheduling chose different orders), the results can differ in the
last bit. For most applications this is tolerable. For anything that demands
bit-exact reproducibility (scientific publishing, distributed training
checkpoints), the fix is a **fixed reduction order**: the optimised reduction
of Chapter 8 is deterministic precisely because its tree order is fixed,
whereas an `atomicAdd`-based reduction is not.

## 5.7 A Decision Procedure

When you see a kernel that writes to shared state, run this checklist:

1. **Who writes?** If more than one thread writes the same location → atomics,
   or restructure so each thread owns its locations.
2. **Who reads after whom?** If a thread reads another's write → a barrier
   (`__syncthreads()`) between write and read, uniformly reachable by all.
3. **Across blocks?** No block barrier exists; use atomics, separate kernels,
   or cooperative groups. Never spin on a non-atomic, non-volatile flag.
4. **Is the order deterministic?** If reproducibility matters, prefer fixed
   tree orders over atomic accumulation.

## Key Takeaways

- A race is two threads accessing one location with a write and no ordering - and on a GPU it fails silently.
- Warp divergence serialises divergent paths; branches uniform across a warp are free.
- __syncthreads() is a block barrier; it must be uniformly reachable or the block deadlocks.
- Atomics are indivisible read-modify-write operations; contention is the cost, privatisation the fix.
- Floating-point addition is not associative: fixed tree orders give bit-reproducible results.

## 5.8 Exercises

1. Explain why the `if (i < n)` boundary guard diverges in at most one warp
   per grid, and why that is negligible.
2. The "cardinal sin" example (divergent barrier) deadlocks. Rewrite the
   pattern so every thread reaches the barrier exactly once.
3. A histogram kernel with 256 global bins runs with heavy contention.
   Sketch the privatised version: per-block shared histograms, block-level
   atomic accumulation, and a final fold. (Chapter 8 gives the full recipe.)
4. `atomicAdd` on `float` returns the old value. Describe an algorithm for a
   global running maximum that does *not* need a lock, using `atomicMax`.
