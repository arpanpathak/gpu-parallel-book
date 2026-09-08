# Chapter 6: Streams, Events & Asynchronous Execution

Chapter 3 noted that a kernel launch is asynchronous: the host does not wait.
This chapter makes that asynchrony useful. The tools are **streams** (ordered
queues in which device work executes), **events** (markers for measuring and
ordering work), and the **double-buffered pipeline** that overlaps transfers
with computation. The chapter closes with **CUDA Graphs**, which capture and
replay fixed pipelines with reduced launch overhead.

## 6.1 The Problem: Serial Execution

Consider the naive vector-add pipeline from Chapter 3, repeated for \\(k\\)
chunks:

```cpp
for (int c = 0; c < k; ++c)
{
    cudaMemcpy(d_in, h_in + c * chunk, chunkBytes, cudaMemcpyHostToDevice);
    kernel<<<grid, block>>>(d_in, d_out, chunkElems);
    cudaMemcpy(h_out + c * chunk, d_out, chunkBytes, cudaMemcpyDeviceToHost);
}
```

On the default (legacy) stream, `cudaMemcpy` is synchronous and each kernel
launch waits for the previous work. The timeline is serial: transfer, kernel,
transfer, kernel. The bus is idle during kernels and the SMs are idle during
transfers. The machine is capable of overlapping these operations, but this
code does not use that capability.

## 6.2 Streams: Ordered Queues of Work

> **Primitive - stream.** An ordered sequence of device operations (copies,
> kernel launches, and events) that executes in FIFO order on the device. Work
> in different streams is unordered and may overlap. A stream is created with
> `cudaStreamCreate` and destroyed with `cudaStreamDestroy`.

Stream semantics:

1. **Order within a stream is guaranteed.** Operations issued to the same
   stream execute in the order issued.
2. **Order across streams is not guaranteed.** Operations in different streams
   may execute in any order, or concurrently if resources permit.
3. **Asynchronous by construction.** `cudaMemcpyAsync`, with pinned host
   memory (Chapter 4), returns immediately; the copy is queued in the stream.

```cpp
// Two streams, each an independent queue.
cudaStream_t s1, s2;
CHECK(cudaStreamCreate(&s1));
CHECK(cudaStreamCreate(&s2));

// Pinned host memory is REQUIRED for async copies (see 4.2).
float *h_pinnedA, *h_pinnedB, *d_A, *d_B, *d_out;
CHECK(cudaMallocHost((void**)&h_pinnedA, chunkBytes));
CHECK(cudaMallocHost((void**)&h_pinnedB, chunkBytes));
CHECK(cudaMalloc((void**)&d_A, chunkBytes));
CHECK(cudaMalloc((void**)&d_B, chunkBytes));
CHECK(cudaMalloc((void**)&d_out, chunkBytes));

// Queue a copy of chunk A in stream 1 and chunk B in stream 2. The copies may
// run concurrently because they are in different streams.
CHECK(cudaMemcpyAsync(d_A, h_pinnedA, chunkBytes,
                      cudaMemcpyHostToDevice, s1));
CHECK(cudaMemcpyAsync(d_B, h_pinnedB, chunkBytes,
                      cudaMemcpyHostToDevice, s2));

// Queue each kernel after its own copy in its own stream. Each kernel must
// write its own output buffer; sharing d_out between streams would be a race.
float *d_outA, *d_outB;
CHECK(cudaMalloc((void**)&d_outA, chunkBytes));
CHECK(cudaMalloc((void**)&d_outB, chunkBytes));
kernel<<<grid, block, 0, s1>>>(d_A, d_outA, chunkElems);
kernel<<<grid, block, 0, s2>>>(d_B, d_outB, chunkElems);
```

The launch syntax gains a fourth argument:
`kernel<<<grid, block, sharedBytes, stream>>>`. `sharedBytes` is dynamic shared
memory (Chapter 7); `stream` selects the queue. Both default to zero, which is
why earlier chapters did not need them.

Async copies require pinned host memory because the DMA engine reads directly
from pinned pages (Chapter 4). A pageable pointer forces the runtime into a
synchronous staging copy and silently removes the asynchrony.

## 6.3 The Default Stream and Implicit Synchronisation

A launch without a stream argument uses the **legacy default stream** (stream
0). This stream synchronises with all other streams: any operation in it waits
for all previously issued work in every stream and blocks other streams from
starting. One streamless launch can therefore serialise an otherwise
concurrent pipeline.

Two remedies exist:

- **Per-thread default stream.** Compile with `--default-stream per-thread`,
  giving each host thread its own non-blocking default stream.
- **Explicit streams.** Name every stream in every launch and copy.

Both are valid; explicit naming is safer because it makes dependencies visible
at the call site.

## 6.4 Events: Markers and Stopwatches

> **Primitive - event.** A marker queued into a stream. It has no payload; it
> records when the stream reaches it. Events measure time, order cross-stream
> dependencies, and let the host wait for specific milestones.

```cpp
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));

// Record "start" into stream s1.
CHECK(cudaEventRecord(start, s1));
kernel<<<grid, block, 0, s1>>>(d_A, d_out, chunkElems);
// Record "stop" into stream s1, after the kernel.
CHECK(cudaEventRecord(stop, s1));

// Block the host until the event is reached (the kernel has finished).
CHECK(cudaEventSynchronize(stop));

// Elapsed time in milliseconds between the two events:
float ms = 0.0f;
CHECK(cudaEventElapsedTime(&ms, start, stop));
std::printf("kernel took %.3f ms\n", ms);
```

Events measure device time. The device records an event when the stream reaches
it, so the elapsed interval excludes host-side launch overhead and queueing
delay. A `std::chrono` measurement around a launch measures host wall time,
which includes whatever the host was doing while the device worked. CUDA events
are the reliable instrument for kernel timing (Chapter 16).

Events also order work across streams. `cudaStreamWaitEvent(stream, event)`
makes one stream wait for an event recorded in another stream, creating a
cross-stream dependency without blocking the host. This is the primitive behind
producer/consumer pipelines.

## 6.5 The Double-Buffered Pipeline

The canonical overlap pattern uses two host buffers. While the GPU computes on
chunk \\(c\\), the DMA engine copies chunk \\(c+1\\) into the other buffer.
The transfer cost moves off the critical path:

![Stream timeline: copies in the copy stream overlap kernels in the compute stream](../../assets/ch06_stream_timeline.svg)

Without double buffering, the timeline is serial: copy, kernel, copy, kernel.
With it, the only serial residue is the first copy (the pipeline prime) and the
last kernel (the pipeline drain).

```cpp
// ---------------------------------------------------------------------------
// Streamed processing of k chunks with double buffering.
// Assumes h_pinned[0] and h_pinned[1] are pinned host buffers, each of
// chunkElems floats, and d_buf[0], d_buf[1] are matching device buffers.
// ---------------------------------------------------------------------------
void runPipelined(int k, int chunkElems, cudaStream_t computeStream,
                  cudaStream_t copyStream)
{
    const size_t chunkBytes = chunkElems * sizeof(float);
    float* h_pinned[2];  float* d_buf[2];  float* d_out;
    cudaEvent_t copyDone[2];   // one event per buffer (created in the omitted setup)
    // ... (allocations omitted for brevity; see 6.2) ...

    // Prime the pipeline: copy chunk 0 into device buffer 0 and record the
    // event the first iteration will wait on.
    CHECK(cudaMemcpyAsync(d_buf[0], h_pinned[0], chunkBytes,
                          cudaMemcpyHostToDevice, copyStream));
    CHECK(cudaEventRecord(copyDone[0], copyStream));

    for (int c = 0; c < k; ++c)
    {
        const int cur = c % 2;        // buffer used for THIS chunk
        const int nxt = (c + 1) % 2;  // buffer used for the NEXT chunk

        // If there is a next chunk, its copy goes into the other buffer in the
        // COPY stream while the kernel runs in the COMPUTE stream. Record an
        // event after it for the next iteration's kernel to wait on.
        if (c + 1 < k)
        {
            CHECK(cudaMemcpyAsync(d_buf[nxt], h_pinned[nxt], chunkBytes,
                                  cudaMemcpyHostToDevice, copyStream));
            CHECK(cudaEventRecord(copyDone[nxt], copyStream));
        }

        // The kernel must wait for its copy. This chunk's copy was queued in
        // the previous iteration (or during the prime), and its completion is
        // marked by copyDone[cur]. cudaStreamWaitEvent installs the dependency
        // without blocking the host.
        CHECK(cudaStreamWaitEvent(computeStream, copyDone[cur], 0));

        kernel<<<grid, block, 0, computeStream>>>(d_buf[cur], d_out,
                                                  chunkElems);
    }
    CHECK(cudaStreamSynchronize(computeStream));
}
```

The essence is the alternation: copy \\(c+1\\) into the idle buffer while
kernel \\(c\\) runs. The two streams provide the queues; events provide the
dependencies; pinned memory provides direct DMA. Chapter 15's capstone uses this
shape for image frames.

One buffer would force copy \\(c+1\\) to wait for kernel \\(c\\), because both
would touch the same data. Two buffers let the DMA engine and the SMs work on
different memory at the same time.

## 6.6 Stream Priorities and Concurrency Limits

Not every pair of operations can overlap. The hardware limits include:

- **One copy engine per direction** (host-to-device and device-to-host) on most
  GPUs. Two simultaneous host-device copies are possible, one in each
  direction.
- **Limited concurrent kernels.** Older GPUs could run only a few kernels
  concurrently; modern GPUs can run more, but SMs time-slice among resident
  work.

Priorities hint the scheduler:

```cpp
int lo = 0, hi = 0;
CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));   // hi = highest priority
cudaStream_t sHigh, sLow;
CHECK(cudaStreamCreateWithPriority(&sHigh, cudaStreamNonBlocking, hi));
CHECK(cudaStreamCreateWithPriority(&sLow,  cudaStreamNonBlocking, lo));
```

Priorities matter when compute and copies compete for the same SMs. Give
latency-critical work the high priority and bulk work the low priority.
`cudaStreamNonBlocking` makes a stream ignore the default-stream
synchronisation rule (§6.3).

## 6.7 CUDA Graphs: Pipelines Without Per-Launch Overhead

Every `kernel<<<>>>` and `cudaMemcpyAsync` call has host-side overhead for
argument marshalling and queueing, roughly 3-10 microseconds per operation. A
pipeline with many operations pays that cost per operation. **CUDA Graphs**
capture the dependency structure once and replay it with one launch:

> **Primitive - CUDA graph.** A captured, reusable description of device work
> (kernel launches, copies, and events) and their dependencies. It is captured
> once, replayed many times, and amortises launch overhead.

```cpp
// Capture phase: record the operations into a graph.
cudaGraph_t graph;
cudaStream_t captureStream;
CHECK(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));
CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeThreadLocal));

// Issue work exactly as in a normal stream, into the capture stream.
kernel<<<grid, block, 0, captureStream>>>(d_A, d_out, chunkElems);
cudaMemcpyAsync(h_out, d_out, chunkBytes, cudaMemcpyDeviceToHost,
                captureStream);

// End capture and instantiate an executable graph.
cudaGraphExec_t exec = nullptr;
CHECK(cudaStreamEndCapture(captureStream, &graph));
CHECK(cudaGraphInstantiate(&exec, graph, 0));

// Replay phase: one call replaces the whole sequence.
for (int frame = 0; frame < 10000; ++frame)
    CHECK(cudaGraphLaunch(exec, /* any stream */ 0));

CHECK(cudaGraphExecDestroy(exec));
CHECK(cudaGraphDestroy(graph));
```

Graphs pay off when launch overhead is a significant fraction of kernel time:
many small kernels, or a fixed pipeline replayed thousands of times, such as
inference loops and render pipelines. For large kernels the overhead is
negligible, and graphs add complexity without benefit. Chapter 15 measures both
regimes.

## 6.8 Synchronisation Reference

| Call | Effect |
|---|---|
| `cudaDeviceSynchronize()` | Wait for all device work issued by this host thread |
| `cudaStreamSynchronize(s)` | Wait for all work queued in stream `s` |
| `cudaEventSynchronize(e)` | Wait for the device to reach event `e` |
| `cudaStreamWaitEvent(s, e)` | No host wait; install a dependency so stream `s` waits for event `e` |
| `cudaMemcpy` (sync) | Wait for the copy itself (in the legacy default stream) |
| `cudaMemcpyAsync(..., stream)` | No wait; queue the copy in `stream` and return |

## Streams as Dependency Graphs

It is natural to think of a stream as a thread or queue and to imagine that two
streams behave like two threads. The hardware model is more precise. The GPU
contains several independent execution engines: copy engines for host-device
transfers, SMs for kernels, and other specialised units. These engines can run
concurrently as long as they do not contend for the same data or resources. A
stream is an ordered sequence of work for one logical device timeline; work in
different streams is unordered and may overlap.

A multi-stream program is therefore a dependency graph. Each operation is a
node; each "must wait for" relationship is an edge. Recording an event in a
copy stream and making a compute stream wait on it adds an edge: the kernel
depends on the copy. Using the legacy default stream implicitly adds edges
between everything because it synchronises with all other streams. The graph
then has no concurrency; it is one long chain.

The dependency-graph view explains the double-buffered pipeline. The copy of
chunk \\(n+1\\) and the kernel of chunk \\(n\\) operate on different buffers,
so no edge connects them. The graph has two independent paths and the hardware
can run them concurrently. Events keep the graph correct without serialising
it: the compute stream waits only for the event that marks its input's copy,
not for all copies.

CUDA Graphs make the same structure explicit and reusable. A graph is the
dependency structure captured once and replayed many times, removing the
host-side cost of issuing each operation and dependency individually. The graph
captures addresses and parameters; if buffers move or launch parameters change,
the graph must be updated or re-captured.

## Common Pitfalls

- Accidentally using the legacy default stream, which synchronises with all
  other streams and serialises the pipeline. Name streams explicitly or compile
  with `--default-stream per-thread`.
- Using pageable memory with `cudaMemcpyAsync`. The call silently becomes
  synchronous and the overlap disappears without an error.
- Recording events on the wrong stream or waiting on the wrong event.
  `cudaStreamWaitEvent` must reference the event that marks the intended
  dependency.
- Replaying a CUDA Graph with changed input pointers. Graphs capture addresses;
  if buffers move, the replay uses stale pointers.

## Check Your Understanding

<details>
<summary>Why does the legacy default stream serialise work?</summary>

The legacy default stream synchronises with all other streams: any operation in
it waits for previously issued work in every stream and blocks other streams
from starting. A single streamless launch therefore injects a full pipeline
barrier.
</details>

<details>
<summary>Why are events better than std::chrono for kernel timing?</summary>

Events are recorded by the device when the stream reaches them, so the elapsed
time excludes host launch overhead and queueing delay. `std::chrono` measures
host wall time around a launch, which includes whatever the host was doing.
</details>

<details>
<summary>Why are two events enough for any number of double-buffered chunks?</summary>

There are two buffers, so there are two copy-completion conditions to track:
the current chunk's copy and the next chunk's copy. Each buffer reuses the same
event slot every two chunks.
</details>

## Key Takeaways

- A stream is an ordered FIFO queue of device work; work in different streams may overlap.
- The legacy default stream synchronises with all other streams; name streams or use cudaStreamNonBlocking.
- Events measure device time and install cross-stream dependencies via cudaStreamWaitEvent.
- Double buffering overlaps the next copy with the current kernel and hides transfer cost.
- CUDA Graphs capture and replay fixed pipelines, amortising launch overhead.

## 6.9 Exercises

1. Explain why `cudaMemcpyAsync` with pageable memory silently becomes
   synchronous. What does that do to a double-buffered pipeline?
2. The legacy default stream synchronises with all other streams. Draw the
   timeline when a pipeline alternates `cudaMemcpyAsync(..., s1)` and
   `kernel<<<...>>>` with no stream argument.
3. The pipeline loop uses one event per buffer (`copyDone[0]`,
   `copyDone[1]`). Explain why two events are enough for any number of chunks,
   then extend the loop to time each kernel with events (§6.4) and report the
   median per-chunk kernel time.
4. A graph captures a sequence of 500 kernel launches of 2 microseconds each.
   Host launch overhead is 5 microseconds per launch. How much time does one
   replay save compared with 500 individual launches?
