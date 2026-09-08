# Chapter 19: NCCL & Multi-GPU Collective Communication

Chapter 18 described the hardware, NVLink and NVSwitch. This chapter describes
the software that turns that hardware into scalable multi-GPU programs:
**NCCL** (NVIDIA Collective Communications Library). NCCL implements the
**collective operations** - all-reduce, broadcast, all-gather, and
reduce-scatter - used by every data-parallel training loop. It is the library
behind PyTorch's `DistributedDataParallel` and TensorFlow's `MirroredStrategy`.

## 19.1 The Multi-GPU Programming Problem

In data-parallel training, each GPU holds a copy of the model and processes a
different batch. At the end of a step, each GPU has computed a local gradient.
The next step requires the same averaged gradient on every GPU, so the GPUs must
combine their gradients and distribute the result. That operation is an
**all-reduce**.

A naive implementation would have one GPU collect all gradients, sum them, and
broadcast the result. That does not scale: the collector becomes the bottleneck
and most links stay idle. Collective algorithms use all links simultaneously in
structured patterns.

> **Primitive - collective operation.** An operation that involves every rank
> in a group and produces a result that depends on data from all ranks.
> Examples: all-reduce, broadcast, reduce, all-gather, and reduce-scatter.

## 19.2 Collective Operations

| Operation | Input | Output |
|---|---|---|
| **Broadcast** | one rank has data | all ranks have the data |
| **Reduce** | all ranks have data | one rank has the combination |
| **All-reduce** | all ranks have data | all ranks have the combination |
| **All-gather** | each rank has a piece | every rank has all pieces |
| **Reduce-scatter** | each rank has data | each rank has one combined piece |

The mathematical operation is usually sum, but collectives generalise to min,
max, product, and other associative operations. NCCL supports `ncclSum`,
`ncclProd`, `ncclMin`, and `ncclMax`.

## 19.3 NCCL Architecture

NCCL:

- discovers the **topology** (which GPUs are NVLink-connected, which use PCIe,
  and which are on different hosts);
- builds a **communication plan** (ring, tree, or hybrid);
- creates **channels**, independent communication paths that can run
  concurrently;
- uses CUDA **streams**, **peer access**, and on modern systems **NVLink
  atomics and multicast** to move data.

NCCL operations are launched on a CUDA stream like kernels. They are
asynchronous and participate in stream ordering:

```cpp
// Each rank does:
ncclCommInitRank(&comm, nranks, ncclUniqueId, rank);
// ... work ...
ncclAllReduce(sendbuff, recvbuff, count,
              ncclFloat, ncclSum,
              comm, stream);
// NCCL is asynchronous; synchronize the stream when results are needed.
```

> **Primitive - rank.** A process or GPU participating in a collective group.
> Ranks are numbered 0..n-1 and each rank has its own `ncclComm`.
> **Primitive - communicator (`ncclComm`).** The NCCL object that represents a
> rank's membership in a group. Every collective call takes a communicator.

## 19.4 Ring All-Reduce

The classic NCCL algorithm is the **ring all-reduce**. \\(N\\) GPUs form a
ring:

```text
GPU0 -> GPU1 -> GPU2 -> ... -> GPU(N-1) -> GPU0
```

The data is split into \\(N\\) chunks. The algorithm has two phases:

1. **Reduce-scatter.** Each GPU sends a chunk to its neighbour, receives a
   chunk from the other neighbour, adds its own contribution, and passes the
   partial result on. After \\(N-1\\) steps, each GPU holds the complete
   reduced value for one chunk.
2. **All-gather.** Each GPU sends its reduced chunk around the ring again.
   After \\(N-1\\) steps, every GPU has every reduced chunk.

Every link is used in every step. For \\(N\\) GPUs and a message of size
\\(M\\):

- data moved per GPU is approximately \\(2M(N-1)/N\\);
- bandwidth utilisation is optimal for large messages;
- latency grows with \\(N\\) because there are \\(N-1\\) steps.

Rings therefore suit large messages rather than tiny ones.

![NCCL ring all-reduce with four GPUs: every link is used in every step of the reduce-scatter and all-gather phases](../../assets/ch19_nccl_ring_allreduce.svg)

## 19.5 Tree All-Reduce

For small messages or many ranks, a **tree** algorithm has lower latency than a
ring because the number of steps is \\(O(\log N)\\) rather than \\(O(N)\\).

```text
GPU0
|-- GPU1
|   |-- GPU3
|   `-- GPU4
`-- GPU2
    |-- GPU5
    `-- GPU6
```

- **Reduce phase:** leaves send data upward; each parent sums its children and
  its own data.
- **Broadcast phase:** the root sends the total down the tree.

Trees place more traffic on the root's links but have lower latency. NCCL
chooses ring or tree, or a hybrid, based on message size, rank count, and
topology. On NVSwitch systems, **NVLS** (NVLink SHARP) lets the switch perform
reduction in flight, so all-reduce can approach the cost of a single send.

> **Primitive - ring all-reduce.** A bandwidth-optimal all-reduce for large
> messages: reduce-scatter around a ring, then all-gather around the ring.
> **Primitive - tree all-reduce.** A latency-optimal all-reduce for small
> messages: a tree reduce followed by a tree broadcast.

## 19.6 A Complete NCCL Example

A minimal all-reduce program requires a multi-GPU machine with NCCL installed
and one process per GPU:

```cpp
// all_reduce.cu - run with one process per GPU, e.g. mpirun -np 4
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e)); exit(1); } } while (0)
#define CHECK_NCCL(call) do { ncclResult_t r = (call); \
    if (r != ncclSuccess) { fprintf(stderr, "NCCL %s\n", ncclGetErrorString(r)); exit(1); } } while (0)

int main(int argc, char** argv)
{
    const int nranks = 4;             // number of GPUs/processes
    const int rank   = atoi(argv[1]); // this process's rank (0..n-1)
    const int count  = 1 << 20;       // floats per rank

    CHECK_CUDA(cudaSetDevice(rank));  // rank i uses GPU i in this simple setup

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    // In real deployments use MPI to broadcast id to all ranks.

    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));

    float *sendbuf, *recvbuf;
    CHECK_CUDA(cudaMalloc(&sendbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(recvbuf, 0, count * sizeof(float)));

    // Each rank's data: rank + 1 (so the all-reduce sum is known).
    std::vector<float> h(count, static_cast<float>(rank + 1));
    CHECK_CUDA(cudaMemcpy(sendbuf, h.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // All-reduce: every rank ends with sum = 1+2+3+4 = 10 per element.
    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count,
                             ncclFloat, ncclSum,
                             comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> out(count);
    CHECK_CUDA(cudaMemcpy(out.data(), recvbuf, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("rank %d: out[0] = %f (expected 10)\n", rank, out[0]);

    CHECK_NCCL(ncclCommDestroy(comm));
    CHECK_CUDA(cudaFree(sendbuf));
    CHECK_CUDA(cudaFree(recvbuf));
    return 0;
}
```

Build and run (simplified; real multi-node runs use MPI or a launcher):

```bash
nvcc -arch=compute_60 all_reduce.cu -o all_reduce -lnccl
# start one process per GPU with the unique ID propagated by MPI
```

The hard parts of NCCL programming are not the arithmetic. They are creating the
communicator, distributing the unique ID, choosing a topology-aware algorithm,
and ensuring every rank calls collectives in the same order on compatible
streams.

## 19.7 NCCL in PyTorch

Most applications call NCCL through a framework. PyTorch's
`DistributedDataParallel` uses NCCL as its backend:

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl", world_size=4, rank=rank)
model = torch.nn.parallel.DistributedDataParallel(model)
# forward/backward
loss.backward()          # DDP hooks an all-reduce of gradients via NCCL
```

The framework handles communicator setup and launches NCCL collectives on the
correct streams. Understanding ring and tree behaviour still matters:
`NCCL_P2P_LEVEL`, `NCCL_ALGO`, and message size affect whether ring or tree is
selected.

## 19.8 Debugging and Tuning NCCL

```bash
# Verbose logs: topology, chosen algorithms, channel count
NCCL_DEBUG=INFO ./train.py

# Detailed trace for a single collective
NCCL_DEBUG=TRACE ./train.py

# Force specific algorithms / transports
NCCL_ALGO=Ring ./train.py
NCCL_P2P_LEVEL=NV ./train.py

# Measure raw collective bandwidth/latency (from nccl-tests)
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

Common issues:

- **Communicator setup hangs** - the unique ID was not distributed correctly or
  ranks disagree on the world size.
- **Slow all-reduce on small messages** - the selected algorithm may be ring
  when tree would be better; try `NCCL_ALGO=Tree`.
- **P2P disabled or blocked** - check `NCCL_P2P_LEVEL` and topology.
- **Stream mismatch** - the NCCL call must be on the same stream as the kernels
  whose results it consumes, or ordered with events.

## Ring versus Centralised All-Reduce

A centralised all-reduce makes the collector read \\(N-1\\) messages, combine
them, and write \\(N-1\\) results. Traffic through one GPU is \\(O(NM)\\) and
all other links idle. A ring all-reduce spreads the work: every GPU sends and
receives \\(N-1\\) chunks of size \\(M/N\\), so total data per GPU is
\\(O(M)\\) and all links are busy in every step. For large \\(M\\), the ring is
bandwidth-optimal. For small \\(M\\), the \\(N-1\\) serial steps make latency
dominate, so a tree with \\(O(\log N)\\) steps wins. NCCL chooses the algorithm
by measuring the hardware, applying the same measure-don't-guess discipline as
Chapter 16.

On NVSwitch systems, NVLink SHARP lets the switch perform arithmetic while
forwarding data. Each GPU sends its data once and receives the reduced result
once; the switch does the combining. This is the multi-GPU analogue of
computation in the memory system, and it is why NVLink/NVSwitch plus NCCL is the
backbone of large-scale training.

## Common Pitfalls

1. **Calling NCCL collectives in different order on different ranks.**
   Collective operations must be matched across ranks; mismatched order
   deadlocks or corrupts data.
2. **Forgetting stream ordering.** NCCL calls are asynchronous. Reading results
   without synchronising or ordering events can race.
3. **Using the default communicator ID everywhere.** In multi-process runs, the
   unique ID must be generated once and broadcast through MPI, a file, or an
   environment variable; every rank must use the same ID.
4. **Ignoring topology.** A ring over PCIe-only GPUs is much slower than a ring
   over NVLink. Check `nvidia-smi topo -m` and `NCCL_P2P_LEVEL`.
5. **Assuming NCCL is only for training.** NCCL supports any all-to-all GPU
   communication: distributed inference, multi-GPU sorts, graph processing, and
   scientific computing.

## Check Your Understanding

<details>
<summary>What is the difference between ring and tree all-reduce?</summary>

Ring all-reduce is bandwidth-optimal for large messages: each GPU sends and
receives \\(N-1\\) chunks and every link stays busy, but latency grows with
\\(N\\). Tree all-reduce has \\(O(\log N)\\) steps and is better for small
messages, but the root and upper links carry more traffic. NCCL chooses between
them based on message size, rank count, and topology.
</details>

<details>
<summary>Why must the ncclUniqueId be shared among ranks?</summary>

The unique ID is the bootstrap token that lets all ranks agree they are joining
the same communicator group. Rank 0 generates it, and it must be distributed
through MPI, a file, or the environment before `ncclCommInitRank`.
</details>

<details>
<summary>Why is all-reduce the core operation of data-parallel training?</summary>

Every GPU computes a local gradient for the same model parameters. The next
step needs the same averaged gradient on every GPU, so the gradients must be
summed or averaged across all GPUs and the result made available to all. That
is exactly an all-reduce.
</details>

## Exercises

1. Trace ring all-reduce for \\(N=4\\) and a four-chunk message: list what each
   GPU sends and receives in each of the six steps (three reduce-scatter and
   three all-gather).
2. Using `nccl-tests`, measure `all_reduce_perf` for 8 bytes versus 128 MB and
   explain which algorithm NCCL chose and why.
3. Modify the example program to use `ncclBroadcast` instead of all-reduce:
   rank 0 sends its buffer and all ranks receive it. Verify with a known value.
4. Explain how NVLink SHARP (NVLS) makes all-reduce cheaper than the ring
   algorithm and why the switch is a natural place to do arithmetic.

## Sources and Further Reading

- NVIDIA, *NCCL User Guide*: <https://docs.nvidia.com/deeplearning/nccl/user-guide/>
- NVIDIA, *NCCL GitHub repository*: <https://github.com/NVIDIA/nccl>
- NVIDIA, *CUDA C++ Programming Guide*, "Peer-to-Peer" and "Streams" sections: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- Patarasuk and Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations," *Journal of Parallel and Distributed Computing*, 2009. Source of the ring all-reduce analysis.
