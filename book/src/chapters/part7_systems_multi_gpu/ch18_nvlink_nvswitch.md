# Chapter 18: NVLink & NVSwitch - The Multi-GPU Interconnect

The previous chapters were single-GPU. This chapter introduces the hardware
that supports multi-GPU programming: **NVLink**, NVIDIA's high-bandwidth
point-to-point interconnect, and **NVSwitch**, a switch that connects many
NVLink ports. The chapter covers topology, memory semantics, peer-to-peer
copies in CUDA, and when NVLink matters compared with PCIe.

## 18.1 Scaling Beyond One GPU

Training a large model or processing a large dataset eventually exceeds one
GPU's memory and compute. The classic scaling strategies are:

1. **Model parallelism** - split the model across GPUs; each GPU holds a piece.
2. **Data parallelism** - each GPU holds a full model copy but processes a
   different batch. Gradients must be reduced across GPUs every step.
3. **Pipeline parallelism** - different model stages live on different GPUs.

All three move data between GPUs. Inter-GPU communication speed determines
whether a multi-GPU system scales or idles waiting for communication. NVLink
exists to make that communication fast.

## 18.2 NVLink: Point-to-Point, High-Bandwidth

**NVLink** is a high-bandwidth, low-latency, point-to-point interconnect between
two GPUs, and on some platforms between a GPU and CPU. It is not PCIe. Its key
properties:

- **Higher bandwidth than PCIe.** An NVLink connection is bidirectional. NVLink
  3.0 (Ampere) provides 25 GB/s per direction per link; an A100 has 12 links,
  for 600 GB/s of total bidirectional bandwidth. PCIe Gen4 x16 provides about
  32 GB/s total. NVLink can be an order of magnitude faster than PCIe for
  peer-to-peer traffic.
- **Lower latency for small messages.** NVLink uses a protocol designed for
  GPU-to-GPU traffic rather than a PCIe root-complex round trip.
- **Direct GPU-to-GPU DMA.** One GPU's DMA engine can read and write another
  GPU's memory without bouncing through host memory.

```text
GPU 0                 GPU 1
+------+   NVLink    +------+
| SM   |<----------->| SM   |
| HBM  |<----------->| HBM  |
+------+             +------+
   P2P, no host involvement
```

> **Primitive - NVLink.** NVIDIA's proprietary high-bandwidth, point-to-point
> GPU interconnect. It carries data and, on supported architectures, coherent
> memory traffic directly between GPUs.

Each GPU has a fixed number of NVLink links. An A100 has 12 links, an H100 has
18, and consumer cards have fewer or none; many consumer cards use PCIe only.
Links can be used as direct GPU-to-GPU connections in two-GPU systems or as
connections through an NVSwitch in larger systems.

## 18.3 NVSwitch and Topologies

A full mesh of \\(N\\) GPUs requires \\(N(N-1)/2\\) links per GPU. For eight
GPUs that is 28 links per GPU, which is impractical. **NVSwitch** provides a
crossbar inside the chassis: each GPU connects to the switch, and the switch
forwards traffic between any pair.

- **2 GPUs:** direct NVLink; no switch is needed.
- **4-GPU HGX baseboards:** four GPUs connected to two NVSwitches.
- **8-GPU HGX baseboards:** eight GPUs connected to NVSwitches in a topology
  that gives every GPU high-bandwidth access to every other GPU.

![NVLink topologies: two GPUs over a direct NVLink versus four GPUs connected through an NVSwitch crossbar](../../assets/ch18_nvlink_topology.svg)

This is why `nvidia-smi topo -m` can report `NV#` for peer paths: the topology
decides whether a copy between two GPUs uses a direct link, a switch hop, or a
PCIe root-complex path.

> **Primitive - NVSwitch.** A crossbar switch that connects many GPUs' NVLink
> ports, giving each GPU high-bandwidth access to every other GPU without a
> full mesh of direct links.

## 18.4 NVLink Memory Semantics

NVLink is not just fast PCIe. It changes the memory model between GPUs:

- **Peer-to-peer (P2P) access.** A kernel on GPU 0 can read and write GPU 1's
  memory when peer access is enabled and the topology allows it. The access
  travels over NVLink rather than host memory.
- **Peer atomics.** Atomic operations can target another GPU's memory. This
  enables lock-free multi-GPU algorithms, but peer-atomic throughput over NVLink
  is lower than local HBM atomics. Use them sparingly.
- **Coherent and address-translation features.** NVLink-C2C (chip-to-chip) on
  Grace-Hopper carries coherent CPU-GPU traffic with hardware-managed cache
  coherence. The CPU and GPU can share one unified memory domain.
- **Unified memory over NVLink.** With `cudaMallocManaged`, pages can migrate
  between GPUs over NVLink. This is convenient, but page migration is a system
  operation and can be slower than explicit P2P copies.

The practical rule is: explicit P2P copies such as `cudaMemcpyPeerAsync` are
predictable and fast; unified memory is convenient but must be measured.

## 18.5 Peer-to-Peer in CUDA

The CUDA API for P2P is small:

```cpp
// 1. Query whether P2P is possible and enable it.
int canAccess = 0;
cudaDeviceCanAccessPeer(&canAccess, device0, device1);
if (canAccess) {
    cudaSetDevice(device0);
    cudaDeviceEnablePeerAccess(device1, 0);
}

// 2. Copy directly between device memories.
cudaMemcpyPeerAsync(d_buf1, device1,
                    d_buf0, device0,
                    bytes, stream);

// 3. Or, once peer access is enabled, a kernel on GPU 0 can read GPU 1's
//    pointer directly (subject to topology and architecture support).

// 4. Disable when done.
cudaDeviceDisablePeerAccess(device1);
```

> **Primitive - peer access.** The CUDA mechanism that lets one device access
> another device's memory. It requires hardware support (NVLink or PCIe P2P), a
> compatible topology, and explicit enabling with
> `cudaDeviceEnablePeerAccess`.

Check the topology first. `cudaDeviceCanAccessPeer` returns true only when the
platform supports it. Two GPUs can sometimes use P2P over PCIe when they share a
root complex; NVLink-connected GPUs always support it. `nvidia-smi topo -m`
shows which GPUs are connected:

```bash
nvidia-smi topo -m
#        GPU0  GPU1  GPU2  GPU3 ...
# GPU0    X    NV#   NV#   NV#
# ...
```

## 18.6 When NVLink Matters

Multi-GPU execution time is approximately:

```text
total_time = compute_time + communication_time
speedup = single_gpu_time / (compute_time/N + communication_time)
```

If communication time is significant, adding GPUs does not scale. NVLink helps
by shrinking communication time.

**When NVLink matters:**

- Frequent all-reduces, such as gradient synchronisation in data-parallel
  training.
- Pipeline parallelism in which activations pass between GPUs.
- Fine-grained P2P reads in multi-GPU databases or graph analytics.

**When PCIe is sufficient:**

- One-time dataset uploads from host to GPU.
- Coarse task parallelism in which GPUs rarely communicate.
- Communication that is small relative to compute, as in embarrassingly
  parallel batches.

Apply the measurement discipline of Chapter 16. Use `cudaMemcpyPeer` with CUDA
events or `nccl-tests` (Chapter 19) to measure the achievable P2P bandwidth on
the actual hardware.

## 18.7 Topology Discovery

```bash
# Matrix of GPU-to-GPU links (NV# = NVLink, PIX/PXB = PCIe paths)
nvidia-smi topo -m

# Detailed NVLink status (link count, active links, errors)
nvidia-smi nvlink -s
```

The NVML API exposes the same information programmatically through functions
such as `nvmlDeviceGetTopologyCommonAncestor`, which is useful in tools that
must adapt to the hardware.

## NVLink and the Data Path

The surface answer is higher bandwidth. The deeper answer concerns the data
path.

PCIe P2P between two GPUs often passes through the root complex: GPU 0, PCIe
switch or root complex, GPU 1. This adds latency and shares the host's PCIe
bandwidth. NVLink is a direct GPU-to-GPU link, or a path through a switch
crossbar. There is no host-memory hop, no root-complex arbitration, and the
protocol is designed for GPU memory semantics, including atomics and, on some
architectures, coherence.

The result is lower latency per message and more concurrent independent
communication streams. NCCL (Chapter 19) prefers NVLink topologies because
collective algorithms require many simultaneous point-to-point transfers.

NVLink-C2C on Grace-Hopper connects CPU and GPU with coherent memory semantics.
The CPU and GPU can share a memory pool with hardware coherence, removing copies
from the programming model. This differs from the discrete PCIe model but uses
the same P2P, atomics, and topology concepts.

## Common Pitfalls

1. **Assuming all GPUs can do P2P.** Check `cudaDeviceCanAccessPeer`; many
   consumer platforms do not support PCIe P2P or require special settings.
2. **Using unified memory instead of explicit P2P on hot paths.** Page
   migration over NVLink is not free. Measure before replacing
   `cudaMemcpyPeerAsync`.
3. **Ignoring topology.** Two GPUs on different PCIe switches can have a much
   slower path than two GPUs sharing a root complex. Read `nvidia-smi topo -m`.
4. **Using peer atomics on hot paths.** NVLink atomics are slower than local
   HBM atomics; use them for rare synchronisation rather than per-element
   updates.
5. **Forgetting to disable peer access** before changing device context or
   shutting down. The runtime can otherwise leave stale mappings.

## Check Your Understanding

<details>
<summary>Why is NVLink faster than PCIe for GPU-to-GPU traffic?</summary>

NVLink is a direct high-bandwidth GPU interconnect with low latency and no
host or root-complex round trip. PCIe P2P often routes through the root complex
and shares host PCIe bandwidth. NVLink is purpose-built for GPU memory
semantics and carries more concurrent traffic.
</details>

<details>
<summary>What does NVSwitch add over direct NVLink?</summary>

It lets more than two GPUs communicate at high bandwidth without a full mesh of
direct links. Each GPU connects to the switch and the switch forwards traffic
between any pair, enabling all-to-all patterns in four- and eight-GPU systems.
</details>

<details>
<summary>What does cudaDeviceEnablePeerAccess do?</summary>

It enables one device to access another device's memory directly, subject to
hardware support and topology. After enabling, `cudaMemcpyPeer` and, in some
cases, kernels can read and write peer memory without going through host
memory.
</details>

## Exercises

1. Run `nvidia-smi topo -m` on a multi-GPU machine, or research a DGX topology
   diagram, and identify which GPU pairs are NVLink-connected.
2. Write a small program that measures `cudaMemcpyPeer` bandwidth between two
   GPUs with CUDA events and compare it with host-device copy bandwidth.
3. Explain why a full mesh of direct NVLink links is impractical for eight GPUs
   and how NVSwitch solves the problem.
4. In data-parallel training, gradients are all-reduced every step. Would you
   prefer NVLink or PCIe for that workload? Justify with the model in §18.6.

## Sources and Further Reading

- NVIDIA, *NVLink & NVSwitch* product documentation: <https://www.nvidia.com/en-us/data-center/nvlink/>
- NVIDIA, *CUDA C++ Programming Guide*, "Peer-to-Peer Access" and "Unified Memory" sections: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA, *NVIDIA Multi-GPU Communication Library (NCCL)* documentation: <https://docs.nvidia.com/deeplearning/nccl/user-guide/>
