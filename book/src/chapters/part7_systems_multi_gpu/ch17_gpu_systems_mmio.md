# Chapter 17: GPU Systems Programming & Memory-Mapped I/O

Chapters 3-16 treated the GPU as an API: `cudaMalloc`, `cudaMemcpy`, and
`kernel<<<...>>>`. This chapter examines the systems side: what happens on the
wire when the CPU tells the GPU to do something, why device memory differs from
host memory, how data moves without the CPU, and what *memory-mapped I/O*,
*BAR*, *DMA*, and *IOMMU* mean for a GPU. This knowledge is not required to
write correct CUDA, but it is required to explain why a kernel or transfer is
slow rather than guessing.

## 17.1 The GPU Is a PCIe Device

A discrete GPU, and an integrated GPU on a Jetson module, connects to the CPU
through a **PCIe** (Peripheral Component Interconnect Express) link. PCIe is a
packet-based serial bus that replaced the parallel PCI bus. Every PCIe device,
whether GPU, NVMe SSD, network card, or USB controller, appears to the CPU as a
device with:

- a **configuration space**, a 4 KB region the CPU reads at boot to discover
  the device, its vendor, BARs, and interrupts;
- one or more **BARs** (Base Address Registers) that tell the CPU where the
  device's control registers and device memory are mapped in the CPU's
  physical address space;
- **MMIO regions** for control registers, doorbells, and mailboxes;
- **DMA capabilities**, so the device can read and write system memory
  directly.

GPUs are PCIe devices with unusually large amounts of device memory and high DMA
bandwidth.

> **Primitive - PCIe.** A packet-switched, point-to-point serial bus. Each lane
> is a differential pair; x16 means 16 lanes. PCIe Gen4 x16 delivers roughly 32
> GB/s raw in each direction, which is why host-device copies top out around
> 20-25 GB/s in practice (Chapter 4).

> **Primitive - BAR (Base Address Register).** A PCIe configuration-space
> register that defines where a device's registers and memory are mapped into
> the CPU's physical address space. The CPU can read and write those addresses
> with ordinary load and store instructions.

CUDA has `cudaMemcpy` because the CPU cannot ordinarily see GPU memory as RAM.
GPU memory lives behind the device's BAR or a DMA mapping, and its cost model
differs from host RAM: a CPU load from a GPU MMIO region can be extremely slow,
while a DMA engine can move gigabytes at near-bus speed without the CPU
touching every byte. The CUDA API manages these two worlds.

## 17.2 Memory-Mapped I/O: The Control Path

**Memory-mapped I/O (MMIO)** exposes a device's control registers as if they
were memory addresses. The CPU writes a command by storing a value to an
address; the PCIe controller converts that store into a PCIe write transaction
that lands in the device's register file.

For a GPU, the MMIO region contains items such as:

- the **doorbell**, a register the CPU writes to tell the GPU that new work has
  been queued;
- **command buffers or push buffers**, rings of commands the CPU writes into
  host memory and then announces with the doorbell;
- **mailbox registers** for small status exchanges;
- **performance, temperature, and power registers**, exposed through NVML and
  `nvidia-smi` but read through the same MMIO path.

The store is **posted**: the CPU does not wait for the GPU to process the
command. This is why kernel launches are asynchronous (Chapter 6): the host
writes the launch command and rings the doorbell, and the GPU processes the work
when it reaches it.

![MMIO control path vs DMA data path: the CPU writes a doorbell through MMIO while the DMA engine moves bulk data](../../assets/ch17_mmio_path.svg)

> **Primitive - MMIO.** A device's control registers are mapped into the CPU's
> physical address space; CPU loads and stores to those addresses become PCIe
> transactions to the device. MMIO is the control path: small, slow, and used
> for commands and status, not bulk data.

Every MMIO read or write is a round trip across the bus into the device's
register file. A CPU store to MMIO can take hundreds of nanoseconds and cannot
be speculatively repeated. Bulk data therefore moves by DMA, not MMIO. MMIO
tells the GPU where the data is; DMA moves the data.

## 17.3 BARs and PCIe Configuration Space

Linux enumerates every PCIe device at boot and reads its configuration space.
The relevant fields for a GPU are:

- **Vendor ID and Device ID**, for example NVIDIA's vendor ID is `0x10de`;
- **Class Code**, `0x03` for display controllers;
- **BARs**, up to six 32-bit or 64-bit base address registers;
- **MSI-X** interrupt configuration.

GPU BARs typically map:

- **BAR0**: MMIO register space (control registers, doorbells);
- **BAR1/BAR2**: a window into the GPU framebuffer (device memory). On some
  systems this window supports legacy framebuffer consoles;
- **BAR3+**: device-specific regions such as UEFI GOP, ATS, or resizable BAR.

On Linux, `lspci` exposes this:

```bash
lspci | grep -i nvidia
# 01:00.0 3D controller: NVIDIA Corporation GA102 [GeForce RTX 3080] (rev a1)

lspci -v -s 01:00.0
# ... Region 0: Memory at f0000000 (64-bit, prefetchable)
#     Region 2: Memory at ... (64-bit, prefetchable)
```

**Resizable BAR.** Modern GPUs support Resizable BAR (called Smart Access
Memory on AMD platforms). The BAR window into device memory can be enlarged so
that the CPU can map a large fraction, or all, of GPU memory into the CPU
address space. This is the physical substrate that makes unified memory and
zero-copy efficient on modern systems: the CPU can reach device memory through
the BAR with ordinary loads and stores, subject to PCIe latency.

> **Primitive - device memory window.** The GPU framebuffer is not directly
> addressable by the CPU by default. A BAR creates a CPU address window that the
> CPU can map and access, but each access crosses PCIe and obeys bus latency and
> ordering rules. `cudaMalloc` returns a device pointer, not a CPU pointer;
> `cudaHostAlloc` returns a host pointer that the GPU's DMA engine can reach.

## 17.4 DMA: The Data Path

If MMIO is the control path, **DMA** (Direct Memory Access) is the data path. A
DMA engine copies data between system memory and device memory without the CPU
touching each byte. The sequence for
`cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice)` is roughly:

1. The user-mode driver pins the host buffer. For pageable memory it may first
   copy into a pinned staging buffer (Chapter 4).
2. The CPU programs a **DMA descriptor** containing the source physical
   address, destination device address, and length.
3. The CPU rings the DMA engine's doorbell.
4. The DMA engine walks the descriptor, reads from system memory, and writes to
   device memory over PCIe. The CPU is free to do other work.
5. An interrupt or doorbell response notifies the driver that the copy is
   complete.

```text
CPU memory        DMA engine          GPU memory
+---------+   +--------------+   +-------------+
| h_a     |-->| read  ------>|-->| d_a         |
+---------+   +--------------+   +-------------+
             no CPU involvement in data movement
```

> **Primitive - DMA (Direct Memory Access).** A hardware engine that copies
> data between memory domains, such as host memory and device memory, without
> CPU per-byte involvement. The CPU sets up a descriptor and the engine performs
> the bulk transfer.

Pageable memory needs a staging copy because the DMA engine requires physical
addresses that remain valid while the transfer is in flight. Ordinary `malloc`
pages can be swapped or moved by the OS. Pinning the pages with
`cudaMallocHost` guarantees the physical pages stay put, so DMA can access them
directly. This is the systems-level reason for Chapter 4's rule: pin what you
stream.

## 17.5 IOMMU/SMMU: The DMA Firewall

An **IOMMU** (Intel/AMD terminology) or **SMMU** (ARM terminology) sits between
devices and system memory. It does for DMA what the CPU's MMU does for CPU
loads: it translates device virtual addresses to physical addresses and
enforces permissions.

Without an IOMMU, a buggy or malicious device could DMA to any physical
address, including kernel memory. With an IOMMU:

- the device receives virtual addresses mapped by the IOMMU to physical pages;
- the CPU can revoke mappings, which matters for hot-unplug and isolation;
- the device cannot touch memory that was not explicitly granted;
- scatter-gather becomes natural: non-contiguous physical pages can be presented
  to the device as a contiguous list.

The trade-off is translation overhead and, historically, lower bandwidth on
some platforms. High-performance GPU users sometimes disable the IOMMU or use a
bypass mode after measuring. `cudaHostAlloc` with pinned memory remains the
fastest path because the driver can pre-map pinned pages in the IOMMU once and
reuse the mapping.

> **Primitive - IOMMU/SMMU.** A hardware unit that translates and validates DMA
> addresses, giving devices virtual addresses and preventing access to
> arbitrary physical memory.

For CUDA programmers this means `cudaMemcpy` is not "memcpy on the GPU"; it is
a sequence of mapping, descriptor programming, doorbell, DMA, and unmap. When
`cudaMemcpyAsync` takes longer than expected, part of the cost can be page-table
and IOMMU work rather than the wire transfer.

## 17.6 User-Mode Driver vs Kernel-Mode Driver

CUDA has two driver layers:

- **Kernel-Mode Driver (KMD)** runs in the kernel, as `nvidia` or `nvgpu` on
  Jetson. It owns the device, MMIO mappings, interrupts, power, and memory
  mappings. Only the kernel can program the device directly.
- **User-Mode Driver (UMD)** runs in the process as `libcuda.so`. It implements
  the CUDA API, launches, memory management, and context state. For
  performance, the UMD avoids kernel round trips by writing commands into
  user-mapped command buffers and ringing the doorbell directly. This is why
  modern GPU launches can be cheap: many do not require a kernel call.

The split explains why:

- a CUDA context is per-process rather than per-thread;
- most CUDA API calls do not enter the kernel; they operate on command buffers
  in user space;
- a GPU crash takes down the context rather than the whole system; the KMD
  resets the GPU and returns an error.

```text
   Your process                     Kernel                     GPU
+------------------+   ioctl    +--------------+  MMIO/DMA +---------+
| libcuda.so (UMD) | ---------> | nvidia (KMD) | --------> | device  |
|  CUDA API        |  (rarely)  |  device mgmt |           |         |
|  command buffers | ---------> | IRQ handling |           |         |
+------------------+  doorbell  +--------------+           +---------+
```

> **Primitive - UMD (User-Mode Driver).** The library linked into the process.
> It implements the CUDA API, manages per-process state, and submits work to the
> GPU with as few kernel transitions as possible.
> **Primitive - KMD (Kernel-Mode Driver).** The kernel module that owns the
> device, handles interrupts and power, and is the only component allowed to
> program the hardware directly.

## 17.7 Observing the System on Linux

Linux exposes this machinery without writing a driver:

```bash
# PCIe topology
lspci -tv

# GPU MMIO regions / BARs
lspci -v -s $(lspci | grep -i nvidia | awk '{print $1}' | head -1)

# Kernel driver in use
lspci -k -s $(lspci | grep -i nvidia | awk '{print $1}' | head -1)

# Interrupts / IOMMU groups
ls /sys/kernel/iommu_groups/
cat /proc/interrupts | grep -i nvidia

# Device memory size as the kernel sees it
cat /sys/bus/pci/devices/*/resource 2>/dev/null | head
```

On a Jetson the GPU is part of the SoC, but the same concepts appear through
`/sys/class/misc/nvhost-*`, debugfs, and the `nvgpu` driver interface.

Inspect the BAR sizes, the driver name, and whether the device is in an IOMMU
group. A GPU behind an IOMMU with a slow translation path explains copies that
run below the link specification.

## 17.8 From MMIO to CUDA APIs

Every CUDA API maps onto the systems concepts above:

| CUDA API | Systems concept |
|---|---|
| `cudaMalloc` | Allocates device memory managed by the KMD; returns a device pointer, not a CPU pointer |
| `cudaMemcpy` | Pins or maps host memory, programs a DMA descriptor, rings a doorbell |
| `cudaMemcpyAsync` | Same, but queued in a stream so DMA can overlap kernels |
| `cudaMallocHost` | Allocates pinned host memory so DMA can access it directly |
| `cudaHostAllocMapped` / `cudaHostGetDevicePointer` | Maps host memory into the device address space (zero-copy) |
| `cudaMallocManaged` | Uses page fault and migration machinery, plus IOMMU or BAR mapping, to present one virtual address space |
| `cudaDeviceEnablePeerAccess` | Programs the GPU's P2P DMA path (Chapter 18) |

CUDA is a systems API presented as a math API. Every call is a transaction with
a driver, a DMA engine, and a memory map. When performance is surprising, the
explanation is usually in this model: MMIO for control, DMA for data, IOMMU for
safety, and pinning for speed.

## CPU Access to Device Memory

The common question "why can't I dereference a device pointer on the host?" has
a precise systems answer:

1. The device pointer is a GPU virtual address, meaningful only in the GPU's
   address space.
2. GPU memory sits behind a BAR. The CPU could map it, but mapping all of
   device memory into the CPU address space costs page-table and IOMMU entries,
   every CPU access crosses PCIe and is slow, and CPU accesses must be ordered
   with GPU accesses.
3. The driver therefore returns a handle, the device pointer, and provides
   explicit copy APIs. The copy API performs the transfer efficiently with DMA,
   and the driver handles ordering.

Unified memory differs because the driver presents one virtual address that both
the CPU and GPU can dereference. The driver uses page faults, migrations, and on
modern systems a BAR window or ATS (Address Translation Services). Every fault
or migration is a systems operation with hidden latency, which is why Chapter 4
recommends explicit prefetching.

## Common Pitfalls

1. **Treating device pointers as host pointers.** Dereferencing a device
   pointer on the CPU is undefined behaviour and usually a segmentation fault.
2. **Not pinning streaming buffers.** Pageable memory forces a staging copy and
   silently destroys async-copy performance.
3. **Believing MMIO is ordinary memory.** MMIO is not cacheable RAM; a CPU
   store to a doorbell is a command, not data storage. MMIO is not for bulk
   data.
4. **Ignoring IOMMU overhead.** DMA through an IOMMU can be measurably slower on
   some platforms. Measure with and without, then decide.
5. **Assuming `cudaMemcpy` is synchronous by hardware design.** It is
   synchronous in the API sense, but underneath it is a DMA operation. The cost
   is descriptor setup plus transfer, which is why async copies on pinned memory
   overlap well.

## Check Your Understanding

<details>
<summary>What is the difference between MMIO and DMA?</summary>

MMIO is the control path: the CPU writes commands and status to device registers
through PCIe transactions, such as a doorbell. DMA is the data path: a hardware
engine moves bulk data between memory domains without CPU per-byte involvement.
MMIO tells the device what to do; DMA moves the data it operates on.
</details>

<details>
<summary>Why does pageable host memory need a staging copy for DMA?</summary>

DMA requires physical addresses that remain valid for the duration of the
transfer. Pageable pages can be swapped or moved by the OS, so the runtime
copies the data into a pinned staging buffer whose physical pages are locked.
Pinned memory (`cudaMallocHost`) skips that staging copy.
</details>

<details>
<summary>What does the IOMMU protect against?</summary>

It prevents a device from DMAing to arbitrary physical memory. The IOMMU
translates device virtual addresses to physical addresses and enforces
permissions, so a device can touch only memory the driver explicitly mapped for
it.
</details>

## Exercises

1. Run `lspci -v` on your machine and identify the GPU's BARs. What is the size
   of the BAR that maps device memory? On Jetson, inspect the SoC memory map
   instead.
2. Explain, using the MMIO/DMA model, why `cudaMemcpyAsync` can overlap with a
   kernel while synchronous `cudaMemcpy` cannot.
3. Draw the full path of a `cudaMemcpy` from a pinned host buffer to device
   memory, naming every component: UMD, KMD, IOMMU, DMA engine, PCIe, and
   device memory.
4. Why is exposing device memory as a plain CPU-mapped BAR and letting
   applications dereference device pointers directly a bad idea? Give two
   reasons from this chapter.

## Sources and Further Reading

- NVIDIA, *CUDA C++ Programming Guide*, "Hardware Implementation" and "Compute Capabilities": <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- PCI-SIG, *PCI Express Base Specification* and `lspci`/`pciutils` documentation for device enumeration.
- Linux kernel documentation, "DMA-API" and "IOMMU" sections: <https://www.kernel.org/doc/html/latest/core-api/dma-api.html>
