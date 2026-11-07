"""The figures of chapter 17, drawn with figkit and figparts.

    python3 tools/draw_ch17.py

Addresses, bytes and interrupt numbers are the ones read on the chapter's 2 machines (a Jetson Orin
NX and a Jetson AGX Thor); register offsets are from NVIDIA's open-source headers.
Times in the queue figure are from code/ch17_systems/systems.cu.
"""
from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, arrow, card, cylinder, legend, line,
                      mono, note, save, sheet, tag, text, wire, zone)
from figkit import must_fit, width_of

OUT = "src/figures/"


def box(x, y, w, h, kind, label="", on=True, cls="mono", fill=None):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        size = 11 if cls == "pill" else 12
        must_fit(label, w - 6, size, mono=cls == "mono", bold=cls == "pill")
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", fill or (accent if on else MUTED)))
    return "\n".join(parts)


def hline(x1, x2, y, kind="flow", label=None, dashed=False):
    """A horizontal arrow from x1 to x2 (either direction), with an optional label above it."""
    colour = WIRE if kind == "flow" else KINDS[kind][0]
    marker = "head" if kind == "flow" else f"head-{kind}"
    end = x2 - 3 if x2 > x1 else x2 + 3
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts = [f'  <path d="M{x1},{y} H{end}" fill="none" stroke="{colour}" stroke-width="1.8"'
             f' stroke-linecap="round"{dash} marker-end="url(#{marker})"/>']
    if label:
        parts.append(text((x1 + x2) / 2, y - 7, label, "mono", "middle", colour))
    return "\n".join(parts)


# 23.1  How a GPU is attached: a discrete card on PCIe, an on-chip PCIe function, a platform device.
cols = [(16, "discrete GPU", "with root complex", "PCIe 4.0/5.0 × 16", "GPU card",
         ["BAR0: registers", "BAR1: memory window"], "own memory", ["GDDR or HBM,", "24 to 80 GB"]),
        (230, "PCIe-function GPU", "with root port", "on-chip PCIe 01:00.0", "GPU, 10de:2b00",
         ["BAR0: 16 MiB registers", "no memory window"], "shared memory", ["LPDDR5X, 128 GB,", "also the CPU's"]),
        (444, "platform-device GPU", "no PCIe link", "on-chip bus, no PCIe", "GPU 17000000.gpu",
         ["3 windows from the", "device tree"], "shared memory", ["LPDDR5, 16 GB,", "also the CPU's"])]
parts = []
for x, title, cpu, link, gpu, gsub, mem, msub in cols:
    parts += [zone(x, 104, 200, 400, title),
              card(x + 8, 132, 184, 54, "client", "CPU", cpu, glyph="chip"),
              arrow(x + 100, 186, 238),
              card(x + 8, 238, 184, 74, "service", gpu, gsub),
              arrow(x + 100, 312, 368),
              cylinder(x + 24, 368, 152, 84, "store", mem, msub)]
    wide = width_of(link, 11, bold=True) + 14
    parts.append(tag(x + 100 - wide / 2, 203, link, "client"))
parts += [note(16, 534, "A discrete GPU is a card at the end of a PCIe link, with its own memory behind a window", room=628),
          note(16, 552, "(BAR1). An integrated GPU is on the chip; it answers as a PCIe function, or it is a", room=628),
          note(16, 570, "platform device described by the device tree. Both kinds share 1 memory with the CPU.", room=628)]
save(OUT + "ch23-attach.svg", 660, 594,
     "3 columns. Discrete GPU: a CPU with a root complex joined by a PCIe 4.0 or 5.0 link of 16 lanes to a GPU card "
     "whose BAR0 holds registers and BAR1 a window on its own GDDR or HBM memory of 24 to 80 GB. Integrated GPU as a PCIe function: the "
     "CPU's root port joined to an on-chip GPU that answers as PCIe function 01:00.0, vendor 10de device 2b00, with a "
     "16 MiB BAR0 of registers and no memory window; memory is 128 GB of LPDDR5X shared with the CPU. Integrated GPU as a platform "
     "device: the CPU joined by an on-chip bus to the GPU platform device 17000000.gpu, with 3 register windows from the device "
     "tree, and 16 GB of LPDDR5 shared with the CPU.",
     "3 ways to attach a GPU", "The GPU is a device the CPU reaches through addresses.", parts)


# 23.2  1 physical address space, many destinations: the windows from a device tree.
ranges = [("0x0_03B4_1000", "4 KiB", "GPU window 3", "service"),
          ("0x0_1700_0000", "16 MiB", "GPU bar0", "service"),
          ("0x0_1800_0000", "16 MiB", "GPU bar1", "service"),
          ("RAM ranges", "16 GB", "DRAM", "store"),
          ("0x24_2800_0000", "16 KiB", "NVMe BAR0", "outside")]
parts = [card(16, 210, 150, 74, "client", "CPU core", ["store [address],", "value"]),
         wire([(166, 247), (204, 247)])]
rows = []
for k, (addr, size, dest, kind) in enumerate(ranges):
    y = 150 + 40 * k
    rows += [box(208, y, 124, 30, kind, addr), box(336, y, 64, 30, kind, size, cls="pill")]
parts += [zone(196, 120, 216, 236, "address decoder")] + rows
dests = [(152, "GPU registers", ["bar0, bar1, window 3", "driver: nvgpu (gk20a)"], "service"),
         (262, "DRAM", ["cached, ordinary", "memory"], "store"),
         (342, "NVMe controller", ["on PCIe 0004:01:00.0", "driver: nvme"], "outside")]
for y, name, sub, kind in dests:
    parts.append(card(452, y, 192, 64, kind, name, sub))
parts += [f'  <path d="M400,165 H412 V245 H400" fill="none" stroke="{WIRE}" stroke-width="1.6"/>',
          f'  <circle cx="412" cy="184" r="3" fill="{WIRE}"/>',
          wire([(412, 184), (452, 184)]),
          wire([(400, 285), (426, 285), (426, 294), (452, 294)]),
          wire([(400, 325), (430, 325), (430, 374), (452, 374)]),
          note(16, 440, "The interconnect sends every load and store by its address. An address in a device's", room=628),
          note(16, 458, "window reaches that device's registers; an address in a RAM range reaches DRAM. The", room=628),
          note(16, 476, "same store instruction does both: the address alone decides.", room=628)]
save(OUT + "ch23-address-map.svg", 660, 498,
     "A CPU core issues store to an address. An address decoder holds 5 ranges read from a device tree: 0x0_03B4_1000, "
     "4 KiB, GPU window 3; 0x0_1700_0000, 16 MiB, GPU bar0; 0x0_1800_0000, 16 MiB, GPU bar1; the RAM ranges, 16 GB, "
     "DRAM; 0x24_2800_0000, 16 KiB, NVMe BAR0. The 3 GPU ranges lead to the GPU's registers, driven by nvgpu; the RAM "
     "ranges to DRAM; the NVMe range to the NVMe controller at PCIe 0004:01:00.0.",
     "1 address space, many destinations", "The windows of an integrated GPU's machine, from its device tree.", parts)


# 23.3  The first 64 bytes of an integrated GPU's configuration space, decoded.
raw = ("de 10 00 2b 06 04 10 00 a1 00 02 03 00 00 00 00 "
       "0c 00 00 c0 d0 00 00 00 00 00 00 00 00 00 00 00 "
       "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 "
       "00 00 00 00 40 00 00 00 00 00 00 00 a5 01 00 00").split()
field = {0: "client", 1: "client", 2: "client", 3: "client", 4: "service", 5: "service",
         9: "model", 10: "model", 11: "model", 52: "outside", 61: "danger"}
field.update({k: "store" for k in range(16, 24)})
cw = 33
parts = [mono(80 + cw * c + cw / 2 - 2, 118, f"{c:x}", "middle", MUTED) for c in range(16)]
for r in range(4):
    parts.append(mono(70, 146 + 34 * r, f"0x{16 * r:02x}", "end", MUTED))
    for c in range(16):
        k = 16 * r + c
        kind = field.get(k)
        parts.append(box(80 + cw * c, 128 + 34 * r, cw - 4, 28, kind or "idle", raw[k], on=bool(kind)))
decode = [("client", "bytes 0-3", "vendor 0x10de (NVIDIA), device 0x2b00"),
          ("service", "bytes 4-5", "command 0x0406: memory on, bus master on"),
          ("model", "bytes 9-11", "class 03, subclass 02: 3D controller"),
          ("store", "bytes 16-23", "BAR0 = 0xd0_c000_000c: address 0xd0_c000_0000"),
          ("outside", "byte 52", "capability list starts at offset 0x40"),
          ("danger", "byte 61", "interrupt pin 1 (INTA), switched off: MSI in use")]
for k, (kind, where, what) in enumerate(decode):
    y = 288 + 30 * k
    parts += [box(40, y, 96, 24, kind, where, cls="pill"), text(150, y + 16.5, what, "sub")]
parts += [note(40, 486, "Numbers wider than 1 byte are stored low byte first: de 10 is 0x10de.", room=580)]
save(OUT + "ch23-config.svg", 660, 508,
     "A grid of 64 bytes in 4 rows of 16, read without root from the configuration space of an integrated GPU presented as a PCIe function. Coloured fields: "
     "bytes 0 to 3, de 10 00 2b, vendor 0x10de and device 0x2b00; bytes 4 and 5, 06 04, command 0x0406 with memory "
     "space and bus master enabled; bytes 9 to 11, 00 02 03, class 3D controller; bytes 16 to 23, 0c 00 00 c0 d0 00 00 "
     "00, BAR0 = 0xd0c000000c, address 0xd0c0000000; byte 52, 40, the capability pointer; byte 61, 01, interrupt pin A.",
     "A configuration header, byte by byte", "The first 64 bytes of a PCIe-function GPU, 0000:01:00.0.", parts)


# 23.4  Sizing a BAR: write all ones, read back, keep the address bits, invert, add 1.
steps = [("BAR holds", "0x4000_0004", "0xC000_000C"),
         ("write all ones", "0xFFFF_FFFF", "0xFFFF_FFFF"),
         ("read back", "0xFFFF_C004", "0xFF00_000C"),
         ("clear flag bits 3:0", "0xFFFF_C000", "0xFF00_0000"),
         ("invert", "0x0000_3FFF", "0x00FF_FFFF"),
         ("add 1: the size", "0x0000_4000", "0x0100_0000")]
parts = [zone(200, 104, 210, 300, "NVMe drive"), zone(430, 104, 210, 300, "PCIe-function GPU")]
for k, (what, a, b) in enumerate(steps):
    y = 140 + 40 * k
    last = k == len(steps) - 1
    parts += [text(24, y + 19, f"{k + 1}. {what}", "sub"),
              box(222, y, 166, 28, "good" if last else "outside", a),
              box(452, y, 166, 28, "good" if last else "service", b)]
    if k < len(steps) - 1:
        parts += [arrow(305, y + 28, y + 40), arrow(535, y + 28, y + 40)]
parts += [tag(222, 422, "16,384 bytes = 16 KiB", "good"), tag(452, 422, "16,777,216 bytes = 16 MiB", "good"),
          note(16, 466, "The device wires the address bits below its size to 0, so they read back 0 whatever is", room=628),
          note(16, 484, "written. Counting those zeros gives the size, which is therefore always a power of 2.", room=628),
          note(16, 502, "Bits 3:0 are flags: 0x4 = 64-bit BAR, 0x8 = prefetchable. The kernel does this at boot.", room=628)]
save(OUT + "ch23-bar-size.svg", 660, 524,
     "6 steps in 2 columns, for an NVMe drive's BAR0 and a PCIe-function GPU's BAR0. 1, the BAR holds 0x40000004 and "
     "0xC000000C. 2, software writes 0xFFFFFFFF. 3, it reads back 0xFFFFC004 and 0xFF00000C. 4, the flag bits 3 to 0 "
     "are cleared: 0xFFFFC000 and 0xFF000000. 5, inverted: 0x00003FFF and 0x00FFFFFF. 6, plus 1: 0x4000 = 16,384 bytes "
     "= 16 KiB, and 0x1000000 = 16,777,216 bytes = 16 MiB.",
     "How the kernel learns a BAR's size", "Write all ones; the bits that stay 0 give the size.", parts)


# 23.5  Control path and data path.
parts = [card(24, 130, 180, 70, "client", "CPU core", ["runs the driver", "and your program"], glyph="chip"),
         cylinder(36, 330, 156, 96, "store", "DRAM", ["data, pushbuffers,", "the work ring"]),
         zone(392, 116, 248, 330, "GPU"),
         card(412, 140, 208, 64, "danger", "registers (BAR0)", ["doorbell, status, setup"]),
         card(412, 236, 208, 64, "service", "host interface", ["reads the work ring"]),
         card(412, 332, 208, 74, "model", "copy engines", ["move data in bulk:", "DMA"]),
         wire([(204, 172), (412, 172)], kind="danger", label="MMIO: a 4-byte store"),
         wire([(192, 362), (300, 362), (300, 268), (412, 268)], label="DMA: 8-byte entries", on=1),
         wire([(192, 386), (412, 386)], label="DMA: 64 MiB"),
         wire([(114, 200), (114, 330)], label="cached"),
         note(24, 494, "Red: the control path, a few bytes stored by the CPU. Blue: the data path, read by the GPU.", room=612),
         note(24, 512, "The CPU tells the GPU where work is with 1 small store; the GPU fetches the work and the", room=612),
         note(24, 530, "data itself, by DMA, while the CPU goes on with other things.", room=612)]
save(OUT + "ch23-two-paths.svg", 660, 552,
     "A CPU core at the top left and DRAM below it, joined by a cached path. On the right, a GPU with registers in "
     "BAR0, a host interface and copy engines. A red control path runs from the CPU to the registers: an MMIO store "
     "of 4 bytes, the doorbell. 2 data paths run from DRAM to the GPU: the host interface reads 8-byte work entries by "
     "DMA, and the copy engines move 64 MiB of data by DMA.",
     "Control path and data path", "MMIO carries commands; DMA carries data.", parts)


# 23.6  The work ring: 8 slots, counters that only grow, slot = counter mod 8.
def ring(y, get, put, title, kind):
    """8 slots; each pending entry c sits in slot c mod 8 and is labelled #c."""
    parts = [text(24, y - 14, title, "name")]
    entry = {c % 8: c for c in range(get, put)}
    for s in range(8):
        x = 120 + 56 * s
        label = f"#{entry[s]}" if s in entry else ""
        parts += [box(x, y, 48, 44, kind, label, on=s in entry, cls="pill"),
                  mono(x + 24, y + 62, str(s), "middle", MUTED)]
    return parts


parts = []
parts += ring(144, 2, 7, "GET = 2, PUT = 7", "outside")
parts += [tag(120 + 56 * 2, 120, "GET", "good"), tag(120 + 56 * 7, 120, "PUT", "client"),
          text(24, 236, "pending = PUT − GET = 7 − 2 = 5 entries, #2 to #6, in slots 2 to 6", "sub")]
parts += ring(300, 2, 10, "after 3 more: PUT = 10", "danger")
parts += [tag(120 + 56 * 2, 276, "GET", "good"), tag(120 + 56 * 2, 254, "PUT = 10: slot 2", "client"),
          text(24, 392, "pending = 10 − 2 = 8 = capacity: the ring is full; the next write must wait", "sub"),
          text(24, 412, "entry #c goes to slot c mod 8: #8 → slot 0, #9 → slot 1, and PUT = 10 → slot 2", "sub"),
          note(24, 452, "The host owns PUT and writes entries; the GPU owns GET and reads them. Each side only", room=612),
          note(24, 470, "moves its own counter forward, so neither needs a lock.", room=612)]
save(OUT + "ch23-ring.svg", 660, 492,
     "2 rows of 8 slots numbered 0 to 7. Top: GET = 2 and PUT = 7, so slots 2 to 6 hold 7 minus 2 = 5 pending entries. "
     "Bottom: after 3 more entries PUT = 10; entries 7, 8 and 9 went to slots 7, 0 and 1, because the slot is the "
     "counter mod 8. Now 10 minus 2 = 8 entries are pending, the ring's capacity, so the next write must wait.",
     "A work ring by hand", "8 slots, 2 counters that only grow, slot = counter mod 8.", parts)


# 23.7  A kernel launch, step by step, across the process, memory and the GPU.
lanes = [(90, "your process", "client"), (350, "DRAM", "store"), (610, "GPU", "service")]
P, D, G = 90, 350, 610
steps = [(P, D, "1. write methods (pushbuffer)"),
         (P, D, "2. write 8-byte ring entry"),
         (P, D, "3. GPPut = n + 1 (USERD 0x8c)"),
         (P, P, "4. barrier: earlier writes first"),
         (P, G, "5. store token to the doorbell"),
         (G, D, "6. read ring entry (DMA)"),
         (G, D, "7. read methods (DMA)"),
         (G, G, "8. SMs run the kernel"),
         (G, D, "9. GPGet = n + 1 (USERD 0x88)")]
parts = []
for x, name, kind in lanes:
    wide = width_of(name, 11, bold=True) + 14
    parts += [line(x, 132, x, 132 + 44 * len(steps) + 4, EDGE, 2, "6 5"), tag(x - wide / 2, 108, name, kind)]
for k, (a, b, label) in enumerate(steps):
    y = 168 + 44 * k
    kind = "danger" if k == 4 else "flow"
    colour = KINDS["danger"][0] if kind == "danger" else INK
    if a == b:
        mark = "danger" if k == 3 else "good"
        anchor, at = ("start", a + 14) if a == P else ("end", a - 14)
        parts += [box(a - 4, y - 12, 8, 16, mark, ""), text(at, y, label, "mono", anchor, KINDS[mark][0])]
    else:
        anchor, at = ("start", a + 10) if b > a else ("end", a - 10)
        parts += [hline(a, b, y, kind), text(at, y - 7, label, "mono", anchor, colour)]
parts += [note(24, 572, "Steps 1 to 5 are ordinary stores from the process, with no system call. The store in step", room=612),
          note(24, 590, "5 is the only one that reaches the GPU; it tells the GPU that new entries are waiting.", room=612)]
save(OUT + "ch23-launch.svg", 660, 612,
     "3 vertical lanes: your process, DRAM and the GPU. 1, the process writes methods to the pushbuffer in DRAM. 2, it "
     "writes an 8-byte entry to the ring. 3, it writes GPPut = n + 1 at offset 0x8c of the USERD page. 4, a barrier "
     "orders the earlier writes. 5, it stores a token to the doorbell register of the GPU. 6, the GPU reads the ring "
     "entry by DMA. 7, it reads the methods by DMA. 8, its SMs run the kernel. 9, it writes GPGet = n + 1 at offset "
     "0x88.",
     "A launch, from store to SMs", "What libcuda and the GPU do for 1 kernel launch.", parts)


# 23.8  1 burst of the work-queue program, ring depth 1 against depth 32.
scale = 3.4          # pixels per microsecond
x0 = 120


def bar_us(start, end, y, kind, label="", dashed=False):
    accent, tint = KINDS[kind]
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts = [f'  <rect x="{x0 + start * scale:.1f}" y="{y}" width="{(end - start) * scale:.1f}" height="26" rx="5"'
             f' fill="{"#FFFFFF" if dashed else tint}" stroke="{accent}" stroke-width="1.3"{dash}/>']
    if label:
        parts.append(text(x0 + (start + end) / 2 * scale, y + 17, label, "pill", "middle", accent))
    return "\n".join(parts)


parts = [zone(16, 104, 628, 150, "depth 1: 1 entry at a time"), zone(16, 284, 628, 150, "depth 32: the whole burst")]
for y, depth in ((136, 1), (316, 32)):
    parts += [text(28, y + 17, "host", "sub"), text(28, y + 57, "device", "sub")]
    if depth == 1:
        parts += [bar_us(0, 64, y, "idle", "submit, wait, submit, wait", dashed=True),
                  bar_us(64, 128, y, "client", "own work 64 µs"),
                  bar_us(0, 64, y + 40, "service", "32 entries × 2 µs"),
                  bar_us(64, 128, y + 40, "idle", "idle", dashed=True),
                  tag(x0 + 128 * scale - 120, y + 76, "128 µs per burst", "danger")]
    else:
        parts += [bar_us(0, 2, y, "idle", "", dashed=True),
                  bar_us(2, 66, y, "client", "own work 64 µs"),
                  bar_us(0, 64, y + 40, "service", "32 entries × 2 µs"),
                  tag(x0 + 66 * scale + 10, y + 76, "64 µs per burst", "good")]
parts += [mono(x0, 266, "0", "middle", MUTED), mono(x0 + 64 * scale, 266, "64 µs", "middle", MUTED),
          mono(x0 + 128 * scale, 266, "128 µs", "middle", MUTED),
          note(16, 466, "200 bursts: depth 1 predicts 200 × 128 µs = 25.6 ms (measured 28.7 to 30.9 ms), depth 32", room=628),
          note(16, 484, "predicts 200 × 64 µs = 12.8 ms (measured 13.5 to 13.7 ms). A deep ring lets the host leave", room=628),
          note(16, 502, "the work queued and go on; the GPU's own command rings exist for the same reason.", room=628)]
save(OUT + "ch23-queue.svg", 660, 524,
     "2 timelines of 1 burst, 0 to 128 microseconds. Depth 1: the host submits and waits for 64 microseconds while the "
     "device runs 32 entries of 2 microseconds, then the host does 64 microseconds of its own work while the device is "
     "idle: 128 microseconds per burst. Depth 32: the host writes all 32 entries at once and does its own work while the "
     "device runs them: 64 microseconds per burst. 200 bursts predict 25.6 ms against 12.8 ms; measured 28.7 to 30.9 ms "
     "and 13.5 to 13.7 ms.",
     "Why a deep ring lets the host run ahead", "1 burst of the chapter's work queue, drawn to scale.", parts)


# 23.9  An IOMMU between a device and memory: contiguous device addresses, scattered pages.
parts = [zone(16, 104, 190, 300, "device addresses"), zone(450, 104, 194, 300, "physical pages")]
iova = ["0x1000_0000", "0x1000_1000", "0x1000_2000", "0x1000_3000", "0x2000_0000"]
phys = {0: 3, 1: 0, 2: 5, 3: 1}
slots = ["0x8_4A21_7000", "0x8_1102_3000", "0x9_0C40_1000", "0x8_4A21_6000", "kernel data", "0x8_77F0_2000"]
for k, a in enumerate(iova):
    y = 140 + 50 * k
    bad = k == 4
    parts.append(box(36, y, 150, 34, "danger" if bad else "client", a))
for k, a in enumerate(slots):
    y = 132 + 44 * k
    used = k in phys.values()
    kind = "danger" if a == "kernel data" else ("store" if used else "idle")
    parts.append(box(470, y, 154, 34, kind, a, on=used or a == "kernel data"))
parts += [zone(236, 104, 184, 300, "SMMU page table")]
for k in range(4):
    y = 140 + 50 * k
    ty = 132 + 44 * phys[k] + 17
    parts += [box(256, y, 144, 34, "outside", f"page {k} → slot {phys[k]}", cls="pill"),
              wire([(186, y + 17), (256, y + 17)], arrow=False),
              wire([(400, y + 17), (426 + 9 * k, y + 17), (426 + 9 * k, ty), (470, ty)])]
parts += [box(256, 340, 144, 34, "danger", "no entry: fault", cls="pill"),
          wire([(186, 357), (256, 357)], kind="danger"),
          note(16, 438, "The device sees 4 neighbouring pages; the SMMU maps each to wherever the OS put it. An", room=628),
          note(16, 456, "address with no entry stops at the SMMU with a fault and never reaches the kernel's data.", room=628),
          note(16, 474, "A 64 MiB buffer in 4 KiB pages needs 64 MiB ÷ 4 KiB = 16,384 entries.", room=628)]
save(OUT + "ch23-iommu.svg", 660, 496,
     "3 columns. Device addresses 0x10000000 to 0x10003000 are 4 neighbouring 4 KiB pages; the SMMU page table maps "
     "page 0 to slot 3, page 1 to slot 0, page 2 to slot 5 and page 3 to slot 1 among scattered physical pages. A fifth "
     "device address, 0x20000000, has no entry: the access stops at the SMMU with a fault and never reaches a page of "
     "kernel data. A 64 MiB buffer in 4 KiB pages needs 16,384 entries.",
     "An IOMMU between the GPU and memory", "Device addresses are translated and checked, page by page (example addresses).", parts)


# 23.10  User-mode and kernel-mode drivers, with the names found on the 2 machines.
parts = [zone(16, 104, 628, 150, "user space: your process"),
         card(36, 136, 280, 96, "client", "libcuda.so (UMD)", ["CUDA API, contexts, launches", "builds pushbuffers and ring",
                                                               "entries in mapped memory"], glyph="gear"),
         card(344, 136, 280, 96, "service", "mapped pages", ["ring, pushbuffer, USERD", "doorbell page (MMIO)",
                                                            "no system call to use"]),
         wire([(316, 184), (344, 184)]),
         zone(16, 290, 628, 150, "kernel space"),
         card(36, 322, 280, 96, "outside", "kernel-mode driver", ["platform: nvgpu, nvmap", "PCIe fn: nvidia, nvidia_uvm",
                                                                 "owns BARs, IRQs, SMMU"], glyph="lock"),
         card(344, 322, 280, 96, "store", "device files it opens", ["platform: /dev/nvgpu/igpu0/ctrl,",
                                                                    "/dev/nvmap. PCIe fn: /dev/nvidia0,", "/dev/nvidiactl, /dev/nvidia-uvm"]),
         wire([(260, 232), (260, 322)], label="ioctl: setup only"),
         wire([(484, 322), (484, 232)], kind="idle", label="mmap"),
         note(16, 470, "Allocating memory, creating a channel and mapping pages go through ioctl calls on the", room=628),
         note(16, 488, "device nodes. Submitting work does not: libcuda writes the mapped pages and the doorbell.", room=628)]
save(OUT + "ch23-drivers.svg", 660, 510,
     "2 bands. User space: libcuda.so, the user-mode driver, which implements the CUDA API and builds pushbuffers and "
     "ring entries, and the pages mapped into the process: ring, pushbuffer, USERD and the doorbell page. Kernel space: "
     "the kernel-mode driver, nvgpu and nvmap for a platform-device GPU and nvidia with nvidia_uvm for a PCIe-function GPU, which owns the BARs, "
     "interrupts and SMMU, and the device files a CUDA process opens: /dev/nvgpu/igpu0/ctrl and /dev/nvmap for the first, "
     "/dev/nvidia0, /dev/nvidiactl and /dev/nvidia-uvm for the second. ioctl calls are for setup only; the pages reach the process through mmap.",
     "2 drivers, 1 GPU", "The kernel sets things up; the process submits work by itself.", parts)
