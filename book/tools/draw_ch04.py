"""The figures of chapter 4, drawn with figkit and figparts.

    python3 tools/draw_ch04.py

The copy speeds in figure 4.3 are the ones measured on a Jetson Orin NX with
code/ch04_memory/transfers.cu (the run quoted in section 4.4).
"""
from figkit import swatches
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, bar, card, cylinder, legend, line,
                      mascot, mono, note, save, tag, text, tile, wire, zone)

OUT = "src/figures/"


# 4.1  2 kinds of system: discrete and integrated.
parts = [zone(16, 104, 628, 166, "discrete: a GPU card"),
         card(36, 136, 170, 56, "client", "CPU", glyph="terminal"),
         cylinder(46, 204, 150, 58, "store", "host memory"),
         line(121, 192, 121, 204, WIRE, 1.8),
         card(454, 136, 170, 56, "service", "GPU", glyph="chip"),
         cylinder(464, 204, 150, 58, "outside", "global memory", "1 to 3 TB/s"),
         line(539, 192, 539, 204, WIRE, 1.8),
         wire([(206, 164), (454, 164)], label="PCIe link, ~32 GB/s"),
         zone(16, 306, 628, 156, "integrated: Jetson, DGX Spark"),
         card(36, 338, 170, 56, "client", "CPU", glyph="terminal"),
         card(454, 338, 170, 56, "service", "GPU", glyph="chip"),
         cylinder(230, 392, 200, 56, "store", "1 shared memory", "102 GB/s on Orin NX"),
         wire([(121, 394), (121, 420), (230, 420)]),
         wire([(539, 394), (539, 420), (430, 420)])]
save(OUT + "ch04-paths.svg", 660, 484,
     "Two kinds of system. Discrete: a CPU with host memory, and a GPU card with its own global memory "
     "of 1 to 3 TB/s; data between them crosses a PCIe link of about 32 GB/s. Integrated, as on Jetson "
     "boards and DGX Spark: the CPU and the GPU both use 1 shared memory, 102 GB/s on the Orin NX.",
     "Where the bytes travel", "A discrete GPU has its own memory across a link; an integrated GPU shares the CPU's.",
     parts)


# 4.2  A pageable copy goes through a pinned staging buffer; a pinned copy does not.
parts = [zone(16, 104, 628, 128, "copy from pageable memory: 2 copies"),
         card(36, 136, 180, 56, "store", "pageable array", "can be moved"),
         card(244, 136, 180, 56, "outside", "staging buffer", "pinned"),
         card(452, 136, 180, 56, "service", "device memory"),
         wire([(216, 164), (244, 164)]), wire([(424, 164), (452, 164)]),
         note(196, 214, "CPU copies", fill=MUTED), note(400, 214, "DMA engine", fill=MUTED),
         zone(16, 268, 628, 128, "copy from pinned memory: 1 copy"),
         card(36, 300, 180, 56, "outside", "pinned array", "stays in place"),
         card(452, 300, 180, 56, "service", "device memory"),
         wire([(216, 328), (452, 328)], label="DMA engine"),
         note(16, 428, "The DMA engine copies by physical address, so it needs pages that cannot move.", room=628)]
save(OUT + "ch04-staging.svg", 660, 450,
     "Two rows. A copy from pageable memory makes 2 copies: the CPU copies the pageable array into a "
     "pinned staging buffer, then the DMA engine copies the staging buffer to device memory. A copy from "
     "pinned memory makes 1: the DMA engine copies the pinned array straight to device memory.",
     "Why pinned copies are faster", "Pageable data takes a detour through a pinned staging buffer.", parts)


# 4.3  Measured copy speeds of 64 MiB on a Jetson Orin NX.
rows = [("pageable host → device", 5.7, "store"), ("pinned host → device", 19.8, "outside"),
        ("device → pageable host", 7.8, "store"), ("device → pinned host", 16.3, "outside")]
parts = []
x0, scale = 230, 360 / 25
for k, (label, gbs, kind) in enumerate(rows):
    y = 116 + k * 52
    parts += [text(x0 - 14, y + 20, label, "sub", "end"),
              f'  <rect x="{x0}" y="{y}" width="{gbs * scale:.1f}" height="30" rx="6" fill="{KINDS[kind][0]}"/>',
              text(x0 + gbs * scale + 10, y + 20, f"{gbs} GB/s", "name")]
for g in (0, 5, 10, 15, 20, 25):
    parts += [line(x0 + g * scale, 104, x0 + g * scale, 116 + 4 * 52, EDGE, 1),
              mono(x0 + g * scale, 116 + 4 * 52 + 18, str(g), "middle", MUTED)]
parts += [text(x0, 116 + 4 * 52 + 40, "GB/s", "sub"),
          card(40, 380, 470, 56, "outside", "pinned copies are 2 to 3.5 times as fast", "same machine, same 64 MiB"),
          mascot(526, 372, 80, "happy")]
save(OUT + "ch04-bandwidth.svg", 660, 458,
     "Bars of copy speed for 64 MiB on a Jetson Orin NX: pageable host to device 5.7 GB/s, pinned host to "
     "device 19.8 GB/s, device to pageable host 7.8 GB/s, device to pinned host 16.3 GB/s. Pinned copies are "
     "about 3 times as fast.",
     "Copy speed, measured on 1 machine", "A Jetson Orin NX: 64 MiB, average of 20 copies, by transfers.cu.", parts)


# 4.5 and 4.6  3 ways data reaches a kernel.
rows = [("explicit copy", "cudaMemcpy before the launch", "once, in bulk", "client"),
        ("unified memory", "pages move on first use", "once per page", "service"),
        ("zero-copy", "the kernel reads host memory", "on every access", "danger")]
parts = [text(330, 116, "when the data moves", "sub", "middle")]
for k, (name, sub, when, kind) in enumerate(rows):
    y = 130 + k * 84
    parts += [card(40, y, 300, 62, kind, name, sub, glyph={"client": "swap", "service": "stack", "danger": "eye"}[kind]),
              wire([(340, y + 31), (430, y + 31)]),
              tag(440, y + 22, when, kind)]
parts.append(note(40, 400, "On a discrete GPU each crossing uses the PCIe link; on an integrated one it", room=600))
parts.append(note(40, 418, "stays in the shared memory, so the 3 differ much less.", room=600))
save(OUT + "ch04-three-ways.svg", 660, 440,
     "Three rows. Explicit copy: cudaMemcpy before the launch, data moves once, in bulk. Unified memory: "
     "pages move on first use, once per page. Zero-copy: the kernel reads host memory, data moves on every "
     "access. On a discrete GPU each crossing uses the PCIe link; on an integrated GPU the 3 differ much less.",
     "3 ways data reaches a kernel", "They differ in when the bytes move.", parts)


# 4.8  Double buffering: copy chunk k + 1 while computing chunk k.
gx, w = 150, 92
parts = [text(gx - 14, 136, "copy", "name", "end"), text(gx - 14, 196, "compute", "name", "end")]
for k in range(4):
    parts.append(bar(gx + k * (w + 6), 116, w, 30, "store", f"copy {k}"))
    parts.append(bar(gx + (k + 1) * (w + 6), 176, w, 30, "service", f"kernel {k}"))
parts += [note(gx, 252, "While kernel k runs, chunk k + 1 is already being copied, so after the", room=480),
          note(gx, 270, "first copy the copies cost no extra time. Chapter 6 builds this.", room=480)]
save(OUT + "ch04-double-buffer.svg", 660, 292,
     "Two rows of a timeline. Copy row: copy 0, copy 1, copy 2, copy 3. Compute row, shifted by 1 step: "
     "kernel 0 runs while copy 1 happens, kernel 1 while copy 2, and so on.",
     "Double buffering", "Copies of the next chunk overlap the kernel on the current chunk.", parts)


# 4.3  The cost of a copy: a fixed latency, then bandwidth. Points measured by transfers.cu.
import math
sizes_kib = [4, 16, 64, 256, 1024, 4096, 16384, 65536]
times_us = [18.6, 16.9, 21.8, 25.1, 74.0, 278.9, 616.5, 2139.7]
t0, bw = 16.0, 30e9                                  # model: 16 us + bytes / 30 GB/s
px0, px1, py0, py1 = 100, 600, 380, 110
lx = lambda kib: px0 + (px1 - px0) * (math.log2(kib) - 2) / 14         # 4 KiB .. 64 MiB
ly = lambda us: py0 - (py0 - py1) * (math.log10(us) - 1) / 3            # 10 us .. 10 ms
cut = t0 * 1e-6 * bw / 1024                          # break-even size in KiB
parts = [f'  <rect x="{px0}" y="{py1}" width="{px1 - px0}" height="{py0 - py1}" fill="#FFFFFF" stroke="{EDGE}"/>',
         f'  <rect x="{px0}" y="{py1}" width="{lx(cut) - px0:.1f}" height="{py0 - py1}" fill="{KINDS["danger"][1]}"/>']
for kib, label in ((4, "4 KiB"), (64, "64 KiB"), (1024, "1 MiB"), (16384, "16 MiB"), (65536, "64 MiB")):
    parts += [line(lx(kib), py0, lx(kib), py1, EDGE, 1), mono(lx(kib), py0 + 18, label, "middle", MUTED)]
for us, label in ((10, "10 µs"), (100, "100 µs"), (1000, "1 ms"), (10000, "10 ms")):
    parts += [line(px0, ly(us), px1, ly(us), EDGE, 1), mono(px0 - 8, ly(us) + 4, label, "end", MUTED)]
model = " ".join(f"{lx(2 ** (2 + i / 10)):.1f},{ly(t0 + 2 ** (2 + i / 10) * 1024 / bw * 1e6):.1f}" for i in range(141))
parts.append(f'  <polyline points="{model}" fill="none" stroke="{INK}" stroke-width="2.4"/>')
for kib, us in zip(sizes_kib, times_us):
    parts.append(f'  <circle cx="{lx(kib):.1f}" cy="{ly(us):.1f}" r="5.5" fill="{KINDS["outside"][0]}" stroke="#FFFFFF" stroke-width="2"/>')
parts += [text(px0 + 12, py1 + 22, "latency-bound", "name"), text(px0 + 12, py1 + 40, "every copy costs ~16 µs", "sub"),
          text(px1 - 12, py1 + 22, "bandwidth-bound", "name", "end"), text(px1 - 12, py1 + 40, "time grows with size", "sub", "end"),
          line(lx(cut), py0, lx(cut), py1, KINDS["danger"][0], 1.6, "6 5"),
          tag(lx(cut) - 8, py1 + 64, "break-even: 16 µs × 30 GB/s ≈ 480 KB", "danger", "end"),
          text((px0 + px1) / 2, py0 + 40, "copy size", "sub", "middle"),
          legend(px0, py0 + 64, [("outside", "measured")]),
          line(px0 + 130, py0 + 59, px0 + 160, py0 + 59, INK, 2.4),
          text(px0 + 168, py0 + 65, "model: 16 µs + size ÷ 30 GB/s", "note")]
save(OUT + "ch04-cost-model.svg", 660, 470,
     "Copy time against copy size on logarithmic axes, for pinned host-to-device copies. Measured points stay "
     "near 16 microseconds from 4 KiB to 64 KiB, then rise with size to about 2 ms at 64 MiB. A line shows the "
     "model 16 microseconds plus size divided by 30 GB/s. A dashed line at about 480 KB, where the 2 terms are "
     "equal, separates the latency-bound region on the left from the bandwidth-bound region on the right.",
     "The cost of a copy: latency, then bandwidth", "Pinned host → device copies of 4 KiB to 64 MiB, measured by transfers.cu.",
     parts)


# 4.1  A PCIe link is a bundle of lanes.
parts = [card(40, 110, 170, 60, "client", "CPU", "host memory", glyph="terminal"),
         card(450, 110, 170, 60, "service", "GPU card", "global memory", glyph="chip")]
for k in range(16):
    y = 100 + k * 5
    parts.append(line(210, y, 450, y, "#7FB5F0" if k % 2 else "#3b6fd4", 2))
parts += [tag(260, 186, "16 lanes, each about 2 GB/s per direction", "client"),
          card(40, 230, 580, 58, "client", "PCIe 4.0 × 16: 16 × 1.97 GB/s ≈ 32 GB/s each way",
               "PCIe 5.0 doubles the rate per lane: about 64 GB/s each way")]
save(OUT + "ch04-pcie.svg", 660, 312,
     "A CPU card and a GPU card joined by 16 parallel lines, the lanes of a PCIe link. Each lane carries "
     "about 2 GB/s in each direction, so PCIe 4.0 with 16 lanes carries about 32 GB/s each way, and PCIe 5.0 "
     "about 64 GB/s.",
     "A PCIe link is a bundle of lanes", "Bandwidth = lanes × rate per lane.", parts)

print("wrote chapter 4 figures")
