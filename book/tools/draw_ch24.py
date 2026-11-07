"""The figures of chapter 18, drawn with figkit and figparts.

    python3 tools/draw_ch18.py

Link rates are NVIDIA's published figures, quoted each way (half of NVIDIA's 2-way totals).
The measured copies are the runs of code/ch18_peer/peer.cu quoted in the chapter.
"""
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, card, fan, legend, line, mono, note,
                      save, tag, text, wire, zone)
from figkit import stacked

OUT = "src/figures/"

# The measured copy inside 1 GPU, the run quoted in section 24.5.5: (label, memory GB/s, GB/s
# copied by cudaMemcpyPeerAsync).
MEASURED = ("measured GPU", 102, 38.8)


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A small rounded box: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def dashed(x1, y1, x2, y2, kind="danger"):
    return line(x1, y1, x2, y2, KINDS[kind][0], 1.8, "6 5")


# 24.1  1 data-parallel training step on 8 GPUs: compute shrinks, the exchange does not.
x0, scale = 176, 1.1
rows = [("1 GPU", 400, 0, None), ("8 GPUs, PCIe 4.0", 50, 54.7, "3.8 × as fast"),
        ("8 GPUs, NVLink 3", 50, 5.8, "7.2 × as fast")]
parts = [legend(x0, 112, [("client", "compute"), ("danger", "gradient exchange")])]
for ms in range(0, 401, 100):
    x = x0 + ms * scale
    parts += [line(x, 132, x, 336, EDGE, 1), mono(x, 354, f"{ms} ms", "middle", MUTED)]
for k, (name, compute, exchange, gain) in enumerate(rows):
    y = 148 + k * 64
    total = compute + exchange
    parts += [text(x0 - 14, y + 16, name, "name", "end"),
              text(x0 - 14, y + 34, f"{total:g} ms per step", "sub", "end"),
              f'  <rect x="{x0}" y="{y}" width="{compute * scale:.1f}" height="36" rx="6" '
              f'fill="{KINDS["client"][1]}" stroke="{KINDS["client"][0]}" stroke-width="1.3"/>',
              mono(x0 + compute * scale / 2, y + 22, f"{compute} ms", "middle", KINDS["client"][0])]
    if exchange:
        w = max(exchange * scale, 4)
        parts += [f'  <rect x="{x0 + compute * scale:.1f}" y="{y}" width="{w:.1f}" height="36" '
                  f'rx="6" fill="{KINDS["danger"][1]}" stroke="{KINDS["danger"][0]}" '
                  f'stroke-width="1.3"/>',
                  mono(x0 + total * scale + 10, y + 22, f"+ {exchange:g} ms", "start",
                       KINDS["danger"][0]),
                  tag(644, y + 9, gain, "good", "end")]
parts += [note(24, 392, "Each GPU computes on 1/8 of the batch, 400 ÷ 8 = 50 ms, then sends and receives "
               "1.75 GB", room=612),
          note(24, 410, "of gradients. At 32 GB/s each way over PCIe 4.0 that takes 1.75 ÷ 32 = 54.7 ms; "
               "at 300 GB/s", room=612),
          note(24, 428, "over NVLink 3, 1.75 ÷ 300 = 5.8 ms. Speedups: 400 ÷ 104.7 = 3.8 and "
               "400 ÷ 55.8 = 7.2.", room=612)]
save(OUT + "ch24-step.svg", 660, 450,
     "Three timelines of 1 training step. On 1 GPU, 400 ms of compute. On 8 GPUs joined by PCIe 4.0, "
     "50 ms of compute and 54.7 ms of gradient exchange, 104.7 ms, 3.8 times as fast. On 8 GPUs joined "
     "by NVLink 3, 50 ms of compute and 5.8 ms of exchange, 55.8 ms, 7.2 times as fast.",
     "1 training step on 8 GPUs", "The compute shrinks 8 times; the gradient exchange does not.",
     parts)


# 24.2  Inside 1 link, and the 18 links of an H100.
parts = [zone(16, 104, 330, 262, "1 link of NVLink 4"), zone(362, 104, 282, 262, "18 links of 1 H100")]
parts += [box(36, 150, 76, 150, "outside", "GPU A", cls="name"),
          box(250, 150, 76, 150, "outside", "GPU B", cls="name")]
for k, y in enumerate((178, 196)):
    parts.append(wire([(112, y), (250, y)]))
for k, y in enumerate((254, 272)):
    parts.append(wire([(250, y), (112, y)]))
parts += [text(181, 166, "A to B: 2 pairs", "sub", "middle"),
          text(181, 242, "B to A: 2 pairs", "sub", "middle"),
          note(36, 326, "2 pairs × 100 Gbit/s = 200 Gbit/s", room=300),
          note(36, 344, "200 Gbit/s ÷ 8 bits = 25 GB/s each way", room=300)]
for k in range(18):
    r, c = divmod(k, 6)
    parts.append(box(386 + c * 40, 140 + r * 40, 32, 32, "client", f"{k + 1}"))
parts += [note(386, 286, "18 × 25 GB/s = 450 GB/s each way", room=246),
          note(386, 304, "both ways: 2 × 450 = 900 GB/s,", room=246),
          note(386, 322, "the figure NVIDIA quotes", room=246),
          note(386, 344, "PCIe 5.0 × 16: 64 GB/s each way", room=246),
          note(24, 398, "A pair is 2 wires carrying 1 signal in 1 direction. A link has pairs both "
               "ways, so it sends", room=612),
          note(24, 416, "and receives at once. A GPU's bandwidth to other GPUs is its number of links "
               "times 1 link.", room=612)]
save(OUT + "ch24-link.svg", 660, 438,
     "Left: 1 link of NVLink 4 between GPU A and GPU B, drawn as 2 wire pairs from A to B and 2 from B "
     "to A. 2 pairs of 100 Gbit/s carry 200 Gbit/s, 25 GB/s each way. Right: the 18 links of 1 H100, "
     "numbered 1 to 18: 18 times 25 GB/s is 450 GB/s each way, 900 GB/s both ways, against 64 GB/s each "
     "way for PCIe 5.0 with 16 lanes.",
     "Links, and how they add up", "1 link carries 25 GB/s each way; an H100 has 18.", parts)


# 24.3  Link rates each way, and the time to move 1 GB.
RATES = [("PCIe 4.0 × 16", 32, "idle"), ("PCIe 5.0 × 16", 64, "idle"),
         ("NVLink 2, V100", 150, "client"), ("NVLink 3, A100", 300, "client"),
         ("NVLink 4, H100", 450, "client"), ("NVLink-C2C, GH200", 450, "outside"),
         ("NVLink 5, B200", 900, "client"), ("H100 memory, a copy", 1675, "store")]
x0, wide = 196, 250
parts = [legend(x0, 112, [("idle", "PCIe"), ("client", "NVLink, GPU to GPU"),
                          ("outside", "CPU to GPU"), ("store", "local memory")])]
for k, (name, rate, kind) in enumerate(RATES):
    y = 136 + k * 40
    ms = 1000 / rate
    took = f"{ms:.1f} ms" if ms < 10 else f"{ms:.0f} ms"
    w = max(wide * rate / 1675, 3)
    parts += [text(x0 - 14, y + 17, name, "sub", "end"),
              f'  <rect x="{x0}" y="{y + 4}" width="{w:.1f}" height="20" rx="5" '
              f'fill="{KINDS[kind][1]}" stroke="{KINDS[kind][0]}" stroke-width="1.3"/>',
              mono(x0 + w + 8, y + 18, f"{rate:,} GB/s", "start", INK),
              mono(644, y + 18, f"1 GB in {took}", "end", KINDS[kind][0])]
parts += [note(24, 474, "Each way, from NVIDIA's figures. A copy inside an H100 reads and writes every "
               "byte, so its", room=612),
          note(24, 492, "3,350 GB/s of memory copy at 3,350 ÷ 2 = 1,675 GB/s. 1 GB ÷ 450 GB/s = 2.2 ms.",
               room=612)]
save(OUT + "ch24-rates.svg", 660, 514,
     "Horizontal bars of bandwidth each way, with the time to move 1 GB. PCIe 4.0 with 16 lanes 32 GB/s, "
     "31 ms. PCIe 5.0 64 GB/s, 16 ms. NVLink 2 on V100 150 GB/s, 6.7 ms. NVLink 3 on A100 300 GB/s, "
     "3.3 ms. NVLink 4 on H100 450 GB/s, 2.2 ms. NVLink-C2C on GH200 450 GB/s, 2.2 ms. NVLink 5 on B200 "
     "900 GB/s, 1.1 ms. A copy inside an H100's own memory 1,675 GB/s, 0.6 ms.",
     "Link rates, each way", "The time to move a 1 GB gradient over each link.", parts)


# 24.4  A full mesh: every pair has its own link.
parts = [zone(16, 104, 250, 330, "4 GPUs: 6 links"), zone(282, 104, 362, 330, "8 GPUs: 28 links")]
corners = [(66, 160), (186, 160), (186, 296), (66, 296)]
for a in range(4):
    for b in range(a + 1, 4):
        (xa, ya), (xb, yb) = corners[a], corners[b]
        parts.append(line(xa + 20, ya + 15, xb + 20, yb + 15, WIRE, 2.2))
for k, (x, y) in enumerate(corners):
    parts.append(box(x, y, 40, 30, "outside", f"{k}"))
parts += [note(40, 366, "each GPU: 3 links", room=210),
          note(40, 384, "4 × 3 ÷ 2 = 6 links in all", room=210),
          note(40, 402, "(÷ 2: each link has 2 ends)", room=210)]
cell, gx, gy = 30, 336, 148
for i in range(8):
    parts += [mono(gx + i * cell + 13, gy - 8, f"{i}", "middle", MUTED),
              mono(gx - 12, gy + i * cell + 19, f"{i}", "middle", MUTED)]
    for j in range(8):
        if i == j:
            parts.append(box(gx + j * cell, gy + i * cell, cell - 4, cell - 4, "idle", "×", cls="pill"))
        else:
            parts.append(box(gx + j * cell, gy + i * cell, cell - 4, cell - 4, "client", on=i < j))
parts += [note(306, 412, "1 link per pair: 8 × 7 ÷ 2 = 28 links, 7 per GPU", room=330)]
save(OUT + "ch24-mesh.svg", 660, 456,
     "Left: 4 GPUs at the corners of a square, joined by 4 sides and 2 diagonals, 6 links, 3 per GPU. "
     "Right: an 8 by 8 table of GPU pairs. The diagonal is crossed out, the upper half is shaded: 28 "
     "cells, 1 link for each pair, 7 links at each GPU.",
     "A full mesh", "With a link for every pair, links grow as N(N − 1) ÷ 2.", parts)


# 24.5  8 H100s: a direct mesh against 4 NVSwitch chips.
parts = [zone(16, 104, 628, 196, "direct: 18 links shared among 7 peers"),
         zone(16, 322, 628, 236, "switched: 4 NVSwitch chips")]
targets = [76 + k * 82 for k in range(7)]
parts += [box(290, 130, 80, 32, "outside", "GPU 0", cls="name"),
          fan(330, 162, targets, 186, 214)]
for k, x in enumerate(targets):
    parts += [box(x - 32, 214, 64, 30, "outside", f"GPU {k + 1}", cls="pill"),
              tag(x - 26, 252, "2 links", "client")]
parts += [note(400, 146, "18 ÷ 7 = 2 per peer, 4 left", room=230),
          note(40, 290, "to any 1 peer: 2 × 25 = 50 GB/s each way", room=400)]
gpus = [56 + k * 76 for k in range(8)]
for k, x in enumerate(gpus):
    parts += [box(x - 30, 350, 60, 30, "outside", f"GPU {k}", cls="pill"),
              arrow(x, 380, 450)]
parts.append(tag(56 - 30, 400, "18 links", "client"))
for s in range(4):
    x = 36 + s * 152
    parts.append(f'  <rect x="{x}" y="450" width="140" height="40" rx="8" fill="#2B3442" '
                 f'stroke="#1B2430" stroke-width="1.3" filter="url(#lift)"/>')
    parts.append(text(x + 70, 475, f"NVSwitch {s}", "pill", "middle", "#DCE3EC"))
parts += [note(40, 516, "to any 1 peer: all 18 links, 450 GB/s each way, and all 8 GPUs at once", room=600),
          note(40, 534, "8 GPUs × 18 links = 144 of the 4 × 64 = 256 switch ports", room=600)]
save(OUT + "ch24-switch.svg", 660, 580,
     "Top: 8 H100s wired directly. GPU 0 fans out to GPUs 1 to 7, each with a tag of 2 links: 18 links "
     "over 7 peers give 2 links per peer, 50 GB/s each way. Bottom: 8 GPUs, each with 18 links going "
     "down into a row of 4 NVSwitch chips. Any pair can use all 18 links, 450 GB/s each way, and all 8 "
     "GPUs can do so at once; 144 of the 256 switch ports are used.",
     "A mesh or a switch, for 8 H100s", "The switch gives every pair the whole 450 GB/s each way.",
     parts)


# 24.6  Bisection bandwidth: cut the 8 GPUs in half and add up what crosses.
parts = [zone(16, 104, 196, 300, "ring"), zone(232, 104, 196, 300, "mesh"),
         zone(448, 104, 196, 300, "switch")]
xs = [36, 78, 130, 172]
for k, x in enumerate(xs):
    parts += [box(x, 150, 24, 22, "outside", f"{k}", cls="pill"),
              box(x, 262, 24, 22, "outside", f"{7 - k}", cls="pill")]
for a, b in zip(xs, xs[1:]):
    parts += [line(a + 24, 161, b, 161, WIRE, 2.2), line(a + 24, 273, b, 273, WIRE, 2.2)]
parts += [line(48, 172, 48, 262, WIRE, 2.2), line(184, 172, 184, 262, WIRE, 2.2),
          dashed(117, 132, 117, 300),
          note(32, 330, "2 edges cross the cut", room=170),
          note(32, 348, "9 links each: 9 × 25 =", room=170),
          note(32, 366, "225 GB/s; 2 × 225 =", room=170),
          tag(32, 378, "450 GB/s", "danger")]
for k in range(4):
    parts += [box(250, 140 + k * 36, 30, 24, "outside", f"{k}", cls="pill"),
              box(380, 140 + k * 36, 30, 24, "outside", f"{k + 4}", cls="pill")]
    for j in range(4):
        parts.append(box(296 + j * 20, 144 + k * 36, 15, 15, "client"))
parts += [dashed(330, 128, 330, 288),
          note(248, 330, "4 × 4 = 16 pairs cross", room=170),
          note(248, 348, "2 links each: 16 × 2 × 25", room=170),
          note(248, 366, "= 800 GB/s", room=170),
          tag(248, 378, "800 GB/s", "client")]
for k in range(4):
    parts += [box(464, 140 + k * 36, 30, 24, "outside", f"{k}", cls="pill"),
              box(598, 140 + k * 36, 30, 24, "outside", f"{k + 4}", cls="pill"),
              line(494, 152 + k * 36, 518, 152 + k * 36, WIRE, 2.2),
              line(574, 152 + k * 36, 598, 152 + k * 36, WIRE, 2.2)]
parts += [f'  <rect x="518" y="136" width="56" height="140" rx="8" fill="#2B3442" stroke="#1B2430" '
          f'stroke-width="1.3"/>',
          text(546, 210, "NVS", "pill", "middle", "#DCE3EC"),
          dashed(546, 126, 546, 290),
          note(464, 330, "4 GPUs send at once", room=170),
          note(464, 348, "4 × 450 GB/s", room=170),
          note(464, 366, "= 1,800 GB/s", room=170),
          tag(464, 378, "1,800 GB/s", "good")]
parts += [note(24, 436, "Every GPU has 18 links of 25 GB/s each way. The dashed line splits the "
               "GPUs into 2 halves", room=612),
          note(24, 454, "of 4; the bisection bandwidth is all that can cross it at once, each way.",
               room=612)]
save(OUT + "ch24-bisection.svg", 660, 476,
     "Three ways to join 8 GPUs that each have 18 links, cut in half by a dashed line. Ring: 2 edges of "
     "9 links cross the cut, 450 GB/s each way. Mesh: 16 pairs of 2 links cross, 800 GB/s. Switch: all "
     "4 GPUs on one side send at 450 GB/s through the switch, 1,800 GB/s.",
     "Bisection bandwidth", "Cut the GPUs into 2 halves; count what crosses, each way.", parts)


# 24.7  3 paths from GPU 0 to GPU 1.
parts = []
rows = [("staged through the host", [("GPU 0", "outside"), ("host memory", "store"),
                                     ("GPU 1", "outside")], ["PCIe", "PCIe"]),
        ("PCIe peer to peer", [("GPU 0", "outside"), ("PCIe switch", "idle"), ("GPU 1", "outside")],
         ["PCIe", "PCIe"]),
        ("NVLink", [("GPU 0", "outside"), ("GPU 1", "outside")], ["NVLink"])]
for k, (name, stops, links) in enumerate(rows):
    y = 116 + k * 112
    parts.append(zone(16, y, 628, 92, name))
    xs = [40, 270, 500] if len(stops) == 3 else [40, 500]
    for (label, kind), x in zip(stops, xs):
        parts.append(box(x, y + 30, 120, 40, kind, label, cls="name"))
    for a, b, body in zip(xs, xs[1:], links):
        parts.append(wire([(a + 120, y + 50), (b, y + 50)], label=body))
parts += [note(24, 456, "Staged: 2 copies, 2 crossings of PCIe, and the host memory carries the data "
               "twice. Peer", room=612),
          note(24, 474, "to peer: 1 crossing, no host memory. NVLink: 1 direct link, 7 times as fast as "
               "PCIe 5.0.", room=612)]
save(OUT + "ch24-paths.svg", 660, 496,
     "Three rows. Staged through the host: GPU 0 to host memory over PCIe, then host memory to GPU 1 "
     "over PCIe. PCIe peer to peer: GPU 0 to a PCIe switch to GPU 1, without host memory. NVLink: GPU 0 "
     "straight to GPU 1 over NVLink.",
     "3 paths from GPU 0 to GPU 1", "The same cudaMemcpyPeer call takes whichever path exists.", parts)


# 24.8  The measured copy inside 1 GPU: each copied byte is read once and written once.
board, memory, copied = MEASURED
traffic = 2 * copied
x0, wide = 176, 440
parts = [legend(x0, 112, [("client", "read"), ("outside", "written"), ("idle", "unused")]),
         text(160, 158, board, "name", "end"),
         text(160, 176, f"{memory} GB/s memory", "sub", "end"),
         stacked(x0, 140, wide, 44, [(copied, KINDS["client"][0], f"{copied:g} GB/s"),
                                     (copied, KINDS["outside"][0], f"{copied:g} GB/s"),
                                     (memory - traffic, "#E4E8EE", "")], total=memory)]
for gb in range(0, memory + 1, 25):
    x = x0 + wide * gb / memory
    parts += [line(x, 190, x, 196, MUTED, 1), mono(x, 212, f"{gb} GB/s", "middle", MUTED)]
parts += [mono(x0, 244, f"copied: 134.2 MB ÷ 3.462 ms = {copied:g} GB/s", "start", INK),
          mono(x0, 264, f"moved:  2 × {copied:g} = {traffic:g} GB/s, {traffic / memory:.0%} of "
               f"{memory} GB/s", "start", INK),
          note(24, 304, "cudaMemcpyPeerAsync of 128 MiB inside 1 GPU. The bar is the GPU's memory "
               "bandwidth; a copy", room=612),
          note(24, 322, "spends it twice, once to read each byte and once to write it.", room=612)]
save(OUT + "ch24-measured.svg", 660, 344,
     f"A bar as long as the GPU's memory bandwidth, {memory} GB/s. The copy ran at "
     f"{copied:g} GB/s, so {copied:g} GB/s were read and {copied:g} GB/s written: {traffic:g} GB/s, "
     f"{traffic / memory:.0%} of the bandwidth. The rest of the bar is unused.",
     "1 copy inside 1 GPU, measured", "The rate that no link to a peer exceeds.", parts)


# 24.9  Reading nvidia-smi topo -m: an example 4-GPU server.
MATRIX = [["X", "NV4", "PIX", "SYS"], ["NV4", "X", "SYS", "PIX"], ["PIX", "SYS", "X", "NV4"],
          ["SYS", "PIX", "NV4", "X"]]
KIND = {"X": "idle", "NV4": "good", "PIX": "client", "SYS": "danger"}
cell, gx, gy = 62, 100, 150
parts = []
for i in range(4):
    parts += [mono(gx + i * cell + 29, gy - 10, f"GPU{i}", "middle", MUTED),
              mono(gx - 14, gy + i * cell + 33, f"GPU{i}", "end", MUTED)]
    for j in range(4):
        body = MATRIX[i][j]
        parts.append(box(gx + j * cell, gy + i * cell, cell - 6, cell - 6, KIND[body], body))
meanings = [("good", "NV4", "4 NVLinks join the pair"), ("client", "PIX", "1 PCIe switch between them"),
            ("danger", "SYS", "PCIe, then across CPU sockets"), ("idle", "X", "the GPU itself")]
for k, (kind, body, meaning) in enumerate(meanings):
    y = 156 + k * 58
    parts += [box(380, y, 56, 36, kind, body), text(448, y + 15, meaning, "sub"),
              text(448, y + 32, ("the fastest path", "usually peer to peer", "the slowest path",
                                 "")[k], "note")]
parts += [note(24, 418, "An example of the matrix's form, for a 2-socket server whose GPUs 0, 1 and 2, 3 "
               "are joined", room=612),
          note(24, 436, "by NVLink. Read a row across: from GPU0, GPU1 is 1 NVLink hop, GPU2 is under the "
               "same", room=612),
          note(24, 454, "PCIe switch, and GPU3 is on the other CPU socket.", room=612)]
save(OUT + "ch24-topo.svg", 660, 476,
     "A 4 by 4 matrix like the one nvidia-smi topo -m prints, for an example server. The diagonal reads "
     "X. GPU0 and GPU1, and GPU2 and GPU3, read NV4: 4 NVLinks. GPU0 and GPU2, and GPU1 and GPU3, read "
     "PIX: 1 PCIe switch. GPU0 and GPU3, and GPU1 and GPU2, read SYS: across the CPU sockets. A legend "
     "explains each code.",
     "Reading a topology matrix", "Each cell names the path between a row GPU and a column GPU.",
     parts)
