"""The figures of chapter 15, drawn with figkit and figparts.

    python3 tools/draw_ch15.py

The rates in figure 15.7 are the ones printed by code/oxide/ch15_dot_product (the runs quoted
in the chapter).
"""
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, bar, card, legend, line, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"


def elbow(x1, y1, x2, y2, colour=WIRE, mid=None):
    """A connector from (x1, y1) down to (x2, y2): down, across at `mid`, down, with an arrow."""
    mid = (y1 + y2) / 2 if mid is None else mid
    return (f'  <path d="M{x1:.1f},{y1:.1f} V{mid:.1f} H{x2:.1f} V{y2 - 3:.1f}" fill="none" '
            f'stroke="{colour}" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#head)"/>')


def cells(x, y, values, kind, w=44, gap=8, kinds=None):
    """A row of value boxes; returns the parts and the centre of each box."""
    parts, centres = [], []
    for k, v in enumerate(values):
        left = x + k * (w + gap)
        parts.append(bar(left, y, w, 28, kinds[k] if kinds else kind, str(v)))
        centres.append(left + w / 2)
    return parts, centres


# 15.1  The dot product of 2 vectors of 4 elements, by hand.
a, b = [1, 2, 3, 4], [2, 0, 1, 3]
prod = [x * y for x, y in zip(a, b)]
parts = [zone(16, 104, 628, 150, "map: 1 product per element"),
         zone(16, 282, 628, 186, "reduce: a tree of additions")]
x0 = 330
row_a, ca = cells(x0, 128, a, "client")
row_b, _ = cells(x0, 164, b, "client")
row_p, cp = cells(x0, 212, prod, "store")
parts += row_a + row_b + row_p
parts += [mono(x0 - 14, 147, "a", "end", MUTED), mono(x0 - 14, 183, "b", "end", MUTED),
          mono(x0 - 14, 231, "a[i] × b[i]", "end", MUTED)]
for k in range(4):
    parts.append(mono(ca[k], 205, "↓", "middle", MUTED))
pairs, cs = cells(x0 + 26, 318, [prod[0] + prod[1], prod[2] + prod[3]], "service", w=96, gap=24)
parts += pairs
for k in range(2):
    for child in (2 * k, 2 * k + 1):
        parts.append(elbow(cp[child], 242, cs[k], 318))
top, ct = cells(x0 + 86, 404, [sum(prod)], "good", w=96)
parts += top
for k in range(2):
    parts.append(elbow(cs[k], 348, ct[0], 404))
parts += [mono(x0 - 14, 337, "step 1", "end", MUTED), mono(x0 - 14, 423, "step 2", "end", MUTED),
          note(40, 148, "4 multiplications,", room=200), note(40, 166, "all independent", room=200),
          note(40, 372, "3 additions in 2 steps:", room=200),
          note(40, 390, "2 + 0 = 2, 3 + 12 = 15,", room=200), note(40, 408, "2 + 15 = 17", room=200),
          note(16, 494, "The products need no communication between threads. The additions do: every", room=628),
          note(16, 512, "product must reach 1 place, which is the reduction of Chapter 8.", room=628)]
save(OUT + "ch15-dot.svg", 660, 534,
     "The vectors a = 1, 2, 3, 4 and b = 2, 0, 1, 3. The map multiplies them element by element into "
     "2, 0, 3, 12. The reduction adds the products in pairs, 2 + 0 = 2 and 3 + 12 = 15, and then 2 + 15 = 17.",
     "A dot product by hand", "a · b = 1 × 2 + 2 × 0 + 3 × 1 + 4 × 3 = 17.", parts)


# 15.2  Bytes per pair: map, then reduce, against the fused kernel.
parts = [zone(16, 104, 300, 300, "map, then reduce"), zone(344, 104, 300, 300, "fused")]
steps_a = [("read a[i]", "client"), ("read b[i]", "client"), ("write p[i]", "danger"),
           ("read p[i]", "danger")]
for k, (name, kind) in enumerate(steps_a):
    parts += [bar(40, 136 + k * 46, 150, 34, kind, name), mono(206, 158 + k * 46, "4 B", "start", MUTED)]
for k, (name, kind) in enumerate(steps_a[:2]):
    parts += [bar(368, 136 + k * 46, 150, 34, kind, name), mono(534, 158 + k * 46, "4 B", "start", MUTED)]
parts += [line(40, 330, 290, 330, EDGE, 1), line(368, 330, 618, 330, EDGE, 1),
          text(40, 356, "16 B per pair", "name"), text(368, 356, "8 B per pair", "name"),
          mono(40, 382, "N = 16,777,216: 268 MB", "start", MUTED),
          mono(368, 382, "N = 16,777,216: 134 MB", "start", MUTED),
          legend(40, 438, [("client", "input, read once"), ("danger", "product array, written and read")]),
          note(16, 476, "The product array p exists only to carry values from the map to the reduce. A", room=628),
          note(16, 494, "kernel that adds each product as soon as it is computed never stores it.", room=628)]
save(OUT + "ch15-traffic.svg", 660, 516,
     "Left, map then reduce: per pair of elements the GPU reads a[i] and b[i], writes p[i] and reads p[i] "
     "back, 4 bytes each, 16 bytes per pair, 268 MB for 16,777,216 pairs. Right, the fused kernel reads only "
     "a[i] and b[i]: 8 bytes per pair, 134 MB.",
     "Memory traffic of the 2 designs", "Bytes moved per pair of elements, with 4-byte floats.", parts)


# 15.3  A static SharedArray: 1 copy per block, visible to the threads of that block only.
parts = [card(150, 100, 360, 60, "outside", "static mut CELLS: SharedArray<f32, 256>",
              "declared once, in the kernel")]
for k, name in enumerate(["block 0", "block 1", "block 2"]):
    x = 24 + k * 210
    parts += [zone(x, 214, 192, 196, f"{name} on an SM"),
              bar(x + 16, 238, 160, 34, "store", "CELLS: 256 floats"),
              mono(x + 96, 300, "threads 0 to 255", "middle", MUTED)]
    for t in range(8):
        parts.append(f'  <circle cx="{x + 30 + t * 19}" cy="{324}" r="6" fill="{KINDS["client"][1]}" '
                     f'stroke="{KINDS["client"][0]}" stroke-width="1.2"/>')
    parts += [mono(x + 96, 360, "1 KiB of the SM's", "middle", MUTED),
              mono(x + 96, 378, "shared memory", "middle", MUTED)]
parts += [wire([(330, 160), (330, 186), (120, 186), (120, 210)]), wire([(330, 186), (330, 210)]),
          wire([(330, 186), (540, 186), (540, 210)]),
          note(16, 440, "Each resident block gets its own copy, created when the block starts and freed", room=628),
          note(16, 458, "when it ends. A thread can read and write the copy of its own block only.", room=628)]
save(OUT + "ch15-shared.svg", 660, 480,
     "1 declaration, static mut CELLS of type SharedArray of 256 floats, at the top. Arrows lead to 3 blocks "
     "on SMs, each holding its own copy of CELLS, 1 KiB of shared memory, used by its threads 0 to 255.",
     "1 shared array per block", "A static SharedArray in a kernel is placed in shared memory.", parts)


# 15.4  The tree in shared memory: 8 cells, half = 4, 2, 1, a barrier after every step.
rows = [[2, 0, 3, 12, 5, 1, 4, 6]]
for half in (4, 2, 1):
    prev = rows[-1][:]
    rows.append([prev[t] + prev[t + half] if t < half else prev[t] for t in range(8)])
parts = []
x0, w, gap = 130, 48, 8
labels = ["products", "half = 4", "half = 2", "half = 1"]
for r, row in enumerate(rows):
    y = 120 + r * 92
    half = [None, 4, 2, 1][r]
    kinds = []
    for t in range(8):
        if r == 0:
            kinds.append("store")
        elif t < half:
            kinds.append("good" if (r == 3 and t == 0) else "service")
        else:
            kinds.append("idle")
    row_parts, centres = cells(x0, y, row, "store", w=w, gap=gap, kinds=kinds)
    parts += row_parts + [mono(x0 - 14, y + 19, labels[r], "end", MUTED)]
    if r:
        for t in range(half):
            src = centres[t + half]
            mid = y - 30 + 7 * t - 3 * (half - 1)
            parts.append(elbow(src, y - 64, centres[t], y, KINDS["service"][0], mid))
    if r < 3:
        parts += [line(x0 - 6, y + 46, x0 + 8 * (w + gap) - 2, y + 46, KINDS["danger"][0], 1.6, "6 4"),
                  mono(x0 + 8 * (w + gap) + 2, y + 50, "barrier", "start", KINDS["danger"][0])]
for t in range(8):
    parts.append(mono(x0 + t * (w + gap) + w / 2, 108, f"cell {t}", "middle", MUTED))
parts += [legend(40, 496, [("service", "written by its own thread"), ("idle", "not touched")]),
          note(16, 530, "In each step thread t writes cell t and reads cell t + half. No cell is written by", room=628),
          note(16, 548, "2 threads, and no cell is read in the step that writes it: the safety argument.", room=628)]
save(OUT + "ch15-tree.svg", 660, 570,
     "8 cells hold the products 2, 0, 3, 12, 5, 1, 4, 6. With half = 4, threads 0 to 3 add cells 4 to 7 into "
     "cells 0 to 3, giving 7, 1, 7, 18. With half = 2, threads 0 and 1 give 14 and 19. With half = 1, thread 0 "
     "gives 33. A dashed barrier line separates the steps.",
     "The block tree in shared memory", "8 cells instead of 256; the steps are the same.", parts)


# 15.5  Combining the block results: a second launch, or an atomic addition.
parts = [zone(16, 104, 300, 340, "a second launch"), zone(344, 104, 300, 340, "an atomic addition")]
parts += [bar(40, 136, 252, 34, "store", "partial[0 … 1023]: 1 per block"),
          arrow(166, 170, 210),
          card(40, 210, 252, 64, "service", "launch 2: 1 block", "256 threads add 1,024 values"),
          arrow(166, 274, 314),
          bar(40, 314, 252, 34, "good", "total"),
          mono(40, 380, "fixed order of additions:", "start", MUTED),
          mono(40, 398, "same bits on every run", "start", MUTED)]
for k, name in enumerate(["block 7", "block 2", "block 900", "…"]):
    parts.append(bar(368, 136 + k * 40, 120, 30, "idle" if name == "…" else "client", name))
    if name != "…":
        parts.append(wire([(488, 151 + k * 40), (540, 151 + k * 40), (540, 312)], arrow=k == 0))
parts += [bar(480, 314, 120, 34, "good", "result[0]"),
          mono(368, 380, "order of arrival changes:", "start", MUTED),
          mono(368, 398, "the bits can change", "start", MUTED),
          note(16, 470, "Both designs add the same 1,024 block sums. The launch boundary orders the", room=628),
          note(16, 488, "additions; the atomic unit adds them in whatever order the blocks finish.", room=628)]
save(OUT + "ch15-combine.svg", 660, 510,
     "Left: the 1,024 block sums are written to an array, and a second launch of 1 block adds them into total "
     "in a fixed order. Right: each block adds its sum into result[0] with an atomic addition, in the order the "
     "blocks finish, for example block 7, then block 2, then block 900.",
     "Combining the block results", "1,024 block sums become 1 number in 1 of 2 ways.", parts)


# 15.6  Float addition depends on the order: 16,777,216 + 1 + 1.
parts = [zone(16, 104, 300, 260, "left to right"), zone(344, 104, 300, 260, "small values first")]
parts += [bar(40, 140, 120, 30, "store", "16,777,216"), bar(176, 140, 50, 30, "store", "1"),
          bar(240, 140, 50, 30, "store", "1"),
          bar(40, 214, 120, 30, "danger", "16,777,216"), mono(176, 234, "+1 rounds away", "start", MUTED),
          bar(40, 288, 120, 30, "danger", "16,777,216"), mono(176, 308, "+1 rounds away", "start", MUTED),
          arrow(100, 170, 214), arrow(100, 244, 288)]
parts += [bar(368, 140, 120, 30, "store", "16,777,216"), bar(504, 140, 50, 30, "store", "1"),
          bar(568, 140, 50, 30, "store", "1"),
          bar(536, 214, 50, 30, "service", "2"),
          elbow(529, 170, 561, 214), elbow(593, 170, 561, 214),
          bar(368, 288, 120, 30, "good", "16,777,218"),
          elbow(428, 170, 428, 288), elbow(561, 244, 428, 288)]
parts += [note(16, 392, "A float has 24 bits of significand. Between 2²⁴ = 16,777,216 and 2²⁵ the floats", room=628),
          note(16, 410, "are 2 apart, so 16,777,216 + 1 is a tie and rounds to the even 16,777,216. Adding", room=628),
          note(16, 428, "the 2 small values first gives 2, which is representable, and the exact 16,777,218.", room=628)]
save(OUT + "ch15-order.svg", 660, 450,
     "Left: 16,777,216 + 1 rounds back to 16,777,216, and so does the second + 1; the result is 16,777,216. "
     "Right: 1 + 1 = 2 first, then 16,777,216 + 2 = 16,777,218, the exact sum.",
     "The order of float additions", "The same 3 numbers, added in 2 orders, in 32-bit floats.", parts)


# 15.7  Measured rates of the 4 designs, against the memory bandwidth of the GPU.
runs = [("map, then reduce", 3.547, 75.7, 16), ("fused, shared tree", 1.430, 93.8, 8),
        ("fused, shuffles", 1.429, 94.0, 8), ("fused, atomic combine", 1.524, 88.1, 8)]
x0, wide, peak = 210, 380, 102.0
parts = []
for k, (name, ms, gb, per_pair) in enumerate(runs):
    y = 128 + k * 62
    kind = "danger" if per_pair == 16 else "service"
    parts += [text(x0 - 12, y + 20, name, "sub", "end"),
              bar(x0, y, wide * gb / peak, 30, kind, f"{gb} GB/s"),
              mono(x0 - 12, y + 40, f"{ms:.3f} ms, {per_pair} B/pair", "end", MUTED)]
px = x0 + wide
parts += [line(px, 112, px, 380, KINDS["danger"][0], 1.6, "6 4"),
          tag(px, 384, "102 GB/s: the memory bandwidth", "danger", "end"),
          legend(40, 436, [("danger", "16 bytes per pair"), ("service", "8 bytes per pair")]),
          note(16, 470, "The fused kernels move half the bytes and take 1.43 ms instead of 3.55 ms. At", room=628),
          note(16, 488, "94 GB/s they run at 92% of the memory bandwidth, so arithmetic is not the limit.", room=628)]
save(OUT + "ch15-rates.svg", 660, 510,
     "Horizontal bars of the measured rate in GB/s, against a dashed line at 102 GB/s. Map then reduce, 16 bytes "
     "per pair: 75.7 GB/s in 3.547 ms. Fused with a shared-memory tree: 93.8 GB/s in 1.430 ms. Fused with "
     "shuffles: 94.0 GB/s in 1.429 ms. Fused with an atomic combine: 88.1 GB/s in 1.524 ms.",
     "Measured rates", "16,777,216 pairs of floats, average of 20 runs, on a GPU with 102 GB/s.", parts)
