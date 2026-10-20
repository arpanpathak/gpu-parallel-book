"""The figures of chapter 8, drawn with figkit and figparts.

    python3 tools/draw_ch08.py

The bandwidths in figures 8.6 and 8.9 are the ones measured with code/ch08_reduction/reduce.cu
on a Jetson Orin NX and a Jetson AGX Thor (the runs quoted in the chapter).
"""
from figkit import width_of
from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, bar, card, legend, line, mono, note, save,
                      tag, text, wire, zone)

OUT = "src/figures/"
VALUES = [3, 1, 4, 1, 5, 9, 2, 6]


def elbow(x1, y1, x2, y2, colour=WIRE):
    """A connector from (x1, y1) down to (x2, y2): down, across, down, with an arrow."""
    mid = (y1 + y2) / 2
    return (f'  <path d="M{x1:.1f},{y1:.1f} V{mid:.1f} H{x2:.1f} V{y2 - 3:.1f}" fill="none" '
            f'stroke="{colour}" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#head)"/>')


# 8.1  A chain of additions against a tree of the same additions.
parts = [zone(16, 104, 262, 330, "1 after another"), zone(294, 104, 350, 330, "as a tree")]
running = 0
for k, v in enumerate(VALUES):
    running += v
    parts.append(bar(36 + k * 26, 136 + k * 34, 40, 26, "store" if k == 0 else "service", str(running)))
parts.append(mono(36, 418, "7 additions, 7 steps", "start", MUTED))

cw, gap, tx = 28, 6, 368
levels = [VALUES, [4, 5, 14, 8], [9, 22], [31]]
centres = [[tx + k * (cw + gap) + cw / 2 for k in range(8)]]
for lv in range(1, 4):
    prev = centres[-1]
    centres.append([(prev[2 * k] + prev[2 * k + 1]) / 2 for k in range(len(prev) // 2)])
for lv, row in enumerate(levels):
    y = 140 + lv * 76
    if lv:
        parts.append(mono(tx - 12, y + 18, f"step {lv}", "end", MUTED))
    for k, v in enumerate(row):
        kind = "store" if lv == 0 else "good" if lv == 3 else "service"
        parts.append(bar(centres[lv][k] - cw / 2, y, cw, 26, kind, str(v)))
        if lv:
            for child in (2 * k, 2 * k + 1):
                parts.append(elbow(centres[lv - 1][child], y - 50, centres[lv][k], y))
parts += [mono(tx, 418, "7 additions, 3 steps", "start", MUTED),
          note(16, 466, "Both orders make the same 7 additions. In the chain each addition waits for the", room=628),
          note(16, 484, "one before it; in the tree the additions of 1 step are independent of each other.", room=628)]
save(OUT + "ch08-chain-tree.svg", 660, 506,
     "Left: the 8 values 3, 1, 4, 1, 5, 9, 2, 6 summed 1 after another, a staircase of running sums 3, 4, 8, "
     "9, 14, 23, 25, 31: 7 additions in 7 steps. Right: the same values summed in pairs, 4, 5, 14, 8, then 9 "
     "and 22, then 31: 7 additions in 3 steps.",
     "A chain and a tree", "The same 7 additions, in 7 dependent steps or in 3 levels.", parts)


# 8.2  A float sum, 1 by 1, stops growing at 2^24 = 16,777,216.
x0, y0, wide, high = 110, 400, 470, 250
scale = high / 34e6
parts = []
for g in (0, 8, 16, 24, 32):
    y = y0 - g * 1e6 * scale
    parts += [line(x0, y, x0 + wide, y, EDGE, 1), mono(x0 - 10, y + 4, f"{g}M", "end", MUTED)]
for g in (0, 16, 32, 48, 64):
    x = x0 + g / 67.1 * wide
    parts.append(mono(x, y0 + 20, f"{g}M", "middle", MUTED))
stall = x0 + 33.55 / 67.1 * wide
top = y0 - 33557630 * scale
level = y0 - 16777216 * scale
parts += [line(x0, y0, x0 + wide, top, KINDS["good"][0], 2.6),
          line(x0, y0, stall, level, KINDS["danger"][0], 2.6, "7 4"),
          line(stall, level, x0 + wide, level, KINDS["danger"][0], 2.6, "7 4"),
          tag(x0 + wide - 4, top - 26, "exact: 33,557,630", "good", "end"),
          tag(x0 + wide - 4, level + 10, "float, 1 by 1: 16,777,216", "danger", "end"),
          text(x0 - 10, 112, "running sum", "sub", "end"),
          text(x0 + wide, y0 + 42, "values added", "sub", "end"),
          note(40, 466, "Above 2²⁴ a float can hold only even numbers, so adding a value below 1 rounds", room=600),
          note(40, 484, "back to the same total. The tree on the GPU gave 33,557,632, off by 2.", room=600)]
save(OUT + "ch08-float-stall.svg", 660, 506,
     "A chart of the running sum against the number of values added, for 67,108,864 values between 0 and 1. "
     "The exact sum rises in a straight line to 33,557,630. The float sum, added 1 by 1, follows it until "
     "16,777,216 and then stays flat.",
     "A float sum that stops growing", "67,108,864 values in [0, 1), added 1 by 1 into a float.", parts)


# 8.3  Coarsening: a grid-stride loop, then 1 tree per block.
parts = [text(40, 124, "the array: 67,108,864 floats, read in 256 passes of 262,144", "sub")]
passes = ["pass 1", "pass 2", "pass 3", "…", "pass 256"]
px, pw = 40, 112
for k, name in enumerate(passes):
    kind = "idle" if name == "…" else "store"
    parts.append(bar(px + k * (pw + 4), 136, pw, 30, kind, name))
    if name != "…":
        cx = px + k * (pw + 4) + 12
        parts += [f'  <rect x="{cx - 4}" y="140" width="8" height="22" rx="2" fill="{KINDS["client"][0]}"/>',
                  line(cx, 166, cx, 206, KINDS["client"][0], 1.6)]
parts += [line(52, 206, px + 4 * (pw + 4) + 12, 206, KINDS["client"][0], 1.6),
          wire([(330, 206), (330, 240)]),
          card(170, 240, 320, 56, "client", "thread 5: 1 register", "sum of elements 5, 5 + 262,144, …"),
          wire([(330, 296), (330, 330)]),
          card(170, 330, 320, 56, "service", "1 tree per block", "over 256 register sums, at the end"),
          note(40, 424, "Thread i reads element i of every pass, so neighbouring threads still read", room=600),
          note(40, 442, "neighbouring floats. Each thread adds 256 values before it touches shared memory.", room=600)]
save(OUT + "ch08-coarsen.svg", 660, 464,
     "The array of 67,108,864 floats as 256 passes of 262,144 elements, 1 per thread. Thread 5 reads element "
     "5 of every pass and adds them in a register. At the end, 1 tree per block combines the 256 register "
     "sums of its threads.",
     "Thread coarsening", "1,024 blocks of 256 threads walk the array in a grid-stride loop.", parts)


# 8.4  A warp sum by shuffles: offsets 16, 8, 4, 2, 1.
cx0, cw = 100, 16
parts = []
rows = [("32 lanes", 32), ("offset 16", 16), ("offset 8", 8), ("offset 4", 4), ("offset 2", 2),
        ("offset 1", 1)]
for r, (label, active) in enumerate(rows):
    y = 130 + r * 54
    parts.append(mono(cx0 - 10, y + 15, label, "end", MUTED))
    for lane in range(32):
        hot = lane < active
        kind = "store" if r == 0 else ("good" if r == 5 and lane == 0 else "client")
        accent, tint = KINDS[kind] if hot else KINDS["idle"]
        parts.append(f'  <rect x="{cx0 + lane * cw:.1f}" y="{y}" width="{cw - 3}" height="20" rx="3" '
                     f'fill="{tint}" stroke="{accent}" stroke-width="1.1"/>')
    if r:
        src = cx0 + active * cw + (cw - 3) / 2
        parts.append(elbow(src, y - 34, cx0 + (cw - 3) / 2, y))
parts += [legend(cx0, 470, [("client", "lane that adds"), ("idle", "result not used"),
                            ("good", "warp total")]),
          note(40, 506, "At offset d, lane k adds the value of lane k + d, read straight from that lane's", room=600),
          note(40, 524, "register. 5 steps leave the sum of all 32 lanes in lane 0, with no shared memory.", room=600)]
save(OUT + "ch08-shuffle.svg", 660, 546,
     "Six rows of 32 lanes. In each of 5 steps, with offsets 16, 8, 4, 2 and 1, the first 16, 8, 4, 2 and 1 "
     "lanes add the value held by the lane offset places to their right. An arrow shows lane 0 receiving "
     "from lane 16, 8, 4, 2 and 1. After the last step lane 0 holds the warp's total.",
     "A warp sum by shuffles", "__shfl_down_sync moves a register from lane k + d to lane k.", parts)


# 8.5  A block sum: 8 warp sums, 1 barrier, 1 more warp sum.
parts = [zone(16, 104, 196, 380, "8 warps shuffle")]
for w in range(8):
    parts.append(bar(36, 132 + w * 42, 156, 30, "client", f"warp {w}: 32 → 1"))
parts += [zone(240, 104, 150, 380, "shared memory")]
for w in range(8):
    parts += [bar(282, 132 + w * 42, 66, 30, "store", f"warps[{w}]"),
              wire([(192, 147 + w * 42), (282, 147 + w * 42)])]
parts += [line(404, 122, 404, 472, KINDS["danger"][0], 2.4),
          tag(404, 494, "__syncthreads()", "danger", "middle"),
          zone(420, 104, 224, 380, "warp 0"),
          card(436, 250, 192, 56, "client", "warp 0: 8 → 1", "shuffles again"),
          wire([(348, 278), (436, 278)]),
          card(436, 360, 192, 56, "good", "partial[block]", "1 value per block"),
          wire([(532, 306), (532, 360)]),
          note(16, 532, "Lane 0 of each warp stores its warp's total; after the 1 barrier, warp 0 adds", room=628),
          note(16, 550, "the 8 totals. The tree of chapter 5 needed 8 barriers for the same 256 values.", room=628)]
save(OUT + "ch08-block-sum.svg", 660, 572,
     "Eight warps each reduce their 32 values to 1 by shuffles and store it in warps[0] to warps[7] in shared "
     "memory. A barrier follows. Warp 0 then reduces the 8 values by shuffles, and thread 0 writes the "
     "block's total to partial[block].",
     "A block sum with 1 barrier", "Shuffles inside each warp; shared memory only between warps.", parts)


# 8.7  An inclusive scan by shifts of 1, 2 and 4 (8 lanes).
rows = [("values", VALUES)]
cur = VALUES[:]
for d in (1, 2, 4):
    cur = [cur[i] + (cur[i - d] if i >= d else 0) for i in range(8)]
    rows.append((f"offset {d}", cur))
cw, gap, gx = 52, 8, 150
parts = []
for r, (label, row) in enumerate(rows):
    y = 128 + r * 76
    parts.append(mono(gx - 12, y + 18, label, "end", MUTED))
    d = (0, 1, 2, 4)[r]
    for i, v in enumerate(row):
        kind = "store" if r == 0 else ("good" if r == 3 else "client" if i >= d else "idle")
        parts.append(bar(gx + i * (cw + gap), y, cw, 26, kind, str(v)))
    if r:
        src = gx + (7 - d) * (cw + gap) + cw / 2
        parts.append(elbow(src, y - 50, gx + 7 * (cw + gap) + cw / 2, y))
parts += [note(40, 442, "At offset d, lane i adds the value lane i − d held after the step before; lanes", room=600),
          note(40, 460, "below d keep theirs. After offsets 1, 2 and 4, lane i holds the sum of lanes 0 to i.", room=600)]
save(OUT + "ch08-scan-steps.svg", 660, 482,
     "Four rows of 8 values. Values 3, 1, 4, 1, 5, 9, 2, 6. After offset 1: 3, 4, 5, 5, 6, 14, 11, 8. After "
     "offset 2: 3, 4, 8, 9, 11, 19, 17, 22. After offset 4: 3, 4, 8, 9, 14, 23, 25, 31, the running sums. "
     "Arrows show lane 7 receiving from lanes 6, 5 and 3.",
     "A scan in 3 steps", "Each lane adds the value d lanes to its left, for d = 1, 2, 4.", parts)


# 8.8  A long scan: scan the blocks, scan their totals, add the offsets.
blocks = [[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8]]
cw, gap, bgap, gx = 34, 4, 20, 150


def scanned(row, start=0):
    out, run = [], start
    for v in row:
        run += v
        out.append(run)
    return out


def strip(y, rows, kind, label):
    out = [mono(gx - 12, y + 18, label, "end", MUTED)]
    for b, row in enumerate(rows):
        for i, v in enumerate(row):
            x = gx + b * (4 * (cw + gap) + bgap) + i * (cw + gap)
            out.append(bar(x, y, cw, 26, kind, str(v)))
    return out


local = [scanned(b) for b in blocks]
totals = [r[-1] for r in local]
offsets = [0] + scanned(totals)[:-1]
final = [scanned(b, o) for b, o in zip(blocks, offsets)]
parts = strip(128, blocks, "store", "values") + strip(196, local, "client", "1. scan blocks")
parts.append(mono(gx - 12, 282, "2. scan totals", "end", MUTED))
for b, t in enumerate(totals):
    parts.append(bar(gx + b * (cw + gap), 264, cw, 26, "outside", str(t)))
    parts.append(bar(gx + 190 + b * (cw + gap), 264, cw, 26, "outside", str(scanned(totals)[b])))
parts.append(wire([(gx + 3 * (cw + gap) + 6, 277), (gx + 184, 277)]))
for b, o in enumerate(offsets):
    parts.append(tag(gx + b * (4 * (cw + gap) + bgap), 326, f"+ {o}", "outside"))
parts += strip(352, final, "good", "3. add offsets")
parts += [note(40, 424, "Each block scans its own values and writes its total. The totals are scanned the same", room=600),
          note(40, 442, "way, and block b adds the scanned total of blocks 0 to b − 1 to each of its values.", room=600)]
save(OUT + "ch08-scan-levels.svg", 660, 464,
     "Twelve values in 3 blocks of 4: 3, 1, 4, 1; 5, 9, 2, 6; 5, 3, 5, 8. Step 1 scans each block: 3, 4, 8, 9; "
     "5, 14, 16, 22; 5, 8, 13, 21, with totals 9, 22 and 21. Step 2 scans the totals: 9, 31, 52. Step 3 adds "
     "0, 9 and 31 to the 3 blocks, giving 3, 4, 8, 9, 14, 23, 25, 31, 36, 39, 44, 52.",
     "A scan of many blocks", "Scan each block, scan the block totals, add each block's offset.", parts)


def measured(path, rows, unit_max, alt, title, lede, ceilings=None, height=None):
    """Horizontal bars, 1 pair per row: Orin NX and Thor, in GB/s."""
    x0, scale = 190, 400 / unit_max
    parts = []
    bottom = 116 + len(rows) * 64
    for g in range(0, unit_max + 1, 50):
        parts += [line(x0 + g * scale, 104, x0 + g * scale, bottom, EDGE, 1),
                  mono(x0 + g * scale, bottom + 18, str(g), "middle", MUTED)]
    for k, (label, orin, thor) in enumerate(rows):
        y = 116 + k * 64
        parts.append(text(x0 - 14, y + 26, label, "sub", "end"))
        for j, (v, kind) in enumerate(((orin, "client"), (thor, "outside"))):
            parts += [f'  <rect x="{x0}" y="{y + j * 24}" width="{v * scale:.1f}" height="20" rx="5" '
                      f'fill="{KINDS[kind][0]}"/>',
                      f'  <rect x="{x0 + v * scale + 4:.1f}" y="{y + j * 24 + 2}" width="{len(str(v)) * 8 + 8}" '
                      f'height="17" rx="3" fill="{PAPER}"/>',
                      mono(x0 + v * scale + 8, y + j * 24 + 15, f"{v}", "start", INK)]
    if ceilings:
        for v, kind in ceilings:
            x = x0 + v * scale
            parts.insert(0, line(x, 104, x, bottom, KINDS[kind][0], 1.6, "5 4"))
            label = f"{v} rated"
            parts.append(tag(x - (width_of(label, 11, bold=True) + 14) / 2, 84, label, kind))
    parts += [text(x0 - 14, bottom + 18, "GB/s", "sub", "end"),
              legend(x0, bottom + 46, [("client", "Jetson Orin NX"), ("outside", "Jetson AGX Thor")])]
    save(OUT + path, 660, height or bottom + 72, alt, title, lede, parts)


# 8.6  Measured sums on 2 boards, against each board's rated memory bandwidth.
measured("ch08-sum-times.svg",
         [("tree, 1 per thread", 25.9, 88.3), ("coarsened", 64.1, 203.0), ("shuffles", 64.4, 203.6),
          ("shuffles, float4", 89.7, 264.2)], 300,
         "Bars of sum bandwidth for 67,108,864 floats. Orin NX: tree 26 GB/s, coarsened 64, shuffles 64, "
         "shuffles with float4 90, against a rated 102. Thor: tree 88 GB/s, coarsened 203, shuffles 204, "
         "shuffles with float4 264, against a rated 273. Dashed lines mark the rated bandwidths.",
         "A sum of 256 MiB, measured on 2 boards",
         "GB/s of floats read; dashed: each board's rated 102 and 273 GB/s.",
         ceilings=[(102, "client"), (273, "outside")])


# 8.9  Measured histograms on 2 boards.
measured("ch08-hist-times.svg",
         [("random, 1 byte", 13.7, 49.7), ("random, 16 bytes", 82.9, 190.9),
          ("all 7, 1 byte", 16.8, 54.1), ("all 7, 16 bytes", 93.4, 210.6)], 300,
         "Bars of histogram bandwidth for 64 MiB of bytes. Random bytes: 1 byte per load 14 GB/s on the "
         "Orin NX and 50 on the Thor; 16 bytes per load 83 and 191. Bytes all 7: 17 and 54 GB/s, then 93 "
         "and 211.",
         "A histogram of 64 MiB, measured on 2 boards",
         "GB/s of bytes read, with 1 byte or 16 bytes per load.")
