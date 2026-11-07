"""The figures of chapter 20, drawn with figkit and figparts.

    python3 tools/draw_ch20.py

Every figure uses the 6 x 6 plate of chapter 19, cut into blocks of 2 x 2 cells for the worked
example. The measured values are the ones printed by code/oxide/ch20_stencil_tiled and
code/cuda/ch20_stencil_tiled in the runs quoted in the text.
"""
import math

from draw_ch19 import STEP1, STEP2, heat, jacobi, outline, plate, short, square, START
from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, line, mono, note, save, tag, text,
                      zone)

OUT = "src/figures/"
RED = KINDS["danger"][0]
BLUE = KINDS["client"][0]
GREEN = KINDS["good"][0]
AMBER = KINDS["store"][0]
N = 6
BLOCK_TINTS = ["#EAF1FD", "#E4F4EF", "#FBF1DC", "#F0EAFB", "#FCEBE6", "#EEF6DC", "#F1F4F7",
               "#E7F5E9", "#FDF3E4"]


def arrow(x1, y1, x2, y2, colour=WIRE, width=1.8):
    """A straight arrow from (x1, y1) to (x2, y2), horizontal or vertical."""
    marker = "head" if colour == WIRE else "head-danger"
    return (f'  <path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{colour}" '
            f'stroke-width="{width}" fill="none" marker-end="url(#{marker})"/>')


def bar(x0, x1, y, label, kind, h=24):
    accent, tint = KINDS[kind]
    return [f'  <rect x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{h}" rx="6" fill="{tint}" '
            f'stroke="{accent}" stroke-width="1.3"/>',
            text((x0 + x1) / 2, y + h / 2 + 4, label, "pill", "middle", accent)]


def wait(x0, x1, y, h=24):
    return line(x0, y + h / 2, x1, y + h / 2, MUTED, 1.2, "3 4")


# 20.1  Each value is read by its neighbours.
side = 40
parts = [zone(32, 96, 280, 330, "step 1: who reads u(2, 2)"),
         zone(328, 96, 280, 330, "reads of each value")]
parts += plate(52, 124, STEP1, side, size=11)
cx, cy = 52 + 2 * side + 18.5, 124 + 2 * side + 18.5
parts.append(outline(52 + 2 * side - 2, 124 + 2 * side - 2, side + 1, side + 1, AMBER, 2.6))
for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    tx, ty = cx + dc * side, cy + dr * side
    parts.append(outline(52 + (2 + dc) * side - 2, 124 + (2 + dr) * side - 2, side + 1, side + 1,
                         BLUE))
    parts.append(arrow(cx + dc * 10, cy + dr * 10, tx - dc * 13, ty - dr * 13, WIRE, 1.6))
counts = [[0] * N for _ in range(N)]
for r in range(1, N - 1):
    for c in range(1, N - 1):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            counts[r + dr][c + dc] += 1
shades = {0: "#FFFFFF", 1: "#F1F4F7", 2: "#E6EEF9", 3: "#CFDEF6", 4: "#B4CBF0"}
for r in range(N):
    for c in range(N):
        edge = r in (0, N - 1) or c in (0, N - 1)
        parts.append(square(348 + c * side, 124 + r * side, side - 3, side - 3, shades[counts[r][c]],
                            INK if edge else EDGE, str(counts[r][c]), width=1.6 if edge else 1.1))
parts += [note(52, 392, "4 threads fetch u(2, 2) from"),
          note(52, 408, "global memory, each on its own."),
          note(348, 392, "16 threads, 64 reads,"),
          note(348, 408, "32 different values.")]
save(OUT + "ch20-reads.svg", 640, 446,
     "Left: the 6 by 6 plate after step 1. Value u(2,2) is highlighted and 4 arrows lead from it to "
     "the threads of cells (1,2), (3,2), (2,1) and (2,3), each of which fetches it from global "
     "memory. Right: the same plate with the number of threads that read each value: 4 in the 4 "
     "centre cells, 3 along the edges of the interior, 2 in the interior corners, 1 in the boundary "
     "cells next to the interior, 0 in the 4 corners of the plate. Totals: 64 reads of 32 different "
     "values.",
     "Every value is read by its neighbours",
     "In the direct kernel, value u(2, 2) is fetched from global memory by 4 threads.", parts)

# 20.2  9 blocks of 2 x 2 threads; block (1, 1) and its tile.
side = 46
x0, y0 = 90, 120
parts = []
for r in range(N):
    for c in range(N):
        b = (r // 2) * 3 + c // 2
        parts.append(square(x0 + c * side, y0 + r * side, side - 4, side - 4, BLOCK_TINTS[b],
                            EDGE, short(STEP1[r][c]), width=1.0))
for k in range(4):
    v = x0 + 2 * k * side - 2
    h = y0 + 2 * k * side - 2
    parts += [line(v, y0 - 2, v, y0 + 6 * side - 2, INK, 2.2),
              line(x0 - 2, h, x0 + 6 * side - 2, h, INK, 2.2)]
for b in range(3):
    parts += [mono(x0 + (2 * b + 1) * side - 2, y0 - 10, f"bx = {b}", "middle", MUTED),
              mono(x0 - 10, y0 + (2 * b + 1) * side + 2, f"by = {b}", "end", MUTED)]
parts += [outline(x0 + 2 * side + 1, y0 + 2 * side + 1, 2 * side - 6, 2 * side - 6, BLUE, 2.6),
          outline(x0 + side - 4, y0 + side - 4, 4 * side + 2, 4 * side + 2, AMBER, 2.6, "7 5"),
          note(390, 150, "blue: block (1, 1), cells (2, 2) to (3, 3)", fill=BLUE),
          note(390, 176, "dashed: its tile, rows 1 to 4, columns 1 to 4", fill=AMBER),
          note(390, 230, "The 9 blocks of 2 × 2 threads cover"),
          note(390, 248, "the 36 cells, 1 thread per cell."),
          note(390, 284, "The tile holds the 4 own cells and"),
          note(390, 302, "the 12 cells around them, which"),
          note(390, 320, "belong to the 8 neighbouring blocks."),
          note(390, 356, "Values shown: the plate after step 1.")]
save(OUT + "ch20-blocks.svg", 640, 420,
     "The 6 by 6 plate cut into 9 blocks of 2 by 2 cells, 1 colour per block. Block (1, 1) owns "
     "cells (2,2), (2,3), (3,2) and (3,3). Its tile, the 4 by 4 square of rows 1 to 4 and columns 1 "
     "to 4, is outlined: the 4 owned cells in the middle and 12 cells around them, which belong to "
     "the 8 neighbouring blocks.",
     "Blocks of 2 × 2 threads on the plate",
     "Block (1, 1) computes the 4 centre cells and needs the 4 × 4 square around them.", parts)

# 20.3  The tile of block (1, 1) in step 2.
side = 56
x0, y0 = 72, 128
parts = [zone(32, 96, 300, 316, "tile of block (1, 1)")]
for r in range(4):
    for c in range(4):
        corner = r in (0, 3) and c in (0, 3)
        inner = 1 <= r <= 2 and 1 <= c <= 2
        kind = "idle" if corner else ("client" if inner else "store")
        accent, tint = KINDS[kind]
        parts.append(square(x0 + c * side, y0 + r * side, side - 5, side - 5, tint, accent,
                            short(STEP1[r + 1][c + 1]), width=1.4))
        parts.append(f'  <text x="{x0 + c * side + 4}" y="{y0 + r * side + 11}" class="note" '
                     f'style="font-size:9px;fill:{MUTED}">{r},{c}</text>')
for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    parts.append(outline(x0 + (1 + dc) * side - 3, y0 + (1 + dr) * side - 3, side + 1, side + 1,
                         RED, 2.2))
parts.append(outline(x0 + side - 3, y0 + side - 3, side + 1, side + 1, GREEN, 2.2, "5 4"))
parts += [zone(352, 96, 256, 316, "cell (2, 2)")]
rows = [("up", "tile[0][1]", "25"), ("down", "tile[2][1]", "0"), ("left", "tile[1][0]", "0"),
        ("right", "tile[1][2]", "0")]
for k, (where, name, value) in enumerate(rows):
    y = 140 + k * 30
    parts += [note(370, y, where), mono(420, y, name), mono(590, y, value, "end")]
parts += [line(370, 262, 592, 262, INK, 1.0),
          mono(370, 284, "sum"), mono(590, 284, "25", "end"),
          mono(370, 308, "× 1/4"), mono(590, 308, "6.25", "end", GREEN),
          tag(370, 330, "own cell", "client"), tag(446, 330, "halo", "store"),
          tag(498, 330, "corner, unused", "idle"),
          note(370, 378, "Same values as in global memory;"),
          note(370, 394, "only the place they are read from moved.")]
save(OUT + "ch20-tile.svg", 640, 432,
     "The 4 by 4 tile of block (1, 1) in step 2, as it sits in shared memory. Its top row holds 25, "
     "25, 25, 25 (row 1 of the plate) and the other 3 rows hold 0. The 4 inner positions are blue "
     "(own cells), the 8 edge positions amber (halo), the 4 corners grey (loaded, never read). For "
     "cell (2, 2), at tile position (1, 1), the 4 neighbours at positions (0, 1), (2, 1), (1, 0) and "
     "(1, 2) are outlined: 25 + 0 + 0 + 0, times 1/4, gives 6.25.",
     "Computing from the tile",
     "Cell (2, 2) reads its 4 neighbours from shared memory instead of global memory.", parts)

# 20.4  2 ways to fill the tile.
side = 50
parts = [zone(32, 96, 280, 340, "extra threads: 4 × 4 threads"),
         zone(328, 96, 280, 340, "loading loop: 2 × 2 threads")]
for r in range(4):
    for c in range(4):
        inner = 1 <= r <= 2 and 1 <= c <= 2
        kind = "good" if inner else "idle"
        accent, tint = KINDS[kind]
        parts.append(square(72 + c * side, 128 + r * side, side - 5, side - 5, tint, accent,
                            f"t{r},{c}", size=11, width=1.3))
loaders = {(0, 0): "client", (0, 1): "service", (1, 0): "store", (1, 1): "outside"}
for r in range(4):
    for c in range(4):
        who = (r % 2, c % 2)
        accent, tint = KINDS[loaders[who]]
        parts.append(square(368 + c * side, 128 + r * side, side - 5, side - 5, tint, accent,
                            f"t{who[0]},{who[1]}", size=11, width=1.3))
parts += [note(52, 352, "16 threads load 1 slot each;"),
          note(52, 368, "the 4 green ones also compute."),
          note(52, 396, "Programs: 34 × 10 = 340 threads,"),
          note(52, 412, "256 of them compute."),
          note(348, 352, "Each thread loads every slot (i, j)"),
          note(348, 368, "with i ≡ ty, j ≡ tx (mod 2): 4 slots."),
          note(348, 396, "Programs: 256 threads; 180 load 1 slot,"),
          note(348, 412, "72 load 2, and 4 load 4.")]
save(OUT + "ch20-loaders.svg", 640, 456,
     "Two ways to load the 4 by 4 tile of block (1, 1). Left, extra threads: the block has 4 by 4 = "
     "16 threads, 1 per slot; all 16 load, and only the inner 4 compute. Right, a loop: the block "
     "has 2 by 2 = 4 threads, and each loads 4 slots, at rows ty and ty + 2 and columns tx and tx + "
     "2; the slots are coloured by the thread that loads them. A note gives the counts for the "
     "programs: 34 by 10 = 340 threads, 256 of which compute; or 256 threads, of which 180 load 1 "
     "slot, 72 load 2 and 4 load 4.",
     "2 ways to load a tile and its halo",
     "Every slot needs exactly 1 loading thread; the block has fewer threads than slots.", parts)

# 20.5  The 3 phases of a block.
barrier = 330
parts = [line(barrier, 104, barrier, 300, INK, 2.4), text(barrier, 318, "barrier", "name", "middle")]
for k, (name, end) in enumerate([("thread (0, 0)", 250), ("thread (0, 1)", 312),
                                  ("thread (1, 0)", 220), ("thread (1, 1)", 286)]):
    y = 120 + k * 44
    parts += [note(32, y + 16, name), *bar(130, end, y, "load 4 slots", "client")]
    if end < barrier - 6:
        parts.append(wait(end + 4, barrier - 4, y))
    parts += bar(barrier + 8, 560, y, "compute 1 cell from the tile", "good")
parts += [note(130, 350, "load: global → shared"), note(barrier + 8, 350, "read: shared only"),
          note(32, 390, "Dotted: threads that finished loading wait until the last one arrives. "
                        "Only then may any"),
          note(32, 406, "thread read the tile.")]
save(OUT + "ch20-phases.svg", 640, 426,
     "A timeline of the 4 threads of block (1, 1). Each thread loads its 4 slots, taking different "
     "times; dotted lines show the early threads waiting at the barrier; then all 4 compute their "
     "cell from the tile and write it to buffer v.",
     "Load, wait, compute",
     "The barrier separates the loads of all threads from the reads of any thread.", parts)

# 20.6  Without the barrier.
parts = [note(32, 136, "thread (0, 0)"), note(32, 196, "thread (0, 1)")]
parts += bar(140, 260, 120, "load 4 slots", "client")
parts += bar(266, 560, 120, "compute (2, 2): reads tile[0][1]", "client")
parts += bar(140, 220, 180, "(delayed)", "idle")
parts += bar(226, 470, 180, "load: tile[0][1] = 25", "store")
parts += [f'  <path d="M330,146 V176" stroke="{RED}" stroke-width="2" '
          f'marker-end="url(#head-danger)"/>',
          f'  <circle cx="330" cy="146" r="4" fill="{RED}"/>',
          note(330, 236, "thread (0, 0) reads tile[0][1] before thread (0, 1) stores 25", "middle",
               fill=RED),
          note(330, 256, "the slot still holds 0, left by an earlier block", "middle", fill=RED),
          text(330, 284, "cell (2, 2) = (0 + 0 + 0 + 0) / 4 = 0 in place of 6.25", "name",
               "middle").replace('class="name"', f'class="name" style="fill:{RED}"', 1)]
save(OUT + "ch20-nobarrier.svg", 640, 310,
     "A timeline of 2 threads of block (1, 1) without the barrier. Thread (0, 0) loads its 4 slots "
     "and goes on to compute cell (2, 2); it reads tile position (0, 1) at a moment marked in red. "
     "Thread (0, 1) is delayed and stores 25 into position (0, 1) only afterwards. The result: thread "
     "(0, 0) reads the stale value 0 and writes 0 instead of 6.25.",
     "Without the barrier",
     "A thread reads a halo slot before the thread that loads it has stored the value.", parts)

# 20.7  Without the halo.
nohalo = [row[:] for row in STEP2]
nohalo[1][1:5] = [25.0, 31.25, 31.25, 25.0]
nohalo[2][1:5] = [0.0, 0.0, 0.0, 0.0]
side = 44
parts = [zone(32, 96, 288, 316, "step 2, correct"), zone(336, 96, 288, 316, "step 2, no halo")]
for x, values in ((50, STEP2), (354, nohalo)):
    parts += plate(x, 124, values, side, size=11)
    for k in (2, 4):
        v = x + k * side - 2
        h = 124 + k * side - 2
        parts += [line(v, 122, v, 124 + 6 * side - 2, INK, 2.4),
                  line(x - 2, h, x + 6 * side - 2, h, INK, 2.4)]
for r in range(N):
    for c in range(N):
        if nohalo[r][c] != STEP2[r][c]:
            parts.append(outline(354 + c * side - 2, 124 + r * side - 2, side + 1, side + 1, RED,
                                 2.2))
parts += [note(50, 400, "Thick lines: borders between blocks of 2 × 2."),
          note(354, 400, "Red: 8 wrong cells, each with a neighbour"),
          note(354, 416, "in another block.")]
save(OUT + "ch20-nohalo.svg", 640, 436,
     "2 versions of the plate after step 2, side by side. Left, correct: row 1 reads 31.25, 37.5, "
     "37.5, 31.25 and row 2 reads 6.25 in each interior cell. Right, without the halo: row 1 reads "
     "25, 31.25, 31.25, 25 and row 2 reads 0; every differing cell is outlined in red. Block "
     "boundaries are drawn as thick lines, showing that each wrong cell has a neighbour in another "
     "block.",
     "Without the halo",
     "Every cell whose neighbour belongs to another block loses that neighbour.", parts)

# 20.8  Loads per result against the block width.
left, right, top, bottom = 90, 590, 110, 330


def lx(t):
    return left + (right - left) * (math.log2(t) - 1) / 4


def ly(v):
    return bottom - (bottom - top) * v / 4.5


parts = [line(left, bottom, right, bottom, INK, 1.2), line(left, top, left, bottom, INK, 1.2)]
for v in range(5):
    parts += [line(left, ly(v), right, ly(v), EDGE, 0.8), mono(left - 8, ly(v) + 4, str(v), "end",
                                                                MUTED)]
for t in (2, 4, 8, 16, 32):
    parts.append(mono(lx(t), bottom + 18, f"T = {t}", "middle", MUTED))
parts.append(note((left + right) / 2, bottom + 38, "block of T × T threads", "middle"))
parts += [line(left, ly(4), right, ly(4), RED, 2.0),
          note(right, ly(4) - 8, "direct kernel: 4 reads requested", "end", fill=RED),
          line(left, ly(1.25), right, ly(1.25), AMBER, 1.8, "6 4"),
          note(left + 6, ly(1.25) + 16, "direct kernel from memory, with caches: 1.25",
               fill=AMBER)]
curve = [(t, (t + 2) ** 2 / t ** 2) for t in [2 ** (k / 8) for k in range(8, 41)]]
trace = " ".join(("M" if k == 0 else "L") + f"{lx(t):.1f},{ly(v):.1f}"
                 for k, (t, v) in enumerate(curve))
parts.append(f'  <path d="{trace}" fill="none" stroke="{BLUE}" stroke-width="2.4"/>')
for t in (2, 4, 8, 16, 32):
    v = (t + 2) ** 2 / t ** 2
    parts += [f'  <circle cx="{lx(t):.1f}" cy="{ly(v):.1f}" r="4.5" fill="{KINDS["client"][1]}" '
              f'stroke="{BLUE}" stroke-width="1.8"/>',
              mono(lx(t) + 8, ly(v) - 8, f"{v:.2f}", fill=BLUE)]
x = lx(math.sqrt(256))
parts += [f'  <rect x="{x - 5:.1f}" y="{ly(340 / 256) - 5:.1f}" width="10" height="10" '
          f'fill="{GREEN}"/>',
          line(x, ly(340 / 256) + 6, x, ly(0.55), GREEN, 1.2),
          note(x, ly(0.55) + 14, "32 × 8: 340 / 256 = 1.33", "middle", fill=GREEN),
          note(left + 10, top + 4, "tile: (T + 2)² / T² loads per result", fill=BLUE)]
save(OUT + "ch20-overhead.svg", 640, 396,
     "A chart of global loads per computed cell against the block width T from 2 to 32. The direct "
     "kernel is a flat line at 4. The tiled kernel follows the curve (T + 2) squared over T squared: "
     "4 at T = 2, 2.25 at 4, 1.56 at 8, 1.27 at 16, 1.13 at 32. A marked point shows the 32 by 8 "
     "block of the programs at 1.33. The cache model of Chapter 19 for the direct kernel, 1.25 "
     "loads from memory per cell, is drawn as a dashed line.",
     "Loads per result",
     "The halo overhead shrinks as the block grows; the caches already reach 1.25.", parts)

# 20.9  Measured effective bandwidth, C++ and Rust, Orin NX.
SPEEDS = [("copy kernel", 76.6, 75.0, "idle"), ("direct", 68.3, 58.0, "client"),
          ("tiled, loading loop", 43.4, 34.6, "store"),
          ("tiled, extra threads", 45.8, 39.2, "outside"),
          ("4 steps per load", 63.5, 54.3, "good")]
scale = 380 / 100
parts = [tag(180, 92, "C++", "client"), tag(226, 92, "Rust", "idle")]
for k, (name, cpp, rust, kind) in enumerate(SPEEDS):
    y = 122 + k * 56
    accent, tint = KINDS[kind]
    parts += [note(40, y + 21, name),
              f'  <rect x="180" y="{y}" width="{cpp * scale:.1f}" height="20" rx="4" '
              f'fill="{tint}" stroke="{accent}" stroke-width="1.4"/>',
              mono(186, y + 14, f"{cpp:.1f} GB/s", fill=accent),
              f'  <rect x="180" y="{y + 22}" width="{rust * scale:.1f}" height="14" rx="4" '
              f'fill="#FFFFFF" stroke="{accent}" stroke-width="1.0" stroke-dasharray="4 3"/>',
              text(186, y + 33, f"{rust:.1f} GB/s", "note", "start", accent)]
x = 180 + 102 * scale
parts += [f'  <path d="M{x:.1f},110 V400" stroke="{RED}" stroke-width="1.4" '
          f'stroke-dasharray="5 4"/>',
          note(x - 4, 418, "rated 102 GB/s", "end", fill=RED),
          note(40, 446, "Solid bars: C++; dashed bars: Rust. 8 bytes per cell per step, 4096 × "
                        "4096 cells.")]
save(OUT + "ch20-bandwidth.svg", 640, 466,
     "Horizontal bars of effective bandwidth on a 4096 by 4096 grid on the Orin NX, for C++ and "
     "Rust: copy, direct kernel, tiled with a loading loop, tiled with extra threads, and 4 steps "
     "per load. In C++: 76.6, 68.3, 43.4, 45.8 and 63.5 GB/s. In Rust: 75.0, 58.0, 34.6, 39.2 and "
     "54.3 GB/s. The direct kernel is the fastest stencil kernel in both languages.",
     "Effective bandwidth of the tiled kernels",
     "Jetson Orin NX: neither tile beats the direct kernel; the caches supply its reuse.", parts)

# 20.10  Temporal blocking: the correct region shrinks by 1 ring per step.
cx, cy, unit = 190, 270, 6.4
parts = []
rings = [(32, "loaded: 32 × 32", "store"), (30, "after step 1: 30 × 30", None),
         (28, "after step 2: 28 × 28", None), (26, "after step 3: 26 × 26", None),
         (24, "after step 4: 24 × 24, written", "good")]
for k, (size, label, kind) in enumerate(rings):
    half = size * unit / 2
    if kind:
        accent, tint = KINDS[kind]
    else:
        accent, tint = KINDS["client"][0], "none"
    fill = tint if kind else "none"
    parts.append(f'  <rect x="{cx - half:.1f}" y="{cy - half:.1f}" width="{2 * half:.1f}" '
                 f'height="{2 * half:.1f}" rx="3" fill="{fill}" stroke="{accent}" '
                 f'stroke-width="{2.2 if kind else 1.2}"/>')
for k, (size, label, kind) in enumerate(rings):
    parts.append(note(380, 140 + k * 24, label))
    accent = KINDS[kind][0] if kind else BLUE
    parts.append(f'  <rect x="362" y="{130 + k * 24}" width="12" height="12" rx="2" '
                 f'fill="{KINDS[kind][1] if kind else "#FFFFFF"}" stroke="{accent}" '
                 f'stroke-width="1.4"/>')
parts += [note(cx, cy + 16 * unit + 22, "halo 4 cells wide, own cells 24 × 24", "middle")]
for k, (src, dst) in enumerate([("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")]):
    y = 284 + k * 34
    parts += [note(362, y + 16, f"step {k + 1}"),
              square(420, y, 44, 24, KINDS["client"][1], BLUE, src, width=1.3),
              arrow(468, y + 12, 530, y + 12),
              square(534, y, 44, 24, KINDS["good"][1], GREEN, dst, width=1.3)]
parts.append(note(362, 438, "2 tiles in shared memory, used in turn"))
save(OUT + "ch20-temporal.svg", 640, 456,
     "A 32 by 32 tile drawn as nested squares. The outer ring 4 cells wide is the halo, loaded from "
     "global memory. After step 1 the correct region is 30 by 30, after step 2 28 by 28, after step 3 "
     "26 by 26, after step 4 24 by 24, the block's own cells, which are written to global memory. "
     "Beside it, 2 tiles in shared memory, A and B, with arrows A to B for steps 1 and 3 and B to A "
     "for steps 2 and 4.",
     "Temporal blocking with K = 4",
     "Each step loses 1 ring; after 4 steps the block's own 24 × 24 cells are correct.",
     parts)

# 20.11  Temporal blocking: traffic, work and speed against K.
SWEEP = [(1, 30.3), (2, 48.4), (4, 62.4), (6, 57.1), (8, 43.7)]
PLAIN = 65.0
left, right, top, bottom = 90, 560, 134, 354
slots = {k: left + 40 + i * 98 for i, (k, _) in enumerate(SWEEP)}
parts = [line(left, bottom, right, bottom, INK, 1.2), line(left, top, left, bottom, INK, 1.2),
         line(right, top, right, bottom, INK, 1.2)]
for v in range(0, 81, 20):
    y = bottom - (bottom - top) * v / 80
    parts += [line(left, y, right, y, EDGE, 0.8), mono(left - 8, y + 4, str(v), "end", MUTED)]
for b in range(0, 13, 2):
    y = bottom - (bottom - top) * b / 12
    parts.append(mono(right + 8, y + 4, str(b), "start", MUTED))
parts += [note(left - 70, top - 30, "GB/s (points)", fill=GREEN),
          note(right + 40, top - 30, "bytes per cell and step (bars)", "end", fill=AMBER)]
points = []
for k, gbs in SWEEP:
    x = slots[k]
    out = 32 - 2 * k
    traffic = 4 / k * (1024 / out ** 2 + 1)
    work = 1024 / out ** 2
    h = (bottom - top) * traffic / 12
    parts += [f'  <rect x="{x - 18}" y="{bottom - h:.1f}" width="36" height="{h:.1f}" rx="3" '
              f'fill="{KINDS["store"][1]}" stroke="{AMBER}" stroke-width="1.2"/>',
              mono(x, bottom - h - 6, f"{traffic:.2f}", "middle", AMBER),
              mono(x, bottom + 18, f"K = {k}", "middle", MUTED),
              note(x, bottom + 36, f"{work:.2f} thread-steps", "middle")]
    points.append((x, bottom - (bottom - top) * gbs / 80, gbs))
trace = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(points))
parts.append(f'  <path d="{trace}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
for x, y, gbs in points:
    parts += [f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{KINDS["good"][1]}" '
              f'stroke="{GREEN}" stroke-width="2"/>',
              mono(x - 9, y - 9, f"{gbs:.1f}", "end", GREEN)]
y = bottom - (bottom - top) * PLAIN / 80
parts += [line(left, y, right, y, BLUE, 1.6, "6 4"),
          note(left + 30, top - 14, f"dashed: direct kernel in the same run, {PLAIN} GB/s",
               fill=BLUE)]
save(OUT + "ch20-temporal-sweep.svg", 640, 420,
     "A chart over K = 1, 2, 4, 6, 8 steps per load. Bars give the modelled bytes per cell and step: "
     "8.55, 4.61, 2.78, 2.37, 2.50. Under each bar, thread-steps per kept cell and step: 1.14, 1.31, "
     "1.78, 2.56, 4.00. Points give the measured effective bandwidth: 30.3, 48.4, 62.4, 57.1, 43.7 "
     "GB/s, highest at K = 4. A dashed line marks the direct kernel in the same run at 65.0 GB/s.",
     "Temporal blocking: traffic against work",
     "Jetson Orin NX, C++: the speed peaks at K = 4, where traffic stops falling.", parts)
