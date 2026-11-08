"""The figures of chapter 19, drawn with figkit and figparts.

    python3 tools/draw_ch19.py

Every figure uses the 6 x 6 plate of the chapter. The measured values in the last figures are the
ones printed by code/oxide/ch19_stencil and code/cuda/ch19_stencil in the runs quoted in the text.
"""
import math

from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, line, mono, note, save, tag, text,
                      zone)

OUT = "src/figures/"
RED = KINDS["danger"][0]
BLUE = KINDS["client"][0]
GREEN = KINDS["good"][0]
AMBER = KINDS["store"][0]


def heat(t):
    """A colour for a temperature t from 0 (cool blue) through yellow to 100 (hot red)."""
    stops = [(0, (214, 228, 247)), (30, (190, 222, 214)), (60, (248, 222, 150)),
             (100, (236, 140, 108))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return "#" + "".join(f"{round(a + f * (b - a)):02x}" for a, b in zip(c0, c1))
    return "#ec8c6c"


def square(x, y, w, h, fill, stroke, label="", cls="mono", ink=INK, width=1.2, dash="", size=None):
    """A cell with a fill, an edge and an optional centred label."""
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>']
    if label:
        body = text(x + w / 2, y + h / 2 + 4, label, cls, "middle", ink)
        if size:
            body = body.replace('style="fill:', f'style="font-size:{size}px;fill:', 1)
        parts.append(body)
    return "\n".join(parts)


def outline(x, y, w, h, colour, width=2.4, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" fill="none" '
            f'stroke="{colour}" stroke-width="{width}"{extra}/>')


def short(v):
    """Whole numbers without decimals, others with 2."""
    return f"{v:.0f}" if v == int(v) else f"{v:.2f}"


def plate(x, y, values, side, labels=None, size=None):
    """A grid of temperatures coloured by value; boundary cells get a darker edge."""
    rows, cols = len(values), len(values[0])
    parts = []
    for r in range(rows):
        for c in range(cols):
            edge = r in (0, rows - 1) or c in (0, cols - 1)
            v = values[r][c]
            label = short(v) if labels is None else labels[r][c]
            fill = "#FFFFFF" if v is None else heat(v)
            parts.append(square(x + c * side, y + r * side, side - 3, side - 3, fill,
                                INK if edge else EDGE, label, width=1.6 if edge else 1.1,
                                size=size))
    return parts


def jacobi(values, steps):
    g = [row[:] for row in values]
    rows, cols = len(g), len(g[0])
    for _ in range(steps):
        new = [row[:] for row in g]
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                new[r][c] = 0.25 * (g[r - 1][c] + g[r + 1][c] + g[r][c - 1] + g[r][c + 1])
        g = new
    return g


N = 6
START = [[100.0] * N] + [[0.0] * N for _ in range(N - 1)]
STEP1 = jacobi(START, 1)
STEP2 = jacobi(START, 2)
FINAL = jacobi(START, 400)

# 19.1  The plate: fixed boundary cells, unknown interior cells.
side = 46
unknown = [[None if 0 < r < N - 1 and 0 < c < N - 1 else START[r][c] for c in range(N)]
           for r in range(N)]
labels = [["?" if unknown[r][c] is None else short(unknown[r][c]) for c in range(N)]
          for r in range(N)]
parts = [zone(32, 96, 340, 340, "6 × 6 plate"), *plate(64, 128, unknown, side, labels)]
for c in range(N):
    parts.append(mono(64 + c * side + 21, 122, str(c), "middle", MUTED))
for r in range(N):
    parts.append(mono(52, 128 + r * side + 27, str(r), "middle", MUTED))
parts += [square(398, 140, 30, 30, heat(100), INK, width=1.6),
          note(440, 152, "boundary cell, top edge:"),
          note(440, 168, "held at 100 degrees"),
          square(398, 196, 30, 30, heat(0), INK, width=1.6),
          note(440, 208, "boundary cell, other edges:"),
          note(440, 224, "held at 0 degrees"),
          square(398, 252, 30, 30, "#FFFFFF", EDGE),
          note(440, 264, "interior cell: unknown,"),
          note(440, 280, "rows 1 to 4, columns 1 to 4"),
          note(398, 330, "36 cells: 20 boundary, 16 interior."),
          note(398, 350, "Each interior cell must equal the"),
          note(398, 366, "mean of its 4 neighbors:"),
          note(398, 382, "16 equations, 16 unknowns.")]
save(OUT + "ch19-plate.svg", 640, 460,
     "A 6 by 6 grid of cells. The 6 cells of row 0 hold 100 and are colored red. The other edge "
     "cells, in column 0, column 5 and row 5, hold 0 and are colored blue. The 16 cells of rows 1 "
     "to 4 and columns 1 to 4 hold question marks. A legend explains that the 20 edge cells are "
     "boundary cells with fixed values and the 16 inner cells are interior cells, the unknowns.",
     "The plate as a grid of cells",
     "20 boundary cells hold fixed temperatures; the 16 interior cells are the unknowns.", parts)

# 19.2  The 5-point stencil, checked on cell (1, 1) of the final plate.
cx, cy, s = 154, 236, 76
cross = [((0, 0), "row, col", "store"), ((0, -1), "row−1, col", "client"), ((0, 1), "row+1, col", "client"),
         ((-1, 0), "row, col−1", "client"), ((1, 0), "row, col+1", "client")]
parts = [zone(32, 96, 250, 320, "5-point stencil")]
for (dx, dy), name, kind in cross:
    accent, tint = KINDS[kind]
    parts.append(square(cx + dx * s - 34, cy + dy * s - 34, 68, 68, tint, accent, name, width=1.6))
    if kind == "client":
        parts.append(note(cx + dx * s, cy + dy * s + 22, "× 1/4", "middle", fill=accent))
parts += [note(48, 372, "weight 1/4 on each neighbor,"),
          note(48, 390, "0 on the center")]
parts += [zone(302, 96, 306, 320, "cell (1, 1), final plate")]
example = [("up", "u(0, 1)", FINAL[0][1]), ("down", "u(2, 1)", FINAL[2][1]),
           ("left", "u(1, 0)", FINAL[1][0]), ("right", "u(1, 2)", FINAL[1][2])]
for k, (where, name, value) in enumerate(example):
    y = 140 + k * 34
    parts += [note(322, y, where), mono(386, y, name), mono(586, y, f"{value:.2f}", "end")]
total = sum(v for _, _, v in example)
parts += [line(322, 272, 590, 272, INK, 1.2),
          mono(322, 296, "sum"), mono(586, 296, f"{total:.2f}", "end"),
          mono(322, 324, "× 1/4"), mono(586, 324, f"{total / 4:.2f}", "end", GREEN),
          note(322, 362, f"The mean of the neighbors, {total / 4:.2f},"),
          note(322, 380, "equals the value of the cell itself.")]
save(OUT + "ch19-stencil.svg", 640, 440,
     "Left: a cross of 5 cells, the center (r, c) and its neighbors (r-1, c) above, (r+1, c) "
     "below, (r, c-1) to the left and (r, c+1) to the right, with weight 1/4 on each neighbor and "
     "0 on the center. Right: cell (1, 1) of the final plate: up 100, down 22.35, left 0, right "
     "59.47, sum 181.82, times 1/4 gives 45.45, the value of the cell.",
     "The 5-point stencil",
     "Each interior value is the mean of its 4 neighbors: 4 reads and 4 operations.", parts)

# 19.3  The first steps of the plate, and the plate after 100 steps.
parts = []
for k, (steps, title) in enumerate([(0, "step 0"), (1, "step 1"), (2, "step 2"),
                                     (100, "step 100")]):
    x = 44 + (k % 2) * 300
    y = 110 + (k // 2) * 300
    parts.append(text(x + 125, y, title, "name", "middle"))
    parts += plate(x, y + 14, jacobi(START, steps), 43, size=11)
save(OUT + "ch19-steps.svg", 640, 690,
     "4 copies of the 6 by 6 plate colored by temperature. Step 0: top row 100, all else 0. Step "
     "1: the 4 cells of row 1 are 25. Step 2: row 1 reads 31.25, 37.5, 37.5, 31.25 and row 2 reads "
     "6.25 in each of its 4 interior cells. Step 100: rows 1 to 4 read 45.45, 59.47, 59.47, 45.45 / "
     "22.35, 32.95, 32.95, 22.35 / 10.98, 17.05, 17.05, 10.98 / 4.55, 7.20, 7.20, 4.55.",
     "Jacobi iteration on the plate",
     "Each step replaces every interior cell by the mean of its old neighbors.", parts)

# 19.4  Updating in place: cell (1, 2) reads a neighbour that step 1 has already overwritten.
parts = [zone(32, 96, 280, 190, "Jacobi: old values"),
         zone(328, 96, 280, 190, "in place: (1, 1) written first")]
for k, (x, left_value, result, kind) in enumerate([(52, "0", "25", "good"),
                                                   (348, "25", "31.25", "danger")]):
    cells = [("100", 1, 0), ("0", 1, 2), (left_value, 0, 1), ("0", 2, 1), ("?", 1, 1)]
    for label, gx, gy in cells:
        centre = gx == 1 and gy == 1
        accent, tint = KINDS["store" if centre else ("danger" if label == "25" and k else
                                                     "client")]
        parts.append(square(x + gx * 46, 120 + gy * 46, 42, 42, tint, accent, label, width=1.5))
    formula = f"(100+0+{left_value}+0)/4"
    parts += [mono(x + 148, 150, "cell (1, 2):"), mono(x + 148, 176, formula),
              mono(x + 148, 204, f"= {result}", fill=KINDS[kind][0])]
parts += [text(32, 318, "the same step on 2 threads", "name")]
lanes = [("thread of (1, 1)", 340, [(150, 290, "read 100, 0, 0, 0", "client"),
                                    (296, 390, "write 25", "danger")]),
         ("thread of (1, 2)", 384, [(150, 250, "load", "idle"),
                                    (360, 540, "read left: 25", "danger")])]
for name, y, bars in lanes:
    parts.append(note(32, y + 16, name))
    for x0, x1, label, kind in bars:
        accent, tint = KINDS[kind]
        parts += [f'  <rect x="{x0}" y="{y}" width="{x1 - x0}" height="24" rx="6" fill="{tint}" '
                  f'stroke="{accent}" stroke-width="1.3"/>',
                  text((x0 + x1) / 2, y + 16, label, "pill", "middle", accent)]
parts += [f'  <path d="M375,365 V381" stroke="{RED}" stroke-width="1.8" '
          f'marker-end="url(#head-danger)"/>',
          note(32, 440, "The write of (1, 1) lands before the read of (1, 2). The thread reads the "
                        "new value 25 instead of"),
          note(32, 458, "the old 0, and stores 31.25 in place of 25. Which happens first depends "
                        "on the scheduling of threads.")]
save(OUT + "ch19-inplace.svg", 640, 480,
     "Left: cell (1, 2) in step 1 computed from the old grid, (100 + 0 + 0 + 0) / 4 = 25. Right: "
     "the same cell computed in place after cell (1, 1) was already overwritten with 25: (100 + 0 + "
     "25 + 0) / 4 = 31.25. Below, a timeline of 2 threads: the thread of cell (1, 1) writes 25; the "
     "thread of cell (1, 2) then reads that 25 as its left neighbor.",
     "Updating in place",
     "In place, a thread can read a neighbor that the same step has already overwritten.", parts)

# 19.5  The work of 1 thread: cell (2, 2) in step 2.
side = 40
parts = [zone(32, 96, 270, 300, "buffer current: step 1"), zone(338, 96, 270, 300, "buffer next: step 2")]
parts += plate(52, 124, STEP1, side, size=11)
after = [row[:] for row in STEP1]
after = [[START[r][c] if r in (0, N - 1) or c in (0, N - 1) else None for c in range(N)]
         for r in range(N)]
after[2][2] = STEP2[2][2]
labels = [["" if v is None else short(v) for v in row] for row in after]
fills = [[0.0 if v is None else v for v in row] for row in after]
parts += plate(358, 124, fills, side, labels, size=11)
for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    parts.append(outline(52 + (2 + dc) * side - 2, 124 + (2 + dr) * side - 2, side + 1, side + 1,
                         BLUE))
parts += [outline(52 + 2 * side - 2, 124 + 2 * side - 2, side + 1, side + 1, AMBER, 1.6, "4 3"),
          outline(358 + 2 * side - 2, 124 + 2 * side - 2, side + 1, side + 1, GREEN),
          f'  <path d="M{52 + 6 * side + 4},{124 + 2.5 * side - 2} H{358 - 6}" '
          f'stroke="{WIRE}" stroke-width="1.8" fill="none" marker-end="url(#head)"/>',
          note(52, 382, "4 reads: 25 + 0 + 0 + 0"),
          note(358, 382, "1 write: 25 / 4 = 6.25"),
          note(32, 424, "Thread (2, 2) reads only buffer current and writes only its own cell of "
                        "buffer next: 4 reads, 1 write.")]
save(OUT + "ch19-thread.svg", 640, 446,
     "Left: buffer current, the plate after step 1, with row 1 at 25. The 4 neighbors of cell (2, 2) are "
     "outlined: 25 above, 0 below, 0 left, 0 right. An arrow leads to buffer next on the right, where "
     "cell (2, 2) receives 6.25. A note counts 4 reads from global memory and 1 write.",
     "The work of 1 thread",
     "Cell (2, 2) in step 2: 4 reads from global memory, 1 write.", parts)

# 19.6  A second buffer without the boundary.
zeroed = [[0.0] * N] + [row[:] for row in STEP1[1:]]
parts = [zone(32, 96, 270, 300, "buffer next after step 1"), zone(338, 96, 270, 300, "step 2")]
parts += plate(52, 124, zeroed, 40, size=11)
for c in range(N):
    parts.append(outline(52 + c * 40 - 1, 123, 39, 39, RED, 2.0))
parts += [mono(358, 150, "cell (1, 1) reads:"),
          mono(358, 176, "up 0 (should be 100)", fill=RED),
          mono(358, 198, "down 0, left 0, right 25"),
          line(358, 214, 590, 214, INK, 1.0),
          mono(358, 238, "(0 + 0 + 0 + 25) / 4"),
          mono(358, 260, "= 6.25", fill=RED),
          mono(358, 296, "correct:"),
          mono(358, 318, "(100 + 0 + 0 + 25) / 4"),
          mono(358, 340, "= 31.25", fill=GREEN),
          note(52, 382, "The kernel never writes row 0,"),
          note(52, 398, "so it stays at 0 in buffer next.")]
save(OUT + "ch19-boundary-bug.svg", 640, 420,
     "Left: buffer next after step 1 when it was created filled with 0: row 1 holds 25 but the top row "
     "holds 0 instead of 100. Right: step 2 reads that buffer; cell (1, 1) gets (0 + 0 + 0 + 25) / 4 "
     "= 6.25 instead of (100 + 0 + 0 + 25) / 4 = 31.25.",
     "A buffer without the boundary",
     "Step 2 reads a cold top edge: cell (1, 1) becomes 6.25 in place of 31.25.", parts)

# 19.7  The invariant of the ping-pong loop.
parts = [text(150, 106, "start of pass k", "name", "middle"),
         text(500, 106, "after the launch, then exchange", "name", "middle")]
for k in range(3):
    y = 124 + k * 92
    parts += [note(32, y + 30, f"pass {k}")]
    for x, name, content, kind in [(90, "current", f"u({k})", "client"),
                                   (210, "next", f"u({max(k - 1, 0)})", "idle"),
                                   (400, "current", f"u({k + 1})", "good"),
                                   (520, "next", f"u({k})", "idle")]:
        accent, tint = KINDS[kind]
        parts.append(square(x, y, 100, 52, tint, accent, f"{name}: {content}", width=1.5))
    parts.append(f'  <path d="M312,{y + 26} H396" stroke="{WIRE}" stroke-width="1.8" fill="none" '
                 f'marker-end="url(#head)"/>')
    parts.append(tag(318, y + 4, "read current", "client"))
    parts.append(tag(318, y + 32, "write next", "good"))
parts += [note(32, 418, "Invariant: at the start of pass k, buffer current holds the Jacobi grid u(k) in "
                        "every cell,"),
          note(32, 436, "and every boundary cell of buffer next holds its given value. Pass 0: "
                        "both buffers hold u(0).")]
save(OUT + "ch19-pingpong.svg", 640, 456,
     "3 rows, one per pass. Before pass 0, buffer current holds u0 and buffer next holds u0. The kernel "
     "reads current and writes the interior of next, which then holds u1; the names are exchanged, so current "
     "holds u1 and next holds u0. Before pass 1, current holds u1; after the launch and the exchange, current holds "
     "u2 and next holds u1. A note states the invariant: at the start of pass k, current holds u(k) and next "
     "holds the boundary values.",
     "Ping-pong buffers and their invariant",
     "Each pass reads current, writes next, then exchanges the 2 names.", parts)

# Measured: the convergence runs and the 4096 x 4096 timings quoted in the chapter.
SMALL = [(1, 8.33), (2, 3.90), (3, 2.52), (4, 1.82), (10, 0.448), (20, 0.0538), (40, 7.75e-4)]
RMS = [(1000, 7.49e-3), (2000, 2.06e-3), (3000, 5.95e-4), (4000, 1.71e-4), (5000, 4.95e-5),
       (6000, 1.44e-5), (7000, 4.57e-6), (8000, 1.79e-6), (9000, 5.59e-7)]


def log_chart(left, right, top, bottom, xmax, lo, hi, points, rho, xticks, xlabel, title):
    def px(x):
        return left + (right - left) * x / xmax

    def py(v):
        return bottom - (bottom - top) * (math.log10(v) - lo) / (hi - lo)

    parts = [line(left, bottom, right, bottom, INK, 1.2), line(left, top, left, bottom, INK, 1.2),
             text((left + right) / 2, top - 16, title, "name", "middle")]
    for e in range(lo, hi + 1):
        y = py(10.0 ** e)
        parts += [line(left, y, right, y, EDGE, 0.8), mono(left - 6, y + 4, f"1e{e}", "end", MUTED)]
    for x in xticks:
        parts.append(mono(px(x), bottom + 16, f"{x:,}", "middle", MUTED))
    parts.append(note((left + right) / 2, bottom + 34, xlabel, "middle"))
    x0, v0 = points[-3]
    model = [(x, v0 * rho ** (x - x0)) for x in range(int(points[0][0]), int(xmax) + 1,
                                                      max(1, int(xmax) // 60))]
    model = [(x, v) for x, v in model if 10.0 ** lo <= v <= 10.0 ** hi]
    trace = " ".join(("M" if k == 0 else "L") + f"{px(x):.1f},{py(v):.1f}"
                     for k, (x, v) in enumerate(model))
    parts.append(f'  <path d="{trace}" fill="none" stroke="{AMBER}" stroke-width="2" '
                 f'stroke-dasharray="5 4"/>')
    for x, v in points:
        parts.append(f'  <circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="4.5" '
                     f'fill="{KINDS["client"][1]}" stroke="{BLUE}" stroke-width="1.8"/>')
    return parts


parts = log_chart(78, 300, 120, 340, 40, -4, 1, SMALL, math.cos(math.pi / 5), [0, 10, 20, 30, 40],
                  "steps", "6 × 6 plate, n = 5")
parts += log_chart(388, 610, 120, 340, 9000, -7, -2, RMS, math.cos(math.pi / 63),
                   [0, 3000, 6000, 9000], "steps", "64 × 64 plate, n = 63")
parts += [tag(78, 400, "measured RMS change per step", "client"),
          tag(300, 400, "predicted: × cos(π / n) per step", "store"),
          note(78, 436, "0.809 per step on the 6 × 6 plate; 0.998757 per step, or 0.288 per 1,000 "
                        "steps, on the 64 × 64 plate.")]
save(OUT + "ch19-convergence.svg", 640, 456,
     "2 charts with a logarithmic vertical axis of RMS change per step. Left, the 6 by 6 plate, "
     "steps 1 to 40: the measured points fall on a straight line of slope 0.809 per step, cos(pi / "
     "5). Right, the 64 by 64 plate, steps 1,000 to 9,000: 9 measured points from 7.49e-3 to "
     "5.59e-7 follow a dashed line of 0.288 per 1,000 steps, cos(pi / 63) to the power 1,000.",
     "Convergence of Jacobi iteration",
     "The RMS change shrinks by cos(π / n) per step: a straight line on a log scale.", parts)

# 19.9  Arithmetic intensity of a step against the ridge points of Chapter 2.
left, right = 150, 560


def ix(v):
    return left + (right - left) * (math.log10(v) + 1) / (math.log10(200) + 1)


RIDGES = [("A100", 10), ("H100", 20), ("Orin NX", 24), ("AGX Thor", 30), ("RTX 5090", 58),
          ("RTX 4090", 82), ("DGX Spark", 114)]
top = 112
parts = [note(left - 10, top - 10, "ridge point of each GPU, FLOP per byte", "start")]
for k, (name, v) in enumerate(RIDGES):
    y = top + k * 34
    accent, tint = KINDS["client"]
    parts += [note(left - 12, y + 17, name, "end"),
              f'  <rect x="{left}" y="{y}" width="{ix(v) - left:.1f}" height="24" rx="5" '
              f'fill="{tint}" stroke="{accent}" stroke-width="1.2"/>',
              mono(ix(v) + 6, y + 16, f"{v}")]
bottom = top + len(RIDGES) * 34
parts.append(line(left, bottom + 4, right, bottom + 4, INK, 1.2))
for v in (0.1, 1, 10, 100):
    parts += [line(ix(v), bottom, ix(v), bottom + 8, INK, 1.2),
              mono(ix(v), bottom + 24, f"{v:g}", "middle", MUTED)]
x = ix(0.5)
parts += [f'  <path d="M{x:.1f},{top - 4} V{bottom + 4}" stroke="{RED}" stroke-width="2" '
          f'stroke-dasharray="5 4"/>',
          f'  <circle cx="{x:.1f}" cy="{bottom + 4}" r="6" fill="{RED}"/>',
          note(x, bottom + 44, "Jacobi step: 0.5 FLOP per byte", "middle", fill=RED),
          note(left, bottom + 76, "Ridge point = peak FLOP/s ÷ memory bandwidth (Chapter 2). Every "
                                  "bar reaches far"),
          note(left, bottom + 94, "past 0.5: the step is memory-bound on each of these GPUs.")]
save(OUT + "ch19-intensity.svg", 640, 470,
     "A horizontal logarithmic axis of FLOP per byte from 0.1 to 200. A red mark at 0.5 FLOP per "
     "byte is the Jacobi step. Marks for the ridge points of 7 GPUs lie between 10 and 114: A100 10, "
     "H100 20, Orin NX 24, AGX Thor 30, RTX 5090 58, RTX 4090 82, DGX Spark 114. The region left of "
     "each ridge point is memory-bound.",
     "Arithmetic intensity of a Jacobi step",
     "4 FLOP per 8 bytes: 20 to 230 times below the ridge points of current GPUs.", parts)

# 19.10  Effective bandwidth on the 4096 x 4096 grid, Orin NX.
SPEEDS = [("copy kernel", 76.4, "idle"), ("Jacobi, C++", 68.3, "client"),
          ("Jacobi, Rust", 57.8, "outside")]
parts = []
scale = 430 / 110
for k, (name, gbs, kind) in enumerate(SPEEDS):
    y = 116 + k * 58
    accent, tint = KINDS[kind]
    parts += [note(40, y + 22, name),
              f'  <rect x="140" y="{y}" width="{gbs * scale:.1f}" height="34" rx="5" fill="{tint}" '
              f'stroke="{accent}" stroke-width="1.4"/>',
              mono(150, y + 22, f"{gbs:.1f} GB/s", fill=accent)]
for value, label, colour, y in [(102, "rated 102 GB/s", RED, 318),
                                (76.4 * 8 / 9, "9 bytes per cell: 67.9 GB/s", AMBER, 336)]:
    x = 140 + value * scale
    parts += [f'  <path d="M{x:.1f},104 V300" stroke="{colour}" stroke-width="1.6" '
              f'stroke-dasharray="5 4"/>',
              note(x, y, label, "middle", fill=colour)]
parts += [note(40, 372, "Effective bandwidth = 8 bytes per cell ÷ time of a step; a step moves at "
                        "least 134.2 MB."),
          note(40, 390, "The C++ step runs where 9 bytes per cell at the copy's rate would put it.")]
save(OUT + "ch19-bandwidth.svg", 640, 410,
     "Horizontal bars of effective bandwidth on a 4096 by 4096 grid, Orin NX: copy kernel 76.4 GB/s, "
     "Jacobi step in C++ 68.3 GB/s, Jacobi step in Rust 57.8 GB/s. A dashed line marks the rated 102 "
     "GB/s. A second dashed line marks 67.9 GB/s, the rate predicted for 9 bytes per cell.",
     "Effective bandwidth of a Jacobi step",
     "4096 × 4096 cells on a Jetson Orin NX rated at 102 GB/s.", parts)

# 19.11  The rows read by 1 warp are the rows of its neighbours.
side = 11
x0, y0 = 140, 112
parts = []
for r in range(10):
    for c in range(34):
        outside = r in (0, 9)
        warp3 = r in (3, 4, 5)
        own = r == 4 and 1 <= c <= 32
        if r == 4 and c in (0, 33):
            kind = "store"
        elif own:
            kind = "good"
        elif warp3 and 1 <= c <= 32:
            kind = "client"
        elif outside:
            kind = "idle"
        else:
            kind = None
        if kind is None:
            fill, stroke = "#FFFFFF", EDGE
        else:
            stroke, fill = KINDS[kind]
        if outside and c in (0, 33):
            fill, stroke = PAPER, EDGE
        parts.append(f'  <rect x="{x0 + c * side}" y="{y0 + r * side * 2}" width="{side - 2}" '
                     f'height="{2 * side - 4}" rx="2" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="0.8"/>')
for r in range(10):
    label = "row above the block" if r == 0 else ("row below the block" if r == 9 else
                                                    f"warp {r - 1}")
    parts.append(note(x0 - 8, y0 + r * 2 * side + 15, label, "end"))
parts += [note(x0 + 34 * side + 8, y0 + 8 * side + 15, "warp 3 reads"),
          note(x0 + 34 * side + 8, y0 + 8 * side + 31, "3 segments"),
          note(32, 380, "The block of 32 × 8 threads computes rows 1 to 8 (warps 0 to 7). Warp 3 "
                        "reads the rows of warps 2"),
          note(32, 398, "and 4 as well as its own. The block as a whole touches 10 rows of the old "
                        "grid: 10 / 8 = 1.25"),
          note(32, 416, "old values per cell computed, if each row segment comes from memory "
                        "once.")]
save(OUT + "ch19-warp-reads.svg", 640, 436,
     "A block of 32 by 8 threads drawn as 8 rows of a grid. For warp 3, its 3 row segments are "
     "highlighted: the row of warp 2 above, its own row with 1 extra cell at each end, and the row "
     "of warp 4 below. The block as a whole touches 10 rows of the old grid, its 8 rows and 1 on "
     "each side, so each old value is fetched from memory about 1.25 times per cell written.",
     "The rows read by a block",
     "Neighboring warps read each other's rows: 8 rows of results need 10 old rows.",
     parts)
