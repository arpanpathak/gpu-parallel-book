"""The figures of chapter 18 (Part V), drawn with figkit and figparts.

    python3 tools/draw_ch18.py

The measured values in the last 3 figures are the ones printed by code/oxide/ch18_stencil in the
run quoted in the chapter.
"""
import math

from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, arrow, line, mono, note, save, tag,
                      text, wire, zone)

OUT = "src/figures/"


def heat(t):
    """A colour for a temperature t from 0 (cool blue) through yellow to 100 (hot red)."""
    stops = [(0, (214, 228, 247)), (30, (190, 222, 214)), (60, (248, 222, 150)),
             (100, (236, 140, 108))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return "#" + "".join(f"{round(a + f * (b - a)):02x}" for a, b in zip(c0, c1))
    return "#ec8c6c"


def square(x, y, w, h, fill, stroke, label="", cls="mono", ink=INK, width=1.2, dash=""):
    """A cell with a fill, an edge and an optional centred label."""
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", ink))
    return "\n".join(parts)


def grid(x, y, values, side, fmt="{:.2f}", interior_only=False):
    """A grid of temperatures coloured by value; boundary cells get a darker edge."""
    rows, cols = len(values), len(values[0])
    parts = []
    for r in range(rows):
        for c in range(cols):
            edge = r in (0, rows - 1) or c in (0, cols - 1)
            v = values[r][c]
            label = "" if v is None else fmt.format(v)
            fill = "#FFFFFF" if v is None else heat(v)
            stroke = INK if edge else EDGE
            parts.append(square(x + c * side, y + r * side, side - 3, side - 3, fill, stroke,
                                label, width=1.6 if edge else 1.1))
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


START = [[100.0] * 5] + [[0.0] * 5 for _ in range(4)]

# 18.1  The plate as a grid: fixed boundary cells, unknown interior cells.
unknown = [[100.0] * 5] + [[0.0, None, None, None, 0.0] for _ in range(3)] + [[0.0] * 5]
parts = [zone(32, 96, 300, 300, "5 x 5 grid"), *grid(62, 124, unknown, 52, "{:.0f}")]
for r in range(1, 4):
    for c in range(1, 4):
        parts.append(text(62 + c * 52 + 24.5, 124 + r * 52 + 30, "?", "name", "middle", MUTED))
parts += [square(370, 140, 30, 30, heat(100), INK, width=1.6),
          note(412, 160, "boundary cell, top edge: held at 100"),
          square(370, 190, 30, 30, heat(0), INK, width=1.6),
          note(412, 210, "boundary cell, other edges: held at 0"),
          square(370, 240, 30, 30, "#FFFFFF", EDGE),
          note(412, 260, "interior cell: unknown, computed"),
          note(370, 310, "Each interior cell must equal"),
          note(370, 328, "the mean of its 4 neighbours."),
          note(370, 356, "9 unknowns, 9 equations.")]
save(OUT + "ch18-grid.svg", 640, 420,
     "A 5 by 5 grid. The top row holds 100 in each cell, the other 3 edges hold 0. The 9 inner cells "
     "are marked with question marks: they are the unknowns. A legend explains boundary and interior "
     "cells, and notes that each interior cell must equal the mean of its 4 neighbours, which gives "
     "9 equations for 9 unknowns.",
     "A heated plate as a grid of cells",
     "The edges hold fixed temperatures; the 9 inner cells are the unknowns.", parts)

# 18.2  The 5-point stencil, and 1 update worked with numbers.
cx, cy, s = 152, 222, 78
cross = [((0, 0), "r, c", "store"), ((0, -1), "r-1, c", "client"), ((0, 1), "r+1, c", "client"),
         ((-1, 0), "r, c-1", "client"), ((1, 0), "r, c+1", "client")]
parts = [zone(32, 96, 244, 300, "stencil")]
for (dx, dy), name, kind in cross:
    accent, tint = KINDS[kind]
    parts.append(square(cx + dx * s - 36, cy + dy * s - 36, 72, 72, tint, accent, name, width=1.6))
parts += [note(48, 352, "weight 1/4 on each blue neighbour,"),
          note(48, 370, "0 on the amber centre")]
parts += [zone(300, 96, 308, 300, "1 update, step 2, cell (1, 1)")]
example = [("up", "u[0][1]", "100.00"), ("down", "u[2][1]", "0.00"), ("left", "u[1][0]", "0.00"),
           ("right", "u[1][2]", "25.00")]
for k, (where, name, value) in enumerate(example):
    y = 140 + k * 34
    parts += [note(320, y, where), mono(380, y, name), mono(560, y, value, "end")]
parts += [line(320, 272, 588, 272, INK, 1.2),
          mono(320, 296, "sum"), mono(560, 296, "125.00", "end"),
          mono(320, 324, "× 1/4"), mono(560, 324, "31.25", "end", KINDS["good"][0]),
          note(320, 360, "The neighbours come from step 1; the new"),
          note(320, 378, "value goes to the other buffer.")]
save(OUT + "ch18-stencil.svg", 640, 420,
     "Left: a cross of 5 cells, the centre (r, c) and its 4 neighbours (r-1, c), (r+1, c), (r, c-1) "
     "and (r, c+1), with weight 1/4 on each neighbour and 0 on the centre. Right: the update of cell "
     "(1, 1) at step 2: up 100, down 0, left 0, right 25, sum 125, times 1/4 is 31.25.",
     "The 5-point stencil",
     "A new value is the mean of the 4 old neighbours: 4 reads, 4 operations.",
     parts)


def short(v):
    """Whole numbers without decimals, others with 2."""
    return f"{v:.0f}" if v == int(v) else f"{v:.2f}"


# 18.3  The first steps of the 5 x 5 example, and the converged grid.
parts = []
for k, (steps, title) in enumerate([(0, "step 0"), (1, "step 1"), (2, "step 2"),
                                     (200, "step 200")]):
    x = 40 + (k % 2) * 300
    y = 114 + (k // 2) * 270
    values = [[short(v) for v in row] for row in jacobi(START, steps)]
    cells = jacobi(START, steps)
    parts.append(text(x + 108, y, title, "name", "middle"))
    for r in range(5):
        for c in range(5):
            edge = r in (0, 4) or c in (0, 4)
            parts.append(square(x + c * 44, y + 12 + r * 44, 41, 41, heat(cells[r][c]),
                                INK if edge else EDGE, values[r][c], width=1.6 if edge else 1.1))
save(OUT + "ch18-steps.svg", 640, 620,
     "4 grids of 5 by 5 cells, coloured by temperature. Step 0: top row 100, the rest 0. Step 1: the "
     "3 cells under the hot edge are 25. Step 2: those cells are 31.25, 37.50 and 31.25, and the row "
     "below is 6.25 in each cell. Step 200: the inner cells are 42.86, 52.68, 42.86 / 18.75, 25, "
     "18.75 / 7.14, 9.82, 7.14.",
     "Jacobi iteration on the 5 x 5 grid",
     "Each step replaces every inner cell by the mean of its old neighbours.",
     parts)

# 18.4  Updating in place: the result depends on the order of the threads.
parts = [zone(32, 96, 280, 274, "old values, step 1"),
         *grid(62, 124, jacobi(START, 1), 46, "{:.0f}")]
parts += [zone(336, 96, 272, 274, "cell (1, 2) at step 2")]
rows = [("(1, 1) not yet written", "(100 + 0 + 25 + 25) / 4", "37.50", "good"),
        ("(1, 1) already written", "(100 + 0 + 31.25 + 25) / 4", "39.06", "danger")]
for k, (case, formula, value, kind) in enumerate(rows):
    y = 140 + k * 92
    parts += [note(356, y, case), mono(356, y + 22, formula),
              mono(588, y + 50, value, "end", KINDS[kind][0])]
parts += [note(32, 398, "In place, a thread may read a neighbour that another thread has already "
                        "replaced. The result"),
          note(32, 416, "then depends on the order in which the threads run, which the hardware "
                        "does not fix.")]
save(OUT + "ch18-inplace.svg", 640, 436,
     "Left: the 5 by 5 grid after step 1, with 25 in the 3 cells under the hot edge. Right: cell (1, 2) "
     "at step 2 computed 2 ways. If cell (1, 1) still holds its old value 25, the result is "
     "(100 + 0 + 25 + 25) / 4 = 37.50. If cell (1, 1) has already been replaced by 31.25, the result is "
     "(100 + 0 + 31.25 + 25) / 4 = 39.06.",
     "Why a step cannot update the grid in place",
     "Two buffers, one read and one written, make every thread see the same old values.", parts)

# 18.5  Ping-pong buffers: the roles of a and b swap after every step.
parts = []
for k, (src, dst) in enumerate([("a", "b"), ("b", "a"), ("a", "b")]):
    y = 110 + k * 82
    parts += [text(40, y + 30, f"step {k + 1}", "name"),
              square(150, y, 140, 50, KINDS["client"][1], KINDS["client"][0],
                     f"{src}: read", "mono", width=1.6),
              square(420, y, 140, 50, KINDS["good"][1], KINDS["good"][0],
                     f"{dst}: written", "mono", width=1.6),
              wire([(290, y + 25), (418, y + 25)], label="jacobi")]
    if k < 2:
        parts.append(note(355, y + 70, "swap(&mut a, &mut b)", "middle"))
parts += [note(40, 372, "std::mem::swap exchanges the 2 DeviceBuffer handles, 40 bytes each: a "
                        "device address, 2 sizes, 2 pointers."),
          note(40, 390, "No grid data moves. After an even number of steps the newest grid is "
                        "in a again.")]
save(OUT + "ch18-pingpong.svg", 640, 410,
     "3 rows, steps 1 to 3. Step 1 reads buffer a and writes buffer b; between steps, swap(&mut a, "
     "&mut b) exchanges the names; step 2 reads b and writes a; step 3 reads a and writes b. A note "
     "says that the swap exchanges the 2 handles and moves no grid data.",
     "Ping-pong buffers",
     "Each step reads 1 buffer and writes the other; the 2 names swap between steps.", parts)

# 18.6  The tile of a 32 x 8 block, with its 1-cell halo: 34 x 10 = 340 cells.
side = 16
x0, y0 = 48, 120
parts = []
for r in range(10):
    for c in range(34):
        halo = r in (0, 9) or c in (0, 33)
        corner = r in (0, 9) and c in (0, 33)
        kind = "idle" if corner else ("store" if halo else "client")
        accent, tint = KINDS[kind]
        parts.append(f'  <rect x="{x0 + c * side}" y="{y0 + r * side}" width="{side - 2}" '
                     f'height="{side - 2}" rx="2" fill="{tint}" stroke="{accent}" '
                     f'stroke-width="0.9"/>')
parts += [tag(48, 310, "256 cells written by the block (32 x 8)", "client"),
          tag(48, 340, "80 halo cells read, written by other blocks", "store"),
          tag(48, 370, "4 corner cells loaded but never read", "idle"),
          note(400, 318, "340 loads from global memory for"),
          note(400, 336, "256 results: 1.33 loads per result,"),
          note(400, 354, "against 4 reads per result when every"),
          note(400, 372, "thread reads its neighbours itself.")]
save(OUT + "ch18-tile.svg", 640, 400,
     "A tile of 34 by 10 small squares. The inner 32 by 8 squares, blue, are the 256 cells the block "
     "writes. A ring of 80 amber squares around them is the halo, cells that neighbouring blocks "
     "write. The 4 corner squares, grey, are loaded but never read by a 5-point stencil. A note says "
     "340 loads serve 256 results, 1.33 loads per result, against 4 reads per result without the tile.",
     "A 32 x 8 block and its halo in shared memory",
     "The block loads a 34 x 10 tile once; threads read neighbours from it.",
     parts)

# Measured: the 64 x 64 run and the 4096 x 4096 timings quoted in the chapter.
RMS = [(1000, 7.49e-3), (2000, 2.06e-3), (3000, 5.95e-4), (4000, 1.71e-4), (5000, 4.95e-5),
       (6000, 1.44e-5), (7000, 4.57e-6), (8000, 1.79e-6), (9000, 5.59e-7)]
FIELD = [[100.00] * 8,
         [0.00, 47.77, 65.71, 71.49, 71.49, 65.71, 47.77, 0.00],
         [0.00, 25.33, 41.16, 47.98, 47.98, 41.16, 25.33, 0.00],
         [0.00, 14.76, 25.53, 30.89, 30.89, 25.53, 14.76, 0.00],
         [0.00, 8.77, 15.53, 19.11, 19.11, 15.53, 8.77, 0.00],
         [0.00, 4.94, 8.84, 10.96, 10.96, 8.84, 4.94, 0.00],
         [0.00, 2.23, 4.01, 4.98, 4.98, 4.01, 2.23, 0.00],
         [0.00] * 8]
SPEEDS = [("copy kernel", 75.3, "idle"), ("jacobi", 58.3, "client"),
          ("jacobi, tiled", 37.8, "store")]

# 18.7  Convergence: the RMS change per step falls by the same factor every 1,000 steps.
left, right, top, bottom = 96, 600, 110, 340
lo, hi = -7, -2


def px(step):
    return left + (right - left) * step / 9000


def py(value):
    return bottom - (bottom - top) * (math.log10(value) - lo) / (hi - lo)


parts = [line(left, bottom, right, bottom, INK, 1.2), line(left, top, left, bottom, INK, 1.2)]
for e in range(lo, hi + 1):
    y = py(10.0 ** e)
    parts += [line(left, y, right, y, EDGE, 0.8), mono(left - 8, y + 4, f"1e{e}", "end", MUTED)]
for s in range(0, 9001, 3000):
    parts.append(mono(px(s), bottom + 18, f"{s:,}", "middle", MUTED))
rho = math.cos(math.pi / 63)
first = RMS[0]
model = [(s, first[1] * rho ** (s - first[0])) for s in range(1000, 9001, 250)]
trace = " ".join(("M" if k == 0 else "L") + f"{px(s):.1f},{py(v):.1f}"
                 for k, (s, v) in enumerate(model))
parts.append(f'  <path d="{trace}" fill="none" stroke="{KINDS["store"][0]}" stroke-width="2" '
             f'stroke-dasharray="5 4"/>')
for s, v in RMS:
    parts.append(f'  <circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="4.5" fill="{KINDS["client"][1]}" '
                 f'stroke="{KINDS["client"][0]}" stroke-width="1.8"/>')
parts += [note(left + 4, top - 14, "RMS change per step"),
          note(right, bottom + 40, "Jacobi steps", "end"),
          tag(110, 270, "measured, every 1,000 steps", "client"),
          tag(110, 300, "cos(π / 63) per step: × 0.288 per 1,000 steps", "store")]
save(OUT + "ch18-convergence.svg", 640, 400,
     "A chart with a logarithmic vertical axis from 1e-7 to 1e-2 and Jacobi steps from 0 to 9,000 on "
     "the horizontal axis. 9 measured points, from 7.49e-3 at step 1,000 to 5.59e-7 at step 9,000, "
     "fall on a straight line. A dashed line, the prediction that the change shrinks by cos(pi / 63) "
     "per step, or 0.288 per 1,000 steps, runs through the points.",
     "Convergence on the 64 x 64 grid",
     "The change shrinks by the same factor every step: a straight line in log scale.",
     parts)

# 18.8  The converged 64 x 64 field, every 9th cell of every 9th row.
side = 56
parts = []
for r, row in enumerate(FIELD):
    for c, v in enumerate(row):
        edge = r in (0, 7) or c in (0, 7)
        parts.append(square(96 + c * side, 100 + r * 38, side - 4, 34, heat(v),
                            INK if edge else EDGE, short(v), width=1.6 if edge else 1.1))
parts += [note(96, 424, "Row 0 is the hot edge at 100; column 0, column 63 and row 63 are held at 0."),
          note(96, 442, "The field is symmetric about the middle column, as the boundary is.")]
save(OUT + "ch18-field.svg", 640, 460,
     "An 8 by 8 table of temperatures coloured from red to blue: the converged 64 by 64 field sampled "
     "every 9th cell of every 9th row. The top row is 100. Row 9 reads 0, 47.77, 65.71, 71.49, 71.49, "
     "65.71, 47.77, 0; the values fall towards the bottom, where row 54 reads 0, 2.23, 4.01, 4.98, "
     "4.98, 4.01, 2.23, 0, and the last row is 0.",
     "The converged temperature field",
     "Every 9th cell of every 9th row of the 64 x 64 grid after 9,000 steps.", parts)

# 18.9  Effective bandwidth on the 4096 x 4096 grid, 8 bytes per cell per step.
parts = []
scale = 420 / 110
for k, (name, gbs, kind) in enumerate(SPEEDS):
    y = 116 + k * 58
    accent, tint = KINDS[kind]
    parts += [note(40, y + 22, name),
              f'  <rect x="140" y="{y}" width="{gbs * scale:.1f}" height="34" rx="5" fill="{tint}" '
              f'stroke="{accent}" stroke-width="1.4"/>',
              mono(148 + gbs * scale, y + 22, f"{gbs:.1f} GB/s")]
x = 140 + 102 * scale
parts += [f'  <path d="M{x:.1f},104 V300" stroke="{KINDS["danger"][0]}" stroke-width="1.6" '
          f'stroke-dasharray="5 4"/>',
          note(x, 318, "rated 102 GB/s", "middle", fill=KINDS["danger"][0]),
          note(40, 352, "1 read and 1 write of 4 bytes per cell: 134.2 MB per step."),
          note(40, 370, "The tiled kernel loads 1.33 cells per result, yet runs slower than the"),
          note(40, 388, "plain kernel, whose neighbour reads the L1 and L2 caches already serve.")]
save(OUT + "ch18-bandwidth.svg", 640, 410,
     "3 horizontal bars of effective bandwidth on a 4096 by 4096 grid: copy kernel 75.3 GB/s, jacobi "
     "58.3 GB/s, jacobi tiled 37.8 GB/s. A dashed line marks the rated memory bandwidth of 102 GB/s. "
     "A note explains that each step moves 134.2 MB, and that the tiled kernel is slower than the "
     "plain one.",
     "Effective bandwidth of 1 Jacobi step",
     "4096 x 4096 cells, 8 bytes per cell per step, on a GPU rated at 102 GB/s.", parts)
