"""The figures of chapter 16 (Part V), drawn with figkit and figparts.

    python3 tools/draw_ch16.py

The speeds in the last 2 figures are the ones measured with code/oxide/ch16_matmul_naive
(the run quoted in the chapter).
"""
from figparts import (EDGE, INK, KINDS, MUTED, arrow, card, legend, line, mono, note, save, tag,
                      text, zone)

OUT = "src/figures/"

# GFLOP/s from the quoted run (the quietest of 6): n -> (f32, f32 unchecked, f64).
MEASURED = {512: (184.4, 182.1, 35.2), 1000: (139.9, 138.9, 34.0), 1024: (134.8, 138.3, 35.3),
            2048: (135.1, 136.1, 35.2), 4096: (129.4, 129.4, None)}
FIXED_1024 = 192.2
PEAK_F32, RATIO = 2400, 32


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def hline(x1, x2, y):
    """A horizontal arrow from x1 to x2 at height y, tip 3 units short."""
    return (f'  <path d="M{x1},{y} H{x2 - 3}" fill="none" stroke="#7F8EA0" stroke-width="1.8"'
            f' stroke-linecap="round" marker-end="url(#head)"/>')


# 16.1  Row-major layout: a 3 x 4 matrix stored as a slice of 12 elements.
cell = 46
parts = [zone(16, 104, 628, 208, "C as a matrix, 3 rows of 4"),
         zone(16, 336, 628, 132, "C as a slice of 12 elements")]
for r in range(3):
    parts.append(mono(84, 163 + r * cell, f"row {r}", "end", MUTED))
    for c in range(4):
        parts.append(box(96 + c * cell, 140 + r * cell, cell - 4, cell - 4, "client" if r == 1 else "idle",
                         f"{r * 4 + c}", (r, c) == (1, 2) or r == 1))
for c in range(4):
    parts.append(mono(117 + c * cell, 288, f"col {c}", "middle", MUTED))
parts += [card(300, 140, 320, 58, "good", "C[1][2] is element 1 × 4 + 2 = 6", "row × width + column"),
          note(300, 230, "The rows lie one after another. Row r starts at", room=320),
          note(300, 248, "r × 4, so column c of row r is at r × 4 + c.", room=320)]
for k in range(12):
    r = k // 4
    parts.append(box(36 + k * 50, 376, 46, 36, "client" if r == 1 else "idle", f"{k}", k == 6 or r == 1))
parts += [tag(36, 424, "row 0: 0 to 3", "idle"), tag(236, 424, "row 1: 4 to 7", "client"),
          tag(436, 424, "row 2: 8 to 11", "idle")]
save(OUT + "ch16-rowmajor.svg", 660, 492,
     "Top: a matrix C of 3 rows and 4 columns whose cells hold their slice index, 0 to 11. Row 1, cells 4 to "
     "7, is highlighted, and the cell in row 1, column 2 holds 6. A card reads C[1][2] is element 1 times 4 "
     "plus 2 = 6. Bottom: the same 12 elements in 1 slice, row 0 at 0 to 3, row 1 at 4 to 7, row 2 at 8 to 11.",
     "Row-major layout", "A matrix is stored row after row in 1 slice.", parts)


# 16.2  A 2-D grid: block (1, 2), thread (5, 3) computes C[21][35] of a 40 x 40 matrix.
parts = [zone(16, 104, 360, 380, "grid of 3 × 3 blocks over C")]
x0, y0, side = 52, 150, 96
for by in range(3):
    for bx in range(3):
        on = (by, bx) == (1, 2)
        parts.append(box(x0 + bx * (side + 6), y0 + by * (side + 6), side, side, "client" if on else "idle",
                         f"({by}, {bx})", on))
covered = 40 / 48 * (3 * side + 12)
parts += [f'  <rect x="{x0 + covered:.1f}" y="{y0}" width="{3 * side + 12 - covered:.1f}" '
          f'height="{3 * side + 12}" fill="#CF4526" opacity="0.10"/>',
          f'  <rect x="{x0}" y="{y0 + covered:.1f}" width="{3 * side + 12}" '
          f'height="{3 * side + 12 - covered:.1f}" fill="#CF4526" opacity="0.10"/>',
          f'  <circle cx="{x0 + 2 * (side + 6) + (3 + 0.5) * side / 16:.1f}" '
          f'cy="{y0 + 1 * (side + 6) + (5 + 0.5) * side / 16:.1f}" r="5" fill="{KINDS["good"][0]}"/>',
          mono(x0, y0 + 3 * side + 36, "blockIdx (y, x); shaded: rows and", "start", MUTED),
          mono(x0, y0 + 3 * side + 54, "columns 40 to 47, outside C", "start", MUTED)]
parts += [card(396, 120, 248, 70, "client", "block (y, x) = (1, 2)", ["blockDim = (16, 16)"]),
          card(396, 206, 248, 70, "outside", "thread (y, x) = (5, 3)", ["threadIdx inside the block"]),
          card(396, 292, 248, 84, "good", "row = 1 × 16 + 5 = 21", ["col = 2 × 16 + 3 = 35"]),
          card(396, 392, 248, 70, "store", "C[21][35] is element", ["21 × 40 + 35 = 875"])]
save(OUT + "ch16-grid.svg", 660, 504,
     "Left: a grid of 3 by 3 blocks of 16 by 16 threads laid over a 40 by 40 matrix C; the last 8 rows and "
     "columns of threads, 40 to 47, are shaded as outside C. Block (1, 2) is highlighted, and a dot marks "
     "thread (5, 3) inside it. Right: row = 1 times 16 plus 5 = 21, column = 2 times 16 plus 3 = 35, and C[21][35] "
     "is element 21 times 40 plus 35 = 875 of the slice.",
     "1 thread, 1 element of C", "Block and thread coordinates give the row and the column.", parts)


# 16.3  3 index spaces: where the row width lives, and when a thread gets no cell.
parts = []
cols = [("idle", "Index1D", ["Chapter 13: 1 dimension", "witness: i", "None if i ≥ len"]),
        ("client", "Index2D<W>", ["W in the type of c", "witness: row × W + col", "None if col ≥ W"]),
        ("good", "Runtime2DIndex", ["width bound by RowWidth", "witness: (row, col)", "None if col ≥ width"])]
for k, (kind, name, lines) in enumerate(cols):
    x = 24 + k * 208
    parts += [card(x, 120, 196, 112, kind, name, lines)]
parts += [zone(16, 262, 628, 168, "the same write in all 3 kernels")]
parts += [mono(40, 312, "if let Some(idx) = <witness>", "start", INK),
          mono(40, 334, "    && let Some(out) = c.get_mut(idx)", "start", INK),
          mono(40, 356, "{", "start", INK),
          mono(40, 378, "    *out = sum;", "start", INK),
          mono(40, 400, "}", "start", INK),
          note(360, 312, "get_mut returns None when the", room=270),
          note(360, 330, "element lies past the end of c.", room=270),
          note(360, 360, "2 threads never hold the same", room=270),
          note(360, 378, "witness, so 2 threads never", room=270),
          note(360, 396, "write the same element.", room=270)]
save(OUT + "ch16-witness.svg", 660, 452,
     "3 cards. Index1D, from Chapter 13: the witness is i, None if i is at least the length. Index2D with "
     "width W: W is part of the type of c, the witness is row times W plus col, None if col is at least W. "
     "Runtime2DIndex: the width is bound on the host by RowWidth, the witness stores row and col, None if col "
     "is at least the width. Below, the same if-let write used by all 3, with the note that get_mut returns "
     "None past the end of c and that 2 threads never hold the same witness.",
     "Index spaces", "A witness names 1 element of c and belongs to 1 thread.", parts)


# 16.4  n = 1,000: 63 blocks per side, 1,008 threads, 8 columns with nothing to do.
parts = [zone(16, 104, 628, 224, "the last block of a row of blocks, n = 1,000")]
for t in range(16):
    col = 992 + t
    inside = col < 1000
    parts.append(box(28 + t * 38, 160, 36, 44, "good" if inside else "danger", "", True))
    parts.append(text(28 + t * 38 + 18, 178, f"{col}", "mono", "middle",
                      KINDS["good" if inside else "danger"][0]))
    parts.append(text(28 + t * 38 + 18, 196, f"x={t}", "sub", "middle", MUTED))
parts += [tag(36, 220, "col 992 to 999: inside C, compute", "good"),
          tag(332, 220, "col 1000 to 1007: col < n fails", "danger"),
          note(36, 264, "1,000 ÷ 16 = 62.5, rounded up to 63 blocks: 63 × 16 = 1,008 threads per row and", room=600),
          note(36, 282, "per column. Block 62 starts at 62 × 16 = 992; its threads x = 8 to 15 fall at", room=600),
          note(36, 300, "columns 1,000 to 1,007, past the last column of C, and the guard stops them.", room=600)]
save(OUT + "ch16-edge.svg", 660, 350,
     "16 cells for the 16 threads of the last block in a row of blocks, at columns 992 to 1,007 with x from 0 "
     "to 15. Columns 992 to 999 are green: inside C. Columns 1,000 to 1,007 are red: the test col < n fails. "
     "The note: 1,000 / 16 = 62.5 rounds up to 63 blocks, 1,008 threads, and block 62 starts at column 992.",
     "The edge of C", "When n is not a multiple of 16, the last block reaches past C.", parts)


# 16.5  The checks on the host before a launch reaches the GPU.
parts = [card(40, 116, 260, 70, "client", "prepare_matmul(config)", ["block = (16, 16, 1)? domain 2?"]),
         arrow(170, 186, 222),
         card(40, 222, 260, 84, "store", "matmul(m, n, k, a, b, c)", ["a.len() ≥ m k", "b.len() ≥ k n, c.len() ≥ m n"]),
         arrow(170, 306, 342),
         card(40, 342, 260, 62, "good", "launch", ["the kernel runs on the GPU"]),
         zone(330, 116, 314, 288, "a launch that claims too much"),
         note(354, 162, "a holds 256 × 256 = 65,536 floats.", room=270),
         note(354, 186, "The call claims m = 256, k = 512:", room=270),
         note(354, 210, "m × k = 131,072 > 65,536.", room=270),
         note(354, 246, "The relation a.len() ≥ m k fails on", room=270),
         note(354, 270, "the CPU, and matmul returns Err.", room=270),
         note(354, 294, "No kernel starts, and no thread", room=270),
         note(354, 318, "reads past the end of a.", room=270),
         tag(354, 356, "checked once per launch", "store")]
save(OUT + "ch16-contract.svg", 660, 428,
     "Left, top to bottom: prepare_matmul checks the block shape (16, 16, 1) and the 2-D domain; the call "
     "matmul checks a.len() at least m k, b.len() at least k n and c.len() at least m n; then the kernel is "
     "launched. Right: with n = 256 and k = 512, a holds 65,536 floats while m times k is 131,072, so the "
     "check fails on the CPU and no kernel starts.",
     "Checks before the launch", "The launch contract tests sizes on the CPU, once per launch.", parts)


# 16.6  Measured GFLOP/s for each size and kernel.
x0, wide, scale = 150, 400, 200
kinds = [("client", "f32"), ("store", "f32 unchecked"), ("outside", "f64"), ("good", "f32 fixed width")]
rows = []
for n, values in MEASURED.items():
    bars = [(kinds[b][0], v) for b, v in enumerate(values) if v is not None]
    if n == 1024:
        bars.append(("good", FIXED_1024))
    rows.append((n, bars))
top, gap, step = 136, 19, 12
heights = [len(bars) * gap + step for _, bars in rows]
bottom = top + sum(heights)
parts = [legend(x0, 104, kinds[:2]), legend(x0 + 250, 104, kinds[2:])]
for g in range(0, 201, 50):
    x = x0 + wide * g / scale
    parts += [line(x, 128, x, bottom, EDGE, 1), mono(x, bottom + 18, f"{g}", "middle", MUTED)]
y = top
for (n, bars), h in zip(rows, heights):
    parts.append(text(x0 - 14, y + len(bars) * gap / 2 + 4, f"n = {n:,}", "sub", "end"))
    for b, (kind, v) in enumerate(bars):
        w = wide * v / scale
        parts += [f'  <rect x="{x0}" y="{y + b * gap}" width="{w:.1f}" height="15" rx="4" '
                  f'fill="{KINDS[kind][0]}"/>',
                  mono(x0 + w + 8, y + b * gap + 12, f"{v:.1f}", "start", INK)]
    y += h
parts += [text(x0 - 14, bottom + 18, "GFLOP/s", "sub", "end"),
          note(40, bottom + 52, "Checked and unchecked f32 differ by no more than 3%, within the spread of", room=600),
          note(40, bottom + 70, "repeated runs. f64 runs at a nearly constant 35 GFLOP/s. With the width 1,024 fixed", room=600),
          note(40, bottom + 88, "at compile time, the kernel ran at 192.2 GFLOP/s against 134.8 at run time.", room=600)]
save(OUT + "ch16-speeds.svg", 660, bottom + 110,
     "Horizontal bars of GFLOP/s for n = 512, 1,000, 1,024, 2,048 and 4,096. f32 checked: 184.4, 139.9, 134.8, "
     "135.1, 129.4. f32 unchecked: 182.1, 138.9, 138.3, 136.1, 129.4. f64: 35.2, 34.0, 35.3, 35.2, and no f64 "
     "run at 4,096. An extra green bar at n = 1,024 shows the fixed-width kernel at 192.2 GFLOP/s.",
     "Measured speed", "1 thread per element of C, on a GPU with 8 SMs and 102 GB/s.", parts)


# 16.7  The 2 precisions against their own peaks.
f32, f64 = MEASURED[2048][0], MEASURED[2048][2]
peak64 = PEAK_F32 / RATIO
parts = [zone(16, 104, 628, 300, "matrices of 2,048 × 2,048")]
for k, (kind, name, v, peak) in enumerate([("client", "f32", f32, PEAK_F32), ("outside", "f64", f64, peak64)]):
    y = 150 + k * 110
    share = v / peak
    parts += [text(40, y + 12, f"{name}: {v} of {peak:,.0f} GFLOP/s", "name"),
              f'  <rect x="40" y="{y + 26}" width="560" height="26" rx="6" fill="#FFFFFF" stroke="{EDGE}"/>',
              f'  <rect x="40" y="{y + 26}" width="{max(560 * share, 4):.1f}" height="26" rx="6" '
              f'fill="{KINDS[kind][0]}"/>',
              mono(40 + max(560 * share, 4) + 10, y + 44, f"{share:.1%}", "start", INK)]
parts += [note(40, 384, f"FP32 peak 2,400 GFLOP/s; the GPU reports 32 times fewer FP64 operations, 75 GFLOP/s.",
               room=590)]
save(OUT + "ch16-precision.svg", 660, 426,
     f"2 bars. f32 at n = 2,048: {f32} of 2,400 GFLOP/s, {f32 / PEAK_F32:.1%} of its peak. f64: {f64} of 75 "
     f"GFLOP/s, {f64 / peak64:.1%} of its peak. The FP64 peak is the FP32 peak divided by the ratio 32 that the "
     "GPU reports.",
     "f32 and f64 against their peaks", "The same kernel is far from 1 peak and close to the other.", parts)
