"""The figures of chapter 17 (Part V), drawn with figkit and figparts.

    python3 tools/draw_ch17.py

Every figure works on the same 4 x 4 example as chapter 16, with tiles of T = 2.
"""
from figparts import EDGE, INK, KINDS, MUTED, must_fit, save, text, width_of, wire

OUT = "src/figures/"

A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
B = [[1, 0, 2, 1], [0, 1, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
C = [[11, 13, 8, 8], [27, 29, 24, 24], [43, 45, 40, 40], [59, 61, 56, 56]]

# 1 colour per thread of a 2 x 2 block: (tint, accent).
THREAD = {(0, 0): ("#DDE3FB", "#5B6BD6"), (0, 1): ("#D7F0E6", "#1F8A70"),
          (1, 0): ("#FBE3D2", "#C8693A"), (1, 1): ("#F9DCE6", "#C2457A")}
PHASE = {0: ("#DDE7FB", "#2F6FDB"), 1: ("#FBF1DC", "#B7791F")}
PLAIN = ("#FFFFFF", EDGE)
FADED = ("#FFFFFF", "#E6EAEF")
RED = KINDS["danger"][0]
GREY = ("#EEF0F3", "#9AA5B1")


def cell(x, y, s, body, colours=PLAIN, faded=False, dashed=False, size=None, bold=False):
    """A square or rectangle with a centred label. `colours` is (fill, stroke)."""
    fill, stroke = FADED if faded else colours
    w, h = (s, s) if not isinstance(s, tuple) else s
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    ink = "#B8C0CA" if faded else (INK if colours in (PLAIN, FADED) else stroke)
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" fill="{fill}"'
             f' stroke="{stroke}" stroke-width="1.3"{dash}/>']
    if body != "":
        body = str(body)
        pt = size or (13 if h >= 30 else 11.5)
        must_fit(body, w - 4, pt, bold=bold)
        weight = "font-weight:700;" if bold else ""
        parts.append(f'  <text x="{x + w / 2:.1f}" y="{y + h / 2 + pt * 0.36:.1f}" text-anchor="middle"'
                     f' style="font-size:{pt}px;{weight}fill:{ink}">{body}</text>')
    return "\n".join(parts)


def grid(x, y, mat, s, gap, style):
    """A matrix of cells; style(r, c) returns (colours, faded)."""
    parts = []
    for r, row in enumerate(mat):
        for c, v in enumerate(row):
            colours, faded = style(r, c)
            parts.append(cell(x + c * (s + gap), y + r * (s + gap), s, v, colours, faded))
    return parts


def box(x, y, w, h, colour, width=2.0, rx=7, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="none"'
            f' stroke="{colour}" stroke-width="{width}"{dashes}/>')


def label(x, y, body, anchor="middle", fill=MUTED, size=12, bold=False, room=None):
    if room:
        must_fit(body, room, size, bold=bold)
    weight = "font-weight:700;" if bold else ""
    return (f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" xml:space="preserve"'
            f' style="font-size:{size}px;{weight}fill:{fill}">{body}</text>')


def bar(x, y, w, h, colours, body="", ink=None):
    fill, stroke = colours
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" fill="{fill}"'
             f' stroke="{stroke}" stroke-width="1.2"/>']
    if body:
        must_fit(body, w - 10, 11.5)
        parts.append(label(x + w / 2, y + h / 2 + 4, body, fill=ink or stroke, size=11.5))
    return "\n".join(parts)


def vline(x, y1, y2, colour=INK, width=2.0, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{colour}"'
            f' stroke-width="{width}"{dashes}/>')


def hline(x1, x2, y, colour=MUTED, width=1.2, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{colour}"'
            f' stroke-width="{width}"{dashes}/>')


def dot(x, y, colour=INK, r=4):
    return f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{colour}"/>'


def shared(x, y, w, h, caption="shared memory"):
    return "\n".join([
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#D9F1E7" stroke="#A9DCC6" stroke-width="1.2"/>',
        label(x + w / 2, y + h - 10, caption, fill="#1F7A5C", size=11.5, bold=True)])


# 17.1  The values needed by block (0, 0).
s, g = 40, 4
ax, bx, cx, y0 = 40, 250, 460, 120
parts = [label(ax + 86, 105, "A", bold=True, fill=INK, size=13),
         label(bx + 86, 105, "B", bold=True, fill=INK, size=13),
         label(cx + 86, 105, "C, block (0, 0)", bold=True, fill=INK, size=13)]
parts += grid(ax, y0, A, s, g, lambda r, c: (PHASE[0], False) if r < 2 else (PLAIN, True))
parts += grid(bx, y0, B, s, g, lambda r, c: (PHASE[1], False) if c < 2 else (PLAIN, True))
for r in range(4):
    for c in range(4):
        if r < 2 and c < 2:
            parts.append(cell(cx + c * (s + g), y0 + r * (s + g), s, f"({r},{c})", THREAD[(r, c)], size=11))
        else:
            parts.append(cell(cx + c * (s + g), y0 + r * (s + g), s, "", PLAIN))
parts += [box(ax - 4, y0 - 4, 4 * s + 3 * g + 8, 2 * s + g + 8, PHASE[0][1]),
          box(bx - 4, y0 - 4, 2 * s + g + 8, 4 * s + 3 * g + 8, PHASE[1][1]),
          box(cx - 4, y0 - 4, 2 * s + g + 8, 2 * s + g + 8, INK),
          label(330, 330, "Block (0, 0) needs rows 0 to 1 of A and columns 0 to 1 of B: 16 values.",
                fill=INK, room=600),
          label(330, 350, "The direct kernel fetches these 16 values 32 times.", room=600)]
save(OUT + "ch17-block-needs.svg", 660, 372,
     "Matrices A, B and C of the 4 by 4 example. Rows 0 and 1 of A (1 2 3 4 and 5 6 7 8) are outlined in blue, "
     "columns 0 and 1 of B (1 0 2 1 and 0 1 1 2) in amber, and the other cells are faded. In C, the top-left 2 by 2 "
     "square is block (0, 0), its 4 cells colored for threads (0,0), (0,1), (1,0) and (1,1). The note: block (0, 0) "
     "needs 16 values, and the direct kernel fetches them 32 times.",
     "The values needed by 1 block", "Block (0, 0) computes 4 entries of C from 2 rows of A and 2 columns of B.",
     parts)


# 17.2  c00 split into 2 groups of 2 terms.
s, g = 40, 4
ax, bx, y0 = 130, 380, 110


def split_a(r, c):
    if r == 0:
        return (PHASE[0] if c < 2 else PHASE[1]), False
    return PLAIN, True


def split_b(r, c):
    if c == 0:
        return (PHASE[0] if r < 2 else PHASE[1]), False
    return PLAIN, True


parts = [label(ax + 86, 98, "A", bold=True, fill=INK, size=13),
         label(bx + 86, 98, "B", bold=True, fill=INK, size=13)]
parts += grid(ax, y0, A, s, g, split_a) + grid(bx, y0, B, s, g, split_b)
parts += [bar(150, 310, 150, 34, PHASE[0], "1·1 + 2·0 = 1", INK),
          label(320, 332, "+", fill=INK, size=16),
          bar(340, 310, 150, 34, PHASE[1], "3·2 + 4·1 = 10", INK),
          label(512, 332, "= 11", fill=INK, size=15, bold=True),
          label(225, 364, "phase 0", fill=PHASE[0][1], bold=True),
          label(415, 364, "phase 1", fill=PHASE[1][1], bold=True)]
save(OUT + "ch17-split.svg", 660, 386,
     "Matrices A and B of the 4 by 4 example. Row 0 of A and column 0 of B are colored in 2 groups: columns 0 and 1 "
     "of A with rows 0 and 1 of B in blue, columns 2 and 3 of A with rows 2 and 3 of B in amber. Below, the blue "
     "group gives 1 times 1 plus 2 times 0 = 1 (phase 0), the amber group 3 times 2 plus 4 times 1 = 10 (phase 1), "
     "and 1 + 10 = 11, the entry c00.",
     "Splitting the sum", "c00 = 11 as 2 groups of 2 terms: each group uses a 2 × 2 piece of A and of B.", parts)


# 17.3  Block (0, 0) in 2 phases.
s, g = 28, 3
side = 2 * s + g


def phase_row(y, t, sums):
    a_style = lambda r, c: ((THREAD[(r, c - 2 * t)], False) if r < 2 and 2 * t <= c < 2 * t + 2
                            else (PLAIN, r >= 2))
    b_style = lambda r, c: ((THREAD[(r - 2 * t, c)], False) if c < 2 and 2 * t <= r < 2 * t + 2
                            else (PLAIN, c >= 2))
    out = [label(30 + 61, y - 10, "A (global)", size=11.5),
           label(176 + 61, y - 10, "B (global)", size=11.5)]
    out += grid(30, y, A, s, g, a_style) + grid(176, y, B, s, g, b_style)
    out += [box(30 + 2 * t * (s + g) - 3, y - 3, side + 6, side + 6, INK),
            box(176 - 3, y + 2 * t * (s + g) - 3, side + 6, side + 6, INK),
            shared(324, y - 26, 196, 132)]
    out += [label(358 + side / 2, y - 6, "tileA", size=11.5, bold=True, fill="#1F7A5C"),
            label(452 + side / 2, y - 6, "tileB", size=11.5, bold=True, fill="#1F7A5C")]
    for r in range(2):
        for c in range(2):
            out.append(cell(358 + c * (s + g), y + 10 + r * (s + g), s, A[r][2 * t + c], THREAD[(r, c)]))
            out.append(cell(452 + c * (s + g), y + 10 + r * (s + g), s, B[2 * t + r][c], THREAD[(r, c)]))
            out.append(cell(560 + c * (s + g), y + 10 + r * (s + g), s, sums[r][c], THREAD[(r, c)]))
    out.append(label(560 + side / 2, y - 6, "sum", size=11.5, bold=True))
    return out


parts = [label(30, 104, "Phase 0", "start", INK, 13, True),
         label(96, 104, "tiles at columns 0 to 1 of A and rows 0 to 1 of B", "start", room=440),
         label(30, 300, "Phase 1", "start", INK, 13, True),
         label(96, 300, "the tile of A moves right, the tile of B moves down", "start", room=440)]
parts += phase_row(140, 0, [[1, 2], [5, 6]])
parts += phase_row(336, 1, [[11, 13], [27, 29]])
save(OUT + "ch17-phases.svg", 660, 480,
     "Block (0, 0) in 2 phases, 1 color per thread. Phase 0: the tile of A at columns 0 and 1 (1 2, 5 6) and the "
     "tile of B at rows 0 and 1 (1 0, 0 1) are outlined in the global matrices; each value is colored by the thread "
     "that copies it into the same slot of tileA or tileB in shared memory; the 4 sums are then 1, 2, 5 and 6. "
     "Phase 1: the tile of A moves to columns 2 and 3 (3 4, 7 8) and the tile of B to rows 2 and 3 (2 1, 1 2); the "
     "sums become 11, 13, 27 and 29, the top-left square of C.",
     "Block (0, 0) in 2 phases", "Each color marks 1 thread: the values it copies, the slots it fills, its sum.",
     parts)


# 17.4  The operands of thread (0, 1) in phase 0.
s, g = 40, 5
parts = [shared(40, 96, 260, 150, "shared memory, phase 0"),
         label(64 + 42, 116, "tileA", size=12, bold=True, fill="#1F7A5C"),
         label(186 + 42, 116, "tileB", size=12, bold=True, fill="#1F7A5C")]
tile_a, tile_b = [[1, 2], [5, 6]], [[1, 0], [0, 1]]
for r in range(2):
    for c in range(2):
        used_a, used_b = r == 0, c == 1
        parts.append(cell(64 + c * (s + g), 128 + r * (s + g), s, tile_a[r][c],
                          THREAD[(r, c)] if used_a else PLAIN))
        parts.append(cell(186 + c * (s + g), 128 + r * (s + g), s, tile_b[r][c],
                          THREAD[(r, c)] if used_b else PLAIN))
parts += [label(322, 140, "thread (0, 1): row 0 of tileA · column 1 of tileB", "start", INK, room=330),
          label(322, 172, "1·0 + 2·1 = 2", "start", "#2F6FDB", 17, True),
          label(322, 204, "loaded by threads (0,0), (0,1), (0,1), (1,1),", "start", room=310),
          label(322, 222, "in the order the operands appear", "start", room=310)]
save(OUT + "ch17-operands.svg", 660, 272,
     "The shared tiles of block (0, 0) in phase 0: tileA holds 1 2 and 5 6, tileB holds 1 0 and 0 1. Row 0 of tileA "
     "and column 1 of tileB are colored by the threads that wrote them: 1 by thread (0,0), 2 by thread (0,1), 0 by "
     "thread (0,1) and 1 by thread (1,1). Thread (0, 1) computes 1 times 0 plus 2 times 1 = 2. It copied 2 of its 4 "
     "operands; its neighbors copied the other 2.",
     "The operands of 1 thread", "Thread (0, 1) in phase 0 uses 4 values and copied only 2 of them.", parts)


# 17.5  A barrier: 4 threads arrive at different times and leave together.
xb = 400
arrive = [250, 390, 330, 205]
colours = [THREAD[(0, 0)], THREAD[(0, 1)], THREAD[(1, 0)], THREAD[(1, 1)]]
parts = [label(xb, 100, "barrier", fill=INK, bold=True)]
for k in range(4):
    y = 116 + k * 44
    fill, stroke = colours[k]
    parts += [label(130, y + 18, f"thread {k}", "end", INK),
              bar(150, y + 4, arrive[k] - 150, 22, (fill, fill)),
              hline(arrive[k], xb - 4, y + 15, MUTED, 1.2, "3 4"),
              dot(arrive[k], y + 15, INK),
              bar(xb + 8, y + 4, 150, 22, (fill, fill))]
parts += [vline(xb, 108, 296, INK, 2.5),
          label(275, 320, "arrive at different times; dotted = waiting", room=290),
          label(xb + 83, 320, "leave together", room=150)]
save(OUT + "ch17-barrier.svg", 660, 340,
     "4 threads of a block as horizontal bars on a time axis, with a vertical line for the barrier. Thread 3 arrives "
     "first, then thread 0, thread 2 and finally thread 1; each early thread waits on a dotted line until the "
     "barrier. After the barrier, all 4 bars start at the same time.",
     "A barrier", "The 4 threads arrive at different times, wait, and leave together.", parts)


# 17.6  The 4 steps of 1 phase.
steps = [("load tiles", "global → shared", PHASE[0]), ("barrier", "all loads complete", GREY),
         ("compute", "T multiply-adds", THREAD[(0, 0)]), ("barrier", "all reads complete", GREY)]
w, gap, x0, y0 = 128, 28, 33, 112
parts = [label(330, 96, "one phase", fill=INK, bold=True)]
for k, (name, sub, (fill, stroke)) in enumerate(steps):
    x = x0 + k * (w + gap)
    parts += [f'  <rect x="{x}" y="{y0}" width="{w}" height="58" rx="10" fill="{fill}" stroke="{stroke}"'
              f' stroke-width="1.3"/>',
              label(x + w / 2, y0 + 25, name, fill=INK, size=13, bold=True, room=w - 10),
              label(x + w / 2, y0 + 44, sub, size=11.5, room=w - 10)]
    if k < 3:
        parts.append(wire([(x + w, y0 + 29), (x + w + gap, y0 + 29)]))
last = x0 + 3 * (w + gap) + w / 2
first = x0 + w / 2
parts.append(wire([(last, y0 + 58), (last, 222), (first, 222), (first, y0 + 58)],
                  label="next phase: phase + 1, while phase < ⌈K/T⌉", on=1))
save(OUT + "ch17-cycle.svg", 660, 250,
     "4 boxes in a row: load tiles, global to shared; barrier, all loads complete; compute, T multiply-adds; "
     "barrier, all reads complete. An arrow runs from the last box back to the first, labeled next phase: phase + 1, "
     "while phase is less than the ceiling of K over T.",
     "One phase of the tiled kernel", "Copy, wait, compute, wait. The cycle repeats ⌈K/T⌉ times.", parts)


# 17.7  Fetches from global memory in the 4 x 4 example.
x0, scale = 190, 2.7
parts = [label(x0 - 14, 128, "direct kernel", "end", INK, bold=True),
         bar(x0, 110, 128 * scale, 28, ("#C9D3F8", "#5B6BD6")),
         label(x0 + 128 * scale + 10, 129, "128 fetches", "start", INK, bold=True),
         label(x0 - 14, 182, "tiled, 2 × 2 tiles", "end", INK, bold=True),
         bar(x0, 164, 64 * scale, 28, ("#CDEEE0", "#1F8A70")),
         label(x0 + 64 * scale + 10, 183, "64 fetches", "start", INK, bold=True),
         vline(x0 + 32 * scale, 98, 206, MUTED, 1.2, "4 4"),
         label(x0 + 32 * scale, 224, "only 32 different numbers exist", fill=MUTED)]
save(OUT + "ch17-fetches.svg", 660, 248,
     "2 horizontal bars. The direct kernel makes 128 fetches from global memory in the 4 by 4 example; the tiled "
     "kernel with 2 by 2 tiles makes 64. A dashed line at 32 marks the number of different values in A and B.",
     "Fetches from global memory", "The 4 × 4 example: tiles of 2 × 2 halve the fetches.", parts)


# 17.8  Without the first barrier: c00 = 7.
parts = [label(150, 128, "thread (0,0)", "end", INK, bold=True),
         bar(170, 112, 80, 26, THREAD[(0, 0)], "load", INK),
         bar(258, 112, 250, 26, THREAD[(0, 0)], "compute: 3·2 + 4·tileB[1][0]", INK),
         label(150, 186, "thread (1,0)", "end", INK, bold=True),
         bar(170, 170, 270, 26, THREAD[(1, 0)], "load (delayed): tileB[1][0] = 1", INK),
         label(170, 100, "phase 1", "start", size=11.5),
         vline(390, 138, 166, RED, 2.2), dot(390, 140, RED, 4.5),
         label(330, 228, "thread (0,0) reads tileB[1][0] before it is refilled", fill=RED, room=600),
         label(330, 248, "the stale value 0 gives 3·2 + 4·0 = 6 in place of 10", fill=RED, room=600),
         label(330, 270, "result: c00 = 7 in place of 11", fill=RED, bold=True, room=600)]
save(OUT + "ch17-no-barrier1.svg", 660, 292,
     "A timeline of phase 1 without the first barrier. Thread (0,0) loads, then computes 3 times 2 plus 4 times "
     "tileB[1][0]. Thread (1,0) is late: its load of tileB[1][0] = 1 is still running when thread (0,0) reads the "
     "slot, marked by a red line. Notes: thread (0,0) reads tileB[1][0] before it is refilled; the stale value 0 "
     "gives 3 times 2 plus 4 times 0 = 6 in place of 10; result c00 = 7 in place of 11.",
     "Without the first barrier", "Thread (0,0) reads a slot before thread (1,0) refills it.", parts)


# 17.9  Without the second barrier: c01 = 15.
parts = [label(150, 128, "thread (0,1)", "end", INK, bold=True),
         bar(170, 112, 300, 26, THREAD[(0, 1)], "phase 0 compute: 1·0 + 2·tileB[1][1]", INK),
         label(150, 186, "thread (1,1)", "end", INK, bold=True),
         bar(170, 170, 56, 26, THREAD[(1, 1)], "load", INK),
         bar(232, 170, 80, 26, THREAD[(1, 1)], "compute", INK),
         bar(318, 170, 220, 26, THREAD[(1, 1)], "phase 1 load: tileB[1][1] = 2", INK),
         vline(420, 138, 166, RED, 2.2), dot(420, 140, RED, 4.5),
         label(330, 228, "thread (1,1) overwrites tileB[1][1] (1 → 2) during phase 0", fill=RED, room=600),
         label(330, 248, "thread (0,1) then computes 1·0 + 2·2 = 4 in place of 2", fill=RED, room=600),
         label(330, 270, "result: c01 = 15 in place of 13", fill=RED, bold=True, room=600)]
save(OUT + "ch17-no-barrier2.svg", 660, 292,
     "A timeline without the second barrier. Thread (0,1) is still computing phase 0, 1 times 0 plus 2 times "
     "tileB[1][1]. Thread (1,1) has finished its phase-0 load and compute and starts the phase-1 load, which "
     "stores tileB[1][1] = 2; a red line marks the moment thread (0,1) reads the slot. Notes: thread (1,1) "
     "overwrites tileB[1][1] from 1 to 2 during phase 0; thread (0,1) then computes 1 times 0 plus 2 times 2 = 4 in "
     "place of 2; result c01 = 15 in place of 13.",
     "Without the second barrier", "Thread (1,1) overwrites a slot that thread (0,1) still needs.", parts)


# 17.10  1 phase with both barriers.
b1, b2 = 270, 470
loads = [218, 256, 236, 206]
nexts = [520, 556, 540, 510]
names = [(0, 0), (0, 1), (1, 0), (1, 1)]
parts = [label(205, 100, "load this phase", room=120), label(370, 100, "compute this phase", room=180),
         label(540, 100, "load next phase", room=120)]
for k, who in enumerate(names):
    y = 114 + k * 40
    fill, _ = THREAD[who]
    parts += [label(140, y + 18, f"thread ({who[0]},{who[1]})", "end", INK),
              bar(156, y + 4, loads[k] - 156, 22, (fill, fill)),
              hline(loads[k] + 4, b1 - 4, y + 15, MUTED, 1.2, "3 4"),
              bar(b1 + 6, y + 4, b2 - b1 - 12, 22, (fill, fill)),
              bar(b2 + 6, y + 4, nexts[k] - b2 - 6, 22, (fill, fill))]
parts += [vline(b1, 106, 280, INK, 2.5), vline(b2, 106, 280, INK, 2.5),
          label(b1, 300, "barrier 1", fill=INK, bold=True), label(b2, 300, "barrier 2", fill=INK, bold=True)]
save(OUT + "ch17-schedule.svg", 660, 322,
     "A timeline of 4 threads over 1 phase with both barriers. The loads of one phase end at different times, and "
     "each thread waits on a dotted line until barrier 1. All 4 threads then compute that phase, meet at barrier 2, "
     "and start the loads of the next phase.",
     "1 phase with both barriers", "The dotted segments show threads waiting for the slowest thread.", parts)


# 17.11  The loop invariant for thread (0, 0).
s, g, x0 = 44, 4, 150
parts = [label(x0 - 14, 124, "row 0 of A", "end"), label(x0 - 14, 176, "column 0 of B", "end")]
for c in range(4):
    colours = PHASE[0] if c < 2 else PHASE[1]
    parts.append(cell(x0 + c * (s + g), 100, s, A[0][c], colours))
    parts.append(cell(x0 + c * (s + g), 152, s, B[c][0], colours))
rows = [("start of phase 0", 0, "s ∈ [0, 0): sum = 0"),
        ("start of phase 1", 2, "s ∈ [0, 2): sum = 1·1 + 2·0 = 1"),
        ("end of the loop", 4, "s ∈ [0, 4): sum = 1 + 3·2 + 4·1 = 11 = c00")]
for k, (name, terms, note) in enumerate(rows):
    y = 226 + k * 36
    parts.append(label(x0 - 14, y + 13, name, "end", INK, bold=True))
    width = max(terms * (s + g) - g, 4)
    parts.append(bar(x0, y, width, 18, ("#DDE3FB", "#5B6BD6")))
    parts.append(label(x0 + 4 * (s + g) + 14, y + 13, note, "start", INK, room=290))
save(OUT + "ch17-invariant.svg", 660, 340,
     "Row 0 of A, 1 2 3 4, and column 0 of B, 1 0 2 1, drawn as 2 rows of cells: the first 2 cells of each in blue "
     "(phase 0), the last 2 in amber (phase 1). Below, 3 bars show how much of the dot product sum covers: at the start "
     "of phase 0 no terms, sum = 0; at the start of phase 1 the terms s in [0, 2), sum = 1 times 1 plus 2 times 0 = 1; at "
     "the end of the loop, the terms s in [0, 4), sum = 1 + 3 times 2 + 4 times 1 = 11 = c00.",
     "The loop invariant", "Thread (0, 0) with T = 2: before each phase, sum holds the terms s ∈ [0, phase · T).", parts)


# 17.12  Matrices whose size is not a multiple of T.
s, g, x0, y0 = 40, 4, 70, 112
block_of = {(0, 0): THREAD[(0, 0)], (0, 1): THREAD[(0, 1)], (1, 0): THREAD[(1, 0)], (1, 1): THREAD[(1, 1)]}
parts = [label(x0 + 2 * (s + g) - 2, 98, "C is 3 × 3, T = 2", fill=INK, bold=True)]
for r in range(4):
    for c in range(4):
        x, y = x0 + c * (s + g), y0 + r * (s + g)
        if r < 3 and c < 3:
            parts.append(cell(x, y, s, "", block_of[(r // 2, c // 2)]))
        else:
            parts.append(cell(x, y, s, "", ("#FFFFFF", "#9AA5B1"), dashed=True))
parts += [box(x0 - 4, y0 - 4, 3 * s + 2 * g + 8, 3 * s + 2 * g + 8, INK, 2.2),
          label(x0 + 2 * (s + g) - 2, 316, "dashed: threads with no entry, 7 of 16", room=300)]
tx0 = 410
parts += [label(tx0 + 92, 98, "tileA, phase 1", fill=INK, bold=True),
          cell(tx0, y0, (88, 40), "A[0][2]", THREAD[(0, 0)]),
          cell(tx0 + 96, y0, (88, 40), "0", GREY),
          cell(tx0, y0 + 48, (88, 40), "A[1][2]", THREAD[(1, 0)]),
          cell(tx0 + 96, y0 + 48, (88, 40), "0", GREY),
          label(tx0 + 92, y0 + 116, "K = 3: column 3 of A does not exist,", room=250),
          label(tx0 + 92, y0 + 134, "so its slots hold 0", room=250)]
save(OUT + "ch17-edges.svg", 660, 340,
     "Left: a 4 by 4 grid of threads from 4 blocks of 2 by 2 laid over a 3 by 3 matrix C, outlined in black. The 9 "
     "cells inside C are colored by block; the 7 cells in row 3 and column 3 are dashed: threads with no entry. "
     "Right: tileA in phase 1 when K = 3. Its left column holds A[0][2] and A[1][2]; its right column would hold "
     "column 3 of A, which does not exist, so both slots hold 0.",
     "Matrices of any size", "Blocks stick out past the edge of C, and a tile past the last column of A holds 0.",
     parts)


# 17.13  A 2 x 2 tile stored as a flat array.
s, g = 50, 6
parts = [label(110 + s + g / 2, 98, "tileA as a 2 × 2 grid", fill=INK, bold=True)]
for r in range(2):
    for c in range(2):
        parts.append(cell(110 + c * (s + g), 110 + r * (s + g), s, f"({r},{c})", THREAD[(r, c)]))
parts.append(label(434, 98, "TILE_A as a flat array", fill=INK, bold=True))
for k in range(4):
    r, c = divmod(k, 2)
    x = 320 + k * (s + g)
    parts += [cell(x, 116, s, f"({r},{c})", THREAD[(r, c)]), label(x + s / 2, 186, str(k))]
parts += [box(320 + 2 * (s + g) - 3, 113, s + 6, s + 6, "#2F6FDB", 2.2),
          label(330, 248, "index = threadIdx.y · T + threadIdx.x = 1 · 2 + 0 = 2", fill="#2F6FDB",
                bold=True, room=600),
          label(330, 270, "with T = 16 the array holds 256 values", room=600)]
save(OUT + "ch17-flat-tile.svg", 660, 292,
     "Left: a 2 by 2 tile with slots (0,0), (0,1), (1,0) and (1,1), 1 color per thread. Right: the same 4 slots in "
     "a flat array at indices 0 to 3, in row-major order; slot (1,0) at index 2 is outlined. The rule: index = threadIdx.y times T "
     "plus threadIdx.x, so (1, 0) is at 1 times 2 plus 0 = 2. With T = 16 the array holds 256 values.",
     "A tile as a flat array", "Slot (threadIdx.y, threadIdx.x) is element threadIdx.y · T + threadIdx.x.", parts)


# 17.14  The path of data through 1 thread of the tiled kernel.
stages = [("global memory", "A and B", GREY), ("shared memory", "tileA, tileB", ("#D9F1E7", "#1F8A70")),
          ("registers", "sum", ("#FBF1DC", "#B7791F")), ("global memory", "C", GREY)]
w, gap, x0, y0 = 128, 30, 24, 110
parts = []
for k, (name, sub, (fill, stroke)) in enumerate(stages):
    x = x0 + k * (w + gap)
    parts += [f'  <rect x="{x}" y="{y0}" width="{w}" height="62" rx="10" fill="{fill}" stroke="{stroke}"'
              f' stroke-width="1.3"/>',
              label(x + w / 2, y0 + 27, name, fill=INK, size=13, bold=True, room=w - 10),
              label(x + w / 2, y0 + 46, sub, size=11.5, room=w - 10)]
    if k < 3:
        parts.append(wire([(x + w, y0 + 31), (x + w + gap, y0 + 31)]))
for k, note in enumerate(["load: 2 per phase", "read: 2T per phase", "store: 1"]):
    xm = x0 + k * (w + gap) + w + gap / 2
    parts.append(label(xm, y0 + 96, note, room=150))
save(OUT + "ch17-path.svg", 660, 236,
     "4 boxes joined by arrows: global memory holding A and B, shared memory holding tileA and tileB, registers "
     "holding sum, and global memory holding C. Under the arrows: load, 2 values per phase; read, 2T values per "
     "phase; store, 1 value.",
     "The path of data through 1 thread", "The tiled kernel stages operands in shared memory and sums in a register.",
     parts)


# 17.15  Measured on the Orin NX at n = 4,096 (quietest of 3 runs of each program).
CPP = {"direct": 184.5, "T = 8": 139.8, "T = 16": 272.1, "T = 32": 215.5}
RUST = {"direct": 138.0, "T = 8": 136.8, "T = 16": 258.5, "T = 32": 205.5}
FETCH = {"direct": 738.0, "T = 8": 69.9, "T = 16": 68.0, "T = 32": 26.9}
LANG = {"C++": ("#DDE3FB", "#5B6BD6"), "Rust": ("#D7F0E6", "#1F8A70")}


def hbar(x, y, w, h, colours, value):
    fill, stroke = colours
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 2):.1f}" height="{h:.1f}" rx="4" fill="{fill}"'
            f' stroke="{stroke}" stroke-width="1.2"/>\n'
            + label(x + max(w, 2) + 6, y + h - 4, value, "start", stroke, 11.5, True)
            .replace('style="', 'style="stroke:#FBFAF6;stroke-width:4px;paint-order:stroke;'))


x0, scale = 130, 300 / 400
limit = x0 + 300 * scale
parts = [label(40, 108, "Rate, GFLOP/s", "start", INK, 13, True), vline(limit, 118, 330, RED, 1.5, "5 4")]
for k, name in enumerate(CPP):
    y = 124 + k * 52
    parts.append(label(x0 - 10, y + 26, name, "end", INK, 12.5))
    parts.append(hbar(x0, y + 4, CPP[name] * scale, 18, LANG["C++"], f"{CPP[name]:.1f}"))
    parts.append(hbar(x0, y + 25, RUST[name] * scale, 18, LANG["Rust"], f"{RUST[name]:.1f}"))
parts += [label(limit + 4, 346, "shared-memory limit, 300", "end", RED, 11.5, True),
          f'  <rect x="40" y="360" width="12" height="12" rx="3" fill="{LANG["C++"][0]}"'
          f' stroke="{LANG["C++"][1]}"/>',
          label(58, 370, "C++", "start", INK, 12),
          f'  <rect x="100" y="360" width="12" height="12" rx="3" fill="{LANG["Rust"][0]}"'
          f' stroke="{LANG["Rust"][1]}"/>',
          label(118, 370, "Rust", "start", INK, 12)]
x1, scale2 = 460, 150 / 800
dram = x1 + 102 * scale2
parts += [label(380, 108, "Requested fetches, GB/s (C++)", "start", INK, 13, True),
          vline(dram, 118, 330, RED, 1.5, "5 4")]
for k, name in enumerate(FETCH):
    y = 124 + k * 52
    parts.append(label(x1 - 10, y + 22, name, "end", INK, 12.5))
    parts.append(hbar(x1, y + 10, FETCH[name] * scale2, 18, ("#FBF1DC", "#B7791F"), f"{FETCH[name]:.1f}"))
parts += [label(dram - 4, 346, "global memory, 102", "start", RED, 11.5, True)]
save(OUT + "ch17-measured.svg", 660, 392,
     "2 panels for n = 4,096 on the Orin NX. Left, the rate in GFLOP/s of each kernel in C++ and in Rust: direct "
     "184.5 and 138.0, T = 8 139.8 and 136.8, T = 16 272.1 and 258.5, T = 32 215.5 and 205.5, with a dashed line "
     "at 300 GFLOP/s, the limit set by 2 shared-memory reads per multiply-add. Right, the requested fetch rate of "
     "the C++ kernels in GB/s: direct 738.0, T = 8 69.9, T = 16 68.0, T = 32 26.9, with a dashed line at 102 GB/s, "
     "the bandwidth of global memory.",
     "The tiled kernels, measured", "16 times fewer fetches, 1.5 times the speed.", parts)
