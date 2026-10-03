"""Hand-drawn figures for chapter 1, in the style of the systems book.

    python3 tools/draw_ch01.py

Each figure is two or three titled panels. Every panel is a stack of labelled
rows, and each panel carries a foot note that states what it shows. Coordinates
are explicit, so no label can run into its neighbour.
"""
from svgkit import *  # noqa: F401,F403
import math

OUT = "src/figures/"
MUTED2 = "#6b7a89"


def panel(f, x, y, w, h, title, color=INK):
    f.rect(x, y, w, h, "#fdfefe", BORDER, rx=8, width=1.4)
    f.line(x + 1, y + 1, x + w - 1, y + 1, color, 3)
    f.text(x + 14, y + 28, title, 15, color, bold=True)


def note(f, x, y, text, size=12):
    f.text(x, y, text, size, MUTED2)


def row(f, x, y, w, h, label, value, fill, value_color=INK):
    f.cell(x, y, w, h, label, fill, size=12)
    if value:
        f.text(x + w + 14, y + h / 2 + 5, value, 12, value_color, mono=True)


# 1. Two answers to a memory wait.
f = Figure(900, 352)
f.text(20, 26, "Two answers to a 500-cycle memory wait", 14, bold=True)

panel(f, 16, 44, 424, 264, "CPU: one fast core, the stall is avoided", MUTED)
row(f, 30, 94, 150, 34, "branch predictor", "", PALE)
row(f, 30, 134, 150, 34, "large cache", "", PALE)
row(f, 30, 174, 150, 34, "out-of-order core", "", PALE)
f.cell(30, 220, 384, 30, "core 0 issues later work across a load", PINK, size=11)
note(f, 30, 284, "the core rarely stalls on a single access")

panel(f, 460, 44, 424, 264, "GPU: many warps resident, the wait is filled", TEAL)
for r in range(4):
    for c in range(8):
        fill = PINK if r == 0 else GREEN
        f.cell(474 + c * 50, 94 + r * 30, 42, 22, "", fill, size=9)
f.text(474, 226, "warp 0 stalls on its load", 10.5, RUST)
f.text(474, 244, "warps 1 to 3 have work to issue", 10.5, TEAL)
note(f, 474, 284, "many warps resident: the wait is hidden under other warps")
f.save(OUT + "ch01-cpu-gpu.svg")


# 2. Little's law: the same latency at two concurrency levels.
f = Figure(900, 352)
f.text(20, 26, "The same 500-cycle latency at two concurrency levels", 14, bold=True)

panel(f, 16, 44, 424, 284, "concurrency = 1: one access in flight", RUST)
f.rect(30, 94, 350, 30, PINK, BORDER, rx=6, width=1.2)
f.text(40, 114, "load a[i]", 11, INK, mono=True)
f.arrow(386, 109, 416, 109, RUST, 2.4)
note(f, 30, 142, "the line is idle until this access returns")
f.cell(30, 162, 380, 30, "1 result, at cycle 500", CREAM, size=11)
note(f, 30, 222, "throughput = 1 result / 500 cycles")

panel(f, 460, 44, 424, 284, "concurrency = 250: enough to fill the line", TEAL)
for r in range(5):
    f.rect(474, 94 + r * 22, 350, 16, PALE, TEAL, rx=4, width=1)
    f.arrow(830, 102 + r * 22, 860, 102 + r * 22, TEAL, 2.4)
f.text(484, 106, "access 1", 10, INK, mono=True)
f.text(484, 150, "access 125", 10, INK, mono=True)
f.text(484, 194, "access 250", 10, INK, mono=True)
f.cell(474, 212, 380, 30, "one result every 2 cycles", CREAM, size=11)
note(f, 474, 272, "throughput = 250 results / 500 cycles")
note(f, 474, 296, "every access still takes 500 cycles")
f.save(OUT + "ch01-little.svg")


# 3. Amdahl on the timeline of the program.
f = Figure(900, 340)
f.text(20, 26, "Amdahl's law on the timeline of the program", 15, bold=True)

scale = 2.6
panel(f, 16, 44, 424, 268, "1 unit: total 130 us", MUTED)
for i, (label, us, color, fill) in enumerate((("read", 20, RUST, PINK),
                                              ("compute", 100, TEAL, GREEN),
                                              ("write", 10, RUST, PINK))):
    y = 88 + i * 52
    f.text(30, y + 22, label, 12, MUTED2, anchor="start", mono=True)
    f.rect(100, y, max(6, us * scale * 0.9), 30, fill, color, rx=5, width=1.4)
    f.text(100 + us * scale * 0.9 + 10, y + 21, "%d us" % us, 12, color, mono=True)
note(f, 30, 256, "read and write 30 us, compute 100 us, f = 30/130 = 0.23")
note(f, 30, 278, "ceiling 1/f = 4.3x")
note(f, 30, 300, "the read and write are a floor under the run time")

panel(f, 460, 44, 424, 268, "8 units: total 42.5 us", TEAL)
for i, (label, us, color, fill) in enumerate((("read", 20, RUST, PINK),
                                              ("compute", 12.5, TEAL, GREEN),
                                              ("write", 10, RUST, PINK))):
    y = 88 + i * 52
    f.text(474, y + 22, label, 12, MUTED2, anchor="start", mono=True)
    f.rect(544, y, max(6, us * scale * 0.9), 30, fill, color, rx=5, width=1.4)
    f.text(544 + us * scale * 0.9 + 10, y + 21, (("%.1f us" % us) if us != int(us) else "%d us" % us),
           12, color, mono=True)
note(f, 474, 256, "the compute divides by 8; the read and write stay at 30 us")
note(f, 474, 278, "speedup 130 / 42.5 = 3.06")
note(f, 474, 300, "more units cannot pass the 4.3x ceiling")
f.save(OUT + "ch01-amdahl-timeline.svg")


# 3b. Amdahl's law as a family of curves: speedup against unit count, one
# curve per serial fraction. The example curve is the four-stage program.
f = Figure(900, 352)
f.text(20, 26, "Amdahl's law: the ceiling for each serial fraction", 14, bold=True)

x0, x1 = 70.0, 470.0
y0, y1 = 52.0, 292.0


def amdahl_s(p, f):
    return 1.0 / (f + (1.0 - f) / p)


def px(p):
    return x0 + (math.log10(p) / 3.0) * (x1 - x0)


def py(s):
    return y1 - (math.log10(s) / 3.0) * (y1 - y0)


f.line(x0, y0, x0, y1, INK, 1.4)
f.line(x0, y1, x1, y1, INK, 1.4)
for p in (1, 4, 16, 64, 256, 1024):
    f.line(px(p), y1, px(p), y1 + 5, BORDER, 1.4)
    f.text(px(p), y1 + 19, str(p), 10, MUTED2, anchor="middle")
f.text((x0 + x1) / 2, y1 + 38, "units (log scale)", 10.5, MUTED2, anchor="middle")
f.text(x0, y0 - 12, "speedup, log scale", 10.5, MUTED2, anchor="start")
for s in (1, 4, 16, 64, 256, 1000):
    f.line(x0 - 5, py(s), x0, py(s), BORDER, 1.4)
    f.text(x0 - 12, py(s) + 4, str(s), 10, MUTED2, anchor="end")
curve_colors = [("#b7c2cc", "f = 0.001"), ("#8fa6b4", "f = 0.01"),
                ("#6b8fa0", "f = 0.05"), ("#4a7a8c", "f = 0.10")]
for color, label in curve_colors:
    fval = float(label.split("= ")[1])
    pts = [(px(p), py(amdahl_s(p, fval))) for p in
           [1, 1.2, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]]
    poly = " ".join("%.1f,%.1f" % (x, y) for x, y in pts)
    f.parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.6"/>' % (poly, color))
    f.text(x1 - 6, py(amdahl_s(1024, fval)) - 8, label, 9.5, color, anchor="end")
example_f = 30.0 / 130.0
pts = [(px(p), py(amdahl_s(p, example_f))) for p in
       [1, 1.2, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]]
poly = " ".join("%.1f,%.1f" % (x, y) for x, y in pts)
f.parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="3.2"/>' % (poly, RUST))
f.text(x1 - 6, py(amdahl_s(1024, example_f)) - 8, "this program, f = 0.23", 9.5, RUST,
       bold=True, anchor="end")
ceiling_y = py(amdahl_s(1024, example_f))
f.line(x0, ceiling_y, x1, ceiling_y, RUST, 1.2, dash=True)
f.text(x0 + 6, ceiling_y - 8, "ceiling 4.3x", 9.5, RUST, bold=True)
for p, sval in ((2, 130.0 / 80.0), (4, 130.0 / 55.0), (8, 130.0 / 42.5)):
    f.circle(px(p), py(sval), 4.5, RUST)

panel(f, 560, 44, 324, 264, "The example run", RUST)
rows = [("read and write", "30 us of 130 us"), ("f", "0.23"),
        ("ceiling 1/f", "4.3x"), ("at p = 4", "S = 2.36"),
        ("at p = 8", "S = 3.06")]
for i, (label, value) in enumerate(rows):
    y = 92 + i * 36
    f.text(578, y + 19, label, 11, MUTED2)
    f.text(578 + 150, y + 19, value, 12, INK, mono=True)
note(f, 578, 292, "more units cannot lift the run past 4.3x")
f.save(OUT + "ch01-amdahl-multi.svg")


# 4. One kernel on four machine types.
f = Figure(900, 456)
f.text(20, 26, "One vector add on four machine types", 14, bold=True)
PW, PH = 424, 184
X0, X1 = 16, 460
Y0, Y1 = 44, 240


def lanes(f, x, y, n, label, fill, cell=32, gap=6, size=9):
    for i in range(n):
        f.cell(x + i * (cell + gap), y, cell, 24, label, fill, size=size)


panel(f, X0, Y0, PW, PH, "SISD: one instruction, one value", MUTED2)
f.text(30, Y0 + 50, "c[i] = a[i] + b[i]", 11, INK, mono=True)
for i in range(4):
    x = 30 + i * 88
    f.cell(x, Y0 + 62, 68, 26, "i = %d" % i, PALE, size=10)
    if i < 3:
        f.arrow(x + 72, Y0 + 75, x + 84, Y0 + 75, MUTED2, 1.4)
note(f, 30, Y0 + 112, "one add at a time, in program order")
note(f, 30, Y0 + 130, "8 adds need 8 instructions")
f.text(30, Y0 + 158, "8 instructions", 12, MUTED2, bold=True)

panel(f, X1, Y0, PW, PH, "SIMD: one instruction, eight values", TEAL)
f.text(474, Y0 + 50, "c[0..7] = a[0..7] + b[0..7]", 11, INK, mono=True)
lanes(f, 474, Y0 + 62, 8, "a+b", GREEN, cell=30, gap=6)
note(f, 474, Y0 + 112, "one 256-bit register holds eight floats")
note(f, 474, Y0 + 130, "one add instruction covers all eight")
f.text(474, Y0 + 158, "1 instruction", 12, TEAL, bold=True)

panel(f, X0, Y1, PW, PH, "MIMD: one stream per core", RUST)
f.text(30, Y1 + 50, "core 0: c[i] = a[i] + b[i]", 10.5, INK, mono=True)
f.text(30, Y1 + 68, "core 1: s += a[i] * b[i]", 10.5, INK, mono=True)
f.cell(30, Y1 + 82, 180, 28, "core 0: add", PINK, size=10.5)
f.cell(224, Y1 + 82, 180, 28, "core 1: dot product", CREAM, size=10.5)
note(f, 30, Y1 + 132, "two instruction streams and two data streams")
note(f, 30, Y1 + 150, "the two cores run different programs")
f.text(30, Y1 + 172, "independent", 12, RUST, bold=True)

panel(f, X1, Y1, PW, PH, "SIMT: one instruction, 32 threads", BRASS)
f.text(474, Y1 + 50, "if (a[t] > 0) c[t] = a[t] + b[t];", 10.5, INK, mono=True)
lanes(f, 474, Y1 + 62, 8, "t", GREEN, cell=30, gap=6)
lanes(f, 474, Y1 + 92, 8, "if", GREY, cell=30, gap=6)
note(f, 474, Y1 + 132, "one instruction is sent to the whole group")
note(f, 474, Y1 + 150, "a branch runs the two sides in turn")
f.text(474, Y1 + 172, "together, then split", 12, "8a6a1f", bold=True)

note(f, 20, 448, "SIMT looks like SIMD while the threads agree, and like MIMD when they branch (Chapter 5).")
f.save(OUT + "ch01-flynn.svg")


# 5. Arithmetic intensity: the same machine, two kernels.
f = Figure(900, 340)
f.text(20, 26, "Arithmetic intensity: FLOPs per byte moved", 14, bold=True)

panel(f, 16, 44, 424, 256, "vector add: 1 FLOP per 12 bytes", TEAL)
for i, (label, val) in enumerate((("read a[i]", "4 bytes"), ("read b[i]", "4 bytes"),
                                  ("write c[i]", "4 bytes"))):
    y = 84 + i * 44
    f.cell(30, y, 130, 32, label, GREEN, size=11)
    f.text(172, y + 21, val, 11, INK, mono=True)
f.line(30, 216, 410, 216, BORDER, 1.2)
f.text(30, 240, "1 FLOP for 12 bytes", 13, INK, bold=True)
f.text(30, 262, "I = 1/12 = 0.08 FLOP/byte", 13, TEAL, bold=True, mono=True)
note(f, 30, 284, "the arithmetic units wait on bytes: memory-bound")

panel(f, 460, 44, 424, 256, "matmul: many FLOPs per byte", BRASS)
f.text(474, 92, "C[ty][tx] = sum over k of A[ty][k] * B[k][tx]", 11.5, INK, mono=True)
f.cell(474, 108, 190, 32, "one row of A", CREAM, size=12)
f.cell(674, 108, 190, 32, "one column of B", CREAM, size=12)
f.arrow(664, 124, 674, 124, INK, 2)
f.text(474, 164, "each loaded value is reused for the whole row", 12, MUTED2)
f.cell(474, 182, 390, 32, "2N^3 FLOPs for 3N^2 values of 4 bytes", PALE, size=12)
f.text(474, 248, "I = 2N^3 / (3N^2 x 4) = N/6 FLOP/byte", 13, BRASS, bold=True, mono=True)
note(f, 474, 272, "680 FLOP/byte at N = 4096: compute-bound")
note(f, 474, 292, "the two programs reuse each loaded value differently")
f.save(OUT + "ch01-intensity.svg")


# 6. Which resource is pinned.
f = Figure(900, 470)
f.text(20, 26, "The bound is the resource pinned near 100%", 14, bold=True)
programs = [("SHA-256 hashing", RUST, [("CPU", 0.98), ("memory", 0.05), ("I/O", 0.02)], "CPU-bound"),
            ("vector add", TEAL, [("CPU", 0.02), ("memory", 0.98), ("I/O", 0.02)], "memory-bound"),
            ("streaming 10 GB", BRASS, [("CPU", 0.02), ("memory", 0.02), ("I/O", 0.98)], "I/O-bound")]
y = 48
for name, col, rows, verdict in programs:
    f.rect(20, y, 860, 118, "#fdfefe", BORDER, rx=8, width=1.2)
    f.line(21, y + 1, 879, y + 1, col, 3)
    f.text(36, y + 28, name, 12.5, INK, bold=True)
    f.text(36, y + 96, verdict, 12, col, bold=True)
    for k, (res, frac) in enumerate(rows):
        ry = y + 44 + k * 24
        f.text(210, ry + 15, res, 10.5, MUTED2)
        f.cell(280, ry, 420, 18, "", PALE)
        f.rect(280, ry, 420 * frac, 18, col, "none", rx=4, width=0)
        f.text(716, ry + 15, "%d%%" % round(frac * 100), 10.5, col, mono=True)
    y += 130
note(f, 20, 442, "Double one resource and re-measure. If the time halves, that resource was the limit.")
f.save(OUT + "ch01-bound.svg")

print("wrote ch01 figures")
