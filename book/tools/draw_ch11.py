"""The figures of chapter 11, drawn with figkit and figparts.

    python3 tools/draw_ch11.py

The measurements are the Jetson Orin NX runs quoted in the chapter, made with the programs in
code/ch11_libraries/.
"""
import math

from figkit import swatches, width_of
from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, arrow, card, legend, line, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"

# Orin NX runs quoted in the chapter.
SUM = [("Chapter 8 kernel", 93.4), ("cub::BlockReduce", 92.5), ("cub::DeviceReduce", 91.8),
       ("thrust::reduce", 93.0)]
SCAN = [("Chapter 8 scan", 43.0), ("thrust::inclusive_scan", 68.2), ("cub::DeviceScan", 74.3)]
GEMM = [("Chapter 9 best", 931.9), ("cuBLAS, FP32", 1504.6), ("cuBLAS, TF32", 6530.4)]
SORT = [("std::sort, 1 CPU core", 10.3, "idle"), ("thrust::sort", 676.7, "store"),
        ("thrust::sort, scratch reused", 1285.5, "client"),
        ("with cuda::std::greater", 542.6, "danger"), ("with own comparison", 686.8, "danger"),
        ("cub::DeviceRadixSort", 1872.8, "client"), ("same, 8-bit keys, bits 0 to 7", 5356.5, "good")]
SMALL = (68.1, 29.9, 8.8)                     # thrust::reduce, CUB + copy, CUB kept, in µs


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def hbar(x, y, w, h, kind, solid=True):
    accent, tint = KINDS[kind]
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 3):.1f}" height="{h}" rx="4" '
            f'fill="{accent if solid else tint}" stroke="{accent}" stroke-width="1.1"/>')


def harrow(x1, x2, y, kind="flow"):
    """A straight horizontal arrow whose tip stops 3 units short of `x2`."""
    return wire([(x1, y), (x2 - 3 if x2 > x1 else x2 + 3, y)], kind)


# 11.1  The levels: Thrust on CUB, from the host down into a kernel; the handle libraries beside.
parts = [zone(16, 104, 400, 404, "Thrust and CUB"), zone(432, 104, 212, 404, "handle libraries")]
levels = [
    ("client", "Thrust", ["thrust::reduce, thrust::sort", "allocates scratch, may wait"], "host"),
    ("service", "CUB, device-wide", ["cub::DeviceReduce, DeviceScan", "caller passes scratch memory"],
     "host"),
    ("outside", "CUB, block and warp", ["cub::BlockReduce, BlockScan", "inside your own kernel"],
     "kernel"),
]
for k, (kind, name, sub, where) in enumerate(levels):
    y = 136 + k * 124
    parts += [card(40, y, 300, 80, kind, name, sub),
              tag(352, y + 31, where, kind)]
    if k < 2:
        parts.append(arrow(190, y + 80, y + 124))
        parts.append(note(200, y + 106, "built on", room=120))
for k, (name, sub) in enumerate((("cuBLAS", ["handle, then", "cublasSgemm"]),
                                 ("cuFFT", ["plan, then", "cufftExecR2C"]),
                                 ("cuRAND", ["generator, then", "curandGenerate"]))):
    parts.append(card(452, 136 + k * 124, 172, 80, "store", name, sub))
parts += [note(40, 540, "Thrust calls CUB's device-wide algorithms, which launch kernels built from CUB's",
               room=600),
          note(40, 558, "block and warp pieces. Lower levels give the caller more control: over scratch",
               room=600),
          note(40, 576, "memory, over waiting, and over what happens around each step.", room=600)]
save(OUT + "ch11-levels.svg", 660, 598,
     "3 levels in a column. At the top, called from the host, Thrust: thrust::reduce, thrust::sort; it "
     "allocates its own scratch memory and launches its own kernels. In the middle, also from the host, CUB "
     "device-wide calls: cub::DeviceReduce, cub::DeviceScan, cub::DeviceRadixSort; the caller passes in "
     "the scratch memory. At the bottom, inside the caller's own kernel, CUB block and warp building "
     "blocks: cub::BlockReduce, cub::BlockScan, cub::WarpReduce. Beside the column, 3 separate libraries "
     "that work through a handle or plan: cuBLAS, cuFFT and cuRAND.",
     "5 libraries, 3 levels", "From a whole-array call on the host down to a step inside a kernel.", parts)


# 11.2  2 passes against 1 fused pass, bytes drawn to scale.
scale = 380 / 768                             # units per MiB
x0 = 216
parts = [zone(16, 104, 628, 150, "transform, then reduce"),
         zone(16, 282, 628, 116, "transform_reduce")]
segments = [("read x", "client"), ("write x²", "danger"), ("read x² again", "store")]
for k, (name, kind) in enumerate(segments):
    x = x0 + k * 256 * scale
    parts += [box(x, 140, 256 * scale - 4, 34, kind, name, cls="pill")]
parts += [text(x0 - 16, 162, "768 MiB", "name", "end"),
          text(40, 212, "2 calls, 2 launches, a 256 MiB array of squares", "sub"),
          tag(470, 200, "measured 10.51 ms", "danger"),
          box(x0, 318, 256 * scale - 4, 34, "good", "read x", cls="pill"),
          text(x0 - 16, 340, "256 MiB", "name", "end"),
          text(x0 + 256 * scale + 12, 340, "x² stays in a register", "sub"),
          tag(470, 368, "measured 3.15 ms", "good"),
          note(40, 430, "Bytes moved for 67,108,864 floats: 3 × 256 MiB against 1 × 256 MiB. The fused", room=600),
          note(40, 448, "call ran 3.3 times as fast; the memory, not the arithmetic, sets the time.", room=600)]
save(OUT + "ch11-fusion.svg", 660, 470,
     "2 rows of bars drawn to scale. Top, transform then reduce: read 256 MiB, write 256 MiB of squares, read "
     "the squares again, 768 MiB in all, measured 10.51 ms. Bottom, transform_reduce: read 256 MiB once, "
     "each square kept in a register, measured 3.15 ms. A note says 3 times the bytes, 3.3 times the time.",
     "Fusion", "The sum of squares of 256 MiB of floats, on a Jetson Orin NX.", parts)


# 11.3  A radix sort by hand: 2 passes over 6 keys of 2 decimal digits.
KEYS = [53, 21, 37, 28, 15, 31]
DIGIT_KIND = {1: "client", 2: "service", 3: "store", 5: "outside", 7: "good", 8: "danger"}


def radix_pass(keys, place):
    digit = lambda k: k // place % 10
    count = [sum(1 for k in keys if digit(k) == d) for d in range(10)]
    start = [sum(count[:d]) for d in range(10)]
    out = sorted(keys, key=digit)              # Python's sort is stable, like the scatter
    return digit, count, start, out


parts = []
cw, gap = 44, 6
for p, place in enumerate((1, 10)):
    keys = KEYS if p == 0 else radix_pass(KEYS, 1)[3]
    digit, count, start, out = radix_pass(keys, place)
    y0 = 104 + p * 286
    parts.append(zone(16, y0, 628, 262, f"pass {p + 1}: the {'ones' if place == 1 else 'tens'} digit"))
    parts.append(text(40, y0 + 46, "keys", "sub"))
    for i, k in enumerate(keys):
        parts.append(box(150 + i * (cw + gap), y0 + 26, cw, 30, DIGIT_KIND[digit(k)], str(k)))
    parts.append(text(40, y0 + 96, "digit", "sub"))
    for d in range(10):
        x = 150 + d * 40
        parts += [mono(x + 18, y0 + 96, str(d), "middle", MUTED)]
    parts.append(text(40, y0 + 128, "count", "sub"))
    parts.append(text(40, y0 + 160, "start", "sub"))
    for d in range(10):
        x = 150 + d * 40
        kind = DIGIT_KIND.get(d, "idle")
        on = count[d] > 0
        parts += [box(x, y0 + 110, 36, 26, kind, str(count[d]), on),
                  box(x, y0 + 142, 36, 26, kind, str(start[d]), on)]
    parts += [text(626, y0 + 128, "histogram", "note", "end"), text(626, y0 + 160, "excl. scan", "note", "end")]
    parts.append(text(40, y0 + 214, "output", "sub"))
    for i, k in enumerate(out):
        parts += [box(150 + i * (cw + gap), y0 + 194, cw, 30, DIGIT_KIND[digit(k)], str(k)),
                  mono(150 + i * (cw + gap) + cw / 2, y0 + 244, str(i), "middle", MUTED)]
    parts.append(text(470, y0 + 214, "keys of 1 colour stay", "sub"))
    parts.append(text(470, y0 + 232, "in input order: stable", "sub"))
parts += [note(40, 690, "Each key goes to start[digit], and start[digit] then grows by 1. After the tens pass the",
               room=600),
          note(40, 708, "keys are in order, because the ones order survives inside each tens group.", room=600)]
save(OUT + "ch11-radix.svg", 660, 730,
     "2 passes of a radix sort on the keys 53, 21, 37, 28, 15, 31. Each pass shows 3 rows. Pass 1, by the "
     "ones digit: the keys coloured by their digit; a histogram row of counts for digits 0 to 9, 0 2 0 1 0 1 "
     "0 1 1 0; the exclusive scan of the counts, 0 0 2 2 3 3 4 4 5 6; and the output 21 31 53 15 37 28 in "
     "groups of colour. Pass 2, by the tens digit: counts 0 1 2 2 0 1 and starts 0 0 1 3 5 5, and the "
     "sorted output 15 21 28 31 37 53.",
     "A radix sort by hand", "Each pass: count the digits, scan the counts, place the keys.", parts)


# 11.4  Sorting speed, millions of keys per second.
x0, wide, top = 236, 300, 122
peak = 6000
parts = []
for g in range(0, peak + 1, 2000):
    x = x0 + wide * g / peak
    parts += [line(x, top - 6, x, top + len(SORT) * 44, EDGE, 1),
              mono(x, top + len(SORT) * 44 + 18, f"{g:,}", "middle", MUTED)]
for k, (name, rate, kind) in enumerate(SORT):
    y = top + k * 44
    w = wide * rate / peak
    parts += [text(x0 - 14, y + 20, name, "sub", "end"), hbar(x0, y + 6, w, 22, kind),
              mono(x0 + max(w, 3) + 8, y + 21, f"{rate:,.1f}", "start", INK)]
parts += [text(x0 - 14, top + len(SORT) * 44 + 18, "Mkeys/s", "sub", "end"),
          swatches(40, top + len(SORT) * 44 + 52, [(KINDS["client"][0], "radix sort"),
                                                   (KINDS["danger"][0], "merge sort"),
                                                   (KINDS["store"][0], "radix, allocating each call")]),
          note(40, top + len(SORT) * 44 + 84, "16,777,216 random 32-bit keys, 64 MiB; the last bar sorts keys from",
               room=600),
          note(40, top + len(SORT) * 44 + 102, "0 to 255 on their low 8 bits only, 1 pass instead of 4.", room=600)]
save(OUT + "ch11-sort.svg", 660, top + len(SORT) * 44 + 124,
     "Horizontal bars of sorting speed in millions of keys per second for 16,777,216 32-bit keys on a Jetson "
     "Orin NX. std::sort on 1 CPU core 10.3. thrust::sort 676.7. thrust::sort with scratch reused 1,285.5. "
     "With cuda::std::greater 542.6 and with its own comparison 686.8, both merge sorts. "
     "cub::DeviceRadixSort 1,872.8. Keys from 0 to 255 sorted on bits 0 to 7 only: 5,356.5.",
     "Sorting 16 Mi keys", "Millions of keys per second on a Jetson Orin NX.", parts)


# 11.5  CUB's 2 calls, and where the time of a small sum goes.
parts = [zone(16, 104, 252, 380, "CUB's 2 calls"), zone(284, 104, 360, 380, "a sum of 4,096 floats")]
steps = [("client", "1. Sum(nullptr, ...)", "writes bytes = 1,279"),
         ("store", "2. cudaMalloc(bytes)", "once, before the loop"),
         ("good", "3. Sum(temp, ...)", "runs; repeat at will")]
for k, (kind, name, sub) in enumerate(steps):
    y = 136 + k * 112
    parts.append(card(32, y, 220, 72, kind, name, sub))
    if k < 2:
        parts.append(arrow(142, y + 72, y + 112))
scale = 250 / 70
rows = [("thrust::reduce", [(38.2, "danger", "allocate, free"), (21.1, "store", "copy, wait"),
                            (8.8, "good", "sum")]),
        ("CUB, result copied", [(21.1, "store", "copy, wait"), (8.8, "good", "sum")]),
        ("CUB, result kept", [(8.8, "good", "sum")])]
for k, (name, segs) in enumerate(rows):
    y = 150 + k * 96
    parts.append(text(304, y, name, "name"))
    x = 304
    for value, kind, label in segs:
        w = value * scale
        parts.append(box(x, y + 12, w - 3, 30, kind, label if w > 70 else "", cls="pill"))
        x += w
    parts.append(mono(x + 6, y + 32, f"{sum(v for v, _, _ in segs):.1f} µs", "start", INK))
parts += [legend(304, 416, [("danger", "allocate"), ("store", "copy, wait"), ("good", "sum")]),
          note(304, 450, "Reading the 16 KiB itself takes 0.18 µs.", room=330),
          note(40, 514, "thrust::reduce allocates its scratch memory, copies the answer to the host and waits", room=600),
          note(40, 532, "for the GPU, on every call. CUB leaves all 3 to the caller.", room=600)]
save(OUT + "ch11-scratch.svg", 660, 554,
     "Left: CUB's 2 calls. Call 1 with nullptr returns 1,279 bytes; the program allocates them once; call 2 "
     "with the memory runs the sum and can be repeated. Right: 3 bars for a sum of 4,096 floats, drawn to "
     "scale. thrust::reduce, 68.1 microseconds: 38.2 allocating and freeing, 21.1 copying the result back "
     "and waiting, 8.8 launching and summing. CUB with the result copied, 29.9 microseconds: 21.1 copy and "
     "wait, 8.8 sum. CUB with the result kept, 8.8 microseconds. The data itself needs 0.18 microseconds.",
     "Scratch memory and waiting", "Measured per call on a Jetson Orin NX.", parts)


# 11.6  The same 6 numbers, read by rows and by columns.
parts = [text(40, 120, "memory", "sub")]
for i in range(6):
    parts.append(box(140 + i * 46, 102, 40, 30, "idle", str(i + 1)))
parts += [zone(16, 168, 300, 210, "read row by row: 2 × 3"),
          zone(344, 168, 300, 210, "read column by column: 3 × 2")]
rows_kind = ("client", "outside")
for r in range(2):
    for c in range(3):
        parts.append(box(70 + c * 50, 210 + r * 44, 44, 36, rows_kind[r], str(r * 3 + c + 1)))
    parts.append(text(240, 233 + r * 44, f"row {r}", "sub"))
for c in range(2):
    for r in range(3):
        parts.append(box(400 + c * 50, 202 + r * 44, 44, 36, rows_kind[c], str(c * 3 + r + 1)))
    parts.append(text(408 + c * 50, 350, f"col {c}", "sub"))
parts += [text(70, 330, "A", "name"), text(110, 330, "as C and C++ store it", "sub"),
          text(530, 248, "Aᵀ:", "name"), text(530, 268, "rows and", "sub"), text(530, 286, "columns", "sub"),
          text(530, 304, "swapped", "sub"),
          zone(16, 404, 628, 108, "the fix"),
          mono(40, 446, "(A B)ᵀ = Bᵀ Aᵀ", "start", INK),
          text(200, 446, "cuBLAS sees our A and B as Aᵀ and Bᵀ. Asked for", "sub"),
          text(200, 466, "Bᵀ Aᵀ = Cᵀ, it writes Cᵀ column by column, which is", "sub"),
          text(200, 486, "exactly C row by row. Swap the operands; copy nothing.", "sub")]
save(OUT + "ch11-layout.svg", 660, 534,
     "Top: 6 numbers in memory, 1 2 3 4 5 6. Left: read row by row as a 2 by 3 matrix, they are A, rows 1 2 3 "
     "and 4 5 6. Right: read column by column as a 3 by 2 matrix, the columns are 1 2 3 and 4 5 6, so the "
     "rows are 1 4, 2 5, 3 6: the transpose of A. Below, the rule: C = A B is the same as C transpose = B "
     "transpose A transpose, so a column-major library asked for B transpose times A transpose, with the "
     "operands swapped, writes C in row-major order.",
     "Row-major and column-major", "1 line of memory, 2 matrices.", parts)


# 11.7  FP32 and TF32 bits, and what they cost and buy.
def bits(y, mantissa, name, value):
    out = [text(40, y + 15, name, "name")]
    x = 120
    fields = [(1, "danger", "sign"), (8, "store", "exponent"), (23, "client", "mantissa")]
    for count, kind, label in fields:
        for b in range(count):
            on = kind != "client" or b < mantissa
            out.append(box(x + b * 14, y, 12, 22, kind, "", on))
        if kind != "client":
            out.append(text(x + count * 7, y + 40, label, "note", "middle"))
        x += count * 14 + 6
    out.append(text(120 + 15 * 14 + 12 + 6 * 14, y + 40, f"mantissa: {mantissa} of 23 bits", "note", "middle"))
    out.append(mono(120 + 32 * 14 + 22, y + 15, "", "start"))
    out.append(mono(624, y + 15, value, "end", INK))
    return out


parts = bits(126, 23, "FP32", "")
parts += bits(206, 10, "TF32", "")
parts += [tag(40, 290, "FP32: 1/3 = 0.33333334", "client"),
          tag(260, 290, "TF32: 1/3 = 0.33325195", "danger"),
          card(40, 336, 280, 80, "client", "cuBLAS, FP32", ["1,504.6 GFLOP/s", "max error 1.9e-5"]),
          card(344, 336, 280, 80, "good", "cuBLAS, TF32 tensor", ["6,530.4 GFLOP/s, 4.3 ×", "max error 4.0e-3, 210 ×"]),
          note(40, 448, "Same 8-bit exponent, so the same range; 13 fewer mantissa bits, so each input is", room=600),
          note(40, 466, "rounded to about 3 decimal digits before it is multiplied. N = 4,096, Jetson Orin NX.",
               room=600)]
save(OUT + "ch11-tf32.svg", 660, 488,
     "The bits of 2 formats as rows of boxes. FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits; 1/3 is "
     "0.33333334. TF32: 1 sign bit, the same 8 exponent bits, 10 mantissa bits, with the other 13 greyed out; "
     "1/3 is 0.33325195. Below, 2 measured results for N = 4,096 on a Jetson Orin NX: FP32 1,504.6 GFLOP/s "
     "with max error 1.9e-5; TF32 on tensor cores 6,530.4 GFLOP/s with max error 4.0e-3.",
     "FP32 and TF32", "What tensor cores keep of each number, and what that buys.", parts)


# 11.8  A signal and its spectrum.
parts = [zone(16, 104, 300, 272, "128 of 1,024 samples"), zone(372, 104, 272, 272, "513 frequencies")]
pts = []
for t in range(129):
    a = 2 * math.pi * t / 1024
    v = math.sin(50 * a) + 0.5 * math.sin(120 * a)
    pts.append(f"{36 + t * 2.1:.1f},{234 - v * 60:.1f}")
parts += [line(36, 234, 306, 234, EDGE, 1),
          f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{KINDS["client"][0]}" stroke-width="1.6"/>',
          text(36, 356, "sin(50 t) + 0.5 sin(120 t)", "sub")]
parts.append(wire([(318, 234), (370, 234)]))
parts.append(text(344, 220, "FFT", "pill", "middle", WIRE))
base, x0, span = 320, 392, 232
parts.append(line(x0, base, x0 + span, base, INK, 1.2))
for f, h, kind in ((50, 512, "client"), (120, 256, "outside")):
    x = x0 + span * f / 512
    hgt = h / 512 * 150
    parts += [f'  <rect x="{x - 4:.1f}" y="{base - hgt:.1f}" width="8" height="{hgt:.1f}" rx="2" '
              f'fill="{KINDS[kind][0]}"/>',
              mono(x + 10, base - hgt + 12, f"frequency {f}: {h}", "start", INK)]
parts += [mono(x0, base + 16, "0", "middle", MUTED), mono(x0 + span, base + 16, "512", "middle", MUTED),
          text(x0, 356, "every other magnitude is 0", "sub"),
          note(40, 408, "A wave of height h with a whole number of cycles in N samples gives 1 peak of h N / 2:", room=600),
          note(40, 426, "1 × 1,024 / 2 = 512 at frequency 50, and 0.5 × 512 = 256 at 120.", room=600)]
save(OUT + "ch11-spectrum.svg", 660, 448,
     "Left: the first 128 samples of the signal sin(50 t) + 0.5 sin(120 t), a fast wobble on a slower wave. "
     "An arrow labelled FFT leads right to the spectrum: 513 frequencies along the bottom, all 0 except a bar "
     "of height 512 at frequency 50 and a bar of height 256 at frequency 120.",
     "A spectrum with cuFFT", "cufftExecR2C on 1,024 real samples.", parts)


# 11.9  The book's kernels against the libraries.
panels = [("sum of 256 MiB", SUM, "GB/s", 100), ("scan of 64 Mi values", SCAN, "GB/s", 100),
          ("SGEMM, N = 4,096", GEMM, "GFLOP/s", 7000)]
x0, wide = 220, 300
parts, y = [], 104
for title, rows, unit, top in panels:
    h = 44 + len(rows) * 30
    parts.append(zone(16, y, 628, h, title))
    for k, (name, value) in enumerate(rows):
        yy = y + 30 + k * 30
        mine = name.startswith("Chapter")
        w = wide * value / top
        parts += [text(x0 - 14, yy + 15, name, "sub", "end"),
                  hbar(x0, yy + 3, w, 18, "idle" if mine else "client"),
                  mono(x0 + w + 8, yy + 16, f"{value:,.1f} {unit}", "start", INK)]
    y += h + 24
parts += [swatches(40, y + 4, [(KINDS["idle"][0], "the book's kernel"), (KINDS["client"][0], "library")])]
save(OUT + "ch11-versus.svg", 660, y + 26,
     "3 panels of horizontal bars, measured on a Jetson Orin NX. Sum of 256 MiB, in GB/s: Chapter 8 kernel "
     "93.4, cub::BlockReduce 92.5, cub::DeviceReduce 91.8, thrust::reduce 93.0. Scan of 64 Mi values, in "
     "GB/s: Chapter 8 43.0, thrust::inclusive_scan 68.2, cub::DeviceScan 74.3. SGEMM with N = 4,096, in "
     "GFLOP/s: Chapter 9 best 931.9, cuBLAS FP32 1,504.6, cuBLAS TF32 6,530.4.",
     "Hand-written against library", "Measured on a Jetson Orin NX, the same day.", parts)
