"""The figures of chapter 16, drawn with figkit and figparts.

    python3 tools/draw_ch16.py

Every number comes from a run quoted in the chapter, on the first GPU (an integrated GPU with 8 SMs): the Nsight Systems
profile of code/ch06_streams, and code/ch16_profiling/{measure,latency,verify,bugs}.cu.
"""
import math

from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, bar, card, legend, line, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"
CLOCK_GHZ = 1.173

# The 2-stream pipeline of Chapter 6 under nsys: (lane, stream, start ms, end ms) per operation.
PIPE8 = [("in", 0, 0.000, 0.663), ("kernel", 0, 0.666, 1.983), ("out", 0, 1.990, 2.828),
         ("in", 1, 0.665, 1.470), ("kernel", 1, 1.978, 3.311), ("out", 1, 3.315, 4.201),
         ("in", 0, 2.835, 3.771), ("kernel", 0, 3.777, 5.105), ("out", 0, 5.108, 5.945),
         ("in", 1, 4.211, 5.038), ("kernel", 1, 5.101, 6.429), ("out", 1, 6.431, 7.321),
         ("in", 0, 5.953, 6.897), ("kernel", 0, 6.906, 8.230), ("out", 0, 8.238, 9.073),
         ("in", 1, 7.330, 8.152), ("kernel", 1, 8.225, 9.551), ("out", 1, 9.553, 10.445),
         ("in", 0, 9.077, 10.021), ("kernel", 0, 10.030, 11.355), ("out", 0, 11.365, 12.207),
         ("in", 1, 10.451, 11.277), ("kernel", 1, 11.350, 12.681), ("out", 1, 12.690, 13.586)]
PIPE1 = [("in", 0, 0.000, 6.176), ("kernel", 0, 6.350, 22.746), ("out", 0, 22.753, 20.947)]

# y = 2 x over 256 MiB: the first 40 runs in ms, from a cold start.
WARMUP = [19.83, 14.43, 10.63, 13.25, 10.64, 11.40, 10.62, 9.67, 8.48, 8.46,
          8.47, 8.47, 8.46, 8.47, 10.43, 8.39, 8.37, 8.38, 8.35, 8.35,
          8.34, 8.35, 8.36, 8.33, 8.32, 8.36, 10.33, 8.38, 8.40, 8.35,
          8.34, 8.34, 12.91, 9.80, 8.36, 8.35, 8.36, 8.36, 8.34, 8.35]


# 22.1  The question decides the tool.
rows = [("Where does the time go?", "the whole program", "Nsight Systems, nsys", "a timeline", "22.2"),
        ("Why is 1 kernel slow?", "1 kernel", "Nsight Compute, ncu", "hardware counters", "22.3"),
        ("Is every access legal?", "every access", "Compute Sanitizer", "memcheck, racecheck …",
         "22.4"),
        ("What does thread 232 hold?", "1 thread", "cuda-gdb", "breakpoints in kernels", "22.5"),
        ("How long does 1 load take?", "a few instructions", "clock64()", "cycles, inside a kernel", "22.6"),
        ("Can the number be trusted?", "a measurement", "a benchmark protocol", "warm-up, events, medians",
         "22.7"),
        ("Is the answer right?", "the output", "a CPU reference", "with a tolerance", "22.8")]
parts = [zone(16, 104, 300, 7 * 62 + 24, "the question"), zone(344, 104, 300, 7 * 62 + 24, "the instrument")]
for k, (question, scope, tool, how, section) in enumerate(rows):
    y = 128 + k * 62
    kind = ("client", "outside", "danger", "danger", "store", "service", "good")[k]
    parts += [card(32, y, 268, 50, "idle", question, scope),
              card(360, y, 268, 50, kind, tool, how, pill=f"§{section}"),
              wire([(300, y + 25), (360, y + 25)])]
parts += [note(16, 586, "A profiler, a debugger and a stopwatch each answer 1 kind of question. The first step is to", room=628),
          note(16, 604, "name the question; the instrument follows from it.", room=628)]
save(OUT + "ch25-tools.svg", 660, 626,
     "Seven rows, each joining a question to an instrument. Where does the time go: Nsight Systems, a "
     "timeline, section 22.2. Why is 1 kernel slow: Nsight Compute, hardware counters, 22.3. Does it touch "
     "Is every access legal: Compute Sanitizer, 22.4. What does thread 232 hold: cuda-gdb, 22.5. How long does 1 "
     "load take: clock64 inside a kernel, 22.6. Can the number be trusted: events, warm-up and medians, 22.7. "
     "Is the answer right: a CPU reference with a tolerance, 22.8.",
     "1 question, 1 instrument", "Each tool of the chapter answers 1 kind of question.", parts)


# 22.2  The nsys timeline of Chapter 6's pipeline: 1 stream, then 2 streams and 8 chunks.
x0, scale = 128, 23.0                          # x of 0 ms, pixels per ms
LANES = {"in": ("copy in", "client"), "kernel": ("kernel", "store"), "out": ("copy out", "good")}


def lanes(y, ops, title, total):
    out = [text(32, y - 10, title, "name"), tag(628, y - 23, f"total {total:.2f} ms", "idle", "end")]
    for row, key in enumerate(("in", "kernel", "out")):
        name, kind = LANES[key]
        ly = y + row * 30
        out += [text(x0 - 12, ly + 15, name, "sub", "end"),
                line(x0, ly + 22, x0 + 21.5 * scale, ly + 22, EDGE, 1)]
        for lane, stream, a, b in ops:
            if lane == key:
                label = "" if (b - a) * scale < 30 else ("A" if stream == 0 else "B")
                out.append(bar(x0 + a * scale, ly, (b - a) * scale - 1.5, 20, kind, label,
                               dashed=stream == 1))
    return out


parts = lanes(130, PIPE1, "1 stream, 1 chunk of 128 MiB", 20.947)
parts += lanes(280, PIPE8, "2 streams, 8 chunks of 16 MiB", 13.586)
for ms in range(0, 22, 5):
    x = x0 + ms * scale
    parts += [line(x, 112, x, 384, EDGE, 1, "3 4"), mono(x, 402, f"{ms} ms", "middle", MUTED)]
parts += [bar(x0, 420, 30, 16, "store"), text(x0 + 38, 433, "stream A", "note"),
          bar(x0 + 120, 420, 30, 16, "store", dashed=True), text(x0 + 158, 433, "stream B", "note"),
          note(32, 462, "Kernel lane busy: 10.40 of 20.95 ms with 1 stream (50%), and 8 × 1.33 = 10.61 of", room=600),
          note(32, 480, "13.59 ms with 2 streams (78%). With 2 streams each kernel starts as soon as its own", room=600),
          note(32, 498, "copy in ends, so the copies hide behind the kernels of the other stream.", room=600)]
save(OUT + "ch25-timeline.svg", 660, 520,
     "Two timelines from Nsight Systems, each with 3 lanes: copy in, kernel, copy out, on a scale from 0 to "
     "21 milliseconds. With 1 stream: copy in from 0 to 6.18 ms, kernel from 6.35 to 16.75 ms, copy out "
     "from 16.75 to 20.95 ms, one after the other. With 2 streams and 8 chunks: 8 kernels of about 1.33 ms "
     "each run almost back to back from 0.67 to 12.68 ms, alternating streams A and B, while copies of the "
     "other stream run beside them; the whole run ends at 13.59 ms.",
     "Chapter 6's pipeline, as nsys saw it", "Measured on a GPU with 8 SMs; stream A solid, stream B dashed.",
     parts)


# 22.3  Why the missing guard went unnoticed: 4,000-byte arrays in 4,096-byte slots.
x0, slot = 40, 580 / 3                        # 1 slot of 4,096 bytes per third of the strip
used = slot * 4000 / 4096
pad = slot - used
parts = [text(x0, 128, "device memory: 3 allocations of 4,000 bytes, in slots of 4,096 bytes", "sub")]
for k, name in enumerate(("a", "b", "c")):
    x = x0 + k * slot
    parts += [bar(x, 140, used - 2, 40, "client" if k < 2 else "good", f"{name}[0] … {name}[999]"),
              bar(x + used, 140, pad - 2, 40, "danger", dashed=True),
              mono(x, 200, f"+{k * 4096:,} bytes", "start", MUTED)]
gap = x0 + used + pad / 2
parts += [zone(16, 240, 628, 150, "enlarged")]
cells = [("a[997]", "client"), ("a[998]", "client"), ("a[999]", "client")]
cells += [("", "danger")] * 24 + [("b[0]", "client")]
at = 40
for k, (name, kind) in enumerate(cells):
    w = 54 if name else 15
    parts.append(bar(at, 296, w - 3, 34, kind, name, dashed=not name))
    if k == 3:
        first = at
    at += w
last = at - 54 - 3
words = "threads 1,000 to 1,023: block 3, threads 232 to 255"
from figkit import width_of
parts += [wire([(gap, 182), (gap, 230), (first + 8, 230), (first + 8, 293)]),
          line(first, 286, last, 286, KINDS["danger"][0], 1.4),
          line(first, 282, first, 290, KINDS["danger"][0], 1.4),
          line(last, 282, last, 290, KINDS["danger"][0], 1.4),
          tag((first + last) / 2 - (width_of(words, 11, bold=True) + 14) / 2, 258, words, "danger"),
          mono(first, 352, "a + 4,000 bytes", "start", MUTED),
          mono(at - 3, 352, "a + 4,096 bytes", "end", MUTED),
          note(40, 374, "24 floats of padding, 96 bytes, that belong to no array", room=560),
          note(40, 420, "cudaMalloc gave each 4,000-byte array a 4,096-byte slot. 1,024 threads − 1,000 elements =", room=600),
          note(40, 438, "24 extra threads; 24 × 4 bytes = 96 bytes, exactly the padding. The stray accesses touched", room=600),
          note(40, 456, "no other array, so the program saw nothing wrong; memcheck reports every one of them.",
               room=600)]
save(OUT + "ch25-bounds.svg", 660, 478,
     "A strip of device memory with 3 allocations a, b and c of 4,000 bytes, each in a slot of 4,096 bytes, "
     "with 96 bytes of padding at the end of each slot. Threads 1,000 to 1,023, which are threads 232 to 255 of "
     "block 3, read and write inside the padding: 24 floats, 96 bytes. An enlarged view shows a[996] to a[999], "
     "the 24 padding floats, then b[0] at a + 4,096 bytes. No other array is touched, so the program sees no "
     "wrong element.",
     "Why the missing guard went unnoticed", "24 stray threads × 4 bytes fit exactly in the padding.", parts)


# 22.4  The race of the block sum without barriers: 2 warps, 1 shared value.
parts = [zone(16, 104, 628, 250, "1 block, no barrier between the steps")]
for k, (name, events) in enumerate([("warp 0, threads 0 to 31", [(0, 3, "s[t] = in[…]", "client"),
                                                                  (3.2, 6.6, "s[t] += s[t + 128]", "danger"),
                                                                  (6.8, 9.6, "s[t] += s[t + 64]", "client")]),
                                    ("warp 4, threads 128 to 159", [(4.4, 7.4, "s[t] = in[…]", "store")])]):
    y = 150 + k * 70
    parts += [text(40, y - 8, name, "sub"), line(40, y + 26, 620, y + 26, EDGE, 1)]
    for a, b, label, kind in events:
        parts.append(bar(40 + a * 58, y, (b - a) * 58 - 3, 24, kind, label))
parts += [line(40 + 3.2 * 58, 176, 40 + 3.2 * 58, 312, KINDS["danger"][0], 1.4, "4 3"),
          line(40 + 4.4 * 58, 246, 40 + 4.4 * 58, 312, KINDS["store"][0], 1.4, "4 3"),
          tag(40 + 3.2 * 58 - 4, 316, "read of s[128]", "danger", "end"),
          tag(40 + 4.4 * 58 + 4, 316, "write of s[128]", "store"),
          mono(620, 340, "time →", "end", MUTED),
          note(16, 384, "Nothing makes warp 4 store s[128] before warp 0 reads it. When warp 0 runs first it adds", room=628),
          note(16, 402, "whatever s[128] held before, and the block sum is wrong. racecheck reports the pair as", room=628),
          note(16, 420, "a write at line 31 and a read at line 35 of the same shared location.", room=628)]
save(OUT + "ch25-race.svg", 660, 442,
     "Two lanes of time for 1 block. Warp 0, threads 0 to 31, stores its own values, then at once adds "
     "s[t + 128] into s[t], which for thread 0 reads s[128]. Warp 4, threads 128 to 159, stores s[128] only later. The read of s[128] comes before "
     "its write: a race, which racecheck reports as a write at line 31 and a read at line 35.",
     "The race in the block sum", "Without a barrier, a read can come before the write it needs.", parts)


# 22.5  Latencies in cycles, on a log scale.
LAT = [("1 multiply-add, dependent", 4.24, None, "store"), ("1 multiply-add, 4 chains", 1.31, None, "good"),
       ("1 load from L1", 39, None, "client"), ("1 load from L2", 191, None, "outside"),
       ("1 load from DRAM", 598, 1137, "danger")]
x0, wide = 210, 300
lx = lambda c: x0 + wide * math.log10(c) / math.log10(2000)
parts = []
for c in (1, 10, 100, 1000):
    parts += [line(lx(c), 120, lx(c), 120 + 5 * 52, EDGE, 1, "3 4"),
              mono(lx(c), 120 + 5 * 52 + 18, f"{c:,}", "middle", MUTED)]
for k, (name, lo, hi, kind) in enumerate(LAT):
    y = 132 + k * 52
    accent = KINDS[kind][0]
    parts += [text(x0 - 14, y + 15, name, "sub", "end"),
              f'  <rect x="{x0}" y="{y}" width="{lx(lo) - x0:.1f}" height="22" rx="5" fill="{accent}"/>']
    if hi:
        parts.append(bar(lx(lo), y, lx(hi) - lx(lo), 22, kind, dashed=True))
    value = f"{lo:g} to {hi:,} cycles" if hi else f"{lo:g} cycles"
    parts.append(mono(lx(hi or lo) + 8, y + 15, value, "start", INK))
parts += [text(x0 - 14, 120 + 5 * 52 + 18, "cycles", "sub", "end"),
          note(40, 430, "Measured with clock64() by 1 thread, at 1,173 MHz: 1 cycle = 0.85 ns.", room=600),
          note(40, 448, "A DRAM load costs about 140 dependent multiply-adds; the range comes from 14 runs.", room=600)]
save(OUT + "ch25-latency.svg", 660, 470,
     "Horizontal bars on a logarithmic scale of clock cycles. A dependent multiply-add takes 4.24 cycles; with "
     "4 independent chains each takes 1.31 cycles. A load from L1 takes 39 cycles, from L2 191 cycles, and "
     "from DRAM 598 to 1,137 cycles over 14 runs.",
     "Latencies, measured inside a kernel", "1 thread, timed with clock64(), on a GPU at 1,173 MHz.", parts)


# 22.6  The first 40 runs of y = 2 x: warm-up, then a plateau, then outliers.
x0, y0, high, step = 72, 400, 260, 11.5
ys = lambda ms: y0 - high * ms / 21
right = x0 + 40 * step
parts = []
for ms in (0, 5, 10, 15, 20):
    parts += [line(x0, ys(ms), right, ys(ms), EDGE, 1, "3 4"), mono(x0 - 8, ys(ms) + 4, f"{ms}", "end", MUTED)]
for r, ms in enumerate(WARMUP):
    kind = "danger" if r < 8 else ("store" if ms > 9 else "client")
    accent, tint = KINDS[kind]
    parts.append(f'  <rect x="{x0 + r * step + 1.5}" y="{ys(ms):.1f}" width="8.5" height="{y0 - ys(ms):.1f}" '
                 f'rx="2" fill="{tint}" stroke="{accent}" stroke-width="1"/>')
mean = sum(WARMUP) / 40
parts += [line(x0, ys(mean), right + 6, ys(mean), KINDS["danger"][0], 1.4, "6 4"),
          line(x0 + 20 * step, ys(8.35), right + 6, ys(8.35), KINDS["good"][0], 1.8),
          tag(right + 10, ys(mean) - 16, f"mean {mean:.2f} ms", "danger"),
          tag(right + 10, ys(8.35) - 2, "median 8.35 ms", "good"),
          note(right + 10, ys(8.35) + 32, "of runs 21 to 40", room=110),
          text(x0 - 8, 112, "ms", "sub", "end"),
          mono(x0 + 6, y0 + 18, "1", "middle", MUTED), mono(x0 + 39 * step + 6, y0 + 18, "40", "middle", MUTED),
          text(x0 + 20 * step, y0 + 18, "run", "sub", "middle"),
          tag(x0 + 20, ys(19.83) - 6, "run 1: 19.83 ms", "danger"),
          note(40, 448, "Runs 1 to 8 (red) pay for the cold start and the low clock; runs 9 to 40 sit near 8.35 ms,", room=600),
          note(40, 466, "with 4 outliers above 9 ms (amber) when other work used the GPU. The mean counts every", room=600),
          note(40, 484, "one of them; the median of the warm runs does not.", room=600)]
save(OUT + "ch25-warmup.svg", 660, 506,
     "Bars of 40 run times of y = 2 x over 256 MiB on 1 GPU. Run 1 takes 19.83 ms and runs 2 to 8 "
     "between 9.67 and 14.43 ms; from run 9 the times sit near 8.35 ms, with outliers of 10.43, 10.33, 12.91 "
     "and 9.80 ms. A dashed line marks the mean of all 40 runs, 9.46 ms; a solid line the median of runs 21 "
     "to 40, 8.35 ms.",
     "The first 40 runs of 1 kernel", "y = 2 x over 256 MiB, timed with events, from a cold start.", parts)


# 22.7  Units in the last place: the spacing of floats doubles at every power of 2.
x0, unit = 50, 35                               # x of 0, pixels per 1
parts = [zone(16, 104, 628, 196, "a toy float: 8 values between 2 powers of 2")]
parts += [line(x0, 190, x0 + 16 * unit, 190, WIRE, 1.6)]
for k, (lo, kind) in enumerate(((1, "store"), (2, "client"), (4, "outside"), (8, "good"))):
    gap = lo / 8
    for t in range(8):
        x = x0 + (lo + t * gap) * unit
        parts.append(line(x, 178, x, 202, KINDS[kind][0], 1.8))
    parts += [mono(x0 + lo * unit, 222, f"{lo}", "middle", INK),
              tag(x0 + lo * unit, 150 if k % 2 == 0 else 128, f"gap {gap:g}", kind)]
parts += [mono(x0 + 16 * unit, 222, "16", "middle", INK), mono(x0, 222, "0", "middle", INK),
          note(40, 256, "Each stretch from 2ᵉ to 2ᵉ⁺¹ holds the same 8 values, so each gap is twice the last.",
               room=590),
          note(40, 274, "A real float holds 2²³ = 8,388,608 values in each stretch.", room=590)]
cards = [("store", "near 1", ["1 ULP = 2⁻²³ ≈ 0.000000119"]),
         ("good", "near e¹⁰ = 22,026.47", ["1 ULP = 2⁻⁹ ≈ 0.00195"]),
         ("client", "expf, measured", ["at most 2 ULPs from the", "nearest float to exp(x)"]),
         ("danger", "__expf, measured", ["at most 8 ULPs, over", "x in [−10, 10)"])]
for k, (kind, name, sub) in enumerate(cards):
    x, y = 16 + (k % 2) * 318, 324 + (k // 2) * 84
    parts.append(card(x, y, 310, 70, kind, name, sub))
parts += [note(16, 508, "An error in ULPs means the same at every size: 2 ULPs is about 2 parts in 8 million.",
               room=628)]
save(OUT + "ch25-ulp.svg", 660, 530,
     "Top: a toy float with 8 values between each 2 powers of 2, on a line from 0 to 16. Between 1 and 2 the "
     "values are 1/8 apart, between 2 and 4 1/4 apart, between 4 and 8 1/2 apart, and between 8 and 16 1 apart. "
     "Bottom: 4 cards. Near 1, 1 ULP of a real float is 2 to the minus 23, about 0.000000119. Near e to the 10, "
     "22,026.47, 1 ULP is 2 to the minus 9, about 0.00195. expf was measured at most 2 ULPs from the nearest "
     "float to exp(x), __expf at most 8 ULPs, over x from minus 10 to 10.",
     "Units in the last place", "The gap between neighboring floats doubles at every power of 2.", parts)


# 22.8  A tolerance that is too wide hides a bug; an input with a known answer does not.
parts = [zone(16, 104, 628, 172, "random inputs: tolerance 42.01"), zone(16, 302, 628, 152, "all ones: tolerance 0")]
cx, half = 330, 260
parts += [bar(cx - half, 160, 2 * half, 30, "good", dashed=True),
          line(cx, 150, cx, 200, INK, 1.6),
          mono(cx, 216, "exact", "middle", INK),
          mono(cx - half, 216, "−42.01", "middle", MUTED), mono(cx + half, 216, "+42.01", "middle", MUTED)]
for err, name, kind, dy in ((0.42, "all n: error 0.42", "client", 0), (0.58, "n − 1: error 0.58", "danger", 1)):
    x = cx + err / 42.01 * half
    parts += [f'  <circle cx="{x:.1f}" cy="{168 + 14 * dy}" r="5" fill="{KINDS[kind][0]}"/>',
              tag(x + 10 if dy else x - 10, 232 + dy * 0, name + ", PASSED", kind, "start" if dy else "end")]
parts += [line(cx, 340, cx, 400, INK, 1.6), mono(cx, 418, "16,777,216", "middle", INK),
          f'  <circle cx="{cx}" cy="370" r="5" fill="{KINDS["client"][0]}"/>',
          f'  <circle cx="{cx - 60}" cy="370" r="5" fill="{KINDS["danger"][0]}"/>',
          mono(cx - 60, 400, "16,777,215", "middle", KINDS["danger"][0]),
          tag(cx + 12, 346, "all n: exact, PASSED", "client"),
          tag(cx - 72, 346, "n − 1: off by 1, FAILED", "danger", "end"),
          note(16, 486, "The missing element is less than 1, far inside a tolerance of 42. With ones, every", room=628),
          note(16, 504, "partial sum is a whole number a float holds exactly, so the right answer is exact and", room=628),
          note(16, 522, "any difference at all is a bug.", room=628)]
save(OUT + "ch25-tolerance.svg", 660, 544,
     "Two panels. Random inputs: a band from minus 42.01 to plus 42.01 around the exact sum; the correct sum is "
     "0.42 away and the sum that misses 1 element is 0.58 away, both inside the band, both PASSED. All ones: "
     "the exact sum is 16,777,216; the correct sum hits it, the sum that misses 1 element gives 16,777,215 and "
     "FAILED.",
     "Tolerance and known answers", "The off-by-one sum passes 1 test and fails the other.", parts)


# 22.9  A kernel report by hand: how close 2 kernels come to the memory and the arithmetic peaks.
REPORT = [("y = 2 x, 256 MiB", [("memory", 64.3 / 102 * 100, "64.3 of 102 GB/s", "client", False),
                                ("FP32", 8.04 / 2400 * 100, "8.0 of 2,400 GFLOP/s", "store", False)]),
          ("4 × 4 per thread (Ch. 9)", [("loads", 56.3 / 102 * 100, "56.3 GB/s, L2 or DRAM", "client", True),
                                        ("FP32", 901.4 / 2400 * 100, "901 of 2,400 GFLOP/s", "store", False)])]
x0, wide = 112, 300
parts = []
for g in range(0, 101, 25):
    x = x0 + wide * g / 100
    parts += [line(x, 128, x, 332, EDGE, 1, "3 4"), mono(x, 350, f"{g}%", "middle", MUTED)]
for k, (name, rows) in enumerate(REPORT):
    y = 124 + k * 108
    parts.append(text(32, y, name, "name"))
    for r, (what, share, label, kind, dashed) in enumerate(rows):
        yy = y + 14 + r * 38
        w = max(wide * share / 100, 3)
        parts += [text(x0 - 12, yy + 16, what, "sub", "end"),
                  bar(x0, yy, w, 24, kind, dashed=dashed),
                  mono(x0 + w + 8, yy + 16, f"{share:.2g}%", "start", INK),
                  text(x0 + wide + 22, yy + 16, label, "sub")]
parts += [note(32, 384, "Percentages of data-sheet peaks of 102 GB/s and 2,400 GFLOP/s. y = 2 x uses 63% of the", room=600),
          note(32, 402, "memory and 0.33% of the arithmetic: memory-bound. The patch kernel uses 38% of the", room=600),
          note(32, 420, "arithmetic; only a profiler's counters can split its loads between L2 and DRAM.", room=600)]
save(OUT + "ch25-report.svg", 660, 442,
     "A chart like a profiler's speed-of-light section, in percent of the peaks of a GPU with 2,400 GFLOP/s and 102 GB/s. y = 2 x over 256 "
     "MiB: memory 64.3 of 102 GB/s, 63%; arithmetic 8.0 of 2,400 GFLOP/s, 0.3%. Chapter 9's 4 by 4 patch "
     "kernel: loads of 56.3 GB/s from L2 or DRAM, 55% of the memory peak if all came from DRAM, drawn dashed; "
     "arithmetic 901 of 2,400 GFLOP/s, 38%.",
     "A speed-of-light report, by hand", "How close 2 kernels come to the memory and the arithmetic peaks.",
     parts)


# 22.10  The engineering loop, with the numbers of the Rust pointer chase.
steps = [("measure", "store", "Rust: 62 cycles per L1 load", "C++: 39 cycles"),
         ("hypothesis", "outside", "the bounds check of next[j]", "sits on the chain"),
         ("change 1 thing", "client", "next.get_unchecked(j)", "with a SAFETY comment"),
         ("measure again", "store", "39 cycles per L1 load", "same as C++"),
         ("verify", "good", "every chain ends at the", "index the host expects")]
pos = [(40, 130), (360, 130), (360, 260), (40, 260), (200, 390)]
parts = []
for (name, kind, a, b), (x, y) in zip(steps, pos):
    parts.append(card(x, y, 260, 72, kind, name, [a, b]))
parts += [wire([(300, 166), (360, 166)]),
          wire([(490, 202), (490, 260)]),
          wire([(360, 296), (300, 296)]),
          wire([(170, 332), (170, 426), (200, 426)]),
          note(40, 498, "1 change between 2 measurements. A 2nd change (a while loop in place of a for loop) was", room=600),
          note(40, 516, "measured separately, so each difference has 1 cause.", room=600)]
save(OUT + "ch25-loop.svg", 660, 538,
     "Five cards joined by arrows in a loop. Measure: the Rust pointer chase takes 62 cycles per L1 load, the "
     "C++ one 39. Hypothesis: the bounds check of next[j] sits on the chain. Change 1 thing: next.get_unchecked(j) "
     "with a SAFETY comment. Measure again: 39 cycles, the same as C++. Verify: every chain ends at the index "
     "the host expects.",
     "The engineering loop, once round", "A real case from this chapter's Rust program.", parts)
