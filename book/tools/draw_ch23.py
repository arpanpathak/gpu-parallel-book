"""The figures of chapter 20, drawn with figkit and figparts.

    python3 tools/draw_ch20.py

The numbers in the hand-worked figures are computed here, so a figure cannot disagree with
its own arithmetic. The measured speeds come from code/oxide/ch23_softmax_layernorm.
"""
import math

from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, bar, card, line, mono, note, save, tag,
                      text, wire, zone)

OUT = "src/figures/"
X = [1.0, 2.0, 3.0, 4.0]


def cell(x, y, w, h, kind, label, on=True):
    """A value in a rounded box: tinted with an accent edge when `on`, white when not."""
    accent, tint = KINDS[kind]
    return "\n".join([
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
        f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>',
        text(x + w / 2, y + h / 2 + 4, label, "mono", "middle", accent if on else MUTED)])


def column(x, y, title, values, kind, w=96, h=30, gap=8):
    """A heading and a column of values, 1 per logit."""
    parts = [text(x + w / 2, y, title, "name", "middle")]
    parts += [cell(x, y + 14 + k * (h + gap), w, h, kind, v) for k, v in enumerate(values)]
    return parts


# 23.1  Softmax of 4 logits, with and without the maximum subtracted.
e = [math.exp(v) for v in X]
total = sum(e)
p = [v / total for v in e]
top = max(X)
e2 = [math.exp(v - top) for v in X]
total2 = sum(e2)
parts = [zone(16, 96, 628, 300, "softmax of 4 logits")]
parts += column(40, 136, "x", [f"{v:.0f}" for v in X], "client", w=56)
parts += column(116, 136, "exp(x)", [f"{v:.3f}" for v in e], "store")
parts += column(232, 136, "x − 4", [f"{v - top:.0f}" for v in X], "outside", w=64)
parts += column(316, 136, "exp(x − 4)", [f"{v:.4f}" for v in e2], "outside")
parts += column(432, 136, "p", [f"{v:.4f}" for v in p], "good")
parts += [mono(164, 318, f"sum {total:.3f}", "middle", KINDS["store"][0]),
          mono(364, 318, f"sum {total2:.4f}", "middle", KINDS["outside"][0]),
          mono(480, 318, f"sum {sum(p):.4f}", "middle", KINDS["good"][0]),
          note(40, 350, f"2.718 ÷ {total:.3f} = {p[0]:.4f} and 0.0498 ÷ {total2:.4f} = {p[0]:.4f}: subtracting "
                        "the maximum, 4, divides", room=590),
          note(40, 368, "every exponential and the sum by the same factor exp(4), so each p is unchanged.",
               room=590)]
save(OUT + "ch23-softmax.svg", 660, 412,
     f"A table for the logits 1, 2, 3 and 4. Their exponentials are 2.718, 7.389, 20.086 and 54.598, which "
     f"sum to {total:.3f}. Subtracting the maximum 4 gives -3, -2, -1 and 0, whose exponentials 0.0498, "
     f"0.1353, 0.3679 and 1 sum to {total2:.4f}. Both routes give the same probabilities 0.0321, 0.0871, "
     f"0.2369 and 0.6439, which sum to 1.",
     "Softmax by hand", "Each probability is exp(x) divided by the sum of the exponentials of its row.", parts)


# 23.2  Overflow: exp(100) does not fit in a float; exp(100 - 103) does.
big = [100.0, 101.0, 102.0, 103.0]
parts = [zone(16, 96, 306, 330, "exp(x)"), zone(338, 96, 306, 330, "exp(x − 103)")]
for k, v in enumerate(big):
    y = 136 + k * 40
    parts += [cell(36, y, 60, 30, "client", f"{v:.0f}"),
              cell(112, y, 92, 30, "danger", f"{math.exp(v):.1e}".replace("e+", "e")),
              cell(220, y, 82, 30, "danger", "inf"),
              cell(358, y, 60, 30, "client", f"{v - 103:.0f}"),
              cell(434, y, 92, 30, "outside", f"{math.exp(v - 103):.4f}"),
              cell(542, y, 82, 30, "good", f"{math.exp(v - 103) / total2:.4f}")]
parts += [text(66, 128, "x", "mono", "middle"), text(158, 128, "exact", "mono", "middle"),
          text(261, 128, "float", "mono", "middle"), text(388, 128, "x − m", "mono", "middle"),
          text(480, 128, "exp", "mono", "middle"), text(583, 128, "p", "mono", "middle"),
          cell(112, 304, 190, 30, "danger", "inf ÷ inf = NaN"),
          cell(434, 304, 190, 30, "good", f"sum {total2:.4f}"),
          note(36, 362, "The largest float is 3.4e38 = exp(88.7).", room=270),
          note(36, 380, "exp(100) = 2.7e43 rounds to infinity.", room=270),
          note(358, 362, "Every x − m is 0 or less, so every", room=270),
          note(358, 380, "exp lies in (0, 1] and cannot overflow.", room=270)]
save(OUT + "ch23-overflow.svg", 660, 442,
     "Left: the logits 100, 101, 102 and 103 have exponentials from 2.7e43 to 5.4e44, all above the "
     "largest float, 3.4e38, so each becomes infinity and the division gives NaN. Right: after "
     "subtracting the maximum 103 the arguments are -3 to 0, the exponentials 0.0498 to 1 sum to 1.5530, "
     "and the probabilities are 0.0321, 0.0871, 0.2369 and 0.6439.",
     "Overflow", "Subtracting the row maximum keeps every exponential between 0 and 1.", parts)


# 23.3  1 block per row: thread t reads columns t, t + 256, ..., then the block reduces.
parts = [zone(16, 96, 628, 400, "1 block of 256 threads per row of 4,096")]
seg = 36
for k in range(16):
    x = 40 + k * seg
    parts.append(cell(x, 132, seg - 4, 26, "client" if k % 2 == 0 else "idle", str(k), True))
parts += [note(40, 178, "The row as 16 groups of 256 columns; group k holds columns 256 k to 256 k + 255.",
               room=590)]
for k in range(16):
    x = 40 + k * seg + (seg - 4) / 2
    parts.append(f'  <circle cx="{x:.1f}" cy="200" r="4" fill="{KINDS["store"][0]}"/>')
parts += [note(40, 228, "Thread 5 reads column 5 of every group: 5, 261, 517, ..., 3,845, 16 values.",
               room=590),
          card(40, 252, 270, 64, "store", "256 partial results", "1 per thread, in registers"),
          card(350, 252, 270, 64, "service", "8 warp results", "5 shuffle steps in each warp"),
          wire([(310, 284), (350, 284)]),
          card(350, 344, 270, 64, "outside", "8 slots in shared memory", "lane 0 of each warp writes"),
          arrow(485, 316, 344),
          card(40, 344, 270, 64, "good", "1 row result", "every thread reads the 8 slots"),
          wire([(350, 376), (310, 376)]),
          note(40, 440, "2 barriers per reduction: after the slots are written, and after they are read,",
               room=590),
          note(40, 458, "so the next reduction of the same block cannot overwrite a slot still in use.",
               room=590)]
save(OUT + "ch23-row-block.svg", 660, 512,
     "A row of 4,096 columns drawn as 16 groups of 256. Thread 5 reads column 5 of every group, 16 values "
     "in all. The 256 partial results become 8 warp results by shuffles, the 8 are written to shared "
     "memory by lane 0 of each warp, and every thread then reads the 8 slots to obtain the row result.",
     "1 block per row", "Each thread reduces 16 values, each warp 32 threads, and the block 8 warps.", parts)


# 23.4  Global memory traffic per element for the 3 softmax kernels.
parts = [zone(16, 96, 628, 300, "global memory traffic per element")]
rows = [("3 passes", [("read", "max"), ("read", "sum"), ("read", "p"), ("write", "p")], 16),
        ("online", [("read", "max + sum"), ("read", "p"), ("write", "p")], 12),
        ("fused", [("read", "registers"), ("write", "p")], 8)]
for r, (name, steps, bytes_) in enumerate(rows):
    y = 140 + r * 80
    parts += [text(40, y + 20, name, "name")]
    x = 140
    for kind, purpose in steps:
        parts.append(bar(x, y, 92, 30, "danger" if kind == "write" else "client", f"{kind} 4 B"))
        parts.append(note(x + 46, y + 48, purpose, "middle"))
        x += 100
    parts.append(tag(600, y + 6, f"{bytes_} B", "store", anchor="end"))
parts += [note(40, 372, "The fused kernel reads each value once and keeps it in a register for all 3 steps.",
               room=590)]
save(OUT + "ch23-passes.svg", 660, 412,
     "3 rows of bars. The 3-pass kernel reads each element 3 times, for the maximum, the sum and the output, "
     "and writes it once: 16 bytes. The online kernel reads it twice and writes once: 12 bytes. The fused "
     "kernel reads once and writes once: 8 bytes, because its values stay in registers.",
     "Bytes per element", "Every pass over the row reads each element from memory again.", parts)


# 23.5  The online softmax by hand on 1, 2, 3, 4.
parts = [zone(16, 96, 628, 330, "online maximum and sum")]
m, s = X[0], 1.0
trace = [("start", X[0], m, s, "m = 1, s = exp(0) = 1")]
for v in X[1:]:
    new = max(m, v)
    s_new = s * math.exp(m - new) + math.exp(v - new)
    trace.append((f"x = {v:.0f}", v, new, s_new, f"{s:.4f} × exp({m:.0f} − {new:.0f}) + exp({v:.0f} − {new:.0f})"))
    m, s = new, s_new
parts += [text(40, 132, "step", "name"), text(130, 132, "m", "name"), text(200, 132, "s", "name"),
          text(300, 132, "how s is updated", "name")]
for k, (label, _, mk, sk, how) in enumerate(trace):
    y = 146 + k * 42
    parts += [text(40, y + 20, label, "mono"),
              cell(112, y, 52, 30, "outside", f"{mk:.0f}"),
              cell(180, y, 92, 30, "store", f"{sk:.4f}"),
              text(300, y + 20, how, "mono")]
parts += [note(40, 380, f"The last s, {s:.4f}, equals the sum of exp(x − 4) from the 2 passes of Figure 23.1.",
               room=590),
          note(40, 398, "When the maximum grows from m to m', the old sum is multiplied by exp(m − m').",
               room=590)]
save(OUT + "ch23-online.svg", 660, 442,
     f"A trace over the logits 1, 2, 3, 4. Start: m = 1, s = 1. At x = 2: m = 2, s = 1 times exp(-1) plus 1 "
     f"= 1.3679. At x = 3: m = 3, s = 1.5032. At x = 4: m = 4, s = {s:.4f}, the same sum as the 2-pass "
     "method gives.",
     "The online softmax", "1 pass keeps a running maximum m and a sum s rescaled to that maximum.", parts)


# 23.6  Welford's running mean and squared deviations, and the merge of 2 halves.
W = [2.0, 4.0, 4.0, 5.0]
parts = [zone(16, 96, 300, 330, "1 value at a time"), zone(332, 96, 312, 330, "merging 2 halves")]
n, mean, q = 0, 0.0, 0.0
parts += [text(40, 132, "x", "name"), text(86, 132, "n", "name"), text(140, 132, "mean", "name"),
          text(236, 132, "q", "name")]
for k, v in enumerate(W):
    n += 1
    delta = v - mean
    mean += delta / n
    q += delta * (v - mean)
    y = 146 + k * 42
    parts += [text(44, y + 20, f"{v:.0f}", "mono"), text(90, y + 20, str(n), "mono"),
              cell(124, y, 84, 30, "client", f"{mean:.4f}"), cell(220, y, 76, 30, "store", f"{q:.4f}")]
parts += [note(40, 336, "delta = x − mean (old)", room=260), note(40, 354, "mean += delta ÷ n", room=260),
          note(40, 372, "q += delta × (x − mean (new))", room=260),
          note(40, 400, f"variance = q ÷ n = {q:.4f} ÷ 4 = {q / 4:.4f}", room=260)]
a, b = (2, 3.0, 2.0), (2, 4.5, 0.5)
d = b[1] - a[1]
merged = (4, a[1] + d * b[0] / 4, a[2] + b[2] + d * d * a[0] * b[0] / 4)
parts += [card(352, 128, 132, 88, "client", "2, 4", ["n = 2", "mean = 3", "q = 2"]),
          card(496, 128, 132, 88, "outside", "4, 5", ["n = 2", "mean = 4.5", "q = 0.5"]),
          line(418, 216, 418, 236, WIRE, 1.6), line(562, 216, 562, 236, WIRE, 1.6),
          line(418, 236, 562, 236, WIRE, 1.6),
          f'  <circle cx="490" cy="236" r="3" fill="{WIRE}"/>',
          arrow(490, 236, 268),
          card(352, 268, 276, 76, "good", "2, 4, 4, 5", [f"n = 4, mean = {merged[1]:.2f}",
                                                           f"q = {merged[2]:.2f}"]),
          note(356, 380, "delta = 4.5 − 3 = 1.5", room=264),
          note(356, 398, "q = 2 + 0.5 + 1.5² × 2 × 2 ÷ 4 = 4.75", room=264)]
save(OUT + "ch23-welford.svg", 660, 442,
     "Left: Welford's method on 2, 4, 4 and 5. The mean goes 2, 3, 3.3333, 3.75 and the sum of squared "
     "deviations q goes 0, 2, 2.6667, 4.75, so the variance is 4.75 / 4 = 1.1875. Right: the halves 2, 4 "
     "(mean 3, q 2) and 4, 5 (mean 4.5, q 0.5) merge to mean 3.75 and q = 2 + 0.5 + 1.5 squared times 2 "
     "times 2 / 4 = 4.75.",
     "Welford's method", "A running mean and sum of squared deviations, and how 2 of them merge.", parts)


# 23.7  Cancellation: the variance as mean of squares minus squared mean, near 1,000.
parts = [zone(16, 96, 628, 330, "variance of values near 1,000, in float")]
rows = [("mean of x²", "1,000,000.33", "0.0625", "client"),
        ("(mean of x)²", "1,000,000.00", "0.0625", "client"),
        ("sum of x², 4,096 values", "4,096,001,365", "256", "store"),
        ("true variance", "0.33", "", "good")]
parts += [text(40, 132, "quantity", "name"), text(330, 132, "value", "name"),
          text(500, 132, "float spacing", "name")]
for k, (name, value, gap, kind) in enumerate(rows):
    y = 146 + k * 44
    parts += [text(40, y + 20, name, "note"), cell(300, y, 170, 30, kind, value)]
    if gap:
        parts.append(cell(500, y, 110, 30, "danger", gap))
parts += [note(40, 340, "Each float near 1,000,000 is exact only to 0.0625, and the sum of 4,096 squares",
               room=590),
          note(40, 358, "only to 256, which is 0.0625 per element after dividing by 4,096. The variance,",
               room=590),
          note(40, 376, "0.33, is the difference of 2 numbers that are each uncertain by several times 0.0625.",
               room=590),
          note(40, 394, "Welford's method never forms x²: it accumulates squared deviations of about 0.33.",
               room=590)]
save(OUT + "ch23-cancel.svg", 660, 442,
     "A table. The mean of the squares of values near 1,000 is about 1,000,000.33 and the square of the mean "
     "1,000,000.00; floats near 1,000,000 are spaced 0.0625 apart. The sum of 4,096 squares, about "
     "4,096,001,365, is held to the nearest 256. The true variance, 0.33, is a small difference between 2 "
     "large uncertain numbers. Welford's method accumulates squared deviations of about 0.33 instead.",
     "Cancellation", "Subtracting 2 nearly equal large numbers leaves mostly rounding error.", parts)


# 23.8  Measured times of the kernels, from the run quoted in the chapter.
MEASURED = [("softmax, 4,096 columns", [("3 passes", 2.028, 16), ("online", 2.245, 12),
                                        ("fused", 1.446, 8)]),
            ("softmax, 65,536 columns", [("3 passes", 3.984, 16), ("online", 3.009, 12)]),
            ("layer norm, 4,096 columns", [("sum of x²", 2.288, 12), ("Welford", 2.168, 12),
                                           ("fused", 1.794, 8)])]
ELEMENTS = 4096 * 4096
parts = []
y = 104
scale = 270 / 4.0
for group, kernels in MEASURED:
    parts.append(text(40, y + 12, group, "name"))
    y += 24
    for name, ms, bytes_ in kernels:
        floor = ELEMENTS * 8 / 102e9 * 1e3
        kind = "danger" if name == "sum of x²" else ("good" if name == "fused" else "client")
        parts += [text(40, y + 17, name, "note"),
                  bar(150, y, ms * scale, 24, kind),
                  mono(158 + ms * scale, y + 17, f"{ms:.3f} ms   {bytes_} B per element")]
        y += 32
    y += 10
floor = ELEMENTS * 8 / 102e9 * 1e3
x = 150 + floor * scale
parts += [line(x, 120, x, y - 8, KINDS["store"][0], 1.4, "5 4"),
          note(x + 6, y + 6, f"{floor:.2f} ms: 8 B per element at 102 GB/s", room=300),
          note(40, y + 30, "The sum of x² gives errors up to 2.97; Welford and fused stay below 1.9e-4.",
               room=590)]
save(OUT + "ch23-measured.svg", 660, y + 52,
     "Horizontal bars of time per matrix of 4,096 by 4,096 floats. Softmax with rows of 4,096: 3 passes "
     "2.028 ms, online 2.245 ms, fused 1.446 ms. With rows of 65,536: 3 passes 3.984 ms, online 3.009 ms. "
     "Layer norm: sum of squares 2.288 ms but wrong, Welford 2.168 ms, fused 1.794 ms. A dashed line marks "
     "1.32 ms, the time to read and write every element once at 102 GB/s.",
     "Measured times", "1 matrix of 4,096 × 4,096 floats, 64 MiB, per kernel.", parts)
