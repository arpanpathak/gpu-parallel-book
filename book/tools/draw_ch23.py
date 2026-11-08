"""The figures of chapter 23, drawn with figkit and figparts.

    python3 tools/draw_ch23.py

Every figure works on the chapter's 2 x 4 example X = [1 2 3 4; 2 4 4 5]. The numbers in the
hand-worked figures are computed here, in float64 or in float32 where the figure is about float32
rounding, so a figure cannot disagree with its own arithmetic. The measured times come from
code/oxide/ch23_softmax_layernorm and code/cuda/ch23_softmax_layernorm.
"""
import math

import numpy as np

from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, bar, card, line, mono, note, save, tag,
                      text, wire, zone)

OUT = "src/figures/"
X = [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 4.0, 5.0]]
ROW_KIND = ["client", "service"]


def cell(x, y, w, h, kind, label, on=True, bold=False):
    """A value in a rounded box: tinted with an accent edge when `on`, white when not."""
    accent, tint = KINDS[kind]
    edge = accent if on else EDGE
    return "\n".join([
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
        f'fill="{tint if on else "#FFFFFF"}" stroke="{edge}" stroke-width="{2.2 if bold else 1.2}"/>',
        text(x + w / 2, y + h / 2 + 4, label, "mono", "middle", accent if on else MUTED)])


def grid(x, y, rows, kinds, w=58, h=28, gap=6, fmt="{:.4f}"):
    """A small matrix, 1 colour per row."""
    parts = []
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            label = v if isinstance(v, str) else fmt.format(v)
            parts.append(cell(x + c * (w + gap), y + r * (h + gap), w, h, kinds[r], label))
    return parts


def softmax(row):
    top = max(row)
    e = [math.exp(v - top) for v in row]
    return [v / sum(e) for v in e]


def layernorm(row, eps=1e-5):
    mean = sum(row) / len(row)
    var = sum((v - mean) ** 2 for v in row) / len(row)
    return [(v - mean) / math.sqrt(var + eps) for v in row]


P = [softmax(r) for r in X]
Y = [layernorm(r) for r in X]


# 23.1  The example: X, its softmax P and its layer norm Y, 1 colour per row.
parts = [zone(16, 96, 628, 276, "the 2 × 4 example")]
parts += [text(36, 140, "X", "name"), text(36, 230, "P = softmax(X)", "name"),
          text(340, 230, "Y = layer norm(X)", "name")]
parts += grid(120, 120, X, ROW_KIND, w=50, gap=6, fmt="{:.0f}")
parts += grid(36, 244, P, ROW_KIND, w=64, gap=6)
parts += grid(340, 244, Y, ROW_KIND, w=64, gap=6, fmt="{:+.4f}")
parts += [note(400, 136, "row 0", room=80, fill=KINDS["client"][0]),
          note(400, 170, "row 1", room=80, fill=KINDS["service"][0]),
          note(36, 336, "Row 0 of P and Y depends only on row 0 of X, and row 1 only on row 1.", room=590),
          note(36, 356, "Each row of P adds up to 1; each row of Y has mean 0 and variance 1 (up to ε).",
               room=590)]
save(OUT + "ch23-example.svg", 660, 388,
     "The 2 by 4 matrix X with rows 1, 2, 3, 4 and 2, 4, 4, 5. Its softmax P has rows 0.0321, 0.0871, "
     "0.2369, 0.6439 and 0.0279, 0.2060, 0.2060, 0.5601. Its layer normalization Y has rows -1.3416, "
     "-0.4472, 0.4472, 1.3416 and -1.6059, 0.2294, 0.2294, 1.1471.",
     "The worked example", "2 rows of 4 values, and the 2 results computed from them row by row.", parts)


# 23.2  Softmax of row 0 by hand.
row = X[0]
e = [math.exp(v) for v in row]
total = sum(e)
parts = [zone(16, 96, 628, 268, "softmax of row 0")]
heads = [("x", 80), ("exp(x)", 210), ("p = exp(x) ÷ sum", 380)]
parts += [text(cx, 130, h, "name", "middle") for h, cx in heads]
for k in range(4):
    y = 142 + k * 36
    hot = k == 3
    parts += [cell(50, y, 60, 28, "client", f"{row[k]:.0f}", bold=hot),
              cell(160, y, 100, 28, "store", f"{e[k]:.3f}", bold=hot),
              cell(320, y, 120, 28, "good", f"{e[k] / total:.4f}", bold=hot)]
parts += [mono(210, 306, f"sum {total:.3f}", "middle", KINDS["store"][0]),
          mono(380, 306, f"sum {sum(e) / total:.4f}", "middle", KINDS["good"][0]),
          note(470, 156, "column 3:", room=160),
          mono(470, 176, f"{e[3]:.3f} ÷ {total:.3f}"),
          mono(470, 196, f"= {e[3] / total:.4f}"),
          note(36, 342, "The largest value, 4, takes 64% of the total: exp(x) magnifies differences.",
               room=590)]
save(OUT + "ch23-softmax.svg", 660, 380,
     f"Row 0 holds 1, 2, 3, 4. Their exponentials are 2.718, 7.389, 20.086 and 54.598, which sum to "
     f"{total:.3f}. Dividing each by the sum gives 0.0321, 0.0871, 0.2369 and 0.6439, which add up to 1.",
     "Softmax by hand", "Each output is exp(x) divided by the sum of the exponentials of the row.", parts)


# 23.3  Overflow: X + 100 in float32, and the same row with its maximum subtracted.
big = [v + 100 for v in X[0]]
top = max(big)
shifted = [math.exp(v - top) for v in big]
parts = [zone(16, 96, 306, 300, "exp(x)"), zone(338, 96, 306, 300, "exp(x − 104)")]
parts += [text(66, 130, "x", "mono", "middle"), text(158, 130, "exact", "mono", "middle"),
          text(261, 130, "float", "mono", "middle"), text(388, 130, "x − m", "mono", "middle"),
          text(480, 130, "exp", "mono", "middle"), text(583, 130, "p", "mono", "middle")]
for k, v in enumerate(big):
    y = 142 + k * 36
    parts += [cell(36, y, 60, 28, "client", f"{v:.0f}"),
              cell(112, y, 92, 28, "danger", f"{math.exp(v):.2e}".replace("e+", "e")),
              cell(220, y, 82, 28, "danger", "inf"),
              cell(358, y, 60, 28, "client", f"{v - top:.0f}"),
              cell(434, y, 92, 28, "outside", f"{shifted[k]:.4f}"),
              cell(542, y, 82, 28, "good", f"{shifted[k] / sum(shifted):.4f}")]
parts += [cell(112, 290, 190, 28, "danger", "inf ÷ inf = NaN"),
          cell(434, 290, 190, 28, "good", f"sum {sum(shifted):.4f}"),
          note(36, 346, "The largest float32 is 3.40e38, exp(88.72).", room=290),
          note(36, 364, "exp(101) = 7.31e43 rounds to infinity.", room=280),
          note(358, 346, "Every x − m is 0 or less, so every", room=270),
          note(358, 364, "exp lies in (0, 1] and cannot overflow.", room=270)]
save(OUT + "ch23-overflow.svg", 660, 412,
     "Left: row 0 of X + 100 holds 101 to 104, whose exponentials 7.31e43 to 1.47e45 exceed the largest "
     "float32, 3.40e38, so each becomes infinity and every output is infinity divided by infinity, NaN. "
     f"Right: after subtracting the maximum 104 the arguments are -3 to 0, the exponentials sum to "
     f"{sum(shifted):.4f}, and the outputs are 0.0321, 0.0871, 0.2369 and 0.6439, the same as for row 0 of X.",
     "Overflow", "Subtracting the row maximum keeps every exponential between 0 and 1.", parts)


# 23.4  The 3 passes over row 1, and what each pass reads.
row = X[1]
m = max(row)
terms = [math.exp(v - m) for v in row]
s = sum(terms)
parts = [zone(16, 96, 628, 300, "3 passes over row 1")]
passes = [("pass 1: maximum", [f"{v:.0f}" for v in row], f"m = {m:.0f}", "outside"),
          ("pass 2: sum", [f"{t:.4f}" for t in terms], f"s = {s:.4f}", "store"),
          ("pass 3: outputs", [f"{t / s:.4f}" for t in terms], "4 writes", "good")]
for k, (name, vals, result, kind) in enumerate(passes):
    y = 132 + k * 74
    parts += [text(36, y + 19, name, "name")]
    for c, v in enumerate(vals):
        parts.append(cell(176 + c * 76, y, 70, 28, kind, v))
    parts += [tag(480, y + 5, "4 reads", "client"), mono(548, y + 19, result)]
    if k < 2:
        parts.append(note(176, y + 50, ["max(2, 4, 4, 5)", "exp(x − 5), added up"][k], room=300))
parts += [note(36, 356, "Pass 2 needs m from pass 1, and pass 3 needs s from pass 2: each pass reads",
               room=590),
          note(36, 374, "the row again. 12 reads and 4 writes of 4 bytes: 64 bytes, 16 bytes per element.",
               room=590)]
save(OUT + "ch23-three-pass.svg", 660, 412,
     f"Row 1 holds 2, 4, 4, 5. Pass 1 reads it and finds the maximum 5. Pass 2 reads it again and adds "
     f"exp(x - 5): 0.0498, 0.3679, 0.3679 and 1, sum {s:.4f}. Pass 3 reads it a third time and writes "
     "0.0279, 0.2060, 0.2060 and 0.5601. 12 reads and 4 writes of 4 bytes, 16 bytes per element.",
     "Softmax in 3 passes", "The maximum, then the sum, then the outputs; each pass reads the row.", parts)


# 23.5  Layer normalization of row 0 by hand.
row = X[0]
mean = sum(row) / 4
dev = [v - mean for v in row]
var = sum(d * d for d in dev) / 4
rstd = 1 / math.sqrt(var + 1e-5)
parts = [zone(16, 96, 628, 288, "layer normalization of row 0")]
heads = [("x", 76), ("x − μ", 186), ("(x − μ)²", 306), ("y", 476)]
parts += [text(cx, 130, h, "name", "middle") for h, cx in heads]
for k in range(4):
    y = 142 + k * 36
    parts += [cell(46, y, 60, 28, "client", f"{row[k]:.0f}"),
              cell(146, y, 80, 28, "outside", f"{dev[k]:+.1f}"),
              cell(266, y, 80, 28, "store", f"{dev[k] ** 2:.2f}"),
              cell(426, y, 100, 28, "good", f"{Y[0][k]:+.4f}")]
parts += [mono(76, 306, f"μ = {mean:.1f}", "middle", KINDS["client"][0]),
          mono(306, 306, f"σ² = {var:.2f}", "middle", KINDS["store"][0]),
          mono(476, 306, f"× {rstd:.4f}", "middle", KINDS["good"][0]),
          note(36, 340, f"μ = 10 ÷ 4 = 2.5;  σ² = 5 ÷ 4 = 1.25;  1 ÷ √(1.25 + 0.00001) = {rstd:.4f}.",
               room=590),
          note(36, 360, f"y = (x − μ) × {rstd:.4f}, with γ = 1 and β = 0.", room=590)]
save(OUT + "ch23-layernorm.svg", 660, 400,
     f"Row 0 holds 1, 2, 3, 4. The mean is 2.5. The deviations are -1.5, -0.5, 0.5, 1.5, their squares "
     f"2.25, 0.25, 0.25, 2.25, and the variance 1.25. Multiplying each deviation by "
     f"1 / sqrt(1.25 + 0.00001) = {rstd:.4f} gives -1.3416, -0.4472, 0.4472 and 1.3416.",
     "Layer normalization by hand", "Subtract the mean, divide by the standard deviation.", parts)


# 23.6  Cancellation: row 0 of X + 10,000, variance as mean of squares minus squared mean, float32.
f = np.float32
far = [f(v + 10000) for v in X[0]]
sq = [f(v * v) for v in far]
running, acc = [], f(0)
for v in sq:
    acc = f(acc + v)
    running.append(acc)
msq = f(acc / f(4))
mu = f(sum(far, f(0)) / f(4))
mu2 = f(mu * mu)
naive = f(msq - mu2)
parts = [zone(16, 96, 628, 352, "row 0 of X + 10,000 in float32")]
parts += [text(80, 130, "x", "name", "middle"), text(206, 130, "x² exact", "name", "middle"),
          text(346, 130, "x² float32", "name", "middle"), text(500, 130, "running sum", "name", "middle")]
for k in range(4):
    y = 142 + k * 36
    exact = int(far[k]) ** 2
    parts += [cell(44, y, 72, 28, "client", f"{int(far[k]):,}"),
              cell(146, y, 120, 28, "idle", f"{exact:,}"),
              cell(286, y, 120, 28, "danger" if int(sq[k]) != exact else "store", f"{int(sq[k]):,}"),
              cell(436, y, 130, 28, "store", f"{int(running[k]):,}")]
lines = [("mean of squares", f"{int(acc):,} ÷ 4", f"{int(msq):,}", "store"),
         ("square of mean", f"{float(mu):,.1f}² = 100,050,006.25", f"{int(mu2):,}", "danger"),
         ("variance", "difference", f"{float(naive):.0f}  (exact 1.25)", "danger")]
for k, (name, how, value, kind) in enumerate(lines):
    y = 300 + k * 36
    parts += [text(44, y + 19, name, "note"), mono(176, y + 19, how),
              cell(436, y, 190, 28, kind, value)]
parts += [note(44, 430, "Near 10^8 float32 values lie 8 apart; near 4 × 10^8, 32 apart.", room=590)]
save(OUT + "ch23-cancel.svg", 660, 464,
     f"Row 0 of X + 10,000 holds 10,001 to 10,004. Their squares round in float32 to 100,020,000, "
     f"100,040,000, 100,060,008 and 100,080,016, and the running sum ends at {int(acc):,}. The mean of the "
     f"squares is {int(msq):,}. The mean is 10,002.5, whose square 100,050,006.25 rounds to {int(mu2):,}. "
     f"The difference is {float(naive):.0f}, a negative variance, where the exact variance is 1.25.",
     "Cancellation", "2 nearly equal large numbers, each rounded, leave a wrong difference.", parts)


# 23.7  The online softmax on row 1: the running maximum m and the rescaled sum s.
row = X[1]
trace = []
m, s = -math.inf, 0.0
for v in row:
    if v > m:
        how = "new maximum: s × exp(m − x) + 1" if m > -math.inf else "first value: s = 1"
        s = s * math.exp(m - v) + 1.0 if m > -math.inf else 1.0
        m = v
    else:
        how = "x ≤ m: s + exp(x − m)"
        s = s + math.exp(v - m)
    trace.append((v, m, s, how))
parts = [zone(16, 96, 628, 300, "online maximum and sum, row 1")]
parts += [text(36, 130, "x", "name"), text(110, 130, "m", "name"), text(184, 130, "s", "name"),
          text(290, 130, "step", "name")]
parts += [mono(36, 160, "–"), cell(84, 142, 60, 28, "outside", "−inf"),
          cell(160, 142, 96, 28, "store", "0"), note(290, 160, "the empty state", room=330)]
for k, (v, mk, sk, how) in enumerate(trace):
    y = 178 + k * 36
    parts += [mono(36, y + 19, f"{v:.0f}"), cell(84, y, 60, 28, "outside", f"{mk:.0f}"),
              cell(160, y, 96, 28, "store", f"{sk:.4f}"), note(290, y + 19, how, room=330)]
parts += [note(36, 350, "Step 2: 1 × exp(2 − 4) + 1 = 1.1353.  Step 3: 1.1353 + exp(0) = 2.1353.", room=590),
          note(36, 370, f"Step 4: 2.1353 × exp(4 − 5) + 1 = {s:.4f}, the sum of pass 2 in Figure 23.5.",
               room=590)]
save(OUT + "ch23-online.svg", 660, 412,
     f"The online softmax over row 1, 2, 4, 4, 5. It starts from m = -inf and s = 0. After 2: m = 2, "
     f"s = 1. After 4: m = 4, s = 1 times exp(-2) plus 1 = 1.1353. After the second 4: m stays 4 and "
     f"s = 2.1353. After 5: m = 5, s = 2.1353 times exp(-1) plus 1 = {s:.4f}, the same sum as 3 passes give.",
     "The online softmax", "1 pass keeps a running maximum m and a sum s taken relative to m.", parts)


# 23.8  The invariant of the online softmax: s_k sums the first k terms relative to m_k.
parts = [zone(16, 96, 628, 286, "the invariant, row 1")]
for c, v in enumerate(row):
    parts.append(cell(176 + c * 64, 120, 56, 28, "service", f"{v:.0f}"))
parts.append(text(36, 139, "row 1", "note"))
for k in range(5):
    y = 170 + k * 34
    mk = max(row[:k]) if k else -math.inf
    sk = sum(math.exp(v - mk) for v in row[:k]) if k else 0.0
    parts.append(text(36, y + 18, f"after {k} values", "note"))
    if k:
        parts.append(bar(176, y + 2, k * 64 - 8, 22, "service"))
    else:
        parts.append(line(176, y + 2, 176, y + 24, KINDS["service"][0], 2))
    terms = " + ".join(f"e^({v:.0f}−{mk:.0f})" for v in row[:k]) if k else "empty sum"
    parts.append(mono(440, y + 18, f"m = {mk:.0f}, s = {sk:.4f}" if k else "m = −inf, s = 0"))
    if k == 4:
        parts.append(note(36, y + 54, f"s = {terms} = {sk:.4f}", room=590))
save(OUT + "ch23-invariant.svg", 660, 398,
     "Row 1, 2, 4, 4, 5, with a bar over the first k values for k = 0 to 4. After k values, m is the "
     "maximum of those k values and s is the sum of exp(x - m) over them: m = 2, s = 1; m = 4, s = 1.1353; "
     "m = 4, s = 2.1353; m = 5, s = 1.7855.",
     "The loop invariant", "After k values, s is the sum of exp(x − m) over exactly those k values.", parts)


# 23.9  Merging 2 partial states: softmax halves of row 1, Welford halves of row 0.
sa = (4.0, math.exp(2 - 4) + 1)
sb = (5.0, math.exp(4 - 5) + 1)
smerged = (5.0, sa[1] * math.exp(sa[0] - 5) + sb[1])
parts = [zone(16, 96, 306, 320, "softmax states, row 1"), zone(338, 96, 306, 320, "Welford states, row 0")]
parts += [card(32, 128, 132, 88, "service", "2, 4", ["m = 4", f"s = {sa[1]:.4f}"]),
          card(174, 128, 132, 88, "service", "4, 5", ["m = 5", f"s = {sb[1]:.4f}"]),
          line(98, 216, 98, 236, WIRE, 1.6), line(240, 216, 240, 236, WIRE, 1.6),
          line(98, 236, 240, 236, WIRE, 1.6), f'  <circle cx="169" cy="236" r="3" fill="{WIRE}"/>',
          arrow(169, 236, 268),
          card(32, 268, 274, 72, "good", "2, 4, 4, 5", ["m = max(4, 5) = 5", f"s = {smerged[1]:.4f}"]),
          note(36, 366, f"s = {sa[1]:.4f} × exp(4 − 5) + {sb[1]:.4f} × exp(0)", room=280),
          note(36, 386, f"  = {smerged[1]:.4f}", room=270)]
wa, wb = (2, 1.5, 0.5), (2, 3.5, 0.5)
d = wb[1] - wa[1]
wm = (4, wa[1] + d * wb[0] / 4, wa[2] + wb[2] + d * d * wa[0] * wb[0] / 4)
parts += [card(354, 128, 132, 88, "client", "1, 2", ["n = 2, μ = 1.5", "q = 0.5"]),
          card(496, 128, 132, 88, "client", "3, 4", ["n = 2, μ = 3.5", "q = 0.5"]),
          line(420, 216, 420, 236, WIRE, 1.6), line(562, 216, 562, 236, WIRE, 1.6),
          line(420, 236, 562, 236, WIRE, 1.6), f'  <circle cx="491" cy="236" r="3" fill="{WIRE}"/>',
          arrow(491, 236, 268),
          card(354, 268, 274, 72, "good", "1, 2, 3, 4", [f"n = 4, μ = {wm[1]:.1f}", f"q = {wm[2]:.0f}"]),
          note(358, 366, "δ = 3.5 − 1.5 = 2,  μ = 1.5 + 2 × 2 ÷ 4", room=270),
          note(358, 386, "q = 0.5 + 0.5 + 2² × 2 × 2 ÷ 4 = 5", room=270)]
save(OUT + "ch23-merge.svg", 660, 432,
     f"Left: the softmax states of 2, 4 (m = 4, s = {sa[1]:.4f}) and of 4, 5 (m = 5, s = {sb[1]:.4f}) merge "
     f"into m = 5 and s = {sa[1]:.4f} times exp(-1) plus {sb[1]:.4f} = {smerged[1]:.4f}. Right: the Welford "
     "states of 1, 2 (n = 2, mean 1.5, q = 0.5) and of 3, 4 (n = 2, mean 3.5, q = 0.5) merge into n = 4, "
     "mean 2.5 and q = 0.5 + 0.5 + 4 times 2 times 2 / 4 = 5.",
     "Merging 2 partial states", "Each half is summarized by a few numbers, and 2 summaries combine.", parts)


# 23.10  Welford's update on row 0 (and on row 0 + 10,000 in float32: the same q).
parts = [zone(16, 96, 628, 300, "Welford's update, row 0")]
parts += [text(36, 130, "x", "name"), text(86, 130, "n", "name"), text(130, 130, "δ = x − μ(old)", "name"),
          text(290, 130, "μ", "name"), text(400, 130, "q", "name"), text(500, 130, "q added", "name")]
n, mu_k, q = 0, 0.0, 0.0
farq = []
nf, muf, qf = f(0), f(0), f(0)
for k, v in enumerate(X[0]):
    n += 1
    delta = v - mu_k
    mu_k += delta / n
    add = delta * (v - mu_k)
    q += add
    vf = f(v + 10000)
    nf = f(nf + f(1))
    df = f(vf - muf)
    muf = f(muf + f(df / nf))
    qf = f(qf + f(df * f(vf - muf)))
    farq.append(float(qf))
    y = 142 + k * 36
    parts += [mono(36, y + 19, f"{v:.0f}"), mono(86, y + 19, str(n)),
              cell(150, y, 80, 28, "outside", f"{delta:.1f}"),
              cell(262, y, 80, 28, "client", f"{mu_k:.1f}"),
              cell(372, y, 80, 28, "store", f"{q:.1f}"),
              mono(500, y + 19, f"{delta:.1f} × {v - mu_k:.1f} = {add:.1f}")]
parts += [note(36, 314, "μ(new) = μ(old) + δ ÷ n,   q(new) = q(old) + δ × (x − μ(new))", room=590),
          note(36, 334, f"variance = q ÷ n = {q:.0f} ÷ 4 = {q / 4:.2f}", room=590),
          note(36, 354, f"On 10,001 to 10,004 in float32 the same steps give q = {farq[-1]:.0f} exactly.",
               room=590)]
save(OUT + "ch23-welford.svg", 660, 412,
     f"Welford's update on 1, 2, 3, 4. The deviations from the old mean are 1, 1, 1.5 and 2. The mean goes "
     f"1, 1.5, 2, 2.5 and q goes 0, 0.5, 2, 5, so the variance is 5 / 4 = 1.25. On 10,001 to 10,004 in "
     f"float32 the same steps give q = {farq[-1]:.0f}.",
     "Welford's method", "A running mean and a running sum of squared deviations.", parts)


# 23.11  1 block per row: the columns of 1 thread, and the example's 2 blocks.
parts = [zone(16, 96, 628, 300, "1 block of 256 threads per row of 4,096")]
seg = 36
for k in range(16):
    x = 40 + k * seg
    parts.append(cell(x, 132, seg - 4, 26, "client" if k % 2 == 0 else "idle", str(k)))
parts += [note(40, 178, "The row as 16 groups of 256 columns; group k holds columns 256k to 256k + 255.",
               room=590)]
for k in range(16):
    x = 40 + k * seg + (seg - 4) / 2
    parts.append(f'  <circle cx="{x:.1f}" cy="196" r="4" fill="{KINDS["store"][0]}"/>')
parts += [note(40, 222, "Thread 5 reads column 5 of every group: 5, 261, 517, ..., 3,845: 16 values.",
               room=590),
          text(40, 262, "The example, block 1 (row 1):", "name")]
for t in range(8):
    x = 40 + t * 72
    label = f"{X[1][t]:.0f}" if t < 4 else "none"
    parts += [cell(x, 276, 64, 28, "service" if t < 4 else "idle", label, on=t < 4),
              note(x + 32, 322, f"thread {t}", "middle")]
parts += [note(40, 350, "Threads 0 to 3 hold 1 value each; threads 4 to 255 hold none and start from",
               room=590),
          note(40, 368, "the empty state, which every merge must leave unchanged.", room=590)]
save(OUT + "ch23-row-block.svg", 660, 412,
     "A row of 4,096 columns drawn as 16 groups of 256. Thread 5 reads column 5 of every group, 16 values in "
     "all. Below, block 1 of the example: threads 0 to 3 hold 2, 4, 4 and 5, and threads 4 to 255 hold no "
     "value.",
     "1 block per row", "Each thread reads every 256th column of its block's row, from threadIdx.x on.", parts)


# 23.12  A butterfly maximum on 8 lanes (a real warp has 32 lanes and 5 steps).
vals = [2.0, 4.0, 4.0, 5.0] + [-math.inf] * 4
steps = [("start", vals)]
for mask in (4, 2, 1):
    vals = [max(vals[i], vals[i ^ mask]) for i in range(8)]
    steps.append((f"mask {mask}", vals))
parts = [zone(16, 96, 628, 330, "maximum over 8 lanes by xor shuffles")]
for i in range(8):
    parts.append(text(150 + i * 60 + 26, 128, f"lane {i}", "note", "middle"))
for r, (name, vs) in enumerate(steps):
    y = 140 + r * 64
    parts.append(text(36, y + 19, name, "name"))
    for i, v in enumerate(vs):
        label = "−inf" if v == -math.inf else f"{v:.0f}"
        parts.append(cell(150 + i * 60, y, 52, 28, "outside" if v > -math.inf else "idle", label,
                          on=v > -math.inf))
    if r < 3:
        mask = (4, 2, 1)[r]
        for i in range(8):
            j = i ^ mask
            if i < j:
                x1, x2 = 150 + i * 60 + 26, 150 + j * 60 + 26
                ym = y + 34 + (i % mask) * 7
                parts.append(f'  <path d="M{x1},{y + 30} V{ym} H{x2} V{y + 30}" fill="none" '
                             f'stroke="{KINDS["outside"][0]}" stroke-width="1.2" opacity="0.6"/>')
parts += [note(36, 404, "In each step lane i takes the larger of its value and the value of lane i xor mask.",
               room=590)]
save(OUT + "ch23-butterfly.svg", 660, 440,
     "8 lanes hold 2, 4, 4, 5 and 4 times minus infinity. With mask 4 each lane takes the maximum of itself "
     "and the lane 4 away: 2, 4, 4, 5, 2, 4, 4, 5. With mask 2: 4, 5, 4, 5, 4, 5, 4, 5. With mask 1 every "
     "lane holds 5.",
     "The butterfly", "After log2 of the lane count steps, every lane holds the maximum.", parts)


# 23.13  The block reduction: 8 warps, 8 slots, 2 barriers.
parts = [zone(16, 96, 628, 286, "block-wide sum, 8 warps")]
for w in range(8):
    x = 36 + w * 74
    parts += [cell(x, 128, 66, 28, "service", f"warp {w}"),
              arrow(x + 33, 158, 186),
              cell(x, 186, 66, 28, "store", f"slot {w}")]
parts += [line(36, 232, 618, 232, INK, 2.4), text(624, 236, "", "note"),
          note(36, 252, "barrier 1: all 8 slots are written", room=300),
          cell(36, 268, 582, 28, "good", "every thread adds slot 0 + slot 1 + ... + slot 7"),
          line(36, 316, 618, 316, INK, 2.4),
          note(36, 336, "barrier 2: all 8 slots are read; only now may the slots be reused", room=590),
          note(36, 362, "Inside a warp: 5 xor shuffles. Across warps: shared memory and 2 barriers.",
               room=590)]
save(OUT + "ch23-block-reduce.svg", 660, 398,
     "8 warps each reduce their 32 values by shuffles, and lane 0 of each writes the result into its slot "
     "in shared memory. After barrier 1 every thread adds the 8 slots. Barrier 2 waits until every thread "
     "has read them, so that a later reduction may write the slots again.",
     "The block reduction", "Shuffles within each warp, then 8 slots in shared memory.", parts)


# 23.14  The missing second barrier: a race in layer norm by the sum of squares, row 94.
sums = [512003.3, 511992.0, 512013.7, 511992.8, 511977.7, 511982.8, 512000.4, 511982.9]
squares5 = 511965734
wrong = sum(sums) - sums[5] + squares5
parts = [zone(16, 96, 628, 300, "row 94, without barrier 2")]
parts += [text(36, 140, "warp 5", "name"), text(36, 214, "warp 0", "name")]
parts += [bar(110, 124, 90, 26, "service", "write slot 5"),
          bar(206, 124, 70, 26, "idle", "barrier 1"),
          bar(282, 124, 110, 26, "service", "read 8 sums"),
          bar(398, 124, 100, 26, "store", "5 shuffles"),
          bar(504, 124, 120, 26, "danger", "slot 5 = 5.1e8"),
          bar(110, 198, 90, 26, "client", "write slot 0"),
          bar(206, 198, 70, 26, "idle", "barrier 1"),
          bar(282, 198, 330, 26, "client", "read slots 0 to 7 (late)"),
          line(580, 152, 580, 196, KINDS["danger"][0], 2),
          f'  <circle cx="580" cy="198" r="4" fill="{KINDS["danger"][0]}"/>',
          note(36, 262, "Warp 0 reads slot 5 after warp 5 has overwritten its sum, 511,982.8, with its",
               room=590),
          note(36, 280, f"sum of squares, {squares5:,}. Warp 0 adds {wrong:,.0f} and takes the mean",
               room=590),
          note(36, 298, f"{wrong / 4096:,.1f} instead of 999.99. Its variance, 999,973.7 − "
                        f"{wrong / 4096:,.0f}², is negative,", room=590),
          note(36, 316, "and the square root makes all 512 outputs of warp 0 NaN.", room=590),
          note(36, 352, "Measured: 11 of 100 launches of this kernel produced NaN rows.", room=590)]
save(OUT + "ch23-race.svg", 660, 412,
     "A timeline for row 94 without the second barrier. Warp 5 finishes reading the 8 sums early, shuffles "
     "its sum of squares and writes its sum of squares, 511,965,734, into slot 5. Warp 0 is still reading the "
     f"slots, reads the new value and computes the mean {wrong / 4096:,.1f} instead of 999.99, a negative "
     "variance, and NaN outputs.",
     "A race on the slots", "Without barrier 2, the next reduction can overwrite a slot still being read.",
     parts)


# 23.15  Global memory traffic per element for the kernels.
parts = [zone(16, 96, 628, 300, "global memory traffic per element")]
rows = [("3 passes", [("read", "max"), ("read", "sum"), ("read", "p"), ("write", "p")], 16),
        ("online", [("read", "max + sum"), ("read", "p"), ("write", "p")], 12),
        ("fused", [("read", "registers"), ("write", "p")], 8)]
for r, (name, steps_, bytes_) in enumerate(rows):
    y = 140 + r * 80
    parts += [text(40, y + 20, name, "name")]
    x = 140
    for kind, purpose in steps_:
        parts.append(bar(x, y, 92, 30, "danger" if kind == "write" else "client", f"{kind} 4 B"))
        parts.append(note(x + 46, y + 48, purpose, "middle"))
        x += 100
    parts.append(tag(600, y + 6, f"{bytes_} B", "store", anchor="end"))
parts += [note(40, 372, "Layer norm: sum of squares and Welford read twice (12 B); fused reads once (8 B).",
               room=590)]
save(OUT + "ch23-passes.svg", 660, 412,
     "3 rows of bars. The 3-pass kernel reads each element 3 times, for the maximum, the sum and the output, "
     "and writes it once: 16 bytes. The online kernel reads it twice and writes once: 12 bytes. The fused "
     "kernel reads once and writes once: 8 bytes, because its values stay in registers.",
     "Bytes per element", "Every pass over the row reads each element from memory again.", parts)


# 23.16  Measured times on the Orin NX (Rust and C++), from the runs quoted in the chapter.
MEASURED = [("softmax, rows of 4,096", [("3 passes", 2.025, 1.434, 16), ("online", 2.255, 1.480, 12),
                                        ("fused", 1.435, 1.417, 8)]),
            ("softmax, rows of 65,536", [("3 passes", 3.985, 3.129, 16), ("online", 3.015, 2.373, 12)]),
            ("layer norm, rows of 4,096", [("sum of x²", 1.808, 1.464, 12), ("Welford", 2.112, 1.920, 12),
                                           ("fused", 1.424, 1.445, 8)])]
ELEMENTS = 4096 * 4096
parts = []
y = 116
scale = 240 / 4.0
for group, kernels in MEASURED:
    parts.append(text(40, y + 12, group, "name"))
    y += 22
    for name, rust, cpp, bytes_ in kernels:
        floor = ELEMENTS * bytes_ / 102e9 * 1e3
        parts += [text(40, y + 14, f"{name}, {bytes_} B", "note"),
                  bar(170, y, rust * scale, 18, "client"),
                  mono(430, y + 14, f"Rust {rust:.3f} ms"),
                  bar(170, y + 20, cpp * scale, 18, "outside"),
                  mono(430, y + 34, f"C++  {cpp:.3f} ms"),
                  line(170 + floor * scale, y - 2, 170 + floor * scale, y + 40, KINDS["store"][0], 1.4,
                       "4 3")]
        y += 50
    y += 6
parts += [note(40, y + 8, "Dashed: bytes per element × 16,777,216 elements ÷ 102 GB/s, the time if every",
               room=590),
          note(40, y + 26, "modeled byte went to memory at peak bandwidth.", room=590)]
save(OUT + "ch23-measured.svg", 660, y + 48,
     "Paired bars of Rust and C++ times per 4,096 by 4,096 matrix on a Jetson Orin NX. Softmax rows of 4,096: "
     "3 passes 2.025 and 1.434 ms, online 2.255 and 1.480 ms, fused 1.435 and 1.417 ms. Rows of 65,536: "
     "3 passes 3.985 and 3.129 ms, online 3.015 and 2.373 ms. Layer norm: sum of squares 1.808 and 1.464 ms, "
     "Welford 2.112 and 1.920 ms, fused 1.424 and 1.445 ms. Dashed lines mark the modeled byte count at "
     "102 GB/s.",
     "Measured times", "1 matrix of 4,096 × 4,096 floats, 64 MiB, per kernel, Jetson Orin NX.", parts)
