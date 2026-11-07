"""The figures of chapter 19, drawn with figkit and figparts.

    python3 tools/draw_ch19.py

The ring steps are the ones code/ch19_collectives prints for 4 ranks of 8 floats. The cost
model uses alpha = 10 µs per message and B = 25 GB/s per link, the values worked in the text.
The measured times are the quoted runs of code/ch19_collectives on a Jetson Orin NX.
"""
import math

from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, arrow, card, fan, legend, line, mono,
                      note, save, tag, text, wire, zone)

OUT = "src/figures/"


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def hline(x1, x2, y, colour=WIRE, width=1.8, tip=True):
    """A horizontal arrow from x1 to x2 whose tip stops 3 units short."""
    end = x2 - 3 if x2 > x1 else x2 + 3
    head = ' marker-end="url(#head)"' if tip else ""
    return (f'  <path d="M{x1},{y} H{end}" fill="none" stroke="{colour}" stroke-width="{width}" '
            f'stroke-linecap="round"{head}/>')


# 25.1  The 5 collectives on 4 ranks, before and after.
DATA = [[1, 0, 2, 1], [2, 1, 0, 3], [0, 3, 1, 0], [1, 2, 4, 1]]
SUM = [sum(DATA[r][i] for r in range(4)) for i in range(4)]       # 4, 6, 7, 5
CASES = [
    ("broadcast", "root = rank 0",
     [DATA[0], None, None, None], [DATA[0]] * 4, [True] * 4),
    ("reduce", "root = rank 0, sum",
     DATA, [SUM, DATA[1], DATA[2], DATA[3]], [True, False, False, False]),
    ("all-reduce", "sum, everywhere",
     DATA, [SUM] * 4, [True] * 4),
    ("reduce-scatter", "rank r keeps piece r of the sum",
     DATA, [[SUM[r] if i == r else None for i in range(4)] for r in range(4)], [True] * 4),
    ("all-gather", "every rank gets every piece",
     [[SUM[r] if i == r else None for i in range(4)] for r in range(4)], [SUM] * 4, [True] * 4),
]
cell = 19
parts = [legend(36, 104, [("client", "input"), ("good", "result"), ("idle", "unchanged")])]
for n, (name, sub, before, after, hot) in enumerate(CASES):
    x0 = 16 + (n % 2) * 322
    y0 = 128 + (n // 2) * 162
    parts += [f'  <rect x="{x0}" y="{y0}" width="306" height="148" rx="12" fill="#FFFFFF" '
              f'stroke="{EDGE}" stroke-width="1.2" filter="url(#lift)"/>',
              text(x0 + 16, y0 + 24, name, "name"), text(x0 + 16, y0 + 41, sub, "sub")]
    for r in range(4):
        y = y0 + 54 + r * (cell + 3)
        parts.append(text(x0 + 16, y + 14, f"rank {r}", "mono", fill=MUTED))
        for i in range(4):
            v = before[r][i] if before[r] else None
            parts.append(box(x0 + 70 + i * cell, y, cell - 3, cell - 3, "client",
                             "" if v is None else str(v), on=v is not None))
            w = after[r][i] if after[r] else None
            kind = "good" if hot[r] else "idle"
            parts.append(box(x0 + 206 + i * cell, y, cell - 3, cell - 3, kind,
                             "" if w is None else str(w), on=w is not None))
    parts.append(hline(x0 + 160, x0 + 198, y0 + 98))
parts += [note(338 + 16, 128 + 2 * 162 + 34, "4 ranks hold 4 numbers each.", room=280),
          note(338 + 16, 128 + 2 * 162 + 52, "The sum of the 4 inputs is 4, 6, 7, 5.", room=280),
          note(338 + 16, 128 + 2 * 162 + 70, "All-reduce = reduce-scatter, then", room=280),
          note(338 + 16, 128 + 2 * 162 + 88, "all-gather: 2 rows of this figure.", room=280)]
save(OUT + "ch28-collectives.svg", 660, 610,
     "Five panels, 1 per collective, each showing 4 ranks before and after. Broadcast: only rank 0 holds "
     "1 0 2 1, and afterwards every rank holds it. Reduce: the ranks hold 1 0 2 1, 2 1 0 3, 0 3 1 0 and 1 2 4 "
     "1, and afterwards rank 0 holds their sum 4 6 7 5. All-reduce: every rank holds 4 6 7 5. Reduce-scatter: "
     "rank r keeps element r of the sum, 4, 6, 7 and 5. All-gather: each rank starts with 1 piece and ends "
     "with all 4, 4 6 7 5.",
     "The 5 collectives", "4 ranks, 4 numbers each: what every rank holds before and after.", parts)


# 25.3  1 collector against a ring: the bytes on each GPU's link.
def gpu(x, y, name, kind="client"):
    return card(x, y, 104, 46, kind, name)


parts = [zone(16, 104, 300, 330, "1 collector"), zone(344, 104, 300, 330, "a ring"),
         gpu(108, 140, "rank 0", "store")]
for k, x in enumerate((30, 116, 202)):
    parts += [card(x, 340, 84, 46, "client", f"rank {k + 1}"),
              wire([(x + 42, 340), (x + 42, 270)], label="100 MB", arrow=False)]
parts += [f'  <path d="M72,270 H244" fill="none" stroke="{WIRE}" stroke-width="1.8"/>',
          f'  <circle cx="160" cy="270" r="3" fill="{WIRE}"/>',
          wire([(160, 270), (160, 189)], label="300 MB"),
          note(36, 408, "Rank 0's link carries 300 MB in,", room=270),
          note(36, 424, "then 300 MB back out: 24 ms.", room=270)]
corners = [(374, 160), (530, 160), (530, 300), (374, 300)]
for k, (x, y) in enumerate(corners):
    parts.append(card(x, y, 88, 46, "client", f"rank {k}"))
parts += [wire([(462, 183), (530, 183)]),
          wire([(574, 206), (574, 300)], label="150 MB"),
          wire([(530, 323), (462, 323)]),
          wire([(418, 300), (418, 206)], label="150 MB"),
          mono(496, 176, "150 MB", "middle", INK), mono(496, 344, "150 MB", "middle", INK),
          note(364, 408, "Every link carries 150 MB each", room=270),
          note(364, 424, "way, all at once: 6 ms.", room=270),
          note(16, 466, "All-reduce of m = 100 MB over 4 GPUs, each with 1 link of 25 GB/s in each direction.",
               room=628),
          note(16, 484, "The collector's link moves (k − 1) m each way; a ring link moves 2 (k − 1) m / k.",
               room=628)]
save(OUT + "ch28-star-ring.svg", 660, 506,
     "Two zones. Left, 1 collector: ranks 1, 2 and 3 each send 100 MB up to rank 0, so rank 0's link carries "
     "300 MB in and then 300 MB back out, 24 ms at 25 GB/s. Right, a ring of 4 ranks: each link carries 150 MB "
     "in each direction, all links at once, 6 ms.",
     "1 collector or a ring", "The same all-reduce of 100 MB: the busiest link decides the time.", parts)


# 25.2  The ring for 4 ranks: 3 steps of reduce-scatter, 3 of all-gather (the program's output).
def ring_states(k=4):
    """The 4 chunk values of each rank after each step, as code/ch19_collectives computes them."""
    data = [[r + 1] * k for r in range(k)]
    states, moved = [[row[:] for row in data]], [None]
    for s in range(k - 1):
        sends = [(r, (r - s) % k, data[r][(r - s) % k]) for r in range(k)]
        for r, c, v in sends:
            data[(r + 1) % k][c] += v
        states.append([row[:] for row in data])
        moved.append({((r + 1) % k, c) for r, c, _ in sends})
    for s in range(k - 1):
        sends = [(r, (r + 1 - s) % k, data[r][(r + 1 - s) % k]) for r in range(k)]
        for r, c, v in sends:
            data[(r + 1) % k][c] = v
        states.append([row[:] for row in data])
        moved.append({((r + 1) % k, c) for r, c, _ in sends})
    return states, moved


states, moved = ring_states()
cw = 26
x0 = 150
parts = []
for r in range(4):
    parts.append(text(x0 + r * 124 + 2 * cw - 2, 112, f"rank {r}", "name", "middle"))
    for c in range(4):
        parts.append(mono(x0 + r * 124 + c * cw + 11, 130, f"c{c}", "middle", MUTED))
labels = ["start", "step 1", "step 2", "step 3", "step 4", "step 5", "step 6"]
for s, (state, hot) in enumerate(zip(states, moved)):
    y = 142 + s * 40 + (14 if s >= 4 else 0)
    phase = "all-gather" if s >= 4 else "reduce-scatter" if s else ""
    parts += [text(28, y + 15, labels[s], "name"), text(28, y + 31, phase, "sub")]
    for r in range(4):
        for c in range(4):
            v = state[r][c]
            kind = "good" if v == 10 else "client" if hot and (r, c) in hot else "idle"
            parts.append(box(x0 + r * 124 + c * cw, y, cw - 4, 26, kind, str(v),
                             on=kind != "idle" or s == 0))
parts += [f'  <path d="M24,{142 + 4 * 40 - 4} H636" fill="none" stroke="{EDGE}" stroke-width="1.2" '
          f'stroke-dasharray="4 4"/>',
          legend(150, 446, [("client", "received this step"), ("good", "complete sum, 10")]),
          note(24, 476, "Rank r starts with r + 1 in all 4 chunks. In step s of reduce-scatter, rank r sends chunk", room=612),
          note(24, 494, "(r − s) mod 4 to rank r + 1, which adds it. After 3 steps each rank holds 1 complete", room=612),
          note(24, 512, "chunk; 3 steps of all-gather pass the complete chunks on, overwriting.", room=612)]
save(OUT + "ch28-ring-steps.svg", 660, 534,
     "A table of 7 rows, start and steps 1 to 6, and 4 columns, ranks 0 to 3, each with 4 chunks c0 to c3. "
     "At the start rank r holds r + 1 in every chunk. In step 1 rank 1's c0 becomes 3, rank 2's c1 5, rank 3's "
     "c2 7 and rank 0's c3 5. After step 3 rank 0 holds 10 in c1, rank 1 in c2, rank 2 in c3 and rank 3 in c0. "
     "Steps 4 to 6 copy each 10 around the ring until every chunk of every rank is 10.",
     "Ring all-reduce, step by step", "4 ranks, 4 chunks: the values the program prints after each step.",
     parts)


# 25.4  The binomial tree for 8 ranks: 3 steps up, 3 steps down.
cols = [64 + r * 72 for r in range(8)]
parts = []
for r, x in enumerate(cols):
    parts.append(box(x - 26, 108, 52, 24, "client", f"rank {r}", cls="pill"))
up = [[(1, 0), (3, 2), (5, 4), (7, 6)], [(2, 0), (6, 4)], [(4, 0)]]
down = [[(0, 4)], [(0, 2), (4, 6)], [(0, 1), (2, 3), (4, 5), (6, 7)]]
for s, pairs in enumerate(up + down):
    y = 160 + s * 44 + (18 if s >= 3 else 0)
    add = s < 3
    parts.append(text(18, y + 4, f"step {s + 1}", "sub"))
    for x in cols:
        parts.append(line(x, y - 14, x, y + 14, EDGE, 1, "3 3"))
    for src, dst in pairs:
        kind = "client" if add else "good"
        parts += [f'  <circle cx="{cols[src]}" cy="{y}" r="5" fill="{KINDS[kind][0]}"/>',
                  hline(cols[src], cols[dst] + (8 if dst < src else -8), y)]
        parts.append(f'  <circle cx="{cols[dst]}" cy="{y}" r="7" fill="#FFFFFF" '
                     f'stroke="{KINDS[kind][0]}" stroke-width="2"/>')
parts += [text(600, 156, "", "sub"),
          f'  <path d="M16,{160 + 3 * 44 - 14} H644" fill="none" stroke="{EDGE}" stroke-width="1.2" '
          f'stroke-dasharray="4 4"/>',
          tag(20, 134, "reduce: add the whole message", "client"),
          tag(20, 160 + 3 * 44 - 8, "broadcast: copy the total down", "good"),
          note(16, 470, "Each arrow moves the whole message, m bytes. 8 = 2³ ranks need log₂ 8 = 3 steps up and 3",
               room=628),
          note(16, 488, "down: 6 steps against the ring's 14. Rank 0 receives 3 messages and sends 3, so its link",
               room=628),
          note(16, 506, "carries 3m each way against 1.75m in a ring of 8.", room=628)]
save(OUT + "ch28-tree.svg", 660, 528,
     "Eight columns, ranks 0 to 7, and 6 rows of arrows. Reduce: in step 1 ranks 1, 3, 5 and 7 send to 0, 2, 4 "
     "and 6; in step 2 ranks 2 and 6 send to 0 and 4; in step 3 rank 4 sends to 0. Broadcast mirrors it: rank 0 "
     "sends to 4, then 0 and 4 send to 2 and 6, then 0, 2, 4 and 6 send to 1, 3, 5 and 7. 6 steps; rank 0 sends "
     "and receives the whole message 3 times.",
     "Tree all-reduce for 8 ranks", "Sums climb a binary tree to rank 0, then the total comes back down.",
     parts)


# 25.5  The alpha-beta model for k = 8: time against message size for the ring and the tree.
ALPHA, B, K = 10e-6, 25e9, 8


def t_ring(m):
    return 2 * (K - 1) * ALPHA + 2 * (K - 1) / K * m / B


def t_tree(m):
    steps = 2 * int(math.log2(K))
    return steps * (ALPHA + m / B)


gx, gy, gw, gh = 96, 120, 500, 300
m_lo, m_hi, t_lo, t_hi = 1e3, 1e9, 1e-5, 1e0


def px(m):
    return gx + gw * (math.log10(m) - math.log10(m_lo)) / (math.log10(m_hi) - math.log10(m_lo))


def py(t):
    return gy + gh - gh * (math.log10(t) - math.log10(t_lo)) / (math.log10(t_hi) - math.log10(t_lo))


parts = [f'  <rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="#FFFFFF" stroke="{EDGE}" stroke-width="1.2"/>']
for e, name in zip(range(3, 10), ["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB", "1 GB"]):
    x = px(10 ** e)
    parts += [line(x, gy, x, gy + gh, EDGE, 1), mono(x, gy + gh + 18, name, "middle", MUTED)]
for e, name in zip(range(-5, 1), ["10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s"]):
    y = py(10 ** e)
    parts += [line(gx, y, gx + gw, y, EDGE, 1), mono(gx - 8, y + 4, name, "end", MUTED)]
for f, kind in ((t_ring, "client"), (t_tree, "store")):
    pts = [(px(10 ** (3 + j / 20)), py(f(10 ** (3 + j / 20)))) for j in range(121)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts if gy <= y <= gy + gh)
    parts.append(f'  <path d="{d}" fill="none" stroke="{KINDS[kind][0]}" stroke-width="2.6"/>')
cross = 8 * ALPHA * B / (6 - 1.75)
cxp, cyp = px(cross), py(t_ring(cross))
parts += [f'  <circle cx="{cxp:.1f}" cy="{cyp:.1f}" r="5" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>',
          line(cxp, cyp + 6, cxp, gy + gh, INK, 1.2, "4 3"),
          tag(cxp + 10, cyp + 10, "crossover: 470 KB, 0.17 ms", "idle"),
          tag(gx + 14, py(t_tree(4e3)) + 8, "tree: 6 steps", "store"),
          tag(gx + 14, py(t_ring(4e3)) - 26, "ring: 14 steps", "client"),
          tag(px(2e7) - 6, py(t_tree(2e7)) - 12, "tree: 6m / B", "store", "end"),
          tag(px(3e7) + 8, py(t_ring(3e7)) + 10, "ring: 1.75m / B", "client"),
          text(gx + gw / 2, gy + gh + 40, "message size m", "sub", "middle"),
          text(gx - 60, gy - 12, "time", "sub"),
          note(16, 484, "k = 8 ranks, α = 10 µs per message, B = 25 GB/s per link. Below 470 KB the tree's 6 steps",
               room=628),
          note(16, 502, "beat the ring's 14; above it the ring's 1.75m of traffic per link beats the tree's 6m.",
               room=628)]
save(OUT + "ch28-alpha-beta.svg", 660, 524,
     "A log-log chart of time against message size from 1 KB to 1 GB for an all-reduce over 8 ranks with "
     "alpha = 10 microseconds and B = 25 GB/s. The tree line starts lower, at 60 microseconds for small "
     "messages, against 140 for the ring. The lines cross at 470 KB and 0.17 ms. For large messages the ring "
     "is lower, 1.75 m / B against 6 m / B for the tree.",
     "Latency or bandwidth", "Time of 1 all-reduce over 8 ranks, from the model T = steps × α + bytes ÷ B.",
     parts)


# 25.6  The simulation: 1 buffer of k slices, and 1 step as 1 launch of transfers.
cw = 34
bx, by = 50, 150
parts = [text(bx, 112, "all: 1 buffer of k × n floats in the GPU's memory", "sub")]
for r in range(4):
    parts += [text(bx + r * 4 * cw + 2 * cw - 2, by - 10, f"rank {r}", "name", "middle")]
    for c in range(4):
        kind = "client" if c == r else "idle"
        parts.append(box(bx + (4 * r + c) * cw, by, cw - 4, 30, kind, f"c{c}", on=c == r))
lanes = [(0, 4), (5, 9), (10, 14), (15, 3)]
for k, (src, dst) in enumerate(lanes):
    y = by + 80 + k * 30
    xs, xd = bx + src * cw + cw / 2 - 2, bx + dst * cw + cw / 2 - 2
    colour = KINDS["client"][0]
    parts += [f'  <circle cx="{xs}" cy="{by + 30}" r="3" fill="{colour}"/>',
              f'  <path d="M{xs},{by + 34} V{y} H{xd} V{by + 33}" fill="none" stroke="{colour}" '
              f'stroke-width="1.6" stroke-dasharray="{"" if dst > src else "5 4"}" marker-end="url(#head)"/>',
              mono(min(xs, xd) + 8, y - 6, f"transfer {k}", "start", colour)]
parts += [card(50, 360, 270, 70, "client", "1 step = 1 launch", ["blocks 0 … per − 1: transfer 0",
                                                                  "blocks per … 2 per − 1: transfer 1"]),
          card(340, 360, 290, 70, "good", "No race inside a step", ["no chunk is both read", "and written within 1 step"]),
          note(16, 462, "Step 1 of reduce-scatter on 4 ranks: rank r adds its chunk r into the same chunk of rank", room=628),
          note(16, 480, "r + 1. The 4 transfers run in 1 kernel launch, like 4 links working at once. The plan of", room=628),
          note(16, 498, "every step, pairs of offsets (from, to), sits in GPU memory before the first launch.", room=628)]
save(OUT + "ch28-plan.svg", 660, 520,
     "A buffer of 16 chunks, 4 per rank. In step 1 of reduce-scatter, 4 arrows leave chunk c0 of rank 0, c1 of "
     "rank 1, c2 of rank 2 and c3 of rank 3, and land on the same chunk of the next rank; the last one wraps "
     "from rank 3 back to rank 0. Below, 2 cards: 1 step is 1 launch, with blocks divided among the transfers; "
     "no element is both read and written within a step, so the transfers cannot race.",
     "Ranks as slices of 1 buffer", "Each step of the plan becomes 1 kernel launch on the GPU.", parts)


# 25.8  NCCL between a framework and the wires.
parts = [card(40, 116, 580, 56, "outside", "framework: PyTorch DistributedDataParallel",
              "an all-reduce of the gradients after each backward pass", glyph="stack"),
         card(40, 206, 580, 74, "client", "NCCL: 1 communicator per rank",
              ["finds the topology, builds rings and trees, runs each collective",
               "as 1 kernel on the rank's CUDA stream"], glyph="nvidia"),
         wire([(330, 172), (330, 206)]),
         fan(330, 280, [102, 252, 402, 552], 300, 330)]
for x, (name, sub, kind) in zip((32, 182, 332, 482),
                                [("NVLink", "NVSwitch", "good"), ("PCIe", "peer to peer", "store"),
                                 ("host memory", "same node", "idle"), ("network", "InfiniBand, RoCE", "danger")]):
    parts.append(card(x, 330, 140, 64, kind, name, sub))
parts += [note(40, 426, "The same ncclAllReduce call runs over whichever transport joins 2 ranks; NCCL picks it", room=580),
          note(40, 444, "from the topology it detects when the communicator is created.", room=580)]
save(OUT + "ch28-nccl-stack.svg", 660, 466,
     "Three layers. At the top, a framework such as PyTorch DistributedDataParallel asks for an all-reduce of "
     "the gradients. In the middle, NCCL keeps 1 communicator per rank, finds the topology, builds rings and "
     "trees, and runs each collective as 1 kernel on the rank's stream. At the bottom, 4 transports: NVLink and "
     "NVSwitch, PCIe peer to peer, host memory within a node, and the network, InfiniBand or RoCE.",
     "Where NCCL sits", "1 call from the framework, 1 kernel per rank, a transport for each pair.",
     parts)


# 25.7  Measured on 1 GPU (Orin NX, C++ run 3 of 3) against T = steps × α + traffic ÷ B_mem.
SIZES = [4096, 65536, 1048576, 16777216]
MEASURED = {"ring": [0.162, 0.226, 0.913, 7.184], "tree": [0.070, 0.143, 0.795, 6.645]}
STEP_US, MEM_GBS = 11.6, 84.0
gx, gy, gw, gh = 96, 120, 500, 280
m_lo, m_hi, t_lo, t_hi = 2e3, 4e7, 1e-2, 2e1          # bytes, milliseconds


def mx(m):
    return gx + gw * (math.log10(m) - math.log10(m_lo)) / (math.log10(m_hi) - math.log10(m_lo))


def my(t):
    return gy + gh - gh * (math.log10(t) - math.log10(t_lo)) / (math.log10(t_hi) - math.log10(t_lo))


def model(steps, m):
    return steps * STEP_US / 1e3 + 35 * m / (MEM_GBS * 1e9) * 1e3


parts = [f'  <rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="#FFFFFF" stroke="{EDGE}" stroke-width="1.2"/>']
for m, name in zip(SIZES, ["4 KiB", "64 KiB", "1 MiB", "16 MiB"]):
    x = mx(m)
    parts += [line(x, gy, x, gy + gh, EDGE, 1), mono(x, gy + gh + 18, name, "middle", MUTED)]
for t, name in [(0.01, "10 µs"), (0.1, "100 µs"), (1, "1 ms"), (10, "10 ms")]:
    y = my(t)
    parts += [line(gx, y, gx + gw, y, EDGE, 1), mono(gx - 8, y + 4, name, "end", MUTED)]
for key, steps, kind in (("ring", 14, "client"), ("tree", 6, "store")):
    colour = KINDS[kind][0]
    pts = [(mx(10 ** (3.4 + j / 40)), my(model(steps, 10 ** (3.4 + j / 40)))) for j in range(164)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts if gx <= x <= gx + gw and gy <= y <= gy + gh)
    parts.append(f'  <path d="{d}" fill="none" stroke="{colour}" stroke-width="1.8" stroke-dasharray="6 4"/>')
    for m, t in zip(SIZES, MEASURED[key]):
        parts.append(f'  <circle cx="{mx(m):.1f}" cy="{my(t):.1f}" r="5.5" fill="{colour}" stroke="#FFFFFF" '
                     f'stroke-width="1.5"/>')
parts += [legend(gx, 104, [("client", "ring, 14 steps"), ("store", "tree, 6 steps")]),
          text(gx + 300, 108, "dots measured, dashes model", "sub"),
          tag(mx(4096) + 12, my(0.162) - 58, "0.162 ms ÷ 14 = 11.6 µs per step", "client"),
          tag(mx(4096) + 12, my(0.070) + 10, "0.070 ms ÷ 6 = 11.7 µs per step", "store"),
          tag(mx(16777216) - 40, my(7.184) - 6, "587 MB ÷ 84 GB/s ≈ 7 ms", "idle", "end"),
          text(gx + gw / 2, gy + gh + 40, "message size", "sub", "middle"),
          note(16, 476, "8 simulated ranks on 1 GPU of 102 GB/s. Small messages cost 1 step time each, so the tree's", room=628),
          note(16, 494, "6 steps win; large ones move 35 bytes of memory traffic per message byte in both, and the", room=628),
          note(16, 512, "times meet, because all the simulated links share 1 memory.", room=628)]
save(OUT + "ch28-measured.svg", 660, 534,
     "A log-log chart of measured all-reduce time against message size on 8 simulated ranks of 1 GPU, "
     "with dashed model lines. Ring: 0.162 ms at 4 KiB, 0.226 at 64 KiB, 0.913 at 1 MiB, 7.184 at 16 MiB. "
     "Tree: 0.070, 0.143, 0.795 and 6.645 ms. At 4 KiB each step costs about 11.6 microseconds; at 16 MiB "
     "both move 587 MB of memory traffic at about 84 GB/s.",
     "Measured on 1 GPU", "Time of 1 all-reduce over 8 simulated ranks, against the model.", parts)
