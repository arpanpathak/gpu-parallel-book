"""The figures of chapter 14, drawn with figkit and figparts.

    python3 tools/draw_ch14.py

The rates in the last figure are the ones printed by code/oxide/ch14_map_saxpy in the run
quoted in the chapter.
"""
from figparts import (EDGE, INK, KINDS, MUTED, arrow, card, fan, legend, line, mono, note, save,
                      tag, text, wire, zone)

OUT = "src/figures/"

PEAK = 102.0                      # GB/s, the measured GPU's memory bandwidth
SMS = 8                           # SMs of the measured GPU
SAXPY, MAP = 90.7, 90.1           # GB/s, 1 thread per element
SLOTS = 6                         # blocks of 256 resident per SM: 1,536 threads
STRIDE = [(1, 32.5), (2, 59.5), (3, 79.1), (6, 93.4), (12, 91.9), (24, 91.4), (48, 88.2)]


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


# 14.1  The map pattern: the same function on every element, each output from its own inputs.
X = [0.0, 0.5, 1.0, 1.5]
Y = [0.0, 1.0, 2.0, 3.0]
parts = [text(40, 122, "x", "name"), text(40, 172, "y", "name"), text(40, 318, "y", "name"),
         text(40, 336, "after", "sub")]
for i in range(4):
    cx = 125 + i * 140
    out = 2.5 * X[i] + Y[i]
    parts += [mono(cx, 100, f"i = {i}", "middle", MUTED),
              box(cx - 45, 106, 90, 26, "client", f"{X[i]}"),
              box(cx - 45, 156, 90, 26, "store", f"{Y[i]}"),
              line(cx, 132, cx, 156, EDGE, 1),
              arrow(cx, 182, 222),
              box(cx - 64, 222, 128, 30, "outside", f"2.5 × {X[i]} + {Y[i]}", cls="mono"),
              arrow(cx, 252, 298),
              box(cx - 45, 298, 90, 30, "good", f"{out}"),
              mono(cx, 352, f"thread {i}", "middle", KINDS["service"][0])]
parts += [note(40, 390, "f(x, y) = a x + y with a = 2.5. Element i of the result needs x[i] and y[i] and", room=600),
          note(40, 408, "nothing else, so the 4 evaluations can run in any order, or all at once.", room=600)]
save(OUT + "ch14-map.svg", 660, 430,
     "4 columns, i = 0 to 3. Each column takes x[i] and y[i], 0 and 0, 0.5 and 1, 1 and 2, 1.5 and 3, "
     "through the same function 2.5 x + y, and gives 0, 2.25, 4.5 and 6.75. Thread i computes column i.",
     "The map pattern", "1 function, applied to every element on its own.", parts)


# 14.2  Bytes and operations of 1 element of SAXPY, and the time each would take for N = 16 Mi.
parts = [zone(16, 104, 300, 250, "1 element"), zone(332, 104, 312, 250, "N = 16,777,216 elements")]
for k, (name, kind) in enumerate((("read x[i]", "client"), ("read y[i]", "store"), ("write y[i]", "good"))):
    parts += [box(40, 140 + k * 44, 120, 32, kind, name, cls="pill"), mono(176, 161 + k * 44, "4 bytes")]
parts += [mono(40, 290, "12 bytes, 2 FLOPs", "start", INK),
          mono(40, 312, "2 / 12 = 0.17 FLOP/B", "start", INK),
          note(40, 336, "1 multiply, 1 add", room=260)]
memory_ms = 12 * 16777216 / PEAK / 1e6
math_ms = 2 * 16777216 / 2.4e12 * 1e3
wide = 250
parts += [text(356, 146, "memory: 201.3 MB at 102 GB/s", "sub"),
          f'  <rect x="356" y="156" width="{wide}" height="22" rx="5" fill="{KINDS["danger"][0]}"/>',
          mono(356, 196, f"{memory_ms:.2f} ms", "start", INK),
          text(356, 236, "arithmetic: 33.6 MFLOP at 2.4 TFLOP/s", "sub"),
          f'  <rect x="356" y="246" width="{max(wide * math_ms / memory_ms, 2):.1f}" height="22" rx="5" '
          f'fill="{KINDS["good"][0]}"/>',
          mono(356, 286, f"{math_ms:.3f} ms", "start", INK),
          note(356, 326, "Memory takes 140 times as long.", room=270)]
save(OUT + "ch14-traffic.svg", 660, 376,
     "Left: 1 element of SAXPY reads x[i] and y[i] and writes y[i], 4 bytes each, 12 bytes for 2 floating-"
     "point operations, 0.17 FLOP per byte. Right: for 16,777,216 elements the memory traffic, 201.3 MB at "
     "102 GB/s, takes 1.97 ms, while the arithmetic, 33.6 MFLOP at 2.4 TFLOP/s, takes 0.014 ms.",
     "SAXPY is memory-bound", "12 bytes move for every 2 operations.", parts)


# 14.3  A closure becomes 1 kernel parameter, and each closure type its own kernel.
parts = [zone(16, 104, 628, 150, "host"),
         zone(16, 274, 628, 160, "device")]
parts += [card(40, 140, 280, 92, "client", "move |x, y| a * x + y", ["captures a = 2.5", "4 bytes"]),
          card(340, 140, 280, 92, "outside", "|x, y| x.max(y)", ["captures nothing", "0 bytes"]),
          arrow(180, 232, 310), arrow(480, 232, 310),
          card(40, 310, 280, 100, "client", "map::<closure 1>", ["f arrives as 1 parameter,", "{ a: 2.5 }"]),
          card(340, 310, 280, 100, "outside", "map::<closure 2>", ["f arrives as 1 parameter,", "{ }, empty"]),
          note(40, 458, "The generic kernel map<F> is compiled once for every closure type it is", room=600),
          note(40, 476, "launched with, and in each one the call f(x, y) runs that closure's code.", room=600)]
save(OUT + "ch14-closure.svg", 660, 498,
     "Top: 2 closures on the host. The first, move x y to a times x plus y, captures a = 2.5, 4 bytes. The "
     "second, x max y, captures nothing. Bottom: each has its own type, and the compiler builds 1 kernel "
     "map for each: the first receives its closure as 1 parameter holding a = 2.5, the second as an empty "
     "parameter.",
     "Closures as kernel arguments", "Each closure has its own type, and gets its own kernel.", parts)


# 14.4  A grid-stride loop: 8 threads, 20 elements, stride 8.
n, stride, cell, gap = 20, 8, 30, 4
colours = ["client", "service", "store", "outside", "danger", "good", "model", "idle"]
parts = [text(40, 120, "grid: 2 blocks of 4 threads, stride = 2 × 4 = 8", "sub")]
for t in range(stride):
    x = 120 + t * (cell + gap)
    parts.append(box(x, 132, cell, 26, colours[t], f"t{t}", cls="pill"))
for p in range(3):
    y = 190 + p * 50
    parts.append(text(40, y + 19, f"pass {p}", "name"))
    for t in range(stride):
        i = p * stride + t
        x = 120 + t * (cell + gap)
        if i < n:
            parts.append(box(x, y, cell, 28, colours[t], str(i)))
        else:
            parts.append(box(x, y, cell, 28, "idle", "", on=False))
    parts.append(mono(120 + stride * (cell + gap) + 10, y + 19,
                      f"i = {p * stride} to {min(p * stride + stride, n) - 1}", "start", MUTED))
parts += [note(40, 368, "Thread 3 handles elements 3, 11 and 19; thread 5 handles 5 and 13, and stops", room=600),
          note(40, 386, "at 21 because 21 ≥ 20. In every pass, neighbouring threads touch neighbouring", room=600),
          note(40, 404, "elements, and no element belongs to 2 threads: i mod 8 is the thread.", room=600)]
save(OUT + "ch14-stride.svg", 660, 426,
     "8 threads, t0 to t7, 2 blocks of 4. 20 elements in 3 passes: pass 0 covers elements 0 to 7, pass 1 "
     "8 to 15, pass 2 16 to 19, and threads 4 to 7 have nothing to do in pass 2. Each element has the colour "
     "of its thread: element i belongs to thread i mod 8.",
     "A grid-stride loop", "8 threads cover 20 elements in 3 passes.", parts)


# 14.5  How much of the GPU a grid occupies: 6 slots of 256 threads per SM.
parts = [text(40, 120, "8 SMs, each with room for 1,536 threads = 6 blocks of 256", "sub")]
rows = [(1, "1 block per SM: 2,048 threads, 1 of 6 slots"),
        (3, "3 blocks per SM: 6,144 threads, half the slots"),
        (6, "6 blocks per SM: 12,288 threads, every slot"),
        (12, "12 blocks per SM: every slot, and a second wave of 6")]
for r, (k, label) in enumerate(rows):
    y = 140 + r * 74
    parts.append(text(40, y + 12, label, "name"))
    for s in range(SMS):
        x = 40 + s * 74
        parts.append(f'  <rect x="{x}" y="{y + 22}" width="68" height="30" rx="5" fill="#FFFFFF" '
                     f'stroke="{EDGE}" stroke-width="1.1"/>')
        for b in range(min(k, SLOTS)):
            accent = KINDS["client" if k <= SLOTS else "outside"][0]
            parts.append(f'  <rect x="{x + 4 + b * 10.5:.1f}" y="{y + 26}" width="8" height="22" '
                         f'rx="1.5" fill="{accent}"/>')
parts += [note(40, 446, "A block of 256 threads occupies 1 of 6 slots. Fewer than 6 blocks per SM leave", room=600),
          note(40, 464, "slots empty; blocks beyond 6 wait for a free slot and run in a later wave.", room=600)]
save(OUT + "ch14-slots.svg", 660, 486,
     "4 rows of 8 SMs, each SM drawn with 6 slots for blocks of 256 threads. With 1 block per SM, 1 slot "
     "of 6 is filled; with 3, half; with 6, all; with 12, all slots are filled and a second wave of 6 "
     "blocks per SM waits.",
     "Blocks per SM", "A grid of 8 SMs × k blocks fills k of 6 slots in each SM.", parts)


# 14.6  Measured rates of the grid-stride kernel against Little's law and the bandwidth.
x0, y0, wide, high = 110, 140, 480, 260
latency_us = 16384 / (STRIDE[0][1] * 1e3)       # bytes / (GB/s) = µs
parts = [legend(x0, 104, [("client", "measured, grid-stride")]),
         legend(x0 + 200, 104, [("danger", "Little's law, capped at 102 GB/s")])]
for g in range(0, 121, 20):
    y = y0 + high - high * g / 120
    parts += [line(x0, y, x0 + wide, y, EDGE, 1), mono(x0 - 8, y + 4, f"{g}", "end", MUTED)]
parts.append(text(x0 - 60, y0 - 8, "GB/s", "sub"))
peak_y = y0 + high - high * PEAK / 120
parts += [line(x0, peak_y, x0 + wide, peak_y, KINDS["danger"][0], 1.6, "6 4"),
          mono(x0 + 4, peak_y - 8, "102 GB/s", "start", KINDS["danger"][0])]
step = wide / len(STRIDE)
prev = None
for k, (per_sm, rate) in enumerate(STRIDE):
    cx = x0 + step * (k + 0.5)
    h = high * rate / 120
    accent = KINDS["client"][0]
    parts += [f'  <rect x="{cx - 20:.1f}" y="{y0 + high - h:.1f}" width="40" height="{h:.1f}" rx="4" '
              f'fill="{KINDS["client"][1]}" stroke="{accent}" stroke-width="1.3"/>',
              mono(cx, y0 + high - h + 18, f"{rate:.0f}", "middle", accent),
              mono(cx, y0 + high + 18, f"{per_sm}", "middle", MUTED)]
    bytes_in_flight = SMS * min(per_sm, SLOTS) * 256 * 8
    model = min(bytes_in_flight / latency_us / 1e3, PEAK)
    py = y0 + high - high * model / 120
    parts.append(f'  <circle cx="{cx:.1f}" cy="{py:.1f}" r="4" fill="{KINDS["danger"][0]}"/>')
    if prev:
        parts.append(line(prev[0], prev[1], cx, py, KINDS["danger"][0], 1.4))
    prev = (cx, py)
parts += [text(x0 + wide / 2, y0 + high + 40, "blocks of 256 threads per SM", "sub", "middle"),
          note(40, 476, "1 block per SM keeps 8 × 256 × 8 bytes = 16 KiB of loads in flight and moved", room=600),
          note(40, 494, f"32.5 GB/s, so a load takes 16,384 B ÷ 32.5 GB/s = {latency_us:.2f} µs. Twice the bytes", room=600),
          note(40, 512, "in flight give about twice the rate, until the memory bandwidth caps it.", room=600)]
save(OUT + "ch14-little.svg", 660, 534,
     "Bars of measured SAXPY rate for the grid-stride kernel against blocks per SM: 1 block 32.5 GB/s, 2 "
     "blocks 59.5, 3 blocks 79.1, 6 blocks 93.4, 12 blocks 91.9, 24 blocks 91.4, 48 blocks 88.2. A red line "
     "shows Little's law with a latency of 0.50 microseconds: 32.5, 65, 97.5, then capped at 102 GB/s.",
     "Enough loads in flight", "SAXPY rate against blocks per SM, 1 GPU of 102 GB/s.", parts)
