"""The figures of chapter 19, drawn with figkit and figparts.

    python3 tools/draw_ch19.py

The small examples are the ones the program code/oxide/ch19_scan_apps prints; the measured
numbers in figures 19.4 and 19.7 are from its run quoted in the chapter.
"""
from figparts import MUTED, WIRE, bar, legend, line, mono, note, save

OUT = "src/figures/"
KEYS = [3, 8, 1, 9, 4, 7, 2, 6]


def elbow(x1, y1, x2, y2, mid, colour=WIRE):
    """A connector from (x1, y1) down to (x2, y2): down to `mid`, across, down, with an arrow."""
    return (f'  <path d="M{x1:.1f},{y1:.1f} V{mid:.1f} H{x2:.1f} V{y2 - 3:.1f}" fill="none" '
            f'stroke="{colour}" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#head)"/>')


def row(y, name, values, kinds, x0=150, step=58, w=48, h=28):
    parts = [mono(x0 - 16, y + h / 2 + 4, name, "end", MUTED)]
    for k, (v, kind) in enumerate(zip(values, kinds)):
        parts.append(bar(x0 + k * step, y, w, h, kind, str(v)))
    return parts


def centre(k, x0=150, step=58, w=48):
    return x0 + k * step + w / 2


def header(y, name):
    """The column numbers 0 to 7 above a table of cells, each centred on its column."""
    return [mono(134, y, name, "end", MUTED)] + [mono(centre(k), y, str(k), "middle", MUTED) for k in range(8)]


# 19.1  Stream compaction of 8 keys: flags, scan, scatter.
flags = [int(k >= 5) for k in KEYS]
inclusive = [sum(flags[:i + 1]) for i in range(8)]
exclusive = [s - f for s, f in zip(inclusive, flags)]
kept = [k for k in KEYS if k >= 5]
kinds = ["good" if f else "idle" for f in flags]

parts = header(100, "index")
parts += row(112, "input", KEYS, kinds)
parts += row(158, "flag", flags, kinds)
parts += row(204, "inclusive", inclusive, ["service"] * 8)
parts += row(250, "exclusive", exclusive, ["client" if f else "idle" for f in flags])
parts += row(372, "output", kept, ["good"] * 4)
for j, k in enumerate(i for i in range(8) if flags[i]):
    parts.append(elbow(centre(k), 278, centre(j), 372, 300 + 14 * j))
parts += [mono(150, 424, "out[exclusive[i]] = input[i] for every kept i", "start", MUTED),
          note(32, 456, "The flag is 1 for x >= 5. The exclusive scan counts the kept keys before each key,", room=596),
          note(32, 474, "so the 4 kept keys go to positions 0, 1, 2 and 3, in their input order.", room=596)]
save(OUT + "ch19-compact.svg", 660, 496,
     "8 keys 3, 8, 1, 9, 4, 7, 2, 6 with flags 0, 1, 0, 1, 0, 1, 0, 1 for x >= 5. The inclusive scan of the "
     "flags is 0, 1, 1, 2, 2, 3, 3, 4 and the exclusive scan 0, 0, 1, 1, 2, 2, 3, 3. Arrows take the kept keys "
     "8, 9, 7 and 6 from their exclusive positions 0, 1, 2, 3 to the output 8, 9, 7, 6.",
     "Stream compaction of 8 keys", "A flag per key, a scan of the flags, a write at the scanned position.",
     parts)


# 19.2  Bytes per element moved by the 2 compaction methods, for a kept fraction f = 0.5.
stages = [
    ("3 passes", [("keep_flags", 8, "client"), ("scan", 16, "service"), ("scatter", 14, "store")]),
    ("warp votes", [("count_kept", 4, "client"), ("compact_votes", 6, "store")]),
    ("minimum", [("read + write kept", 6, "good")]),
]
parts, scale, x0 = [], 11, 130
for r, (name, pieces) in enumerate(stages):
    y = 120 + r * 92
    parts.append(mono(x0 - 14, y + 20, name, "end", MUTED))
    at = x0
    for label, size, kind in pieces:
        w = size * scale
        parts.append(bar(at, y, w, 30, kind, f"{size} B"))
        at += w
    total = sum(size for _, size, _ in pieces)
    parts.append(mono(at + 10, y + 20, f"= {total} B", "start"))
    names = " + ".join(f"{label} {size} B" for label, size, _ in pieces)
    parts.append(mono(x0, y + 52, names, "start", MUTED))
parts += [note(32, 420, "Per input element of 4 bytes, with half of the elements kept (f = 0.5). The", room=596),
          note(32, 438, "scan reads and writes the flags twice; the scatter reads input, flag and scan.", room=596)]
save(OUT + "ch19-bytes.svg", 660, 460,
     "Bars of bytes moved per input element when half the elements are kept. 3 passes: keep_flags 8 bytes, "
     "scan 16 bytes, scatter 14 bytes, total 38 bytes. Warp votes: count_kept 4 bytes, compact_votes 6 "
     "bytes, total 10 bytes. The minimum, reading the input and writing the kept half, is 6 bytes.",
     "Bytes moved per element", "3 passes move 38 bytes per element, warp votes 10, the minimum is 6.",
     parts)


# 19.3  A warp vote, drawn for 8 lanes: ballot, the lanes below, and the rank.
keep = [0, 1, 1, 0, 1, 0, 0, 1]
parts = header(100, "lane")
parts += row(112, "keep", keep, ["good" if k else "idle" for k in keep])
below = [1 if lane < 4 else 0 for lane in range(8)]
both = [a & b for a, b in zip(keep, below)]
parts += row(170, "votes", keep, ["client"] * 8)
parts += row(216, "below 4", below, ["store" if b else "idle" for b in below])
parts += row(262, "votes & below", both, ["service" if b else "idle" for b in both])
parts += [bar(150, 330, 236, 34, "good", "lane 4: popcount 2 = rank 2"),
          mono(410, 352, "next + before + rank", "start", MUTED),
          note(32, 404, "ballot returns 1 bit per lane of the warp, here drawn for 8 lanes. A kept lane's", room=596),
          note(32, 422, "rank is the number of kept lanes to its left: the 1 bits of the votes below it.", room=596)]
save(OUT + "ch19-ballot.svg", 660, 444,
     "8 lanes with keep flags 0, 1, 1, 0, 1, 0, 0, 1. The ballot is the same 8 bits. For lane 4 the mask of "
     "lanes below is 1 for lanes 0 to 3; the votes and the mask give 0, 1, 1, 0, whose popcount 2 is the "
     "rank of lane 4 among the kept lanes.",
     "A rank from a warp vote", "The kept lanes to the left of lane 4, counted from 1 ballot.", parts)


# 19.4  A stable split by bit 0, and the 3 passes that sort 3-bit keys.
keys = [5, 2, 7, 0, 3, 6, 1, 4]
zero = [1 - (k & 1) for k in keys]
upto = [sum(zero[:i + 1]) for i in range(8)]
zeros = upto[-1]
at = [u - 1 if z else zeros + i - u for i, (z, u) in enumerate(zip(zero, upto))]
out = [0] * 8
for i, a in enumerate(at):
    out[a] = keys[i]
parts = header(100, "index")
parts += row(112, "key", keys, ["good" if z else "store" for z in zero])
parts += row(158, "bit 0 is 0", zero, ["good" if z else "idle" for z in zero])
parts += row(204, "inclusive", upto, ["service"] * 8)
parts += row(250, "position", at, ["good" if z else "store" for z in zero])
parts += row(340, "output", out, ["good" if k % 2 == 0 else "store" for k in out])
parts += [mono(150, 306, "zero bit: inclusive - 1     one bit: 4 + i - inclusive", "start", MUTED),
          note(32, 404, "4 keys have bit 0 = 0, so they take positions 0 to 3 and the others 4 to 7.", room=596),
          note(32, 422, "Within each group the input order is kept: the split is stable.", room=596)]
save(OUT + "ch19-split.svg", 660, 444,
     "Keys 5, 2, 7, 0, 3, 6, 1, 4. Flags for bit 0 equal to 0: 0, 1, 0, 1, 0, 1, 0, 1. Inclusive scan 0, 1, 1, "
     "2, 2, 3, 3, 4, so 4 keys have a zero bit. Positions 4, 0, 5, 1, 6, 2, 7, 3. Output 2, 0, 6, 4, 5, 7, 3, 1: "
     "even keys first, odd keys after, each group in input order.",
     "A stable split by 1 bit", "Zero-bit keys to the front, one-bit keys after them, in order.", parts)

passes = [keys]
for bit in range(3):
    cur = passes[-1]
    passes.append([k for k in cur if not (k >> bit) & 1] + [k for k in cur if (k >> bit) & 1])
parts = []
for p, values in enumerate(passes):
    y = 104 + p * 76
    name = "input" if p == 0 else f"after bit {p - 1}"
    if p < 3:
        kinds = ["good" if not (k >> p) & 1 else "store" for k in values]
    else:
        kinds = ["client"] * 8
    parts += row(y, name, values, kinds)
    for k, v in enumerate(values):
        parts.append(mono(centre(k), y + 44, format(v, "03b"), "middle", MUTED))
parts.append(legend(150, 410, [("good", "next bit is 0"), ("store", "next bit is 1")]))
parts += [note(32, 436, "Each pass is a stable split on the next bit. Because the split keeps the order of", room=596),
          note(32, 454, "the earlier passes, after the last pass the keys are in order: 0 to 7.", room=596)]
save(OUT + "ch19-radix.svg", 660, 476,
     "Keys 5, 2, 7, 0, 3, 6, 1, 4 after a split on bit 0: 2, 0, 6, 4, 5, 7, 3, 1. After bit 1: 0, 4, 5, 1, 2, "
     "6, 7, 3. After bit 2: 0, 1, 2, 3, 4, 5, 6, 7, sorted.",
     "A radix sort of 3-bit keys", "3 stable splits, least significant bit first, sort the 8 keys.", parts)


# 19.6  Equalisation of a 16-pixel image with 8 grey levels.
counts = [0, 0, 4, 6, 4, 2, 0, 0]
cdf = [sum(counts[:v + 1]) for v in range(8)]
low, n = 4, 16
table = [((7 * max(c - low, 0)) + (n - low) // 2) // (n - low) for c in cdf]
parts = header(100, "level")
parts += row(112, "count", counts, ["store" if c else "idle" for c in counts])
parts += row(158, "cdf", cdf, ["service"] * 8)
parts += row(204, "new level", table, ["good" if c else "idle" for c in counts])
base = 384
for panel, x0 in (("before", 70), ("after", 380)):
    for v in range(8):
        x = x0 + v * 30
        parts.append(mono(x + 10, base + 16, str(v), "middle", MUTED))
    parts.append(line(x0 - 6, base, x0 + 236, base, MUTED))
for v in range(8):
    h = counts[v] * 14
    if h:
        parts.append(bar(70 + v * 30, base - h, 20, h, "store"))
        parts.append(bar(380 + table[v] * 30, base - h, 20, h, "good"))
parts += [mono(70, 262, "before: levels 2 to 5", "start", MUTED),
          mono(380, 262, "after: levels 0, 4, 6, 7", "start", MUTED),
          note(32, 436, "new level = round(7 (cdf - 4) / (16 - 4)). The 4 occupied levels 2, 3, 4, 5", room=596),
          note(32, 454, "move to 0, 4, 6, 7: the narrow range is stretched over the whole scale.", room=596)]
save(OUT + "ch19-cdf.svg", 660, 476,
     "A 16-pixel image with 8 grey levels has counts 0, 0, 4, 6, 4, 2, 0, 0. The cumulative counts are 0, 0, 4, "
     "10, 14, 16, 16, 16. With the first occupied cdf 4 and 16 pixels, the new levels of the occupied levels "
     "2, 3, 4, 5 are 0, 4, 6, 7. 2 small histograms show the counts before, on levels 2 to 5, and after, "
     "spread from 0 to 7.",
     "Histogram equalisation by hand", "The cumulative histogram becomes the table of new levels.", parts)


# 19.7  The measured image: shares of the pixels in 8 groups of 32 levels, before and after.
groups = ["0-31", "32-63", "64-95", "96-127", "128-159", "160-191", "192-223", "224-255"]
before = [0.0, 0.0, 0.0, 51.6, 48.4, 0.0, 0.0, 0.0]
after = [11.7, 13.0, 12.2, 11.5, 11.9, 12.7, 13.7, 13.3]
parts = []
for name, shares, kind, y0 in (("before", before, "store", 250), ("after", after, "good", 430)):
    parts.append(mono(32, y0 - 128, name, "start", MUTED))
    parts.append(line(100, y0, 620, y0, MUTED))
    for k, share in enumerate(shares):
        x = 110 + k * 64
        h = share * 2.2
        if h:
            parts.append(bar(x, y0 - h, 44, h, kind))
        parts.append(mono(x + 22, y0 - h - 6, f"{share:.1f} %", "middle", MUTED))
        parts.append(mono(x + 22, y0 + 16, groups[k], "middle", MUTED))
parts += [note(32, 470, "8,192 x 8,192 pixels. Before: all pixels on levels 96 to 158. After: about 1/8", room=596),
          note(32, 488, "of the pixels in each group of 32 levels; level 96 maps to 0, 127 to 131, 158 to 255.",
               room=596)]
save(OUT + "ch19-shares.svg", 660, 510,
     "2 bar charts of the measured image, shares of the pixels in 8 groups of 32 grey levels. Before: 51.6 % "
     "in 96 to 127 and 48.4 % in 128 to 159, 0 elsewhere. After: 11.7, 13.0, 12.2, 11.5, 11.9, 12.7, 13.7 "
     "and 13.3 % in the 8 groups.",
     "An image before and after equalisation", "Measured: the 2 occupied groups spread over all 8.", parts)


# 19.5  Measured compaction time, 3 passes against warp votes, for 3 kept fractions.
runs = [("10 % kept", 10.08, 2.27), ("50 % kept", 9.35, 2.49), ("90 % kept", 9.59, 2.57)]
parts, scale = [], 32
for r, (name, three, votes) in enumerate(runs):
    y = 112 + r * 96
    parts.append(mono(32, y + 20, name, "start", MUTED))
    parts.append(bar(150, y, three * scale, 26, "store"))
    parts.append(mono(156 + three * scale, y + 18, f"{three:.2f} ms  3 passes", "start"))
    parts.append(bar(150, y + 34, votes * scale, 26, "good"))
    parts.append(mono(156 + votes * scale, y + 52, f"{votes:.2f} ms  warp votes", "start"))
parts += [note(32, 418, "16,777,216 floats, average of 20 runs. At 50 % kept the ratio is 9.35 / 2.49 = 3.8,",
               room=596),
          note(32, 436, "the ratio of the bytes moved, 38 / 10 = 3.8: both versions run at the memory rate.",
               room=596)]
save(OUT + "ch19-measured.svg", 660, 458,
     "Bars of measured time for compacting 16,777,216 floats. 10 % kept: 3 passes 10.08 ms, warp votes 2.27 "
     "ms. 50 % kept: 9.35 ms and 2.49 ms. 90 % kept: 9.59 ms and 2.57 ms.",
     "Measured compaction time", "Warp votes take about 1/4 of the time of 3 passes.", parts)
