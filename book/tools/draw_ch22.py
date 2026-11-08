"""The figures of chapter 22, drawn with figkit and figparts.

    python3 tools/draw_ch22.py

Every figure uses the worked example of the chapter, the 8 keys of 3 bits 5, 2, 7, 2, 0, 6,
3, 5. The measured numbers in the last 2 figures are from the runs quoted in the chapter.
"""
from draw_ch21 import (STEP, W, X0, across, bar, cx, down, elbow, header, inclusive, marker,
                       merge, mono, notes, row, save, small, text)
from figparts import KINDS, MUTED, line

OUT = "src/figures/"
K = [5, 2, 7, 2, 0, 6, 3, 5]
SORTED = sorted(K)


def exclusive(values):
    return [s - v for s, v in zip(inclusive(values), values)]


def stable_split(keys, bit):
    return [k for k in keys if not (k >> bit) & 1] + [k for k in keys if (k >> bit) & 1]


def bits(keys, width=3):
    return [format(k, f"0{width}b") for k in keys]


def ex_keys():
    order = sorted(range(8), key=lambda i: K[i])
    parts = header(100)
    parts += row(112, "key", K, ["client"] * 8, subs=bits(K))
    parts += row(214, "sorted", SORTED, ["good"] * 8, subs=[f"was {i}" for i in order])
    parts += notes(290, ["8 keys of 3 bits, with their binary digits. The sorted order keeps the 2 keys 2",
                         "(from positions 1 and 3) and the 2 keys 5 (from 0 and 7) in their input order."])
    save(OUT + "ch22-keys.svg", 660, 334,
         "The keys 5, 2, 7, 2, 0, 6, 3, 5 with their binary digits 101, 010, 111, 010, 000, 110, 011, 101, and "
         "the sorted keys 0, 2, 2, 3, 5, 5, 6, 7, labeled with the positions they came from: 4, 1, 3, 6, 0, 7, "
         "5, 2.",
         "The worked example", "8 keys of 3 bits, before and after sorting.", parts)


def ex_split():
    b0 = [k & 1 for k in K]
    zero = [1 - b for b in b0]
    ez = exclusive(zero)
    eo = exclusive(b0)
    zeros = sum(zero)
    pos = [ez[i] if zero[i] else zeros + eo[i] for i in range(8)]
    out = [0] * 8
    for i, p in enumerate(pos):
        out[p] = K[i]
    zk = ["good" if z else "idle" for z in zero]
    ok = ["store" if b else "idle" for b in b0]
    parts = header(100)
    parts += row(112, "key", K, ["good" if z else "store" for z in zero], subs=bits(K))
    parts += row(172, "bit 0 is 0", zero, zk)
    parts += row(214, "exclusive", ez, zk)
    parts += row(268, "bit 0 is 1", b0, ok)
    parts += row(310, "4 + exclusive", [zeros + e for e in eo], ok)
    parts += row(364, "position", pos, ["good" if z else "store" for z in zero])
    parts += row(426, "output", out, ["good" if not k & 1 else "store" for k in out])
    parts += notes(488, ["Compaction 1 sends the 4 keys whose bit 0 is 0 to positions 0 to 3; compaction 2 sends",
                         "the other 4 to 4 + their exclusive scan, positions 4 to 7. Both keep the input order."])
    save(OUT + "ch22-split.svg", 660, 534,
         "A split of 5, 2, 7, 2, 0, 6, 3, 5 on bit 0 as 2 compactions. Flags for bit 0 equal to 0: 0, 1, 0, 1, 1, "
         "1, 0, 0, exclusive scan 0, 0, 1, 1, 2, 3, 4, 4. Flags for bit 0 equal to 1: 1, 0, 1, 0, 0, 0, 1, 1, with "
         "4 plus their exclusive scan 4, 5, 5, 6, 6, 6, 6, 7. Positions 4, 0, 5, 1, 2, 3, 6, 7; output 2, 2, 0, 6, "
         "5, 7, 3, 5.",
         "A split on bit 0", "2 compactions: the zero-bit keys first, the one-bit keys after them.", parts)


def ex_passes():
    passes = [K]
    for bit in range(3):
        passes.append(stable_split(passes[-1], bit))
    parts = header(96)
    for p, values in enumerate(passes):
        y = 106 + p * 74
        name = "input" if p == 0 else f"after bit {p - 1}"
        if p < 3:
            kinds = ["good" if not (k >> p) & 1 else "store" for k in values]
        else:
            kinds = ["client"] * 8
        parts += row(y, name, values, kinds, subs=bits(values))
    parts += [bar(X0, 418, 160, 24, "good", "next bit is 0"), bar(X0 + 176, 418, 160, 24, "store", "next bit is 1")]
    parts += notes(470, ["Each pass is a stable split on the next bit. After the pass on a bit, the keys are in",
                         "order of their bits up to that one; after bit 2 they are sorted: 0, 2, 2, 3, 5, 5, 6, 7."])
    save(OUT + "ch22-passes.svg", 660, 514,
         "Keys 5, 2, 7, 2, 0, 6, 3, 5. After a split on bit 0: 2, 2, 0, 6, 5, 7, 3, 5. After bit 1: 0, 5, 5, 2, 2, "
         "6, 7, 3. After bit 2: 0, 2, 2, 3, 5, 5, 6, 7, sorted. Binary digits are written under every key.",
         "A radix sort in 3 passes", "3 stable splits, least significant bit first, sort the 8 keys.", parts)


def ex_unstable():
    a0 = stable_split(K, 0)
    good1 = stable_split(a0, 1)
    good2 = stable_split(good1, 2)
    bad1 = [k for k in a0 if not (k >> 1) & 1] + [k for k in a0 if (k >> 1) & 1][::-1]
    bad2 = stable_split(bad1, 2)
    wrong = [i for i in range(8) if bad2[i] != SORTED[i]]
    parts = header(96)
    parts += row(106, "after bit 0", a0, ["client"] * 8, subs=bits(a0))
    parts += row(188, "bit 1, stable", good1, ["good"] * 8, subs=bits(good1))
    parts += row(250, "then bit 2", good2, ["good"] * 8)
    parts += row(332, "bit 1, unstable", bad1, ["good"] * 3 + ["danger"] * 5, subs=bits(bad1))
    parts += row(394, "then bit 2", bad2, ["danger" if i in wrong else "good" for i in range(8)])
    parts += [small(X0, 172, "one-bit keys 2, 2, 6, 7, 3 in their order", "start"),
              small(X0, 316, "one-bit keys in reverse order: 3, 7, 6, 2, 2", "start", KINDS["danger"][0])]
    parts += notes(458, ["Both splits put the keys with bit 1 = 0 before the others. The unstable one loses the",
                         "order by bit 0 inside each group, and the pass on bit 2 cannot restore it: 3 now comes",
                         "before 2, and 7 before 6."])
    save(OUT + "ch22-unstable.svg", 660, 520,
         "From 2, 2, 0, 6, 5, 7, 3, 5, a stable split on bit 1 gives 0, 5, 5, 2, 2, 6, 7, 3 and then 0, 2, 2, 3, 5, 5, "
         "6, 7 after bit 2. An unstable split that reverses the one-bit keys gives 0, 5, 5, 3, 7, 6, 2, 2 and then "
         "0, 3, 2, 2, 5, 5, 7, 6 after bit 2: positions 1, 3, 6 and 7 are wrong.",
         "An unstable pass breaks the sort", "A split that reorders equal bits destroys the earlier passes.",
         parts)


def ex_atomics():
    counts = [K.count(v) for v in range(8)]
    parts = header(100, "thread")
    parts += row(112, "key", K, ["client"] * 8)
    for v in range(8):
        threads = [i for i, k in enumerate(K) if k == v]
        kind = "danger" if len(threads) > 1 else "store"
        for j, i in enumerate(threads):
            parts.append(bar(cx(v) - 20, 222 - 30 * j, 40, 24, kind, f"t{i}"))
    parts += [mono(X0 - 14, 239, "requests", "end", MUTED)]
    parts += row(264, "bin", counts, ["danger" if c > 1 else ("store" if c else "idle") for c in counts],
                 subs=[f"bin {v}" for v in range(8)])
    parts += [small(cx(2), 180, "2nd waits", "middle", KINDS["danger"][0]),
              small(cx(5), 180, "2nd waits", "middle", KINDS["danger"][0])]
    parts += notes(336, ["Thread i adds 1 to the bin of its key with an atomic addition; the chips above each bin",
                         "are the requests it receives. Bins 2 and 5 receive 2 requests each, which the hardware",
                         "applies 1 after the other: the second request waits for the first."])
    save(OUT + "ch22-atomics.svg", 660, 396,
         "8 threads, 1 per key 5, 2, 7, 2, 0, 6, 3, 5, each sending an atomic addition to the bin of its key. "
         "Bin 2 receives the requests of threads 1 and 3, bin 5 those of threads 0 and 7, and each other "
         "occupied bin 1 request. The bins end with the counts 1, 0, 2, 1, 0, 2, 1, 1.",
         "A histogram with atomic additions", "8 keys counted into 8 bins; equal keys contend for 1 bin.", parts)


def ex_privatize():
    step, w = 30, 26
    parts = []
    blocks = [(K[:4], 60), (K[4:], 360)]
    totals = [0] * 8
    for b, (keys, x0) in enumerate(blocks):
        local = [keys.count(v) for v in range(8)]
        totals = [t + c for t, c in zip(totals, local)]
        parts.append(mono(x0 + 4 * step - 4, 100, f"block {b}: keys " + ", ".join(map(str, keys)), "middle", MUTED))
        parts += row(146, "", local, ["store" if c else "idle" for c in local], x0=x0, step=step, w=w)
        parts.append(mono(x0 + 4 * step - 4, 134, "private bins in shared memory", "middle", MUTED))
        for v in range(8):
            parts.append(small(x0 + v * step + w / 2, 190, str(v)))
    gx = 210
    parts += row(276, "", totals, ["good"] * 8, x0=gx, step=step, w=w)
    parts.append(mono(gx + 4 * step - 4, 320, "global bins 0 to 7", "middle", MUTED))
    for b, (_, x0) in enumerate(blocks):
        xs = [x0 + v * step + w / 2 for v in range(8)]
        parts += [line(xs[0], 206, xs[-1], 206, KINDS["store"][0], 1.4)]
        parts += [line(x, 196, x, 206, KINDS["store"][0], 1.0) for x in xs]
        mid = (xs[0] + xs[-1]) / 2
        parts.append(elbow(mid, 206, gx + 4 * step - 4 + (-40 if b == 0 else 40), 276, 238 + 8 * b, "store"))
    parts.append(marker("store"))
    parts += notes(360, ["Each block counts its keys into its own 8 bins in shared memory, where only its own",
                         "threads contend, and then adds each bin to the global bins once. For 16,777,216 keys in",
                         "1,024 blocks with 16 bins: 16,384 atomic additions on global memory, not 16,777,216."])
    save(OUT + "ch22-privatize.svg", 660, 420,
         "2 blocks of 4 keys. Block 0, keys 5, 2, 7, 2, counts into private bins 0, 0, 2, 0, 0, 1, 0, 1; block 1, "
         "keys 0, 6, 3, 5, into 1, 0, 0, 1, 0, 1, 1, 0. Both add their bins to the global bins, which end at 1, 0, "
         "2, 1, 0, 2, 1, 1.",
         "Privatized bins", "Count in shared memory, merge once per block.", parts)


def ex_digit_pass():
    d = [k & 3 for k in K]
    hist = [d.count(v) for v in range(4)]
    start = exclusive(hist)
    seen = [0] * 4
    rank, pos = [], []
    for v in d:
        rank.append(seen[v])
        pos.append(start[v] + seen[v])
        seen[v] += 1
    out = [0] * 8
    for i, p in enumerate(pos):
        out[p] = K[i]
    dk = ["client", "outside", "store", "service"]
    parts = header(100)
    parts += row(112, "key", K, [dk[v] for v in d], subs=bits(K))
    parts += row(172, "digit", d, [dk[v] for v in d])
    parts += row(238, "histogram", hist, dk, subs=[f"digit {v}" for v in range(4)])
    parts += row(298, "start", start, dk, solid=(0, 1, 2, 3))
    parts += row(364, "rank", rank, [dk[v] for v in d])
    parts += row(412, "position", pos, [dk[v] for v in d])
    parts += row(470, "output", out, [dk[k & 3] for k in out])
    parts += notes(530, ["The digit is the 2 low bits. The histogram of the digits, 1, 2, 3, 2, scans to the starts",
                         "0, 1, 3, 6. A key goes to the start of its digit plus its rank: the number of earlier",
                         "keys with the same digit. The output equals 2 passes of 1 bit."])
    save(OUT + "ch22-digit-pass.svg", 660, 594,
         "A pass on the 2-bit digit of 5, 2, 7, 2, 0, 6, 3, 5. Digits 1, 2, 3, 2, 0, 2, 3, 1. Histogram 1, 2, 3, 2; "
         "exclusive scan 0, 1, 3, 6. Ranks among equal digits 0, 0, 0, 1, 0, 2, 1, 1. Positions 1, 3, 6, 4, 0, 5, 7, "
         "2. Output 0, 5, 5, 2, 2, 6, 7, 3.",
         "A pass on a 2-bit digit", "A histogram and an exclusive scan give each digit its start.", parts)


def ex_tiles():
    tiles = [K[:4], K[4:]]
    dk = ["client", "outside", "store", "service"]
    counts = [[[k & 3 for k in t].count(v) for t in tiles] for v in range(4)]
    table = [counts[v][t] for v in range(4) for t in range(2)]
    starts = exclusive(table)
    subs = [f"d{v} t{t}" for v in range(4) for t in range(2)]
    kinds = [dk[v] for v in range(4) for _ in range(2)]
    parts = [mono(X0, 100, "tile 0: keys 5, 2, 7, 2, digits 1, 2, 3, 2", "start", MUTED),
             mono(X0, 120, "tile 1: keys 0, 6, 3, 5, digits 0, 2, 3, 1", "start", MUTED)]
    parts += row(146, "counts", table, kinds, subs=subs)
    parts += row(222, "exclusive", starts, kinds, solid=tuple(range(8)))
    for k in range(8):
        parts.append(down(cx(k), 190, 222))
    parts += notes(290, ["The counts are stored digit by digit: digit 0 of tile 0, digit 0 of tile 1, digit 1 of",
                         "tile 0, and so on. 1 exclusive scan of this table gives the start of every digit in every",
                         "tile: in tile 1, digit 1 starts at 2, after the keys 0 (tile 1) and 5 (tile 0)."])
    save(OUT + "ch22-tiles.svg", 660, 350,
         "The digit counts of 2 tiles stored digit by digit: 0, 1, 1, 1, 2, 1, 1, 1 for digit 0 of tiles 0 and 1, "
         "digit 1 of tiles 0 and 1, and so on. Their exclusive scan 0, 0, 1, 2, 3, 5, 6, 7 is the start of each "
         "digit in each tile.",
         "Digit starts for many tiles", "1 scan of the count table, digit-major, places every tile.", parts)


def ex_match():
    d = [k & 3 for k in K]
    peers = [1 if v == 2 else 0 for v in d]
    below = [1 if i < 5 else 0 for i in range(8)]
    both = [a & b for a, b in zip(peers, below)]
    parts = header(100, "lane")
    parts += row(112, "digit", d, ["store" if v == 2 else "idle" for v in d])
    parts += row(160, "match", peers, ["store" if p else "idle" for p in peers])
    parts += row(208, "lanes < 5", below, ["client" if b else "idle" for b in below])
    parts += row(256, "both", both, ["service" if b else "idle" for b in both])
    parts += [bar(X0, 318, 300, 30, "service", "lane 5: rank = popcount 2", solid=True),
              bar(X0, 360, 300, 30, "store", "digit 2 in the warp: popcount 3", solid=True)]
    parts += notes(432, ["match_any(digit) gives each lane the set of lanes with the same digit, here lanes 1, 3",
                         "and 5 for digit 2. Lane 5 counts the matching lanes below it: its rank is 2."])
    save(OUT + "ch22-match.svg", 660, 476,
         "8 lanes with digits 1, 2, 3, 2, 0, 2, 3, 1. For lane 5, whose digit is 2, match_any returns lanes 1, 3 and 5. "
         "The mask of lanes below 5 keeps lanes 1 and 3, so the rank of lane 5 among the lanes with digit 2 is 2, "
         "and the warp holds 3 keys with digit 2.",
         "Ranks among equal digits in a warp", "1 match and 1 popcount give each lane its rank.", parts)


def ex_hist_measured(rows):
    """rows: (keys, global ms, shared ms, aggregated ms)."""
    parts = []
    names = ["global bins", "shared bins", "shared, aggregated"]
    kinds = ["danger", "store", "good"]
    scale = 24
    for r, (name, *ms) in enumerate(rows):
        y0 = 100 + r * 150
        parts.append(mono(32, y0, f"keys: {name}", "start", MUTED))
        for k, t in enumerate(ms):
            y = y0 + 14 + k * 38
            parts.append(mono(196, y + 18, names[k], "end", MUTED))
            parts.append(bar(206, y, max(t * scale, 4), 26, kinds[k]))
            parts.append(mono(212 + max(t * scale, 4), y + 18, f"{t:.2f} ms", "start"))
    parts += notes(404, ["Histogram of the low 4 bits of 16,777,216 keys, 16 bins, average of 20 runs. With all",
                         "keys equal, every global atomic targets 1 address."])
    save(OUT + "ch22-hist-measured.svg", 700, 446,
         "Measured histogram times. " + "; ".join(
             f"{n} keys: global {a:.2f} ms, shared {b:.2f} ms, aggregated {c:.2f} ms" for n, a, b, c in rows) + ".",
         "Measured histograms", "Jetson Orin NX, C++ program.", parts)


def ex_sort_measured(rows):
    """rows: (name, ms)."""
    parts = []
    kinds = ["store", "good", "idle"]
    scale = 0.9
    for r, (name, ms) in enumerate(rows):
        y = 104 + r * 46
        parts.append(mono(196, y + 18, name, "end", MUTED))
        parts.append(bar(206, y, ms * scale, 28, kinds[r]))
        parts.append(mono(212 + ms * scale, y + 19, f"{ms:.1f} ms", "start"))
    parts += notes(260, ["16,777,216 random 32-bit keys, best of 3 runs. The CUB time is quoted from Chapter 11."])
    save(OUT + "ch22-sort-measured.svg", 700, 290,
         "Measured sort times for 16,777,216 keys: " + "; ".join(f"{n} {m:.1f} ms" for n, m in rows) + ".",
         "Measured radix sorts", "Jetson Orin NX, C++ program.", parts)


if __name__ == "__main__":
    ex_keys()
    ex_split()
    ex_passes()
    ex_unstable()
    ex_atomics()
    ex_privatize()
    ex_digit_pass()
    ex_tiles()
    ex_match()
    ex_hist_measured([("random", 3.23, 1.02, 1.91), ("all 7", 15.15, 1.01, 1.31)])
    ex_sort_measured([("32 passes of 1 bit", 262.26), ("8 passes of 4 bits", 40.13),
                      ("CUB, Chapter 11", 8.96)])
