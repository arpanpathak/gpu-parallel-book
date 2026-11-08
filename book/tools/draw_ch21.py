"""The figures of chapter 21, drawn with figkit and figparts.

    python3 tools/draw_ch21.py

Every figure uses the worked example of the chapter, the 8 values 3, 1, 7, 0, 4, 1, 6, 3.
The measured numbers in the last figure are from the runs quoted in the chapter. The helpers
at the top are shared with tools/draw_ch22.py.
"""
from figparts import KINDS, MUTED, WIRE, bar, line, mono, note, save, text

OUT = "src/figures/"
X = [3, 1, 7, 0, 4, 1, 6, 3]
STEP, W, H = 58, 48, 30
X0 = 170


def cx(k, x0=X0, step=STEP, w=W):
    """The centre of cell k of a row."""
    return x0 + k * step + w / 2


def small(x, y, body, anchor="middle", fill=MUTED, size=10.5):
    return (f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"'
            f' style="fill:{fill}" xml:space="preserve">{body}</text>')


def row(y, name, values, kinds, x0=X0, step=STEP, w=W, h=H, subs=None, solid=()):
    """A labelled row of cells. `kinds` gives the colour of each cell; None leaves a gap."""
    parts = [mono(x0 - 14, y + h / 2 + 4, name, "end", MUTED)] if name else []
    for k, (v, kind) in enumerate(zip(values, kinds)):
        if kind is None:
            continue
        parts.append(bar(x0 + k * step, y, w, h, kind, str(v), solid=k in solid))
        if subs and subs[k]:
            parts.append(small(x0 + k * step + w / 2, y + h + 13, subs[k]))
    return parts


def header(y, name="i", n=8, x0=X0, step=STEP, w=W):
    parts = [mono(x0 - 14, y, name, "end", MUTED)]
    return parts + [mono(cx(k, x0, step, w), y, str(k), "middle", MUTED) for k in range(n)]


def marker(kind):
    colour = KINDS[kind][0]
    return (f'  <defs><marker id="m-{kind}" viewBox="0 0 10 10" refX="8.5" refY="5"'
            f' markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto">'
            f'<path d="M0.5,1 L9,5 L0.5,9 Z" fill="{colour}"/></marker></defs>')


def elbow(x1, y1, x2, y2, mid, kind=None, width=1.4):
    """Down from (x1, y1) to `mid`, across to x2, down to y2, ending in an arrow."""
    colour = KINDS[kind][0] if kind else WIRE
    head = f"m-{kind}" if kind else "head"
    return (f'  <path d="M{x1:.1f},{y1:.1f} V{mid:.1f} H{x2:.1f} V{y2 - 3:.1f}" fill="none" '
            f'stroke="{colour}" stroke-width="{width}" stroke-linejoin="round" '
            f'marker-end="url(#{head})"/>')


def down(x, y1, y2, kind=None, width=1.4):
    colour = KINDS[kind][0] if kind else WIRE
    head = f"m-{kind}" if kind else "head"
    return (f'  <path d="M{x:.1f},{y1:.1f} V{y2 - 3:.1f}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" marker-end="url(#{head})"/>')


def across(x1, x2, y, kind=None, width=1.4):
    colour = KINDS[kind][0] if kind else WIRE
    head = f"m-{kind}" if kind else "head"
    end = x2 - 3 if x2 > x1 else x2 + 3
    return (f'  <path d="M{x1:.1f},{y:.1f} H{end:.1f}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" marker-end="url(#{head})"/>')


def merge(xs, y_top, y_bus, x_to, y_to, kind=None):
    """Lines down from several cells to a bus, and 1 arrow from the bus to the target."""
    colour = KINDS[kind][0] if kind else WIRE
    parts = [line(x, y_top, x, y_bus, colour) for x in xs]
    parts.append(line(min(xs + [x_to]), y_bus, max(xs + [x_to]), y_bus, colour))
    parts.append(down(x_to, y_bus, y_to, kind))
    return parts


def notes(y, lines, room=600):
    return [note(32, y + 18 * k, body, room=room) for k, body in enumerate(lines)]


def rng(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}..{hi}"


def inclusive(values):
    out, run = [], 0
    for v in values:
        run += v
        out.append(run)
    return out


def hillis_steele(values):
    """The rows of a Hillis-Steele scan: the input, then 1 row per offset."""
    rows, d = [list(values)], 1
    while d < len(values):
        a = rows[-1]
        rows.append([a[i] + a[i - d] if i >= d else a[i] for i in range(len(a))])
        d *= 2
    return rows


def ex_example():
    y_inc = inclusive(X)
    y_exc = [s - x for s, x in zip(y_inc, X)]
    parts = header(100)
    parts += row(112, "x", X, ["client" if k <= 4 else "idle" for k in range(8)])
    parts += row(166, "inclusive", y_inc, ["service"] * 8, solid=(4,))
    parts += row(220, "exclusive", y_exc, ["store"] * 8, solid=(4,))
    parts += notes(294, ["inclusive y4 = 3 + 1 + 7 + 0 + 4 = 15: the sum of x0 to x4.",
                         "exclusive y4 = 3 + 1 + 7 + 0 = 11: the sum of x0 to x3, without x4 itself."])
    save(OUT + "ch21-example.svg", 660, 340,
         "The 8 values 3, 1, 7, 0, 4, 1, 6, 3 with their inclusive scan 3, 4, 11, 11, 15, 16, 22, 25 and "
         "their exclusive scan 0, 3, 4, 11, 11, 15, 16, 22. Entry 4 is highlighted: 15 inclusive, the sum of "
         "x0 to x4, and 11 exclusive, the sum of x0 to x3.",
         "The worked example and its 2 scans", "Each output is the sum of the inputs up to its position.",
         parts)


def ex_chain():
    x0, step, w = 110, 68, 44
    y_inc = inclusive(X)
    parts = header(100, "i", x0=x0, step=step, w=w)
    parts += row(112, "x", X, ["client"] * 8, x0=x0, step=step, w=w)
    parts += row(206, "y", y_inc, ["service"] * 8, x0=x0, step=step, w=w)
    for k in range(8):
        parts.append(down(cx(k, x0, step, w), 142, 206))
    for k in range(1, 8):
        a, b = x0 + (k - 1) * step + w, x0 + k * step
        parts.append(across(a + 3, b, 221, "danger"))
        parts.append(small(cx(k, x0, step, w), 268, f"step {k}"))
    parts.append(marker("danger"))
    parts += notes(310, ["y[i] = y[i - 1] + x[i]: every addition (red) needs the result of the one before,",
                         "so the 7 additions form a chain of 7 steps, whatever the number of processors."])
    save(OUT + "ch21-chain.svg", 670, 356,
         "The sequential scan of the 8 values. Each x[i] flows down into y[i], and a red arrow carries y[i - 1] "
         "into y[i]. The outputs 3, 4, 11, 11, 15, 16, 22, 25 are produced 1 after the other in 7 steps.",
         "The dependence chain of the sequential scan", "Output i cannot start before output i - 1 is known.",
         parts)


def ex_regroup():
    x0, step, w = 170, 58, 48
    levels = [X, [4, 7, 5, 9], [11, 14], [25]]
    ranges = [[rng(k, k) for k in range(8)], ["0..1", "2..3", "4..5", "6..7"], ["0..3", "4..7"], ["0..7"]]
    kinds = ["client", "outside", "store", "good"]
    parts = []
    centres = [[cx(k, x0, step, w) for k in range(8)]]
    for lv in range(1, 4):
        prev = centres[-1]
        centres.append([(prev[2 * j] + prev[2 * j + 1]) / 2 for j in range(len(levels[lv]))])
    for lv, (vals, subs) in enumerate(zip(levels, ranges)):
        y = 100 + lv * 76
        for j, v in enumerate(vals):
            c = centres[lv][j]
            parts.append(bar(c - w / 2, y, w, 30, kinds[lv], str(v)))
            parts.append(small(c, y + 43, subs[j]))
        parts.append(mono(x0 - 14, y + 19, ["x", "pairs", "halves", "y7"][lv], "end", MUTED))
        if lv:
            for j in range(len(vals)):
                kids = centres[lv - 1][2 * j:2 * j + 2]
                parts += merge(kids, y - 76 + 48, y - 14, centres[lv][j], y, kinds[lv])
    parts += notes(426, ["y7 = (3 + 1) + (7 + 0) + (4 + 1) + (6 + 3) = 11 + 14 = 25. The same 7 additions,",
                         "grouped in pairs, need 3 steps instead of 7: the 4 pair sums are independent."])
    save(OUT + "ch21-regroup.svg", 660, 470,
         "The sum y7 = 25 of the 8 values computed as a tree. Pairs give 4, 7, 5, 9 for the ranges 0..1, 2..3, "
         "4..5, 6..7; halves give 11 for 0..3 and 14 for 4..7; the root gives 25 for 0..7. 3 levels of "
         "additions replace the chain of 7.",
         "Regrouping the sum", "Addition is associative, so the sums of 2 ranges combine into 1.", parts)


def ex_hillis():
    rows = hillis_steele(X)
    kinds = ["client", "outside", "store", "service"]
    names = ["x", "distance = 1", "distance = 2", "distance = 4"]
    parts = header(96)
    for r, vals in enumerate(rows):
        y = 106 + r * 86
        span = 2 ** r
        subs = [rng(max(0, i - span + 1), i) for i in range(8)]
        ks = [kinds[r] if (r == 0 or i >= 2 ** (r - 1)) else "idle" for i in range(8)]
        parts += row(y, names[r], vals, ks, subs=subs, solid=(7,) if r else ())
        if r:
            d = 2 ** (r - 1)
            top = y - 86 + 30 + 18
            parts.append(elbow(cx(7 - d), top, cx(7) - 6, y, y - 16, kinds[r]))
            parts.append(down(cx(7) + 6, top, y, kinds[r]))
    parts.append(marker("outside"))
    parts.append(marker("store"))
    parts.append(marker("service"))
    parts += notes(470, ["Element i adds element i - distance of the previous row (gray cells, i < distance,",
                         "are copied). The small labels give the range of x that each cell sums; the arrows",
                         "show element 7: 9 = 6 + 3, then 14 = 9 + 5, then 25 = 14 + 11."])
    save(OUT + "ch21-hillis.svg", 680, 536,
         "The Hillis-Steele scan of 3, 1, 7, 0, 4, 1, 6, 3. After offset 1: 3, 4, 8, 7, 4, 5, 7, 9. After "
         "offset 2: 3, 4, 11, 11, 12, 12, 11, 14. After offset 4: 3, 4, 11, 11, 15, 16, 22, 25. Under each "
         "cell, the range of inputs it sums doubles at every step. Arrows show element 7 adding elements 6, 5 "
         "and 3 of the rows above.",
         "The Hillis-Steele scan of the 8 values", "3 steps; at each step every element adds the one at that distance to its left.",
         parts)


def blelloch_rows():
    a = list(X)
    up = [list(a)]
    d = 1
    while d < 8:
        for i in range(2 * d - 1, 8, 2 * d):
            a[i] += a[i - d]
        up.append(list(a))
        d *= 2
    a[7] = 0
    dn = [list(a)]
    d = 4
    while d >= 1:
        for i in range(2 * d - 1, 8, 2 * d):
            a[i - d], a[i] = a[i], a[i] + a[i - d]
        dn.append(list(a))
        d //= 2
    return up, dn


def ex_upsweep():
    up, _ = blelloch_rows()
    kinds = ["client", "outside", "store", "service"]
    names = ["x", "distance = 1", "distance = 2", "distance = 4"]
    subs_by_level = [[rng(i, i) for i in range(8)]]
    for lv in range(1, 4):
        d = 2 ** (lv - 1)
        prev = list(subs_by_level[-1])
        for i in range(2 * d - 1, 8, 2 * d):
            prev[i] = rng(i - 2 * d + 1, i)
        subs_by_level.append(prev)
    parts = header(96)
    for lv, vals in enumerate(up):
        y = 106 + lv * 86
        d = 2 ** (lv - 1) if lv else 0
        changed = set(range(2 * d - 1, 8, 2 * d)) if lv else set(range(8))
        ks = [kinds[lv] if i in changed else "idle" for i in range(8)]
        parts += row(y, names[lv], vals, ks, subs=subs_by_level[lv])
        if lv:
            top = y - 86 + 30 + 18
            for i in sorted(changed):
                parts.append(elbow(cx(i - d), top, cx(i) - 6, y, y - 16, kinds[lv]))
                parts.append(down(cx(i) + 6, top, y, kinds[lv]))
    parts += [marker("outside"), marker("store"), marker("service")]
    parts += notes(470, ["Element i = 2 · distance - 1, 4 · distance - 1, ... adds element i - distance.",
                         "4 + 2 + 1 = 7 additions build a tree of partial sums; element 7 ends with 25."])
    save(OUT + "ch21-upsweep.svg", 680, 516,
         "The up-sweep of the Blelloch scan on 3, 1, 7, 0, 4, 1, 6, 3. Distance 1: elements 1, 3, 5, 7 become 4, "
         "7, 5, 9. Distance 2: elements 3 and 7 become 11 and 14. Distance 4: element 7 becomes 25. The small "
         "labels give the range each cell sums.",
         "The up-sweep", "7 additions leave the sums of ranges of length 1, 2, 4 and 8 in place.", parts)


def ex_downsweep():
    _, dn = blelloch_rows()
    kinds = ["danger", "service", "store", "outside"]
    names = ["clear 7", "distance = 4", "distance = 2", "distance = 1"]
    excl = [s - x for s, x in zip(inclusive(X), X)]
    parts = header(96)
    for lv, vals in enumerate(dn):
        y = 106 + lv * 86
        if lv == 0:
            ks = ["idle"] * 7 + ["danger"]
        else:
            d = 2 ** (3 - lv)
            touched = set()
            for i in range(2 * d - 1, 8, 2 * d):
                touched |= {i, i - d}
            ks = [kinds[lv] if i in touched else "idle" for i in range(8)]
            top = y - 86 + 30 + 4
            for i in range(2 * d - 1, 8, 2 * d):
                parts.append(elbow(cx(i - d) - 6, top, cx(i) - 6, y, y - 30, kinds[lv]))
                parts.append(elbow(cx(i) + 6, top, cx(i - d) + 6, y, y - 16, kinds[lv]))
        parts += row(y, names[lv], vals, ks)
    parts += [marker("service"), marker("store"), marker("outside")]
    parts += notes(458, ["Each pair (i - distance, i) swaps and adds: the left element receives the right",
                         "one's value, and the right one adds the old left value. The last row,",
                         ", ".join(map(str, excl)) + ", is the exclusive scan of the 8 values."])
    save(OUT + "ch21-downsweep.svg", 680, 520,
         "The down-sweep of the Blelloch scan. Element 7 is set to 0. Distance 4: elements 3 and 7 become 0 and "
         "11. Distance 2: elements 1, 3, 5, 7 become 0, 4, 11, 16. Distance 1: all 8 become 0, 3, 4, 11, 11, "
         "15, 16, 22, the exclusive scan.",
         "The down-sweep", "7 more additions push the left totals down the same tree.", parts)


def ex_work():
    rows_ = [("sequential", 7, 7, "idle"), ("Hillis-Steele", 17, 3, "client"), ("Blelloch", 14, 6, "store")]
    parts = [mono(32, 104, "additions for n = 8", "start", MUTED), mono(360, 104, "steps for n = 8", "start", MUTED)]
    for r, (name, adds, steps, kind) in enumerate(rows_):
        y = 120 + r * 44
        parts.append(mono(150, y + 19, name, "end", MUTED))
        parts.append(bar(160, y, adds * 9, 28, kind))
        parts.append(mono(166 + adds * 9, y + 19, str(adds), "start"))
        parts.append(bar(380, y, steps * 24, 28, kind))
        parts.append(mono(386 + steps * 24, y + 19, str(steps), "start"))
    big = [("sequential", "n - 1", "1,023", "n - 1", "1,023"),
           ("Hillis-Steele", "n log n - (n - 1)", "9,217", "log n", "10"),
           ("Blelloch", "2 (n - 1)", "2,046", "2 log n", "20")]
    y0 = 290
    parts += [mono(32, 266, "in general, and for n = 1,024", "start", MUTED)]
    for r, (name, f_add, n_add, f_step, n_step) in enumerate(big):
        y = y0 + 24 + r * 22
        parts += [mono(150, y, name, "end", MUTED), mono(166, y, f"{f_add} = {n_add}", "start"),
                  mono(396, y, f"{f_step} = {n_step}", "start")]
    parts += [mono(166, y0 + 2, "additions", "start", MUTED), mono(396, y0 + 2, "steps", "start", MUTED)]
    parts += notes(398, ["Hillis-Steele takes the fewest steps but does about log n times the work of the chain.",
                         "Blelloch does twice the work of the chain in twice the steps of Hillis-Steele."])
    save(OUT + "ch21-work.svg", 660, 436,
         "Additions and steps of the 3 scans. For n = 8: sequential 7 additions in 7 steps, Hillis-Steele 17 in "
         "3, Blelloch 14 in 6. For n = 1,024: 1,023 in 1,023, 9,217 in 10, and 2,046 in 20.",
         "Work and steps of 3 scans", "Fewer steps cost more additions, unless the algorithm is work-efficient.",
         parts)


def timeline(y, name, pieces, x0=170, scale=1.0):
    """1 thread's row: (start, length, kind, label) pieces along the time axis."""
    parts = [mono(x0 - 14, y + 19, name, "end", MUTED)]
    for start, length, kind, label in pieces:
        parts.append(bar(x0 + start * scale, y, length * scale, 28, kind, label))
    return parts


def ex_race_barrier():
    red = KINDS["danger"][0]
    parts = [mono(170, 100, "step with distance 1", "start", MUTED),
             mono(470, 100, "step with distance 2", "start", MUTED)]
    parts += timeline(112, "thread 2", [(0, 260, "client", "next[2] = current[2] + current[1] = 8"),
                                        (300, 270, "client", "current[2] = next[2] + next[0]")])
    parts += timeline(176, "thread 0", [(0, 70, "idle", "late"),
                                        (360, 210, "store", "next[0] = current[0] = 3")])
    parts += [line(460, 92, 460, 216, MUTED, 1.2, "4 4"),
              small(454, 230, "barrier removed", "end"),
              f'  <circle cx="480" cy="140" r="4" fill="{red}"/>',
              line(480, 140, 480, 216, red, 1.6, "3 3"),
              small(488, 230, "read of next[0]", "start", red),
              f'  <circle cx="530" cy="190" r="4" fill="{red}"/>',
              text(450, 266, "thread 2 reads next[0] before thread 0 has written it, and finds the 0", "note",
                   "middle", red),
              text(450, 286, "left in the slot: current[2] = 8 + 0 = 8, and y2 = 8 in place of 11", "note",
                   "middle", red)]
    parts += notes(324, ["With the barrier, no thread starts the step with distance 2 until every thread has",
                         "written its slot of next in the step with distance 1, and thread 2 reads next[0] = 3:",
                         "current[2] = 8 + 3 = 11. The names are those of the step with distance 1."])
    save(OUT + "ch21-race-barrier.svg", 820, 390,
         "A timeline of threads 0 and 2 without the barrier at the end of a step, with the buffer names of "
         "step 1. Thread 2 finishes step 1 with next[2] = 8 and starts step 2, where it reads next[0]. Thread 0 "
         "is late and writes next[0] = 3 only afterward. Thread 2 reads the 0 left in the slot, and y2 becomes 8 "
         "instead of 11.",
         "Without the barrier: a read before the write", "Thread 2 runs ahead into the next step.", parts)


def ex_race_inplace():
    red = KINDS["danger"][0]
    parts = [mono(170, 100, "step with distance 1, 1 buffer current", "start", MUTED)]
    parts += timeline(112, "thread 1", [(0, 300, "store", "current[1] = current[1] + current[0] = 4")])
    parts += timeline(176, "thread 2", [(0, 150, "idle", "late"),
                                        (320, 290, "client", "current[2] = current[2] + current[1]")])
    parts += [f'  <circle cx="470" cy="126" r="4" fill="{red}"/>',
              line(470, 126, 470, 216, red, 1.6, "3 3"),
              small(466, 230, "current[1] = 4 written", "end", red),
              f'  <circle cx="500" cy="190" r="4" fill="{red}"/>',
              small(504, 230, "current[1] read", "start", red),
              text(470, 264, "thread 2 reads current[1] after thread 1 has already added current[0] to it:",
                   "note", "middle", red),
              text(470, 284, "current[2] = 7 + 4 = 11 in place of 8, so x0 = 3 is counted twice; after the",
                   "note", "middle", red),
              text(470, 304, "step with distance 2, current[2] = 11 + 3 = 14, and y2 = 14 in place of 11",
                   "note", "middle", red)]
    parts += notes(340, ["With 2 buffers, the step with distance 1 writes only next and reads only current,",
                         "so thread 2 reads current[1] = 1 whether thread 1 has finished or not."])
    save(OUT + "ch21-race-inplace.svg", 820, 400,
         "A timeline of threads 1 and 2 when the scan updates 1 buffer in place. Thread 1 adds current[0] to "
         "current[1], making it 4, before the late thread 2 reads current[1]. Thread 2 computes 7 + 4 = 11 "
         "instead of 8, and after the next step y2 is 14 instead of 11.",
         "In place: a read after the write", "Thread 2 reads a value of the current step, not of the last one.",
         parts)


def ex_invariant():
    parts = header(96)
    parts += row(106, "x", X, ["client" if k <= 6 else "idle" for k in range(8)])
    windows = [(6, 6, 6, "distance 1", "6"), (5, 6, 7, "distance 2", "1 + 6 = 7"),
               (3, 6, 11, "distance 4", "0 + 4 + 1 + 6 = 11"),
               (0, 6, 22, "end", "3 + 1 + 7 + 0 + 4 + 1 + 6 = 22")]
    for r, (lo, hi, _, name, label) in enumerate(windows):
        y = 168 + r * 44
        left = X0 + lo * STEP
        width = (hi - lo) * STEP + W
        parts.append(mono(X0 - 14, y + 17, name, "end", MUTED))
        parts.append(bar(left, y, width, 24, "service", label))
    parts += notes(362, ["Before each step, current[6] holds the sum of the last distance inputs that end",
                         "at position 6 (fewer at the left edge). Each step doubles the window; once",
                         "distance > 6, it reaches x0, and current[6] = y6 = 22."])
    save(OUT + "ch21-invariant.svg", 660, 420,
         "The window of inputs summed by current[6] in the Hillis-Steele scan. At the start of offset 1 it is x6 alone, "
         "6; at offset 2 it is x5 and x6, 7; at offset 4 it is x3 to x6, 11; at the end it is x0 to x6, 22.",
         "The invariant for thread 6", "The window of inputs that current[6] sums doubles at every step.", parts)


def ex_blocks():
    step, w = 58, 48
    parts = []
    left, right = 120, 400
    for b, (x0, vals) in enumerate(((left, X[:4]), (right, X[4:]))):
        sc = inclusive(vals)
        parts.append(mono(x0 + 2 * step - 5, 100, f"tile {b}", "middle", MUTED))
        parts += row(112, "", vals, ["client"] * 4, x0=x0)
        parts += row(176, "", sc, ["outside"] * 4, x0=x0, solid=(3,))
        for k in range(4):
            parts.append(down(x0 + k * step + w / 2, 142, 176))
    parts += [mono(100, 131, "x", "end", MUTED), mono(100, 195, "scan", "end", MUTED),
              mono(100, 271, "totals", "end", MUTED), mono(100, 347, "result", "end", MUTED)]
    parts += row(252, "", [11, 25], ["store", "store"], x0=262)
    parts += [elbow(left + 3 * step + w / 2, 206, 262 + w / 2, 252, 226, "store"),
              elbow(right + 3 * step + w / 2, 206, 262 + step + w / 2, 252, 234, "store"),
              small(262 + step + w + 12, 271, "scan of 11, 14", "start")]
    parts += row(328, "", inclusive(X[:4]), ["good"] * 4, x0=left)
    parts += row(328, "", inclusive(X)[4:], ["good"] * 4, x0=right)
    parts += [elbow(262 + w / 2, 282, right + 1.5 * step + w / 2, 316, 298, "danger"),
              line(right, 316, right + 3 * step + w, 316, KINDS["danger"][0], 1.6),
              line(right, 316, right, 322, KINDS["danger"][0], 1.6),
              line(right + 3 * step + w, 316, right + 3 * step + w, 322, KINDS["danger"][0], 1.6),
              small(330, 292, "+ 11 to every element of tile 1", "start", KINDS["danger"][0])]
    parts.append(marker("store"))
    parts.append(marker("danger"))
    parts += notes(392, ["Step 1 scans each tile on its own. Step 2 scans the tile totals 11 and 14 into 11 and 25.",
                         "Step 3 adds to every element of tile 1 the total of the tiles before it, 11."])
    save(OUT + "ch21-blocks.svg", 660, 436,
         "The 8 values as 2 tiles of 4. Tile 0, 3, 1, 7, 0, scans to 3, 4, 11, 11 with total 11; tile 1, 4, 1, 6, 3, "
         "scans to 4, 5, 11, 14 with total 14. The totals scan to 11 and 25, and adding 11 to tile 1 gives 15, 16, "
         "22, 25: the whole scan is 3, 4, 11, 11, 15, 16, 22, 25.",
         "A scan of 2 tiles", "Scan the tiles, scan their totals, add each tile's offset.", parts)


def ex_compact():
    flags = [int(v >= 3) for v in X]
    excl = [s - f for s, f in zip(inclusive(flags), flags)]
    kept = [v for v in X if v >= 3]
    kinds = ["good" if f else "idle" for f in flags]
    parts = header(100)
    parts += row(112, "x", X, kinds)
    parts += row(160, "flag", flags, kinds)
    parts += row(208, "exclusive", excl, ["store" if f else "idle" for f in flags])
    parts += row(352, "output", kept, ["good"] * 5)
    order = [k for k in range(8) if flags[k]]
    for j, k in enumerate(order):
        parts.append(elbow(cx(k), 238, cx(j), 352, 262 + 16 * j, "good"))
    parts.append(marker("good"))
    parts += notes(410, ["The flag is 1 for x >= 3. The exclusive scan of the flags counts the kept values before",
                         "each value, so the 5 kept values go to positions 0 to 4, in their input order."])
    save(OUT + "ch21-compact.svg", 660, 456,
         "Stream compaction of 3, 1, 7, 0, 4, 1, 6, 3, keeping x >= 3. Flags 1, 0, 1, 0, 1, 0, 1, 1; exclusive scan "
         "0, 1, 1, 2, 2, 3, 3, 4. Arrows take the kept values 3, 7, 4, 6, 3 from positions 0, 2, 4, 6, 7 to output "
         "positions 0, 1, 2, 3, 4.",
         "Stream compaction of the 8 values", "A flag per value, a scan of the flags, a write at the scanned position.",
         parts)


def ex_ballot():
    flags = [int(v >= 3) for v in X]
    below = [1 if k < 4 else 0 for k in range(8)]
    both = [a & b for a, b in zip(flags, below)]
    parts = header(100, "lane")
    parts += row(112, "x", X, ["client"] * 8)
    parts += row(160, "ballot", flags, ["good" if f else "idle" for f in flags])
    parts += row(208, "lanes < 4", below, ["store" if b else "idle" for b in below])
    parts += row(256, "both", both, ["service" if b else "idle" for b in both])
    parts += [bar(X0, 318, 300, 30, "service", "lane 4: popcount 2 = position 2", solid=True),
              bar(X0, 360, 300, 30, "good", "count: popcount of ballot = 5", solid=True)]
    parts += notes(432, ["ballot(x >= 3) gives every lane the same word, 1 bit per lane, drawn here for 8 lanes.",
                         "Lane 4 keeps the bits of lanes 0 to 3 and counts them: its exclusive scan is 2."])
    save(OUT + "ch21-ballot.svg", 660, 476,
         "8 lanes holding 3, 1, 7, 0, 4, 1, 6, 3. The ballot of x >= 3 is 1, 0, 1, 0, 1, 0, 1, 1. The mask of the "
         "lanes below lane 4 is 1, 1, 1, 1, 0, 0, 0, 0; their combination 1, 0, 1, 0, 0, 0, 0, 0 has 2 bits, so "
         "lane 4 writes at position 2. The ballot has 5 bits: 5 values are kept.",
         "Positions from a warp vote", "1 ballot gives every lane its exclusive scan and the count.", parts)


def ex_bytes():
    stages = [
        ("3 passes", [("flags", 8, "client"), ("scan", 16, "service"), ("scatter", 14, "store")]),
        ("warp votes", [("count", 4, "client"), ("write", 6, "store")]),
        ("minimum", [("read + write kept", 6, "good")]),
    ]
    parts, scale, x0 = [], 11, 130
    for r, (name, pieces) in enumerate(stages):
        y = 112 + r * 80
        parts.append(mono(x0 - 14, y + 20, name, "end", MUTED))
        at = x0
        for label, size, kind in pieces:
            wdt = size * scale
            parts.append(bar(at, y, wdt, 30, kind, f"{size} B"))
            at += wdt
        total = sum(size for _, size, _ in pieces)
        parts.append(mono(at + 10, y + 20, f"= {total} B", "start"))
        names = " + ".join(f"{label} {size} B" for label, size, _ in pieces)
        parts.append(small(x0, y + 48, names, "start"))
    parts += notes(370, ["Bytes moved per input value of 4 bytes, with half of the values kept. The 3 passes",
                         "write the flags and the scan to memory and read them back; the votes never store them."])
    save(OUT + "ch21-bytes.svg", 660, 414,
         "Bars of bytes moved per input value when half the values are kept. 3 passes: flags 8 bytes, scan 16 "
         "bytes, scatter 14 bytes, total 38 bytes. Warp votes: count 4 bytes, write 6 bytes, total 10 bytes. "
         "The minimum, reading the input and writing the kept half, is 6 bytes.",
         "Bytes moved per value", "3 passes move 38 bytes per value, warp votes 10, the minimum is 6.", parts)


def ex_measured(scans, compaction):
    """scans: (name, tiles ms, whole ms); compaction: (kept, 3-pass ms, votes ms)."""
    parts = [mono(32, 100, "scan of 16,777,216 values: 1 level (tiles) and whole", "start", MUTED)]
    scale = 52
    for r, (name, tiles, whole) in enumerate(scans):
        y = 114 + r * 62
        parts.append(mono(170, y + 18, name, "end", MUTED))
        parts.append(bar(180, y, tiles * scale, 24, "client"))
        parts.append(mono(186 + tiles * scale, y + 17, f"{tiles:.2f} ms", "start"))
        parts.append(bar(180, y + 28, whole * scale, 24, "store"))
        parts.append(mono(186 + whole * scale, y + 45, f"{whole:.2f} ms", "start"))
    y0 = 114 + 3 * 62 + 30
    parts.append(mono(32, y0, "compaction of 16,777,216 values, 50 % kept", "start", MUTED))
    kept, three, votes = compaction
    parts.append(mono(170, y0 + 30, "3 passes", "end", MUTED))
    parts.append(bar(180, y0 + 12, three * scale, 26, "store"))
    parts.append(mono(186 + three * scale, y0 + 30, f"{three:.2f} ms", "start"))
    parts.append(mono(170, y0 + 64, "warp votes", "end", MUTED))
    parts.append(bar(180, y0 + 46, votes * scale, 26, "good"))
    parts.append(mono(186 + votes * scale, y0 + 64, f"{votes:.2f} ms", "start"))
    parts.append(f'  <rect x="180" y="{y0 + 92}" width="14" height="14" rx="4" fill="{KINDS["client"][1]}"'
                 f' stroke="{KINDS["client"][0]}"/>')
    parts.append(note(200, y0 + 104, "tiles only"))
    parts.append(f'  <rect x="290" y="{y0 + 92}" width="14" height="14" rx="4" fill="{KINDS["store"][1]}"'
                 f' stroke="{KINDS["store"][0]}"/>')
    parts.append(note(310, y0 + 104, "whole scan, or 3 passes"))
    save(OUT + "ch21-measured.svg", 700, y0 + 130,
         "Measured times. Scans of 16,777,216 values, tiles only and whole: " +
         "; ".join(f"{n} {t:.2f} ms and {w:.2f} ms" for n, t, w in scans) +
         f". Compaction with 50 % kept: 3 passes {three:.2f} ms, warp votes {votes:.2f} ms.",
         "Measured scans and compaction", "Jetson Orin NX, C++ program, average of 20 runs.", parts)


if __name__ == "__main__":
    ex_example()
    ex_chain()
    ex_regroup()
    ex_hillis()
    ex_upsweep()
    ex_downsweep()
    ex_work()
    ex_race_barrier()
    ex_race_inplace()
    ex_invariant()
    ex_blocks()
    ex_compact()
    ex_ballot()
    ex_bytes()
    ex_measured([("Hillis-Steele", 5.540, 7.118), ("Blelloch", 4.722, 6.328), ("warp shuffles", 1.750, 3.404)],
                (8387729, 8.05, 2.41))
