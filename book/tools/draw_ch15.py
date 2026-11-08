"""The figures of chapter 15 (Reduction and the Dot Product), in the style of the author's
paper, with the drawing helpers of draw_ch14.py.

    python3 tools/draw_ch15.py

The measured rates in the last figure are those quoted in the chapter (Jetson Orin NX,
code/oxide/ch15_dot_product and code/cuda/ch15_dot_product).
"""
from draw_ch14 import (EDGE, GREEN, GREY, HI, INK, PAL, RED, THREADS, T, arrow, bars, brace,
                       cell, curve, line, rect, save)

A = [1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 2, 3, 1, 3, 2, 1]
B = [2, 3, 1, 4, 1, 1, 4, 1, 1, 2, 2, 2, 1, 3, 1, 4]
P = [a * b for a, b in zip(A, B)]
BLOCK_COL = ["blue", "mint"]


def tree_levels(values):
    """The cells of the shared-memory tree after each step s = n/2, ..., 1."""
    cells = list(values)
    levels = [list(cells)]
    s = len(cells) // 2
    while s >= 1:
        for t in range(s):
            cells[t] += cells[t + s]
        levels.append(list(cells))
        s //= 2
    return levels


def example():
    cw, ch, pitch = 30, 26, 34
    x0, gap, w = 64, 12, 640
    parts = []

    def cx(i):
        return x0 + i * pitch + (gap if i >= 8 else 0)

    for i in range(16):
        parts.append(T(cx(i) + cw / 2, 22, i, 10.5, GREY))
    rows = [("a", A, 30, None), ("b", B, 64, None), ("a·b", P, 112, True)]
    for name, values, y, coloured in rows:
        parts.append(T(x0 - 12, y + 18, name, 12.5, INK, "end", 600, italic=True))
        for i, v in enumerate(values):
            col = BLOCK_COL[i // 8] if coloured else "white"
            parts.append(cell(cx(i), y, cw, ch, v, col, 12))
    parts.append(line(x0, 100, cx(15) + cw, 100, EDGE, 1))
    parts += [brace(cx(0), cx(7) + cw, 142, "products 0 to 7 add up to 32"),
              brace(cx(8), cx(15) + cw, 142, "products 8 to 15 add up to 31"),
              T(x0 + 8 * pitch + gap / 2 - 2, 196, "a · b = 32 + 31 = 63", 13, HI, weight=700)]
    save("ch15-example.svg", w, 210,
         "The worked example: vectors a = 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 2, 3, 1, 3, 2, 1 and "
         "b = 2, 3, 1, 4, 1, 1, 4, 1, 1, 2, 2, 2, 1, 3, 1, 4. Their products are 2, 6, 3, 4, 5, 1, "
         "8, 3 (sum 32) and 3, 2, 4, 6, 1, 9, 2, 4 (sum 31). The dot product is 63.", parts)


def chain():
    cw, ch, pitch = 30, 26, 34
    x0, gap, w = 64, 12, 640
    parts = []

    def cx(i):
        return x0 + i * pitch + (gap if i >= 8 else 0)

    s, sums = 0, []
    for p in P:
        s += p
        sums.append(s)
    parts += [T(x0 - 12, 42, "a·b", 12, INK, "end", 600, italic=True),
              T(x0 - 12, 108, "sum", 12, INK, "end", 600, italic=True)]
    for i in range(16):
        parts += [cell(cx(i), 24, cw, ch, P[i], "white", 12),
                  arrow(cx(i) + cw / 2, 51, cx(i) + cw / 2, 88, "grey", 1.1),
                  cell(cx(i), 90, cw, ch, sums[i], "red", 12),
                  T(cx(i) + cw / 2, 136, i + 1, 10.5, GREY)]
        nxt = cx(i + 1) if i < 15 else None
        if nxt is not None:
            parts.append(line(cx(i) + cw, 103, nxt, 103, RED, 1.6))
    parts += [T(x0 - 12, 136, "step", 10.5, GREY, "end"),
              T(x0, 166, "Line 3 runs 16 times. Each addition needs the sum of the one before,",
                12, INK, "start"),
              T(x0, 184, "so the 16 additions form a chain of 16 steps, whatever the hardware.",
                12, INK, "start")]
    save("ch15-chain.svg", w, 200,
         "The sequential sum of the 16 products. The running sums are 2, 8, 11, 15, 20, 21, 29, "
         "32, 35, 37, 41, 47, 48, 57, 59, 63, each joined to the next: 16 additions in a chain "
         "of 16 steps.", parts)


def regroup():
    w = 640
    cw, ch = 26, 24
    x0 = 20
    span = 470 / 16
    parts = []
    level = list(P)
    y = 20
    ys = []
    xs = [x0 + (i + 0.5) * span for i in range(16)]
    for depth in range(5):
        n = len(level)
        for k, v in enumerate(level):
            cx = xs[k] if depth == 0 else None
            if depth > 0:
                width = 16 // n
                cx = (xs[k * width] + xs[k * width + width - 1]) / 2
            col = BLOCK_COL[0] if (k * (16 // n)) < 8 else BLOCK_COL[1]
            if depth == 4:
                col = "hi"
            parts.append(cell(cx - cw / 2, y, cw, ch, v, col, 11.5, outline=depth == 4))
            if depth > 0:
                width = 16 // n
                left = (xs[k * width] + xs[k * width + width // 2 - 1]) / 2
                right = (xs[k * width + width // 2] + xs[k * width + width - 1]) / 2
                d = f"M{left:.1f},{y - 30 + ch} V{y - 9} H{right:.1f} V{y - 30 + ch}"
                parts.append(f'  <path d="{d}" fill="none" stroke="{GREY}" stroke-width="1.1"/>')
                parts.append(line(cx, y - 9, cx, y, GREY, 1.1))
        ys.append(y)
        if depth < 4:
            level = [level[2 * k] + level[2 * k + 1] for k in range(n // 2)]
        y += 54
    labels = ["16 products", "step 1: 8 additions", "step 2: 4 additions",
              "step 3: 2 additions", "step 4: 1 addition"]
    for yy, lab in zip(ys[1:], labels[1:]):
        parts.append(T(506, yy + 16, lab, 10.5, GREY, "start"))
    parts.append(T(x0, ys[-1] + ch + 30, "The same 15 additions as the chain, regrouped: 4 steps "
                   "instead of 16.", 12, INK, "start"))
    save("ch15-regroup.svg", w, ys[-1] + ch + 44,
         "The 16 products regrouped in pairs. Step 1 makes 8, 7, 6, 11, 5, 10, 10, 6; step 2 "
         "makes 15, 17, 15, 16; step 3 makes 32 and 31; step 4 makes 63. 15 additions in 4 "
         "steps.", parts)


def plus_badge(x, y, col):
    """A small round "+" badge: the place where 2 values are added."""
    accent = PAL[col][1]
    return "\n".join([
        f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#FFFFFF" stroke="{accent}" '
        f'stroke-width="1.6" filter="url(#elev)"/>',
        T(x, y + 4.2, "+", 12, accent, weight=700)])


def steps():
    w = 660
    cw, ch, pitch = 44, 32, 54
    x0 = 170
    gap = 104
    levels = tree_levels(P[:8])
    parts = [T(20, 24, "Shared memory of block 0, cells 0 to 7", 13, INK, "start", 700)]
    for k in range(8):
        parts.append(T(x0 + k * pitch + cw / 2, 44, f"cell {k}", 10, GREY))
    labels = ["after line 9", "step with stride = 4", "step with stride = 2", "step with stride = 1"]
    actives = [None, "threads 0 to 3 add", "threads 0 and 1 add", "thread 0 adds"]
    y = 52
    for depth, cells in enumerate(levels):
        live = 8 if depth == 0 else 8 >> depth
        for k, v in enumerate(cells):
            if depth == 0:
                col, faded = "white", False
            elif k < live:
                col, faded = THREADS[k % 4], False
            else:
                col, faded = "grey", True
            parts.append(cell(x0 + k * pitch, y, cw, ch, v, col, 14, faded=faded,
                              outline=(depth == 3 and k == 0), weight=600))
        parts.append(T(x0 - 18, y + 15, labels[depth], 12, INK, "end", 700))
        if actives[depth]:
            parts.append(T(x0 - 18, y + 30, actives[depth], 10.5, GREY, "end"))
        if depth == 3:
            y += ch
            break
        ny = y + gap
        s = 4 >> depth
        bar_y = y + ch + 16
        parts.append(line(x0 - 8, bar_y, x0 + 7 * pitch + cw + 8, bar_y, "#5C6B7E", 1.6, "6 4"))
        parts.append(T(x0 + 7 * pitch + cw + 14, bar_y + 4, "barrier", 11, "#5C6B7E", "start", 600))
        for t in range(s):
            col = THREADS[t % 4]
            xt = x0 + t * pitch + cw / 2
            xs = x0 + (t + s) * pitch + cw / 2
            by = ny - 20
            parts.append(curve(f"M{xt:.1f},{y + ch + 3} L{xt:.1f},{by - 10}", col, 2.2))
            parts.append(curve(f"M{xs:.1f},{y + ch + 3} C{xs:.1f},{y + ch + 44} "
                               f"{xt + 34:.1f},{by - 30} {xt + 8:.1f},{by - 7}", col, 2.2))
            parts.append(curve(f"M{xt:.1f},{by + 8} L{xt:.1f},{ny - 2}", col, 2.2))
            parts.append(plus_badge(xt, by, col))
        y = ny
    parts.append(T(x0, y + 30, "Each new value is the sum of 2 cells: its own cell and the cell "
                   "stride places to the right.", 11.5, GREY, "start"))
    parts.append(T(x0, y + 47, "Gray cells are no longer read. After step with stride = 1, cell 0 holds "
                   "32.", 11.5, GREY, "start"))
    save("ch15-steps.svg", w, y + 62,
         "The tree in the shared memory of block 0. The 8 cells start as 2, 6, 3, 4, 5, 1, 8, 3. "
         "In step with stride = 4, threads 0 to 3 each add 2 cells, its own cell and cell threadIndex + 4, into its own cell, "
         "giving 2 + 5 = 7, 6 + 1 = 7, 3 + 8 = 11 and 4 + 3 = 7. In step with stride = 2, threads 0 and 1 "
         "give 7 + 11 = 18 and 7 + 7 = 14. In step with stride = 1, thread 0 gives 18 + 14 = 32. Two arrows "
         "lead into every new value, and a barrier separates the steps.", parts)


def broken():
    w = 640
    x0 = 130
    parts = [T(x0 - 12, 44, "thread 0", 11.5, INK, "end"),
             T(x0 - 12, 104, "thread 2", 11.5, INK, "end")]
    parts += [cell(x0, 30, 120, 26, "stride = 4: 2 + 5 = 7", "blue", 11),
              cell(x0 + 128, 30, 160, 26, "stride = 2: 7 + cell 2", "blue", 11),
              cell(x0 + 296, 30, 130, 26, "stride = 1: 10 + 14", "blue", 11),
              cell(x0, 90, 230, 26, "stride = 4 (delayed): 3 + 8 = 11", "peach", 11)]
    rx = x0 + 128 + 140
    parts += [line(rx, 57, rx, 76, RED, 1.4),
              f'  <circle cx="{rx}" cy="57" r="3.5" fill="{RED}"/>',
              T(rx + 8, 76, "reads cell 2 = 3, before thread 2 stores 11", 11, RED, "start"),
              arrow(x0, 150, x0 + 470, 150, "grey", 1.1),
              T(x0 + 470, 166, "time", 11, GREY, "end"),
              T(x0, 190, "cell 0 = 7 + 3 = 10 in place of 18", 12, RED, "start", 600),
              T(x0, 210, "block sum = 10 + 14 = 24 in place of 32", 13, RED, "start", 700)]
    save("ch15-broken.svg", w, 226,
         "Without the barrier after step with stride = 4. Thread 0 finishes stride = 4 early and in step with stride = 2 "
         "reads cell 2 while it still holds 3, because thread 2 has not yet stored 3 + 8 = 11. "
         "Cell 0 becomes 7 + 3 = 10 in place of 18, and the block sum 10 + 14 = 24 in place of "
         "32.", parts)


def invariant():
    w = 640
    cw, ch, pitch = 34, 24, 40
    x0 = 210
    parts = [T(x0 - 14, 36, "products of block 0", 11, GREY, "end")]
    for k in range(8):
        parts.append(cell(x0 + k * pitch, 20, cw, ch, P[k], "white", 12))
    rows = [("start of stride = 4", [0], "cell 0 = p0 = 2"),
            ("start of stride = 2", [0, 4], "cell 0 = p0 + p4 = 7"),
            ("start of stride = 1", [0, 2, 4, 6], "cell 0 = p0 + p2 + p4 + p6 = 18"),
            ("after stride = 1", list(range(8)), "cell 0 = all 8 products = 32")]
    y = 62
    for label, members, text in rows:
        parts.append(T(x0 - 14, y + 16, label, 11.5, INK, "end", 600))
        for k in range(8):
            on = k in members
            parts.append(cell(x0 + k * pitch, y, cw, ch, P[k] if on else "",
                              "blue" if on else "white", 11, faded=not on))
        parts.append(T(x0, y + ch + 15, text, 11, HI, "start"))
        y += 58
    parts.append(T(20, y + 10, "At the start of the step with a given stride, cell 0 holds the products "
                   "with index j ≡ 0 (mod 2 × stride).", 11.5, GREY, "start"))
    save("ch15-invariant.svg", w, y + 24,
         "The loop invariant for cell 0 of block 0. At the start of step with stride = 4 it holds p0 = 2; "
         "of stride = 2, p0 + p4 = 7; of stride = 1, p0 + p2 + p4 + p6 = 18; after stride = 1, all 8 products, "
         "32.", parts)


def shuffle():
    w = 640
    cw, ch, pitch = 40, 28, 54
    x0 = 150
    levels = [list(P[:8])]
    v = list(P[:8])
    for off in (4, 2, 1):
        v = [v[k] + (v[k + off] if k + off < 8 else 0) for k in range(8)]
        levels.append(list(v))
    parts = [T(20, 22, "The register value of lanes 0 to 7 (no shared memory, no barrier)", 12, INK,
               "start", 700)]
    for k in range(8):
        parts.append(T(x0 + k * pitch + cw / 2, 40, f"lane {k}", 10.5, GREY))
    labels = ["start", "distance 4", "distance 2", "distance 1"]
    y = 48
    for depth, vals in enumerate(levels):
        used = 8 >> depth if depth else 8
        for k, val in enumerate(vals):
            on = k < used
            parts.append(cell(x0 + k * pitch, y, cw, ch, val if on else "",
                              THREADS[k % 4] if on else "grey", 12,
                              outline=(depth == 3 and k == 0)))
        parts.append(T(x0 - 14, y + 18, labels[depth], 11.5, INK, "end", 600))
        if depth < 3:
            off = 4 >> depth
            ny = y + 92
            for k in range(off):
                col = THREADS[k % 4]
                xs = x0 + (k + off) * pitch + cw / 2
                xt = x0 + k * pitch + cw / 2
                by = ny - 20
                parts.append(curve(f"M{xt:.1f},{y + ch + 3} L{xt:.1f},{by - 10}", col, 2.2))
                parts.append(curve(f"M{xs:.1f},{y + ch + 3} C{xs:.1f},{y + ch + 36} "
                                   f"{xt + 34:.1f},{by - 30} {xt + 8:.1f},{by - 7}", col, 2.2))
                parts.append(curve(f"M{xt:.1f},{by + 8} L{xt:.1f},{ny - 2}", col, 2.2))
                parts.append(plus_badge(xt, by, col))
            y = ny
        else:
            y += ch
    parts.append(T(x0, y + 24, "Each lane adds the value of the lane distance places to its right. A real warp"
                   " has 32 lanes", 11.5, GREY, "start"))
    parts.append(T(x0, y + 42, "and 5 steps, with distances 16, 8, 4, 2 and 1. Gray lanes hold values we do not use.",
                   11.5, GREY, "start"))
    save("ch15-shuffle.svg", w, y + 56,
         "A warp sum by shuffles, drawn on 8 lanes. The registers start as 2, 6, 3, 4, 5, 1, 8, 3. "
         "With distance 4, lanes 0 to 3 add the values of lanes 4 to 7: 7, 7, 11, 7. With distance 2: "
         "18, 14. With distance 1: lane 0 holds 32. The same numbers as the tree in shared memory, "
         "with no shared memory and no barrier.", parts)


def block_sum():
    w = 640
    x0 = 30
    bw, gap = 62, 6
    parts = [T(20, 22, "1 block of 256 threads = 8 warps", 12, INK, "start", 700)]
    for k in range(8):
        x = x0 + k * (bw + gap)
        parts += [cell(x, 34, bw, 28, f"warp {k}", THREADS[k % 4], 11),
                  arrow(x + bw / 2, 63, x + bw / 2, 92, "grey", 1.1),
                  cell(x + 8, 94, bw - 16, 26, f"w[{k}]", "white", 11)]
    parts += [T(x0 + bw / 2 + 6, 82, "5 shuffles", 10.5, GREY, "start"),
              rect(x0 - 6, 88, 8 * (bw + gap) + 6, 38, "none", PAL["mint"][1], 1.2, 8,
                   dash="4 3"),
              T(x0 + 8 * (bw + gap) + 4, 112, "shared", 10.5, PAL["mint"][1], "start"),
              line(x0 - 6, 140, x0 + 8 * (bw + gap), 140, INK, 1.4),
              T(x0 + 8 * (bw + gap) + 4, 144, "barrier", 10.5, INK, "start")]
    parts += [arrow(x0 + bw / 2, 127, x0 + bw / 2, 166, "grey", 1.1),
              cell(x0, 168, 300, 28, "warp 0: lanes 0 to 7 read w[0..8), 5 shuffles", "blue", 11),
              arrow(x0 + 150, 197, x0 + 150, 222, "grey", 1.1),
              cell(x0 + 60, 224, 180, 28, "thread 0: block sum", "hi", 11, outline=True),
              T(x0 + 330, 186, "1 barrier and 8 floats of shared memory,", 11.5, INK, "start"),
              T(x0 + 330, 204, "against 9 barriers and 256 floats for the tree", 11.5, INK,
                "start")]
    save("ch15-blocksum.svg", w, 266,
         "The block sum with shuffles. Each of the 8 warps of a 256-thread block sums its 32 values "
         "with 5 shuffles, and lane 0 stores the warp sum in w[0] to w[7] in shared memory. After 1 "
         "barrier, warp 0 reads the 8 values and sums them with 5 more shuffles; thread 0 holds the "
         "block sum.", parts)


def combine():
    w = 640
    parts = [T(20, 22, "a second launch", 12.5, INK, "start", 700),
             T(340, 22, "atomic additions", 12.5, INK, "start", 700)]
    parts += [cell(30, 40, 110, 28, "block 0: 32", "blue", 11.5),
              cell(160, 40, 110, 28, "block 1: 31", "mint", 11.5),
              arrow(85, 69, 85, 98, "grey", 1.1), arrow(215, 69, 215, 98, "grey", 1.1),
              cell(30, 100, 110, 26, "partial[0] = 32", "white", 11),
              cell(160, 100, 110, 26, "partial[1] = 31", "white", 11),
              T(150, 146, "end of kernel 1: every block is done", 10.5, GREY),
              arrow(150, 152, 150, 176, "grey", 1.1),
              cell(60, 178, 180, 28, "kernel 2: 32 + 31 = 63", "hi", 11.5),
              T(150, 226, "fixed order, same bits every run", 11, GREEN, weight=600)]
    parts += [cell(350, 40, 110, 28, "block 1: 31", "mint", 11.5),
              cell(480, 40, 110, 28, "block 0: 32", "blue", 11.5),
              T(405, 84, "finishes first", 10.5, GREY), T(535, 84, "finishes second", 10.5, GREY),
              arrow(405, 89, 405, 116, "mint", 1.3), arrow(535, 89, 535, 116, "blue", 1.3),
              cell(350, 118, 240, 26, "result: 0 → 31 → 63", "white", 11.5),
              T(470, 168, "1 kernel, no second launch;", 11, INK),
              T(470, 186, "the order of additions depends on timing", 11, INK),
              T(470, 226, "order can change between runs", 11, RED, weight=600)]
    save("ch15-combine.svg", w, 242,
         "2 ways to combine the block sums 32 and 31. Left: each block writes its partial sum, and "
         "a second kernel, which starts after every block of the first is done, adds them in a "
         "fixed order: 63. Right: each block adds its sum into result with an atomic addition, in "
         "the order the blocks finish, here block 1 then block 0: 0, 31, 63.", parts)


def race():
    w = 640
    x0 = 130
    parts = [T(x0 - 12, 44, "block 0", 11.5, INK, "end"),
             T(x0 - 12, 104, "block 1", 11.5, INK, "end")]
    parts += [cell(x0, 30, 130, 26, "read result = 0", "blue", 11),
              cell(x0 + 200, 30, 150, 26, "write 0 + 32 = 32", "blue", 11),
              cell(x0 + 30, 90, 130, 26, "read result = 0", "mint", 11),
              cell(x0 + 380, 90, 110, 26, "write 0 + 31", "mint", 11),
              arrow(x0, 150, x0 + 490, 150, "grey", 1.1),
              T(x0 + 490, 166, "time", 11, GREY, "end"),
              T(x0, 194, "Both blocks read 0 before either writes. The later write wins:", 12, INK,
                "start"),
              T(x0, 216, "result = 31 in place of 63", 13, RED, "start", 700)]
    save("ch15-race.svg", w, 232,
         "A plain addition in place of an atomic one. Block 0 reads result = 0, then block 1 reads "
         "0. Block 0 writes 32, and block 1 later writes 0 + 31. The result is 31 in place of "
         "63.", parts)


def order():
    w = 640
    parts = [T(20, 22, "left to right", 12.5, INK, "start", 700),
             T(340, 22, "small values first", 12.5, INK, "start", 700)]
    parts += [cell(20, 36, 280, 28, "16,777,216 + 1 = 16,777,217", "white", 11.5, mono=False),
              T(160, 82, "not a float: rounds to 16,777,216", 11, RED),
              cell(20, 92, 280, 28, "16,777,216 + 1 = 16,777,217", "white", 11.5),
              T(160, 138, "rounds again to 16,777,216", 11, RED),
              cell(20, 150, 280, 28, "result 16,777,216", "red", 12)]
    parts += [cell(340, 36, 280, 28, "1 + 1 = 2", "white", 11.5),
              T(480, 82, "exact", 11, GREEN),
              cell(340, 92, 280, 28, "16,777,216 + 2 = 16,777,218", "white", 11.5),
              T(480, 138, "a float: exact", 11, GREEN),
              cell(340, 150, 280, 28, "result 16,777,218", "mint", 12)]
    parts.append(T(20, 206, "Above 2²⁴ = 16,777,216 the floats are 2 apart, so 16,777,217 does "
                   "not exist.", 11.5, GREY, "start"))
    save("ch15-order.svg", w, 222,
         "The same 3 floats, 16,777,216, 1 and 1, added in 2 orders. Left to right, each + 1 gives "
         "16,777,217, which is not a float and rounds back to 16,777,216. With the small values "
         "first, 1 + 1 = 2 and 16,777,216 + 2 = 16,777,218 exactly.", parts)


def traffic():
    w = 640
    parts = [T(20, 22, "map, then reduce: 16 bytes per pair", 12.5, INK, "start", 700)]
    items = [("read a[i]", "blue"), ("read b[i]", "blue"), ("write p[i]", "peach"),
             ("read p[i]", "blue")]
    for k, (name, col) in enumerate(items):
        parts += [cell(20 + k * 104, 36, 96, 28, name, col, 11), T(68 + k * 104, 80, "4 B", 11,
                                                                  GREY)]
    parts.append(T(450, 56, "2 kernels, p in between", 11, GREY, "start"))
    parts.append(T(20, 116, "fused: 8 bytes per pair", 12.5, INK, "start", 700))
    for k, (name, col) in enumerate(items[:2]):
        parts += [cell(20 + k * 104, 130, 96, 28, name, col, 11), T(68 + k * 104, 174, "4 B", 11,
                                                                   GREY)]
    parts.append(T(240, 150, "product added to a register, never stored", 11, GREEN, "start"))
    rows = [("map, then reduce", 268, "peach"), ("fused", 134, "mint")]
    b, y = bars(20, 200, rows, 1.4, "MB", label_w=130)
    parts += b
    parts.append(T(20, y + 12, "for n = 16,777,216 pairs: at 102 GB/s, 2.63 ms and 1.32 ms",
                   11.5, GREY, "start"))
    save("ch15-traffic.svg", w, y + 26,
         "Bytes moved per pair. Map then reduce reads a[i] and b[i], writes p[i] and reads it back: "
         "16 bytes. The fused kernel reads a[i] and b[i] and adds the product to a register: 8 "
         "bytes. For 16,777,216 pairs, 268 MB against 134 MB, 2.63 ms against 1.32 ms at 102 "
         "GB/s.", parts)


# Measured on a Jetson Orin NX (run quoted in the chapter): Rust, C++ in GB/s.
MEASURED = [("map, then reduce", 80.5, 86.7), ("fused, shared tree", 95.2, 95.1),
            ("fused, shuffles", 95.5, 92.2), ("fused, atomic", 95.5, 96.6)]


def measured():
    w = 640
    rows = []
    for name, rust, cpp in MEASURED:
        rows += [(f"{name}, Rust", rust, "peach"), (f"{name}, C++", cpp, "blue")]
    parts, y = bars(20, 22, rows, 3.9, "GB/s", label_w=170, h=17, gap=7, peak=102,
                    peak_label="102 GB/s, the memory bandwidth", inside=True)
    save("ch15-measured.svg", w, y + 22,
         "Measured rates on the Jetson Orin NX, Rust and C++: map then reduce 80.5 and 86.7 GB/s; "
         "fused with a shared-memory tree 95.2 and 95.1 GB/s; fused with shuffles 95.5 and 92.2 "
         "GB/s; fused with an atomic combine 95.5 and 96.6 GB/s. A dashed line marks 102 GB/s.",
         parts)


def main():
    example()
    chain()
    regroup()
    steps()
    broken()
    invariant()
    shuffle()
    block_sum()
    combine()
    race()
    order()
    traffic()
    measured()


if __name__ == "__main__":
    main()
