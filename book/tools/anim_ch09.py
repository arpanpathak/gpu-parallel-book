"""The animations of chapter 9: tiles through shared memory, and a 4 x 4 patch in registers.

    python3 tools/anim_ch09.py            # both
    python3 tools/anim_ch09.py tiles      # a block walks its strips of A and B
    python3 tools/anim_ch09.py outer      # 4 values of A and 4 of B make 16 products

Tiles and values travel as small cards, so a reader sees what each step copies, and where.
"""
import sys

from motion import (INK, LINE, MUTED, PAPER, STAGE, TEAL, TEAL_LT, Timeline, caption, clamp,
                    progress, render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
VIOLET, VIOLET_LT = "#7a4fd0", "#f0eafb"
GREY, GREY_LT = "#b9c3ce", "#f1f4f7"


def cell(p, x, y, w, colour, fill, label="", opacity=1.0, size=13):
    p.rect(x, y, w - 4, w - 4, fill, colour, 5, 1.4, opacity=opacity)
    if label:
        p.text(x + (w - 4) / 2, y + (w - 4) / 2 + size * 0.36, label, size, colour, 700, "middle",
               mono=True, opacity=opacity)


def travel(a, b, u):
    return a + (b - a) * clamp(u)


def footer(p, tl, s, total, y):
    caption(p, tl, s.t, y)
    progress(p, tl, s.t, total, y + 2 * 23 + 26 + 22)
    return y + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- tiles

def tiles():
    tl = Timeline(caption="", kind="step", s0=0.0, s1=0.0, s2=0.0, s3=0.0)

    tl.chapter("the strips")
    tl.say("8 × 8 matrices, tiles of 2 × 2. The block owns the green tile of C. It needs rows 2 "
           "and 3 of A and columns 4 and 5 of B: 8 values along each.")
    tl.wait(4.5)

    tl.chapter("step 1")
    tl.say("Step 1: the block copies 1 tile of A and 1 tile of B into shared memory, 4 values each.")
    tl.to(2.2, s0=1.0)
    tl.say("Each of its 4 threads adds 2 products to its sum: 2 of the 8 it needs.")
    tl.wait(1.6)

    tl.chapter("steps 2 to 4")
    tl.say("Step 2: the next pair of tiles overwrites shared memory, and every sum grows by 2 "
           "more products.")
    tl.to(2.2, s1=1.0)
    tl.say("Steps 3 and 4: the tiles walk on along the strips.")
    tl.to(2.0, s2=1.0)
    tl.to(2.0, s3=1.0)
    tl.say("4 steps, 8 products per sum. Each value copied was read by 2 threads; with tiles of "
           "16 × 16, by 16.", "insight")
    tl.wait(3.0)

    w, n = 26, 8
    ax, bx, cx, my = 40, 306, 572, 116
    sx, sy = 150, 366

    def draw(p, s, total):
        title_block(p, "Tiles through shared memory", "A block walks its strips of A and B, 1 pair of "
                    "tiles per step.")
        steps = (s.s0, s.s1, s.s2, s.s3)
        done = sum(1 for u in steps if u >= 1)
        live = next((k for k, u in enumerate(steps) if 0 < u < 1), None)
        current = live if live is not None else (done - 1 if done else None)
        for name, x in (("A", ax), ("B", bx), ("C", cx)):
            p.text(x, my - 12, name, 15, INK, 700)
        for r in range(n):
            for c in range(n):
                ka = c // 2 if r in (2, 3) else None
                kb = r // 2 if c in (4, 5) else None
                for kk, x, colour, fill, row, col in ((ka, ax, BLUE, BLUE_LT, r, c),
                                                       (kb, bx, VIOLET, VIOLET_LT, r, c)):
                    if kk is None:
                        cell(p, x + col * w, my + row * w, w, LINE, PAPER)
                    elif kk == current:
                        cell(p, x + col * w, my + row * w, w, colour, fill)
                    elif current is not None and kk < current:
                        cell(p, x + col * w, my + row * w, w, GREY, GREY_LT)
                    else:
                        cell(p, x + col * w, my + row * w, w, colour, PAPER, opacity=0.55)
                hot = r in (2, 3) and c in (4, 5)
                cell(p, cx + c * w, my + r * w, w, TEAL if hot else LINE, TEAL_LT if hot else PAPER)
        products = 2 * done + (2 * clamp(steps[live] * 2 - 1) if live is not None else 0)
        for r in (2, 3):
            for c in (4, 5):
                h = (w - 8) * products / 8
                p.rect(cx + c * w + 2, my + r * w + w - 6 - h, w - 8, h, TEAL, "none", 2, opacity=0.8)
        p.text(cx, my + n * w + 22, f"each sum: {round(products)} of 8 products", 13, TEAL, 700)

        p.rect(sx, sy - 34, 348, 116, STAGE, LINE, 12, 1.2)
        p.text(sx + 14, sy - 14, "shared memory", 12, MUTED, 700)
        p.text(sx + 56, sy + 2 * w + 20, "tileA", 11.5, BLUE, 700, "middle", mono=True)
        p.text(sx + 292, sy + 2 * w + 20, "tileB", 11.5, VIOLET, 700, "middle", mono=True)
        if current is not None:
            u = steps[current] if live is not None else 1.0
            m = clamp(u * 2)
            for r in (0, 1):
                for c in (0, 1):
                    cell(p, travel(ax + (2 * current + c) * w, sx + 30 + c * w, m),
                         travel(my + (2 + r) * w, sy - 4 + r * w, m), w, BLUE, BLUE_LT, "A", size=11)
                    cell(p, travel(bx + (4 + c) * w, sx + 266 + c * w, m),
                         travel(my + (2 * current + r) * w, sy - 4 + r * w, m), w, VIOLET, VIOLET_LT,
                         "B", size=11)
            p.text(sx + 174, sy + 26, f"k0 = {2 * current}", 13, INK, 700, "middle", mono=True)
        return footer(p, tl, s, total, 482)

    return tl, draw, 482 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- outer product

A_COLS = [[1, 2, 0, 3], [0, 1, 1, 2], [2, 0, 1, 1]]   # tileA[4ty + i][k] for k = 0, 1, 2
B_ROWS = [[2, 1, 0, 1], [1, 3, 2, 0], [0, 1, 1, 2]]   # tileB[k][4tx + j]


def outer():
    tl = Timeline(caption="", kind="step", k0=0.0, k1=0.0, k2=0.0)

    tl.chapter("k = 0")
    tl.say("A thread owns a 4 × 4 patch of C: 16 sums in registers, all 0 at the start.")
    tl.wait(2.0)
    tl.say("k = 0: it reads 4 values of A, 1, 2, 0, 3, and 4 values of B, 2, 1, 0, 1, from "
           "shared memory.")
    tl.to(2.4, k0=1.0)
    tl.say("Each sum adds its product a[i] × b[j]: row 3 gets 3 × 2, 3 × 1, 3 × 0, 3 × 1 = "
           "6, 3, 0, 3.")
    tl.wait(2.4)

    tl.chapter("k = 1, 2")
    tl.say("k = 1: the next column of A and the next row of B. 8 reads, 16 more multiply-adds.")
    tl.to(2.4, k1=1.0)
    tl.wait(1.0)
    tl.say("k = 2: the same again.")
    tl.to(2.4, k2=1.0)
    tl.say("1 element per thread read 2 values for each multiply-add. The patch reads 8 for 16, "
           "4 times fewer.", "insight")
    tl.wait(3.0)

    sums = []
    acc = [[0] * 4 for _ in range(4)]
    for a, b in zip(A_COLS, B_ROWS):
        acc = [[acc[i][j] + a[i] * b[j] for j in range(4)] for i in range(4)]
        sums.append(acc)
    w, gx, gy = 54, 340, 170
    tw = 40

    def draw(p, s, total):
        title_block(p, "A 4 × 4 patch per thread", "Each k: 8 reads from shared memory, 16 multiply-adds.")
        steps = (s.k0, s.k1, s.k2)
        done = sum(1 for u in steps if u >= 1)
        live = next((k for k, u in enumerate(steps) if 0 < u < 1), None)
        p.text(40, 134, "tileA, column k", 12, MUTED, 700)
        p.text(626, 134, "tileB, row k", 12, MUTED, 700)
        for i in range(4):
            for k in range(3):
                on = k == live
                cell(p, 40 + k * tw, gy + i * w + 7, tw, BLUE if on else LINE, BLUE_LT if on else PAPER,
                     str(A_COLS[k][i]), size=12)
        for k in range(3):
            for j in range(4):
                on = k == live
                cell(p, 626 + j * tw, gy + k * tw, tw, VIOLET if on else LINE,
                     VIOLET_LT if on else PAPER, str(B_ROWS[k][j]), size=12)
        for k in range(3):
            p.text(40 + k * tw + 18, gy - 12, f"k={k}", 10.5, MUTED, 600, "middle", mono=True)
            p.text(626 - 10, gy + k * tw + 23, f"k={k}", 10.5, MUTED, 600, "end", mono=True)
        u = steps[live] if live is not None else 0.0
        for i in range(4):
            for j in range(4):
                landed = live is not None and clamp(u * 2.2 - 1.1 - (i * 4 + j) * 0.03) >= 1
                shown = sums[done - 1][i][j] if done else 0
                if landed:
                    shown = sums[live][i][j]
                fresh = landed and u < 0.98
                cell(p, gx + j * w, gy + i * w, w, TEAL, TEAL_LT if fresh else PAPER, str(shown),
                     size=16)
        p.text(gx + 2 * w - 2, gy + 4 * w + 20, "acc[i][j], 16 sums in registers", 12, MUTED, 600,
               "middle")
        if live is not None:
            m = clamp(u * 2.2)
            for i in range(4):
                cell(p, travel(40 + live * tw, gx - 48, m), gy + i * w + 7, tw, BLUE, BLUE_LT,
                     str(A_COLS[live][i]), size=12)
                cell(p, travel(626 + i * tw, gx + i * w + 7, m), travel(gy + live * tw, gy - 50, m), tw,
                     VIOLET, VIOLET_LT, str(B_ROWS[live][i]), size=12)
        return footer(p, tl, s, total, 430)

    return tl, draw, 430 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"tiles": ("ch09-tiles-anim.gif", tiles), "outer": ("ch09-outer-anim.gif", outer)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
