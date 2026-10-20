"""The animations of chapter 9: tiles through shared memory, and a 4 x 4 patch in registers.

    python3 tools/anim_ch09.py            # both
    python3 tools/anim_ch09.py tiles      # a block walks its strips of A and B
    python3 tools/anim_ch09.py outer      # 4 values of A and 4 of B make 16 products

Tiles and values travel as small cards, so a reader sees what each step copies, and where.
"""
import sys

from motion import (BRASS, BRASS_LT, INK, LINE, MUTED, STAGE, TEAL, TEAL_LT, Timeline, caption,
                    clamp, progress, render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
VIOLET, VIOLET_LT = "#7a4fd0", "#f0eafb"


def cell(p, x, y, w, colour, fill, label="", opacity=1.0):
    p.rect(x, y, w - 3, w - 3, fill, colour, 4, 1.3, opacity=opacity)
    if label:
        p.text(x + (w - 3) / 2, y + w / 2 + 3, label, 11.5, colour, 700, "middle", mono=True,
               opacity=opacity)


def travel(a, b, u):
    return a + (b - a) * clamp(u)


# ------------------------------------------------------------- tiles

def tiles():
    tl = Timeline(caption="", kind="step", s0=0.0, s1=0.0, s2=0.0, s3=0.0)

    tl.chapter("the strips")
    tl.say("The block computes the 2 x 2 tile of C in rows 2 and 3, columns 4 and 5. It needs "
           "rows 2 and 3 of A and columns 4 and 5 of B, all 8 values long.")
    tl.wait(7.0)

    tl.chapter("tile by tile")
    for k, name in enumerate(("s0", "s1", "s2", "s3")):
        if k == 0:
            tl.say("Step k0 = 0: the block copies 1 tile of A and 1 tile of B into shared memory, "
                   "and every thread adds their products to its element of C.")
        elif k == 1:
            tl.say("Step k0 = 2: the next tiles along the strips. The same shared memory is reused.")
        elif k == 2:
            tl.say("Step k0 = 4: each step adds 2 more products to every element of C.")
        elif k == 3:
            tl.say("After 4 steps the strips are used up and C holds the full sums.")
        tl.to(1.8, **{name: 1.0})
        tl.wait(0.5)
    tl.say("Each value copied to shared memory was read by 2 threads here; with tiles of 16 x 16, "
           "by 16. Each C cell counts the steps added so far.", "insight")
    tl.wait(2.6)

    w, n = 22, 8
    ax, bx, cx, my, sy = 40, 300, 560, 120, 330

    def draw(p, s, total):
        title_block(p, "Tiles through shared memory", "A block walks its strips of A and B.")
        steps = (s.s0, s.s1, s.s2, s.s3)
        done = sum(1 for u in steps if u >= 1)
        live = next((k for k, u in enumerate(steps) if 0 < u < 1), None)
        for name, x in (("A", ax), ("B", bx), ("C", cx)):
            p.text(x, my - 10, name, 14, INK, 700)
        for r in range(n):
            for c in range(n):
                in_a = r in (2, 3)
                in_b = c in (4, 5)
                cell(p, ax + c * w, my + r * w, w, BLUE if in_a else LINE, BLUE_LT if in_a else "#ffffff")
                cell(p, bx + c * w, my + r * w, w, VIOLET if in_b else LINE,
                     VIOLET_LT if in_b else "#ffffff")
                hot = r in (2, 3) and c in (4, 5)
                cell(p, cx + c * w, my + r * w, w, TEAL if hot else LINE,
                     TEAL_LT if hot else "#ffffff", str(done) if hot and done else "")
        p.rect(240, sy - 30, 300, 84, STAGE, LINE, 10, 1.2)
        p.text(254, sy - 12, "shared memory", 12, MUTED, 700)
        k = live if live is not None else (done - 1 if done else None)
        if k is not None:
            u = steps[k] if live is not None else 1.0
            for r in (0, 1):
                for c in (0, 1):
                    sxA, syA = ax + (2 * k + c) * w, my + (2 + r) * w
                    sxB, syB = bx + (4 + c) * w, my + (2 * k + r) * w
                    m = clamp(u * 1.6)
                    cell(p, travel(sxA, 300 + c * w, m), travel(syA, sy + r * w + 2, m), w, BLUE,
                         BLUE_LT, "A")
                    cell(p, travel(sxB, 420 + c * w, m), travel(syB, sy + r * w + 2, m), w, VIOLET,
                         VIOLET_LT, "B")
            p.text(380, sy + 30, f"k0 = {2 * k}", 12, MUTED, 700, "middle")
        caption(p, tl, s.t, 410)
        progress(p, tl, s.t, total, 410 + 2 * 23 + 26 + 22)

    return tl, draw, 410 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- outer product

def outer():
    tl = Timeline(caption="", kind="step", k0=0.0, k1=0.0, k2=0.0)

    tl.chapter("k = 0")
    tl.say("A thread owns a 4 x 4 patch of C: 16 sums, kept in registers.")
    tl.wait(1.6)
    tl.say("For k = 0 it reads 4 values of A, from a column of the A tile, and 4 values of B, "
           "from a row of the B tile.")
    tl.to(2.0, k0=1.0)
    tl.say("Every pair multiplies once: 16 multiply-adds from 8 reads of shared memory.")
    tl.wait(1.6)

    tl.chapter("k = 1, 2, …")
    tl.say("k = 1: the next column of A, the next row of B, 16 more multiply-adds.")
    tl.to(2.0, k1=1.0)
    tl.to(2.0, k2=1.0)
    tl.say("16 values of k per tile, then the next tile. 1 element per thread needed 2 reads "
           "for each multiply-add; the patch needs 1 for every 2.", "insight")
    tl.wait(2.6)

    w = 44
    gx, gy = 330, 150

    def draw(p, s, total):
        title_block(p, "A 4 x 4 patch per thread", "8 reads from shared memory, 16 multiply-adds.")
        steps = (s.k0, s.k1, s.k2)
        done = sum(1 for u in steps if u >= 1)
        live = next((k for k, u in enumerate(steps) if 0 < u < 1), None)
        p.text(60, 128, "A tile, column k", 12, MUTED, 700)
        p.text(560, 128, "B tile, row k", 12, MUTED, 700)
        for r in range(4):
            for c in range(3):
                on = live == c or (live is None and done and c == done - 1)
                cell(p, 60 + c * 34, gy + r * w, 34, BLUE if on else LINE, BLUE_LT if on else "#ffffff",
                     f"a{r}")
        for r in range(3):
            for c in range(4):
                on = live == r or (live is None and done and r == done - 1)
                cell(p, 560 + c * 34, gy + r * 34, 34, VIOLET if on else LINE,
                     VIOLET_LT if on else "#ffffff", f"b{c}")
        k = live
        u = steps[k] if k is not None else 0.0
        for i in range(4):
            for j in range(4):
                lit = done + (1 if k is not None and clamp(u * 2 - 0.8 - (i * 4 + j) * 0.03) >= 1 else 0)
                fill = TEAL_LT if lit else "#ffffff"
                cell(p, gx + j * w, gy + i * w, w, TEAL if lit else LINE, fill, f"{lit}" if lit else "")
        if k is not None:
            m = clamp(u * 2)
            for i in range(4):
                cell(p, travel(60 + k * 34, gx - 40, m), gy + i * w + 4, 34, BLUE, BLUE_LT, f"a{i}")
                cell(p, travel(560 + i * 34, gx + i * w + 4, m), travel(gy + k * 34, gy - 40, m), 34,
                     VIOLET, VIOLET_LT, f"b{i}")
        p.text(gx + 2 * w, gy + 4 * w + 22, "products added per accumulator", 12, MUTED, 600, "middle")
        caption(p, tl, s.t, 400)
        progress(p, tl, s.t, total, 400 + 2 * 23 + 26 + 22)

    return tl, draw, 400 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"tiles": ("ch09-tiles-anim.gif", tiles), "outer": ("ch09-outer-anim.gif", outer)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
