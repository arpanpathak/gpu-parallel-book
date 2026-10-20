"""The animations of chapter 8: a scan inside a warp, and a scan of many blocks.

    python3 tools/anim_ch08.py            # both
    python3 tools/anim_ch08.py warp       # 8 lanes scanned by offsets 1, 2 and 4
    python3 tools/anim_ch08.py levels     # scan the blocks, scan the totals, add the offsets

Values travel as small cards, so a reader sees which lane's value each lane adds, and when.
"""
import sys

from motion import (BRASS, BRASS_LT, INK, LINE, MUTED, STAGE, TEAL, TEAL_LT, Timeline, caption,
                    clamp, progress, render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
VIOLET, VIOLET_LT = "#7a4fd0", "#f0eafb"
VALUES = [3, 1, 4, 1, 5, 9, 2, 6]


def value(p, x, y, label, colour, fill, w=56, opacity=1.0):
    p.rect(x, y, w, 30, fill, colour, 7, 1.6, opacity=opacity)
    p.text(x + w / 2, y + 20, label, 14, colour, 700, "middle", mono=True, opacity=opacity)


def travel(a, b, u):
    return a + (b - a) * clamp(u)


def scanned(row, start=0):
    out, run = [], start
    for v in row:
        run += v
        out.append(run)
    return out


# ------------------------------------------------------------- warp scan

def warp():
    tl = Timeline(caption="", kind="step", s1=0.0, s2=0.0, s3=0.0)

    tl.chapter("offset 1")
    tl.say("8 lanes of a warp hold 3, 1, 4, 1, 5, 9, 2, 6. The goal: lane i ends with the sum "
           "of lanes 0 to i.")
    tl.wait(1.0)
    tl.say("Offset 1: every lane from 1 up adds the value of the lane 1 to its left, read "
           "straight from that lane's register.")
    tl.to(2.4, s1=1.0)
    tl.wait(0.6)

    tl.chapter("offset 2")
    tl.say("Offset 2: each lane now holds the sum of 2 values, and adds the pair that ends "
           "2 lanes to its left.")
    tl.to(2.4, s2=1.0)
    tl.wait(0.6)

    tl.chapter("offset 4")
    tl.say("Offset 4: sums of 4 values add sums of 4. Lane 7 now holds 3 + 1 + … + 6 = 31.")
    tl.to(2.4, s3=1.0)
    tl.say("3 steps for 8 lanes; a warp of 32 needs 5, with offsets 1, 2, 4, 8 and 16.",
           "insight")
    tl.wait(2.4)

    rows = [VALUES]
    for d in (1, 2, 4):
        rows.append([rows[-1][i] + (rows[-1][i - d] if i >= d else 0) for i in range(8)])
    gx, cw, gap, y = 104, 68, 12, 160

    def x_of(i):
        return gx + i * (cw + gap)

    def draw(p, s, total):
        title_block(p, "A scan inside a warp", "Each lane adds the value d lanes to its left.")
        for i in range(8):
            p.text(x_of(i) + cw / 2, y - 26, f"lane {i}", 12, MUTED, 600, "middle")
        steps = (s.s1, s.s2, s.s3)
        done = sum(1 for u in steps if u >= 1)
        live = next((k for k, u in enumerate(steps) if 0 < u < 1), None)
        shown = rows[done]
        for i in range(8):
            colour, fill = (BRASS, BRASS_LT) if done == 0 else (TEAL, TEAL_LT)
            value(p, x_of(i), y, str(shown[i]), colour, fill, cw)
        if live is not None:
            d, u = (1, 2, 4)[live], steps[live]
            src = rows[live]
            for i in range(d, 8):
                move = clamp(u * 1.5 - (i - d) * 0.06)
                if move < 1:
                    value(p, travel(x_of(i - d), x_of(i), move), y + 54 - 20 * move, str(src[i - d]),
                          BLUE, BLUE_LT, cw)
            p.text(x_of(0), y + 120, f"offset {d}", 13, BLUE, 700)
        caption(p, tl, s.t, 330)
        progress(p, tl, s.t, total, 330 + 2 * 23 + 26 + 22)

    return tl, draw, 330 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- many blocks

BLOCKS = [[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8]]


def levels():
    tl = Timeline(caption="", kind="step", p1=0.0, up=0.0, p2=0.0, p3=0.0)

    tl.chapter("scan blocks")
    tl.say("12 values in 3 blocks of 4. Step 1: each block scans its own values, all 3 at "
           "the same time.")
    tl.to(1.8, p1=1.0)
    tl.say("Each block writes its total: 9, 22 and 21.")
    tl.to(1.6, up=1.0)
    tl.wait(0.4)

    tl.chapter("scan totals")
    tl.say("Step 2: the 3 totals are scanned by the same code: 9, 31, 52. Block b now knows "
           "the sum of every block before it.")
    tl.to(1.8, p2=1.0)
    tl.wait(1.6)

    tl.chapter("add offsets")
    tl.say("Step 3: block 1 adds 9 and block 2 adds 31 to each of their values. Block 0 "
           "adds nothing.")
    tl.to(2.2, p3=1.0)
    tl.say("The 12 running sums, 3 to 52. For 67,108,864 values a block holds 1,024, and its "
           "65,536 totals are scanned by the same 3 steps.", "insight")
    tl.wait(2.6)

    local = [scanned(b) for b in BLOCKS]
    totals = [r[-1] for r in local]
    offsets = [0] + scanned(totals)[:-1]
    final = [scanned(b, o) for b, o in zip(BLOCKS, offsets)]
    gx, cw, gap, bgap, y, ty = 60, 50, 6, 30, 120, 236

    def x_of(b, i):
        return gx + b * (4 * (cw + gap) + bgap) + i * (cw + gap)

    def draw(p, s, total):
        title_block(p, "A scan of many blocks", "Scan the blocks, scan their totals, add the offsets.")
        for b in range(3):
            left = x_of(b, 0) - 8
            p.rect(left, y - 30, 4 * (cw + gap) + 10, 86, STAGE, LINE, 10, 1.2)
            p.text(left + 10, y - 12, f"block {b}", 12, MUTED, 700)
            for i in range(4):
                if s.p3 > 0 and b > 0:
                    row, colour, fill = final[b] if s.p3 >= 1 else local[b], TEAL, TEAL_LT
                elif s.p1 >= 1:
                    row, colour, fill = local[b], BLUE, BLUE_LT
                else:
                    row, colour, fill = BLOCKS[b], BRASS, BRASS_LT
                if s.p3 >= 1:
                    colour, fill = TEAL, TEAL_LT
                pulse = 0 < s.p1 < 1 and abs(s.p1 * 4 - i - 0.5) < 0.5
                value(p, x_of(b, i), y, str(row[i]), colour, fill, cw, 0.6 if pulse else 1.0)
        p.text(gx, ty - 10, "block totals", 12, MUTED, 700)
        for b in range(3):
            tx = gx + b * (cw + gap)
            if s.up > 0:
                label = str(scanned(totals)[b] if s.p2 >= 1 else totals[b])
                if s.up < 1:
                    value(p, travel(x_of(b, 3), tx, s.up), travel(y, ty, s.up), str(totals[b]),
                          VIOLET, VIOLET_LT, cw)
                else:
                    value(p, tx, ty, label, VIOLET, VIOLET_LT, cw, 0.6 if 0 < s.p2 < 1 else 1.0)
        if 0 < s.p3 < 1:
            for b in (1, 2):
                for i in range(4):
                    move = clamp(s.p3 * 1.6 - i * 0.1)
                    if move < 1:
                        value(p, travel(gx + (b - 1) * (cw + gap), x_of(b, i), move),
                              travel(ty, y, move), f"+{offsets[b]}", VIOLET, VIOLET_LT, cw)
        caption(p, tl, s.t, 320)
        progress(p, tl, s.t, total, 320 + 2 * 23 + 26 + 22)

    return tl, draw, 320 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"warp": ("ch08-warp-scan.gif", warp), "levels": ("ch08-scan-levels-anim.gif", levels)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
