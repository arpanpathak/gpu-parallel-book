"""The animation of chapter 19: a ring all-reduce over 4 ranks.

    python3 tools/anim_ch19.py

4 ranks sit at the corners of a square, each with 4 chunks. In every step each rank hands 1
chunk to the next rank around the ring, as a card that travels; the values are the ones
code/ch19_collectives prints for 4 ranks of 8 floats.
"""
from motion import (INK, LINE, MUTED, PAPER, TEAL, TEAL_LT, Timeline, caption, clamp, progress,
                    render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
GREY_LT = "#f1f4f7"
K = 4


def states():
    """Each rank's 4 chunk values after every step, and the (source rank, chunk) of every send."""
    data = [[r + 1] * K for r in range(K)]
    out, sends = [[row[:] for row in data]], [[]]
    for s in range(2 * (K - 1)):
        add = s < K - 1
        step = []
        for r in range(K):
            c = (r - s) % K if add else (r + 1 - (s - K + 1)) % K
            step.append((r, c, data[r][c]))
        for r, c, v in step:
            dst = (r + 1) % K
            data[dst][c] = data[dst][c] + v if add else v
        out.append([row[:] for row in data])
        sends.append(step)
    return out, sends


STATES, SENDS = states()
CW = 50
CORNERS = [(120, 120), (500, 120), (500, 300), (120, 300)]


def cell(p, x, y, value, colour, fill, size=17):
    p.rect(x, y, CW - 6, CW - 6, fill, colour, 7, 1.6)
    p.text(x + (CW - 6) / 2, y + (CW - 6) / 2 + size * 0.36, str(value), size, colour, 700, "middle",
           mono=True)


def ring():
    tl = Timeline(caption="", kind="step", **{f"s{k}": 0.0 for k in range(1, 7)})

    tl.chapter("the ring")
    tl.say("4 ranks in a ring, each with 4 chunks of the message. Rank r starts with r + 1 in "
           "every chunk; the sum of all 4 is 10.")
    tl.wait(4.0)

    tl.chapter("reduce-scatter")
    tl.say("Step 1: every rank sends 1 chunk to the next rank, which adds it. 4 sends at once, "
           "1 on each link.")
    tl.to(2.4, s1=1.0)
    tl.say("Steps 2 and 3: each rank passes on the chunk it just received, with its own value "
           "added.")
    tl.to(2.4, s2=1.0)
    tl.to(2.4, s3=1.0)
    tl.say("After 3 steps every rank holds 1 complete chunk, 10: the reduce-scatter is done.",
           "insight")
    tl.wait(3.0)

    tl.chapter("all-gather")
    tl.say("Steps 4 to 6: the complete chunks travel on around the ring and overwrite.")
    tl.to(2.4, s4=1.0)
    tl.to(2.4, s5=1.0)
    tl.to(2.4, s6=1.0)
    tl.say("6 steps, 2 (k − 1). Each rank sent 6 chunks of 2 floats: 12 floats, 1.5 times the "
           "message of 8.", "insight")
    tl.wait(3.5)

    def draw(p, s, total):
        title_block(p, "Ring all-reduce, 4 ranks", "Each step, every rank hands 1 chunk to the next.")
        steps = [getattr(s, f"s{k}") for k in range(1, 7)]
        done = sum(1 for u in steps if u >= 1)
        live = next((k for k, u in enumerate(steps) if 0 < u < 1), None)

        p.rect(100, 140, 620, 200, "none", LINE, 30, 2.0, dash="7 6")
        for x, y, dx, dy in ((410, 140, 1, 0), (720, 240, 0, 1), (410, 340, -1, 0), (100, 240, 0, -1)):
            p.line(x - 9 * dx - 7 * dy, y - 9 * dy - 7 * dx, x, y, MUTED, 2.4)
            p.line(x - 9 * dx + 7 * dy, y - 9 * dy + 7 * dx, x, y, MUTED, 2.4)
        for r, (x, y) in enumerate(CORNERS):
            p.rect(x - 14, y - 34, 4 * CW + 22, CW + 48, PAPER, LINE, 12, 1.4)
            p.text(x, y - 14, f"rank {r}", 13, MUTED, 700)
            for c in range(K):
                p.text(x + c * CW + (CW - 6) / 2, y + CW + 4, f"c{c}", 10.5, MUTED, 600, "middle",
                       mono=True)

        shown = STATES[done]
        landed = set()
        if live is not None:
            u = steps[live]
            if u >= 0.65:
                shown = STATES[live + 1]
                landed = {((r + 1) % K, c) for r, c, _ in SENDS[live + 1]}
        for r, (x, y) in enumerate(CORNERS):
            for c in range(K):
                v = shown[r][c]
                if (r, c) in landed:
                    colour, fill = BLUE, BLUE_LT
                elif v == 10:
                    colour, fill = TEAL, TEAL_LT
                else:
                    colour, fill = INK, GREY_LT
                cell(p, x + c * CW, y, v, colour, fill)

        if live is not None:
            m = clamp(steps[live] / 0.6)
            if m < 1:
                for r, c, v in SENDS[live + 1]:
                    (x0, y0), (x1, y1) = CORNERS[r], CORNERS[(r + 1) % K]
                    x = x0 + c * CW + (x1 - x0) * m
                    y = y0 - 10 + (y1 - y0) * m
                    cell(p, x, y, v, BLUE, BLUE_LT, 15)
            phase = "reduce-scatter: add" if live < K - 1 else "all-gather: overwrite"
            p.text(410, 238, f"step {live + 1}", 16, INK, 700, "middle")
            p.text(410, 258, phase, 12.5, MUTED, 600, "middle")
        return footer(p, tl, s, total, 400)

    return tl, draw, 400 + 2 * 23 + 26 + 22 + 40


def footer(p, tl, s, total, y):
    caption(p, tl, s.t, y)
    progress(p, tl, s.t, total, y + 2 * 23 + 26 + 22)
    return y + 2 * 23 + 26 + 22 + 40


if __name__ == "__main__":
    tl, draw, height = ring()
    print(render("ch25-ring-anim.gif", tl, draw, height))
