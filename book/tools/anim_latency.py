"""Animation 1 of chapter 1: hiding the wait for memory.

Groups of threads share one arithmetic unit. Every group reads its operands
from memory for 6 cycles, then needs the unit for 2 cycles to add them, then
reads again. A chart grows cycle by cycle: one row per group, and a row for the
unit that turns green while some group is adding. The run is shown with 1, 2
and 4 groups. With 4, the unit never waits, and 8 / 2 = 4 is Little's law.

    python3 tools/anim_latency.py
"""

from motion import (BRASS, FAINT, INK, LINE, MUTED, PAPER, STAGE, TEAL, TEAL_LT, W, Timeline,
                    caption, clamp, linear, progress, render, title_block)

READ, ADD, HORIZON = 6, 2, 24        # cycles per read, cycles per add, cycles drawn

X0, PX = 170, 25.0                   # left edge of the chart, pixels per cycle
TOP, ROW, BAR = 104, 46, 30          # first row, row pitch, bar height
GREY, GREY_LT = "#7f8ea0", "#eef1f5"


def simulate(groups):
    """The blocks of a run: (row, kind, start, end), row -1 being the unit.

    Every group starts its first read at cycle 0. A group whose read is done
    waits for the unit if another group is adding, then adds, then reads again."""
    ready = [READ] * groups           # when each group's read completes
    free, blocks = 0, []
    for g in range(groups):
        blocks.append((g, "read", 0, READ))
    while True:
        g = min(range(groups), key=lambda k: (ready[k], k))
        start = max(ready[g], free)
        if start >= HORIZON:
            break
        if start > ready[g]:
            blocks.append((g, "queue", ready[g], start))
        end = start + ADD
        blocks.append((g, "add", start, end))
        blocks.append((-1, "add", start, end))
        blocks.append((g, "read", end, end + READ))
        free, ready[g] = end, end + READ
    return blocks


RUNS = {n: simulate(n) for n in (1, 2, 4)}


def busy(groups, upto):
    """Cycles the unit has spent adding, up to cycle `upto`."""
    return sum(max(0.0, min(e, upto) - s) for r, k, s, e in RUNS[groups] if r == -1)


def hide_latency():
    tl = Timeline(caption="", kind="step", groups=1.0, cyc=0.0)

    def run(groups, line, kind="step", follow=None):
        tl.set(groups=float(groups), cyc=0.0)
        tl.say(line, kind)
        tl.to(7.0, ease=linear, cyc=float(HORIZON))
        if follow:
            tl.say(*follow)
        tl.wait(1.2)

    tl.chapter("1 group")
    run(1, "One group reads for 6 cycles, then adds for 2, then reads again. "
           "The arithmetic unit works 2 cycles out of every 8.",
        follow=("While the group waits for memory, the unit has nothing to do: "
                "it is busy 25% of the time.", "fail"))
    tl.chapter("2 groups")
    run(2, "Two groups. When group 0 starts its next read, group 1 has its values "
           "and adds. The unit now works 4 cycles out of every 8.")
    tl.chapter("4 groups")
    run(4, "Four groups. While one group adds, the others are reading or waiting their turn at the unit.",
        follow=("From cycle 6 on, some group always has its values ready. "
                "The unit never waits, although every read still takes 6 cycles.", "insight"))
    tl.chapter("why 4")
    tl.say("A group needs 8 cycles per round and the unit 2 cycles per add, "
           "so 8 / 2 = 4 groups keep the unit busy. This is Little's law.", "insight")
    tl.wait(2.0)

    def draw(p, s, total):
        t = s.t
        groups, cyc = int(round(s.groups)), s.cyc
        title_block(p, "Hiding the wait for memory",
                    "Each group of threads reads for 6 cycles, then adds for 2. One arithmetic unit.")

        # The cycle axis, with a tick every 2 cycles.
        for c in range(0, HORIZON + 1, 2):
            x = X0 + c * PX
            p.text(x, TOP - 12, str(c), 11, FAINT, 600, "middle", mono=True)
            p.line(x, TOP - 4, x, TOP + 5 * ROW - 8, "#edf1f5", 1)

        rows = list(range(groups)) + [-1]
        for i, r in enumerate(rows):
            y = TOP + (i if r >= 0 else 4) * ROW
            name = "group %d" % r if r >= 0 else "arithmetic unit"
            p.text(X0 - 14, y + BAR / 2 + 5, name, 13.5, INK, 700 if r < 0 else 600, "end")
            p.rect(X0, y, HORIZON * PX, BAR, STAGE, LINE, 7, 1)
            for row, kind, a, b in RUNS[groups]:
                if row != r or a >= cyc:
                    continue
                b = min(b, cyc, HORIZON)
                x, w = X0 + a * PX + 1.5, (b - a) * PX - 3
                if kind == "add":
                    p.rect(x, y + 2, w, BAR - 4, TEAL, "none", 6)
                    if w > 34:
                        p.text(x + w / 2, y + BAR / 2 + 4.5, "add", 12, PAPER, 700, "middle")
                elif kind == "read":
                    p.rect(x, y + 2, w, BAR - 4, GREY_LT, GREY, 6, 1.2)
                    if w > 60:
                        p.text(x + w / 2, y + BAR / 2 + 4.5, "read", 12, GREY, 700, "middle")
                else:
                    p.rect(x, y + 2, w, BAR - 4, PAPER, BRASS, 6, 1.2, dash="4 3")

        # The unit row's frame turns teal, and idle stretches read as gaps.
        uy = TOP + 4 * ROW
        p.rect(X0, uy, HORIZON * PX, BAR, "none", TEAL, 7, 1.6)

        # The playhead.
        if 0 < cyc < HORIZON:
            x = X0 + cyc * PX
            p.line(x, TOP - 4, x, uy + BAR + 4, BRASS, 2)

        # The tally: how much of the elapsed time the unit has worked.
        done = busy(groups, cyc)
        share = 0 if cyc < 0.5 else round(100 * done / cyc)
        ty = uy + BAR + 40
        p.text(X0, ty, "unit busy", 13.5, MUTED, 600)
        p.rect(X0 + 84, ty - 13, 300, 16, "#e6ebf0", "none", 8)
        p.rect(X0 + 84, ty - 13, 300 * clamp(share / 100), 16, TEAL, "none", 8)
        p.text(X0 + 398, ty, "%d of %d cycles" % (round(done), int(cyc)), 13.5, INK, 700, mono=True)

        legend_y = ty + 30
        for k, (fill, edge, label) in enumerate(((GREY_LT, GREY, "waiting for a read"),
                                                 (TEAL, TEAL, "adding"),
                                                 (PAPER, BRASS, "values ready, unit taken"))):
            x = X0 + k * 190
            p.rect(x, legend_y - 11, 16, 14, fill, edge, 4, 1.2, dash="4 3" if k == 2 else None)
            p.text(x + 24, legend_y + 1, label, 12.5, MUTED)

        caption(p, tl, t, legend_y + 26)
        progress(p, tl, t, total, legend_y + 26 + 2 * 23 + 26 + 22)

    height = TOP + 4 * ROW + BAR + 40 + 30 + 26 + 2 * 23 + 26 + 22 + 40
    return tl, draw, height


if __name__ == "__main__":
    tl, draw, height = hide_latency()
    print(render("ch01-hide-latency.gif", tl, draw, height))
