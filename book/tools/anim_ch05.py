"""The animations of chapter 5: a lost update, and a tree sum with and without barriers.

    python3 tools/anim_ch05.py            # both
    python3 tools/anim_ch05.py race       # plain add against atomicAdd
    python3 tools/anim_ch05.py tree       # tree sum in shared memory

Values travel as small cards, so a reader sees which value each thread read, and when.
"""
import sys

from motion import (BRASS, BRASS_LT, INK, LINE, MUTED, RUST, RUST_LT, STAGE, TEAL, TEAL_LT, W,
                    Timeline, caption, clamp, linear, progress, render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
VIOLET, VIOLET_LT = "#7a4fd0", "#f0eafb"


def box(p, x, y, w, h, title, edge=LINE, fill=STAGE):
    p.rect(x, y, w, h, fill, edge, 12, 1.4)
    p.text(x + 14, y + 22, title, 13.5, INK, 700)


def value(p, x, y, label, colour, fill, w=46, opacity=1.0):
    p.rect(x, y, w, 30, fill, colour, 7, 1.6, opacity=opacity)
    p.text(x + w / 2, y + 20, label, 14, colour, 700, "middle", mono=True, opacity=opacity)


def travel(a, b, u):
    return a + (b - a) * clamp(u)


# ------------------------------------------------------------- lost update

def race():
    tl = Timeline(caption="", kind="step", ra=0.0, rb=0.0, wa=0.0, wb=0.0, phase=1.0,
                  qa=0.0, qbw=0.0, qb=0.0)

    tl.chapter("plain add")
    tl.say("counter = counter + 1 is 3 steps: read the counter, add 1 in a register, "
           "write it back. The counter holds 5.")
    tl.wait(1.2)
    tl.say("Thread A reads 5. Before A writes, thread B also reads 5.")
    tl.to(1.4, ra=1.0)
    tl.to(1.4, rb=1.0)
    tl.wait(0.6)
    tl.say("A adds 1 and writes 6. Then B adds 1 to its own 5 and writes 6 as well.")
    tl.to(1.4, wa=1.0)
    tl.to(1.4, wb=1.0)
    tl.say("2 additions, and the counter grew by 1. B's write replaced A's.", "fail")
    tl.wait(2.0)

    tl.chapter("atomicAdd")
    tl.set(phase=2.0, ra=0.0, rb=0.0, wa=0.0, wb=0.0)
    tl.say("With atomicAdd, each thread sends a request. The memory system reads, adds and "
           "writes as 1 step, 1 request at a time per address.")
    tl.to(2.0, qa=1.0, qbw=1.0)
    tl.wait(0.4)
    tl.say("B's request waits until A's is done, then adds 1 to 6.")
    tl.to(1.4, qb=1.0)
    tl.say("Counter 7: both additions counted. On a real GPU, 1,048,576 plain adds left 495; "
           "1,048,576 atomic adds left 1,048,576.", "insight")
    tl.wait(2.6)

    cx, cy = 340, 104
    ax, bx, ty = 60, 580, 250

    def draw(p, s, total):
        title_block(p, "A lost update", "2 threads add 1 to the same counter.")
        box(p, cx, cy, 140, 96, "counter", BRASS, BRASS_LT)
        box(p, ax, ty, 180, 96, "thread A", BLUE, BLUE_LT)
        box(p, bx, ty, 180, 96, "thread B", VIOLET, VIOLET_LT)
        p.text(ax + 14, ty + 74, "register", 11.5, MUTED, 600)
        p.text(bx + 14, ty + 74, "register", 11.5, MUTED, 600)
        slot = (cx + 47, cy + 46)
        rega, regb = (ax + 110, ty + 50), (bx + 110, ty + 50)

        if s.phase < 2:
            stored = "6" if s.wa >= 1 else "5"
            colour = RUST if s.wb >= 1 else BRASS
            value(p, *slot, stored, colour, RUST_LT if s.wb >= 1 else "#ffffff")
            for r, w, reg, col, lt in ((s.ra, s.wa, rega, BLUE, BLUE_LT), (s.rb, s.wb, regb, VIOLET, VIOLET_LT)):
                if 0 < r < 1:
                    value(p, travel(slot[0], reg[0], r), travel(slot[1], reg[1], r), "5", col, lt)
                elif r >= 1 and w <= 0:
                    value(p, *reg, "5", col, lt)
                elif 0 < w < 1:
                    value(p, travel(reg[0], slot[0], w), travel(reg[1], slot[1], w), "6", col, lt)
            if s.wb >= 1:
                p.text(cx + 70, cy + 128, "1 addition lost", 13, RUST, 700, "middle")
        else:
            done = (s.qa >= 1) + (s.qb >= 1)
            value(p, *slot, str(5 + done), TEAL, TEAL_LT)
            p.text(cx + 70, cy + 128, "1 request at a time", 12, TEAL, 700, "middle")
            spot = (cx + 160, cy + 46)
            if 0 < s.qa < 1:
                value(p, travel(rega[0], slot[0], s.qa), travel(rega[1], slot[1], s.qa), "+1", BLUE, BLUE_LT)
            if s.qb <= 0 and s.qbw > 0:
                value(p, travel(regb[0], spot[0], s.qbw), travel(regb[1], spot[1], s.qbw), "+1", VIOLET, VIOLET_LT)
                if s.qbw >= 1:
                    p.text(spot[0] + 23, spot[1] + 48, "waits", 11.5, VIOLET, 700, "middle")
            elif 0 < s.qb < 1:
                value(p, travel(spot[0], slot[0], s.qb), slot[1], "+1", VIOLET, VIOLET_LT)

        caption(p, tl, s.t, 380)
        progress(p, tl, s.t, total, 380 + 2 * 23 + 26 + 22)

    return tl, draw, 380 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- tree sum

START = [3, 1, 4, 1, 5, 9, 2, 6]


def tree():
    tl = Timeline(caption="", kind="step", l1=0.0, l2=0.0, l3=0.0, gate=0.0, phase=1.0)

    tl.chapter("with barriers")
    tl.say("8 threads sum 8 values in shared memory. At stride 4, thread k adds s[k + 4] into s[k].")
    tl.to(2.0, l1=1.0)
    tl.say("The barrier holds every thread until all 4 additions are written.")
    tl.to(0.8, gate=1.0)
    tl.to(0.8, gate=0.0)
    tl.say("Stride 2, barrier, stride 1. Each level reads only finished values.")
    tl.to(1.8, l2=1.0)
    tl.to(0.6, gate=1.0)
    tl.to(0.6, gate=0.0)
    tl.to(1.6, l3=1.0)
    tl.say("s[0] = 31, the right sum.", "insight")
    tl.wait(1.8)

    tl.chapter("without barriers")
    tl.set(phase=2.0, l1=0.0, l2=0.0, l3=0.0)
    tl.say("Without barriers, threads 0 and 1 race ahead to stride 2 while threads 2 and 3 "
           "are still adding.")
    tl.to(1.6, l2=1.0)
    tl.say("Thread 0 reads s[2] = 4 instead of 6, and thread 1 reads s[3] = 1 instead of 7.")
    tl.to(1.6, l1=1.0)
    tl.to(1.6, l3=1.0)
    tl.say("s[0] = 23, not 31. On a real GPU, 2,450 of 4,096 blocks got a wrong sum.", "fail")
    tl.wait(2.4)

    gx, cw, gap = 120, 64, 12
    rows = [110, 190, 270, 350]

    def x_of(k):
        return gx + k * (cw + gap)

    def draw(p, s, total):
        title_block(p, "A tree sum in shared memory", "Each level halves the active threads.")
        right = [8, 10, 6, 7]
        lv2 = [14, 17] if s.phase < 2 else [12, 11]
        final = 31 if s.phase < 2 else 23
        names = ["s[0..7]", "stride 4", "stride 2", "stride 1"]
        for i, y in enumerate(rows):
            p.text(gx - 16, y + 20, names[i], 12.5, MUTED, 600, "end")
        for k, v in enumerate(START):
            value(p, x_of(k), rows[0], str(v), BRASS, BRASS_LT, cw)

        def level(u, src_row, dst_row, srcs, outs, stride, wrong=()):
            for k, out in enumerate(outs):
                if u <= 0:
                    continue
                moving = clamp(u * 1.4 - k * 0.12)
                src = srcs[k + stride]
                label = str(src)
                col, lt = (RUST, RUST_LT) if k in wrong else (TEAL, TEAL_LT)
                if moving < 1:
                    value(p, travel(x_of(k + stride), x_of(k), moving), travel(src_row, dst_row, moving),
                          label, col, lt, cw)
                else:
                    value(p, x_of(k), dst_row, str(out), col, lt, cw)

        if s.phase < 2:
            level(s.l1, rows[0], rows[1], START, right, 4)
            level(s.l2, rows[1], rows[2], right, lv2, 2)
            level(s.l3, rows[2], rows[3], lv2, [final], 1)
        else:
            stale = START[:4]
            level(s.l2, rows[0], rows[2], [8, 10] + stale[2:4], lv2, 2, wrong=(0, 1))
            level(s.l1, rows[0], rows[1], START, right, 4)
            level(s.l3, rows[2], rows[3], lv2, [final], 1, wrong=(0,))

        for i in range(3):
            y = (rows[i] + rows[i + 1]) / 2 + 15
            on = s.phase < 2
            p.line(gx - 6, y, x_of(7) + cw + 6, y, RUST if on else LINE, 2.2 if s.gate > 0.05 and on else 1.4,
                   dash="6 5")
        if s.phase < 2:
            p.text(x_of(7) + cw + 16, rows[1] - 18, "barriers", 12, RUST, 700)
        else:
            p.text(x_of(7) + cw + 16, rows[1] - 18, "no barriers", 12, MUTED, 700)

        caption(p, tl, s.t, 410)
        progress(p, tl, s.t, total, 410 + 2 * 23 + 26 + 22)

    return tl, draw, 410 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"race": ("ch05-lost-update.gif", race), "tree": ("ch05-tree-sum.gif", tree)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
