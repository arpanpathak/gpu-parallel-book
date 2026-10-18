"""The animations of chapter 4: bytes moving between host and device.

    python3 tools/anim_ch04.py            # both
    python3 tools/anim_ch04.py copy       # pageable and pinned copies
    python3 tools/anim_ch04.py unified    # unified memory pages on first touch

Pages are drawn as small cards that travel between memories, so a reader sees
how many trips each kind of copy makes and when the bytes actually move.
"""
import sys

from motion import (BRASS, BRASS_LT, FAINT, INK, LINE, MUTED, PAPER, RUST, RUST_LT, STAGE, TEAL,
                    TEAL_LT, W, Timeline, caption, clamp, linear, progress, render, title_block)

PAGES = 8
BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
CELL, GAP = 30, 6


def slots(x0):
    return [x0 + k * (CELL + GAP) for k in range(PAGES)]


def box(p, x, y, w, h, title, sub="", edge=LINE, fill=STAGE):
    p.rect(x, y, w, h, fill, edge, 12, 1.4)
    p.text(x + 14, y + 22, title, 13.5, INK, 700)
    if sub:
        p.text(x + w - 14, y + 22, sub, 12, MUTED, 500, "end")


def page(p, x, y, colour, fill, label="", opacity=1.0):
    p.rect(x, y, CELL, CELL, fill, colour, 6, 1.6, opacity=opacity)
    if label:
        p.text(x + CELL / 2, y + CELL / 2 + 4.5, label, 11, colour, 700, "middle", opacity=opacity)


def travel(a, b, u):
    return a + (b - a) * clamp(u)


# ------------------------------------------------------------- pageable vs pinned

def copy_paths():
    tl = Timeline(caption="", kind="step", lift=0.0, cpu=0.0, dma_p=0.0, dma_q=0.0)

    tl.chapter("pages")
    tl.say("Host memory is managed in pages. The operating system may move a pageable page "
           "while the program runs, so a copy engine cannot read it by address.")
    tl.to(1.2, lift=1.0)
    tl.wait(0.8)
    tl.to(1.2, lift=0.0)
    tl.wait(0.6)

    tl.chapter("pageable")
    tl.say("From pageable memory, the CPU first copies every page into a pinned staging buffer: "
           "2 time units per page.")
    tl.to(6.0, ease=linear, cpu=float(PAGES))
    tl.say("Then the copy engine moves the staging buffer to the device: 1 unit per page. "
           "8 pages took 8 × 2 + 8 × 1 = 24 units.")
    tl.to(2.4, ease=linear, dma_p=float(PAGES))
    tl.wait(0.6)

    tl.chapter("pinned")
    tl.say("Pinned pages cannot move, so the copy engine reads them directly: 1 trip, 8 units.")
    tl.to(2.4, ease=linear, dma_q=float(PAGES))
    tl.wait(0.6)

    tl.chapter("result")
    tl.say("Pinned: 8 units. Pageable: 24 units, 3 times as long, because of the CPU's extra copy. "
           "Measured copies show the same ratio.", "insight")
    tl.wait(2.0)

    host_x, stage_x, dev_x = 40, 330, 520

    def draw(p, s, total):
        t = s.t
        title_block(p, "Copying 8 pages to the device", "From pageable memory: 2 copies. From pinned memory: 1.")

        # Pageable row.
        y = 96
        p.text(40, y - 6, "pageable", 14, RUST, 700)
        # host memory drawn as 2 rows of 4, so 3 boxes fit across
        def hpos(k):
            return host_x + (k % 4) * (CELL + GAP), y + 30 + (k // 4) * (CELL + 4) - 18
        box(p, 28, y + 4, 4 * (CELL + GAP) + 18, 108, "host", "pageable")
        box(p, stage_x - 12, y + 4, 4 * (CELL + GAP) + 18, 108, "staging", "pinned", BRASS, BRASS_LT)
        box(p, dev_x + 60, y + 4, 4 * (CELL + GAP) + 18, 108, "device", "", TEAL, TEAL_LT)
        now, later = [], []
        for k in range(PAGES):
            hx, hy = 40 + (k % 4) * (CELL + GAP), y + 38 + (k // 4) * (CELL + 4)
            sx, sy = stage_x + (k % 4) * (CELL + GAP), hy
            dx, dy = dev_x + 72 + (k % 4) * (CELL + GAP), hy
            if k == 2:                           # the page the operating system moves
                hx += 0
                hy -= 0
            u1 = clamp(s.cpu - k)
            u2 = clamp(s.dma_p - k)
            if u2 > 0:
                x, yy = travel(sx, dx, u2), sy
            else:
                x, yy = travel(hx, sx, u1), hy
            if k == 2 and s.cpu == 0:
                x += 0
            lifted = s.lift if k == 2 else 0.0
            moving = 0 < u1 < 1 or 0 < u2 < 1
            (later if moving else now).append((x + 14 * lifted, yy - 6 * lifted, str(k)))
        for x, yy, label in now + later:
            page(p, x, yy, RUST, RUST_LT, label)
        if s.lift > 0.05:
            p.text(28, y + 128, "the operating system may move page 2", 11.5, RUST, 700, opacity=s.lift)

        units_p = 2 * min(s.cpu, PAGES) + min(s.dma_p, PAGES)
        p.text(W - 30, y + 128, "time: %d units" % round(units_p), 14, RUST, 700, "end", mono=True)
        p.text(254, y + 54, "CPU", 11.5, MUTED, 700, "middle")
        p.text(254, y + 68, "→", 14, MUTED, 700, "middle")
        p.text(530, y + 54, "DMA", 11.5, MUTED, 700, "middle")
        p.text(530, y + 68, "→", 14, MUTED, 700, "middle")

        # Pinned row.
        y = 266
        p.text(40, y - 6, "pinned", 14, BLUE, 700)
        box(p, 28, y + 4, 4 * (CELL + GAP) + 18, 108, "host", "pinned", BLUE, BLUE_LT)
        box(p, dev_x + 60, y + 4, 4 * (CELL + GAP) + 18, 108, "device", "", TEAL, TEAL_LT)
        now_q, later_q = [], []
        for k in range(PAGES):
            hx, hy = 40 + (k % 4) * (CELL + GAP), y + 38 + (k // 4) * (CELL + 4)
            dx = dev_x + 72 + (k % 4) * (CELL + GAP)
            (later_q if 0 < s.dma_q - k < 1 else now_q).append((travel(hx, dx, s.dma_q - k), hy, str(k)))
        for x, yy, label in now_q + later_q:
            page(p, x, yy, BLUE, BLUE_LT, label)
        p.text(380, y + 54, "DMA, 1 trip", 11.5, MUTED, 700, "middle")
        p.text(380, y + 68, "→", 14, MUTED, 700, "middle")
        p.text(W - 30, y + 128, "time: %d units" % round(min(s.dma_q, PAGES)), 14, BLUE, 700, "end", mono=True)

        caption(p, tl, t, 418)
        progress(p, tl, t, total, 418 + 2 * 23 + 26 + 22)

    return tl, draw, 418 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- unified memory

def unified():
    tl = Timeline(caption="", kind="step", touch=0.0, again=0.0, bulk=0.0, run3=0.0, phase=0.0)

    tl.chapter("1st launch")
    tl.set(phase=1.0)
    tl.say("The host wrote all 8 pages, so they sit in host memory. A kernel now reads them "
           "in order. Each first touch is a page fault: the page must cross the link.")
    tl.to(8.0, ease=linear, touch=float(PAGES))
    tl.say("8 faults, 1 per page. The first launch pays for moving every page.", "fail")
    tl.wait(1.0)

    tl.chapter("2nd launch")
    tl.set(phase=2.0)
    tl.say("A 2nd launch finds every page already on the device: no faults, full speed.")
    tl.to(2.4, ease=linear, again=float(PAGES))
    tl.wait(1.0)

    tl.chapter("prefetch")
    tl.set(phase=3.0, touch=0.0, again=0.0)
    tl.say("With cudaMemPrefetchAsync, the pages move in 1 bulk transfer before the kernel starts.")
    tl.to(2.0, bulk=1.0)
    tl.say("The kernel then runs with no faults at all, even on its first launch.", "insight")
    tl.to(2.4, ease=linear, run3=float(PAGES))
    tl.wait(1.6)

    host_x, dev_x, y = 40, 470, 120

    def draw(p, s, total):
        t = s.t
        title_block(p, "Unified memory: pages move on first touch", "1 pointer, 8 pages, and a kernel that reads them.")
        box(p, 28, y, 4 * (CELL + GAP) + 18, 112, "host memory")
        box(p, dev_x - 12, y, 4 * (CELL + GAP) + 18, 112, "device memory", "", TEAL, TEAL_LT)
        p.rect(200, y + 46, 254, 6, "#e6ebf0", "none", 3)
        p.text(327, y + 70, "link", 11.5, MUTED, 700, "middle")
        faults = 0
        now, later = [], []
        for k in range(PAGES):
            hx, hy = host_x + (k % 4) * (CELL + GAP), y + 34 + (k // 4) * (CELL + 4)
            dx = dev_x + (k % 4) * (CELL + GAP)
            if s.phase < 3:
                u = clamp(s.touch - k)
                faulted = s.touch > k
            else:
                u = clamp(s.bulk)
                faulted = False
            if s.phase < 3 and faulted:
                faults += 1
            lane = y + 34                     # the link's height, above the first row
            if u <= 0.2:
                x, yy = hx, travel(hy, lane, u / 0.2)
            elif u <= 0.8:
                x, yy = travel(hx, dx, (u - 0.2) / 0.6), lane
            else:
                x, yy = dx, travel(lane, hy, (u - 0.8) / 0.2)
            colour, fill = (RUST, RUST_LT) if (s.phase < 3 and 0 < u < 1) else (BLUE, BLUE_LT)
            (later if 0 < u < 1 else now).append((x, yy, colour, fill, str(k)))
            if s.phase < 3 and 0 < u < 1:
                p.text(327, y + 30, "page fault: page %d crosses the link" % k, 12, RUST, 700, "middle")

        for x, yy, colour, fill, label in now + later:
            page(p, x, yy, colour, fill, label)

        # The kernel's reading head over device memory.
        head = s.touch if s.phase == 1 else s.again if s.phase == 2 else s.run3
        if 0 < head < PAGES:
            k = int(head)
            hx = dev_x + (k % 4) * (CELL + GAP)
            hy = y + 34 + (k // 4) * (CELL + 4)
            p.rect(hx - 4, hy - 4, CELL + 8, CELL + 8, "none", BRASS, 8, 2.6)
            p.text(dev_x + 4 * (CELL + GAP) + 10, y + 70, "kernel reads page %d" % k, 12, BRASS, 700)

        rows = [("faults", str(faults if s.phase < 3 else 0), RUST),
                ("phase", ["", "1st launch", "2nd launch", "prefetch, then launch"][int(s.phase)], INK)]
        for i, (k, v, c) in enumerate(rows):
            p.text(40, y + 150 + i * 22, k, 13, MUTED, 600)
            p.text(110, y + 150 + i * 22, v, 13, c, 700, mono=True)

        caption(p, tl, t, y + 210)
        progress(p, tl, t, total, y + 210 + 2 * 23 + 26 + 22)

    return tl, draw, y + 210 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"copy": ("ch04-copy-paths.gif", copy_paths), "unified": ("ch04-unified.gif", unified)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
