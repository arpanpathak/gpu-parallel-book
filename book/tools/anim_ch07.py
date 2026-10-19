"""The animations of chapter 7: sectors fetched by a warp, and a transpose through a tile.

    python3 tools/anim_ch07.py            # both
    python3 tools/anim_ch07.py sectors    # stride 1, 2 and 8: sectors moved and bytes used
    python3 tools/anim_ch07.py tile       # a 4 x 4 tile: rows in, columns out

Sectors and elements travel as small cards, so a reader sees what memory actually moves.
"""
import sys

from motion import (BRASS, BRASS_LT, INK, LINE, MUTED, RUST, RUST_LT, STAGE, TEAL, TEAL_LT, W,
                    Timeline, caption, clamp, linear, progress, render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"


def box(p, x, y, w, h, title, edge=LINE, fill=STAGE):
    p.rect(x, y, w, h, fill, edge, 12, 1.4)
    p.text(x + 14, y + 22, title, 13.5, INK, 700)


def lerp(a, b, u):
    return a + (b - a) * clamp(u)


# ------------------------------------------------------------- sectors

SECTORS = 32


def sectors():
    tl = Timeline(caption="", kind="step", moved=0.0, stride=1.0)

    for stride, needed, line in ((1, 4, "Stride 1: the warp's 32 floats are 128 neighbouring bytes, 4 sectors. "
                                           "Every byte moved is used."),
                                 (2, 8, "Stride 2: the floats spread over 256 bytes, 8 sectors, and half of each "
                                        "sector is unused."),
                                 (8, 32, "Stride 8: every float sits in a sector of its own. 32 sectors move 1,024 "
                                         "bytes to deliver 128.")):
        tl.chapter("stride %d" % stride)
        tl.set(stride=float(stride), moved=0.0)
        tl.say(line, "fail" if stride == 8 else "step")
        tl.to(0.35 * needed + 1.4, ease=linear, moved=float(needed))
        tl.wait(1.8)

    x0, sw, my = 40, 23, 300

    def draw(p, s, total):
        title_block(p, "Sectors fetched by 1 warp", "Memory moves in 32-byte sectors, whatever the warp asks for.")
        stride = int(s.stride)
        needed = min(SECTORS, 4 * stride)
        box(p, x0 - 10, 90, SECTORS * sw + 20, 96, "SM: the warp's 32 lanes")
        box(p, x0 - 10, my - 30, SECTORS * sw + 20, 74, "memory: 32-byte sectors")

        for lane in range(32):
            p.rect(x0 + lane * sw + 6, 128, 10, 10, BLUE_LT, BLUE, 3, 1.2)

        used = min(8, 8 // stride) if stride < 8 else 1
        for k in range(SECTORS):
            wanted = k < needed
            p.rect(x0 + k * sw, my, sw - 3, 28, RUST_LT if wanted else "#ffffff", RUST if wanted else LINE, 4, 1.2)
            if wanted:
                p.rect(x0 + k * sw + 1, my + 1, (sw - 5) * used / 8, 26, RUST, "none", 3)

        for k in range(needed):
            u = clamp(s.moved - k)
            if 0 < u < 1:
                y = lerp(my, 150, u)
                p.rect(x0 + k * sw, y, sw - 3, 22, TEAL_LT, TEAL, 4, 1.4)
                p.rect(x0 + k * sw + 1, y + 1, (sw - 5) * used / 8, 20, TEAL, "none", 3)
            elif u >= 1:
                p.rect(x0 + k * sw, 150, sw - 3, 22, TEAL_LT, TEAL, 4, 1.2)
                p.rect(x0 + k * sw + 1, 151, (sw - 5) * used / 8, 20, TEAL, "none", 3)

        arrived = int(min(s.moved, needed))
        p.text(W - 30, 398, "moved %4d bytes   used %3d bytes" % (32 * arrived, 4 * min(32, arrived * used)),
               14, INK, 700, "end", mono=True)
        caption(p, tl, s.t, 420)
        progress(p, tl, s.t, total, 420 + 2 * 23 + 26 + 22)

    return tl, draw, 420 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- tile

def tile():
    tl = Timeline(caption="", kind="step", load=0.0, gate=0.0, store=0.0)

    tl.chapter("load rows")
    tl.say("Each row of threads reads a row of the input tile, neighbouring floats, and writes it into "
           "the same row of the shared tile.")
    tl.to(4.0, ease=linear, load=4.0)
    tl.say("A barrier: no thread reads the tile until every row is in.")
    tl.to(0.8, gate=1.0)
    tl.wait(0.6)

    tl.chapter("store columns")
    tl.say("Now each row of threads reads a column of the tile and writes it as a row of the output. "
           "Global memory sees only rows.")
    tl.to(4.0, ease=linear, store=4.0)
    tl.say("Rows in, columns out: the output is the transpose, and both global accesses were "
           "coalesced.", "insight")
    tl.wait(2.2)

    cell, gap = 46, 6
    ix, sx, ox, gy = 50, 320, 590, 120

    def at(x0, r, c):
        return x0 + c * (cell + gap), gy + r * (cell + gap)

    def draw(p, s, total):
        title_block(p, "A transpose through a tile", "4 × 4 for clarity; the real tile is 32 × 32.")
        box(p, ix - 14, gy - 34, 4 * (cell + gap) + 22, 4 * (cell + gap) + 46, "input")
        box(p, sx - 14, gy - 34, 4 * (cell + gap) + 22, 4 * (cell + gap) + 46, "shared tile",
            BRASS if s.gate > 0.5 else LINE, BRASS_LT if s.gate > 0.5 else STAGE)
        box(p, ox - 14, gy - 34, 4 * (cell + gap) + 22, 4 * (cell + gap) + 46, "output")

        for r in range(4):
            for c in range(4):
                label = "%d%d" % (r, c)
                x, y = at(ix, r, c)
                p.rect(x, y, cell, cell, BLUE_LT, BLUE, 7, 1.4)
                p.text(x + cell / 2, y + cell / 2 + 5, label, 14, BLUE, 700, "middle", mono=True)

                u = clamp(s.load - r)
                if u > 0:
                    tx, ty = at(sx, r, c)
                    x2, y2 = lerp(x, tx, u), lerp(y, ty, u)
                    p.rect(x2, y2, cell, cell, BRASS_LT, BRASS, 7, 1.4)
                    p.text(x2 + cell / 2, y2 + cell / 2 + 5, label, 14, BRASS, 700, "middle", mono=True)

        for c in range(4):
            v = clamp(s.store - c)
            for r in range(4):
                if v <= 0:
                    continue
                label = "%d%d" % (r, c)
                x, y = at(sx, r, c)
                tx, ty = at(ox, c, r)
                x2, y2 = lerp(x, tx, v), lerp(y, ty, v)
                p.rect(x2, y2, cell, cell, TEAL_LT, TEAL, 7, 1.4)
                p.text(x2 + cell / 2, y2 + cell / 2 + 5, label, 14, TEAL, 700, "middle", mono=True)

        if s.gate > 0.05:
            p.text(sx + 2 * (cell + gap) - 3, gy + 4 * (cell + gap) + 30, "__syncthreads()", 13, RUST, 700,
                   "middle", mono=True, opacity=min(1.0, s.gate * 2))
        caption(p, tl, s.t, 380)
        progress(p, tl, s.t, total, 380 + 2 * 23 + 26 + 22)

    return tl, draw, 380 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"sectors": ("ch07-sectors-anim.gif", sectors), "tile": ("ch07-tile-anim.gif", tile)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
