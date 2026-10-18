"""The figures of chapter 1, drawn with figkit.

    python3 tools/draw_ch01.py

Each figure works one small example with numbers that can be checked by hand.
figkit refuses to write a figure whose text does not fit its box.
"""
import math

from figkit import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, card, cylinder, icon, legend, must_fit, save,
                    tag, text, width_of, wire, zone)
from draw_mascots import bee as bee_svg

OUT = "src/figures/"


def bar(x, y, w, h, kind, label="", dashed=False, solid=False):
    """A rounded bar in the colours of `kind`, with a centred label that must fit."""
    accent, tint = KINDS[kind]
    fill = accent if solid else tint
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" fill="{fill}"'
             f' stroke="{accent}" stroke-width="1.3"{dash}/>']
    if label:
        must_fit(label, w - 8, 11, bold=True)
        parts.append(text(x + w / 2, y + h / 2 + 4, label, "pill", "middle",
                          "#FFFFFF" if solid else accent))
    return "\n".join(parts)


def line(x1, y1, x2, y2, colour=WIRE, width=1.4, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}"'
            f' stroke-width="{width}"{dashes}/>')


def arrow(x, y1, y2, kind="flow"):
    """A straight vertical arrow whose tip stops 3 units short of `y2`."""
    colour = {"flow": WIRE}.get(kind, KINDS.get(kind, (WIRE,))[0])
    marker = "head" if colour == WIRE else f"head-{kind}"
    return (f'  <path d="M{x},{y1} V{y2 - 3}" fill="none" stroke="{colour}" stroke-width="1.8"'
            f' stroke-linecap="round" marker-end="url(#{marker})"/>')


def fan(x, y, targets, bus, top, kind="flow"):
    """One source fanned out to several targets: a stem down from (x, y) to the
    bus line at `bus`, the bus across the targets, and an arrow down to each."""
    colour = WIRE if kind == "flow" else KINDS[kind][0]
    left, right = min(targets + [x]), max(targets + [x])
    parts = [f'  <path d="M{x},{y} V{bus} M{left},{bus} H{right}" fill="none" stroke="{colour}"'
             f' stroke-width="1.8" stroke-linecap="round"/>',
             f'  <circle cx="{x}" cy="{bus}" r="3" fill="{colour}"/>']
    marker = "head"
    if kind != "flow":
        marker = f"tip-{kind}-{int(x)}"
        parts.append(f'  <defs><marker id="{marker}" viewBox="0 0 10 10" refX="8.5" refY="5"'
                     f' markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">'
                     f'<path d="M0.5,1 L9,5 L0.5,9 Z" fill="{colour}"/></marker></defs>')
    for t in targets:
        parts.append(f'  <path d="M{t},{bus} V{top - 3}" fill="none" stroke="{colour}" stroke-width="1.6"'
                     f' stroke-linecap="round" marker-end="url(#{marker})"/>')
    return "\n".join(parts)


def tile(x, y, glyph, kind="client", side=28):
    """A rounded tile in the tint of `kind` holding one glyph, like the tiles on figkit cards."""
    accent, tint = KINDS[kind]
    mark = side - 10
    return (f'  <rect x="{x}" y="{y}" width="{side}" height="{side}" rx="{side / 4:.0f}" fill="{tint}"/>\n'
            + icon(glyph, x + 5, y + 5, mark, accent))


_bees = 0


def mascot(x, y, width, mood):
    """One of the book's bees, `width` wide, top-left at (x, y)."""
    global _bees
    _bees += 1
    body = bee_svg(mood)
    body = body[body.index(">") + 1:body.rindex("</svg>")].replace('id="body"', f'id="bee{_bees}"')
    body = body.replace("url(#body)", f"url(#bee{_bees})")
    return f'  <g transform="translate({x},{y}) scale({width / 240:.4f})">{body}</g>'


def actors(y, left_sub, right_sub, label):
    """The processor card, the memory store and the wire between them, on one row."""
    return "\n".join([
        card(40, y, 230, 64, "client", "processor", left_sub, glyph="chip"),
        cylinder(430, y - 4, 170, 72, "store", "main memory", right_sub),
        wire([(270, y + 32), (430, y + 32)], label=label),
    ])


def note(x, y, body, anchor="start", room=None, fill=None):
    if room:
        must_fit(body, room)
    return text(x, y, body, "note", anchor, fill)


def mono(x, y, body, anchor="start", fill=None):
    return text(x, y, body, "mono", anchor, fill)


# 1.1  One processor adds four rows, and waits for memory almost all the time.
x0, add_w, gap, top = 40, 9, 5, 206
wait_w = (560 - 4 * add_w - 3 * gap) / 4
parts = [actors(96, ["adds two numbers", "in 4 cycles"], "500 cycles per read", "a[i], b[i]")]
for k in range(4):
    left = x0 + k * (wait_w + add_w + gap)
    parts.append(bar(left, top, wait_w, 40, "idle", f"row {k}: waiting"))
    parts.append(bar(left + wait_w, top, add_w, 40, "danger", solid=True))
parts += [
    line(x0, top + 56, x0 + 560, top + 56, MUTED, 1),
    mono(x0, top + 74, "0", "start", MUTED),
    mono(x0 + 560, top + 74, "2,016 cycles", "end", MUTED),
    legend(x0, top + 110, [("idle", "waiting for a[i] and b[i]: 500 cycles"), ("danger", "add: 4 cycles")]),
    card(x0, top + 134, 470, 58, "danger", "The adder works 16 of 2,016 cycles",
         "under 1% of the time; add bars drawn wider than to scale"),
    mascot(522, top + 126, 82, "sleepy"),
]
save(OUT + "ch01-one-processor.svg", 640, top + 218,
     "A processor card, which adds two numbers in 4 cycles, is wired to a main memory store, which takes "
     "500 cycles per read. Below, the processor adds four rows of the vector one after another: each row "
     "is a long grey bar, waiting 500 cycles for a[i] and b[i], and a thin red bar, the 4-cycle add. The "
     "four rows take 2,016 cycles and the adder is busy for 16 of them. A sleepy bee sits beside the result.",
     "One processor adds 4 rows", "Each row waits about 500 cycles for memory, then adds for 4.", parts)


# 1.2  Reads that overlap: every read takes 10 cycles, one starts every 2 cycles.
gx, step, top, pitch = 110, 19, 214, 28
cut = gx + 10.5 * step
parts = [actors(96, ["sends a new read", "every 2 cycles"], "10 cycles per read", "reads")]
parts += [mono(gx + t * step, top - 10, str(t), "middle", MUTED) for t in range(0, 27, 2)]
for r in range(8):
    start = 2 * r
    y = top + r * pitch
    live = start <= 10.5 < start + 10
    parts.append(text(gx - 12, y + 16, f"read {r}", "sub", "end"))
    parts.append(bar(gx + start * step, y + 3, 10 * step, 20, "good" if live else "idle"))
bottom = top + 8 * pitch
parts += [
    line(cut, top - 4, cut, bottom + 6, KINDS["danger"][0], 2, "6 5"),
    tag(cut + 8, bottom + 2, "at cycle 10.5: 5 reads in flight", "danger"),
    legend(gx, bottom + 50, [("good", "in flight at cycle 10.5"), ("idle", "finished, or not started")]),
    card(40, bottom + 70, 560, 58, "service", "reads in flight = 10 / 2 = 5",
         "time one read takes, divided by the time between two starts"),
]
save(OUT + "ch01-overlap.svg", 640, bottom + 152,
     "A processor that sends a new read every 2 cycles is wired to a memory that takes 10 cycles per "
     "read. Below, eight reads drawn as bars on a time axis from 0 to 26 cycles. Each bar is 10 cycles long, and "
     "each starts 2 cycles after the one before. A dashed line at cycle 10.5 crosses five bars, drawn "
     "green: five reads are in flight at that moment. A card below reads: reads in flight = 10 / 2 = 5.",
     "Reads that overlap", "Every read takes 10 cycles, and a new read starts every 2 cycles.", parts)


# 1.3  The same 500-cycle read, with 1 in flight and with 250 in flight.
s = 0.5                                         # pixels per cycle
parts = [zone(24, 104, 592, 128, "1 read in flight")]
for k in range(2):
    parts.append(bar(48 + k * 500 * s, 140, 500 * s - 4, 26, "idle", f"read {k}: 500 cycles"))
parts += [
    tag(48 + 500 * s - 70, 176, "value 1", "service"),
    tag(48 + 1000 * s - 74, 176, "value 2", "service"),
    note(48, 218, "one value every 500 cycles", room=540),
    zone(24, 268, 592, 196, "250 reads in flight"),
]
for k in range(8):
    parts.append(bar(48 + k * 16, 296 + k * 14, 500 * s - 4, 12, "good"))
parts += [
    note(48 + 8 * 16 + 500 * s + 4, 330, "8 of the 250 reads"),
    note(48 + 8 * 16 + 500 * s + 4, 348, "are drawn"),
    note(48, 430, "a new read starts every 2 cycles, so after the first 500 cycles", room=540),
    note(48, 448, "one value arrives every 2 cycles: 250 times as many", room=540),
    mascot(526, 372, 76, "happy"),
]
save(OUT + "ch01-little.svg", 640, 490,
     "Two panels. Top, 1 read in flight: two reads of 500 cycles run one after the other, and a value "
     "arrives every 500 cycles. Bottom, 250 reads in flight: the reads are drawn as staggered bars that "
     "start 2 cycles apart, each still 500 cycles long, and after the first 500 cycles a value arrives "
     "every 2 cycles.",
     "One read in flight, or 250", "Each read takes 500 cycles in both panels. Only the number in flight changes.",
     parts)


# 1.4  Amdahl on a timeline: 20 + 100 + 10 microseconds on 1 processor and on 8.
s, x0 = 3.6, 48                                 # pixels per microsecond
parts = [
    tile(x0, 88, "chip"), text(x0 + 36, 108, "1 processor", "name"),
    bar(x0, 120, 20 * s, 34, "danger", "read 20"),
    bar(x0 + 20 * s, 120, 100 * s, 34, "good", "compute 100 µs"),
    bar(x0 + 120 * s, 120, 10 * s, 34, "danger", "10"),
    tag(x0 + 130 * s + 10, 128, "130 µs", "idle"),
    tile(x0, 182, "chip"), text(x0 + 36, 202, "8 processors", "name"),
    bar(x0, 263, 20 * s - 2, 34, "danger", "read 20"),
]
for lane in range(8):
    parts.append(bar(x0 + 20 * s, 214 + lane * 17, 12.5 * s - 2, 14, "good"))
parts += [
    bar(x0 + 32.5 * s, 263, 10 * s, 34, "danger", "10"),
    tag(x0 + 42.5 * s + 10, 271, "42.5 µs", "idle"),
    note(x0 + 42.5 * s + 90, 238, "the compute phase: 12.5 µs"),
    note(x0 + 42.5 * s + 90, 256, "on each of the 8 processors,"),
    note(x0 + 42.5 * s + 90, 274, "all at the same time"),
    legend(x0, 396, [("danger", "serial: one processor at a time"), ("good", "parallel: divides by p")]),
]
save(OUT + "ch01-amdahl-timeline.svg", 640, 424,
     "Top, one processor: a 20 microsecond read, a 100 microsecond compute phase and a 10 microsecond "
     "write, 130 microseconds in all. Bottom, eight processors: the same read and write, and between "
     "them eight short compute bars of 12.5 microseconds that run side by side, 42.5 microseconds in all.",
     "The same program on 1 processor and on 8", "The compute phase divides by 8. The read and the write do not divide.",
     parts)


# 1.5  Amdahl's law as curves, one for each serial share f.
px0, px1, py0, py1 = 84, 500, 392, 100          # plot box: left, right, bottom, top
lx = lambda p: px0 + (px1 - px0) * math.log2(p) / 10           # p from 1 to 1,024
ly = lambda v: py0 - (py0 - py1) * math.log2(v) / 10           # speedup from 1 to 1,024
speed = lambda f, p: 1 / (f + (1 - f) / p)
parts = [f'  <rect x="{px0}" y="{py1}" width="{px1 - px0}" height="{py0 - py1}" fill="#FFFFFF" stroke="{EDGE}"/>']
for k in range(0, 11, 2):
    parts += [line(lx(2 ** k), py0, lx(2 ** k), py1, EDGE, 1), mono(lx(2 ** k), py0 + 18, f"{2 ** k:,}", "middle", MUTED),
              line(px0, ly(2 ** k), px1, ly(2 ** k), EDGE, 1), mono(px0 - 8, ly(2 ** k) + 4, f"{2 ** k:,}", "end", MUTED)]
parts += [text((px0 + px1) / 2, py0 + 40, "processors p", "sub", "middle"),
          text(px0 - 56, py1 - 14, "speedup S", "sub")]
curves = [(0.001, "outside", "1,000"), (0.01, "client", "100"), (0.05, "service", "20"), (0.10, "store", "10"),
          (0.23, "danger", "4.3")]
for f, kind, limit in curves:
    points = " ".join(f"{lx(2 ** (i / 8)):.1f},{ly(speed(f, 2 ** (i / 8))):.1f}" for i in range(81))
    width = 3 if f == 0.23 else 1.8
    parts.append(f'  <polyline points="{points}" fill="none" stroke="{KINDS[kind][0]}" stroke-width="{width}"/>')
    end = speed(f, 1024)
    parts.append(text(px1 + 10, ly(end) + 4, f"f = {f:g}: limit {limit}", "pill", "start", KINDS[kind][0]))
parts.append(line(px0, ly(1 / 0.23), px1, ly(1 / 0.23), KINDS["danger"][0], 1.2, "6 5"))
for p in (2, 4, 8):
    parts.append(f'  <circle cx="{lx(p):.1f}" cy="{ly(speed(0.23, p)):.1f}" r="4.5" fill="{KINDS["danger"][0]}"/>')
parts.append(tag(lx(8) + 10, ly(speed(0.23, 8)) + 6, "p = 8: S = 3.06", "danger"))
save(OUT + "ch01-amdahl-multi.svg", 680, 456,
     "Speedup against the number of processors, both on logarithmic scales from 1 to 1,024. Five curves, "
     "for serial shares 0.001, 0.01, 0.05, 0.10 and 0.23, rise and then flatten at their limits of 1,000, "
     "100, 20, 10 and 4.3. The thick f = 0.23 curve has points at 2, 4 and 8 processors, the last at a "
     "speedup of 3.06, under a dashed line at 4.3.",
     "Amdahl's law, one curve for each serial share", "Every curve flattens at 1/f. The thick curve is the example program.",
     parts)


# 1.6  The two laws side by side: shrink the time, or grow the job.
parts = [zone(16, 104, 316, 300, "Amdahl: the same job")]
s = 1.9
for k, (p, par) in enumerate(((1, 100), (4, 25), (8, 12.5))):
    y = 136 + k * 74
    parts += [text(32, y, f"{p} processor" + ("s" if p > 1 else ""), "name"),
              bar(32, y + 10, 30 * s, 30, "danger", "30 µs"),
              bar(32 + 30 * s, y + 10, par * s, 30, "good", f"{par:g} µs" if par > 20 else "")]
    if par < 20:
        parts.append(note(32 + 30 * s + par * s + 8, y + 30, f"{par:g} µs"))
parts += [note(32, 372, "the 30 µs never shrinks, so no run", room=290),
          note(32, 390, "can be shorter than 30 µs", room=290),
          zone(348, 104, 316, 300, "Gustafson: the same time")]
parts.append(text(364, 136, "4 processors, a 10 s run", "name"))
parts.append(bar(364, 146, 26, 4 * 24 - 4, "danger", "1"))
for lane in range(4):
    parts.append(bar(394, 146 + lane * 24, 130, 20, "good", "9 s"))
parts += [text(364, 278, "the same work on 1 processor", "name"),
          bar(364, 288, 26, 30, "danger", "1")]
for k in range(4):
    parts.append(bar(394 + k * 62, 288, 58, 30, "good", "9 s"))
parts += [note(364, 350, "1 + 4 × 9 = 37 s on 1 processor", room=290),
          note(364, 370, "4 processors: 37 / 10 = 3.7 times faster", room=290),
          note(364, 390, "(bars not to scale)", room=290)]
save(OUT + "ch01-two-laws.svg", 680, 424,
     "Two panels. Left, Amdahl, the same job: a 30 microsecond serial bar and a parallel bar of 100, 25 "
     "and 12.5 microseconds on 1, 4 and 8 processors; the serial bar never shrinks. Right, Gustafson, the "
     "same time: 4 processors run 1 serial second and four 9-second bars side by side, 10 seconds in all; "
     "the same work on one processor is 1 + 4 × 9 = 37 seconds, so the speedup is 3.7.",
     "Two questions about the same kind of program", "Amdahl keeps the job fixed. Gustafson keeps the time fixed.",
     parts)


# 1.7  A three-stage pipeline over four items.
gx, cw = 142, 40
kinds = {"L": "client", "C": "good", "S": "store"}
parts = [mono(gx + t * cw + cw / 2, 102, str(t + 1), "middle", MUTED) for t in range(12)]
parts.append(text(gx - 12, 132, "one at a time", "sub", "end"))
for item in range(4):
    for st, name in enumerate("LCS"):
        t = item * 3 + st
        parts.append(bar(gx + t * cw + 2, 112, cw - 4, 30, kinds[name], f"{name}{item + 1}"))
parts.append(note(gx, 166, "4 items × 3 stages = 12 steps"))
for st, (name, label) in enumerate((("L", "load"), ("C", "compute"), ("S", "store"))):
    y = 204 + st * 40
    parts.append(tile(34, y + 2, {"L": "stack", "C": "chip", "S": "doc"}[name], kinds[name], 26))
    parts.append(text(68, y + 20, label, "sub"))
    for item in range(4):
        parts.append(bar(gx + (item + st) * cw + 2, y, cw - 4, 30, kinds[name], f"{name}{item + 1}"))
parts += [
    f'  <rect x="{gx + 2 * cw - 2}" y="196" width="{2 * cw + 4}" height="126" rx="8" fill="none"'
    f' stroke="{KINDS["outside"][0]}" stroke-width="1.6" stroke-dasharray="6 5"/>',
    tag(gx + 2 * cw, 326, "all 3 stages busy", "outside"),
    note(gx + 6 * cw + 12, 230, "pipelined: 3 steps until the"),
    note(gx + 6 * cw + 12, 248, "first item is done, then 1 more"),
    note(gx + 6 * cw + 12, 266, "item every step: 3 + 3 = 6 steps"),
    card(40, 362, 560, 58, "service", "speedup = 12 / 6 = 2",
         "with many more items it approaches 3, the number of stages"),
]
save(OUT + "ch01-pipeline.svg", 640, 444,
     "Top row, one at a time: twelve steps, L1 C1 S1 L2 C2 S2 and so on, 4 items times 3 stages. Below, "
     "pipelined: three rows for load, compute and store. Load runs L1 to L4 in steps 1 to 4, compute "
     "runs C1 to C4 in steps 2 to 5, store runs S1 to S4 in steps 3 to 6. In steps 3 and 4 all three "
     "stages are busy. Pipelined, the four items take 6 steps, a speedup of 2.",
     "Three stages, four items", "One item at a time, then the stages overlapped.", parts)


# 1.8  Flynn's classes and SIMT: one add over 8 numbers on four kinds of machine.
def instr(cx, y, body, kind="client"):
    """An instruction: a pill in the code face, 26 tall, centred on `cx`."""
    wide = width_of(body, 12, mono=True) + 26
    accent, tint = KINDS[kind]
    return (f'  <rect x="{cx - wide / 2:.1f}" y="{y}" width="{wide:.1f}" height="26" rx="13" fill="{tint}"'
            f' stroke="{accent}" stroke-width="1.4"/>\n' + mono(cx, y + 17.5, body, "middle", accent))


def cells(x, y, n, kind="store", labels=None, w=26, gap=4, faint_from=None):
    """`n` square cells in a row; `labels` are drawn inside them."""
    out = []
    for i in range(n):
        faint = faint_from is not None and i >= faint_from
        out.append(bar(x + i * (w + gap), y, w, 26, "idle" if faint else kind,
                       labels[i] if labels else ""))
    return "\n".join(out)


def centres(x, n, w=26, gap=4):
    return [x + i * (w + gap) + w / 2 for i in range(n)]


parts = []
boxes = [(16, 100, "SISD", "one instruction, one number at a time"),
         (336, 100, "SIMD", "one instruction, 8 numbers at once"),
         (16, 300, "MIMD", "two cores, each with its own program"),
         (336, 300, "SIMT", "one instruction, 8 threads")]
for x, y, name, sub in boxes:
    parts.append(zone(x, y, 304, 180, name))
    parts.append(note(x + 18, y + 32, sub, room=270))
row = 210                                        # top of the cells in the upper panels
# SISD: the instruction reaches the first number; the other seven wait their turn.
parts += [instr(63, 146, "add"), arrow(63, 172, row),
          cells(50, row, 8, faint_from=1),
          note(34, 262, "8 adds, one after another", room=270)]
# SIMD: one instruction, one wide register that holds all 8 numbers.
parts += [instr(488, 146, "add × 8"), arrow(488, 172, row - 6),
          f'  <rect x="366" y="{row - 6}" width="244" height="38" rx="9" fill="none"'
          f' stroke="{KINDS["client"][0]}" stroke-width="1.4" stroke-dasharray="5 4"/>',
          cells(371, row, 8),
          note(354, 262, "one register holds all 8 numbers", room=270)]
row = 410
# MIMD: two instructions, each fanned out to its own four numbers.
parts += [instr(98, 346, "add"), fan(98, 372, centres(40, 4), 390, row), cells(40, row, 4),
          instr(238, 346, "dot", "outside"), fan(238, 372, centres(180, 4), 390, row, "outside"),
          cells(180, row, 4, "outside"),
          note(34, 462, "each core follows its own instructions", room=270)]
# SIMT: one instruction fanned out to 8 threads, each with its own number.
parts += [instr(488, 346, "add"), fan(488, 372, centres(371, 8), 390, row),
          cells(371, row, 8, "good", [f"t{i}" for i in range(8)]),
          note(354, 462, "each thread has its own registers", room=270)]
save(OUT + "ch01-flynn.svg", 656, 500,
     "Four panels for one add over 8 numbers. SISD: one add instruction reaches the first number, and "
     "the other seven wait their turn. SIMD: one add instruction reaches a wide register that holds all "
     "8 numbers. MIMD: two cores, one running add and one running dot, each fanned out to its own four "
     "numbers. SIMT: one add instruction fanned out to 8 threads, t0 to t7, each with its own registers.",
     "One add over 8 numbers, on four kinds of machine", "Flynn's SISD, SIMD and MIMD, and NVIDIA's SIMT.",
     parts)


# 1.9  Arithmetic intensity: FLOPs per byte for vector addition and a 2 x 2 matrix multiply.
def strip(x, y, n, kind, w, label, group=0):
    """`n` small bars; with `group`, a wider gap after every `group` of them."""
    at = lambda i: x + i * (w + 2) + (i // group) * 6 if group else x + i * (w + 2)
    return "\n".join([text(x - 12, y + 17, label, "sub", "end")]
                     + [bar(at(i), y, w, 24, kind) for i in range(n)])


parts = [zone(16, 104, 608, 140, "vector addition, one row")]
parts += [strip(150, 136, 12, "store", 18, "bytes moved: 12", 4),
          strip(150, 172, 1, "danger", 18, "FLOPs: 1"),
          note(150, 224, "a[i], b[i] and c[i], 4 bytes each, for 1 add:  I = 1 / 12 ≈ 0.08", room=460)]
parts += [zone(16, 280, 608, 172, "2 × 2 matrix multiply")]
parts += [strip(150, 312, 48, "store", 6, "bytes moved: 48", 4),
          strip(150, 348, 16, "danger", 18, "FLOPs: 16"),
          note(150, 400, "3 matrices of 4 numbers, 4 bytes each; 2 multiplies and 2 adds", room=460),
          note(150, 420, "for each of the 4 results:  I = 16 / 48 = 1/3", room=460)]
parts.append(card(16, 480, 608, 58, "service", "N × N matrices: I = 2N³ / (3 × 4 × N²) = N / 6",
                  "at N = 4,096 that is about 680 FLOPs for every byte moved"))
save(OUT + "ch01-intensity.svg", 640, 562,
     "Two rows of squares. Vector addition: 12 bytes moved, drawn as 12 small squares, against 1 FLOP, "
     "an intensity of 1/12, about 0.08. A 2 by 2 matrix multiply: 48 bytes moved against 16 FLOPs, an "
     "intensity of 1/3. A card below gives the general case: for N by N matrices the intensity is N/6, "
     "about 680 at N = 4,096.",
     "Arithmetic intensity: FLOPs per byte moved", "Count the bytes that cross from memory, and the arithmetic done on them.",
     parts)


# 1.10  The roofline of a machine with 40 TFLOP/s and 1 TB/s.
px0, px1, py0, py1 = 90, 600, 380, 100
lo_i, hi_i, lo_p, hi_p = -4, 10, -4, 7          # log2 ranges: intensity 1/16..1024, TFLOP/s 1/16..128
lx = lambda i: px0 + (px1 - px0) * (math.log2(i) - lo_i) / (hi_i - lo_i)
ly = lambda v: py0 - (py0 - py1) * (math.log2(v) - lo_p) / (hi_p - lo_p)
parts = [f'  <rect x="{px0}" y="{py1}" width="{px1 - px0}" height="{py0 - py1}" fill="#FFFFFF" stroke="{EDGE}"/>']
for k in range(lo_i, hi_i + 1, 2):
    label = f"{2 ** k:g}" if k >= 0 else f"1/{2 ** -k}"
    parts += [line(lx(2 ** k), py0, lx(2 ** k), py1, EDGE, 1), mono(lx(2 ** k), py0 + 18, label, "middle", MUTED)]
for k in range(lo_p, hi_p + 1, 2):
    label = f"{2 ** k:g}" if k >= 0 else f"1/{2 ** -k}"
    parts += [line(px0, ly(2 ** k), px1, ly(2 ** k), EDGE, 1), mono(px0 - 8, ly(2 ** k) + 4, label, "end", MUTED)]
parts += [text((px0 + px1) / 2, py0 + 40, "arithmetic intensity, FLOP per byte", "sub", "middle"),
          text(px0 - 70, py1 - 14, "TFLOP/s", "sub")]
roof = [(2 ** (lo_p), 2 ** lo_p), (40, 40), (2 ** hi_i, 40)]
points = " ".join(f"{lx(i):.1f},{ly(p):.1f}" for i, p in roof)
parts += [f'  <polyline points="{points}" fill="none" stroke="{INK}" stroke-width="3"/>',
          line(lx(40), py0, lx(40), ly(40), KINDS["danger"][0], 1.4, "6 5"),
          tag(lx(40) + 8, ly(1 / 4), "ridge: 40 FLOP/byte", "danger"),
          tile(lx(1 / 4) - 66, ly(16) - 21, "stack", "store", 30),
          text(lx(1 / 4) + 28, ly(16), "memory-bound", "name", "middle"),
          text(lx(1 / 4) + 28, ly(16) + 18, "limit = I × 1 TB/s", "sub", "middle"),
          tile(lx(170) - 98, ly(40) + 9, "chip", "client", 30),
          text(lx(170), ly(40) + 30, "compute-bound", "name", "middle"),
          text(lx(170), ly(40) + 48, "limit = 40 TFLOP/s", "sub", "middle")]
for i, kind, label, dx, dy in ((1 / 12, "client", "vector add: 0.08", 10, -30), (64 / 6, "service", "matmul, N = 64", 10, 10),
                               (4096 / 6, "outside", "matmul, N = 4,096", -10, -30)):
    p = min(i, 40)
    parts.append(f'  <circle cx="{lx(i):.1f}" cy="{ly(p):.1f}" r="6" fill="{KINDS[kind][0]}" stroke="#FFFFFF" stroke-width="2"/>')
    parts.append(tag(lx(i) + dx, ly(p) + dy, label, kind, "start" if dx > 0 else "end"))
save(OUT + "ch01-roofline.svg", 640, 444,
     "A roofline on logarithmic axes. The limit rises along a diagonal, intensity times 1 TB/s, until the "
     "ridge at 40 FLOP per byte, then stays flat at 40 TFLOP/s. Vector addition sits far down the "
     "diagonal at 0.08 FLOP per byte. A 64 by 64 matrix multiply sits on the diagonal at about 11. A "
     "4,096 by 4,096 multiply sits under the flat roof at about 680.",
     "The roofline of a 40 TFLOP/s, 1 TB/s machine", "Left of the ridge memory sets the limit; right of it, arithmetic does.",
     parts)


# 1.11  Three programs, each with one resource near 100 percent.
programs = [("SHA-256 hashing", (98, 12, 3), "CPU-bound", "key"), ("vector addition", (15, 98, 2), "memory-bound", "stack"),
            ("copying 10 GB", (6, 10, 98), "I/O-bound", "doc")]
parts = []
for k, (name, use, verdict, glyph) in enumerate(programs):
    x = 16 + k * 208
    parts.append(f'  <rect x="{x}" y="100" width="196" height="196" rx="12" fill="#FFFFFF" stroke="{EDGE}" filter="url(#lift)"/>')
    parts.append(text(x + 16, 128, name, "name"))
    for r, (res, pct) in enumerate(zip(("CPU", "memory", "I/O"), use)):
        y = 148 + r * 34
        hot = pct > 90
        parts += [text(x + 16, y + 15, res, "sub"),
                  f'  <rect x="{x + 76}" y="{y + 2}" width="100" height="16" rx="5" fill="#F1F4F7"/>',
                  f'  <rect x="{x + 76}" y="{y + 2}" width="{pct}" height="16" rx="5" fill="{KINDS["danger" if hot else "idle"][0]}"/>',
                  mono(x + 176, y + 34 - 4, f"{pct}%", "end", MUTED)]
    parts.append(tag(x + 16, 264, verdict, "danger"))
    parts.append(tile(x + 196 - 16 - 30, 258, glyph, "danger", 30))
save(OUT + "ch01-bound.svg", 640, 320,
     "Three programs, each with bars for CPU, memory and I/O use. SHA-256 hashing keeps the CPU at 98 "
     "percent and is CPU-bound. Vector addition keeps memory at 98 percent and is memory-bound. Copying "
     "10 GB keeps the I/O at 98 percent and is I/O-bound.",
     "The busy resource names the bound", "Three programs, each with one resource near 100%.", parts)


# 1.12  A barrier: every processor waits for the slowest one.
gx, s = 130, 0.36                               # pixels per cycle
work = [900, 1000, 950, 1200]
parts = []
for u, w in enumerate(work):
    y = 104 + u * 44
    parts.append(text(gx - 12, y + 21, f"processor {u}", "sub", "end"))
    parts.append(tile(16, y + 3, "chip", "good", 24))
    parts.append(bar(gx, y, w * s, 30, "good", f"work {w:,}"))
    if w < 1200:
        parts.append(bar(gx + w * s, y, (1200 - w) * s, 30, "idle", "wait" if 1200 - w >= 150 else "", dashed=True))
    parts.append(bar(gx + 1200 * s, y, 100 * s, 30, "danger", "100"))
bx = gx + 1200 * s
parts += [line(bx, 96, bx, 104 + 4 * 44 - 6, KINDS["danger"][0], 2.4),
          tag(bx - 34, 72, "barrier at cycle 1,200", "danger", "end"),
          tile(bx - 26, 67, "lock", "danger", 26),
          note(gx, 300, "the barrier opens when processor 3, the slowest, arrives", room=480),
          card(40, 318, 560, 58, "danger", "useful work 4,050 of 4 × 1,300 = 5,200",
               "efficiency 4,050 / 5,200 = 0.78")]
save(OUT + "ch01-barrier.svg", 640, 400,
     "Four processors on a time line. Processor 0 works 900 cycles, processor 1 works 1,000, processor 2 "
     "works 950 and processor 3 works 1,200. A vertical barrier line stands at 1,200 cycles. Processors "
     "0, 1 and 2 wait in dashed boxes until 1,200. Then all four spend 100 cycles at the barrier. Useful "
     "work is 4,050 of 5,200 processor-cycles, an efficiency of 0.78.",
     "At a barrier, every processor waits for the slowest", "Four processors with uneven work meet at one barrier.",
     parts)


# The chapter map: five questions, in the order the chapter answers them.
questions = [
    ("client", "clock", "Why is one processor slow on vector addition?", "1.1  latency, throughput, Little's law"),
    ("service", "chip", "How much faster can p processors be?", "1.2 to 1.5  speedup, Amdahl, Gustafson"),
    ("store", "stack", "How does work divide, and on which machines?", "1.6 and 1.7  task, data, pipeline; Flynn"),
    ("outside", "curve", "Does memory or arithmetic set the limit?", "1.8  arithmetic intensity, the roofline"),
    ("danger", "lock", "What does it cost for processors to wait?", "1.9  barriers"),
]
parts, y = [], 100
for k, (kind, glyph, name, sub) in enumerate(questions):
    parts.append(card(80, y, 480, 60, kind, name, sub, glyph=glyph))
    if k < len(questions) - 1:
        parts.append(arrow(320, y + 60, y + 84))
    y += 84
save(OUT + "ch01-map.svg", 640, y + 8,
     "Five cards, one below the other, joined by arrows. 1: why is one processor slow on vector "
     "addition? Section 1.1. 2: how much faster can p processors be? Sections 1.2 to 1.5. 3: how does "
     "work divide, and on which machines? Sections 1.6 and 1.7. 4: does memory or arithmetic set the "
     "limit? Section 1.8. 5: what does it cost for processors to wait? Section 1.9.",
     "The questions of this chapter", "Each card names the sections that answer it.", parts)

print("wrote chapter 1 figures")
