"""The figures of chapter 14 (Map and SAXPY), in the style of the author's paper: small
cells with real numbers, 1 pastel colour per thread or block, captions outside the picture.

    python3 tools/draw_ch14.py

The helpers at the top are also used by tools/draw_ch15.py. The measured rates in the last 2
figures are those quoted in the chapter (Jetson Orin NX, code/oxide/ch14_map_saxpy and
code/cuda/ch14_map_saxpy).
"""
from html import escape
from pathlib import Path


from fig_arrows import materialize, sharpen

OUT = Path(__file__).resolve().parent.parent / "src" / "figures"

INK = "#1F2633"
GREY = "#8E99A8"
EDGE = "#D6DCE4"
HI = "#3F51B5"
RED = "#C2412D"
GREEN = "#2E8B57"

# name: (fill, stroke)
PAL = {
    # Material tonal pairs: (container fill, accent). Accents are the 600 shades.
    "blue": ("#E3E8FD", "#3F51B5"),
    "mint": ("#DDF3EC", "#00897B"),
    "peach": ("#FFE6D6", "#E65100"),
    "pink": ("#FCE0EA", "#C2185B"),
    "amber": ("#FFF0CC", "#B26A00"),
    "grey": ("#F1F3F6", "#B0B8C4"),
    "white": ("#FFFFFF", "#C9D0D9"),
    "red": ("#FDE3E0", "#C62828"),
    "hi": ("#3F51B5", "#283593"),
}
THREADS = ["blue", "mint", "peach", "pink", "amber"]

MARKERS = {"hi": HI, "grey": GREY, "red": RED, "ink": INK, "green": GREEN,
           "blue": PAL["blue"][1], "mint": PAL["mint"][1], "peach": PAL["peach"][1],
           "pink": PAL["pink"][1]}

STYLE = """  <style>
    text { font-family: Roboto, Inter, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
    .mono { font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace; }
  </style>"""


def defs():
    out = ["  <defs>"]
    for name, colour in MARKERS.items():
        out.append(f'    <marker id="m-{name}" viewBox="0 0 10 10" refX="10" refY="5" '
                   f'markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" '
                   f'orient="auto"><path d="M0,0.5 L10,5 L0,9.5 Z" fill="{colour}"/></marker>')
    out.append('    <filter id="elev" x="-20%" y="-20%" width="140%" height="160%">'
               '<feDropShadow dx="0" dy="1.5" stdDeviation="1.6" flood-color="#1F2633" '
               'flood-opacity="0.18"/></filter>')
    out.append("  </defs>")
    return "\n".join(out)


def T(x, y, body, size=12, fill=INK, anchor="middle", weight=400, italic=False, mono=False):
    """A line of text; `y` is the baseline."""
    style = ' font-style="italic"' if italic else ""
    cls = ' class="mono"' if mono else ""
    return (f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{style}{cls} '
            f'xml:space="preserve">{escape(str(body))}</text>')


def rect(x, y, w, h, fill="#FFFFFF", stroke=EDGE, sw=1.2, rx=5, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def cell(x, y, w, h, label=None, col="white", size=13, outline=False, dash=False,
         fill_text=INK, weight=400, faded=False, mono=False):
    """A rounded cell holding 1 value, in the Material style: a tonal fill on a soft shadow.
    `outline` marks the highlighted result with a solid primary fill and white text."""
    fill, accent = PAL[col]
    if outline:
        fill, accent, fill_text, weight = HI, HI, "#FFFFFF", 700
    shadow = "" if faded or dash else ' filter="url(#elev)"'
    stroke = accent if col in ("white", "grey") or dash else fill
    d = ' stroke-dasharray="4 3"' if dash else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"{d}{shadow}/>']
    if col not in ("white", "grey") and not outline:
        parts.append(f'  <rect x="{x:.1f}" y="{y + h - 3:.1f}" width="{w:.1f}" height="3" '
                     f'rx="1.5" fill="{accent}"/>')
    if label is not None:
        colour = "#A7B0BC" if faded else fill_text
        parts.append(T(x + w / 2, y + h / 2 + size * 0.36, label, size, colour,
                       weight=weight if not outline else 700, mono=mono))
    return "\n".join(parts)


def line(x1, y1, x2, y2, col=GREY, w=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{col}" stroke-width="{w}"{d}/>')


def arrow(x1, y1, x2, y2, col="hi", w=2.0, dash=None):
    """A straight arrow; the tip stops on (x2, y2)."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{MARKERS[col]}" stroke-width="{w}"{d} '
            f'marker-end="url(#m-{col})"/>')


def curve(d, col="hi", w=2.0, head=True, dash=None):
    """An arrow along an SVG path."""
    m = f' marker-end="url(#m-{col})"' if head else ""
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <path d="{d}" fill="none" stroke="{MARKERS[col]}" stroke-width="{w}"{dd}{m}/>')


def brace(x1, x2, y, label, size=11):
    """A bracket under a range of cells, with a label below it."""
    mid = (x1 + x2) / 2
    d = f"M{x1},{y} v5 H{x2} v-5 M{mid},{y + 5} v4"
    return "\n".join([f'  <path d="{d}" fill="none" stroke="{GREY}" stroke-width="1"/>',
                      T(mid, y + 22, label, size, GREY)])


# The title and one-line explanation printed inside each figure, as in the rest of the book.
HEADS = {
    "ch14-example.svg": ("SAXPY on 12 elements", "Each result is 2 times x plus y at the same position."),
    "ch14-sequential.svg": ("The sequential loop", "One pass per element, one after another."),
    "ch14-independence.svg": ("Which results depend on others",
                              "SAXPY uses only its own column; a running sum needs the result before it."),
    "ch14-grid.svg": ("A grid that covers 12 elements",
                      "2 blocks of 8 threads; the last 4 threads have no element."),
    "ch14-thread.svg": ("What one thread reads and writes", "2 reads, 1 write and 2 FLOP for element 9."),
    "ch14-assign.svg": ("Two ways to give 12 elements to 4 threads",
                        "Chunks spread each step over 40 bytes; a stride keeps it in 16."),
    "ch14-twice.svg": ("A stride that is too small", "Two threads update element 2, and it ends at 8, not 6."),
    "ch14-overrun.svg": ("Threads past the end of the array",
                         "Without the guard, threads 12 to 15 write into other data."),
    "ch14-intensity.svg": ("Why SAXPY waits for memory",
                           "0.17 FLOP per byte, far below the ridge point of any GPU."),
    "ch14-closure.svg": ("A closure passed to a kernel",
                         "Each closure type gets its own kernel; its captured values ride along."),
    "ch14-measured.svg": ("Measured SAXPY rates", "Every version reaches about 90% of the memory bandwidth."),
    "ch14-blocks.svg": ("Blocks per SM in a grid-stride loop",
                        "The rate follows the bytes in flight until the bandwidth caps it."),
    "ch15-example.svg": ("The dot product on 16 pairs", "The 16 products add up to 63."),
    "ch15-chain.svg": ("A running sum is a chain", "Each addition waits for the one before it: 16 steps."),
    "ch15-regroup.svg": ("The same additions in pairs", "15 additions in 4 steps instead of 16."),
    "ch15-steps.svg": ("A tree sum in shared memory",
                       "Each step adds 2 cells into 1; a barrier separates the steps."),
    "ch15-broken.svg": ("A tree sum without its barrier", "Thread 0 reads cell 2 too early: 24, not 32."),
    "ch15-invariant.svg": ("What cell 0 holds before each step",
                           "The products whose index leaves remainder 0 when divided by 2 × stride."),
    "ch15-shuffle.svg": ("A warp sum with shuffles", "The same additions as the tree, in registers."),
    "ch15-blocksum.svg": ("A block sum with shuffles", "5 shuffles per warp, 1 barrier, 5 shuffles in warp 0."),
    "ch15-combine.svg": ("Two ways to combine block sums",
                         "A second launch has a fixed order; atomic additions do not."),
    "ch15-race.svg": ("A plain addition loses an update", "Both blocks read 0, and the later write wins."),
    "ch15-order.svg": ("Float sums depend on the order", "The same 3 floats in 2 orders give 2 sums."),
    "ch15-traffic.svg": ("Bytes moved per pair", "16 when the products are stored, 8 when they stay in registers."),
    "ch15-measured.svg": ("Measured dot-product rates", "The fused kernels reach about 95% of the bandwidth."),
}
PAPER = "#FBFAF6"
DANGER = "#CF4526"
HEAD_ROOM = 78


def save(name, w, h, alt, parts):
    """Writes a figure in the book's house style: a paper card with a title and a one-line
    explanation inside the figure, then the drawing."""
    title, lede = HEADS[name]
    body = "\n".join(parts)
    total = h + HEAD_ROOM
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {total}" width="{w}" '
           f'height="{total}" role="img" aria-label="{escape(alt)}">\n{STYLE}\n{defs()}\n'
           f'  <rect width="{w}" height="{total}" rx="14" fill="{PAPER}"/>\n'
           f'{T(28, 38, title, 18, INK, "start", 700)}\n'
           f'{T(28, 60, lede, 12.5, "#5D6B7A", "start")}\n'
           f'  <g transform="translate(0,{HEAD_ROOM})">\n{body}\n  </g>\n</svg>\n')
    (OUT / name).write_text(materialize(sharpen(svg)))


def barrier(x1, x2, y):
    """A barrier across the threads of a block: a red dashed line with a label."""
    return "\n".join([
        line(x1, y, x2, y, DANGER, 1.6, "6 4"),
        f'  <rect x="{x2 + 6:.1f}" y="{y - 10:.1f}" width="62" height="20" rx="10" '
        f'fill="#FCEBE6"/>',
        T(x2 + 37, y + 4, "barrier", 11, DANGER, weight=700)])


def pill(x, y, label, size=11):
    """A dark capsule label with white capital letters, as in the rest of the book."""
    body = str(label).upper()
    width = 8.2 * len(body) * size / 11 + 24
    return "\n".join([
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="22" rx="11" fill="#1B2430"/>',
        T(x + width / 2, y + 15, body, size, "#FFFFFF", weight=700)])


GLYPH = {
    "stack": "M12 2 2 7l10 5 10-5zm-7.8 8.6L2 11.7l10 5 10-5-2.2-1.1L12 14.5zm0 4.7L2 16.4l10 5 10-5-2.2-1.1L12 19.2z",
    "chip": "M8 2h2v3h4V2h2v3h2a1 1 0 0 1 1 1v2h3v2h-3v4h3v2h-3v2a1 1 0 0 1-1 1h-2v3h-2v-3h-4v3H8v-3H6a1 1 0 0 1-1-1v-2H2v-2h3v-4H2V8h3V6a1 1 0 0 1 1-1h2zm-1 5v10h10V7z",
}


def icon_tile(x, y, glyph, col="blue"):
    """A rounded tile with an icon, like the cards of the rest of the book."""
    fill, accent = PAL[col]
    return "\n".join([
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="26" height="26" rx="7" fill="{fill}"/>',
        f'  <path transform="translate({x + 4:.1f},{y + 4:.1f}) scale(0.75)" d="{GLYPH[glyph]}" '
        f'fill="{accent}"/>'])


def bars(x0, y0, rows, scale, unit, label_w=170, h=20, gap=10, peak=None, peak_label="",
         inside=False):
    """Horizontal bars: rows of (label, value, colour). Returns parts and the bottom y. With
    `inside`, the value is written inside the right end of its bar."""
    parts = []
    y = y0
    for label, value, col in rows:
        fill, stroke = PAL[col]
        end = x0 + label_w + value * scale
        parts += [T(x0 + label_w - 10, y + h / 2 + 4, label, 11.5, INK, "end"),
                  rect(x0 + label_w, y, value * scale, h, fill, stroke, 1.1, rx=4)]
        if inside:
            parts.append(T(end - 6, y + h / 2 + 4, f"{value} {unit}", 11, INK, "end", 600))
        else:
            parts.append(T(end + 6, y + h / 2 + 4, f"{value} {unit}", 11.5, INK, "start", 600))
        y += h + gap
    if peak is not None:
        px = x0 + label_w + peak * scale
        parts += [line(px, y0 - 8, px, y - gap + 6, RED, 1.3, "4 3"),
                  T(px + 30, y - gap + 20, peak_label, 11, RED, "end")]
    return parts, y


# The worked example of the chapter.
X = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5]
Y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
A = 2
Z = [A * x + y for x, y in zip(X, Y)]


def example():
    cw, ch, pitch = 34, 28, 38
    x0, w = 96, 640
    parts = []
    for i in range(12):
        parts.append(T(x0 + i * pitch + cw / 2, 30, i, 11, GREY))
    rows = [("x", X, 40), ("y", Y, 78), ("2x + y", Z, 136)]
    for name, values, y in rows:
        parts.append(T(x0 - 14, y + 19, name, 13, INK, "end", 600, italic=True))
        for i, v in enumerate(values):
            on = i == 9
            col = "hi" if (on and name == "2x + y") else ("blue" if on else "white")
            parts.append(cell(x0 + i * pitch, y, cw, ch, v, col, outline=on and name != "2x + y"))
    parts.append(T(x0 - 14, 26 + 4, "i", 11, GREY, "end", italic=True))
    parts.append(line(x0, 122, x0 + 11 * pitch + cw, 122, EDGE, 1))
    parts.append(T(x0 + 9 * pitch + cw / 2 - 48, 196, "element 9:", 12, GREY, "end"))
    parts.append(T(x0 + 9 * pitch + cw / 2 - 40, 196, "2 · 8 + 3 = 19", 12.5, HI, "start", 600))
    save("ch14-example.svg", w, 214,
         "The worked example. Row i holds the indices 0 to 11. Row x holds 2, 7, 1, 8, 2, 8, 1, 8, "
         "2, 8, 4, 5; row y holds 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8. With a = 2 the result 2x + y "
         "is 7, 15, 6, 17, 9, 25, 4, 22, 9, 19, 13, 18. Column 9 is highlighted: 2 times 8 plus 3 "
         "is 19.", parts)


def sequential():
    cw, ch, pitch = 34, 26, 38
    x0, w = 96, 640
    parts = [T(x0 - 14, 46, "visit", 11, GREY, "end")]
    for i in range(12):
        x = x0 + i * pitch
        on = i == 9
        parts.append(cell(x, 28, cw, ch, i + 1, "hi" if on else "blue", 12, outline=on))
    parts.append(arrow(x0, 74, x0 + 11 * pitch + cw, 74, "grey", 1.1))
    parts.append(T(x0 + 11 * pitch + cw, 90, "time", 11, GREY, "end"))
    bx = x0 + 9 * pitch + cw / 2
    parts += [line(bx, 56, bx, 104, HI, 1.2, "3 3"),
              rect(x0 + 4 * pitch, 108, 7 * pitch + cw - 4, 92, "#F7F7FE", PAL["hi"][1], 1.2, 8),
              T(x0 + 4 * pitch + 14, 130, "visit 10, i = 9 (line 2 of the loop):", 12, INK,
                "start", 600),
              T(x0 + 4 * pitch + 14, 152, "read x[9] = 8 and y[9] = 3", 12, INK, "start"),
              T(x0 + 4 * pitch + 14, 172, "y[9] = 2 · 8 + 3 = 19, written back", 12, HI,
                "start", 600),
              T(x0 + 4 * pitch + 14, 191, "2 reads, 1 write, 2 FLOP", 11.5, GREY, "start"),
              T(x0 - 50, 130, "12 visits, 1 after another", 11.5, GREY, "start"),
              T(x0 - 50, 150, "each visit: 2 reads, 1 write", 11.5, GREY, "start"),
              T(x0 - 50, 170, "total: 24 FLOP, 144 bytes", 11.5, GREY, "start")]
    save("ch14-sequential.svg", w, 216,
         "The sequential loop visits the 12 elements in order, 1 to 12 along a time axis. Visit "
         "10 handles element 9: it reads x[9] = 8 and y[9] = 3 and writes y[9] = 19, with 2 reads, "
         "1 write and 2 floating-point operations. In total 12 visits make 24 FLOP and move 144 "
         "bytes.", parts)


def independence():
    cw, ch, pitch = 34, 26, 38
    x0, w = 96, 640
    parts = [T(20, 22, "SAXPY: each result comes from its own column", 12.5, INK, "start", 700)]
    for i in range(12):
        x = x0 + i * pitch
        col = THREADS[i % 4]
        parts += [cell(x, 34, cw, ch, Y[i], "white", 12),
                  arrow(x + cw / 2, 61, x + cw / 2, 80, "grey", 1.1),
                  cell(x, 82, cw, ch, Z[i], col, 12)]
    parts += [T(x0 - 12, 51, "y", 12, INK, "end", 600, italic=True),
              T(x0 - 12, 99, "2x + y", 12, INK, "end", 600, italic=True),
              T(x0, 130, "12 results in 1 step: no arrow runs between columns", 11.5, GREEN,
                "start", 600)]
    sums, s = [], 0
    for v in Y:
        s += v
        sums.append(s)
    parts.append(T(20, 168, "For contrast, a running sum: each result needs the one before",
                   12.5, INK, "start", 700))
    for i in range(12):
        x = x0 + i * pitch
        parts += [cell(x, 180, cw, ch, Y[i], "white", 12),
                  arrow(x + cw / 2, 207, x + cw / 2, 226, "grey", 1.1),
                  cell(x, 228, cw, ch, sums[i], "red" if i else "grey", 12)]
        if i < 11:
            parts.append(line(x + cw, 241, x + pitch, 241, RED, 1.6))
    parts += [T(x0 - 12, 197, "y", 12, INK, "end", 600, italic=True),
              T(x0 - 12, 245, "sum", 12, INK, "end", 600, italic=True),
              T(x0, 276, "12 results in 12 steps: a chain", 11.5, RED, "start", 600)]
    save("ch14-independence.svg", w, 290,
         "Top: SAXPY on the example. Each result 7, 15, 6, ... 18 comes from its own column only, "
         "so all 12 can be computed in 1 step. Bottom, for contrast: the running sum of y, 3, 4, "
         "8, 9, 14, 23, 25, 31, 36, 39, 44, 52, where each result needs the previous one, a chain "
         "of 12 steps.", parts)


def grid():
    cw, ch, pitch = 30, 26, 33
    x0, gap, w = 64, 22, 640
    parts = []
    for b in range(2):
        bx = x0 + b * (8 * pitch + gap)
        col = THREADS[b]
        parts += [rect(bx - 8, 22, 8 * pitch + 13, 70, PAL[col][0], PAL[col][1], 1.2, 9,
                       opacity=0.55),
                  T(bx + 4 * pitch - 4, 40, f"block {b}", 11.5, INK, weight=600)]
        for t in range(8):
            i = b * 8 + t
            x = bx + t * pitch
            on = (b, t) == (1, 1)
            out = i >= 12
            parts.append(cell(x, 52, cw, ch, f"t{t}", "white" if out else col, 11,
                              outline=on, dash=out, faded=out))
            parts.append(T(x + cw / 2, 108, i, 11, RED if out else GREY))
            if not out:
                parts.append(arrow(x + cw / 2, 112, x + cw / 2, 132, "grey", 1))
                parts.append(cell(x, 134, cw, ch, Y[i], "white", 11))
            else:
                parts.append(T(x + cw / 2, 152, "—", 12, RED))
    parts += [T(x0 - 14, 112, "i =", 11, GREY, "end"),
              T(x0 - 14, 152, "y[i]", 11, GREY, "end", italic=True),
              T(626, 190, "threads 12 to 15: i ≥ n = 12, no element", 11.5, RED, "end"),
              T(x0, 190, "block 1, thread 1:  i = 1 · 8 + 1 = 9", 12, HI, "start", 600)]
    save("ch14-grid.svg", w, 206,
         "The grid for n = 12 with blocks of 8 threads: 2 blocks, 16 threads, indices i = 0 to 15. "
         "Threads with i from 0 to 11 each own 1 element of y. Threads 12 to 15, the last 4 of "
         "block 1, are dashed: i is at least n and they have no element. Thread 1 of block 1 is "
         "highlighted: i = 1 times 8 plus 1 = 9.", parts)


def one_thread():
    cw, ch, pitch = 30, 24, 34
    x0, w = 96, 640
    parts = [T(x0 - 14, 36, "x", 13, INK, "end", 600, italic=True),
             T(x0 - 14, 216, "y", 13, INK, "end", 600, italic=True),
             T(20, 128, "global", 11, GREY, "start"), T(20, 142, "memory", 11, GREY, "start")]
    for i in range(12):
        on = i == 9
        parts.append(cell(x0 + i * pitch, 18, cw, ch, X[i], "blue" if on else "white", 12,
                          outline=on))
        parts.append(cell(x0 + i * pitch, 198, cw, ch, Y[i], "blue" if on else "white", 12,
                          outline=on))
    cx = x0 + 9 * pitch + cw / 2
    bx, bw = cx - 95, 190
    parts += [rect(bx, 84, bw, 72, "#F7F7FE", HI, 1.4, 8),
              T(cx, 104, "block 1, thread 1, i = 9", 11.5, GREY),
              T(cx, 126, "2 · 8 + 3 = 19", 14, HI, weight=700),
              T(cx, 146, "2 FLOP in registers", 11, GREY),
              arrow(cx, 43, cx, 83, "hi", 1.4),
              T(cx - 8, 68, "read x[9] = 8", 11, HI, "end"),
              arrow(cx - 30, 197, cx - 30, 157, "hi", 1.4),
              T(cx - 38, 184, "read y[9] = 3", 11, HI, "end"),
              arrow(cx + 30, 157, cx + 30, 196, "red", 1.4),
              T(cx + 38, 184, "write y[9] = 19", 11, RED, "start"),
              T(x0, 250, "2 reads of 4 bytes, 1 write of 4 bytes: 12 bytes for 2 FLOP", 12, INK,
                "start", 600)]
    save("ch14-thread.svg", w, 266,
         "The work of thread 1 of block 1. It reads x[9] = 8 and y[9] = 3 from global memory, "
         "computes 2 times 8 plus 3 = 19 in registers with 2 floating-point operations, and "
         "writes 19 to y[9]: 12 bytes moved for 2 FLOP.", parts)


def assignments():
    cw, ch, pitch = 30, 22, 34
    x0, w = 108, 640
    parts = []
    panels = [("chunks: thread t takes 3t, 3t + 1, 3t + 2", lambda t, k: 3 * t + k, 22),
              ("stride 4: thread t takes t, t + 4, t + 8", lambda t, k: t + 4 * k, 168)]
    for title, owner_of, top in panels:
        parts.append(T(20, top + 4, title, 12, INK, "start", 700))
        owner = {}
        for t in range(4):
            for k in range(3):
                owner[owner_of(t, k)] = t
        for k in range(3):
            y = top + 16 + k * 30
            parts.append(T(x0 - 12, y + 15, f"step {k}", 11, GREY, "end"))
            touched = {owner_of(t, k) for t in range(4)}
            for i in range(12):
                on = i in touched
                parts.append(cell(x0 + i * pitch, y, cw, ch, i,
                                  THREADS[owner[i]] if on else "white", 11, faded=not on))
            xs = sorted(touched)
            span = (xs[-1] - xs[0] + 1) * 4
            parts.append(T(x0 + 12 * pitch + 6, y + 15, f"{span} bytes", 11,
                           GREEN if span == 16 else RED, "start", 600))
    lx = x0
    for t in range(4):
        parts += [rect(lx, 284, 14, 14, *PAL[THREADS[t]], rx=3),
                  T(lx + 20, 295, f"thread {t}", 11, INK, "start")]
        lx += 92
    save("ch14-assign.svg", w, 310,
         "2 ways to give 12 elements to 4 threads, shown as 3 steps. Top, chunks: in step 0 the "
         "4 threads touch elements 0, 3, 6 and 9, spread over 40 bytes. Bottom, stride 4: in step "
         "0 they touch elements 0 to 3, then 4 to 7, then 8 to 11, each step 16 neighboring "
         "bytes. Colors mark threads 0 to 3.", parts)


def twice():
    w = 640
    x0 = 150
    parts = [T(x0 - 12, 50, "global thread 0", 11.5, INK, "end"),
             T(x0 - 12, 64, "(block 0, t0)", 10.5, GREY, "end"),
             T(x0 - 12, 112, "global thread 2", 11.5, INK, "end"),
             T(x0 - 12, 126, "(block 1, t0)", 10.5, GREY, "end")]
    seq0 = [0, 2, 4, 6, 8, 10]
    seq2 = [2, 4, 6, 8, 10]
    bw, gap = 56, 6
    for k, i in enumerate(seq0):
        x = x0 + k * (bw + gap)
        on = i == 2
        parts.append(cell(x, 40, bw, 26, f"i = {i}", "hi" if on else "blue", 11, outline=on))
    for k, i in enumerate(seq2):
        x = x0 + (k + 1.5) * (bw + gap)
        on = i == 2
        parts.append(cell(x, 102, bw, 26, f"i = {i}", "red" if on else "mint", 11,
                          outline=False))
    ax = x0 + 1 * (bw + gap) + bw / 2
    bx = x0 + 1.5 * (bw + gap) + bw / 2
    parts += [arrow(x0, 160, x0 + 6 * (bw + gap), 160, "grey", 1.1),
              T(x0 + 6 * (bw + gap), 176, "time", 11, GREY, "end"),
              T(ax, 30, "y[2] = 2 · 1 + 4 = 6", 11.5, HI, weight=600),
              T(bx + 40, 146, "y[2] = 2 · 1 + 6 = 8", 11.5, RED, "start", 600),
              T(x0, 206, "With threads = blockDim.x = 2, global threads 0 and 2 start 2 apart", 12,
                INK, "start"),
              T(x0, 224, "and both visit elements 2, 4, 6, 8 and 10. Each visit adds 2x again:",
                12, INK, "start"),
              T(x0, 248, "y[2] = 8 in place of 6", 13, RED, "start", 700)]
    save("ch14-twice.svg", w, 266,
         "A grid-stride loop with the wrong stride, 2 in place of 4. Global thread 0 visits "
         "elements 0, 2, 4, 6, 8, 10 and global thread 2 visits 2, 4, 6, 8, 10. Thread 0 computes "
         "y[2] = 2 times 1 plus 4 = 6; thread 2 later reads the 6 and computes 2 times 1 plus 6 = 8. "
         "The result is 8 in place of 6.", parts)


def overrun():
    cw, ch, pitch = 30, 26, 33
    x0, w = 60, 640
    parts = [T(x0, 24, "device memory, 4-byte slots", 11, GREY, "start")]
    for i in range(16):
        x = x0 + i * pitch + (14 if i >= 12 else 0)
        if i < 12:
            parts.append(cell(x, 34, cw, ch, Z[i], "white", 11))
        else:
            parts.append(cell(x, 34, cw, ch, "?", "red", 11))
            parts.append(arrow(x + cw / 2, 106, x + cw / 2, 62, "red", 1.3))
            parts.append(cell(x, 108, cw, ch, f"t{i - 8}", "mint", 11, dash=True))
    parts += [brace(x0, x0 + 11 * pitch + cw, 66, "y[0] to y[11]: the 48 bytes of y"),
              brace(x0 + 12 * pitch + 14, x0 + 15 * pitch + 14 + cw, 140,
                    "block 1, threads 4 to 7"),
              T(x0 + 12 * pitch + 14, 24, "after y: other data", 11, RED, "start"),
              T(x0, 200, "Without the test i < n, threads 12 to 15 compute addresses 48 to 60",
                12, INK, "start"),
              T(x0, 218, "bytes past the start of y and write 4 values into memory that does",
                12, INK, "start"),
              T(x0, 236, "not belong to y.", 12, INK, "start")]
    save("ch14-overrun.svg", w, 254,
         "Device memory as 4-byte slots. The first 12 hold y[0] to y[11]. The next 4 belong to "
         "other data. Without the test i less than n, threads 4 to 7 of block 1, global indices "
         "12 to 15, write into those 4 slots.", parts)


def intensity():
    w = 640
    parts = [T(20, 26, "1 element of SAXPY", 12.5, INK, "start", 700)]
    items = [("read x[i]", "4 B", "blue"), ("read y[i]", "4 B", "blue"),
             ("write y[i]", "4 B", "peach")]
    for k, (name, size, col) in enumerate(items):
        parts += [cell(20 + k * 92, 38, 84, 26, name, col, 11),
                  T(20 + k * 92 + 42, 80, size, 11, GREY)]
    parts += [T(300, 56, "+ 2 FLOP (1 multiply, 1 add)", 12, INK, "start"),
              T(300, 80, "2 FLOP / 12 B = 0.17 FLOP/B", 13, HI, "start", 700)]
    parts.append(T(20, 124, "FLOP per byte a GPU can do at its peak (ridge point)", 12.5, INK,
                   "start", 700))
    rows = [("SAXPY", 0.17, "hi"), ("Jetson Orin NX", 23.5, "grey"),
            ("Jetson AGX Thor", 29.7, "grey")]
    b, y = bars(20, 138, rows, 13, "FLOP/B", label_w=130)
    parts += b
    parts.append(T(20, y + 22, "Time for n = 16,777,216 on the Orin NX (102 GB/s, 2.4 TFLOP/s)",
                   12.5, INK, "start", 700))
    rows = [("201 MB of memory", 1.97, "peach"), ("33.6 MFLOP", 0.014, "blue")]
    b, y2 = bars(20, y + 36, rows, 200, "ms", label_w=130)
    parts += b
    save("ch14-intensity.svg", w, y2 + 6,
         "Top: 1 element of SAXPY reads x[i] and y[i] and writes y[i], 4 bytes each, for 2 "
         "floating-point operations: 0.17 FLOP per byte. Middle: the ridge points, 23.5 FLOP per "
         "byte for the Jetson Orin NX and 29.7 for the Jetson AGX Thor, dwarf SAXPY's 0.17. "
         "Bottom: for 16,777,216 elements on the Orin NX the memory traffic takes 1.97 ms and the "
         "arithmetic 0.014 ms.", parts)


def closures():
    w = 640
    parts = [T(20, 26, "host", 12, GREY, "start", 600), T(20, 168, "device code", 12, GREY,
                                                            "start", 600)]
    rows = [("move |x, y| a * x + y", "captures a = 2", "4 bytes", "blue", 60),
            ("|x, y| x.max(y)", "captures nothing", "0 bytes", "mint", 340)]
    for code, cap, size, col, x in rows:
        parts += [rect(x, 40, 250, 70, PAL[col][0], PAL[col][1], 1.2, 8),
                  T(x + 125, 64, code, 12, INK, mono=True),
                  T(x + 125, 86, f"{cap}: {size}", 11.5, GREY),
                  T(x + 125, 102, "1 closure type, no name", 10.5, GREY),
                  arrow(x + 125, 111, x + 125, 178, col, 1.4),
                  T(x + 133, 140, f"{size} as 1 kernel parameter", 11, INK, "start"),
                  rect(x, 180, 250, 52, "#FFFFFF", PAL[col][1], 1.4, 8),
                  T(x + 125, 202, "map::<type of the closure>", 12, INK, mono=True),
                  T(x + 125, 222, "its own compiled kernel", 11, GREY)]
    parts.append(T(20, 260, "C++ is the same: a lambda [a] __device__ (float x, float y) {...} "
                   "has its own type,", 11.5, INK, "start"))
    parts.append(T(20, 278, "and the template map<F> is instantiated once per lambda type.",
                   11.5, INK, "start"))
    save("ch14-closure.svg", w, 294,
         "2 closures on the host: move |x, y| a * x + y, which captures a = 2 (4 bytes), and "
         "|x, y| x.max(y), which captures nothing (0 bytes). Each has its own type, and the "
         "compiler builds 1 kernel map for each type; the captured bytes travel as 1 kernel "
         "parameter. C++ lambdas and the template map<F> work the same way.", parts)


# Measured on a Jetson Orin NX (run quoted in the chapter): Rust, C++ in GB/s.
MEASURED = [("saxpy", 89.9, 91.2), ("map, closure", 90.9, 90.7),
            ("loop, 12 blocks/SM", 92.1, 92.4), ("device copy", 85.0, 85.0)]
LOOP = [(1, 32.9, 33.8), (2, 59.7, 61.1), (3, 78.2, 80.5), (6, 88.3, 93.1),
        (12, 92.1, 92.4), (48, 91.9, 90.2)]


def measured():
    w = 640
    parts = []
    rows = []
    for name, rust, cpp in MEASURED:
        rows += [(f"{name}, Rust", rust, "peach"), (f"{name}, C++", cpp, "blue")]
    b, y = bars(20, 22, rows, 3.9, "GB/s", label_w=170, h=17, gap=7, peak=102,
                peak_label="102 GB/s, the memory bandwidth", inside=True)
    parts += b
    save("ch14-measured.svg", w, y + 22,
         "Measured rates on the Jetson Orin NX, Rust and C++: saxpy 89.9 and 91.2 GB/s; map with "
         "a closure or lambda 90.9 and 90.7 GB/s; grid-stride loop with 12 blocks per SM 92.1 and "
         "92.4 GB/s; the driver's device-to-device copy 85.0 GB/s in both. A dashed line marks "
         "102 GB/s.",
         parts)


def blocks_per_sm():
    w, x0, y0, h = 640, 70, 30, 200
    parts = [T(20, 20, "GB/s", 11, GREY, "start")]
    for v in (0, 25, 50, 75, 100):
        y = y0 + h - v * h / 110
        parts += [line(x0, y, 600, y, "#EEF0F3", 1), T(x0 - 8, y + 4, v, 10.5, GREY, "end")]
    peak_y = y0 + h - 102 * h / 110
    parts += [line(x0, peak_y, 600, peak_y, RED, 1.2, "4 3"),
              T(600, peak_y - 6, "102 GB/s", 10.5, RED, "end")]
    slot = (600 - x0) / len(LOOP)
    pred = []
    for k, (blocks, rust, cpp) in enumerate(LOOP):
        cx = x0 + k * slot + slot / 2
        for j, (v, col) in enumerate(((rust, "peach"), (cpp, "blue"))):
            bh = v * h / 110
            parts.append(rect(cx - 22 + j * 23, y0 + h - bh, 21, bh, *PAL[col], rx=3))
        parts += [T(cx, y0 + h + 16, blocks, 11.5, INK, weight=600)]
        little = min(102, 32.9 * blocks)
        pred.append((cx, y0 + h - little * h / 110))
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pred)
    parts += [curve(d, "hi", 1.6, head=False, dash="5 3"),
              T(x0 + (600 - x0) / 2, y0 + h + 36, "blocks of 256 threads per SM", 11.5, GREY)]
    lx = x0
    for col, name in (("peach", "Rust"), ("blue", "C++")):
        parts += [rect(lx, y0 + h + 50, 14, 14, *PAL[col], rx=3),
                  T(lx + 20, y0 + h + 61, name, 11, INK, "start")]
        lx += 80
    parts += [line(lx, y0 + h + 57, lx + 26, y0 + h + 57, HI, 1.6, "5 3"),
              T(lx + 32, y0 + h + 61, "Little's law: 16,384 B in flight per block per SM, "
                "latency 0.50 µs", 11, INK, "start")]
    save("ch14-blocks.svg", w, y0 + h + 76,
         "Measured rate of the grid-stride SAXPY against the number of blocks per SM on the "
         "Jetson Orin NX. Rust and C++: 1 block 32.9 and 33.8 GB/s, 2 blocks 59.7 and 61.1, 3 "
         "blocks 78.2 and 80.5, 6 blocks 88.3 and 93.1, 12 blocks 92.1 and 92.4, 48 blocks 91.9 "
         "and 90.2. A dashed line from Little's law rises in proportion to the blocks until it "
         "meets 102 GB/s.", parts)


def main():
    example()
    sequential()
    independence()
    grid()
    one_thread()
    assignments()
    twice()
    overrun()
    intensity()
    closures()
    measured()
    blocks_per_sm()


if __name__ == "__main__":
    main()
