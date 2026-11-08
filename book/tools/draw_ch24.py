"""The figures of chapter 24, drawn with figkit and figparts.

    python3 tools/draw_ch24.py [path/to/target/images]

Every grid is computed here from the worked example of the chapter, an 8 x 8 colour image,
with the same integer arithmetic as the programs in code/oxide/ch24_capstone and
code/cuda/ch24_capstone. The times are the ones the programs printed in the run quoted in
the chapter. With a path, the script also turns the program's output images (PPM/PGM) into
the PNG montage of figure 24.13.
"""
import math
import sys

from figkit import KINDS, card, legend, wire
from figparts import EDGE, INK, MUTED, bar, line, mono, note, save, tag, text, zone

OUT = "src/figures/"

# ------------------------------------------------------------------ the worked example

N = 8
SKY, HILL, HOUSE, GROUND = (120, 150, 180), (70, 140, 80), (170, 100, 90), (120, 90, 60)
COLOURS = [("sky", SKY), ("hill", HILL), ("house", HOUSE), ("ground", GROUND)]
TAPS = [1, 4, 6, 4, 1]


def colour_at(x, y):
    if y < 2:
        return SKY
    if y >= 6:
        return GROUND
    return HOUSE if 2 <= x <= 5 else HILL


def gray_of(c):
    r, g, b = c
    return (77 * r + 150 * g + 29 * b + 128) >> 8


def clamp(i, n=N):
    return min(max(i, 0), n - 1)


RGB = [[colour_at(x, y) for x in range(N)] for y in range(N)]
GRAY = [[gray_of(c) for c in row] for row in RGB]
BINS = [0] * 256
for row in GRAY:
    for v in row:
        BINS[v] += 1
CDF = []
_c = 0
for v in range(256):
    _c += BINS[v]
    CDF.append(_c)
CMIN = next(CDF[v] for v in range(256) if BINS[v])
SPAN = N * N - CMIN
TABLE = [(max(CDF[v] - CMIN, 0) * 255 + SPAN // 2) // SPAN for v in range(256)]
LEVELS = [v for v in range(256) if BINS[v]]
EQ = [[TABLE[v] for v in row] for row in GRAY]


def row_pass(img, tile=None):
    """Row sums; `tile(y)` may return the 12 slots of a block of 8 for the broken variant."""
    out = []
    for y in range(N):
        slots = tile(y) if tile else [img[y][clamp(s - 2)] for s in range(N + 4)]
        out.append([sum(w * slots[x + k] for k, w in enumerate(TAPS)) for x in range(N)])
    return out


def col_pass(rows):
    return [[(sum(w * rows[clamp(y + k - 2)][x] for k, w in enumerate(TAPS)) + 128) >> 8
             for x in range(N)] for y in range(N)]


ROWS = row_pass(EQ)
BLUR = col_pass(ROWS)
BAD_ROWS = row_pass(EQ, lambda y: [0, 0] + EQ[y] + [0, 0])
BAD_BLUR = col_pass(BAD_ROWS)


def sobel_parts(img, x, y):
    p = lambda r, c: img[clamp(r)][clamp(c)]
    gx = (p(y - 1, x + 1) + 2 * p(y, x + 1) + p(y + 1, x + 1)) \
        - (p(y - 1, x - 1) + 2 * p(y, x - 1) + p(y + 1, x - 1))
    gy = (p(y + 1, x - 1) + 2 * p(y + 1, x) + p(y + 1, x + 1)) \
        - (p(y - 1, x - 1) + 2 * p(y - 1, x) + p(y - 1, x + 1))
    return gx, gy


def edge_of(gx, gy):
    return min(int(math.sqrt(gx * gx + gy * gy) * 0.25 + 0.5), 255)


EDGES = [[edge_of(*sobel_parts(BLUR, x, y)) for x in range(N)] for y in range(N)]

# ------------------------------------------------------------------ measurements
# From the runs quoted in the chapter (Orin NX). Milliseconds per stage, Rust then C++.
STAGE_MS = [("grayscale", 0.200, 0.183), ("histogram", 0.056, 0.055), ("table", 0.007, 0.007),
            ("apply table", 0.185, 0.192), ("blur rows", 0.301, 0.208),
            ("blur columns", 0.280, 0.210), ("sobel", 0.350, 0.225)]
COPY_IN, COPY_OUT, KERNELS = 0.201, 0.070, 1.165    # ms per image, C++ run
RATES = [("1 stream", 695.3), ("2 streams", 835.3), ("3 streams", 859.5),
         ("CPU, 8 cores", 37.7)]


# ------------------------------------------------------------------ drawing helpers

def hexcol(c):
    return "#%02X%02X%02X" % c


def cell(x, y, w, h, body="", fill="#FFFFFF", stroke=EDGE, ink=INK, size=11.5, bold=False,
         width=1.2):
    weight = ' font-weight="700"' if bold else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>']
    if body != "":
        parts.append(f'  <text x="{x + w / 2:.1f}" y="{y + h / 2 + size * 0.36:.1f}" '
                     f'text-anchor="middle" class="mono" font-size="{size}"{weight} '
                     f'style="fill:{ink}">{body}</text>')
    return "\n".join(parts)


def shade(level, top=255):
    """A gray fill for a pixel level, light enough to read; dark text on light cells."""
    v = int(248 - 98 * (1 - level / top))
    return "#%02X%02X%02X" % (v, v, v)


def grid(x, y, values, side=30, top=255, marks=None, size=11.5, title=None, shaded=True):
    """An 8 x 8 grid of numbers; `marks` maps (col, row) to a kind for an outline."""
    marks = marks or {}
    parts = []
    if title:
        parts.append(text(x, y - 10, title, "name"))
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            fill = shade(v, top) if shaded else "#FFFFFF"
            kind = marks.get((c, r))
            stroke = KINDS[kind][0] if kind else "#C9D1DB"
            parts.append(cell(x + c * side, y + r * side, side - 2, side - 2, v, fill, stroke,
                              INK, size, bool(kind), 2.2 if kind else 1.0))
    return "\n".join(parts)


def axis_labels(x, y, side=30, n=N):
    parts = []
    for k in range(n):
        parts.append(text(x + k * side + side / 2 - 1, y - 4, str(k), "note", "middle"))
        parts.append(text(x - 8, y + k * side + side / 2 + 4, str(k), "note", "end"))
    return "\n".join(parts)


# ------------------------------------------------------------------ figures

def example():
    """The 8 x 8 colour image, its 4 colours, and its gray levels."""
    side, x0, x1, y0 = 30, 52, 360, 112
    parts = [text(x0, 92, "color image, 3 bytes per pixel", "name"),
             text(x1, 92, "gray image, 1 byte per pixel", "name"),
             axis_labels(x0, y0), axis_labels(x1, y0)]
    for r in range(N):
        for c in range(N):
            parts.append(cell(x0 + c * side, y0 + r * side, side - 2, side - 2, "",
                              hexcol(RGB[r][c]), "#FFFFFF"))
    parts.append(grid(x1, y0, GRAY, side, top=144))

    y = y0 + N * side + 30
    for k, (name, col) in enumerate(COLOURS):
        r, g, b = col
        yy = y + k * 24
        parts.append(cell(x0, yy - 13, 16, 16, "", hexcol(col), "#FFFFFF"))
        parts.append(text(x0 + 26, yy, name, "name"))
        parts.append(mono(x0 + 86, yy, f"({r}, {g}, {b})"))
        sum3 = 77 * r + 150 * g + 29 * b
        parts.append(mono(x0 + 230, yy, f"({sum3:,} + 128) >> 8 = {gray_of(col)}"))
        parts.append(note(x0 + 470, yy, "16 pixels"))
    save(OUT + "ch24-example.svg", 640, y + 4 * 24 + 6,
         "Left: an 8 by 8 color image. Rows 0 and 1 are sky blue, rows 6 and 7 are brown "
         "ground, and rows 2 to 5 hold a red-brown house in columns 2 to 5 between green "
         "hill pixels in columns 0, 1, 6 and 7. Right: the gray levels of the same pixels: "
         "144 for the sky, 112 for the hill, 120 for the house and 96 for the ground. Below, "
         "the 4 colors: sky (120, 150, 180) gives (36,960 + 128) >> 8 = 144; hill (70, 140, "
         "80) gives 112; house (170, 100, 90) gives 120; ground (120, 90, 60) gives 96. Each "
         "color covers 16 pixels.",
         "The worked example", "An 8 x 8 image of 4 colors, 16 pixels each, and its gray "
         "levels.", parts)


def thread_work():
    """1 thread of the gray kernel: 3 bytes in, 1 byte out."""
    side, x0, y0 = 40, 60, 120
    parts = [text(x0, y0 - 22, "rgb: row 0 of the color image, as it sits in memory",
                  "name")]
    kinds = ["client", "service", "store", "outside"]
    for i in range(12):
        px = i // 3
        on = px == 3
        accent, tint = KINDS[kinds[px]]
        parts.append(cell(x0 + i * side, y0, side - 4, 30, SKY[i % 3], tint,
                          accent, INK, 12, on, 2.2 if on else 1.2))
        parts.append(text(x0 + i * side + 18, y0 + 46, str(i), "note", "middle"))
    for px in range(4):
        parts.append(text(x0 + px * 3 * side + 58, y0 + 64, f"pixel {px}", "note", "middle"))

    gy = y0 + 130
    parts.append(text(x0, gy - 22, "gray", "name"))
    for i in range(4):
        accent, tint = KINDS[kinds[i]]
        on = i == 3
        parts.append(cell(x0 + i * side, gy, side - 4, 30, GRAY[0][i], tint, accent, INK,
                          12, on, 2.2 if on else 1.2))
        parts.append(text(x0 + i * side + 18, gy + 46, str(i), "note", "middle"))
    ax = x0 + 9 * side + 18
    parts.append(wire([(ax, y0 + 72), (ax, gy - 46), (x0 + 3 * side + 18, gy - 46),
                       (x0 + 3 * side + 18, gy - 3)], kind="flow"))
    tx = x0 + 4 * side + 30
    parts += [mono(tx, gy + 4, "thread 3: p = 3 x 3 = 9, bytes 9, 10, 11"),
              mono(tx, gy + 24, "77 x 120 + 150 x 150 + 29 x 180 = 36,960"),
              mono(tx, gy + 44, "(36,960 + 128) >> 8 = 144 = gray[3]"),
              note(tx, gy + 66, "3 reads from global memory, 1 write; no other thread "
                   "is involved")]
    save(OUT + "ch24-thread.svg", 640, gy + 92,
         "Row 0 of the color image as 12 bytes in memory, 120, 150, 180 repeated for pixels "
         "0 to 3, one color per pixel. Thread 3 computes p = 9, reads bytes 9, 10 and 11, "
         "computes 77 times 120 plus 150 times 150 plus 29 times 180 = 36,960, adds 128 and "
         "shifts right by 8 to get 144, and writes element 3 of the gray image.",
         "The work of 1 thread of the gray kernel", "Thread i reads bytes 3i to 3i + 2 and "
         "writes byte i.", parts)


def pipeline():
    """The 7 kernels and the buffers between them, with the pattern of each."""
    w, h, gap = 268, 52, 76
    left, right, top = 34, 342, 104
    lx, rx = left + w / 2, right + w / 2
    stages_left = [
        ("store", "RGB image", "3 bytes per pixel, 6.2 MB"),
        ("client", "grayscale", "map (Chapter 14)"),
        ("service", "histogram", "histogram (Chapter 22)"),
        ("service", "equalize table", "scan of 256 counts (Chapter 21)"),
        ("client", "apply table", "map with a lookup (Chapter 14)"),
    ]
    stages_right = [
        ("model", "blur rows", "stencil with a halo (Chapter 20)"),
        ("model", "blur columns", "stencil with a halo (Chapter 20)"),
        ("model", "sobel", "stencil (Chapter 19)"),
        ("store", "edge image", "1 byte per pixel, 2.1 MB"),
    ]
    links_left = ["rgb", "gray", "bins", "table"]
    links_right = ["rows, 2 B", "blurred", "edges"]

    parts = [zone(18, 92, 302, 400, "Equalise"), zone(326, 92, 298, 324, "Blur and edges")]
    for k, (kind, name, sub) in enumerate(stages_left):
        y = top + k * gap
        parts.append(card(left, y, w, h, kind, name, sub))
        if k < len(links_left):
            parts.append(wire([(lx, y + h), (lx, y + gap - 3)], label=links_left[k]))
    for k, (kind, name, sub) in enumerate(stages_right):
        y = top + k * gap
        parts.append(card(right, y, w, h, kind, name, sub))
        if k < len(links_right):
            parts.append(wire([(rx, y + h), (rx, y + gap - 3)], label=links_right[k]))

    turn = top + 4 * gap + h / 2
    parts.append(wire([(left + w, turn), (322, turn), (322, top + h / 2),
                       (right - 3, top + h / 2)]))
    parts.append(note(334, 444, "the equalized image goes to blur rows"))
    parts.append(legend(334, 474, [("client", "map"), ("service", "global"),
                                   ("model", "neighbourhood")]))
    save(OUT + "ch24-pipeline.svg", 640, 512,
         "The pipeline in 2 columns. Equalize: the RGB image of 3 bytes per pixel goes to "
         "grayscale, a map; histogram counts the gray levels into 256 bins; equalize table "
         "scans the 256 counts; apply table maps each pixel through the table. Blur and "
         "edges: blur rows and blur columns are stencils with a halo, sobel is a stencil, "
         "and the edge image has 1 byte per pixel, 2.1 MB.",
         "The 7 kernels of the pipeline", "Each arrow is a buffer in device memory; each "
         "kernel is a pattern of Part V.", parts)


def equalise():
    """The histogram, the scan and the table of the example, and the equalised image."""
    parts = [text(40, 96, "histogram of the gray image", "name")]
    base, x0 = 238, 70
    for k, v in enumerate(LEVELS):
        x = x0 + k * 62
        parts.append(bar(x, base - BINS[v] * 7, 44, BINS[v] * 7, "service", str(BINS[v])))
        parts.append(mono(x + 22, base + 16, str(v), "middle"))
    parts.append(line(56, base, 316, base, EDGE, 1.2))
    parts.append(note(56, base + 36, "every other level of 0 to 255 has count 0"))

    ty = 300
    rows = [("level v", LEVELS, "idle"), ("count", [BINS[v] for v in LEVELS], "service"),
            ("c(v), scan", [CDF[v] for v in LEVELS], "client"),
            ("new level", [TABLE[v] for v in LEVELS], "good")]
    for r, (name, values, kind) in enumerate(rows):
        y = ty + r * 34
        parts.append(text(132, y + 19, name, "name", "end"))
        accent, tint = KINDS[kind]
        for k, value in enumerate(values):
            parts.append(cell(142 + k * 46, y, 42, 28, value, tint, accent, INK, 12))
    parts.append(mono(40, ty + 4 * 34 + 18, "new(112) = round((32 - 16) x 255 / (64 - 16))"))
    parts.append(mono(40, ty + 4 * 34 + 38, "         = (16 x 255 + 24) / 48 = 85"))
    parts.append(grid(352, 112, EQ, 34, title="equalized image"))
    save(OUT + "ch24-equalise.svg", 640, ty + 4 * 34 + 60,
         "Left top: the histogram of the gray image has 4 bars of 16 pixels at levels 96, "
         "112, 120 and 144. Left bottom: a table with level, count, cumulative count c(v) "
         "and new level: 96, 16, 16, 0; 112, 16, 32, 85; 120, 16, 48, 170; 144, 16, 64, "
         "255. new(112) = round((32 - 16) times 255 / (64 - 16)) = 85. Right: the equalized "
         "image: rows 0 and 1 are 255, rows 2 to 5 are 85 85 170 170 170 170 85 85, rows 6 "
         "and 7 are 0.",
         "Histogram equalization of the example", "4 levels, 16 pixels each, move to 0, 85, "
         "170 and 255.", parts)


def separable():
    """25 weights as the product of 2 x 5, and pixel (3, 3) both ways."""
    parts = [text(40, 96, "the 25 weights of 1 pass", "name")]
    x0, y0, s = 92, 150, 34
    parts.append(note(x0 + 5 * s + 2, y0 - 14, "b(j)"))
    parts.append(note(x0 - 40, y0 - 14, "a(i)"))
    for k, t in enumerate(TAPS):
        parts.append(cell(x0 + k * s, y0 - 30, s - 4, 24, t, KINDS["model"][1],
                          KINDS["model"][0], INK, 12))
        parts.append(cell(x0 - 40, y0 + k * s, s - 4, 26, t, KINDS["client"][1],
                          KINDS["client"][0], INK, 12))
    for r, a in enumerate(TAPS):
        for c, b in enumerate(TAPS):
            parts.append(cell(x0 + c * s, y0 + r * s, s - 4, 26, a * b, "#FFFFFF",
                              "#C9D1DB", INK, 12))
    parts.append(note(40, y0 + 5 * s + 14, "weight (i, j) = a(i) x b(j); the 25 sum to "
                                           "16 x 16 = 256"))

    tx = 330
    parts += [text(tx, 96, "pixel (3, 3), 2 passes", "name"),
              mono(tx, 122, "row pass, row 1:  4,080"),
              mono(tx, 142, "row pass, rows 2 to 5:  2,635 each"),
              mono(tx, 162, "column pass:"),
              mono(tx, 182, " 4,080 + 4 x 2,635 + 6 x 2,635"),
              mono(tx, 202, "   + 4 x 2,635 + 2,635 = 43,605"),
              mono(tx, 222, "(43,605 + 128) >> 8 = 170"),
              text(tx, 258, "the same pixel, 1 pass", "name"),
              mono(tx, 284, "sum of 25 weights x pixels = 43,605"),
              note(tx, 314, "work per pixel: 5 + 5 = 10 multiply-adds"),
              note(tx, 334, "instead of 5 x 5 = 25")]
    save(OUT + "ch24-separable.svg", 640, 372,
         "Left: the column weights 1, 4, 6, 4, 1 and the row weights 1, 4, 6, 4, 1, and "
         "their 25 products from 1 at the corners to 36 in the center, summing to 256. "
         "Right: pixel (3, 3) of the example. The row pass gives 4,080 in row 1 and 2,635 in "
         "rows 2 to 5; the column pass gives 4,080 + 15 times 2,635 = 43,605, and (43,605 + "
         "128) >> 8 = 170. 1 pass over 25 weights gives the same 43,605. 2 passes cost 10 "
         "multiply-adds per pixel instead of 25.",
         "A separable blur", "2 passes of 5 weights give the 25 weights of 1 pass.", parts)


def blur_grids():
    """Equalised image, row sums, blurred image, with pixel (3, 3) and its window."""
    side = 34
    marks_eq = {(c, r): "client" for r in range(1, 6) for c in range(1, 6)}
    marks_rows = {(3, r): "model" for r in range(1, 6)}
    marks_blur = {(3, 3): "good"}
    parts = [grid(40, 112, EQ, side, marks=marks_eq, title="equalized image"),
             grid(340, 112, ROWS, side, top=4080, marks=marks_rows, size=10,
                  title="row pass: sums, 0 to 4,080"),
             grid(40, 432, BLUR, side, marks=marks_blur, title="column pass: blurred image")]
    parts += [note(340, 450, "Blue: the 5 x 5 window of pixel (3, 3)."),
              note(340, 470, "Green: the 5 row sums that the column"),
              note(340, 488, "pass reads for (3, 3):"),
              mono(340, 512, "4,080 + 15 x 2,635 = 43,605"),
              mono(340, 532, "(43,605 + 128) >> 8 = 170"),
              note(340, 562, "The image is clamped at its border:"),
              note(340, 580, "row -1 repeats row 0, column 8 repeats"),
              note(340, 598, "column 7.")]
    save(OUT + "ch24-blur.svg", 640, 712,
         "3 grids of the example. The equalized image, with the 5 by 5 window of pixel "
         "(3, 3) outlined. The row sums: 4,080 in rows 0 and 1; 1,445, 1,785, 2,295, 2,635, "
         "2,635, 2,295, 1,785, 1,445 in rows 2 to 5; 0 in rows 6 and 7, with the 5 sums "
         "that pixel (3, 3) reads outlined. The blurred image, from 245 at the top left "
         "down to 6 at the bottom left, with pixel (3, 3) = 170 outlined.",
         "The blur of the example, pass by pass", "Each output of the column pass reads 5 "
         "outputs of the row pass.", parts)


def tile_slots(values, y, missing=False, who=True):
    parts = []
    x0, s = 40, 46
    for t, v in enumerate(values):
        halo = t < 2 or t >= N + 2
        kind = "danger" if (halo and missing) else ("store" if halo else "client")
        accent, tint = KINDS[kind]
        body = "?" if (halo and missing) else v
        parts.append(cell(x0 + t * s, y, s - 4, 30, body, tint, accent, INK, 12))
        parts.append(text(x0 + t * s + 21, y - 8, str(t), "note", "middle"))
        pixel = clamp(t - 2)
        parts.append(text(x0 + t * s + 21, y + 46, f"x={pixel}", "note", "middle"))
    return "\n".join(parts)


def tile():
    """The shared tile of row 2 for a block of 8 threads, and the windows of 2 threads."""
    slots = [EQ[2][clamp(t - 2)] for t in range(N + 4)]
    y = 122
    parts = [text(40, 92, "slot s of the tile of row 2 holds pixel x = s - 2, clamped",
                  "name"),
             tile_slots(slots, y)]
    x0, s = 40, 46
    for k, (tx, kind) in enumerate([(0, "outside"), (3, "good"), (7, "model")]):
        yy = y + 70 + k * 30
        accent = KINDS[kind][0]
        parts.append(f'  <rect x="{x0 + tx * s:.1f}" y="{yy}" width="{5 * s - 4}" '
                     f'height="20" rx="6" fill="{KINDS[kind][1]}" stroke="{accent}" '
                     f'stroke-width="1.5"/>')
        total = sum(w * slots[tx + j] for j, w in enumerate(TAPS))
        parts.append(text(x0 + tx * s + 8, yy + 14, f"thread {tx}: slots {tx} to {tx + 4}",
                          "pill", "start", accent))
        side = x0 + tx * s + 5 * s + 4
        if side + 120 > 630:
            parts.append(mono(x0 + tx * s - 8, yy + 14, f"{total:,}", "end"))
        else:
            parts.append(mono(side, yy + 14, f"sum {total:,}"))
    parts.append(legend(40, y + 178, [("store", "halo: loaded by threads 0 and 1"),
                                      ("client", "own pixels: thread t loads slot t + 2")]))
    parts.append(note(40, y + 204, "8 threads load 12 values once; the 8 sums read 40 "
                                   "values, all from shared memory."))
    save(OUT + "ch24-tile.svg", 640, y + 220,
         "12 slots of a shared tile for row 2 of the equalized image, block of 8 threads. "
         "Slots 0 and 1 hold pixel 0, value 85, the clamped left halo; slots 2 to 9 hold "
         "pixels 0 to 7: 85, 85, 170, 170, 170, 170, 85, 85; slots 10 and 11 hold pixel 7, "
         "85, the clamped right halo. Thread 0 reads slots 0 to 4 and sums 1,445; thread 3 "
         "reads slots 3 to 7 and sums 2,635; thread 7 reads slots 7 to 11 and sums 1,445.",
         "The tile of the row pass, with its halo", "A block of 8 outputs needs 8 + 4 = 12 "
         "inputs.", parts)


def halo_bug():
    """The same tile without the halo loads, and the wrong blurred pixels."""
    slots = [EQ[2][clamp(t - 2)] for t in range(N + 4)]
    y = 122
    parts = [text(40, 92, "the same tile when the halo loads are left out", "name"),
             tile_slots(slots, y, missing=True)]
    parts += [mono(40, y + 82, "thread 0, correct:  85 + 4 x 85 + 6 x 85 + 4 x 85 + 170 "
                               "= 1,445"),
              mono(40, y + 102, "thread 0, slots 0 and 1 hold 0:  0 + 0 + 6 x 85 + 4 x 85 "
                                "+ 170 = 1,020")]
    marks = {(c, r): "danger" for r in range(N) for c in range(N)
             if BAD_BLUR[r][c] != BLUR[r][c]}
    gy = y + 160
    parts.append(grid(40, gy, BAD_BLUR, 34, marks=marks, title="blurred image, halo left out"))
    wrong = len(marks)
    parts += [note(340, gy + 10, f"Red: the {wrong} of 64 pixels that differ"),
              note(340, gy + 30, "from the correct blur of figure 24.7."),
              mono(340, gy + 60, f"pixel (0, 3): {BAD_BLUR[3][0]} instead of {BLUR[3][0]}"),
              mono(340, gy + 80, f"pixel (7, 2): {BAD_BLUR[2][7]} instead of {BLUR[2][7]}"),
              note(340, gy + 110, "Only columns 0, 1, 6 and 7 can be"),
              note(340, gy + 128, "wrong: their windows reach into a"),
              note(340, gy + 146, "halo slot. Rows 6 and 7 are 0 in the"),
              note(340, gy + 164, "equalized image, so 0 in a halo slot"),
              note(340, gy + 182, "does no harm there.")]
    save(OUT + "ch24-halo-bug.svg", 640, gy + 8 * 34 + 14,
         "The tile of row 2 with slots 0, 1, 10 and 11 marked as never written. Thread 0 "
         "should sum 85 + 4 times 85 + 6 times 85 + 4 times 85 + 170 = 1,445; with 0 in "
         "slots 0 and 1 it sums 1,020. Below, the blurred image computed this way, with the "
         f"{wrong} wrong pixels outlined in red, all in columns 0, 1, 6 and 7; pixel (0, 3) "
         f"is {BAD_BLUR[3][0]} instead of {BLUR[3][0]}.",
         "A missing halo", "The 4 halo slots are never written, and 4 threads of 8 read "
         "them.", parts)


def sobel():
    """Pixel (3, 1) of the blurred image through the 2 masks, and the edge image."""
    x, y = 3, 1
    hood = [[BLUR[clamp(y + r - 1)][clamp(x + c - 1)] for c in range(3)] for r in range(3)]
    gxm = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gym = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    gx, gy = sobel_parts(BLUR, x, y)
    parts = []
    for k, (title, g, kind) in enumerate([("blurred, around (3, 1)", hood, "idle"),
                                          ("Gx mask", gxm, "client"),
                                          ("Gy mask", gym, "model")]):
        x0 = 40 + k * 196
        parts.append(text(x0, 96, title, "name"))
        accent, tint = KINDS[kind]
        for r in range(3):
            for c in range(3):
                v = g[r][c]
                on = kind == "idle" or v != 0
                fill = shade(v) if kind == "idle" else (tint if on else "#FFFFFF")
                parts.append(cell(x0 + c * 44, 108 + r * 34, 40, 30, v, fill,
                                  accent if on else EDGE, INK if on else MUTED, 12))
    m = math.sqrt(gx * gx + gy * gy)
    parts += [
        mono(40, 236, f"gx = (249 + 2 x 227 + 193) - (248 + 2 x 220 + 178) = 896 - 866 "
                      f"= {gx}"),
        mono(40, 256, f"gy = (178 + 2 x 193 + 193) - (248 + 2 x 249 + 249) = 757 - 995 "
                      f"= {gy}"),
        mono(40, 276, f"sqrt({gx} x {gx} + {gy} x {gy}) = sqrt({gx * gx + gy * gy:,}) "
                      f"= {m:.2f}"),
        mono(40, 296, f"edge = round({m:.2f} / 4) = round({m / 4:.2f}) = {EDGES[y][x]}"),
    ]
    parts.append(grid(40, 340, EDGES, 34, top=100, marks={(x, y): "good"},
                      title="edge image"))
    parts += [note(340, 352, "The strongest edges lie in rows 1 and 2,"),
              note(340, 370, "where the sky meets the house and hill,"),
              note(340, 388, "and in rows 5 and 6, where they meet"),
              note(340, 406, "the ground. Shading here runs from 0"),
              note(340, 424, "(dark) to 100 (light).")]
    save(OUT + "ch24-sobel.svg", 640, 340 + 8 * 34 + 14,
         "Top: the 3 by 3 neighborhood of pixel (3, 1) in the blurred image, 248 249 249, "
         "220 227 227, 178 193 193, beside the Gx mask, -1 0 1 with the middle row doubled, "
         "and the Gy mask, -1 -2 -1 on top and 1 2 1 at the bottom. gx = 896 - 866 = 30, "
         "gy = 757 - 995 = -238, the square root of 57,544 is 239.88, and 239.88 / 4 rounds "
         "to 60. Below: the edge image of the example, with values from 24 to 100; pixel "
         "(3, 1) = 60 is outlined.",
         "Sobel edges of the example", "gx measures the change across, gy the change "
         "down.", parts)


def race():
    """2 threads add 1 to the same shared bin without an atomic."""
    parts = []
    rows = [("thread 5", "client", [(0, 1.2, "read 7"), (1.3, 2.2, "+1"),
                                    (2.3, 3.5, "write 8")]),
            ("thread 9", "model", [(0.4, 1.6, "read 7"), (1.7, 2.6, "+1"),
                                   (3.6, 4.8, "write 8")])]
    x0, s = 140, 92
    for k, (name, kind, steps) in enumerate(rows):
        y = 108 + k * 46
        parts.append(text(x0 - 14, y + 18, name, "name", "end"))
        for a, b, label in steps:
            parts.append(bar(x0 + a * s, y, (b - a) * s - 4, 28, kind, label))
    parts.append(line(x0, 214, x0 + 5 * s, 214, EDGE, 1.2))
    parts.append(note(x0, 232, "time"))
    parts += [mono(40, 262, "bin 144 held 7; 2 pixels of level 144 were counted; the bin "
                            "now holds 8, not 9"),
              note(40, 286, "Both threads read the bin before either wrote it. "
                            "An atomic add makes read, add and write 1 step.")]
    save(OUT + "ch24-race.svg", 640, 304,
         "A timeline of 2 threads. Thread 5 reads bin 144, which holds 7, adds 1 and "
         "writes 8. Thread 9 reads the same bin before thread 5 has written it, also reads "
         "7, adds 1 and writes 8. 2 pixels were counted and the bin grew by 1.",
         "A histogram update without an atomic", "2 threads count a level of 144 at the "
         "same time.", parts)


def stages():
    """Measured time per stage, Rust and C++."""
    top, h, scale = 104, 14, 300 / max(max(r, c) for _, r, c in STAGE_MS)
    parts = [legend(170, 92, [("client", "Rust"), ("store", "C++")])]
    for k, (name, rs, cc) in enumerate(STAGE_MS):
        y = top + 10 + k * 40
        parts.append(text(150, y + h + 4, name, "name", "end"))
        for j, (ms, kind) in enumerate([(rs, "client"), (cc, "store")]):
            yy = y + j * (h + 2)
            parts.append(bar(162, yy, max(ms * scale, 2), h, kind))
            parts.append(note(168 + max(ms * scale, 2), yy + 11, f"{ms:.3f} ms"))
    end = top + 10 + 7 * 40
    parts.append(note(36, end + 10, "Sum of the 7 stages, as printed: 1.380 ms in Rust, "
                                    "1.079 ms in C++, per image of 1,920 x 1,080 pixels."))
    save(OUT + "ch24-stages.svg", 640, end + 30,
         "Paired horizontal bars of the measured time of each stage for 1 image, Rust then "
         "C++: " + "; ".join(f"{n} {r:.3f} ms and {c:.3f} ms" for n, r, c in STAGE_MS)
         + ".",
         "Time per stage", "1 image of 1,920 x 1,080 pixels on the Jetson Orin NX.", parts)


def streams():
    """The batch on 1 and on 2 streams as lanes of copies and kernels, and the rates."""
    parts = [text(36, 104, "1 stream", "name"), text(36, 178, "2 streams", "name")]
    seq = [(COPY_IN, "client"), (KERNELS, "service"), (COPY_OUT, "store")]
    scale, x0 = 110, 120

    def image(start, y, n):
        bars, at = [], start
        for ms, kind in seq:
            label = f"image {n}" if kind == "service" else ""
            bars.append(bar(x0 + at * scale, y, ms * scale - 1.5, 22, kind, label))
            at += ms
        return bars

    one = COPY_IN + KERNELS + COPY_OUT
    for k in range(3):
        parts += image(k * one, 90, k)
    # 2 streams: the kernels of consecutive images run back to back, the copies overlap.
    for k in range(3):
        parts += image(k * KERNELS, 152 + (k % 2) * 28, k)
    parts.append(legend(120, 236, [("client", f"copy in, {COPY_IN:.2f} ms"),
                                   ("service", f"7 kernels, {KERNELS:.2f} ms"),
                                   ("store", f"copy out, {COPY_OUT:.2f} ms")]))
    top = 262
    peak = max(rate for _, rate in RATES)
    for k, (name, rate) in enumerate(RATES):
        y = top + k * 32
        parts.append(text(150, y + 16, name, "name", "end"))
        kind = "idle" if name.startswith("CPU") else "good"
        parts.append(bar(162, y, max(rate / peak * 340, 2), 22, kind))
        parts.append(note(170 + max(rate / peak * 340, 2), y + 15, f"{rate:.1f} images/s"))
    save(OUT + "ch24-streams.svg", 640, top + 4 * 32 + 14,
         "Top: on 1 stream each image is copied in, processed by the 7 kernels and copied "
         "out before the next one starts. On 2 streams the copies of 1 image run while the "
         "kernels of the other run. Bottom, measured images per second for a batch of 128 "
         "images in C++: " + "; ".join(f"{n} {r:.1f}" for n, r in RATES) + ".",
         "A batch of images on streams", "The copies of 1 image overlap the kernels of "
         "another.", parts)


def montage(folder):
    """The program's real output images: 3 panels reduced to 480 x 270 pixels, and 1 panel
    of 2 crops of 60 x 68 pixels around the window, enlarged 4 times without smoothing."""
    from PIL import Image, ImageDraw, ImageFont

    open_rgb = lambda file: Image.open(f"{folder}/{file}").convert("RGB")
    crop = (400, 580, 460, 648)
    zoom = lambda file: open_rgb(file).crop(crop).resize((236, 270), Image.NEAREST)
    pair = Image.new("RGB", (480, 270), "#FBFAF6")
    pair.paste(zoom("equalised.pgm"), (0, 0))
    pair.paste(zoom("blurred.pgm"), (244, 0))

    panels = [(open_rgb("input.ppm").resize((480, 270)), "input (color)"),
              (open_rgb("equalised.pgm").resize((480, 270)), "equalised"),
              (pair, "equalized | blurred, 4 x zoom on the window"),
              (open_rgb("edges.pgm").resize((480, 270)), "edges")]
    sheet = Image.new("RGB", (2 * 480 + 30, 2 * 300 + 30), "#FBFAF6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for k, (picture, title) in enumerate(panels):
        x, y = 10 + (k % 2) * 490, 10 + (k // 2) * 305
        sheet.paste(picture, (x, y + 26))
        draw.text((x, y + 2), title, fill="#1B2430", font=font)
    sheet.save(OUT + "ch24-images.png")


if __name__ == "__main__":
    example()
    thread_work()
    pipeline()
    equalise()
    separable()
    blur_grids()
    tile()
    halo_bug()
    sobel()
    race()
    if all(r > 0 for _, r, _ in STAGE_MS):
        stages()
    if KERNELS > 0:
        streams()
    if len(sys.argv) > 1:
        montage(sys.argv[1])
