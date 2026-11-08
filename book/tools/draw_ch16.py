"""The figures of chapter 16 (Part V), drawn with figkit.

    python3 tools/draw_ch16.py

Every figure uses the 4 x 4 example of the chapter: A holds 1 to 16 row by row, B the
rows (1 0 2 1) (0 1 1 2) (2 1 0 1) (1 2 1 0), and C = A B. Each thread or block has
its own colour, the same colour in every figure.
"""
from figkit import EDGE, INK, MUTED, WIRE, must_fit, save, text

OUT = "src/figures/"

A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
B = [[1, 0, 2, 1], [0, 1, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
C = [[sum(A[i][s] * B[s][j] for s in range(4)) for j in range(4)] for i in range(4)]

# (tint, accent) for the 4 threads or blocks, then the amber of B's columns.
LAV, MINT, PEACH, PINK = ("#DDE3FB", "#5B6BD6"), ("#D7F0E6", "#1F8A70"), ("#FBE3D2", "#C8693A"), \
    ("#F9DCE6", "#C2457A")
PALETTE = [LAV, MINT, PEACH, PINK]
AMBER = ("#FCEFCF", "#B7791F")
GREY = ("#F1F4F7", "#7F8EA0")
RED = "#CF4526"
DEEP = "#3B47B0"

MARKERS = """  <defs>
    <marker id="tip-ink" viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse"
      markerWidth="10" markerHeight="10" orient="auto"><path d="M0.5,1 L9,5 L0.5,9 Z" fill="#3F5873"/></marker>
    <marker id="tip-deep" viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse"
      markerWidth="10" markerHeight="10" orient="auto"><path d="M0.5,1 L9,5 L0.5,9 Z" fill="#3B47B0"/></marker>
    <marker id="tip-mint" viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse"
      markerWidth="10" markerHeight="10" orient="auto"><path d="M0.5,1 L9,5 L0.5,9 Z" fill="#1F8A70"/></marker>
  </defs>"""


def label(x, y, body, anchor="start", fill=MUTED, size=12, bold=False, mono=False, room=None):
    """A line of text; with `room` it must fit that width."""
    if room is not None:
        must_fit(body, room, size, mono=mono, bold=bold)
    font = ' font-family="JetBrains Mono, SF Mono, Menlo, Consolas, monospace"' if mono else ""
    weight = ' font-weight="650"' if bold else ""
    return (f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{font}{weight}'
            f' style="fill:{fill}" xml:space="preserve">{escape(body)}</text>')


def escape(body):
    return body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, body="", colour=None, outline=False, size=13, rx=5, dashed=False, bold=False):
    """A cell. `colour` is a (tint, accent) pair; None draws a plain white cell."""
    tint, accent = colour if colour else ("#FFFFFF", EDGE)
    ink = INK if colour else "#8794A3"
    stroke, width = (DEEP, 2.4) if outline else (accent if colour else EDGE, 1.2)
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{tint}"'
             f' stroke="{stroke}" stroke-width="{width}"{dash}/>']
    if body != "":
        must_fit(str(body), w - 4, size, bold=bold)
        parts.append(label(x + w / 2, y + h / 2 + size * 0.36, str(body), "middle", ink, size, bold))
    return "\n".join(parts)


def matrix(x, y, values, side, colour_of=lambda r, c: None, outline=None, size=13):
    """A grid of cells with top-left corner (x, y); `side` includes a 4-unit gap."""
    parts = []
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            parts.append(box(x + c * side, y + r * side, side - 4, side - 4, v, colour_of(r, c),
                             outline == (r, c), size))
    return parts


def arrow(x1, y1, x2, y2, tip="ink", colour=WIRE, width=1.6, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" fill="none" stroke="{colour}"'
            f' stroke-width="{width}" stroke-linecap="round"{dashes} marker-end="url(#tip-{tip})"/>')


def rounded(x, y, w, h, fill, stroke, rx=12, width=1.3, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"'
            f' stroke="{stroke}" stroke-width="{width}"{dash}/>')


def brace(x1, x2, y):
    """A flat brace under the span x1..x2, opening upwards."""
    mid = (x1 + x2) / 2
    return (f'  <path d="M{x1},{y} q0,8 8,8 H{mid - 6} q6,0 6,6 q0,-6 6,-6 H{x2 - 8} q8,0 8,-8"'
            f' fill="none" stroke="#8794A3" stroke-width="1.2"/>')


def product_line(cx, y, pairs, total, colours=(LAV, AMBER)):
    """The chips a1 · b1 + a2 · b2 + ... = total, centred on cx."""
    chip, gap = 26, 6
    unit = chip + 12 + chip + 18
    width = len(pairs) * unit - 18 + 22 + 34
    x = cx - width / 2
    parts = []
    for k, (a, b) in enumerate(pairs):
        parts.append(box(x, y, chip, chip, a, colours[0], size=12))
        parts.append(label(x + chip + 6, y + 18, "·", "middle", INK, 14))
        parts.append(box(x + chip + 12, y + 0, chip, chip, b, colours[1], size=12))
        x += 2 * chip + 12
        if k < len(pairs) - 1:
            parts.append(label(x + 9, y + 18, "+", "middle", INK, 13))
            x += 18
    parts.append(label(x + 11, y + 18, "=", "middle", INK, 13))
    parts.append(box(x + 22, y, 34, chip, total, LAV, size=12, bold=True))
    return parts


def figure(name, width, height, alt, title, lede, parts):
    save(OUT + name, width, height, alt, title, lede, [MARKERS] + parts)


# 16.1  The worked example and the entry c00.
side = 38
ax, bx, cx, top = 52, 250, 448, 128
parts = [label(ax + 74, top - 14, "A", "middle", MUTED, 13, True),
         label(bx + 74, top - 14, "B", "middle", MUTED, 13, True),
         label(cx + 74, top - 14, "C = A B", "middle", MUTED, 13, True),
         label(ax + 4 * side + 23, top + 78, "×", "middle", INK, 18),
         label(bx + 4 * side + 23, top + 78, "=", "middle", INK, 18)]
parts += matrix(ax, top, A, side, lambda r, c: LAV if r == 0 else None)
parts += matrix(bx, top, B, side, lambda r, c: AMBER if c == 0 else None)
parts += matrix(cx, top, [[v if (i, j) == (0, 0) else "" for j, v in enumerate(row)]
                          for i, row in enumerate(C)], side, lambda r, c: LAV if (r, c) == (0, 0) else None)
parts += product_line(330, 300, [(1, 1), (2, 0), (3, 2), (4, 1)], 11)
parts.append(label(330, 350, "row 0 of A times column 0 of B: 1 + 0 + 6 + 4 = 11", "middle", MUTED, 12))
figure("ch16-example.svg", 660, 372,
       "The 4 by 4 matrices A, B and C = A B. A holds 1 to 16 row by row. Row 0 of A, 1 2 3 4, is "
       "shaded lavender, column 0 of B, 1 0 2 1, is shaded amber, and the entry c00 = 11 is shaded in C. "
       "Below, the 4 products 1 times 1, 2 times 0, 3 times 2 and 4 times 1 are added: 1 + 0 + 6 + 4 = 11.",
       "The worked example", "c00 pairs row 0 of A with column 0 of B and adds 4 products.", parts)


# 16.2  Row-major order.
side = 36
gx, gy = 262, 106
parts = [label(gx + 70, gy - 10, "A, as we picture it", "middle", MUTED, 12)]
parts += matrix(gx, gy, A, side, lambda r, c: PALETTE[r], outline=(2, 1))
parts += [label(140, gy + 70, "row 2, column 1", "middle", DEEP, 12, True),
          label(140, gy + 88, "is stored at location", "middle", DEEP, 12),
          label(140, gy + 106, "2 · 4 + 1 = 9", "middle", DEEP, 13, True),
          arrow(206, gy + 2 * side + 16, gx - 6, gy + 2 * side + 16, "deep", DEEP)]
my = 312
parts.append(label(330, my - 14, "A, as it lies in memory", "middle", MUTED, 12))
step = 38
mx = 330 - 8 * step + 2
for k in range(16):
    r, c = divmod(k, 4)
    parts.append(box(mx + k * step, my, step - 4, 34, A[r][c], PALETTE[r], (r, c) == (2, 1)))
    parts.append(label(mx + k * step + 17, my + 50, str(k), "middle", MUTED, 11))
for r in range(4):
    x1, x2 = mx + 4 * r * step, mx + (4 * r + 4) * step - 4
    parts += [brace(x1, x2, my + 58), label((x1 + x2) / 2, my + 90, f"row {r}", "middle", MUTED, 11)]
figure("ch16-rowmajor.svg", 660, 424,
       "Top: A as a 4 by 4 grid, 1 color per row; the entry 10 in row 2, column 1 is outlined. A note "
       "reads: row 2, column 1 is stored at location 2 times 4 plus 1 = 9. Bottom: the same 16 values in "
       "memory, locations 0 to 15, row 0 at 0 to 3, row 1 at 4 to 7, row 2 at 8 to 11 and row 3 at 12 to "
       "15. The value 10 is outlined at location 9.",
       "Row-major order", "The rows of A lie one after another; entry (r, c) is at r · 4 + c.", parts)


# 16.3  The order of visits of the sequential algorithm.
side = 44
gx, gy = 60, 112
parts = [label(gx + 86, gy - 12, "order of visits to C", "middle", MUTED, 12)]
parts += matrix(gx, gy, [[4 * r + c + 1 for c in range(4)] for r in range(4)], side,
                lambda r, c: ("#B9C3F5", DEEP) if (r, c) == (1, 2) else LAV)
tx = 280
parts += [label(tx, 130, "Lines 1 and 2 visit the 16 entries of C row by row:", "start", INK, 12.5,
                room=360),
          label(tx, 149, "visit 1 is c00, visit 16 is c33.", "start", INK, 12.5),
          label(tx, 186, "At visit 7 (c12), line 5 runs inner = 0, 1, 2, 3:", "start", INK, 12.5,
                room=360),
          label(tx, 208, "c12 = 5·2 + 6·1 + 7·0 + 8·1 = 24", "start", DEEP, 13, True, room=360),
          label(tx, 248, "In all: 16 visits × 4 steps = 64 multiply-adds,", "start", MUTED, 12.5,
                room=360),
          label(tx, 267, "performed one after another.", "start", MUTED, 12.5)]
figure("ch16-sequential.svg", 660, 316,
       "C as a 4 by 4 grid whose cells hold the order in which the sequential algorithm visits them, 1 "
       "to 16 row by row. Visit 7, entry c12, is highlighted. The text says that at visit 7 line 5 runs "
       "for inner = 0 to 3 and computes c12 = 5 times 2 + 6 times 1 + 7 times 0 + 8 times 1 = 24, and that "
       "16 visits of 4 steps make 64 multiply-adds, performed one after another.",
       "The sequential algorithm", "Matrix-Multiply fills C 1 entry at a time.", parts)


# 16.4  Host and device: the 5 steps, and the system.
steps = [("Host", "creates A and B", ("#FFFFFF", "#7F8EA0"), "host"),
         ("Copy in", "A, B to the GPU", LAV, "device"),
         ("Launch", "threads compute C", ("#E9E2FA", "#7A4FD0"), "device"),
         ("Copy out", "C to the host", LAV, "device"),
         ("Verify", "compare on host", MINT, "host")]
parts = []
w, gap, x0 = 114, 10, 27
for k, (name, sub, (tint, accent), side_name) in enumerate(steps):
    x = x0 + k * (w + gap)
    parts += [rounded(x, 96, w, 58, tint, accent, 10),
              label(x + w / 2, 120, name, "middle", INK, 13, True, room=w - 10),
              label(x + w / 2, 139, sub, "middle", MUTED, 10.5, room=w - 6),
              label(x + w / 2, 172, side_name, "middle", "#8794A3", 11)]
    if k:
        parts.append(arrow(x - gap + 1, 125, x - 2, 125))
# The system.
parts += [rounded(26, 196, 200, 220, "#FAFAF7", "#C9CED6", 14, dashed=True),
          label(126, 216, "Host", "middle", INK, 13, True),
          rounded(56, 232, 140, 56, "#FFFFFF", "#9AA6B4", 10),
          label(126, 256, "CPU", "middle", INK, 13, True),
          label(126, 274, "runs main()", "middle", MUTED, 11),
          f'  <path d="M126,288 V340" stroke="#9AA6B4" stroke-width="1.4"/>',
          rounded(46, 340, 160, 56, "#ECEEF2", "#C9CED6", 10),
          label(126, 364, "Host memory", "middle", INK, 13, True),
          label(126, 382, "a_host, b_host", "middle", MUTED, 11, mono=True),
          rounded(318, 196, 316, 220, "#FAFAF7", "#C9CED6", 14, dashed=True),
          label(476, 216, "Device (GPU)", "middle", INK, 13, True)]
for k, name in enumerate(["SM 0", "SM 1", "SM p − 1"]):
    x = 336 + k * 98
    parts += [rounded(x, 232, 86, 56, "#FFFFFF", "#9AA6B4", 10),
              label(x + 43, 256, name, "middle", INK, 12.5, True),
              label(x + 43, 274, "runs blocks", "middle", MUTED, 11),
              f'  <path d="M{x + 43},288 V340" stroke="#9AA6B4" stroke-width="1.4"/>']
parts += [rounded(336, 340, 282, 56, "#ECEEF2", "#C9CED6", 10),
          label(477, 364, "Global memory", "middle", INK, 13, True),
          label(477, 382, "a_gpu, b_gpu, c_gpu", "middle", MUTED, 11, mono=True),
          arrow(210, 358, 332, 358), arrow(332, 380, 210, 380),
          label(271, 350, "link", "middle", INK, 11, True),
          label(271, 398, "copies only", "middle", MUTED, 11)]
figure("ch16-host-device.svg", 660, 440,
       "Top: the 5 steps of every program in a row: Host creates A and B; Copy in, A and B to the GPU; "
       "Launch, threads compute C; Copy out, C to the host; Verify, compare on the host. The first and "
       "last steps run on the host, the middle 3 on the device. Bottom: the system. The host holds a CPU "
       "running main() and the host memory with a_host and b_host. The device holds SMs 0 to p minus 1, "
       "which run blocks, and the global memory with a_gpu, b_gpu and c_gpu. A link between the 2 "
       "memories carries only explicit copies.",
       "Host and device", "Data move between the 2 memories only when the program copies them.", parts)


# 16.5  A 2-D grid of 2 x 2 blocks of 2 x 2 threads over C.
cell, inner = 44, 4
parts = [label(170, 98, "grid: 2 × 2 blocks of 2 × 2 threads over C", "middle", MUTED, 12)]
for by in range(2):
    for bx in range(2):
        tint, accent = PALETTE[2 * by + bx]
        x, y = 48 + bx * 128, 112 + by * 128
        parts += [rounded(x, y, 116, 116, "#FFFFFF", accent, 12, 1.4),
                  label(x + 58, y + 18, f"block ({by}, {bx})", "middle", accent, 11.5, True)]
        for ty in range(2):
            for tx in range(2):
                hit = (by, bx, ty, tx) == (0, 1, 1, 0)
                parts.append(box(x + 12 + tx * 48, y + 28 + ty * 42, cell, 38, f"({ty}, {tx})",
                                 (tint, accent), hit, 11.5))
parts.append(label(170, 384, "(threadIdx.y, threadIdx.x) repeats in every block", "middle", MUTED, 11.5))
tx0 = 340
parts += [rounded(tx0, 128, 296, 222, "#FFFFFF", EDGE, 12),
          label(tx0 + 20, 158, "block (0, 1), thread (1, 0)", "start", DEEP, 13, True, room=260),
          label(tx0 + 20, 186, "The block covers rows 0–1 and", "start", INK, 12.5, room=260),
          label(tx0 + 20, 204, "columns 2–3 of C.", "start", INK, 12.5),
          label(tx0 + 20, 232, "row = blockIdx.y · blockDim.y", "start", INK, 12, mono=True, room=262),
          label(tx0 + 20, 250, "      + threadIdx.y = 0 · 2 + 1 = 1", "start", INK, 12, mono=True, room=262),
          label(tx0 + 20, 272, "col = blockIdx.x · blockDim.x", "start", INK, 12, mono=True, room=262),
          label(tx0 + 20, 290, "      + threadIdx.x = 1 · 2 + 0 = 2", "start", INK, 12, mono=True, room=262),
          label(tx0 + 20, 322, "so this thread computes c12 = 24", "start", DEEP, 12.5, True, room=260)]
figure("ch16-grid.svg", 660, 404,
       "A grid of 2 by 2 blocks over the 4 by 4 matrix C, 1 color per block: block (0, 0) lavender, "
       "(0, 1) mint, (1, 0) peach and (1, 1) pink. Each block holds 4 threads labeled (threadIdx.y, threadIdx.x) from "
       "(0, 0) to (1, 1). Thread (1, 0) of block (0, 1) is outlined. The panel on the right: the block "
       "covers rows 0 to 1 and columns 2 to 3; row = 0 times 2 + 1 = 1, col = 1 times 2 + 0 = 2, so the "
       "thread computes c12 = 24.",
       "Threads and blocks in 2 dimensions", "A thread's row and column come from its block and its position.",
       parts)


# 16.6  1 dimension: 12 elements, 3 blocks of 4 threads.
parts = []
step, x0, y0 = 46, 52, 132
for b in range(3):
    tint, accent = PALETTE[b]
    xb = x0 + b * (4 * step + 14)
    parts.append(label(xb + 2 * step - 2, y0 - 16, f"block {b}", "middle", accent, 12, True))
    for t in range(4):
        x = xb + t * step
        parts += [box(x, y0, step - 6, 34, str(t), (tint, accent), (b, t) == (2, 1), 12),
                  label(x + (step - 6) / 2, y0 + 54, str(4 * b + t), "middle",
                        DEEP if (b, t) == (2, 1) else MUTED, 12, (b, t) == (2, 1))]
parts += [label(x0 - 14, y0 + 54, "index", "end", MUTED, 12),
          label(330, y0 + 92, "block 2, thread 1:  index = 2 · 4 + 1 = 9", "middle", DEEP, 13, True)]
figure("ch16-1d.svg", 660, 260,
       "12 elements in 3 blocks of 4 threads, 1 color per block. Each block's threads are labeled "
       "threadIdx.x = 0 to 3, and below them the element index runs from 0 to 11. Thread 1 of block 2 "
       "is outlined, above index 9, and a line reads: block 2, thread 1: index = 2 times 4 plus 1 = 9.",
       "Threads and blocks in 1 dimension", "A thread handles element blockIdx.x · blockDim.x + threadIdx.x.", parts)


# 16.7  3 dimensions: a 4 x 2 x 4 volume in blocks of 2 x 2 x 2.
s, ox, oy = 40, 70, 236
dx, dy = 0.55 * s, -0.42 * s


def point(x, y, z):
    return ox + x * s + z * dx, oy + y * s + z * dy


def face(corners, fill, stroke, width=1.1):
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in corners)
    return f'  <polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def shade(colour, amount):
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{int(v * amount):02x}" for v in (r, g, b))


def block_colour(x, z):
    return PALETTE[2 * (z // 2) + x // 2]


parts = []
hit = (3, 0, 2)
for z in range(3, -1, -1):          # top face, back to front
    for x in range(4):
        tint, accent = block_colour(x, z)
        on = (x, 0, z) == hit
        parts.append(face([point(x, 0, z), point(x + 1, 0, z), point(x + 1, 0, z + 1), point(x, 0, z + 1)],
                          "#B9C3F5" if on else shade(tint, 0.97), DEEP if on else "#FFFFFF", 2 if on else 1.1))
for z in range(3, -1, -1):          # right face, x = 4
    for y in range(2):
        tint, accent = block_colour(3, z)
        on = (3, y, z) == hit
        parts.append(face([point(4, y, z), point(4, y, z + 1), point(4, y + 1, z + 1), point(4, y + 1, z)],
                          "#9EABEF" if on else shade(tint, 0.88), DEEP if on else "#FFFFFF", 2 if on else 1.1))
for y in range(2):                  # front face, z = 0
    for x in range(4):
        tint, accent = block_colour(x, 0)
        parts.append(face([point(x, y, 0), point(x + 1, y, 0), point(x + 1, y + 1, 0), point(x, y + 1, 0)],
                          tint, "#FFFFFF"))
parts += [arrow(ox, oy + 2 * s + 18, ox + 4 * s, oy + 2 * s + 18), label(ox + 2 * s, oy + 2 * s + 36, "x",
                                                                         "middle", INK, 13, True),
          arrow(ox - 18, oy, ox - 18, oy + 2 * s), label(ox - 30, oy + s + 4, "y", "middle", INK, 13, True),
          arrow(*point(4.25, 2, 0), *point(4.25, 2, 4)), label(*point(4.5, 2.05, 2.2), "z", "start", INK, 13,
                                                                 True)]
tx0 = 372
parts += [label(tx0, 128, "4 × 2 × 4 volume", "start", INK, 12.5, True, room=270),
          label(tx0, 146, "blocks of 2 × 2 × 2, 1 color per block", "start", MUTED, 12, room=270),
          label(tx0, 170, "outlined: block (1, 0, 1),", "start", MUTED, 12, room=270),
          label(tx0, 188, "thread (1, 0, 0), written as (x, y, z)", "start", MUTED, 12, room=270),
          label(tx0, 214, "x = 1 · 2 + 1 = 3", "start", DEEP, 12.5, mono=True),
          label(tx0, 236, "y = 0 · 2 + 0 = 0", "start", DEEP, 12.5, mono=True),
          label(tx0, 258, "z = 1 · 2 + 0 = 2", "start", DEEP, 12.5, mono=True)]
figure("ch16-3d.svg", 660, 380,
       "A box of 4 by 2 by 4 cells drawn in perspective, cut into 4 blocks of 2 by 2 by 2 threads, 1 "
       "color per block. Axes: x to the right, y downwards, z into the page. The cell of thread (1, 0, 0) "
       "in block (1, 0, 1) is outlined on the top and right faces. The text computes its position: "
       "x = 1 times 2 + 1 = 3, y = 0 times 2 + 0 = 0, z = 1 times 2 + 0 = 2.",
       "Threads and blocks in 3 dimensions", "Each of the 3 coordinates is computed on its own axis.", parts)


# 16.8  8 blocks on 2 SMs that hold 2 blocks each.
parts = [label(92, 118, "blocks waiting", "middle", MUTED, 11.5)]
cols = [LAV, MINT, PEACH, PINK, LAV, MINT, PEACH, PINK]
for k in range(8):
    parts.append(box(26 + k * 17, 130, 13, 26, "", cols[k], rx=3))
parts.append(arrow(172, 143, 214, 143))
for sm in range(2):
    y = 104 + sm * 94
    parts += [rounded(220, y, 414, 78, "#FFFFFF", EDGE, 12),
              label(250, y + 44, f"SM {sm}", "middle", INK, 13, True)]
    for wave in range(2):
        for slot in range(2):
            b = 4 * wave + 2 * slot + sm
            parts.append(box(290 + wave * 172, y + 10 + slot * 30, 156, 26, f"block {b}", cols[b], size=11.5))
parts += [arrow(290, 302, 628, 302), label(632, 316, "time", "end", MUTED, 11),
          label(368, 322, "first wave", "middle", MUTED, 11.5),
          label(540, 322, "second wave", "middle", MUTED, 11.5)]
figure("ch16-waves.svg", 660, 346,
       "8 blocks wait on the left. 2 SMs each hold 2 blocks at a time. In the first wave SM 0 runs "
       "blocks 0 and 2 and SM 1 runs blocks 1 and 3; in the second wave SM 0 runs blocks 4 and 6 and "
       "SM 1 runs blocks 5 and 7. A time axis runs underneath.",
       "Scheduling blocks", "8 blocks on 2 SMs that hold 2 blocks each: 2 waves.", parts)


# 16.9  Threads of 1 block cooperate; blocks do not.
parts = []
for k, (tint, accent) in enumerate([LAV, MINT]):
    x = 40 + k * 330
    parts += [rounded(x, 98, 250, 176, tint, accent, 14, 1.4),
              label(x + 125, 120, f"block {k}", "middle", INK, 13, True)]
    for t in range(4):
        cxp = x + 44 + t * 54
        parts += [f'  <circle cx="{cxp}" cy="148" r="12" fill="#FFFFFF" stroke="{accent}" stroke-width="1.4"/>',
                  arrow(cxp, 162, cxp, 190, "mint", "#1F8A70")]
    parts += [rounded(x + 22, 194, 206, 30, "#C9EBDD", "#1F8A70", 8),
              label(x + 125, 214, "shared memory of the block", "middle", "#1F8A70", 11.5, True, room=196),
              f'  <path d="M{x + 22},246 H{x + 228}" stroke="{INK}" stroke-width="2.2"/>',
              label(x + 125, 262, "barrier: all 4 threads of the block", "middle", INK, 10.5, room=204)]
parts += [f'  <path d="M292,186 H368" stroke="{RED}" stroke-width="1.8" stroke-dasharray="6 5"/>',
          label(330, 178, "×", "middle", RED, 18, True),
          rounded(40, 304, 580, 40, "#ECEEF2", "#C9CED6", 10),
          label(330, 329, "global memory: the only memory that threads of different blocks share", "middle",
                INK, 12, room=560),
          arrow(165, 276, 165, 302), arrow(495, 276, 495, 302)]
figure("ch16-cooperation.svg", 660, 368,
       "2 blocks, each with 4 threads drawn as circles. In each block, arrows run from the 4 threads to "
       "the shared memory of that block, and a line underneath marks a barrier for the 4 threads of the "
       "block. A red dashed line with a cross between the 2 blocks shows that they share neither shared "
       "memory nor a barrier. Both blocks reach global memory below, the only memory that threads of "
       "different blocks share.",
       "The block is the unit of cooperation", "Threads of 1 block share shared memory and a barrier.", parts)


# 16.10  The memories of the device.
parts = []
for k, (name, blocks) in enumerate([("SM 0", "block (0, 0), block (1, 0)"),
                                     ("SM 1", "block (0, 1), block (1, 1)"), ("SM p − 1", "…")]):
    x = 40 + k * 200
    parts += [rounded(x, 96, 180, 196, "#FFFFFF", EDGE, 12),
              label(x + 90, 118, name, "middle", INK, 13, True)]
    for r in range(2):
        for c in range(8):
            parts.append(box(x + 20 + c * 18, 128 + r * 16, 14, 12, "", ("#E6E9FB", "#C5CCF3"), rx=3))
    parts += [label(x + 90, 172, "arithmetic units", "middle", MUTED, 11),
              rounded(x + 14, 180, 152, 36, "#FCEFCF", "#E3B55A", 8),
              label(x + 90, 196, "registers", "middle", "#8A5A00", 12, True),
              label(x + 90, 210, "1 thread's variables", "middle", "#8A5A00", 10.5, room=146),
              rounded(x + 14, 222, 152, 36, "#D7F0E6", "#7CC6AA", 8),
              label(x + 90, 238, "shared memory", "middle", "#1F6F5A", 12, True),
              label(x + 90, 252, "1 block's tiles", "middle", "#1F6F5A", 10.5),
              label(x + 90, 280, blocks, "middle", DEEP, 11, room=166),
              f'  <path d="M{x + 90},292 V318" stroke="#9AA6B4" stroke-width="1.4"/>']
parts += [rounded(40, 318, 580, 34, "#F1F2F5", "#D5DBE3", 8),
          label(330, 340, "L2 cache: recently used values of global memory, shared by all SMs", "middle", INK,
                11.5, room=560),
          f'  <path d="M330,352 V370" stroke="#9AA6B4" stroke-width="1.4"/>',
          rounded(40, 370, 580, 48, "#E4E7EC", "#C9CED6", 8),
          label(330, 390, "Global memory (device DRAM)", "middle", INK, 13, True),
          label(330, 407, "A, B and C, visible to every thread", "middle", MUTED, 11)]
figure("ch16-memories.svg", 660, 440,
       "3 SM cards side by side: SM 0, SM 1 and SM p minus 1. Each holds a grid of arithmetic units, "
       "registers that hold 1 thread's variables, and shared memory that holds 1 block's tiles. Blue labels "
       "show 1 possible placement: blocks (0, 0) and (1, 0) on SM 0, blocks (0, 1) and (1, 1) on SM 1. "
       "All SMs connect to an L2 cache shared by all SMs, which holds recently used values of global "
       "memory, and below it to global memory, the device DRAM, which holds A, B and C.",
       "The memories of the device", "Registers and shared memory sit in each SM; global memory is shared.",
       parts)


# 16.11  The work of 1 thread of the direct kernel.
side = 38
ax, bx, cx, top = 52, 250, 448, 112
parts = [label(ax + 74, top - 14, "A", "middle", MUTED, 13, True),
         label(bx + 74, top - 14, "B", "middle", MUTED, 13, True),
         label(cx + 74, top - 14, "C", "middle", MUTED, 13, True)]
parts += matrix(ax, top, A, side, lambda r, c: LAV if r == 1 else None)
parts += matrix(bx, top, B, side, lambda r, c: AMBER if c == 2 else None)
parts += matrix(cx, top, [[24 if (i, j) == (1, 2) else "" for j in range(4)] for i in range(4)], side,
                lambda r, c: ("#B9C3F5", DEEP) if (r, c) == (1, 2) else None)
parts += [label(330, 290, "thread (1, 0) of block (0, 1): row = 1, col = 2", "middle", INK, 12.5, True),
          label(330, 312, "sum = 5·2 + 6·1 + 7·0 + 8·1 = 24, stored in c12", "middle", DEEP, 13, True),
          label(330, 334, "8 reads from global memory, 1 write", "middle", MUTED, 12)]
figure("ch16-one-thread.svg", 660, 356,
       "Matrices A, B and C. Row 1 of A, 5 6 7 8, is shaded lavender; column 2 of B, 2 1 0 1, is shaded "
       "amber; in C only the entry c12 = 24 is filled. Text below: thread (1, 0) of block (0, 1) has "
       "row 1 and column 2; sum = 5 times 2 + 6 times 1 + 7 times 0 + 8 times 1 = 24, stored in c12; "
       "8 reads from global memory and 1 write.",
       "The work of 1 thread", "Thread (1, 0) of block (0, 1) computes c12 from global memory.", parts)


# 16.12  The same value fetched 4 times.
side = 38
ax, top = 52, 186
cxs = 400
parts = [label(ax + 74, top + 4 * side + 16, "A", "middle", MUTED, 13, True),
         label(cxs + 74, top + 4 * side + 16, "C", "middle", MUTED, 13, True)]
parts += matrix(ax, top, A, side, lambda r, c: ("#B9C3F5", DEEP) if (r, c) == (0, 1) else
                (LAV if r == 0 else None))
parts += matrix(cxs, top, [[f"c0{j}" if i == 0 else "" for j in range(4)] for i in range(4)], side,
                lambda r, c: LAV if r == 0 else None, size=11.5)
x0 = ax + side + 17
for j in range(4):
    x1 = cxs + j * side + 17
    lift = 40 + 16 * j
    parts.append(f'  <path d="M{x0 - 6 + 4 * j},{top - 2} C{x0 - 6 + 4 * j},{top - lift - 20} {x1},{top - lift - 20}'
                 f' {x1},{top - 6}" fill="none" stroke="{DEEP}" stroke-width="1.6" marker-end="url(#tip-deep)"/>')
parts += [label(300, 252, "4 separate fetches", "middle", DEEP, 12.5, True, room=170),
          label(300, 270, "of a01 = 2, 1 by each", "middle", MUTED, 11.5, room=170),
          label(300, 287, "thread of row 0 of C", "middle", MUTED, 11.5, room=170),
          rounded(40, 382, 580, 56, "#FFFFFF", EDGE, 10),
          label(330, 404, "In all: 16 threads × 8 fetches = 128 fetches from global memory,", "middle",
                INK, 12.5, room=560),
          label(330, 424, "although A and B hold only 32 distinct values.", "middle", INK, 12.5, room=560)]
figure("ch16-waste.svg", 660, 460,
       "Matrix A on the left with the value a01 = 2 highlighted, and C on the right with the 4 entries "
       "c00, c01, c02 and c03 of row 0 shaded. 4 separate curved arrows run from a01 to the 4 entries: "
       "each of the 4 threads that compute row 0 of C fetches a01 from global memory on its own. A note "
       "below: in all, 16 threads times 8 fetches = 128 fetches from global memory, although A and B "
       "hold only 32 distinct values.",
       "The same value, fetched 4 times", "In the direct kernel every thread fetches its operands itself.",
       parts)


# 16.13  Measured on the Orin NX: rate against the peak, fetch rate against DRAM.
# Numbers from code/cuda/ch16_matmul_naive and code/oxide/ch16_matmul_naive (quietest of 3 runs).
def hbar(x, y, w, h, colour, value_text, name, room=170, value_x=None):
    tint, accent = colour
    vx = value_x if value_x is not None else x + max(w, 2) + 8
    return [rounded(x, y, max(w, 2), h, tint, accent, 4),
            label(x - 10, y + h / 2 + 4, name, "end", INK, 12, room=room),
            label(vx, y + h / 2 + 4, value_text, "start", accent, 12, True)]


x0, scale = 210, 380 / 2400
parts = [label(40, 110, "Rate, n = 4,096 (n = 1,024 for the fixed width), GFLOP/s", "start", INK, 13, True,
               room=580)]
rows = [("C++, f32", 185.0, LAV), ("Rust, f32", 137.8, MINT), ("Rust, f32, fixed width", 194.0, MINT),
        ("f64, n = 2,048", 35.2, PEACH)]
for k, (name, v, col) in enumerate(rows):
    vx = x0 + 75 * scale + 8 if v < 50 else None
    parts += hbar(x0, 128 + k * 34, v * scale, 22, col, f"{v:.1f}", name, value_x=vx)
peak_x = x0 + 2400 * scale
parts += [f'  <path d="M{peak_x},{122} V{262}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="5 4"/>',
          label(peak_x, 278, "FP32 peak 2,400", "middle", RED, 11.5, True),
          f'  <path d="M{x0 + 75 * scale:.1f},{224} V{258}" stroke="{PEACH[1]}" stroke-width="1.5"'
          f' stroke-dasharray="3 3"/>',
          label(x0 + 75 * scale + 6, 272, "FP64 peak 75", "start", PEACH[1], 11.5)]
x0, scale = 210, 380 / 800
parts += [label(40, 318, "Requested fetch rate, n = 4,096, GB/s", "start", INK, 13, True, room=580)]
for k, (name, v, col) in enumerate([("C++, f32", 739.8, LAV), ("Rust, f32", 551.3, MINT)]):
    parts += hbar(x0, 336 + k * 34, v * scale, 22, col, f"{v:.1f}", name)
dram_x = x0 + 102 * scale
parts += [f'  <path d="M{dram_x:.1f},{330} V{404}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="5 4"/>',
          label(dram_x, 420, "global memory 102", "middle", RED, 11.5, True),
          label(40, 452, "At least 1 − 102 / 740 = 86% of the C++ kernel's fetches were answered by the caches.",
                "start", MUTED, 12, room=590)]
figure("ch16-measured.svg", 660, 474,
       "2 panels of horizontal bars for the direct kernel on the Orin NX. Top, the rate in GFLOP/s at n = 4,096: "
       "C++ f32 185.0, Rust f32 137.8, Rust f32 with the width fixed at compile time 194.0 (n = 1,024), and f64 "
       "35.2 at n = 2,048, against a dashed line for the FP32 peak of 2,400 and a short mark for the FP64 peak of 75. Bottom, "
       "the requested fetch rate in GB/s: C++ 739.8 and Rust 551.3, against a dashed line at 102 GB/s, the "
       "bandwidth of global memory. A note: at least 1 minus 102 / 740 = 86% of the C++ kernel's fetches were "
       "answered by the caches.",
       "The direct kernel, measured", "Far below the arithmetic peak, far above the bandwidth of memory.",
       parts)
