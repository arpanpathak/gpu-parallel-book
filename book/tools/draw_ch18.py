"""The figures of chapter 18 (register tiling), drawn with figkit and figparts.

    python3 tools/draw_ch18.py

Every figure uses the 4 x 4 example of chapters 16 and 17:
A = 1..16 row by row, B = (1 0 2 1)(0 1 1 2)(2 1 0 1)(1 2 1 0), C = A B.
"""
from figparts import EDGE, INK, MUTED, card, must_fit, save, text, zone

OUT = "src/figures/"

A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
B = [[1, 0, 2, 1], [0, 1, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
C = [[sum(A[i][s] * B[s][j] for s in range(4)) for j in range(4)] for i in range(4)]
assert C == [[11, 13, 8, 8], [27, 29, 24, 24], [43, 45, 40, 40], [59, 61, 56, 56]]

# 1 colour per thread: (fill, edge). Thread id = ty * 2 + tx.
THREAD = [("#DDE3FB", "#5B6BD6"), ("#D7F0E6", "#1F8A70"), ("#FBE3D2", "#C8693A"), ("#F9DCE6", "#C2457A")]
LAV, MINT, PEACH, PINK = THREAD
AMBER = ("#FBF1DC", "#B7791F")
IDLE = ("#FFFFFF", EDGE)


def cell(x, y, w, h, colours, label="", cls="mono", bold=False, lw=1.2):
    fill, edge = colours
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
             f'fill="{fill}" stroke="{edge}" stroke-width="{lw}"/>']
    if label != "":
        body = str(label)
        must_fit(body, w - 4, 12, mono=True)
        colour = MUTED if colours == IDLE else INK
        parts.append(text(x + w / 2, y + h / 2 + 4.2, body, cls + (" strong" if bold else ""), "middle",
                          colour))
    return "\n".join(parts)


def grid(x, y, values, colour_of, side=30, gap=4):
    """A matrix of cells; colour_of(i, j) gives the colours of cell (i, j)."""
    parts = []
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            parts.append(cell(x + j * (side + gap), y + i * (side + gap), side, side, colour_of(i, j), v))
    return "\n".join(parts)


def outline(x, y, w, h, colour=INK, width=2.0, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="7" fill="none" '
            f'stroke="{colour}" stroke-width="{width}"{dashes}/>')


def harrow(x1, x2, y, colour="#7F8EA0"):
    return (f'  <path d="M{x1},{y} H{x2 - 3}" fill="none" stroke="{colour}" stroke-width="1.8"'
            f' stroke-linecap="round" marker-end="url(#head)"/>')


def varrow(x, y1, y2, colour="#7F8EA0"):
    return (f'  <path d="M{x},{y1} V{y2 - 3}" fill="none" stroke="{colour}" stroke-width="1.8"'
            f' stroke-linecap="round" marker-end="url(#head)"/>')


def label(x, y, body, anchor="start", cls="note", fill=None, room=None):
    if room:
        must_fit(body, room, 12, mono="mono" in cls)
    return text(x, y, body, cls, anchor, fill)


def matrix_title(x, y, body):
    return text(x, y, body, "tab", "middle", MUTED)


# 18.1  Chapter 17's thread (0, 0) in phase 0: 2 reads from shared memory per multiply-add.
parts = [zone(16, 100, 250, 196, "shared memory, phase 0")]
parts += [matrix_title(82, 136, "TILEA"), matrix_title(200, 136, "TILEB")]
parts.append(grid(50, 148, [[1, 2], [5, 6]], lambda i, j: LAV if i == 0 else IDLE, side=30))
parts.append(grid(168, 148, [[1, 0], [0, 1]], lambda i, j: AMBER if j == 0 else IDLE, side=30))
parts += [label(82, 240, "row 0", "middle"), label(200, 240, "column 0", "middle"),
          label(36, 274, "thread (0, 0) computes c00", room=220)]
rows = [("offset = 0", "read tileA[0][0] = 1, tileB[0][0] = 1", "sum += 1 · 1"),
        ("offset = 1", "read tileA[0][1] = 2, tileB[1][0] = 0", "sum += 2 · 0")]
for r, (u, reads, op) in enumerate(rows):
    y = 120 + r * 78
    parts += [cell(290, y, 352, 62, ("#FFFFFF", EDGE)),
              text(304, y + 22, u, "name"),
              label(360, y + 22, reads, room=276),
              label(360, y + 44, op, cls="mono", fill=INK),
              text(626, y + 44, "2 reads, 1 FMA", "pill", "end", "#5B6BD6")]
parts += [card(16, 318, 304, 76, "client", "1 thread, 1 phase", ["4 reads from shared memory", "2 multiply-adds, 4 FLOP"]),
          card(338, 318, 304, 76, "store", "the block of 4 threads", ["16 reads from shared memory", "8 multiply-adds, 16 FLOP"]),
          label(16, 424, "Thread (0, 1) reads the same row 0 of tileA again, for c01: each value of tileA is",
                room=626),
          label(16, 442, "read by 2 threads, and each read feeds 1 multiply-add.", room=626)]
save(OUT + "ch18-tiled-reads.svg", 660, 462,
     "Left: tileA = 1 2 / 5 6 and tileB = 1 0 / 0 1 in shared memory during phase 0; row 0 of tileA and column 0 "
     "of tileB are highlighted. Right: thread (0, 0) of the tiled kernel of Chapter 17 at offset = 0 reads 1 and 1 and "
     "adds 1 times 1; at offset = 1 it reads 2 and 0 and adds 2 times 0: 2 reads from shared memory for each "
     "multiply-add. Below: 1 thread makes 4 reads for 2 multiply-adds, 4 FLOP, in a phase; the block of 4 "
     "threads makes 16 reads for 8 multiply-adds. Thread (0, 1) reads the same row 0 of tileA again.",
     "1 read per FLOP in the tiled kernel", "Thread (0, 0) of Chapter 17 in phase 0: 2 shared-memory reads per multiply-add.",
     parts)


# 18.2  1 thread computes c00 and c10: b00 is read once and used twice.
def small(x, y, values, colour_of, title):
    return "\n".join([matrix_title(x + 2 * 26 - 2, y - 10, title), grid(x, y, values, colour_of, side=24, gap=3)])


parts = [small(24, 116, A, lambda i, j: LAV if (i, j) in ((0, 0), (1, 0)) else IDLE, "A"),
         text(144, 172, "×", "name", "middle"),
         small(158, 116, B, lambda i, j: AMBER if (i, j) == (0, 0) else IDLE, "B"),
         text(278, 172, "=", "name", "middle"),
         small(292, 116, [[11, "", "", ""], [27, "", "", ""], ["", "", "", ""], ["", "", "", ""]],
               lambda i, j: MINT if (i, j) in ((0, 0), (1, 0)) else IDLE, "C")]
parts += [zone(420, 106, 222, 112, "registers, offset = 0"),
          label(436, 140, "a_reg", cls="mono"), cell(490, 126, 28, 24, LAV, 1), cell(522, 126, 28, 24, LAV, 5),
          label(436, 178, "b_reg", cls="mono"), cell(490, 164, 28, 24, AMBER, 1),
          label(436, 206, "3 reads from shared memory", room=196)]
parts += [cell(24, 248, 300, 34, MINT, ""), label(36, 270, "c00 += a_reg[0] · b_reg[0] = 1 · 1 = 1", cls="mono", fill=INK),
          cell(24, 290, 300, 34, MINT, ""), label(36, 312, "c10 += a_reg[1] · b_reg[0] = 5 · 1 = 5", cls="mono", fill=INK),
          card(344, 248, 298, 76, "good", "2 × 1 per thread", ["3 reads, 2 multiply-adds", "b00 is read once, used twice"]),
          card(24, 344, 300, 62, "idle", "1 × 1 per thread (Chapter 17)", ["4 reads for the same 2 multiply-adds"]),
          card(344, 344, 298, 62, "client", "per multiply-add", ["2 reads before, 1.5 reads now"])]
save(OUT + "ch18-two-by-one.svg", 660, 426,
     "A, B and C of the 4 by 4 example. a00 = 1 and a10 = 5 of A, b00 = 1 of B, and c00 and c10 of C are "
     "highlighted. In registers at offset = 0: a_reg = 1, 5 and b_reg = 1, 3 reads from shared memory. The thread "
     "adds 1 times 1 = 1 to c00 and 5 times 1 = 5 to c10: 3 reads for 2 multiply-adds, because b00 is read once "
     "and used twice. 2 threads of Chapter 17 needed 4 reads for the same 2 multiply-adds: 1.5 reads per "
     "multiply-add instead of 2.",
     "1 thread, 2 entries of C", "The thread computes c00 and c10 and reads b00 only once.", parts)


# 18.3  1 thread computes a 2 x 2 block: 4 reads, 4 products, the outer product.
parts = [label(24, 112, "offset = 0: column 0 of A, rows 0 and 1, and row 0 of B, columns 0 and 1", room=600)]
ox, oy, s = 200, 196, 64
parts += [matrix_title(ox - 56, oy - 14, "A_REG"), matrix_title(ox + s + 2, oy - 64, "B_REG")]
parts += [cell(ox - 72, oy, 40, s - 4, LAV, 1), cell(ox - 72, oy + s, 40, s - 4, LAV, 5),
          cell(ox, oy - 52, s - 4, 40, AMBER, 1), cell(ox + s, oy - 52, s - 4, 40, AMBER, 0)]
products = [["1 · 1 = 1", "1 · 0 = 0"], ["5 · 1 = 5", "5 · 0 = 0"]]
for i in range(2):
    for j in range(2):
        parts.append(cell(ox + j * s, oy + i * s, s - 4, s - 4, THREAD[0], ""))
        parts += [text(ox + j * s + (s - 4) / 2, oy + i * s + 26, products[i][j].split(" = ")[0], "mono", "middle"),
                  text(ox + j * s + (s - 4) / 2, oy + i * s + 44, "= " + products[i][j].split(" = ")[1], "mono strong", "middle")]
parts += [label(ox + s - 2, oy + 2 * s + 18, "c00  c01 / c10  c11", "middle", fill=MUTED)]
parts += [card(380, 150, 262, 68, "client", "4 reads", ["2 values of A, 2 values of B"]),
          card(380, 230, 262, 68, "good", "4 multiply-adds", ["every value read is used twice"]),
          card(380, 310, 262, 52, "store", "1 read per multiply-add", [])]
save(OUT + "ch18-two-by-two.svg", 660, 384,
     "For offset = 0, the values 1 and 5 from column 0 of A run down the left, and the values 1 and 0 from row 0 of "
     "B run across the top. The 4 cells hold the products 1 times 1 = 1, 1 times 0 = 0, 5 times 1 = 5 and 5 times "
     "0 = 0, which are added to c00, c01, c10 and c11. Cards: 4 reads, 2 values of A and 2 of B; 4 "
     "multiply-adds, every value read is used twice; 1 read per multiply-add.",
     "1 thread, a 2 × 2 block of C", "4 values read, 4 products: the outer product of a column and a row.", parts)


# 18.4  The 2 x 2 block of C as the sum of 4 outer products.
parts = []
pw, px0 = 142, 16
for u in range(4):
    x = px0 + u * (pw + 18)
    a = (A[0][u], A[1][u])
    b = (B[u][0], B[u][1])
    prod = [[a[0] * b[0], a[0] * b[1]], [a[1] * b[0], a[1] * b[1]]]
    parts += [cell(x, 102, pw, 196, ("#FFFFFF", EDGE)),
              text(x + pw / 2, 126, f"s = {u}", "name", "middle"),
              label(x + pw / 2, 148, f"({a[0]}, {a[1]}) × ({b[0]}, {b[1]})", "middle", cls="mono", fill=INK, room=pw - 10),
              cell(x + 14, 196, 26, 30, LAV, a[0]), cell(x + 14, 230, 26, 30, LAV, a[1]),
              cell(x + 52, 160, 34, 26, AMBER, b[0]), cell(x + 90, 160, 34, 26, AMBER, b[1])]
    for i in range(2):
        for j in range(2):
            parts.append(cell(x + 52 + j * 38, 196 + i * 34, 34, 30, LAV if True else IDLE, prod[i][j], bold=True))
    parts.append(label(x + pw / 2, 284, "A column s, B row s", "middle", room=pw - 6))
    if u < 3:
        parts.append(text(x + pw + 9, 236, "+", "name", "middle"))
total = [[sum(A[i][u] * B[u][j] for u in range(4)) for j in range(2)] for i in range(2)]
assert total == [[11, 13], [27, 29]]
parts += [text(160, 352, "sum over s = 0, 1, 2, 3", "name", "middle"),
          text(300, 352, "=", "name", "middle")]
for i in range(2):
    for j in range(2):
        parts.append(cell(320 + j * 46, 322 + i * 40, 42, 36, MINT, total[i][j], bold=True))
parts += [label(430, 344, "c00 = 1 + 0 + 6 + 4 = 11", cls="mono", fill=INK),
          label(430, 364, "c11 = 0 + 6 + 7 + 16 = 29", cls="mono", fill=INK),
          label(430, 392, "c01 = 13, c10 = 27", fill=MUTED)]
save(OUT + "ch18-outer-sum.svg", 660, 420,
     "4 panels, 1 for each s from 0 to 3. In each, the 2 values of column s of A, rows 0 and 1, multiply the 2 "
     "values of row s of B, columns 0 and 1. s = 0: (1, 5) times (1, 0) gives 1 0 / 5 0. s = 1: (2, 6) times "
     "(0, 1) gives 0 2 / 0 6. s = 2: (3, 7) times (2, 1) gives 6 3 / 14 7. s = 3: (4, 8) times (1, 2) gives 4 8 "
     "/ 8 16. Their sum is 11 13 / 27 29, the 2 by 2 block of C in rows 0 and 1 and columns 0 and 1: for "
     "example c00 = 1 + 0 + 6 + 4 = 11 and c11 = 0 + 6 + 7 + 16 = 29.",
     "A block of C as a sum of outer products", "Each term index s adds 1 outer product to the thread's 4 sums.",
     parts)


# 18.5  Shared-memory reads per multiply-add for each thread tile.
tiles = [("1 × 1", 1, 1), ("2 × 1", 2, 1), ("2 × 2", 2, 2), ("4 × 4", 4, 4), ("8 × 8", 8, 8)]
x0, scale = 150, 200   # 1 read per FMA = 200 px
parts = [label(24, 104, "reads per multiply-add = (TM + TN) / (TM · TN)", cls="mono", fill=INK),
         text(x0 + 2 * scale / 2, 132, "shared-memory reads per multiply-add", "tab", "middle", MUTED),
         text(600, 132, "SUMS", "tab", "middle", MUTED)]
for k, (name, tm, tn) in enumerate(tiles):
    y = 146 + k * 46
    reads = (tm + tn) / (tm * tn)
    w = reads * scale
    parts += [text(x0 - 14, y + 21, name, "name", "end"),
              f'  <rect x="{x0}" y="{y + 4}" width="{w:.1f}" height="26" rx="6" fill="{THREAD[0][0]}" '
              f'stroke="{THREAD[0][1]}" stroke-width="1.3"/>',
              label(x0 + 10 if w > 200 else x0 + w + 8, y + 21, f"({tm} + {tn}) / {tm * tn} = {reads:g}",
                    cls="mono", fill=INK),
              cell(578, y + 4, 44, 26, MINT, tm * tn)]
for v in (0, 1, 2):
    parts += [f'  <line x1="{x0 + v * scale}" y1="142" x2="{x0 + v * scale}" y2="380" stroke="{EDGE}" '
              f'stroke-width="1" stroke-dasharray="3 4"/>',
              text(x0 + v * scale, 396, str(v), "note", "middle")]
parts += [label(24, 426, "SUMS: the accumulators of 1 thread, TM · TN registers. Chapter 17's kernel is the 1 × 1 row.",
                room=620)]
save(OUT + "ch18-reads-per-fma.svg", 660, 446,
     "Horizontal bars of shared-memory reads per multiply-add, (TM + TN) divided by TM times TN, for thread "
     "tiles 1 by 1: 2; 2 by 1: 1.5; 2 by 2: 1; 4 by 4: 0.5; 8 by 8: 0.25. Beside each bar, the number of sums a "
     "thread keeps in registers: 1, 2, 4, 16 and 64. The 1 by 1 row is the tiled kernel of Chapter 17.",
     "Reads per multiply-add", "A larger block of C per thread reads shared memory less often per multiply-add.",
     parts)


# 18.6  3 levels: global memory, shared memory, registers (4 x 4 per thread: BM = BN = 64, BK = 16).
parts = [card(16, 112, 196, 116, "idle", "global memory", ["A, B and C", "all threads", "hundreds of cycles"]),
         card(232, 112, 196, 116, "service", "shared memory", ["tileA 64 × 16", "tileB 16 × 64", "1 block, 1 phase"]),
         card(448, 112, 196, 116, "store", "registers", ["a_reg: 4, b_reg: 4", "acc: 4 × 4 = 16", "1 thread"])]
parts += [harrow(212, 232, 170), harrow(428, 448, 170)]
parts += [zone(16, 262, 628, 168, "counts for 4 × 4 per thread, blocks of 64 × 64, BK = 16")]
rows = [("global → shared", "per phase, 1 block copies 64 · 16 + 16 · 64 = 2,048 floats"),
        ("", "each of the 256 threads copies 8 of them"),
        ("shared → registers", "per offset, 1 thread reads 4 + 4 = 8 floats"),
        ("registers", "per offset, 1 thread does 4 · 4 = 16 multiply-adds"),
        ("", "0.5 shared-memory reads per multiply-add")]
for r, (head, body) in enumerate(rows):
    y = 298 + r * 26
    if head:
        parts.append(text(36, y, head, "name"))
    parts.append(label(200, y, body, room=430))
save(OUT + "ch18-levels.svg", 660, 452,
     "3 cards joined by arrows. Global memory holds A, B and C, is seen by all threads and answers in hundreds "
     "of cycles. Shared memory holds tileA of 64 by 16 and tileB of 16 by 64 floats for 1 block during 1 phase. "
     "Registers hold a_reg of 4 values, b_reg of 4 values and acc of 4 by 4 = 16 sums for 1 thread. Below, the "
     "counts for 4 by 4 per thread with blocks of 64 by 64 and BK = 16: per phase a block copies 64 times 16 "
     "plus 16 times 64 = 2,048 floats, 8 per thread; per offset a thread reads 8 floats from shared memory "
     "and does 16 multiply-adds, 0.5 reads per multiply-add.",
     "3 levels of memory", "Each value moves once per level and is reused at the level below it.", parts)


# 18.7  The example: 1 block of 2 x 2 threads, each thread a 2 x 2 block of C; who copies what.
owner = lambda i, j: THREAD[(i // 2) * 2 + j // 2]
parts = [matrix_title(86, 116, "C, OWNED BY THREADS")]
parts.append(grid(24, 128, C, owner, side=30))
names = ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]
parts += [label(24, 280, "thread (1, 0) owns rows 2 and 3", room=220),
          label(24, 298, "and columns 0 and 1", room=236)]
for t in range(4):
    parts += [cell(24 + (t % 2) * 100, 318 + (t // 2) * 30, 18, 18, THREAD[t]),
              label(48 + (t % 2) * 100, 332 + (t // 2) * 30, f"thread {names[t]}")]
tileA0 = [[A[r][s] for s in range(2)] for r in range(4)]
tileB0 = [[B[r][c] for c in range(4)] for r in range(2)]
copierA = lambda i, j: THREAD[(i * 2 + j) % 4]
copierB = lambda i, j: THREAD[(i * 4 + j) % 4]
parts += [matrix_title(304, 116, "TILEA, 4 × 2"),
          grid(270, 128, tileA0, copierA, side=30),
          matrix_title(486, 116, "TILEB, 2 × 4"),
          grid(420, 128, tileB0, copierB, side=30),
          label(420, 212, "phase 0: columns 0-1 of A,", room=220),
          label(420, 230, "rows 0-1 of B", room=220),
          label(420, 262, "thread (0, 1) has number 1;", room=220),
          label(420, 280, "it copies slots 1 and 5", room=220),
          label(420, 298, "of each tile (row-major)", room=220)]
parts += [card(270, 324, 372, 76, "idle", "Chapter 17, T = 2",
               ["the 4 threads of block (0, 0)", "computed c00, c01, c10 and c11"]),
          label(270, 422, "Here thread (0, 0) alone computes those 4 entries.", room=372)]
save(OUT + "ch18-ownership.svg", 660, 440,
     "Left: C of the example, 11 13 8 8 / 27 29 24 24 / 43 45 40 40 / 59 61 56 56, colored in 4 blocks of 2 by "
     "2, 1 color per thread of the single block of 2 by 2 threads: thread (1, 0), for example, owns rows 2 "
     "and 3 and columns 0 and 1. Right: the tiles of phase 0, tileA of 4 by 2 = columns 0 and 1 of A, 1 2 / 5 "
     "6 / 9 10 / 13 14, and tileB of 2 by 4 = rows 0 and 1 of B, 1 0 2 1 / 0 1 1 2, colored by the thread "
     "that copies each value: thread (0, 1), whose number is 1, copies slots 1 and 5 of each tile in row-major "
     "order. A card recalls that in Chapter 17 with T = 2 the 4 threads of block (0, 0) computed c00, c01, c10 "
     "and c11; here thread (0, 0) alone computes them.",
     "1 block, 4 threads, 4 entries each", "Each thread owns a 2 × 2 block of C and copies 2 values of each tile.",
     parts)


# 18.8  Thread (0, 0)'s 4 sums over the 2 phases.
def acc_grid(x, y, values, colours):
    return grid(x, y, values, lambda i, j: colours, side=34, gap=4)


def phase_row(y, t, before, added, after):
    rows_a = lambda i, j: LAV if (i < 2 and 2 * t <= j < 2 * t + 2) else (AMBER if False else IDLE)
    rows_b = lambda i, j: AMBER if (2 * t <= i < 2 * t + 2 and j < 2) else IDLE
    shown_a = lambda i, j: (LAV if i < 2 else ("#F1F4F7", "#B9C3CF")) if 2 * t <= j < 2 * t + 2 else IDLE
    shown_b = lambda i, j: (AMBER if j < 2 else ("#F1F4F7", "#B9C3CF")) if 2 * t <= i < 2 * t + 2 else IDLE
    parts = [text(24, y + 8, f"phase {t}", "name"),
             label(24, y + 26, f"s = {2 * t}, {2 * t + 1}", fill=MUTED),
             grid(104, y - 6, A, shown_a, side=22, gap=3),
             outline(104 + 2 * t * 25 - 3, y - 9, 50, 103, "#5B6BD6", 1.6),
             grid(222, y - 6, B, shown_b, side=22, gap=3),
             outline(222 - 3, y - 6 + 2 * t * 25 - 3, 103, 50, AMBER[1], 1.6),
             acc_grid(350, y + 8, before, ("#F1F4F7", "#B9C3CF")),
             text(436, y + 50, "+", "name", "middle"),
             acc_grid(450, y + 8, added, LAV),
             text(538, y + 50, "=", "name", "middle"),
             acc_grid(552, y + 8, after, MINT)]
    return "\n".join(parts)


def block_sum(us):
    return [[sum(A[i][u] * B[u][j] for u in us) for j in range(2)] for i in range(2)]


p0 = block_sum([0, 1])
p1 = block_sum([2, 3])
assert p0 == [[1, 2], [5, 6]] and p1 == [[10, 11], [22, 23]]
done = [[p0[i][j] + p1[i][j] for j in range(2)] for i in range(2)]
assert done == [[11, 13], [27, 29]]
parts = [matrix_title(152, 108, "A"), matrix_title(270, 108, "B"),
         matrix_title(388, 108, "ACC BEFORE"), matrix_title(488, 108, "ADDED"), matrix_title(590, 108, "ACC AFTER"),
         phase_row(132, 0, [[0, 0], [0, 0]], p0, p0),
         phase_row(256, 1, p0, p1, done),
         label(24, 384, "Highlighted: values thread (0, 0) reads; gray: tile values that other threads read.",
               room=620),
         label(24, 404, "After phase 1 the 4 sums are c00 = 11, c01 = 13, c10 = 27 and c11 = 29.", room=620)]
save(OUT + "ch18-phases.svg", 660, 424,
     "2 rows, 1 per phase. Phase 0 uses columns 0 and 1 of A and rows 0 and 1 of B, s = 0 and 1; thread (0, 0) "
     "reads rows 0 and 1 of that part of A and columns 0 and 1 of that part of B. Its 4 sums go from 0 0 / 0 0 "
     "to 1 2 / 5 6. Phase 1 uses columns 2 and 3 of A and rows 2 and 3 of B, s = 2 and 3, and adds 10 11 / 22 23, "
     "so the sums become 11 13 / 27 29: c00, c01, c10 and c11.",
     "Thread (0, 0) over 2 phases", "The 4 sums grow by 2 outer products in each phase.", parts)


# 18.9  Measured on the Orin NX at n = 4,096 (quietest of 3 runs of each program), with the
# shared-memory limit 300 x 2 / (reads per multiply-add) of each thread tile.
ROWS = [("Ch. 17, T = 16", 2.0, 271.9, 274.4), ("1 × 1", 2.0, 220.4, 219.4), ("2 × 1", 1.5, 391.5, 389.8),
        ("2 × 2", 1.0, 583.5, 568.5), ("4 × 4", 0.5, 1009.9, 875.5), ("8 × 8", 0.25, 773.1, 827.5)]
CPP_C, RUST_C = ("#DDE3FB", "#5B6BD6"), ("#D7F0E6", "#1F8A70")
RED = "#CF4526"


def bar18(x, y, w, h, colours, value):
    fill, stroke = colours
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}"'
            f' stroke="{stroke}" stroke-width="1.2"/>\n'
            + f'  <text x="{x + w + 6:.1f}" y="{y + h - 4:.1f}" style="font-size:11.5px;font-weight:700;'
            f'fill:{stroke};stroke:#FBFAF6;stroke-width:4px;paint-order:stroke">{value}</text>')


x0, scale = 150, 420 / 2400
parts = [label(40, 106, "Rate at n = 4,096, GFLOP/s; red mark: shared-memory limit", "start", "name", room=600)]
for k, (name, reads, cpp, rust) in enumerate(ROWS):
    y = 122 + k * 50
    lim = min(300 * 2 / reads, 2400)
    parts.append(label(x0 - 10, y + 24, name, "end", "note", INK))
    parts.append(f'  <line x1="{x0 + lim * scale:.1f}" y1="{y}" x2="{x0 + lim * scale:.1f}" y2="{y + 44}"'
                 f' stroke="{RED}" stroke-width="2"/>')
    parts.append(bar18(x0, y + 3, cpp * scale, 17, CPP_C, f"{cpp:.1f}"))
    parts.append(bar18(x0, y + 23, rust * scale, 17, RUST_C, f"{rust:.1f}"))
y = 122 + 6 * 50 + 8
parts += [f'  <rect x="40" y="{y}" width="12" height="12" rx="3" fill="{CPP_C[0]}" stroke="{CPP_C[1]}"/>',
          label(58, y + 10, "C++", "start", "note", INK),
          f'  <rect x="100" y="{y}" width="12" height="12" rx="3" fill="{RUST_C[0]}" stroke="{RUST_C[1]}"/>',
          label(118, y + 10, "Rust", "start", "note", INK),
          f'  <line x1="170" y1="{y + 6}" x2="186" y2="{y + 6}" stroke="{RED}" stroke-width="2"/>',
          label(192, y + 10, "300 × 2 / (reads per multiply-add); 2,400 is the FP32 peak", "start", "note", INK,
                room=440)]
save(OUT + "ch18-measured.svg", 660, y + 34,
     "Horizontal bars of the rate in GFLOP/s at n = 4,096 on the Orin NX, C++ and Rust for each kernel, with a "
     "red mark at the shared-memory limit of 300 times 2 divided by the reads per multiply-add. Chapter 17's "
     "kernel with T = 16: 271.9 and 274.4, limit 300. 1 by 1: 220.4 and 219.4, limit 300. 2 by 1: 391.5 and "
     "389.8, limit 400. 2 by 2: 583.5 and 568.5, limit 600. 4 by 4: 1,009.9 and 875.5, limit 1,200. 8 by 8: "
     "773.1 and 827.5, limit 2,400, the FP32 peak.",
     "Register tiles, measured", "Up to 2 × 2 the rate follows the shared-memory limit; beyond, other limits.",
     parts)
