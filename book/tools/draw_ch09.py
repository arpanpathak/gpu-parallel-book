"""The figures of chapter 9, drawn with figkit and figparts.

    python3 tools/draw_ch09.py

The speeds in the last figure are the ones measured with code/ch09_sgemm/sgemm.cu on a Jetson
Orin NX and a Jetson AGX Thor (the runs quoted in the chapter).
"""
from figkit import width_of
from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, arrow, card, fan, legend, line, mono,
                      note, save, tag, text, wire, zone)

OUT = "src/figures/"

# GFLOP/s for N = 4,096, from the runs quoted in the chapter: (Orin NX, Thor).
PEAKS = (2400, 8100)
SPEEDS = [("by column", 34.2, 116.2), ("by row", 185.9, 596.2), ("tiles 16 × 16", 273.9, 949.9),
          ("tiles 32 × 32", 222.1, 783.4), ("4 × 4 per thread", 901.4, 3222.0),
          ("4 × 4, best bounds", 931.9, 3264.2), ("cuBLAS", 1495.3, 6176.5)]


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A matrix cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def matrix(x, y, values, cell, hot, kind):
    """A small matrix of numbers; the cells in `hot` take the colours of `kind`."""
    return [box(x + c * cell, y + r * cell, cell - 3, cell - 3, kind, str(v), (r, c) in hot)
            for r, row in enumerate(values) for c, v in enumerate(row)]


def gather(xs, y, bus, x, bottom):
    """Several sources joined into 1: a drop from each to the bus line, then 1 arrow down."""
    left, right = min(xs + [x]), max(xs + [x])
    parts = [f'  <path d="M{left},{bus} H{right}" fill="none" stroke="{WIRE}" stroke-width="1.8"/>']
    parts += [f'  <path d="M{sx},{y} V{bus}" fill="none" stroke="{WIRE}" stroke-width="1.6"/>'
              for sx in xs]
    parts += [f'  <circle cx="{x}" cy="{bus}" r="3" fill="{WIRE}"/>', arrow(x, bus, bottom)]
    return "\n".join(parts)


def squares(x, y, count, kind, side=10, gap=3):
    """`count` small squares in a row; a fraction draws the last one half full."""
    accent, tint = KINDS[kind]
    parts, whole = [], int(count)
    for k in range(whole):
        parts.append(f'  <rect x="{x + k * (side + gap):.1f}" y="{y}" width="{side}" height="{side}" '
                     f'rx="2.5" fill="{tint}" stroke="{accent}" stroke-width="1.1"/>')
    if count > whole:
        at = x + whole * (side + gap)
        parts += [f'  <rect x="{at:.1f}" y="{y}" width="{side}" height="{side}" rx="2.5" fill="#FFFFFF" '
                  f'stroke="{accent}" stroke-width="1.1" stroke-dasharray="2 2"/>',
                  f'  <rect x="{at:.1f}" y="{y}" width="{side / 2}" height="{side}" rx="1.5" fill="{accent}" '
                  f'opacity="0.45"/>']
    return "\n".join(parts)


# 9.1  An element of C is the dot product of a row of A and a column of B.
A = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 1], [1, 1, 0, 2]]
B = [[2, 1, 0, 1], [1, 0, 2, 1], [0, 3, 1, 0], [1, 2, 1, 1]]
C = [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
cell = 32
parts = [text(40, 118, "A", "name"), text(250, 118, "B", "name"), text(460, 118, "C = A B", "name")]
parts += matrix(40, 130, A, cell, {(1, k) for k in range(4)}, "client")
parts += matrix(250, 130, B, cell, {(k, 2) for k in range(4)}, "outside")
parts += matrix(460, 130, C, cell, {(1, 2)}, "good")
parts += [mono(203, 196, "×", "middle", INK), mono(413, 196, "=", "middle", INK)]
xs = []
for k in range(4):
    a, b = A[1][k], B[k][2]
    x = 70 + k * 130
    parts += [mono(x + 52, 284, f"k = {k}", "middle", MUTED),
              box(x, 292, 104, 30, "idle", f"{a} × {b} = {a * b}")]
    xs.append(x + 52)
parts += [gather(xs, 322, 344, 330, 368),
          box(220, 368, 220, 32, "good", "C[1][2] = 0 + 2 + 3 + 2 = 7"),
          note(40, 434, "Row 1 of A meets column 2 of B 1 pair at a time: 4 multiplications and 4", room=600),
          note(40, 452, "additions into a running sum. Every element of C is 1 such sum.", room=600)]
save(OUT + "ch09-matmul.svg", 660, 474,
     "Three 4 by 4 matrices A, B and C. Row 1 of A, 0 1 3 2, and column 2 of B, 0 2 1 1, are highlighted. "
     "Below, their 4 products 0 times 0, 1 times 2, 3 times 1 and 2 times 1 are joined into the sum "
     "0 + 2 + 3 + 2 = 7, which is element C[1][2].",
     "1 element of C", "C[1][2] is row 1 of A times column 2 of B, pair by pair.", parts)


# 9.2  1 value of A is used by a whole row of C: reuse, and how much of it a GPU needs.
parts = [zone(16, 104, 340, 340, "1 value, 4 uses"), zone(372, 104, 272, 340, "uses needed")]
targets = [66 + j * 76 for j in range(4)]
parts += [box(126, 140, 100, 34, "client", "A[1][1] = 1"),
          fan(176, 174, targets, 214, 246)]
for j in range(4):
    parts += [box(targets[j] - 33, 246, 66, 40, "good", f"C[1][{j}]"),
              mono(targets[j], 310, f"+ 1 × {B[1][j]}", "middle", INK)]
parts += [note(40, 352, "A[1][1] multiplies each value of row 1 of B,", room=300),
          note(40, 370, "1, 0, 2 and 1, and joins all 4 sums of row 1", room=300),
          note(40, 388, "of C. With N = 4, every value has 4 uses.", room=300),
          card(392, 136, 232, 64, "client", "N = 4,096", "every value: 4,096 uses"),
          card(392, 220, 232, 80, "store", "Orin NX: 94 FLOP", ["per float from memory", "to reach the FP32 peak"]),
          card(392, 320, 232, 80, "outside", "Thor: 119 FLOP", ["per float from memory", "to reach the FP32 peak"]),
          note(392, 426, "peak FLOP/s ÷ floats per second", room=232)]
save(OUT + "ch09-reuse.svg", 660, 466,
     "Left: in a 4 by 4 product, element A[1][1] = 1 is multiplied by each value of row 1 of B, "
     "1, 0, 2 and 1, and joins all 4 sums of row 1 of C. Right: for N = 4,096 each value has 4,096 uses; "
     "to reach the FP32 peak, each float fetched from memory must feed 94 operations on the Orin NX and "
     "119 on the Thor.",
     "Reuse", "Every value of A and B is needed N times.", parts)


# 9.3  Which sectors 1 warp touches at 1 step k: threadIdx.x down a column of C, or along a row.
parts = [zone(16, 104, 304, 400, "x down a column of C"), zone(340, 104, 304, 400, "x along a row of C")]
sw = 30


def sector(x, y, on, kind):
    accent, tint = KINDS[kind]
    return (f'  <rect x="{x:.1f}" y="{y}" width="{sw - 3}" height="10" rx="2" '
            f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1"/>')


for x0, by_row in ((40, False), (364, True)):
    kind = "good" if by_row else "danger"
    a_rows, b_sectors = (2, 2) if by_row else (16, 1)
    total = a_rows + b_sectors
    parts.append(text(x0, 142, "A: rows 0 to 15, first 8 sectors", "sub"))
    for r in range(16):
        for s in range(8):
            parts.append(sector(x0 + s * sw, 152 + r * 13, s == 0 and r < a_rows, kind))
    parts.append(text(x0, 386, "B: row k, first 8 sectors", "sub"))
    for s in range(8):
        parts.append(sector(x0 + s * sw, 396, s < b_sectors, kind))
    parts += [tag(x0, 428, f"A {a_rows} + B {b_sectors} = {total} sectors", kind),
              mono(x0, 474, f"{total} × 32 bytes = {total * 32} bytes", "start", INK)]
parts += [note(16, 534, "1 sector is 32 bytes, 8 floats. A warp is 16 values of x for 2 of y. Down a column,", room=628),
          note(16, 552, "x lands on 16 rows of A. Along a row, the 16 lanes of each y share 1 element of A", room=628),
          note(16, 570, "and read 16 neighbouring elements of B: 64 bytes, 2 sectors.", room=628)]
save(OUT + "ch09-mapping.svg", 660, 592,
     "Two panels, each showing the 32-byte sectors 1 warp touches at 1 step of k. Left, with threadIdx.x down "
     "a column of C: 1 sector in each of 16 rows of A and 1 sector of B, 17 sectors, 544 bytes. Right, with "
     "threadIdx.x along a row of C: 1 sector in each of 2 rows of A and 2 sectors of B, 4 sectors, 128 bytes.",
     "Sectors per step of k", "The same kernel with x and y swapped touches 17 or 4 sectors.", parts)


# 9.4  Tiling: global memory to shared memory to the block's tile of C.
n, cell = 8, 13
ax, ay, by = 66, 150, 300
parts = [zone(16, 104, 222, 356, "global memory"), zone(258, 104, 170, 356, "shared memory"),
         zone(448, 104, 196, 356, "the block")]
parts += [text(40, ay + 4 * cell + 4, "A", "name"), text(40, by + 4 * cell + 4, "B", "name"),
          note(40, ay + 8 * cell + 22, "strip: rows 2 and 3", room=190),
          note(40, by + 8 * cell + 22, "strip: columns 4 and 5", room=190)]
for r in range(n):
    for c in range(n):
        in_a, in_b = r in (2, 3), c in (4, 5)
        parts += [box(ax + c * cell, ay + r * cell, cell - 2, cell - 2,
                      "client" if in_a and c in (2, 3) else "idle", on=in_a),
                  box(ax + c * cell, by + r * cell, cell - 2, cell - 2,
                      "outside" if in_b and r in (2, 3) else "idle", on=in_b)]
for r in range(2):
    for c in range(2):
        parts += [box(298 + c * 40, 160 + r * 40, 36, 36, "client", "×2"),
                  box(298 + c * 40, 318 + r * 40, 36, 36, "outside", "×2")]
cy = 217
for r in range(n):
    for c in range(n):
        parts.append(box(476 + c * 18, cy + r * 18, 15, 15, "good", on=r in (2, 3) and c in (4, 5)))
parts += [text(298, 152, "tileA", "mono"), text(298, 310, "tileB", "mono"), text(476, cy - 10, "C", "name"),
          wire([(ax + 8 * cell + 2, ay + 2.5 * cell), (296, ay + 2.5 * cell)]),
          wire([(ax + 8 * cell + 2, by + 2.5 * cell), (296, by + 2.5 * cell)]),
          wire([(376, 198), (406, 198), (406, cy + 54)], arrow=False),
          wire([(376, 356), (406, 356), (406, cy + 54)], arrow=False),
          wire([(406, cy + 54), (473, cy + 54)]),
          f'  <circle cx="406" cy="{cy + 54}" r="3" fill="{WIRE}"/>',
          note(476, cy + 8 * 18 + 20, "2 × 2 threads,", room=160),
          note(476, cy + 8 * 18 + 38, "1 sum each", room=160),
          note(16, 492, "At step k0 = 2 the block copies 1 tile of A and 1 tile of B, 4 values each. Every", room=628),
          note(16, 510, "value in shared memory is then read by 2 threads (×2): by T threads for a T × T tile.",
               room=628)]
save(OUT + "ch09-tiles.svg", 660, 532,
     "Three zones. Global memory holds A and B as 8 by 8 grids; the block needs rows 2 and 3 of A and columns "
     "4 and 5 of B, and at the step shown a 2 by 2 tile of each is copied into shared memory. There each value "
     "is marked times 2: it is read by 2 threads. The block's 2 by 2 tile of C sits in rows 2 and 3, "
     "columns 4 and 5 of C.",
     "Tiles through shared memory", "Copy a tile once, read it T times.", parts)


# 9.5  The reads behind 16 multiply-adds of 1 thread, in each kernel.
rows = [("1 per thread, by row", "by row", 32, 0), ("tiles of 16 × 16", "tiles 16 × 16", 2, 32),
        ("4 × 4 per thread", "4 × 4 per thread", 0.5, 8)]
speed = {name: orin for name, orin, _ in SPEEDS}
parts = [legend(186, 120, [("store", "1 float from global memory"), ("client", "1 float from shared memory")])]
for g, (name, key, glob, shared) in enumerate(rows):
    y = 150 + g * 96
    orin = speed[key]
    parts += [f'  <rect x="24" y="{y}" width="612" height="82" rx="12" fill="#FFFFFF" stroke="{EDGE}" '
              f'stroke-width="1.2" filter="url(#lift)"/>',
              text(40, y + 26, name, "name"),
              tag(620, y + 13, f"{orin:,.0f} GFLOP/s on the Orin NX", "good", "end"),
              text(40, y + 52, f"{glob:g} from global", "sub"),
              text(40, y + 70, f"{shared:g} from shared", "sub"),
              squares(186, y + 42, glob, "store"), squares(186, y + 60, shared, "client")]
parts += [note(24, 456, "Each multiply-add needs 2 operands. 1 thread per element fetches both from global memory;", room=612),
          note(24, 474, "a tile fetches 2 of every 32 from global memory and the rest from shared memory; a 4 × 4", room=612),
          note(24, 492, "patch keeps the sums in registers and fetches 8 operands for 16 multiply-adds.", room=612)]
save(OUT + "ch09-reads.svg", 660, 514,
     "Three cards, 1 per kernel, counting the floats 1 thread reads for 16 multiply-adds. 1 per thread: 32 "
     "from global memory. Tiles of 16 by 16: 2 from global memory and 32 from shared memory. 4 by 4 per "
     "thread: half a float from global memory and 8 from shared memory.",
     "The reads behind 16 multiply-adds", "Fewer reads per multiply-add, from closer memory, at each stage.",
     parts)


# 9.6  From C to a block's 64 x 64 tile to a thread's 4 x 4 patch.
parts = [zone(16, 104, 196, 320, "C, 4,096 × 4,096"), zone(232, 104, 214, 320, "1 block: 64 × 64"),
         zone(466, 104, 178, 320, "1 thread: 4 × 4")]
for r in range(8):
    for c in range(8):
        parts.append(box(40 + c * 19, 150 + r * 19, 16, 16, "client", on=(r, c) == (2, 7)))
for r in range(16):
    for c in range(16):
        parts.append(box(256 + c * 11, 150 + r * 11, 9, 9, "outside", on=(r, c) == (6, 15)))
for r in range(4):
    for c in range(4):
        parts.append(box(490 + c * 34, 150 + r * 34, 30, 30, "good", f"{r}{c}"))
parts += [wire([(40 + 8 * 19, 150 + 2.5 * 19), (252, 150 + 2.5 * 19)]),
          wire([(256 + 16 * 11, 150 + 6.5 * 11), (486, 150 + 6.5 * 11)]),
          note(40, 352, "64 blocks per side", room=160), note(40, 370, "64 × 64 = 4,096 blocks", room=160),
          note(256, 352, "64 ÷ 4 = 16 threads a side", room=180),
          note(256, 370, "16 × 16 = 256 threads", room=180),
          note(490, 352, "acc[0][0] … acc[3][3]", room=140), note(490, 370, "16 sums in registers", room=140),
          note(16, 456, "A block of 256 threads owns a 64 × 64 tile of C; thread (tx, ty) owns rows 4ty to 4ty + 3", room=628),
          note(16, 474, "and columns 4tx to 4tx + 3 of that tile. The shaded thread is tx = 15, ty = 6.", room=628)]
save(OUT + "ch09-zoom.svg", 660, 496,
     "Three zones zooming in. C, 4,096 by 4,096, is cut into 64 by 64 tiles, 1 per block, 4,096 blocks. A "
     "block's tile is cut into 16 by 16 patches of 4 by 4, 1 per thread. A thread's patch holds 16 sums, "
     "acc[0][0] to acc[3][3], in registers.",
     "Who computes what", "A block owns a 64 × 64 tile; each of its 256 threads a 4 × 4 patch.", parts)


# 9.7  A thread's 4 x 4 patch for 1 value of k: 4 values of A and 4 of B give 16 products.
av, bv = [1, 2, 0, 3], [2, 1, 0, 1]
cell, x0, y0 = 40, 300, 176
parts = [text(x0, 130, "b = tileB[k][4tx … 4tx + 3]", "mono")]
for j in range(4):
    parts.append(box(x0 + j * cell, 140, cell - 4, 28, "outside", str(bv[j])))
parts.append(text(x0 - 64, y0 - 10, "a", "mono"))
for i in range(4):
    parts.append(box(x0 - 64, y0 + i * cell, cell - 4, cell - 4, "client", str(av[i])))
    for j in range(4):
        parts.append(box(x0 + j * cell, y0 + i * cell, cell - 4, cell - 4, "good", str(av[i] * bv[j])))
parts += [text(x0 - 80, y0 + 60, "tileA[4ty … 4ty + 3][k]", "mono", "end"),
          text(x0 + 4 * cell + 16, y0 + 54, "acc[i][j] += a[i] × b[j]", "mono"),
          text(x0 + 4 * cell + 16, y0 + 76, "16 multiply-adds", "sub"),
          text(x0 + 4 * cell + 16, y0 + 94, "from 8 reads", "sub"),
          note(40, 376, "For each k, a thread reads 4 values of A and 4 of B from shared memory and adds all", room=600),
          note(40, 394, "16 products a[i] × b[j] into its 16 sums. Row 3 of the patch adds 6, 3, 0 and 3.",
               room=600)]
save(OUT + "ch09-outer.svg", 660, 416,
     "A thread's 4 by 4 patch for 1 value of k. The 4 values of A, 1, 2, 0 and 3, run down the left; the 4 "
     "values of B, 2, 1, 0 and 1, run across the top. Each cell holds the product a[i] times b[j], from 2, 1, "
     "0, 1 in row 0 to 6, 3, 0, 3 in row 3.",
     "A 4 × 4 patch for 1 value of k", "8 reads from shared memory feed 16 multiply-adds.", parts)


# 9.8  Registers per thread decide how many blocks fit on an SM.
total = 65536
x0, wide = 196, 300
parts = [text(x0, 124, "65,536 registers in 1 SM", "sub")]
for k, (regs, spill) in enumerate([(80, 0), (64, 0), (40, 16)]):
    y = 144 + k * 64
    per = regs * 256
    count = min(total // per, 6)
    parts += [text(x0 - 14, y + 15, f"{regs} registers", "name", "end"),
              text(x0 - 14, y + 33, f"× 256 = {per:,}", "sub", "end")]
    for b in range(count):
        parts.append(box(x0 + b * per / total * wide, y, per / total * wide - 3, 36,
                         "danger" if spill else "client", "block" if per / total * wide > 50 else "",
                         cls="pill"))
    parts += [f'  <rect x="{x0}" y="{y}" width="{wide}" height="36" rx="6" fill="none" stroke="{EDGE}" '
              f'stroke-width="1.2"/>',
              text(x0 + wide + 14, y + 15, f"{count} blocks", "name"),
              text(x0 + wide + 14, y + 33, f"{count * 256:,} threads" + (", spills" if spill else ""), "sub")]
parts += [note(40, 358, "1 block of 256 threads at 80 registers each needs 80 × 256 = 20,480 registers, and", room=600),
          note(40, 376, "65,536 ÷ 20,480 = 3.2, so 3 blocks fit. At 40 registers 6 blocks fit, the limit of", room=600),
          note(40, 394, "1,536 threads, but on the Orin NX the compiler spilled 16 bytes per thread to stay at 40.",
               room=600)]
save(OUT + "ch09-occupancy.svg", 660, 416,
     "Three bars of 65,536 registers each, filled with blocks of 256 threads. At 80 registers per thread a "
     "block needs 20,480 registers and 3 blocks fit, 768 threads. At 64, 16,384 per block and 4 fit, 1,024 "
     "threads. At 40, 10,240 per block and 6 fit, 1,536 threads, with values spilled to memory.",
     "Registers decide how many blocks fit", "Blocks of 256 threads on 1 SM of the Orin NX.", parts)


# 9.9  Measured speeds on 2 boards, as a share of each board's FP32 peak.
x0, wide = 176, 360
parts = [legend(x0, 104, [("client", "Jetson Orin NX, peak 2,400 GFLOP/s")]),
         legend(x0 + 260, 104, [("outside", "Jetson AGX Thor, peak 8,100")])]
for g in range(0, 101, 25):
    x = x0 + wide * g / 100
    parts += [line(x, 132, x, 132 + 7 * 54, EDGE, 1), mono(x, 132 + 7 * 54 + 18, f"{g}%", "middle", MUTED)]
for k, (name, orin, thor) in enumerate(SPEEDS):
    y = 140 + k * 54
    parts.append(text(x0 - 14, y + 24, name, "sub", "end"))
    for b, (v, kind) in enumerate(((orin, "client"), (thor, "outside"))):
        share = v / PEAKS[b]
        colour = KINDS[kind][0]
        w = max(wide * share, 3)
        parts += [f'  <rect x="{x0}" y="{y + b * 20}" width="{w:.1f}" height="16" rx="4" fill="{colour}"/>',
                  mono(x0 + w + 8, y + b * 20 + 12.5, f"{v:,.0f} GFLOP/s", "start", INK)]
parts += [text(x0 - 14, 132 + 7 * 54 + 18, "of FP32 peak", "sub", "end"),
          note(40, 552, "N = 4,096. Each bar is the measured speed as a share of its board's FP32 peak. The", room=600),
          note(40, 570, "best kernel of the chapter reached 39% and 40% of peak, cuBLAS 62% and 76%.", room=600)]
save(OUT + "ch09-gflops.svg", 660, 592,
     "Horizontal bars of SGEMM speed for N = 4,096 as a share of each board's FP32 peak. Orin NX, peak "
     "2,400 GFLOP/s: by column 34, by row 186, tiles of 16 274, tiles of 32 222, 4 by 4 per thread 901, "
     "4 by 4 with the best launch bounds 932, cuBLAS 1,495 GFLOP/s. Thor, peak 8,100: 116, 596, 950, 783, "
     "3,222, 3,264 and 6,177 GFLOP/s.",
     "Every stage, measured on 2 boards", "GFLOP/s as a share of each board's FP32 peak.", parts)
