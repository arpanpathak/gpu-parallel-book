"""The figures of chapter 9, drawn with figkit and figparts.

    python3 tools/draw_ch09.py

The speeds in figure 9.6 are the ones measured with code/ch09_sgemm/sgemm.cu on a Jetson
Orin NX and a Jetson AGX Thor (the runs quoted in the chapter).
"""
from figkit import width_of
from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, bar, card, legend, line, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"


def rect(x, y, w, h, fill, stroke="none", sw=1.2, rx=3):
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def grid(x, y, n, cell, hot=(), kind="client", labels=None):
    """An n x n matrix of cells; `hot` cells take the colours of `kind`."""
    out = []
    for r in range(n):
        for c in range(n):
            on = (r, c) in hot
            out.append(rect(x + c * cell, y + r * cell, cell - 2, cell - 2,
                            KINDS[kind][1] if on else "#FFFFFF", KINDS[kind][0] if on else EDGE))
            if labels:
                out.append(mono(x + c * cell + (cell - 2) / 2, y + r * cell + cell / 2 + 2,
                                labels[r][c], "middle", INK))
    return out


# 9.1  An element of C is the dot product of a row of A and a column of B.
A = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 1], [1, 1, 0, 2]]
B = [[2, 1, 0, 1], [1, 0, 2, 1], [0, 3, 1, 0], [1, 2, 1, 1]]
C = [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
cell = 30
s = lambda m: [[str(v) for v in row] for row in m]
parts = [text(40, 120, "A", "name"), text(240, 120, "B", "name"), text(440, 120, "C = A B", "name")]
parts += grid(40, 132, 4, cell, {(1, k) for k in range(4)}, "client", s(A))
parts += grid(240, 132, 4, cell, {(k, 2) for k in range(4)}, "outside", s(B))
parts += grid(440, 132, 4, cell, {(1, 2)}, "good", s(C))
parts += [mono(184, 196, "×", "middle", INK), mono(384, 196, "=", "middle", INK),
          note(40, 290, "C[1][2] = 0 × 0 + 1 × 2 + 3 × 1 + 2 × 1 = 7: row 1 of A against column 2 of B.", room=600),
          note(40, 316, "Each of the N² elements of C takes N multiplications and N additions, so the", room=600),
          note(40, 334, "product makes 2N³ operations; it reads 2N² values and writes N².", room=600)]
save(OUT + "ch09-matmul.svg", 660, 356,
     "Three 4 by 4 matrices A, B and C. Row 1 of A, 0 1 3 2, and column 2 of B, 0 2 1 1, are highlighted; "
     "their dot product 0 + 2 + 3 + 2 = 7 is element C[1][2].",
     "Matrix multiplication", "Element (i, j) of C is row i of A times column j of B.", parts)


# 9.2  Which addresses 1 warp reads: threadIdx.x down a column of C, or along a row.
cell = 13
parts = [zone(16, 104, 300, 300, "threadIdx.x down a column"), zone(344, 104, 300, 300, "threadIdx.x along a row")]
for z, (x0, by_row) in enumerate([(40, False), (368, True)]):
    a_hot = {(r, 0) for r in range(16)} if not by_row else {(0, 0), (1, 0)}
    parts.append(text(x0, 146, "A, at step k = 0", "sub"))
    parts += grid(x0, 156, 16, cell, a_hot, "danger" if not by_row else "good")
    parts.append(text(x0, 384, "16 rows: 16 sectors" if not by_row else "2 elements, each broadcast",
                      "sub"))
parts += [note(16, 436, "A warp is 16 values of threadIdx.x for 2 of threadIdx.y. As the row index, x puts", room=628),
          note(16, 454, "the warp on 16 rows of A, 16 sectors. As the column index, the warp covers only", room=628),
          note(16, 472, "2 rows, and the 16 lanes of each row read the same element of A.", room=628)]
save(OUT + "ch09-mapping.svg", 660, 494,
     "Two panels showing the part of matrix A that 1 warp reads at step k = 0. Left, with threadIdx.x down a "
     "column of C, the lanes read 16 elements of A in 16 different rows, 1 sector each. Right, with "
     "threadIdx.x along a row of C, the warp reads 2 elements of A, each broadcast to 16 lanes.",
     "A warp's reads of A", "The same kernel with x and y swapped reads A very differently.", parts)


# 9.3  Tiling: a block's tile of C needs a strip of A and a strip of B, walked 1 tile at a time.
cell = 16
parts = [text(40, 124, "A", "name"), text(280, 124, "B", "name"), text(480, 124, "C", "name")]
n = 8
rowband = {(r, c) for r in (2, 3) for c in range(n)}
colband = {(r, c) for r in range(n) for c in (4, 5)}
tileA = {(r, c) for r in (2, 3) for c in (2, 3)}
tileB = {(r, c) for r in (2, 3) for c in (4, 5)}
parts += grid(40, 136, n, cell, rowband, "idle") + grid(280, 136, n, cell, colband, "idle")
parts += grid(480, 136, n, cell, {(r, c) for r in (2, 3) for c in (4, 5)}, "good")
for (r, c) in tileA:
    parts.append(rect(40 + c * cell, 136 + r * cell, cell - 2, cell - 2, KINDS["client"][1], KINDS["client"][0]))
for (r, c) in tileB:
    parts.append(rect(280 + c * cell, 136 + r * cell, cell - 2, cell - 2, KINDS["outside"][1], KINDS["outside"][0]))
parts += [mono(232, 204, "×", "middle", INK), mono(456, 204, "=", "middle", INK),
          legend(40, 300, [("client", "tile of A at step k0"), ("outside", "tile of B at step k0"),
                           ("good", "the block's tile of C")]),
          note(40, 342, "The block's T × T tile of C needs a strip of T rows of A and a strip of T columns", room=600),
          note(40, 360, "of B. At each step k0 the block copies 1 T × T tile of each into shared memory;", room=600),
          note(40, 378, "every value copied is then read by T threads, so global reads fall T times.", room=600)]
save(OUT + "ch09-tiles.svg", 660, 400,
     "Matrices A, B and C as 8 by 8 grids of cells. A block's 2 by 2 tile of C, highlighted, needs rows 2 "
     "and 3 of A and columns 4 and 5 of B. At the step shown, 1 2 by 2 tile of each strip is in shared memory.",
     "Tiles of A and B", "A block walks its strips of A and B 1 tile at a time.", parts)


# 9.4  A thread's 4 x 4 patch: 4 values of A and 4 of B give 16 products.
cell = 34
x0, y0 = 250, 170
parts = [text(x0, 128, "4 values of B: b0 … b3, from 1 row of tileB", "sub")]
for j in range(4):
    parts.append(bar(x0 + j * cell, 140, cell - 4, 24, "outside", f"b{j}"))
for i in range(4):
    parts.append(bar(x0 - cell - 10, y0 + i * cell, cell - 4, 24, "client", f"a{i}"))
    for j in range(4):
        parts.append(bar(x0 + j * cell, y0 + i * cell, cell - 4, 24, "good", f"{i}{j}"))
parts += [text(x0 - cell - 22, y0 + 60, "4 values of A", "sub", "end"),
          text(x0 - cell - 22, y0 + 78, "from 1 column", "sub", "end"),
          text(x0 + 4 * cell + 14, y0 + 60, "acc[i][j] += a[i] × b[j]", "mono"),
          text(x0 + 4 * cell + 14, y0 + 78, "16 sums, in registers", "sub"),
          note(40, 344, "For each k, a thread reads 8 values from shared memory and makes 16 multiply-adds.", room=600),
          note(40, 362, "With 1 element per thread it read 2 values for 1. Shared-memory reads fall 4 times.", room=600)]
save(OUT + "ch09-outer.svg", 660, 384,
     "A 4 by 4 grid of accumulators, acc 0 0 to acc 3 3. To its left a column of 4 values a0 to a3 from the "
     "A tile; above it a row of 4 values b0 to b3 from the B tile. Each accumulator adds a[i] times b[j].",
     "A 4 × 4 patch per thread", "8 reads from shared memory feed 16 multiply-adds.", parts)


# 9.5  Registers per thread decide how many blocks fit on an SM.
total = 65536
x0, wide = 180, 360
parts = [text(x0, 124, "65,536 registers per SM, blocks of 256 threads", "sub")]
for k, (label, regs, spill) in enumerate([("80 registers", 80, False), ("64 registers", 64, False),
                                          ("40 registers", 40, True)]):
    y = 144 + k * 60
    per = regs * 256
    count = min(total // per, 6)
    parts.append(text(x0 - 14, y + 20, label, "sub", "end"))
    for b in range(count):
        parts.append(bar(x0 + b * per / total * wide, y, per / total * wide - 3, 30,
                         "danger" if spill else "client", f"{regs * 256:,}" if per / total * wide > 60 else ""))
    parts.append(rect(x0, y, wide, 30, "none", EDGE, 1.2, 6))
    parts.append(mono(x0 + wide + 12, y + 20, f"{count} blocks", "start", INK))
parts += [note(40, 342, "A block of 256 threads at 80 registers each takes 20,480 registers, so 3 fit in", room=600),
          note(40, 360, "65,536. At 64, 4 fit. At 40, 6 fit, the limit of 1,536 threads, but the compiler", room=600),
          note(40, 378, "had to spill values to local memory to stay under 40 on the Orin NX.", room=600)]
save(OUT + "ch09-occupancy.svg", 660, 400,
     "Three bars of 65,536 registers each. At 80 registers per thread, a block of 256 threads uses 20,480 "
     "registers and 3 blocks fit. At 64, 16,384 per block and 4 fit. At 40, 10,240 per block and 6 fit.",
     "Registers and occupancy", "Registers per thread set the number of resident blocks.", parts)


# 9.6  Measured speeds on 2 boards, against cuBLAS and the FP32 peak.
rows = [("by column", 33.6, 117.7), ("by row", 179.6, 594.0), ("tiles 16 × 16", 277.7, 945.3),
        ("tiles 32 × 32", 219.0, 773.8), ("4 × 4, 4 blocks", 954.2, 3272.5), ("cuBLAS", 1517.9, 6172.6)]
x0 = 170
parts = []
for b, (board, peak, kind, top) in enumerate([("Jetson Orin NX", 2400, "client", 104),
                                              ("Jetson AGX Thor", 8100, "outside", 104)]):
    left = x0 + b * 240
    scale = 220 / peak
    parts.append(text(left, top + 8, board, "name"))
    for k, (label, orin, thor) in enumerate(rows):
        v = (orin, thor)[b]
        y = top + 24 + k * 40
        if b == 0:
            parts.append(text(x0 - 14, y + 18, label, "sub", "end"))
        parts += [rect(left, y, 220, 24, "none", EDGE, 1, 5),
                  f'  <rect x="{left}" y="{y}" width="{v * scale:.1f}" height="24" rx="5" '
                  f'fill="{KINDS["good" if label == "cuBLAS" else kind][0]}"/>',
                  mono(left + 220 - 6, y + 17, f"{v:,.0f}", "end", INK)]
    parts.append(mono(left, top + 24 + 6 * 40 + 14, f"bar = peak, {peak:,} GFLOP/s", "start", MUTED))
parts += [note(40, 418, "GFLOP/s for N = 4,096. Each bar's full width is the board's FP32 peak; the best", room=600),
          note(40, 436, "kernel of the chapter reached 40% of it on both boards, cuBLAS 63% and 76%.", room=600)]
save(OUT + "ch09-gflops.svg", 660, 458,
     "Bars of SGEMM speed for N = 4096 on 2 boards, each bar's full width the board's FP32 peak. Orin NX, "
     "peak 2,400 GFLOP/s: by column 34, by row 180, tiles of 16 278, tiles of 32 219, 4 by 4 per thread "
     "954, cuBLAS 1,518. Thor, peak 8,100: 118, 594, 945, 774, 3,273 and 6,173.",
     "SGEMM speed, measured on 2 boards", "From 1 thread per element to register tiles, and cuBLAS.", parts)
