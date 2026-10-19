"""The figures of chapter 7, drawn with figkit and figparts.

    python3 tools/draw_ch07.py

The bandwidths in figures 7.2 and 7.6 are the ones measured with code/ch07_memory/memory.cu
on a Jetson Orin NX and a Jetson AGX Thor (the runs quoted in the chapter).
"""
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, bar, card, legend, line, mascot, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"


def rect(x, y, w, h, fill, stroke="none", sw=1.2, rx=3):
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


# 7.1  Sectors fetched by 1 warp: stride 1, 2 and 8.
parts = []
x0, sw = 150, 15.5                    # 1 sector of 32 bytes = 8 floats
for k, (stride, kind, summary) in enumerate([(1, "good", "4 sectors, 128 B, all used"),
                                             (2, "store", "8 sectors, 256 B, half used"),
                                             (8, "danger", "32 sectors, 1,024 B, 1/8 used")]):
    y = 128 + k * 96
    sectors = min(32, 4 * stride)
    parts.append(text(x0 - 16, y + 16, f"stride {stride}", "name", "end"))
    for s in range(32):
        fetched = s < sectors
        parts.append(rect(x0 + s * sw, y, sw - 2, 24, KINDS[kind][1] if fetched else "#FFFFFF",
                          KINDS[kind][0] if fetched else EDGE))
        if fetched:
            used = min(8, max(0, 8 // stride)) if stride < 8 else 1
            parts.append(rect(x0 + s * sw + 1, y + 1, (sw - 4) * used / 8, 22, KINDS[kind][0], rx=2))
    parts.append(text(x0, y + 46, summary, "sub"))
parts += [legend(x0, 418, [("good", "sector fetched"), ("idle", "not fetched")]),
          note(40, 456, "Memory moves in 32-byte sectors. A warp's 32 floats fill 4 sectors when neighbours", room=600),
          note(40, 474, "are adjacent; at stride 8 or more every float needs a sector of its own.", room=600)]
save(OUT + "ch07-sectors.svg", 660, 496,
     "Three rows of 32 memory sectors of 32 bytes. Stride 1: the warp's 32 floats fill 4 sectors completely. "
     "Stride 2: 8 sectors, each half used. Stride 8: 32 sectors, each with 1 float of 4 useful bytes. The dark part "
     "of each sector is the bytes the warp uses.",
     "Sectors fetched by 1 warp", "Each square is a 32-byte sector; the dark part is what the warp uses.", parts)


# 7.2  Measured useful bandwidth against stride, on 2 boards.
strides = [1, 2, 4, 8, 16, 32]
orin = [68.0, 61.0, 35.1, 20.9, 14.4, 17.6]
thor = [228.6, 172.9, 104.6, 58.0, 30.0, 50.9]
x0, y0, h, scale = 110, 380, 250, 250 / 240
parts = []
for g in (0, 50, 100, 150, 200):
    parts += [line(x0, y0 - g * scale, x0 + 520, y0 - g * scale, EDGE, 1),
              mono(x0 - 10, y0 - g * scale + 4, str(g), "end", MUTED)]
for k, s in enumerate(strides):
    cx = x0 + 30 + k * 84
    for j, (v, kind) in enumerate(((orin[k], "client"), (thor[k], "outside"))):
        parts.append(f'  <rect x="{cx + j * 26}" y="{y0 - v * scale:.1f}" width="22" height="{v * scale:.1f}" '
                     f'rx="4" fill="{KINDS[kind][0]}"/>')
    parts.append(mono(cx + 24, y0 + 20, f"stride {s}", "middle", MUTED))
parts += [text(x0 - 10, 112, "GB/s useful", "sub", "end"),
          legend(x0, y0 + 46, [("client", "Jetson Orin NX"), ("outside", "Jetson AGX Thor")]),
          note(40, 466, "Useful bandwidth falls with the sectors wasted, from stride 1 to stride 16.", room=600)]
save(OUT + "ch07-stride-times.svg", 660, 488,
     "Grouped bars of useful bandwidth against stride. Orin NX: 68, 61, 35, 21, 14 and 18 GB/s for strides 1, 2, "
     "4, 8, 16 and 32. Thor: 229, 173, 105, 58, 30 and 51 GB/s.",
     "Useful bandwidth against stride", "Measured by memory.cu, 16,777,216 reads per run.", parts)


# 7.3  The naive transpose: a warp reads a row (coalesced) and writes a column (strided).
def grid(x, y, n, cell, highlight=(), kind="client"):
    out = []
    for r in range(n):
        for c in range(n):
            hot = (r, c) in highlight
            out.append(rect(x + c * cell, y + r * cell, cell - 2, cell - 2,
                            KINDS[kind][1] if hot else "#FFFFFF", KINDS[kind][0] if hot else EDGE))
    return out


cell = 22
row = {(2, c) for c in range(8)}
col = {(r, 2) for r in range(8)}
parts = [text(60, 120, "input: 1 warp reads a row", "sub")] + grid(60, 132, 8, cell, row, "good")
parts += [text(440, 120, "output: it writes a column", "sub")] + grid(440, 132, 8, cell, col, "danger")
parts += [wire([(240, 132 + 2.5 * cell), (440, 132 + 2.5 * cell)]),
          mono(340, 132 + 2.5 * cell - 12, "out[x][y] = in[y][x]", "middle", INK),
          note(40, 340, "The read is coalesced: 32 neighbouring floats, 4 sectors. The write is not: 32 floats", room=600),
          note(40, 358, "in 32 different rows, 32 sectors, and 7 of every 8 bytes written to memory are waste.", room=600)]
save(OUT + "ch07-transpose-naive.svg", 660, 380,
     "Two 8 by 8 grids. In the input, a warp reads row 2: neighbouring elements, coalesced. In the output, the "
     "same warp writes column 2: 1 element in each of 8 rows, strided.",
     "The naive transpose", "Reading rows is coalesced; writing them as columns is not.", parts)


# 7.4  The tiled transpose: through a tile in shared memory, both global accesses are rows.
cell = 22
parts = [zone(16, 104, 200, 236, "input"), zone(230, 104, 200, 236, "shared memory tile"),
         zone(444, 104, 200, 236, "output")]
parts += grid(46, 150, 6, cell, {(1, c) for c in range(6)}, "good")
parts += grid(262, 150, 6, cell, {(1, c) for c in range(6)} | {(r, 1) for r in range(6)}, "store")
parts += grid(474, 150, 6, cell, {(1, c) for c in range(6)}, "good")
parts += [text(46, 300, "read a row", "sub"), text(262, 300, "write a row, read a column", "sub"),
          text(474, 300, "write a row", "sub"),
          wire([(180, 172), (262, 172)]), wire([(394, 172), (474, 172)]),
          note(16, 372, "Global memory sees only rows, so both the read and the write are coalesced. The column", room=628),
          note(16, 390, "access happens in shared memory, which has no sectors to waste; it has banks (figure 7.5).", room=628)]
save(OUT + "ch07-tile.svg", 660, 412,
     "Three panels. A block reads a row of its input tile, coalesced, and writes it as a row of a tile in shared "
     "memory. After a barrier it reads a column of the tile and writes it as a row of the output tile, also "
     "coalesced.",
     "The tiled transpose", "A tile in shared memory turns the strided write into a coalesced one.", parts)


# 7.5  Banks: a column of a 32 x 32 tile hits 1 bank 32 times; with 33 columns it hits 32 banks.
parts = []
for p, (pad, kind, title) in enumerate([(0, "danger", "tile[32][32]: column 0 is all in bank 0"),
                                        (1, "good", "tile[32][33]: tile[r][0] falls in bank r")]):
    y0 = 130 + p * 168
    parts.append(text(60, y0 - 12, title, "sub"))
    for b in range(16):
        parts.append(mono(80 + b * 34 + 15, y0 + 6, str(b), "middle", MUTED))
    for r in range(8):
        bank = (r * (32 + pad)) % 32
        for b in range(16):
            hot = b == bank
            parts.append(rect(80 + b * 34, y0 + 14 + r * 15, 30, 12,
                              KINDS[kind][1] if hot else "#FFFFFF", KINDS[kind][0] if hot else EDGE))
        parts.append(mono(72, y0 + 24 + r * 15, f"r{r}", "end", MUTED))
parts += [note(40, 470, "Banks 0 to 15 of 32, rows 0 to 7 of the tile. Element tile[r][0] sits at word", room=600),
          note(40, 488, "r × 32 or r × 33; its bank is that word mod 32: always 0, or r.", room=600)]
save(OUT + "ch07-banks.svg", 660, 510,
     "Two grids of shared-memory banks. Top, a tile of 32 by 32 floats: tile[r][0] for rows 0 to 7 all fall in "
     "bank 0, so reading the column takes 32 turns. Bottom, a tile of 32 by 33: tile[r][0] falls in bank r, a "
     "diagonal, so the column is read in 1 turn.",
     "Bank conflicts and padding", "1 extra column moves each row of the tile to the next bank.", parts)


# 7.6  Measured transpose speeds on 2 boards.
rows = [("copy", 78.3, 230.6), ("naive", 16.5, 80.7), ("tiled", 29.2, 96.4), ("tiled, padded", 44.8, 143.1)]
x0, scale = 170, 400 / 240
parts = []
for g in (0, 50, 100, 150, 200):
    parts += [line(x0 + g * scale, 104, x0 + g * scale, 116 + 4 * 64, EDGE, 1),
              mono(x0 + g * scale, 116 + 4 * 64 + 18, str(g), "middle", MUTED)]
for k, (label, orin_v, thor_v) in enumerate(rows):
    y = 116 + k * 64
    parts.append(text(x0 - 14, y + 26, label, "sub", "end"))
    for j, (v, kind) in enumerate(((orin_v, "client"), (thor_v, "outside"))):
        parts += [f'  <rect x="{x0}" y="{y + j * 24}" width="{v * scale:.1f}" height="20" rx="5" fill="{KINDS[kind][0]}"/>',
                  mono(x0 + v * scale + 8, y + j * 24 + 15, f"{v} GB/s", "start", INK)]
parts += [legend(x0, 116 + 4 * 64 + 44, [("client", "Jetson Orin NX"), ("outside", "Jetson AGX Thor")])]
save(OUT + "ch07-transpose-times.svg", 660, 460,
     "Bars of transpose bandwidth for a 4096 by 4096 matrix. Orin NX: copy 78 GB/s, naive 17, tiled 29, tiled "
     "and padded 45. Thor: copy 231 GB/s, naive 81, tiled 96, tiled and padded 143.",
     "Transpose speed, measured on 2 boards", "4096 × 4096 floats, average of 20 runs, by memory.cu.", parts)
