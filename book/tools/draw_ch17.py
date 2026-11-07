"""The figures of chapter 17 (tiled matrix multiplication in Rust), drawn with figkit and figparts.

    python3 tools/draw_ch17.py

The speeds in the last figure are the ones measured with code/oxide/ch17_matmul_tiled.
"""
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, card, legend, line, mono, note, save,
                      tag, text, wire, zone)

OUT = "src/figures/"

# GFLOP/s for N = 4,096 on the measured GPU, from the run quoted in the chapter, with the
# memory limit of each tile width: T x 25.5 GFLOP/s.
SPEEDS = [("naive", 193.0, 25.5), ("tiled16, while", 274.0, 408), ("tiled16, for", 201.8, 408),
          ("tiled::<8>", 146.2, 204), ("tiled::<16>", 271.6, 408), ("tiled::<32>", 217.9, 816),
          ("tiled_fn::<16>", 168.2, 408)]
SPEEDS_NOTE = ("Tiles of 16 reach 67% of their memory limit and match the C++ kernel of Chapter 9;",
               "tiles of 32 reach 27% of theirs, with 1 block of 1,024 threads per SM.")
SPEEDS_ALT = ("Horizontal bars of measured speed in GFLOP/s for N = 4,096, on a scale from 0 to 1,600. "
              "This chapter's kernels: naive 193.0, tiled16 with a while loop 274.0, tiled16 with a "
              "for loop 201.8, tiled with T = 8 146.2, T = 16 271.6, T = 32 217.9, tiled_fn with "
              "T = 16 168.2. Dashed marks show the memory limit of each tile width: 25.5 for the "
              "naive kernel, 204 for T = 8, 408 for T = 16 and 816 for T = 32. Below, 2 results of "
              "Chapter 9: the CUDA C++ kernel with 16 by 16 tiles, 273.9, and cuBLAS, 1,495.3 GFLOP/s.")


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def grid(x, y, side, cell, kind, hot=(), labels=False):
    """A side x side tile; the cells in `hot` take the colours of `kind`."""
    parts = []
    for r in range(side):
        for c in range(side):
            body = str(r * side + c) if labels else ""
            parts.append(box(x + c * cell, y + r * cell, cell - 3, cell - 3, kind, body, (r, c) in hot))
    return parts


# 17.1  1 copy of each tile per block.
parts = [zone(16, 104, 300, 300, "block (0, 0)"), zone(344, 104, 300, 300, "block (1, 0)")]
for x0, row in ((40, 0), (368, 1)):
    parts += [text(x0, 146, "TILE_A", "mono"), text(x0 + 140, 146, "TILE_B", "mono")]
    parts += grid(x0, 156, 4, 28, "client", {(r, c) for r in range(4) for c in range(4)})
    parts += grid(x0 + 140, 156, 4, 28, "outside", {(r, c) for r in range(4) for c in range(4)})
    parts += [note(x0, 300, "rows 0 to 3 of A,", room=130),
              note(x0, 318, "4 columns per step", room=130),
              note(x0 + 140, 300, "4 rows of B,", room=130),
              note(x0 + 140, 318, f"columns {4 * row} to {4 * row + 3}", room=130),
              tag(x0, 350, "16 threads see this copy", "good")]
parts += [note(16, 436, "static mut TILE_A: SharedArray<f32, 16> declares 1 array per block, not 1 for the", room=628),
          note(16, 454, "kernel. The 16 threads of a block share their copy; another block never sees it.", room=628),
          note(16, 472, "Both blocks copy rows 0 to 3 of A, each into its own TILE_A. With T = 16 a tile holds", room=628),
          note(16, 490, "256 floats, 1,024 bytes, so a block uses 2,048 bytes of shared memory.", room=628)]
save(OUT + "ch17-copies.svg", 660, 512,
     "2 zones, block (0, 0) and block (1, 0), each holding its own TILE_A and TILE_B as 4 by 4 grids. "
     "Both blocks load rows 0 to 3 of A, each into its own copy; block (0, 0) loads columns 0 to 3 of B and "
     "block (1, 0) columns 4 to 7. Each copy is seen by the 16 threads of its block only.",
     "1 copy of each tile per block", "A static SharedArray is allocated once for every block.", parts)


# 17.2  Slot y * T + x belongs to thread (x, y).
parts = [zone(16, 104, 300, 330, "T = 4: 16 threads, 16 slots"), zone(344, 104, 300, 330, "T = 16")]
parts += grid(56, 150, 4, 52, "client", {(1, 3)}, labels=True)
parts += [text(56, 142, "slot = y × 4 + x", "mono"),
          note(40, 382, "Thread (3, 1) writes slot 1 × 4 + 3 = 7.", room=270),
          note(40, 400, "No other (x, y) gives 7.", room=270),
          card(368, 140, 252, 74, "client", "256 threads", ["x and y from 0 to 15"]),
          card(368, 228, 252, 74, "good", "largest slot 255", ["15 × 16 + 15 = 255 < 256"]),
          card(368, 316, 252, 74, "outside", "256 distinct slots", ["1 writer for each slot"])]
parts += [note(16, 466, "The formula y × T + x gives a different number for every thread of a T × T block, and", room=628),
          note(16, 484, "the largest is (T − 1) × T + (T − 1) = T² − 1. Each slot of the tile has exactly 1 writer.",
               room=628)]
save(OUT + "ch17-slots.svg", 660, 506,
     "Left, a 4 by 4 tile with its slots numbered 0 to 15 row by row; slot 7 is highlighted, written by "
     "thread x = 3, y = 1, because 1 times 4 plus 3 is 7. Right, 3 cards for T = 16: 256 threads with x and y "
     "from 0 to 15; the largest slot is 15 times 16 plus 15, 255, less than 256; 256 distinct slots, 1 writer "
     "each.",
     "Each slot has 1 writer", "Thread (x, y) writes slot y × T + x, and no other thread does.", parts)


# 17.3  The 2 barriers and the 2 dependences they enforce.
cols = [("write slot", 92), ("barrier 1", 196), ("read row/col", 300), ("barrier 2", 404),
        ("write slot", 508)]
parts = [text(92, 124, "step k0", "sub"), text(508, 124, "step k0 + 16", "sub")]
for name, x in cols:
    parts.append(mono(x + 40, 144, name, "middle", MUTED))
rows = [("thread (0, 0)", 176), ("thread (5, 0)", 250)]
for name, y in rows:
    parts.append(text(80, y + 22, name, "sub", "end"))
    for label, x in cols:
        if label.startswith("barrier"):
            continue
        parts.append(box(x, y, 80, 34, "client" if label == "write slot" else "outside",
                         label.split()[0]))
for _, x in cols[1::2]:
    parts.append(f'  <rect x="{x + 34}" y="170" width="12" height="128" rx="4" fill="{INK}"/>')
parts += [wire([(132, 284), (132, 322), (270, 322), (270, 193), (296, 193)]),
          tag(148, 313, "read after write", "idle"),
          wire([(340, 176), (340, 160), (494, 160), (494, 267), (504, 267)], kind="danger"),
          tag(374, 151, "write after read", "danger")]
parts += [note(16, 362, "Thread (5, 0) writes slot 5 of TILE_A; thread (0, 0) reads it in its row. Barrier 1 keeps", room=628),
          note(16, 380, "the read after the write. At the next step thread (5, 0) overwrites slot 5; barrier 2", room=628),
          note(16, 398, "keeps that write after every read of the old value. Without either barrier the result", room=628),
          note(16, 416, "depends on the order in which warps happen to run.", room=628)]
save(OUT + "ch17-barriers.svg", 660, 438,
     "A timeline of 2 threads over 1 step and the start of the next. Each step is write slot, barrier 1, "
     "read, barrier 2. An arrow labelled read after write runs from thread (5, 0)'s write of slot 5 to "
     "thread (0, 0)'s read, across barrier 1. An arrow labelled write after read runs from thread (0, 0)'s "
     "read to thread (5, 0)'s next write, across barrier 2.",
     "Why 2 barriers", "Barrier 1 orders write then read; barrier 2 orders read then overwrite.", parts)


# 17.4  The 3 obligations behind the unsafe block, and who discharges each.
parts = [card(24, 112, 612, 74, "client", "1. Every index is less than 256",
              ["no bounds check on shared memory: the launch contract fixes x, y < 16"]),
         card(24, 202, 612, 74, "outside", "2. Each slot has 1 writer per step",
              ["slot y × 16 + x is different for every thread (figure 17.2)"]),
         card(24, 292, 612, 74, "danger", "3. No slot is read while it is written",
              ["thread::sync_threads() before the reads and after them"]),
         note(24, 400, "The compiler cannot check any of the 3: a static mut is shared by every thread of a", room=612),
         note(24, 418, "block, and the barriers are invisible to Rust's type system. The SAFETY comments in", room=612),
         note(24, 436, "the kernel record which of the 3 each unsafe block relies on.", room=612)]
save(OUT + "ch17-obligations.svg", 660, 458,
     "3 cards. 1: every index is less than 256; shared memory is not bounds-checked, and the launch "
     "contract fixes x and y below 16. 2: each slot has 1 writer per step, because y times 16 plus x differs "
     "for every thread. 3: no slot is read while it is written, which the 2 calls of sync_threads ensure.",
     "What the unsafe block promises", "3 obligations that the compiler cannot check.", parts)


# 17.5  1 generic source, 3 kernels.
parts = [card(186, 112, 288, 64, "idle", "tiled<const T: usize>", ["1 source"])]
targets = [118, 330, 542]
parts.append(f'  <path d="M330,176 V196 M{targets[0]},196 H{targets[-1]}" fill="none" '
             f'stroke="{WIRE}" stroke-width="1.8"/>')
parts.append(f'  <circle cx="330" cy="196" r="3" fill="{WIRE}"/>')
parts += [arrow(t, 196, 220) for t in targets]
for t, (size, kind) in zip(targets, ((8, "client"), (16, "good"), (32, "outside"))):
    threads, smem, blocks = size * size, 2 * size * size * 4, (4096 // size) ** 2
    parts += [box(t - 96, 224, 192, 34, kind, f"tiled::<{size}>"),
              mono(t, 284, f"{size} × {size} = {threads:,} threads", "middle", INK),
              mono(t, 304, f"2 × {threads:,} × 4 = {smem:,} B", "middle", INK),
              mono(t, 324, f"{blocks:,} blocks", "middle", INK),
              mono(t, 344, f"{size} FLOP per float", "middle", KINDS[kind][0])]
parts += [note(16, 384, "The compiler builds 1 kernel for each value of T that the host program names. The block", room=628),
          note(16, 402, "shape, the shared memory and the reuse all grow with T; for N = 4,096 the number of", room=628),
          note(16, 420, "blocks is (4,096 ÷ T)².", room=628)]
save(OUT + "ch17-generic.svg", 660, 442,
     "1 card, tiled with a const generic T, fans out into 3 kernels. tiled::<8>: 64 threads, 512 bytes of "
     "shared memory, 262,144 blocks, 8 FLOP per float. tiled::<16>: 256 threads, 2,048 bytes, 65,536 blocks, "
     "16 FLOP per float. tiled::<32>: 1,024 threads, 8,192 bytes, 16,384 blocks, 32 FLOP per float.",
     "1 source, 3 kernels", "A const generic tile size, specialised by the host program.", parts)


# 17.6  The dynamic shared memory of tiled::<16>.
x0, wide = 60, 540
parts = [text(x0, 128, "dynamic shared memory of 1 block, 2,048 bytes", "sub")]
parts += [box(x0, 140, wide / 2 - 3, 44, "client", "tile A: bytes 0 to 1,023"),
          box(x0 + wide / 2, 140, wide / 2 - 3, 44, "outside", "tile B: bytes 1,024 to 2,047")]
for g in (0, 1024, 2048):
    x = x0 + wide * g / 2048
    parts += [line(x, 192, x, 202, EDGE, 1.2), mono(x, 218, f"{g:,}", "middle", MUTED)]
parts += [wire([(x0 + 60, 262), (x0 + 60, 188)]), wire([(x0 + wide / 2 + 60, 262), (x0 + wide / 2 + 60, 188)]),
          mono(x0, 282, "DynamicSharedArray::<f32>::get()", "start", INK),
          mono(x0 + wide / 2, 282, "::offset(T * T * 4) = offset(1,024)", "start", INK),
          note(x0, 326, "The launch configuration asks for 2 × 16 × 16 × 4 = 2,048 bytes; the contract", room=540),
          note(x0, 344, "dynamic_shared_range = (512, 8192) rejects a launch outside that range.", room=540)]
save(OUT + "ch17-dynamic.svg", 660, 366,
     "A bar of 2,048 bytes of dynamic shared memory for tiled with T = 16. Bytes 0 to 1,023 hold tile A, "
     "reached with DynamicSharedArray get; bytes 1,024 to 2,047 hold tile B, reached with offset of T times T "
     "times 4, which is 1,024. The launch asks for 2,048 bytes and the contract allows 512 to 8,192.",
     "Tiles in dynamic shared memory", "The size is chosen when the kernel is launched.", parts)


# 17.7  Measured speeds, with the memory limit of each tile size and 2 results of chapter 9.
x0, wide, top = 196, 330, 128
scale = 1600
rows = [(name, value, limit, "client") for name, value, limit in SPEEDS]
rows += [("C++ tiles 16 (Ch. 9)", 273.9, None, "store"), ("cuBLAS (Ch. 9)", 1495.3, None, "outside")]
parts = [legend(x0, 104, [("client", "this chapter"), ("store", "Chapter 9, CUDA C++"),
                          ("outside", "cuBLAS")])]
for g in range(0, scale + 1, 400):
    x = x0 + wide * g / scale
    parts += [line(x, top, x, top + len(rows) * 36, EDGE, 1),
              mono(x, top + len(rows) * 36 + 18, f"{g:,}", "middle", MUTED)]
for k, (name, value, limit, kind) in enumerate(rows):
    y = top + 8 + k * 36
    w = wide * value / scale
    parts += [text(x0 - 12, y + 13, name, "sub", "end"),
              f'  <rect x="{x0}" y="{y}" width="{w:.1f}" height="18" rx="4" fill="{KINDS[kind][0]}"/>',
              mono(x0 + wide + 12, y + 13, f"{value:,.1f} GFLOP/s", "start", INK)]
    if limit:
        lx = x0 + wide * limit / scale
        parts.append(f'  <path d="M{lx:.1f},{y - 4} V{y + 22}" stroke="{KINDS["danger"][0]}" '
                     f'stroke-width="2" stroke-dasharray="3 2"/>')
bottom = top + len(rows) * 36 + 18
parts += [text(x0 - 12, bottom, "GFLOP/s", "sub", "end"),
          legend(x0, bottom + 30, [("danger", "memory limit: T × 25.5 GFLOP/s")]),
          note(16, bottom + 62, SPEEDS_NOTE[0], room=628),
          note(16, bottom + 80, SPEEDS_NOTE[1], room=628)]
save(OUT + "ch17-speeds.svg", 660, bottom + 102, SPEEDS_ALT,
     "Measured speeds, N = 4,096", "GFLOP/s on the measured GPU; dashed marks: the memory limit.", parts)
