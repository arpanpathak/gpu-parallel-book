"""The figures of chapter 3, drawn with figkit and figparts.

    python3 tools/draw_ch03.py

Each figure shows one step of a CUDA program with numbers small enough to check
by hand: a launch of 3 blocks of 8 threads, an 8 × 8 image, n = 1,000.
"""
from figkit import swatches
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, bar, card, cylinder, legend, line,
                      mascot, mono, note, save, tag, text, tile, width_of, wire, zone)

OUT = "src/figures/"


# 3.1  Host and device: 2 processors, 2 memories, explicit copies.
parts = [zone(16, 104, 280, 300, "host"), zone(364, 104, 280, 300, "device"),
         card(36, 136, 240, 60, "client", "CPU", "runs main() and launches", glyph="terminal"),
         cylinder(56, 262, 200, 92, "store", "host memory", ["h_a, h_b, h_c"]),
         card(384, 136, 240, 60, "service", "GPU", "runs the kernel's threads", glyph="chip"),
         cylinder(404, 262, 200, 92, "outside", "global memory", ["d_a, d_b, d_c"]),
         wire([(156, 196), (156, 262)], arrow=False), wire([(504, 196), (504, 262)], arrow=False),
         wire([(276, 166), (384, 166)], label="launch"),
         wire([(256, 292), (404, 292)], label="copy in"),
         wire([(404, 330), (256, 330)], label="copy out"),
         note(16, 436, "Copies use cudaMemcpy. A host pointer must not be read on the device, nor a", room=628),
         note(16, 454, "device pointer on the host.", room=628)]
save(OUT + "ch03-host-device.svg", 660, 476,
     "Two zones. Host: a CPU card that runs main and launches the kernel, above host memory holding h_a, "
     "h_b and h_c. Device: a GPU card that runs the kernel's threads, above global memory holding d_a, d_b "
     "and d_c. An arrow labelled launch runs from the CPU to the GPU. An arrow labelled copy in runs from "
     "host memory to global memory, and one labelled copy out runs back.",
     "Host and device", "A CUDA program runs on 2 processors, each with its own memory.", parts)


# 3.3  A 1-D launch: kernel<<<3, 8>>>.
parts = []
for b in range(3):
    zx = 16 + b * 212
    parts.append(zone(zx, 112, 200, 196, f"block {b}"))
    parts.append(mono(zx + 14, 140, f"blockIdx.x = {b}", "start", INK))
    for t in range(8):
        x = zx + 14 + t * 22
        hot = b == 2 and t == 5
        kind = "outside" if hot else "service"
        parts.append(bar(x, 154, 21, 30, kind, str(t), solid=hot))
        parts.append(bar(x, 228, 21, 30, kind, str(b * 8 + t), solid=hot))
    parts += [note(zx + 14, 204, "threadIdx.x", fill=MUTED), note(zx + 14, 278, "index i", fill=MUTED)]
parts += [card(40, 338, 580, 62, "outside", "i = blockIdx.x × blockDim.x + threadIdx.x",
               "the highlighted thread: 2 × 8 + 5 = 21")]
save(OUT + "ch03-launch-1d.svg", 660, 424,
     "Three zones, blockIdx.x = 0, 1 and 2, each with 8 threads. In each block a row shows threadIdx.x from "
     "0 to 7 and a row below shows the index i: 0 to 7 in block 0, 8 to 15 in block 1, 16 to 23 in block 2. "
     "Thread 5 of block 2, index 21, is highlighted. A card reads i = blockIdx.x × blockDim.x + threadIdx.x, "
     "and for the highlighted thread 2 × 8 + 5 = 21.",
     "The launch kernel<<<3, 8>>>", "3 blocks of 8 threads: 24 threads with indexes 0 to 23.", parts)


# 3.5  Rounding the grid up: n = 1,000 with blocks of 256.
parts = [text(40, 120, "n = 1,000 elements, 256 threads per block", "name"),
         text(40, 140, "blocks = (1,000 + 255) / 256 = 1,255 / 256 = 4, rounded down", "sub")]
x0, scale = 40, 580 / 1024
for b in range(4):
    x = x0 + b * 256 * scale
    parts.append(bar(x + 1, 164, 256 * scale - 2, 34, "service", f"block {b}"))
masked = 1000 * scale
parts += [f'  <rect x="{x0 + masked:.1f}" y="164" width="{24 * scale:.1f}" height="34" rx="3" fill="{KINDS["danger"][1]}"'
          f' stroke="{KINDS["danger"][0]}" stroke-width="1.3"/>',
          line(x0, 216, x0 + 1000 * scale, 216, KINDS["good"][0], 3),
          text(x0, 236, "indexes 0 to 999: one thread per element", "sub"),
          tag(x0 + 1000 * scale, 246, "indexes 1,000 to 1,023: i < n is false", "danger", "end"),
          card(40, 290, 580, 60, "danger", "24 threads have no element",
               "the guard if (i < n) makes them do nothing")]
save(OUT + "ch03-rounding.svg", 660, 374,
     "A bar of 1,024 threads in 4 blocks of 256, for n = 1,000 elements. The number of blocks is "
     "(1,000 + 255) / 256 = 4, rounded down. A green line marks indexes 0 to 999, one thread per element. "
     "A red sliver at the end marks indexes 1,000 to 1,023, 24 threads for which i < n is false; the guard "
     "makes them do nothing.",
     "Rounding the grid up", "Enough whole blocks to cover every element, and a guard for the rest.", parts)


# 3.3  A 2-D launch over an 8 × 8 image: 2 × 2 blocks of 4 × 4 threads.
colours = {(0, 0): "client", (1, 0): "service", (0, 1): "store", (1, 1): "outside"}
gx, gy, cs = 60, 140, 34
parts = []
for ix in range(8):
    parts.append(mono(gx + ix * cs + cs / 2, gy - 10, str(ix), "middle", MUTED))
for iy in range(8):
    parts.append(mono(gx - 12, gy + iy * cs + cs / 2 + 4, str(iy), "end", MUTED))
for iy in range(8):
    for ix in range(8):
        b = (ix // 4, iy // 4)
        hot = (ix, iy) == (6, 1)
        accent, tint = KINDS[colours[b]]
        parts.append(f'  <rect x="{gx + ix * cs + 1}" y="{gy + iy * cs + 1}" width="{cs - 2}" height="{cs - 2}" rx="4"'
                     f' fill="{accent if hot else tint}" stroke="{accent}" stroke-width="1"/>')
        parts.append(text(gx + ix * cs + cs / 2, gy + iy * cs + cs / 2 + 4, str(iy * 8 + ix), "pill", "middle",
                          "#FFFFFF" if hot else accent))

px = gx + 8 * cs + 30
parts += [tag(px, 140, "block (1, 0)", "service"),
          note(px, 182, "blockIdx = (1, 0)"), note(px, 202, "threadIdx = (2, 1)"),
          note(px, 222, "blockDim = (4, 4)"),
          note(px, 258, "ix = 1 × 4 + 2 = 6"), note(px, 278, "iy = 0 × 4 + 1 = 1"),
          note(px, 314, "pixel number"), note(px, 334, "iy × width + ix"),
          note(px, 354, "= 1 × 8 + 6 = 14"),
          legend(gx, gy + 8 * cs + 30, [("client", "block (0, 0)"), ("service", "(1, 0)"),
                                        ("store", "(0, 1)"), ("outside", "(1, 1)")]),
          note(gx, gy + 8 * cs + 60, "Numbers above the grid are ix, the column; numbers on its left are iy, the row.",
               room=560)]
save(OUT + "ch03-image-2d.svg", 660, gy + 8 * cs + 74,
     "An 8 by 8 image, its pixels numbered 0 to 63 row by row, coloured by the 4 blocks of 4 by 4 threads "
     "that cover it. Pixel 14, at column 6 and row 1, is highlighted. It belongs to block (1, 0), where the "
     "thread is at position (2, 1). Its column is 1 × 4 + 2 = 6, its row 0 × 4 + 1 = 1, and its pixel number "
     "1 × 8 + 6 = 14.",
     "A 2-D launch over an 8 × 8 image", "4 blocks of 4 × 4 threads; each thread handles 1 pixel.", parts)


# 3.8  A launch returns at once; the host waits only where it asks to.
gx, s = 150, 1.0
parts = [text(gx - 14, 136, "host", "name", "end"), text(gx - 14, 214, "device", "name", "end"),
         bar(gx, 116, 90, 30, "store", "copy in"),
         bar(gx + 96, 116, 50, 30, "client", "launch"),
         bar(gx + 152, 116, 130, 30, "idle", "other host work"),
         bar(gx + 288, 116, 100, 30, "danger", "synchronize", dashed=True),
         bar(gx + 394, 116, 80, 30, "store", "copy out"),
         bar(gx, 194, 90, 30, "store", "copy in"),
         bar(gx + 104, 194, 284, 30, "service", "kernel runs on the GPU"),
         bar(gx + 394, 194, 80, 30, "store", "copy out"),
         wire([(gx + 121, 146), (gx + 121, 194)]),
         note(gx, 262, "The launch returns at once, before the kernel has run.", room=490),
         note(gx, 280, "cudaDeviceSynchronize waits until the kernel finishes, and", room=490),
         note(gx, 298, "reports any error that happened inside it.", room=490),
         note(gx + 300, 106, "host waits", fill=KINDS["danger"][0])]
save(OUT + "ch03-async.svg", 660, 322,
     "Two timelines. Host: copy in, a short launch, other host work, a dashed wait in cudaDeviceSynchronize, "
     "copy out. Device: copy in, then the kernel runs, starting just after the launch and ending when the "
     "host's wait ends, then copy out. The launch returns at once; the host waits only in "
     "cudaDeviceSynchronize, where errors inside the kernel are reported.",
     "A launch is asynchronous", "The host carries on while the GPU runs the kernel.", parts)


# 3.9  How nvcc builds 1 executable from a .cu file.
parts = [card(220, 96, 220, 54, "store", "vector_add.cu", "host code and kernels", glyph="doc"),
         card(220, 186, 220, 54, "outside", "nvcc", "splits the file in 2", glyph="gear"),
         wire([(330, 150), (330, 186)]),
         card(40, 290, 250, 54, "client", "host code", "compiled by g++", glyph="terminal"),
         card(370, 290, 250, 54, "service", "kernels", "compiled to PTX", glyph="chip"),
         wire([(330, 240), (330, 262), (165, 262), (165, 290)]),
         wire([(330, 240), (330, 262), (495, 262), (495, 290)]),
         f'  <circle cx="330" cy="262" r="3" fill="{WIRE}"/>',
         card(370, 378, 250, 54, "service", "SASS for sm_87", "machine code of this GPU", glyph="chip"),
         wire([(495, 344), (495, 378)]),
         card(160, 482, 340, 54, "good", "vector_add", "1 executable, with both parts", glyph="terminal"),
         wire([(165, 344), (165, 456), (330, 456)], arrow=False),
         wire([(495, 432), (495, 456), (330, 456)], arrow=False),
         f'  <circle cx="330" cy="456" r="3" fill="{WIRE}"/>',
         arrow(330, 456, 482)]
save(OUT + "ch03-nvcc.svg", 660, 560,
     "vector_add.cu, holding host code and kernels, goes into nvcc, which splits it in 2. The host code is "
     "compiled by g++. The kernels are compiled to PTX, a portable form, and then to SASS for sm_87, the "
     "machine code of this GPU. Both parts end up in 1 executable, vector_add.",
     "How nvcc builds a CUDA program", "1 source file, 2 compilers, 1 executable.", parts)


# 3.3  A 3-D block, dim3(4, 4, 2): 2 layers of 4 × 4 threads, numbered x first, then y, then z.
cs = 40
parts = []
for z in (0, 1):
    ox, oy = 60 + z * 300, 150
    kind = "store" if z == 0 else "service"
    accent, tint = KINDS[kind]
    parts.append(zone(ox - 20, oy - 34, 4 * cs + 40, 4 * cs + 56, f"layer z = {z}"))
    for y in range(4):
        for x in range(4):
            lin = x + 4 * y + 16 * z
            hot = (x, y, z) == (2, 1, 1)
            parts.append(f'  <rect x="{ox + x * cs + 2}" y="{oy + y * cs + 2}" width="{cs - 4}" height="{cs - 4}" rx="5"'
                         f' fill="{accent if hot else tint}" stroke="{accent}" stroke-width="1"/>')
            parts.append(text(ox + x * cs + cs / 2, oy + y * cs + cs / 2 + 4, str(lin), "pill", "middle",
                              "#FFFFFF" if hot else accent))
    for x in range(4):
        parts.append(mono(ox + x * cs + cs / 2, oy - 6 + 4 * cs + 28, f"x={x}", "middle", MUTED))
parts += [wire([(240, 230), (340, 230)], label="then"),
          card(40, 392, 580, 60, "service", "thread (x, y, z) = (2, 1, 1) is number 2 + 4 × 1 + 16 × 1 = 22",
               "x counts 1 each, a row of 4 counts 4, a layer of 16 counts 16"),
          note(40, 480, "blockDim = (4, 4, 2): 4 × 4 × 2 = 32 threads, exactly 1 warp.", room=580)]
save(OUT + "ch03-block-3d.svg", 660, 500,
     "A block with blockDim (4, 4, 2) drawn as 2 layers of 4 by 4 threads side by side. Layer z = 0 is "
     "numbered 0 to 15 row by row, and layer z = 1, which comes after it, 16 to 31. Thread (2, 1, 1) in "
     "layer z = 1 is highlighted: 2 + 4 × 1 + 16 × 1 = 22. 4 × 4 × 2 = 32 threads, exactly 1 warp.",
     "A 3-D block: 2 layers of 4 × 4 threads", "Threads are numbered x first, then y, then z.", parts)


# 3.3  A 3-D block of 4 × 4 × 4 threads drawn as a cube; the 2 warps are slabs of z.
def shade(hex_colour, f):
    """`hex_colour` mixed towards black (f < 1) or white (f > 1)."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    if f < 1:
        r, g, b = (int(v * f) for v in (r, g, b))
    else:
        r, g, b = (int(v + (255 - v) * (f - 1)) for v in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


cs, dzx, dzy = 42, 26, -22                       # cube edge, and the screen shift of 1 step in z
ox, oy = 70, 196                                 # front-top-left corner of the z = 0 layer
warp_colour = {0: "#7FB5F0", 1: "#F2B36B"}       # warp 0: z = 0, 1; warp 1: z = 2, 3
hot = (2, 0, 3)
parts = []
for z in (3, 2, 1, 0):                           # back to front
    for y in (3, 2, 1, 0):                       # bottom to top
        for x in range(4):                       # left to right
            X, Y = ox + x * cs + z * dzx, oy + y * cs + z * dzy
            base = "#7A4FD0" if (x, y, z) == hot else warp_colour[(x + 4 * y + 16 * z) // 32]
            front, top, side = base, shade(base, 1.35), shade(base, 0.78)
            parts += [
                f'  <path d="M{X},{Y} L{X + dzx},{Y + dzy} L{X + cs + dzx},{Y + dzy} L{X + cs},{Y} Z" fill="{top}" stroke="#FFFFFF" stroke-width="1.2"/>',
                f'  <path d="M{X + cs},{Y} L{X + cs + dzx},{Y + dzy} L{X + cs + dzx},{Y + cs + dzy} L{X + cs},{Y + cs} Z" fill="{side}" stroke="#FFFFFF" stroke-width="1.2"/>',
                f'  <rect x="{X}" y="{Y}" width="{cs}" height="{cs}" fill="{front}" stroke="#FFFFFF" stroke-width="1.2"/>',
            ]
            if z == 0:
                parts.append(text(X + cs / 2, Y + cs / 2 + 4, str(x + 4 * y), "pill", "middle", INK))
# the highlighted thread's top face carries its number
X, Y = ox + 2 * cs + 3 * dzx, oy + 3 * dzy
parts.append(text(X + cs / 2 + dzx / 2, Y + dzy / 2 + 4, "50", "pill", "middle", "#FFFFFF"))
# axes
parts += [wire([(ox, oy + 4 * cs + 18), (ox + 4 * cs, oy + 4 * cs + 18)]),
          mono(ox + 2 * cs, oy + 4 * cs + 38, "x", "middle", INK),
          wire([(ox - 18, oy), (ox - 18, oy + 4 * cs)]),
          mono(ox - 32, oy + 2 * cs + 4, "y", "middle", INK),
          f'  <path d="M{ox - 4},{oy - 12} L{ox - 4 + 3 * dzx},{oy - 12 + 3 * dzy}" stroke="{WIRE}" stroke-width="1.8" marker-end="url(#head)"/>',
          mono(ox - 4 + 3 * dzx - 14, oy - 12 + 3 * dzy - 6, "z", "middle", INK)]
nx = 400
parts += [swatches(nx, 130, [(warp_colour[0], "warp 0: z = 0 and 1")]),
          swatches(nx, 156, [(warp_colour[1], "warp 1: z = 2 and 3")]),
          swatches(nx, 182, [("#7A4FD0", "the highlighted thread")]),
          note(nx, 214, "blockDim = (4, 4, 4): 64 threads"),
          note(nx, 252, "highlighted: (x, y, z) = (2, 0, 3)"),
          note(nx, 272, "number = x + 4y + 16z"),
          note(nx, 292, "= 2 + 0 + 48 = 50"),
          note(nx, 312, "50 ÷ 32 = 1 remainder 18:"),
          note(nx, 332, "warp 1, lane 18"),
          note(nx, 364, "Each z-layer holds 16 threads,"),
          note(nx, 384, "so 1 warp of 32 is 2 layers.")]
save(OUT + "ch03-cube.svg", 660, 420,
     "A block of 4 by 4 by 4 threads drawn as a cube of 64 small cubes, with x to the right, y downward and "
     "z going back. The front layer z = 0 is numbered 0 to 15. Layers z = 0 and 1 are coloured blue, warp 0, "
     "and layers z = 2 and 3 orange, warp 1. The thread at (2, 0, 3), on the top face at the back, is "
     "highlighted and numbered 50: 2 + 4 × 0 + 16 × 3 = 50, which is warp 1, lane 18.",
     "A 3-D block as a cube", "4 × 4 × 4 threads, numbered x first, then y, then z.", parts)

print("wrote chapter 3 figures")
