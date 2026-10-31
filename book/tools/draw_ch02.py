"""The figures of chapter 2, drawn with figkit and figparts.

    python3 tools/draw_ch02.py

Every figure shrinks a real case to small numbers (8 lanes, 8 banks, a load of 8
cycles) so that each value can be checked by hand. Lanes are drawn directly above
the data they read, so every arrow runs straight down.
"""
import math

from figparts import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, arrow, bar, card, fan, legend, line,
                      mascot, mono, note, save, sheet, tag, text, tile, width_of, wire, zone)

OUT = "src/figures/"


def pill(cx, y, body, kind="client"):
    """An instruction: a pill in the code face, 26 tall, centred on `cx`."""
    wide = width_of(body, 12, mono=True) + 26
    accent, tint = KINDS[kind]
    return (f'  <rect x="{cx - wide / 2:.1f}" y="{y}" width="{wide:.1f}" height="26" rx="13" fill="{tint}"'
            f' stroke="{accent}" stroke-width="1.4"/>\n' + mono(cx, y + 17.5, body, "middle", accent))


def dot(cx, cy, r, kind="good"):
    accent, tint = KINDS[kind]
    return f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{tint}" stroke="{accent}" stroke-width="1.3"/>'


# 2.1  From the whole GPU down to one thread.
parts = [zone(16, 104, 300, 228, "the GPU"), zone(344, 104, 300, 228, "one SM"),
         zone(344, 384, 300, 228, "one warp"), zone(16, 384, 300, 228, "one thread")]
parts.append(tile(276, 120, "chip", "client", 28))
for r in range(4):
    for c in range(8):
        parts.append(bar(40 + c * 32, 160 + r * 32, 26, 26, "client", "SM"))
parts += [note(40, 296, "32 of an H100's 132 SMs drawn;", room=260),
          note(40, 314, "a Jetson Orin NX has 8", room=260)]
for k in range(4):
    y = 140 + k * 36
    parts.append(text(364, y + 17, f"scheduler {k}", "sub"))
    for w in range(4):
        parts.append(bar(448 + w * 44, y + 2, 38, 22, "service", f"w{k + 4 * w}"))
parts += [note(364, 296, "up to 16 warps per scheduler,", room=270),
          note(364, 314, "64 per SM; 4 per scheduler drawn", room=270)]
for r in range(4):
    for c in range(8):
        parts.append(dot(384 + c * 30, 430 + r * 28, 9))
parts += [note(364, 560, "32 threads that carry out", room=270),
          note(364, 578, "each instruction together", room=270)]
parts.append(sheet(36, 410, 260, 176, "store", "registers",
                   [("", "i  = this thread's element"), ("", "r0 = a[i]"), ("", "r1 = b[i]"), ("", "r2 = r0 + r1")],
                   ["its own registers hold r0, r1, r2"]))
parts += [wire([(316, 218), (344, 218)]), wire([(494, 332), (494, 384)]), wire([(344, 498), (316, 498)])]
save(OUT + "ch02-zoom.svg", 660, 632,
     "Four panels joined by arrows. The GPU: a grid of identical streaming multiprocessors, SMs; an H100 "
     "has 132 and the Jetson Orin NX has 8. One SM: four schedulers, each looking after its own warps. One "
     "warp: 32 threads that carry out each instruction together. One thread: its own registers r0, r1 and "
     "r2, holding a[i], b[i] and their sum.",
     "From the whole GPU to one thread", "H100: 132 SMs, 4 schedulers per SM, 32 threads per warp.", parts)


# 2.2  One instruction reaches every lane of a warp: 1 control unit, 8 lanes with their own registers.
A, B = [1, 4, 0, 3, 2, 7, 5, 6], [2, 2, 5, 1, 8, 0, 3, 3]
x0, w, gap = 36, 64, 8
centres = [x0 + k * (w + gap) + w / 2 for k in range(8)]
parts = [card(130, 96, 380, 58, "client", "control unit: fetch and decode once",
              "instruction c = a + b, sent to all 8 lanes", glyph="gear"),
         fan(320, 154, centres, 176, 204)]
for k in range(8):
    x = x0 + k * (w + gap)
    parts += [f'  <rect x="{x}" y="204" width="{w}" height="118" rx="10" fill="#FFFFFF" stroke="{EDGE}" filter="url(#lift)"/>',
              text(x + w / 2, 224, f"lane {k}", "pill", "middle", MUTED),
              mono(x + w / 2, 248, f"a = {A[k]}", "middle"), mono(x + w / 2, 268, f"b = {B[k]}", "middle"),
              bar(x + 6, 282, w - 12, 28, "good", f"c = {A[k] + B[k]}")]
parts += [note(36, 350, "Each lane reads a and b from its own registers: 1 instruction, 8 different sums.", room=580),
          note(36, 370, "Drawn with 8 lanes; a real warp has 32.", room=580)]
save(OUT + "ch02-lockstep.svg", 640, 396,
     "One instruction, c = a + b, fanned out to eight lanes. Lane 0 has a = 1 and b = 2 and gets c = 3; "
     "lane 1 has a = 4, b = 2, c = 6; lane 2 has a = 0, b = 5, c = 5; and so on to lane 7, a = 6, b = 3, "
     "c = 9. Same instruction, different data in each lane. Drawn with 8 lanes; a real warp has 32.",
     "One instruction, every lane at once", "Each lane applies the instruction to its own registers.", parts)


# 2.3  A branch on which the lanes disagree.
X = [5, 1, 7, 2, 9, 0, 3, 4]
gx, cw = 150, 60
parts = [text(gx - 14, 132, "x", "name", "end")]
for k in range(8):
    parts += [mono(gx + k * cw + 26, 106, f"lane {k}", "middle", MUTED),
              bar(gx + k * cw, 114, 52, 26, "store", str(X[k]))]
steps = [("step 1", "if x > 2:  y = x * 2", lambda v: v > 2, lambda v: str(v * 2)),
         ("step 2", "else:  y = 0", lambda v: v <= 2, lambda v: "0"),
         ("step 3", "after the if", lambda v: True, lambda v: "runs")]
for s_, (name, code, on, out) in enumerate(steps):
    y = 170 + s_ * 74
    parts += [text(24, y + 4, name, "name"), mono(90, y + 4, code)]
    for k in range(8):
        if on(X[k]):
            parts.append(bar(gx + k * cw, y + 14, 52, 28, "good", out(X[k])))
        else:
            parts.append(bar(gx + k * cw, y + 14, 52, 28, "idle", "wait", dashed=True))
parts.append(card(40, 400, 470, 58, "danger", "Two sides of the branch, two steps",
                  "while one side runs, the lanes of the other side wait"))
parts.append(mascot(526, 392, 80, "shocked"))
save(OUT + "ch02-divergence.svg", 640, 480,
     "Eight lanes with x values 5, 1, 7, 2, 9, 0, 3 and 4. Step 1 runs if x > 2: y = x * 2. Lanes 0, 2, 4, "
     "6 and 7 compute 10, 14, 18, 6 and 8 while lanes 1, 3 and 5 wait. Step 2 runs else: y = 0, and lanes "
     "1, 3 and 5 compute 0 while the others wait. Step 3, after the if, all lanes run together.",
     "When the lanes disagree at a branch", "The warp runs each side in turn, with the other lanes switched off.", parts)


# 2.4  A block is cut into warps of 32.
def warp_row(y, number, used):
    first = number * 32
    out = [text(32, y + 14, f"warp {number}", "name"),
           text(100, y + 14, f"threads {first} to {first + 31}", "sub")]
    for t in range(32):
        kind = "good" if t < used else "idle"
        out.append(f'  <rect x="{250 + t * 11.5:.1f}" y="{y}" width="9.5" height="20" rx="3" fill="{KINDS[kind][1]}"'
                   f' stroke="{KINDS[kind][0]}" stroke-width="1"/>')
    return "\n".join(out)


parts = [zone(16, 104, 608, 134, "a block of 96 threads")]
parts += [warp_row(132 + k * 32, k, 32) for k in range(3)]
parts += [zone(16, 270, 608, 166, "a block of 100 threads")]
parts += [warp_row(298 + k * 32, k, 32) for k in range(3)] + [warp_row(298 + 3 * 32, 3, 4)]
parts += [tag(250, 446, "warp 3: 4 lanes used, 28 idle, a whole warp's place", "danger"),
          legend(32, 494, [("good", "a thread of the block"), ("idle", "a lane with no thread")])]
save(OUT + "ch02-block-warps.svg", 640, 516,
     "A block of 96 threads becomes three full warps: threads 0 to 31, 32 to 63 and 64 to 95. A block of "
     "100 threads becomes four warps: three full ones, and warp 3 with 4 lanes used and 28 idle, which still "
     "takes a whole warp's place on the SM.",
     "A block is cut into warps of 32 threads", "Threads 0 to 31 form warp 0, threads 32 to 63 warp 1, and so on.", parts)


# 2.5  The memory hierarchy of an H100.
rungs = [("client", "chip", "registers", "256 KB per SM, private to a thread", 1, "about 1 cycle"),
         ("service", "stack", "shared memory and L1", "up to 228 KB per SM", 30, "about 30 cycles"),
         ("store", "stack", "L2 cache", "50 MB, shared by all SMs", 200, "about 200 cycles"),
         ("outside", "stack", "global memory", "80 GB, beside the GPU chip", 500, "about 500 cycles"),
         ("idle", "laptop", "host memory", "the CPU's memory, over a link", 2000, "thousands of cycles")]
parts = [zone(24, 108, 412, 156, "on the SM")]
y = 126
for k, (kind, glyph, name, sub, cycles, label) in enumerate(rungs):
    if k == 2:
        y += 22
    parts.append(card(40, y, 380, 56, kind, name, sub, glyph=glyph))
    length = max(6, 4 * math.sqrt(cycles))
    parts += [f'  <rect x="448" y="{y + 16}" width="{length:.1f}" height="14" rx="5" fill="{KINDS[kind][0]}"/>',
              text(448, y + 48, label, "sub")]
    y += 70
parts.append(note(448, 112, "time to fetch", fill=MUTED))
save(OUT + "ch02-memory-ladder.svg", 660, y + 10,
     "Five cards from fast to slow. Registers: 256 KB per SM, private to a thread, about 1 cycle. Shared "
     "memory and L1: up to 228 KB per SM, about 30 cycles. These two are on the SM. L2 cache: 50 MB shared "
     "by all SMs, about 200 cycles. Global memory: 80 GB beside the GPU chip, about 500 cycles. Host memory: "
     "the CPU's memory, over a link, thousands of cycles. Bar lengths grow with the square root of the time.",
     "Where a thread's data can live", "The memory of an H100, from the fastest and smallest to the slowest.", parts)


# 2.6  Coalescing: which sectors a warp's load touches.
parts = [zone(16, 104, 608, 196, "lane k reads float k")]
for k in range(8):
    cx = 44 + k * 52 + 23
    parts += [tag(cx - (width_of(f"L{k}", 11, bold=True) + 14) / 2, 132, f"L{k}", "client"), arrow(cx, 152, 186),
              bar(44 + k * 52, 186, 46, 30, "good", str(k))]
parts += [f'  <rect x="38" y="180" width="420" height="42" rx="9" fill="none" stroke="{KINDS["good"][0]}" stroke-width="1.6"/>',
          text(44, 240, "sector 0: floats 0 to 7", "sub"),
          note(474, 196, "sectors 1 to 7"), note(474, 214, "are not fetched"),
          tag(44, 262, "1 sector fetched: 32 bytes, all used", "good")]
parts += [zone(16, 332, 608, 220, "lane k reads float 8k")]
for k in range(8):
    x = 40 + k * 72
    cx = x + 32
    parts += [tag(cx - (width_of(f"L{k}", 11, bold=True) + 14) / 2, 360, f"L{k}", "client"), arrow(cx, 380, 414)]
    parts.append(f'  <rect x="{x}" y="414" width="64" height="30" rx="7" fill="{KINDS["danger"][1]}" stroke="{KINDS["danger"][0]}" stroke-width="1.3"/>')
    for f in range(8):
        hot = f == 0
        parts.append(f'  <rect x="{x + 4 + f * 7.2:.1f}" y="420" width="5.6" height="18" rx="1.5"'
                     f' fill="{KINDS["good"][0] if hot else "#FFFFFF"}" stroke="{KINDS["danger"][0]}" stroke-width="0.6"/>')
    parts.append(text(cx, 462, f"float {8 * k}", "sub", "middle"))
parts += [tag(40, 490, "8 sectors fetched: 256 bytes for 32 useful", "danger"),
          legend(40, 530, [("good", "a float a lane asked for"), ("danger", "a sector fetched")])]
save(OUT + "ch02-coalesce.svg", 640, 572,
     "Two ways for 8 lanes to read 8 floats. Top: lane k reads float k. The lanes sit above floats 0 to 7, "
     "which all lie in sector 0, so one sector of 32 bytes is fetched and every byte is used. Bottom: lane k "
     "reads float 8k. Each lane sits above a different sector and uses one float of its eight, so 8 sectors, "
     "256 bytes, are fetched for 32 useful bytes.",
     "Two ways for a warp to read 8 floats", "Memory delivers 32-byte sectors; a load costs one fetch per sector it touches.",
     parts)


# 2.7  Shared-memory banks: three access patterns.
cases = [("word k", "lane k reads word k", lambda r, c: r == 0, "good", "1 step"),
         ("word 8k", "lane k reads word 8k", lambda r, c: c == 0, "danger", "8 steps: one bank"),
         ("word 5", "every lane reads word 5", lambda r, c: r == 0 and c == 5, "store", "1 step: broadcast")]
parts = []
for p_, (title, sub, hot, kind, result) in enumerate(cases):
    zx = 16 + p_ * 208
    parts.append(zone(zx, 104, 196, 324, title))
    parts.append(note(zx + 18, 136, sub, room=170))
    gx, gy = zx + 18, 176
    for c in range(8):
        parts.append(mono(gx + c * 20 + 9, 166, f"b{c}", "middle", MUTED))
    for r in range(8):
        for c in range(8):
            on = hot(r, c)
            parts.append(f'  <rect x="{gx + c * 20}" y="{gy + r * 22}" width="18" height="20" rx="3"'
                         f' fill="{KINDS[kind][0] if on else "#FFFFFF"}" stroke="{EDGE if not on else KINDS[kind][0]}" stroke-width="1"/>')
            if on:
                parts.append(text(gx + c * 20 + 9, gy + r * 22 + 14, str(r * 8 + c), "pill", "middle", "#FFFFFF"))
    parts.append(tag(zx + 18, 376, result, kind))
parts.append(note(16, 456, "Words are laid out in rows of 8, so each column is one bank: bank = word mod 8.", room=608))
save(OUT + "ch02-banks.svg", 640, 476,
     "Three panels, each a grid of words 0 to 63 in rows of 8, with columns labelled banks b0 to b7. Left, "
     "lane k reads word k: words 0 to 7, one in each bank, 1 step. Middle, lane k reads word 8k: words 0, 8, "
     "16 up to 56, all in column b0, 8 steps. Right, every lane reads word 5: one word, read once and given "
     "to every lane, 1 step.",
     "Three ways for 8 lanes to read shared memory", "8 banks, each serving one word per step.", parts)


# 2.8  The register file as a fixed budget.
rows = [(32, 8, "100%", 32), (33, 6, "75%", 40), (64, 4, "50%", 64), (128, 2, "25%", 128)]
parts = []
for k, (regs, blocks, occ, alloc) in enumerate(rows):
    y = 112 + k * 92
    parts += [tile(32, y - 6, "chip", "client", 26),
              text(66, y + 12, f"{regs} registers per thread" + (f", allocated as {alloc}" if alloc != regs else ""), "name"),
              text(66, y + 30, f"a block of 256 threads needs {256 * alloc:,}", "sub")]
    seg = 560 / (65536 / (256 * alloc))
    for b in range(blocks):
        parts.append(bar(40 + b * seg, y + 42, seg - 3, 28, "service", "block" if seg > 50 else ""))
    used = blocks * seg
    if used < 559:
        parts.append(bar(40 + used, y + 42, 560 - used, 28, "idle", "", dashed=True))
    parts.append(tag(600, y - 2, f"{blocks} blocks, {blocks * 256:,} threads, {occ}", "service", "end"))
parts.append(note(40, 482, "Each bar is the SM's 65,536 registers; dashed: registers left unused.", room=560))
save(OUT + "ch02-occupancy-budget.svg", 640, 506,
     "One SM has 65,536 registers, and each block has 256 threads. With 32 registers per thread a block "
     "needs 8,192, so 8 blocks fit: 2,048 threads, 100 percent occupancy. With 33 registers per thread, "
     "allocated as 40, a block needs 10,240, so 6 blocks fit: 1,536 threads, 75 percent. With 64 registers per thread a "
     "block needs 16,384, so 4 blocks fit: 1,024 threads, 50 percent. With 128 registers per thread a block "
     "needs 32,768, so 2 blocks fit: 512 threads, 25 percent.",
     "The register file is a fixed budget", "Blocks of 256 threads on one SM with 65,536 registers.", parts)


# 2.9  One scheduler, four warps, a load of 8 cycles.
order = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 0, 10: 3, 11: 1, 12: 2}
kinds = ["danger", "client", "service", "store"]
gx, cw = 120, 39
parts = [mono(gx + c * cw + cw / 2, 106, str(c), "middle", MUTED) for c in range(13)]
for wid in range(4):
    y = 116 + wid * 40
    parts += [tile(16, y + 2, "chip", kinds[wid], 24), text(46, y + 19, f"warp {wid}", "name")]
    parts.append(f'  <rect x="{gx}" y="{y}" width="{13 * cw}" height="30" rx="7" fill="#F6F8FA" stroke="{EDGE}"/>')
parts += [bar(gx + 2, 118, cw - 4, 26, "danger", "load", solid=True),
          bar(gx + cw + 2, 118, 8 * cw - 4, 26, "idle", "waiting for its data: cycles 1 to 8", dashed=True),
          bar(gx + 9 * cw + 2, 118, cw - 4, 26, "danger", "add", solid=True)]
for c, wid in order.items():
    if wid:
        parts.append(bar(gx + c * cw + 2, 116 + wid * 40 + 2, cw - 4, 26, kinds[wid], "i", solid=True))
y = 116 + 4 * 40 + 12
parts.append(text(46, y + 19, "issued", "name"))
parts.append(bar(gx + 2, y, cw - 4, 30, "danger", "w0", solid=True))
for c, wid in order.items():
    parts.append(bar(gx + c * cw + 2, y, cw - 4, 30, kinds[wid], f"w{wid}", solid=True))
parts.append(note(gx, y + 46, "i: an instruction that does not need the data warp 0 is waiting for", room=500))
parts += [card(40, y + 64, 470, 58, "service", "An instruction is issued in every cycle",
               "while warp 0 waits, warps 1 to 3 fill the gaps"),
          mascot(526, y + 56, 80, "happy")]
save(OUT + "ch02-time-slicing.svg", 640, y + 148,
     "A timeline of cycles 0 to 12 for warps 0 to 3. In cycle 0 warp 0 issues a load, then waits for its "
     "data in cycles 1 to 8. Meanwhile the scheduler issues an instruction from warp 1 in cycle 1, warp 2 in "
     "cycle 2, warp 3 in cycle 3, warp 1 again in cycle 4, and so on. In cycle 9 warp 0's data has arrived "
     "and it issues an add. The bottom row shows one instruction issued in every cycle.",
     "One scheduler, four warps", "A load takes 8 cycles here, instead of about 500, so the whole wait fits.", parts)


# 2.1.1  A CPU spends its chip on a few fast cores; a GPU on many simple units.
parts = [zone(16, 104, 300, 300, "a CPU"), zone(344, 104, 300, 300, "a GPU")]
for k in range(4):
    x, y = 36 + (k % 2) * 136, 132 + (k // 2) * 76
    parts.append(card(x, y, 124, 62, "client", f"core {k}", glyph="chip"))
parts += [card(36, 290, 260, 56, "store", "large cache", "keeps data near the cores", glyph="stack"),
          note(36, 380, "a few large cores, each fast alone", room=260)]
for r in range(8):
    for c in range(16):
        parts.append(f'  <rect x="{364 + c * 16.5:.1f}" y="{134 + r * 17}" width="13" height="13" rx="3"'
                     f' fill="{KINDS["good"][1]}" stroke="{KINDS["good"][0]}" stroke-width="1"/>')
parts += [card(364, 290, 260, 56, "store", "small caches", "most area goes to arithmetic", glyph="stack"),
          note(364, 380, "thousands of simple arithmetic units", room=260)]
save(OUT + "ch02-cpu-gpu.svg", 660, 424,
     "Two panels. A CPU: 4 large cores, each fast for 1 thread, above a large cache that keeps data near "
     "the cores. A GPU: a field of 128 small squares standing for thousands of simple arithmetic units, "
     "above small caches; most of its area goes to arithmetic.",
     "Where a CPU and a GPU spend their chip area", "A CPU avoids waiting for memory; a GPU hides the wait behind other threads.",
     parts)


# 2.1.2  A launch: a grid of 4 blocks of 256 threads, and thread 300 computing c[300].
parts = [card(40, 96, 250, 60, "client", "program on the CPU", "launches the kernel vec_add", glyph="terminal"),
         card(400, 96, 220, 60, "service", "1,024 threads", "4 blocks of 256", glyph="chip"),
         wire([(290, 126), (400, 126)], label="launch"),
         zone(16, 196, 628, 108, "the grid")]
for b in range(4):
    x = 32 + b * 152
    kind = "outside" if b == 1 else "service"
    parts.append(card(x, 224, 140, 60, kind, f"block {b}", f"threads {256 * b}–{256 * b + 255}"))
cells_x = [96 + k * 52 for k in range(9)]
parts += [f'  <rect x="98" y="326" width="476" height="44" rx="10" fill="none" stroke="{KINDS["outside"][0]}"'
          f' stroke-width="1.5" stroke-dasharray="6 5"/>',
          arrow(254, 284, 326, "flow"),
          text(32, 352, "block 1", "name")]
for k, i in enumerate(range(296, 305)):
    parts.append(bar(cells_x[k] - 23 + 30, 334, 46, 28, "outside" if i == 300 else "idle", str(i), solid=(i == 300)))
parts += [note(32, 380, "thread index", fill=MUTED),
          arrow(cells_x[4] + 30, 362, 410),
          text(32, 432, "array c", "name")]
for k, i in enumerate(range(296, 305)):
    parts.append(bar(cells_x[k] - 24 + 30, 412, 48, 28, "outside" if i == 300 else "idle", f"c[{i}]" if i == 300 else ""))
parts += [card(40, 468, 580, 58, "outside", "thread 300 computes c[300] = a[300] + b[300]",
               "every thread uses its own index to choose its element")]
save(OUT + "ch02-launch.svg", 660, 548,
     "A program on the CPU launches the kernel vec_add as 1,024 threads in 4 blocks of 256. The grid holds "
     "block 0 with threads 0 to 255, block 1 with threads 256 to 511, block 2 with 512 to 767 and block 3 "
     "with 768 to 1,023. Inside block 1, threads 296 to 304 are drawn, and thread 300 is highlighted; an "
     "arrow runs from it to element c[300] of the array c. Thread 300 computes c[300] = a[300] + b[300].",
     "A launch of vector addition", "1,024 threads in 4 blocks of 256; each thread computes 1 element.", parts)


# 2.2  Inside one SM of an H100.
parts = [zone(16, 104, 628, 516, "one SM")]
for q in range(4):
    x, y = 36 + (q % 2) * 304, 134 + (q // 2) * 166
    parts.append(f'  <rect x="{x}" y="{y}" width="284" height="150" rx="12" fill="#FFFFFF" stroke="{EDGE}" filter="url(#lift)"/>')
    parts += [tile(x + 14, y + 12, "gear", "client", 26), text(x + 48, y + 30, f"processing block {q}", "name")]
    for r in range(4):
        for c in range(8):
            parts.append(f'  <rect x="{x + 14 + c * 20:.1f}" y="{y + 50 + r * 18}" width="15" height="13" rx="3"'
                         f' fill="{KINDS["good"][1]}" stroke="{KINDS["good"][0]}" stroke-width="1"/>')
    parts += [note(x + 182, y + 66, "32 arithmetic"), note(x + 182, y + 84, "units"),
              bar(x + 14, y + 124, 256, 18, "client", "16,384 registers")]
parts += [card(36, 468, 588, 44, "store", "shared memory and L1 cache: 256 KB, used by all 4 processing blocks", glyph="stack"),
          card(36, 532, 588, 58, "service", "holds up to 2,048 threads: 64 warps, 16 per processing block",
               "2,048 threads / 128 arithmetic units = 16 threads for each unit", glyph="chip")]
save(OUT + "ch02-sm.svg", 660, 640,
     "One SM of an H100, divided into 4 processing blocks. Each has its own control logic, 32 arithmetic units drawn as "
     "small squares, and 16,384 registers. Below the 4 processing blocks, one pool of 256 KB serves as shared memory "
     "and L1 cache for the whole SM.",
     "Inside one SM of an H100", "4 processing blocks of 32 arithmetic units, each with its own control logic.", parts)


# 2.1.2  Where a thread's index comes from: block number × block size + position in the block.
pos = [0, 1, 2, None, 43, 44, 45, None, 255]
parts = [card(40, 96, 270, 56, "service", "block 0", "positions 0–255, indexes 0–255"),
         card(350, 96, 270, 56, "outside", "block 1", "positions 0–255, indexes 256–511"),
         arrow(485, 152, 222),
         f'  <rect x="30" y="222" width="600" height="54" rx="10" fill="none" stroke="{KINDS["outside"][0]}"'
         f' stroke-width="1.5" stroke-dasharray="6 5"/>',
         text(40, 208, "position in block 1", "name")]
xs = [40 + k * 64 for k in range(9)]
for k, v in enumerate(pos):
    if v is None:
        parts.append(text(xs[k] + 28, 254, "…", "name", "middle"))
        parts.append(text(xs[k] + 28, 330, "…", "name", "middle"))
        continue
    hot = v == 44
    parts.append(bar(xs[k], 234, 56, 30, "outside" if hot else "idle", str(v), solid=hot))
    parts.append(arrow(xs[k] + 28, 264, 310))
    parts.append(bar(xs[k], 310, 56, 30, "outside" if hot else "idle", str(256 + v), solid=hot))
parts += [text(40, 366, "index of the thread", "name"),
          note(200, 366, "each index is 256 more than the position", room=420),
          card(40, 392, 580, 60, "outside", "index = block number × block size + position",
               "thread 300: 1 × 256 + 44 = 300")]
save(OUT + "ch02-index.svg", 660, 476,
     "Block 0 holds threads with indexes 0 to 255, and block 1 holds indexes 256 to 511. Inside block 1, "
     "each thread has a position from 0 to 255, drawn in a row: 0, 1, 2, then 43, 44, 45, then 255. An arrow "
     "runs from each position down to the thread's index, 256 more: 256, 257, 258, 299, 300, 301 and 511. "
     "Position 44 and index 300 are highlighted. A card reads: index = block number × block size + position; "
     "thread 300 is 1 × 256 + 44.",
     "Where a thread's index comes from", "Block 1 starts after the 256 threads of block 0.", parts)

print("wrote chapter 2 figures")


# 2.4  Blocks reach the SMs in waves: 20 blocks on 4 SMs and on 8 SMs, 2 places per SM.
def waves(y0, sms, label):
    places = sms * 2
    lane, sm_gap, wave_w, x0 = 19, 8, 150, 120
    out = [zone(16, y0, 628, 46 + sms * (2 * lane + 2 + sm_gap) + 50, label)]
    top = y0 + 40
    nwaves = -(-20 // places)
    for w in range(nwaves + 1):
        out.append(line(x0 + w * wave_w, top - 6, x0 + w * wave_w, top + sms * (2 * lane + 2 + sm_gap) - 4, EDGE, 1, "3 4"))
    for w in range(nwaves):
        out.append(mono(x0 + w * wave_w + wave_w / 2, top - 10, f"wave {w + 1}", "middle", MUTED))
    for m in range(sms):
        y = top + m * (2 * lane + 2 + sm_gap)
        out.append(text(x0 - 12, y + lane + 6, f"SM {m}", "sub", "end"))
        for slot in range(2):
            for w in range(nwaves):
                b = w * places + m * 2 + slot
                yy = y + slot * (lane + 2)
                if b < 20:
                    out.append(bar(x0 + w * wave_w + 3, yy, wave_w - 6, lane, "service", f"block {b}"))
                else:
                    out.append(bar(x0 + w * wave_w + 3, yy, wave_w - 6, lane, "idle", "idle", dashed=True))
    bottom = top + sms * (2 * lane + 2 + sm_gap) + 8
    last = 20 - (nwaves - 1) * places
    out.append(note(x0, bottom + 14, f"20 blocks / {places} places = {20 / places:g} waves; the last wave fills "
                    f"{last} of {places} places", room=500))
    out.append(note(x0, bottom + 32, f"busy: 20 of {nwaves} × {places} = {nwaves * places} block-times, "
                    f"{100 * 20 / (nwaves * places):.1f}%", room=500))
    return out, y0 + 46 + sms * (2 * lane + 2 + sm_gap) + 50
parts, y = waves(100, 4, "4 SMs, 2 blocks each: 8 places")
more, y = waves(y + 30, 8, "8 SMs, 2 blocks each: 16 places")
parts += more
save(OUT + "ch02-waves.svg", 660, y + 14,
     "The same grid of 20 equal blocks on 2 GPUs whose SMs each hold 2 blocks at a time. Top, 4 SMs, 8 places: "
     "blocks 0 to 7 run in wave 1, 8 to 15 in wave 2, and 16 to 19 in wave 3 on SMs 0 and 1 while SMs 2 and 3 "
     "are idle; 20 blocks in 3 waves of 8 places, 83.3 percent busy. Bottom, 8 SMs, 16 places: blocks 0 to 15 run "
     "in wave 1 and 16 to 19 in wave 2 on SMs 0 and 1 while 6 SMs are idle; 20 blocks in 2 waves of 16 places, "
     "62.5 percent busy.",
     "20 blocks on 4 SMs and on 8 SMs", "Each SM holds 2 blocks at a time; every block takes the same time.", parts)


# 2.1.4  Memory bandwidth = bytes per transfer x transfers per second, for 4 GPUs.
gpus = [("Jetson Orin NX", "LPDDR5", 128, 6.4, 102),
        ("Jetson AGX Thor", "LPDDR5X", 256, 8.53, 273),
        ("GeForce RTX 4090", "GDDR6X", 384, 21.0, 1008),
        ("H100 SXM", "HBM3", 5120, 5.23, 3350)]
x0, full = 40, 580
parts = []
for k, (name, kind, bits, rate, bw) in enumerate(gpus):
    y = 104 + k * 92
    wbar = max(6, full * bits / 5120)
    parts += [text(x0, y + 14, f"{name}: {kind}", "name"),
              text(x0 + full, y + 14, f"{bw:,} GB/s", "name", "end"),
              bar(x0, y + 24, wbar, 22, "store", f"{bits:,} bits" if wbar > 80 else ""),
              note(x0, y + 66, f"{bits // 8:,} bytes per transfer × {rate:g} billion transfers/s"
                   f" = {bw:,} GB/s", room=full)]
    if wbar <= 80:
        parts.append(note(x0 + wbar + 8, y + 40, f"{bits:,} bits"))
parts.append(note(x0, 104 + 4 * 92 + 4, "Bar length: the width of the memory bus, in bits carried by 1 transfer.",
                  room=full))
save(OUT + "ch02-bus.svg", 660, 104 + 4 * 92 + 28,
     "4 horizontal bars, one per GPU, whose length is the width of its memory bus. Jetson Orin NX, LPDDR5: 128 bits, "
     "16 bytes per transfer at 6.4 billion transfers per second, 102 GB/s. Jetson AGX Thor, LPDDR5X: 256 bits, 32 "
     "bytes at 8.53 billion, 273 GB/s. GeForce RTX 4090, GDDR6X: 384 bits, 48 bytes at 21 billion, 1,008 GB/s. H100 "
     "SXM, HBM3: 5,120 bits, 640 bytes at 5.23 billion, 3,350 GB/s; its bar is 40 times as long as the Orin's.",
     "Memory bandwidth = bus width × transfer rate", "The H100 moves 640 bytes per transfer; the Orin NX moves 16.", parts)


# 2.7.4  A row-major matrix of 4 x 8 floats: a warp reading a row, and reading a column.
CW, CH = 19, 22
def matrix_panel(y0, label, picked, lanes):
    out = [zone(16, y0, 648, 238, label)]
    gx, gy = 40, y0 + 40
    out.append(text(gx, gy - 8, "the matrix, 4 rows × 8 columns", "sub"))
    for r in range(4):
        for c in range(8):
            n = r * 8 + c
            hot = n in picked
            out.append(f'  <rect x="{gx + c * 26}" y="{gy + r * 24}" width="24" height="{CH}" rx="4"'
                       f' fill="{KINDS["good"][1] if hot else "#FFFFFF"}" stroke="{KINDS["good"][0] if hot else EDGE}"/>')
            out.append(mono(gx + c * 26 + 12, gy + r * 24 + 15, str(n), "middle", KINDS["good"][0] if hot else MUTED))
    nx = gx + 8 * 26 + 18
    out += [note(nx, gy + 14, f"{len(picked)} lanes read the green elements,"),
            note(nx, gy + 32, lanes),
            note(nx, gy + 58, "each number is the element's address"),
            note(nx, gy + 76, "in floats: row × 8 + column")]
    sy = gy + 4 * 24 + 34
    out.append(text(40, sy - 10, "the same 32 floats in memory, as 4 sectors of 8 floats", "sub"))
    touched = sorted({n // 8 for n in picked})
    for sec in range(4):
        sx = 40 + sec * (8 * CW + 8)
        hot = sec in touched
        out.append(f'  <rect x="{sx - 3}" y="{sy - 3}" width="{8 * CW + 4}" height="{CH + 6}" rx="6" fill="none"'
                   f' stroke="{KINDS["danger"][0] if hot else EDGE}" stroke-width="{1.8 if hot else 1}"/>')
        for c in range(8):
            n = sec * 8 + c
            green = n in picked
            out.append(f'  <rect x="{sx + c * CW}" y="{sy}" width="{CW - 2}" height="{CH}" rx="3"'
                       f' fill="{KINDS["good"][1] if green else "#FFFFFF"}" stroke="{KINDS["good"][0] if green else EDGE}"/>')
            out.append(mono(sx + c * CW + (CW - 2) / 2, sy + 15, str(n), "middle", KINDS["good"][0] if green else MUTED))
        out.append(text(sx + 4 * CW, sy + CH + 20, f"sector {sec}", "pill", "middle", KINDS["danger"][0] if hot else MUTED))
    kind = "good" if len(touched) == 1 else "danger"
    plural = "s" if len(touched) > 1 else ""
    out.append(tag(40, sy + CH + 32, f"{len(touched)} sector{plural} fetched: {32 * len(touched)} bytes "
                   f"for {4 * len(picked)} useful bytes", kind))
    return out
parts = matrix_panel(100, "lanes 0 to 7 read row 1", list(range(8, 16)), "row 1, columns 0 to 7")
parts += matrix_panel(368, "lanes 0 to 3 read column 5", [5, 13, 21, 29], "column 5, rows 0 to 3")
save(OUT + "ch02-matrix-access.svg", 680, 630,
     "2 panels, each showing a matrix of 4 rows and 8 columns of floats, numbered by address in row-major order, and "
     "the same 32 floats laid out in memory as 4 sectors of 8. Top: 8 lanes read row 1, elements 8 to 15, which "
     "fill sector 1 exactly: 1 sector fetched, 32 bytes, all used. Bottom: 4 lanes read column 5, elements 5, 13, 21 "
     "and 29, which lie 8 floats apart in 4 different sectors: 4 sectors fetched, 128 bytes, for 16 useful bytes.",
     "A row and a column of the same matrix", "Neighbours in a row are neighbours in memory; neighbours in a column are 8 floats apart.",
     parts)
