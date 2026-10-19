"""The figures of chapter 5, drawn with figkit and figparts.

    python3 tools/draw_ch05.py

The histogram times in figure 5.7 are the ones measured on a Jetson Orin NX with
code/ch05_sync/sync.cu (the run quoted in section 5.5).
"""
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, bar, card, legend, line, mascot, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"


# 5.1  A lost update: 2 threads read the same value, and 1 addition disappears.
steps = [("A", "read 5", 0), ("B", "read 5", 1), ("A", "write 6", 2), ("B", "write 6", 3)]
gx, w = 170, 104
parts = [text(gx - 16, 148, "thread A", "name", "end"), text(gx - 16, 214, "thread B", "name", "end"),
         text(gx - 16, 286, "counter", "name", "end")]
for k in range(4):
    parts.append(mono(gx + k * (w + 8) + w / 2, 116, f"step {k + 1}", "middle", MUTED))
for who, what, k in steps:
    y = 126 if who == "A" else 192
    parts.append(bar(gx + k * (w + 8), y, w, 36, "client" if who == "A" else "outside", what))
for k, value in enumerate(["5", "5", "6", "6"]):
    parts.append(bar(gx + k * (w + 8), 264, w, 36, "danger" if k == 3 else "store", value))
parts += [note(40, 346, "2 threads each added 1, and the counter grew by 1: thread B read 5 before", room=600),
          note(40, 364, "thread A wrote 6, so B's write replaced A's. 1 addition is lost.", room=600)]
save(OUT + "ch05-race.svg", 660, 386,
     "A timeline of 4 steps. Step 1: thread A reads the counter, 5. Step 2: thread B reads it, also 5. "
     "Step 3: A writes 6. Step 4: B writes 6. Both threads added 1, but the counter grew from 5 to 6 only.",
     "A lost update", "Read, add and write are 3 steps, and another thread can come between them.", parts)


# 5.2  An atomic operation: the memory system performs read, add and write as 1 step per address.
parts = [card(24, 130, 190, 50, "client", "thread A", "atomicAdd(c, 1)", glyph="terminal"),
         card(24, 196, 190, 50, "outside", "thread B", "atomicAdd(c, 1)", glyph="terminal"),
         card(24, 262, 190, 50, "client", "thread C", "atomicAdd(c, 1)", glyph="terminal"),
         card(256, 186, 236, 70, "service", "atomic unit in L2", "1 operation at a time", glyph="lock"),
         card(520, 196, 116, 50, "store", "counter c", "5 → 8"),
         wire([(214, 155), (234, 155), (234, 210), (256, 210)]),
         wire([(214, 221), (256, 221)]),
         wire([(214, 287), (234, 287), (234, 232), (256, 232)]),
         wire([(492, 221), (520, 221)]),
         note(36, 352, "Each request reads, adds and writes the counter before the next begins, so", room=600),
         note(36, 370, "none is lost. Requests to the same address wait for each other.", room=600)]
save(OUT + "ch05-atomic.svg", 660, 392,
     "Three threads each call atomicAdd on counter c. Their requests go to an atomic unit in the L2 cache, "
     "which performs 1 operation at a time on an address: read, add and write as 1 step. The counter goes "
     "from 5 to 8 and no addition is lost.",
     "An atomic add", "The memory system performs read, add and write as 1 indivisible step.", parts)


# 5.3  Divergence: alternate lanes run both paths one after another; whole warps run 1 each.
gx, unit = 210, 170
parts = [zone(16, 104, 628, 128, "lanes alternate: odd and even lanes in every warp"),
         text(gx - 14, 160, "warp 0", "name", "end"),
         bar(gx, 140, unit, 34, "client", "sin path, 16 lanes"),
         bar(gx + unit + 6, 140, unit, 34, "outside", "sqrt path, 16 lanes"),
         mono(gx, 200, "time: 2 units per warp", "start", MUTED),
         zone(16, 252, 628, 168, "warps agree: warp 0 takes 1 path, warp 1 the other"),
         text(gx - 14, 308, "warp 0", "name", "end"), text(gx - 14, 356, "warp 1", "name", "end"),
         bar(gx, 288, unit, 34, "client", "sin path, 32 lanes"),
         bar(gx, 336, unit, 34, "outside", "sqrt path, 32 lanes"),
         mono(gx, 398, "time: 1 unit per warp", "start", MUTED),
         note(16, 452, "Measured, 16,777,216 threads: Orin NX 54.8 ms against 28.4 ms; Thor 12.2 ms", room=628),
         note(16, 470, "against 6.2 ms. The same work, about twice the time when lanes disagree.", room=628)]
save(OUT + "ch05-divergence.svg", 660, 492,
     "Two cases. Lanes alternate: every warp has 16 lanes on the sin path and 16 on the sqrt path, and runs "
     "the 2 paths one after the other, 2 time units. Warps agree: warp 0 runs only the sin path and warp 1 "
     "only the sqrt path, 1 unit each. Measured: 54.8 ms against 28.4 ms on an Orin NX, 12.2 against 6.2 ms "
     "on a Thor.",
     "Divergence", "A warp runs every path that any of its lanes takes.", parts)


# 5.4  A barrier: warps reach __syncthreads() at different times; none continues until all arrive.
gx, gate = 150, 430
arrive = [190, 250, 120, 270, 210, 160]
parts = [text(gx, 118, "phase 1: write shared memory", "sub"), text(gate + 16, 118, "phase 2: read it", "sub")]
for k, length in enumerate(arrive):
    y = 132 + k * 40
    parts += [text(gx - 14, y + 20, f"warp {k}", "name", "end"),
              bar(gx, y, length, 28, "client", ""),
              line(gx + length, y + 14, gate - 4, y + 14, KINDS["idle"][0], 1.6, "4 4"),
              bar(gate + 10, y, 150, 28, "service", "")]
parts += [line(gate + 3, 124, gate + 3, 132 + 6 * 40, KINDS["danger"][0], 2.4),
          tag(gate + 3, 132 + 6 * 40 + 14, "__syncthreads()", "danger", "middle"),
          legend(gx, 132 + 6 * 40 + 54, [("client", "working"), ("idle", "waiting"), ("service", "after the barrier")])]
save(OUT + "ch05-barrier.svg", 660, 462,
     "Six warps of a block write shared memory and reach __syncthreads at different times. Each early warp "
     "waits, shown dashed, at a vertical line marking the barrier. When the last warp arrives, all 6 start "
     "phase 2 together and can read what the others wrote.",
     "A block barrier", "No thread of the block passes __syncthreads() until all of them reach it.", parts)


# 5.5  A tree sum in shared memory: each level halves the active threads; a barrier between levels.
vals = [3, 1, 4, 1, 5, 9, 2, 6]
cw, gap, gx = 46, 8, 130
levels = [vals, [8, 10, 6, 7], [14, 17], [31]]
names = ["s[0..7]", "stride 4", "stride 2", "stride 1"]
parts = []
for lv, row in enumerate(levels):
    y = 120 + lv * 88
    parts.append(text(gx - 14, y + 22, names[lv], "sub", "end"))
    for k, v in enumerate(row):
        kind = "store" if lv == 0 else "service" if lv < 3 else "good"
        parts.append(bar(gx + k * (cw + gap), y, cw, 34, kind, str(v)))
    if lv < 3:
        stride = 4 >> lv
        right = gx + 8 * (cw + gap) - gap
        parts.append(line(gx - 6, y + 60, right + 6, y + 60, KINDS["danger"][0], 1.8, "6 4"))
        parts.append(tag(right + 14, y + 51, "barrier", "danger"))
        for k in range(stride):
            x1 = gx + k * (cw + gap) + cw / 2
            x2 = gx + (k + stride) * (cw + gap) + cw / 2
            parts.append(line(x1, y + 34, x1, y + 88, WIRE, 1.2))
            parts.append(line(x2, y + 34, x1 + 8, y + 88, WIRE, 1.2))
parts.append(note(gx, 120 + 3 * 88 + 64, "Thread k adds s[k + stride] into s[k]. Without the barrier, it can read", room=500))
parts.append(note(gx, 120 + 3 * 88 + 82, "s[k + stride] before the thread that owns it has written its new value.", room=500))
save(OUT + "ch05-tree.svg", 660, 530,
     "Eight values 3, 1, 4, 1, 5, 9, 2, 6 in shared memory. Stride 4: thread k adds s[k + 4] into s[k], giving "
     "8, 10, 6, 7. Stride 2 gives 14, 17. Stride 1 gives 31. A barrier separates each level from the next.",
     "A tree sum in shared memory", "Each level halves the number of active threads; a barrier separates the levels.", parts)


# 5.6  Privatisation: each block counts into its own bins in shared memory, then adds them once.
parts = [zone(16, 104, 300, 290, "phase 1: count in shared memory"),
         card(36, 140, 260, 56, "client", "block 0", "256 shared bins", glyph="stack"),
         card(36, 216, 260, 56, "client", "block 1", "256 shared bins", glyph="stack"),
         card(36, 292, 260, 56, "client", "block 1,023", "256 shared bins", glyph="stack"),
         zone(344, 104, 300, 290, "phase 2: add the bins once"),
         card(374, 216, 240, 56, "store", "global bins", "256 counters", glyph="doc"),
         wire([(296, 168), (334, 168), (334, 234), (374, 234)]),
         wire([(296, 244), (374, 244)]),
         wire([(296, 320), (334, 320), (334, 254), (374, 254)]),
         note(16, 426, "Each block counts its 65,536 bytes with shared-memory atomics, then adds 256 totals.", room=628),
         note(16, 444, "Global atomics fall from 67,108,864 to 1,024 × 256 = 262,144.", room=628)]
save(OUT + "ch05-privatise.svg", 660, 466,
     "Phase 1: blocks 0, 1 and so on to 1,023 each count their 65,536 bytes into 256 bins in shared memory. "
     "Phase 2: each block adds its 256 counts to the 256 global bins. The global atomics drop from "
     "67,108,864 to 262,144.",
     "A privatised histogram", "Each block counts in its own shared memory, then adds its bins once.", parts)


# 5.7  Measured histogram times on a Jetson Orin NX.
rows = [("random bytes, global bins", 33.4, "danger"), ("random bytes, shared bins", 5.2, "good"),
        ("all 7, global bins", 134.7, "danger"), ("all 7, shared bins", 4.8, "good")]
x0, scale = 240, 340 / 140
parts = []
for g in (0, 20, 40, 60, 80, 100, 120, 140):
    parts += [line(x0 + g * scale, 104, x0 + g * scale, 116 + 4 * 52, EDGE, 1),
              mono(x0 + g * scale, 116 + 4 * 52 + 18, str(g), "middle", MUTED)]
for k, (label, ms, kind) in enumerate(rows):
    y = 116 + k * 52
    parts += [text(x0 - 14, y + 20, label, "sub", "end"),
              f'  <rect x="{x0}" y="{y}" width="{ms * scale:.1f}" height="30" rx="6" fill="{KINDS[kind][0]}"/>',
              text(x0 + ms * scale + 10, y + 20, f"{ms} ms", "name")]
parts += [text(x0, 116 + 4 * 52 + 40, "ms", "sub"),
          card(40, 380, 470, 56, "good", "shared bins: 6 to 28 times as fast", "the gain grows with contention"),
          mascot(526, 372, 80, "happy")]
save(OUT + "ch05-hist-times.svg", 660, 458,
     "Bars of histogram time for 64 MiB on a Jetson Orin NX: random bytes with global bins 33.4 ms, with "
     "shared bins 5.2 ms; all bytes equal to 7 with global bins 134.7 ms, with shared bins 4.8 ms.",
     "Histogram time, measured on 1 machine", "A Jetson Orin NX: 64 MiB, average of 10 runs, by sync.cu.", parts)
