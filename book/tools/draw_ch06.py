"""The figures of chapter 6, drawn with figkit and figparts.

    python3 tools/draw_ch06.py

The times in figures 6.3, 6.5 and 6.6 are the ones measured with code/ch06_streams/streams.cu
on a Jetson Orin NX and a Jetson AGX Thor (the runs quoted in the chapter).
"""
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, bar, card, legend, line, mascot, mono, note,
                      save, tag, text, wire, zone)

OUT = "src/figures/"


# 6.1  An asynchronous launch: the host call returns at once; the GPU runs the kernel later.
gx, scale = 150, 40          # 1 ms = 40 px
parts = [text(gx - 14, 150, "host thread", "name", "end"), text(gx - 14, 236, "GPU", "name", "end"),
         bar(gx, 130, 0.07 * scale + 18, 34, "client", ""),
         bar(gx + 0.07 * scale + 24, 130, 9.6 * scale, 34, "idle", "free for other work", dashed=True),
         bar(gx + 9.9 * scale, 130, 66, 34, "danger", "wait"),
         bar(gx + 0.3 * scale, 216, 10.4 * scale, 34, "service", "kernel, 10.4 ms"),
         line(gx, 266, gx + 10.8 * scale + 30, 266, EDGE, 1)]
for ms in (0, 2, 4, 6, 8, 10):
    parts += [line(gx + ms * scale, 262, gx + ms * scale, 270, MUTED, 1),
              mono(gx + ms * scale, 288, f"{ms} ms", "middle", MUTED)]
parts += [tag(gx - 4, 104, "launch returns: 0.07 ms", "client"),
          tag(gx + 9.9 * scale + 70, 104, "cudaDeviceSynchronize()", "danger", "end"),
          note(40, 330, "The launch only puts the kernel in a queue. The host can prepare the next work", room=600),
          note(40, 348, "while the GPU computes, and waits only when it needs the result.", room=600)]
save(OUT + "ch06-async.svg", 660, 370,
     "Two timelines. Host thread: the launch call takes 0.07 ms, then the host is free for about 10 ms, then "
     "it waits in cudaDeviceSynchronize. GPU: the kernel runs for 10.4 ms, starting just after the launch.",
     "An asynchronous launch", "The launch call returns long before the kernel finishes.", parts)


# 6.2  Streams: each is an ordered queue; work in different streams may overlap.
parts = [zone(16, 104, 400, 210, "2 streams, filled by the host"),
         text(36, 152, "stream 0", "name"), text(36, 238, "stream 1", "name")]
for k, (name, kind) in enumerate([("copy in 0", "store"), ("kernel 0", "service"), ("copy out 0", "outside")]):
    parts.append(bar(120 + k * 96, 132, 88, 32, kind, name))
for k, (name, kind) in enumerate([("copy in 1", "store"), ("kernel 1", "service"), ("copy out 1", "outside")]):
    parts.append(bar(120 + k * 96, 218, 88, 32, kind, name))
parts += [note(36, 290, "in order within a stream", room=360),
          card(460, 112, 176, 50, "store", "copy engine", "host → device", glyph="swap"),
          card(460, 184, 176, 50, "service", "SMs", "kernels", glyph="chip"),
          card(460, 256, 176, 50, "outside", "copy engine", "device → host", glyph="swap"),
          wire([(416, 148), (438, 148), (438, 137), (460, 137)]),
          wire([(416, 209), (460, 209)]),
          wire([(416, 270), (438, 270), (438, 281), (460, 281)]),
          note(16, 348, "Work in 1 stream runs in the order it was queued. Work in different streams has no", room=628),
          note(16, 366, "order between them, so the GPU may run a copy of 1 stream during a kernel of another.", room=628)]
save(OUT + "ch06-streams.svg", 660, 388,
     "Two streams, each a queue of copy in, kernel and copy out. The queues feed 3 units of the GPU: a copy "
     "engine for host to device, the SMs for kernels, and a copy engine for device to host. Work in 1 stream "
     "runs in order; work in different streams may overlap.",
     "Streams", "Each stream is an ordered queue; different streams are independent.", parts)


# 6.3  The pipeline: 1 stream runs the stages one after another; chunks on 2 streams overlap them.
def stages(y, chunks, label):
    out = [text(150, y - 12, label, "sub")]
    rows = [("copy in", "store"), ("kernel", "service"), ("copy out", "outside")]
    unit_in, unit_k = 430 / (2 * 4.3 + 10.4) * 4.3 / chunks, 430 / (2 * 4.3 + 10.4) * 10.4 / chunks
    for r, (name, kind) in enumerate(rows):
        out.append(text(136, y + 18 + r * 34, name, "sub", "end"))
    for c in range(chunks):
        if chunks == 1:
            x_in, x_k, x_out = 150, 150 + unit_in, 150 + unit_in + unit_k
        else:
            x_in = 150 + c * unit_in
            x_k = 150 + unit_in + c * unit_k
            x_out = x_k + unit_k
        out += [bar(x_in, y, unit_in - 2, 26, "store", "" if chunks > 1 else "4.3 ms"),
                bar(x_k, y + 34, unit_k - 2, 26, "service", "" if chunks > 1 else "10.4 ms"),
                bar(x_out, y + 68, unit_in - 2, 26, "outside", "" if chunks > 1 else "4.3 ms")]
    return out


parts = stages(130, 1, "1 stream, 1 chunk: the 3 stages run one after another")
parts += stages(290, 4, "2 streams, 4 chunks: copies of 1 chunk overlap the kernel of another")
parts += [tag(600, 236, "22.6 ms", "danger", "end"), tag(468, 396, "12.9 ms", "good", "end"),
          note(40, 438, "Measured on a Jetson Orin NX with 8 chunks. The kernel, 10.4 ms, is the longest", room=600),
          note(40, 456, "stage, so no overlap can bring the total below about 10.4 ms.", room=600)]
save(OUT + "ch06-pipeline.svg", 660, 478,
     "Two timelines with rows copy in, kernel and copy out. 1 stream, 1 chunk: copy in 4.3 ms, then the kernel "
     "10.4 ms, then copy out 4.3 ms, one after another. 2 streams with 4 chunks: the copy of each chunk overlaps "
     "the kernel of the previous chunk and the copy out of the one before, so the rows overlap in a staircase. "
     "Measured on a Jetson Orin NX: 22.6 ms against 12.9 ms.",
     "A pipeline on 2 streams", "Chunks let the copy engines and the SMs work at the same time.", parts)


# 6.4  Events: markers in a stream that record the GPU's clock when the GPU reaches them.
parts = [text(40, 160, "stream", "name"),
         bar(120, 140, 90, 32, "store", "copy in"),
         bar(250, 140, 220, 32, "service", "kernel"),
         bar(510, 140, 100, 32, "outside", "copy out"),
         line(230, 120, 230, 196, KINDS["danger"][0], 2.4), line(490, 120, 490, 196, KINDS["danger"][0], 2.4),
         tag(230, 98, "event start", "danger", "middle"), tag(490, 98, "event stop", "danger", "middle"),
         wire([(230, 210), (490, 210)], label="cudaEventElapsedTime: 10.4 ms"),
         note(40, 268, "An event is a marker queued in a stream. The GPU records its own clock when it", room=600),
         note(40, 286, "reaches the marker, so the elapsed time covers only the work between the 2 markers.", room=600)]
save(OUT + "ch06-events.svg", 660, 308,
     "A stream holding copy in, kernel and copy out. An event marker named start sits just before the kernel and "
     "one named stop just after it. The elapsed time between the 2 events, 10.4 ms, is the kernel's time on the "
     "GPU.",
     "Events", "2 markers in a stream time the work between them on the GPU's own clock.", parts)


# 6.5  CUDA graphs: 1,000 launches one by one against 1 replay of a captured graph.
parts = [zone(16, 104, 628, 120, "1,000 launches, one by one"),
         zone(16, 248, 628, 120, "1 graph of 1,000 kernels, replayed")]
for k in range(9):
    parts += [bar(40 + k * 64, 146, 24, 30, "client", ""), bar(66 + k * 64, 146, 34, 30, "service", "")]
parts += [text(40, 200, "launch cost, then kernel, 1,000 times: 3.8 µs per kernel on Orin NX", "sub")]
parts += [bar(40, 290, 40, 30, "client", "")]
for k in range(14):
    parts.append(bar(84 + k * 38, 290, 34, 30, "service", ""))
parts += [text(40, 344, "1 launch, then the kernels back to back: 0.9 µs per kernel", "sub"),
          legend(40, 398, [("client", "host launch work"), ("service", "kernel on the GPU")])]
save(OUT + "ch06-graph.svg", 660, 420,
     "Two rows. One by one: each of 1,000 empty kernels is preceded by its own launch work on the host, 3.8 "
     "microseconds per kernel on an Orin NX. Graph: 1 launch submits all 1,000 kernels, which run back to back, "
     "0.9 microseconds per kernel.",
     "A CUDA graph", "Capture the launches once, then replay them with 1 call.", parts)


# 6.6  Measured pipeline times on 2 boards.
rows = [("Orin NX, 1 stream", 22.6, "danger"), ("Orin NX, 2 streams", 12.9, "good"),
        ("Thor, 1 stream", 5.2, "danger"), ("Thor, 2 streams", 4.5, "good")]
x0, scale = 220, 380 / 25
parts = []
for g in (0, 5, 10, 15, 20, 25):
    parts += [line(x0 + g * scale, 104, x0 + g * scale, 116 + 4 * 52, EDGE, 1),
              mono(x0 + g * scale, 116 + 4 * 52 + 18, str(g), "middle", MUTED)]
for k, (label, ms, kind) in enumerate(rows):
    y = 116 + k * 52
    parts += [text(x0 - 14, y + 20, label, "sub", "end"),
              f'  <rect x="{x0}" y="{y}" width="{ms * scale:.1f}" height="30" rx="6" fill="{KINDS[kind][0]}"/>',
              text(x0 + ms * scale + 10, y + 20, f"{ms} ms", "name")]
parts += [text(x0, 116 + 4 * 52 + 40, "ms, copy in + kernel + copy out of 128 MiB", "sub"),
          card(40, 380, 470, 56, "good", "overlap pays most when copies are long", "Orin NX: 43% less time; Thor: 13%"),
          mascot(526, 372, 80, "happy")]
save(OUT + "ch06-overlap-times.svg", 660, 458,
     "Bars of pipeline time for 128 MiB. Orin NX: 22.6 ms on 1 stream, 12.9 ms on 2 streams. Thor: 5.2 ms on "
     "1 stream, 4.5 ms on 2 streams.",
     "Pipeline time, measured on 2 boards", "Average of 5 runs, by streams.cu.", parts)
