"""The animations of chapter 6: a launch queue, and a pipeline of chunks on 2 streams.

    python3 tools/anim_ch06.py            # both
    python3 tools/anim_ch06.py queue      # the host fills a stream; the GPU drains it
    python3 tools/anim_ch06.py pipeline   # 1 chunk on 1 stream, then 4 chunks on 2

Work travels as small cards, so a reader sees which unit of the GPU is busy at each moment.
The stage times are the ones measured on a Jetson Orin NX: copy 4.3 ms, kernel 10.4 ms.
"""
import sys

from motion import (BRASS, BRASS_LT, INK, LINE, MUTED, RUST, RUST_LT, STAGE, TEAL, TEAL_LT, W,
                    Timeline, caption, clamp, linear, progress, render, title_block)

BLUE, BLUE_LT = "#3b6fd4", "#e6eefb"
VIOLET, VIOLET_LT = "#7a4fd0", "#f0eafb"
COPY, KERNEL = 4.3, 10.4


def box(p, x, y, w, h, title, sub="", edge=LINE, fill=STAGE, busy=0.0):
    p.rect(x, y, w, h, fill, edge, 12, 1.4 + 1.6 * busy)
    p.text(x + w / 2, y + 22, title, 13.5, INK, 700, "middle")
    if sub:
        p.text(x + w / 2, y + 40, sub, 11.5, MUTED, 500, "middle")


def card(p, x, y, w, h, label, colour, fill):
    p.rect(x, y, w, h, fill, colour, 7, 1.6)
    if label:
        p.text(x + w / 2, y + h / 2 + 4.5, label, 12, colour, 700, "middle")


def lerp(a, b, u):
    return a + (b - a) * clamp(u)


def through(a, mid, b, u):
    """Move from a into the station at mid, stay there while it works, then on to b."""
    if u < 0.15:
        return lerp(a, mid, u / 0.15)
    if u < 0.85:
        return mid
    return lerp(mid, b, (u - 0.85) / 0.15)


# ------------------------------------------------------------- launch queue

def queue():
    tl = Timeline(caption="", kind="step", sent=0.0, done=0.0, wait=0.0)

    tl.chapter("launch")
    tl.say("The host launches 3 kernels. Each launch only puts a command in the stream's queue "
           "and returns.")
    tl.to(2.4, ease=linear, sent=3.0)
    tl.say("The host goes on with its own work while the GPU takes the commands in order.")
    tl.to(5.4, ease=linear, done=2.0)

    tl.chapter("wait")
    tl.say("When the host needs the results, it calls cudaStreamSynchronize and waits for the "
           "queue to empty.")
    tl.set(wait=1.0)
    tl.to(2.7, ease=linear, done=3.0)
    tl.set(wait=0.0)
    tl.say("The queue is empty: the host continues, and the results are ready.", "insight")
    tl.wait(2.2)

    hx, qx, gx, y = 40, 300, 590, 110
    slots = [qx + 20 + k * 74 for k in range(3)]

    def draw(p, s, total):
        title_block(p, "A stream is a queue", "The host fills it; the GPU drains it in order.")
        box(p, hx, y, 200, 150, "host thread")
        box(p, qx, y, 250, 150, "stream 0", "commands, in order")
        box(p, gx, y, 190, 150, "GPU", "", TEAL, TEAL_LT, busy=float(s.sent > s.done))

        if s.wait > 0.5:
            p.text(hx + 100, y + 90, "waiting", 14, RUST, 700, "middle")
        else:
            p.text(hx + 100, y + 90, "own work", 14, BLUE, 700, "middle")

        finished = [k for k in range(3) if s.done - k >= 1]
        if finished:
            p.text(gx + 95, y + 134, "done: " + ", ".join("k%d" % k for k in finished), 12, TEAL, 700, "middle")

        first = int(s.done)
        for k in range(3):
            sent, ran = clamp(s.sent - k), clamp(s.done - k)
            if sent <= 0 or ran >= 1:
                continue
            if ran > 0:
                card(p, gx + 20, y + 62, 150, 30, "kernel %d runs" % k, TEAL, TEAL_LT)
                p.rect(gx + 20, y + 98, 150 * ran, 6, TEAL, "none", 3)
            else:
                x = lerp(hx + 60, slots[k - first], sent)
                card(p, x, y + 70, 66, 30, "k%d" % k, BRASS, BRASS_LT)

        caption(p, tl, s.t, 300)
        progress(p, tl, s.t, total, 300 + 2 * 23 + 26 + 22)

    return tl, draw, 300 + 2 * 23 + 26 + 22 + 40


# ------------------------------------------------------------- pipeline

def schedule(chunks):
    """(start, end) of copy in, kernel and copy out for every chunk, in ms."""
    cin, ker = COPY / chunks, KERNEL / chunks
    out, k_end = [], 0.0
    for c in range(chunks):
        a = c * cin
        b = max(a + cin, k_end)
        k_end = b + ker
        out.append(((a, a + cin), (b, k_end), (k_end, k_end + cin)))
    return out


def pipeline():
    tl = Timeline(caption="", kind="step", clock=0.0, phase=1.0)
    rate = 0.45                                   # seconds of animation per ms of GPU time

    tl.chapter("1 stream")
    tl.say("1 stream, 1 chunk of 128 MiB: copy in, then the kernel, then copy out. Each unit "
           "waits for the one before it.")
    serial = COPY * 2 + KERNEL
    tl.to(serial * rate, ease=linear, clock=serial)
    tl.say("19 ms of work, and at every moment 2 of the 3 units stood idle.", "fail")
    tl.wait(1.6)

    tl.chapter("2 streams")
    tl.set(phase=2.0, clock=0.0)
    tl.say("4 chunks on 2 streams. While the SMs work on 1 chunk, a copy engine brings in the next "
           "and the other sends back the last.")
    piped = schedule(4)[-1][2][1]
    tl.to(piped * rate, ease=linear, clock=piped)
    tl.say("12.6 ms: the copies hide behind the kernel. Measured: 22.6 ms against 12.9 ms.", "insight")
    tl.wait(2.2)

    y = 110
    stations = [(40, "host", "inputs"), (190, "copy engine", "host → device"), (350, "SMs", "kernel"),
                (510, "copy engine", "device → host"), (660, "host", "results")]
    centre = [x + 60 for x, _, _ in stations]

    def draw(p, s, total):
        title_block(p, "A pipeline of chunks", "3 units of the GPU, busy at the same time.")
        chunks = 1 if s.phase < 2 else 4
        plan = schedule(chunks)
        busy = [0.0] * 5
        for stages in plan:
            for k, (a, b) in enumerate(stages):
                if a <= s.clock < b:
                    busy[k + 1] = 1.0
        colours = [(LINE, STAGE), (BRASS, BRASS_LT), (TEAL, TEAL_LT), (VIOLET, VIOLET_LT), (LINE, STAGE)]
        for k, (x, title, sub) in enumerate(stations):
            edge, fill = colours[k]
            box(p, x, y, 120, 170, title, sub, edge if busy[k] or k in (0, 4) else LINE,
                fill if busy[k] or k in (0, 4) else STAGE, busy[k])

        h = 110 if chunks == 1 else 26
        w = 70 if chunks == 1 else 56
        for c, ((ia, ib), (ka, kb), (oa, ob)) in enumerate(plan):
            cy = y + 52 + (0 if chunks == 1 else c * 29)
            t = s.clock
            if t < ia:
                x = centre[0]
            elif t < ib:
                x = through(centre[0], centre[1], centre[2], (t - ia) / (ib - ia))
            elif t < kb:
                x = centre[2]
            elif t < oa:
                x = centre[2]
            elif t < ob:
                x = through(centre[2], centre[3], centre[4], (t - oa) / (ob - oa))
            else:
                x = centre[4]
            done = t >= kb
            label = ("128 MiB" if chunks == 1 else "c%d" % c)
            card(p, x - w / 2, cy, w, h, label, TEAL if done else BLUE, TEAL_LT if done else BLUE_LT)

        p.text(W - 30, y + 210, "GPU time: %5.1f ms" % s.clock, 15, INK, 700, "end", mono=True)
        caption(p, tl, s.t, 350)
        progress(p, tl, s.t, total, 350 + 2 * 23 + 26 + 22)

    return tl, draw, 350 + 2 * 23 + 26 + 22 + 40


BUILDERS = {"queue": ("ch06-queue.gif", queue), "pipeline": ("ch06-pipeline-anim.gif", pipeline)}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILDERS:
        name, fn = BUILDERS[key]
        tl, draw, height = fn()
        print(render(name, tl, draw, height))
