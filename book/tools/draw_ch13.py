#!/usr/bin/env python3
"""Figures for Chapter 13, Your First Rust Kernel.

    python3 tools/draw_ch13.py
"""

import os

from svgkit import (BRASS, CREAM, GREEN, GREY, INK, MUTED, PALE, PINK, RUST, TEAL, Figure)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "figures")


def loop_vs_threads():
    f = Figure(760, 290)

    for px, title, foot in ((20, "One CPU core: a loop", "8 steps, one after another"),
                            (400, "The GPU: one thread per element", "1 step, all at the same time")):
        f.panel(px, 20, 340, 250, title)
        for k in range(8):
            f.cell(px + 18 + k * 39, 160, 34, 34, str(k), CREAM)
        f.text(px + 170, 250, foot, 12, MUTED, anchor="middle")

    # The CPU: one worker walks the row.
    f.cell(38, 62, 52, 30, "core", PINK, size=12)
    f.arrow(64, 96, 55, 152, RUST, 1.8)
    f.arrow(80, 140, 330, 140, RUST, 2.0, dash=True)
    f.text(205, 128, "i = 0, then 1, then 2, …", 12, RUST, mono=True, anchor="middle", halo=True)

    # The GPU: a thread above every cell.
    for k in range(8):
        x = 418 + k * 39
        f.cell(x, 70, 34, 26, "t%d" % k, GREEN, size=11)
        f.arrow(x + 17, 100, x + 17, 154, TEAL, 1.6)

    f.text(570, 222, "each thread: c[i] = a[i] + b[i]", 12, TEAL, mono=True, anchor="middle")
    f.save(os.path.join(OUT, "ch13-loop-vs-threads.svg"))


def finding_i():
    f = Figure(760, 330)
    n, block = 20, 8
    w, gap, x0 = 28, 2, 20

    f.text(380, 26, "N = 20 elements, blocks of 8 threads, so ceil(20 / 8) = 3 blocks", 13, INK, anchor="middle", bold=True)

    for b in range(3):
        bx = x0 + b * block * (w + gap) - 4
        f.zone(bx, 70, block * (w + gap) + 2, 86, "block %d" % b, TEAL, TEAL, 0.07, 12)

    for i in range(24):
        x = x0 + i * (w + gap)
        idle = i >= n
        f.text(x + w / 2, 96, str(i % block), 11, MUTED, mono=True, anchor="middle")
        f.cell(x, 110, w, 34, str(i), GREY if idle else CREAM, MUTED if idle else INK, size=12, dash=idle,
               color=MUTED if idle else INK)

    f.text(380, 182, "small number: the thread's place in its block.  Number in the cell: its global index i.", 11, MUTED, anchor="middle")

    f.rect(20, 210, 720, 96, PALE, INK, 8, width=1.1)
    f.text(40, 238, "i = block × 8 + thread", 15, INK, mono=True, bold=True)
    f.text(40, 264, "block 2, thread 3:   i = 2 × 8 + 3 = 19   → adds a[19] + b[19]", 13, INK, mono=True)
    f.text(40, 290, "block 2, thread 4:   i = 2 × 8 + 4 = 20   → 20 ≥ N, so it does nothing", 13, RUST, mono=True)
    f.save(os.path.join(OUT, "ch13-finding-i.svg"))


def round_trip():
    f = Figure(760, 320)
    f.panel(20, 20, 250, 280, "HOST: CPU memory")
    f.panel(490, 20, 250, 280, "DEVICE: GPU memory")

    a, b, c = (60, 70, 170, 40), (60, 130, 170, 40), (60, 230, 170, 40)
    ad, bd, cd = (530, 70, 170, 40), (530, 130, 170, 40), (530, 230, 170, 40)
    for box, text, fill in ((a, "a: Vec<f32>", CREAM), (b, "b: Vec<f32>", CREAM), (c, "c: Vec<f32>", GREEN),
                            (ad, "a_dev: DeviceBuffer", PALE), (bd, "b_dev: DeviceBuffer", PALE),
                            (cd, "c_dev: DeviceBuffer", PALE)):
        f.cell(*box, text, fill, size=12)

    f.link(a, ad, "1  from_host", TEAL)
    f.link(b, bd, "1  from_host", TEAL)
    f.link(cd, c, "3  to_host_vec", RUST)

    k = (560, 184, 110, 30)
    f.cell(*k, "vector_add", BRASS, size=12)
    f.text(552, 204, "2  launch", 11, INK, mono=True, anchor="end")
    f.link(bd, k)
    f.link(k, cd)
    f.line(700, 90, 718, 90, INK, 1.4)
    f.line(718, 90, 718, 199, INK, 1.4)
    f.arrow(718, 199, 676, 199, INK, 1.4)
    f.save(os.path.join(OUT, "ch13-round-trip.svg"))


def one_file():
    f = Figure(760, 280)
    src = (20, 115, 120, 46)
    cc = (180, 108, 170, 60)
    host = (400, 40, 170, 50)
    dev = (400, 190, 170, 50)
    exe = (610, 105, 135, 66)

    f.cell(*src, "main.rs", CREAM, size=14)
    for box, lines, fill in ((cc, ["rustc with the", "cuda-oxide backend"], PALE),
                             (host, ["host code:", "main, buffers, launch"], PINK),
                             (dev, ["#[kernel] code:", "vector_add"], GREEN),
                             (exe, ["one executable,", "GPU code", "inside"], BRASS)):
        f.rect(*box, fill, INK, 4)
        top = box[1] + box[3] / 2 - (len(lines) - 1) * 8 + 4
        for k, text in enumerate(lines):
            f.text(box[0] + box[2] / 2, top + 16 * k, text, 12, INK, anchor="middle")

    f.link(src, cc)
    f.link(cc, host, "CPU", label_offset=(-14, -6))
    f.link(cc, dev, "GPU", label_offset=(-14, 10))
    f.link(host, exe)
    f.link(dev, exe)
    f.text(380, 270, "cargo oxide run builds both halves from one file; kernels::load hands the GPU half to the driver", 11,
           MUTED, anchor="middle")
    f.save(os.path.join(OUT, "ch13-one-file.svg"))


def disjoint():
    f = Figure(760, 250)
    f.text(380, 26, "c.get_mut(idx): each thread gets its own cell, or None", 13, INK, anchor="middle", bold=True)

    for k in range(8):
        x = 60 + k * 84
        idle = k >= 6
        f.cell(x, 50, 60, 28, "t%d" % k, GREY if idle else GREEN, MUTED if idle else INK, size=12)
        if idle:
            f.text(x + 30, 118, "None", 13, RUST, mono=True, anchor="middle", bold=True)
            f.text(x + 30, 140, "skips the add", 10, MUTED, anchor="middle")
        else:
            f.arrow(x + 30, 82, x + 30, 160, TEAL, 1.6)
            f.text(x + 30, 118, "Some", 11, TEAL, mono=True, anchor="middle", halo=True)
            f.cell(x, 166, 60, 34, "c[%d]" % k, CREAM, size=12)

    f.text(60, 230, "N = 6: cells 0 to 5 exist. No two threads can hold the same cell, so no two can write it.", 12, MUTED)
    f.save(os.path.join(OUT, "ch13-disjoint.svg"))


if __name__ == "__main__":
    for draw in (loop_vs_threads, finding_i, round_trip, one_file, disjoint):
        draw()
        print("  " + draw.__name__)
