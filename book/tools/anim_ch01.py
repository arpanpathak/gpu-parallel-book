"""The chapter 1 animations, drawn with `motion`: things move, rather than swap.

Each builder scripts one run of the chapter's example and paints every frame from
the timeline's sampled values, in the manner of `anim_async.py`. Values travel
between panels as pills, a worker is drawn as a robot that sleeps while a read is
in flight and wakes when the value lands, and each run ends on a failing case.

    python3 tools/animations.py wait amdahl-multi flynn roofline
"""

import math

from motion import *  # noqa: F401,F403
from motion import Timeline, render

CODE_Y = 330

WAIT_CODE = [
    "i = this worker's element",
    "load a[i], load b[i]",
    "c[i] = a[i] + b[i]",
]

A_VALS = [3, 1, 4, 1]
B_VALS = [2, 7, 1, 8]
LANE_X = [360, 470, 580, 690]
DESK_Y = 236
PANEL_R = 288
ARC_Y = 70
CELL_A = [126, 166, 206, 246]
ROW_A, ROW_B, ROW_C = 128, 170, 282


def layout_for(code_lines, code_y=CODE_Y):
    code_h = 30 + code_lines * 17.5 + 8
    caption_y = code_y + code_h + 14
    rail_y = caption_y + 2 * 23.0 + 26 + 18
    return caption_y, rail_y, rail_y + 36


def furniture(p, tl, t, total, panel_title, title, sub, code, code_lines, cap_y, rail_y,
              tint=TEAL, code_y=CODE_Y):
    title_block(p, title, sub)
    code_panel(p, 26, code_y, W - 52, panel_title, code_lines, code,
               size=11.2, lead=17.5, tint=tint, reveal=tl.reached("code", t))
    caption(p, tl, t, cap_y)
    progress(p, tl, t, total, rail_y)


# ------------------------------------------------------------------ the wait

WAIT_CODE_Y = 520        # the wait scene needs room for two warp rows
WARP0_Y = 240            # warp 0's desk
WARP1_Y = 380            # warp 1's desk
CROW_Y = 425             # the c[i] row
ROW_CX = 525.0           # centre of a warp's four lanes


def route(points, u):
    """A point at fraction `u` of a polyline's total length."""
    total = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if total <= 0:
        return points[-1]
    target = u * total
    for a, b in zip(points, points[1:]):
        d = math.dist(a, b)
        if d <= 0:
            continue
        if target <= d:
            return (lerp(a[0], b[0], target / d), lerp(a[1], b[1], target / d))
        target -= d
    return points[-1]


def wait():
    """Two units and one scheduler. Unit 0 asks for its operands and waits; the
    scheduler runs unit 1 in the gap; unit 0 resumes when its values arrive."""
    tl = Timeline(
        code=-1.0,
        u=0.0, a=0.0, label="", color=TEAL, fill=TEAL_LT,
        fx=0.0, fy=0.0, tx=0.0, ty=0.0, arc=-1.0,
        runs=0.0, w0wait=0.0, w1work=0.0, w1done=0.0,
        cycles=0, results=0, writes=0, broken=0.0,
    )

    def trip(fx, fy, tx, ty, label, color, fill, dur=1.2, arc=-1.0):
        tl.set(u=0.0, a=0.0, label=label, color=color, fill=fill,
               fx=fx, fy=fy, tx=tx, ty=ty, arc=arc)
        # The pill fades in after it leaves the source, so it never covers the
        # cells it just read.
        tl.also(0.25, ease_out, delay=0.15 * dur, a=1.0)
        tl.to(dur, in_out, u=1.0)
        tl.to(0.16, ease_in, a=0.0)

    tl.chapter("stall")
    tl.to(0.3, code=0.0)
    tl.say("Unit 0 asks for a[] and b[]. The add needs those values, so unit 0 waits.")
    tl.wait(2.6)
    tl.set(w0wait=1.0, runs=0.0)
    trip(PANEL_R, ROW_A + 14, ROW_CX, WARP0_Y - 70, "load a[]", TEAL, TEAL_LT, dur=1.5)
    trip(PANEL_R, ROW_B + 14, ROW_CX, WARP0_Y - 70, "load b[]", TEAL, TEAL_LT, dur=1.3)
    tl.to(0.5, cycles=500)
    tl.say("Five hundred cycles pass before the values return. Unit 0 has nothing else to run "
           "in that time.")
    tl.wait(2.2)

    tl.chapter("switch")
    tl.set(runs=1.0, w1work=1.0)
    tl.say("The scheduler runs unit 1 instead. Unit 1 has independent work, so the arithmetic "
           "units keep running while unit 0 waits.")
    tl.to(2.2, in_out, w1done=4.0)
    tl.say("Unit 1 finishes four operations during the wait. Unit 0's read still takes its "
           "500 cycles.")
    tl.wait(2.4)

    tl.chapter("resume")
    tl.set(w0wait=0.0, runs=0.0, w1work=0.0)
    tl.to(0.3, code=2.0)
    tl.say("The values arrive. The scheduler returns to unit 0, which adds them and writes c[i].")
    tl.wait(1.0)
    trip(ROW_CX, WARP0_Y - 70, ROW_CX, CROW_Y + 14, "sums", BRASS, BRASS_LT, dur=1.2)
    tl.set(results=4)
    tl.wait(2.6)

    tl.chapter("bad index")
    tl.set(broken=1.0, results=0, writes=0)
    tl.to(0.3, code=0.0)
    tl.say("Now the index is wrong, and every worker of unit 0 takes element 0.", "fail")
    tl.wait(1.4)
    for k in range(4):
        pace = 1.0 if k == 0 else 0.45
        trip(LANE_X[k], WARP0_Y - 70, LANE_X[0], CROW_Y + 14,
             "%d" % (A_VALS[0] + B_VALS[0]), RUST, RUST_LT, dur=0.5 * pace)
        tl.set(writes=tl._at("writes", tl.now) + 1)
        if k in (0, 3):
            tl.say("Worker %d writes c[0] as well. %d writes have gone to the same cell."
                   % (k, k + 1), "fail")
        else:
            tl.wait(0.4)
    tl.say("Only c[0] holds a value, so c[1] to c[3] were never written.", "fail")
    tl.wait(2.4)

    def draw(p, s, total):
        t = s.t
        p.rect(26, 104, 262, 116, STAGE, LINE, 10, 1.2)
        p.text(40, 128, "memory", 12.5, MUTED, 600)
        for k in range(4):
            for vals, y in ((A_VALS, ROW_A), (B_VALS, ROW_B)):
                p.rect(CELL_A[k] - 17, y, 34, 28, PAPER, LINE, 6, 1.5)
                p.text(CELL_A[k], y + 19, str(vals[k]), 15, INK, 700, "middle", mono=True)
        p.text(40, 212, "a[i]  and  b[i]", 11.5, FAINT, 500)

        # The scheduler, and the unit it is running.
        p.rect(360, 92, 360, 28, "#fbfcfd", LINE, 8, 1.4)
        p.text(376, 111, "scheduler", 12, MUTED, 700)
        which = int(round(s.runs))
        edge = TEAL if which == 0 else BRASS
        p.rect(612, 97, 98, 18, TEAL_LT if which == 0 else BRASS_LT, edge, 6, 1.2)
        p.text(661, 110, "unit %d" % which, 11, edge, 700, "middle", mono=True)

        # Two units. Unit 0 sleeps while its read is out; unit 1 keeps computing,
        # and the row of markers beside it records the work it finishes.
        for row, dy in ((0, WARP0_Y), (1, WARP1_Y)):
            waiting = s.w0wait if row == 0 else 0.0
            active = (1.0 - waiting) if row == 0 else s.w1work
            body = RUST if (row == 0 and s.broken > 0.5) else (TEAL if row == 0 else NIGHT)
            bob = 5.0 * math.sin(math.floor(t * 3.0) / 3.0 * 6.0) if (row == 1 and s.w1work > 0.5) else 0.0
            for k in range(4):
                robot(p, LANE_X[k], dy, body=body, awake=1.0 - waiting, lit=active, bob=bob)
                zzz(p, LANE_X[k] + 26, dy - 70, t, waiting)
            p.text(744, dy - 46, "unit %d" % row, 12.5, body, 700, "start")
            if row == 1:
                done = int(round(s.w1done))
                for i in range(4):
                    on = i < done
                    p.rect(744 + i * 18, dy - 30, 14, 14, TEAL_LT if on else PAPER,
                           TEAL if on else LINE, 3, 1.2)
                p.text(744, dy - 8, "%d done" % done, 10.5, MUTED, 600, "start", mono=True)

        # The c[i] row.
        for k in range(4):
            written = (s.results >= 4 and s.broken < 0.5) or (s.broken > 0.5 and k == 0)
            fill, edge = (TEAL_LT, TEAL) if written and s.broken < 0.5 else (PAPER, LINE)
            if s.broken > 0.5 and k == 0:
                fill, edge = RUST_LT, RUST
            p.rect(LANE_X[k] - 20, CROW_Y, 40, 30, fill, edge, 7, 1.5)
            if written:
                value = A_VALS[0] + B_VALS[0] if s.broken > 0.5 else A_VALS[k] + B_VALS[k]
                p.text(LANE_X[k], CROW_Y + 21, str(value), 16, INK, 700, "middle", mono=True)
        p.text(LANE_X[0] - 62, CROW_Y + 20, "c[i]", 12, MUTED, 600, "end")

        # The pill in flight.
        if s.a > 0.01:
            if s.arc > 0.0:
                x, y = route([(s.fx, s.fy), (s.fx, s.arc), (s.tx, s.arc), (s.tx, s.ty)], s.u)
            else:
                mx, my = (s.fx + s.tx) / 2, (s.fy + s.ty) / 2 - 60
                x, y = bezier((s.fx, s.fy), (mx, my), (s.tx, s.ty), s.u)
            pill(p, x, y, s.label, s.color, s.fill, size=13, opacity=s.a,
                 scale=lerp(0.75, 1.0, clamp(s.u * 4)))

        cycles = int(round(s.cycles))
        scoreboard(p, [("cycles", cycles, TEAL), ("results", "%d/4" % s.results, BRASS),
                       ("writes to c[0]", s.writes if s.broken > 0.5 else 0, RUST)])
        meter(p, 26, 84, 186, min(1.0, s.cycles / 2000.0), TEAL, "cycles elapsed", str(cycles))

        furniture(p, tl, t, total, "the operation", "Two units, one scheduler",
                  "Unit 0 waits on its read while unit 1 keeps the arithmetic units busy.",
                  s.code, WAIT_CODE, cap_y, rail_y, tint=RUST if s.broken > 0.5 else TEAL,
                  code_y=WAIT_CODE_Y)

    cap_y, rail_y, height = layout_for(len(WAIT_CODE), WAIT_CODE_Y)
    return tl, draw, height



# ---------------------------------------------------------------- the roofline


ROOF_CODE = [
    "for k = 0 to N - 1",
    "acc = acc + A[ty][k] * B[k][tx]",
    "C[ty][tx] = acc",
]


def roofline():
    tl = Timeline(code=-1.0, k=-1.0, i=0.25, tile=0.0, hot=-1.0, broken=0.0, miss=0.0)
    tl.chapter("the loop")
    tl.to(0.3, code=0.0)
    tl.say("A matrix multiply computes one output element as a sum over k. The inner loop reads "
           "A[ty][k] and B[k][tx].")
    tl.wait(2.6)
    for k in range(4):
        tl.set(k=float(k), hot=float(k))
        tl.to(0.25, code=1.0)
        tl.say("k = %d: load A[ty][%d] and B[%d][tx] from memory." % (k, k, k))
        tl.wait(0.5)
    tl.chapter("reuse")
    tl.set(hot=-1.0, k=-1.0)
    tl.to(0.3, code=2.0)
    tl.say("Four k steps moved eight values for eight FLOPs, so the operation sits at 0.25 "
           "FLOP/byte on the bandwidth diagonal.", "insight")
    tl.wait(2.6)
    tl.say("Hold a 16 by 16 tile of C in registers, and each loaded value is reused 16 times.")
    tl.to(1.4, in_out, tile=1.0)
    tl.to(1.2, i=4.0)
    tl.say("The intensity rises sixteenfold, to about 4 FLOP/byte. The ridge is at 40, so the "
           "operation is still memory-bound.", "insight")
    tl.wait(2.4)
    tl.chapter("no tile")
    tl.set(broken=1.0)
    tl.to(1.2, in_out, tile=0.0)
    tl.to(1.0, i=0.25)
    tl.say("Remove the tile, and every k re-reads A and B from memory. The operation falls back "
           "to the diagonal.", "fail")
    tl.wait(1.0)
    tl.set(miss=1.0)
    tl.say("On the diagonal the bandwidth ceiling allows 0.6% of the arithmetic peak. The "
           "arithmetic units idle.", "fail")
    tl.wait(2.6)

    def draw(p, s, total):
        t = s.t
        x0, x1, y0, y1 = 120.0, 640.0, 96.0, 300.0
        lo, hi, plo, phi = 0.1, 100.0, 0.001, 1.0

        def X(v):
            return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

        def Y(v):
            frac = (math.log10(v) - math.log10(plo)) / (math.log10(phi) - math.log10(plo))
            return y1 - frac * (y1 - y0)

        def perf(intensity):
            return min(1.0, intensity / 40.0)

        p.line(x0, y0, x0, y1, INK, 2.0)
        p.line(x0, y1, x1, y1, INK, 2.0)
        for v in (0.1, 1.0, 10.0, 100.0):
            p.line(X(v), y1, X(v), y1 + 5, FAINT, 1.2)
            p.text(X(v), y1 + 19, "%g" % v, 10.5, FAINT, 400, "middle", mono=True)
        p.text((x0 + x1) / 2, y1 + 40, "arithmetic intensity, FLOP/byte", 12, MUTED, 600, "middle")
        for v in (0.001, 0.01, 0.1, 1.0):
            p.line(x0 - 5, Y(v), x0, Y(v), FAINT, 1.2)
            p.text(x0 - 10, Y(v) + 4, "%g" % v, 10, FAINT, 400, "end", mono=True)
        p.text(x0, y0 - 12, "performance, fraction of peak", 11, MUTED, 600, "start")
        p.line(X(lo), Y(perf(lo)), X(40.0), Y(1.0), TEAL, 3.4)
        p.line(X(40.0), Y(1.0), x1, Y(1.0), BRASS, 3.4)
        p.circle(X(40.0), Y(1.0), 6, RUST, PAPER, 2.0)
        p.text(X(40.0), Y(1.0) - 14, "ridge 40", 11.5, RUST, 700, "middle", mono=True)
        p.text(X(0.15), Y(perf(0.15)) + 26, "memory-bound", 11.5, TEAL, 700, "start")
        p.text(X(48.0), Y(1.0) + 22, "compute-bound", 11.5, BRASS, 700, "middle")
        for u in (0.25, 0.5, 0.75):
            xi = max(0.1, 0.25 + (s.i - 0.25) * u)
            p.circle(X(xi), Y(perf(xi)), 3.5, FAINT, "none", 0, opacity=0.5 * s.tile)
        px, py = X(max(0.1, s.i)), Y(perf(max(0.1, s.i)))
        p.circle(px, py, 7, RUST, PAPER, 2.0)
        pulse = math.floor(t * 3.0) / 3.0        # idle motion, quantised so frames merge
        p.circle(px, py, 15 + 4 * math.sin(pulse * 3.0), "none", RUST, 1.6, dash="4 3")
        p.text(px + 24, py + 14, "the operation", 11.5, RUST, 700, "start", mono=True)
        if s.miss > 0.5:
            p.text(px + 24, py + 32, "0.6% of peak", 11.5, RUST, 700, "start", mono=True)

        gx, gy, gs = 660.0, 112.0, 26.0
        p.text(gx + 2 * gs, gy - 14, "tile in registers", 11.5, MUTED, 600, "middle")
        for r in range(4):
            for c in range(4):
                on = s.tile > 0.5
                p.rect(gx + c * gs, gy + r * gs, gs - 4, gs - 4, TEAL_LT if on else "#f6f8fa",
                       TEAL if on else LINE, 4, 1.2)
        if s.hot >= 0:
            k = int(s.hot)
            p.rect(gx + k * gs - 2, gy - 2, gs, gs, "none", BRASS, 4, 2.2)
            p.rect(gx - 2, gy + k * gs - 2, gs, gs, "none", BRASS, 4, 2.2)

        scoreboard(p, [("intensity", "%.2f" % s.i, TEAL), ("ridge", "40", RUST)])
        furniture(p, tl, t, total, "the inner loop", "The roofline",
                  "Below the ridge, bandwidth limits it. Above it, the arithmetic units do.",
                  s.code, ROOF_CODE, cap_y, rail_y, tint=RUST if s.broken else TEAL)

    cap_y, rail_y, height = layout_for(len(ROOF_CODE))
    return tl, draw, height


# ------------------------------------------------------- Amdahl: the curves


AMD_CODE = [
    "S(p) = 1 / (f + (1 - f) / p)",
    "f = read+write / total = 30 / 130 = 0.23",
    "ceiling = 1 / f = 4.3",
]

FAMILY = [0.001, 0.01, 0.05, 0.10]
FAMILY_COLOR = ["#b7c2cc", "#8fa6b4", "#6b8fa0", "#4a7a8c"]


def amdahl_s(p, f):
    return 1.0 / (f + (1.0 - f) / p)


def amdahl_multi():
    """Amdahl's law as a family of curves: speedup against unit count, one
    curve per serial fraction, with the example run climbing its own curve."""
    tl = Timeline(code=-1.0, family=0.0, f=30.0 / 130.0, p=1.0, slow=0.0)
    tl.chapter("the curves")
    tl.to(0.3, code=0.0)
    tl.say("Amdahl's law as a family of curves. Each curve is S(p) = 1 / (f + (1 - f) / p), "
           "for one serial fraction f.")
    tl.wait(2.6)
    tl.to(0.5, family=1.0)
    tl.say("f = 0.001: one part in a thousand is serial, so the ceiling is 1000x.")
    tl.wait(1.4)
    tl.to(0.6, family=3.0)
    tl.say("f = 0.01 and f = 0.05 lower the ceiling to 100x and 20x.")
    tl.wait(1.8)
    tl.to(0.6, family=4.0)
    tl.say("f = 0.10 sits below 10x. A larger serial part bends the curve down.")
    tl.wait(2.2)
    tl.chapter("our run")
    tl.to(0.4, code=1.0)
    tl.say("This program: the read and write are 30 of 130 microseconds, so f = 0.23 and the "
           "ceiling is 4.3x.", "insight")
    tl.wait(2.6)
    tl.say("On one unit S = 1.00.")
    tl.wait(0.8)
    tl.to(1.0, in_out, p=2.0)
    tl.say("Two units halve the parallel work: S = 1.63, not 2.")
    tl.wait(1.2)
    tl.to(0.9, in_out, p=4.0)
    tl.say("Four units: S = 2.36.")
    tl.wait(1.0)
    tl.to(0.9, in_out, p=8.0)
    tl.say("Eight units: S = 3.06 against the 4.3 ceiling.")
    tl.wait(1.8)
    tl.chapter("slow read")
    tl.set(slow=1.0)
    tl.say("Now the input read is slow: 50 microseconds, because the file sits on a slower "
           "device.", "fail")
    tl.to(1.0, in_out, f=60.0 / 160.0, p=1.0)
    tl.say("The serial part is 60 of 160 microseconds, so f = 0.375 and the ceiling falls "
           "to 2.7x.", "fail")
    tl.to(0.9, in_out, p=8.0)
    tl.say("Eight units now reach 2.2x. The read and write set the pace.", "fail")
    tl.wait(2.4)

    def draw(p, s, total):
        t = s.t
        x0, x1, y0, y1 = 96.0, 470.0, 96.0, 300.0

        def X(v):
            return x0 + (math.log10(max(v, 1.0)) / 3.0) * (x1 - x0)

        def Y(v):
            return y1 - (math.log10(max(v, 1.0)) / 3.0) * (y1 - y0)

        p.line(x0, y0, x0, y1, INK, 2.0)
        p.line(x0, y1, x1, y1, INK, 2.0)
        for v in (1, 4, 16, 64, 256, 1024):
            p.line(X(v), y1, X(v), y1 + 5, FAINT, 1.2)
            p.text(X(v), y1 + 19, str(v), 10.5, FAINT, 400, "middle", mono=True)
        p.text((x0 + x1) / 2, y1 + 40, "units, log scale", 12, MUTED, 600, "middle")
        for v in (1, 4, 16, 64, 256, 1000):
            p.line(x0 - 5, Y(v), x0, Y(v), FAINT, 1.2)
            p.text(x0 - 10, Y(v) + 4, str(v), 10.5, FAINT, 400, "end", mono=True)
        p.text(x0, y0 - 16, "speedup, log scale", 12, MUTED, 600, "start")
        # Family curves appear one by one, labelled inside the plot.
        for k in range(len(FAMILY)):
            if s.family < k + 0.5:
                continue
            pts = [(X(pv), Y(amdahl_s(pv, FAMILY[k]))) for pv in
                   [10 ** (i / 20.0) for i in range(61)]]
            d = "M " + " L ".join("%.2f %.2f" % pt for pt in pts)
            p.path(d, "none", FAMILY_COLOR[k], 1.8)
            p.text(x1 - 6, Y(amdahl_s(1024, FAMILY[k])) - 8, "f = %.3g" % FAMILY[k],
                   10.5, FAMILY_COLOR[k], 600, "end", mono=True)
        # The example curve, rust when the run is on it.
        f = s.f
        pts = [(X(pv), Y(amdahl_s(pv, f))) for pv in [10 ** (i / 20.0) for i in range(61)]]
        d = "M " + " L ".join("%.2f %.2f" % pt for pt in pts)
        curve_color = RUST if s.slow > 0.5 else TEAL
        p.path(d, "none", curve_color, 3.2)
        p.text(x1 - 6, Y(amdahl_s(1024, f)) - 8, "this program, f = %.3g" % f, 11,
               curve_color, 700, "end", mono=True)
        ceiling = 1.0 / f
        cy = Y(ceiling)
        p.line(x0, cy, x1, cy, curve_color, 1.6, dash="5 4")
        p.text(x0 + 6, cy - 8, "ceiling %.1fx" % ceiling, 11, curve_color, 700, "start",
               mono=True)
        # The running point, with a fixed readout in the empty top-left corner.
        speedup = amdahl_s(s.p, f)
        px, py = X(s.p), Y(speedup)
        p.circle(px, py, 7, curve_color, PAPER, 2.0)
        p.text(x0 + 8, y0 + 22, "p = %g, S = %.2f" % (s.p, speedup), 13, curve_color, 700,
               "start", mono=True)

        # Readout panel.
        rx, ry = 596.0, 108.0
        p.text(rx, ry, "the program", 12, MUTED, 600)
        for label, value, color in (("f =", "%.3g" % f, curve_color),
                                    ("ceiling 1/f =", "%.1fx" % ceiling, curve_color),
                                    ("measured S(p) =", "%.2fx" % speedup, TEAL)):
            p.text(rx, ry + 30, label, 12.5, MUTED, 600)
            p.text(rx + 150, ry + 30, value, 15, color, 700, "start", mono=True)
            ry += 30
        meter(p, rx, ry + 52, 196, min(1.0, speedup / ceiling), TEAL, "share of the ceiling",
              "%.0f%%" % (100 * speedup / ceiling))

        scoreboard(p, [("units", "%g" % s.p, TEAL), ("S(p)", "%.2f" % speedup, INK)])
        furniture(p, tl, t, total, "the law", "Amdahl's law",
                  "The serial fraction f sets the ceiling. Only the divisible part shrinks.",
                  s.code, AMD_CODE, cap_y, rail_y, tint=RUST if s.slow > 0.5 else TEAL)

    cap_y, rail_y, height = layout_for(len(AMD_CODE))
    return tl, draw, height


# -------------------------------------------------------- Flynn's taxonomy


FLYNN_CODE = {
    "sisd": ["ADD c[i], a[i], b[i]", "i = i + 1", "repeat for i = 0 .. 7"],
    "simd": ["VADDPS ymm0, ymm1, ymm2", "one instruction, 8 floats", ""],
    "mimd": ["core 0: ADD c[i], a[i], b[i]", "core 1: DOT acc, a[i], b[i]", ""],
    "simt": ["c[i] = a[i] + b[i]", "sent to 32 threads", ""],
    "diverge": ["if (a[i] > 2) c[i] = a[i] + b[i]", "else c[i] = b[i]", ""],
}

FLYNN_SUMS = [5, 8, 5, 9, 7, 17, 3, 14]
THEN_LANES = {0, 2, 4, 5, 7}

CELL_X0, CELL_Y, CELL_W, CELL_H = 130.0, 252.0, 56.0, 44.0
CELL_PITCH = 68.0
SLOT_Y = 186.0


def cell_center(k):
    return CELL_X0 + k * CELL_PITCH + CELL_W / 2.0, CELL_Y + CELL_H / 2.0


def flynn():
    """The same add over eight elements on the four machine types of Flynn's
    taxonomy, then a divergent group as the failing case."""
    tl = Timeline(code=-1.0, mode="sisd", issue=0.0, hot=-1.0, done=0.0,
                  pill_u=0.0, pill_a=0.0, pill_from=0.0, pill_tx=0.0, pill_ty=0.0,
                  pill_label="", pill_color=TEAL, pill_fill=TEAL_LT,
                  lanes_a=0.0, lanes_b=0.0)

    def trip(k, label, color, fill, dur=0.6, frm=0.0):
        tx, ty = cell_center(k)
        tl.set(pill_u=0.0, pill_a=0.0, pill_from=frm, pill_tx=tx, pill_ty=ty,
               pill_label=label, pill_color=color, pill_fill=fill)
        tl.to(0.10, ease_out, pill_a=1.0)
        tl.to(dur, in_out, pill_u=1.0)
        tl.to(0.12, ease_in, pill_a=0.0)

    def trip_all(label, color, fill, dur=0.7, frm=0.0):
        tx = CELL_X0 + 4 * CELL_PITCH - CELL_W / 2.0
        ty = CELL_Y + CELL_H / 2.0
        tl.set(pill_u=0.0, pill_a=0.0, pill_from=frm, pill_tx=tx, pill_ty=ty,
               pill_label=label, pill_color=color, pill_fill=fill)
        tl.to(0.10, ease_out, pill_a=1.0)
        tl.to(dur, in_out, pill_u=1.0)
        tl.to(0.12, ease_in, pill_a=0.0)

    # SISD.
    tl.chapter("SISD")
    tl.to(0.3, code=0.0)
    tl.say("SISD, single instruction single data: one scalar core performs one ADD per element.")
    tl.wait(2.6)
    for k in range(8):
        first = k == 0
        pace = 1.0 if first else 0.32
        tl.set(hot=float(k))
        trip(k, "ADD", TEAL, TEAL_LT, dur=0.55 * pace)
        tl.set(done=float(k + 1), issue=float(k + 1))
        if first:
            tl.say("The first ADD reads a[0] and b[0] and writes c[0]. One instruction, "
                   "one data pair.")
            tl.wait(1.0)
        else:
            tl.wait(0.15)
    tl.say("Eight elements need eight instructions. The data streams one value at a time.")
    tl.wait(2.4)

    # SIMD.
    tl.chapter("SIMD")
    tl.set(mode="simd", code=1.0, issue=1.0, done=0.0, hot=-1.0)
    tl.say("SIMD, single instruction multiple data: the compiler packs eight floats into one "
           "256-bit register.")
    tl.wait(2.6)
    trip_all("VADDPS", TEAL, TEAL_LT, dur=0.8)
    tl.set(done=8.0)
    tl.say("One VADDPS instruction adds all eight pairs at once. One instruction, eight results.")
    tl.wait(2.8)

    # MIMD.
    tl.chapter("MIMD")
    tl.set(mode="mimd", code=2.0, issue=1.0, done=0.0)
    tl.say("MIMD, multiple instruction multiple data: two cores, two instruction streams.")
    tl.wait(2.4)
    trip(0, "ADD", TEAL, TEAL_LT, dur=0.6, frm=0.0)
    tl.set(done=1.0)
    trip(4, "DOT", BRASS, BRASS_LT, dur=0.6, frm=1.0)
    tl.set(done=2.0)
    tl.say("Core 0 runs ADD over its elements while core 1 runs a dot product. Each core "
           "fetches its own instruction stream.")
    tl.wait(3.2)

    # SIMT.
    tl.chapter("SIMT")
    tl.set(mode="simt", code=3.0, issue=1.0, done=0.0)
    tl.say("SIMT, single instruction multiple threads: one instruction is sent to a group "
           "of 32 threads.")
    tl.wait(2.6)
    trip_all("ADD", TEAL, TEAL_LT, dur=0.7)
    tl.set(done=8.0)
    tl.say("Each thread holds its own data and its own registers. The threads run together "
           "while their control flow agrees.")
    tl.wait(3.2)

    # Divergence, the failing case.
    tl.chapter("diverge")
    tl.set(mode="diverge", code=4.0, issue=2.0, done=0.0)
    tl.say("Now the code branches on a[i] > 2. Threads 0, 2, 4, 5, and 7 take the then side.",
           "fail")
    tl.wait(2.6)
    tl.set(lanes_a=1.0, lanes_b=0.0)
    trip_all("then", TEAL, TEAL_LT, dur=0.7)
    tl.set(done=5.0)
    tl.say("The other three threads wait. When the then side finishes, the else side runs.",
           "fail")
    tl.wait(2.6)
    tl.set(lanes_a=0.0, lanes_b=1.0)
    trip_all("else", RUST, RUST_LT, dur=0.7)
    tl.set(done=8.0)
    tl.say("Both paths run in turn, so a split group costs two passes. Chapter 5 measures "
           "the cost.", "fail")
    tl.wait(2.8)

    def draw(p, s, total):
        t = s.t
        lines = FLYNN_CODE[s.mode]
        mode = s.mode

        # The instruction slot, or two of them for MIMD.
        if mode == "mimd":
            p.rect(150, SLOT_Y, 250, 40, "#fbfcfd", TEAL, 8, 1.5)
            p.text(162, SLOT_Y + 26, "core 0: ADD", 12, TEAL, 700, mono=True)
            chip(p, 275, SLOT_Y + 56, "instructions: 1", TEAL, TEAL_LT, size=10.5)
            p.rect(420, SLOT_Y, 250, 40, "#fbfcfd", BRASS, 8, 1.5)
            p.text(432, SLOT_Y + 26, "core 1: DOT", 12, BRASS, 700, mono=True)
            chip(p, 545, SLOT_Y + 56, "instructions: 1", BRASS, BRASS_LT, size=10.5)
        else:
            slot_text = {
                "sisd": "ADD c[i], a[i], b[i]",
                "simd": "VADDPS ymm0, ymm1, ymm2",
                "simt": "c[i] = a[i] + b[i]",
                "diverge": "if (a[i] > 2) c[i] = a[i] + b[i]",
            }[mode]
            p.rect(150, SLOT_Y, 520, 40, "#fbfcfd", RUST if mode == "diverge" else TEAL, 8, 1.5)
            p.text(162, SLOT_Y + 26, slot_text, 12, INK, 600, mono=True)
            chip(p, 410, SLOT_Y + 54, "instructions: %d" % int(s.issue), RUST if mode == "diverge" else TEAL,
                 RUST_LT if mode == "diverge" else TEAL_LT, size=10.5)

        # SIMD: a frame around the eight cells is the packed register.
        if mode == "simd":
            p.rect(CELL_X0 - 10, CELL_Y - 8, 8 * CELL_PITCH - 24, CELL_H + 16, "none", TEAL, 10,
                   1.8, dash="5 4")
            p.text(CELL_X0 + 4 * CELL_PITCH - 28, CELL_Y + CELL_H + 18, "one 256-bit register, 8 floats",
                   10.5, TEAL, 600, "middle")

        # The eight elements.
        for k in range(8):
            x = CELL_X0 + k * CELL_PITCH
            fill, edge = STAGE, LINE
            value = FLYNN_SUMS[k]
            if mode == "sisd":
                if k < s.done:
                    fill, edge = TEAL_LT, TEAL
                elif abs(s.hot - k) < 0.5:
                    fill, edge = BRASS_LT, BRASS
            elif mode in ("simd", "simt"):
                if s.done >= 8:
                    fill, edge = TEAL_LT, TEAL
                elif s.pill_a > 0.01:
                    fill, edge = BRASS_LT, BRASS
            elif mode == "mimd":
                if k == 0:
                    fill, edge = TEAL_LT, TEAL
                elif k == 4:
                    fill, edge = BRASS_LT, BRASS
            elif mode == "diverge":
                is_then = k in THEN_LANES
                if is_then and s.lanes_a > 0.5:
                    fill, edge = TEAL_LT, TEAL
                elif (not is_then) and s.lanes_b > 0.5:
                    fill, edge = RUST_LT, RUST
                else:
                    fill, edge = "#eef2f5", LINE
            p.rect(x, CELL_Y, CELL_W, CELL_H, fill, edge, 8, 1.5)
            p.text(x + CELL_W / 2, CELL_Y + 17, "i=%d" % k, 10.5, MUTED, 600, "middle", mono=True)
            show_value = (mode == "sisd" and k < s.done) or \
                         (mode in ("simd", "simt") and s.done >= 8) or \
                         (mode == "mimd" and k in (0, 4)) or \
                         (mode == "diverge" and ((k in THEN_LANES and s.lanes_a > 0.5) or
                                                 (k not in THEN_LANES and s.lanes_b > 0.5)))
            if show_value:
                p.text(x + CELL_W / 2, CELL_Y + 34, str(value), 12, INK, 700, "middle", mono=True)
            if mode in ("simt", "diverge"):
                p.text(x + CELL_W / 2, CELL_Y + CELL_H + 14, "thread %d" % k, 10, MUTED, 500,
                       "middle", mono=True)

        # The pill in flight.
        if s.pill_a > 0.01:
            if mode == "mimd":
                sx = 275.0 if s.pill_from < 0.5 else 545.0
            else:
                sx = 410.0
            sy = SLOT_Y + 20
            mx, my = (sx + s.pill_tx) / 2.0, (sy + s.pill_ty) / 2.0 - 52
            x, y = bezier((sx, sy), (mx, my), (s.pill_tx, s.pill_ty), s.pill_u)
            pill(p, x, y, s.pill_label, s.pill_color, s.pill_fill, size=12,
                 opacity=s.pill_a, scale=lerp(0.75, 1.0, clamp(s.pill_u * 4)))

        scoreboard(p, [("instructions", "%d" % int(s.issue), TEAL), ("results", "%d/8" % int(s.done),
                      BRASS if mode != "diverge" else RUST)])
        furniture(p, tl, t, total, "the machine", "Flynn's taxonomy",
                  "One add over eight elements, on each of the four machine types.",
                  s.code, lines, cap_y, rail_y, tint=RUST if mode == "diverge" else TEAL)

    cap_y, rail_y, height = layout_for(3)
    return tl, draw, height


def build(name, fn, extra=()):
    def run(only=None):
        tl, draw, height = fn()
        return render(name, tl, draw, height, only=only, extra_colors=extra)
    return run


BUILDERS = {
    "wait": build("ch01-vadd.gif", wait),
    "amdahl-multi": build("ch01-amdahl-multi.gif", amdahl_multi),
    "flynn": build("ch01-flynn.gif", flynn),
    "roofline": build("ch01-roofline.gif", roofline),
}

if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or BUILDERS:
        print(BUILDERS[name]())
