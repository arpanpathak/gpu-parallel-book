"""Animations that move: a timeline of eased values, drawn at 20 frames a second.

The older animations in this book are keyframe slides. This module draws every
frame from a timeline instead, so a message travels from one actor to another,
a value flips in place, and a thread visibly falls asleep. The reader's eye
follows the thing that moved, which is the thing the caption is talking about.

A builder does three things:

  1. Build a `Timeline`: set values, tween them, and wait, in script order.
  2. Write `draw(p, s)`, which paints one frame from the sampled values `s`.
  3. Call `render(...)`, which samples the timeline, rasterises each distinct
     frame with rsvg-convert in parallel, and writes one GIF with one palette.

Frames whose SVG is identical to the previous one are merged into a longer
hold, so a pause costs nothing in the file.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "figures"

FPS = 20
SCALE = 1.85          # output pixels per drawing unit, as in the slide animations
W = 820

# ------------------------------------------------------------------ colours

INK = "#1d2733"
MUTED = "#5b6b7c"
FAINT = "#9aa8b6"
LINE = "#d5dde5"
PAPER = "#ffffff"
STAGE = "#f6f8fa"
TEAL = "#2f7f86"
TEAL_LT = "#e3f1ea"
BRASS = "#c89b3c"
BRASS_LT = "#fdf6e8"
RUST = "#d9622b"
RUST_LT = "#fbe3d6"
NIGHT = "#4b5a8c"      # a parked thread
NIGHT_LT = "#e8eaf5"
SCREEN = "#22303d"     # a robot's face
GLOW = "#7fe0c8"       # a robot's eyes

SANS = "Avenir Next, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO_FONT = "Menlo, monospace"


# ------------------------------------------------------------------- easing

def linear(u):
    return u


def in_out(u):
    return 4 * u * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_out(u):
    return 1 - (1 - u) ** 3


def ease_in(u):
    return u * u * u


def back(u, s=1.7):
    u -= 1
    return u * u * ((s + 1) * u + s) + 1


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, u):
    return a + (b - a) * u


def hex_rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, u):
    ra, rb = hex_rgb(a), hex_rgb(b)
    return "#%02x%02x%02x" % tuple(round(lerp(x, y, clamp(u))) for x, y in zip(ra, rb))


def _interp(a, b, u):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return lerp(a, b, u)
    if isinstance(a, str) and isinstance(b, str) and a.startswith("#") and b.startswith("#") \
            and len(a) == 7 and len(b) == 7:
        return mix(a, b, u)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(_interp(x, y, u) for x, y in zip(a, b))
    return b if u >= 1.0 else a


# ----------------------------------------------------------------- timeline


class Timeline:
    """Named values that change over time, written as a script.

    `to` tweens values and moves the cursor to the end of the tween. `also`
    tweens without moving the cursor, for things that happen together. `set`
    changes values at the cursor. `event` records a moment, and `age` tells a
    drawing how long ago it happened, which is how one-shot effects like a
    ringing bell or a flash are drawn.

    Pacing is built in, for readers who need time to read before they watch.
    `say` holds the previous caption until it has been on screen long enough to
    read slowly, shows the new one, and waits a beat before anything moves.
    Every scripted duration is also stretched by `pace`.
    """

    pace = 1.2           # motion runs this much slower than it is scripted
    lead = 0.9           # seconds a new caption is on screen before anything moves
    read_rate = 0.36     # seconds of reading per word: about 165 words a minute
    min_read = 2.6       # no caption is on screen for less than this

    def __init__(self, **initial):
        self.initial = dict(initial)
        self.initial.setdefault("caption", "")
        self.initial.setdefault("kind", "step")
        self.segments: dict[str, list] = {k: [] for k in self.initial}
        self.events: dict[str, list] = {}
        self.chapters: list[tuple[float, str]] = []
        self.now = 0.0
        self._said = None

    def _at(self, name, t):
        value = self.initial[name]
        for start, end, a, b, ease in self.segments[name]:
            if t < start:
                break
            if t >= end:
                value = b
            else:
                value = _interp(a, b, ease((t - start) / (end - start)))
        return value

    def _tween(self, start, dur, ease, values):
        for name, target in values.items():
            if name not in self.initial:
                raise KeyError("timeline has no value named %r" % name)
            a = self._at(name, start)
            segs = self.segments[name]
            segs.append((start, start + max(dur, 0.0), a, target, ease))
            segs.sort(key=lambda seg: seg[0])

    def set(self, **values):
        self._tween(self.now, 0.0, linear, values)
        return self

    def to(self, dur, ease=in_out, **values):
        dur *= self.pace
        self._tween(self.now, dur, ease, values)
        self.now += dur
        return self

    def also(self, dur, ease=in_out, delay=0.0, **values):
        self._tween(self.now + delay * self.pace, dur * self.pace, ease, values)
        return self

    def wait(self, dur):
        self.now += dur * self.pace
        return self

    def event(self, name, delay=0.0):
        self.events.setdefault(name, []).append(self.now + delay * self.pace)
        return self

    def reading_time(self, text):
        return max(self.min_read, self.read_rate * len(str(text).split()))

    def say(self, text, kind="step"):
        """Change the narration, but only once the last line has been read."""
        if self._said is not None:
            when, previous = self._said
            self.now = max(self.now, when + self.reading_time(previous))
        self.set(caption=text, kind=kind)
        self._said = (self.now, text)
        self.now += self.lead
        return self

    def end(self):
        """When the script is over, including time to read the last caption."""
        if self._said is None:
            return self.now
        when, previous = self._said
        return max(self.now, when + self.reading_time(previous))

    def reached(self, name, t):
        """The highest value `name` has taken at or before `t`: how far a code
        panel's highlight has got, so the lines below it can stay hidden."""
        best = self.initial[name]
        for start, _end, _a, b, _ease in self.segments[name]:
            if start <= t and isinstance(b, (int, float)):
                best = max(best, b)
        return best

    def chapter(self, label=""):
        self.chapters.append((self.now, label))
        return self

    def sample(self, t):
        s = Sample({name: self._at(name, t) for name in self.initial})
        s.t = t
        s.timeline = self
        return s

    def age(self, name, t):
        """Seconds since the latest `name` event at or before `t`, or None."""
        latest = None
        for when in self.events.get(name, []):
            if when <= t:
                latest = when
        return None if latest is None else t - latest

    def changed(self, name, t):
        """The time `name` last started changing, at or before `t`, and its old value."""
        when, old = 0.0, None
        for start, _end, a, _b, _ease in self.segments[name]:
            if start <= t:
                when, old = start, a
        return when, old


class Sample(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# ------------------------------------------------------------ SVG painting


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def f2(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def text_width(s, size, mono=False):
    return len(str(s)) * size * (0.602 if mono else 0.56)


class Pic:
    """An SVG canvas with the handful of primitives the motion builders use."""

    def __init__(self, width, height, background=PAPER):
        self.width, self.height = width, height
        self.parts: list[str] = []
        self.defs: list[str] = [
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="160%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1d2733" '
            'flood-opacity="0.16"/></filter>',
            '<filter id="lift" x="-30%" y="-30%" width="160%" height="190%">'
            '<feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#1d2733" '
            'flood-opacity="0.22"/></filter>',
            '<filter id="blur" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur stdDeviation="8"/></filter>',
        ]
        self.background = background

    def add(self, markup):
        self.parts.append(markup)

    @contextmanager
    def group(self, opacity=1.0, dx=0.0, dy=0.0, rotate=0.0, cx=0.0, cy=0.0, scale=1.0):
        attrs = []
        transform = []
        if dx or dy:
            transform.append("translate(%s %s)" % (f2(dx), f2(dy)))
        if rotate:
            transform.append("rotate(%s %s %s)" % (f2(rotate), f2(cx), f2(cy)))
        if scale != 1.0:
            transform.append("translate(%s %s) scale(%s) translate(%s %s)"
                             % (f2(cx), f2(cy), f2(scale), f2(-cx), f2(-cy)))
        if transform:
            attrs.append('transform="%s"' % " ".join(transform))
        if opacity < 0.999:
            attrs.append('opacity="%s"' % f2(max(0.0, opacity)))
        self.parts.append("<g %s>" % " ".join(attrs))
        try:
            yield self
        finally:
            self.parts.append("</g>")

    def rect(self, x, y, w, h, fill=PAPER, stroke="none", rx=6, width=1.5, opacity=1.0,
             dash=None, shadow=None):
        extra = ""
        if dash:
            extra += ' stroke-dasharray="%s"' % dash
        if opacity < 0.999:
            extra += ' opacity="%s"' % f2(max(0.0, opacity))
        if shadow:
            extra += ' filter="url(#%s)"' % shadow
        self.add('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s" stroke="%s" '
                 'stroke-width="%s"%s/>' % (f2(x), f2(y), f2(max(w, 0)), f2(max(h, 0)), f2(rx),
                                            fill, stroke, f2(width), extra))

    def circle(self, cx, cy, r, fill=PAPER, stroke="none", width=1.5, opacity=1.0, dash=None,
               filter=None):
        extra = ""
        if dash:
            extra += ' stroke-dasharray="%s"' % dash
        if opacity < 0.999:
            extra += ' opacity="%s"' % f2(max(0.0, opacity))
        if filter:
            extra += ' filter="url(#%s)"' % filter
        self.add('<circle cx="%s" cy="%s" r="%s" fill="%s" stroke="%s" stroke-width="%s"%s/>'
                 % (f2(cx), f2(cy), f2(max(r, 0)), fill, stroke, f2(width), extra))

    def line(self, x1, y1, x2, y2, color=INK, width=1.5, dash=None, opacity=1.0, cap="round"):
        extra = ' stroke-dasharray="%s"' % dash if dash else ""
        if opacity < 0.999:
            extra += ' opacity="%s"' % f2(max(0.0, opacity))
        self.add('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s" '
                 'stroke-linecap="%s"%s/>' % (f2(x1), f2(y1), f2(x2), f2(y2), color, f2(width),
                                              cap, extra))

    def path(self, d, fill="none", stroke=INK, width=1.5, opacity=1.0, dash=None, join="round"):
        extra = ' stroke-dasharray="%s"' % dash if dash else ""
        if opacity < 0.999:
            extra += ' opacity="%s"' % f2(max(0.0, opacity))
        self.add('<path d="%s" fill="%s" stroke="%s" stroke-width="%s" stroke-linecap="round" '
                 'stroke-linejoin="%s"%s/>' % (d, fill, stroke, f2(width), join, extra))

    def text(self, x, y, s, size=13, color=INK, weight=400, anchor="start", mono=False,
             opacity=1.0, italic=False):
        extra = ' opacity="%s"' % f2(max(0.0, opacity)) if opacity < 0.999 else ""
        if italic:
            extra += ' font-style="italic"'
        self.add('<text x="%s" y="%s" font-size="%s" fill="%s" font-weight="%d" '
                 'text-anchor="%s" font-family="%s" xml:space="preserve"%s>%s</text>'
                 % (f2(x), f2(y), f2(size), color, weight, anchor,
                    MONO_FONT if mono else SANS, extra, esc(s)))

    def svg(self):
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" '
                'height="%d"><defs>%s</defs><rect width="100%%" height="100%%" fill="%s"/>%s</svg>'
                % (self.width, self.height, self.width, self.height, "".join(self.defs),
                   self.background, "".join(self.parts)))


# ------------------------------------------------------------- the actors


def robot(p, x, y, body=TEAL, awake=1.0, lit=1.0, label=None, sub=None, look=0.0, bob=0.0):
    """A thread, drawn as a small robot at a desk. `x, y` is the desk centre.

    `awake` opens the eyes (1) or closes them (0). `lit` turns the antenna light
    on. `look` shifts the eyes left or right, so the robot can watch a message it
    has sent.
    """
    y += bob
    # desk
    p.rect(x - 58, y, 116, 8, "#e6ebf0", "none", 3)
    # body
    p.rect(x - 30, y - 44, 60, 44, body, "none", 12)
    p.rect(x - 16, y - 34, 32, 16, mix(body, PAPER, 0.25), "none", 4)
    # neck, head
    p.rect(x - 6, y - 52, 12, 10, mix(body, INK, 0.35), "none", 2)
    p.rect(x - 36, y - 98, 72, 50, body, "none", 14)
    p.rect(x - 28, y - 91, 56, 36, SCREEN, "none", 9)
    # antenna
    p.line(x, y - 98, x, y - 110, mix(body, INK, 0.35), 3)
    if lit > 0.02:
        p.circle(x, y - 113, 10, GLOW, opacity=0.35 * lit, filter="blur")
    p.circle(x, y - 113, 5, mix("#6b7785", GLOW, lit))
    # eyes
    open_h = lerp(1.8, 10.0, clamp(awake))
    eye = mix("#8391a0", GLOW, clamp(awake))
    for sx in (-1, 1):
        cx = x + sx * 11 + look * 4
        p.rect(cx - 5, y - 73 - open_h / 2, 10, open_h, eye, "none", min(5, open_h / 2))
    if label:
        p.text(x, y + 30, label, 14, INK, 600, "middle")
    if sub:
        p.text(x, y + 47, sub, 11.5, MUTED, 400, "middle", mono=True)


def zzz(p, x, y, t, strength):
    """Three letters drifting up from a sleeping robot."""
    if strength <= 0.01:
        return
    t = math.floor(t * 8) / 8.0      # drift at 8 steps a second; it is idle motion
    for i in range(3):
        phase = ((t * 0.55) + i / 3.0) % 1.0
        size = 11 + 6 * phase
        a = math.sin(phase * math.pi) * strength
        p.text(x + 10 * phase + i * 2, y - 34 * phase, "z", size, NIGHT, 700, "middle",
               opacity=a)


def bell(p, x, y, size=1.0, angle=0.0, fill=BRASS, opacity=1.0):
    """The waker, drawn as a hand bell. `x, y` is the centre of its mouth."""
    if opacity <= 0.01:
        return
    s = size
    with p.group(opacity=opacity, rotate=angle, cx=x, cy=y - 14 * s):
        p.circle(x, y + 3 * s, 3.4 * s, mix(fill, INK, 0.3))
        d = ("M %s %s C %s %s %s %s %s %s C %s %s %s %s %s %s Z"
             % (f2(x - 12 * s), f2(y),
                f2(x - 11 * s), f2(y - 20 * s), f2(x - 7 * s), f2(y - 24 * s), f2(x), f2(y - 24 * s),
                f2(x + 7 * s), f2(y - 24 * s), f2(x + 11 * s), f2(y - 20 * s), f2(x + 12 * s), f2(y)))
        p.path(d, fill, mix(fill, INK, 0.35), 1.2 * s)
        p.rect(x - 14 * s, y - 2 * s, 28 * s, 4 * s, mix(fill, INK, 0.2), "none", 2 * s)
        p.circle(x, y - 26 * s, 3 * s, mix(fill, INK, 0.2))
        p.path("M %s %s Q %s %s %s %s" % (f2(x - 6 * s), f2(y - 17 * s), f2(x - 5 * s),
                                          f2(y - 21 * s), f2(x - 1 * s), f2(y - 22 * s)),
               "none", "#f7e3b5", 1.6 * s)


def ring_waves(p, x, y, age, color=BRASS, span=0.9, count=3):
    """Arcs spreading from a ringing bell."""
    if age is None or age > span + 0.4:
        return
    for i in range(count):
        u = (age - i * 0.13) / span
        if 0 <= u <= 1:
            r = 16 + 34 * u
            a = (1 - u) * 0.9
            for side in (-1, 1):
                a0, a1 = (-50, 50) if side > 0 else (130, 230)
                x0 = x + r * math.cos(math.radians(a0))
                y0 = y + r * math.sin(math.radians(a0))
                x1 = x + r * math.cos(math.radians(a1))
                y1 = y + r * math.sin(math.radians(a1))
                p.path("M %s %s A %s %s 0 0 1 %s %s" % (f2(x0), f2(y0), f2(r), f2(r), f2(x1), f2(y1)),
                       "none", color, 2.4, opacity=a)


def pill(p, x, y, label, color=INK, fill=PAPER, size=13, mono=True, opacity=1.0, scale=1.0,
         icon=None, shadow="lift", weight=700):
    """A message in flight, centred on `x, y`."""
    if opacity <= 0.01:
        return
    w = text_width(label, size, mono) + 26 + (26 if icon else 0)
    h = size + 16
    with p.group(opacity=opacity, scale=scale, cx=x, cy=y):
        p.rect(x - w / 2, y - h / 2, w, h, fill, color, h / 2, 1.8, shadow=shadow)
        tx = x - (13 if icon else 0)
        p.text(tx, y + size * 0.36, label, size, color, weight, "middle", mono=mono)
        if icon == "bell":
            bell(p, x + w / 2 - 20, y + 8, 0.55)
    return w


def chip(p, x, y, label, color, fill, size=12.5, mono=True, opacity=1.0, weight=700,
         anchor="middle", pad=10):
    w = text_width(label, size, mono) + 2 * pad
    h = size + 10
    left = x - w / 2 if anchor == "middle" else x
    p.rect(left, y - h / 2, w, h, fill, color, 6, 1.4, opacity=opacity)
    p.text(left + w / 2, y + size * 0.36, label, size, color, weight, "middle", mono=mono,
           opacity=opacity)
    return w


def meter(p, x, y, w, frac, color, label, value):
    """A labelled horizontal gauge, used for CPU use."""
    p.text(x, y - 7, label, 11, MUTED, 600)
    p.text(x + w, y - 7, value, 11, color, 700, "end", mono=True)
    p.rect(x, y, w, 8, "#e6ebf0", "none", 4)
    if frac > 0.004:
        p.rect(x, y, w * clamp(frac), 8, color, "none", 4)


def focus(p, x, y, w, h, strength, color=BRASS, pad=7):
    """A soft ring that tells the eye where to look."""
    if strength <= 0.01:
        return
    p.rect(x - pad, y - pad, w + 2 * pad, h + 2 * pad, "none", color, 10, 2.4,
           opacity=0.85 * strength, dash="6 5")


def bezier(p0, p1, p2, u):
    return ((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
            (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1])


def code_panel(p, x, y, w, title, lines, active, size=11.2, lead=17.5, strike=None,
               tint=TEAL, opacity=1.0, reveal=None):
    """A few lines of code with a highlight bar on the line that is running.

    `active` is a float, so the bar can slide between lines; a negative value
    hides it. `strike` crosses out a line, for the variant with a line removed.
    `reveal` is the last line the reader has been shown: lines below it are
    drawn as faint bars, so the panel keeps its shape without asking the reader
    to take in code the animation has not reached yet.
    """
    h = 30 + len(lines) * lead + 8
    with p.group(opacity=opacity):
        p.rect(x, y, w, h, "#fbfcfd", LINE, 8, 1.2)
        p.text(x + 12, y + 19, title, 11, MUTED, 600)
        if active >= 0:
            by = y + 28 + active * lead
            p.rect(x + 6, by, w - 12, lead, mix(tint, PAPER, 0.84), "none", 4)
            p.rect(x + 6, by, 3, lead, tint, "none", 1.5)
        for i, row in enumerate(lines):
            near = active >= 0 and abs(active - i) < 0.5
            ty = y + 28 + i * lead + lead * 0.7
            if reveal is not None and i > reveal + 0.01 and not near:
                body = row.strip()
                if body and body.strip("{}();,") :
                    indent = text_width(row[:len(row) - len(row.lstrip())], size, True)
                    p.rect(x + 16 + indent, ty - size * 0.62, text_width(body, size, True),
                           size * 0.7, "#eef2f5", "none", 3)
                continue
            color = INK if near else MUTED
            if strike is not None and i == strike:
                color = RUST
            p.text(x + 16, ty, row, size, color, 600 if near else 400, mono=True)
            if strike is not None and i == strike:
                p.line(x + 14, ty - size * 0.33, x + 16 + text_width(row, size, True) + 2,
                       ty - size * 0.33, RUST, 1.6)
    return h


# ------------------------------------------------------- frame furniture


def scoreboard(p, pairs, y=34):
    """`name = value` pairs, right-aligned beside the title."""
    x = W - 26
    for name, value, color in reversed(pairs):
        v = str(value)
        p.text(x, y, v, 14, color, 700, "end", mono=True)
        x -= text_width(v, 14, True)
        label = "%s = " % name
        p.text(x, y, label, 14, MUTED, 400, "end", mono=True)
        x -= text_width(label, 14, True) + 22


def title_block(p, title, sub):
    p.text(26, 34, title, 20, INK, 700)
    if sub:
        p.text(26, 58, sub, 13, MUTED)


def caption(p, tl, t, y, name="caption", kind="kind", width=W - 52, size=16.0, lead=23.0):
    """The narration line. It fades up when it changes, so the eye notices.

    The band behind it is plain for a step, brass for the insight, and rust for
    the failing case, the same code the slide animations use.
    """
    text = tl._at(name, t)
    started, old = tl.changed(name, t)
    # The first caption is shown at once, so the loop opens on a sentence.
    u = 1.0 if started <= 0.0 else clamp((t - started) / 0.3)
    k = tl._at(kind, t)
    fill, edge = {"insight": (BRASS_LT, BRASS), "fail": (RUST_LT, RUST)}.get(k, (STAGE, LINE))
    p.rect(26, y, width, 2 * lead + 26, fill, edge, 10, 1.4)
    indent = 0
    if k in ("insight", "fail"):
        indent = 34
        cx, cy = 50, y + lead + 13
        p.circle(cx, cy, 11, edge)
        if k == "insight":
            p.text(cx, cy + 5.5, "!", 15, PAPER, 700, "middle")
        else:
            p.line(cx - 4, cy - 4, cx + 4, cy + 4, PAPER, 2.4)
            p.line(cx - 4, cy + 4, cx + 4, cy - 4, PAPER, 2.4)
    rows = wrap(text, size, width - 40 - indent)
    a = ease_out(u)
    top = y + 30 if len(rows) > 1 else y + 30 + lead / 2
    for i, row in enumerate(rows[:2]):
        p.text(46 + indent, top + i * lead + (1 - a) * 6, row, size, INK, 500, opacity=a)


def wrap(s, size, width, mono=False):
    out, line = [], ""
    for word in str(s).split():
        candidate = word if not line else line + " " + word
        if not line or text_width(candidate, size, mono) <= width:
            line = candidate
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def progress(p, tl, t, total, y):
    """A rail at the foot: how far through the loop this frame is, by chapter."""
    x0, x1 = 26, W - 26
    # The rail advances in half-second steps rather than on every frame: a bar
    # that creeps every frame makes every frame of the GIF different.
    fill = math.floor(t * 2) / 2.0 / total
    p.rect(x0, y, x1 - x0, 4, "#e6ebf0", "none", 2)
    p.rect(x0, y, (x1 - x0) * clamp(fill), 4, TEAL, "none", 2)
    right = -1e9
    for when, label in tl.chapters:
        x = x0 + (x1 - x0) * when / total
        done = when <= t
        p.circle(x, y + 2, 4.2, TEAL if done else PAPER, TEAL if done else FAINT, 1.4)
        if label and x + 7 > right + 8:
            p.text(x + 7, y + 19, label, 10.5, INK if done else FAINT, 600 if done else 400)
            right = x + 7 + text_width(label, 10.5) + 10


# ------------------------------------------------------------------ render

def _raster(job):
    svg, width, height, palette_bytes = job[:4]
    keep = job[4] if len(job) > 4 else None
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "f.svg")
        dst = os.path.join(tmp, "f.png")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(svg)
        # rsvg-convert is not installed on this machine; ffmpeg is built with
        # librsvg and rasterises the same SVG to PNG at the requested size.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-vf", "scale=%d:%d" % (width, height),
                        "-frames:v", "1", dst], check=True)
        im = Image.open(dst).convert("RGB")
    if keep:
        im.save(keep, "PNG")
        return None
    if palette_bytes is None:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()
    return (im.size, nearest(im, palette_bytes))


def nearest(im, palette_bytes):
    """Map every pixel to its exact nearest palette entry.

    Pillow's own conversion looks colours up through a reduced-precision cache,
    which turns the page white into a near-white grey. Frames hold only a few
    thousand distinct colours, so an exact search over those is cheap.
    """
    import numpy as np

    pixels = np.asarray(im, dtype=np.int32).reshape(-1, 3)
    packed = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    colours, inverse = np.unique(packed, return_inverse=True)
    rgb = np.stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255], axis=1)
    pal = np.asarray(palette_bytes, dtype=np.int32).reshape(-1, 3)
    distance = ((rgb[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    index = distance.argmin(axis=1).astype(np.uint8)
    return index[inverse].tobytes()


def render(name, tl, draw, height, tail=1.8, fade=0.6, fps=FPS, scale=SCALE, only=None,
           extra_colors=()):
    """Sample `tl`, draw every frame, and write `name` into src/figures.

    The last `fade` seconds cross-fade the final picture into the first, so the
    loop does not jump. `only` renders a list of times to PNG files instead,
    which is how a single frame is checked while a builder is being written.
    """
    from PIL import Image

    total = tl.end() + tail
    width, px_h = int(round(W * scale)), int(round(height * scale))

    def frame(t):
        p = Pic(W, height)
        if t > total - fade:
            a = in_out((t - (total - fade)) / fade)
            with p.group(opacity=1 - a):
                draw(p, tl.sample(min(t, total)), total)
            with p.group(opacity=a):
                draw(p, tl.sample(0.0), total)
        else:
            draw(p, tl.sample(t), total)
        return p.svg()

    if only is not None:
        with ProcessPoolExecutor() as pool:
            pngs = list(pool.map(_raster, [(frame(t), width, px_h, None) for t in only]))
        paths = []
        for t, png in zip(only, pngs):
            folder = pathlib.Path(os.environ.get("MOTION_PREVIEW", tempfile.gettempdir()))
            path = folder / ("%s-%05.2f.png" % (name, t))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
            paths.append(path)
        return paths

    count = int(round(total * fps))
    svgs, waits = [], []
    step = 1000.0 / fps
    last = None
    for i in range(count):
        svg = frame(i / fps)
        digest = hashlib.sha1(svg.encode()).digest()
        if digest == last:
            waits[-1] += step
            continue
        last = digest
        svgs.append(svg)
        waits.append(step)

    # Every distinct frame is rasterised once, in parallel, to a PNG. ffmpeg then
    # builds both outputs from those stills: one GIF with a single palette taken
    # from every frame and delta frames, and one MP4. Encoding the GIF in Python
    # was the slow, single-threaded part of the old pipeline.
    stills = tempfile.mkdtemp(prefix="motion-")
    with ProcessPoolExecutor() as pool:
        list(pool.map(_raster, [(s, width, px_h, None, os.path.join(stills, "f%05d.png" % i))
                                for i, s in enumerate(svgs)], chunksize=4))
    # GIF delays are in centiseconds, so round each hold to a multiple of 10 ms.
    waits = [max(20, int(round(w / 10.0)) * 10) for w in waits]
    OUT.mkdir(parents=True, exist_ok=True)
    _gif(stills, waits, OUT / name)
    video = OUT / name.replace(".gif", ".mp4")
    _video(stills, waits, video)
    shutil.rmtree(stills, ignore_errors=True)
    (OUT / name.replace(".gif", ".json")).write_text(json.dumps({
        "duration": round(total, 2),
        "chapters": [[round(when, 2), label] for when, label in tl.chapters],
    }) + "\n")
    print("  %s: %d frames, %.1f s, gif %.1f MB, mp4 %.1f MB" % (
        name, len(svgs), total, (OUT / name).stat().st_size / 1e6, video.stat().st_size / 1e6))
    return OUT / name


def _concat_list(stills, waits):
    listing = os.path.join(stills, "frames.txt")
    names = sorted(f for f in os.listdir(stills) if f.endswith(".png"))
    with open(listing, "w") as fh:
        for f, ms in zip(names, waits):
            fh.write("file '%s'\nduration %.3f\n" % (os.path.join(stills, f), ms / 1000.0))
        fh.write("file '%s'\n" % os.path.join(stills, names[-1]))
    return listing


def _gif(stills, waits, path):
    """One palette from every frame, no dithering, and only the changed
    rectangle stored per frame."""
    listing = _concat_list(stills, waits)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                    "-vf", "split[a][b];[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=1[p];"
                           "[b][p]paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", str(path)], check=True)


def _video(stills, waits, path):
    """The same frames as an MP4, which a reader can pause, scrub, and slow down.

    The frames are held for the same times as in the GIF, through ffmpeg's
    concat demuxer, and resampled to a constant 20 frames a second.
    """
    listing = _concat_list(stills, waits)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-vf",
                    "fps=%d,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" % FPS,
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-tune", "animation",
                    "-movflags", "+faststart", str(path)], check=True)
