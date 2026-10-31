#!/usr/bin/env python3
"""Draw the cover and the chapter-opener plates.

The cast lives in a hive. The GPU is a honeycomb of streaming multiprocessors,
its threads are bees that fly in warps of 32, and the Rust chapters are worked
by a crew of crabs. Each chapter opens with one member of the cast at work on
the chapter's subject. The SVG goes to src/art/, a PNG beside it (for the PDF),
and plates.json holds the descriptions the chapters use as alternative text.

    python3 tools/art.py              # every plate
    python3 tools/art.py cover ch13   # some of them
"""

import json
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "src" / "art"
W, H = 960, 420
GROUND = 322

INK, SNOW, CREAM = "#1b2430", "#ffffff", "#f6efe0"
RUST, BRASS, TEAL, STEEL, WOOD = "#d9622b", "#c89b3c", "#2f7f86", "#5b6b7c", "#8a6a44"
HONEY, HONEY_DARK, NIGHT = "#f5c542", "#c98f12", "#12263d"
SANS = "Avenir Next Condensed, Avenir Next, Helvetica Neue, Arial, sans-serif"
MONO = "JetBrains Mono, SF Mono, Menlo, DejaVu Sans Mono, monospace"
SERIF = "Georgia, Iowan Old Style, Times New Roman, serif"

PLATES = {}
_ids = [0]


def uid(prefix):
    _ids[0] += 1
    return f"{prefix}{_ids[0]}"


def plate(name, title, crew, description, size=(W, H)):
    """Registers a function that returns the parts of one plate."""

    def register(draw):
        PLATES[name] = (title, crew, description, draw, size)
        return draw

    return register


# ------------------------------------------------------------------ depth
# Characters are shaded with radial gradients, lit from the top left, so they look rounded.
# Gradients stay vectors in a PDF; blur filters would be rasterised, so none are used.
_SHADES = set()


def _rgb(c):
    c = c.lstrip("#")
    return [int(c[i:i + 2], 16) for i in (0, 2, 4)]


def _mix(c, target, k):
    a, b = _rgb(c), _rgb(target)
    return "#" + "".join(f"{round(x + (y - x) * k):02x}" for x, y in zip(a, b))


def make_3d(body):
    """Replaces each mid-tone fill in `body` with a gradient of the same colour."""
    def swap(m):
        c = m.group(1).lower()
        r, g, b = _rgb(c)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        if luma < 70 or luma > 232:
            return m.group(0)
        _SHADES.add(c)
        return f'fill="url(#d3-{c[1:]})"'
    return re.sub(r'fill="(#[0-9a-fA-F]{6})"', swap, body)


def defs_3d():
    grads = [f'<radialGradient id="d3-{c[1:]}" cx="0.36" cy="0.30" r="0.78">'
             f'<stop offset="0" stop-color="{_mix(c, "#ffffff", 0.42)}"/><stop offset="0.5" stop-color="{c}"/>'
             f'<stop offset="1" stop-color="{_mix(c, "#000000", 0.32)}"/></radialGradient>' for c in sorted(_SHADES)]
    grads.append('<radialGradient id="d3-shadow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#0d1a28" stop-opacity="0.38"/>'
                 '<stop offset="1" stop-color="#0d1a28" stop-opacity="0"/></radialGradient>')
    return "<defs>" + "".join(grads) + "</defs>"


# ---------------------------------------------------------------- primitives

def rect(x, y, w, h, fill, rx=0, stroke="none", sw=0, extra=""):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"'
            f' stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def ellipse(cx, cy, rx, ry, fill, extra=""):
    return f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" {extra}/>'


def circle(cx, cy, r, fill, extra=""):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" {extra}/>'


def path(d, fill="none", stroke="none", sw=0, extra=""):
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"'
            f' stroke-linejoin="round" {extra}/>')


def line(x1, y1, x2, y2, stroke, sw=3, extra=""):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"'
            f' stroke-linecap="round" {extra}/>')


def textw(body, size, font=SANS):
    """The width a label is drawn at. Fonts differ between machines, so every label that sits
    in a box is forced to this width with textLength: it can never spill out of its box."""
    if font == MONO:
        ratio = 0.60
    elif body.upper() == body:
        ratio = 0.70
    else:
        ratio = 0.56
    return len(body) * size * ratio


def label(x, y, body, size=20, fill=SNOW, anchor="middle", font=SANS, weight=700, extra="", fit=None):
    """A line of text. `fit` forces its width in units, so it lands the same in every font."""
    if fit:
        extra += f' textLength="{fit:.1f}" lengthAdjust="spacingAndGlyphs"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}"'
            f' fill="{fill}" text-anchor="{anchor}" {extra}>{escape(body)}</text>')


def tag(cx, cy, body, size=17, fill="#ffffffdd", ink="#27405a", font=SANS, weight=700, pad=16, rx=None):
    """A label in a rounded box sized to the text, centred on (cx, cy)."""
    w = textw(body, size, font)
    h = size * 1.8
    return (rect(cx - w / 2 - pad, cy - h / 2, w + 2 * pad, h, fill, h / 2 if rx is None else rx)
            + label(cx, cy + size * 0.36, body, size, ink, font=font, weight=weight, fit=w))


def rot(angle, cx, cy):
    return f'transform="rotate({angle} {cx:.1f} {cy:.1f})"'


def shadow(cx, base, rx):
    return ellipse(cx, base + 3, rx * 1.15, rx * 0.17, "url(#d3-shadow)")


def eye(cx, cy, r, look=0.0, pupil=0.52):
    return "\n".join([circle(cx, cy, r, SNOW), circle(cx + look * r * 0.32, cy + r * 0.06, r * pupil, INK),
                      circle(cx + look * r * 0.32 + r * 0.18, cy - r * 0.16, r * 0.16, SNOW)])


def hexagon(cx, cy, r, fill, stroke="none", sw=0, extra=""):
    pts = " ".join(f"{cx + r * math.cos(math.radians(60 * k + 30)):.1f},{cy + r * math.sin(math.radians(60 * k + 30)):.1f}"
                   for k in range(6))
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def sparkle(cx, cy, r, fill=SNOW, opacity=0.9):
    return path(f"M{cx},{cy - r} Q{cx},{cy} {cx + r},{cy} Q{cx},{cy} {cx},{cy + r} Q{cx},{cy} {cx - r},{cy} Q{cx},{cy} {cx},{cy - r} Z",
                fill, extra=f'opacity="{opacity}"')


def cell(x, y, text, fill="#e8c877", ink=NIGHT, size=34, font_size=17):
    """A numbered square: one element of an array."""
    return rect(x - size / 2, y - size / 2, size, size, fill, 6) + label(x, y + font_size * 0.36, text, font_size, ink, font=MONO)


# ------------------------------------------------------------------ the cast

def bee(cx, cy, s=1.0, look=1.0, face=1, wings=True, hat=None):
    """A worker bee seen from the side, flying, centred on (cx, cy). `face` 1 looks right, -1 left."""
    h = 100 * s
    x = lambda f: cx + face * f * h
    clip = uid("bee")
    body_rx, body_ry = 0.40 * h, 0.27 * h
    parts = []

    if wings:
        parts += [ellipse(x(-0.06), cy - 0.36 * h, 0.17 * h, 0.27 * h, "#dff3ff", f'opacity="0.78" {rot(-face * 22, x(-0.06), cy - 0.36 * h)}'),
                  ellipse(x(0.12), cy - 0.34 * h, 0.14 * h, 0.23 * h, "#eef9ff", f'opacity="0.85" {rot(face * 16, x(0.12), cy - 0.34 * h)}')]

    parts += [
        path(f"M{x(-0.38):.1f},{cy - 0.06 * h:.1f} L{x(-0.56):.1f},{cy + 0.02 * h:.1f} L{x(-0.38):.1f},{cy + 0.08 * h:.1f} Z", INK),
        f'<clipPath id="{clip}"><ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{body_rx:.1f}" ry="{body_ry:.1f}"/></clipPath>',
        ellipse(cx, cy, body_rx, body_ry, HONEY),
        f'<g clip-path="url(#{clip})">' + "".join(rect(x(f) - 0.05 * h, cy - h, 0.10 * h, 2 * h, INK) for f in (-0.22, -0.02)) + "</g>",
        ellipse(cx, cy, body_rx, body_ry, "none", f'stroke="{HONEY_DARK}" stroke-width="{1.2 * s + 0.6:.1f}"'),
        circle(x(0.42), cy - 0.04 * h, 0.21 * h, INK),
        path(f"M{x(0.46):.1f},{cy - 0.22 * h:.1f} Q{x(0.52):.1f},{cy - 0.48 * h:.1f} {x(0.66):.1f},{cy - 0.50 * h:.1f}", stroke=INK, sw=2.2 * s + 0.6),
        path(f"M{x(0.38):.1f},{cy - 0.22 * h:.1f} Q{x(0.40):.1f},{cy - 0.50 * h:.1f} {x(0.30):.1f},{cy - 0.58 * h:.1f}", stroke=INK, sw=2.2 * s + 0.6),
        circle(x(0.66), cy - 0.50 * h, 0.04 * h, HONEY), circle(x(0.30), cy - 0.58 * h, 0.04 * h, HONEY),
        eye(x(0.48), cy - 0.07 * h, 0.085 * h, face * look),
        path(f"M{x(0.44):.1f},{cy + 0.07 * h:.1f} q{face * 0.05 * h:.1f},{0.04 * h:.1f} {face * 0.10 * h:.1f},0", stroke="#f0a35a", sw=1.6 * s + 0.6),
    ]

    if hat == "hard":
        hx, hy = x(0.42), cy - 0.20 * h
        parts += [path(f"M{hx - 0.20 * h:.1f},{hy:.1f} A{0.20 * h:.1f},{0.15 * h:.1f} 0 0 1 {hx + 0.20 * h:.1f},{hy:.1f} Z", BRASS),
                  rect(hx - 0.25 * h, hy - 0.02 * h, 0.50 * h, 0.04 * h, "#a57f2c", 3)]

    return "\n".join(parts)


def ant(cx, base=GROUND, s=1.0, face=1, look=1.0, carry=None):
    """An ant walking, facing right (`face` 1) or left. `carry` is the colour of a crumb on its back."""
    h = 70 * s
    x = lambda f: cx + face * f * h
    coat, leg = "#7a3b2e", "#4f241b"
    legs = [line(x(dx), base - 0.32 * h, x(dx + 0.14 * k), base, leg, 2.4 * s + 0.8) for dx in (-0.10, 0.06, 0.20) for k in (-1, 1)]
    parts = [shadow(cx, base, 0.62 * h), *legs,
             ellipse(x(-0.42), base - 0.38 * h, 0.30 * h, 0.22 * h, coat),
             ellipse(x(0.06), base - 0.36 * h, 0.17 * h, 0.12 * h, coat),
             circle(x(0.40), base - 0.46 * h, 0.19 * h, coat),
             path(f"M{x(0.46):.1f},{base - 0.62 * h:.1f} Q{x(0.56):.1f},{base - 0.92 * h:.1f} {x(0.74):.1f},{base - 0.90 * h:.1f}", stroke=leg, sw=2.2 * s + 0.6),
             path(f"M{x(0.36):.1f},{base - 0.62 * h:.1f} Q{x(0.40):.1f},{base - 0.96 * h:.1f} {x(0.56):.1f},{base - 1.02 * h:.1f}", stroke=leg, sw=2.2 * s + 0.6),
             eye(x(0.46), base - 0.50 * h, 0.075 * h, face * look)]

    if carry:
        parts += [circle(x(-0.30), base - 0.82 * h, 0.22 * h, carry, f'stroke="{WOOD}" stroke-width="1.5"')]

    return "\n".join(parts)


def crab(cx, base=GROUND, s=1.0, mood="happy", look=0.0, shade=True, hat=None):
    """A crab in the orange of Rust, in one of 5 moods: happy, cute, angry, sleepy or shocked."""
    h, coat, dark = 120 * s, "#f46a35", "#c2441a"
    y = lambda f: base - f * h
    up = 0.98 if mood in ("happy", "angry", "shocked") else 0.70
    parts = [shadow(cx, base, 0.62 * h)] if shade else []
    parts += [line(cx + side * (0.30 + 0.12 * k) * h, y(0.24), cx + side * (0.44 + 0.16 * k) * h, y(0.0), dark, 5 * s + 1.5)
              for side in (-1, 1) for k in range(3)]

    for side in (-1, 1):
        tip = (cx + side * 0.80 * h, y(up))
        parts += [path(f"M{cx + side * 0.46 * h:.1f},{y(0.46):.1f} Q{cx + side * 0.80 * h:.1f},{y(0.44):.1f} {tip[0]:.1f},{tip[1] + 0.10 * h:.1f}", stroke=coat, sw=0.09 * h),
                  circle(tip[0], tip[1], 0.15 * h, coat),
                  path(f"M{tip[0]:.1f},{tip[1]:.1f} l{side * 0.02 * h:.1f},{-0.19 * h:.1f} l{-side * 0.13 * h:.1f},{0.08 * h:.1f} Z", "#fff1e6")]

    parts += [
        path(f"M{cx - 0.56 * h:.1f},{y(0.40):.1f} L{cx - 0.50 * h:.1f},{y(0.70):.1f} L{cx - 0.34 * h:.1f},{y(0.66):.1f} L{cx - 0.22 * h:.1f},{y(0.82):.1f}"
             f" L{cx:.1f},{y(0.74):.1f} L{cx + 0.22 * h:.1f},{y(0.82):.1f} L{cx + 0.34 * h:.1f},{y(0.66):.1f} L{cx + 0.50 * h:.1f},{y(0.70):.1f} L{cx + 0.56 * h:.1f},{y(0.40):.1f} Z", coat),
        ellipse(cx, y(0.38), 0.58 * h, 0.30 * h, coat),
        ellipse(cx, y(0.24), 0.36 * h, 0.12 * h, "#ffb08a", 'opacity="0.55"'),
    ]

    for side in (-1, 1):
        ex, ey = cx + side * 0.17 * h, y(0.50)

        if mood == "sleepy":
            parts.append(path(f"M{ex - 0.08 * h:.1f},{ey:.1f} q{0.08 * h:.1f},{0.07 * h:.1f} {0.16 * h:.1f},0", stroke=INK, sw=3 * s + 1.5))
            continue

        parts.append(ellipse(ex, ey, 0.10 * h, 0.13 * h, SNOW))
        size = {"cute": 0.085, "shocked": 0.035}.get(mood, 0.055) * h
        parts += [circle(ex + look * 0.03 * h, ey + 0.01 * h, size, INK), circle(ex + look * 0.03 * h + size * 0.35, ey - size * 0.3, size * 0.32, SNOW)]

        if mood == "cute":
            parts += [circle(ex + look * 0.03 * h - size * 0.3, ey + size * 0.4, size * 0.18, SNOW), ellipse(ex + side * 0.16 * h, y(0.34), 0.07 * h, 0.04 * h, "#ff9aa8", 'opacity="0.8"')]

        if mood == "angry":
            parts.append(line(ex - side * 0.12 * h, y(0.62), ex + side * 0.10 * h, y(0.70), INK, 4 * s + 1.5))

    parts.append({
        "happy": path(f"M{cx - 0.12 * h:.1f},{y(0.30):.1f} Q{cx:.1f},{y(0.12):.1f} {cx + 0.12 * h:.1f},{y(0.30):.1f} Z", "#7a1f0a"),
        "cute": path(f"M{cx - 0.07 * h:.1f},{y(0.30):.1f} q{0.035 * h:.1f},{0.05 * h:.1f} {0.07 * h:.1f},0 q{0.035 * h:.1f},{0.05 * h:.1f} {0.07 * h:.1f},0", stroke="#7a1f0a", sw=2.5 * s + 1),
        "angry": path(f"M{cx - 0.11 * h:.1f},{y(0.22):.1f} Q{cx:.1f},{y(0.34):.1f} {cx + 0.11 * h:.1f},{y(0.22):.1f}", stroke="#7a1f0a", sw=3 * s + 1.5),
        "sleepy": ellipse(cx, y(0.27), 0.035 * h, 0.03 * h, "#7a1f0a"),
        "shocked": ellipse(cx, y(0.25), 0.06 * h, 0.08 * h, "#7a1f0a"),
    }[mood])

    if mood == "sleepy":
        parts += [label(cx + (0.46 + 0.14 * k) * h, y(0.86 + 0.16 * k), "z", (14 + 6 * k) * s + 4, "#cfe1f0", weight=800) for k in range(3)]

    if hat == "hard":
        parts += [path(f"M{cx - 0.30 * h:.1f},{y(0.80):.1f} A{0.30 * h:.1f},{0.22 * h:.1f} 0 0 1 {cx + 0.30 * h:.1f},{y(0.80):.1f} Z", BRASS),
                  rect(cx - 0.36 * h, y(0.82), 0.72 * h, 0.05 * h, "#a57f2c", 3)]

    return make_3d("\n".join(parts))


def honeybee(cx, base, s=1.0, look=0.0, role="worker", arms="down", hat=None, carry=None, fur="#ffd23f"):
    """A bee facing the reader, standing (or hovering) with its feet at `base`. 1.0 is 200 units tall.

    `role` is "worker" or "queen". The queen is longer, wears a crown and a sash, and holds a sceptre.
    `arms` is "down", "up" or "hold"; `carry` puts a honey drop in a worker's hands.
    """
    h = 200 * s
    y = lambda f: base - f * h
    queen = role == "queen"
    body_y, body_rx, body_ry = (0.34, 0.25, 0.32) if queen else (0.30, 0.24, 0.25)
    clip = uid("hb")
    dark, line_w = "#2a1d10", 2.2 * s + 1.2
    parts = [shadow(cx, base, 0.32 * h)]

    for side in (-1, 1):
        wx, wy = cx + side * 0.30 * h, y(0.66 if not queen else 0.70)
        parts += [ellipse(wx, wy, 0.22 * h, 0.13 * h, "#e3f4ff", f'opacity="0.82" stroke="#a9cfe8" stroke-width="{line_w:.1f}" {rot(side * -32, wx, wy)}'),
                  ellipse(cx + side * 0.25 * h, y(0.50), 0.14 * h, 0.08 * h, "#eef9ff", f'opacity="0.75" stroke="#a9cfe8" stroke-width="{line_w:.1f}" {rot(side * -12, cx + side * 0.25 * h, y(0.50))}')]

    parts += [ellipse(cx - 0.09 * h, y(0.02), 0.07 * h, 0.035 * h, dark), ellipse(cx + 0.09 * h, y(0.02), 0.07 * h, 0.035 * h, dark),
              line(cx - 0.08 * h, y(0.05), cx - 0.09 * h, y(0.10), dark, line_w * 1.4), line(cx + 0.08 * h, y(0.05), cx + 0.09 * h, y(0.10), dark, line_w * 1.4)]

    by = y(body_y)
    stripes = [rect(cx - h, by + f * h - 0.04 * h, 2 * h, 0.08 * h, dark) for f in ((-0.04, 0.12) if not queen else (-0.08, 0.06, 0.20))]
    parts += [f'<clipPath id="{clip}"><ellipse cx="{cx:.1f}" cy="{by:.1f}" rx="{body_rx * h:.1f}" ry="{body_ry * h:.1f}"/></clipPath>',
              ellipse(cx, by, body_rx * h, body_ry * h, fur),
              f'<g clip-path="url(#{clip})">' + "".join(stripes) + ellipse(cx - 0.08 * h, by - 0.10 * h, 0.06 * h, 0.10 * h, SNOW, 'opacity="0.25"') + "</g>",
              ellipse(cx, by, body_rx * h, body_ry * h, "none", f'stroke="{dark}" stroke-width="{line_w:.1f}"')]

    if queen:
        parts.append(path(f"M{cx - 0.22 * h:.1f},{y(0.52):.1f} L{cx + 0.20 * h:.1f},{y(0.16):.1f} L{cx + 0.24 * h:.1f},{y(0.24):.1f} L{cx - 0.16 * h:.1f},{y(0.58):.1f} Z", "#8e4fd1", dark, line_w * 0.6))
        parts.append(circle(cx - 0.02 * h, y(0.38), 0.035 * h, BRASS, f'stroke="{dark}" stroke-width="{line_w * 0.6:.1f}"'))

    # Arms, with a round hand at the end of each.
    pose = {"down": ((-0.30, 0.30), (0.30, 0.30)), "up": ((-0.34, 0.66), (0.34, 0.66)), "hold": ((-0.12, 0.34), (0.12, 0.34))}[arms]
    if queen:
        pose = ((-0.32, 0.36), (0.34, 0.50))

    for (hx, hy), side in zip(pose, (-1, 1)):
        sx, sy = cx + side * 0.20 * h, y(0.44)
        ex, ey = cx + hx * h, y(hy)
        parts += [path(f"M{sx:.1f},{sy:.1f} Q{(sx + ex) / 2 + side * 0.05 * h:.1f},{(sy + ey) / 2:.1f} {ex:.1f},{ey:.1f}", stroke=dark, sw=line_w * 1.5),
                  circle(ex, ey, 0.035 * h, dark)]

    if queen:
        tx, ty = cx + 0.34 * h, y(0.50)
        parts += [line(tx, ty + 0.30 * h, tx, ty - 0.34 * h, "#a57f2c", line_w * 1.6),
                  hexagon(tx, ty - 0.40 * h, 0.07 * h, "#ffcf4d", dark, line_w * 0.7), circle(tx, ty - 0.40 * h, 0.025 * h, "#e2704a")]

    if carry:
        dx, dy = cx, y(0.30)
        parts += [path(f"M{dx:.1f},{dy - 0.10 * h:.1f} C{dx + 0.08 * h:.1f},{dy - 0.01 * h:.1f} {dx + 0.06 * h:.1f},{dy + 0.06 * h:.1f} {dx:.1f},{dy + 0.06 * h:.1f}"
                       f" C{dx - 0.06 * h:.1f},{dy + 0.06 * h:.1f} {dx - 0.08 * h:.1f},{dy - 0.01 * h:.1f} {dx:.1f},{dy - 0.10 * h:.1f} Z", carry, dark, line_w * 0.6),
                  ellipse(dx - 0.02 * h, dy - 0.01 * h, 0.012 * h, 0.022 * h, SNOW, 'opacity="0.7"')]

    # Head, face and antennae.
    hx, hy, hr = cx, y(0.72 if not queen else 0.78), 0.20 * h
    for side in (-1, 1):
        tip = (cx + side * 0.17 * h, hy - 0.34 * h)
        parts += [path(f"M{cx + side * 0.06 * h:.1f},{hy - 0.16 * h:.1f} Q{cx + side * 0.06 * h:.1f},{hy - 0.30 * h:.1f} {tip[0]:.1f},{tip[1]:.1f}", stroke=dark, sw=line_w),
                  circle(tip[0], tip[1], 0.035 * h, dark)]

    parts += [circle(hx, hy, hr, fur, f'stroke="{dark}" stroke-width="{line_w:.1f}"'),
              eye(hx - 0.075 * h, hy + 0.01 * h, 0.06 * h, look), eye(hx + 0.075 * h, hy + 0.01 * h, 0.06 * h, look),
              ellipse(hx - 0.13 * h, hy + 0.08 * h, 0.035 * h, 0.022 * h, "#ff8fa3", 'opacity="0.8"'),
              ellipse(hx + 0.13 * h, hy + 0.08 * h, 0.035 * h, 0.022 * h, "#ff8fa3", 'opacity="0.8"'),
              path(f"M{hx - 0.05 * h:.1f},{hy + 0.09 * h:.1f} Q{hx:.1f},{hy + 0.14 * h:.1f} {hx + 0.05 * h:.1f},{hy + 0.09 * h:.1f}", stroke=dark, sw=line_w)]

    if queen:
        top = hy - hr * 0.80
        crown = (f"M{hx - 0.15 * h:.1f},{top:.1f} L{hx - 0.17 * h:.1f},{top - 0.14 * h:.1f} L{hx - 0.08 * h:.1f},{top - 0.06 * h:.1f}"
                 f" L{hx:.1f},{top - 0.17 * h:.1f} L{hx + 0.08 * h:.1f},{top - 0.06 * h:.1f} L{hx + 0.17 * h:.1f},{top - 0.14 * h:.1f} L{hx + 0.15 * h:.1f},{top:.1f} Z")
        parts += [path(crown, "#ffcf4d", dark, line_w),
                  circle(hx, top - 0.05 * h, 0.022 * h, "#e2704a"), circle(hx - 0.10 * h, top - 0.03 * h, 0.016 * h, "#5fc9b4"),
                  circle(hx + 0.10 * h, top - 0.03 * h, 0.016 * h, "#5fc9b4")]

    if hat == "hard":
        top = hy - hr * 0.62
        parts += [path(f"M{hx - 0.17 * h:.1f},{top:.1f} A{0.17 * h:.1f},{0.13 * h:.1f} 0 0 1 {hx + 0.17 * h:.1f},{top:.1f} Z", "#f08a3c", dark, line_w * 0.8),
                  rect(hx - 0.21 * h, top - 0.015 * h, 0.42 * h, 0.035 * h, "#d9622b", 3, dark, line_w * 0.6)]

    return make_3d("\n".join(parts))


def hornet(cx, base, s=1.0, look=0.0, arms="down", brows=True):
    """A hornet facing the reader, feet at `base`; 1.0 is 230 units tall. Amber and dark brown,
    with a narrow waist, large eyes and longer wings than a bee."""
    h = 230 * s
    y = lambda f: base - f * h
    coat, dark, face, line_w = "#d07a32", "#2b1a12", "#e6b25c", 2.2 * s + 1.2
    parts = [shadow(cx, base, 0.26 * h)]
    for side in (-1, 1):
        wx, wy = cx + side * 0.28 * h, y(0.60)
        parts += [ellipse(wx, wy, 0.26 * h, 0.10 * h, "#e6f1f8", f'opacity="0.8" stroke="#9fbcd2" stroke-width="{line_w:.1f}" {rot(side * -38, wx, wy)}'),
                  ellipse(cx + side * 0.24 * h, y(0.46), 0.17 * h, 0.065 * h, "#eef6fb", f'opacity="0.75" stroke="#9fbcd2" stroke-width="{line_w:.1f}" {rot(side * -16, cx + side * 0.24 * h, y(0.46))}')]
    for side in (-1, 1):
        parts += [line(cx + side * 0.06 * h, y(0.10), cx + side * 0.10 * h, y(0.0), dark, line_w * 1.4),
                  ellipse(cx + side * 0.11 * h, y(0.0), 0.06 * h, 0.025 * h, dark)]
    # Abdomen with bands, a narrow waist, then the thorax.
    clip = uid("hn")
    ay = y(0.24)
    parts += [f'<clipPath id="{clip}"><ellipse cx="{cx:.1f}" cy="{ay:.1f}" rx="{0.17 * h:.1f}" ry="{0.20 * h:.1f}"/></clipPath>',
              ellipse(cx, ay, 0.17 * h, 0.20 * h, coat),
              f'<g clip-path="url(#{clip})">' + "".join(rect(cx - h, ay + f * h, 2 * h, 0.055 * h, dark) for f in (-0.10, 0.0, 0.10)) + "</g>",
              ellipse(cx, ay, 0.17 * h, 0.20 * h, "none", f'stroke="{dark}" stroke-width="{line_w:.1f}"'),
              ellipse(cx, y(0.46), 0.05 * h, 0.04 * h, dark),
              ellipse(cx, y(0.54), 0.13 * h, 0.09 * h, dark),
              ellipse(cx - 0.04 * h, y(0.57), 0.04 * h, 0.025 * h, SNOW, 'opacity="0.2"')]
    pose = {"down": ((-0.24, 0.30), (0.24, 0.30)), "up": ((-0.30, 0.74), (0.30, 0.74)), "fold": ((-0.10, 0.44), (0.10, 0.44))}[arms]
    for (hx, hy), side in zip(pose, (-1, 1)):
        sx, sy = cx + side * 0.10 * h, y(0.54)
        ex, ey = cx + hx * h, y(hy)
        parts += [path(f"M{sx:.1f},{sy:.1f} Q{(sx + ex) / 2 + side * 0.05 * h:.1f},{(sy + ey) / 2:.1f} {ex:.1f},{ey:.1f}", stroke=dark, sw=line_w * 1.4),
                  circle(ex, ey, 0.03 * h, dark)]
    hx, hy, hr = cx, y(0.74), 0.15 * h
    for side in (-1, 1):
        tip = (cx + side * 0.15 * h, hy - 0.30 * h)
        parts += [path(f"M{cx + side * 0.05 * h:.1f},{hy - 0.12 * h:.1f} Q{cx + side * 0.04 * h:.1f},{hy - 0.26 * h:.1f} {tip[0]:.1f},{tip[1]:.1f}", stroke=dark, sw=line_w),
                  circle(tip[0], tip[1], 0.025 * h, dark)]
    parts += [ellipse(hx, hy, hr, hr * 1.08, face, f'stroke="{dark}" stroke-width="{line_w:.1f}"'),
              ellipse(hx, hy - hr * 0.55, hr * 0.9, hr * 0.45, coat, f'stroke="{dark}" stroke-width="{line_w * 0.8:.1f}"')]
    for side in (-1, 1):
        ex, ey = hx + side * 0.075 * h, hy + 0.0 * h
        parts += [ellipse(ex, ey, 0.05 * h, 0.065 * h, dark),
                  circle(ex + look * 0.012 * h - side * 0.012 * h, ey - 0.025 * h, 0.014 * h, SNOW)]
        if brows:
            parts.append(line(ex - side * 0.055 * h, ey - 0.085 * h, ex + side * 0.035 * h, ey - 0.065 * h, dark, line_w * 1.2))
    parts += [path(f"M{hx - 0.04 * h:.1f},{hy + 0.09 * h:.1f} Q{hx:.1f},{hy + 0.12 * h:.1f} {hx + 0.04 * h:.1f},{hy + 0.09 * h:.1f}", stroke=dark, sw=line_w),
              line(hx - 0.03 * h, hy + 0.13 * h, hx - 0.02 * h, hy + 0.16 * h, dark, line_w),
              line(hx + 0.03 * h, hy + 0.13 * h, hx + 0.02 * h, hy + 0.16 * h, dark, line_w)]
    return make_3d("\n".join(parts))


# ----------------------------------------------------------------- background

def backdrop(sky=("#12263d", "#2c5c86"), ground=("#5f9a62", "#86bd7c"), seed=0, size=(W, H), stars=True):
    """A sky with stars and a band of light, over ground of 2 tones."""
    width, height = size
    base = height - (H - GROUND)
    sid, gid = uid("sky"), uid("glow")
    dots = [circle((k * 197 + seed * 53) % width, (k * 89 + seed * 31) % (base - 130) + 12, 1.1 + (k % 3) * 0.6, SNOW,
                   f'opacity="{0.35 + (k % 4) * 0.15:.2f}"') for k in range(46 if stars else 0)]

    return "\n".join([
        f'<defs><linearGradient id="{sid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{sky[0]}"/>'
        f'<stop offset="1" stop-color="{sky[1]}"/></linearGradient>'
        f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#f5c542" stop-opacity="0"/>'
        '<stop offset="0.5" stop-color="#f5c542" stop-opacity="0.22"/><stop offset="1" stop-color="#7aa6ff" stop-opacity="0"/>'
        '</linearGradient></defs>',
        rect(0, 0, width, height, f"url(#{sid})"),
        path(f"M0,{120 + seed % 3 * 12} C240,{40 + seed % 5 * 8} 520,{190 - seed % 4 * 10} {width},{80 + seed % 3 * 14} L{width},{150 + seed % 3 * 14}"
             f" C520,{250 - seed % 4 * 10} 240,{110 + seed % 5 * 8} 0,{190 + seed % 3 * 12} Z", f"url(#{gid})"),
        *dots,
        path(f"M0,{base - 26} C200,{base - 52} 420,{base - 8} 640,{base - 34} S900,{base - 20} {width},{base - 40} L{width},{height} L0,{height} Z", ground[0]),
        path(f"M0,{base - 6} C260,{base - 26} 520,{base + 8} {width},{base - 14} L{width},{height} L0,{height} Z", ground[1]),
    ])


MEADOW = dict(sky=("#17324a", "#3f7f8f"), ground=("#5f9a62", "#86bd7c"))
SAND = dict(sky=("#1d2f4d", "#b7684a"), ground=("#d6b47c", "#ecd4a4"))
DUSK = dict(sky=("#1b2540", "#6a4a7a"), ground=("#a58e6a", "#c9b48e"))
SHORE = dict(sky=("#102a44", "#2f6f9a"), ground=("#d9c08e", "#efdcb2"))


def ribbon(number, title, crew, size=(W, H)):
    """The plate's caption: the chapter and the name of the character."""
    height = size[1]
    name = textw(crew, 17)
    room = size[0] - 60 - name - 12 - 140      # space left of the name tag, after the box padding
    font = min(21, 21 * room / textw(title.upper(), 21))
    words = textw(title.upper(), font)

    return "\n".join([
        rect(28, height - 60, 72, 40, RUST, 6), label(64, height - 32, number, 22, fit=textw(number, 22)),
        rect(100, height - 60, words + 40, 40, NIGHT, 6),
        label(120, height - 33, title.upper(), font, anchor="start", fit=words),
        rect(size[0] - 60 - name, height - 56, name + 32, 32, "#ffffffdd", 16),
        label(size[0] - 44 - name / 2, height - 34, crew, 17, "#27405a", weight=600, fit=name),
    ])


def flower(cx, base, s=1.0, petal="#f7a8c4"):
    return "\n".join([line(cx, base, cx, base - 34 * s, "#3f7a4a", 3),
                      *[circle(cx + 8 * s * math.cos(k * 1.2566), base - 34 * s + 8 * s * math.sin(k * 1.2566), 6 * s, petal) for k in range(5)],
                      circle(cx, base - 34 * s, 5 * s, HONEY)])


def pelican(cx, base, s=1.0, look=1):
    """A standing pelican facing `look` (1 right, -1 left), its feet at `base`. 1.0 is 220 units tall."""
    f = look
    X = lambda dx: cx + f * dx * s
    Y = lambda dy: base - dy * s
    white, grey, beak, pouch, ink = "#f6f3ec", "#c9ccd3", "#f2a03d", "#f7c66b", "#2a1d10"
    return "".join([
        shadow(cx, base, 60 * s),
        line(X(-12), Y(0), X(-12), Y(40), beak, 6 * s), line(X(12), Y(0), X(12), Y(40), beak, 6 * s),
        ellipse(cx, Y(80), 62 * s, 46 * s, white, f'stroke="{ink}" stroke-width="{2.5 * s}"'),
        path(f"M{X(-50)},{Y(92)} Q{X(-10)},{Y(130)} {X(40)},{Y(84)} Q{X(0)},{Y(70)} {X(-50)},{Y(92)} Z", grey, ink, 2.5 * s),
        path(f"M{X(30)},{Y(110)} Q{X(40)},{Y(150)} {X(28)},{Y(176)}", "none", ink, 22 * s),
        path(f"M{X(30)},{Y(110)} Q{X(40)},{Y(150)} {X(28)},{Y(176)}", "none", white, 17 * s),
        circle(X(30), Y(184), 22 * s, white, f'stroke="{ink}" stroke-width="{2.5 * s}"'),
        path(f"M{X(46)},{Y(188)} L{X(132)},{Y(176)} L{X(46)},{Y(176)} Z", beak, ink, 2.2 * s),
        path(f"M{X(48)},{Y(176)} Q{X(96)},{Y(128)} {X(126)},{Y(176)} Z", pouch, ink, 2.2 * s),
        eye(X(34), Y(190), 6 * s, look * 0.6),
    ])


def meerkat(cx, base, s=1.0, look=1):
    """A meerkat standing upright on its hind legs, facing `look`, with a whistle. 1.0 is 200 units tall."""
    f = look
    X = lambda dx: cx + f * dx * s
    Y = lambda dy: base - dy * s
    fur, dark, belly, ink = "#d6ad74", "#6b4a2b", "#f2dfb8", "#2a1d10"
    stroke = f'stroke="{ink}" stroke-width="{2.4 * s}"'
    return "".join([
        shadow(cx, base, 46 * s),
        path(f"M{X(-22)},{Y(18)} Q{X(-60)},{Y(10)} {X(-70)},{Y(-2)}", "none", ink, 9 * s),
        path(f"M{X(-22)},{Y(18)} Q{X(-60)},{Y(10)} {X(-70)},{Y(-2)}", "none", fur, 5.5 * s),
        ellipse(X(-12), Y(4), 16 * s, 6 * s, fur, stroke), ellipse(X(14), Y(4), 16 * s, 6 * s, fur, stroke),
        ellipse(cx, Y(68), 34 * s, 64 * s, fur, stroke),
        ellipse(X(8), Y(64), 19 * s, 46 * s, belly),
        path(f"M{X(14)},{Y(104)} q{f * 18 * s},{4 * s} {f * 20 * s},{-14 * s}", "none", ink, 7 * s),
        path(f"M{X(14)},{Y(104)} q{f * 18 * s},{4 * s} {f * 20 * s},{-14 * s}", "none", fur, 4 * s),
        ellipse(X(6), Y(150), 27 * s, 23 * s, fur, stroke),
        ellipse(X(30), Y(144), 15 * s, 10 * s, fur, stroke),
        circle(X(44), Y(147), 4 * s, ink),
        circle(X(-14), Y(166), 6.5 * s, dark, stroke),
        ellipse(X(12), Y(155), 9 * s, 7 * s, dark),
        eye(X(13), Y(155), 4.2 * s, look * 0.6),
        rect(min(X(40), X(58)), Y(138), 18 * s, 8 * s, "#c9d3de", 3 * s, ink, 1.6 * s),
        line(X(30), Y(136), X(10), Y(116), "#c94a3b", 2 * s),
    ])


def otter(cx, cy, s=1.0, look=1, carry=True):
    """An otter floating on its back, head towards `look`, a crate on its belly. 1.0 is 170 units long."""
    f = look
    X = lambda dx: cx + f * dx * s
    Y = lambda dy: cy - dy * s
    coat, belly, ink = "#7b5233", "#d8b48a", "#2a1d10"
    stroke = f'stroke="{ink}" stroke-width="{2.2 * s}"'
    return "".join([
        path(f"M{X(-60)},{Y(0)} Q{X(-100)},{Y(4)} {X(-112)},{Y(18)}", "none", ink, 13 * s),
        path(f"M{X(-60)},{Y(0)} Q{X(-100)},{Y(4)} {X(-112)},{Y(18)}", "none", coat, 9 * s),
        ellipse(cx, Y(0), 68 * s, 22 * s, coat, stroke),
        ellipse(X(4), Y(6), 48 * s, 12 * s, belly),
        circle(X(68), Y(8), 21 * s, coat, stroke),
        ellipse(X(80), Y(4), 11 * s, 8 * s, belly),
        circle(X(89), Y(6), 3.5 * s, ink),
        circle(X(58), Y(26), 5 * s, coat, stroke),
        eye(X(72), Y(14), 3.6 * s, look * 0.5),
        line(X(84), Y(2), X(100), Y(-2), "#3a2a1e", 1.2 * s), line(X(84), Y(4), X(101), Y(6), "#3a2a1e", 1.2 * s),
        plain_crate(X(-18) - (16 * s if f > 0 else 0), Y(48), 36 * s, 30 * s) if carry else "",
        ellipse(X(30), Y(22), 7 * s, 5 * s, coat, stroke), ellipse(X(-30), Y(22), 7 * s, 5 * s, coat, stroke),
    ])


def caterpillar(x0, base, segments=8, r=18, look=1):
    """A caterpillar walking to the right along `base`: `segments` body circles and a head."""
    body, belly, ink = "#7cc06a", "#b9e3a2", "#24401c"
    stroke = f'stroke="{ink}" stroke-width="2.2"'
    parts = []
    for k in range(segments):
        cx = x0 + k * 1.6 * r
        cy = base - r - (4 if k % 2 else 0)
        parts += [line(cx - 4, cy + r - 4, cx - 6, base, ink, 3), line(cx + 4, cy + r - 4, cx + 6, base, ink, 3),
                  circle(cx, cy, r, body, stroke), circle(cx, cy + 4, r * 0.55, belly)]
    hx, hy = x0 + segments * 1.6 * r + 4, base - 1.6 * r
    parts += [line(hx - 4, hy - r, hx - 12, hy - 2 * r, ink, 2.5), line(hx + 6, hy - r, hx + 12, hy - 2 * r, ink, 2.5),
              circle(hx - 12, hy - 2 * r, 4, "#f5c542"), circle(hx + 12, hy - 2 * r, 4, "#f5c542"),
              circle(hx, hy, r * 1.15, body, stroke), eye(hx + 7, hy - 4, 5, look * 0.6),
              path(f"M{hx + 4},{hy + 9} q6,5 12,0", "none", ink, 2.2)]
    return "".join(parts)


def acorn(cx, cy, r=12):
    """An acorn: a brown nut under a textured cap with a stalk, centred on (cx, cy)."""
    ink = "#3a2410"
    return "".join([ellipse(cx, cy + r * 0.25, r * 0.82, r, "#b8732e", f'stroke="{ink}" stroke-width="1.8"'),
                    ellipse(cx - r * 0.25, cy + r * 0.1, r * 0.2, r * 0.45, "#d99a52"),
                    path(f"M{cx - r},{cy - r * 0.35} Q{cx},{cy - r * 1.3} {cx + r},{cy - r * 0.35} Z", "#7a5a32", ink, 1.8),
                    line(cx, cy - r * 0.95, cx + r * 0.25, cy - r * 1.4, ink, 2.2)])


def squirrel(cx, base, s=1.0, look=1):
    """A red squirrel sitting up, facing `look`, holding an acorn. 1.0 is 200 units tall."""
    f = look
    X = lambda dx: cx + f * dx * s
    Y = lambda dy: base - dy * s
    fur, dark, belly, ink = "#c8642a", "#8a3f17", "#f3d9b4", "#2a1d10"
    stroke = f'stroke="{ink}" stroke-width="{2.4 * s}"'
    return "".join([
        shadow(cx, base, 60 * s),
        path(f"M{X(-30)},{Y(20)} C{X(-110)},{Y(10)} {X(-120)},{Y(120)} {X(-70)},{Y(170)} "
             f"C{X(-40)},{Y(200)} {X(-100)},{Y(230)} {X(-60)},{Y(232)} "
             f"C{X(-20)},{Y(236)} {X(-20)},{Y(180)} {X(-50)},{Y(140)} "
             f"C{X(-74)},{Y(100)} {X(-50)},{Y(50)} {X(-10)},{Y(40)} Z", fur, ink, 2.4 * s),
        path(f"M{X(-62)},{Y(160)} C{X(-46)},{Y(186)} {X(-70)},{Y(206)} {X(-58)},{Y(222)}", "none", dark, 4 * s),
        ellipse(X(-4), Y(8), 22 * s, 8 * s, fur, stroke), ellipse(X(26), Y(8), 18 * s, 7 * s, fur, stroke),
        ellipse(X(6), Y(60), 36 * s, 56 * s, fur, stroke),
        ellipse(X(16), Y(56), 20 * s, 40 * s, belly),
        ellipse(X(20), Y(132), 28 * s, 24 * s, fur, stroke),
        path(f"M{X(4)},{Y(150)} L{X(0)},{Y(178)} L{X(20)},{Y(156)} Z", fur, ink, 2.2 * s),
        path(f"M{X(6)},{Y(154)} L{X(4)},{Y(170)} L{X(15)},{Y(157)} Z", "#f0a3a0"),
        ellipse(X(40), Y(126), 12 * s, 9 * s, belly, f'stroke="{ink}" stroke-width="{1.6 * s}"'),
        circle(X(50), Y(128), 3.6 * s, ink),
        eye(X(28), Y(138), 5 * s, look * 0.6),
        acorn(X(40), Y(92), 13 * s),
        ellipse(X(30), Y(86), 9 * s, 6 * s, fur, stroke), ellipse(X(46), Y(80), 9 * s, 6 * s, fur, stroke),
    ])


def tortoise(cx, base, s=1.0, look=1):
    """A tortoise walking towards `look`, its domed shell laid with square tiles. 1.0 is 300 units long."""
    f = look
    X = lambda dx: cx + f * dx * s
    Y = lambda dy: base - dy * s
    skin, shell, rim, ink = "#9cc27a", "#c9a24a", "#8a6a2a", "#2a2410"
    stroke = f'stroke="{ink}" stroke-width="{2.4 * s}"'
    sid = uid("shell")
    tiles = []
    for r in range(4):
        for c in range(6):
            tx, ty = cx - 105 * s + c * 35 * s, base - 150 * s + r * 26 * s
            tone = ("#e3c26a", "#d4ad52", "#b98f3a")[(r + c) % 3]
            tiles.append(rect(tx + 2 * s, ty + 2 * s, 31 * s, 22 * s, tone, 4 * s, ink, 1.6 * s,
                              f'clip-path="url(#{sid})"'))
    dome = f"M{cx - 120 * s},{base - 44 * s} Q{cx - 116 * s},{base - 160 * s} {cx},{base - 162 * s} Q{cx + 116 * s},{base - 160 * s} {cx + 120 * s},{base - 44 * s} Z"
    return "".join([
        shadow(cx, base, 140 * s),
        *[ellipse(X(dx), Y(14), 18 * s, 22 * s, skin, stroke) for dx in (-80, -30, 40, 88)],
        path(f"M{X(-118)},{Y(56)} L{X(-150)},{Y(40)} L{X(-118)},{Y(42)} Z", skin, ink, 2.2 * s),
        path(f"M{X(110)},{Y(70)} Q{X(150)},{Y(110)} {X(178)},{Y(96)}", "none", ink, 30 * s),
        path(f"M{X(110)},{Y(70)} Q{X(150)},{Y(110)} {X(178)},{Y(96)}", "none", skin, 25 * s),
        ellipse(X(184), Y(98), 30 * s, 24 * s, skin, stroke),
        eye(X(194), Y(106), 6 * s, look * 0.6),
        path(f"M{X(196)},{Y(88)} q{f * 10 * s},{4 * s} {f * 18 * s},{-2 * s}", "none", ink, 2.2 * s),
        f'<defs><clipPath id="{sid}"><path d="{dome}"/></clipPath></defs>',
        path(dome, shell, ink, 2.8 * s),
        *tiles,
        path(dome, "none", ink, 2.8 * s),
        rect(cx - 126 * s, base - 50 * s, 252 * s, 14 * s, rim, 7 * s, ink, 2.2 * s),
    ])


def plain_crate(x, y, w, h, fill="#c9a46a"):
    """A small crate with no label: a box and its 2 diagonal straps."""
    return "\n".join([rect(x, y, w, h, fill, 4, "#7a5a32", 2.5),
                      line(x + 4, y + 4, x + w - 4, y + h - 4, "#7a5a32", 2, 'opacity="0.6"'),
                      line(x + w - 4, y + 4, x + 4, y + h - 4, "#7a5a32", 2, 'opacity="0.6"')])


def crate(x, y, w, h, text, fill="#c9a46a"):
    return "\n".join([rect(x, y, w, h, fill, 5, "#7a5a32", 3),
                      line(x + 6, y + 6, x + w - 6, y + h - 6, "#7a5a32", 2.5, 'opacity="0.6"'),
                      line(x + w - 6, y + 6, x + 6, y + h - 6, "#7a5a32", 2.5, 'opacity="0.6"'),
                      rect(x + 10, y + h / 2 - 15, w - 20, 30, "#fff6e2", 4, "#7a5a32", 1.5),
                      label(x + w / 2, y + h / 2 + 7, text, 18, NIGHT, font=MONO)])


# --------------------------------------------------------------------- plates

COVER_W, COVER_H = 960, 1280


def iso_cube(sx, sy, a, top, left, right, edge="#0b1520", extra=""):
    """An isometric cube whose top face has its upper corner at (sx, sy) and edge length a."""
    w, h = a * 0.866, a * 0.5
    pts = lambda ps: " ".join(f"{x:.1f},{y:.1f}" for x, y in ps)
    return "".join([
        f'<polygon points="{pts([(sx - w, sy + h), (sx, sy + 2 * h), (sx, sy + 2 * h + a), (sx - w, sy + h + a)])}" fill="{left}" stroke="{edge}" stroke-width="1.2" {extra}/>',
        f'<polygon points="{pts([(sx, sy + 2 * h), (sx + w, sy + h), (sx + w, sy + h + a), (sx, sy + 2 * h + a)])}" fill="{right}" stroke="{edge}" stroke-width="1.2" {extra}/>',
        f'<polygon points="{pts([(sx, sy), (sx + w, sy + h), (sx, sy + 2 * h), (sx - w, sy + h)])}" fill="{top}" stroke="{edge}" stroke-width="1.2" {extra}/>'])


@plate("cover", "", "", "The cover of GPU Architecture & Parallel Programming from First Principles. A glowing perspective "
       "grid runs to the horizon under a night sky. In the middle stands an isometric 3D grid of thread blocks, 8 cubes wide, 4 "
       "deep and 3 high; on its top layer 31 worker bees in hard hats, one warp, each stand on their own amber cube, and 1 cube "
       "is empty because its bee has wandered off on a branch with a question mark. On the left the queen bee, on a pillar, "
       "gives the one order c[i] = a[i] + b[i]. On the right 4 large cubes carry 4 hornets: the few fast cores of a CPU. A "
       "crab in a hard hat waves from the corner, because the book writes every program in CUDA C++ and in Rust.",
       size=(COVER_W, COVER_H))
def cover():
    FUR, HONEY_SOFT, CREAM_T, AMBER, NAVY = "#e7bf5f", "#d9a441", "#f1e6cc", "#d6a85a", "#0f1d2b"
    defs = """<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0a1420"/><stop offset="0.45" stop-color="#16304a"/>
<stop offset="0.62" stop-color="#1d4363"/><stop offset="1" stop-color="#08121c"/></linearGradient>
<radialGradient id="halo" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#5fb3c9" stop-opacity="0.35"/><stop offset="1" stop-color="#5fb3c9" stop-opacity="0"/></radialGradient>
<radialGradient id="warm" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#e9c47a" stop-opacity="0.30"/><stop offset="1" stop-color="#e9c47a" stop-opacity="0"/></radialGradient>
<linearGradient id="band" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#08121c" stop-opacity="0"/><stop offset="0.4" stop-color="#08121c"/></linearGradient>
<linearGradient id="honey" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ecc777"/><stop offset="1" stop-color="#c98f32"/></linearGradient>
</defs>"""
    parts = [defs, rect(0, 0, COVER_W, COVER_H, "url(#sky)")]

    # Stars, then the perspective floor grid running to a horizon at y = 560.
    for k in range(70):
        x, y = (k * 137) % COVER_W, 20 + (k * 89) % 520
        parts.append(circle(x, y, 0.8 + (k % 3) * 0.5, "#cfe3f0", f'opacity="{0.25 + 0.15 * (k % 4):.2f}"'))
    hz, vx = 560, 480
    parts.append(ellipse(vx, hz, 560, 90, "url(#halo)"))
    for k in range(-14, 15):
        parts.append(line(vx, hz, vx + k * 140, COVER_H, "#4ea3bd", 1.4, 'opacity="0.45"'))
    for k in range(1, 16):
        y = hz + (COVER_H - hz) * (k / 15) ** 2.2
        parts.append(line(0, y, COVER_W, y, "#4ea3bd", 1.2, f'opacity="{0.15 + 0.35 * k / 15:.2f}"'))

    parts += [
        label(480, 128, "GPU ARCHITECTURE", 92, CREAM_T, weight=900, fit=860, extra='stroke="#08121c" stroke-width="3" paint-order="stroke"'),
        label(480, 206, "& PARALLEL PROGRAMMING", 60, AMBER, weight=900, fit=860, extra='stroke="#08121c" stroke-width="2" paint-order="stroke"'),
        label(480, 278, "from First Principles", 44, "#d8cbb0", font=SERIF, weight=700, fit=600, extra='font-style="italic"'),
    ]

    # The GPU: an isometric grid of thread blocks, 8 x 4 x 3, with a warp of bees on its top layer.
    a, ox, oy = 40, 470, 640
    w, h = a * 0.866, a * 0.5
    parts.append(ellipse(ox, oy + 330, 330, 60, "url(#warm)"))
    cubes = [(i, j, k) for k in range(3) for i in range(8) for j in range(4)]
    cubes.sort(key=lambda c: (c[2], c[0] + c[1]))
    bees = []
    for i, j, k in cubes:
        sx, sy = ox + (i - j) * w - 2 * w, oy + (i + j) * h - k * a + 2 * a
        top_layer = k == 2
        if top_layer:
            parts.append(iso_cube(sx, sy, a, "#e8bf62", "#b9852c", "#94661c"))
            if (i, j) != (5, 2):
                bees.append((sx, sy + h))
            else:
                parts.append(f'<polygon points="{sx:.1f},{sy + 4:.1f} {sx + w - 6:.1f},{sy + h:.1f} {sx:.1f},{sy + 2 * h - 4:.1f} {sx - w + 6:.1f},{sy + h:.1f}" fill="#0b1520" stroke="{AMBER}" stroke-width="2" stroke-dasharray="4 4"/>')
        else:
            parts.append(iso_cube(sx, sy, a, "#3d7393", "#26506d", "#173a52"))
    for bx, by in sorted(bees, key=lambda b: b[1]):
        parts.append(honeybee(bx, by + 4, 0.20, look=0, hat="hard", arms="hold", carry=HONEY_SOFT, fur=FUR))
    # The bee that took the other branch.
    ex, ey = ox + (5 - 2) * w - 2 * w, oy + (5 + 2) * h - 2 * a + 2 * a
    parts += [path(f"M{ex + 10:.1f},{ey:.1f} Q{ex + 120:.1f},{ey - 90:.1f} {ex + 205:.1f},{ey - 70:.1f}", stroke=AMBER, sw=2.5, extra='stroke-dasharray="5 6" opacity="0.85"'),
              honeybee(ex + 230, ey - 40, 0.30, look=-1, hat="hard", arms="up", fur=FUR),
              circle(ex + 262, ey - 120, 16, "#f4ecdc"), label(ex + 262, ey - 111, "?", 24, RUST, weight=900)]
    parts.append(tag(380, 1040, "the GPU: a grid of blocks, a warp of 32", 18, "#f4ecdc", NAVY, MONO, 800))

    # The queen on a pillar, with her one order.
    parts += [iso_cube(150, 640, 64, "#e8bf62", "#b9852c", "#94661c"),
              iso_cube(150, 704, 64, "#3d7393", "#26506d", "#173a52"),
              honeybee(150, 676, 0.95, look=1, role="queen", fur=FUR),
              path("M230,452 L206,488 L244,474", "#f4ecdc", "#08121c", 4),
              rect(196, 374, 300, 104, "#f4ecdc", 20, "#08121c", 4), rect(208, 458, 34, 18, "#f4ecdc"),
              label(346, 410, "One order, every bee:", 20, "#5a4a30", weight=700, fit=250),
              label(346, 456, "c[i] = a[i] + b[i]", 30, NAVY, font=MONO, weight=800, fit=270)]

    # The CPU: 4 large cubes, each with a hornet.
    big, cx0, cy0 = 88, 806, 770
    W2, H2 = big * 0.866, big * 0.5
    spots = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for i, j in sorted(spots, key=lambda c: c[0] + c[1]):
        sx, sy = cx0 + (i - j) * W2, cy0 + (i + j) * H2
        parts.append(iso_cube(sx, sy, big, "#c9733a", "#9c5222", "#7a3d17"))
    for i, j in sorted(spots, key=lambda c: c[0] + c[1]):
        sx, sy = cx0 + (i - j) * W2, cy0 + (i + j) * H2
        parts.append(hornet(sx, sy + H2 + 6, 0.29, look=-1 if i else 1, arms="fold"))
    parts.append(tag(806, 1080, "the CPU: 4 fast hornets", 18, "#f4ecdc", NAVY, MONO, 800))

    parts += [rect(0, 1110, COVER_W, 170, "url(#band)"),
              rect(0, 1158, COVER_W, 3, AMBER, extra='opacity="0.7"'),
              label(48, 1248, "Arpan Pathak", 40, CREAM_T, "start", SERIF, 700, fit=textw("Arpan Pathak", 40)),
              rect(640, 1214, 290, 46, AMBER, 10), label(785, 1244, "IN CUDA C++ AND RUST", 19, NAVY, weight=800, fit=250),
              crab(785, 1208, 0.42, "happy", look=-1, hat="hard", shade=False)]
    return parts


@plate("ch01", "The Mathematics of Parallelism", "Tally the ant", "Tally, an ant, leads a column of ants carrying crumbs across a "
       "meadow. The column is wide where the ground is open, and every ant walks at once. Then it narrows to a single twig "
       "bridge over a stream, where only one ant can cross at a time and the rest wait in a queue. The bridge is the serial "
       "part of the work.")
def ch01():
    crumbs = ["#e8c877", "#f0b27a", "#e8c877", "#f7d9a0"]
    field = [ant(70 + 82 * (k % 4), GROUND - 64 + 58 * (k // 4), 0.6, carry=crumbs[k % 4]) for k in range(8)]
    queue = [ant(420 + 66 * k, GROUND - 6, 0.55, carry=crumbs[k % 4]) for k in range(3)]

    return [backdrop(seed=1, **MEADOW),
            path(f"M640,{GROUND - 44} L800,{GROUND - 44} L870,{H} L590,{H} Z", "#3c8dc5"),
            *[path(f"M{640 + 30 * k},{GROUND - 10 + 22 * k} q14,-6 28,0 t28,0", stroke="#bfe3f7", sw=3, extra='opacity="0.7"') for k in range(3)],
            rect(616, GROUND - 52, 210, 12, WOOD, 6, "#5a3d28", 2),
            *field, *queue, ant(722, GROUND - 52, 0.55, carry="#e8c877"),
            ant(900, GROUND - 6, 0.6, carry="#f0b27a"),
            label(722, GROUND - 104, "one at a time", 17, "#ffe7a3", font=MONO, weight=600),
            label(190, GROUND - 150, "all at once", 17, "#ffe7a3", font=MONO, weight=600),
            rect(420, 96, 400, 56, "#fff6e2", 10, WOOD, 3),
            label(620, 131, "time = serial + parallel / p", 17, NIGHT, font=MONO, weight=600)]


@plate("ch02", "GPU Hardware from First Principles", "Comb the worker bee", "Comb, a worker bee in a hard hat, stands beside a "
       "hive with its side cut away. Inside hang three wooden comb frames, labelled SM 0, SM 1 and SM 2. On each frame small worker "
       "bees stand in straight rows that move together, a formation called a warp. Comb points at the frames like a guide on a tour.")
def ch02():
    parts = [backdrop(seed=2, **DUSK),
             rect(292, 34, 640, 300, "#7a4f16", 18, "#2a1a10", 4),
             rect(308, 50, 608, 268, "#2a1a10", 10)]

    for k in range(3):
        fx = 324 + k * 198
        parts += [rect(fx, 76, 180, 228, "#c9a46a", 6, "#5a3d28", 3), rect(fx + 10, 86, 160, 208, "#e8b84a", 3)]
        for r in range(5):
            for c in range(6 - r % 2):
                hexagon_x = fx + 24 + c * 26 + (r % 2) * 13
                parts.append(hexagon(hexagon_x, 104 + r * 24, 12, "#f5c542", "#c98f12", 1.5, 'opacity="0.55"'))
        for row in range(2):
            for c in range(6):
                parts.append(honeybee(fx + 27 + c * 25, 190 + row * 80, 0.20, look=0))
        parts.append(tag(fx + 90, 64, "SM %d" % k, 16, "#fff6e2", "#2a1a10", MONO, 800))

    parts += [honeybee(160, 330, 1.0, look=1, arms="up", hat="hard"),
              tag(160, 70, "a warp moves as one", 17, "#fff6e2", "#2a1a10", SANS, 700)]
    return parts


@plate("ch03", "The CUDA Programming Model", "The Queen", "The Queen bee, in her crown and sash, holds up a scroll that "
       "reads <<<4, 256>>>: a launch of 4 blocks of 256 threads. Beside her a wooden board holds 4 comb frames, labelled block 0 "
       "to block 3, and on each frame rows of worker bees wait for the order. The Queen gives one order and every bee on every "
       "frame carries it out.")
def ch03():
    parts = [backdrop(seed=3, **MEADOW),
             rect(368, 40, 572, 282, "#7a4f16", 18, "#2a1a10", 4),
             rect(382, 54, 544, 254, "#2a1a10", 10)]
    for k in range(4):
        fx = 394 + k * 132
        parts += [rect(fx, 78, 122, 218, "#c9a46a", 6, "#5a3d28", 3), rect(fx + 8, 86, 106, 202, "#e8b84a", 3)]
        for r in range(3):
            for c in range(4):
                parts.append(honeybee(fx + 25 + c * 24, 140 + r * 64, 0.165, look=0))
        parts.append(tag(fx + 61, 66, "block %d" % k, 15, "#fff6e2", "#2a1a10", MONO, 800))
    parts += [honeybee(150, 332, 1.0, look=1, role="queen"),
              rect(200, 72, 156, 52, "#fff6e2", 10, WOOD, 3),
              label(278, 105, "<<<4, 256>>>", 16, "#2a1a10", font=MONO, weight=800, fit=textw("<<<4, 256>>>", 16, MONO)),
              tag(150, 30, "one order, every thread", 16, "#fff6e2", "#2a1a10", SANS, 700)]
    return parts


@plate("ch04", "Memory Management & Data Movement", "Ferry the pelican", "Ferry, a pelican, stands on a wooden raft in the "
       "middle of a river, its beak pouch full of crates of bytes. On the left bank crates are stacked under a sign that reads host "
       "memory; on the right bank a sign reads device memory. A tag says pinned: 1 trip, because pinned data crosses in a single "
       "copy.")
def ch04():
    river = path(f"M300,{GROUND - 30} L660,{GROUND - 30} L720,{H} L240,{H} Z", "#3c8dc5")
    waves = [path(f"M{330 + 60 * k},{GROUND + 10 + 14 * (k % 2)} q14,-6 28,0 t28,0", stroke="#bfe3f7", sw=3, extra='opacity="0.7"')
             for k in range(5)]
    host = [crate(60 + 58 * (k % 3), GROUND - 52 - 48 * (k // 3), 52, 44, "") for k in range(5)]
    return [backdrop(seed=4, **SHORE), river, *waves,
            *host,
            rect(60, 96, 180, 44, "#fff6e2", 10, WOOD, 3), label(150, 125, "host memory", 18, "#2a1a10", fit=textw("host memory", 18)),
            rect(720, 96, 200, 44, "#fff6e2", 10, WOOD, 3), label(820, 125, "device memory", 18, "#2a1a10", fit=textw("device memory", 18)),
            rect(760, GROUND - 120, 150, 120, "#7a4f16", 8, "#2a1a10", 3), rect(800, GROUND - 70, 70, 70, "#2a1a10", 4),
            rect(380, GROUND - 6, 210, 22, WOOD, 6, "#5a3d28", 3),
            crate(392, GROUND - 40, 40, 34, ""), crate(536, GROUND - 40, 40, 34, ""),
            pelican(470, GROUND - 6, 0.9, look=1),
            tag(480, 70, "pinned: 1 trip", 17, "#fff6e2", "#2a1a10", SANS, 700)]


@plate("ch05", "Synchronisation, Atomics & Race Conditions", "Whistle the meerkat", "Whistle, a meerkat with a whistle, "
       "stands on watch beside a striped gate across a path. 3 worker bees have reached the gate and wait in a row; a 4th is "
       "still flying in from the left. The gate stays down until every bee has arrived. A sign on the gate reads "
       "__syncthreads(), and a tag says all arrive, then all go.")
def ch05():
    gate_x = 600
    waiting = [honeybee(370 + 70 * k, GROUND - 4, 0.42, look=1) for k in range(3)]
    return [backdrop(seed=5, **SAND),
            path(f"M0,{GROUND - 20} L{W},{GROUND - 20} L{W},{GROUND + 30} L0,{GROUND + 30} Z", "#c9a46a", extra='opacity="0.55"'),
            rect(gate_x - 8, GROUND - 150, 16, 150, WOOD, 4, "#2a1a10", 3),
            rect(gate_x - 300, GROUND - 120, 300, 16, "#f6efe0", 8, "#2a1a10", 3),
            *[rect(gate_x - 290 + 50 * k, GROUND - 119, 24, 14, RUST, 2) for k in range(6)],
            rect(gate_x - 220, GROUND - 196, 210, 44, "#fff6e2", 10, WOOD, 3),
            label(gate_x - 115, GROUND - 167, "__syncthreads()", 17, "#2a1a10", font=MONO, weight=700,
                  fit=textw("__syncthreads()", 17, MONO)),
            *waiting,
            bee(150, GROUND - 120, 0.9, look=1),
            path(f"M40,{GROUND - 104} q30,-18 60,-6 t50,-4", stroke=SNOW, sw=2.5, extra='stroke-dasharray="4 7" opacity="0.7"'),
            meerkat(720, GROUND - 4, 1.05, look=-1),
            tag(480, 60, "all arrive, then all go", 17, "#fff6e2", "#2a1a10", SANS, 700),
            tag(150, GROUND - 170, "still on its way", 15, "#ffffffdd", "#27405a", SANS, 700)]


@plate("ch06", "Streams, Events & Asynchronous Execution", "The otter twins", "The otter twins float on their backs "
       "down 2 rivers that run side by side, each with a crate on its belly. Crates bob along both rivers at staggered "
       "places, so both streams carry work at the same time. Signs on the bank read stream 0 and stream 1, and a tag "
       "says 2 streams, at the same time.")
def ch06():
    rivers = []
    for k, top in enumerate((150, 240)):
        rivers += [rect(0, top, W, 64, "#3c8dc5", 0, "none", 0, 'opacity="0.95"'),
                   *[path(f"M{40 + 120 * j + 60 * k},{top + 18 + 22 * (j % 2)} q14,-6 28,0 t28,0", stroke="#bfe3f7", sw=3,
                          extra='opacity="0.7"') for j in range(8)]]
    floats = [plain_crate(250 + 190 * j, 166, 34, 28) for j in range(3)] + \
             [plain_crate(330 + 190 * j, 256, 34, 28) for j in range(3)]
    return [backdrop(seed=6, **MEADOW), *rivers, *floats,
            otter(150, 182, 0.75, look=1), otter(230, 272, 0.75, look=1),
            rect(16, 96, 132, 40, "#fff6e2", 10, WOOD, 3), label(82, 123, "stream 0", 17, "#2a1a10", font=MONO, fit=textw("stream 0", 17, MONO)),
            rect(16, 312, 132, 40, "#fff6e2", 10, WOOD, 3), label(82, 339, "stream 1", 17, "#2a1a10", font=MONO, fit=textw("stream 1", 17, MONO)),
            tag(620, 70, "2 streams, at the same time", 17, "#fff6e2", "#2a1a10", SANS, 700)]


@plate("ch07", "Memory Optimisation Deep Dive", "Inch the caterpillar", "Inch, a caterpillar, walks along a branch "
       "laid with numbered cells, 0 to 9 in order. Each of its segments stands over the next cell, so all of them step "
       "together and none reaches past its neighbour. A tag says neighbours read neighbours: the rule of coalesced "
       "memory access.")
def ch07():
    r, x0 = 22, 318
    cells = [cell(x0 + 1.6 * r * k, GROUND - 26, str(k), "#f7d9a0", NIGHT, 32, 16) for k in range(10)]
    return [backdrop(seed=7, **MEADOW),
            rect(250, GROUND - 6, 460, 22, WOOD, 10, "#5a3d28", 3),
            *cells,
            caterpillar(x0, GROUND - 42, segments=9, r=r),
            tag(480, 80, "neighbours read neighbours", 17, "#fff6e2", "#2a1a10", SANS, 700)]


@plate("ch08", "Reduction, Scan & Histogram", "Stash the squirrel", "Stash, a red squirrel, sits on a tree stump "
       "beside a pyramid of acorns. The bottom row holds 8 acorns, the row above 4, then 2, then 1 at the top: each acorn "
       "stands for the 2 below it, so 8 values become 1 in 3 steps. Stash holds 1 more acorn in its paws. A tag says pairs, "
       "then pairs of pairs: the tree of a reduction.")
def ch08():
    acorns = []
    for row, count in enumerate((8, 4, 2, 1)):
        gap = 40 * 2 ** row
        x0 = 480 - gap * (count - 1) / 2
        y = GROUND - 22 - row * 46
        acorns += [acorn(x0 + gap * k, y, 15) for k in range(count)]
    return [backdrop(seed=8, **DUSK),
            rect(300, GROUND - 4, 360, 20, WOOD, 8, "#5a3d28", 3),
            *acorns,
            rect(720, GROUND - 56, 140, 56, "#8a6a44", 10, "#5a3d28", 3),
            ellipse(790, GROUND - 56, 70, 12, "#c9a46a", 'stroke="#5a3d28" stroke-width="3"'),
            squirrel(780, GROUND - 60, 0.9, look=-1),
            tag(480, 64, "pairs, then pairs of pairs", 17, "#fff6e2", "#2a1a10", SANS, 700)]


@plate("ch09", "Optimised Matrix Multiplication", "Shell the tortoise", "Shell, a tortoise, walks across a meadow. "
       "Its domed shell is laid with square tiles in rows, like a matrix cut into blocks. Ahead of it lie 2 stacks of "
       "tiles, 1 marked A and 1 marked B, and behind it a finished row of tiles marked C. A tag says load a tile once, "
       "use it many times: the idea of a tiled matrix multiplication.")
def ch09():
    stacks = []
    for k, name in enumerate(("A", "B")):
        x = 640 + k * 130
        stacks += [rect(x, GROUND - 22 - 18 * j, 90, 16, ("#e3c26a", "#d4ad52", "#b98f3a")[j % 3], 4, "#2a2410", 2)
                   for j in range(4)]
        stacks.append(tag(x + 45, GROUND - 110, name, 18, "#fff6e2", "#2a1a10", MONO, 700))
    done = [rect(40 + 34 * k, GROUND - 22, 30, 20, "#c8f0d0", 4, "#2a2410", 2) for k in range(5)]
    return [backdrop(seed=9, **MEADOW), *stacks, *done,
            tag(120, GROUND - 60, "C", 18, "#fff6e2", "#2a1a10", MONO, 700),
            tortoise(380, GROUND - 2, 0.95, look=1),
            tag(480, 64, "load a tile once, use it many times", 17, "#fff6e2", "#2a1a10", SANS, 700)]


@plate("ch13", "Your First Rust Kernel", "Pinch the crab", "Pinch, a crab in a hard hat, stands at a long table of numbered "
       "cells. Each claw picks up one input, a[i] in one and b[i] in the other, and the sum goes into the row c below. At the "
       "end of the row two small crabs have no cell to fill: they are the spare threads of the last block, and they doze.")
def ch13():
    a = ["0", "1", "2", "3", "4"]
    b = ["0", "2", "4", "6", "8"]
    c = ["0", "3", "6", "9", "12"]
    rows = []

    for k in range(5):
        x = 380 + 70 * k
        rows += [cell(x, 104, a[k], "#bfe3f7"), cell(x, 158, b[k], "#f7d9a0"), cell(x, 222, c[k], "#c8f0d0" if k < 4 else "#ffe7a3")]

    rows += [label(330, 110, "a", 22, SNOW, font=MONO), label(330, 164, "b", 22, SNOW, font=MONO), label(330, 228, "c", 22, SNOW, font=MONO),
             path("M660,124 C690,150 690,190 660,214", stroke=SNOW, sw=3, extra='stroke-dasharray="5 6" opacity="0.8"'),
             label(706, 176, "+", 34, HONEY, weight=900)]

    return [backdrop(seed=13, **SHORE),
            rect(350, 76, 330, 186, "#0f1c2b", 12, "#6f8196", 3, 'opacity="0.55"'),
            *rows,
            crab(190, s=1.15, mood="happy", look=1, hat="hard"),
            crab(790, s=0.42, mood="sleepy", shade=True), crab(880, s=0.42, mood="sleepy", shade=True),
            rect(752, 222, 168, 30, "#ffffffcc", 15), label(836, 243, "i ≥ N: no work", 16, "#27405a", font=MONO, weight=600)]


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)

    for name in names or list(PLATES):
        title, crew, description, draw, size = PLATES[name]
        parts = draw()

        if name.startswith("ch"):
            parts.append(ribbon(str(int(name[2:])), title, crew, size))

        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size[0]} {size[1]}" width="{size[0]}" height="{size[1]}" role="img"'
               f' aria-label="{escape(description)}">\n' + "\n".join(parts) + "\n" + defs_3d() + "\n</svg>\n")
        (OUT / (name + ".svg")).write_text(svg)

        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / (name + ".png")), output_width=size[0] * 2)
        except ImportError:
            pass

        print("  " + name)

    (OUT / "plates.json").write_text(json.dumps({name: p[2] for name, p in PLATES.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
