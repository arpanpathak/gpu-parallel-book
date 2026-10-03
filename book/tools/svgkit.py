"""Small helpers for the hand-drawn SVG figures (see tools/draw_*.py).

A figure is an SVG built from primitives. The same primitives build the frames
of an animation, so the picture in a chapter and the picture in its GIF come
from one set of calls.

Two things make an animation frame different from a printed one:

  * Every fragment of a figure is tagged `art` or `text`, so a transition can
    move the drawing while the words stay still. `Animation` rasterises the two
    layers apart and cross-fades only the drawing.
  * The GIF is written with one palette for the whole animation, built from the
    figure colours themselves rather than guessed from the pixels. Text stays
    sharp because the palette already holds the colours that antialiasing makes.

Labels go through `text_width`, so a box can be sized to the text it holds and a
long caption can be wrapped instead of running off the canvas. `lint_figures.py`
checks the result.
"""

import math
import os
import subprocess
import tempfile

INK = "#1d2733"; MUTED = "#5b6b7c"; PALE = "#eef3f7"; GREEN = "#e3f1ea"; TEAL = "#2f7f86"
CREAM = "#fdf6e8"; BRASS = "#c89b3c"; PINK = "#fbe3d6"; RUST = "#d9622b"; GREY = "#f1f3f5"
WHITE = "#ffffff"; BORDER = "#c9d4de"; HALO = WHITE
MONO = 'font-family="Menlo, monospace"'

# Every colour the figures draw with. The animation palette is grown from this
# list, so the names here and the palette cannot drift apart.
PALETTE_COLORS = [WHITE, INK, MUTED, PALE, GREEN, TEAL, CREAM, BRASS, PINK, RUST, GREY, BORDER]

# Mean advance width as a fraction of the font size. Menlo is monospace, so the
# estimate is exact for it; the sans-serif figure is an average that slightly
# over-reports, which keeps a label inside its box.
MONO_RATIO = 0.601
SANS_RATIO = 0.556


def text_width(s, size, mono=False):
    """Approximate the advance width of `s` at `size` points."""
    ratio = MONO_RATIO if mono else SANS_RATIO
    return max((len(line) for line in str(s).split("\n")), default=0) * size * ratio


def wrap(s, size, max_width, mono=False):
    """Break `s` on spaces so every line fits `max_width`. Honors explicit \\n."""
    out = []
    for paragraph in str(s).split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = word if not line else line + " " + word
            if not line or text_width(candidate, size, mono) <= max_width:
                line = candidate
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def fit_size(s, size, max_width, mono=False, floor=8.0):
    """Shrink `size` until the longest line of `s` fits `max_width`."""
    widest = max((text_width(line, size, mono) for line in str(s).split("\n")), default=0.0)
    if widest <= max_width or widest == 0:
        return size
    return max(floor, size * max_width / widest)


def box_edge(box, toward):
    """The point where the ray from a box's centre to `toward` leaves the box."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2.0) / abs(dx) if dx else math.inf
    sy = (h / 2.0) / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def rgb(value):
    """`#rrggbb` as three integers."""
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def polar(cx, cy, r, deg):
    """A point on a circle. Zero degrees is at the top; angles grow clockwise."""
    a = math.radians(deg) - math.pi / 2.0
    return cx + r * math.cos(a), cy + r * math.sin(a)


def mix(a, b, t):
    """Blend two hex colours in sRGB, the space the rasteriser blends in."""
    ca, cb = rgb(a), rgb(b)
    return "#%02x%02x%02x" % tuple(round(ca[i] * (1 - t) + cb[i] * t) for i in range(3))


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _Parts(list):
    """The SVG fragments of a figure, each remembering its layer.

    `Figure.text` writes into the `text` layer; everything else lands in `art`.
    `Animation` rasterises the two apart, so a cross-fade between two frames can
    move the drawing while the words change in one clean step.
    """

    def __init__(self):
        super().__init__()
        self.current = "art"
        self.layers = []

    def append(self, markup):
        super().append(markup)
        self.layers.append(self.current)


class Figure:
    # A type multiplier and a height multiplier, so one layout can be drawn at
    # the size a printed page wants or the larger size a GIF wants without
    # touching every call. The static chapter figures leave them at 1.0.
    type_scale = 1.0
    height_scale = 1.0

    def __init__(self, width, height):
        self.width = width
        self.height = height * self.height_scale
        self.parts = _Parts()
        self.defs = []

    # ------------------------------------------------------------- primitives

    def rect(self, x, y, w, h, fill=PALE, stroke=INK, rx=4, dash=False, width=1.4, opacity=1.0,
             scrim=False, layer="art"):
        d = ' stroke-dasharray="6,4"' if dash else ""
        o = ' opacity="%s"' % opacity if opacity != 1.0 else ""
        s = ' data-scrim="1"' if scrim else ""
        self.parts.current = layer
        try:
            self.parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{width}"{d}{o}{s}/>')
        finally:
            self.parts.current = "art"

    def circle(self, cx, cy, r, fill="none", stroke=INK, width=1.4, dash=False, opacity=1.0):
        d = ' stroke-dasharray="6,4"' if dash else ""
        o = ' opacity="%s"' % opacity if opacity != 1.0 else ""
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{width}"{d}{o}/>')

    def glow(self, cx, cy, r, color=BRASS, opacity=0.5):
        """A soft light behind the thing the frame is pointing at."""
        gid = "glow%d" % len(self.defs)
        self.defs.append(
            f'<radialGradient id="{gid}">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="{opacity:.3f}"/>'
            f'<stop offset="0.45" stop-color="{color}" stop-opacity="{opacity * 0.45:.3f}"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>')
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#{gid})"/>')

    def text(self, x, y, s, size=11, fill=INK, mono=False, anchor="start", bold=False, halo=False,
             halo_pad=2.6, scaled=True, layer="art"):
        """A label. `layer` is `art` for a name inside the drawing, `text` for prose."""
        if scaled:
            size = size * self.type_scale
            halo_pad = halo_pad * self.type_scale
        s = esc(s)
        f = MONO + ' xml:space="preserve"' if mono else ""
        b = ' font-weight="bold"' if bold else ""
        self.parts.current = layer
        try:
            if halo:
                # The halo is a second copy of the glyphs, drawn underneath. It is
                # marked so the figure linter does not report it as a duplicate label.
                self.parts.append(
                    f'<text data-halo="1" x="{x}" y="{y}" font-size="{size}" fill="{WHITE}" '
                    f'text-anchor="{anchor}" stroke="{WHITE}" stroke-width="{halo_pad}" '
                    f'stroke-linejoin="round" {f}{b}>{s}</text>')
            self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
                              f'text-anchor="{anchor}" {f}{b}>{s}</text>')
        finally:
            self.parts.current = "art"

    def cell(self, x, y, w, h, label, fill=PALE, stroke=INK, size=12, mono=True, dash=False,
             color=INK, fit=True, pad=8, rx=4, width=1.4, layer="art"):
        self.rect(x, y, w, h, fill, stroke, rx, dash, width, layer=layer)
        if label is None or label == "":
            return
        text = str(label)
        size = size * self.type_scale
        if fit and "\n" not in text:
            size = fit_size(text, size, w - pad, mono)
        self.text(x + w / 2, y + h / 2 + size * 0.36, text, size, color, mono, "middle",
                  scaled=False, layer=layer)

    def arrow(self, x1, y1, x2, y2, color=INK, width=1.6, dash=False, head=8.0, opacity=1.0):
        a = math.atan2(y2 - y1, x2 - x1)
        p1 = (x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45))
        p2 = (x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45))
        d = ' stroke-dasharray="6,4"' if dash else ""
        o = ' opacity="%s"' % opacity if opacity != 1.0 else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{width}"{d}{o}/>')
        self.parts.append(f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} '
                          f'{p2[0]:.1f},{p2[1]:.1f}" fill="{color}"{o}/>')

    def link(self, box_a, box_b, label=None, color=INK, width=1.6, dash=False, gap=5.0,
             label_size=10, label_mono=True, label_offset=(0, -7), halo=True):
        """Arrow from the edge of `box_a` to the edge of `box_b`, never into either."""
        ca, cb = center(box_a), center(box_b)
        ax, ay = box_edge(box_a, cb)
        bx, by = box_edge(box_b, ca)
        a = math.atan2(by - ay, bx - ax)
        ax, ay = ax + gap * math.cos(a), ay + gap * math.sin(a)
        bx, by = bx - (gap + 1.0) * math.cos(a), by - (gap + 1.0) * math.sin(a)
        self.arrow(ax, ay, bx, by, color, width, dash)
        if label is not None:
            self.text((ax + bx) / 2 + label_offset[0], (ay + by) / 2 + label_offset[1],
                      label, label_size, color, label_mono, "middle", halo=halo)

    def line(self, x1, y1, x2, y2, color=INK, width=1.4, dash=False, opacity=1.0):
        d = ' stroke-dasharray="6,4"' if dash else ""
        o = ' opacity="%s"' % opacity if opacity != 1.0 else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{width}"{d}{o}/>')

    def shape(self, points, fill=INK, opacity=1.0):
        """A filled shape from a list of `(x, y)`, for a caret or a small triangle.

        This is a path rather than a `polygon` on purpose. `lint_figures.py` reads
        a small polygon as an arrowhead and insists it sit on a shaft, which a
        caret under a value has no reason to do.
        """
        o = ' opacity="%s"' % opacity if opacity != 1.0 else ""
        head = "M %.2f %.2f" % points[0]
        body = "".join(" L %.2f %.2f" % p for p in points[1:])
        self.parts.append(f'<path d="{head}{body} Z" fill="{fill}" stroke="none"{o}/>')

    def arc(self, cx, cy, r, start_deg, end_deg, color=TEAL, width=2.4, dash=False, opacity=1.0):
        """A stroked arc. Zero degrees is at the top; angles grow clockwise.

        A ring of nodes and keys needs this: which node owns a key is read by
        travelling clockwise, and a straight line cannot show a wrap-around.
        """
        x0, y0 = polar(cx, cy, r, start_deg)
        x1, y1 = polar(cx, cy, r, end_deg)
        sweep = end_deg - start_deg
        large = 1 if abs(sweep) > 180 else 0
        flag = 1 if sweep > 0 else 0
        d = ' stroke-dasharray="6,4"' if dash else ""
        o = ' opacity="%s"' % opacity if opacity != 1.0 else ""
        self.parts.append(f'<path d="M {x0:.2f} {y0:.2f} A {r} {r} 0 {large} {flag} '
                          f'{x1:.2f} {y1:.2f}" fill="none" stroke="{color}" '
                          f'stroke-width="{width}"{d}{o}/>')

    def panel(self, x, y, w, h, title=None, fill="#fbfcfd", stroke=INK, rx=6, size=10.5):
        """A soft container with an optional caption in its top-left corner."""
        self.rect(x, y, w, h, fill, stroke, rx, width=1.1)
        if title is not None:
            self.text(x + 10, y + 16 * self.type_scale, title, size, MUTED, bold=True)

    def caption(self, x, y, s, size=10, fill=MUTED, width=None, leading=None, mono=False,
                anchor="start"):
        """A muted line, wrapped to `width` when one is given. Returns the last baseline."""
        size = size * self.type_scale
        lead = leading * self.type_scale if leading is not None else size * 1.45
        if width is None:
            self.text(x, y, s, size, fill, mono, anchor, scaled=False)
            return y
        rows = wrap(s, size, width, mono)
        for i, row in enumerate(rows):
            self.text(x, y + i * lead, row, size, fill, mono, anchor, scaled=False)
        return y + (len(rows) - 1) * lead

    def zone(self, x, y, w, h, label=None, color=TEAL, fill=TEAL, opacity=0.10, size=11,
             rx=8, label_above=True, dash=True):
        """A tinted band that marks a region, with its name in large type.

        This is the shape a reader's eye can hold on to: the sliding window, the
        live range of a binary search, the frontier of a BFS, the matched prefix.
        """
        self.rect(x, y, w, h, fill, color, rx, dash=dash, width=2.0, opacity=opacity)
        if label:
            ly = y - 9 if label_above else y + h + size * 1.35
            self.text(x + w / 2, ly, label, size, color, anchor="middle", bold=True,
                      halo=True, scaled=False)

    def scrim(self, boxes, frame_box):
        """Fade everything outside `frame_box` so the eye lands inside it.

        The dimming is drawn as four pale panels around the region. A reader sees
        the same picture with the focus brought up, which is the effect of a
        spotlight, without hiding any of the surrounding state.
        """
        x, y, w, h = frame_box
        x0, y0 = 0.0, 0.0
        x1, y1 = float(self.width), float(self.height)
        for box in ((x0, y0, x1 - x0, y - y0),
                    (x0, y + h, x1 - x0, y1 - (y + h)),
                    (x0, y, x - x0, h),
                    (x + w, y, x1 - (x + w), h)):
            bx, by, bw, bh = box
            if bw > 0.5 and bh > 0.5:
                self.rect(bx, by, bw, bh, WHITE, "none", 0, width=0, opacity=0.66, scrim=True)

    # ------------------------------------------------------------------ output

    def svg(self, layer=None, background=True):
        body = [m for m, at in zip(self.parts, self.parts.layers)
                if layer is None or at == layer]
        head = "".join(self.defs)
        if background:
            head += f'<rect width="100%" height="100%" fill="{WHITE}"/>'
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} '
                f'{self.height}" width="{self.width}" height="{self.height}" '
                f'font-family="Helvetica, Arial, sans-serif">\n{head}\n'
                f'{"".join(chr(10) + b for b in body)}\n</svg>\n')

    def save(self, path, layer=None, background=True):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.svg(layer, background))


def _ramp(a, b, steps):
    return [mix(a, b, (i + 1) / (steps + 1.0)) for i in range(steps)]


def animation_palette(colors=256):
    """A palette built from the figure colours instead of guessed from pixels.

    Antialiased glyphs are blends between a stroke colour and the fill behind it.
    A clustered palette spends its entries on whatever covers the most pixels,
    which is the flat background, so the blend ramp gets truncated and the type
    looks chewed. Listing the ramps up front keeps every edge clean, and costs
    nothing at run time because it is the same list for every animation.
    """
    from PIL import Image

    flat = list(PALETTE_COLORS)
    entries = list(flat)

    # Every colour fading into the two grounds the figures use: white and ink.
    for c in flat:
        if c != WHITE:
            entries.extend(_ramp(c, WHITE, 12))
    for c in (WHITE, PALE, GREEN, CREAM, GREY, BORDER, BRASS):
        entries.extend(_ramp(c, INK, 7))
    # The accent pairs that actually meet on a cell or a band.
    for a, b in ((RUST, GREEN), (TEAL, GREEN), (TEAL, CREAM), (BRASS, CREAM),
                 (INK, GREEN), (INK, PALE), (MUTED, PALE)):
        entries.extend(_ramp(a, b, 5))

    seen, unique = set(), []
    for value in entries:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    while len(unique) < colors:
        unique.append(unique[len(unique) % len(flat)])
    unique = unique[:colors]

    img = Image.new("P", (len(unique), 1))
    img.putpalette([round(v) for c in unique for v in rgb(c)])
    return img


class Animation:
    """A sequence of `Figure` frames saved as a slow, looping GIF.

    Each keyframe is held long enough to read. Between keyframes the drawing is
    cross-faded in a couple of short steps while the words cut straight to their
    new value, so a pointer glides without the sentence blurring.

    The class settings tune every animation at once:

      * `speed` scales every hold, so the whole book can be slowed together.
      * `scale` rasterises the figure at more than one output pixel per drawing
        unit, which is what keeps the type sharp on a retina screen.
      * `tween` and `tween_ms` set how a transition looks and how long it lasts.
    """

    speed = 1.0
    scale = 1.85          # output pixels per drawing unit
    tween = 2             # drawing cross-fades inserted between two keyframes
    tween_ms = 110
    colors = 256
    hold = 1900           # default hold in milliseconds, before `speed`

    def __init__(self, frames, durations=None, default_duration=None):
        self.frames = list(frames)
        base = default_duration or self.hold
        if durations is None:
            durations = [base] * len(self.frames)
        elif isinstance(durations, (int, float)):
            durations = [durations] * len(self.frames)
        if len(durations) != len(self.frames):
            raise ValueError("expected %d durations, got %d" % (len(self.frames), len(durations)))
        self.durations = [d * self.speed for d in durations]

    def _layer(self, frame, target_w, tmp, tag, layer, background):
        from PIL import Image

        svg = os.path.join(tmp, "%s.svg" % tag)
        png = os.path.join(tmp, "%s.png" % tag)
        frame.save(svg, layer=layer, background=background)
        height = max(1, round(frame.height * target_w / frame.width))
        subprocess.run(["rsvg-convert", "-w", str(target_w), "-h", str(height), "-o", png, svg],
                       check=True)
        return Image.open(png)

    def save(self, path, width=None, colors=None):
        from PIL import Image

        if not self.frames:
            raise ValueError("an animation needs at least one frame")
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        target_w = int(round((width or self.frames[0].width) * self.scale))
        with tempfile.TemporaryDirectory() as tmp:
            art, words = [], []
            for i, frame in enumerate(self.frames):
                art.append(self._layer(frame, target_w, tmp, "a%03d" % i, "art", True)
                           .convert("RGBA"))
                words.append(self._layer(frame, target_w, tmp, "t%03d" % i, "text", False)
                             .convert("RGBA"))

            def compose(back, front):
                return Image.alpha_composite(back, front).convert("RGB")

            shots, waits = [], []
            for i in range(len(self.frames)):
                shots.append(compose(art[i], words[i]))
                waits.append(self.durations[i])
                if self.tween and i + 1 < len(self.frames):
                    for step in range(1, self.tween + 1):
                        eased = (1 - math.cos(math.pi * step / (self.tween + 1.0))) / 2.0
                        moving = Image.blend(art[i], art[i + 1], eased).convert("RGBA")
                        shots.append(compose(moving, words[i + 1]))
                        waits.append(self.tween_ms)

            base = animation_palette(colors or self.colors)
            paletted = [shot.quantize(palette=base, dither=Image.Dither.NONE) for shot in shots]
            paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=waits,
                             loop=0, optimize=True, disposal=1)
