"""Sharp arrowheads for the book's figures.

Every marker becomes a swept-back head 12 units long, and every line or path that carries
a marker is shortened by half the head, so that the line ends inside the head and the tip
still lands exactly on the original end point. A line that ended at the tip showed its
round cap past the head, which made small heads look like dots."""
import math
import re

HEAD = 12.0          # length of the head along the line
HALF_WIDTH = 4.5     # half of its width
INSET = HEAD / 2     # how far the tip lies beyond the shortened line


def _marker(match):
    tag = match.group(0)
    ident = re.search(r'id="([^"]+)"', tag).group(1)
    fill = re.search(r'fill="([^"]+)"', tag)
    colour = fill.group(1) if fill else "#3F5873"
    return (f'<marker id="{ident}" viewBox="0 0 {HEAD:g} {2 * HALF_WIDTH:g}" refX="{INSET:g}" '
            f'refY="{HALF_WIDTH:g}" markerUnits="userSpaceOnUse" markerWidth="{HEAD:g}" '
            f'markerHeight="{2 * HALF_WIDTH:g}" orient="auto">'
            f'<path d="M0,0 L{HEAD:g},{HALF_WIDTH:g} L0,{2 * HALF_WIDTH:g} L3.5,{HALF_WIDTH:g} Z" '
            f'fill="{colour}"/></marker>')


def _back(x0, y0, x1, y1):
    """The point INSET units before (x1, y1) on the way from (x0, y0)."""
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= INSET:
        return x1, y1
    return x1 - (x1 - x0) * INSET / length, y1 - (y1 - y0) * INSET / length


def _line(tag):
    def get(name):
        return float(re.search(rf'\b{name}="([-\d.]+)"', tag).group(1))
    x2, y2 = _back(get("x1"), get("y1"), get("x2"), get("y2"))
    tag = re.sub(r'\bx2="[-\d.]+"', f'x2="{x2:.1f}"', tag)
    return re.sub(r'\by2="[-\d.]+"', f'y2="{y2:.1f}"', tag)


def _path(tag):
    d = re.search(r'\bd="([^"]+)"', tag).group(1)
    numbers = list(re.finditer(r'-?\d+(?:\.\d+)?', d))
    if len(numbers) < 4 or re.search(r'[a-z]', re.sub(r'e-?\d', '', d)):
        return tag                                   # relative commands: leave alone
    values = [float(n.group(0)) for n in numbers]
    x1, y1 = values[-2], values[-1]
    k = len(values) - 4
    while k >= 0 and (values[k], values[k + 1]) == (x1, y1):
        k -= 2
    if k < 0:
        return tag
    x2, y2 = _back(values[k], values[k + 1], x1, y1)
    start, end = numbers[-2].start(), numbers[-1].end()
    new_d = d[:start] + f"{x2:.1f}" + d[numbers[-2].end():numbers[-1].start()] + f"{y2:.1f}" + d[end:]
    return tag.replace(f'd="{d}"', f'd="{new_d}"')


def sharpen(svg):
    svg = re.sub(r'<marker\b.*?</marker>', _marker, svg, flags=re.S)

    def element(match):
        tag = match.group(0)
        if "marker-end" not in tag:
            return tag
        if tag.startswith("<line"):
            return _line(tag)
        if tag.startswith("<path"):
            return _path(tag)
        return tag

    return re.sub(r'<(?:line|path)\b[^>]*>', element, svg)


# Material 3 containers --------------------------------------------------------------

SHADOW = ('<filter id="m3-shadow" x="-10%" y="-20%" width="120%" height="160%">'
          '<feDropShadow dx="0" dy="1" stdDeviation="1" flood-color="#000" flood-opacity="0.18"/>'
          '<feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#000" flood-opacity="0.10"/>'
          '</filter>')
KEEP_STROKES = {"#1B2430", "#C9D1DB"}          # pixel grids of the image figures


def _attr(tag, name):
    found = re.search(rf'\b{name}="([^"]*)"', tag)
    return found.group(1) if found else None


def _set(tag, name, value):
    if _attr(tag, name) is None:
        return tag.replace("<rect", f'<rect {name}="{value}"', 1)
    return re.sub(rf'\b{name}="[^"]*"', f'{name}="{value}"', tag, count=1)


def _mix(fill, accent, amount):
    """`fill` moved `amount` of the way toward `accent`: a deeper tonal container."""
    if not (fill.startswith("#") and accent.startswith("#") and len(fill) == len(accent) == 7):
        return fill
    a = [int(fill[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(accent[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


def _container(match):
    tag = match.group(0)
    fill, stroke = _attr(tag, "fill"), _attr(tag, "stroke")
    if not fill or fill == "none" or not stroke or stroke == "none":
        return tag
    if "stroke-dasharray" in tag or stroke.upper() in KEEP_STROKES:
        return tag
    if float(_attr(tag, "stroke-width") or 1) >= 2:
        return tag                                       # an emphasis ring stays
    height = float(_attr(tag, "height") or 0)
    if height and float(_attr(tag, "rx") or 0) >= height / 2 - 1:
        return tag                                       # a chip keeps its outline
    tag = _set(tag, "stroke", "none")
    if fill.upper() not in ("#FFFFFF", "#F8F8F8") and not fill.upper().startswith("#F1F"):
        tag = _set(tag, "fill", _mix(fill, stroke, 0.18))
    if height >= 24 and float(_attr(tag, "rx") or 0) < 8:
        tag = _set(tag, "rx", "8")
    if fill.upper() in ("#FFFFFF", "#F8F8F8") or fill.upper().startswith("#F1F"):
        tag = re.sub(r'\sfilter="[^"]*"', "", tag)
        tag = tag.replace("<rect", '<rect filter="url(#m3-shadow)"', 1)
    return tag


def materialize(svg):
    svg = re.sub(r'<rect\b[^>]*>', _container, svg)
    return re.sub(r'(<svg\b[^>]*>)', lambda m: m.group(1) + f'\n  <defs>{SHADOW}</defs>', svg, count=1)
