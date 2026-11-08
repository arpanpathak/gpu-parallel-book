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
    return _head(ident, colour, 1.0) + _head(ident + "-small", colour, 0.6)


def _head(ident, colour, scale):
    HEAD, HALF_WIDTH, INSET = 12.0 * scale, 4.5 * scale, 6.0 * scale
    return (f'<marker id="{ident}" viewBox="0 0 {HEAD:g} {2 * HALF_WIDTH:g}" refX="{INSET:g}" '
            f'refY="{HALF_WIDTH:g}" markerUnits="userSpaceOnUse" markerWidth="{HEAD:g}" '
            f'markerHeight="{2 * HALF_WIDTH:g}" orient="auto">'
            f'<path d="M0,0 L{HEAD:g},{HALF_WIDTH:g} L0,{2 * HALF_WIDTH:g} L{3.5 * scale:g},{HALF_WIDTH:g} Z" '
            f'fill="{colour}"/></marker>')


SHORT = 18.0          # a last segment shorter than this gets the small head


def _back(x0, y0, x1, y1, inset=INSET):
    """The point INSET units before (x1, y1) on the way from (x0, y0); for a segment
    shorter than that, a point just after (x0, y0), so the direction is kept."""
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return x1, y1
    step = inset if length > inset + 0.5 else length - 0.5
    return x1 - (x1 - x0) * step / length, y1 - (y1 - y0) * step / length


def _line(tag):
    def get(name):
        return float(re.search(rf'\b{name}="([-\d.]+)"', tag).group(1))
    x1, y1, x2, y2 = get("x1"), get("y1"), get("x2"), get("y2")
    short = math.hypot(x2 - x1, y2 - y1) < SHORT
    x2, y2 = _back(x1, y1, x2, y2, INSET * (0.6 if short else 1))
    if short:
        tag = re.sub(r'marker-end="url\(#([^)]+)\)"', r'marker-end="url(#\1-small)"', tag)
    tag = re.sub(r'\bx2="[-\d.]+"', f'x2="{x2:.1f}"', tag)
    return re.sub(r'\by2="[-\d.]+"', f'y2="{y2:.1f}"', tag)


TOKEN = re.compile(r'[MLHVCSQTZ]|-?\d+(?:\.\d+)?')
ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "Z": 0}


def _path(tag):
    d = re.search(r'\bd="([^"]+)"', tag).group(1)
    if re.search(r'[a-df-z]', d):
        return tag                                   # relative commands: leave alone
    tokens = TOKEN.findall(d)
    commands, x, y, command = [], 0.0, 0.0, None
    i = 0
    while i < len(tokens):
        if tokens[i] in ARITY:
            command = tokens[i]
            i += 1
            if command == "Z":
                continue
        args = [float(v) for v in tokens[i:i + ARITY[command]]]
        i += ARITY[command]
        if command == "H":
            points = [(args[0], y)]
        elif command == "V":
            points = [(x, args[0])]
        else:
            points = [(args[k], args[k + 1]) for k in range(0, len(args), 2)]
        commands.append((command, (x, y), points))
        x, y = points[-1]
    if not commands:
        return tag
    command, start, points = commands[-1]
    end = points[-1]
    before = points[-2] if len(points) > 1 and points[-2] != end else start
    short = math.hypot(end[0] - before[0], end[1] - before[1]) < SHORT
    nx, ny = _back(before[0], before[1], end[0], end[1], INSET * (0.6 if short else 1))
    if short:
        tag = re.sub(r'marker-end="url\(#([^)]+)\)"', r'marker-end="url(#\1-small)"', tag)
    if command in ("H", "V"):                        # keep the straight segment exact
        command, points = "L", [(nx, ny)]
    else:
        points = points[:-1] + [(nx, ny)]
    last = command + " " + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    head = " ".join(_render(c, p) for c, _, p in commands[:-1])
    return tag.replace(f'd="{d}"', f'd="{head} {last}".strip()'.replace('".strip()', '"'))


def _render(command, points):
    if command == "H":
        return f"H{points[0][0]:g}"
    if command == "V":
        return f"V{points[0][1]:g}"
    return command + " " + " ".join(f"{px:g},{py:g}" for px, py in points)


def sharpen(svg):
    svg = re.sub(r'<marker\b.*?</marker>', _marker, svg, flags=re.S)

    def element(match):
        tag = match.group(0)
        if "marker-end" not in tag:
            return tag
        tag = re.sub(r'\sstroke-linecap="[^"]*"', "", tag)
        tag = tag.replace("marker-end=", 'stroke-linecap="butt" marker-end=', 1)
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
    if height < 20:
        return tag                                       # tiny cells keep their outline
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
