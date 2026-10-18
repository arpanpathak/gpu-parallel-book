"""The drawing kit for the book's system diagrams.

One visual language for every figure: paper cards with a soft shadow and a
coloured edge, cylinders for stores, sheets for text, and wires that only run
horizontally and vertically. A figure is a list of parts handed to `save`.

Two rules are checked while drawing, so a figure that breaks one does not get
written: text has to fit the box it is in, and a wire may not run at an angle.

Brand marks in `icons/` are from Simple Icons (CC0). The trademarks are their
owners'; they are used here only to say which product a box stands for.
"""
import os
import re
from html import escape
from pathlib import Path

ICONS = Path(__file__).resolve().parent / "icons"

PAPER = "#FBFAF6"
CARD = "#FFFFFF"
INK = "#1B2430"
MUTED = "#5D6B7A"
EDGE = "#D5DBE3"
WIRE = "#3F5873"

# kind: (accent, tint)
KINDS = {
    "client": ("#2F6FDB", "#EAF1FD"),
    "service": ("#1F8A70", "#E4F4EF"),
    "model": ("#5E9400", "#EEF6DC"),
    "store": ("#B7791F", "#FBF1DC"),
    "outside": ("#7A4FD0", "#F0EAFB"),
    "danger": ("#CF4526", "#FCEBE6"),
    "idle": ("#7F8EA0", "#F1F4F7"),
    "good": ("#2E8B3D", "#E7F5E9"),
}

BRAND = {
    "nvidia": "#76B900", "tailscale": "#242424", "rust": "#A8431A", "huggingface": "#E8A100",
    "python": "#3776AB", "firefoxbrowser": "#E66000", "github": "#181717", "anthropic": "#191919",
    "claude": "#D97757", "openai": "#412991", "linux": "#1B2430", "sqlite": "#003B57",
    "searxng": "#3050FF", "ollama": "#1B2430", "googlechrome": "#4285F4", "gnubash": "#3E474A",
    "git": "#F03C2E", "letsencrypt": "#003A70", "namecheap": "#DE3723", "alibabacloud": "#FF6A00",
}

# Glyphs drawn for this book, on the same 24 x 24 grid as the brand marks.
GLYPHS = {
    "terminal": "M3 4h18a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zm3.3 4.3-1.4 1.4L7.2 12l-2.3 2.3 1.4 1.4L10 12zM11 14v2h7v-2z",
    "browser": "M3 4h18a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zm1 5v9h16V9zm1.5-3a1 1 0 1 0 0 2 1 1 0 0 0 0-2zm3 0a1 1 0 1 0 0 2 1 1 0 0 0 0-2zm3 0a1 1 0 1 0 0 2 1 1 0 0 0 0-2z",
    "gear": "M10.3 2h3.4l.5 2.6 1.7.7 2.2-1.5 2.4 2.4-1.5 2.2.7 1.7 2.6.5v3.4l-2.6.5-.7 1.7 1.5 2.2-2.4 2.4-2.2-1.5-1.7.7-.5 2.6h-3.4l-.5-2.6-1.7-.7-2.2 1.5-2.4-2.4 1.5-2.2-.7-1.7L2 13.7v-3.4l2.6-.5.7-1.7-1.5-2.2 2.4-2.4 2.2 1.5 1.7-.7zM12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z",
    "chip": "M8 2h2v3h4V2h2v3h2a1 1 0 0 1 1 1v2h3v2h-3v4h3v2h-3v2a1 1 0 0 1-1 1h-2v3h-2v-3h-4v3H8v-3H6a1 1 0 0 1-1-1v-2H2v-2h3v-4H2V8h3V6a1 1 0 0 1 1-1h2zM7 7v10h10V7zm2.5 2.5h5v5h-5z",
    "doc": "M6 2h8l5 5v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm7 1.5V8h4.5zM8 11v1.6h8V11zm0 3.2v1.6h8v-1.6zm0 3.2V19h5v-1.6z",
    "key": "M14.5 2a7.5 7.5 0 0 0-7.2 9.6L2 16.9V22h5.1v-2.5h2.5V17h2.4l1-1a7.5 7.5 0 1 0 1.5-14zm2.2 4.3a2 2 0 1 1 0 4 2 2 0 0 1 0-4z",
    "search": "M10.5 3a7.5 7.5 0 0 1 5.9 12.1l4.7 4.8-1.8 1.8-4.8-4.7A7.5 7.5 0 1 1 10.5 3zm0 2.5a5 5 0 1 0 0 10 5 5 0 0 0 0-10z",
    "clock": "M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20zm0 2.4a7.6 7.6 0 1 0 0 15.2 7.6 7.6 0 0 0 0-15.2zM11 7h2v4.6l3.3 2-1 1.7-4.3-2.6z",
    "stack": "M12 2 2 7l10 5 10-5zm-7.8 8.6L2 11.7l10 5 10-5-2.2-1.1L12 14.5zm0 4.7L2 16.4l10 5 10-5-2.2-1.1L12 19.2z",
    "swap": "M7 4 2 9l5 5v-3.5h9V8H7zm10 6v3.5H8V16h9v3.5l5-5z",
    "mask": "M4 3h16v9a8 8 0 0 1-16 0zM7.5 7h3v2.2h-3zM13.5 7h3v2.2h-3zM8.2 12.6h7.6a3.8 3.8 0 0 1-7.6 0z",
    "listen": "M4 3h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-8.5L6 21.5V17H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM12 14.3l-3.4-3.2a2.1 2.1 0 0 1 3.4-2.4 2.1 2.1 0 0 1 3.4 2.4z",
    "passport": "M6 2h12a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zM12 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 8.2a1.8 1.8 0 1 1 0 3.6 1.8 1.8 0 0 1 0-3.6zM8.5 16.5h7v1.6h-7z",
    "pencil": "M3 21l1.5-5.5L15.2 4.8l4 4L8.5 19.5zM16.6 3.4l1.3-1.3a1 1 0 0 1 1.4 0l2.6 2.6a1 1 0 0 1 0 1.4l-1.3 1.3z",
    "note": "M10.5 3H19v4h-6v10.5a3.5 3.5 0 1 1-2.5-3.35z",
    "curve": "M10 2h4v4h-4zM2 15h5v5H2zM17 15h5v5h-5zM3.6 15a8.4 8.4 0 0 1 16.8 0h-1.8a6.6 6.6 0 0 0-13.2 0z",
    "laptop": "M5 4h14a1 1 0 0 1 1 1v10H4V5a1 1 0 0 1 1-1zm1 2v7h12V6zM2 16.5h20V18a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z",
    "lock": "M7 10V7a5 5 0 0 1 10 0v3h1a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1zm2.2 0h5.6V7a2.8 2.8 0 0 0-5.6 0zM12 13.5a1.6 1.6 0 0 0-.8 3v2h1.6v-2a1.6 1.6 0 0 0-.8-3z",
    "eye": "M12 5c5 0 8.5 3.5 10 7-1.5 3.5-5 7-10 7S3.5 15.5 2 12c1.5-3.5 5-7 10-7zm0 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0 2.2a1.8 1.8 0 1 1 0 3.6 1.8 1.8 0 0 1 0-3.6z",
    "cup": "M4 8h13v2h1.5a3.5 3.5 0 0 1 0 7H17a5 5 0 0 1-5 4H9a5 5 0 0 1-5-5zm13 4v3h1.5a1.5 1.5 0 0 0 0-3zM7 2h1.6v4H7zm3.2 0h1.6v4h-1.6zm3.2 0H15v4h-1.6z",
}

# A person's look: (skin, hair colour, hair style, top colour, extra).
PEOPLE = {
    "long": ("#C68B5E", "#2B1D16", "long", "#1F8A70", None),
    "beard": ("#F0C8A0", "#3A2A1E", "short", "#2F6FDB", "beard"),
    "curly": ("#8A5A3C", "#1B1512", "curly", "#D9A21B", None),
    "glasses": ("#F4D3B5", "#B5532A", "short", "#5E9400", "glasses"),
    "bun": ("#E0AC84", "#5A3A22", "bun", "#7A4FD0", None),
    "cap": ("#A9744B", "#1B1512", "short", "#CF4526", "cap"),
}

STYLE = f"""  <style>
    text {{ font-family: Inter, "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; fill: {INK}; }}
    .title {{ font-size: 18px; font-weight: 650; }}
    .lede {{ font-size: 12.5px; fill: {MUTED}; }}
    .name {{ font-size: 14px; font-weight: 650; }}
    .sub {{ font-size: 12px; fill: {MUTED}; }}
    .note {{ font-size: 12px; fill: {MUTED}; }}
    .tab {{ font-size: 11.5px; font-weight: 650; letter-spacing: 0.04em; fill: {MUTED}; }}
    .mono {{ font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }}
    .dim {{ fill: #8794A3; }}
    .strong {{ font-weight: 700; }}
    .trap {{ fill: {KINDS['danger'][0]}; font-weight: 700; }}
    .pill {{ font-size: 11px; font-weight: 600; }}
    .wire {{ fill: none; stroke: {WIRE}; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }}
    .wire.danger {{ stroke: {KINDS['danger'][0]}; stroke-dasharray: 6 5; }}
    .wire.idle {{ stroke: {KINDS['idle'][0]}; stroke-dasharray: 2 5; }}
  </style>
  <defs>
    <filter id="lift" x="-10%" y="-10%" width="120%" height="135%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1B2430" flood-opacity="0.13"/>
    </filter>
    <linearGradient id="thor-body" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#C6CCD5"/><stop offset="1" stop-color="#979EA9"/>
    </linearGradient>
    <linearGradient id="switch-body" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3B4656"/><stop offset="1" stop-color="#232B37"/>
    </linearGradient>
    <marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">
      <path d="M0.5,1 L9,5 L0.5,9 Z" fill="{WIRE}"/>
    </marker>
    <marker id="head-danger" viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">
      <path d="M0.5,1 L9,5 L0.5,9 Z" fill="{KINDS['danger'][0]}"/>
    </marker>
    <marker id="head-idle" viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">
      <path d="M0.5,1 L9,5 L0.5,9 Z" fill="{KINDS['idle'][0]}"/>
    </marker>
  </defs>"""


class DoesNotFit(Exception):
    """Text that is wider than the room it was given, or a wire at an angle."""


def width_of(body, size=12.0, mono=False, bold=False):
    """A cautious estimate of how wide `body` renders, in pixels."""
    per_char = 0.61 if mono else (0.6 if bold else 0.55)
    return len(body) * size * per_char


def must_fit(body, room, size=12.0, mono=False, bold=False):
    """Raises when `body` is wider than `room`. With FIGKIT_REPORT set it prints instead,
    so one run lists every label that needs more room."""
    if width_of(body, size, mono, bold) <= room:
        return
    problem = f"{body!r} needs {width_of(body, size, mono, bold):.0f} px, has {room:.0f}"
    if not os.environ.get("FIGKIT_REPORT"):
        raise DoesNotFit(problem)
    print("does not fit:", problem)


def text(x, y, body, cls="note", anchor="start", fill=None):
    colour = f' style="fill:{fill}"' if fill else ""
    return (f'  <text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{colour}'
            f' xml:space="preserve">{escape(body)}</text>')


def icon(name, x, y, size=20, fill=None):
    """A brand mark from icons/, or one of the kit's own glyphs, drawn `size` wide."""
    if name in GLYPHS:
        path, colour = GLYPHS[name], fill or INK
    else:
        source = (ICONS / f"{name}.svg").read_text()
        path, colour = re.search(r'<path d="([^"]+)"', source).group(1), fill or BRAND.get(name, INK)
    return (f'  <g transform="translate({x:.1f},{y:.1f}) scale({size / 24:.4f})">'
            f'<path d="{path}" fill="{colour}" fill-rule="evenodd"/></g>')


def person(x, y, look, size=56):
    """A friendly head-and-shoulders portrait in a round badge, `size` wide, with its
    top-left corner at (x, y). `look` is a key of PEOPLE."""
    skin, hair, style, top, extra = PEOPLE[look]
    clip = f"person-{look}-{int(x)}-{int(y)}"
    fringe = "M21.5 26 C20 15 26 12 32 12 C38 12 44 15 42.5 26 C41 20.5 38 18.5 32 18.5 C26 18.5 23 20.5 21.5 26 Z"
    parts = [
        f'  <g transform="translate({x:.1f},{y:.1f}) scale({size / 64:.4f})">',
        f'<clipPath id="{clip}"><circle cx="32" cy="32" r="30"/></clipPath>',
        f'<circle cx="32" cy="32" r="30" fill="#FFFFFF" stroke="{EDGE}" stroke-width="1.6" filter="url(#lift)"/>',
        f'<g clip-path="url(#{clip})">',
        f'<circle cx="32" cy="32" r="30" fill="{top}" opacity="0.14"/>',
    ]
    if style == "long":
        parts.append(f'<path d="M18 50 C15 28 19 11 32 11 C45 11 49 28 46 50 C42 50 41 44 41 36 L23 36 C23 44 22 50 18 50 Z" fill="{hair}"/>')
    parts += [
        f'<path d="M8 66 C8 49 19 43 32 43 C45 43 56 49 56 66 Z" fill="{top}"/>',
        f'<path d="M27.5 34 h9 v8.5 a4.5 4.5 0 0 1 -9 0 Z" fill="{skin}"/>',
        f'<path d="M27.5 37.5 a8 8 0 0 0 9 0 v3 a6 6 0 0 1 -9 0 Z" fill="{INK}" opacity="0.13"/>',
        f'<ellipse cx="32" cy="26.5" rx="10.2" ry="11.2" fill="{skin}"/>',
    ]
    if extra == "beard":
        parts.append(f'<path d="M22 27 C22 38.5 27.5 40 32 40 C36.5 40 42 38.5 42 27 C40 32.5 36.5 33.2 32 33.2 C27.5 33.2 24 32.5 22 27 Z" fill="{hair}"/>')
    if style == "curly":
        parts += [f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{hair}"/>'
                  for cx, cy, r in ((21.5, 22, 5.2), (24.5, 16, 5.6), (30, 12.5, 5.8), (36, 13, 5.6), (41, 17, 5.4), (43, 23, 4.8))]
    else:
        parts.append(f'<path d="{fringe}" fill="{hair}"/>')
    if style == "bun":
        parts.append(f'<circle cx="32" cy="9.5" r="5.2" fill="{hair}"/>')
    if extra == "cap":
        parts.append(f'<path d="M21 21 C21 12 26 9.5 32 9.5 C38 9.5 43 12 43 21 Z M43 18.5 h7.5 a1.6 1.6 0 0 1 0 3.2 H42 Z" fill="{top}"/>')
    parts += [
        f'<circle cx="28" cy="27" r="1.25" fill="{INK}"/><circle cx="36" cy="27" r="1.25" fill="{INK}"/>',
        f'<path d="M28.3 31.4 Q32 34.6 35.7 31.4" fill="none" stroke="{"#FFFFFF" if extra == "beard" else INK}" stroke-width="1.3" stroke-linecap="round"/>',
    ]
    if extra == "glasses":
        parts.append(f'<path d="M24.6 27 a3.4 3.4 0 1 0 6.8 0 a3.4 3.4 0 1 0 -6.8 0 M32.6 27 a3.4 3.4 0 1 0 6.8 0 a3.4 3.4 0 1 0 -6.8 0 M31.4 26.6 h1.2"'
                     f' fill="none" stroke="{INK}" stroke-width="1.1"/>')
    parts += ["</g>", "</g>"]
    return "\n".join(parts)


# The five cubs of the cluster: one cub for each Thor, each a different kind.
# colours: fur light, fur dark, fur flat, markings, cloth top, cloth bottom, accent, eyes.
CUBS = {
    "volt": dict(colours=("#B3C0D9", "#8F9DBA", "#97A5C1", "#3E4F78", "#4A6396", "#2E4170", "#5ED3F3", "#F2B84B"),
                 head=(116, 98), body=(72, 62), foot=27, ears="round", cheeks="jag", hat="helm", marks="stripes", eyes="round",
                 mouth="fang", tail="curl", outfit="cape", pose="hammer", emblem="bolt"),
    "ember": dict(colours=("#F9C48E", "#E88D48", "#EFA060", "#A3421D", "#D0482F", "#8E2A1E", "#FFC93C", "#FFE08A"),
                  head=(108, 94), body=(64, 58), foot=24, ears="flame", cheeks="jag", hat="tuft", marks="blaze", eyes="fierce",
                  mouth="grin", tail="flame", outfit="bandana", pose="cheer", emblem="flame"),
    "frost": dict(colours=("#F5FAFE", "#CCE1F2", "#DCEBF7", "#6FA3C7", "#3AA6CC", "#1F7397", "#9BE7FF", "#63C1F5"),
                  head=(128, 102), body=(90, 66), foot=32, ears="small", cheeks="fluff", hat="beanie", marks="spots", eyes="sleepy",
                  mouth="tiny", tail="fluffy", outfit="scarf", pose="hold", emblem="crystal"),
    "leaf": dict(colours=("#D2E8B0", "#9DC972", "#B0D686", "#4E7F2E", "#2F8551", "#1D5A35", "#F4A6B6", "#F2B84B"),
                 head=(102, 98), body=(58, 56), foot=22, ears="leaf", cheeks="none", hat="sprout", marks="freckles", eyes="happy",
                 mouth="open", tail="sprig", outfit="collar", pose="wave", emblem="leaf"),
    "nova": dict(colours=("#C3B2E4", "#8E79C0", "#A08CCD", "#47327D", "#4A3690", "#22174A", "#FFD86B", "#F7A8D8"),
                 head=(112, 94), body=(68, 60), foot=26, ears="bat", cheeks="jag", hat="wizard", marks="moon", eyes="star",
                 mouth="smirk", tail="star", outfit="cloak", pose="wand", emblem="star"),
}

# Emblems, each drawn about 30 units tall around the origin.
EMBLEMS = {
    "bolt": "M3,-15 L-9,2 L-1,2 L-6,15 L9,-4 L1,-4 Z",
    "flame": "M0,-16 C7,-8 11,-2 10,6 C9,12 5,15 0,15 C-5,15 -10,11 -10,5 C-10,0 -7,-3 -5,-7 C-3,-4 -2,-3 -1,-3 C1,-7 1,-12 0,-16 Z",
    "crystal": "M0,-16 L9,-5 L5,15 L-5,15 L-9,-5 Z",
    "leaf": "M0,-16 C11,-9 12,5 0,16 C-12,5 -11,-9 0,-16 Z",
    "star": "M0,-16 L4.2,-5.8 L15.2,-4.9 L6.8,2.2 L9.4,12.9 L0,7.2 L-9.4,12.9 L-6.8,2.2 L-15.2,-4.9 L-4.2,-5.8 Z",
}
FIRE, GREEN = "#F2622E", "#6FB13F"


def emblem(name, cx, cy, scale, fill, width=2.5, turn=0):
    return (f'<path d="{EMBLEMS[name]}" transform="translate({cx},{cy}) rotate({turn}) scale({scale})" fill="{fill}"'
            f' stroke-width="{width / scale:.2f}"/>')


def both(shape):
    """A drawing and its mirror image across the cub's centre line, x = 210."""
    return shape + f'<g transform="translate(420,0) scale(-1,1)">{shape}</g>'


def cub(style, x, y, size=150):
    """One of the cluster's five cubs, `size` wide and 1.05 `size` tall, top-left at (x, y).

    `style` is a key of CUBS. The five differ in build, ears, eyes, mouth, tail, clothes,
    headgear and pose, so each has a silhouette of its own."""
    spec = CUBS[style]
    light, deep, flat, stripe, cloth_top, cloth_low, accent, iris = spec["colours"]
    (hx, hy), (bx, by), foot, mark = spec["head"], spec["body"], spec["foot"], spec["emblem"]
    ink, steel, white = "#131B2E", f"url(#cub-steel-{style})", "#FFFFFF"
    fur, cloth = f"url(#cub-fur-{style})", f"url(#cub-cloth-{style})"
    limb = lambda path, wide: (f'<path d="{path}" fill="none" stroke-width="{wide + 10}"/>'
                               f'<path d="{path}" fill="none" stroke="{flat}" stroke-width="{wide}"/>')
    on_head = lambda art: f'<g transform="translate(210,206) scale({hx / 116:.3f},{hy / 98:.3f}) translate(-210,-206)">{art}</g>'
    ring = lambda cx, cy, r=7: f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{stripe}" stroke-width="4"/>'

    ears = {
        "round": both(f'<circle cx="124" cy="112" r="34" fill="{flat}"/><circle cx="124" cy="115" r="17" fill="#E9B7C1" stroke-width="3"/>'),
        "flame": both(f'<path d="M96,150 L98,56 L168,108 Z" fill="{flat}"/><path d="M112,128 L113,84 L146,108 Z" fill="#E9B7C1" stroke-width="3"/>'
                      + emblem("flame", 98, 48, 1.3, FIRE, 3, -10)),
        "small": both(f'<circle cx="122" cy="126" r="25" fill="{flat}"/><circle cx="122" cy="128" r="11" fill="#E9B7C1" stroke-width="3"/>'),
        "leaf": both(emblem("leaf", 86, 142, 4.2, GREEN, 4, -64) + '<path d="M54,158 L120,128" fill="none" stroke="#4E7F2E" stroke-width="3"/>'),
        "bat": both(f'<path d="M100,162 L76,28 L178,100 Z" fill="{flat}"/><path d="M112,134 L96,62 L152,104 Z" fill="{iris}" stroke-width="3"/>'),
    }[spec["ears"]]
    cheeks = {
        "jag": both(f'<path d="M104,184 L78,206 L100,214 L80,238 L106,240 L94,262 L126,250 Z" fill="{flat}"/>'),
        "spiky": both(f'<path d="M100,176 L58,194 L92,210 L56,238 L98,242 L78,274 L128,254 Z" fill="{flat}"/>'),
        "fluff": both(f'<circle cx="84" cy="222" r="23" fill="{flat}"/><circle cx="96" cy="254" r="21" fill="{flat}"/>'),
        "none": "",
    }[spec["cheeks"]]
    band = lambda fill: f'<rect x="112" y="140" width="196" height="24" rx="11" fill="{fill}"/>'
    hat = {
        "helm": (f'<path d="M118,152 C120,86 300,86 302,152 Z" fill="{steel}"/>'
                 f'<path d="M150,104 C176,92 206,90 226,92" fill="none" stroke="{white}" stroke-width="6" opacity="0.8"/>'
                 f'<path d="M202,94 C204,80 206,72 210,62 C214,72 216,80 218,94 Z" fill="{cloth_low}" stroke-width="4"/>'
                 + band("#8E9AB3") + f'<path d="M210,138 L224,152 L210,166 L196,152 Z" fill="{accent}" stroke-width="4"/>'
                 + both(f'<path d="M118,146 C92,138 70,112 58,74 C80,90 92,96 104,100 C92,86 86,70 88,54 C106,80 122,108 130,136 Z" fill="{steel}"/>')),
        "tuft": (emblem("flame", 180, 118, 1.9, FIRE, 4, -24) + emblem("flame", 240, 118, 1.9, FIRE, 4, 24)
                 + emblem("flame", 210, 100, 3.0, FIRE, 4) + emblem("flame", 210, 112, 1.6, accent, 3)),
        "beanie": (f'<path d="M114,152 C118,70 302,70 306,152 Z" fill="{cloth}"/>'
                   f'<path d="M136,124 L154,110 L172,124 L191,110 L210,124 L229,110 L248,124 L266,110 L284,124" fill="none" stroke="{white}" stroke-width="5"/>'
                   f'<rect x="106" y="138" width="208" height="28" rx="13" fill="{white}"/><circle cx="210" cy="70" r="23" fill="{white}"/>'),
        "sprout": ('<path d="M210,116 C206,98 210,84 210,70" fill="none" stroke-width="13"/>'
                   '<path d="M210,116 C206,98 210,84 210,70" fill="none" stroke="#4E7F2E" stroke-width="5"/>'
                   + emblem("leaf", 182, 58, 2.5, GREEN, 4, -60) + emblem("leaf", 240, 52, 3.0, GREEN, 4, 54)
                   + "".join(f'<circle cx="{210 + 12 * dx}" cy="{118 + 12 * dy}" r="8" fill="{accent}" stroke-width="3"/>'
                             for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                   + f'<circle cx="210" cy="118" r="7" fill="#FFD86B" stroke-width="3"/>'),
        "wizard": (f'<path d="M122,152 C150,112 198,56 268,10 C256,62 278,112 298,152 Z" fill="{cloth}"/>'
                   + emblem("star", 226, 104, 1.1, accent, 3, 10) + emblem("star", 258, 50, 0.5, white, 2)
                   + f'<ellipse cx="210" cy="152" rx="130" ry="17" fill="{cloth_low}"/>'),
    }[spec["hat"]]
    marks = {
        "stripes": (f'<g stroke="{stripe}" stroke-width="6"><path d="M186,170 L190,182"/><path d="M234,170 L230,182"/>'
                    + both('<path d="M100,196 L124,202"/><path d="M98,218 L120,220"/>') + "</g>" + emblem("bolt", 210, 183, 0.72, accent, 2.5)),
        "blaze": (emblem("flame", 210, 178, 1.0, FIRE, 2.5)
                  + f'<g stroke="{stripe}" stroke-width="6" fill="none">' + both('<path d="M102,198 L122,206 L102,216"/>') + "</g>"),
        "spots": "".join(ring(cx, cy) for cx, cy in ((184, 174), (236, 174), (210, 164))) + both(ring(108, 204) + ring(116, 228, 6)),
        "freckles": (emblem("leaf", 210, 176, 0.7, GREEN, 2.5)
                     + f'<g fill="{stripe}" stroke="none">' + both('<circle cx="124" cy="238" r="3.5"/><circle cx="136" cy="246" r="3.5"/>') + "</g>"),
        "moon": (f'<path d="M214,164 A14,14 0 1 0 224,186 A11,11 0 1 1 214,164 Z" fill="{accent}" stroke-width="2.5"/>'
                 + both(emblem("star", 114, 212, 0.5, accent, 2))),
    }[spec["marks"]]
    pupil = lambda cx, px, rx=12: (f'<ellipse cx="{px}" cy="217" rx="{rx}" ry="16" fill="{ink}" stroke="none"/>'
                                   f'<circle cx="{px + 6}" cy="207" r="6" fill="{white}" stroke="none"/><circle cx="{px - 6}" cy="225" r="3" fill="{white}" stroke="none"/>')
    ball = lambda cx, rx=23, ry=26: f'<ellipse cx="{cx}" cy="214" rx="{rx}" ry="{ry}" fill="{iris}" stroke-width="4"/>'
    eyes = {
        "round": ball(164) + pupil(164, 166) + ball(256) + pupil(256, 254),
        "fierce": (ball(164) + pupil(164, 168, 8) + ball(256) + pupil(256, 252, 8)
                   + both('<path d="M136,182 L192,200" fill="none" stroke-width="10"/>')),
        "sleepy": both(f'<path d="M141,212 A23,24 0 0 0 187,212 Z" fill="{iris}" stroke-width="4"/>'
                       f'<path d="M153,212 A12,14 0 0 0 177,212 Z" fill="{ink}" stroke="none"/><path d="M136,212 H192" fill="none" stroke-width="6"/>'),
        "happy": both('<path d="M142,224 Q164,190 186,224" fill="none" stroke-width="7"/>'),
        "star": (ball(164, 26, 29) + pupil(164, 166, 14) + ball(256, 26, 29) + pupil(256, 254, 14)
                 + f'<g stroke="none">{emblem("star", 160, 220, 0.42, white)}{emblem("star", 248, 220, 0.42, white)}</g>'),
    }[spec["eyes"]]
    nose = '<path d="M196,236 Q210,229 224,236 Q219,250 210,252 Q201,250 196,236 Z" fill="#D98C99" stroke-width="4"/>'
    mouth = nose + {
        "fang": ('<path d="M210,252 L210,260" stroke-width="4"/><path d="M190,259 Q200,272 210,260 Q220,272 230,259" fill="none" stroke-width="4"/>'
                 f'<path d="M216,265 L220,276 L224,264 Z" fill="{white}" stroke-width="2.5"/>'),
        "grin": ('<path d="M184,257 Q210,296 236,257 Z" fill="#7A2330" stroke-width="4"/>'
                 f'<path d="M193,259 L199,273 L205,260 Z" fill="{white}" stroke-width="2.5"/><path d="M215,260 L221,273 L227,259 Z" fill="{white}" stroke-width="2.5"/>'),
        "tiny": '<path d="M210,252 L210,258" stroke-width="4"/><path d="M201,261 Q210,268 219,261" fill="none" stroke-width="4"/>',
        "open": ('<path d="M190,257 Q210,296 230,257 Z" fill="#C7455E" stroke-width="4"/>'
                 '<ellipse cx="210" cy="276" rx="10" ry="6" fill="#F4A6B6" stroke="none"/>'),
        "smirk": ('<path d="M210,252 L210,259" stroke-width="4"/><path d="M194,260 Q212,272 232,255" fill="none" stroke-width="4"/>'
                  f'<path d="M218,264 L222,275 L227,261 Z" fill="{white}" stroke-width="2.5"/>'),
    }[spec["mouth"]]
    curl = "M152,388 C102,392 66,370 64,328 C62,298 84,282 104,292"
    tail = {
        "curl": (limb(curl, 20) + f'<g stroke="{stripe}" stroke-width="5"><path d="M120,384 L116,396"/><path d="M90,372 L80,380"/>'
                 '<path d="M70,346 L58,348"/><path d="M70,312 L58,306"/></g>'),
        "flame": limb("M152,392 C112,404 74,400 50,372", 18) + emblem("flame", 42, 336, 2.7, FIRE, 4, -16) + emblem("flame", 44, 346, 1.4, accent, 3, -16),
        "fluffy": (limb("M150,394 C96,402 54,376 52,328 C52,298 76,284 100,298", 30) + ring(118, 394, 6) + ring(84, 380) + ring(60, 350, 6)
                   + ring(58, 316) + ring(82, 296, 5)),
        "sprig": limb("M152,392 C112,398 80,382 70,342", 9) + emblem("leaf", 64, 318, 2.3, GREEN, 4, -22),
        "star": limb("M152,390 C100,398 62,372 60,332 C60,306 76,296 90,304", 11) + emblem("star", 98, 298, 1.5, accent, 3, -12),
    }[spec["tail"]]
    back = {
        "cape": f'<path d="M150,282 C128,330 112,384 104,428 C170,442 250,442 316,428 C308,384 292,330 270,282 Z" fill="{cloth}"/>',
        "cloak": (f'<path d="M144,280 C112,328 92,384 84,430 C170,446 250,446 336,430 C328,384 308,328 276,280 Z" fill="{cloth}"/>'
                  + emblem("star", 112, 404, 0.6, accent, 2) + emblem("star", 308, 396, 0.7, accent, 2) + emblem("star", 296, 350, 0.4, white, 2)),
    }.get(spec["outfit"], "")
    front = {
        "cape": f'<circle cx="210" cy="328" r="15" fill="{steel}" stroke-width="4"/>' + emblem("bolt", 210, 328, 0.62, accent, 2),
        "cloak": emblem("star", 210, 308, 0.8, accent, 2.5),
        "bandana": (f'<path d="M150,290 C184,304 236,304 270,290 L258,304 L210,350 L162,304 Z" fill="{cloth}"/>' + emblem("flame", 210, 318, 0.7, accent, 2)),
        "scarf": (f'<path d="M236,300 L266,298 L274,374 L244,378 Z" fill="{cloth}"/>'
                  f'<path d="M240,340 L270,336 M242,358 L272,354" fill="none" stroke="{white}" stroke-width="5"/>'
                  f'<rect x="136" y="284" width="148" height="28" rx="14" fill="{cloth}"/>'),
        "collar": (emblem("leaf", 176, 302, 1.3, GREEN, 3, -52) + emblem("leaf", 244, 302, 1.3, GREEN, 3, 52)
                   + f'<circle cx="210" cy="308" r="9" fill="{accent}" stroke-width="3"/>'),
    }[spec["outfit"]]
    arm = "M262,326 C290,312 318,292 336,262"
    raised = limb(arm, 22)
    stick = '<path d="M340,256 L356,172" stroke-width="16"/><path d="M340,256 L356,172" stroke="#8A5A33" stroke-width="8"/>'
    paw = f'<circle cx="338" cy="260" r="18" fill="{flat}"/>'
    idle_arm = f'<ellipse cx="146" cy="356" rx="17" ry="27" fill="{flat}" transform="rotate(14 146 356)"/>'
    pose = {
        "hammer": (idle_arm, raised + stick + '<g transform="rotate(12 360 156)">'
                   f'<rect x="322" y="134" width="76" height="42" rx="7" fill="{steel}"/><rect x="322" y="163" width="76" height="13" rx="4" fill="#8E9AB3" stroke-width="4"/>'
                   f'<path d="M363,139 L352,155 L361,155 L354,169 L370,150 L361,150 Z" fill="{accent}" stroke-width="2.5"/></g>' + paw
                   + emblem("bolt", 412, 112, 1.3, accent, 3, 8) + emblem("bolt", 424, 180, 0.7, accent, 3, -6)),
        "wand": (idle_arm, raised + stick + emblem("star", 360, 142, 2.5, accent, 4, 10) + paw
                 + emblem("star", 414, 104, 0.9, accent, 3) + emblem("star", 424, 186, 0.6, white, 2.5) + emblem("star", 28, 150, 0.7, accent, 3)),
        "cheer": ("", both(raised + paw) + both(emblem("flame", 372, 212, 1.5, FIRE, 3) + emblem("flame", 372, 218, 0.8, accent, 2))),
        "hold": ("", emblem("crystal", 210, 336, 2.3, accent, 3.5) + f'<path d="M204,318 L202,348" fill="none" stroke="{white}" stroke-width="4" opacity="0.9"/>'
                 + both(f'<ellipse cx="176" cy="358" rx="20" ry="16" fill="{flat}"/>')
                 + emblem("crystal", 396, 150, 0.9, accent, 3, 12) + emblem("crystal", 30, 180, 0.7, accent, 3, -12)
                 + f'<g fill="{white}">' + "".join(f'<circle cx="{cx}" cy="{cy}" r="5" stroke-width="2.5"/>' for cx, cy in ((412, 196), (380, 110), (44, 130))) + "</g>"),
        "wave": (idle_arm, raised + f'<circle cx="338" cy="258" r="21" fill="{flat}"/>'
                 '<path d="M330,246 L328,254 M340,243 L340,252 M350,247 L351,255" fill="none" stroke-width="3"/>'
                 f'<g fill="none" stroke="{stripe}" stroke-width="4"><path d="M372,236 Q384,256 374,276"/><path d="M390,226 Q406,256 392,286"/></g>'
                 + "".join(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{accent}" stroke-width="2.5"/>' for cx, cy in ((404, 150), (34, 170), (56, 120)))),
    }[spec["pose"]]
    body_spots = ring(152, 344) + ring(268, 340) + ring(160, 372, 6) if spec["marks"] == "spots" else ""
    return "\n".join([
        f'  <g transform="translate({x:.1f},{y:.1f}) scale({size / 440:.4f})">',
        f'<defs><linearGradient id="cub-cloth-{style}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{cloth_top}"/><stop offset="1" stop-color="{cloth_low}"/></linearGradient>'
        f'<linearGradient id="cub-steel-{style}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F4F7FC"/><stop offset="1" stop-color="#C9D3E4"/></linearGradient>'
        f'<radialGradient id="cub-fur-{style}" cx="0.45" cy="0.35" r="0.75"><stop offset="0" stop-color="{light}"/><stop offset="1" stop-color="{deep}"/></radialGradient></defs>',
        f'<g stroke="{ink}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round">',
        back, tail,
        f'<ellipse cx="210" cy="{418 - by}" rx="{bx}" ry="{by}" fill="{fur}"/>'
        f'<ellipse cx="210" cy="{430 - by}" rx="{bx * 0.58:.0f}" ry="{by * 0.6:.0f}" fill="{white}" opacity="0.75" stroke="none"/>',
        body_spots,
        both(f'<ellipse cx="{210 - bx * 0.53:.0f}" cy="414" rx="{foot}" ry="16" fill="{flat}"/>'
             f'<path d="M{202 - bx * 0.53:.0f},410 v10 M{218 - bx * 0.53:.0f},410 v10" fill="none" stroke-width="3"/>'),
        front, pose[0], cheeks, on_head(ears),
        f'<ellipse cx="210" cy="206" rx="{hx}" ry="{hy}" fill="{fur}"/>',
        on_head(hat + marks),
        f'<ellipse cx="210" cy="252" rx="54" ry="38" fill="{white}" opacity="0.85"/>',
        eyes,
        both('<ellipse cx="140" cy="252" rx="15" ry="8" fill="#F4A6B6" stroke="none" opacity="0.7"/>'),
        mouth,
        f'<g fill="{ink}" stroke="none">' + both('<circle cx="182" cy="256" r="2.5"/><circle cx="176" cy="264" r="2.5"/><circle cx="188" cy="266" r="2.5"/>') + "</g>",
        '<g stroke="#5D6C8D" stroke-width="3">' + both('<path d="M160,262 L108,256"/><path d="M160,270 L112,276"/>') + "</g>",
        pose[1],
        "</g>", "</g>",
    ])


def thor_kit(x, y, w=150):
    """The Jetson AGX Thor developer kit, seen from above its long side: a low silver-grey
    slab on a dark base, with the black ribbed fan grille on top and the ports on the short
    end. `w` wide and 0.38 `w` tall, top-left at (x, y); the front face is the left 0.79 of it."""
    top = lambda u, v: f"{u + 40 * v:.1f},{28 - 26 * v:.1f}"
    ribs = "".join(f'<path d="M{top(11, v / 100)} L{top(73, v / 100)}" stroke="#3A3F48" stroke-width="0.9"/>' for v in range(22, 84, 9))
    return "\n".join([
        f'  <g transform="translate({x:.1f},{y:.1f}) scale({w / 190:.4f})">',
        '<ellipse cx="96" cy="68" rx="94" ry="5" fill="#1B2430" opacity="0.14"/>',
        '<path d="M0,47 H150 V66 H128 V60 H98 V66 H0 Z" fill="#25282E"/>',
        '<rect x="8" y="51" width="78" height="10" rx="2" fill="#0B0C0F"/>',
        '<path d="M150,47 L190,21 V40 L150,66 Z" fill="#17191D"/>',
        '<g transform="matrix(1,-0.65,0,1,150,47)"><rect x="5" y="6" width="6" height="6" rx="1" fill="#9AA2AE"/>'
        '<rect x="14" y="6" width="9" height="5" rx="1" fill="#5D6470"/><rect x="26" y="5" width="5" height="8" rx="1" fill="#5D6470"/>'
        '<rect x="33" y="8" width="4" height="4" rx="1" fill="#5D6470"/></g>',
        '<rect x="0" y="28" width="150" height="19" fill="url(#thor-body)"/>',
        '<path d="M150,28 L190,2 V21 L150,47 Z" fill="#7B828D"/>',
        f'<path d="M{top(0, 0)} L{top(150, 0)} L{top(150, 1)} L{top(0, 1)} Z" fill="#50545D"/>',
        f'<path d="M{top(9, 0.15)} L{top(75, 0.15)} L{top(75, 0.85)} L{top(9, 0.85)} Z" fill="#121418" stroke="#121418" stroke-width="4" stroke-linejoin="round"/>',
        ribs,
        '<path d="M0,28 H150 L190,2" fill="none" stroke="#E3E7EC" stroke-width="0.9"/>',
        '<path d="M118,28 V47" stroke="#737A85" stroke-width="0.7"/>',
        icon("nvidia", 127, 32, 11, BRAND["nvidia"]).strip(),
        "</g>",
    ])


def switch(x, y, w, ports, label, uplink=None, flip=False):
    """A network switch, 60 tall: one socket above each x in `ports` along the bottom edge
    (the top edge when `flip`), `label` printed on its face, and a socket for an uplink at
    x `uplink` along the top edge."""
    must_fit(label, (uplink - 24 if uplink else x + w - 20) - (x + 20), 12, bold=True)
    socket = lambda cx, top, light: (f'  <rect x="{cx - 12}" y="{top}" width="24" height="15" rx="3" fill="#0F151D" stroke="#5A6779" stroke-width="1"/>\n'
                                     f'  <rect x="{cx - 5}" y="{top + (0 if flip else 9)}" width="10" height="6" fill="#5A6779"/>\n'
                                     f'  <circle cx="{cx + 18}" cy="{top + 7.5}" r="2.2" fill="{light}"/>')
    return "\n".join([
        f'  <rect x="{x}" y="{y}" width="{w}" height="60" rx="10" fill="url(#switch-body)" stroke="#1B2430" stroke-width="1.3" filter="url(#lift)"/>',
        text(x + 20, y + (46 if flip else 22), label, "pill", fill="#DCE3EC"),
        *([socket(uplink, y + 7, "#F2B544")] if uplink else []),
        *[socket(cx, y + (7 if flip else 38), "#8FE000") for cx in ports],
    ])


def card(x, y, w, h, kind, name, sub=(), glyph=None, glyph_fill=None, dashed=False, pill=None, tile=None, big=False):
    """A paper card: a coloured edge on the left, an icon tile, a name and up to three lines.

    `tile` is the colour behind the icon; the card's tint when it is not given. `big` draws
    the icon half as large again, for a card whose picture is the point."""
    accent, tint = KINDS[kind]
    lines = [sub] if isinstance(sub, str) else list(sub)
    side = 48 if big else 32
    left = x + (26 + side if glyph else 20)
    room = x + w - left - 12 - (width_of(pill, 11, bold=True) + 22 if pill else 0)
    must_fit(name, room, 14, bold=True)
    for line in lines:
        must_fit(line, x + w - left - 12)

    outline = f'stroke="{accent}" stroke-dasharray="6 5"' if dashed else f'stroke="{EDGE}"'
    shadow = "" if dashed else ' filter="url(#lift)"'
    fill = tint if dashed else CARD
    parts = [
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" {outline} stroke-width="1.3"{shadow}/>',
        f'  <path d="M{x + 12},{y + 0.7} H{x + 6} V{y + h - 0.7} H{x + 12} A11.3,11.3 0 0 1 {x + 0.7},{y + h - 12}'
        f' V{y + 12} A11.3,11.3 0 0 1 {x + 12},{y + 0.7} Z" fill="{accent}"/>',
    ]
    if glyph:
        mark = side - 12 if not big else 30
        parts += [f'  <rect x="{x + 16}" y="{y + h / 2 - side / 2}" width="{side}" height="{side}" rx="{side / 4:.0f}" fill="{tile or tint}"/>',
                  icon(glyph, x + 16 + (side - mark) / 2, y + h / 2 - mark / 2, mark,
                       glyph_fill or (None if glyph not in GLYPHS else accent))]
    top = y + h / 2 - 9 * len(lines) + 5
    parts.append(text(left, top, name, "name"))
    parts += [text(left, top + 18 * (row + 1), line, "sub") for row, line in enumerate(lines)]
    if pill:
        parts.append(tag(x + w - 12, y + 12, pill, kind, anchor="end"))
    return "\n".join(parts)


def tag(x, y, body, kind="idle", anchor="start"):
    """A small rounded label, such as a port number."""
    accent, tint = KINDS[kind]
    wide = width_of(body, 11, bold=True) + 14
    left = x - wide if anchor == "end" else x
    return (f'  <rect x="{left:.1f}" y="{y}" width="{wide:.1f}" height="18" rx="9" fill="{tint}"/>\n'
            + text(left + wide / 2, y + 12.6, body, "pill", "middle", accent))


def cylinder(x, y, w, h, kind, name, sub=()):
    """A store: a database, a file on disk, a model's weights."""
    accent, tint = KINDS[kind]
    lines = [sub] if isinstance(sub, str) else list(sub)
    must_fit(name, w - 20, 14, bold=True)
    for line in lines:
        must_fit(line, w - 16)
    ry, cx = 9, x + w / 2
    body = (f'M{x},{y + ry} A{w / 2},{ry} 0 0 1 {x + w},{y + ry} V{y + h - ry}'
            f' A{w / 2},{ry} 0 0 1 {x},{y + h - ry} Z')
    top = y + 2 * ry + (h - 2 * ry - 4) / 2 - 9 * len(lines) + 5
    parts = [
        f'  <path d="{body}" fill="{tint}" stroke="{accent}" stroke-width="1.4" filter="url(#lift)"/>',
        f'  <ellipse cx="{cx}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{CARD}" stroke="{accent}" stroke-width="1.4"/>',
        text(cx, top, name, "name", "middle"),
    ]
    parts += [text(cx, top + 18 * (row + 1), line, "sub", "middle") for row, line in enumerate(lines)]
    return "\n".join(parts)


def sheet(x, y, w, h, kind, heading, lines, footer=()):
    """A sheet of paper with a folded corner: a heading, lines of code, then plain lines.

    `lines` are (class, text) pairs set in the code face; `footer` are plain strings."""
    accent, tint = KINDS[kind]
    fold = 18
    must_fit(heading, w - 36 - fold, 12, bold=True)
    for _, body in lines:
        must_fit(body, w - 32, 12, mono=True)
    for body in footer:
        must_fit(body, w - 32)
    outline = (f'M{x + 10},{y} H{x + w - fold} L{x + w},{y + fold} V{y + h - 10} A10,10 0 0 1 {x + w - 10},{y + h}'
               f' H{x + 10} A10,10 0 0 1 {x},{y + h - 10} V{y + 10} A10,10 0 0 1 {x + 10},{y} Z')
    parts = [
        f'  <path d="{outline}" fill="{CARD}" stroke="{EDGE}" stroke-width="1.3" filter="url(#lift)"/>',
        f'  <path d="M{x + w - fold},{y} V{y + fold} H{x + w} Z" fill="{tint}" stroke="{EDGE}" stroke-width="1.3" stroke-linejoin="round"/>',
        f'  <rect x="{x + 16}" y="{y + 16}" width="4" height="14" rx="2" fill="{accent}"/>',
        text(x + 28, y + 27.5, heading, "tab", fill=accent),
    ]
    parts += [text(x + 16, y + 56 + 18 * row, body, f"mono {cls}".strip()) for row, (cls, body) in enumerate(lines)]
    base = y + h - 16 - 17 * (len(footer) - 1)
    parts += [text(x + 16, base + 17 * row, body, "note", fill=accent) for row, body in enumerate(footer)]
    return "\n".join(parts)


def zone(x, y, w, h, label):
    """A tinted area that groups cards, with its label on a tab."""
    must_fit(label, w - 40, 11.5, bold=True)
    wide = width_of(label.upper(), 11.5, bold=True) * 1.14 + 28
    return "\n".join([
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#F3F1EA" stroke="#E4E0D4" stroke-width="1.2"/>',
        f'  <rect x="{x + 18}" y="{y - 11}" width="{wide:.1f}" height="22" rx="11" fill="{INK}"/>',
        text(x + 18 + wide / 2, y + 4.2, label.upper(), "tab", "middle", "#FFFFFF"),
    ])


def wire(points, kind="flow", label=None, on=None, arrow=True):
    """A connection through `points`. Every segment is horizontal or vertical; corners are rounded.

    `label` sits on segment `on` (the longest by default), on a patch of paper that hides the line under it."""
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        if ax != bx and ay != by:
            raise DoesNotFit(f"wire runs at an angle from {(ax, ay)} to {(bx, by)}")
    radius, path = 9, [f"M{points[0][0]},{points[0][1]}"]
    for before, corner, after in zip(points, points[1:], points[2:]):
        path.append("L{},{} Q{},{} {},{}".format(*toward(corner, before, radius), *corner, *toward(corner, after, radius)))
    path.append(f"L{points[-1][0]},{points[-1][1]}")
    cls = "wire" if kind == "flow" else f"wire {kind}"
    marker = {"flow": "head", "danger": "head-danger", "idle": "head-idle"}[kind]
    end = f' marker-end="url(#{marker})"' if arrow else ""
    parts = [f'  <path class="{cls}" d="{" ".join(path)}"{end}/>']
    if label:
        segments = list(zip(points, points[1:]))
        (ax, ay), (bx, by) = segments[on] if on is not None else max(segments, key=lambda s: abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]))
        if abs(ax - bx) + abs(ay - by) < width_of(label, 11, bold=True) + 34 and ay == by:
            raise DoesNotFit(f"label {label!r} is longer than the segment it sits on")
        colour = {"flow": WIRE, "danger": KINDS["danger"][0], "idle": KINDS["idle"][0]}[kind]
        wide, cx, cy = width_of(label, 11, bold=True) + 14, (ax + bx) / 2, (ay + by) / 2
        parts += [f'  <rect x="{cx - wide / 2:.1f}" y="{cy - 9}" width="{wide:.1f}" height="18" rx="9" fill="{PAPER}" stroke="{colour}" stroke-width="1"/>',
                  text(cx, cy + 3.8, label, "pill", "middle", colour)]
    return "\n".join(parts)


def toward(corner, other, distance):
    """The point `distance` from `corner` on the way to `other`, or halfway when they are closer than that."""
    (cx, cy), (ox, oy) = corner, other
    reach = min(distance, (abs(ox - cx) + abs(oy - cy)) / 2)
    step = lambda a, b: a + reach * ((b > a) - (b < a))
    return step(cx, ox), step(cy, oy)


def legend(x, y, items):
    """A row of (kind, text) pairs, each with a swatch."""
    parts, at = [], x
    for kind, body in items:
        accent, tint = KINDS[kind]
        parts += [f'  <rect x="{at}" y="{y - 10}" width="14" height="14" rx="4" fill="{tint}" stroke="{accent}" stroke-width="1.3"/>',
                  text(at + 21, y + 1.5, body, "note")]
        at += 21 + width_of(body) + 22
    return "\n".join(parts)


def save(path, width, height, label, title, lede, parts):
    """Writes the figure. `label` is read aloud by a screen reader in place of the picture."""
    must_fit(title, width - 64, 18, bold=True)
    must_fit(lede, width - 64, 12.5)
    body = "\n".join(parts)
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}"'
        f' role="img" aria-label="{escape(label)}">\n{STYLE}\n'
        f'  <rect width="{width}" height="{height}" rx="14" fill="{PAPER}"/>\n'
        f'{text(32, 42, title, "title")}\n{text(32, 63, lede, "lede")}\n{body}\n</svg>\n')


def stacked(x, y, w, h, segments, total=None):
    """A horizontal bar of (value, colour, label) segments. A label is drawn inside its
    segment when it fits and left out when it does not, so a caller lists small ones in a legend."""
    total = total or sum(value for value, _, _ in segments)
    parts, at = [], x
    for value, colour, label in segments:
        wide = w * value / total
        parts.append(f'  <rect x="{at:.1f}" y="{y}" width="{wide:.1f}" height="{h}" fill="{colour}"/>')
        if label and width_of(label, 12, bold=True) + 16 <= wide:
            parts.append(text(at + wide / 2, y + h / 2 + 4.2, label, "pill", "middle", "#FFFFFF"))
        at += wide
    frame = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{EDGE}" stroke-width="1.2"/>'
    clip = f"clip-{int(x)}-{int(y)}"
    return "\n".join([f'  <clipPath id="{clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/></clipPath>',
                      f'  <g clip-path="url(#{clip})" filter="url(#lift)">'] + parts + ["  </g>", frame])


def swatches(x, y, items):
    """A row of (colour, text) pairs."""
    parts, at = [], x
    for colour, body in items:
        parts += [f'  <rect x="{at:.1f}" y="{y - 10}" width="14" height="14" rx="4" fill="{colour}"/>',
                  text(at + 21, y + 1.5, body, "note")]
        at += 21 + width_of(body) + 24
    return "\n".join(parts)
