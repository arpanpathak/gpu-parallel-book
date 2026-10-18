"""Shared drawing parts for the chapter figures, on top of figkit: bars, straight
arrows, fan-outs, icon tiles, the memory store, and the book's bee mascots.

    from figparts import *
"""
from figkit import (EDGE, INK, KINDS, MUTED, PAPER, WIRE, card, cylinder, icon, legend, must_fit, save,
                    sheet, tag, text, width_of, wire, zone)
from draw_mascots import bee as bee_svg

def bar(x, y, w, h, kind, label="", dashed=False, solid=False):
    """A rounded bar in the colours of `kind`, with a centred label that must fit."""
    accent, tint = KINDS[kind]
    fill = accent if solid else tint
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" fill="{fill}"'
             f' stroke="{accent}" stroke-width="1.3"{dash}/>']
    if label:
        must_fit(label, w - 8, 11, bold=True)
        parts.append(text(x + w / 2, y + h / 2 + 4, label, "pill", "middle",
                          "#FFFFFF" if solid else accent))
    return "\n".join(parts)


def line(x1, y1, x2, y2, colour=WIRE, width=1.4, dash=""):
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}"'
            f' stroke-width="{width}"{dashes}/>')


def arrow(x, y1, y2, kind="flow"):
    """A straight vertical arrow whose tip stops 3 units short of `y2`."""
    colour = {"flow": WIRE}.get(kind, KINDS.get(kind, (WIRE,))[0])
    marker = "head" if colour == WIRE else f"head-{kind}"
    return (f'  <path d="M{x},{y1} V{y2 - 3}" fill="none" stroke="{colour}" stroke-width="1.8"'
            f' stroke-linecap="round" marker-end="url(#{marker})"/>')


def fan(x, y, targets, bus, top, kind="flow"):
    """One source fanned out to several targets: a stem down from (x, y) to the
    bus line at `bus`, the bus across the targets, and an arrow down to each."""
    colour = WIRE if kind == "flow" else KINDS[kind][0]
    left, right = min(targets + [x]), max(targets + [x])
    parts = [f'  <path d="M{x},{y} V{bus} M{left},{bus} H{right}" fill="none" stroke="{colour}"'
             f' stroke-width="1.8" stroke-linecap="round"/>',
             f'  <circle cx="{x}" cy="{bus}" r="3" fill="{colour}"/>']
    marker = "head"
    if kind != "flow":
        marker = f"tip-{kind}-{int(x)}"
        parts.append(f'  <defs><marker id="{marker}" viewBox="0 0 10 10" refX="8.5" refY="5"'
                     f' markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto">'
                     f'<path d="M0.5,1 L9,5 L0.5,9 Z" fill="{colour}"/></marker></defs>')
    for t in targets:
        parts.append(f'  <path d="M{t},{bus} V{top - 3}" fill="none" stroke="{colour}" stroke-width="1.6"'
                     f' stroke-linecap="round" marker-end="url(#{marker})"/>')
    return "\n".join(parts)


def tile(x, y, glyph, kind="client", side=28):
    """A rounded tile in the tint of `kind` holding one glyph, like the tiles on figkit cards."""
    accent, tint = KINDS[kind]
    mark = side - 10
    return (f'  <rect x="{x}" y="{y}" width="{side}" height="{side}" rx="{side / 4:.0f}" fill="{tint}"/>\n'
            + icon(glyph, x + 5, y + 5, mark, accent))


_bees = 0


def mascot(x, y, width, mood):
    """One of the book's bees, `width` wide, top-left at (x, y)."""
    global _bees
    _bees += 1
    body = bee_svg(mood)
    body = body[body.index(">") + 1:body.rindex("</svg>")].replace('id="body"', f'id="bee{_bees}"')
    body = body.replace("url(#body)", f"url(#bee{_bees})")
    return f'  <g transform="translate({x},{y}) scale({width / 240:.4f})">{body}</g>'


def actors(y, left_sub, right_sub, label):
    """The processor card, the memory store and the wire between them, on one row."""
    return "\n".join([
        card(40, y, 230, 64, "client", "processor", left_sub, glyph="chip"),
        cylinder(430, y - 4, 170, 72, "store", "main memory", right_sub),
        wire([(270, y + 32), (430, y + 32)], label=label),
    ])


def note(x, y, body, anchor="start", room=None, fill=None):
    if room:
        must_fit(body, room)
    return text(x, y, body, "note", anchor, fill)


def mono(x, y, body, anchor="start", fill=None):
    return text(x, y, body, "mono", anchor, fill)


