"""The bee mascots: one bee in five moods, drawn flat like the crabs of
Democratizing Linux. A bee sits beside a note, a callout and the summary.

    python3 tools/draw_mascots.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "art"
INK = "#1b2430"
GOLD = "#f2b84b"
GOLD_DARK = "#d9932a"
STRIPE = "#3a2a1e"
WING = "#d8ecfa"
WING_EDGE = "#9cc3e0"
CHEEK = "#f4a6b6"


def eye(cx, cy, mood):
    if mood == "sleepy":
        return (f'<path d="M{cx - 10},{cy} Q{cx},{cy + 8} {cx + 10},{cy}" fill="none" stroke="{INK}"'
                f' stroke-width="3.6" stroke-linecap="round"/>')
    rx, ry, r = (12, 15, 6.4) if mood != "shocked" else (13, 17, 5.2)
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#ffffff"/>'
            f'<circle cx="{cx}" cy="{cy + 1.5}" r="{r}" fill="{INK}"/>'
            f'<circle cx="{cx + 2.3}" cy="{cy - 1.8}" r="2.1" fill="#ffffff"/>')


def mouth(mood):
    return {
        "happy": f'<path d="M106,140 Q120,160 134,140 Z" fill="#7a1f0a"/>',
        "cute": f'<path d="M111,142 Q120,150 129,142" fill="none" stroke="#7a1f0a" stroke-width="3.6" stroke-linecap="round"/>',
        "shocked": f'<ellipse cx="120" cy="147" rx="7" ry="10" fill="#7a1f0a"/>',
        "sleepy": f'<ellipse cx="120" cy="146" rx="5" ry="4" fill="#7a1f0a"/>',
        "angry": f'<path d="M108,150 Q120,138 132,150" fill="none" stroke="#7a1f0a" stroke-width="3.8" stroke-linecap="round"/>',
    }[mood]


def bee(mood):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 190" width="240" height="190" role="img" aria-hidden="true">',
        # wings behind the body
        f'<ellipse cx="78" cy="56" rx="34" ry="24" fill="{WING}" stroke="{WING_EDGE}" stroke-width="3" transform="rotate(-28 78 56)"/>',
        f'<ellipse cx="162" cy="56" rx="34" ry="24" fill="{WING}" stroke="{WING_EDGE}" stroke-width="3" transform="rotate(28 162 56)"/>',
        # antennae
        f'<path d="M104,72 Q96,40 82,30" fill="none" stroke="{STRIPE}" stroke-width="4.5" stroke-linecap="round"/>',
        f'<path d="M136,72 Q144,40 158,30" fill="none" stroke="{STRIPE}" stroke-width="4.5" stroke-linecap="round"/>',
        f'<circle cx="82" cy="30" r="7" fill="{STRIPE}"/><circle cx="158" cy="30" r="7" fill="{STRIPE}"/>',
        # sting and legs
        f'<path d="M206,132 L226,138 L206,144 Z" fill="{STRIPE}"/>',
        *(f'<line x1="{x}" y1="160" x2="{x + d}" y2="178" stroke="{STRIPE}" stroke-width="6" stroke-linecap="round"/>'
          for x, d in ((86, -8), (110, -4), (130, 4), (154, 8))),
        # body with stripes, clipped to the body
        '<clipPath id="body"><ellipse cx="120" cy="124" rx="92" ry="46"/></clipPath>',
        f'<ellipse cx="120" cy="124" rx="92" ry="46" fill="{GOLD}"/>',
        f'<g clip-path="url(#body)" fill="{STRIPE}"><rect x="40" y="70" width="14" height="110"/>'
        f'<rect x="186" y="70" width="14" height="110"/><rect x="164" y="70" width="10" height="110"/></g>',
        f'<ellipse cx="120" cy="152" rx="58" ry="12" fill="{GOLD_DARK}" opacity="0.45"/>',
        eye(99, 118, mood), eye(141, 118, mood),
    ]
    if mood == "angry":
        parts.append(f'<path d="M84,98 L110,108 M156,98 L130,108" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>')
    if mood in ("cute", "happy"):
        parts.append(f'<ellipse cx="78" cy="138" rx="10" ry="6" fill="{CHEEK}" opacity="0.8"/>'
                     f'<ellipse cx="162" cy="138" rx="10" ry="6" fill="{CHEEK}" opacity="0.8"/>')
    if mood == "sleepy":
        parts.append(f'<text x="196" y="66" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="{WING_EDGE}">z</text>'
                     f'<text x="212" y="44" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700" fill="{WING_EDGE}">z</text>')
    parts += [mouth(mood), "</svg>"]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    for mood in ("happy", "cute", "shocked", "sleepy", "angry"):
        (OUT / f"bee-{mood}.svg").write_text(bee(mood))
    print("wrote 5 bee mascots")
