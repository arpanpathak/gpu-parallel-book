"""The cast in the margin boxes: a different character for each mood, drawn with the
same functions as the plates, so the boxes show the book's whole cast and not one bee.

    python3 tools/draw_cast_mascots.py

happy: Ferry's friend the otter; cute: the squirrel with an acorn; shocked: the meerkat;
sleepy: the tortoise; angry: a hornet; note: an ant with a crumb.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import art  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "src" / "art"
W, H = 240, 190

CAST = {
    "happy": lambda: art.otter(130, 108, 0.86, look=1),
    "cute": lambda: art.squirrel(126, 184, 0.68, look=1),
    "shocked": lambda: art.meerkat(120, 186, 0.86, look=1),
    "sleepy": lambda: art.tortoise(104, 170, 0.60, look=1),
    "angry": lambda: art.hornet(120, 186, 0.74, look=0, arms="fold"),
    "note": lambda: art.ant(112, 158, 1.45, carry="#e8c877"),
}

for mood, draw in CAST.items():
    art._SHADES.clear()
    body = art.make_3d(draw())
    (OUT / f"cast-{mood}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-hidden="true">\n{body}\n{art.defs_3d()}\n</svg>\n')
    print("  cast-" + mood)
