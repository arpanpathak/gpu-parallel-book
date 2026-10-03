#!/usr/bin/env python3
"""Point every motion animation in the chapters at its video.

`motion.render` writes three files per animation: the GIF, an MP4 of the same
frames, and a JSON file with the loop's steps. In the HTML book the MP4 is used,
because a reader can pause it, slow it down, and jump between steps
(theme/anim.js adds those controls); the GIF stays inside the <video> tag as the
fallback. This rewrites the <img> of each such animation into that <video>, and
refreshes the step list when an animation is rendered again.

    python3 tools/anim_markup.py
"""

import html
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "src" / "figures"

FIGURE = re.compile(r'(<figure class="anim">\s*)(.*?)(\s*<figcaption>)', re.S)
GIF = re.compile(r'src="([^"]*)figures/([a-z0-9-]+)\.gif" alt="([^"]*)"')


def markup(stem, alt, prefix):
    steps = json.loads((FIGURES / (stem + ".json")).read_text())["chapters"]
    return ('<video class="motion" src="%sfigures/%s.mp4" autoplay loop muted playsinline '
            'preload="metadata" aria-label="%s" data-chapters="%s">'
            '<img src="%sfigures/%s.gif" alt="%s"></video>'
            % (prefix, stem, alt, html.escape(json.dumps(steps), quote=True),
               prefix, stem, alt))


def pages():
    yield from sorted((ROOT / "src").glob("*.md"))
    yield from sorted((ROOT / "src" / "chapters").glob("**/*.md"))


def main():
    """Rebuild the media part of every animation figure from its GIF's name and
    alt text. The whole part is regenerated, so running this twice changes
    nothing, and a figure that was wrapped more than once is repaired."""
    changed = 0
    for page in pages():
        text = page.read_text()

        def swap(m):
            found = GIF.search(m.group(2))
            if not found:
                return m.group(0)
            prefix, stem, alt = found.groups()
            if not (FIGURES / (stem + ".mp4")).exists() or not (FIGURES / (stem + ".json")).exists():
                return m.group(0)
            return m.group(1) + markup(stem, alt, prefix) + m.group(3)

        new = FIGURE.sub(swap, text)
        if new != text:
            page.write_text(new)
            changed += 1
    print("updated %d chapter(s)" % changed)


if __name__ == "__main__":
    main()
