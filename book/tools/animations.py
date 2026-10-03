#!/usr/bin/env python3
"""Every animation in the book, written to src/figures/.

Each module below holds the animations for one or more chapters, drawn with
`motion`. `render` writes three files per animation: a GIF, an MP4 of the same
frames, and a JSON step list that the HTML book uses for its step buttons.

    python3 tools/animations.py              # every animation
    python3 tools/animations.py roofline amdahl  # some of them

After a render, `python3 tools/anim_markup.py` points the chapters at the new
videos.
"""

import sys

import anim_ch01
from motion import OUT

MOTION = {}
for module in (anim_ch01,):
    MOTION.update(module.BUILDERS)


def main(argv):
    names = argv or list(MOTION)
    unknown = [n for n in names if n not in MOTION]
    if unknown:
        print("unknown animation(s): %s" % ", ".join(unknown), file=sys.stderr)
        print("available: %s" % ", ".join(MOTION), file=sys.stderr)
        return 2
    for name in names:
        MOTION[name]()
        print("  %s" % name)
    print("wrote %d animations to %s" % (len(names), OUT.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
