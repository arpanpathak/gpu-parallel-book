#!/usr/bin/env python3
"""Draw the cover and chapter plates for the GPU book.

The design is deliberately minimal: a dark board, one warp of lane cells, and
one roofline schematic. Every coordinate is fixed and checked, so nothing
overlaps. The SVG is written to src/art/ and rasterised to PNG with ffmpeg.

    python3 tools/art.py
"""

from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "art"

BG = "#0c1726"
PANEL = "#101f2e"
EDGE = "#2b4b66"
BAND = "#060d15"
INK_LIGHT = "#e8eef4"
MUTED = "#8fa3b8"
BRASS = "#e8c56b"
TEAL = "#5fc9b4"
TEAL_DIM = "#1d4a44"
RUST = "#e2704a"
RUST_DIM = "#3d2118"
LANE_OFF = "#1b3349"

SERIF = "Liberation Serif, Georgia, serif"
SANS = "Liberation Sans, DejaVu Sans, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, monospace"


# ---------------------------------------------------------------- primitives


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none",
         width: float = 0, rx: float = 0) -> str:
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" fill="%s" '
            'stroke="%s" stroke-width="%.1f"/>' % (x, y, w, h, rx, fill, stroke, width))


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 2) -> str:
    return ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f" '
            'stroke-linecap="round"/>' % (x1, y1, x2, y2, stroke, width))


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none",
           width: float = 0) -> str:
    return ('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" stroke-width="%.1f"/>'
            % (cx, cy, r, fill, stroke, width))


def text(x: float, y: float, body: str, size: float, fill: str, family: str = SANS,
         anchor: str = "start", weight: str = "normal", spacing: float = 0) -> str:
    body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return ('<text x="%.1f" y="%.1f" font-family="%s" font-size="%s" fill="%s" text-anchor="%s" '
            'font-weight="%s" letter-spacing="%s">%s</text>'
            % (x, y, family, size, fill, anchor, weight, spacing, body))


# ---------------------------------------------------------------- components


def lane_row(x: float, y: float, cell: float, gap: float, n: int = 32) -> str:
    """One row of lane cells. Lanes 0-11 ready, 12 stalled, 13-15 running,
    the rest parked."""
    out = ""
    for k in range(n):
        cx = x + k * (cell + gap) + cell / 2
        if k < 12:
            fill, edge = TEAL_DIM, TEAL
        elif k == 12:
            fill, edge = RUST_DIM, RUST
        elif k < 16:
            fill, edge = "#3a3220", BRASS
        else:
            fill, edge = LANE_OFF, EDGE
        out += rect(cx - cell / 2, y, cell, cell, fill, edge, 1.4, rx=3)
    return out


def roofline(x0: float, y0: float, x1: float, y1: float, big: bool = False) -> str:
    """A roofline: bandwidth ramp, flat compute roof, and a ridge point."""
    size = 20 if big else 14
    axis = 3 if big else 2
    rx, ry = x0 + 0.62 * (x1 - x0), y0 + 0.16 * (y1 - y0)
    out = line(x0, y0, x0, y1, EDGE, axis)
    out += line(x0, y1, x1, y1, EDGE, axis)
    out += line(x0 + 20, y1 - 20, rx, ry, TEAL, 6 if big else 4)
    out += line(rx, ry, x1 - 20, ry, BRASS, 6 if big else 4)
    out += circle(rx, ry, 10 if big else 7, RUST, INK_LIGHT, 2)
    out += text((x0 + 20 + rx) / 2, y1 - 40 if big else y1 - 30, "memory-bound",
                size, TEAL, SANS, "middle")
    out += text((rx + x1 - 20) / 2, ry - 20 if big else ry - 16, "compute-bound",
                size, BRASS, SANS, "middle")
    out += text(rx + 16, ry + 28 if big else ry + 24, "ridge", size, RUST, SANS, "start")
    out += text((x0 + x1) / 2, y1 + 34 if big else y1 + 26,
                "arithmetic intensity, FLOP / byte", size, MUTED, SANS, "middle")
    out += text(x0 - 10, y0 - 12, "perf", size, MUTED, SANS, "end")
    return out


# ------------------------------------------------------------------- plates


def plate_ch01() -> str:
    width, height = 1000, 320
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">'
        % (width, height, width, height),
        rect(0, 0, width, height, BG),
        text(48, 70, "WARP", 15, MUTED, SANS, "start", spacing=3),
        lane_row(48, 86, 13, 3, 32),
        text(48, 136, "32 lanes: lane 12 waits on its load, the rest ready or parked",
             13, MUTED, SANS, "start"),
        text(48, 196, "S(p) = 1 / (f + (1 - f) / p)", 17, INK_LIGHT, MONO, "start"),
        text(48, 228, "ceiling = 1 / f", 17, BRASS, MONO, "start"),
        roofline(600, 60, 950, 230),
        rect(0, 268, width, 4, RUST),
        rect(0, 272, width, 48, BAND),
        text(24, 303, "CHAPTER", 15, MUTED, SANS, "start", spacing=4),
        text(150, 305, "01", 20, BRASS, SANS, "start", "bold", spacing=2),
        text(226, 302, "RIDGE", 18, INK_LIGHT, SANS, "start", spacing=3),
        text(width - 24, 303, "READS WHERE THE ROOF MEETS THE BAND", 13, MUTED, SANS, "end",
             spacing=1.5),
        "</svg>",
    ]
    return "\n".join(out)


def cover() -> str:
    w, h = 1400, 1750
    lanes_w = 32 * 34 + 31 * 8
    lane_x = (w - lanes_w) / 2
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">'
        % (w, h, w, h),
        rect(0, 0, w, h, BG),
        text(90, 200, "CUDA Kernels", 100, INK_LIGHT, SANS, "start", "bold"),
        text(90, 330, "GPU & Parallel", 100, INK_LIGHT, SANS, "start", "bold"),
        text(90, 460, "Programming", 100, BRASS, SANS, "start", "bold"),
        text(94, 530, "From the warp to the roofline, one kernel at a time", 32, "#9fc1d6",
             SERIF, "start"),
        rect(0, 580, w, 12, RUST),
        lane_row(lane_x, 700, 34, 8, 32),
        text(w / 2, 830, "one warp: 32 lanes", 26, INK_LIGHT, SANS, "middle", "bold"),
        text(w / 2, 866, "lane 12 waits on its load; the other lanes run", 20, RUST, SANS,
             "middle"),
        roofline(150, 950, 1250, 1300, big=True),
        rect(150, 1380, 180, 60, PANEL, EDGE, 2, rx=6),
        text(240, 1418, "HBM", 20, MUTED, MONO, "middle"),
        rect(1070, 1380, 180, 60, PANEL, EDGE, 2, rx=6),
        text(1160, 1418, "PTX", 20, BRASS, MONO, "middle"),
        text(90, 1690, "Arpan Pathak", 50, INK_LIGHT, SANS, "start", "bold"),
        rect(1010, 1610, 300, 80, PANEL, EDGE, 2, rx=8),
        text(1160, 1662, "IN CUDA", 34, BRASS, SANS, "middle", spacing=8),
        "</svg>",
    ]
    return "\n".join(out)


def rasterise(svg: pathlib.Path, width: int) -> None:
    png = svg.with_suffix(".png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(svg),
                    "-vf", "scale=%d:-1" % width, "-frames:v", "1", str(png)], check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    target = OUT / "cover.svg"
    target.write_text(cover(), encoding="utf-8")
    rasterise(target, 1400)

    target = OUT / "ch01.svg"
    target.write_text(plate_ch01(), encoding="utf-8")
    rasterise(target, 2000)

    print("wrote cover and chapter plate to %s" % OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
