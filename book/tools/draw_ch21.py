"""The figures of chapter 21, drawn with figkit and figparts.

    python3 tools/draw_ch21.py [path/to/target/images]

The times in the measured figures are the ones printed by code/oxide/ch21_capstone in the
run quoted in the chapter. With a path, the script also converts the program's output
images (PPM/PGM) into the PNG montage of figure 21.2.
"""
import sys

from figkit import KINDS, card, legend, wire
from figparts import EDGE, INK, MUTED, WIRE, bar, line, mono, note, save, tag, text, zone

OUT = "src/figures/"

# From the quoted run (milliseconds per stage for 1 image of 1,920 x 1,080 pixels).
STAGE_MS = [("grayscale", 0.201), ("histogram", 0.123), ("table", 0.007),
            ("apply table", 0.186), ("blur rows", 0.263), ("blur columns", 0.264),
            ("sobel", 0.350)]
BYTES = [4, 1, 0, 2, 3, 3, 2]
PIXELS = 1920 * 1080
# Images per second for the batch of 32 images, and for the CPU.
RATES = [("1 stream", 587.6), ("2 streams", 692.8), ("3 streams", 699.9),
         ("CPU, 8 cores", 46.3)]


def cell(x, y, w, h, body, kind="idle", on=True, cls="mono"):
    """A table cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" '
             f'stroke-width="1.2"/>']
    if body != "":
        parts.append(text(x + w / 2, y + h / 2 + 4, str(body), cls, "middle",
                          accent if on else MUTED))
    return "\n".join(parts)


def pipeline():
    """The 7 kernels and the buffers between them, in 2 columns."""
    w, h, gap = 264, 52, 76
    left, right, top = 34, 344, 104
    lx, rx = left + w / 2, right + w / 2
    stages_left = [
        ("store", "RGB image", "3 bytes per pixel, 6.2 MB"),
        ("service", "grayscale", "reads 3 B, writes 1 B per pixel"),
        ("service", "histogram", "reads 1 B per pixel, 256 counts"),
        ("service", "table", "1 block: scan of 256 counts"),
        ("service", "apply table", "reads gray and the table"),
    ]
    stages_right = [
        ("model", "blur rows", "reads 1 B, writes 2 B per pixel"),
        ("model", "blur columns", "reads 2 B, writes 1 B per pixel"),
        ("outside", "sobel", "reads 1 B, writes 1 B per pixel"),
        ("store", "edge image", "1 byte per pixel, 2.1 MB"),
    ]
    links_left = ["rgb", "gray", "bins", "table"]
    links_right = ["row sums", "blurred", "edges"]

    parts = [zone(18, 92, 300, 400, "Equalise"), zone(328, 92, 294, 324, "Blur and edges")]
    for k, (kind, name, sub) in enumerate(stages_left):
        y = top + k * gap
        parts.append(card(left, y, w, h, kind, name, sub))
        if k < len(links_left):
            parts.append(wire([(lx, y + h), (lx, y + gap - 3)], label=links_left[k]))
    for k, (kind, name, sub) in enumerate(stages_right):
        y = top + k * gap
        parts.append(card(right, y, w, h, kind, name, sub))
        if k < len(links_right):
            parts.append(wire([(rx, y + h), (rx, y + gap - 3)], label=links_right[k]))

    turn = top + 4 * gap + h / 2
    parts.append(wire([(left + w, turn), (323, turn), (323, top + h / 2),
                       (right - 3, top + h / 2)]))
    parts.append(note(334, 444, "the equalised image goes to blur rows"))
    save(OUT + "ch21-pipeline.svg", 640, 512,
         "The pipeline in 2 columns. Equalise: an RGB image of 3 bytes per pixel goes to the "
         "grayscale kernel, which writes a gray image of 1 byte per pixel. The histogram kernel "
         "counts its levels into 256 bins, the table kernel scans the 256 counts in 1 block, and "
         "the apply-table kernel reads the gray image and the table and writes the equalised "
         "image. Blur and edges: blur rows reads 1 byte and writes 2 bytes per pixel, blur "
         "columns reads 2 bytes and writes 1, and sobel writes the edge image of 1 byte per "
         "pixel, 2.1 MB.",
         "The 7 kernels of the pipeline", "Each arrow is a buffer in device memory.", parts)


def equalise():
    """Histogram equalisation of 16 pixels with 8 levels, by hand."""
    counts = [0, 0, 3, 5, 4, 2, 2, 0]
    cdf, total = [], 0
    for c in counts:
        total += c
        cdf.append(total)
    first = 3
    span = 16 - first
    levels = [0 if c < first else ((c - first) * 7 + span // 2) // span for c in cdf]

    x0, cw, ch = 168, 52, 30
    rows = [("level v", list(range(8)), "idle"), ("count", counts, "client"),
            ("cumulative", cdf, "service"), ("new level", levels, "good")]
    parts = []
    for r, (name, values, kind) in enumerate(rows):
        y = 100 + r * (ch + 8)
        parts.append(text(x0 - 14, y + ch / 2 + 4, name, "name", "end"))
        for v, value in enumerate(values):
            on = not (kind == "client" and value == 0)
            parts.append(cell(x0 + v * (cw + 4), y, cw, ch, value, kind, on))

    base, scale = 400, 18
    for k, (title, hist, kind) in enumerate([("before", counts, "client"),
                                              ("after", None, "good")]):
        gx = 60 + k * 300
        if hist is None:
            hist = [0] * 8
            for v, c in enumerate(counts):
                hist[levels[v]] += c
        parts.append(text(gx, 278, f"histogram {title}", "name"))
        parts.append(line(gx, base, gx + 8 * 30, base, EDGE, 1.2))
        for v, c in enumerate(hist):
            if c:
                parts.append(bar(gx + v * 30 + 4, base - c * scale, 22, c * scale, kind))
            parts.append(mono(gx + v * 30 + 15, base + 16, str(v), "middle"))
    parts.append(note(32, 452, "new level = round((cumulative - 3) x 7 / (16 - 3)); 3 is the "
                               "cumulative count of the first level in use."))
    parts.append(note(32, 472, "Levels 2 to 6 spread to 0 to 7; the order of the levels is "
                               "kept, so no 2 pixels change places."))
    save(OUT + "ch21-equalise.svg", 640, 496,
         "A table with 8 gray levels, 0 to 7, for an image of 16 pixels. Counts: 0, 0, 3, 5, 4, "
         "2, 2, 0. Cumulative counts: 0, 0, 3, 8, 12, 14, 16, 16. New levels: 0, 0, 0, 3, 5, 6, "
         "7, 7. Below, the histogram before occupies levels 2 to 6; after, the same pixels "
         "occupy levels 0, 3, 5, 6 and 7. The formula is new level = round((cumulative minus 3) "
         "times 7 divided by 13).",
         "Histogram equalisation by hand", "16 pixels, 8 levels: each level moves to its rank.",
         parts)


def blur():
    """The 5 x 5 weights as a product of 2 passes, and the tile of blur_rows."""
    taps = [1, 4, 6, 4, 1]
    parts = [text(36, 104, "row pass", "name"), text(36, 128, "column pass", "name")]
    for k, t in enumerate(taps):
        parts.append(cell(150 + k * 30, 88, 26, 24, t, "model"))
        parts.append(cell(150 + k * 30, 116, 26, 24, t, "client"))
    parts.append(note(310, 104, "each pass sums to 16"))
    parts.append(note(310, 128, "together 16 x 16 = 256"))

    parts.append(text(36, 172, "the same weights in 1 pass: 25 products", "name"))
    for r, a in enumerate(taps):
        for c, b in enumerate(taps):
            parts.append(cell(150 + c * 34, 184 + r * 26, 30, 22, a * b, "outside",
                              cls="mono"))
    parts.append(note(340, 220, "5 + 5 = 10 multiply-adds per pixel"))
    parts.append(note(340, 240, "instead of 5 x 5 = 25"))

    y = 352
    parts.append(text(36, y - 12, "tile of 1 block of blur_rows, 128 threads, x0 = 0",
                      "name"))
    slots = [("0", "0"), ("1", "0"), ("2", "0"), ("3", "1"), ("...", ""), ("129", "127"),
             ("130", "128"), ("131", "129")]
    for k, (slot, pixel) in enumerate(slots):
        halo = k in (0, 1, 6, 7)
        parts.append(cell(36 + k * 70, y, 64, 26, slot, "danger" if halo else "model"))
        parts.append(text(36 + k * 70 + 32, y + 44, pixel and f"pixel {pixel}", "note",
                          "middle"))
    parts.append(note(36, y + 72, "Thread t loads slot t; threads 0 to 3 also load slots "
                                  "128 to 131. Slot t holds pixel t - 2,"))
    parts.append(note(36, y + 90, "clamped to the row, so slots 0 and 1 repeat pixel 0. "
                                  "Output x reads slots x to x + 4."))
    save(OUT + "ch21-blur.svg", 640, 470,
         "Top: the row weights 1, 4, 6, 4, 1 and the column weights 1, 4, 6, 4, 1, each "
         "summing to 16. Middle: their products form a 5 by 5 grid from 1 at the corners to 36 "
         "in the centre, summing to 256; 2 passes need 10 multiply-adds per pixel instead of "
         "25. Bottom: the shared tile of 1 block of blur rows, slots 0 to 131. Slot t holds "
         "pixel t minus 2 clamped to the row: slots 0 and 1 hold pixel 0, slot 129 holds pixel "
         "127, slots 130 and 131 hold pixels 128 and 129, loaded by threads 0 to 3.",
         "A separable Gaussian blur", "2 passes of 5 weights give the 25 weights of 1 pass.",
         parts)


def sobel():
    """The 2 Sobel masks applied to 1 neighbourhood, by hand."""
    block = [[20, 20, 60], [20, 20, 60], [20, 30, 60]]
    gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    parts = []
    for k, (title, grid, kind) in enumerate([("pixels around (x, y)", block, "idle"),
                                              ("Gx mask", gx, "client"),
                                              ("Gy mask", gy, "model")]):
        x = 40 + k * 200
        parts.append(text(x, 104, title, "name"))
        for r in range(3):
            for c in range(3):
                value = grid[r][c]
                on = kind == "idle" or value != 0
                parts.append(cell(x + c * 44, 116 + r * 34, 40, 30, value, kind, on))
    parts += [
        mono(40, 252, "gx = (60 + 2 x 60 + 60) - (20 + 2 x 20 + 20) = 240 - 80 = 160"),
        mono(40, 276, "gy = (20 + 2 x 30 + 60) - (20 + 2 x 20 + 60) = 140 - 120 = 20"),
        mono(40, 300, "magnitude = sqrt(160 x 160 + 20 x 20) = sqrt(26,000) = 161.2"),
        mono(40, 324, "edge pixel = round(161.2) = 161, clamped to 255"),
    ]
    save(OUT + "ch21-sobel.svg", 640, 348,
         "Left: a 3 by 3 neighbourhood with columns 20, 20, 60 and a bottom row 20, 30, 60. "
         "Middle: the Gx mask, -1 0 1 in each row with the middle row doubled. Right: the Gy "
         "mask, -1 -2 -1 on top and 1 2 1 at the bottom. Worked: gx = 240 - 80 = 160, gy = "
         "140 - 120 = 20, magnitude = square root of 26,000 = 161.2, rounded to 161.",
         "Sobel edges by hand", "gx measures the change across, gy the change down.", parts)


def stages():
    """Measured time per stage, with the memory traffic each one moves."""
    top, h, scale = 104, 26, 320 / max(ms for _, ms in STAGE_MS)
    parts = []
    for k, ((name, ms), b) in enumerate(zip(STAGE_MS, BYTES)):
        y = top + k * (h + 12)
        parts.append(text(150, y + h / 2 + 4, name, "name", "end"))
        parts.append(bar(162, y, max(ms * scale, 2), h, "service"))
        rate = b * PIXELS / 1e9 / (ms / 1e3) if b else 0
        body = f"{ms:.3f} ms" + (f", {rate:.0f} GB/s" if b else "")
        parts.append(note(170 + max(ms * scale, 2), y + h / 2 + 4, body))
    total = sum(ms for _, ms in STAGE_MS)
    parts.append(note(36, top + 7 * (h + 12) + 14,
                      f"Sum of the 7 stages: {total:.2f} ms per image of 1,920 x 1,080 "
                      "pixels."))
    save(OUT + "ch21-stages.svg", 640, top + 7 * (h + 12) + 40,
         "Horizontal bars of the measured time of each stage for 1 image: "
         + "; ".join(f"{n} {ms:.3f} ms" for n, ms in STAGE_MS)
         + f". The 7 stages sum to {total:.2f} ms.",
         "Time per stage", "1 image of 1,920 x 1,080 pixels on the measured GPU.", parts)


def streams():
    """The batch on 1 and on 3 streams, as lanes of copies and kernels, and the rates."""
    parts = [text(36, 106, "1 stream", "name"), text(36, 190, "3 streams", "name")]
    # Milliseconds per image, from the quoted run: copy in, the 7 kernels, copy out.
    seq = [(0.22, "client"), (1.41, "service"), (0.08, "store")]
    scale, x0 = 90, 120

    def image(start, y):
        bars, at = [], start
        for ms, kind in seq:
            label = f"{ms:.2f} ms" if ms > 1 else ""
            bars.append(bar(x0 + at * scale, y, ms * scale - 1.5, 22, kind, label))
            at += ms
        return bars

    for k in range(3):
        parts += image(k * 1.71, 92)
        parts += image(k * 1.41, 140 + k * 28)
    parts.append(legend(120, 248, [("client", "copy in, 6.2 MB"), ("service", "7 kernels"),
                                   ("store", "copy out, 2.1 MB")]))
    top = 272
    peak = max(rate for _, rate in RATES)
    for k, (name, rate) in enumerate(RATES):
        y = top + k * 34
        parts.append(text(150, y + 17, name, "name", "end"))
        kind = "idle" if name.startswith("CPU") else "good"
        parts.append(bar(162, y, max(rate / peak * 340, 2), 24, kind))
        parts.append(note(170 + max(rate / peak * 340, 2), y + 16, f"{rate:.1f} images/s"))
    save(OUT + "ch21-streams.svg", 640, top + 4 * 34 + 20,
         "Top: on 1 stream each image is copied in, processed and copied out before the next "
         "starts. On 3 streams the copy of 1 image overlaps the kernels of another. Bottom, "
         "measured images per second for a batch of 128 images: "
         + "; ".join(f"{n} {r:.1f}" for n, r in RATES) + ".",
         "A batch of images on streams", "Copies of 1 image run while another is processed.",
         parts)


def montage(folder):
    """The program's real output images: 3 panels reduced to 480 x 270 pixels, and 1 panel
    of 2 crops of 60 x 68 pixels around the window, enlarged 4 times without smoothing."""
    from PIL import Image, ImageDraw, ImageFont

    open_rgb = lambda file: Image.open(f"{folder}/{file}").convert("RGB")
    crop = (400, 580, 460, 648)
    zoom = lambda file: open_rgb(file).crop(crop).resize((236, 270), Image.NEAREST)
    pair = Image.new("RGB", (480, 270), "#FBFAF6")
    pair.paste(zoom("equalised.pgm"), (0, 0))
    pair.paste(zoom("blurred.pgm"), (244, 0))

    panels = [(open_rgb("input.ppm").resize((480, 270)), "input (colour)"),
              (open_rgb("equalised.pgm").resize((480, 270)), "equalised"),
              (pair, "equalised | blurred, 4 x zoom on the window"),
              (open_rgb("edges.pgm").resize((480, 270)), "edges")]
    sheet = Image.new("RGB", (2 * 480 + 30, 2 * 300 + 30), "#FBFAF6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for k, (picture, title) in enumerate(panels):
        x, y = 10 + (k % 2) * 490, 10 + (k // 2) * 305
        sheet.paste(picture, (x, y + 26))
        draw.text((x, y + 2), title, fill="#1B2430", font=font)
    sheet.save(OUT + "ch21-images.png")


if __name__ == "__main__":
    pipeline()
    equalise()
    blur()
    sobel()
    if all(ms > 0 for _, ms in STAGE_MS):
        stages()
        streams()
    if len(sys.argv) > 1:
        montage(sys.argv[1])
