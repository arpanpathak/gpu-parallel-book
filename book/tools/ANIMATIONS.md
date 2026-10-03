# Animations

Read this file before you add or change an animation. It describes the engine in
`tools/motion.py`, the rules every animation follows, and problems already solved.

## 1. Purpose

An animation shows a change over time that a still figure cannot show: a message sent
from one thread to another, a value that changes, a thread that stops running. Each
animation follows one run of the program in the listing beside it.

Every animation in the book uses the engine below. The earlier slide animations, which
swapped whole pictures every few seconds, have been removed.

## 2. Pacing and reader control

The reader may have ADHD, may be neurodivergent, or may be new to the topic. The engine
gives that reader time and control by default.

- `tl.say(text)` changes the caption. It first waits until the previous caption has been
  on screen for 0.36 s per word, and at least 2.6 s. After the new caption appears, nothing
  moves for `lead` seconds. Every scripted duration is multiplied by `pace` (1.2). Set
  captions only with `tl.say`.
- A code panel shows a line only after the highlight has reached it. Lines below that
  point are drawn as grey bars. Pass `reveal=s.timeline.reached(track, t)`.
- `render` writes an MP4 and a JSON step list next to each GIF. `tools/anim_markup.py`
  replaces the chapter's `<img>` with a `<video>` that keeps the GIF as a fallback.
  `theme/anim.js` adds a play button, speeds of 0.5x, 0.75x, and 1x, a button for each
  step, and a mode that pauses at the end of each step. When the reader's system asks for
  reduced motion, the video starts paused.

## 3. Drawing rules

1. Draw a message as an object that moves from the sender to the receiver. Draw a changed
   value in the place it is stored.
2. Use a physical object for each actor, and label it with the name from the code. The
   async chapter draws a thread as a robot at a desk, which closes its eyes when parked, and
   a waker as a bell.
3. Show each actor's code in a panel, with a bar on the line that actor is running.
4. Keep one caption at the foot of the frame. Use the plain band for a step, the brass band
   for the result the animation exists to show, and the rust band for the failing case.
5. Mark the element the caption names with a ring or a glow. Never draw a marker on top of
   the element it marks.
6. End with a failing case: the same run with one line removed or one rule broken.
7. Follow the code exactly. Events happen in the order the code runs them, and numbers
   match the program's output.
8. Show a progress rail with a label for each step.
9. Write captions by the rules in `../anti_ai_slop.md` and `WRITING_GUIDE.md`. Each caption
   states what is happening in the frame. Run the deletion test on each one.
10. Show the first occurrence of a repeated step slowly, and later ones faster. The builders
    take a speed factor `k` for this.

`tools/anim_async.py` shows the actors (threads, wakers, messages). `tools/anim_ch03.py`
shows the algorithm style built on `motion_kit`.

## 4. The engine: `tools/motion.py`

- `Timeline(**initial)`: named values over time, scripted in order.
  - `set(**v)` changes values at the cursor. `to(dur, ease, **v)` tweens and advances
    the cursor. `also(dur, ease, delay, **v)` tweens without advancing (parallel motion).
    `wait(s)`. `event(name)` marks a moment; `age(name, t)` in the draw function gives the
    seconds since, for one-shot effects (a bell ringing, a shake). `chapter(label)` adds a
    stop on the progress rail. `changed(name, t)` gives when a value last changed (used for
    caption fades and the split-flap tag flip).
  - Numbers, `(x, y)` tuples, and `#rrggbb` colours interpolate. Strings switch.
  - Easing: `in_out`, `ease_out`, `ease_in`, `linear`, `back` (overshoot, for things that
    land). Do not use `back` on gauges or counters: it overshoots (a CPU gauge read 110%).
- `draw(p, s, total)`: paints one frame from the sampled values `s` onto a `Pic` (an SVG
  canvas with `rect`, `circle`, `line`, `path`, `text`, and `group(opacity, dx, dy, rotate,
  scale)`). Keep it a pure function of `s`.
- Actors and furniture: `robot`, `zzz`, `bell`, `ring_waves`, `pill` (a message in
  flight), `chip`, `meter`, `focus`, `code_panel`, `title_block`, `scoreboard`, `caption`,
  `progress`, `bezier` (for arcs).
- `render(name, tl, draw, height)`: samples at 20 fps, merges identical frames into longer
  holds, rasterises the distinct frames with `rsvg-convert` in parallel, builds one palette
  for the whole GIF, and writes `src/figures/<name>`, plus `<stem>.mp4` (same frames,
  H.264) and `<stem>.json` (duration and chapter times for the step buttons). The last
  0.6 s cross-fades into the first frame so the loop does not jump.
- Builders register in the module's `BUILDERS`, and `animations.py` merges them into
  `MOTION`. `make animations` runs them.

## 4a. Algorithm animations: `tools/motion_kit.py`

An algorithm animation is written as a simulation, not as a hand-placed script. The
builder runs the chapter's algorithm on the chapter's input and appends one step per
thing the reader should see. Each step is a dict: `say` (the caption), optional `kind`,
`chapter`, `dur`, and `hold`, and the state after the step. `play(steps, defaults)` turns
the list into a timeline: numbers slide, everything else changes when the step starts.

- `source(path, first, last)` reads lines from `rust-interview-lab`, so the code panel
  always shows the listing's code. `last="}"` ends at the brace that closes the item.
  `line_of(code, text)` finds the line to highlight.
- `cells` draws a row of boxes with indexes; `pointer` draws a named arrow above or below
  a cell, never on it, and stacks two pointers at one index with `slot`.
- `kv_panel` draws a map, `stack_panel` a stack, `node` and `edge` a tree or a graph,
  `heap_positions` a complete binary tree.
- `layout(code_y, lines)` places the caption and the progress rail below the code panel
  from its real height, so the three cannot overlap.
- Do not name a state field `t`: the draw function's `s.t` is the sample time.

Workflow:

```bash
# stills of chosen moments in one contact sheet, while scripting
python3 tools/preview.py /tmp/sheet.png two-sum:12 window:40
# render one animation (about a minute)
python3 tools/animations.py poll-wake
python3 tools/anim_markup.py          # point the chapter at the new video and steps
```

Print the timeline length and chapter times with `tl.now` and `tl.chapters` before
rendering. Preview frames at the busiest moments and look for collisions: labels on
arcs, flying objects crossing the title, captions wrapping onto a third line.

## 5. Problems already solved

- **File size.** Anything that changes on every frame makes every frame a new image. The
  progress rail steps twice a second, and idle motion (drifting z's, a stopwatch hand) is
  quantised to a few steps a second, so held frames merge.
- **Encoding speed.** Encoding the GIF in Pillow ran on one core and held every frame in
  memory: about 20 minutes per animation. `render` now writes PNG stills in parallel and
  lets ffmpeg build both the GIF (one palette from all frames, no dithering, changed
  rectangles only) and the MP4. An animation takes about a minute.
- **Whitespace.** SVG collapses leading spaces, which flattened the code panels. Text is
  written with `xml:space="preserve"`.
- **`set` at t = 0** must take effect at t = 0 (zero-length segments count as done at their
  start). Otherwise the first frame shows the old value.
- **Workers re-import the modules.** `render` uses a process pool, and on macOS the workers
  import the main module again. Do not edit `motion.py` or the builders while a render is
  running. To keep working, render from a copy: copy `tools/` into `<tmp>/book/tools`,
  link `<tmp>/book/src` to `src` and `<tmp>/rust-interview-lab` to the lab, and run
  `python3 tools/animations.py` from `<tmp>/book`.
- **Lifted arcs** have to stay below the subtitle: keep a flying object's top edge under
  y = 64.

## 6. Status

All 46 animations in chapters 3, 5, 6, 9, 11, 12, 13, 14, 16, 17, and 19 to 29 use the
motion engine. Chapter 21 presents its code in small excerpts, with the complete files
at the end. The other chapters still show some long listings first.

## 7. The print edition

Animations do not work in print. The PDF (`tools/build_pdf.py`, WeasyPrint) shows a GIF's
first frame at best, and the HTML edition in `book/` is where the animations live.

The print edition, if it is revived as a printable book, needs its own approach to
intuition: a strip of three to five numbered still panels per animation (the key
moments, drawn with the same actors), or a static figure that shows all states at once
with arrows for the order. Those can be rendered from the same timelines by sampling
chosen times with `render(..., only=[t1, t2, ...])`. That work has not been done.
