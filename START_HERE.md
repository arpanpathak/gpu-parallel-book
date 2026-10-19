# Start here: briefing for the next session

Paste this into a new session to continue the revamp of the book.

```text
You are continuing the revamp of the mdBook "GPU Architecture & Parallel Programming from First Principles" in
/home/jetson/Projects/gpu-parallel-book (book in book/, Rust code in code/oxide/).

Read, in full and in this order, before changing anything:
  1. START_HERE.md   rules, standards, tools, checks (this file)
  2. WORKLOG.md      what is done, the plan, the next item
  3. book/src/chapters/part5_oxide/ch13_your_first_rust_kernel.md
                     the reference chapter: match its shape, depth and voice

Take the next unchecked item in WORKLOG.md. Work one chapter at a time. Keep
`mdbook serve -n 0.0.0.0 -p 3000` running in book/ so the user reads changes live,
and stop after each chapter for their verdict.
```

## What the user asked for

A professional, modern rewrite that builds intuition from first principles and keeps cognitive load low (the user names readers with ADHD explicitly). The model is the user's two other books:

- `~/Projects/nvidia-cloud-software-engineer-interview/cracking-the-systems-programming-interview`
- `~/Projects/nvidia-cloud-software-engineer-interview/democratizing-linux`

Take from them: a character plate on every chapter, a "This chapter covers" box, the problem in plain words, a hand-worked example before any code, creative diagrams, code in small explained excerpts, the complete program at the end with its real output, a summary, and exercises.

## Standing rules (from the user)

- **The Rust part is pure cuda-oxide Rust.** Never show PTX, LLVM IR or assembly in the book, and do not dump it in the session either. Verify a kernel by running it and reading the PASSED line.
- **Every Rust chapter gets a standalone crate** in `code/oxide/<name>/` with `Cargo.toml`, `rust-toolchain.toml` and `src/main.rs`. Copy the manifest from `code/oxide/vector_add/` (cuda-oxide pinned to rev `b7fe543b863cdef46362e5f69797906b84d499c7`, toolchain `nightly-2026-04-03`). The chapter explains every line of `Cargo.toml` the first time a new dependency appears.
- **Every listing is real.** Build and run with `cargo oxide run` on this Jetson Orin NX 16GB, Super (sm_87, 8 SMs, CUDA 12.6). Paste the real output. When showing a mistake, really run the broken variant (copy the crate to the scratchpad, set `CARGO_TARGET_DIR` to the crate's own `target/`, then rebuild the correct version afterwards).
- **Use the safe launch path**: `#[launch_contract]` + `module.prepare_<kernel>(LaunchConfigND::new(..))` + the safe launch call. Only `kernels::load` stays `unsafe`, with a `// SAFETY:` comment.
- **Text never leaves its box** in art or figures. In `tools/art.py`, every label inside a box uses `fit=` (`textLength`), and `tag()` sizes the box from the text. In figures, `Figure.cell` fits text automatically, and multi-line boxes draw one `f.text` per line (a `\n` inside `cell` does not wrap).
- **Look at every picture** you make the way the reader's browser draws it: screenshot it with the headless Chromium at `~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux-arm64/chrome-headless-shell --no-sandbox --hide-scrollbars --window-size=W,H --screenshot=out.png file:///path.svg` and Read the PNG. cairosvg uses different fonts and hides overflow.
- Do not run `mdbook build` while `mdbook serve` is running: both write `book/book/` and the server's rebuild fails. Touch a source file instead.
- Do not rewrite the Foreword. Keep its opening line. (See also the user's memory notes.)
- **Code style.** No macros, few comments, a blank line between logical groups (setup, copies, launch, checks, printing). No `unwrap` or `expect` in Rust: functions return `Result<_, DriverError>` and use `?`. Keep lines short enough to fit the page (about 95 characters); split long signatures and `printf` calls.
- **Do not overfit the author's Jetson.** Explain on any GPU first (a table across Orin NX, AGX Thor, DGX Spark, RTX 4090/5090, A100, H100, GH200 where it applies); the Jetson's measured run is 1 example, with "run it on yours".
- **Every chapter ends with its complete programs.** After the last numbered section comes "N.M The complete programs": the whole C++ file and the whole Rust `main.rs`, as listings, so a reader can see each excerpt in place. Keep every source line at 92 characters or fewer so whole files fit the page.
- **Excerpts are whole blocks.** An excerpt that starts or ends inside a function shows the enclosing header, `// ...` and the closing brace. `tools/excerpts.py` does this (see Tools).

Read the "Agent conduct log" section of WORKLOG.md first: it records where an earlier agent
overstepped, and the rules that came out of it.

## How to write or rewrite a chapter

The user rejected two attempts at their own style (2026-10-18: "aggressive chatty AI slop", "you didn't read my books"). The model to copy is the user's own books. Read one chapter of each before writing:

- `~/Projects/nvidia-cloud-software-engineer-interview/democratizing-linux/src/07-virtual-memory.md`
- `~/Projects/nvidia-cloud-software-engineer-interview/cracking-the-systems-programming-interview/src/ch16-threads-locks.md`, its `WRITING_GUIDE.md`, and `../anti_ai_slop.md`
- `~/Projects/thor-thunder-tigress-platform/book/src/ch26-context-window.md`

ch01 is the reference chapter in this book. Rules:

- Calm, declarative sentences, each carrying one fact. No coaching ("Imagine", "Think of", "Now ask", "Look at"), no punchy openers, no analogies carrying the explanation (no ants, crumbs, "workers"), no vague "this"/"that" at the start of a sentence.
- **Define every term before its first use**, in a formal sentence, even if an earlier chapter defined it (restate briefly). In ch02 the user caught "kernel" and "block" used before any definition. Before showing a chapter, list its terms and check where each is first defined.
- Build up in order: basic quantities first (for example cycle, then latency, then throughput, then concurrency), each with a small example worked by hand, then the law, checked against the example.
- Boxes (CSS in `theme/custom.css`): `<div class="covers" markdown="1">`; `<div class="callout note|tip|warning" markdown="1">` with `**NOTE:**` etc.; a bee box `<div class="bee bee-shocked" markdown="1">` (moods: happy, cute, shocked, sleepy, angry) for the one fact to keep from a section; `<div class="lld" markdown="1">` for a design table; `<div class="summary" markdown="1">`. No "In short"/recap boxes.
- Figures: `tools/figkit.py` (copied from democratizing-linux): `card`, `tag`, `zone`, `wire`, `legend`, `save(path, w, h, alt, title, lede, parts)`. Draw at about 640 wide. `FIGKIT_REPORT=1` lists text that does not fit. One script per chapter, `tools/draw_chNN.py`. Replace old `assets/*.svg` and svgkit figures.
- Keep every correct fact and the section numbers (other chapters cite them; grep for `§N.` and `section N.` first). Use numbered subsections (1.1.1) like the reference books.
- Give every section the same rhythm, so a reader always knows what comes next: the question it answers; the definition; a worked example in `<div class="example" markdown="1">` (first line `Worked example: ...`, then numbered steps or a table); the figure; the formula with a `Symbol | Meaning | In the example` table; one bee box with the fact to keep; sometimes a `<details class="check"><summary>question</summary>answer</details>`.
- Open a long chapter with a map figure of its questions (see `ch01-map.svg`).
- Connectors in figures: straight vertical or horizontal only; a fan-out is a stem, a bus line, a junction dot and one arrow per target (`fan()` and `arrow()` in `tools/draw_ch01.py`). Arrow tips stop 3 units short of the target.
- Animations: calm card style like democratizing-linux (a chart or a few cards, one caption, a step rail; see `tools/anim_latency.py`). No robots, no empty code panels, no scoreboards. Prefer one animation that shows something a still figure cannot; drop animations that repeat a figure.
- Every number gets its reason in the same place: why that number, why divide by it (e.g. ridge point = FLOPs the GPU can do while memory delivers 1 byte; 100 threads need 4 warps because 100 ÷ 32 rounds up to 4). Work the arithmetic in the text.
- Write numbers as numerals (4 schedulers, 8 lanes), except "one" as a pronoun and ordinals. Careful with regex replacements: they also change file names (ch01-two-laws.svg was broken once); check every image path afterwards.
- Keep "the", "this" and "that" to a minimum. No vague "This is ..." openings: name the thing.
- Every new idea gets a diagram when a picture can carry it (e.g. kernel/launch/grid/block, thread index, the inside of an SM). Explain a term the moment a figure shows it (the thread index was once drawn but never explained).
- Compare GPUs broadly (embedded Jetson Orin NX and AGX Thor, DGX Spark, RTX 4090/5090, A100, H100) with sourced figures; never centre the text on the author's own board. Drop any figure that sources disagree on (B200 FP32).
- Before a figure goes in, screenshot it in headless Chromium and check every arrow lands on its target.
- Show every program in CUDA C++ and in cuda-oxide Rust side by side (C++ listing, then Rust listing, same step). Keep code lean: no macros, a comment only where a line is not obvious. Build and run both on the board and print the real output (see ch03: `code/ch03_vector_add/`, `code/oxide/ch03_basics/`).
- For indexing, draw it: 1-D blocks, 2-D image tiles, and 3-D blocks both as unfolded layers and as a cube with warps coloured (ch03-cube).
- Run `tools/lint_prose.py`; fix every banned phrase and the worst long sentences. A line that starts with `1.` inside a paragraph becomes a list; reflow it.
- Add a plate (WORKLOG cast plan), a "This chapter covers" box, a Summary and exercises.

## Chapter shape for new Rust chapters (copy ch13)

1. Plate (`<img class="plate" src="../../art/chNN.svg" alt="...">`), H1, the "This chapter covers" box.
2. Two short paragraphs: where the chapter sits, and what you build.
3. The problem in plain words, with a small table worked by hand.
4. The big idea, with a figure. One analogy from the hive cast, at most.
5. The tricky part (indexing, tiling, the tree of a reduction), worked by hand with small numbers and a figure.
6. Pseudocode in a `text` block.
7. Setup and `Cargo.toml`, for anything new.
8. Kernel, then host, in excerpts of a few lines, each explained before or after.
9. Run it: the real output, read line by line against the hand-worked numbers.
10. Performance, honestly: what the numbers mean (arithmetic intensity from Chapter 1).
11. When it goes wrong: real broken variants and their real output.
12. The complete program, a summary list, one sentence on the next chapter, then exercises.

## Second board: Jetson AGX Thor

`ssh thor` (sm_110, CUDA 13.2, 2 copy engines; reads pageable memory through host page tables).
Validate every chapter's code there too. C++: `nvcc -arch=native`. Rust: `. ~/.local/clang18/env.sh`
first (clang 18 unpacked into the home folder without root, for bindgen), then `cargo oxide run`.
The backend source in `~/.cargo/cuda-oxide/src` must stay checked out at rev b7fe543; a fresh
clone takes `main`, whose layout no longer builds. Scratch copies live in `~/gpu-book-check/`.

## Measuring on the boards

- **Warm up before timing.** The Jetsons raise their GPU clocks under sustained load; run the kernel for about 0.4 s first, then average many runs. Expect 10 to 20% drift between runs and say so in the text.
- **Exceed the L2 cache.** The Thor has 32 MiB of L2; data smaller than that measures the cache, not the memory. Print `prop.l2CacheSize` and size the test beyond it.
- **Check results, not just times.** Every program verifies its output (a total, a transpose element by element, PASSED).
- **Run on both boards** and quote both, with the Thor's integrated-memory differences explained where they change the conclusion.
- **One issue, then one PR per chapter, always into `main`.** Build each chapter's branch on the previous one so the history is linear, but never set a PR's base to another chapter's branch: GitHub then merges it there and it never reaches `main` (this happened to PRs #7, #9 and #11). State the merge order in each PR. No Claude attribution in commits, PRs or issues (user's instruction).

## Tools

Run from `book/`.

| Command | What it does |
|---|---|
| `python3 tools/art.py [names]` | cover and plates into `src/art/` (SVG + PNG) and `plates.json` |
| `python3 tools/draw_chNN.py` | that chapter's figures into `src/figures/` |
| `mdbook serve -n 0.0.0.0 -p 3000` | live preview at http://192.168.0.6:3000 |
| `mdbook build` | builds into `book/book/` |

Code excerpts are mdBook line-range includes, for example `{{#include ../../../../code/oxide/vector_add/src/main.rs:19:35}}`. mdBook does not strip `// ANCHOR` lines from a whole-file include, so do not use anchors. After editing a source file, re-check that every range still starts and ends where the prose expects (print the first and last line of each range).

`python3 tools/excerpts.py check | wrap | remap` keeps excerpts whole: `wrap` adds enclosing headers, `// ...` and closing braces around a partial excerpt (it owns every fence whose lines are all includes); `remap` moves ranges after a committed source changes (it diffs against `git HEAD`, so for untracked files shift the ranges by hand). Run `wrap` then `check` after any range change.

## Checks before every commit

1. `cargo oxide run` passes in every crate you touched.
2. `mdbook build` prints no warning, and the page has no `{{#include` or `ANCHOR` text.
3. Every new or changed figure and plate was rasterised and looked at.
