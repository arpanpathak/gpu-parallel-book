#!/usr/bin/env python3
"""Keep the book's code excerpts whole and pointing at the right lines.

The chapters show code with mdBook includes, `{{#include path:start:end}}`. Two
things go wrong with line ranges, and this tool fixes both.

* An excerpt that starts inside a block (a method inside an `impl`, a few lines
  inside a function) reads as broken code. `wrap` puts the enclosing headers
  around it, copied from the source, marks the code it skips with `// ...`, and
  closes every brace it opened.
* Editing or formatting a source file moves its lines. `remap` finds each
  excerpt's code again in the edited file, by aligning the committed version
  with the working copy character by character (ignoring whitespace), and
  rewrites the range and the "(lines A to B)" of the listing caption above it.

    python3 tools/excerpts.py check    # report excerpts that start or end mid-block
    python3 tools/excerpts.py wrap     # add or refresh the wrappers
    python3 tools/excerpts.py remap    # after editing sources: remap ranges, then wrap

Adapted from cracking-the-systems-programming-interview for this book: chapters live in
nested folders, so includes resolve relative to each page, and C++ fences are handled too.

A code fence whose lines are all includes is owned by this tool: `wrap`
regenerates every other line in it. Write prose outside the fence.
"""

from __future__ import annotations

import bisect
import difflib
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INCLUDE = re.compile(r"^\{\{#include ([^}:]+)(?::(\d+):(\d+))?\}\}$")
KEYWORD = re.compile(r"^(pub(\([^)]*\))?\s+)?(unsafe\s+)?(async\s+)?(impl|mod|trait|fn|struct|enum|loop|match|for|while|if|else)\b")
ELIDE = "// ..."


def indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def code_only(line: str) -> str:
    line = re.sub(r"//.*", "", line)
    line = re.sub(r'r#"(.*?)"#', '""', line)
    line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
    return re.sub(r"'(\\.|[^'\\])'", "''", line)


def depth_change(line: str) -> int:
    body = code_only(line)
    return body.count("{") - body.count("}")


def enclosing(src: list[str], start: int) -> list[tuple[int, int]]:
    """The blocks around line `start` (1-based), outermost first, as
    (first header line, the line holding the opening brace), both 0-based."""
    body = [line for line in src[start - 1:] if line.strip()]
    if not body:
        return []
    level = indent(body[0])
    chain = []
    j = start - 2
    while level > 0 and j >= 0:
        line = src[j]
        text = line.strip()
        if text and indent(line) < level and not text.startswith(("//", "#[")) and text.endswith("{"):
            k = j
            # a header can span lines (a where clause): walk up to its first line
            while k > 0 and not (KEYWORD.match(src[k].strip()) and indent(src[k]) == indent(line)) \
                    and src[k - 1].strip() and not src[k - 1].rstrip().endswith((";", "}")):
                k -= 1
            chain.insert(0, (k, j))
            level = indent(line)
        j -= 1
    return chain


def block_end(src: list[str], opener: int) -> int:
    """The 0-based line that closes the block opened on line `opener`."""
    depth = 0
    for i in range(opener, len(src)):
        depth += depth_change(src[i])
        if depth <= 0 and i > opener:
            return i
    return len(src) - 1


def wrapped(src: list[str], a: int, b: int, include_line: str) -> list[str]:
    """The include line with its enclosing headers, elisions, and closers."""
    chain = enclosing(src, a)
    before, after = [], []
    for depth, (first, brace) in enumerate(chain):
        before += src[first:brace + 1]
        inner = chain[depth + 1][0] if depth + 1 < len(chain) else a - 1
        # something sits between this header and the next thing shown
        between = [line for line in src[brace + 1:inner] if line.strip()]
        if between:
            before.append(" " * indent(between[0]) + ELIDE)
    for depth, (first, brace) in reversed(list(enumerate(chain))):
        close = block_end(src, brace)
        inner_end = block_end(src, chain[depth + 1][1]) if depth + 1 < len(chain) else b - 1
        between = [line for line in src[inner_end + 1:close] if line.strip()]
        if between:
            after.append(" " * indent(between[0]) + ELIDE)
        after.append(src[close])
    if not chain:
        # an excerpt at the top level that stops before its closing braces
        open_braces = sum(depth_change(line) for line in src[a - 1:b])
        if open_braces > 0:
            after = [" " * 4 * open_braces + ELIDE]
            after += [" " * 4 * depth + "}" for depth in reversed(range(open_braces))]
    return before + [include_line] + after


PAGE = SRC


def source(path: str) -> list[str]:
    """The lines of an included file; `path` is relative to the page being processed."""
    return (PAGE / path).resolve().read_text().split("\n")


def fences(lines: list[str]):
    """(start, end) of every ```rust fence whose non-blank lines are includes
    or lines this tool generated."""
    i = 0
    while i < len(lines):
        if lines[i].startswith("```") and lines[i].strip() != "```":
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            body = lines[i + 1:j]
            if any(INCLUDE.match(line.strip()) and INCLUDE.match(line.strip()).group(2) for line in body):
                yield i, j
            i = j
        i += 1


def rebuild_fence(body: list[str]) -> list[str]:
    includes = [line.strip() for line in body if INCLUDE.match(line.strip())]
    out = []
    for n, line in enumerate(includes):
        m = INCLUDE.match(line)
        if n:
            out.append("")
        if m.group(2):
            out += wrapped(source(m.group(1)), int(m.group(2)), int(m.group(3)), line)
        else:
            out.append(line)
    return out


def wrap() -> int:
    changed = 0
    global PAGE
    for page in sorted(SRC.rglob("*.md")):
        PAGE = page.parent
        lines = page.read_text().split("\n")
        out, last = [], 0
        for start, end in fences(lines):
            out += lines[last:start + 1] + rebuild_fence(lines[start + 1:end])
            last = end
        out += lines[last:]
        if out != lines:
            page.write_text("\n".join(out))
            changed += 1
    print("wrap: %d chapter(s) changed" % changed)
    return 0


def remap() -> int:
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    cache: dict[str, tuple] = {}

    def mapper(path: str):
        if path not in cache:
            real = (SRC / path).resolve()
            rel = os.path.relpath(real, top)
            old = subprocess.check_output(["git", "show", "HEAD:" + rel], text=True).split("\n")
            new = real.read_text().split("\n")

            def stream(lines):
                chars, where = [], []
                for n, line in enumerate(lines, 1):
                    for ch in line:
                        if not ch.isspace():
                            chars.append(ch)
                            where.append(n)
                return "".join(chars), where

            old_text, old_where = stream(old)
            new_text, new_where = stream(new)
            first, last = {}, {}
            for i, n in enumerate(old_where):
                first.setdefault(n, i)
                last[n] = i
            blocks = [b for b in difflib.SequenceMatcher(None, old_text, new_text, autojunk=False)
                      .get_matching_blocks() if b.size]
            starts = [b.a for b in blocks]

            def to_new(i):
                k = max(bisect.bisect_right(starts, i) - 1, 0)
                b = blocks[k]
                if b.a <= i < b.a + b.size:
                    return new_where[b.b + i - b.a]
                return new_where[blocks[min(k + 1, len(blocks) - 1)].b]

            cache[path] = (first, last, to_new)
        return cache[path]

    moved = 0
    for page in sorted(SRC.glob("*.md")):
        lines = page.read_text().split("\n")
        for i, line in enumerate(lines):
            m = INCLUDE.match(line.strip())
            if not (m and m.group(2)):
                continue
            a, b = int(m.group(2)), int(m.group(3))
            first, last, to_new = mapper(m.group(1))
            na, nb = to_new(first[a]), to_new(last[b])
            if (na, nb) == (a, b):
                continue
            moved += 1
            lines[i] = line.replace(":%d:%d}}" % (a, b), ":%d:%d}}" % (na, nb))
            for k in range(i - 1, max(i - 12, -1), -1):
                if "(lines %d to %d)" % (a, b) in lines[k]:
                    lines[k] = lines[k].replace("(lines %d to %d)" % (a, b), "(lines %d to %d)" % (na, nb))
                    break
        page.write_text("\n".join(lines))
    print("remap: %d excerpt(s) moved" % moved)
    return wrap()


def check() -> int:
    global PAGE
    problems = 0
    for page in sorted(SRC.rglob("*.md")):
        PAGE = page.parent
        for n, line in enumerate(page.read_text().split("\n"), 1):
            m = INCLUDE.match(line.strip())
            if not (m and m.group(2)):
                continue
            src = source(m.group(1))
            a, b = int(m.group(2)), int(m.group(3))
            if b > len(src) or a < 1 or a > b:
                print("%s:%d: range %d:%d is outside the file" % (page.name, n, a, b))
                problems += 1
                continue
            shown = src[a - 1:b]
            if not shown[0].strip() or not shown[-1].strip():
                print("%s:%d: %s:%d:%d starts or ends on a blank line" % (page.name, n, m.group(1), a, b))
                problems += 1
            total = sum(depth_change(x) for x in shown)
            low = run = 0
            for x in shown:
                run += depth_change(x)
                low = min(low, run)
            if total < 0 or low < 0:
                print("%s:%d: %s:%d:%d closes a block it did not open" % (page.name, n, m.group(1), a, b))
                problems += 1
    print("check: %d problem(s)" % problems)
    return 1 if problems else 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    sys.exit({"check": check, "wrap": wrap, "remap": remap}[command]())
