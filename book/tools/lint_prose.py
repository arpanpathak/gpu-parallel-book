#!/usr/bin/env python3
"""Flag slop in chapter prose: banned phrases and long sentences.

    python3 tools/lint_prose.py src/ch01-*.md

Code blocks, HTML tags, tables, and include directives are skipped.
"""
import re
import sys

BANNED = [
    "why it matters", "what matters", "it matters", "matters", "the key idea", "the key point", "is the key", "key idea", "the point is", "is the point", "that is the point",
    "worth", "important", "crucial", "powerful", "elegant", "simply", " just ", "note that",
    "interestingly", "surprisingly", "here's", "here is where", "let's be honest", "the truth is",
    "delve", "tapestry", "landscape", "unlock", "testament", "intersection of", "fast-paced",
    "nuanced", "one-size", "depends on your needs", "have their merits", "in summary",
    "at the end of the day", "bottom line", "hope this helps", "game-changer", "makes all the difference",
    "earns its place", "real story", "not just", "no fluff", "—", "–",
    # The same families, phrased the ways they actually turn up. Every entry here
    # is a sentence that can be deleted without losing a fact.
    "is what makes", "what makes it", "is what lets", "all the difference", "the whole point",
    "the point of", "is really the", "this is where", "is where the", "not merely", "simply put",
    "let that sink in", "here is the thing", "it is worth", "is worth noting", "important to note",
    "crucially", "seamlessly", "leverage", "deep dive", "at a high level", "the beauty of",
    "at its core", "in essence", "essentially", "boils down", "the magic", "superpower",
    "rewrite the rules", "cannot be overstated", "a must", "paradigm", "holistic", "synergy",
    "dive into", "embark", "journey", "realm", "plethora", "myriad", "it depends",
]
MAX_WORDS = 24


def prose_lines(text):
    in_code = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.startswith("|") or "{{#include" in line:
            continue
        if line.lstrip().startswith("<"):
            # Captions are prose too; every other HTML line is markup.
            caption = re.search(r"<figcaption>(.*?)</figcaption>", line)
            if caption:
                text = re.sub(r"<b>[^<]*</b>", "", caption.group(1))
                text = re.sub(r"<code>([^<]*)</code>", r"`\1`", text)
                yield number, re.sub(r"<[^>]+>", "", text)
            continue
        yield number, line


def main(paths):
    problems = 0
    for path in paths:
        text = open(path, encoding="utf-8").read()
        lines = list(prose_lines(text))
        for number, line in lines:
            lower = re.sub(r"`[^`]*`", "", line).lower()
            for phrase in BANNED:
                pattern = re.escape(phrase.strip())
                if phrase.strip()[:1].isalnum():
                    pattern = r"\b" + pattern
                if phrase.strip()[-1:].isalnum():
                    pattern = pattern + r"\b"
                if re.search(pattern, lower):
                    print(f"{path}:{number}: banned phrase {phrase.strip()!r}: {line.strip()[:90]}")
                    problems += 1
        paragraphs, current = [], []
        for _, line in lines:
            stripped = line.strip()
            starts_block = stripped.startswith(("#", "- ", "* ", ">")) or re.match(r"\d+\. ", stripped)
            if not stripped or starts_block:
                if current:
                    paragraphs.append(" ".join(current))
                current = [] if not stripped else [stripped.lstrip("#->*0123456789. ")]
                continue
            current.append(stripped)
        if current:
            paragraphs.append(" ".join(current))
        for paragraph in paragraphs:
            prose = re.sub(r"`[^`]*`", "CODE", paragraph)
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z*`])", prose):
                words = sentence.split()
                if len(words) > MAX_WORDS:
                    print(f"{path}: long sentence ({len(words)} words): {sentence.strip()[:110]}")
                    problems += 1
    print(f"{problems} finding(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
