# Chapter edit prompt

Run this once per chapter, in a new conversation each time. Attach `WRITING_STANDARDS.md`, `CODING_STANDARDS.md` and the chapter's Markdown file, then paste the prompt below.

---

You are editing one chapter of a technical book on CUDA C++ and Rust GPU programming. The draft was written by an AI model and has the usual habits: dramatic openings, "not X but Y" contrasts, numbers without sources, repeated material and closing lines that restate the section.

The attached files are:
- `WRITING_STANDARDS.md`: the prose rules. Follow them, and make your edits read like the "After" examples in section 4.
- `CODING_STANDARDS.md`: the code rules.
- The chapter to edit.

Work in two passes.

**Pass 1: technical review. Do not edit anything yet.** List:
1. Every number, benchmark or performance claim, and whether the chapter shows where it comes from.
2. Every CUDA term, API name or function signature you cannot confirm from NVIDIA's documentation.
3. Every claim made about all GPUs rather than a named architecture or CUDA version.
4. Every place where the prose and the code beside it disagree.
5. Every code block that breaks a rule in `CODING_STANDARDS.md`.

For each item, give the line or heading, the problem, and what a human should check.

**Pass 2: edit the prose.**
1. Cut before you rewrite. Delete any sentence whose removal loses no information. The chapter should get shorter.
2. Keep every technical fact, definition, heading and code block. Do not change code; report code problems in the change log.
3. Do not add facts, numbers, examples or analogies that are not in the draft.
4. For each unverified claim from Pass 1: if it carries information, keep it and add `<!-- VERIFY: reason -->` after it; if it is decoration, delete it.
5. Keep the book's terminology and British spelling. Do not rename a term, even one flagged in Pass 1.
6. If a passage repeats material from another chapter, replace it with one sentence and a `<!-- LINK: topic -->` marker.

**Output, in this order:**
1. The Pass 1 list.
2. The complete edited chapter in Markdown, in one code block.
3. A change log with one line per substantive change, followed by the word count before and after.

---

## Follow-up check

Open another new conversation, attach `WRITING_STANDARDS.md` and the edited chapter, and paste:

> List every sentence in this chapter that breaks a rule in `WRITING_STANDARDS.md`. Quote the sentence, name the rule, and suggest the smallest fix. Do not rewrite anything else.

Apply the fixes you agree with, then run the grep checks in section 5 of `WRITING_STANDARDS.md`.
