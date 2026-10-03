# WRITING_STANDARDS.md

These rules apply to every paragraph of prose in the book. `CODING_STANDARDS.md` covers the code.

## 1. Accuracy

1. Every number has a source or a derivation shown in the text. If it has neither, remove it or mark it with `<!-- VERIFY: reason -->`.
2. State the scope of every general claim: which architecture, which CUDA version, which compute capability. Never write "every GPU" or "all GPUs".
3. CUDA terms, API names and signatures match NVIDIA's documentation exactly. A term the book introduces itself is defined once and spelled the same way everywhere.
4. Prose next to a code block describes that code. If the two disagree, the code is the reference and the prose is fixed.

## 2. Sentences

1. One idea per sentence. Put the subject and verb near the start.
2. Be concrete. Write "Pageable memory forces the driver to copy data into a pinned staging buffer", not "Memory choices can affect performance".
3. Use second person, present tense and active voice: "You allocate the buffer", not "The buffer is allocated".
4. Use British spelling throughout (synchronisation, optimisation, greyscale, programme for a scheme, program for software).
5. Do not use these constructions:
   - "not X but Y", "X, not Y" and "not just X, it's Y" as a rhetorical device. State Y.
   - A dramatic opening line that states no fact ("Computing is entering its parallel age").
   - A closing line that restates the section or draws a moral ("...and neither should the book").
   - Rhetorical questions followed by their answer.
   - Groups of three adjectives or phrases added for rhythm.
   - Hype: "revolutionary", "game-changing", "powerful", "incredibly", "blazing".
   - Filler: "it is worth noting that", "in essence", "at its core", "simply put", "let's dive in".
   - AI vocabulary: delve, tapestry, testament, seamless, robust, leverage, landscape, realm.
6. Use a dash only where a comma, colon or full stop will not do. No more than one per paragraph.
7. Use a metaphor only when it explains a mechanism the reader could not otherwise picture. Never use one for decoration.

## 3. Paragraphs and structure

1. Open each section with what the reader will learn or do, in one sentence. No scene-setting.
2. End a section when its content ends. No summary paragraph unless the chapter is longer than 5,000 words.
3. Say a thing once. If another chapter needs it, link to it.
4. Use a list only for parallel items the reader will scan: steps, options, prerequisites. Reasoning goes in prose.
5. Use bold only for a term at the point where it is defined.
6. Before a code block, say what it does and why. After it, point out only what the reader might miss. Do not repeat the code comments in prose.

## 4. Examples

These pairs are taken from the current Foreword. Edits should read like the "After" column.

**Before:** Computing is entering its parallel age, and the GPU is the machine that defines it.
**After:** *(deleted: it states no fact)*

**Before:** That path ran into a wall in the mid-2000s, when clock speeds stopped climbing and silicon stopped cooperating. The industry's answer was not surrender but a change of question - from "how fast can one core go?" to "how many cores can we set free at once?"
**After:** Around 2005, single-core clock speeds stopped rising because of power and heat limits. Since then, most performance gains have come from adding cores.

**Before:** ...and keeps more threads in flight than there are people on Earth...
**After:** *(deleted, or replaced with a sourced thread count for a named GPU)*

**Before:** Treat those chapters as a map of the territory, not a surveyor's certificate.
**After:** Expect the code in those chapters to need changes as CUDA-Oxide develops.

**Before:** These rules are not here to annoy you. Every violation listed above has been, at some point, a production incident or a silent numerical error.
**After:** Each rule targets a class of bug that produces wrong results without an error message.

**Before:** The GPU does not stop changing, and neither should the book.
**After:** *(deleted: the previous sentence already asks for bug reports)*

## 5. Mechanical checks

Run these before merging. Every hit needs a human look, since some are false positives.

```sh
# Slop vocabulary and filler
grep -rniE "delve|tapestry|testament|seamless|robust|leverage|landscape|realm|game.?chang|revolutionar|it is worth noting|in essence|at its core|simply put|dive in" book/src

# "not X but Y" and "not just X" contrasts
grep -rniE "not (just|only|merely) |, not (a |an |the )?[a-z]+[.;]| not [a-z]+ but " book/src

# Mid-sentence dashes (list markers at line start are excluded)
grep -rnE "[a-z,] - [a-z]" book/src

# Universal claims
grep -rniE "every gpu|all gpus|ever built|always faster|never fails" book/src

# Unresolved verification markers
grep -rn "VERIFY:" book/src
```
