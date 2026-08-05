# Contributing to the GPU Parallel Book

Thank you for considering a contribution. This repository follows standard
open-source practice: discussions in issues, review in pull requests, and a
code of conduct that applies to everyone (see CODE_OF_CONDUCT.md).

## What kind of contributions are welcome

- **Corrections.** A kernel that is wrong, a formula that is misleading, a
  number that does not match the hardware: these are the highest-value
  contributions. If you find one, open an issue or submit a PR.
- **Clarifications.** If a section confused you, it will confuse the next
  reader. Propose rewording.
- **Missing concepts.** The book's promise is "every primitive is explained".
  If a term is used without being defined, that is a bug in the book.
- **Examples and exercises.** Additional worked examples and exercises (with
  or without solutions) that follow the existing style.

## Before you start

1. Read `CODING_STANDARDS.md`. Every code block in the book follows it, and
   so must yours.
2. Check open issues and PRs to avoid duplicating work.
3. For substantial changes, open an issue first to discuss the approach.

## Style conventions

- **No em dashes or en dashes.** The book uses ASCII hyphens only. Run
  `rg "[—–]" book/src` before submitting; it must return nothing.
- **Formal British English** (American spellings tolerated), no contractions
  in prose.
- **Every primitive explained.** If your section introduces a new CUDA API,
  type, or built-in variable, define it before using it.
- **Every code block commented line by line**, with the reasoning for each
  non-obvious line.
- **No magic numbers** (see CODING_STANDARDS.md).
- **Fully commented kernels** that a reviewer can verify from the comments
  alone: index arithmetic, memory accesses, synchronisation rationale.

## Building locally

Requires [mdBook](https://rust-lang.github.io/mdBook/) v0.5.x:

```bash
mdbook build book      # compiles to book/book/
mdbook serve book      # live preview at http://localhost:3000
```

The book must build with zero warnings before a PR is merged.

## Verifying code in the book

CUDA kernels cannot be compiled on machines without the NVIDIA CUDA toolkit.
Contributors with access to `nvcc` are encouraged to compile the `.cu`
snippets (`nvcc -arch=sm_90 -o /tmp/x <snippet>.cu`) and note the result in
the PR. Contributors without `nvcc` should state so; the maintainers will
verify on CI hardware.

## Submitting a pull request

1. Fork the repository and create a feature branch (`git checkout -b
   fix/chapter-7-coalescing`).
2. Make your change, following the style conventions above.
3. Verify: `mdbook build book` produces zero warnings.
4. Commit with a clear message, e.g. `Fix bank-conflict arithmetic in
   Chapter 7`.
5. Open the PR against `main`. In the description, summarise the change and
   (if applicable) how you verified it.

## Review process

- Every PR is reviewed by at least one maintainer.
- The book is a living document: the reviewer may request changes that keep
  the voice and structure consistent with the rest of the text.
- Be patient and kind; this project is a volunteer effort.

## Reporting bugs

Open an issue with the **chapter**, the **section**, and a description of the
problem. For security-related reports, see SECURITY.md instead.
