# Security Policy

This repository hosts a book: Markdown sources, CSS, and a CI workflow. It
does not ship executable CUDA kernels, and the code snippets are teaching
material, not production libraries. Nonetheless, we take security reports
seriously.

## Reporting a vulnerability

Do **not** open a public issue for security problems. Instead, report them
privately by email to the repository owner (see the git author information in
the repository) with the subject line `[gpu-parallel-book] security report`.

Please include:

1. A description of the issue and its impact.
2. The affected file and section (for content) or workflow (for CI).
3. A minimal reproduction, if applicable.

## What is in scope

- Malicious or broken instructions in the book's code snippets that could
  cause data loss or system damage if followed literally.
- Compromised dependencies in the GitHub Actions workflow
  (`actions/checkout`, `actions/configure-pages`, etc.).
- Phishing or malicious links in the book or README.

## What is out of scope

- General CUDA programming mistakes (the book exists to teach these; report
  them as normal issues instead).
- Vulnerabilities in NVIDIA's CUDA toolkit, drivers, or libraries; report
  those to NVIDIA directly.

## Response

We will acknowledge receipt within 7 days and aim to resolve or triage the
issue within 30 days. As this is a volunteer project, response times may
vary.
