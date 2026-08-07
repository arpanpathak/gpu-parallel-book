#!/usr/bin/env python3
"""Build the print-ready PDF of the book with the Night Owl theme intact.

Pipeline:
  1. `mdbook build` the book (source -> book/book/)
  2. Inject a print stylesheet into book/book/print.html that forces
     Chrome to keep background colors (print-color-adjust: exact) and
     constrains SVG diagrams to the printable width.
  3. Serve book/book/ locally and render print.html to PDF with headless
     Chrome, giving MathJax time to typeset every formula.
  4. Write the result to <repo-root>/gpu-parallel-book.pdf

Why the injection is needed:
  Chrome's --print-to-pdf strips CSS background-color by default, turning
  the Night Owl dark pages white while the light body text stays light.
  The print.css below is embedded directly into print.html so the rule
  is present at render time regardless of mdBook's print CSS handling.

Usage:
  python3 make-pdf.py            # full rebuild + PDF
  python3 make-pdf.py --no-build # skip mdbook build (use existing output)
"""
import argparse
import http.server
import os
import re
import socketserver
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.join(ROOT, "book")          # source book (book.toml lives here)
OUT_DIR = os.path.join(BOOK_DIR, "book")       # mdBook output
PRINT_HTML = os.path.join(OUT_DIR, "print.html")
PDF_OUT = os.path.join(ROOT, "gpu-parallel-book.pdf")

# The print stylesheet injected into print.html. Everything is scoped to
# @media print so the on-screen site is untouched.
PRINT_CSS = """
@page { margin: 0; }  /* remove Chrome's white print margins: dark bg must fill the whole page */
@media print {
  /* Force Chrome to render backgrounds in the PDF (Night Owl theme).
     Explicit hex values, never var(--bg): the CSS variable override does
     not survive the print pipeline, so we hard-code every colour. */
  * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  html, body, .page-wrapper, #mdbook-page-wrapper, .content {
    background-color: #011627 !important; color: #d6deeb !important;
  }
  h1, h2, h3, h4, h5, h6 { color: #82aaff !important; }
  pre { background: #01111f !important; border: 1px solid #0b2942 !important;
        color: #d6deeb !important; white-space: pre-wrap !important; word-wrap: break-word !important; }
  pre code, code { background: transparent !important; color: #addb67 !important; }
  .content code { background: rgba(130,170,255,0.10) !important; color: #addb67 !important; }
  table { border-color: #1d3b53 !important; }
  th, td { border-color: #1d3b53 !important; background: transparent !important; }
  th { color: #82aaff !important; }
  blockquote { background: rgba(130,170,255,0.06) !important; border-left: 3px solid #82aaff !important; }
  /* Content needs breathing room now that @page margin is 0. */
  .content { padding: 1.5em 2.5em !important; }
  /* Diagrams: constrain to printable width, centre, never split across pages. */
  .content img { max-width: 100% !important; height: auto !important; display: block;
                 margin: 1.2em auto !important; page-break-inside: avoid; break-inside: avoid; }
  .content img[src$=".svg"] { max-width: 92% !important; }
  /* Clean pagination. */
  h1 { page-break-before: always; }
  h2, h3, h4 { page-break-after: avoid; }
  blockquote, table, pre, ul, ol { page-break-inside: avoid; }
}
"""


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def inject_print_css():
    with open(PRINT_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    marker = "<!-- injected:gpu-print-css -->"
    if marker in html:
        html = html.split(marker, 1)[0]
    style = f"<style id=\"gpu-print-css\">{PRINT_CSS}</style><!-- {marker} -->"
    # insert after the first <link rel="stylesheet" ... print-*.css> if present,
    # otherwise right after <head>
    m = re.search(r'<link rel="stylesheet" href="css/print[^"]*"[^>]*>', html)
    if m:
        html = html[: m.end()] + style + html[m.end():]
    else:
        html = html.replace("<head>", "<head>" + style, 1)
    with open(PRINT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print("injected print CSS into", PRINT_HTML)


def serve_and_print():
    os.chdir(OUT_DIR)

    handler = http.server.SimpleHTTPRequestHandler

    class Quiet(handler):
        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 8123), Quiet) as httpd:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        url = "http://127.0.0.1:8123/print.html"
        print("serving", url)
        # Chrome's CLI --print-to-pdf strips backgrounds; the CDP
        # Page.printToPDF call with printBackground=true is the only way to
        # keep the Night Owl dark pages. print-to-pdf.cjs drives Chrome via
        # the DevTools Protocol and waits for MathJax to finish typesetting.
        run(["node", os.path.join(ROOT, "print-to-pdf.cjs"), url, PDF_OUT])
        httpd.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build:
        run(["mdbook", "build", BOOK_DIR], cwd=ROOT)
    inject_print_css()
    serve_and_print()
    print("PDF written to", PDF_OUT)


if __name__ == "__main__":
    main()
