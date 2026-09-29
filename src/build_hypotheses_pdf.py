"""Render report/hypotheses.md (+ its figures) to a single, self-contained
PDF: report/hypotheses.pdf.

Markdown -> styled, self-contained HTML (images inlined as base64 data
URIs) -> PDF via headless Chrome's native --print-to-pdf. No pandoc /
wkhtmltopdf / weasyprint dependency -- just the `markdown` package (already
in .venv) and a local Chrome install.
"""
from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
MD_PATH = REPORT_DIR / "hypotheses.md"
HTML_PATH = REPORT_DIR / "hypotheses.html"
PDF_PATH = REPORT_DIR / "hypotheses.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
    font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: #20242B;
    line-height: 1.55;
    font-size: 10.5pt;
    max-width: 760px;
    margin: 0 auto;
}
h1 { font-size: 19pt; border-bottom: 2.5px solid #3B6EA5; padding-bottom: 6px; }
h2 {
    font-size: 14.5pt; margin-top: 26px; color: #C1443C;
    border-bottom: 1px solid #D8DCE1; padding-bottom: 3px;
}
h3 { font-size: 12pt; margin-top: 18px; color: #20242B; }
hr { border: none; border-top: 1px solid #D8DCE1; margin: 22px 0; }
p, li { font-size: 10.3pt; }
code {
    background: #F1F3F5; padding: 1px 5px; border-radius: 3px;
    font-size: 9.3pt; font-family: "SF Mono", Menlo, monospace;
}
pre { background: #F1F3F5; padding: 10px; border-radius: 5px; overflow-x: auto; }
strong { color: #20242B; }
table {
    border-collapse: collapse; width: 100%; margin: 12px 0 18px;
    font-size: 9.3pt;
}
th, td { border: 1px solid #D8DCE1; padding: 5px 8px; text-align: left; }
th { background: #EFF3F8; font-weight: 600; }
tr:nth-child(even) td { background: #FAFBFC; }
img {
    max-width: 100%; display: block; margin: 14px auto;
    border: 1px solid #D8DCE1; border-radius: 4px;
    page-break-inside: avoid;
}
a { color: #3B6EA5; text-decoration: none; }
blockquote {
    border-left: 3px solid #E1A948; margin: 10px 0; padding: 4px 14px;
    color: #444; background: #FFFBF2;
}
h2 { page-break-before: auto; }
"""


def _find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    found = shutil.which("chromium") or shutil.which("chromium-browser")
    if found:
        return found
    raise RuntimeError("No Chrome/Chromium/Edge install found for --print-to-pdf.")


def _inline_images(html: str, base_dir: Path) -> str:
    """Replace <img src="relative/path.png"> with a base64 data: URI so
    the HTML (and the PDF Chrome renders from it) is fully self-contained."""
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

    def repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = (base_dir / src).resolve()
        ext = path.suffix.lower()
        if not path.exists() or ext not in mime:
            print(f"  ! image not inlined (missing): {src}", file=sys.stderr)
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return m.group(0).replace(src, f"data:{mime[ext]};base64,{data}")

    return re.sub(r'<img[^>]*\ssrc="([^"]+)"', repl, html)


def build_html() -> Path:
    md_text = MD_PATH.read_text()
    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    body = _inline_images(body, base_dir=REPORT_DIR)
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Assignment 3 -- Hypotheses</title><style>{CSS}</style>"
        f"</head><body>{body}</body></html>"
    )
    HTML_PATH.write_text(html)
    print("wrote", HTML_PATH.relative_to(ROOT))
    return HTML_PATH


def build_pdf() -> Path:
    chrome = _find_chrome()
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--print-to-pdf={PDF_PATH}", "--no-pdf-header-footer",
         "--print-to-pdf-no-header", HTML_PATH.resolve().as_uri()],
        check=True, capture_output=True,
    )
    print("wrote", PDF_PATH.relative_to(ROOT))
    return PDF_PATH


if __name__ == "__main__":
    build_html()
    build_pdf()
