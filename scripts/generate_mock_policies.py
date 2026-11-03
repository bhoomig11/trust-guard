"""Render the mock policy corpus to PDF and TXT files in data/policies/.

Usage:
    python scripts/generate_mock_policies.py [--out data/policies]

Each source page becomes exactly one PDF page (or one form-feed separated
section in TXT files), so page numbers in citations are deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_policy_content import DOCUMENTS, INSTITUTION  # noqa: E402


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=16, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, spaceBefore=8),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10, leading=13),
        "meta": ParagraphStyle("meta", parent=base["BodyText"], fontSize=8, textColor="#555555"),
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_pdf(pages: list[str], path: Path) -> None:
    styles = _styles()
    story: list[object] = []
    for page_number, page in enumerate(pages):
        if page_number:
            story.append(PageBreak())
        story.append(Paragraph(_escape(INSTITUTION), styles["meta"]))
        story.append(Spacer(1, 4))
        for line in page.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                story.append(Paragraph(_escape(line[2:]), styles["title"]))
            elif line.startswith("## "):
                story.append(Paragraph(_escape(line[3:]), styles["h2"]))
            else:
                story.append(Paragraph(_escape(line), styles["body"]))
                story.append(Spacer(1, 3))
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=path.stem.replace("_", " ").title(),
        author="Cobalt Ridge Financial (fictional)",
    )
    doc.build(story)


def render_txt(pages: list[str], path: Path) -> None:
    rendered = []
    for page in pages:
        lines = [INSTITUTION, ""]
        for line in page.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                lines.append(stripped[2:].upper())
            elif stripped.startswith("## "):
                lines.extend(["", stripped[3:]])
            else:
                lines.append(stripped)
        rendered.append("\n".join(lines).strip() + "\n")
    path.write_text("\f".join(rendered), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/policies"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCUMENTS.items():
        path = args.out / name
        if path.suffix == ".pdf":
            render_pdf(pages, path)
        else:
            render_txt(pages, path)
        print(f"wrote {path} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
