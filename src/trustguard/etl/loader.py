"""Document loaders for PDF and plain-text policy files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".txt"})


@dataclass(frozen=True)
class PageText:
    r"""Text extracted from a single page of a source document.

    Attributes:
        text: Cleaned page text.
        source: File name of the document the page came from.
        page: 1-indexed page number. Plain-text files use form feeds
            (``\\f``) as page breaks and otherwise count as a single page.
    """

    text: str
    source: str
    page: int


def _normalize(text: str) -> str:
    """Collapse runs of spaces and blank lines produced by PDF extraction."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SOFT_WRAP = re.compile(r"(?<=[^\n.:;!?])\n(?=[a-z$(])")


def _unwrap(text: str) -> str:
    """Join lines that PDF layout wrapped mid-sentence.

    A newline is treated as a soft wrap when the previous line does not end a
    sentence and the next line starts with a lowercase letter, ``$`` or ``(``.
    Clause numbers and headings start with digits or capitals, so they keep
    their line breaks.
    """
    return _SOFT_WRAP.sub(" ", text)


def load_pdf(path: Path) -> list[PageText]:
    """Extract text from each page of a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        One `PageText` per non-empty page.
    """
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = _unwrap(_normalize(page.extract_text() or ""))
        if text:
            pages.append(PageText(text=text, source=path.name, page=index))
    return pages


def load_txt(path: Path) -> list[PageText]:
    """Read a UTF-8 text file, splitting pages on form-feed characters.

    Args:
        path: Path to the text file.

    Returns:
        One `PageText` per non-empty page.
    """
    raw = path.read_text(encoding="utf-8")
    pages: list[PageText] = []
    for index, chunk in enumerate(raw.split("\f"), start=1):
        text = _normalize(chunk)
        if text:
            pages.append(PageText(text=text, source=path.name, page=index))
    return pages


def load_document(path: Path) -> list[PageText]:
    """Load a supported document based on its file extension.

    Args:
        path: Path to a `.pdf` or `.txt` file.

    Returns:
        The document's pages.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}")


def discover_documents(directory: Path) -> list[Path]:
    """List supported documents under a directory (recursively), sorted by path."""
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
