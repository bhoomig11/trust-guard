"""Clause-aware text chunking built on LangChain's recursive character splitter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from trustguard.etl.loader import PageText

# Ordered from strongest to weakest boundary so that numbered policy clauses
# and paragraphs stay intact whenever they fit inside one chunk.
CLAUSE_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# Top-level numbered section headings, e.g. "7. Suspicious Activity Reports (SAR)".
_HEADING = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][^\n]{2,80})$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance metadata.

    Attributes:
        chunk_id: Deterministic id (stable across re-ingestion of the same text).
        text: The chunk text shown to the model and returned as evidence.
        source: Source document file name.
        page: 1-indexed page number.
        chunk_index: Position of the chunk within its page.
        title: Human-readable document title.
        section: Nearest preceding numbered section heading, if any.
    """

    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int
    title: str = ""
    section: str = ""

    @property
    def embedding_text(self) -> str:
        """Text used for the embedding: the chunk prefixed with its document context.

        Prefixing the title and section ("contextual chunk headers") lets a clause
        such as "filed within 30 days" match a query about SARs even when the
        word SAR only appears in the heading.
        """
        header = " > ".join(part for part in (self.title, self.section) if part)
        return f"{header}\n{self.text}" if header else self.text

    def to_metadata(self) -> dict[str, str | int]:
        """Return metadata in the flat shape expected by the vector store."""
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "section": self.section,
        }


def title_from_filename(source: str) -> str:
    """Turn ``aml_kyc_policy.pdf`` into ``AML KYC Policy``."""
    acronyms = {"aml", "kyc", "ai", "ctr", "sar", "ach", "fincen", "ofac"}
    words = Path(source).stem.replace("-", "_").split("_")
    pretty = {"fincen": "FinCEN"}
    return " ".join(
        pretty.get(w, w.upper()) if w.lower() in acronyms else w.capitalize() for w in words if w
    )


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Build a deterministic id so re-ingesting a file upserts instead of duplicating."""
    digest = hashlib.sha256(f"{source}|{page}|{index}|{text}".encode()).hexdigest()[:16]
    return f"{source}:p{page}:c{index}:{digest}"


def split_sections(text: str, inherited: str = "") -> list[tuple[str, str]]:
    """Split page text at numbered section headings.

    Args:
        text: Page text.
        inherited: Section in effect at the top of the page (carried from the
            previous page), used for any text before the first heading.

    Returns:
        ``(section_heading, section_text)`` pairs in page order. Each section's
        text starts with its own heading line.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(inherited, text)] if text.strip() else []
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((inherited, preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = f"{match.group(1)}. {match.group(2).strip()}"
        sections.append((heading, text[match.start() : end].strip()))
    return sections


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split pages into overlapping chunks that respect clause boundaries.

    Pages are first split at numbered section headings, then each section is
    split recursively, so a chunk never mixes two sections and never spans
    pages (every chunk maps to exactly one citation). A section that starts on
    one page carries over to text at the top of the next page.

    Args:
        pages: Pages produced by the loader, in document order.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        Chunks in document order.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CLAUSE_SEPARATORS,
        keep_separator="end",
        strip_whitespace=True,
    )
    chunks: list[Chunk] = []
    carried_section: dict[str, str] = {}
    for page in pages:
        title = title_from_filename(page.source)
        sections = split_sections(page.text, carried_section.get(page.source, ""))
        index = 0
        for section, section_text in sections:
            for piece in splitter.split_text(section_text):
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(page.source, page.page, index, piece),
                        text=piece,
                        source=page.source,
                        page=page.page,
                        chunk_index=index,
                        title=title,
                        section=section,
                    )
                )
                index += 1
        if sections:
            carried_section[page.source] = sections[-1][0]
    return chunks
