"""End-to-end ingestion: load -> chunk -> embed -> store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trustguard.etl.chunker import Chunk, chunk_pages
from trustguard.etl.loader import discover_documents, load_document
from trustguard.vectorstore import PolicyVectorStore


@dataclass(frozen=True)
class IngestReport:
    """Summary of an ingestion run."""

    documents: int
    pages: int
    chunks: int
    per_document: list[dict[str, str | int]]


def ingest_files(
    paths: list[Path],
    store: PolicyVectorStore,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> IngestReport:
    """Ingest documents into the vector store.

    Previously indexed chunks of the same document are replaced, so
    re-ingesting an edited file never leaves stale clauses behind.

    Args:
        paths: PDF or TXT files to ingest.
        store: Destination vector store.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Counts per document and in total.
    """
    all_chunks: list[Chunk] = []
    page_count = 0
    for path in paths:
        pages = load_document(path)
        page_count += len(pages)
        store.delete_source(path.name)
        all_chunks.extend(chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    store.add(all_chunks)

    frame = pd.DataFrame(
        [{"source": c.source, "page": c.page, "chars": len(c.text)} for c in all_chunks],
        columns=["source", "page", "chars"],
    )
    per_document: list[dict[str, str | int]] = []
    if not frame.empty:
        summary = (
            frame.groupby("source")
            .agg(pages=("page", "nunique"), chunks=("page", "size"), avg_chars=("chars", "mean"))
            .reset_index()
        )
        per_document = [
            {
                "source": str(row.source),
                "pages": int(row.pages),
                "chunks": int(row.chunks),
                "avg_chunk_chars": int(row.avg_chars),
            }
            for row in summary.itertuples(index=False)
        ]
    return IngestReport(
        documents=len(paths),
        pages=page_count,
        chunks=len(all_chunks),
        per_document=per_document,
    )


def ingest_directory(directory: Path, store: PolicyVectorStore, **kwargs: int) -> IngestReport:
    """Ingest every supported file in a directory."""
    return ingest_files(discover_documents(directory), store, **kwargs)
