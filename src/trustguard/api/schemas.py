"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body of `POST /query`."""

    query: str = Field(..., min_length=1, max_length=4000, description="The user's question.")


class CitationOut(BaseModel):
    """A supporting source."""

    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    """Answer returned by `POST /query`."""

    answer: str
    citations: list[CitationOut]
    masked_query: str
    pii_detected: list[str]
    tools_used: list[str]
    latency_ms: dict[str, float]
    model: str
    request_id: str


class DocumentSummary(BaseModel):
    """Per-document ingestion counts."""

    source: str
    pages: int
    chunks: int
    avg_chunk_chars: int


class IngestResponse(BaseModel):
    """Result of `POST /ingest`."""

    documents: int
    pages: int
    chunks: int
    total_chunks_indexed: int
    per_document: list[DocumentSummary]


class HealthResponse(BaseModel):
    """Result of `GET /health`."""

    status: str
    version: str
    indexed_chunks: int
    indexed_documents: list[str]
    llm_configured: bool
    model: str
