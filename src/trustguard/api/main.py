"""FastAPI application exposing `/health`, `/ingest` and `/query`."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from trustguard import __version__
from trustguard.agent.agent import AgentError, ComplianceAgent
from trustguard.agent.tools import ToolExecutor
from trustguard.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from trustguard.config import Settings, get_settings
from trustguard.etl.loader import SUPPORTED_SUFFIXES
from trustguard.etl.pipeline import IngestReport, ingest_directory, ingest_files
from trustguard.governance.logger import GovernanceLogger
from trustguard.privacy.shield import PrivacyShield, get_shield
from trustguard.vectorstore import PolicyVectorStore, SentenceTransformerEmbedder

logger = logging.getLogger("trustguard")


@dataclass
class Services:
    """Long-lived objects shared by all requests."""

    settings: Settings
    store: PolicyVectorStore
    shield: PrivacyShield
    audit: GovernanceLogger
    agent: ComplianceAgent | None


def build_services(settings: Settings) -> Services:
    """Construct the vector store, privacy shield, logger and agent from settings."""
    store = PolicyVectorStore(
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    shield = get_shield(settings.spacy_model)
    audit = GovernanceLogger(settings.log_path)
    agent: ComplianceAgent | None = None
    if settings.anthropic_api_key:
        agent = ComplianceAgent(
            client=anthropic.Anthropic(api_key=settings.anthropic_api_key),
            executor=ToolExecutor(store, settings.top_k, settings.min_similarity),
            shield=shield,
            logger=audit,
            model=settings.llm_model,
            effort=settings.llm_effort,
            thinking=settings.llm_thinking,
            max_tokens=settings.llm_max_tokens,
            max_steps=settings.agent_max_steps,
            prefetch=settings.prefetch_context,
        )
    else:
        logger.warning("ANTHROPIC_API_KEY is not set; /query will return 503")
    return Services(settings=settings, store=store, shield=shield, audit=audit, agent=agent)


def _ingest_report(report: IngestReport, store: PolicyVectorStore) -> IngestResponse:
    return IngestResponse(
        documents=report.documents,
        pages=report.pages,
        chunks=report.chunks,
        total_chunks_indexed=store.count(),
        per_document=report.per_document,  # type: ignore[arg-type]
    )


def create_app(services: Services | None = None) -> FastAPI:
    """Create the application.

    Args:
        services: Pre-built services (used by tests). When omitted they are
            built from environment settings at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services or build_services(get_settings())
        app.state.services = svc
        # Load the embedding and NER models now so the first request is not slow.
        await run_in_threadpool(svc.store.embedder.embed, ["warmup"])
        await run_in_threadpool(svc.shield.mask, "warmup")
        data_dir = svc.settings.data_dir
        if svc.settings.auto_ingest and svc.store.count() == 0 and data_dir.is_dir():
            report = await run_in_threadpool(
                ingest_directory,
                data_dir,
                svc.store,
                chunk_size=svc.settings.chunk_size,
                chunk_overlap=svc.settings.chunk_overlap,
            )
            logger.info("Auto-ingested %d chunks from %s", report.chunks, data_dir)
        yield

    app = FastAPI(
        title="TrustGuard AI",
        version=__version__,
        description="Secure retrieval and compliance agent with PII masking and audit logging.",
        lifespan=lifespan,
    )

    def get_services(request: Request) -> Services:
        return request.app.state.services  # type: ignore[no-any-return]

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report service status and index size."""
        svc = get_services(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            indexed_chunks=svc.store.count(),
            indexed_documents=svc.store.sources(),
            llm_configured=svc.agent is not None,
            model=svc.settings.llm_model,
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        request: Request,
        files: Annotated[list[UploadFile] | None, File()] = None,
    ) -> IngestResponse:
        """Ingest uploaded PDF/TXT files, or re-ingest the configured data directory."""
        svc = get_services(request)
        chunk_args = {
            "chunk_size": svc.settings.chunk_size,
            "chunk_overlap": svc.settings.chunk_overlap,
        }
        if not files:
            if not svc.settings.data_dir.is_dir():
                raise HTTPException(404, f"Data directory {svc.settings.data_dir} not found")
            report = await run_in_threadpool(
                ingest_directory, svc.settings.data_dir, svc.store, **chunk_args
            )
            return _ingest_report(report, svc.store)

        max_bytes = svc.settings.max_upload_mb * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for upload in files:
                name = Path(upload.filename or "").name
                if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                    raise HTTPException(415, f"Unsupported file '{name}'. Use PDF or TXT.")
                content = await upload.read()
                if len(content) > max_bytes:
                    raise HTTPException(413, f"'{name}' exceeds {svc.settings.max_upload_mb} MB")
                path = Path(tmp) / name
                path.write_bytes(content)
                paths.append(path)
            try:
                report = await run_in_threadpool(ingest_files, paths, svc.store, **chunk_args)
            except Exception as exc:  # malformed PDFs raise a variety of parser errors
                raise HTTPException(422, f"Could not parse document: {exc}") from exc
        return _ingest_report(report, svc.store)

    @app.post("/query", response_model=QueryResponse)
    async def query(request: Request, body: QueryRequest) -> QueryResponse:
        """Answer a question with PII masking, tool use, citations and audit logging."""
        svc = get_services(request)
        if svc.agent is None:
            raise HTTPException(503, "LLM is not configured. Set ANTHROPIC_API_KEY.")
        try:
            result = await run_in_threadpool(svc.agent.answer, body.query)
        except AgentError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        return QueryResponse(
            answer=result.answer,
            citations=[CitationOut(**c.to_dict()) for c in result.citations],  # type: ignore[arg-type]
            masked_query=result.masked_query,
            pii_detected=result.pii_detected,
            tools_used=result.tools_used,
            latency_ms=result.latency_ms,
            model=result.model,
            request_id=result.request_id,
        )

    return app


app = create_app()
