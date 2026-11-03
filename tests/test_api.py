from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ScriptedClient, make_message, text, tool_use
from trustguard.agent.agent import ComplianceAgent
from trustguard.agent.tools import POLICY_RETRIEVER, ToolExecutor
from trustguard.api.main import Services, build_services, create_app
from trustguard.config import Settings
from trustguard.governance.logger import GovernanceLogger
from trustguard.vectorstore import PolicyVectorStore


def make_services(tmp_path, embedder, shield, client=None, data_dir=None, auto_ingest=True):
    settings = Settings(
        anthropic_api_key="test-key" if client else None,
        data_dir=data_dir or tmp_path / "empty",
        chroma_dir=tmp_path / "chroma",
        log_path=tmp_path / "audit.jsonl",
        auto_ingest=auto_ingest,
        max_upload_mb=1,
    )
    store = PolicyVectorStore(embedder, persist_dir=settings.chroma_dir, collection_name="api")
    audit = GovernanceLogger(settings.log_path)
    agent = None
    if client is not None:
        agent = ComplianceAgent(
            client=client,  # type: ignore[arg-type]
            executor=ToolExecutor(store, min_similarity=0.0),
            shield=shield,
            logger=audit,
        )
    return Services(settings=settings, store=store, shield=shield, audit=audit, agent=agent)


@pytest.fixture()
def corpus_dir(tmp_path: Path, sample_txt: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / sample_txt.name).write_text(sample_txt.read_text(), encoding="utf-8")
    return directory


def test_health_and_auto_ingest(tmp_path, embedder, shield, corpus_dir) -> None:
    services = make_services(tmp_path, embedder, shield, data_dir=corpus_dir)
    with TestClient(create_app(services)) as api:
        body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0
    assert body["indexed_documents"] == ["wire_policy.txt"]
    assert body["llm_configured"] is False


def test_ingest_directory_endpoint(tmp_path, embedder, shield, corpus_dir) -> None:
    services = make_services(tmp_path, embedder, shield, data_dir=corpus_dir, auto_ingest=False)
    with TestClient(create_app(services)) as api:
        assert api.get("/health").json()["indexed_chunks"] == 0
        body = api.post("/ingest").json()
    assert body["documents"] == 1
    assert body["total_chunks_indexed"] == body["chunks"] > 0
    assert body["per_document"][0]["source"] == "wire_policy.txt"


def test_ingest_missing_directory(tmp_path, embedder, shield) -> None:
    with TestClient(create_app(make_services(tmp_path, embedder, shield))) as api:
        assert api.post("/ingest").status_code == 404


def test_ingest_upload(tmp_path, embedder, shield, sample_txt) -> None:
    with TestClient(create_app(make_services(tmp_path, embedder, shield))) as api:
        response = api.post(
            "/ingest", files=[("files", ("wire_policy.txt", sample_txt.read_bytes(), "text/plain"))]
        )
        assert response.status_code == 200
        assert api.get("/health").json()["indexed_documents"] == ["wire_policy.txt"]


def test_ingest_rejects_bad_uploads(tmp_path, embedder, shield) -> None:
    with TestClient(create_app(make_services(tmp_path, embedder, shield))) as api:
        bad_type = api.post("/ingest", files=[("files", ("a.docx", b"x", "application/msword"))])
        too_big = api.post(
            "/ingest", files=[("files", ("a.txt", b"x" * (2 * 1024 * 1024), "text/plain"))]
        )
        corrupt = api.post("/ingest", files=[("files", ("a.pdf", b"not a pdf", "application/pdf"))])
    assert bad_type.status_code == 415
    assert too_big.status_code == 413
    assert corrupt.status_code == 422


def test_query_without_llm(tmp_path, embedder, shield) -> None:
    with TestClient(create_app(make_services(tmp_path, embedder, shield))) as api:
        response = api.post("/query", json={"query": "What is the wire fee?"})
    assert response.status_code == 503


def test_query_validation(tmp_path, embedder, shield) -> None:
    client = ScriptedClient([])
    with TestClient(create_app(make_services(tmp_path, embedder, shield, client))) as api:
        assert api.post("/query", json={"query": ""}).status_code == 422


def test_query_end_to_end(tmp_path, embedder, shield, corpus_dir) -> None:
    client = ScriptedClient(
        [
            make_message(
                [tool_use(POLICY_RETRIEVER, {"query": "international wire fee"})],
                stop_reason="tool_use",
            ),
            make_message([text("The international wire fee is $45 [wire_policy.txt, p. 2].")]),
        ]
    )
    services = make_services(tmp_path, embedder, shield, client, data_dir=corpus_dir)
    with TestClient(create_app(services)) as api:
        response = api.post(
            "/query", json={"query": "Account 4401928374: what is the international wire fee?"}
        )
    assert response.status_code == 200
    body = response.json()
    assert "$45" in body["answer"]
    assert body["citations"] == [
        {"source": "wire_policy.txt", "page": 2, "score": body["citations"][0]["score"]}
    ]
    assert body["masked_query"].startswith("[ACCOUNT_NUMBER]")
    assert body["pii_detected"] == ["ACCOUNT_NUMBER"]
    assert body["tools_used"] == [POLICY_RETRIEVER]
    assert set(body["latency_ms"]) == {"masking_ms", "llm_ms", "tools_ms", "total_ms"}


def test_query_provider_error(tmp_path, embedder, shield) -> None:
    import anthropic
    import httpx

    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x.test"))
    services = make_services(tmp_path, embedder, shield, ScriptedClient([error]))
    with TestClient(create_app(services)) as api:
        assert api.post("/query", json={"query": "hi"}).status_code == 503


def test_build_services(tmp_path, monkeypatch) -> None:
    settings = Settings(
        anthropic_api_key="sk-test",
        chroma_dir=tmp_path / "chroma",
        log_path=tmp_path / "audit.jsonl",
    )
    services = build_services(settings)
    assert services.agent is not None
    assert services.agent.model == settings.llm_model
    assert build_services(settings.model_copy(update={"anthropic_api_key": None})).agent is None
