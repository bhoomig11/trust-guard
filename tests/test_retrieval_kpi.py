"""Accuracy KPI: >90% hit@k on the labeled question set with the real embedding model."""

import json
from pathlib import Path

import pytest

from trustguard.config import Settings
from trustguard.etl.pipeline import ingest_directory
from trustguard.vectorstore import PolicyVectorStore, SentenceTransformerEmbedder

EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "retrieval_eval.jsonl"


@pytest.mark.slow
def test_retrieval_hit_rate_meets_target(policy_dir: Path, shield) -> None:
    settings = Settings()
    store = PolicyVectorStore(
        SentenceTransformerEmbedder(settings.embedding_model), collection_name="kpi"
    )
    store.reset()
    ingest_directory(policy_dir, store)
    indexed = set(store.sources())

    items = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    # Public FinCEN documents are downloaded on demand; skip their questions if absent.
    items = [item for item in items if item["source"] in indexed]
    assert len(items) >= 40

    hits = 0
    for item in items:
        results = store.search(shield.mask(item["question"]).masked_text, settings.top_k)
        hits += any(
            r.source == item["source"] and item["expect"].lower() in r.text.lower() for r in results
        )
    assert hits / len(items) > 0.90
