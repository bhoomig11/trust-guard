import json
from pathlib import Path

from trustguard.etl.chunker import Chunk
from trustguard.governance.logger import AuditRecord, GovernanceLogger


def _record(query: str = "q") -> AuditRecord:
    return AuditRecord(
        original_query=query,
        masked_query=query,
        pii_entities=[],
        retrieved_chunks=[{"score": 0.8}, {"score": 0.5}],
        tool_calls=[],
        llm_prompt={},
        llm_response="a",
        latency_ms={"total_ms": 1.0},
        model="m",
        stop_reason="end_turn",
    )


def test_logger_writes_json_lines(tmp_path: Path) -> None:
    log = GovernanceLogger(tmp_path / "nested" / "audit.jsonl")
    assert log.tail() == []
    log.write(_record("first"))
    log.write(_record("second"))
    lines = (tmp_path / "nested" / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[1])
    assert parsed["original_query"] == "second"
    assert parsed["similarity_scores"] == [0.8, 0.5]
    assert parsed["request_id"] and parsed["timestamp"]
    assert [r["original_query"] for r in log.tail(1)] == ["second"]


def _chunks() -> list[Chunk]:
    return [
        Chunk("a1", "Wire transfers require dual authorization.", "a.txt", 1, 0, "A", "1. Wires"),
        Chunk("b1", "The late payment fee is thirty dollars.", "b.txt", 2, 0, "B", "3. Fees"),
    ]


def test_store_add_search_and_delete(store) -> None:
    assert store.search("anything") == []
    assert store.add(_chunks()) == 2
    hits = store.search("late payment fee", top_k=2)
    assert hits[0].source == "b.txt"
    assert hits[0].page == 2
    assert hits[0].score > hits[1].score
    assert hits[0].to_dict()["source"] == "b.txt"
    assert store.sources() == ["a.txt", "b.txt"]
    store.delete_source("a.txt")
    assert store.sources() == ["b.txt"]
    store.reset()
    assert store.count() == 0


def test_store_top_k_larger_than_collection(store) -> None:
    store.add(_chunks())
    assert len(store.search("fee", top_k=10)) == 2
