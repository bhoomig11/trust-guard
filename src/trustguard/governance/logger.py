"""Structured JSON audit logging for every query transaction."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditRecord:
    """One end-to-end query transaction (FR-4.1, FR-4.2)."""

    original_query: str
    masked_query: str
    pii_entities: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    llm_prompt: dict[str, Any]
    llm_response: str
    latency_ms: dict[str, float]
    model: str
    stop_reason: str | None
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def similarity_scores(self) -> list[float]:
        """Scores of the retrieved chunks, in retrieval order."""
        return [float(c["score"]) for c in self.retrieved_chunks]

    def to_dict(self) -> dict[str, Any]:
        """Serialize, including the derived similarity scores."""
        data = asdict(self)
        data["similarity_scores"] = self.similarity_scores
        return data


class GovernanceLogger:
    """Appends audit records as JSON Lines to a local file."""

    def __init__(self, path: Path) -> None:
        """Create the log directory if needed.

        Args:
            path: Destination `.jsonl` file.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> None:
        """Append one record. Writes are serialized across threads."""
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent records, newest last."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
