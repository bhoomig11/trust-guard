"""Shared fixtures: deterministic embedder, scripted Claude client, sample corpus."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from trustguard.privacy.shield import PrivacyShield
from trustguard.vectorstore import PolicyVectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO_ROOT / "data" / "policies"


class HashingEmbedder:
    """Bag-of-words hashing embedder: fast, deterministic, no model download."""

    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9$]+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def make_message(
    content: list[Any],
    stop_reason: str = "end_turn",
    model: str = "claude-haiku-5-5",
) -> Message:
    """Build a real SDK `Message` for scripted responses."""
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        content=content,
        usage=Usage(input_tokens=100, output_tokens=20),
    )


def text(value: str) -> TextBlock:
    return TextBlock(type="text", text=value)


def tool_use(name: str, tool_input: dict[str, Any], tool_id: str = "tu_1") -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=tool_id, name=name, input=tool_input)


class _Endpoint:
    def __init__(self, owner: ScriptedClient) -> None:
        self.owner = owner

    def create(self, **kwargs: Any) -> Message:
        # Snapshot messages: the agent mutates the list after the call.
        self.owner.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.owner.responses:
            raise AssertionError("ScriptedClient ran out of responses")
        response = self.owner.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _Beta:
    def __init__(self, owner: ScriptedClient) -> None:
        self.messages = _Endpoint(owner)


class ScriptedClient:
    """Stand-in for `anthropic.Anthropic` that replays queued responses."""

    def __init__(self, responses: list[Message | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = _Endpoint(self)
        self.beta = _Beta(self)


@pytest.fixture(scope="session")
def shield() -> PrivacyShield:
    return PrivacyShield(spacy_model="en_core_web_md")


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture()
def store(embedder: HashingEmbedder) -> Iterator[PolicyVectorStore]:
    store = PolicyVectorStore(embedder, persist_dir=None, collection_name="test")
    store.reset()
    yield store
    store.reset()


@pytest.fixture()
def policy_dir() -> Path:
    return POLICY_DIR


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "wire_policy.txt"
    path.write_text(
        "WIRE POLICY\n\n"
        "1. Limits\n"
        "1.1 Consumer wires are limited to $25,000 per business day.\n"
        "1.2 Wires of $50,000 or more require dual authorization.\n"
        "\f"
        "2. Fees\n"
        "2.1 Outgoing international wire fee is $45.\n",
        encoding="utf-8",
    )
    return path
