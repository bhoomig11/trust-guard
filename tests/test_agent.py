import json
from pathlib import Path

import anthropic
import httpx
import pytest

from tests.conftest import ScriptedClient, make_message, text, tool_use
from trustguard.agent.agent import (
    SERVER_FALLBACK_BETA,
    AgentError,
    ComplianceAgent,
    extract_citations,
)
from trustguard.agent.tools import MATH_CALCULATOR, POLICY_RETRIEVER, ToolExecutor
from trustguard.etl.pipeline import ingest_files
from trustguard.governance.logger import GovernanceLogger
from trustguard.vectorstore import RetrievedChunk

REQUIRED_LOG_FIELDS = {
    "original_query",
    "masked_query",
    "retrieved_chunks",
    "similarity_scores",
    "llm_prompt",
    "llm_response",
    "latency_ms",
}


@pytest.fixture()
def audit(tmp_path: Path) -> GovernanceLogger:
    return GovernanceLogger(tmp_path / "logs" / "audit.jsonl")


def build_agent(client, store, shield, audit, **kwargs) -> ComplianceAgent:
    return ComplianceAgent(
        client=client,  # type: ignore[arg-type]
        executor=ToolExecutor(store, top_k=3, min_similarity=0.0),
        shield=shield,
        logger=audit,
        **kwargs,
    )


def test_retrieval_flow_masks_pii_and_cites(store, shield, audit, sample_txt) -> None:
    ingest_files([sample_txt], store)
    client = ScriptedClient(
        [
            make_message(
                [tool_use(POLICY_RETRIEVER, {"query": "dual authorization wire threshold"})],
                stop_reason="tool_use",
            ),
            make_message(
                [text("Wires of $50,000 or more need dual authorization [wire_policy.txt, p. 1].")]
            ),
        ]
    )
    agent = build_agent(client, store, shield, audit)
    query = "John Smith (SSN 123-45-6789) wants to wire $60,000. Is dual approval needed?"

    result = agent.answer(query)

    assert "dual authorization" in result.answer
    assert [c.source for c in result.citations] == ["wire_policy.txt"]
    assert result.citations[0].page == 1
    assert result.tools_used == [POLICY_RETRIEVER]
    assert {"SSN", "PERSON"} <= set(result.pii_detected)
    assert "123-45-6789" not in result.masked_query

    # Nothing sent to the LLM contains the raw PII.
    sent = json.dumps([c["messages"][0]["content"] for c in client.calls])
    assert "123-45-6789" not in sent
    assert "John Smith" not in sent
    assert "[SSN]" in sent

    # The second call carries the tool result back to the model.
    second_call = client.calls[1]["messages"]
    assert second_call[-1]["content"][0]["type"] == "tool_result"
    assert "wire_policy.txt" in second_call[-1]["content"][0]["content"]

    record = audit.tail(1)[0]
    assert set(record) >= REQUIRED_LOG_FIELDS
    assert record["original_query"] == query
    assert record["masked_query"] == result.masked_query
    assert record["similarity_scores"] == [c["score"] for c in record["retrieved_chunks"]]
    assert record["llm_response"] == result.answer
    assert record["latency_ms"]["total_ms"] >= record["latency_ms"]["llm_ms"]
    assert record["usage"]["input_tokens"] == 200
    assert record["llm_prompt"]["tools"] == [POLICY_RETRIEVER, MATH_CALCULATOR]


def test_request_shape(store, shield, audit) -> None:
    client = ScriptedClient([make_message([text("Hello.")])])
    build_agent(client, store, shield, audit, effort="low").answer("hi")
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["thinking"] == {"type": "adaptive"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "tool_choice" not in call
    assert "betas" not in call


def test_math_flow(store, shield, audit) -> None:
    client = ScriptedClient(
        [
            make_message(
                [tool_use(MATH_CALCULATOR, {"expression": "max(10, 0.05 * 1200)"})],
                stop_reason="tool_use",
            ),
            make_message([text("The fee is $60.")]),
        ]
    )
    result = build_agent(client, store, shield, audit).answer("5% of $1,200 or $10?")
    assert result.answer == "The fee is $60."
    assert result.tools_used == [MATH_CALCULATOR]
    assert result.citations == []
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["content"] == "60"
    assert tool_result["is_error"] is False


def test_answer_without_tools(store, shield, audit) -> None:
    client = ScriptedClient([make_message([text("4")])])
    result = build_agent(client, store, shield, audit).answer("What is 2+2?")
    assert result.answer == "4"
    assert result.tools_used == []
    assert len(client.calls) == 1


def test_step_limit_forces_final_answer(store, shield, audit) -> None:
    looping = make_message(
        [tool_use(MATH_CALCULATOR, {"expression": "1+1"})], stop_reason="tool_use"
    )
    client = ScriptedClient([looping, looping, make_message([text("Done.")])])
    result = build_agent(client, store, shield, audit, max_steps=2).answer("loop")
    assert result.answer == "Done."
    assert client.calls[2]["tool_choice"] == {"type": "none"}
    assert "tool_choice" not in client.calls[0]


def test_pause_turn_is_resumed(store, shield, audit) -> None:
    client = ScriptedClient(
        [make_message([text("...")], stop_reason="pause_turn"), make_message([text("Final.")])]
    )
    assert build_agent(client, store, shield, audit).answer("q").answer == "Final."


def test_refusal(store, shield, audit) -> None:
    client = ScriptedClient([make_message([], stop_reason="refusal")])
    result = build_agent(client, store, shield, audit).answer("q")
    assert "declined" in result.answer
    assert result.stop_reason == "refusal"


def test_fallback_models_use_beta_endpoint(store, shield, audit) -> None:
    client = ScriptedClient([make_message([text("ok")], model="claude-opus-5-5")])
    build_agent(client, store, shield, audit, model="claude-opus-5-5").answer("q")
    assert client.calls[0]["betas"] == [SERVER_FALLBACK_BETA]
    assert client.calls[0]["fallbacks"] == "default"


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("error", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (_status_error(anthropic.AuthenticationError, 401), 500),
        (_status_error(anthropic.RateLimitError, 429), 429),
        (_status_error(anthropic.BadRequestError, 400), 502),
        (_status_error(anthropic.InternalServerError, 500), 502),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            503,
        ),
    ],
)
def test_provider_errors_are_mapped_and_audited(store, shield, audit, error, status) -> None:
    agent = build_agent(ScriptedClient([error]), store, shield, audit)
    with pytest.raises(AgentError) as info:
        agent.answer("SSN 123-45-6789 question")
    assert info.value.status_code == status
    record = audit.tail(1)[0]
    assert record["error"]
    assert record["masked_query"] == "SSN [SSN] question"


def _chunk(source: str, page: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source}{page}", text="t", source=source, page=page, score=score
    )


def test_extract_citations_prefers_inline_references() -> None:
    retrieved = [_chunk("a.pdf", 1, 0.9), _chunk("b.txt", 2, 0.5), _chunk("a.pdf", 1, 0.7)]
    cites = extract_citations("Per policy [b.txt, p. 2].", retrieved)
    assert [(c.source, c.page) for c in cites] == [("b.txt", 2)]


def test_extract_citations_ignores_unretrieved_sources() -> None:
    retrieved = [_chunk("a.pdf", 1, 0.9), _chunk("b.txt", 2, 0.5)]
    cites = extract_citations("See [made_up.pdf, p. 9].", retrieved)
    assert [(c.source, c.page, c.score) for c in cites] == [("a.pdf", 1, 0.9), ("b.txt", 2, 0.5)]


def test_extract_citations_page_variants() -> None:
    retrieved = [_chunk("a.pdf", 3, 0.9)]
    assert extract_citations("[a.pdf, page 3]", retrieved)[0].page == 3
    assert extract_citations("[a.pdf, p3]", retrieved)[0].page == 3
    assert extract_citations("no refs", []) == []
