"""Agentic RAG workflow: mask -> reason with tools -> cite -> audit."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from trustguard.agent.prompts import SYSTEM_PROMPT
from trustguard.agent.tools import (
    POLICY_RETRIEVER,
    TOOL_DEFINITIONS,
    ToolExecutor,
    ToolResult,
)
from trustguard.governance.logger import AuditRecord, GovernanceLogger
from trustguard.privacy.shield import PLACEHOLDERS, PrivacyShield
from trustguard.vectorstore import RetrievedChunk

# Models that accept server-side refusal fallbacks. Claude Haiku 5.5 (the
# default) does not, so it uses the standard Messages endpoint.
SERVER_FALLBACK_MODELS: frozenset[str] = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
)
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_CITATION_PATTERN = re.compile(r"\[([^\[\],]+?\.(?:pdf|txt)),\s*p(?:age|\.)?\s*(\d+)\]", re.I)


class AgentError(RuntimeError):
    """Raised when the LLM provider cannot serve the request."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        """Store an HTTP-style status code alongside the message."""
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Citation:
    """A source document and page that supports the answer."""

    source: str
    page: int
    score: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Serialize for API responses."""
        return {"source": self.source, "page": self.page, "score": round(self.score, 4)}


@dataclass
class AgentAnswer:
    """What the API returns for one query."""

    answer: str
    citations: list[Citation]
    masked_query: str
    pii_detected: list[str]
    tools_used: list[str]
    latency_ms: dict[str, float]
    request_id: str
    model: str
    stop_reason: str | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an SDK content block (or a plain dict) into JSON-safe data for the audit log."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return dict(block.model_dump(mode="json", exclude_none=True))
    return {"type": type(block).__name__, "repr": repr(block)}


def _serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = [_block_to_dict(b) for b in content]
        serialized.append({"role": message["role"], "content": content})
    return serialized


def extract_citations(
    answer: str,
    retrieved: list[RetrievedChunk],
    fallback: list[RetrievedChunk] | None = None,
) -> list[Citation]:
    """Build citations from the answer text, falling back to retrieved chunks.

    Inline references like ``[aml_kyc_policy.pdf, p. 2]`` are kept only when
    they match a chunk that was actually retrieved, so the model cannot cite a
    source it never saw.

    Args:
        answer: Final model answer.
        retrieved: Every chunk the model saw during the run.
        fallback: Chunks to cite when the answer has no valid inline reference.
            Defaults to `retrieved`. The agent passes only the chunks from
            explicit retriever calls, so an answer that ignored the pre-fetched
            context (for example pure arithmetic) gets no citations.

    Returns:
        Unique citations, best score first.
    """
    best: dict[tuple[str, int], float] = {}
    for chunk in retrieved:
        key = (chunk.source, chunk.page)
        best[key] = max(best.get(key, float("-inf")), chunk.score)

    cited = {
        (m.group(1).strip(), int(m.group(2)))
        for m in _CITATION_PATTERN.finditer(answer)
        if (m.group(1).strip(), int(m.group(2))) in best
    }
    if cited:
        keys = cited
    else:
        pool = retrieved if fallback is None else fallback
        keys = {(c.source, c.page) for c in pool}
    citations = [Citation(source=s, page=p, score=best[(s, p)]) for s, p in keys]
    return sorted(citations, key=lambda c: (-c.score, c.source, c.page))


class ComplianceAgent:
    """Runs the privacy-first, tool-using answer loop for one query at a time."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        executor: ToolExecutor,
        shield: PrivacyShield,
        logger: GovernanceLogger,
        model: str = "claude-haiku-5-5",
        effort: str = "low",
        thinking: str = "adaptive",
        max_tokens: int = 4096,
        max_steps: int = 4,
        prefetch: bool = True,
    ) -> None:
        """Wire the agent's dependencies.

        Args:
            client: Anthropic SDK client.
            executor: Tool implementations.
            shield: PII masking middleware.
            logger: Audit log sink.
            model: Claude model id.
            effort: ``output_config.effort`` level.
            thinking: ``"adaptive"`` or ``"disabled"`` (disabled lowers latency; it is
                only accepted at effort ``high`` or below on Claude Haiku 5.5).
            max_tokens: Per-response output cap.
            max_steps: Maximum model calls that may request tools before a final answer is forced.
            prefetch: Retrieve passages for the masked query before the first model call
                and include them in the prompt. Most policy questions can then be
                answered in one call instead of two; the model can still call
                either tool.
        """
        self.client = client
        self.executor = executor
        self.shield = shield
        self.logger = logger
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.thinking = thinking
        self.prefetch = prefetch
        self._system = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]

    def _create(self, messages: list[dict[str, Any]], allow_tools: bool) -> Any:
        """Send one Messages API request."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._system,
            "tools": TOOL_DEFINITIONS,
            "messages": messages,
            "output_config": {"effort": self.effort},
            "thinking": {"type": self.thinking},
        }
        if not allow_tools:
            params["tool_choice"] = {"type": "none"}
        if self.model in SERVER_FALLBACK_MODELS:
            return self.client.beta.messages.create(
                betas=[SERVER_FALLBACK_BETA], fallbacks="default", **params
            )
        return self.client.messages.create(**params)

    def _call_llm(self, messages: list[dict[str, Any]], allow_tools: bool) -> Any:
        """Call the model, translating SDK errors into `AgentError`."""
        try:
            return self._create(messages, allow_tools)
        except anthropic.AuthenticationError as exc:
            raise AgentError("LLM authentication failed; check ANTHROPIC_API_KEY", 500) from exc
        except anthropic.RateLimitError as exc:
            raise AgentError("LLM rate limit reached; retry shortly", 429) from exc
        except anthropic.BadRequestError as exc:
            raise AgentError(f"LLM rejected the request: {exc.message}", 502) from exc
        except anthropic.APIStatusError as exc:
            raise AgentError(f"LLM provider error ({exc.status_code})", 502) from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError("Could not reach the LLM provider", 503) from exc

    def answer(self, query: str) -> AgentAnswer:
        """Answer a user question end to end and write an audit record.

        Args:
            query: Raw user question, possibly containing PII.

        Returns:
            The grounded answer with citations and timing.

        Raises:
            AgentError: If the LLM provider fails. The failure is still audited.
        """
        start = time.perf_counter()
        masked = self.shield.mask(query)
        masking_ms = (time.perf_counter() - start) * 1000

        tool_results: list[ToolResult] = []
        llm_ms = 0.0
        tool_ms = 0.0
        user_content = masked.masked_text
        if self.prefetch:
            t_pre = time.perf_counter()
            prefetched = self.executor.run(POLICY_RETRIEVER, {"query": masked.masked_text})
            tool_ms += (time.perf_counter() - t_pre) * 1000
            prefetched.prefetch = True
            tool_results.append(prefetched)
            if prefetched.chunks:
                user_content = (
                    f"<retrieved_context>\n{prefetched.output}\n</retrieved_context>\n\n"
                    f"Question: {masked.masked_text}"
                )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        response: Any = None
        error: str | None = None
        answer_text = ""

        try:
            for step in range(self.max_steps + 1):
                allow_tools = step < self.max_steps
                t0 = time.perf_counter()
                response = self._call_llm(messages, allow_tools)
                llm_ms += (time.perf_counter() - t0) * 1000
                for key in usage:
                    usage[key] += int(getattr(response.usage, key, 0) or 0)

                if response.stop_reason == "pause_turn":
                    messages.append({"role": "assistant", "content": response.content})
                    continue
                if response.stop_reason != "tool_use":
                    break

                messages.append({"role": "assistant", "content": response.content})
                results_content: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    t1 = time.perf_counter()
                    result = self.executor.run(block.name, dict(block.input))
                    tool_ms += (time.perf_counter() - t1) * 1000
                    tool_results.append(result)
                    results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    )
                messages.append({"role": "user", "content": results_content})

            if response is not None and response.stop_reason == "refusal":
                answer_text = (
                    "This request was declined by the model's safety system. Please rephrase "
                    "the question or contact the compliance team."
                )
            elif response is not None:
                answer_text = "".join(
                    b.text for b in response.content if getattr(b, "type", None) == "text"
                ).strip()
        except AgentError as exc:
            error = str(exc)
            raise
        finally:
            retrieved = [c for r in tool_results for c in r.chunks]
            total_ms = (time.perf_counter() - start) * 1000
            latency = {
                "masking_ms": round(masking_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "tools_ms": round(tool_ms, 1),
                "total_ms": round(total_ms, 1),
            }
            record = AuditRecord(
                original_query=query,
                masked_query=masked.masked_text,
                pii_entities=[e.to_dict() for e in masked.entities],
                retrieved_chunks=[c.to_dict() for c in retrieved],
                tool_calls=[
                    {
                        "tool": r.tool,
                        "input": r.tool_input,
                        "is_error": r.is_error,
                        "prefetch": r.prefetch,
                    }
                    for r in tool_results
                ],
                llm_prompt={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "tools": [t["name"] for t in TOOL_DEFINITIONS],
                    "messages": _serialize_messages(messages),
                },
                llm_response=answer_text,
                latency_ms=latency,
                model=getattr(response, "model", self.model) if response else self.model,
                stop_reason=getattr(response, "stop_reason", None),
                usage=usage,
                error=error,
            )
            self.logger.write(record)

        return AgentAnswer(
            answer=answer_text,
            citations=extract_citations(
                answer_text,
                retrieved,
                fallback=[c for r in tool_results if not r.prefetch for c in r.chunks],
            ),
            masked_query=masked.masked_text,
            pii_detected=sorted({PLACEHOLDERS[e.entity_type].strip("[]") for e in masked.entities}),
            tools_used=[r.tool for r in tool_results if not r.prefetch],
            latency_ms=latency,
            request_id=record.request_id,
            model=record.model,
            stop_reason=record.stop_reason,
            retrieved=retrieved,
        )
