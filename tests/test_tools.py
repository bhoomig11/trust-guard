import pytest

from trustguard.agent.tools import (
    MATH_CALCULATOR,
    POLICY_RETRIEVER,
    TOOL_DEFINITIONS,
    CalculatorError,
    ToolExecutor,
    format_number,
    safe_eval,
)
from trustguard.etl.pipeline import ingest_files


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("25000 * 0.027 + 45", 720.0),
        ("$1,250 * 0.027", 33.75),
        ("max(10, 0.05 * 150)", 10),
        ("max(10, 0.05 * 1200)", 60.0),
        ("(1000 - 250) / 3", 250.0),
        ("2 ** 10", 1024),
        ("-5 + abs(-3)", -2),
        ("round(10 / 3, 2)", 3.33),
        ("floor(7.9) + ceil(1.1)", 9),
        ("17 // 5 + 17 % 5", 5),
    ],
)
def test_safe_eval(expression: str, expected: float) -> None:
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "a + 1",
        "(1).__class__",
        "2 ** 1000",
        "1 / 0",
        "",
        "1 +",
        "max(1, key=abs)",
        "'a' * 3",
        "1" * 400,
    ],
)
def test_safe_eval_rejects(expression: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expression)


def test_format_number() -> None:
    assert format_number(675.0) == "675"
    assert format_number(0.1 + 0.2) == "0.3"
    assert format_number(7) == "7"
    assert format_number(float("inf")) == "inf"


def test_tool_definitions_are_strict() -> None:
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert names == {POLICY_RETRIEVER, MATH_CALCULATOR}
    for tool in TOOL_DEFINITIONS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False


class TestExecutor:
    def test_retriever_returns_cited_passages(self, store, sample_txt) -> None:
        ingest_files([sample_txt], store)
        result = ToolExecutor(store, top_k=2, min_similarity=0.0).run(
            POLICY_RETRIEVER, {"query": "dual authorization for wires"}
        )
        assert not result.is_error
        assert 'source="wire_policy.txt"' in result.output
        assert result.chunks and result.chunks[0].source == "wire_policy.txt"

    def test_retriever_filters_low_similarity(self, store, sample_txt) -> None:
        ingest_files([sample_txt], store)
        result = ToolExecutor(store, min_similarity=0.99).run(
            POLICY_RETRIEVER, {"query": "unrelated quantum gardening"}
        )
        assert result.chunks == []
        assert "No relevant policy passages" in result.output

    def test_retriever_requires_query(self, store) -> None:
        assert ToolExecutor(store).run(POLICY_RETRIEVER, {"query": "  "}).is_error

    def test_calculator(self, store) -> None:
        result = ToolExecutor(store).run(MATH_CALCULATOR, {"expression": "1200 * 0.05"})
        assert result.output == "60"
        assert not result.is_error

    def test_calculator_error_is_returned_not_raised(self, store) -> None:
        result = ToolExecutor(store).run(MATH_CALCULATOR, {"expression": "import os"})
        assert result.is_error
        assert result.output.startswith("Error")

    def test_calculator_overflow(self, store) -> None:
        result = ToolExecutor(store).run(MATH_CALCULATOR, {"expression": "10.0 ** 99 ** 1"})
        assert not result.is_error or result.output.startswith("Error")

    def test_unknown_tool(self, store) -> None:
        assert ToolExecutor(store).run("Shell", {}).is_error
