"""Tools available to the agent: policy retrieval and safe arithmetic."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from trustguard.vectorstore import PolicyVectorStore, RetrievedChunk

POLICY_RETRIEVER = "Policy_Retriever"
MATH_CALCULATOR = "Math_Calculator"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": POLICY_RETRIEVER,
        "description": (
            "Search the indexed enterprise policy documents (transaction limits, fees, AML/KYC, "
            "disputes, data privacy, AI governance, and public regulatory guidance) and return "
            "the most relevant passages with their source document and page number. Use it for "
            "any question about what a policy says, including the figures needed for a "
            "calculation. Call it again with a reworded query if the first results are off-topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query describing the policy content needed.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": MATH_CALCULATOR,
        "description": (
            "Evaluate an arithmetic expression exactly, e.g. for fees, percentages, limits and "
            "remaining balances. Supports + - * / // % ** parentheses and the functions min, max, "
            "round, abs, floor, ceil. Use it instead of doing arithmetic yourself. Percentages "
            "must be written as decimals (2.7% -> 0.027)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. 'max(10, 0.05 * 1200)'.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class CalculatorError(ValueError):
    """Raised for expressions the calculator refuses to evaluate."""


_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "round": round,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
}
_MAX_EXPONENT = 100
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_MAX_EXPRESSION_LENGTH = 300


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression without `eval`.

    Only numeric literals, arithmetic operators and a small function whitelist
    are allowed, so model-generated input cannot execute code.

    Args:
        expression: Expression such as ``"25000 * 0.027 + 45"``. Thousands
            separators (``1,250``), ``$`` and ``_`` are stripped first; commas
            between function arguments are kept.

    Returns:
        The numeric result.

    Raises:
        CalculatorError: If the expression is invalid or uses anything outside the whitelist.
    """
    cleaned = _THOUSANDS.sub("", expression).replace("$", "").replace("_", "").strip()
    if not cleaned:
        raise CalculatorError("Empty expression")
    if len(cleaned) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression too long")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {expression!r}") from exc
    result = _eval_node(tree.body)
    if not isinstance(result, int | float):
        raise CalculatorError("Expression did not produce a number")
    return result


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise CalculatorError("Exponent too large")
        try:
            return _BINARY_OPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval_node(arg) for arg in node.args))
    raise CalculatorError(f"Unsupported syntax: {ast.dump(node)[:80]}")


def format_number(value: float | int) -> str:
    """Render a result without float noise (e.g. 675.0000000001 -> 675)."""
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


@dataclass
class ToolResult:
    """Outcome of one tool invocation, kept for governance logging."""

    tool: str
    tool_input: dict[str, Any]
    output: str
    is_error: bool = False
    chunks: list[RetrievedChunk] = field(default_factory=list)


class ToolExecutor:
    """Dispatches tool calls from the model to their implementations."""

    def __init__(
        self, store: PolicyVectorStore, top_k: int = 4, min_similarity: float = 0.25
    ) -> None:
        """Create an executor bound to a vector store.

        Args:
            store: Vector store queried by `Policy_Retriever`.
            top_k: Number of chunks to return per retrieval.
            min_similarity: Chunks below this cosine similarity are dropped.
        """
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def run(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        """Execute a tool by name. Errors are returned, not raised, so the model can recover."""
        if name == POLICY_RETRIEVER:
            return self._retrieve(tool_input)
        if name == MATH_CALCULATOR:
            return self._calculate(tool_input)
        return ToolResult(name, tool_input, f"Unknown tool '{name}'", is_error=True)

    def _retrieve(self, tool_input: dict[str, Any]) -> ToolResult:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return ToolResult(POLICY_RETRIEVER, tool_input, "Missing 'query'", is_error=True)
        hits = [c for c in self.store.search(query, self.top_k) if c.score >= self.min_similarity]
        if not hits:
            return ToolResult(
                POLICY_RETRIEVER,
                tool_input,
                "No relevant policy passages were found for this query.",
            )
        blocks = [
            f'<passage id="{i}" source="{c.source}" page="{c.page}" score="{c.score:.3f}">\n'
            f"{c.text}\n</passage>"
            for i, c in enumerate(hits, start=1)
        ]
        return ToolResult(POLICY_RETRIEVER, tool_input, "\n".join(blocks), chunks=hits)

    def _calculate(self, tool_input: dict[str, Any]) -> ToolResult:
        expression = str(tool_input.get("expression", ""))
        try:
            value = safe_eval(expression)
        except CalculatorError as exc:
            return ToolResult(MATH_CALCULATOR, tool_input, f"Error: {exc}", is_error=True)
        except (OverflowError, TypeError, ValueError) as exc:
            return ToolResult(MATH_CALCULATOR, tool_input, f"Error: {exc}", is_error=True)
        return ToolResult(MATH_CALCULATOR, tool_input, format_number(value))
