"""Tools available to the agent: knowledge-base search (W15 RAG), weather (W15 mock), calculator (new)."""
from __future__ import annotations

import ast
import operator
from pathlib import Path

from app.rag.ingestion import chunk_text
from app.rag.vector_store import RAGVectorStore
from app.tools import get_current_weather

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def safe_eval(expression: str) -> float:
    """Evaluate a plain arithmetic expression without using eval()."""
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return _ev(ast.parse(expression.strip().replace(",", ""), mode="eval").body)


TOOL_DESCRIPTIONS = {
    "search_docs": 'search_docs(query: str) - search the Nimbus Cloud knowledge base; returns the most relevant text passages.',
    "get_current_weather": 'get_current_weather(location: str) - current weather for a city.',
    "calculator": 'calculator(expression: str) - evaluate an arithmetic expression such as "29 * 12 * 0.8"; returns the number.',
}


class ToolBox:
    """Owns the vector store (built once from data/kb) and dispatches tool calls."""

    def __init__(self, top_k: int, chunk_size: int, chunk_overlap: int = 50, store: RAGVectorStore | None = None,
                 error_hints: bool = False):
        self.top_k = top_k
        self.error_hints = error_hints  # v5+: tell the model HOW to fix a failed tool call
        self.store = store or RAGVectorStore()
        self.store._init_collection()  # fresh collection: chunk_size changes between experiments
        chunks: list[str] = []
        for path in sorted(KB_DIR.glob("*.md")):
            chunks += chunk_text(path.read_text(encoding="utf-8"), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.store.add_documents(chunks)
        self.n_chunks = len(chunks)

    @staticmethod
    def describe() -> str:
        return "\n".join(f"- {d}" for d in TOOL_DESCRIPTIONS.values())

    def call(self, name: str, args: dict) -> str:
        """Run a tool and return the raw observation string (errors are returned, not raised)."""
        try:
            if name == "search_docs":
                passages = self.store.retrieve(str(args["query"]), top_k=self.top_k)
                return "\n---\n".join(passages) if passages else "No results."
            if name == "get_current_weather":
                return get_current_weather(str(args["location"]))
            if name == "calculator":
                result = safe_eval(str(args["expression"]))
                return str(round(result, 6))
            return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(TOOL_DESCRIPTIONS)}"
        except KeyError as e:
            return f"ERROR: missing argument {e} for tool '{name}'"
        except Exception as e:  # noqa: BLE001 - surfaced to the model as an observation
            msg = f"ERROR: {type(e).__name__}: {e}"
            if self.error_hints and name == "calculator":
                msg += (". The calculator accepts ONLY digits, + - * / ( ) and a decimal point - no words, units, "
                        "percent signs or comments. Rewrite the expression (e.g. 20% off = * 0.8) and call calculator again.")
            return msg
