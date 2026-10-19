"""Unit tests for the agent loop, parser, calculator and harness - no LLM or vector store needed."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.llm import LLMReply  # noqa: E402
from agent.loop import parse_action, run_agent  # noqa: E402
from agent.tools import safe_eval  # noqa: E402
from harness import aggregate, is_refusal, score_trace  # noqa: E402


class ScriptedLLM:
    """Returns pre-baked replies in order, like a deterministic model."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMReply(content=reply, prompt_tokens=10, completion_tokens=5, latency_s=0.01)


class FakeToolBox:
    def describe(self):
        return "- search_docs(query)"

    def call(self, name, args):
        return f"result-of-{name}"


FENCED = '```json\n{"thought": "t", "action": "final_answer", "args": {"answer": "42"}}\n```'


def test_safe_eval_arithmetic_and_rejects_code():
    assert safe_eval("29 * 12 * 0.8") == pytest.approx(278.4)
    assert safe_eval("10 * 12 * 9.6") == pytest.approx(1152)
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_strict_parser_rejects_markdown_fences_but_lenient_accepts():
    assert parse_action(FENCED, lenient=False)[0] is None
    obj, err = parse_action(FENCED, lenient=True)
    assert err is None and obj["action"] == "final_answer"


def test_lenient_parser_keeps_only_first_object_and_ignores_invented_steps():
    first = '{"thought": "t", "action": "search_docs", "args": {"query": "x"}}'
    invented = '{"thought": "made up", "action": "final_answer", "args": {"answer": "$15"}}'
    fence = "```"
    raw = f"{fence}json\n{first}\n{fence}\n\n{fence}json\n{invented}\n{fence}"
    obj, err = parse_action(raw, lenient=True)
    assert err is None and obj["action"] == "search_docs" and obj["args"] == {"query": "x"}


def test_string_args_are_rejected_even_in_lenient_mode():
    obj, err = parse_action('{"thought": "t", "action": "search_docs", "args": "pricing pro plan"}', lenient=True)
    assert obj is None and "args" in err


def test_lenient_parser_repairs_missing_closing_brace():
    obj, err = parse_action('{"thought": "x", "action": "search_docs", "args": {"query": "pro price"}', lenient=True)
    assert err is None and obj["args"] == {"query": "pro price"}


def test_loop_tool_call_then_final_answer():
    llm = ScriptedLLM([
        json.dumps({"thought": "look up", "action": "search_docs", "args": {"query": "pro"}}),
        json.dumps({"thought": "done", "action": "final_answer", "args": {"answer": "$29"}}),
    ])
    trace = run_agent("price?", "sys {tools}", FakeToolBox(), llm, max_iterations=4)
    assert trace["termination_reason"] == "final_answer" and trace["final_answer"] == "$29"
    assert trace["iterations"] == 2 and trace["tools_used"] == ["search_docs"]
    assert trace["steps"][0]["result"] == "result-of-search_docs"
    assert trace["prompt_tokens"] == 20


def test_loop_stops_at_max_iterations_and_counts_parse_errors():
    trace = run_agent("q", "sys", FakeToolBox(), ScriptedLLM([FENCED]), max_iterations=3, lenient_json=False)
    assert trace["termination_reason"] == "max_iterations" and trace["iterations"] == 3
    assert trace["n_parse_errors"] == 3 and trace["final_answer"] is None


def test_harness_scoring():
    q = {"id": "q", "category": "single_hop", "expected_tools": ["search_docs"],
         "must_contain": [["$29"]], "expect_refusal": False}
    good = {"final_answer": "It costs $29.", "tools_used": ["search_docs"], "termination_reason": "final_answer",
            "iterations": 2, "n_parse_errors": 0, "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 1.0}
    assert score_trace(good, q)["task_success"] is True
    bad = dict(good, final_answer=None, termination_reason="max_iterations")
    s = score_trace(bad, q)
    assert s["task_success"] is False and s["failure_reason"] == "no_answer"
    refuse_q = dict(q, must_contain=[], expect_refusal=True)
    assert score_trace(dict(good, final_answer="I don't know."), refuse_q)["task_success"] is True
    assert score_trace(good, refuse_q)["failure_reason"] == "answered_instead_of_refusing"
    assert is_refusal("I don’t know")  # curly apostrophes are normalised
    assert is_refusal("I do not have access to the required information.")
    agg = aggregate([score_trace(good, q), s])
    assert agg["task_success_rate"] == 0.5 and agg["max_iterations_rate"] == 0.5
