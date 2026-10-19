"""Evaluation harness (W16-style): deterministic scoring of agent traces against eval/test_queries.json.

Per query:
  task_success    answer contains every required fact (any-of alternatives), none of the forbidden strings;
                  for unanswerable queries: the agent refuses ("I don't know") instead of inventing an answer
  tool_ok         every expected tool was called at least once
  terminated_ok   loop ended with a final answer (not max_iterations / llm_error)
Aggregates are what get logged to MLflow.
"""
from __future__ import annotations

import re
from statistics import mean

REFUSAL_PATTERNS = ["i don't know", "i do not know", "i dont know", "cannot find", "not mentioned",
                    "no information", "not specified", "does not say", "doesn't say", "not provided",
                    "unable to find", "not available in", "do not have access", "don't have access",
                    "cannot answer", "can't answer", "does not contain", "do not contain",
                    "doesn't contain", "no mention"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("’", "'")).strip()


def is_refusal(answer: str) -> bool:
    a = _norm(answer)
    return any(p in a for p in REFUSAL_PATTERNS)


def score_trace(trace: dict, q: dict) -> dict:
    answer = _norm(trace.get("final_answer") or "")
    has_answer = bool(answer)
    if q["expect_refusal"]:
        success = has_answer and is_refusal(answer)
        failure_reason = None if success else ("no_answer" if not has_answer else "answered_instead_of_refusing")
    else:
        missing = [alts for alts in q["must_contain"] if not any(_norm(a).replace(",", "") in answer.replace(",", "") for a in alts)]
        forbidden = [s for s in q.get("must_not_contain", []) if _norm(s) in answer]
        success = has_answer and not missing and not forbidden and not is_refusal(answer)
        failure_reason = (None if success else "no_answer" if not has_answer
                          else "refused_answerable" if is_refusal(answer)
                          else "missing_fact" if missing else "forbidden_fact")
    used = set(trace["tools_used"])
    return {
        "id": q["id"], "category": q["category"], "task_success": success, "failure_reason": failure_reason,
        "tool_ok": set(q["expected_tools"]) <= used,
        "terminated_ok": trace["termination_reason"] == "final_answer",
        "iterations": trace["iterations"], "n_parse_errors": trace["n_parse_errors"],
        "tokens": trace["prompt_tokens"] + trace["completion_tokens"], "latency_s": trace["latency_s"],
        "termination_reason": trace["termination_reason"],
    }


def aggregate(per_query: list[dict], usd_per_1k_tokens: float = 0.0) -> dict[str, float]:
    n = len(per_query)
    tokens = sum(r["tokens"] for r in per_query)
    out = {
        "task_success_rate": sum(r["task_success"] for r in per_query) / n,
        "tool_selection_accuracy": sum(r["tool_ok"] for r in per_query) / n,
        "termination_rate": sum(r["terminated_ok"] for r in per_query) / n,
        "max_iterations_rate": sum(r["termination_reason"] == "max_iterations" for r in per_query) / n,
        "parse_error_rate": sum(r["n_parse_errors"] > 0 for r in per_query) / n,
        "avg_iterations": mean(r["iterations"] for r in per_query),
        "avg_tokens_per_query": tokens / n,
        "total_tokens": tokens,
        "avg_latency_s": mean(r["latency_s"] for r in per_query),
        "est_cost_usd": tokens / 1000 * usd_per_1k_tokens,
    }
    for cat in sorted({r["category"] for r in per_query}):
        rows = [r for r in per_query if r["category"] == cat]
        out[f"success_{cat}"] = sum(r["task_success"] for r in rows) / len(rows)
    return out
