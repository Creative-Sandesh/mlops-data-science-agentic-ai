"""The W16 agentic loop, instrumented to emit a full structured trace for every query.

Protocol: on each turn the model must reply with ONE JSON object
    {"thought": "...", "action": "<tool name>" | "final_answer", "args": {...}}
Tool results are fed back as "Observation: ..." user messages. The loop ends on `final_answer`,
or when `max_iterations` model calls have been used.

Trace schema (one dict per query, JSON-serialisable):
    question, prompt_version, final_answer, iterations, termination_reason
        (final_answer | max_iterations | llm_error), n_parse_errors, tools_used,
    prompt_tokens, completion_tokens, latency_s,
    steps: [{step, raw_output, thought, action, args, result, parse_error, latency_s, tokens}]
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from agent.llm import ChatLLM
from agent.tools import ToolBox

PARSE_ERROR_FEEDBACK = (
    "Your last reply was not a single valid JSON object. Reply again with ONLY a JSON object of the form "
    '{"thought": "...", "action": "<tool name or final_answer>", "args": {...}}.'
)


def parse_action(raw: str, lenient: bool) -> tuple[dict | None, str | None]:
    """Parse the model's reply. Strict mode = the whole reply must be valid JSON.

    Lenient mode (added for prompt v2 after v1 traces showed 14/14 replies wrapped in ```json fences with the
    model role-playing several steps in one reply): strips ``` fences, decodes ONLY the first JSON object and
    ignores anything after it (so invented follow-up steps/observations are discarded), and repairs missing
    closing braces.
    """
    text = raw.strip()
    try:
        if lenient:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            start = text.find("{")
            if start == -1:
                return None, "invalid JSON: no JSON object found"
            text = text[start:]
            try:
                obj, _ = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                depth = text.count("{") - text.count("}")
                text = text.split("```")[0].rstrip() + "}" * max(depth, 0)
                obj, _ = json.JSONDecoder().raw_decode(text)
        else:
            obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg} (char {e.pos})"
    if not isinstance(obj, dict) or "action" not in obj:
        return None, "JSON object has no 'action' field"
    if not isinstance(obj.get("args", {}), dict):
        return None, "'args' must be a JSON object, e.g. {\"query\": \"...\"}"
    return obj, None


def run_agent(question: str, system_prompt: str, toolbox: ToolBox, llm: ChatLLM,
              max_iterations: int, lenient_json: bool = False, prompt_version: str = "") -> dict[str, Any]:
    system = system_prompt.replace("{tools}", toolbox.describe()).replace("{max_iterations}", str(max_iterations))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    trace: dict[str, Any] = {
        "question": question, "prompt_version": prompt_version, "final_answer": None, "iterations": 0,
        "termination_reason": "max_iterations", "n_parse_errors": 0, "tools_used": [],
        "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "steps": [],
    }
    t0 = time.perf_counter()

    for i in range(1, max_iterations + 1):
        step: dict[str, Any] = {"step": i, "raw_output": None, "thought": None, "action": None, "args": None,
                                "result": None, "parse_error": None, "latency_s": 0.0, "tokens": 0}
        trace["iterations"] = i
        try:
            reply = llm.chat(messages)
        except Exception as e:  # noqa: BLE001
            step["parse_error"] = f"llm_error: {type(e).__name__}: {e}"
            trace["steps"].append(step)
            trace["termination_reason"] = "llm_error"
            break
        step.update(raw_output=reply.content, latency_s=round(reply.latency_s, 3),
                    tokens=reply.prompt_tokens + reply.completion_tokens)
        trace["prompt_tokens"] += reply.prompt_tokens
        trace["completion_tokens"] += reply.completion_tokens

        obj, err = parse_action(reply.content, lenient_json)
        messages.append({"role": "assistant", "content": reply.content})
        if err:
            step["parse_error"] = err
            trace["n_parse_errors"] += 1
            messages.append({"role": "user", "content": PARSE_ERROR_FEEDBACK})
            trace["steps"].append(step)
            continue

        step.update(thought=obj.get("thought"), action=obj["action"], args=obj.get("args", {}))
        if obj["action"] == "final_answer":
            answer = obj.get("args", {}).get("answer", obj.get("answer"))
            step["result"] = "(loop terminated with final answer)"
            trace["steps"].append(step)
            trace["final_answer"] = str(answer) if answer is not None else None
            trace["termination_reason"] = "final_answer" if answer is not None else "final_answer_missing"
            break

        observation = toolbox.call(obj["action"], step["args"])
        step["result"] = observation
        trace["tools_used"].append(obj["action"])
        trace["steps"].append(step)
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    trace["latency_s"] = round(time.perf_counter() - t0, 3)
    return trace
