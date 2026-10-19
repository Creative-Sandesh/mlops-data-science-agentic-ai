"""Run the eval queries through the agent for one version config and return traces + harness scores.

Standalone smoke test:  uv run python src/run_queries.py --version v1 --limit 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.llm import ChatLLM  # noqa: E402
from agent.loop import run_agent  # noqa: E402
from agent.tools import ToolBox  # noqa: E402
from harness import aggregate, score_trace  # noqa: E402

QUERIES = ROOT / "eval" / "test_queries.json"
VERSIONS = ROOT / "configs" / "versions.yaml"


def load_version(version: str) -> dict:
    cfg = yaml.safe_load(VERSIONS.read_text(encoding="utf-8"))[version]
    cfg["prompt_text"] = (ROOT / cfg["prompt_file"]).read_text(encoding="utf-8")
    return cfg


def run_version(version: str, limit: int | None = None, verbose: bool = True) -> dict:
    cfg = load_version(version)
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))[:limit]
    toolbox = ToolBox(top_k=cfg["top_k"], chunk_size=cfg["chunk_size"], error_hints=cfg.get("tool_error_hints", False))
    llm = ChatLLM(model=cfg["model"], temperature=cfg["temperature"])

    traces, scores = [], []
    for q in queries:
        trace = run_agent(q["question"], cfg["prompt_text"], toolbox, llm, cfg["max_iterations"],
                          cfg["lenient_json"], prompt_version=version)
        trace["query_id"] = q["id"]
        s = score_trace(trace, q)
        traces.append(trace)
        scores.append(s)
        if verbose:
            print(f"[{version}] {q['id']} success={s['task_success']!s:<5} tools={trace['tools_used']} "
                  f"iters={trace['iterations']} parse_err={trace['n_parse_errors']} "
                  f"end={trace['termination_reason']} {trace['latency_s']:.0f}s", flush=True)
    return {"version": version, "config": cfg, "queries": queries, "traces": traces,
            "per_query": scores, "metrics": aggregate(scores), "n_chunks": toolbox.n_chunks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    res = run_version(args.version, args.limit)
    print(json.dumps(res["metrics"], indent=2))
    print(json.dumps(res["traces"][0], indent=2)[:3000])
