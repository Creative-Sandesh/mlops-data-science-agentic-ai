"""Re-run ONLY the Evidently LLM regression suite on the traces already saved for a version.

Useful to iterate on the judge (prompts, model) without paying for the agent runs again.
Usage:  uv run python src/rejudge.py --version v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from regression import run_regression  # noqa: E402


def main(version: str, out_dir: Path | None = None) -> dict:
    run_dir = ROOT / "reports" / "runs" / version
    traces = [json.loads(line) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = {q["id"]: q for q in json.loads((ROOT / "eval" / "test_queries.json").read_text(encoding="utf-8"))}
    golden = json.loads((ROOT / "eval" / "golden_responses.json").read_text(encoding="utf-8"))
    res = run_regression(version, [queries[t["query_id"]] for t in traces], traces, golden, out_dir or run_dir)
    print(res["verdicts"][["query_id", "Correctness", "Grounding", "case_pass"]].to_string(index=False))
    print({k: v for k, v in res.items() if k not in ("verdicts", "html_path")})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    main(ap.parse_args().version)
