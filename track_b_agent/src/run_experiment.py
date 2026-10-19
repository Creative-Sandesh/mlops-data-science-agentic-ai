"""Run ONE prompt/config version end to end and log it as one MLflow run.

    queries -> agent traces -> harness metrics -> Evidently LLM regression suite -> MLflow

Usage:  uv run python src/run_experiment.py --version v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.config import settings  # noqa: E402
from regression import JUDGE_MODEL, run_regression  # noqa: E402
from run_queries import run_version  # noqa: E402

EXPERIMENT = "agent-prompt-versions"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
GOLDEN = ROOT / "eval" / "golden_responses.json"
REPORTS = ROOT / "reports"


def init_mlflow() -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    return exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)


def pick_representative_traces(traces: list[dict], per_query: list[dict], k: int = 3) -> list[tuple[str, dict]]:
    """One clean success (prefer one that used a tool) + up to two distinct-cause failures."""
    chosen: list[tuple[str, dict]] = []
    ok = [(t, s) for t, s in zip(traces, per_query) if s["task_success"]]
    ok.sort(key=lambda ts: (-len(ts[0]["tools_used"]), ts[0]["iterations"]))
    if ok:
        chosen.append(("success", ok[0][0]))
    seen_reasons: set[str] = set()
    bad = [(t, s) for t, s in zip(traces, per_query) if not s["task_success"]]
    bad.sort(key=lambda ts: (-ts[0]["iterations"], -ts[0]["n_parse_errors"]))
    for t, s in bad:
        reason = f"{s['failure_reason']}/{s['termination_reason']}"
        if reason not in seen_reasons and len(chosen) < k:
            seen_reasons.add(reason)
            chosen.append((f"failure_{s['failure_reason']}", t))
    for t, s in bad:  # fill up if failures were all of the same kind
        if len(chosen) >= k:
            break
        if all(t is not c[1] for c in chosen):
            chosen.append((f"failure_{s['failure_reason']}", t))
    return chosen


def main(version: str, extra_tags: dict[str, str] | None = None) -> dict:
    exp_id = init_mlflow()
    res = run_version(version)
    cfg, queries, traces, per_query = res["config"], res["queries"], res["traces"], res["per_query"]
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    out = REPORTS / "runs" / version
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{version}] running Evidently regression suite with judge {JUDGE_MODEL} ...", flush=True)
    reg = run_regression(version, queries, traces, golden, out)

    (out / "traces.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traces), encoding="utf-8")
    pd.DataFrame(per_query).to_csv(out / "harness_per_query.csv", index=False)
    rep = pick_representative_traces(traces, per_query)
    rep_dir = out / "representative_traces"
    rep_dir.mkdir(exist_ok=True)
    for label, t in rep:
        (rep_dir / f"{t['query_id']}_{label}.json").write_text(json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8")

    with mlflow.start_run(experiment_id=exp_id, run_name=version):
        mlflow.set_tags({"prompt_version": version, "note": cfg.get("note", ""),
                         "promotable": str(reg["gate_passed"]), "failed_cases": ",".join(reg["failed_cases"]),
                         **(extra_tags or {})})
        mlflow.log_params({
            "prompt_version": version, "prompt_file": cfg["prompt_file"], "model": cfg["model"],
            "temperature": cfg["temperature"], "top_k": cfg["top_k"], "chunk_size": cfg["chunk_size"],
            "max_iterations": cfg["max_iterations"], "lenient_json": cfg["lenient_json"],
            "tool_error_hints": cfg.get("tool_error_hints", False),
            "embedding_model": settings.EMBEDDING_MODEL_NAME, "n_kb_chunks": res["n_chunks"],
            "n_queries": len(queries), "judge_model": JUDGE_MODEL, "pass_threshold": reg["threshold"],
        })
        metrics = dict(res["metrics"])
        metrics.update({k: reg[k] for k in ("pct_tests_passed", "pct_correct", "pct_grounded")})
        metrics["gate_passed"] = float(reg["gate_passed"])
        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(ROOT / cfg["prompt_file"]), artifact_path="prompt")
        mlflow.log_artifact(str(out / "traces.jsonl"), artifact_path="traces")
        mlflow.log_artifacts(str(rep_dir), artifact_path="traces/representative")
        mlflow.log_artifact(str(out / "harness_per_query.csv"), artifact_path="harness")
        mlflow.log_artifact(str(reg["html_path"]), artifact_path="evidently")
        mlflow.log_artifact(str(out / f"judge_verdicts_{version}.csv"), artifact_path="evidently")

    print(f"\n[{version}] harness: " + ", ".join(f"{k}={v:.3f}" for k, v in res["metrics"].items()
                                                if k in ("task_success_rate", "tool_selection_accuracy", "avg_iterations", "parse_error_rate")))
    print(f"[{version}] regression: pct_tests_passed={reg['pct_tests_passed']:.2f} correct={reg['pct_correct']:.2f} "
          f"grounded={reg['pct_grounded']:.2f} gate={'PASS' if reg['gate_passed'] else 'FAIL'} failed={reg['failed_cases']}")
    return {"harness": res["metrics"], "regression": {k: v for k, v in reg.items() if k not in ("verdicts", "html_path")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    main(ap.parse_args().version)
