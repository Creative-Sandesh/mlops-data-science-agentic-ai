"""Compare all prompt/config versions in MLflow, export the table, and promote the best passing version.

Promotion rule: a version is *promotable* only if the Evidently regression gate passed
(pct_tests_passed >= 80%). Among promotable versions the winner has the highest pct_tests_passed, then
the highest harness task_success_rate, then the fewest tokens per query (cost).

Usage:  uv run python src/compare_versions.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_experiment import EXPERIMENT, REPORTS, init_mlflow  # noqa: E402

PARAMS = ["prompt_version", "temperature", "top_k", "chunk_size", "max_iterations", "lenient_json", "tool_error_hints"]
METRICS = ["pct_tests_passed", "pct_correct", "pct_grounded", "task_success_rate", "tool_selection_accuracy",
           "termination_rate", "parse_error_rate", "avg_iterations", "avg_tokens_per_query", "avg_latency_s"]


def main() -> None:
    exp_id = init_mlflow()
    runs = mlflow.search_runs(experiment_ids=[exp_id], order_by=["attributes.start_time ASC"])
    if runs.empty:
        raise SystemExit("no runs found - run src/run_experiment.py first")
    cols = {f"params.{p}": p for p in PARAMS} | {f"metrics.{m}": m for m in METRICS}
    df = runs[["run_id", "tags.note"] + [c for c in cols if c in runs.columns]].rename(columns=cols)
    df = df.rename(columns={"tags.note": "note"})
    df["promotable"] = runs["metrics.gate_passed"].fillna(0).astype(float).values == 1.0

    cand = df[df["promotable"]].sort_values(["pct_tests_passed", "task_success_rate", "avg_tokens_per_query"],
                                            ascending=[False, False, True])
    winner = cand.iloc[0]["prompt_version"] if not cand.empty else None

    client = MlflowClient()
    for _, r in df.iterrows():
        client.set_tag(r["run_id"], "promoted", str(r["prompt_version"] == winner))
    df["promoted"] = df["prompt_version"] == winner

    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "version_comparison.csv", index=False)
    table = df[["prompt_version"] + METRICS + ["promotable", "promoted"]].copy()
    table[METRICS] = table[METRICS].astype(float).round(3)
    md = table.to_markdown(index=False)
    config_md = df[["prompt_version"] + [p for p in PARAMS if p != "prompt_version"] + ["note"]].to_markdown(index=False)
    (REPORTS / "version_comparison.md").write_text(
        f"# Track B - MLflow comparison of prompt/config versions\n\nExperiment `{EXPERIMENT}`. "
        f"Promoted version: **{winner}**\n\n{md}\n\n## Configuration and rationale\n\n{config_md}\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = df["prompt_version"]
    for m in ("pct_tests_passed", "task_success_rate", "tool_selection_accuracy"):
        axes[0].plot(x, df[m].astype(float), marker="o", label=m)
    axes[0].axhline(0.8, color="grey", ls="--", lw=1)
    axes[0].set(title="Quality (dashed = 80% promotion gate)", ylim=(-0.05, 1.05))
    axes[0].legend(fontsize=8)
    axes[1].bar(x, df["avg_tokens_per_query"].astype(float), color="#4c72b0")
    axes[1].set(title="Cost: avg tokens per query")
    axes[2].bar(x, df["avg_latency_s"].astype(float), color="#dd8452")
    axes[2].set(title="Latency: avg seconds per query")
    fig.tight_layout()
    fig.savefig(REPORTS / "version_comparison.png", dpi=130)
    print(md)
    print(f"\npromoted version: {winner}")


if __name__ == "__main__":
    main()
