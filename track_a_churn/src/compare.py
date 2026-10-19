"""Export the side-by-side MLflow run comparison (params + metrics) to reports/.

Run:  uv run python src/compare.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from common import REPORTS, TRAIN_EXPERIMENT, init_mlflow

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "fit_seconds"]
PARAM_COLS = ["model_family", "C", "penalty", "n_estimators", "max_depth", "learning_rate",
              "min_samples_leaf", "class_weight", "scale_pos_weight"]


def comparison_table(experiment: str = TRAIN_EXPERIMENT) -> pd.DataFrame:
    exp_id = init_mlflow(experiment)
    runs = mlflow.search_runs(experiment_ids=[exp_id], order_by=["metrics.f1 DESC", "metrics.roc_auc DESC"])
    keep = {"run_id": "run_id", "tags.mlflow.runName": "run"}
    keep.update({f"params.{p}": p for p in PARAM_COLS})
    keep.update({f"metrics.{m}": m for m in METRICS})
    df = runs[[c for c in keep if c in runs.columns]].rename(columns=keep)
    return df.reset_index(drop=True)


def plot_comparison(df: pd.DataFrame) -> None:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = df.set_index("run")[cols].astype(float).plot.bar(figsize=(11, 5), width=0.8, colormap="viridis")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("score (held-out test)")
    ax.set_title("Track A - run comparison (sorted by F1). Accuracy alone would favour rf_depth12.")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(REPORTS / "run_comparison.png", dpi=130)
    plt.close()


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = comparison_table()
    df.to_csv(REPORTS / "run_comparison.csv", index=False)
    plot_comparison(df)

    metrics_md = df[["run"] + METRICS].copy()
    metrics_md[METRICS] = metrics_md[METRICS].astype(float).round(4)
    table = metrics_md.to_markdown(index=False)
    (REPORTS / "run_comparison.md").write_text(
        "# Track A - MLflow run comparison (held-out test split, threshold 0.5)\n\n"
        "Sorted by F1 then ROC-AUC.\n\n" + table + "\n\n## Hyperparameters\n\n"
        + df[["run"] + [c for c in PARAM_COLS if c in df.columns]].fillna("-").to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(table)


if __name__ == "__main__":
    main()
