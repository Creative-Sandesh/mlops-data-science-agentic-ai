"""Register the best run in the MLflow Model Registry and promote it Staging -> Production.

Selection rule: highest F1 on the held-out test set (ROC-AUC breaks ties). Accuracy is *not*
used because only ~26.5% of customers churn, so accuracy rewards under-predicting churn.

Before the Staging -> Production promotion, the model is loaded back *from the registry* and
must pass a smoke test (right output shape, probabilities in [0, 1]).

Run:  uv run python src/register.py
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from common import DATA_PROCESSED, FEATURES, REGISTERED_MODEL, REPORTS, TRAIN_EXPERIMENT, init_mlflow

warnings.filterwarnings("ignore", message=".*stage.*deprecated.*", category=FutureWarning)


def main() -> None:
    exp_id = init_mlflow(TRAIN_EXPERIMENT)
    client = MlflowClient()

    runs = mlflow.search_runs(experiment_ids=[exp_id], order_by=["metrics.f1 DESC", "metrics.roc_auc DESC"])
    best = runs.iloc[0]
    run_id, run_name = best["run_id"], best["tags.mlflow.runName"]
    print(f"best run: {run_name} ({run_id}) f1={best['metrics.f1']:.4f} roc_auc={best['metrics.roc_auc']:.4f}")

    log: list[dict] = []

    def record(event: str, **kw) -> None:
        log.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **kw})
        print(f"  - {event} {kw}")

    mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL)
    version = mv.version
    record("registered", model=REGISTERED_MODEL, version=version, run=run_name, f1=round(float(best["metrics.f1"]), 4),
           roc_auc=round(float(best["metrics.roc_auc"]), 4))
    client.set_model_version_tag(REGISTERED_MODEL, version, "source_run_name", run_name)
    client.update_model_version(
        REGISTERED_MODEL, version,
        description=f"{run_name}: best F1 ({best['metrics.f1']:.3f}) / ROC-AUC ({best['metrics.roc_auc']:.3f}) "
                    f"of {len(runs)} runs on the held-out test split.",
    )

    client.transition_model_version_stage(REGISTERED_MODEL, version, stage="Staging")
    record("transition", version=version, to="Staging")

    # Validation gate: load from the registry exactly as a consumer would and score real rows.
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/Staging")
    sample = pd.read_csv(DATA_PROCESSED / "reference.csv")[FEATURES].head(50)
    proba = model.predict_proba(sample)[:, 1]
    assert proba.shape == (50,) and ((proba >= 0) & (proba <= 1)).all(), "smoke test failed"
    record("smoke_test_passed", stage="Staging", rows=len(sample))

    client.transition_model_version_stage(REGISTERED_MODEL, version, stage="Production", archive_existing_versions=True)
    record("transition", version=version, to="Production", archived_previous_versions=True)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "registry_transitions.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\nRegistry state:")
    for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
        print(f"  {v.name} v{v.version}: stage={v.current_stage} run={v.tags.get('source_run_name')}")


if __name__ == "__main__":
    main()
