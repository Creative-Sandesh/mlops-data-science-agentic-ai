"""Drift monitoring with Evidently AI: reference (70%) vs current (30%, with injected drift).

For each scenario ("clean" = untouched 30% split, "drifted" = 30% split with synthetic drift) this:
  1. builds three Evidently HTML reports: data drift, target drift, custom metrics (+tests);
  2. checks that the columns we perturbed were flagged;
  3. scores the Production model on the current data to show the performance impact;
  4. logs everything (metrics, HTML, JSON summary) to the ``churn-monitoring`` MLflow experiment;
  5. decides whether retraining should be recommended.

Run:  uv run python src/monitor.py            (both scenarios)
      uv run python src/monitor.py --scenario drifted
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from evidently.tests import lt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from common import (
    CATEGORICAL, DATA_PROCESSED, FEATURES, MONITOR_EXPERIMENT, NUMERIC, REGISTERED_MODEL, REPORTS, TARGET,
    init_mlflow, split_train_test, xy,
)
from custom_metrics import MeanShift, SegmentRateShift
from drift import M2M, inject_drift

warnings.filterwarnings("ignore")

# Columns that inject_drift() deliberately perturbs (+ the target). Used to verify detection.
INJECTED = {"MonthlyCharges", "tenure", "Contract", TARGET}

# Retraining policy (also used by the Airflow DAG).
DRIFT_SHARE_THRESHOLD = 0.30      # >= 30% of feature columns drifted
MEAN_SHIFT_LIMIT = 5.0            # |mean MonthlyCharges shift| must stay under $5
SEGMENT_SHIFT_LIMIT_PP = 5.0      # |churn-rate shift in month-to-month segment| under 5 pp


def _definition() -> DataDefinition:
    return DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL + [TARGET])


def _dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[FEATURES + [TARGET]], data_definition=_definition())


def _column_drift(snapshot) -> pd.DataFrame:
    """Per-column drift verdicts parsed from the ValueDrift metrics of a snapshot."""
    rows = []
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if not cfg["type"].endswith(":ValueDrift"):
            continue
        method, thr, val = cfg["method"], cfg["threshold"], float(m["value"])
        # p-value tests flag drift when p < threshold; distance measures when distance >= threshold.
        drifted = val < thr if "p_value" in method.lower() or "p-value" in method.lower() else val >= thr
        rows.append({"column": cfg["column"], "method": method, "score": val, "threshold": thr, "drifted": drifted})
    return pd.DataFrame(rows)


def build_reports(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref, cur = _dataset(reference), _dataset(current)
    features = [c for c in FEATURES]

    data_drift = Report([DataDriftPreset(columns=features, drift_share=DRIFT_SHARE_THRESHOLD)]).run(cur, ref)
    target_drift = Report([
        ValueDrift(column=TARGET),
        SegmentRateShift(segment_column="Contract", segment_value=M2M, label_column=TARGET,
                         positive_label="Yes", tests=[lt(SEGMENT_SHIFT_LIMIT_PP)]),
    ]).run(cur, ref)
    custom = Report([
        MeanShift(column="MonthlyCharges", tests=[lt(MEAN_SHIFT_LIMIT)]),
        MeanShift(column="tenure"),
        SegmentRateShift(segment_column="Contract", segment_value=M2M, label_column=TARGET,
                         positive_label="Yes", tests=[lt(SEGMENT_SHIFT_LIMIT_PP)]),
    ]).run(cur, ref)
    return {"data_drift": data_drift, "target_drift": target_drift, "custom_metrics": custom}


def _performance(model, df: pd.DataFrame) -> dict[str, float]:
    X, y = xy(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba),
        "predicted_churn_rate": float(pred.mean()), "actual_churn_rate": float(y.mean()),
    }


def run_drift_check(scenario: str = "drifted", current: pd.DataFrame | None = None) -> dict:
    """Run one drift check and return a JSON-serialisable summary.

    ``current`` lets a caller (e.g. Airflow, or real production logs) supply the incoming batch;
    otherwise the 30% split is used, with synthetic drift injected when ``scenario == 'drifted'``.
    """
    reference = pd.read_csv(DATA_PROCESSED / "reference.csv")
    if current is None:
        current = pd.read_csv(DATA_PROCESSED / "current_clean.csv")
        if scenario == "drifted":
            current = inject_drift(current)

    reports = build_reports(reference, current)

    # ---- interpret -------------------------------------------------------------------------------
    col_drift = _column_drift(reports["data_drift"])
    col_drift = col_drift[col_drift["column"] != TARGET]  # the target is judged by its own report
    drifted_cols = sorted(col_drift.loc[col_drift["drifted"], "column"])
    share = len(drifted_cols) / len(col_drift)

    target_row = _column_drift(reports["target_drift"]).query("column == @TARGET").iloc[0]
    custom = {m["metric_name"].split("(")[0] + ":" + m["config"].get("column", m["config"].get("segment_value", "")):
              float(m["value"]) for m in reports["custom_metrics"].dict()["metrics"]}
    mean_shift = custom["MeanShift:MonthlyCharges"]
    seg_shift = custom["SegmentRateShift:Month-to-month"]
    failed_tests = [t["name"] for t in reports["custom_metrics"].dict()["tests"] if t["status"] != "SUCCESS"]

    # "Did Evidently catch what we injected?" only makes sense for the drifted scenario.
    injected = INJECTED - {TARGET}
    detected = sorted(injected & set(drifted_cols)) if scenario == "drifted" else None
    missed = sorted(injected - set(drifted_cols)) if scenario == "drifted" else None
    side_effects = sorted(set(drifted_cols) - INJECTED)

    reasons = []
    if share >= DRIFT_SHARE_THRESHOLD:
        reasons.append(f"{share:.0%} of feature columns drifted (>= {DRIFT_SHARE_THRESHOLD:.0%})")
    if bool(target_row["drifted"]):
        reasons.append(f"target '{TARGET}' drifted (score {target_row['score']:.3f})")
    if failed_tests:
        reasons.append(f"{len(failed_tests)} custom metric test(s) failed")
    retrain = bool(reasons)

    # ---- model impact ----------------------------------------------------------------------------
    exp_id = init_mlflow(MONITOR_EXPERIMENT)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/Production")
    _, ref_holdout = split_train_test(reference)  # rows the model never trained on
    perf_ref, perf_cur = _performance(model, ref_holdout), _performance(model, current)

    summary = {
        "scenario": scenario,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reference": len(reference), "n_current": len(current),
        "drifted_columns": drifted_cols, "drifted_share": round(share, 4),
        "injected_columns_detected": detected, "injected_columns_missed": missed,
        "side_effect_drifted_columns": side_effects,
        "target_drift": {"score": round(float(target_row["score"]), 4), "drifted": bool(target_row["drifted"]),
                         "churn_rate_reference": round(float((reference[TARGET] == "Yes").mean()), 4),
                         "churn_rate_current": round(float((current[TARGET] == "Yes").mean()), 4)},
        "custom_metrics": {"mean_monthly_charges_shift": round(mean_shift, 3),
                           "month_to_month_churn_shift_pp": round(seg_shift, 3),
                           "mean_tenure_shift": round(custom["MeanShift:tenure"], 3)},
        "failed_tests": failed_tests,
        "model_performance": {"reference": {k: round(v, 4) for k, v in perf_ref.items()},
                              "current": {k: round(v, 4) for k, v in perf_cur.items()}},
        "retrain_recommended": retrain, "reasons": reasons,
    }

    # ---- persist: local files + MLflow -----------------------------------------------------------
    out = REPORTS / "evidently"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, snap in reports.items():
        p = out / f"{scenario}_{name}.html"
        snap.save_html(str(p))
        files.append(p)
    col_drift.to_csv(out / f"{scenario}_column_drift.csv", index=False)
    (out / f"{scenario}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow.start_run(experiment_id=exp_id, run_name=f"drift-check-{scenario}"):
        mlflow.set_tags({"scenario": scenario, "retrain_recommended": str(retrain),
                         "drifted_columns": ",".join(drifted_cols)})
        mlflow.log_params({"drift_share_threshold": DRIFT_SHARE_THRESHOLD, "mean_shift_limit": MEAN_SHIFT_LIMIT,
                           "segment_shift_limit_pp": SEGMENT_SHIFT_LIMIT_PP, "registered_model": REGISTERED_MODEL,
                           "n_reference": len(reference), "n_current": len(current)})
        mlflow.log_metrics({
            "n_drifted_columns": len(drifted_cols), "drifted_share": share,
            "target_drift_score": float(target_row["score"]),
            "churn_rate_reference": summary["target_drift"]["churn_rate_reference"],
            "churn_rate_current": summary["target_drift"]["churn_rate_current"],
            "mean_monthly_charges_shift": mean_shift, "month_to_month_churn_shift_pp": seg_shift,
            "failed_custom_tests": len(failed_tests), "retrain_recommended": float(retrain),
            **{f"current_{k}": v for k, v in perf_cur.items()},
            **{f"reference_{k}": v for k, v in perf_ref.items()},
        })
        for p in files:
            mlflow.log_artifact(str(p), artifact_path="evidently_reports")
        mlflow.log_artifact(str(out / f"{scenario}_summary.json"), artifact_path="evidently_reports")
        mlflow.log_artifact(str(out / f"{scenario}_column_drift.csv"), artifact_path="evidently_reports")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["clean", "drifted", "both"], default="both")
    args = ap.parse_args()
    for sc in (["clean", "drifted"] if args.scenario == "both" else [args.scenario]):
        s = run_drift_check(sc)
        print(f"\n=== scenario: {sc} ===")
        print(json.dumps({k: s[k] for k in ("drifted_columns", "drifted_share", "injected_columns_detected",
                                            "injected_columns_missed", "side_effect_drifted_columns", "target_drift",
                                            "custom_metrics", "retrain_recommended", "reasons")}, indent=2))
        print("model f1 reference -> current:", s["model_performance"]["reference"]["f1"], "->",
              s["model_performance"]["current"]["f1"])


if __name__ == "__main__":
    main()
