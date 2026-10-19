"""Train several churn models and log every run to MLflow.

Training data = the 70% *reference* split (the "training-time" data used later for drift
monitoring). It is further split 80/20 (stratified) into train / held-out test. The 30%
*current* split is never seen here.

Run:  uv run python src/train.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (
    DATA_PROCESSED, SEED, TRAIN_EXPERIMENT, init_mlflow, load_raw, make_preprocessor,
    split_reference_current, split_train_test, xy,
)

# Imbalance handling: ~26.5% of customers churn. Linear/forest models use class_weight,
# XGBoost uses scale_pos_weight (~ negatives / positives).
POS_WEIGHT = 2.77

CONFIGS: list[dict] = [
    {"run_name": "logreg_C0.1", "family": "logistic_regression",
     "params": {"C": 0.1, "penalty": "l2", "class_weight": "balanced", "max_iter": 1000}},
    {"run_name": "logreg_C1.0", "family": "logistic_regression",
     "params": {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "max_iter": 1000}},
    {"run_name": "rf_depth5", "family": "random_forest",
     "params": {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 5, "class_weight": "balanced"}},
    {"run_name": "rf_depth12", "family": "random_forest",
     "params": {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"}},
    {"run_name": "xgb_depth3", "family": "xgboost",
     "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "scale_pos_weight": POS_WEIGHT}},
    {"run_name": "xgb_depth6", "family": "xgboost",
     "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": POS_WEIGHT}},
]


def build_estimator(family: str, params: dict):
    if family == "logistic_regression":
        return LogisticRegression(solver="lbfgs", random_state=SEED, **params)
    if family == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    if family == "xgboost":
        return XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, **params)
    raise ValueError(family)


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def log_plots(pipe: Pipeline, X_test, y_test, tmp: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, display_labels=["No churn", "Churn"], ax=ax)
    ax.set_title("Confusion matrix (held-out test)")
    fig.tight_layout()
    fig.savefig(tmp / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curve (held-out test)")
    fig.tight_layout()
    fig.savefig(tmp / "roc_curve.png", dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(tmp / "confusion_matrix.png"))
    mlflow.log_artifact(str(tmp / "roc_curve.png"))


def main() -> None:
    exp_id = init_mlflow(TRAIN_EXPERIMENT)

    reference, current = split_reference_current(load_raw())
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    reference.to_csv(DATA_PROCESSED / "reference.csv", index=False)
    current.to_csv(DATA_PROCESSED / "current_clean.csv", index=False)

    train_df, test_df = split_train_test(reference)
    X_train, y_train = xy(train_df)
    X_test, y_test = xy(test_df)
    print(f"reference={len(reference)} (train={len(X_train)}, test={len(X_test)}) current={len(current)}")

    for cfg in CONFIGS:
        with mlflow.start_run(experiment_id=exp_id, run_name=cfg["run_name"]):
            mlflow.set_tags({"model_family": cfg["family"], "stage": "experiment"})
            mlflow.log_param("model_family", cfg["family"])
            mlflow.log_params(cfg["params"])
            mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test), "seed": SEED})

            pipe = Pipeline([("prep", make_preprocessor()), ("clf", build_estimator(cfg["family"], cfg["params"]))])
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t0

            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = evaluate(y_test, pred, proba)
            metrics["fit_seconds"] = fit_seconds
            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as d:
                log_plots(pipe, X_test, y_test, Path(d))

            mlflow.sklearn.log_model(
                pipe, artifact_path="model",
                signature=infer_signature(X_test, pred),
                input_example=X_test.head(3),
            )
            print(f"{cfg['run_name']:<14} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
