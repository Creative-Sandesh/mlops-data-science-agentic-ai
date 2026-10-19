"""Shared paths, constants and data helpers for the churn pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# Keep MLflow quiet and stop it from recording the names of unrelated env vars in model metadata.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", "false")

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()

TRAIN_EXPERIMENT = "churn-training"
MONITOR_EXPERIMENT = "churn-monitoring"
REGISTERED_MODEL = "churn-classifier"

TARGET = "Churn"
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL

SEED = 42
REFERENCE_FRACTION = 0.70  # reference ("training-time") share; the rest is "incoming" data


def init_mlflow(experiment: str) -> str:
    """Point MLflow at the local SQLite store and return the experiment id (created on demand)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
    return exp.experiment_id


def load_raw() -> pd.DataFrame:
    """Load and clean the raw Telco CSV. Churn stays as 'Yes'/'No' here."""
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"{DATA_RAW} missing - run `uv run python src/get_data.py` first")
    df = pd.read_csv(DATA_RAW)
    # TotalCharges is blank for brand-new customers (tenure == 0) -> not numeric in the raw file.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    return df.drop(columns=["customerID"])


def split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 70/30 split: reference = training-time data, current = incoming production data."""
    reference = df.sample(frac=REFERENCE_FRACTION, random_state=SEED)
    current = df.drop(reference.index)
    return reference.reset_index(drop=True), current.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], (df[TARGET] == "Yes").astype(int)


def split_train_test(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split of the reference data. Shared by training and monitoring so that
    'reference performance' is always measured on rows the model never trained on."""
    return train_test_split(reference, test_size=0.2, stratify=reference[TARGET], random_state=SEED)
