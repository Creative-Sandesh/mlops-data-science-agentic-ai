"""Thin FastAPI wrapper that serves the model currently in the registry's Production stage.

Run:  uv run uvicorn --app-dir src serve:app --port 8000
Try:  curl localhost:8000/health ;  curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json

(Alternative with no custom code: `mlflow models serve -m models:/churn-classifier/Production -p 5001 --env-manager local`
with MLFLOW_TRACKING_URI set to the sqlite URI from src/common.py.)
"""
from __future__ import annotations

import os
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from common import FEATURES, REGISTERED_MODEL, TRACKING_URI

warnings.filterwarnings("ignore", category=FutureWarning)

STAGE = os.getenv("MODEL_STAGE", "Production")

app = FastAPI(title="Churn model API", version="1.0.0")
_state: dict = {}


class Customer(BaseModel):
    """One customer record. Field names/values follow the Telco CSV columns."""
    model_config = {"extra": "forbid"}

    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


@app.on_event("startup")
def load_model() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{REGISTERED_MODEL}/{STAGE}"
    _state["model"] = mlflow.sklearn.load_model(uri)
    versions = MlflowClient().get_latest_versions(REGISTERED_MODEL, stages=[STAGE])
    _state["info"] = {"model_uri": uri, "version": versions[0].version if versions else None,
                      "source_run": versions[0].tags.get("source_run_name") if versions else None}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/model-info")
def model_info() -> dict:
    return _state["info"]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if "model" not in _state:
        raise HTTPException(503, "model not loaded")
    df = pd.DataFrame([c.model_dump() for c in req.customers])[FEATURES]
    proba = _state["model"].predict_proba(df)[:, 1]
    return {
        "model": _state["info"],
        "predictions": [
            {"churn_probability": round(float(p), 4), "churn": bool(p >= 0.5)} for p in proba
        ],
    }
