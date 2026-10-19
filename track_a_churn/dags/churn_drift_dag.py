"""Airflow DAG (optional bonus): scheduled drift check that triggers retraining on drift.

    check_drift --> decide --+--> retrain_model --> promote_model
                             +--> no_action

* Schedule: daily (`@daily`). Change `schedule` for weekly/monthly cadence.
* Trigger condition: `monitor.run_drift_check()` sets `retrain_recommended` when
  >= 30% of feature columns drifted, OR the target (Churn) drifted, OR a custom metric test failed
  (mean MonthlyCharges shift >= $5, or month-to-month churn-rate shift >= 5 pp).
* On a positive signal: re-train all candidate models (train.py), refresh the comparison table,
  and register + promote the winner (register.py: Staging -> smoke test -> Production).
  Set the Airflow Variable `churn_auto_retrain=false` to only *log a recommendation* instead.

In this demo the "incoming" batch is the 30% current split with synthetic drift injected. In a real
deployment `_load_incoming_batch()` would read the last day's logged production requests.
Set the environment variable CHURN_PROJECT_DIR to the track_a_churn folder (defaults to this file's parent).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator

PROJECT_DIR = Path(os.environ.get("CHURN_PROJECT_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_DIR / "src"))


@dag(
    dag_id="churn_drift_monitoring",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "churn", "evidently"],
)
def churn_drift_monitoring():
    @task
    def check_drift() -> dict:
        import monitor  # imported lazily so the scheduler can parse this file quickly

        return monitor.run_drift_check(scenario="drifted")

    @task.branch
    def decide(summary: dict) -> str:
        print(f"drifted columns: {summary['drifted_columns']} | reasons: {summary['reasons']}")
        if not summary["retrain_recommended"]:
            return "no_action"
        if Variable.get("churn_auto_retrain", default_var="true").lower() != "true":
            print("RECOMMENDATION: retrain the churn model (auto-retrain disabled)")
            return "no_action"
        return "retrain_model"

    @task
    def retrain_model() -> None:
        import compare
        import train

        train.main()
        compare.main()

    @task
    def promote_model() -> None:
        import register

        register.main()

    no_action = EmptyOperator(task_id="no_action")

    retrain = retrain_model()
    decision = decide(check_drift())
    decision >> [retrain, no_action]
    retrain >> promote_model()


churn_drift_monitoring()
