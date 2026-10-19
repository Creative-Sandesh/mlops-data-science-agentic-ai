"""Airflow DAG (optional bonus): nightly regression evaluation of the agent, with degradation alerting.

    run_eval --> check_degradation --+--> alert_degradation
                                     +--> all_clear

* Schedule: nightly at 02:00 (`0 2 * * *`).
* run_eval: re-runs the harness + Evidently LLM regression suite for the currently PROMOTED prompt version
  (Airflow Variable `agent_prompt_version`, default "v3") and logs an MLflow run tagged `scheduled=true`.
* check_degradation: FAIL if any of these hold (thresholds are Airflow Variables with defaults):
    - pct_tests_passed        < 0.80   (`agent_min_pct_tests_passed`)
    - task_success_rate       drops by more than 0.10 vs the promoted baseline (`agent_max_success_drop`)
* On degradation: `alert_degradation` raises AirflowException so the run is marked failed (hook up
  email/Slack `on_failure_callback` in your deployment); otherwise `all_clear` only logs.

Set the project path with the env var AGENT_PROJECT_DIR (defaults to this file's parent folder).
Needs a running LLM endpoint (Ollama or any OpenAI-compatible server) reachable from the Airflow worker.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator

PROJECT_DIR = Path(os.environ.get("AGENT_PROJECT_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))


@dag(
    dag_id="agent_nightly_regression",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["mlops", "agent", "evidently", "regression"],
)
def agent_nightly_regression():
    @task
    def run_eval() -> dict:
        import run_experiment  # lazy import: heavy (torch, mlflow)

        version = Variable.get("agent_prompt_version", default_var="v3")
        return {"version": version, **run_experiment.main(version, extra_tags={"scheduled": "true"})}

    @task.branch
    def check_degradation(result: dict) -> str:
        min_pct = float(Variable.get("agent_min_pct_tests_passed", default_var="0.8"))
        max_drop = float(Variable.get("agent_max_success_drop", default_var="0.10"))
        baseline = float(Variable.get("agent_baseline_task_success", default_var="0.0"))
        pct = result["regression"]["pct_tests_passed"]
        success = result["harness"]["task_success_rate"]
        problems = []
        if pct < min_pct:
            problems.append(f"pct_tests_passed {pct:.2f} < {min_pct:.2f}")
        if baseline and baseline - success > max_drop:
            problems.append(f"task_success_rate {success:.2f} dropped > {max_drop:.2f} from baseline {baseline:.2f}")
        print("degradation:", problems or "none")
        return "alert_degradation" if problems else "all_clear"

    @task
    def alert_degradation() -> None:
        raise AirflowException("Agent regression suite degraded - see the latest MLflow run tagged scheduled=true")

    all_clear = EmptyOperator(task_id="all_clear")

    decision = check_degradation(run_eval())
    decision >> [alert_degradation(), all_clear]


agent_nightly_regression()
