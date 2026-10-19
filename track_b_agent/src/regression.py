"""Regression test suite (Evidently AI + LLM-as-a-judge) for the agent's answers.

Every prompt/config version answers the same fixed queries; each answer is compared with an approved
golden reference (eval/golden_responses.json) by an LLM judge with TWO checks:

  1. Correctness (reference-based, BinaryClassificationPromptTemplate): does the response contradict the
     reference or lose information that the reference contains?      -> "correct" / "incorrect"
  2. Grounding (no fabrication): does the response assert concrete facts / numbers / names that are NOT
     supported by the reference?                                       -> "supported" / "unsupported"

A query (test case) PASSES only if both checks pass. `pct_tests_passed` = share of passing queries; the
version is promotable only if it reaches PASS_THRESHOLD. Evidently's Report additionally evaluates
suite-level tests ("share of passed row tests >= threshold") for each check and renders the HTML report.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.descriptors import LLMEval
from evidently.legacy.options.base import Options
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently.llm.utils.wrapper import OllamaOptions
from evidently.metrics import MeanValue, UniqueValueCount
from evidently.tests import eq, gte

PASS_THRESHOLD = 0.80  # minimum share of test cases that must pass for a version to be promotable
NO_ANSWER = "(no answer produced: the agent stopped without a final answer)"

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "ollama")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma:latest")
JUDGE_API_URL = os.getenv("JUDGE_API_URL", "http://localhost:11434")

CORRECTNESS = BinaryClassificationPromptTemplate(
    criteria="""A RESPONSE is correct when it agrees with the REFERENCE on every fact, name and number it contains, even if it is worded differently or is shorter/longer.
A RESPONSE is incorrect when it contradicts the REFERENCE, gives a different number, or leaves out information that the REFERENCE contains and that is needed to answer the question. A response that says it does not know is only correct if the REFERENCE also says the answer is unknown.
REFERENCE:
=====
{reference}
=====""",
    target_category="incorrect", non_target_category="correct", uncertainty="target",
    include_reasoning=True,
    pre_messages=[("system", "You are a strict evaluator that compares an answer with an approved reference answer.")],
)

GROUNDING = BinaryClassificationPromptTemplate(
    criteria="""A RESPONSE is supported when every concrete fact, number, price, date or name it states can be found in (or directly calculated from) the REFERENCE.
A RESPONSE is unsupported when it states any concrete fact, number, price, date or name that does NOT appear in the REFERENCE and cannot be derived from it (an invented or fabricated detail).
REFERENCE:
=====
{reference}
=====""",
    target_category="unsupported", non_target_category="supported", uncertainty="target",
    include_reasoning=True,
    pre_messages=[("system", "You are a strict fact-checker that looks for fabricated details in an answer.")],
)

CHECKS = {"Correctness": "correct", "Grounding": "supported"}


def _options() -> Options:
    return Options(custom={OllamaOptions: OllamaOptions(api_url=JUDGE_API_URL)})


def run_regression(version: str, queries: list[dict], traces: list[dict], golden: dict[str, str],
                   out_dir: Path, threshold: float = PASS_THRESHOLD) -> dict:
    """Judge one version's answers against the golden set; write the Evidently HTML + verdict CSV to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "query_id": [q["id"] for q in queries],
        "question": [q["question"] for q in queries],
        "response": [t["final_answer"] or NO_ANSWER for t in traces],
        "reference": [golden[q["id"]] for q in queries],
    })
    descriptors = [
        LLMEval("response", provider=JUDGE_PROVIDER, model=JUDGE_MODEL, template=tmpl,
                additional_columns={"reference": "reference"}, alias=name,
                tests=[eq(positive, column=name)])
        for (name, positive), tmpl in zip(CHECKS.items(), (CORRECTNESS, GROUNDING))
    ]
    definition = DataDefinition(text_columns=["question", "response", "reference"], id_column="query_id")
    judged = Dataset.from_pandas(df, data_definition=definition, descriptors=descriptors, options=_options())
    verdicts = judged.as_dataframe()

    # The judge is a small local LLM: it varies verdict casing ("CORRECT", "Unsupported") and may abstain
    # ("UNKNOWN"). Normalise, and count anything that is not the positive verdict as a FAILED check.
    # An empty answer can never pass - the judge must not be able to "agree" with a refusal reference
    # for an answer that was never produced (observed for v1/q12).
    no_answer = df["response"].eq(NO_ANSWER).to_numpy()
    for name, positive in CHECKS.items():
        verdicts[name] = verdicts[name].astype(str).str.strip().str.lower()
        verdicts[f"{name}_pass"] = (verdicts[name] == positive) & ~no_answer
    verdicts["case_pass"] = verdicts[[f"{n}_pass" for n in CHECKS]].all(axis=1)

    # Evidently test suite over the (normalised) verdicts: each check and the overall case must reach the threshold.
    suite = verdicts[["query_id", "question", "response", "reference"] + list(CHECKS)].copy()
    for name in CHECKS:
        suite[f"{name} pass"] = verdicts[f"{name}_pass"].astype(int)
    suite["Case pass (all checks)"] = verdicts["case_pass"].astype(int)
    result_def = DataDefinition(text_columns=["question", "response", "reference"],
                                categorical_columns=list(CHECKS),
                                numerical_columns=[f"{n} pass" for n in CHECKS] + ["Case pass (all checks)"],
                                id_column="query_id")
    suite_ds = Dataset.from_pandas(suite, data_definition=result_def)
    metrics = [MeanValue(column=f"{n} pass", tests=[gte(threshold, alias=f"{n}: share of cases passing >= {threshold:.0%}")])
               for n in CHECKS]
    metrics.append(MeanValue(column="Case pass (all checks)",
                             tests=[gte(threshold, alias=f"Overall: share of cases passing all checks >= {threshold:.0%}")]))
    metrics += [UniqueValueCount(column=n) for n in CHECKS]
    snapshot = Report(metrics, include_tests=True).run(suite_ds, None)
    html_path = out_dir / f"evidently_regression_{version}.html"
    snapshot.save_html(str(html_path))
    suite_tests = [{"name": t["description"], "status": str(t["status"]).split(".")[-1]}
                   for t in snapshot.dict()["tests"] if "share of cases" in t["description"]]

    keep = ["query_id", "question", "response", "reference"] + [c for n in CHECKS for c in (n, f"{n} reasoning")] \
        + [f"{n}_pass" for n in CHECKS] + ["case_pass"]
    verdicts[keep].to_csv(out_dir / f"judge_verdicts_{version}.csv", index=False)

    pct = float(verdicts["case_pass"].mean())
    return {
        "pct_tests_passed": pct,
        "pct_correct": float(verdicts["Correctness_pass"].mean()),
        "pct_grounded": float(verdicts["Grounding_pass"].mean()),
        "gate_passed": pct >= threshold,
        "threshold": threshold,
        "failed_cases": verdicts.loc[~verdicts["case_pass"], "query_id"].tolist(),
        "suite_tests": suite_tests,
        "verdicts": verdicts[keep],
        "html_path": html_path,
    }
