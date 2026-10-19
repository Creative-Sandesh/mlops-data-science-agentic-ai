import pandas as pd
import pytest
from evidently import DataDefinition, Dataset, Report
from evidently.tests import lt

from common import CATEGORICAL, DATA_PROCESSED, FEATURES, NUMERIC, TARGET, split_reference_current, split_train_test
from custom_metrics import MeanShift, SegmentRateShift
from drift import M2M, inject_drift

pytestmark = pytest.mark.skipif(
    not (DATA_PROCESSED / "reference.csv").exists(), reason="run src/train.py first to create the splits"
)


@pytest.fixture(scope="module")
def frames():
    return (pd.read_csv(DATA_PROCESSED / "reference.csv"), pd.read_csv(DATA_PROCESSED / "current_clean.csv"))


def test_split_is_70_30_and_disjoint(frames):
    ref, cur = frames
    assert abs(len(ref) / (len(ref) + len(cur)) - 0.70) < 0.01


def test_train_test_split_is_deterministic_and_stratified(frames):
    ref, _ = frames
    a, b = split_train_test(ref), split_train_test(ref)
    assert a[1].index.equals(b[1].index)
    assert abs((a[1][TARGET] == "Yes").mean() - (ref[TARGET] == "Yes").mean()) < 0.01


def test_inject_drift_changes_what_it_claims(frames):
    _, cur = frames
    d = inject_drift(cur)
    assert d["MonthlyCharges"].mean() - cur["MonthlyCharges"].mean() > 10
    assert d["tenure"].mean() < cur["tenure"].mean() - 5
    assert (d["Contract"] == M2M).mean() > 0.70
    assert (d["Churn"] == "Yes").mean() > (cur["Churn"] == "Yes").mean() + 0.08
    assert inject_drift(cur).equals(d), "injection must be reproducible"


def test_custom_metrics_values_and_tests(frames):
    ref, cur = frames
    dd = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL + [TARGET])
    mk = lambda df: Dataset.from_pandas(df[FEATURES + [TARGET]], data_definition=dd)  # noqa: E731
    drifted = inject_drift(cur)
    snap = Report([
        MeanShift(column="MonthlyCharges", tests=[lt(5)]),
        SegmentRateShift(segment_column="Contract", segment_value=M2M, label_column=TARGET,
                         positive_label="Yes", tests=[lt(5)]),
    ]).run(mk(drifted), mk(ref))
    values = [float(m["value"]) for m in snap.dict()["metrics"]]
    assert values[0] == pytest.approx(drifted["MonthlyCharges"].mean() - ref["MonthlyCharges"].mean())
    assert all(t["status"] == "FAIL" for t in snap.dict()["tests"])
