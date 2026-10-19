"""Custom Evidently metrics (beyond the built-in defaults) for the churn monitoring report.

* ``MeanShift``           - current mean minus reference mean of a numeric column
                            (used for MonthlyCharges: how many dollars did the average bill move?).
* ``SegmentRateShift``    - change (percentage points) in the positive-label rate inside one
                            segment, e.g. churn rate among ``Contract == 'Month-to-month'``.

Both are ``SingleValueMetric`` s with attached tests, so they show up in the HTML report
and in Evidently's pass/fail test summary.
"""
from __future__ import annotations

from evidently.core.datasets import Dataset
from evidently.core.metric_types import SingleValue, SingleValueCalculation, SingleValueMetric
from evidently.core.report import Context


class MeanShift(SingleValueMetric):
    column: str


class MeanShiftCalculation(SingleValueCalculation[MeanShift]):
    def calculate(self, context: Context, current_data: Dataset, reference_data: Dataset | None) -> SingleValue:
        if reference_data is None:
            raise ValueError("MeanShift needs reference data")
        cur = current_data.column(self.metric.column).data.astype(float).mean()
        ref = reference_data.column(self.metric.column).data.astype(float).mean()
        return self.result(float(cur - ref))

    def display_name(self) -> str:
        return f"Mean shift of '{self.metric.column}' (current - reference)"


class SegmentRateShift(SingleValueMetric):
    segment_column: str
    segment_value: str
    label_column: str
    positive_label: str


class SegmentRateShiftCalculation(SingleValueCalculation[SegmentRateShift]):
    def _rate(self, data: Dataset) -> float:
        m = self.metric
        df = data.as_dataframe()
        seg = df[df[m.segment_column] == m.segment_value]
        return float((seg[m.label_column] == m.positive_label).mean()) if len(seg) else float("nan")

    def calculate(self, context: Context, current_data: Dataset, reference_data: Dataset | None) -> SingleValue:
        if reference_data is None:
            raise ValueError("SegmentRateShift needs reference data")
        return self.result((self._rate(current_data) - self._rate(reference_data)) * 100.0)

    def display_name(self) -> str:
        m = self.metric
        return (f"Shift (pp) in {m.label_column}={m.positive_label} rate for "
                f"{m.segment_column}={m.segment_value} (current - reference)")
