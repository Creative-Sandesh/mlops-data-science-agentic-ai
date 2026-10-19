"""Deliberately inject synthetic drift into the 'current' (incoming production) data."""
from __future__ import annotations

import numpy as np
import pandas as pd

M2M = "Month-to-month"


def inject_drift(current: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Return a drifted copy of ``current``. Four perturbations:

    1. MonthlyCharges: + random per-row offset ~ N(20, 8), floored at 18 (numeric drift).
    2. tenure: minus a random 0-18 months, floored at 0 (numeric drift: newer customer base).
    3. Contract: over-sample 'Month-to-month' rows until they are ~75% of the data
       (natural share is ~55%) - categorical distribution skew.
    4. Churn label: flip 12% of 'No' -> 'Yes' (label / concept drift, raises churn rate).

    TotalCharges is never touched directly, so if Evidently flags it that is a spill-over effect
    of the Contract resampling (month-to-month customers have lower lifetime charges).
    """
    rng = np.random.default_rng(seed)
    df = current.copy()

    df["MonthlyCharges"] = (df["MonthlyCharges"] + rng.normal(20, 8, len(df))).clip(lower=18).round(2)
    df["tenure"] = (df["tenure"] - rng.integers(0, 19, len(df))).clip(lower=0)

    is_m2m = df["Contract"] == M2M
    target_share = 0.75
    n_extra = int((target_share * len(df) - is_m2m.sum()) / (1 - target_share))
    if n_extra > 0:
        extra = df[is_m2m].sample(n=n_extra, replace=True, random_state=seed)
        df = pd.concat([df, extra], ignore_index=True)

    no_idx = df.index[df["Churn"] == "No"]
    flip = rng.choice(no_idx, size=int(0.12 * len(no_idx)), replace=False)
    df.loc[flip, "Churn"] = "Yes"
    return df
