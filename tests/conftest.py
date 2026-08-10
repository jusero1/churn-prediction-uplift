"""Shared fixtures. Tests never download data; they build synthetic frames with
a known structure so the metrics and models can be checked against ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def uplift_frame() -> pd.DataFrame:
    """Randomized treatment/control data with a KNOWN heterogeneous effect.

    Persuadables are the ``x0 > 0`` group: the treatment lifts their outcome
    probability, and it does nothing for the rest. A working uplift model must
    rank the persuadables at the top.
    """
    rng = np.random.default_rng(2026)
    n = 6000
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)

    base = 0.15 + 0.05 * (x1 > 0)
    effect = np.where(x0 > 0, 0.20, 0.0)  # only x0>0 responds to treatment
    prob = base + treatment * effect
    outcome = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {"x0": x0, "x1": x1, "treatment": treatment, "outcome": outcome}
    )


@pytest.fixture
def survival_frame() -> pd.DataFrame:
    """Synthetic tenure/event data where a covariate raises the hazard."""
    rng = np.random.default_rng(7)
    n = 2000
    risk = rng.normal(size=n)  # higher risk -> shorter tenure
    duration = np.clip(np.round(rng.exponential(scale=30) / np.exp(0.4 * risk)), 1, 72)
    event = (duration < 60).astype(int)  # long tenures are right-censored
    return pd.DataFrame({"MonthsInService": duration, "churn": event, "risk": risk})
