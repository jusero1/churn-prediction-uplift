"""Propensity discrimination metrics, from the shared riskkit implementation."""

from __future__ import annotations

import numpy.typing as npt
from riskkit import auc, gini, ks


def metric_summary(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> dict[str, float]:
    """AUC, Gini and KS for a churn score, in one dict."""
    return {
        "auc": auc(y_true, y_score),
        "gini": gini(y_true, y_score),
        "ks": ks(y_true, y_score),
    }
