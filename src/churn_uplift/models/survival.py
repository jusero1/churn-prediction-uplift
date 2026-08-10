"""Survival analysis of time-to-churn: Kaplan-Meier and Cox.

A propensity score says *whether* a customer will churn; survival says *when*.
Duration is tenure (``MonthsInService``); churners are events and still-active
customers are right-censored at their current tenure. Kaplan-Meier gives the
non-parametric retention curve (optionally by segment); Cox gives covariate
hazard ratios — how each driver multiplies the instantaneous churn risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier(
    durations: np.ndarray,
    events: np.ndarray,
    label: str = "all",
):
    """Fit a Kaplan-Meier estimator and return the fitted object.

    Parameters
    ----------
    durations : array-like
        Observed tenure per customer.
    events : array-like
        1 if churn was observed, 0 if right-censored (still active).
    label : str
        Curve label (useful when overlaying segments).

    Returns
    -------
    lifelines.KaplanMeierFitter
        Fitted estimator; ``.survival_function_`` and ``.median_survival_time_``
        hold the retention curve and median lifetime.
    """
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter(label=label)
    kmf.fit(durations, event_observed=events)
    return kmf


def kaplan_meier_by_group(
    durations: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Fit one Kaplan-Meier curve per group value. Returns {group: fitter}."""
    durations = np.asarray(durations)
    events = np.asarray(events)
    groups = np.asarray(groups)
    return {
        g: kaplan_meier(durations[groups == g], events[groups == g], label=str(g))
        for g in pd.unique(groups)
    }


def cox_model(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    covariates: list[str],
):
    """Fit a Cox proportional-hazards model on the given covariates.

    Returns
    -------
    lifelines.CoxPHFitter
        Fitted model; ``.summary`` holds coefficients, hazard ratios
        (``exp(coef)``) and p-values. A hazard ratio > 1 means the covariate
        increases churn risk.
    """
    from lifelines import CoxPHFitter

    data = df[[duration_col, event_col, *covariates]].dropna()
    model = CoxPHFitter()
    model.fit(data, duration_col=duration_col, event_col=event_col)
    return model


def hazard_ratios(cox) -> pd.DataFrame:
    """Return a tidy table of hazard ratios sorted by effect size."""
    summary = cox.summary[["coef", "exp(coef)", "p"]].copy()
    summary = summary.rename(columns={"exp(coef)": "hazard_ratio"})
    summary["direction"] = np.where(
        summary["hazard_ratio"] > 1, "increases churn", "reduces churn"
    )
    return summary.sort_values("hazard_ratio", ascending=False)
