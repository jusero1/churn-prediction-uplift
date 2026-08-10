"""Matplotlib figures for the report and notebooks (backend-agnostic)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from riskkit import auc
from sklearn.metrics import roc_curve

_TEAL = "#0EA5E9"
_AMBER = "#F59E0B"
_GREY = "#94A3B8"
_GREEN = "#10B981"


def plot_propensity_roc(y_true, y_score) -> plt.Figure:
    """ROC curve for the churn propensity model."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    label = f"Churn (AUC {auc(y_true, y_score):.3f})"
    ax.plot(fpr, tpr, color=_TEAL, lw=2, label=label)
    ax.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Churn propensity — ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_km(fitters: dict) -> plt.Figure:
    """Overlay one or more Kaplan-Meier retention curves."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [_TEAL, _AMBER, _GREEN, _GREY]
    for (label, kmf), color in zip(fitters.items(), colors, strict=False):
        surv = kmf.survival_function_
        ax.step(
            surv.index,
            surv.iloc[:, 0],
            where="post",
            color=color,
            lw=2,
            label=str(label),
        )
    ax.set_xlabel("Tenure (months in service)")
    ax.set_ylabel("Retention S(t)")
    ax.set_title("Kaplan-Meier retention curve")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_qini(curve: pd.DataFrame) -> plt.Figure:
    """Qini curve vs the random baseline."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = curve["fraction"].to_numpy()
    q = curve["qini"].to_numpy()
    ax.plot(x, q, color=_TEAL, lw=2, label="Uplift model")
    ax.plot([0, 1], [0, q[-1]], color=_GREY, ls="--", lw=1.5, label="Random")
    ax.set_xlabel("Fraction targeted (by predicted uplift)")
    ax.set_ylabel("Incremental responses (Qini)")
    ax.set_title("Qini curve")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_uplift_by_decile(table: pd.DataFrame) -> plt.Figure:
    """Observed uplift per decile of predicted uplift."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = np.where(table["observed_uplift"] >= 0, _TEAL, _AMBER)
    ax.bar(table["decile"], table["observed_uplift"], color=colors)
    ax.axhline(0, color=_GREY, lw=1)
    ax.set_xlabel("Predicted-uplift decile (1 = highest)")
    ax.set_ylabel("Observed uplift")
    ax.set_title("Uplift by decile (validation)")
    fig.tight_layout()
    return fig


def plot_campaign(simulation: pd.DataFrame) -> plt.Figure:
    """Net value vs targeting fraction from the campaign simulation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        simulation["fraction"],
        simulation["net_value"],
        color=_GREEN,
        marker="o",
        lw=2,
    )
    ax.axhline(0, color=_GREY, lw=1)
    ax.set_xlabel("Fraction of base contacted")
    ax.set_ylabel("Net value")
    ax.set_title("Campaign net value vs reach")
    fig.tight_layout()
    return fig
