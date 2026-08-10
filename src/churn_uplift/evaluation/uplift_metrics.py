"""Uplift evaluation: Qini curve, Qini coefficient and uplift by decile.

These are the uplift analogues of the ROC/gains curves. They compare, along the
population ranked by predicted uplift, the response of the treated vs. the
control group — so they measure *incremental* effect, which a standard accuracy
metric cannot see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_treatment_effect(treatment: np.ndarray, outcome: np.ndarray) -> float:
    """Overall uplift (ATE): mean outcome under treatment minus under control."""
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    return float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def qini_curve(
    uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> pd.DataFrame:
    """Cumulative Qini curve over the population ranked by predicted uplift.

    At each prefix of the ranked population the Qini value is the treated
    responders minus the control responders rescaled to the treated count:
    ``Y_t - Y_c * N_t / N_c``. It is the incremental responders captured by
    targeting that fraction.

    Returns
    -------
    pandas.DataFrame
        Columns ``fraction`` (share targeted) and ``qini`` (incremental
        responders), starting at the origin (0, 0).
    """
    uplift = np.asarray(uplift, dtype=float)
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)

    order = np.argsort(-uplift, kind="mergesort")
    t = treatment[order]
    y = outcome[order]

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))

    safe_c = np.where(cum_c == 0, 1, cum_c)
    qini = cum_yt - cum_yc * (cum_t / safe_c)

    n = len(uplift)
    fraction = np.arange(1, n + 1) / n
    return pd.DataFrame(
        {
            "fraction": np.concatenate([[0.0], fraction]),
            "qini": np.concatenate([[0.0], qini]),
        }
    )


def qini_coefficient(
    uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> float:
    """Area between the model Qini curve and the random (diagonal) baseline.

    Positive means the model ranks persuadables above the average; 0 is random.
    Normalised by the number of treated so it reads as a fraction.
    """
    curve = qini_curve(uplift, treatment, outcome)
    x = curve["fraction"].to_numpy()
    q = curve["qini"].to_numpy()
    # Random baseline: straight line from origin to the final Qini value.
    random_line = x * q[-1]
    area = np.trapezoid(q - random_line, x)
    n_treated = int(np.sum(np.asarray(treatment) == 1))
    return float(area / n_treated) if n_treated else 0.0


def uplift_by_decile(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed uplift within deciles of predicted uplift (decile 1 = highest).

    The validation an operator trusts: if the model works, observed uplift
    (treated response minus control response) falls monotonically from the top
    decile down, and can go negative for "sleeping dogs".
    """
    df = pd.DataFrame(
        {
            "uplift": np.asarray(uplift, dtype=float),
            "treatment": np.asarray(treatment),
            "outcome": np.asarray(outcome),
        }
    )
    ranks = df["uplift"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(ranks, q=n_bins, labels=range(1, n_bins + 1))

    rows = []
    for decile, g in df.groupby("decile", observed=True):
        treated = g.loc[g["treatment"] == 1, "outcome"]
        control = g.loc[g["treatment"] == 0, "outcome"]
        rows.append(
            {
                "decile": int(decile),
                "n": len(g),
                "resp_treated": round(treated.mean(), 4) if len(treated) else np.nan,
                "resp_control": round(control.mean(), 4) if len(control) else np.nan,
                "observed_uplift": round(
                    (treated.mean() if len(treated) else 0)
                    - (control.mean() if len(control) else 0),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
