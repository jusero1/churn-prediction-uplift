"""Campaign simulation: contact cost vs. incremental value under uplift targeting.

The closing question of the project: given a per-contact cost and a value per
incremental response, how much of the base should we actually contact, and what
does targeting by predicted uplift earn over contacting everyone?

Incremental response rates are estimated from the experiment itself: within the
top-fraction selected by predicted uplift, the incremental rate is the treated
response minus the control response for that same subgroup. Cost and value are
scenario inputs, not figures read from the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def campaign_simulation(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    cost_per_contact: float,
    value_per_response: float,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
) -> pd.DataFrame:
    """Net value of contacting the top-``fraction`` customers ranked by uplift.

    Parameters
    ----------
    uplift, treatment, outcome : array-like
        Predicted uplift and the experiment's treatment flag and observed
        outcome, on a held-out set.
    cost_per_contact : float
        Cost of contacting one customer.
    value_per_response : float
        Value of one incremental response (retained customer / conversion).
    fractions : tuple of float
        Fractions of the base to target, highest predicted uplift first.

    Returns
    -------
    pandas.DataFrame
        One row per fraction with the estimated incremental rate, incremental
        responses, cost, value and net value. The best row is the recommended
        campaign size.
    """
    uplift = np.asarray(uplift, dtype=float)
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    n = len(uplift)
    order = np.argsort(-uplift, kind="mergesort")

    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        idx = order[:k]
        t = treatment[idx]
        y = outcome[idx]
        # Estimated incremental response rate in the targeted subgroup.
        rate_t = y[t == 1].mean() if (t == 1).any() else 0.0
        rate_c = y[t == 0].mean() if (t == 0).any() else 0.0
        incremental_rate = rate_t - rate_c

        incremental_responses = incremental_rate * k
        cost = k * cost_per_contact
        value = incremental_responses * value_per_response
        net = value - cost
        rows.append(
            {
                "fraction": frac,
                "n_targeted": k,
                "incremental_rate": round(incremental_rate, 4),
                "incremental_responses": round(incremental_responses, 1),
                "cost": round(cost, 1),
                "value": round(value, 1),
                "net_value": round(net, 1),
                "roi": round(net / cost, 3) if cost else np.nan,
            }
        )
    return pd.DataFrame(rows)


def best_fraction(simulation: pd.DataFrame) -> float:
    """Return the targeting fraction that maximises net value."""
    return float(simulation.loc[simulation["net_value"].idxmax(), "fraction"])
