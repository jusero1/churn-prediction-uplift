"""Churn propensity model (LightGBM) on the Cell2Cell features.

Answers "who is likely to churn?" — the ranking a retention team needs before it
decides who to act on. Discrimination is reported with the shared riskkit metrics
so it is comparable across the portfolio.
"""

from __future__ import annotations

import pandas as pd


def train_propensity(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 400,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    random_state: int = 42,
):
    """Train a LightGBM churn classifier.

    Categorical columns (pandas ``category`` dtype) are passed natively. Class
    imbalance is handled with ``scale_pos_weight``.

    Returns
    -------
    lightgbm.LGBMClassifier
        The fitted classifier.
    """
    from lightgbm import LGBMClassifier

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=neg / pos if pos else 1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def churn_proba(model, X: pd.DataFrame) -> pd.Series:
    """Predicted churn probability from a fitted propensity model."""
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="p_churn")
