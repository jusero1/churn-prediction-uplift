"""Uplift modeling: estimate the *incremental* effect of contacting a customer.

Propensity ranks who will act; uplift ranks who will act *because* of the
treatment. The distinction matters for a retention campaign: a customer who
churns no matter what (a "lost cause"), or stays no matter what (a "sure thing"),
is wasted contact. Only the *persuadables* — positive uplift — are worth the
cost.

This is a **T-learner** (two-model): fit one outcome model on the treated group
and one on the control group, and take the difference of their predicted
probabilities as the estimated uplift. It needs randomized treatment/control
data, which the Hillstrom experiment provides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class UpliftTLearner:
    """Two-model (T-learner) uplift estimator.

    Parameters
    ----------
    n_estimators, learning_rate, num_leaves : LightGBM hyper-parameters.
    random_state : int
        Seed for both base models.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        random_state: int = 42,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_treated_ = None
        self.model_control_ = None

    def fit(
        self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series
    ) -> UpliftTLearner:
        """Fit the treated-group and control-group outcome models."""
        from lightgbm import LGBMClassifier

        treatment = np.asarray(treatment)
        self.model_treated_ = LGBMClassifier(**self.params).fit(
            X[treatment == 1], np.asarray(y)[treatment == 1]
        )
        self.model_control_ = LGBMClassifier(**self.params).fit(
            X[treatment == 0], np.asarray(y)[treatment == 0]
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Estimated uplift = P(outcome | treat) - P(outcome | control)."""
        if self.model_treated_ is None or self.model_control_ is None:
            raise RuntimeError("UpliftTLearner must be fitted before predicting")
        p_treated = self.model_treated_.predict_proba(X)[:, 1]
        p_control = self.model_control_.predict_proba(X)[:, 1]
        return p_treated - p_control
