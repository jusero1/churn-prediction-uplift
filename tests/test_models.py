"""Tests for the uplift T-learner and the survival models."""

import numpy as np
import pytest

from churn_uplift.models import UpliftTLearner, cox_model, hazard_ratios, kaplan_meier


def test_uplift_tlearner_ranks_persuadables_higher(uplift_frame):
    X = uplift_frame[["x0", "x1"]]
    model = UpliftTLearner(n_estimators=150).fit(
        X, uplift_frame["treatment"], uplift_frame["outcome"]
    )
    uplift = model.predict_uplift(X)
    # Persuadables (x0>0) must get higher predicted uplift than the rest.
    persuadable = uplift[(uplift_frame["x0"] > 0).to_numpy()].mean()
    other = uplift[(uplift_frame["x0"] <= 0).to_numpy()].mean()
    assert persuadable > other


def test_uplift_predict_before_fit_raises(uplift_frame):
    with pytest.raises(RuntimeError):
        UpliftTLearner().predict_uplift(uplift_frame[["x0", "x1"]])


def test_kaplan_meier_is_monotone_non_increasing(survival_frame):
    kmf = kaplan_meier(
        survival_frame["MonthsInService"].to_numpy(),
        survival_frame["churn"].to_numpy(),
    )
    surv = kmf.survival_function_.iloc[:, 0].to_numpy()
    assert (np.diff(surv) <= 1e-9).all()
    assert surv[0] <= 1.0 and surv[-1] >= 0.0


def test_cox_recovers_risk_direction(survival_frame):
    cox = cox_model(
        survival_frame,
        duration_col="MonthsInService",
        event_col="churn",
        covariates=["risk"],
    )
    hr = hazard_ratios(cox)
    # 'risk' was built to shorten tenure -> hazard ratio above 1.
    assert hr.loc["risk", "hazard_ratio"] > 1.0
