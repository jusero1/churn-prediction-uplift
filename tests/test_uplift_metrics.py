"""Tests for the uplift evaluation metrics against a known-effect fixture."""

import numpy as np

from churn_uplift.evaluation import (
    average_treatment_effect,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
)


def test_ate_recovers_the_planted_effect(uplift_frame):
    ate = average_treatment_effect(uplift_frame["treatment"], uplift_frame["outcome"])
    # Half the population (x0>0) gets +0.20, so the ATE is around +0.10.
    assert 0.06 < ate < 0.14


def test_qini_curve_spans_origin_to_one(uplift_frame):
    # An oracle uplift score = true persuadability indicator.
    oracle = (uplift_frame["x0"] > 0).astype(float).to_numpy()
    curve = qini_curve(oracle, uplift_frame["treatment"], uplift_frame["outcome"])
    assert curve["fraction"].iloc[0] == 0.0
    assert curve["qini"].iloc[0] == 0.0
    assert curve["fraction"].iloc[-1] == 1.0


def test_qini_coefficient_positive_for_informative_ranking(uplift_frame):
    oracle = uplift_frame["x0"].to_numpy()  # higher x0 -> more persuadable
    random_score = np.random.default_rng(0).normal(size=len(uplift_frame))
    q_model = qini_coefficient(
        oracle, uplift_frame["treatment"], uplift_frame["outcome"]
    )
    q_random = qini_coefficient(
        random_score, uplift_frame["treatment"], uplift_frame["outcome"]
    )
    assert q_model > q_random
    assert q_model > 0


def test_uplift_by_decile_top_beats_bottom(uplift_frame):
    oracle = uplift_frame["x0"].to_numpy()
    table = uplift_by_decile(oracle, uplift_frame["treatment"], uplift_frame["outcome"])
    assert len(table) == 10
    top = table.loc[table["decile"] == 1, "observed_uplift"].iloc[0]
    bottom = table.loc[table["decile"] == 10, "observed_uplift"].iloc[0]
    assert top > bottom
