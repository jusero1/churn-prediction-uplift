"""Tests for the campaign simulation."""

import numpy as np

from churn_uplift.evaluation import best_fraction, campaign_simulation


def test_campaign_columns_and_arithmetic(uplift_frame):
    oracle = uplift_frame["x0"].to_numpy()
    sim = campaign_simulation(
        oracle,
        uplift_frame["treatment"],
        uplift_frame["outcome"],
        cost_per_contact=0.5,
        value_per_response=5.0,
    )
    assert {"fraction", "n_targeted", "net_value", "roi"} <= set(sim.columns)
    # net_value = value - cost must be internally consistent.
    recomputed = sim["value"] - sim["cost"]
    assert np.allclose(sim["net_value"], recomputed, atol=1e-6)


def test_targeting_concentrates_uplift_at_the_top(uplift_frame):
    oracle = uplift_frame["x0"].to_numpy()
    sim = campaign_simulation(
        oracle,
        uplift_frame["treatment"],
        uplift_frame["outcome"],
        cost_per_contact=0.5,
        value_per_response=5.0,
        fractions=(0.1, 1.0),
    )
    # The top-10% by uplift has a higher incremental rate than the whole base.
    top = sim.loc[sim["fraction"] == 0.1, "incremental_rate"].iloc[0]
    everyone = sim.loc[sim["fraction"] == 1.0, "incremental_rate"].iloc[0]
    assert top > everyone


def test_best_fraction_is_argmax_of_net_value(uplift_frame):
    oracle = uplift_frame["x0"].to_numpy()
    sim = campaign_simulation(
        oracle,
        uplift_frame["treatment"],
        uplift_frame["outcome"],
        cost_per_contact=0.9,
        value_per_response=2.0,
    )
    frac = best_fraction(sim)
    assert frac == sim.loc[sim["net_value"].idxmax(), "fraction"]
