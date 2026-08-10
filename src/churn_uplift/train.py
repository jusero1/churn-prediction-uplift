"""Reproducible pipeline: churn propensity + survival, and uplift + campaign.

Rebuilds every artefact the report and notebooks use, under ``reports/`` and
``models/``. Target behind ``make train``.

Usage
-----
    python -m churn_uplift.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

from churn_uplift.data import (
    CHURN_TARGET,
    DURATION_COL,
    cell2cell_xy,
    hillstrom_parts,
    load_cell2cell,
    load_hillstrom,
    stratified_split,
)
from churn_uplift.evaluation import (
    average_treatment_effect,
    best_fraction,
    campaign_simulation,
    metric_summary,
    plots,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
)
from churn_uplift.models import (
    UpliftTLearner,
    churn_proba,
    cox_model,
    hazard_ratios,
    kaplan_meier,
    kaplan_meier_by_group,
    train_propensity,
)

RANDOM_STATE = 42
COX_COVARIATES = [
    "MonthlyRevenue",
    "MonthlyMinutes",
    "CustomerCareCalls",
    "DroppedCalls",
    "OverageMinutes",
]
# Illustrative retention-campaign economics (scenario inputs, not data figures).
CONTACT_COST = 0.40
RESPONSE_VALUE = 5.0


def _dirs() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    reports, figures, models = (
        root / "reports",
        root / "reports" / "figures",
        root / "models",
    )
    for d in (reports, figures, models):
        d.mkdir(parents=True, exist_ok=True)
    return reports, figures, models


def _churn_component(reports, figures, models) -> dict:
    df = load_cell2cell()
    train, test = stratified_split(df, stratify_cols=[CHURN_TARGET])
    x_train, y_train = cell2cell_xy(train)
    x_test, y_test = cell2cell_xy(test)

    model = train_propensity(x_train, y_train, random_state=RANDOM_STATE)
    p_test = churn_proba(model, x_test)
    metrics = {k: round(v, 4) for k, v in metric_summary(y_test, p_test).items()}
    plots.plot_propensity_roc(y_test, p_test).savefig(
        figures / "01_propensity_roc.png", dpi=120
    )

    # Survival: overall and split on median monthly minutes (usage intensity).
    durations = df[DURATION_COL].to_numpy()
    events = df[CHURN_TARGET].to_numpy()
    heavy = (df["MonthlyMinutes"] > df["MonthlyMinutes"].median()).map(
        {True: "heavy usage", False: "light usage"}
    )
    km = kaplan_meier_by_group(durations, events, heavy.to_numpy())
    plots.plot_km(km).savefig(figures / "02_kaplan_meier.png", dpi=120)

    cox = cox_model(
        df, duration_col=DURATION_COL, event_col=CHURN_TARGET, covariates=COX_COVARIATES
    )
    hr = hazard_ratios(cox)
    hr.to_csv(reports / "cox_hazard_ratios.csv")

    joblib.dump(model, models / "propensity.joblib")
    return {
        "propensity_test": metrics,
        "median_survival_months": float(
            kaplan_meier(durations, events).median_survival_time_
        ),
        "cox_top_driver": hr.index[0],
    }


def _uplift_component(reports, figures, models) -> dict:
    df = load_hillstrom()  # Men's e-mail vs control (canonical uplift benchmark)
    train, test = stratified_split(
        df, stratify_cols=["treatment", "outcome"], random_state=RANDOM_STATE
    )
    x_train, t_train, y_train = hillstrom_parts(train)
    x_test, t_test, y_test = hillstrom_parts(test)

    model = UpliftTLearner(random_state=RANDOM_STATE).fit(x_train, t_train, y_train)
    uplift = model.predict_uplift(x_test)

    ate = average_treatment_effect(df["treatment"], df["outcome"])
    qini = qini_coefficient(uplift, t_test, y_test)
    plots.plot_qini(qini_curve(uplift, t_test, y_test)).savefig(
        figures / "03_qini.png", dpi=120
    )
    decile = uplift_by_decile(uplift, t_test, y_test)
    decile.to_csv(reports / "uplift_by_decile.csv", index=False)
    plots.plot_uplift_by_decile(decile).savefig(
        figures / "04_uplift_by_decile.png", dpi=120
    )

    sim = campaign_simulation(
        uplift,
        t_test,
        y_test,
        cost_per_contact=CONTACT_COST,
        value_per_response=RESPONSE_VALUE,
    )
    sim.to_csv(reports / "campaign_simulation.csv", index=False)
    plots.plot_campaign(sim).savefig(figures / "05_campaign.png", dpi=120)

    joblib.dump(model, models / "uplift.joblib")
    top = decile.loc[decile["decile"] == 1, "observed_uplift"].iloc[0]
    return {
        "ate_visit": round(ate, 4),
        "qini_coefficient": round(qini, 5),
        "top_decile_uplift": float(top),
        "best_targeting_fraction": best_fraction(sim),
    }


def main() -> dict:
    matplotlib.use("Agg")
    reports, figures, models = _dirs()

    churn = _churn_component(reports, figures, models)
    uplift = _uplift_component(reports, figures, models)
    metrics = {"churn": churn, "uplift": uplift}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== churn-prediction-uplift ===")
    print(
        f"  Propensity (test): AUC {churn['propensity_test']['auc']} | "
        f"KS {churn['propensity_test']['ks']}"
    )
    print(f"  Median survival: {churn['median_survival_months']:.0f} months")
    print(f"  Top churn hazard driver: {churn['cox_top_driver']}")
    print(
        f"  Uplift ATE (visit): {uplift['ate_visit']:.3f} | "
        f"Qini {uplift['qini_coefficient']} | "
        f"top-decile uplift {uplift['top_decile_uplift']:.3f}"
    )
    print("  artefacts written to reports/ and models/")
    return metrics


if __name__ == "__main__":
    main()
