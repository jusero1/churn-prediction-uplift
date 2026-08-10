"""churn_uplift: churn propensity, survival analysis and uplift modeling.

Two real datasets, two components:

- Cell2Cell telecom churn -> propensity (:mod:`churn_uplift.models.propensity`)
  and survival (:mod:`churn_uplift.models.survival`).
- Hillstrom randomized e-mail experiment -> uplift
  (:mod:`churn_uplift.models.uplift`) and the campaign simulation
  (:mod:`churn_uplift.evaluation.campaign`).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("churn-prediction-uplift")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
