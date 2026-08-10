"""Load and clean the two real datasets.

Event and window definitions (the part most churn portfolios skip):

- **Cell2Cell** — the event is contractual churn (``Churn = 'Yes'``) observed at
  the data-collection snapshot. For survival, the duration is ``MonthsInService``
  (tenure) and churners are events while still-active customers are right
  *censored* at their current tenure. This is a snapshot-with-tenure
  approximation, not a longitudinal cohort; the limitation is documented in the
  model card.
- **Hillstrom** — a randomized experiment. The treatment is receiving a
  marketing e-mail (men's or women's) versus control (no e-mail); the outcome is
  a site ``visit`` within the two weeks after the send.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHURN_TARGET = "churn"
DURATION_COL = "MonthsInService"

# High-cardinality / identifier columns dropped from the churn feature set.
_CELL2CELL_DROP = ["CustomerID", "ServiceArea"]


def _raw_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "raw"


# --------------------------------------------------------------------------
# Cell2Cell — churn propensity and survival
# --------------------------------------------------------------------------
def load_cell2cell(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the Cell2Cell churn training table.

    Returns
    -------
    pandas.DataFrame
        Cleaned frame with a binary ``churn`` column (1 = churned), numeric NaNs
        median-filled and object columns cast to ``category`` for LightGBM.
    """
    if path is None:
        path = _raw_dir() / "cell2celltrain.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m churn_uplift.data.download` first."
        )
    df = pd.read_csv(path)

    df[CHURN_TARGET] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
    df = df.drop(columns=[c for c in ["Churn", *_CELL2CELL_DROP] if c in df.columns])

    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].fillna(df[numeric].median())

    obj = df.select_dtypes(include="object").columns
    for col in obj:
        df[col] = df[col].fillna("Unknown").astype("category")

    return df


def cell2cell_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned churn frame into features and target."""
    y = df[CHURN_TARGET]
    x = df.drop(columns=[CHURN_TARGET])
    return x, y


# --------------------------------------------------------------------------
# Hillstrom — uplift
# --------------------------------------------------------------------------
_HILLSTROM_CATEGORICAL = ["history_segment", "zip_code", "channel"]
_HILLSTROM_NUMERIC = ["recency", "history", "mens", "womens", "newbie"]


def load_hillstrom(
    path: str | Path | None = None,
    outcome: str = "visit",
    treatment_arm: str | None = "Mens E-Mail",
) -> pd.DataFrame:
    """Load the Hillstrom experiment and derive treatment and outcome columns.

    Parameters
    ----------
    outcome : {"visit", "conversion"}, default="visit"
        The response to model. ``visit`` has far more signal (~15%) than the
        rare ``conversion`` (~1%).
    treatment_arm : str or None, default="Mens E-Mail"
        Which e-mail arm to treat as the treatment against the ``No E-Mail``
        control. The men's-vs-control contrast is the canonical uplift benchmark
        (the clearest heterogeneous effect); pass ``None`` to pool both e-mail
        arms as a single treatment (a weaker, more diluted signal).

    Returns
    -------
    pandas.DataFrame
        Frame with one-hot features plus ``treatment`` (1 = e-mail) and
        ``outcome``.
    """
    if path is None:
        path = _raw_dir() / "hillstrom.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m churn_uplift.data.download` first."
        )
    df = pd.read_csv(path)

    if treatment_arm is not None:
        df = df[df["segment"].isin([treatment_arm, "No E-Mail"])].reset_index(drop=True)
        df["treatment"] = (df["segment"] == treatment_arm).astype(int)
    else:
        df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    df["outcome"] = df[outcome].astype(int)

    features = pd.get_dummies(
        df[_HILLSTROM_NUMERIC + _HILLSTROM_CATEGORICAL],
        columns=_HILLSTROM_CATEGORICAL,
        drop_first=True,
    ).astype(float)
    # One-hot names carry '$', ')', spaces (e.g. "2) $100 - $200") that LightGBM
    # rejects; normalise to safe identifiers.
    features.columns = [_safe_name(c) for c in features.columns]

    return pd.concat(
        [features, df[["treatment", "outcome"]].reset_index(drop=True)], axis=1
    )


def _safe_name(name: str) -> str:
    """Reduce a column name to a LightGBM-safe identifier."""
    import re

    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")


def hillstrom_parts(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, treatment, outcome) from a loaded Hillstrom frame."""
    treatment = df["treatment"]
    outcome = df["outcome"]
    features = df.drop(columns=["treatment", "outcome"])
    return features, treatment, outcome


def stratified_split(
    df: pd.DataFrame,
    *,
    stratify_cols: list[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded split stratified on one or more columns (joined into a key)."""
    from sklearn.model_selection import train_test_split

    key = df[stratify_cols].astype(str).agg("|".join, axis=1)
    train, test = train_test_split(
        df, test_size=test_size, stratify=key, random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def churn_event_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (durations, events) for survival: tenure and the churn flag."""
    return df[DURATION_COL].to_numpy(), df[CHURN_TARGET].to_numpy()
