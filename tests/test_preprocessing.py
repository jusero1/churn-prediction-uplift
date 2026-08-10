"""Tests for preprocessing helpers that do not require the downloaded data."""

from churn_uplift.data.preprocessing import _safe_name, stratified_split


def test_safe_name_strips_special_characters():
    assert _safe_name("2) $100 - $200") == "2_100_200"
    assert _safe_name("channel_Multichannel") == "channel_Multichannel"


def test_stratified_split_preserves_rate_and_size(uplift_frame):
    train, test = stratified_split(
        uplift_frame, stratify_cols=["treatment", "outcome"], test_size=0.25
    )
    assert len(train) + len(test) == len(uplift_frame)
    base = uplift_frame["outcome"].mean()
    assert abs(test["outcome"].mean() - base) < 0.05
