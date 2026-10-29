"""Tests for forecastforge.schemas.validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.core.errors import DataValidationError
from forecastforge.schemas.validation import Gap, find_gaps, validate_frame


def _frame(periods: int = 5, *, freq: str = "D") -> pd.DataFrame:
    """Build a minimal valid frame with a 'close' target."""
    index = pd.date_range("2024-01-01", periods=periods, freq=freq)
    return pd.DataFrame({"close": np.linspace(100.0, 104.0, periods)}, index=index)


# ---------------------------------------------------------------------------
# validate_frame — happy path
# ---------------------------------------------------------------------------


def test_valid_frame_passes() -> None:
    validate_frame(_frame(), target="close")


def test_valid_frame_with_covariates_passes() -> None:
    frame = _frame()
    frame["volume"] = np.arange(len(frame), dtype="float64")

    validate_frame(frame, target="close", covariates=["volume"])


def test_extra_columns_are_allowed() -> None:
    """Undeclared columns are ignored rather than rejected."""
    frame = _frame()
    frame["open"] = frame["close"]

    validate_frame(frame, target="close")


def test_integer_target_is_accepted() -> None:
    frame = _frame()
    frame["close"] = np.arange(len(frame), dtype="int64")

    validate_frame(frame, target="close")


# ---------------------------------------------------------------------------
# validate_frame — structural rejections
# ---------------------------------------------------------------------------


def test_rejects_non_dataframe() -> None:
    with pytest.raises(DataValidationError, match="expected a pandas DataFrame"):
        validate_frame({"close": [1.0, 2.0]}, target="close")  # type: ignore[arg-type]


def test_rejects_empty_frame() -> None:
    empty = pd.DataFrame({"close": pd.Series(dtype="float64")})

    with pytest.raises(DataValidationError, match="time series is empty"):
        validate_frame(empty, target="close")


def test_rejects_non_datetime_index() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.Index([1, 2]))

    with pytest.raises(DataValidationError, match="must be a pandas DatetimeIndex"):
        validate_frame(frame, target="close")


def test_rejects_nat_in_index() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-01", None]))

    with pytest.raises(DataValidationError, match="NaT"):
        validate_frame(frame, target="close")


def test_rejects_duplicate_timestamps() -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-02"])
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        validate_frame(frame, target="close")


def test_duplicate_error_names_the_offending_timestamp() -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-02"])
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)

    with pytest.raises(DataValidationError, match="2024-01-02"):
        validate_frame(frame, target="close")


def test_rejects_unsorted_index() -> None:
    frame = _frame().iloc[::-1]

    with pytest.raises(DataValidationError, match="not sorted in increasing order"):
        validate_frame(frame, target="close")


# ---------------------------------------------------------------------------
# validate_frame — column contract
# ---------------------------------------------------------------------------


def test_rejects_missing_target_and_lists_columns() -> None:
    with pytest.raises(DataValidationError, match="target column 'price' is missing"):
        validate_frame(_frame(), target="price")


def test_missing_target_error_lists_available_columns() -> None:
    with pytest.raises(DataValidationError, match="'close'"):
        validate_frame(_frame(), target="price")


def test_rejects_missing_covariate() -> None:
    with pytest.raises(DataValidationError, match=r"covariate column\(s\) \['volume'\]"):
        validate_frame(_frame(), target="close", covariates=["volume"])


def test_rejects_non_numeric_target() -> None:
    frame = _frame()
    frame["close"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_frame(frame, target="close")


def test_rejects_nan_in_target() -> None:
    frame = _frame()
    frame.iloc[2, 0] = np.nan

    with pytest.raises(DataValidationError, match="1 NaN value"):
        validate_frame(frame, target="close")


def test_nan_error_names_the_first_offending_timestamp() -> None:
    frame = _frame()
    frame.iloc[2, 0] = np.nan

    with pytest.raises(DataValidationError, match="2024-01-03"):
        validate_frame(frame, target="close")


def test_rejects_infinite_values() -> None:
    frame = _frame()
    frame.iloc[1, 0] = np.inf

    with pytest.raises(DataValidationError, match="infinite value"):
        validate_frame(frame, target="close")


def test_rejects_nan_in_covariate() -> None:
    frame = _frame()
    frame["volume"] = [1.0, np.nan, 3.0, 4.0, 5.0]

    with pytest.raises(DataValidationError, match="'volume' contains 1 NaN value"):
        validate_frame(frame, target="close", covariates=["volume"])


# ---------------------------------------------------------------------------
# find_gaps
# ---------------------------------------------------------------------------


def test_complete_index_has_no_gaps() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)

    assert find_gaps(index, "B") == ()


def test_single_missing_period_is_one_gap() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)
    broken = index.delete(3)

    gaps = find_gaps(broken, "B")

    assert len(gaps) == 1
    assert gaps[0].missing == 1
    assert gaps[0].start == index[3]
    assert gaps[0].end == index[3]


def test_contiguous_missing_periods_form_one_gap() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)
    broken = index.delete([3, 4, 5])

    gaps = find_gaps(broken, "B")

    assert len(gaps) == 1
    assert gaps[0].missing == 3
    assert gaps[0].start == index[3]
    assert gaps[0].end == index[5]


def test_separated_missing_periods_form_separate_gaps() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)
    broken = index.delete([2, 6])

    gaps = find_gaps(broken, "B")

    assert [gap.missing for gap in gaps] == [1, 1]
    assert gaps[0].start == index[2]
    assert gaps[1].start == index[6]


def test_weekend_is_a_gap_at_daily_frequency() -> None:
    """The same business-day index is incomplete when read as calendar days.

    Six business days from Monday 1 January 2024 reach Monday the 8th, so a
    calendar-day expectation sees Saturday and Sunday as missing.
    """
    index = pd.bdate_range("2024-01-01", periods=6)

    gaps = find_gaps(index, "D")

    assert len(gaps) == 1
    assert gaps[0].missing == 2
    assert gaps[0].start == pd.Timestamp("2024-01-06")
    assert gaps[0].end == pd.Timestamp("2024-01-07")


@pytest.mark.parametrize("periods", [0, 1])
def test_too_short_indexes_have_no_gaps(periods: int) -> None:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")

    assert find_gaps(index, "D") == ()


def test_invalid_frequency_raises_a_clear_error() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)

    with pytest.raises(DataValidationError, match="not a valid pandas offset alias"):
        find_gaps(index, "every-other-tuesday")


def test_gap_renders_human_readable_summary() -> None:
    gap = Gap(
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-04"),
        missing=3,
    )

    assert str(gap) == "2024-01-02..2024-01-04 (3 missing)"


# ---------------------------------------------------------------------------
# validate_frame — completeness policy
# ---------------------------------------------------------------------------


def test_require_complete_rejects_gaps() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)
    frame = pd.DataFrame({"close": np.arange(9, dtype="float64")}, index=index.delete(4))

    with pytest.raises(DataValidationError, match="incomplete at frequency 'B'"):
        validate_frame(frame, target="close", frequency="B", require_complete=True)


def test_require_complete_accepts_a_complete_series() -> None:
    frame = _frame(periods=5, freq="D")

    validate_frame(frame, target="close", frequency="D", require_complete=True)


def test_require_complete_without_frequency_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="needs a frequency"):
        validate_frame(_frame(), target="close", require_complete=True)
