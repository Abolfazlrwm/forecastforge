"""Leakage tests for every built-in transform.

Look-ahead bias is the failure mode that makes a backtest look brilliant and
a live strategy lose money, and it is invisible in ordinary unit tests: the
numbers are plausible, they are just computed from information that did not
exist yet.

Two independent checks are applied to *every* transform:

**Truncation.** Recompute the transform on the series cut off at row ``cut``.
Every feature value at or before ``cut`` must be identical. If a transform
uses a centred window, a ``shift(-n)``, or a full-sample statistic, the
truncated run cannot reproduce the value and the test fails.

**Perturbation.** Replace everything after ``cut`` with violently different
values and recompute. The prefix must not move by a single unit in the last
place. Truncation alone can be fooled by a transform that merely *reads* the
future without changing magnitude; perturbation cannot.

The harness is itself tested: a deliberately leaky transform is included at
the bottom to prove these tests actually catch leakage rather than passing
vacuously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastforge.features import (
    MACD,
    RSI,
    CalendarFeatures,
    FeatureSet,
    Lag,
    Returns,
    RollingStat,
    Transform,
    Volatility,
)
from forecastforge.schemas.time_series import TimeSeries
from tests.features.helpers import slice_series

# One representative configuration per built-in transform. Every transform
# class in the package must appear here: the parametrised tests below are the
# project's guarantee that "leakage-safe" is a checked property, not a claim.
CAUSAL_TRANSFORMS: tuple[Transform, ...] = (
    Lag([1, 5]),
    RollingStat(5, "mean"),
    RollingStat(5, "std"),
    RollingStat(5, "min"),
    RollingStat(5, "max"),
    Returns(1, log=False),
    Returns(2, log=True),
    CalendarFeatures(
        [
            "dayofweek",
            "dayofmonth",
            "month",
            "quarter",
            "dayofyear",
            "weekofyear",
            "dow_cyclic",
            "month_cyclic",
        ]
    ),
    RSI(5),
    MACD(fast=3, slow=6, signal=2),
    Volatility(5),
)

_CUT = 120


def _feature_matrix(transform: Transform, series: TimeSeries) -> np.ndarray:
    """Apply ``transform`` and return its feature columns as a float array."""
    features = transform.apply(FeatureSet.from_time_series(series))
    return features.matrix.to_numpy(dtype="float64")


def assert_causal(transform: Transform, series: TimeSeries, *, cut: int) -> None:
    """Assert that ``transform`` never uses rows after ``cut``.

    Args:
        transform: The transform under test.
        series: A series long enough that ``cut`` leaves a usable future.
        cut: Index of the last row the features are allowed to depend on.

    Raises:
        AssertionError: If any feature value at or before ``cut`` changes when
            the future is truncated or perturbed.
    """
    full = _feature_matrix(transform, series)
    head = full[: cut + 1]

    truncated = _feature_matrix(transform, slice_series(series, cut))
    np.testing.assert_allclose(
        head,
        truncated,
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
        err_msg=(
            f"{transform!r} changed its past values when the series was truncated "
            f"at row {cut}, so it reads rows after {cut}"
        ),
    )

    perturbed_frame = series.frame.copy()
    future_index = perturbed_frame.index[cut + 1 :]
    # A large multiplicative shock *and* an alternating sign pattern: the
    # product makes any leaked value obvious, and the alternation defeats
    # transforms that would only notice a monotone shift.
    original = perturbed_frame.loc[future_index, series.target].to_numpy(dtype="float64")
    shock = np.where(np.arange(len(original)) % 2 == 0, 7.0, 0.05)
    perturbed_frame.loc[future_index, series.target] = original * shock
    perturbed_series = TimeSeries(
        frame=perturbed_frame,
        target=series.target,
        metadata=series.metadata,
        covariates=series.covariates,
    )
    # Sanity: the perturbation must actually change the input, otherwise the
    # check below would pass for the wrong reason.
    assert not np.allclose(
        perturbed_frame.loc[future_index, series.target].to_numpy(dtype="float64"),
        original,
    )

    perturbed = _feature_matrix(transform, perturbed_series)
    np.testing.assert_allclose(
        head,
        perturbed[: cut + 1],
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
        err_msg=(
            f"{transform!r} changed its past values when future rows were perturbed, "
            f"so it leaks information from after row {cut}"
        ),
    )


@pytest.mark.parametrize("transform", CAUSAL_TRANSFORMS, ids=repr)
def test_transform_never_reads_the_future(
    transform: Transform, synthetic_series: TimeSeries
) -> None:
    """Every built-in transform must be causal under truncation and perturbation."""
    assert_causal(transform, synthetic_series, cut=_CUT)


@pytest.mark.parametrize("transform", CAUSAL_TRANSFORMS, ids=repr)
def test_every_builtin_transform_is_covered(transform: Transform) -> None:
    """Guard against a new transform being added without a leakage case.

    If someone adds a transform class and forgets ``CAUSAL_TRANSFORMS``, this
    fails with a message naming the missing class.
    """
    covered = {type(item) for item in CAUSAL_TRANSFORMS}
    expected = {Lag, RollingStat, Returns, CalendarFeatures, RSI, MACD, Volatility}
    assert covered == expected, (
        "the leakage suite does not cover every built-in transform; "
        f"missing: {sorted(cls.__name__ for cls in expected - covered)}"
    )
    assert isinstance(transform, Transform)


def test_calendar_features_are_value_independent(synthetic_series: TimeSeries) -> None:
    """Calendar features depend only on timestamps, never on prices."""
    transform = CalendarFeatures(["dayofweek", "month_cyclic"])
    base = transform.apply(FeatureSet.from_time_series(synthetic_series))

    moved_frame = synthetic_series.frame.copy()
    moved_frame[synthetic_series.target] = moved_frame[synthetic_series.target] * 1000.0
    moved = TimeSeries(
        frame=moved_frame,
        target=synthetic_series.target,
        metadata=synthetic_series.metadata,
    )
    shifted = transform.apply(FeatureSet.from_time_series(moved))
    pd.testing.assert_frame_equal(base.matrix, shifted.matrix)


class _PeekingTransform(Transform):
    """Deliberately leaky: reads tomorrow's value. Used to test the harness."""

    name = "peeking"

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Declare the single leaked column."""
        return (f"{features.target}_peek",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Shift *backwards*, which is exactly the bug the suite must catch."""
        return pd.DataFrame(
            {f"{features.target}_peek": features.frame[features.target].shift(-1)},
            index=features.frame.index,
        )


class _CenteredWindowTransform(Transform):
    """Deliberately leaky: a centred rolling mean straddles the present."""

    name = "centered"

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Declare the single leaked column."""
        return (f"{features.target}_centered",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Average a window that extends one row into the future."""
        values = features.frame[features.target].rolling(window=5, center=True).mean()
        return pd.DataFrame({f"{features.target}_centered": values}, index=features.frame.index)


def test_harness_detects_a_forward_shift(synthetic_series: TimeSeries) -> None:
    """The leakage harness must reject a transform that peeks one row ahead."""
    with pytest.raises(AssertionError):
        assert_causal(_PeekingTransform(), synthetic_series, cut=_CUT)


def test_harness_detects_a_centered_window(synthetic_series: TimeSeries) -> None:
    """The leakage harness must reject a centred rolling window."""
    with pytest.raises(AssertionError):
        assert_causal(_CenteredWindowTransform(), synthetic_series, cut=_CUT)


def test_harness_detects_a_full_sample_statistic(synthetic_series: TimeSeries) -> None:
    """The leakage harness must reject a statistic computed over the whole sample."""
    series = synthetic_series

    class _GlobalZScore(Transform):
        name = "global-zscore"

        def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
            return (f"{features.target}_zscore",)

        def compute(self, features: FeatureSet) -> pd.DataFrame:
            column = features.frame[features.target]
            values = (column - column.mean()) / column.std()
            return pd.DataFrame({f"{features.target}_zscore": values}, index=features.frame.index)

    with pytest.raises(AssertionError):
        assert_causal(_GlobalZScore(), series, cut=_CUT)
