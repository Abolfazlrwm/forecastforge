"""The built-in, leakage-safe feature transforms.

Every transform here is causal: the value at row *t* is computed from rows
``<= t`` only. That property is verified by ``tests/features/test_leakage.py``
for each class, using both a truncation test (recompute on a prefix) and a
perturbation test (scramble the future and confirm the past does not move).

The set is deliberately small. It covers the features that actually matter
for daily price series — memory, trend, dispersion, momentum, and seasonality
— and nothing that would smuggle future information in. Levels (raw price) are
not emitted as features: they are non-stationary and a model that receives
them will happily learn a spurious trend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataValidationError
from forecastforge.features.frame import FeatureSet
from forecastforge.features.transform import Transform

__all__ = [
    "MACD",
    "RSI",
    "CalendarFeatures",
    "Lag",
    "Returns",
    "RollingStat",
    "Volatility",
]

RollingStatistic = Literal["mean", "std", "min", "max"]
CalendarComponent = Literal[
    "dayofweek",
    "dayofmonth",
    "month",
    "quarter",
    "dayofyear",
    "weekofyear",
    "dow_cyclic",
    "month_cyclic",
]


def _resolve_column(features: FeatureSet, column: str | None) -> str:
    """Return ``column``, defaulting to the target, and check it exists.

    Args:
        features: The set whose columns are searched.
        column: Explicit column name, or ``None`` to use the target.

    Returns:
        The resolved column name.

    Raises:
        DataValidationError: If an explicit ``column`` is not present.
    """
    if column is None:
        return features.target
    if column not in features.frame.columns:
        available = ", ".join(repr(str(name)) for name in features.frame.columns[:6])
        raise DataValidationError(
            f"column {column!r} is not present; available columns: {available}"
        )
    return column


def _require_positive(series: pd.Series, *, transform: str) -> None:
    """Reject non-positive values, which a log transform cannot represent.

    Args:
        series: The input column.
        transform: Transform name, used in the error message.

    Raises:
        DataValidationError: If any value is zero or negative.
    """
    if (series <= 0).any():
        count = int((series <= 0).sum())
        raise DataValidationError(
            f"{transform} requires strictly positive values, found {count} value(s) <= 0"
        )


class Lag(Transform):
    """Add lagged copies of a column — the series' own memory.

    Args:
        periods: One or more positive look-back lengths, in rows. Duplicates
            are collapsed and the result is sorted ascending.
        column: Column to lag. Defaults to the target.

    Raises:
        DataValidationError: If ``periods`` is empty or contains a
            non-positive value.

    Note:
        Row *t* receives the value from row ``t - p``, so the first ``p`` rows
        are NaN. This is the canonical warm-up: it is unavoidable and it is
        reported by :attr:`FeatureSet.warmup`.
    """

    name: ClassVar[str] = "lag"

    def __init__(self, periods: int | Sequence[int], column: str | None = None) -> None:
        values = [periods] if isinstance(periods, int) else list(periods)
        if not values:
            raise DataValidationError("Lag needs at least one period")
        if any(period <= 0 for period in values):
            raise DataValidationError(f"lag periods must be positive, got {values}")
        self.periods: tuple[int, ...] = tuple(sorted(set(values)))
        self.column = column

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return one ``{column}_lag{p}`` name per requested period."""
        column = _resolve_column(features, self.column)
        return tuple(f"{column}_lag{period}" for period in self.periods)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Shift the column forward, producing NaN for the warm-up rows."""
        column = _resolve_column(features, self.column)
        series = features.frame[column]
        return pd.DataFrame(
            {f"{column}_lag{period}": series.shift(period) for period in self.periods},
            index=features.frame.index,
        )

    def _describe(self) -> str:
        return f"periods={self.periods}, column={self.column!r}"


class RollingStat(Transform):
    """Add a trailing rolling statistic of a column.

    The window ends at the current row, so the value at *t* summarises
    ``(t - window + 1) .. t`` and never looks ahead.

    Args:
        window: Number of rows in the window.
        stat: One of ``"mean"``, ``"std"``, ``"min"``, ``"max"``.
        min_periods: Observations required before a value is emitted.
            Defaults to ``window``, which makes the warm-up region explicit
            instead of letting a half-filled window leak a misleading value.
        column: Column to summarise. Defaults to the target.

    Raises:
        DataValidationError: If ``window`` is not positive, ``stat`` is
            unknown, or ``min_periods`` is outside ``1..window``.
    """

    name: ClassVar[str] = "rolling"
    _STATS: ClassVar[frozenset[str]] = frozenset({"mean", "std", "min", "max"})

    def __init__(
        self,
        window: int,
        stat: RollingStatistic = "mean",
        min_periods: int | None = None,
        column: str | None = None,
    ) -> None:
        if window <= 0:
            raise DataValidationError(f"window must be positive, got {window}")
        if stat not in self._STATS:
            raise DataValidationError(
                f"unknown rolling statistic {stat!r}; valid: {sorted(self._STATS)}"
            )
        resolved_min = window if min_periods is None else min_periods
        if not 1 <= resolved_min <= window:
            raise DataValidationError(
                f"min_periods must be between 1 and {window}, got {resolved_min}"
            )
        self.window = window
        self.stat = stat
        self.min_periods = resolved_min
        self.column = column

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return the single ``{column}_rolling_{stat}_{window}`` name."""
        column = _resolve_column(features, self.column)
        return (f"{column}_rolling_{self.stat}_{self.window}",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Apply the trailing rolling statistic."""
        column = _resolve_column(features, self.column)
        rolling = features.frame[column].rolling(window=self.window, min_periods=self.min_periods)
        match self.stat:
            case "mean":
                values = rolling.mean()
            case "std":
                values = rolling.std()
            case "min":
                values = rolling.min()
            case "max":
                values = rolling.max()
        return pd.DataFrame({self.output_columns(features)[0]: values}, index=features.frame.index)

    def _describe(self) -> str:
        return (
            f"window={self.window}, stat={self.stat!r}, "
            f"min_periods={self.min_periods}, column={self.column!r}"
        )


class Returns(Transform):
    """Add simple or log returns of a column.

    Returns are scale-free, which makes them comparable across assets and
    across time — a property raw prices do not have.

    Args:
        periods: Look-back distance in rows. ``1`` is the one-day return.
        log: When true, emit ``log(x[t]) - log(x[t - periods])``; otherwise
            emit ``x[t] / x[t - periods] - 1``.
        column: Column to differentiate. Defaults to the target.

    Raises:
        DataValidationError: If ``periods`` is not positive, or a log return
            is requested for a column containing non-positive values.
    """

    name: ClassVar[str] = "returns"

    def __init__(
        self,
        periods: int = 1,
        log: bool = False,
        column: str | None = None,
    ) -> None:
        if periods <= 0:
            raise DataValidationError(f"periods must be positive, got {periods}")
        self.periods = periods
        self.log = log
        self.column = column

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return ``{column}_logret{p}`` or ``{column}_ret{p}``."""
        column = _resolve_column(features, self.column)
        prefix = "logret" if self.log else "ret"
        return (f"{column}_{prefix}{self.periods}",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Differentiate the column, leaving the warm-up rows as NaN."""
        column = _resolve_column(features, self.column)
        series = features.frame[column]
        if self.log:
            _require_positive(series, transform=type(self).__name__)
            values = np.log(series).diff(self.periods)
        else:
            # fill_method=None keeps pandas from padding across gaps, which
            # would silently invent a return that never happened.
            values = series.pct_change(periods=self.periods, fill_method=None)
        return pd.DataFrame({self.output_columns(features)[0]: values}, index=features.frame.index)

    def _describe(self) -> str:
        return f"periods={self.periods}, log={self.log}, column={self.column!r}"


class CalendarFeatures(Transform):
    """Add calendar features derived from the timestamp index.

    Seasonality in financial series is real (day-of-week effects, month-end
    flows), and calendar columns are the cheapest way to let a model see it.
    These depend only on the timestamp itself, so they are causal by
    construction.

    Args:
        components: Which calendar components to emit. Supported values are
            ``dayofweek``, ``dayofmonth``, ``month``, ``quarter``,
            ``dayofyear``, ``weekofyear``, ``dow_cyclic`` and
            ``month_cyclic``. The two ``*_cyclic`` options emit sine/cosine
            pairs, which preserve the wrap-around distance that a raw integer
            (December = 12, January = 1) destroys.

    Raises:
        DataValidationError: If ``components`` is empty or names an unknown
            component.
    """

    name: ClassVar[str] = "calendar"
    _VALID: ClassVar[frozenset[str]] = frozenset(
        {
            "dayofweek",
            "dayofmonth",
            "month",
            "quarter",
            "dayofyear",
            "weekofyear",
            "dow_cyclic",
            "month_cyclic",
        }
    )

    def __init__(self, components: Sequence[CalendarComponent]) -> None:
        values = tuple(components)
        if not values:
            raise DataValidationError("CalendarFeatures needs at least one component")
        invalid = sorted(set(values) - self._VALID)
        if invalid:
            raise DataValidationError(
                f"unknown calendar component(s) {invalid}; valid: {sorted(self._VALID)}"
            )
        self.components = values

    def _names(self, component: str) -> tuple[str, ...]:
        """Return the output column names for one component."""
        if component == "dow_cyclic":
            return ("dt_dow_sin", "dt_dow_cos")
        if component == "month_cyclic":
            return ("dt_month_sin", "dt_month_cos")
        return (f"dt_{component}",)

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return every column this transform will emit, in order."""
        names: list[str] = []
        for component in self.components:
            names.extend(self._names(component))
        return tuple(names)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Extract the requested components from the index."""
        index = features.index
        columns: dict[str, pd.Series] = {}
        for component in self.components:
            if component == "dayofweek":
                columns["dt_dayofweek"] = pd.Series(index.dayofweek, index=index)
            elif component == "dayofmonth":
                columns["dt_dayofmonth"] = pd.Series(index.day, index=index)
            elif component == "month":
                columns["dt_month"] = pd.Series(index.month, index=index)
            elif component == "quarter":
                columns["dt_quarter"] = pd.Series(index.quarter, index=index)
            elif component == "dayofyear":
                columns["dt_dayofyear"] = pd.Series(index.dayofyear, index=index)
            elif component == "weekofyear":
                columns["dt_weekofyear"] = pd.Series(
                    index.isocalendar().week.to_numpy(), index=index
                )
            elif component == "dow_cyclic":
                angle = 2.0 * np.pi * index.dayofweek.to_numpy() / 7.0
                columns["dt_dow_sin"] = pd.Series(np.sin(angle), index=index)
                columns["dt_dow_cos"] = pd.Series(np.cos(angle), index=index)
            else:  # month_cyclic
                angle = 2.0 * np.pi * (index.month.to_numpy() - 1) / 12.0
                columns["dt_month_sin"] = pd.Series(np.sin(angle), index=index)
                columns["dt_month_cos"] = pd.Series(np.cos(angle), index=index)
        return pd.DataFrame(columns, index=index)

    def _describe(self) -> str:
        return f"components={self.components}"


class RSI(Transform):
    """Add Wilder's Relative Strength Index.

    RSI maps recent gains against recent losses onto ``0..100`` and is the
    standard overbought/oversold indicator.

    Args:
        window: Smoothing period. The conventional default is 14.
        column: Column to measure. Defaults to the target.

    Raises:
        DataValidationError: If ``window`` is not positive.

    Note:
        Wilder's original recursion is expressed as an exponentially weighted
        mean with ``alpha = 1 / window`` and ``adjust=False``. The recursion
        runs forward in time only, which is what keeps the feature causal.
        When the window contains no movement at all the ratio is undefined;
        those rows report the neutral value ``50``.
    """

    name: ClassVar[str] = "rsi"

    def __init__(self, window: int = 14, column: str | None = None) -> None:
        if window <= 0:
            raise DataValidationError(f"window must be positive, got {window}")
        self.window = window
        self.column = column

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return the single ``{column}_rsi{window}`` name."""
        column = _resolve_column(features, self.column)
        return (f"{column}_rsi{self.window}",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Compute RSI from one-step price changes."""
        column = _resolve_column(features, self.column)
        delta = features.frame[column].diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)

        alpha = 1.0 / self.window
        avg_gain = gains.ewm(alpha=alpha, adjust=False, min_periods=self.window).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False, min_periods=self.window).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = avg_gain / avg_loss
            values = 100.0 - (100.0 / (1.0 + ratio))

        # A window with no gains and no losses has no defined ratio; report
        # the neutral midpoint instead of a NaN in the middle of the series.
        flat = (avg_gain == 0.0) & (avg_loss == 0.0)
        values = values.mask(flat, 50.0)
        return pd.DataFrame({self.output_columns(features)[0]: values}, index=features.frame.index)

    def _describe(self) -> str:
        return f"window={self.window}, column={self.column!r}"


class MACD(Transform):
    """Add the Moving Average Convergence Divergence indicator.

    MACD compares a fast and a slow exponential moving average; the signal
    line smooths MACD itself, and the histogram is their difference.

    Args:
        fast: Span of the fast EMA.
        slow: Span of the slow EMA. Must exceed ``fast``.
        signal: Span of the signal EMA.
        column: Column to measure. Defaults to the target.

    Raises:
        DataValidationError: If a span is not positive or ``slow <= fast``.
    """

    name: ClassVar[str] = "macd"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str | None = None,
    ) -> None:
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise DataValidationError(
                f"MACD spans must be positive, got fast={fast}, slow={slow}, signal={signal}"
            )
        if slow <= fast:
            raise DataValidationError(
                f"slow span must exceed fast span, got fast={fast}, slow={slow}"
            )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.column = column

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return the MACD, signal, and histogram column names."""
        column = _resolve_column(features, self.column)
        return (
            f"{column}_macd",
            f"{column}_macd_signal",
            f"{column}_macd_hist",
        )

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Compute the MACD line, its signal line, and the histogram."""
        column = _resolve_column(features, self.column)
        series = features.frame[column]
        fast_ema = series.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        slow_ema = series.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()

        macd_name, signal_name, hist_name = self.output_columns(features)
        return pd.DataFrame(
            {macd_name: macd, signal_name: signal, hist_name: macd - signal},
            index=features.frame.index,
        )

    def _describe(self) -> str:
        return f"fast={self.fast}, slow={self.slow}, signal={self.signal}, column={self.column!r}"


class Volatility(Transform):
    """Add a trailing rolling volatility of log returns.

    Volatility is the standard risk input: it says how much the series has
    been moving, which is often more predictable than its direction.

    Args:
        window: Number of returns in the rolling window.
        annualize: Multiply by ``sqrt(periods_per_year)`` so the value reads
            as an annualised figure.
        periods_per_year: Trading periods in a year; 252 for daily bars.
        column: Column to measure. Defaults to the target.

    Raises:
        DataValidationError: If ``window`` or ``periods_per_year`` is not
            positive, or the column contains non-positive values (a log
            return is undefined for them).
    """

    name: ClassVar[str] = "volatility"

    def __init__(
        self,
        window: int = 20,
        annualize: bool = True,
        periods_per_year: int = 252,
        column: str | None = None,
    ) -> None:
        if window <= 0:
            raise DataValidationError(f"window must be positive, got {window}")
        if periods_per_year <= 0:
            raise DataValidationError(f"periods_per_year must be positive, got {periods_per_year}")
        self.window = window
        self.annualize = annualize
        self.periods_per_year = periods_per_year
        self.column = column

    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return ``{column}_vol{w}``, suffixed ``_raw`` when not annualised."""
        column = _resolve_column(features, self.column)
        suffix = "" if self.annualize else "_raw"
        return (f"{column}_vol{self.window}{suffix}",)

    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Take the rolling standard deviation of one-step log returns."""
        column = _resolve_column(features, self.column)
        series = features.frame[column]
        _require_positive(series, transform=type(self).__name__)
        log_returns = np.log(series).diff()
        values = log_returns.rolling(window=self.window, min_periods=self.window).std()
        if self.annualize:
            values = values * float(np.sqrt(self.periods_per_year))
        return pd.DataFrame({self.output_columns(features)[0]: values}, index=features.frame.index)

    def _describe(self) -> str:
        return (
            f"window={self.window}, annualize={self.annualize}, "
            f"periods_per_year={self.periods_per_year}, column={self.column!r}"
        )
