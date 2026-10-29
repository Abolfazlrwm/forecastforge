"""Deterministic synthetic time-series generator.

The synthetic source produces realistic-looking OHLCV price paths from a fixed
seed, so every run — local, in CI, or on a contributor's laptop — yields the
identical series. This is the project's default offline dataset and the basis
for all deterministic tests.

The model is a log-price random walk with:

* a constant per-period drift (trend),
* sinusoidal seasonality,
* gaussian noise, and
* optional regime changes (sudden level shifts and/or sustained drift shifts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecastforge.core.errors import DataError
from forecastforge.data.source import BaseDataSource

__all__ = ["Regime", "SyntheticConfig", "SyntheticSource"]

_DEFAULT_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class Regime:
    """A structural break applied at a specific position in the series.

    Attributes:
        at: Zero-based index where the regime takes effect. Negative values
            count from the end of the series.
        level_shift: One-time additive jump in log-price applied at ``at``.
        drift_shift: Sustained additive change to the per-period log-return
            applied from ``at`` onward.
    """

    at: int
    level_shift: float = 0.0
    drift_shift: float = 0.0


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Knobs for the deterministic generator.

    Attributes:
        seed: Seed for the numpy generator. Same seed → same series.
        length: Number of periods to generate.
        frequency: pandas offset alias for the index spacing.
        start: First timestamp of the index.
        initial_price: Opening price at index 0.
        drift: Constant per-period log-return (trend).
        seasonality_amplitude: Peak sinusoidal amplitude in log space.
        seasonality_period: Period of the sinusoid in samples.
        volatility: Standard deviation of the gaussian log-return noise.
        regimes: Structural breaks applied in order.
        volume_base: Mean daily volume before multiplicative noise.
        columns: Columns to emit; must include whatever target/covariates a
            later :meth:`~forecastforge.data.source.BaseDataSource.fetch` requests.
    """

    seed: int = 42
    length: int = 1000
    frequency: str = "B"
    start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2015-01-02"))
    initial_price: float = 100.0
    drift: float = 0.0002
    seasonality_amplitude: float = 0.0015
    seasonality_period: int = 252
    volatility: float = 0.01
    regimes: tuple[Regime, ...] = ()
    volume_base: float = 1_000_000.0
    columns: tuple[str, ...] = _DEFAULT_COLUMNS


def _build_frame(config: SyntheticConfig) -> pd.DataFrame:
    """Generate the full OHLCV frame for ``config`` (no slicing)."""
    if config.length <= 0:
        raise DataError(f"length must be positive, got {config.length}")
    if config.volatility < 0:
        raise DataError(f"volatility must be non-negative, got {config.volatility}")

    index = pd.date_range(start=config.start, periods=config.length, freq=config.frequency)
    n = len(index)
    rng = np.random.default_rng(config.seed)

    t = np.arange(n)
    seasonal = config.seasonality_amplitude * np.sin(2.0 * np.pi * t / config.seasonality_period)
    noise = rng.normal(0.0, config.volatility, size=n)
    log_returns = np.full(n, config.drift) + seasonal + noise

    for regime in config.regimes:
        pos = regime.at if regime.at >= 0 else n + regime.at
        if not 0 <= pos < n:
            raise DataError(f"regime at={regime.at} is out of range for length {n}")
        if regime.level_shift:
            log_returns[pos] += regime.level_shift
        if regime.drift_shift and pos + 1 < n:
            log_returns[pos + 1 :] += regime.drift_shift

    log_price = np.log(config.initial_price) + np.cumsum(log_returns)
    close = np.exp(log_price)

    intraday = rng.normal(0.0, config.volatility * 0.5, size=n)
    open_ = np.empty(n)
    open_[0] = config.initial_price * np.exp(intraday[0])
    open_[1:] = close[:-1] * np.exp(intraday[1:])

    upper = np.abs(rng.normal(0.0, config.volatility * 0.6, size=n))
    lower = np.abs(rng.normal(0.0, config.volatility * 0.6, size=n))
    high = np.maximum(open_, close) * np.exp(upper)
    low = np.minimum(open_, close) * np.exp(-lower)

    volume = config.volume_base * np.exp(rng.normal(0.0, 0.3, size=n))
    volume = np.round(volume).astype("int64")

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    frame.index.name = "date"

    unknown = set(config.columns) - set(frame.columns)
    if unknown:
        available = ", ".join(map(str, frame.columns))
        raise DataError(f"unknown synthetic columns {sorted(unknown)}; available: {available}")
    return frame[list(config.columns)]


@dataclass(frozen=True, slots=True)
class SyntheticSource(BaseDataSource):
    """A fully deterministic, offline :class:`DataSource`.

    Args:
        config: Generator configuration. Defaults to a 1000-business-day,
            SPY-like path seeded with ``42``.
    """

    config: SyntheticConfig = field(default_factory=SyntheticConfig)

    source_name = "synthetic"
    default_frequency = "B"

    def _load_frame(self, *, symbol: str, force_refresh: bool) -> pd.DataFrame:
        return _build_frame(self.config)

    def _attributes(self, *, symbol: str) -> dict[str, Any]:
        cfg = self.config
        return {
            "seed": cfg.seed,
            "length": cfg.length,
            "frequency": cfg.frequency,
            "initial_price": cfg.initial_price,
            "drift": cfg.drift,
            "volatility": cfg.volatility,
            "regimes": [r.at for r in cfg.regimes],
        }
