"""ForecastForge: a production-grade, open-source forecasting platform.

Implemented:
  * Core exception hierarchy and validated ``TimeSeries`` container.
  * Structural validation (sorted DatetimeIndex, numeric/finite/NaN checks,
    optional gap detection).
  * Data ingestion layer: ``DataSource`` protocol, deterministic synthetic
    OHLCV generator, optional Yahoo Finance connector (cached to Parquet),
    and CSV/Parquet file loaders.
  * Feature engineering: ``FeatureSet`` container, causal transforms (lags,
    rolling statistics, returns, volatility, RSI, MACD, calendar features),
    and an ordered ``FeaturePipeline`` with a leakage test suite.

Still planned:
  * Model training, evaluation, and serving layers.
  * Experiment tracking, monitoring, and containerised deployment.

This module exposes package metadata so that the package is importable and
its version can be introspected by tooling (pip, build backends, CI, docs).
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__: str = metadata.version("forecastforge")
except metadata.PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g. running from a source checkout without
    # `pip install -e .`). Fall back to a development sentinel version.
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
