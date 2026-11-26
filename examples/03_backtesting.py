"""Demo: walk-forward backtesting of the four baseline forecasters.

Runs the leakage-safe
:class:`~forecastforge.evaluation.WalkForwardBacktester` over the same
deterministic 504-day series used by ``01_synthetic_data.py``, then prints the
per-model metrics table and plots the fold layout and the resulting MASE
scores. Every fold is scored against a target strictly after its cutoff, and
each model receives a fresh clone per fold, so no state can carry across.

Install the optional dependencies first::

    pip install -e ".[viz]"

Then run::

    python examples/03_backtesting.py

The statsmodels and LightGBM forecasters are intentionally left out: the
demo runs on the dependency-free baselines so that the core install plus
matplotlib is all it needs. See ``make benchmark`` for the full comparison.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from forecastforge.config._optional import optional_import
from forecastforge.data import Regime, SyntheticConfig, SyntheticSource
from forecastforge.evaluation import BacktestConfig, WalkForwardBacktester
from forecastforge.models import Drift, MovingAverage, Naive, SeasonalNaive

if TYPE_CHECKING:
    from forecastforge.evaluation import BacktestResult
    from forecastforge.evaluation.backtest import Fold

_PYPLOT = optional_import("matplotlib.pyplot")

# Same generator settings as 01_synthetic_data.py and 02_features.py, so all
# three demos describe the identical series.
_SEED = 2024
_LENGTH = 504
_REGIMES: tuple[Regime, ...] = (
    Regime(at=126, drift_shift=-0.0006, volatility_shift=0.008),
    Regime(at=252, drift_shift=0.0006, volatility_shift=-0.006),
    Regime(at=378, drift_shift=-0.0002, volatility_shift=0.010),
)

# Five-day-ahead forecasts issued every five days, with a one-row embargo
# between each training window and its origin. 252 rows is one trading year
# of warm-up; the remaining 252 rows yield 50 forecast origins.
_HORIZON = 5

CONFIG = BacktestConfig(initial_train=252, horizon=_HORIZON, step=5, gap=1, seasonal_period=5)

# Every model must be constructed with the same horizon the backtester uses;
# the base class enforces the match and rejects a mismatched model outright.
MODELS: dict[str, Drift | MovingAverage | Naive | SeasonalNaive] = {
    "naive": Naive(horizon=_HORIZON),
    "seasonal_naive": SeasonalNaive(season=5, horizon=_HORIZON),
    "moving_average": MovingAverage(window=20, horizon=_HORIZON),
    "drift": Drift(window=60, horizon=_HORIZON),
}


def _require_pyplot() -> ModuleType:
    """Return ``matplotlib.pyplot``, raising a helpful error when it is absent."""
    if _PYPLOT is None:
        raise ImportError(
            "matplotlib is required for the demo plot. "
            'Install with: pip install -e ".[viz]"'
        )
    return _PYPLOT


def _build_series():
    """Generate the synthetic OHLCV series and return the validated target."""
    source = SyntheticSource(
        config=SyntheticConfig(
            seed=_SEED,
            length=_LENGTH,
            initial_price=450.0,
            drift=0.0003,
            volatility=0.012,
            seasonality_amplitude=0.002,
            regimes=_REGIMES,
        )
    )
    series = source.fetch(symbol="SPY")
    return series.target_series


def _report(result: BacktestResult, folds: tuple[Fold, ...]) -> None:
    """Print the fold layout and the per-model metrics table."""
    print(f"Series length : {_LENGTH} rows")
    print(f"Backtest      : {len(folds)} folds, expanding window")
    print(f"Horizon       : {result.config.horizon} rows per origin")
    print(f"Embargo (gap) : {result.config.gap} row(s)")
    print(f"First origin  : {folds[0].origin}  Last origin: {folds[-1].origin}")
    print()
    print("Per-model metrics over all folds:")
    summary = result.summary()
    print(summary.round(4).to_string())
    print()
    print("MASE reads against the seasonal naive one-step error on the")
    print("training window: 1.0 means no better than naive, below 1.0 better.")


def _plot(result: BacktestResult, series, folds: tuple[Fold, ...]) -> None:
    """Render the fold layout and the per-model MASE comparison."""
    plt = _require_pyplot()
    mase = result.summary()["mase"]
    order = [name for name in MODELS if name in mase.index]
    values = mase.loc[order]

    fig, (ax_series, ax_mase) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1.4]},
    )

    ax_series.plot(series.index, series.values, linewidth=0.8, color="#1f77b4")
    for fold in folds[::5]:  # one marker per five folds, for legibility
        ax_series.axvline(
            series.index[fold.origin],
            color="#bbbbbb",
            linewidth=0.7,
            linestyle=":",
        )
    ax_series.set_title(
        f"Walk-forward folds over the synthetic series - {len(folds)} "
        f"origins, horizon {result.config.horizon} (seed={_SEED})"
    )
    ax_series.set_ylabel("Price ($)")
    ax_series.set_xlabel("Date")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][: len(order)]
    bars = ax_mase.bar(order, values.values, color=colors, width=0.6)
    ax_mase.axhline(1.0, color="#d62728", linewidth=1.0, linestyle="--", label="naive parity (1.0)")
    for bar, value in zip(bars, values.values, strict=True):
        ax_mase.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize="small",
        )
    ax_mase.set_ylabel("MASE (lower is better)")
    ax_mase.set_ylim(0.0, max(values.max() * 1.25, 1.1))
    ax_mase.legend(loc="upper right", fontsize="small")

    plt.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "docs" / "assets" / "backtest_overview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main() -> None:
    """Run the backtest, report the metrics, and save the figure."""
    _require_pyplot()  # fail fast, before doing any work
    series = _build_series()
    backtester = WalkForwardBacktester(config=CONFIG)
    result = backtester.run(series, MODELS)
    _report(result, backtester.folds(len(series)))
    print()
    _plot(result, series, backtester.folds(len(series)))


if __name__ == "__main__":
    main()
