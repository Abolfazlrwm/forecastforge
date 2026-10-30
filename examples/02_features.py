"""Demo: build leakage-safe features on the synthetic series.

Runs a :class:`~forecastforge.features.FeaturePipeline` over the same
deterministic 504-day series used by ``01_synthetic_data.py``, prints what the
pipeline produced and where each column becomes usable, and saves a four-panel
figure of the derived indicators.

Install the optional dependencies first::

    pip install -e ".[viz]"

Then run::

    python examples/02_features.py

"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from forecastforge.config._optional import optional_import
from forecastforge.data import Regime, SyntheticConfig, SyntheticSource
from forecastforge.features import (
    MACD,
    RSI,
    CalendarFeatures,
    FeaturePipeline,
    Lag,
    Returns,
    RollingStat,
    Volatility,
)

if TYPE_CHECKING:
    from forecastforge.features import FeatureSet

_PYPLOT = optional_import("matplotlib.pyplot")

# Same generator settings as 01_synthetic_data.py, so both demos show the
# identical series and the figures can be compared side by side.
_SEED = 2024
_LENGTH = 504
_REGIMES: tuple[Regime, ...] = (
    Regime(at=126, level_shift=-0.08, drift_shift=-0.0004),
    Regime(at=252, level_shift=0.10, drift_shift=0.0006),
    Regime(at=378, drift_shift=-0.0002),
)

# Order matters: the volatility step reads the return column created above it.
PIPELINE = FeaturePipeline(
    [
        Lag([1, 5, 10]),
        RollingStat(window=20, stat="mean"),
        Returns(periods=1, log=True),
        Volatility(window=20),
        RSI(window=14),
        MACD(),
        CalendarFeatures(["dayofweek", "month_cyclic"]),
    ]
)


def _require_pyplot() -> ModuleType:
    """Return ``matplotlib.pyplot``, raising a helpful error when it is absent."""
    if _PYPLOT is None:
        raise ImportError(
            "matplotlib is required for the demo plot. "
            'Install with: pip install -e ".[viz]"'
        )
    return _PYPLOT


def _build_features() -> FeatureSet:
    """Generate the synthetic series and run the feature pipeline over it."""
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
    return PIPELINE.apply(series)


def _report(features: FeatureSet) -> None:
    """Print the pipeline steps, the resulting columns, and the warm-up."""
    print(f"Source series : {len(features)} rows")
    print(f"Pipeline      : {len(PIPELINE)} steps")
    for step in PIPELINE:
        print(f"  - {step!r}")
    print()
    print(f"Feature columns: {len(features.features)}")
    print(f"{'column':34s} {'first valid':12s}")
    print("-" * 48)
    for name in features.features:
        column = features.frame[name]
        first_valid = column.first_valid_index()
        label = "never" if first_valid is None else str(first_valid.date())
        print(f"{name:34s} {label:12s}")
    print()
    print(f"Warm-up rows  : {features.warmup} of {len(features)}")
    usable = features.drop_warmup()
    print(f"Usable rows   : {len(usable)} ({len(usable) / len(features):.0%} of the series)")
    print(f"Complete      : {usable.is_complete}")


def _plot(features: FeatureSet) -> None:
    """Render price, RSI, MACD, and volatility for the generated features."""
    plt = _require_pyplot()
    frame = features.frame
    warmup = features.warmup

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 1.2, 1.5, 1.2]},
    )
    ax_price, ax_rsi, ax_macd, ax_vol = axes

    ax_price.plot(frame.index, frame["close"], linewidth=0.8, color="#1f77b4", label="Close")
    ax_price.plot(
        frame.index,
        frame["close_rolling_mean_20"],
        linewidth=1.1,
        color="#ff7f0e",
        label="20-day rolling mean",
    )
    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(
        f"Leakage-safe features on the synthetic series - {len(frame)} days (seed={_SEED})"
    )
    ax_price.legend(loc="upper left", fontsize="small")

    ax_rsi.plot(frame.index, frame["close_rsi14"], linewidth=0.9, color="#9467bd")
    ax_rsi.axhline(70.0, color="#999999", linestyle="--", linewidth=0.7)
    ax_rsi.axhline(30.0, color="#999999", linestyle="--", linewidth=0.7)
    ax_rsi.set_ylabel("RSI(14)")
    ax_rsi.set_ylim(0.0, 100.0)

    ax_macd.plot(frame.index, frame["close_macd"], linewidth=0.9, label="MACD")
    ax_macd.plot(frame.index, frame["close_macd_signal"], linewidth=0.9, label="Signal")
    ax_macd.bar(
        frame.index,
        frame["close_macd_hist"],
        width=1.0,
        color="#888888",
        alpha=0.55,
        label="Histogram",
    )
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left", fontsize="small", ncol=3)

    ax_vol.plot(frame.index, frame["close_vol20"], linewidth=0.9, color="#2ca02c")
    ax_vol.set_ylabel("Vol (ann.)")
    ax_vol.set_xlabel("Date")

    # Mark the warm-up region so the reader can see what is not usable yet.
    if warmup > 0:
        boundary = frame.index[warmup]
        for axis in axes:
            axis.axvspan(frame.index[0], boundary, color="#cccccc", alpha=0.45)
        low, high = ax_price.get_ylim()
        ax_price.text(
            frame.index[warmup // 2],
            low + 0.06 * (high - low),
            f"warm-up\n{warmup} rows",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize="small",
            color="#444444",
        )

    plt.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "docs" / "assets" / "feature_overview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main() -> None:
    """Build the features, report them, and save the figure."""
    _require_pyplot()  # fail fast, before doing any work
    features = _build_features()
    _report(features)
    print()
    _plot(features)


if __name__ == "__main__":
    main()
