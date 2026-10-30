"""Demo: generate and inspect a synthetic SPY-like time series.

This script demonstrates the offline, deterministic synthetic data source
that is the default dataset for ForecastForge. It fetches a 504-business-day
(roughly 2-year) series with visible structural regimes, prints summary
statistics, and saves a price-plus-volume plot with the regimes shaded.

Install the optional dependencies first::

    pip install -e ".[viz]"

Then run::

    python examples/01_synthetic_data.py

"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from forecastforge.config._optional import optional_import
from forecastforge.data import Regime, SyntheticConfig, SyntheticSource

if TYPE_CHECKING:
    from forecastforge.schemas.time_series import TimeSeries

# ``optional_import`` must target the submodule: importing the top-level
# ``matplotlib`` package does not expose ``pyplot``.
_PYPLOT = optional_import("matplotlib.pyplot")

# Regimes that produce visible structural breaks in the synthetic series.
# They are chosen so the shaded spans are wide enough to read on the plot.
_DEMO_REGIMES: tuple[Regime, ...] = (
    Regime(at=126, level_shift=-0.08, drift_shift=-0.0004),  # ~6 months in: correction
    Regime(at=252, level_shift=0.10, drift_shift=0.0006),  # ~12 months in: recovery + uptrend
    Regime(at=378, drift_shift=-0.0002),  # ~18 months in: cooling
)


def _require_pyplot() -> ModuleType:
    """Return ``matplotlib.pyplot``, raising a helpful error when it is absent."""
    if _PYPLOT is None:
        raise ImportError(
            "matplotlib is required for the demo plot. "
            'Install with: pip install -e ".[viz]"'
        )
    return _PYPLOT


def main() -> None:
    """Generate the synthetic series, print stats, and save the demo plot."""
    _require_pyplot()  # fail fast, before doing any work

    source = SyntheticSource(
        config=SyntheticConfig(
            seed=2024,
            length=504,  # 2 years of business days
            initial_price=450.0,
            drift=0.0003,
            volatility=0.012,
            seasonality_amplitude=0.002,
            regimes=_DEMO_REGIMES,
        ),
    )

    ts = source.fetch(symbol="SPY")
    print(f"Symbol       : {ts.metadata.symbol}")
    print(f"Source       : {ts.metadata.source}")
    print(f"Frequency    : {ts.metadata.frequency}")
    print(f"Observations : {len(ts)}")
    print(f"Date range   : {ts.index.min().date()} -> {ts.index.max().date()}")
    print(f"Regimes      : {[r.at for r in _DEMO_REGIMES]}")
    print()
    print(ts.frame.describe().round(4))
    print()

    _plot(ts, regimes=_DEMO_REGIMES)


def _plot(ts: TimeSeries, *, regimes: tuple[Regime, ...]) -> None:
    """Render the price/volume chart with regime spans shaded."""
    plt = _require_pyplot()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price, ax_volume = axes
    frame = ts.frame

    # Price line
    ax_price.plot(frame.index, frame["close"], label="Close", linewidth=0.8, color="#1f77b4")
    ax_price.fill_between(
        frame.index,
        frame["low"],
        frame["high"],
        alpha=0.15,
        label="Range (low-high)",
        color="#1f77b4",
    )

    # Shade regime spans
    regime_colors = ["#ff9999", "#99cc99", "#9999cc"]
    for i, regime in enumerate(regimes):
        start_idx = regime.at
        end_idx = regimes[i + 1].at if i + 1 < len(regimes) else len(frame) - 1
        start_date = frame.index[start_idx]
        end_date = frame.index[end_idx]
        ax_price.axvspan(
            start_date,
            end_date,
            alpha=0.12,
            color=regime_colors[i % len(regime_colors)],
            label=f"Regime {i + 1} (at={regime.at})",
        )

    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(
        f"Synthetic asset ({ts.metadata.symbol}-like) - {len(ts)} days "
        f"(seed={ts.metadata.attributes['seed']})"
    )
    ax_price.legend(loc="upper left", fontsize="small")

    # Volume bars
    ax_volume.bar(frame.index, frame["volume"], width=0.8, color="gray", alpha=0.5)
    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel("Date")

    plt.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "docs" / "assets" / "synthetic_spy_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
