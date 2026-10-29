"""Demo: generate and inspect a synthetic SPY-like time series.

This script demonstrates the offline, deterministic synthetic data source
that is the default dataset for ForecastForge. It fetches a 2-year
business-day series and prints summary statistics alongside a
price-plus-volume plot.

Install the optional dependencies first::

    pip install 'forecastforge[viz]'

"""

from __future__ import annotations

from pathlib import Path

from forecastforge.data import SyntheticConfig, SyntheticSource


def main() -> None:
    source = SyntheticSource(
        config=SyntheticConfig(
            seed=2024,
            length=252 * 2,  # ~2 years of business days
            initial_price=450.0,
            drift=0.0003,
            volatility=0.012,
            seasonality_amplitude=0.002,
        ),
    )

    ts = source.fetch(symbol="SPY")
    print(f"Symbol       : {ts.metadata.symbol}")
    print(f"Source       : {ts.metadata.source}")
    print(f"Frequency    : {ts.metadata.frequency}")
    print(f"Observations : {len(ts)}")
    print(f"Date range   : {ts.index.min().date()} → {ts.index.max().date()}")
    print()
    print(ts.frame.describe().round(4))
    print()

    _plot(ts)


def _plot(ts) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'forecastforge[viz]'"
        ) from exc

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price, ax_volume = axes
    frame = ts.frame

    ax_price.plot(frame.index, frame["close"], label="Close", linewidth=0.8)
    ax_price.fill_between(
        frame.index,
        frame["low"],
        frame["high"],
        alpha=0.2,
        label="Range (low–high)",
    )
    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(
        f"Synthetic {ts.metadata.symbol} — {len(ts)} days (seed={ts.metadata.attributes['seed']})"
    )
    ax_price.legend(loc="upper left")

    ax_volume.bar(frame.index, frame["volume"], width=0.8, color="gray", alpha=0.5)
    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel("Date")

    plt.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "docs" / "assets" / "synthetic_spy_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
