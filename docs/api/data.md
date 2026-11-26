# forecastforge.data

Data ingestion: a `DataSource` protocol implemented as load, slice, validate,
and wrap; a deterministic synthetic OHLCV generator as the default offline
dataset; CSV and Parquet file loaders that preserve a `DatetimeIndex`; and an
optional Yahoo Finance connector behind the `[data]` extra.

## Synthetic data

::: forecastforge.data.synthetic

## Sources

::: forecastforge.data.source

## Yahoo Finance connector

::: forecastforge.data.yfinance_source

## File loaders

::: forecastforge.data.loaders
