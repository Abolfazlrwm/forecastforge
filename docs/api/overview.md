# API reference

Every public symbol is documented here, generated from the docstrings in the
source tree. The package is split by stage:

| Module | Contents |
|---|---|
| [`forecastforge.config`](config.md) | Settings model, environment detection, paths, device, seeding |
| [`forecastforge.core`](core.md) | The exception hierarchy shared across stages |
| [`forecastforge.data`](data.md) | Sources, loaders, and the synthetic generator |
| [`forecastforge.schemas`](schemas.md) | The validated `TimeSeries` container |
| [`forecastforge.features`](features.md) | Transforms, `FeatureSet`, and `FeaturePipeline` |
| [`forecastforge.models`](models.md) | The `Forecaster` interface, baselines, optional models |
| [`forecastforge.evaluation`](evaluation.md) | Metrics, the walk-forward backtester, and the benchmark |

The `tracking`, `serving`, `monitoring`, and `pipelines` subpackages are
placeholders for planned phases and intentionally have no public API yet.

## Install

```bash
pip install -e .              # core, fully offline
pip install -e ".[data]"      # + Yahoo Finance connector and Parquet engine
pip install -e ".[models]"    # + statsmodels and LightGBM forecasters
pip install -e ".[viz]"       # + matplotlib for the example figures
```

The optional models import cleanly without their dependency and raise an
error naming the extra when you construct one.
