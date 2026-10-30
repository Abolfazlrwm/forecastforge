"""Composable feature pipelines.

A :class:`FeaturePipeline` chains transforms so that later steps can consume
columns produced by earlier ones — the common case being "compute returns,
then take the rolling volatility of those returns".

Ordering matters and is the caller's responsibility: a pipeline is a list, not
a dependency graph. Keeping it explicit is deliberate — a pipeline that
silently reorders itself is much harder to review for leakage.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from forecastforge.features.frame import FeatureSet
from forecastforge.features.transform import Transform
from forecastforge.schemas.time_series import TimeSeries

__all__ = ["FeaturePipeline"]


class FeaturePipeline:
    """An ordered sequence of transforms.

    Args:
        steps: Transforms to apply, first to last.

    Example::

        pipeline = FeaturePipeline(
            [
                Returns(periods=1, log=True),
                RollingStat(window=20, stat="std", column="close_logret1"),
                CalendarFeatures(["dayofweek", "month_cyclic"]),
            ]
        )
        features = pipeline.apply(series)
    """

    def __init__(self, steps: Sequence[Transform]) -> None:
        self.steps: tuple[Transform, ...] = tuple(steps)

    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self.steps)

    def __iter__(self) -> Iterator[Transform]:
        """Iterate over the steps in order."""
        return iter(self.steps)

    def __repr__(self) -> str:
        """Return a representation listing every step."""
        inner = ", ".join(repr(step) for step in self.steps)
        return f"FeaturePipeline([{inner}])"

    def apply(self, data: TimeSeries | FeatureSet) -> FeatureSet:
        """Apply every step in order.

        Args:
            data: A validated series, or an existing set to extend. Passing a
                :class:`FeatureSet` lets callers compose pipelines
                incrementally.

        Returns:
            The enriched set. Feature columns may contain NaN in the warm-up
            region; inspect :attr:`FeatureSet.warmup`.
        """
        features = data if isinstance(data, FeatureSet) else FeatureSet.from_time_series(data)
        for step in self.steps:
            features = step.apply(features)
        return features

    def feature_names(self, data: TimeSeries | FeatureSet) -> tuple[str, ...]:
        """Return the column names :meth:`apply` would produce.

        This runs the pipeline, because output names can depend on upstream
        columns (for example a rolling statistic of a return column). It is a
        convenience for introspection, not a cheap schema lookup.

        Args:
            data: A validated series, or an existing set to extend.
        """
        return self.apply(data).features
