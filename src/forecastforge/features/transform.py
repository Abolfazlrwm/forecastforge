"""The base contract for feature transforms.

A transform derives new columns from an existing
:class:`~forecastforge.features.frame.FeatureSet`. Two rules define the
framework:

**Causality.** The value of a feature at row *t* may depend only on rows
``<= t``. This is what makes walk-forward backtesting honest, and it is the
single easiest thing to get wrong — a centred rolling window, a
``shift(-1)``, or a z-score computed over the whole sample all look
reasonable and all leak the future. The framework cannot enforce causality
statically, so it is verified by tests: see ``tests/features/test_leakage.py``,
which truncates and perturbs the future for every built-in transform.

**Declared output.** :meth:`Transform.output_columns` must exactly match the
columns :meth:`Transform.compute` returns. :meth:`Transform.apply` checks
this on every call, so a mismatch surfaces at the point of the mistake rather
than as a confusing column error three stages downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from forecastforge.core.errors import DataValidationError
from forecastforge.features.frame import FeatureSet

__all__ = ["Transform"]


class Transform(ABC):
    """Base class for a causal, stateless feature transform.

    Subclasses set :attr:`name` for display purposes and implement
    :meth:`output_columns` and :meth:`compute`. :meth:`apply` is provided and
    should not normally be overridden.

    Transforms must be **stateless**: the same input always produces the same
    output, and a transform keeps no fitted parameters. Anything that needs
    to learn from data (scalers, encoders) belongs to the model pipeline,
    where the fit/transform split makes leakage auditable.
    """

    name: ClassVar[str] = "transform"

    @abstractmethod
    def output_columns(self, features: FeatureSet) -> tuple[str, ...]:
        """Return the column names :meth:`compute` will produce, in order.

        Args:
            features: The input set. Output names may depend on its column
                names (e.g. the target name) but never on its values.
        """

    @abstractmethod
    def compute(self, features: FeatureSet) -> pd.DataFrame:
        """Compute the new columns.

        Args:
            features: The input set. May be read freely, including columns
                added by earlier transforms.

        Returns:
            A frame containing *only* the new columns, indexed like
            ``features.frame``. NaN is allowed (warm-up).
        """

    def apply(self, features: FeatureSet) -> FeatureSet:
        """Apply the transform and return the extended set.

        Args:
            features: The input set.

        Returns:
            A new set with this transform's columns appended.

        Raises:
            DataValidationError: If the produced columns do not match
                :meth:`output_columns`, or if they would overwrite an existing
                column.
        """
        added = self.compute(features)
        expected = self.output_columns(features)
        actual = tuple(str(column) for column in added.columns)
        if actual != expected:
            raise DataValidationError(
                f"{type(self).__name__} declared columns {expected} but produced {actual}"
            )
        return features.extend(added)

    def _describe(self) -> str:
        """Return the parameter summary used by :meth:`__repr__`.

        Subclasses override this so that a pipeline prints its configuration
        rather than a list of bare class names.
        """
        return ""

    def __repr__(self) -> str:
        """Return a debug representation including the configuration."""
        return f"{type(self).__name__}({self._describe()})"
