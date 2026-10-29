"""Deterministic random seed configuration for ForecastForge.

Seeds Python's :mod:`random` module and, if installed, NumPy and
PyTorch. Neither NumPy nor PyTorch is a hard dependency of
ForecastForge at this stage of the project, so both are resolved
through :func:`forecastforge.config._optional.optional_import` and their
absence is handled gracefully rather than raising an
:class:`ImportError`.
"""

from __future__ import annotations

import os
import random

from forecastforge.config._optional import optional_import

__all__ = ["set_seed"]


def set_seed(seed: int) -> None:
    """Seed every available random number generator for reproducibility.

    Always seeds:

      - the ``PYTHONHASHSEED`` environment variable (affects hash-based
        iteration order in *subprocesses* spawned afterward; it has no
        effect on the already-running interpreter, but is set here so
        that any child process inherits a consistent seed)
      - the standard library :mod:`random` module

    Additionally seeds NumPy's global RNG and PyTorch's CPU, CUDA, and
    MPS generators if the respective library is installed.

    Args:
        seed: The seed value to apply everywhere.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy = optional_import("numpy")
    if numpy is not None:
        numpy.random.seed(seed)

    torch = optional_import("torch")
    if torch is None:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mps_backend = getattr(torch.backends, "mps", None)
    mps_module = getattr(torch, "mps", None)
    if mps_backend is not None and mps_backend.is_available() and mps_module is not None:
        mps_module.manual_seed(seed)
