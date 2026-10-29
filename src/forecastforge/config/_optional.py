"""Access to third-party modules that may or may not be installed.

NumPy and PyTorch are *optional* for ForecastForge: the package must import
and behave correctly whether or not they are present. A plain top-level
``import numpy`` cannot express that requirement, while an ``import`` hidden
inside a function body hides the dependency from readers and from tooling
that builds an import graph.

This module centralises the optional-import policy in one documented place so
that call sites stay uniform and reviewable.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["optional_import"]


def optional_import(module_name: str) -> ModuleType | None:
    """Import an optional module, returning ``None`` when it is unavailable.

    A module that is installed but fails to import (for example a broken
    native build) is treated as unavailable rather than propagating the
    error: optional accelerators must never be able to break the core
    package, and callers already have a well-defined fallback path.

    Args:
        module_name: Fully qualified name of the module to import, e.g.
            ``"numpy"`` or ``"torch.backends"``.

    Returns:
        The imported module, or ``None`` if it could not be imported.
    """
    try:
        return import_module(module_name)
    except ImportError:
        return None
