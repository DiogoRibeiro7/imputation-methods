"""Helpers for renaming public API without breaking existing code.

Every deprecation message starts with ``imputation-methods:`` so callers can
filter or escalate them, e.g. ``-W "error:imputation-methods:FutureWarning"``.
"""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

REMOVAL_VERSION = "1.0.0"
MESSAGE_PREFIX = "imputation-methods:"


def warn_renamed(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Emit the standard ``FutureWarning`` for a renamed API element."""
    warnings.warn(
        f"{MESSAGE_PREFIX} {old} is deprecated and will be removed in "
        f"{REMOVAL_VERSION}; use {new} instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def renamed_parameters(**renames: str) -> Callable[[F], F]:
    """Accept deprecated keyword arguments under their new names.

    Use as ``@renamed_parameters(old_name="new_name")`` on a function or an
    ``__init__`` method. Passing the old name emits a ``FutureWarning`` and
    forwards the value to the new parameter; passing both raises ``TypeError``.
    The decorated signature, as seen by :func:`inspect.signature`, only shows the
    new names.
    """

    def decorator(func: F) -> F:
        owner = func.__qualname__.removesuffix(".__init__")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(
                        f"{owner}() got both {old!r} (deprecated) and {new!r}; "
                        f"use only {new!r}"
                    )
                warn_renamed(f"{owner}({old}=...)", f"{new}=...")
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# Renamed classes and functions, by the module they can still be imported from.
RENAMED_ATTRIBUTES: dict[str, dict[str, str]] = {
    "imputation_methods": {
        "KNNImputerMethod": "KNNImputer",
        "BayesianPCAImputer": "PPCAImputer",
        "predictive_mean_matching": "pmm_impute",
        "bayesian_pca_impute": "ppca_impute",
    },
    "imputation_methods.neighbors": {"KNNImputerMethod": "KNNImputer"},
    "imputation_methods.matrix": {"BayesianPCAImputer": "PPCAImputer"},
    "imputation_methods.functional": {
        "predictive_mean_matching": "pmm_impute",
        "bayesian_pca_impute": "ppca_impute",
    },
}


def renamed_module_attributes(module_name: str) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that resolves renamed names with a warning.

    Assign the result to ``__getattr__`` at module level (PEP 562); the renames
    come from :data:`RENAMED_ATTRIBUTES`. Old names stay out of ``__all__`` and
    ``dir()``, so ``import *`` and API listings only show the new names.
    """
    # Look up by the package-relative name, so modules imported under another
    # prefix (e.g. ``src.imputation_methods`` by some test runners) still resolve.
    package_index = module_name.rfind("imputation_methods")
    renames = RENAMED_ATTRIBUTES.get(module_name[package_index:], {})

    def __getattr__(name: str) -> Any:
        if name in renames:
            new = renames[name]
            warn_renamed(f"{module_name}.{name}", f"{module_name}.{new}")
            return getattr(sys.modules[module_name], new)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
