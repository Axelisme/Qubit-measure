"""Notebook-neutral analysis kernels, loaded independently by their callers."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.analysis import fluxdep

__all__ = ["fluxdep"]


def __getattr__(name: str) -> ModuleType:
    """Load the flux-dependence tools only when explicitly selected."""
    if name != "fluxdep":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".fluxdep", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include the public subpackages without loading them."""
    return sorted(set(globals()) | set(__all__))
