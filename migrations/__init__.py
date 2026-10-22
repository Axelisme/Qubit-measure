"""Explicit offline migrations; never imported by the measurement runtime."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .container_upgrade import main as upgrade_main
    from .container_upgrade import upgrade_containers

__all__ = ["upgrade_main", "upgrade_containers"]


def __getattr__(name: str) -> object:
    """Load an explicitly selected upgrade entry without preloading its CLI module."""
    exports = {"upgrade_main": "main", "upgrade_containers": "upgrade_containers"}
    if name not in exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".container_upgrade", __name__), exports[name])
    globals()[name] = value
    return value
