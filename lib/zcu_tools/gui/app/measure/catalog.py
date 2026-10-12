"""Stable, Qt-free experiment-catalog reload contract."""

from __future__ import annotations

from typing import Protocol

from zcu_tools.gui.catalog_loader import CatalogReloadError as CatalogReloadError
from zcu_tools.gui.catalog_loader import PreparedCatalogReload as PreparedCatalogReload
from zcu_tools.gui.expected_error import FailedPreconditionError

from .registry import Registry


class ExperimentAccess:
    """Shared driving-entry gate, separate from hardware exclusion."""

    def __init__(self) -> None:
        self.switching = False
        self.available = True
        self.shutting_down = False

    def require_available(self) -> None:
        if self.shutting_down or self.switching or not self.available:
            raise FailedPreconditionError(
                "Experiments are unavailable during reload or shutdown"
            )


class ExperimentCatalogLoader(Protocol):
    def prepare(self) -> PreparedCatalogReload:
        """Validate source without replacing modules or the live registry."""
        ...

    def load(self, plan: PreparedCatalogReload) -> Registry:
        """Consume a prepared revision and return an unpublished catalog."""
        ...
