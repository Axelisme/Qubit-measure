"""Stable, Qt-free experiment-catalog reload contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


@dataclass(frozen=True, eq=False)
class PreparedCatalogReload:
    """Opaque, loader-owned, single-use source revision token."""


class CatalogReloadError(RuntimeError):
    """A reload failure with an explicit recovery disposition."""

    def __init__(self, message: str, *, restart_required: bool = False) -> None:
        super().__init__(message)
        self.restart_required = restart_required


class ExperimentCatalogLoader(Protocol):
    def prepare(self) -> PreparedCatalogReload:
        """Validate source without replacing modules or the live registry."""
        ...

    def load(self, plan: PreparedCatalogReload) -> Registry:
        """Consume a prepared revision and return an unpublished catalog."""
        ...
