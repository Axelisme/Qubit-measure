"""Narrow progress-control facet for app-local progress widgets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.services.progress import ProgressService


class ProgressControlPort(Protocol):
    """Owner-keyed progress subscription/query surface for UI consumers."""

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]: ...
    def progress_bars(
        self, owner_id: str
    ) -> tuple[tuple[int, ProgressBarModel], ...]: ...


class ProgressControlFacet:
    """Adapter over ProgressService's owner-keyed progress surface."""

    def __init__(self, progress: ProgressService) -> None:
        self._progress = progress

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        return self._progress.attach_by_owner(owner_id, listener)

    def progress_bars(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._progress.bars_for_owner(owner_id)
