"""Narrow progress-control facet for app-local progress widgets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.session.pbar_host import ProgressBarModel


class OwnerProgressPort(Protocol):
    """Owner-thread progress service surface shared by control facets.

    ``owner_id`` is the existing tab ID or device name. A no-argument listener
    re-queries bars when notified and follows that owner across operations.
    The returned disposer detaches the listener and is safe to call repeatedly.
    Queries return current ``(handle_id, live model)`` pairs, or an empty tuple
    when the owner has no live bars; they do not copy the models.
    """

    def attach_by_owner(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]: ...
    def bars_for_owner(
        self, owner_id: str
    ) -> tuple[tuple[int, ProgressBarModel], ...]: ...


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

    def __init__(self, progress: OwnerProgressPort) -> None:
        self._progress = progress

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        return self._progress.attach_by_owner(owner_id, listener)

    def progress_bars(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._progress.bars_for_owner(owner_id)
