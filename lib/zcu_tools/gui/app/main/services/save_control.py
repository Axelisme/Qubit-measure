"""App-facing save control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.catalog import ExperimentAccess
from zcu_tools.gui.expected_error import FailedPreconditionError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .guard import GuardService
    from .save import SaveService
    from .tab import TabService


class SaveControlPort(Protocol):
    """App-facing save surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def save_data(
        self, tab_id: str, data_path: str | None = None, comment: str = ""
    ) -> str: ...

    def save_image(self, tab_id: str, image_path: str | None = None) -> str: ...

    def save_post_image(self, tab_id: str, image_path: str | None = None) -> str: ...


class SaveControlFacet:
    """Composite adapter over save guards, save service, and save path state."""

    def __init__(
        self,
        *,
        state: State,
        bus: EventBus,
        guard: GuardService,
        tab: TabService,
        save: SaveService,
        notify_info: Callable[[str], None],
        access: ExperimentAccess | None = None,
    ) -> None:
        self._state = state
        self._bus = bus
        self._guard = guard
        self._tab = tab
        self._save = save
        self._notify_info = notify_info
        self._access = access if access is not None else ExperimentAccess()

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def save_data(
        self, tab_id: str, data_path: str | None = None, comment: str = ""
    ) -> str:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        resolved = data_path or self._tab.get_tab_data_path(tab_id)
        if resolved is None:
            raise FailedPreconditionError(f"Tab {tab_id!r} has no data path configured")
        return self._save.start_save_data(permit, resolved, comment=comment)

    def save_image(self, tab_id: str, image_path: str | None = None) -> str:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        resolved = image_path or self._tab.get_tab_analysis_image_path(tab_id)
        if resolved is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no analysis image path configured"
            )
        self._save.save_image_sync(permit, resolved)
        self._notify_info(f"Image saved to {resolved}")
        return resolved

    def save_post_image(self, tab_id: str, image_path: str | None = None) -> str:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        resolved = image_path or self._tab.get_tab_post_analysis_image_path(tab_id)
        if resolved is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no post-analysis image path configured"
            )
        self._save.save_post_image_sync(permit, resolved)
        self._notify_info(f"Post-analysis image saved to {resolved}")
        return resolved

    def _require_tab_idle(self, tab_id: str) -> None:
        self._access.require_available()
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")
