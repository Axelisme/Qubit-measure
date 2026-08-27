"""App-facing save control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.expected_error import FailedPreconditionError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import SavePaths
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

    def save_result(
        self,
        tab_id: str,
        data_path: str | None = None,
        image_path: str | None = None,
        comment: str = "",
    ) -> tuple[str, str]: ...

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None: ...


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
    ) -> None:
        self._state = state
        self._bus = bus
        self._guard = guard
        self._tab = tab
        self._save = save
        self._notify_info = notify_info

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def _resolve_save_paths(self, tab_id: str) -> SavePaths:
        paths = self._tab.get_tab_save_paths(tab_id)
        if paths is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no save paths configured — "
                "set paths via the Save panel or update_tab_save_paths()."
            )
        return paths

    def save_data(
        self, tab_id: str, data_path: str | None = None, comment: str = ""
    ) -> str:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        resolved = data_path or self._resolve_save_paths(tab_id).data_path
        return self._save.start_save_data(permit, resolved, comment=comment)

    def save_image(self, tab_id: str, image_path: str | None = None) -> str:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        resolved = image_path or self._resolve_image_path(tab_id, post=False)
        self._save.save_image_sync(permit, resolved)
        self._notify_info(f"Image saved to {resolved}")
        return resolved

    def save_post_image(self, tab_id: str, image_path: str | None = None) -> str:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        resolved = image_path or self._resolve_image_path(tab_id, post=True)
        self._save.save_post_image_sync(permit, resolved)
        self._notify_info(f"Post-analysis image saved to {resolved}")
        return resolved

    def _require_tab_idle(self, tab_id: str) -> None:
        """Apply the dynamic same-tab gate before resolving destinations."""
        busy = getattr(self._state, "is_tab_busy", None)
        if callable(busy) and busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def _resolve_image_path(self, tab_id: str, *, post: bool) -> str:
        """Resolve the image destination owned by the requested pane.

        New ``TabService`` callers expose independent Analysis and Post-Analysis
        paths. The combined SavePaths fallback is retained only for older injected
        tab ports during the caller migration.
        """
        method_name = (
            "get_tab_post_analysis_image_path"
            if post
            else "get_tab_analysis_image_path"
        )
        getter = getattr(self._tab, method_name, None)
        if callable(getter):
            resolve = cast(Callable[[str], str | None], getter)
            resolved = resolve(tab_id)
            if resolved is None:
                raise FailedPreconditionError(
                    f"Tab {tab_id!r} has no image path configured"
                )
            return resolved
        return self._resolve_save_paths(tab_id).image_path

    def save_result(
        self,
        tab_id: str,
        data_path: str | None = None,
        image_path: str | None = None,
        comment: str = "",
    ) -> tuple[str, str]:
        permit = self._guard.acquire_save_permit(tab_id)
        self._require_tab_idle(tab_id)
        paths = self._resolve_save_paths(tab_id)
        resolved_data = data_path or paths.data_path
        resolved_image = image_path or paths.image_path
        written_data = self._save.start_save_result(
            permit, resolved_data, resolved_image, comment=comment
        )
        return written_data, resolved_image

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._tab.update_tab_save_path_overrides(tab_id, data_path, image_path)
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id,
                fact=TabInteractionFact.SAVE_PATHS_CHANGED,
            )
        )
