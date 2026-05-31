from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.services.remote.dialogs import DialogName


class _ViewQueryTarget(Protocol):
    """The subset of the View the remote query surface depends on."""

    def get_view_snapshot(self) -> dict[str, object]: ...
    def take_screenshot(self, tab_id: Optional[str] = None) -> bytes: ...
    def take_figure_screenshot(self, tab_id: str) -> bytes: ...
    def take_dialog_screenshot(self, dialog_name: Any) -> bytes: ...


class ViewQueryService:
    """Remote-only View projection: snapshot / screenshot.

    Owns the pure-read surface the remote adapter needs from the View, so the
    Controller façade no longer carries remote-only methods. The actual pixel
    grab / snapshot lives in the View (it needs real Qt widgets); this service
    resolves the view lazily through a provider and centralises the defensive
    checks.

    Note: cfg field editing is *not* here — agents edit a tab's cfg through the
    tab's CfgEditorService session (``editor.set_field``), the same draft the
    open form attaches to. See ADR-0013 (F11).
    """

    def __init__(self, view_provider: Callable[[], _ViewQueryTarget]) -> None:
        self._view_provider = view_provider

    def snapshot(self) -> dict[str, object]:
        return self._view_provider().get_view_snapshot()

    def screenshot(self, tab_id: Optional[str] = None) -> bytes:
        return self._view_provider().take_screenshot(tab_id)

    def figure_screenshot(self, tab_id: str) -> bytes:
        return self._view_provider().take_figure_screenshot(tab_id)

    def dialog_screenshot(self, dialog_name: "DialogName") -> bytes:
        return self._view_provider().take_dialog_screenshot(dialog_name)
