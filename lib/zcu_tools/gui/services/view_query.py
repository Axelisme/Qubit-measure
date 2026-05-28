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
    def get_tab_live_model_root(self, tab_id: str) -> Any: ...


class ViewQueryService:
    """Remote-only View projection: snapshot / screenshot / cfg field edit.

    Owns the read/edit surface the RemoteControlService needs from the View, so
    the Controller façade no longer carries remote-only methods nor reverse-
    imports the remote ``path_resolver``. The actual pixel grab / snapshot lives
    in the View (it needs real Qt widgets); this service resolves the view
    lazily through a provider and centralises the defensive checks.

    ``set_field`` is a *View-coupled* edit: it mutates the tab's live LiveModel
    so the change is visible in the open form (WYSIWYG) and auto-commits via the
    form's ``schema_changed`` chain. It fails fast if the form is not populated.
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

    def live_model_root(self, tab_id: str) -> Any:
        return self._view_provider().get_tab_live_model_root(tab_id)

    def set_field(self, tab_id: str, path: str, value: object) -> None:
        from zcu_tools.gui.services.remote.path_resolver import resolve_and_set

        root = self._view_provider().get_tab_live_model_root(tab_id)
        resolve_and_set(root, path, value)
