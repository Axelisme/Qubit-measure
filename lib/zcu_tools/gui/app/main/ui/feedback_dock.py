"""Docking controller for the operation feedback panel."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

from .feedback_widget import FeedbackPanel

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller


class FeedbackHostTab(Protocol):
    """Narrow tab capability required by the dock controller."""

    def mount_feedback_panel(self, panel: QWidget) -> None: ...

    def unmount_feedback_panel(self, panel: QWidget) -> None: ...


class FeedbackDockController:
    """Owns feedback-panel target resolution, docking, and gating."""

    def __init__(
        self,
        ctrl: Controller,
        parent: QWidget,
        *,
        tab_by_id: Callable[[str], FeedbackHostTab | None],
        running_tab_id: Callable[[], str | None],
        active_tab_id: Callable[[], str | None],
    ) -> None:
        self._ctrl = ctrl
        self._tab_by_id = tab_by_id
        self._running_tab_id = running_tab_id
        self._active_tab_id = active_tab_id
        self._panel = FeedbackPanel(ctrl, parent=parent)
        self._panel.hide()
        self._host_tab: FeedbackHostTab | None = None

    @property
    def panel(self) -> FeedbackPanel:
        """Return the owned feedback panel for focused UI tests."""
        return self._panel

    @property
    def host_tab(self) -> FeedbackHostTab | None:
        """Return the current dock host for focused UI tests."""
        return self._host_tab

    def refresh(self) -> None:
        """Mount or unmount the panel from op-count and agent-presence gates."""
        target = self._target_tab() if self._should_mount() else None
        if target is None:
            self._unmount()
            return

        if self._host_tab is not target:
            if self._host_tab is not None:
                self._host_tab.unmount_feedback_panel(self._panel)
            target.mount_feedback_panel(self._panel)
            self._host_tab = target

        self._panel.refresh_gating()

    def _should_mount(self) -> bool:
        return (
            self._ctrl.active_operation_count() > 0 and self._ctrl.has_agent_connected()
        )

    def _target_tab(self) -> FeedbackHostTab | None:
        tab_id = self._running_tab_id() or self._active_tab_id()
        if tab_id is None:
            return None
        return self._tab_by_id(tab_id)

    def _unmount(self) -> None:
        if self._host_tab is not None:
            self._host_tab.unmount_feedback_panel(self._panel)
            self._host_tab = None
        self._panel.hide()
        self._panel.clear_input()


__all__ = ["FeedbackDockController", "FeedbackHostTab"]
