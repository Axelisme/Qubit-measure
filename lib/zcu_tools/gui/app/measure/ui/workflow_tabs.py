"""Synchronize facet-created workflow tabs, local close gestures and status navigation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTabBar, QTabWidget, QToolButton, QWidget

from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import GateChangedPayload

from .workflow_pane import WorkflowPane

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller

logger = logging.getLogger(__name__)


class WorkflowTabs(QTabWidget):
    """Mount tabs opened by either GUI or remote callers without a second registry.

    badge is the parent-owned status-bar navigation button for the active run.
    controller supplies the shared facet/session. activate reveals Workflow tabs;
    lock_changed refreshes the experiment/dialog controls from the shared gate.
    dispose only after producers stop. Closing a tab asks the facet first, then
    detaches its pane before Qt deletion; a refused close retains every widget.
    """

    def __init__(
        self,
        controller: Controller,
        activate: Callable[[], None],
        lock_changed: Callable[[], None],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workflow.tabs")
        self.setTabsClosable(True)
        self._ctrl = controller
        self._control = controller.workflow_control()
        self._activate = activate
        self._lock_changed = lock_changed
        self._writable = True
        self._disposed = False
        self._panes: dict[str, WorkflowPane] = {}
        self._labels: dict[str, str] = {}
        self._numbers: dict[str, int] = {}
        self._badge_target: str | None = None
        self.badge = QToolButton(parent)
        self.badge.setObjectName("workflow.badge")
        self.badge.setToolTip("Go to active workflow")
        self.badge.hide()
        self.badge.clicked.connect(self._navigate)
        self.tabCloseRequested.connect(self._close_requested)
        self._unsubscribe = self._control.subscribe(self._sync)
        self._subscriptions = EventSubscriptions()
        self._subscriptions.subscribe(
            controller.get_bus(), GateChangedPayload, self._gate_changed
        )
        # Engine acceptance is off-owner and has no separate status event. Poll
        # only its read-only status; terminal delivery still comes via subscribe.
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def synchronize(self) -> None:
        """Mount tabs that were opened before this view was constructed."""
        self._sync()

    def _gate_changed(self, _payload: GateChangedPayload) -> None:
        self._refresh()
        self._lock_changed()

    def _sync(self) -> None:
        if self._disposed:
            return
        snapshots = self._control.tabs()
        live = {s.tab_id for s in snapshots}
        for tab_id in tuple(self._panes):
            if tab_id not in live:
                pane = self._panes.pop(tab_id)
                pane.dispose()
                self.removeTab(self.indexOf(pane))
                self._labels.pop(tab_id)
                pane.deleteLater()
        for snapshot in snapshots:
            if snapshot.tab_id not in self._panes:
                number = self._numbers.get(snapshot.name, 0) + 1
                self._numbers[snapshot.name] = number
                label = f"{snapshot.name} ({number})"
                pane = WorkflowPane(
                    self._ctrl,
                    snapshot.tab_id,
                    self,
                )
                pane.set_writable(writable=self._writable)
                self._panes[snapshot.tab_id] = pane
                self._labels[snapshot.tab_id] = label
                index = self.addTab(pane, label)
                bar = self.tabBar()
                if bar is not None:
                    for side in (
                        QTabBar.ButtonPosition.LeftSide,
                        QTabBar.ButtonPosition.RightSide,
                    ):
                        button = bar.tabButton(index, side)
                        if button is not None:
                            button.setToolTip(f"Close {label}")
                self.setCurrentIndex(index)
                self._activate()
        self._refresh()
        self._lock_changed()

    def _refresh(self) -> None:
        if self._disposed:
            return
        snapshots = self._control.tabs()
        active = next((s for s in snapshots if s.busy), None)
        self._badge_target = None if active is None else active.tab_id
        self.badge.setVisible(active is not None)
        if active is not None:
            status = "pending" if active.status is None else active.status.lifecycle
            self.badge.setText(
                f"{self._labels.get(active.tab_id, active.name)}: {status}"
            )
        for pane in self._panes.values():
            pane.set_writable(writable=self._writable)
            pane.refresh()

    def _navigate(self) -> None:
        pane = self._panes.get(self._badge_target or "")
        if pane is not None:
            self._activate()
            self.setCurrentWidget(pane)

    def _close_requested(self, index: int) -> None:
        if not self._writable:
            return
        pane = self.widget(index)
        tab_id = next(
            (key for key, value in self._panes.items() if value is pane), None
        )
        if tab_id is None:
            return
        try:
            self._control.close(tab_id)
        except Exception:
            # Release failures retain both owner and pane for retry.
            logger.exception("Workflow tab close failed: %s", tab_id)

    def set_writable(self, *, writable: bool) -> None:
        """Project local authority to close gestures and pane commands."""
        self._writable = writable
        self.setTabsClosable(writable)
        self._refresh()

    def dispose(self) -> None:
        """Stop the poll and detach all view subscriptions after producer drain."""
        if self._disposed:
            return
        self._disposed = True
        self._timer.stop()
        self._unsubscribe()
        self._subscriptions.unsubscribe_all()
        for pane in self._panes.values():
            pane.dispose()
