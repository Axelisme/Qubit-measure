"""Tests for FeedbackDockController target selection and panel docking."""

from __future__ import annotations

from typing import Any, cast

from qtpy.QtWidgets import QLineEdit, QPushButton, QWidget
from zcu_tools.gui.app.main.ui.feedback_dock import FeedbackDockController
from zcu_tools.gui.app.main.ui.feedback_widget import FeedbackPanel


class _Ctrl:
    def __init__(
        self,
        *,
        op_count: int,
        agent_connected: bool,
        can_cancel: bool = False,
    ) -> None:
        self.op_count = op_count
        self.agent_connected = agent_connected
        self.can_cancel = can_cancel
        self.feedback: list[tuple[str, bool]] = []

    def active_operation_count(self) -> int:
        return self.op_count

    def has_agent_connected(self) -> bool:
        return self.agent_connected

    def can_cancel_active_operation(self) -> bool:
        return self.can_cancel

    def send_feedback(self, text: str, *, stop: bool) -> None:
        self.feedback.append((text, stop))


class _FakeHost:
    def __init__(self, name: str, events: list[str] | None = None) -> None:
        self.name = name
        self.events = events
        self.panels: list[QWidget] = []

    def mount_feedback_panel(self, panel: QWidget) -> None:
        if panel in self.panels:
            return
        self.panels.append(panel)
        if self.events is not None:
            self.events.append(f"{self.name}:mount")
        panel.show()

    def unmount_feedback_panel(self, panel: QWidget) -> None:
        if panel not in self.panels:
            return
        self.panels.remove(panel)
        if self.events is not None:
            self.events.append(f"{self.name}:unmount")
        panel.setParent(None)  # type: ignore[arg-type]


def _ctrl(value: _Ctrl) -> Any:
    return cast(Any, value)


def _input(panel: FeedbackPanel) -> QLineEdit:
    inputs = panel.findChildren(QLineEdit)
    assert len(inputs) == 1
    return inputs[0]


def _button(panel: FeedbackPanel, text: str) -> QPushButton:
    for button in panel.findChildren(QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"button not found: {text!r}")


def test_refresh_waits_for_operation_and_agent(qapp):
    del qapp
    ctrl = _Ctrl(op_count=1, agent_connected=False)
    host = _FakeHost("active")
    parent = QWidget()
    dock = FeedbackDockController(
        _ctrl(ctrl),
        parent=parent,
        tab_by_id=lambda tab_id: host if tab_id == "active" else None,
        running_tab_id=lambda: None,
        active_tab_id=lambda: "active",
    )

    dock.refresh()
    assert dock.host_tab is None
    assert dock.panel not in host.panels

    ctrl.agent_connected = True
    dock.refresh()

    assert dock.host_tab is host
    assert dock.panel in host.panels


def test_running_tab_wins_over_active_tab(qapp):
    del qapp
    ctrl = _Ctrl(op_count=1, agent_connected=True)
    running = _FakeHost("running")
    active = _FakeHost("active")
    parent = QWidget()
    dock = FeedbackDockController(
        _ctrl(ctrl),
        parent=parent,
        tab_by_id={"running": running, "active": active}.get,
        running_tab_id=lambda: "running",
        active_tab_id=lambda: "active",
    )

    dock.refresh()

    assert dock.host_tab is running
    assert dock.panel in running.panels
    assert dock.panel not in active.panels


def test_remount_unmounts_old_target_before_mounting_new_target(qapp):
    del qapp
    ctrl = _Ctrl(op_count=1, agent_connected=True)
    events: list[str] = []
    host_a = _FakeHost("a", events)
    host_b = _FakeHost("b", events)
    running_id: str | None = "a"
    parent = QWidget()
    dock = FeedbackDockController(
        _ctrl(ctrl),
        parent=parent,
        tab_by_id={"a": host_a, "b": host_b}.get,
        running_tab_id=lambda: running_id,
        active_tab_id=lambda: "b",
    )

    dock.refresh()
    running_id = None
    dock.refresh()

    assert events == ["a:mount", "a:unmount", "b:mount"]
    assert dock.host_tab is host_b
    assert dock.panel not in host_a.panels
    assert dock.panel in host_b.panels


def test_unmount_hides_and_clears_input_when_gate_drops(qapp):
    del qapp
    ctrl = _Ctrl(op_count=1, agent_connected=True)
    host = _FakeHost("active")
    parent = QWidget()
    dock = FeedbackDockController(
        _ctrl(ctrl),
        parent=parent,
        tab_by_id=lambda tab_id: host if tab_id == "active" else None,
        running_tab_id=lambda: None,
        active_tab_id=lambda: "active",
    )
    dock.refresh()
    _input(dock.panel).setText("pending")

    ctrl.agent_connected = False
    dock.refresh()

    assert dock.host_tab is None
    assert dock.panel not in host.panels
    assert dock.panel.isHidden()
    assert _input(dock.panel).text() == ""


def test_unmount_hides_and_clears_input_when_target_disappears(qapp):
    del qapp
    ctrl = _Ctrl(op_count=1, agent_connected=True)
    host = _FakeHost("active")
    hosts: dict[str, _FakeHost] = {"active": host}
    parent = QWidget()
    dock = FeedbackDockController(
        _ctrl(ctrl),
        parent=parent,
        tab_by_id=hosts.get,
        running_tab_id=lambda: None,
        active_tab_id=lambda: "active",
    )
    dock.refresh()
    _input(dock.panel).setText("pending")

    hosts.clear()
    dock.refresh()

    assert dock.host_tab is None
    assert dock.panel not in host.panels
    assert dock.panel.isHidden()
    assert _input(dock.panel).text() == ""


def test_refresh_updates_stop_button_gating(qapp):
    del qapp
    ctrl = _Ctrl(op_count=1, agent_connected=True, can_cancel=False)
    host = _FakeHost("active")
    parent = QWidget()
    dock = FeedbackDockController(
        _ctrl(ctrl),
        parent=parent,
        tab_by_id=lambda tab_id: host if tab_id == "active" else None,
        running_tab_id=lambda: None,
        active_tab_id=lambda: "active",
    )
    dock.refresh()
    _input(dock.panel).setText("please stop")
    assert not _button(dock.panel, "Send & Stop").isEnabled()

    ctrl.can_cancel = True
    dock.refresh()

    assert _button(dock.panel, "Send & Stop").isEnabled()
