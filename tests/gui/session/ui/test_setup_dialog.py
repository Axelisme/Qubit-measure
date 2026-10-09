"""Smoke tests for SetupDialog."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QComboBox,
    QInputDialog,
    QLabel,
    QListWidget,
    QPushButton,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
)
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    SetupPreferences,
)
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog


def _prefs(
    *,
    entry_name: str = "",
    ip: str = "192.168.10.1",
    port: int = 8887,
) -> SetupPreferences:
    return SetupPreferences(
        entry_name=entry_name,
        ip=ip,
        port=port,
    )


def _make_ctrl(**overrides: object) -> MagicMock:
    ctrl = MagicMock()
    # SetupControlPort.get_setup_preferences is non-Optional (defaults when
    # nothing is remembered), so the double honours that contract rather than
    # returning None.
    ctrl.get_setup_preferences.return_value = _prefs()
    ctrl.list_devices.return_value = []
    ctrl.get_context_labels.return_value = []
    ctrl.get_active_context_label.return_value = None
    ctrl.get_soccfg.return_value = None
    ctrl.get_project_root.return_value = "/tmp"
    ctrl.list_entries.return_value = ()
    ctrl.get_entry_info.return_value = None
    ctrl.list_points.return_value = ()
    for k, v in overrides.items():
        getattr(ctrl, k).return_value = v
    return ctrl


def test_setup_dialog_cached_point_list_uses_entry_queries_on_show_and_fact(qapp):
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import WorkingPointInfo
    from zcu_tools.gui.session.events import ContextSwitchedPayload

    # Keep unrelated legacy project controls outside this read-side seam.
    ctrl = MagicMock()
    prefs = MagicMock()
    prefs.entry_name = ""
    prefs.ip = "192.168.10.1"
    prefs.port = 8887
    ctrl.get_setup_preferences.return_value = prefs
    ctrl.list_entries.return_value = ()
    ctrl.get_entry_info.return_value = None
    ctrl.list_devices.return_value = []
    ctrl.get_soccfg.return_value = None
    bus = BaseEventBus()
    ctrl.get_bus.return_value = bus
    ctrl.list_points.return_value = (
        WorkingPointInfo("base", "first"),
        WorkingPointInfo("next", "second"),
    )
    # Legacy collaborators must not mask a live call to the deleted owner.
    ctrl.get_active_context_label.side_effect = AttributeError("deleted context label")
    ctrl.get_context_labels.side_effect = AttributeError("deleted context labels")
    dialog = SetupDialog(ctrl)
    try:
        dialog.show()
        qapp.processEvents()
        lists = dialog.findChildren(QListWidget)
        assert len(lists) == 1
        labels = []
        for index in range(lists[0].count()):
            item = lists[0].item(index)
            assert item is not None
            labels.append(item.text())
        assert labels == ["base", "next"]
        ctrl.list_points.return_value = (WorkingPointInfo("updated", ""),)
        bus.emit(ContextSwitchedPayload(binding_id=None, sources=None))
        assert lists[0].count() == 1
        updated = lists[0].item(0)
        assert updated is not None and updated.text() == "updated"
        dialog.hide()
        ctrl.list_points.return_value = ()
        dialog.show()
        qapp.processEvents()
        assert lists[0].count() == 0
    finally:
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()


def test_setup_dialog_init(qapp):
    ctrl = _make_ctrl(
        get_context_labels=["ctx1", "ctx2"],
        get_active_context_label="ctx1",
    )

    dialog = SetupDialog(ctrl)

    assert dialog._ctx_list.count() == 2
    # The first item should be active (index 0)
    assert dialog._ctx_list.currentRow() == 0


def test_setup_dialog_switch_context(qapp):
    ctrl = _make_ctrl(
        get_context_labels=["ctx1", "ctx2"],
        get_active_context_label="ctx1",
    )
    dialog = SetupDialog(ctrl)

    dialog._ctx_list.setCurrentRow(1)  # select ctx2
    dialog._on_switch_clicked()

    ctrl.use_context.assert_called_with("ctx2")


def test_setup_dialog_new_context_clone_from_dropdown(qapp):
    # No bound device (empty device combo) -> bind_device=None; the clone
    # dropdown is populated from the active project's context labels and the
    # picked label flows through as clone_from.
    ctrl = _make_ctrl(
        get_context_labels=["ctx_a", "ctx_b"],
        get_active_context_label="ctx_a",
    )
    dialog = SetupDialog(ctrl)

    # index 0 == "(none)"; pick "ctx_b".
    idx = dialog._clone_combo.findData("ctx_b")
    assert idx > 0
    dialog._clone_combo.setCurrentIndex(idx)
    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(bind_device=None, clone_from="ctx_b")


def test_setup_dialog_new_context_clone_none_default(qapp):
    # Default clone selection "(none)" -> clone_from=None.
    ctrl = _make_ctrl(get_context_labels=["ctx_a"])
    dialog = SetupDialog(ctrl)

    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(bind_device=None, clone_from=None)


def test_setup_dialog_connect_mock_dispatches_request(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Connect"
    ).click()

    ctrl.start_simulated_environment.assert_called_once_with()


def test_setup_dialog_connect_remote_dispatches_request(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(False)
    dialog._ip_edit.setText("10.0.0.1")
    dialog._port_spin.setValue(7000)
    dialog._on_connect_clicked()

    ctrl.start_connect.assert_called_once()
    (req,) = ctrl.start_connect.call_args.args
    assert isinstance(req, ConnectRemoteRequest)
    assert req.ip == "10.0.0.1"
    assert req.port == 7000
    ctrl.remember_connection.assert_called_once_with(
        ConnectionPreferences(ip="10.0.0.1", port=7000)
    )


def test_setup_dialog_connect_failure_signal_updates_status(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    dialog._on_connect_clicked()
    on_failed = ctrl.bind_connection_outcome.call_args.args[1]
    on_failed("network bad")

    assert "network bad" in dialog._conn_status.text()
    assert dialog._connect_btn.isEnabled()


class SetupController(Controller):
    """Expose owned background teardown for this real-session widget fixture."""

    def quiesce(self) -> None:
        """Settle workers and queued terminal callbacks before Qt teardown."""
        self._background_svc.quiesce()


@pytest.fixture
def entry_dialog(
    qapp: QApplication, tmp_path: Path
) -> Generator[tuple[SetupDialog, Controller, State]]:
    """Own a real offline session; opening Setup never connects hardware."""
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = SetupController(
        state, Registry(), None, BaseEventBus(), project_root=str(tmp_path)
    )
    controller.setup_control.new_entry(EntryRequest("remembered"))
    dialog = SetupDialog(controller.setup_control)
    try:
        dialog.show()
        qapp.processEvents()
        yield dialog, controller, state
    finally:
        dialog.reject()
        controller.quiesce()
        dialog.deleteLater()
        qapp.processEvents()


def entry_button(dialog: SetupDialog, text: str) -> QPushButton:
    """Locate the visible action through the public Qt widget tree."""
    return next(
        button for button in dialog.findChildren(QPushButton) if button.text() == text
    )


def entry_selector(dialog: SetupDialog) -> QComboBox:
    """Result entry selector has a stable object name for Qt consumers."""
    combo = dialog.findChild(QComboBox, "result_entry")
    assert combo is not None
    return combo


def test_entry_show_and_reopen_prefill_without_changing_active_point(
    entry_dialog: tuple[SetupDialog, Controller, State], qapp: QApplication
) -> None:
    dialog, controller, state = entry_dialog
    selected = controller.setup_control.new_point(
        PointCreateRequest("bias", "", CommandOrigin("gui"))
    )
    dialog.hide()
    dialog.show()
    qapp.processEvents()
    assert entry_selector(dialog).currentText() == "remembered"
    assert controller.context_control.read_context().target == selected
    assert state.session_env.readiness == ContextReadiness.ACTIVE
    assert state.session_env.soc is None
    assert state.session_env.soccfg is None


def test_entry_new_arbitrary_name_displays_owner_paths_and_empty_draft(
    entry_dialog: tuple[SetupDialog, Controller, State],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog, controller, state = entry_dialog

    def chosen_name(*args: object, **kwargs: object) -> tuple[str, bool]:
        return "cooldown-A", True

    monkeypatch.setattr(QInputDialog, "getText", chosen_name)
    entry_button(dialog, "New entry…").click()
    info = controller.setup_control.get_entry_info()
    assert info is not None and info.name == "cooldown-A"
    assert entry_selector(dialog).currentText() == info.name
    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert str(info.result_path) in labels
    assert str(info.database_path) in labels
    assert state.session_env.readiness == ContextReadiness.DRAFT
    assert controller.context_control.read_context().document().components == {}
    assert controller.setup_control.list_points() == ()
    assert state.session_env.soc is None
    assert controller.device_control.list_devices() == []


def test_entry_open_republishes_draft_without_selecting_a_point(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    selected = controller.setup_control.new_point(
        PointCreateRequest("bias", "", CommandOrigin("gui"))
    )
    entry_selector(dialog).setCurrentText("remembered")
    entry_button(dialog, "Open").click()
    assert state.session_env.readiness == ContextReadiness.DRAFT
    assert controller.context_control.read_context().target != selected
    assert controller.setup_control.list_points()[0].label == "bias"


def test_entry_rejected_new_shows_error_and_keeps_active_point(
    entry_dialog: tuple[SetupDialog, Controller, State],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog, controller, state = entry_dialog
    selected = controller.setup_control.new_point(
        PointCreateRequest("bias", "", CommandOrigin("gui"))
    )

    def duplicate_name(*args: object, **kwargs: object) -> tuple[str, bool]:
        return "remembered", True

    monkeypatch.setattr(QInputDialog, "getText", duplicate_name)
    entry_button(dialog, "New entry…").click()
    assert controller.context_control.read_context().target == selected
    assert state.session_env.readiness == ContextReadiness.ACTIVE
    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "exists" in labels.lower()


def test_entry_cancel_new_keeps_binding(
    entry_dialog: tuple[SetupDialog, Controller, State], monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog, controller, state = entry_dialog
    before = controller.setup_control.get_entry_info()

    def cancelled(*args: object, **kwargs: object) -> tuple[str, bool]:
        return "", False

    monkeypatch.setattr(QInputDialog, "getText", cancelled)
    entry_button(dialog, "New entry…").click()
    assert controller.setup_control.get_entry_info() == before
    assert state.session_env.readiness == ContextReadiness.DRAFT


def test_entry_owner_switch_updates_visible_identity_and_paths(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    info = controller.setup_control.new_entry(EntryRequest("external"))
    assert entry_selector(dialog).currentText() == info.name
    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert str(info.result_path) in labels
    assert str(info.database_path) in labels
    assert state.session_env.readiness == ContextReadiness.DRAFT
    assert state.session_env.soc is None
