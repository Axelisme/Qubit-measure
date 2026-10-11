"""Smoke tests for SetupDialog."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
)
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.events import DeviceChangedPayload
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
)
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    SetupPreferences,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
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
    ctrl.devices.list_devices.return_value = []
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


def test_entry_new_arbitrary_name_displays_owner_paths_without_point(
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
    assert entry_path_row(dialog, "result_path").toolTip() == str(info.result_path)
    assert entry_path_row(dialog, "database_path").toolTip() == str(info.database_path)
    assert state.session_env.readiness == ContextReadiness.EMPTY
    assert state.session_env.parameters is None
    assert controller.setup_control.list_points() == ()
    assert state.session_env.soc is None
    assert controller.device_control.list_devices() == []


def test_entry_open_does_not_select_a_point(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    controller.setup_control.new_point(
        PointCreateRequest("bias", "", CommandOrigin("gui"))
    )
    entry_selector(dialog).setCurrentText("remembered")
    entry_button(dialog, "Open").click()
    assert state.session_env.readiness == ContextReadiness.EMPTY
    assert state.session_env.parameters is None
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
    assert state.session_env.readiness == ContextReadiness.EMPTY


def test_entry_owner_switch_updates_visible_identity_and_paths(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    info = controller.setup_control.new_entry(EntryRequest("external"))
    assert entry_selector(dialog).currentText() == info.name
    assert entry_path_row(dialog, "result_path").toolTip() == str(info.result_path)
    assert entry_path_row(dialog, "database_path").toolTip() == str(info.database_path)
    assert state.session_env.readiness == ContextReadiness.EMPTY
    assert state.session_env.soc is None


def entry_path_row(dialog: SetupDialog, name: str) -> QLabel:
    """Find the Result or Database path label by its public object name."""
    label = dialog.findChild(QLabel, name)
    assert label is not None
    return label


def point_field(dialog: SetupDialog, name: str) -> QLineEdit:
    """Find an editable working-point field by its public object name."""
    field = dialog.findChild(QLineEdit, name)
    assert field is not None
    return field


def point_list(dialog: SetupDialog) -> QListWidget:
    """Read working-point selection through the public Qt tree."""
    listing = dialog.findChild(QListWidget, "working_points")
    assert listing is not None
    return listing


def point_combo(dialog: SetupDialog, name: str) -> QComboBox:
    """Find the Track device or Clone from selector."""
    combo = dialog.findChild(QComboBox, name)
    assert combo is not None
    return combo


def setup_status(dialog: SetupDialog) -> str:
    """Read the visible owner-result message."""
    status = dialog.findChild(QLabel, "setup_status")
    assert status is not None
    return status.text()


def test_working_point_switch_uses_label_and_displays_description(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    first = controller.setup_control.new_point(
        PointCreateRequest("bias-a", "near sweet spot", CommandOrigin("gui"))
    )
    controller.setup_control.new_point(
        PointCreateRequest("bias-b", "away from sweet spot", CommandOrigin("gui"))
    )
    listing = point_list(dialog)
    assert listing.count() == 2
    first_item, second_item = listing.item(0), listing.item(1)
    assert first_item is not None and second_item is not None
    assert first_item.toolTip() == "near sweet spot"
    assert second_item.toolTip() == "away from sweet spot"
    listing.setCurrentRow(0)
    entry_button(dialog, "Switch to selected").click()
    assert controller.context_control.read_context().target == first
    assert state.session_env.readiness == ContextReadiness.ACTIVE


@pytest.mark.parametrize("description", ["new description", ""])
@pytest.mark.parametrize("source_label", ["source", "src"])
def test_working_point_create_clones_source_and_replaces_description(
    entry_dialog: tuple[SetupDialog, Controller, State],
    description: str,
    source_label: str,
) -> None:
    dialog, controller, state = entry_dialog
    source = controller.setup_control.new_point(
        PointCreateRequest(source_label, "source description", CommandOrigin("gui"))
    )
    controller.context_control.set_parameter(source, "general.ext.flux_value", 0.27)
    clone = point_combo(dialog, "clone_from")
    clone.setCurrentIndex(clone.findData(source_label))
    point_field(dialog, "point_label").setText("cloned")
    point_field(dialog, "point_description").setText(description)
    entry_button(dialog, "Create").click()
    capture = controller.context_control.read_context()
    assert capture.target != source
    assert capture.value("general.ext.flux_value") == 0.27
    assert capture.document().general.description == description
    assert {p.label: p.description for p in controller.setup_control.list_points()} == {
        "cloned": description,
        source_label: "source description",
    }
    assert state.session_env.readiness == ContextReadiness.ACTIVE


def test_working_point_empty_create_uses_setup_seed(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    assert state.session_env.parameters is None
    assert point_combo(dialog, "clone_from").currentText() == "(empty)"
    point_field(dialog, "point_label").setText("fresh")
    entry_button(dialog, "Create").click()
    capture = controller.context_control.read_context()
    assert not capture.lookup("general.ext.flux_value").present
    assert capture.document().general.description == ""
    assert capture.document().components == {}
    assert state.session_env.readiness == ContextReadiness.ACTIVE


@pytest.mark.parametrize("label", ["duplicate", "", "../unsafe"])
def test_working_point_create_owner_rejection_preserves_binding(
    entry_dialog: tuple[SetupDialog, Controller, State],
    label: str,
) -> None:
    dialog, controller, state = entry_dialog
    selected = controller.setup_control.new_point(
        PointCreateRequest("duplicate", "kept", CommandOrigin("gui"))
    )
    before = controller.setup_control.list_points()
    point_field(dialog, "point_label").setText(label)
    entry_button(dialog, "Create").click()
    assert controller.context_control.read_context().target == selected
    assert controller.setup_control.list_points() == before
    assert state.session_env.readiness == ContextReadiness.ACTIVE
    assert "Failed" in setup_status(dialog)


def test_working_point_missing_clone_preserves_binding(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, _state = entry_dialog
    selected = controller.setup_control.new_point(
        PointCreateRequest("kept", "", CommandOrigin("gui"))
    )
    clone = point_combo(dialog, "clone_from")
    clone.addItem("removed", "removed")
    clone.setCurrentIndex(clone.count() - 1)
    point_field(dialog, "point_label").setText("not-created")
    entry_button(dialog, "Create").click()
    assert controller.context_control.read_context().target == selected
    assert [p.label for p in controller.setup_control.list_points()] == ["kept"]
    assert "Failed" in setup_status(dialog)


def test_working_point_switch_missing_source_preserves_binding(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    from qtpy.QtCore import Qt

    dialog, controller, _state = entry_dialog
    selected = controller.setup_control.new_point(
        PointCreateRequest("kept", "", CommandOrigin("gui"))
    )
    listing = point_list(dialog)
    # A stale cached row still passes the exact label to the owning command.
    item = listing.item(0)
    assert item is not None
    item.setData(Qt.ItemDataRole.UserRole, "removed")
    listing.setCurrentRow(0)
    entry_button(dialog, "Switch to selected").click()
    assert controller.context_control.read_context().target == selected
    assert "Failed" in setup_status(dialog)


def test_working_points_follow_entry_changes_and_reopen_without_selecting(
    entry_dialog: tuple[SetupDialog, Controller, State],
    qapp: QApplication,
) -> None:
    dialog, controller, state = entry_dialog
    controller.setup_control.new_point(
        PointCreateRequest("old", "old entry", CommandOrigin("gui"))
    )
    controller.setup_control.new_entry(EntryRequest("other"))
    listing = point_list(dialog)
    assert listing.count() == 0
    assert point_combo(dialog, "clone_from").count() == 1
    controller.setup_control.new_point(
        PointCreateRequest("new", "other entry", CommandOrigin("gui"))
    )
    controller.setup_control.open_entry(EntryRequest("other"))
    before = controller.get_session_env()
    dialog.hide()
    dialog.show()
    qapp.processEvents()
    assert listing.count() == 1
    item = listing.item(0)
    assert item is not None and item.toolTip() == "other entry"
    assert controller.get_session_env() == before
    assert before.parameters is None
    assert state.session_env.readiness == ContextReadiness.EMPTY


def test_working_point_create_without_entry_shows_owner_failure(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = SetupController(
        state, Registry(), None, BaseEventBus(), project_root=str(tmp_path)
    )
    dialog = SetupDialog(controller.setup_control)
    try:
        point_field(dialog, "point_label").setText("bias")
        entry_button(dialog, "Create").click()
        assert state.session_env.readiness == ContextReadiness.EMPTY
        assert controller.setup_control.get_entry_info() is None
        assert "Failed" in setup_status(dialog)
    finally:
        dialog.reject()
        controller.quiesce()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("cached", [None, 0.0035, 0.005])
def test_working_point_device_suggestion_is_cached_and_preserves_manual_text(
    entry_dialog: tuple[SetupDialog, Controller, State],
    monkeypatch: pytest.MonkeyPatch,
    cached: float | None,
) -> None:
    dialog, controller, state = entry_dialog

    def forbidden_live_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("Setup must only read cached device facts")

    for name in ("get_device_info", "poll_device_info", "try_poll_device_info"):
        monkeypatch.setattr(controller.device_control, name, forbidden_live_read)
    state.put_device(
        DeviceState(
            name="flux",
            type_name="YOKOGS200",
            address="offline",
            status=DeviceStatus.CONNECTED,
            remember=False,
            info=YOKOGS200Info(address="offline", mode="current", value=cached)
            if cached is not None
            else None,
        )
    )
    controller.get_bus().emit(DeviceChangedPayload(name="flux"))
    device = point_combo(dialog, "track_device")
    device.setCurrentIndex(device.findData("flux"))
    label = point_field(dialog, "point_label")
    assert label.text() == (f"{cached}A" if cached is not None else "")
    label.selectAll()
    label.insert("my-bias")
    assert label.isModified()
    state.set_device_info(
        "flux", YOKOGS200Info(address="offline", mode="current", value=0.004)
    )
    controller.get_bus().emit(DeviceChangedPayload(name="flux"))
    assert label.text() == "my-bias"
    device.setCurrentIndex(0)
    assert label.text() == "my-bias"
    entry_button(dialog, "Create").click()
    assert [p.label for p in controller.setup_control.list_points()] == ["my-bias"]
    assert state.session_env.soc is None


def test_working_point_cached_none_keeps_previous_suggestion(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, state = entry_dialog
    state.put_device(
        DeviceState(
            name="flux",
            type_name="YOKOGS200",
            address="offline",
            status=DeviceStatus.CONNECTED,
            remember=False,
            info=YOKOGS200Info(address="offline", mode="voltage", value=0.12),
        )
    )
    controller.get_bus().emit(DeviceChangedPayload(name="flux"))
    device = point_combo(dialog, "track_device")
    device.setCurrentIndex(device.findData("flux"))
    label = point_field(dialog, "point_label")
    assert label.text() == "0.12V"
    state.set_device_info("flux", None)
    controller.get_bus().emit(DeviceChangedPayload(name="flux"))
    assert label.text() == "0.12V"


def point_texts(dialog: SetupDialog) -> list[str]:
    listing = point_list(dialog)
    items = [listing.item(row) for row in range(listing.count())]
    return [item.text() for item in items if item is not None]


def current_point_text(dialog: SetupDialog) -> str | None:
    item = point_list(dialog).currentItem()
    return item.text() if item is not None else None


def selected_description(dialog: SetupDialog) -> QLabel:
    label = dialog.findChild(QLabel, "selected_point_description")
    assert label is not None
    return label


def test_long_status_message_wraps_without_widening_dialog(
    entry_dialog: tuple[SetupDialog, Controller, State], qapp: QApplication
) -> None:
    dialog, _controller, _state = entry_dialog
    dialog.resize(900, 600)
    qapp.processEvents()
    ip = dialog.findChild(QLineEdit)
    widths = (dialog.width(), dialog.minimumSizeHint().width())
    combo = entry_selector(dialog)
    combo.addItem("missing_" + "x" * 160)
    combo.setCurrentIndex(combo.count() - 1)
    entry_button(dialog, "Open").click()
    qapp.processEvents()
    status = dialog.findChild(QLabel, "setup_status")
    assert status is not None and status.text()
    assert status.wordWrap()
    assert dialog.width() == widths[0]
    assert dialog.minimumSizeHint().width() <= max(widths[1], 900)
    assert ip is not None


def test_entry_paths_elide_middle_keep_name_and_open_folder(
    entry_dialog: tuple[SetupDialog, Controller, State], qapp: QApplication
) -> None:
    dialog, controller, _state = entry_dialog
    info = controller.setup_control.get_entry_info()
    assert info is not None
    dialog.resize(520, 600)
    qapp.processEvents()
    for name, path in (
        ("result_path", info.result_path),
        ("database_path", info.database_path),
    ):
        row = entry_path_row(dialog, name)
        assert row.toolTip() == str(path)
        assert row.text().endswith(info.name)
        assert row.text() != str(path)
    folders = [b for b in dialog.findChildren(QPushButton) if b.text() == "Open folder"]
    assert len(folders) == 2 and all(button.isEnabled() for button in folders)


def test_point_rows_show_labels_and_selected_description_below(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    from qtpy.QtCore import Qt

    dialog, controller, _state = entry_dialog
    long_text = "JPA on, readout -30 dBm " * 8
    controller.setup_control.new_point(
        PointCreateRequest("bias-a", long_text, CommandOrigin("gui"))
    )
    controller.setup_control.new_point(
        PointCreateRequest("bias-b", "", CommandOrigin("gui"))
    )
    listing = point_list(dialog)
    assert point_texts(dialog) == ["bias-a", "● bias-b"]
    assert listing.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    listing.setCurrentRow(0)
    view = selected_description(dialog)
    assert long_text.strip() in view.text()
    assert view.wordWrap()
    listing.setCurrentRow(1)
    assert long_text.strip() not in view.text()


def test_selection_and_marker_follow_current_point(
    entry_dialog: tuple[SetupDialog, Controller, State], qapp: QApplication
) -> None:
    dialog, controller, _state = entry_dialog
    for label in ("p1", "p2", "p3"):
        point_field(dialog, "point_label").setText(label)
        entry_button(dialog, "Create").click()
    assert point_texts(dialog) == ["p1", "p2", "● p3"]
    assert current_point_text(dialog) == "● p3"
    listing = point_list(dialog)
    listing.setCurrentRow(1)
    entry_button(dialog, "Switch to selected").click()
    assert point_texts(dialog) == ["p1", "● p2", "p3"]
    assert current_point_text(dialog) == "● p2"
    current = listing.currentItem()
    assert current is not None and current.font().bold()
    first = listing.item(0)
    assert first is not None and not first.font().bold()
    controller.setup_control.use_point(PointUseRequest("p1"))
    assert current_point_text(dialog) == "● p1"
    dialog.hide()
    dialog.show()
    qapp.processEvents()
    assert current_point_text(dialog) == "● p1"


def test_entry_without_selected_point_selects_no_row(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, controller, _state = entry_dialog
    controller.setup_control.new_point(
        PointCreateRequest("p1", "", CommandOrigin("gui"))
    )
    controller.setup_control.open_entry(EntryRequest("remembered"))
    assert point_texts(dialog) == ["p1"]
    assert current_point_text(dialog) is None


def test_create_clears_form_on_success_and_keeps_it_on_failure(
    entry_dialog: tuple[SetupDialog, Controller, State],
) -> None:
    dialog, _controller, _state = entry_dialog
    label = point_field(dialog, "point_label")
    description = point_field(dialog, "point_description")
    label.setText("p1")
    description.setText("first")
    entry_button(dialog, "Create").click()
    assert (label.text(), description.text()) == ("", "")
    label.setText("p1")
    description.setText("duplicate")
    entry_button(dialog, "Create").click()
    assert (label.text(), description.text()) == ("p1", "duplicate")


def test_missing_result_root_shows_hint_instead_of_error(
    qapp: QApplication, tmp_path: Path
) -> None:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = SetupController(
        state, Registry(), None, BaseEventBus(), project_root=str(tmp_path)
    )
    dialog = SetupDialog(controller.setup_control)
    try:
        dialog.show()
        qapp.processEvents()
        status = dialog.findChild(QLabel, "setup_status")
        assert status is not None
        assert str(tmp_path / "result") in status.toolTip()
        assert "red" not in status.styleSheet()
        assert entry_selector(dialog).count() == 0
        (tmp_path / "result").write_text("not a directory", encoding="utf-8")
        next(
            b
            for b in dialog.findChildren(QPushButton)
            if b.toolTip() == "Refresh result entries"
        ).click()
        assert "red" in status.styleSheet()
    finally:
        dialog.reject()
        controller.quiesce()
        dialog.deleteLater()
        qapp.processEvents()
