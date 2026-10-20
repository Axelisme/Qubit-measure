"""Public UI and persistence contracts for PredictorDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetSelectionRange,
    QTabWidget,
)
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.session.device_models import DeviceEntry
from zcu_tools.gui.session.events import DeviceChangedPayload
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.session.ui.predictor_dialog import (
    PredictorDialog,
    PredictorDialogState,
)
from zcu_tools.gui.widgets.spinbox import FiniteFloatSpinBox, TrimDoubleSpinBox
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams


def _make_dialog(**kwargs) -> PredictorDialog:
    # These UI contracts do not require a completed numerical trial.
    return PredictorDialog(
        PredictorControlFacet(PredictorService()), params=FluxoniumParams(), **kwargs
    )


def _make_device_ctrl(
    *,
    entries: list[DeviceEntry] | None = None,
    cached: dict[str, float | None] | None = None,
    units: dict[str, str] | None = None,
    live: dict[str, FakeDeviceInfo | None] | None = None,
) -> MagicMock:
    ctrl = MagicMock()
    value_cache = cached if cached is not None else {}
    unit_map = units if units is not None else {}
    live_map = live if live is not None else {}
    ctrl.list_devices.return_value = list(entries or [])
    ctrl.get_cached_device_value.side_effect = lambda name: value_cache.get(name)
    ctrl.get_device_unit.side_effect = lambda name: unit_map.get(name, "none")
    ctrl.get_device_info.side_effect = lambda name: live_map.get(name)
    ctrl.try_poll_device_info.return_value = True
    ctrl.on_device_changed.return_value = MagicMock(name="unsubscribe_device")
    return ctrl


def _table(dialog: PredictorDialog) -> QTableWidget:
    table = dialog.findChild(QTableWidget)
    assert table is not None
    return table


def _button(dialog: PredictorDialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


def _status(dialog: PredictorDialog) -> str:
    label = dialog.findChild(QLabel, "predictor_status")
    assert label is not None
    return label.text()


def _device_combo(dialog: PredictorDialog) -> QComboBox:
    return next(
        combo for combo in dialog.findChildren(QComboBox) if combo.findData("flux") >= 0
    )


def _value_spin(dialog: PredictorDialog) -> TrimDoubleSpinBox:
    spin = dialog.findChild(TrimDoubleSpinBox)
    assert spin is not None
    return spin


def test_live_mode_locks_model_and_transition_controls(qapp):
    dialog = _make_dialog(device=_make_device_ctrl())

    dialog.set_live_mode(enabled=True)

    for widget_type in (
        QDoubleSpinBox,
        FiniteFloatSpinBox,
        QSpinBox,
        QCheckBox,
        QComboBox,
        EvalNumericField,
    ):
        assert all(
            not widget.isEnabled() for widget in dialog.findChildren(widget_type)
        )
    for text in ("Load", "Refresh files", "Add", "Remove", "Calibrate", "Use"):
        assert not _button(dialog, text).isEnabled()

    dialog.set_live_mode(enabled=False)

    assert _value_spin(dialog).isEnabled()
    assert _button(dialog, "Add").isEnabled()
    assert not _button(dialog, "Use").isEnabled()


def test_live_device_value_updates_read_only_control(qapp):
    dialog = _make_dialog()
    dialog.set_live_mode(enabled=True)

    dialog.set_live_device_value(0.25)

    assert _value_spin(dialog).value() == 0.25
    assert not _value_spin(dialog).isEnabled()
    assert "0.25" in _status(dialog)


def test_predictor_dialog_default_tracked_count(qapp):
    dialog = _make_dialog()

    assert len(dialog.capture_state().tracked_transitions) == 5
    assert _table(dialog).rowCount() == 5


def test_predictor_dialog_optional_device_facet_no_controls(qapp):
    dialog = _make_dialog()

    assert all(button.text() != "Use" for button in dialog.findChildren(QPushButton))


def test_predictor_dialog_device_selector_lists_connected_cached_values(qapp):
    entries = [
        DeviceEntry("flux", "YOKOGS200", DeviceStatus.CONNECTED.value),
        DeviceEntry("fake", "FakeDevice", DeviceStatus.CONNECTED.value),
        DeviceEntry("lo", "RohdeSchwarzSGS100A", DeviceStatus.CONNECTED.value),
        DeviceEntry("memory", "FakeDevice", DeviceStatus.MEMORY_ONLY.value),
        DeviceEntry("busy", "FakeDevice", DeviceStatus.SETTING_UP.value),
    ]
    device = _make_device_ctrl(
        entries=entries,
        cached={"flux": 0.1, "fake": 0.2, "lo": None, "memory": 0.3, "busy": 0.4},
        units={"flux": "A"},
    )
    dialog = _make_dialog(device=device)

    combo = _device_combo(dialog)
    assert [combo.itemData(i) for i in range(combo.count())] == [None, "flux", "fake"]
    assert "YOKOGS200" in combo.itemText(1)
    assert "(A)" in combo.itemText(1)


def test_predictor_dialog_read_selected_device_updates_value_immediately(qapp):
    device = _make_device_ctrl(
        entries=[DeviceEntry("flux", "FakeDevice", DeviceStatus.CONNECTED.value)],
        cached={"flux": 0.1},
        live={"flux": FakeDeviceInfo(address="none", value=0.456)},
    )
    dialog = _make_dialog(device=device)
    combo = _device_combo(dialog)
    combo.setCurrentIndex(combo.findData("flux"))

    _button(dialog, "Use").click()

    assert _value_spin(dialog).value() == 0.456
    assert "Read device value from flux" in _status(dialog)
    device.get_device_info.assert_called_with("flux")


def test_predictor_dialog_show_refreshes_selected_cached_value(qapp):
    cached: dict[str, float | None] = {"flux": 0.1}
    device = _make_device_ctrl(
        entries=[DeviceEntry("flux", "FakeDevice", DeviceStatus.CONNECTED.value)],
        cached=cached,
    )
    dialog = _make_dialog(device=device)
    combo = _device_combo(dialog)
    combo.setCurrentIndex(combo.findData("flux"))
    device.try_poll_device_info.reset_mock()
    cached["flux"] = 0.3

    dialog.show()
    qapp.processEvents()
    try:
        assert _value_spin(dialog).value() == 0.3
        device.try_poll_device_info.assert_called_with("flux")
    finally:
        dialog.close()


def test_predictor_dialog_selected_device_changed_applies_cached_value(qapp):
    cached: dict[str, float | None] = {"flux": 0.1, "other": 0.2}
    device = _make_device_ctrl(
        entries=[
            DeviceEntry("flux", "FakeDevice", DeviceStatus.CONNECTED.value),
            DeviceEntry("other", "FakeDevice", DeviceStatus.CONNECTED.value),
        ],
        cached=cached,
    )
    dialog = _make_dialog(device=device)
    combo = _device_combo(dialog)
    combo.setCurrentIndex(combo.findData("flux"))
    on_changed = device.on_device_changed.call_args.args[0]
    dialog.show()
    qapp.processEvents()
    try:
        cached["other"] = 0.9
        on_changed(DeviceChangedPayload(name="other"))
        assert _value_spin(dialog).value() == 0.1

        cached["flux"] = 0.7
        on_changed(DeviceChangedPayload(name="flux"))
        assert _value_spin(dialog).value() == 0.7
    finally:
        dialog.close()


def test_predictor_dialog_persistent_reject_hides_without_finishing(qapp):
    device = _make_device_ctrl()
    unsubscribe = device.on_device_changed.return_value
    dialog = _make_dialog(device=device, persistent_on_close=True)
    finished = MagicMock()
    dialog.finished.connect(finished)
    dialog.show()
    qapp.processEvents()

    dialog.reject()
    qapp.processEvents()

    assert not dialog.isVisible()
    finished.assert_not_called()
    unsubscribe.assert_not_called()


def test_predictor_dialog_capture_restore_state_roundtrip(qapp):
    source = _make_dialog()
    state = PredictorDialogState(
        tracked_transitions=((2, 3), (3, 5)),
        tab_index=2,
    )

    source.restore_state(state)
    captured = source.capture_state()
    restored = _make_dialog()
    restored.restore_state(captured)

    assert captured == state
    assert restored.capture_state() == state
    assert _table(restored).rowCount() == 2


def test_predictor_dialog_restore_state_allows_empty_transitions(qapp):
    dialog = _make_dialog()
    state = PredictorDialogState(tracked_transitions=(), tab_index=1)

    dialog.restore_state(state)

    assert dialog.capture_state() == state
    assert _table(dialog).rowCount() == 0


def test_predictor_dialog_restore_state_skips_invalid_transitions(qapp):
    dialog = _make_dialog()

    dialog.restore_state(
        PredictorDialogState(
            tracked_transitions=((0, 1), (0, 1), (2, 2), (-1, 3), (3, 5)),
            tab_index=99,
        )
    )

    assert dialog.capture_state() == PredictorDialogState(
        tracked_transitions=((0, 1), (3, 5)), tab_index=2
    )
    assert _table(dialog).rowCount() == 2


def test_predictor_dialog_state_changed_only_for_persisted_ui_state(qapp):
    dialog = _make_dialog()
    changed = MagicMock()
    dialog.state_changed.connect(changed)
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None

    tabs.setCurrentIndex(1)
    frm, to = dialog.findChildren(QSpinBox)
    frm.setValue(2)
    to.setValue(5)
    _button(dialog, "Add").click()
    _table(dialog).selectRow(0)
    _button(dialog, "Remove").click()

    assert changed.call_count == 3

    changed.reset_mock()
    dialog.set_live_device_value(0.25)
    _value_spin(dialog).setValue(0.5)

    changed.assert_not_called()


def test_predictor_dialog_add_valid_transition(qapp):
    dialog = _make_dialog()
    before = dialog.capture_state().tracked_transitions
    frm, to = dialog.findChildren(QSpinBox)
    frm.setValue(2)
    to.setValue(4)

    _button(dialog, "Add").click()

    assert dialog.capture_state().tracked_transitions == (*before, (2, 4))
    assert _table(dialog).rowCount() == len(before) + 1


def test_predictor_dialog_add_invalid_from_ge_to(qapp):
    dialog = _make_dialog()
    before = dialog.capture_state()
    frm, to = dialog.findChildren(QSpinBox)
    frm.setValue(3)
    to.setValue(2)

    _button(dialog, "Add").click()

    assert dialog.capture_state() == before
    assert _table(dialog).rowCount() == len(before.tracked_transitions)


def test_predictor_dialog_add_duplicate_transition(qapp):
    dialog = _make_dialog()
    before = dialog.capture_state()
    frm, to = dialog.findChildren(QSpinBox)
    frm.setValue(0)
    to.setValue(1)

    _button(dialog, "Add").click()

    assert dialog.capture_state() == before
    assert _table(dialog).rowCount() == len(before.tracked_transitions)


def test_predictor_dialog_remove_selected_single_row(qapp):
    dialog = _make_dialog()
    before = dialog.capture_state().tracked_transitions
    _table(dialog).selectRow(1)

    _button(dialog, "Remove").click()

    assert dialog.capture_state().tracked_transitions == before[:1] + before[2:]
    assert _table(dialog).rowCount() == len(before) - 1


def test_predictor_dialog_remove_selected_multiple_rows(qapp):
    dialog = _make_dialog()
    before = dialog.capture_state().tracked_transitions
    table = _table(dialog)
    for row in (0, 2):
        table.setRangeSelected(
            QTableWidgetSelectionRange(row, 0, row, table.columnCount() - 1), True
        )

    _button(dialog, "Remove").click()

    assert dialog.capture_state().tracked_transitions == (before[1], *before[3:])
    assert table.rowCount() == len(before) - 2


def test_predictor_dialog_remove_no_selection_is_noop(qapp):
    dialog = _make_dialog()
    before = dialog.capture_state()
    _table(dialog).clearSelection()

    _button(dialog, "Remove").click()

    assert dialog.capture_state() == before


def test_predictor_dialog_remove_all_empties_table(qapp):
    dialog = _make_dialog()
    table = _table(dialog)
    table.selectAll()

    _button(dialog, "Remove").click()

    assert dialog.capture_state().tracked_transitions == ()
    assert table.rowCount() == 0


def test_predictor_dialog_value_column_runtime_error_is_logged(qapp, caplog):
    ctrl = MagicMock()
    # A query-port failure exercises UI recovery without replacing a successful kernel.
    ctrl.predict_freq.side_effect = RuntimeError("compute failed")
    dialog = PredictorDialog(ctrl, params=FluxoniumParams())

    with caplog.at_level("ERROR"):
        dialog.set_live_device_value(0.1)

    table = _table(dialog)
    for row in range(table.rowCount()):
        for column in range(1, table.columnCount()):
            item = table.item(row, column)
            assert item is not None and item.text() == "—"
    assert "compute failed" in _status(dialog)
    assert (
        sum(
            record.message == "predictor value-column update failed"
            for record in caplog.records
        )
        == 1
    )


def test_predictor_dialog_predict_curve_error_shown_in_status(qapp):
    ctrl = MagicMock()
    # The failing query is the collaborator boundary; no successful result is stubbed.
    ctrl.predict_freq_curve.side_effect = ValueError("oops")

    dialog = PredictorDialog(ctrl, params=FluxoniumParams(4.0, 1.0, 1.0, 0.0, 1.0, 0.0))

    assert "oops" in _status(dialog)


def test_predictor_params_row_keeps_full_control_height(qapp):
    dialog = _make_dialog()
    dialog.show()
    qapp.processEvents()
    try:
        row = [_button(dialog, text) for text in ("Load", "Refresh files")]
        for widget in row:
            assert widget.height() >= widget.sizeHint().height()
    finally:
        dialog.close()


def test_transition_value_columns_fit_their_text(qapp):
    dialog = _make_dialog()
    dialog.show()
    qapp.processEvents()
    try:
        table = _table(dialog)
        assert table.rowCount() > 0
        metrics = table.fontMetrics()
        for column in range(table.columnCount() - 1):
            needed = max(
                metrics.horizontalAdvance(item.text())
                for row in range(table.rowCount())
                if (item := table.item(row, column)) is not None
            )
            assert table.columnWidth(column) >= needed, column
        viewport = table.viewport()
        assert viewport is not None
        used = sum(table.columnWidth(c) for c in range(table.columnCount()))
        assert used >= viewport.width() - 1
    finally:
        dialog.close()
