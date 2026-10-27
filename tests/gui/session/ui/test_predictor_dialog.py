"""Public UI and persistence contracts for PredictorDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetSelectionRange,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.session.ui.predictor_curves import PredictorCurveTabs
from zcu_tools.gui.session.ui.predictor_dialog import (
    PredictorDialog,
    PredictorDialogState,
)
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox
from zcu_tools.resources.predictor_file import PredictorFlux
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from tests.gui.session.ui._predictor_support import (
    make_device_control as _make_device_ctrl,
)


def test_parameter_editor_returns_energy_only_trial(qapp):
    from zcu_tools.gui.session.ui.predictor_parameters import PredictorParameterEditor

    editor = PredictorParameterEditor(
        PredictorControlFacet(PredictorService()),
        on_edited=lambda: None,
        on_replaced=lambda: None,
        on_status=lambda _message, _error: None,
    )
    try:
        assert editor.parameters() == FluxoniumParams(EJ=4.0, EC=1.0, EL=1.0)
    finally:
        editor.deleteLater()


def test_dialog_loads_frame_and_every_cached_device_value(qapp):
    from zcu_tools.resources.predictor_file import (
        PredictorFile,
        PredictorFlux,
        PredictorSource,
    )

    flux = PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"})
    control = MagicMock(wraps=PredictorControlFacet(PredictorService()))
    control.load_file.return_value = PredictorFile(
        "fluxonium", FluxoniumParams(), flux, PredictorSource(None, None)
    )
    dialog = PredictorDialog(
        control,
        params=FluxoniumParams(),
        device=_make_device_ctrl(cached={"coil": 0.3, "trim": -0.4}),
    )
    try:
        dialog.load_file("two.json")
        assert dialog.flux() == flux
        assert dialog.device_values() == {"coil": 0.3, "trim": -0.4}
    finally:
        dialog.close()
        dialog.deleteLater()


def test_curve_tabs_project_all_devices_onto_selected_axis(qapp):
    from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
    from zcu_tools.gui.session.ui.predictor_curves import CurveCoordinates
    from zcu_tools.simulate.transmon import TransmonParams

    from tests.gui._user_input import ComponentDriver

    tabs = PredictorCurveTabs(
        PredictorControlFacet(PredictorService()),
        on_follow=lambda _value: None,
        on_lock=lambda _value: None,
        on_status=lambda _text, _error: None,
        on_clear_values=lambda: None,
        on_matrix_capability=lambda _supported: None,
    )
    try:
        rendered = tabs.refresh_curves(
            TransmonParams(20.0, 0.3, 0.1),
            ((0, 1),),
            (),
            0.3,
            coordinates=CurveCoordinates(
                frame=FluxFrame(
                    0.2,
                    {"coil": DevicePeriod(2.0, "A"), "trim": DevicePeriod(-4.0, "V")},
                ),
                axis_device="coil",
                device_values={"coil": 0.3, "trim": -0.4},
            ),
        )
        tabs.show()
        qapp.processEvents()
        assert rendered
        assert any(
            axis.visible_curves > 0 for axis in ComponentDriver(tabs).plots().axes
        )
    finally:
        tabs.close()
        tabs.deleteLater()


def test_calibrate_adds_phi0_correction_to_nonzero_local_offset(qapp):
    from zcu_tools.resources.predictor_file import PredictorFlux
    from zcu_tools.simulate.transmon import TransmonParams

    params = TransmonParams(20.0, 0.3, 0.1)
    dialog = PredictorDialog(PredictorControlFacet(PredictorService()), params=params)
    try:
        dialog.set_flux(
            PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}),
            {"coil": 0.3, "trim": -0.4},
        )
        measured = params.build_predictor().predict_freq(0.48, (0, 1))
        result = dialog.calibrate(measured)
        assert result.offset_correction == pytest.approx(0.03, abs=1e-5)
        assert dialog.flux().offset == pytest.approx(0.23, abs=1e-5)
        assert dialog.predict_frequency() == pytest.approx(measured, abs=1e-3)
        assert not result.clamped
    finally:
        dialog.close()
        dialog.deleteLater()


def test_cache_restores_frame_values_and_axis_with_energy_trial(qapp, tmp_path):
    from zcu_tools.resources.predictor_file import PredictorFlux

    from tests.gui._user_input import ComponentDriver

    port = PredictorControlFacet(PredictorService())
    flux = PredictorFlux(0.23, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"})
    dialog = PredictorDialog(port, params=FluxoniumParams())
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path)
        dialog.set_flux(flux, {"coil": 0.3, "trim": -0.4})
        dialog.show()
        qapp.processEvents()
        ComponentDriver(dialog).click(object_name="predictor_axis_trim")
    finally:
        dialog.close()
    reopened = PredictorDialog(port, params=FluxoniumParams())
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path)
        assert reopened.flux() == flux
        assert reopened.device_values() == {"coil": 0.3, "trim": -0.4}
        reopened.show()
        qapp.processEvents()
        assert (
            ComponentDriver(reopened).observe(object_name="predictor_axis_trim").checked
        )
    finally:
        reopened.close()
        reopened.deleteLater()
        dialog.deleteLater()


def test_live_marker_changes_only_the_selected_axis_value(qapp):
    dialog = _make_dialog()
    try:
        dialog.set_flux(
            PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}),
            {"coil": 0.3, "trim": -0.4},
        )
        dialog.set_live_mode(enabled=True)
        dialog.set_live_device_value(0.25)
        assert dialog.device_values() == {"coil": 0.25, "trim": -0.4}
    finally:
        dialog.close()
        dialog.deleteLater()


def test_show_preserves_full_precision_native_values(qapp):
    dialog = _make_dialog()
    values = {"coil": 0.12345678901234567}
    try:
        dialog.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), values)
        dialog.show()
        qapp.processEvents()
        assert dialog.device_values() == values
    finally:
        dialog.close()
        dialog.deleteLater()


def test_supplied_frame_clears_stale_missing_frame_status(qapp):
    from zcu_tools.simulate.transmon import TransmonParams

    dialog = PredictorDialog(
        PredictorControlFacet(PredictorService()), params=TransmonParams(20.0, 0.3, 0.1)
    )
    try:
        assert "flux frame" in _status(dialog)
        dialog.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.3})
        assert _status(dialog) == ""
    finally:
        dialog.close()
        dialog.deleteLater()


def test_empty_snapshot_replaces_a_previously_loaded_flux_frame(qapp):
    from zcu_tools.gui.session.ui.predictor_cache import PredictorSnapshot
    from zcu_tools.simulate.predictor import PredictorInputError

    dialog = _make_dialog()
    try:
        dialog.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.3})
        dialog.restore_persisted_state(PredictorSnapshot())
        with pytest.raises(PredictorInputError, match="flux frame"):
            dialog.flux()
        assert dialog.device_values() == {}
    finally:
        dialog.close()
        dialog.deleteLater()


def _make_dialog(**kwargs) -> PredictorDialog:
    # These UI contracts do not require a completed numerical trial.
    return PredictorDialog(
        PredictorControlFacet(PredictorService()), params=FluxoniumParams(), **kwargs
    )


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


def _value_spin(dialog: PredictorDialog) -> TrimDoubleSpinBox:
    spin = dialog.findChild(TrimDoubleSpinBox)
    assert spin is not None
    return spin


def test_live_mode_locks_model_and_transition_controls(qapp):
    dialog = _make_dialog(device=_make_device_ctrl())

    dialog.set_live_mode(enabled=True)

    for widget_type in (
        QDoubleSpinBox,
        QLineEdit,
        QSpinBox,
        QComboBox,
        EvalNumericField,
    ):
        assert all(
            not widget.isEnabled() for widget in dialog.findChildren(widget_type)
        )
    for text in ("Load", "Refresh files", "Add", "Remove", "Calibrate", "Save as…"):
        assert not _button(dialog, text).isEnabled()

    dialog.set_live_mode(enabled=False)

    assert _value_spin(dialog).isEnabled()
    assert _button(dialog, "Add").isEnabled()


def test_live_device_value_updates_read_only_control(qapp):
    dialog = _make_dialog()
    dialog.set_live_mode(enabled=True)

    dialog.set_live_device_value(0.25)

    assert _value_spin(dialog).value() == 0.25
    assert not _value_spin(dialog).isEnabled()
    assert "0.25" in _status(dialog)


def test_predictor_dialog_curve_and_controls_share_width_on_open_and_resize(qapp):
    dialog = _make_dialog()
    curves = dialog.findChild(PredictorCurveTabs)
    controls = _button(dialog, "Close").parentWidget()
    layout = dialog.layout()
    assert curves is not None and controls is not None and layout is not None

    dialog.show()
    qapp.processEvents()
    try:
        initial_curve_width = curves.width()
        assert abs(curves.width() - controls.width()) <= 2 * layout.spacing()

        dialog.resize(dialog.width() + 400, dialog.height())
        qapp.processEvents()

        assert curves.width() > initial_curve_width
        assert abs(curves.width() - controls.width()) <= 2 * layout.spacing()
    finally:
        dialog.close()


def test_predictor_dialog_default_tracked_count(qapp):
    dialog = _make_dialog()

    assert len(dialog.capture_state().tracked_transitions) == 5
    assert _table(dialog).rowCount() == 5


def test_predictor_dialog_persistent_reject_hides_without_finishing(qapp):
    device = _make_device_ctrl()
    dialog = _make_dialog(device=device, persistent_on_close=True)
    finished = MagicMock()
    dialog.finished.connect(finished)
    dialog.show()
    qapp.processEvents()

    dialog.reject()
    qapp.processEvents()

    assert not dialog.isVisible()
    finished.assert_not_called()


def test_persistent_predictor_hide_reopen_keeps_callbacks_until_done(qapp, tmp_path):
    from tests.gui._user_input import ComponentDriver

    host = QWidget()
    layout = QVBoxLayout(host)
    opener = QPushButton("Open predictor")
    layout.addWidget(opener)
    provider = MagicMock(return_value=None)
    dialog = _make_dialog(
        parent=host, parameter_provider=provider, persistent_on_close=True
    )
    opener.clicked.connect(dialog.show)
    expression_name = "predictor_calibration_frequency_expression"
    edit = dialog.findChild(QLineEdit, expression_name)
    assert edit is not None
    field = next(
        field
        for field in dialog.findChildren(EvalNumericField)
        if edit in field.findChildren(QLineEdit)
    )
    changes: list[None] = []
    field.raw_changed.connect(lambda: changes.append(None))
    ui = ComponentDriver(host)
    calibration_ui = ComponentDriver(field)
    try:
        host.show()
        ui.click("Open predictor")
        ui.screenshot(tmp_path / "01-open.png")
        calibration_ui.enter_text(
            object_name=expression_name, value="1+1", commit=False
        )
        assert calibration_ui.observe(object_name="eval.preview").text == "= 2.0"
        ui.screenshot(tmp_path / "02-expression.png")
        ui.click("Close")
        assert not calibration_ui.observe(object_name=expression_name).visible
        ui.screenshot(tmp_path / "03-hidden.png")
        ui.click("Open predictor")
        assert calibration_ui.text_input(object_name=expression_name).text == "1+1"
        ui.screenshot(tmp_path / "04-reopened.png")
        provider.reset_mock()
        changes.clear()
        calibration_ui.enter_text(
            object_name=expression_name, value="2+2", commit=False
        )
        assert calibration_ui.observe(object_name="eval.preview").text == "= 4.0"
        assert provider.called
        assert changes
        ui.screenshot(tmp_path / "05-reopened-edit.png")

        dialog.done(0)
        dialog.done(0)
        provider.reset_mock()
        changes.clear()
        # Keep the C++ field alive: a deleted-owner guard cannot satisfy this.
        edit.setText("3+3")
        provider.assert_not_called()
        assert changes == []
    finally:
        dialog.done(0)
        host.close()
        host.deleteLater()
        qapp.processEvents()


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
    dialog.set_flux(PredictorFlux(0.2, {"coil": 1.0}, {"coil": "A"}), {"coil": 0.1})

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

    dialog = PredictorDialog(ctrl, params=FluxoniumParams(4.0, 1.0, 1.0))
    dialog.set_flux(PredictorFlux(0.2, {"coil": 1.0}, {"coil": "A"}), {"coil": 0.1})

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
