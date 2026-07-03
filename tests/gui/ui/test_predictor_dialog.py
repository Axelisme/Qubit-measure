"""Tests for PredictorDialog — tracked-transitions table UX."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.session.events import DeviceChangedPayload
from zcu_tools.gui.session.services.device import DeviceEntry
from zcu_tools.gui.session.services.predictor import (
    PredictCurveResult,
    PredictMatrixCurveResult,
    PredictorLoadError,
    PredictorNotLoaded,
    SetModelParamsRequest,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.gui.session.ui.predictor_dialog import (
    _COL_FREQ,
    _COL_MAG_N,
    _COL_MAG_PHI,
    _DEFAULT_EC,
    _DEFAULT_EJ,
    _DEFAULT_EL,
    _DEFAULT_FLUX_BIAS,
    _DEFAULT_FLUX_HALF,
    _DEFAULT_FLUX_PERIOD,
    _DEFAULT_TRANSITIONS,
    PredictorDialog,
)


def _make_freq_result(
    transitions: list[tuple[int, int]], n: int = 20
) -> PredictCurveResult:
    """Build a minimal PredictCurveResult for the given transitions."""
    values = np.linspace(-0.5, 0.5, n)
    labels = tuple(f"{f}→{t}" for f, t in transitions)
    freqs = np.ones((len(transitions), n), dtype=np.float64) * 1000.0
    return PredictCurveResult(
        labels=labels,
        values=values,
        fluxs=values.copy(),
        freqs_mhz=freqs,
    )


def _make_mat_result(
    transitions: list[tuple[int, int]], n: int = 20
) -> PredictMatrixCurveResult:
    """Build a minimal PredictMatrixCurveResult for the given transitions."""
    values = np.linspace(-0.5, 0.5, n)
    labels = tuple(f"{f}→{t}" for f, t in transitions)
    mags = np.ones((len(transitions), n), dtype=np.float64) * 0.3
    return PredictMatrixCurveResult(
        labels=labels,
        values=values,
        fluxs=values.copy(),
        mags=mags,
    )


def _make_ctrl(
    *,
    has_predictor: bool = False,
    flux_half: float = 0.5,
    flux_period: float = 1.0,
    flux_bias: float = 0.0,
    ej: float = 4.0,
    ec: float = 1.0,
    el: float = 1.0,
    path: str | None = None,
) -> MagicMock:
    """Build a MagicMock predictor-control port pre-configured for dialog tests."""
    ctrl = MagicMock()
    if has_predictor:
        ctrl.get_predictor_info.return_value = {
            "path": path,
            "flux_bias": flux_bias,
            "flux_half": flux_half,
            "flux_period": flux_period,
            "EJ": ej,
            "EC": ec,
            "EL": el,
        }
    else:
        ctrl.get_predictor_info.return_value = None

    def _freq_side(req):  # type: ignore[no-untyped-def]
        return _make_freq_result(list(req.transitions))

    def _mat_side(req):  # type: ignore[no-untyped-def]
        return _make_mat_result(list(req.transitions))

    ctrl.predict_freq_curve.side_effect = _freq_side
    ctrl.predict_matrix_element_curve.side_effect = _mat_side
    ctrl.predict_freq.return_value = 1234.5
    ctrl.on_predictor_changed.return_value = MagicMock(name="unsubscribe_predictor")
    return ctrl


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
    ctrl.poll_device_info.return_value = None
    ctrl.on_device_changed.return_value = MagicMock(name="unsubscribe_device")
    return ctrl


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------


def test_predictor_dialog_default_field_values_no_predictor(qapp):
    """With no predictor loaded, the six spinboxes show the documented defaults."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    assert dialog._ej_spin.value() == pytest.approx(_DEFAULT_EJ)  # 4.0
    assert dialog._ec_spin.value() == pytest.approx(_DEFAULT_EC)  # 1.0
    assert dialog._el_spin.value() == pytest.approx(_DEFAULT_EL)  # 1.0
    assert dialog._flux_half_spin.value() == pytest.approx(_DEFAULT_FLUX_HALF)  # 0.0
    assert dialog._flux_period_spin.value() == pytest.approx(
        _DEFAULT_FLUX_PERIOD
    )  # 0.005
    assert dialog._flux_bias_spin.value() == pytest.approx(_DEFAULT_FLUX_BIAS)  # 0.0


def test_live_mode_locks_model_marker_and_transition_controls(qapp):
    ctrl = _make_ctrl(has_predictor=True, flux_bias=0.12)
    device = _make_device_ctrl()
    dialog = PredictorDialog(ctrl, device=device)

    dialog.set_live_mode(True)

    for widget in (
        dialog._params_path_edit,
        dialog._load_btn,
        dialog._browse_btn,
        dialog._ej_spin,
        dialog._ec_spin,
        dialog._el_spin,
        dialog._flux_half_spin,
        dialog._flux_period_spin,
        dialog._flux_bias_spin,
        dialog._apply_btn,
        dialog._predict_value_spin,
        dialog._add_from_spin,
        dialog._add_to_spin,
        dialog._add_btn,
        dialog._remove_btn,
        dialog._device_combo,
        dialog._device_refresh_btn,
        dialog._device_read_btn,
    ):
        assert widget is not None
        assert not widget.isEnabled()
    assert all(not canvas._interaction_enabled for canvas in dialog._all_canvases)

    dialog._on_apply_model_params()
    ctrl.set_predictor_model_params.assert_not_called()

    dialog.set_live_mode(False)

    assert dialog._apply_btn.isEnabled()
    assert dialog._predict_value_spin.isEnabled()
    assert dialog._device_read_btn is not None
    assert not dialog._device_read_btn.isEnabled()
    assert all(canvas._interaction_enabled for canvas in dialog._all_canvases)


def test_live_device_value_updates_marker_without_enabling_manual_drag(qapp):
    ctrl = _make_ctrl(has_predictor=True)
    dialog = PredictorDialog(ctrl)
    dialog.set_live_mode(True)

    dialog.set_live_device_value(0.25)

    assert dialog._predict_value_spin.value() == pytest.approx(0.25)
    assert all(
        canvas._marker_value == pytest.approx(0.25) for canvas in dialog._all_canvases
    )

    dialog._on_canvas_follow(0.5)

    assert dialog._predict_value_spin.value() == pytest.approx(0.25)


def test_predictor_dialog_param_spinboxes_do_not_pad_trailing_zeros(qapp):
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    assert dialog._ej_spin.text() == "4.0"
    assert dialog._flux_period_spin.text() == "0.005"


# ---------------------------------------------------------------------------
# Init / load
# ---------------------------------------------------------------------------


def test_predictor_dialog_init_populates_fields_from_info(qapp):
    """With a predictor loaded at init, the six spinboxes reflect it."""
    ctrl = _make_ctrl(
        has_predictor=True,
        path="/p.json",
        ej=4.2,
        ec=1.1,
        el=0.7,
        flux_half=0.3,
        flux_period=0.8,
        flux_bias=0.05,
    )
    dialog = PredictorDialog(ctrl)

    assert dialog._ej_spin.value() == pytest.approx(4.2)
    assert dialog._ec_spin.value() == pytest.approx(1.1)
    assert dialog._el_spin.value() == pytest.approx(0.7)
    assert dialog._flux_half_spin.value() == pytest.approx(0.3)
    assert dialog._flux_period_spin.value() == pytest.approx(0.8)
    assert dialog._flux_bias_spin.value() == pytest.approx(0.05)
    assert dialog._params_path_edit.text() == "/p.json"


# ---------------------------------------------------------------------------
# Browse button — populates six fields AND auto-applies (installs)
# ---------------------------------------------------------------------------


def test_predictor_dialog_browse_populates_six_fields(qapp):
    """Browse picks a file and populates EJ/EC/EL/flux_half/flux_period spinboxes."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    fake_params = SetModelParamsRequest(
        EJ=5.5, EC=1.3, EL=0.6, flux_half=0.4, flux_period=0.7
    )
    with (
        patch(
            "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
            return_value=("/chosen/params.json", ""),
        ),
        patch(
            "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
            return_value=fake_params,
        ),
    ):
        dialog._on_browse_file()

    assert dialog._params_path_edit.text() == "/chosen/params.json"
    assert dialog._ej_spin.value() == pytest.approx(5.5)
    assert dialog._ec_spin.value() == pytest.approx(1.3)
    assert dialog._el_spin.value() == pytest.approx(0.6)
    assert dialog._flux_half_spin.value() == pytest.approx(0.4)
    assert dialog._flux_period_spin.value() == pytest.approx(0.7)


def test_predictor_dialog_browse_auto_applies(qapp):
    """Browse must install the predictor (auto-apply) using the loaded fields."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    fake_params = SetModelParamsRequest(
        EJ=5.5, EC=1.3, EL=0.6, flux_half=0.4, flux_period=0.7
    )
    with (
        patch(
            "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
            return_value=("/chosen/params.json", ""),
        ),
        patch(
            "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
            return_value=fake_params,
        ),
    ):
        dialog._on_browse_file()

    ctrl.set_predictor_model_params.assert_called_once()
    (req,) = ctrl.set_predictor_model_params.call_args.args
    assert isinstance(req, SetModelParamsRequest)
    assert req.EJ == pytest.approx(5.5)
    assert req.flux_period == pytest.approx(0.7)


def test_predictor_dialog_browse_install_failure_keeps_fields_and_shows_error(qapp):
    """A failed auto-apply (service error) keeps the fields populated, no crash."""
    ctrl = _make_ctrl(has_predictor=False)
    ctrl.set_predictor_model_params.side_effect = PredictorLoadError("bad model")
    dialog = PredictorDialog(ctrl)

    fake_params = SetModelParamsRequest(
        EJ=5.5, EC=1.3, EL=0.6, flux_half=0.4, flux_period=0.7
    )
    with (
        patch(
            "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
            return_value=("/chosen/params.json", ""),
        ),
        patch(
            "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
            return_value=fake_params,
        ),
    ):
        dialog._on_browse_file()

    # Fields still populated despite the failed install.
    assert dialog._ej_spin.value() == pytest.approx(5.5)
    assert "bad model" in dialog._status_label.text()


def test_predictor_dialog_browse_cancel_is_noop(qapp):
    """Cancelling the file dialog (empty path) leaves the fields unchanged."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    before_ej = dialog._ej_spin.value()

    with patch(
        "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ):
        dialog._on_browse_file()

    assert dialog._ej_spin.value() == before_ej
    ctrl.set_predictor_model_params.assert_not_called()


def test_predictor_dialog_browse_error_shown_in_status(qapp):
    """A read error during Browse is surfaced in the status label, fields untouched."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    with (
        patch(
            "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
            return_value=("/bad/params.json", ""),
        ),
        patch(
            "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
            side_effect=PredictorLoadError("no fluxdep_fit"),
        ),
    ):
        dialog._on_browse_file()

    assert "no fluxdep_fit" in dialog._status_label.text()
    # Read failure means no install attempt.
    ctrl.set_predictor_model_params.assert_not_called()


def test_predictor_dialog_typed_path_load_populates_and_applies(qapp):
    """Typing a params.json path exercises the same load+auto-apply path as Browse."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    dialog._params_path_edit.setText("/typed/params.json")

    fake_params = SetModelParamsRequest(
        EJ=6.0, EC=1.4, EL=0.8, flux_half=0.2, flux_period=0.9
    )
    with patch(
        "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
        return_value=fake_params,
    ) as read_params:
        dialog._on_load_path_clicked()

    read_params.assert_called_once_with("/typed/params.json")
    assert dialog._ej_spin.value() == pytest.approx(6.0)
    ctrl.set_predictor_model_params.assert_called_once()
    (req,) = ctrl.set_predictor_model_params.call_args.args
    assert req.EJ == pytest.approx(6.0)
    assert req.flux_period == pytest.approx(0.9)


def test_predictor_dialog_typed_path_load_requires_path(qapp):
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._on_load_path_clicked()

    assert "params.json path" in dialog._status_label.text()
    ctrl.set_predictor_model_params.assert_not_called()


# ---------------------------------------------------------------------------
# Apply button
# ---------------------------------------------------------------------------


def test_predictor_dialog_apply_installs_from_fields(qapp):
    """Apply builds a SetModelParamsRequest from the fields and installs it."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._ej_spin.setValue(4.0)
    dialog._ec_spin.setValue(1.0)
    dialog._el_spin.setValue(1.0)
    dialog._flux_half_spin.setValue(0.25)
    dialog._flux_period_spin.setValue(0.9)
    dialog._flux_bias_spin.setValue(0.05)

    dialog._on_apply_model_params()

    ctrl.set_predictor_model_params.assert_called_once()
    (req,) = ctrl.set_predictor_model_params.call_args.args
    assert isinstance(req, SetModelParamsRequest)
    assert req.EJ == pytest.approx(4.0)
    assert req.EC == pytest.approx(1.0)
    assert req.EL == pytest.approx(1.0)
    assert req.flux_half == pytest.approx(0.25)
    assert req.flux_period == pytest.approx(0.9)
    assert req.flux_bias == pytest.approx(0.05)


def test_predictor_dialog_apply_zero_period_guarded(qapp):
    """flux_period == 0 must NOT install — the dialog fast-fails with a message."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._flux_period_spin.setValue(0.0)
    dialog._on_apply_model_params()

    ctrl.set_predictor_model_params.assert_not_called()
    assert "flux_period" in dialog._status_label.text()


def test_predictor_dialog_apply_surfaces_service_error(qapp):
    """A PredictorLoadError from the controller is shown in the status label."""
    ctrl = _make_ctrl(has_predictor=False)
    ctrl.set_predictor_model_params.side_effect = PredictorLoadError("bad model")
    dialog = PredictorDialog(ctrl)

    dialog._flux_period_spin.setValue(1.0)
    dialog._on_apply_model_params()

    assert "bad model" in dialog._status_label.text()


# ---------------------------------------------------------------------------
# Default tracked transitions — table layout
# ---------------------------------------------------------------------------


def test_predictor_dialog_default_tracked_count(qapp):
    """Dialog initialises with 5 default tracked transitions, table has 5 rows."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


def test_predictor_dialog_table_has_4_columns(qapp):
    """Table must have 4 columns: Transition | f (MHz) | |n| | |phi|."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert dialog._table.columnCount() == 4
    headers: list[str] = []
    for i in range(dialog._table.columnCount()):
        item = dialog._table.horizontalHeaderItem(i)
        assert item is not None
        headers.append(item.text())
    assert headers == ["Transition", "f (MHz)", "|n|", "|phi|"]


def test_predictor_dialog_right_side_has_three_tabs(qapp):
    """Right panel is a QTabWidget with 3 tabs: Frequency, |n|, |phi|."""
    from qtpy.QtWidgets import QTabWidget  # type: ignore[attr-defined]

    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert isinstance(dialog._tab_widget, QTabWidget)
    assert dialog._tab_widget.count() == 3
    assert dialog._tab_widget.tabText(0) == "Frequency"
    assert dialog._tab_widget.tabText(1) == "|n|"
    assert dialog._tab_widget.tabText(2) == "|phi|"


def test_predictor_dialog_device_value_label_no_unit(qapp):
    """The predict-position control is labelled 'Device value' with no unit."""
    from qtpy.QtWidgets import QLabel  # type: ignore[attr-defined]

    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert "Device value:" in labels
    # No Ampere / "(A)" unit on any label.
    assert not any("(A)" in t for t in labels)
    assert not any("Flux value" in t for t in labels)


def test_predictor_dialog_optional_device_facet_no_controls(qapp):
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    assert dialog._dev is None
    assert dialog._device_combo is None


def test_predictor_dialog_device_and_transition_controls_are_separate_rows(qapp):
    ctrl = _make_ctrl(has_predictor=False)
    device = _make_device_ctrl(
        entries=[DeviceEntry("flux", "YOKOGS200", DeviceStatus.CONNECTED.value)],
        cached={"flux": 0.1},
    )

    dialog = PredictorDialog(ctrl, device=device)

    assert dialog._device_combo is not None
    assert dialog._value_controls_row.indexOf(dialog._predict_value_spin) >= 0
    assert dialog._value_controls_row.indexOf(dialog._device_combo) >= 0
    assert dialog._value_controls_row.indexOf(dialog._add_from_spin) == -1
    assert dialog._transition_controls_row.indexOf(dialog._add_from_spin) >= 0
    assert dialog._transition_controls_row.indexOf(dialog._add_to_spin) >= 0
    assert dialog._transition_controls_row.indexOf(dialog._device_combo) == -1


def test_predictor_dialog_device_selector_lists_connected_cached_values(qapp):
    ctrl = _make_ctrl(has_predictor=False)
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

    dialog = PredictorDialog(ctrl, device=device)

    combo = dialog._device_combo
    assert combo is not None
    assert [combo.itemData(i) for i in range(combo.count())] == [
        None,
        "flux",
        "fake",
    ]
    assert "YOKOGS200" in combo.itemText(1)
    assert "(A)" in combo.itemText(1)


def test_predictor_dialog_read_selected_device_updates_value_immediately(qapp):
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    device = _make_device_ctrl(
        entries=[DeviceEntry("flux", "FakeDevice", DeviceStatus.CONNECTED.value)],
        cached={"flux": 0.1},
        live={"flux": FakeDeviceInfo(address="none", value=0.456)},
    )
    dialog = PredictorDialog(ctrl, device=device)
    combo = dialog._device_combo
    assert combo is not None
    combo.setCurrentIndex(combo.findData("flux"))

    with patch.object(dialog, "_update_value_columns") as update_columns:
        dialog._on_read_device_clicked()

    assert dialog._predict_value_spin.value() == pytest.approx(0.456)
    update_columns.assert_called_once_with()
    device.get_device_info.assert_called_with("flux")


def test_predictor_dialog_show_refreshes_selected_cached_value_and_columns(qapp):
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    cached: dict[str, float | None] = {"flux": 0.1}
    device = _make_device_ctrl(
        entries=[DeviceEntry("flux", "FakeDevice", DeviceStatus.CONNECTED.value)],
        cached=cached,
    )
    dialog = PredictorDialog(ctrl, device=device)
    combo = dialog._device_combo
    assert combo is not None
    combo.setCurrentIndex(combo.findData("flux"))
    ctrl.predict_freq.reset_mock()
    ctrl.predict_freq_curve.reset_mock()
    device.poll_device_info.reset_mock()
    cached["flux"] = 0.3

    dialog.showEvent(None)

    assert dialog._predict_value_spin.value() == pytest.approx(0.3)
    device.poll_device_info.assert_called_with("flux")
    ctrl.predict_freq.assert_called()
    ctrl.predict_freq_curve.assert_not_called()


def test_predictor_dialog_selected_device_changed_applies_cached_value(qapp):
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    cached: dict[str, float | None] = {"flux": 0.1, "other": 0.2}
    device = _make_device_ctrl(
        entries=[
            DeviceEntry("flux", "FakeDevice", DeviceStatus.CONNECTED.value),
            DeviceEntry("other", "FakeDevice", DeviceStatus.CONNECTED.value),
        ],
        cached=cached,
    )
    dialog = PredictorDialog(ctrl, device=device)
    combo = dialog._device_combo
    assert combo is not None
    combo.setCurrentIndex(combo.findData("flux"))
    dialog.show()
    qapp.processEvents()

    cached["other"] = 0.9
    dialog._on_device_changed(DeviceChangedPayload(name="other"))
    assert dialog._predict_value_spin.value() == pytest.approx(0.1)

    cached["flux"] = 0.7
    dialog._on_device_changed(DeviceChangedPayload(name="flux"))
    assert dialog._predict_value_spin.value() == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Predictor active/inactive indicator
# ---------------------------------------------------------------------------


def test_predictor_dialog_active_label_inactive_when_no_predictor(qapp):
    """With no predictor installed, the active label reads 'not installed'."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert "not installed" in dialog._active_label.text()


def test_predictor_dialog_active_label_active_when_predictor_loaded(qapp):
    """With a predictor installed at init, the active label reads 'active'."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    assert "active" in dialog._active_label.text()


def test_predictor_dialog_active_label_updates_on_apply(qapp):
    """After a successful Apply the active label flips to 'active'."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert "not installed" in dialog._active_label.text()

    # Simulate the install making a predictor available afterwards.
    ctrl.get_predictor_info.return_value = {
        "path": None,
        "flux_bias": 0.0,
        "flux_half": 0.5,
        "flux_period": 1.0,
        "EJ": 4.0,
        "EC": 1.0,
        "EL": 1.0,
    }
    dialog._flux_period_spin.setValue(1.0)
    dialog._on_apply_model_params()

    assert "active" in dialog._active_label.text()


def test_predictor_dialog_active_label_updates_on_clear_event(qapp):
    """A cleared predictor event flips the active label back to 'not installed'."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    assert "active" in dialog._active_label.text()

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    assert "not installed" in dialog._active_label.text()


def test_predictor_dialog_default_predict_freq_curve_call(qapp):
    """With predictor loaded at init, predict_freq_curve is called with the 5 defaults."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_called()
    last_req = ctrl.predict_freq_curve.call_args.args[0]
    assert set(last_req.transitions) == set(map(tuple, _DEFAULT_TRANSITIONS))


def test_predictor_dialog_default_predict_matrix_element_curve_call(qapp):
    """With predictor loaded at init, predict_matrix_element_curve covers both operators.

    _refresh_curves now calls the full-set curve once per operator (n and phi);
    _update_value_columns then calls it once per (transition, operator) for the
    single-point column fill.  We verify each operator got a full-set call.
    """
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    PredictorDialog(ctrl)
    ctrl.predict_matrix_element_curve.assert_called()
    all_reqs = [
        call.args[0] for call in ctrl.predict_matrix_element_curve.call_args_list
    ]
    full_default = set(map(tuple, _DEFAULT_TRANSITIONS))
    for op in ("n", "phi"):
        full_call = next(
            (
                r
                for r in all_reqs
                if r.operator == op and set(r.transitions) == full_default
            ),
            None,
        )
        assert full_call is not None, f"No full-set curve call for operator {op!r}"
    # Both operators must appear and no others.
    assert {r.operator for r in all_reqs} == {"n", "phi"}


def test_predictor_dialog_init_no_predictor_no_curve_call(qapp):
    """With no predictor at init, neither curve function is called."""
    ctrl = _make_ctrl(has_predictor=False)
    PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_not_called()
    ctrl.predict_matrix_element_curve.assert_not_called()


def test_predictor_dialog_persistent_reject_hides_without_finishing(qapp):
    """Persistent host mode hides on close so cached curves and subscription remain."""
    from qtpy.QtWidgets import QApplication

    ctrl = _make_ctrl(has_predictor=False)
    unsubscribe = ctrl.on_predictor_changed.return_value
    dialog = PredictorDialog(ctrl, persistent_on_close=True)
    finished = MagicMock()
    dialog.finished.connect(finished)
    dialog.show()
    QApplication.processEvents()

    dialog.reject()
    QApplication.processEvents()

    assert dialog.isVisible() is False
    finished.assert_not_called()
    unsubscribe.assert_not_called()


def test_predictor_dialog_init_with_predictor_all_canvases_have_axes(qapp):
    """After init with a loaded predictor, all three canvases have axes (curves drawn)."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    assert len(dialog._freq_canvas.figure.get_axes()) > 0
    assert len(dialog._mat_n_canvas.figure.get_axes()) > 0
    assert len(dialog._mat_phi_canvas.figure.get_axes()) > 0


# ---------------------------------------------------------------------------
# Add transition
# ---------------------------------------------------------------------------


def test_predictor_dialog_add_valid_transition(qapp):
    """Adding a valid, non-duplicate transition increments row count and calls refresh."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    dialog._add_from_spin.setValue(2)
    dialog._add_to_spin.setValue(4)
    dialog._on_add_clicked()

    assert len(dialog._tracked) == 6
    assert dialog._table.rowCount() == 6
    assert (2, 4) in dialog._tracked
    assert ctrl.predict_freq_curve.call_count > initial_count


def test_predictor_dialog_add_invalid_from_ge_to(qapp):
    """from >= to: not added, row count unchanged, no crash."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._add_from_spin.setValue(3)
    dialog._add_to_spin.setValue(2)
    dialog._on_add_clicked()

    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


def test_predictor_dialog_add_duplicate_transition(qapp):
    """Adding a duplicate (0,1): not added, row count unchanged."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._add_from_spin.setValue(0)
    dialog._add_to_spin.setValue(1)
    dialog._on_add_clicked()

    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


# ---------------------------------------------------------------------------
# Delete transition (via _delete_transition internal API — still used internally)
# ---------------------------------------------------------------------------


def test_predictor_dialog_delete_transition(qapp):
    """Deleting a tracked transition removes it and triggers a curve refresh."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    transition_to_delete = dialog._tracked[0]
    dialog._delete_transition(transition_to_delete)

    assert len(dialog._tracked) == 4
    assert dialog._table.rowCount() == 4
    assert transition_to_delete not in dialog._tracked
    assert ctrl.predict_freq_curve.call_count > initial_count


def test_predictor_dialog_delete_all_transitions_clears_all_canvases(qapp):
    """Deleting all tracked transitions results in an empty table and cleared canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    for t in list(dialog._tracked):
        dialog._delete_transition(t)

    assert dialog._table.rowCount() == 0
    # All canvases clear() resets marker_value.
    assert dialog._freq_canvas._marker_value is None
    assert dialog._mat_n_canvas._marker_value is None
    assert dialog._mat_phi_canvas._marker_value is None


# ---------------------------------------------------------------------------
# Multi-select Remove button
# ---------------------------------------------------------------------------


def test_predictor_dialog_remove_selected_single_row(qapp):
    """Selecting one row and clicking Remove deletes that transition."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    target = dialog._tracked[1]  # (0, 2)
    dialog._table.selectRow(1)
    dialog._on_remove_selected()

    assert target not in dialog._tracked
    assert len(dialog._tracked) == 4
    assert dialog._table.rowCount() == 4
    assert ctrl.predict_freq_curve.call_count > initial_count


def test_predictor_dialog_remove_selected_multiple_rows(qapp):
    """Selecting multiple rows and clicking Remove deletes all selected transitions."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    # Select rows 0 and 2.
    targets = [dialog._tracked[0], dialog._tracked[2]]
    dialog._table.selectRow(0)
    # Extend selection to row 2 via selectRow on the QTableWidget
    # (selectRow replaces selection; use setRangeSelected for multi-select).
    from qtpy.QtWidgets import QTableWidgetSelectionRange  # type: ignore[attr-defined]

    dialog._table.setRangeSelected(
        QTableWidgetSelectionRange(0, 0, 0, dialog._table.columnCount() - 1), True
    )
    dialog._table.setRangeSelected(
        QTableWidgetSelectionRange(2, 0, 2, dialog._table.columnCount() - 1), True
    )
    dialog._on_remove_selected()

    for t in targets:
        assert t not in dialog._tracked
    assert len(dialog._tracked) == 3
    assert dialog._table.rowCount() == 3


def test_predictor_dialog_remove_no_selection_is_noop(qapp):
    """Clicking Remove with nothing selected does nothing."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_freq_count = ctrl.predict_freq_curve.call_count

    dialog._table.clearSelection()
    dialog._on_remove_selected()

    assert len(dialog._tracked) == 5
    assert ctrl.predict_freq_curve.call_count == initial_freq_count


def test_predictor_dialog_remove_all_clears_canvases(qapp):
    """Removing all transitions via multi-select Remove clears all canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    from qtpy.QtWidgets import QTableWidgetSelectionRange  # type: ignore[attr-defined]

    n = dialog._table.rowCount()
    dialog._table.setRangeSelected(
        QTableWidgetSelectionRange(0, 0, n - 1, dialog._table.columnCount() - 1), True
    )
    dialog._on_remove_selected()

    assert dialog._table.rowCount() == 0
    assert dialog._freq_canvas._marker_value is None
    assert dialog._mat_n_canvas._marker_value is None
    assert dialog._mat_phi_canvas._marker_value is None


# ---------------------------------------------------------------------------
# Spinbox / debounce → column update
# ---------------------------------------------------------------------------


def test_predictor_dialog_spinbox_change_does_not_recompute_curves(qapp):
    """Changing the Device value spinbox must NOT call predict_freq_curve again."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    dialog._on_spinbox_changed(0.3)

    assert ctrl.predict_freq_curve.call_count == initial_count


def test_predictor_dialog_spinbox_change_updates_all_markers(qapp):
    """Changing spinbox calls set_marker on all three canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with (
        patch.object(dialog._freq_canvas, "set_marker") as freq_mock,
        patch.object(dialog._mat_n_canvas, "set_marker") as mat_n_mock,
        patch.object(dialog._mat_phi_canvas, "set_marker") as mat_phi_mock,
    ):
        dialog._on_spinbox_changed(0.3)

    freq_mock.assert_called_with(0.3)
    mat_n_mock.assert_called_with(0.3)
    mat_phi_mock.assert_called_with(0.3)


def test_predictor_dialog_value_columns_updated_after_debounce(qapp):
    """After _update_value_columns, each row's f / |n| / |phi| cells are filled."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    dialog._predict_value_spin.setValue(0.1)
    dialog._update_value_columns()

    # Each row should now show a numeric freq, |n| and |phi| (not "—").
    for row in range(dialog._table.rowCount()):
        freq_item = dialog._table.item(row, _COL_FREQ)
        assert freq_item is not None
        assert freq_item.text() != "—"
        mag_n_item = dialog._table.item(row, _COL_MAG_N)
        assert mag_n_item is not None
        assert mag_n_item.text() != "—"
        mag_phi_item = dialog._table.item(row, _COL_MAG_PHI)
        assert mag_phi_item is not None
        assert mag_phi_item.text() != "—"


def test_predictor_dialog_value_columns_fill_both_operators(qapp):
    """_update_value_columns fills |n| and |phi| via per-operator single-point calls."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    ctrl.predict_matrix_element_curve.reset_mock()

    dialog._predict_value_spin.setValue(0.1)
    dialog._update_value_columns()

    operators = {
        call.args[0].operator
        for call in ctrl.predict_matrix_element_curve.call_args_list
    }
    assert operators == {"n", "phi"}


def test_predictor_dialog_columns_dash_when_not_loaded(qapp):
    """predict_freq raising PredictorNotLoaded → all value columns show '—', no crash."""
    ctrl = _make_ctrl(has_predictor=False)
    ctrl.predict_freq.side_effect = PredictorNotLoaded("no predictor")
    ctrl.predict_matrix_element_curve.side_effect = PredictorNotLoaded("no predictor")
    dialog = PredictorDialog(ctrl)

    dialog._update_value_columns()

    for row in range(dialog._table.rowCount()):
        for col in (_COL_FREQ, _COL_MAG_N, _COL_MAG_PHI):
            item = dialog._table.item(row, col)
            assert item is not None and item.text() == "—"


def test_predictor_dialog_canvas_follow_updates_spinbox_no_recompute(qapp):
    """on_follow must update the spinbox without triggering a full curve recompute."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    freq_curve_count_before = ctrl.predict_freq_curve.call_count

    dialog._on_canvas_follow(0.77)

    assert dialog._predict_value_spin.value() == pytest.approx(0.77)
    # on_follow only debounces the column update — no full curve recompute.
    assert ctrl.predict_freq_curve.call_count == freq_curve_count_before


def test_predictor_dialog_canvas_lock_immediate_column_update(qapp):
    """on_lock updates the spinbox and recomputes the columns immediately (no recompute)."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    freq_curve_count_before = ctrl.predict_freq_curve.call_count

    dialog._on_canvas_lock(0.42)

    assert dialog._predict_value_spin.value() == pytest.approx(0.42)
    # The immediate update fills the freq column via the scalar predict_freq...
    ctrl.predict_freq.assert_called()
    # ...without re-running the full curve computation.
    assert ctrl.predict_freq_curve.call_count == freq_curve_count_before


# ---------------------------------------------------------------------------
# Table selection → all canvas highlights
# ---------------------------------------------------------------------------


def test_predictor_dialog_row_select_calls_set_highlight_all_canvases(qapp):
    """Selecting a table row calls set_highlight on ALL three canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with (
        patch.object(dialog._freq_canvas, "set_highlight") as freq_hi,
        patch.object(dialog._mat_n_canvas, "set_highlight") as mat_n_hi,
        patch.object(dialog._mat_phi_canvas, "set_highlight") as mat_phi_hi,
    ):
        dialog._table.selectRow(1)
        dialog._on_selection_changed()

    expected = dialog._tracked[1]
    freq_hi.assert_called_with(expected)
    mat_n_hi.assert_called_with(expected)
    mat_phi_hi.assert_called_with(expected)


def test_predictor_dialog_no_selection_calls_set_highlight_none_all(qapp):
    """Clearing selection calls set_highlight(None) on all three canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with (
        patch.object(dialog._freq_canvas, "set_highlight") as freq_hi,
        patch.object(dialog._mat_n_canvas, "set_highlight") as mat_n_hi,
        patch.object(dialog._mat_phi_canvas, "set_highlight") as mat_phi_hi,
    ):
        dialog._table.clearSelection()
        dialog._on_selection_changed()

    freq_hi.assert_called_with(None)
    mat_n_hi.assert_called_with(None)
    mat_phi_hi.assert_called_with(None)


# ---------------------------------------------------------------------------
# Predictor events
# ---------------------------------------------------------------------------


def test_predictor_dialog_on_predictor_changed_event_refreshes_curves(qapp):
    """PredictorChangedPayload with a loaded predictor must call predict_freq_curve."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert ctrl.predict_freq_curve.call_count == 0

    ctrl.get_predictor_info.return_value = {
        "path": "/new.json",
        "flux_bias": 0.0,
        "flux_half": 0.5,
        "flux_period": 1.0,
        "EJ": 4.0,
        "EC": 1.0,
        "EL": 1.0,
    }
    dialog._on_predictor_changed(object())

    assert ctrl.predict_freq_curve.call_count >= 1
    assert ctrl.predict_matrix_element_curve.call_count >= 1


def test_predictor_dialog_on_predictor_changed_cleared_clears_all_canvases(qapp):
    """Predictor event with no predictor must blank all canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    assert dialog._freq_canvas._marker_line is None
    assert dialog._mat_n_canvas._marker_line is None
    assert dialog._mat_phi_canvas._marker_line is None


def test_predictor_dialog_on_predictor_changed_cleared_resets_all_columns(qapp):
    """After a cleared predictor event, all f / |n| / |phi| cells show '—'."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    for row in range(dialog._table.rowCount()):
        for col in (_COL_FREQ, _COL_MAG_N, _COL_MAG_PHI):
            item = dialog._table.item(row, col)
            assert item is not None and item.text() == "—"


def test_predictor_dialog_predict_curve_error_shown_in_status(qapp):
    """If predict_freq_curve raises, the error is shown in the status label."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    ctrl.predict_freq_curve.side_effect = PredictorNotLoaded("oops")
    dialog = PredictorDialog(ctrl)

    assert "oops" in dialog._status_label.text()
