"""Detached energy/frame trials, nullable drafts, cache and calibration displays."""

import json
import os
from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.predictor_file import PredictorFlux
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.predictor import PredictorInputError
from zcu_tools.simulate.transmon import TransmonParams

import tests.gui._component_fakes as components
from tests.gui._predictor_fakes import FailingPredictorService
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.session.ui._predictor_support import SyntheticCurves


@pytest.mark.parametrize(
    "trial, name",
    [(FluxoniumParams(5.0, 1.0, None), "EL"), (TransmonParams(20.0, 1.0, None), "d")],
    ids=["fluxonium", "transmon"],
)
def test_blank_parameters_can_be_completed_cleared_and_corrected(
    qapp, tmp_path: Path, trial, name
) -> None:
    dialog = PredictorDialog(PredictorControlFacet(SyntheticCurves()), params=trial)
    dialog.set_flux(PredictorFlux(0.2, {"coil": 1.0}, {"coil": "A"}), {"coil": 0.0})
    evidence = (
        Path(os.environ.get("PREDICTOR_BLANK_EVIDENCE", str(tmp_path))) / trial.model
    )
    field_name = f"predictor_input_{name}"
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.screenshot(evidence / "01-missing.png")
        field = ui.observe(object_name=field_name)
        assert field.visible and field.enabled and field.text == ""
        assert (
            f"Missing parameters: {name}"
            in ui.observe(object_name="predictor_status").text
        )
        ui.enter_text(object_name=field_name, value="0.1", commit=False)
        qt_until(
            lambda: any(axis.visible_curves > 0 for axis in ui.plots().axes),
            label="completed predictor curves",
        )
        ui.screenshot(evidence / "02-complete.png")
        assert (
            "Missing parameters" not in ui.observe(object_name="predictor_status").text
        )
        ui.enter_text(object_name=field_name, value="", commit=False)
        ui.click(object_name="predictor_offset")
        ui.screenshot(evidence / "03-cleared.png")
        assert ui.observe(object_name=field_name).text == ""
        assert (
            f"Missing parameters: {name}"
            in ui.observe(object_name="predictor_status").text
        )
        assert all(axis.visible_curves == 0 for axis in ui.plots().axes)
        for index, invalid in enumerate(("oops", "1e", "nan", "inf"), start=4):
            ui.enter_text(object_name=field_name, value=invalid, commit=False)
            ui.screenshot(evidence / f"{index:02}-invalid-{invalid}.png")
            field = ui.observe(object_name=field_name)
            assert field.text == invalid
            assert name in field.tooltip and "finite number" in field.tooltip
            assert name in ui.observe(object_name="predictor_status").text
            assert not ui.observe("Calibrate").enabled
            rows = ui.table_rows(object_name="predictor_transitions")
            assert rows and rows[0].columns[1] == "—"
            assert all(value in ("—", "") for row in rows for value in row.columns[1:])
            assert all(axis.visible_curves == 0 for axis in ui.plots().axes)
        ui.enter_text(object_name=field_name, value="0.1", commit=False)
        qt_until(
            lambda: any(axis.visible_curves > 0 for axis in ui.plots().axes),
            label="corrected predictor curves",
        )
        ui.screenshot(evidence / "08-corrected.png")
        assert "finite number" not in ui.observe(object_name=field_name).tooltip
        ui.click("Close")
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.parametrize(
    "flux, values, status",
    [
        (None, {}, "Load a predictor file to set the flux frame"),
        (PredictorFlux(0.2, {}, {}), {}, "Select a horizontal axis instrument"),
        (
            PredictorFlux(0.2, {"coil": 1.0}, {"coil": "A"}),
            {},
            "coil: enter a finite number",
        ),
        (
            PredictorFlux(0.2, {"coil": 1.0, "bias": 2.0}, {"coil": "A", "bias": "V"}),
            {"coil": 0.0},
            "bias: enter a finite number",
        ),
    ],
    ids=["missing-frame", "missing-axis", "missing-axis-value", "missing-fixed-value"],
)
def test_curve_status_prioritizes_coordinates_but_scalar_prioritizes_energy(
    qapp,
    tmp_path: Path,
    flux: PredictorFlux | None,
    values: dict[str, float],
    status: str,
) -> None:
    dialog = PredictorDialog(
        PredictorControlFacet(SyntheticCurves()),
        params=FluxoniumParams(float("nan"), 1.0, 1.0),
    )
    if flux is not None:
        dialog.set_flux(flux, values)
    evidence = Path(os.environ.get("PREDICTOR_PRECEDENCE_EVIDENCE", str(tmp_path)))
    case = (
        "missing-frame"
        if flux is None
        else "missing-axis"
        if not flux.period
        else "missing-axis-value"
        if not values
        else "missing-fixed-value"
    )
    energy_error = "EJ: enter a finite number or leave blank."
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.screenshot(evidence / case / "01-invalid-energy-and-coordinates.png")
        assert ui.observe(object_name="predictor_status").text == status
        assert ui.observe(object_name="predictor_input_EJ").tooltip == energy_error
        assert all(axis.visible_curves == 0 for axis in ui.plots().axes)
        with pytest.raises(PredictorInputError, match="^EJ:") as failure:
            dialog.predict_frequency()
        assert str(failure.value) == energy_error

        ui.enter_text(object_name="predictor_input_EJ", value="5.0", commit=False)
        qt_until(
            lambda: ui.observe(object_name="predictor_status").text == status,
            label="coordinate error after energy correction",
        )
        ui.screenshot(evidence / case / "02-energy-corrected.png")
        assert (
            ui.observe(object_name="predictor_input_EJ").tooltip
            == "Leave blank for unknown."
        )
        assert all(axis.visible_curves == 0 for axis in ui.plots().axes)
        ui.click("Close")
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_dialog_logs_unexpected_value_error_with_traceback(qapp, caplog) -> None:
    failure = ValueError("solver invariant")
    service = FailingPredictorService(failure)
    dialog = PredictorDialog(
        PredictorControlFacet(service), params=FluxoniumParams(5.0, 1.0, 0.5)
    )
    with caplog.at_level("ERROR"):
        dialog.set_flux(PredictorFlux(0.2, {"coil": 1.0}, {"coil": "A"}), {"coil": 0.1})
    try:
        assert service.requests
        assert any(
            record.exc_info is not None and record.exc_info[1] is failure
            for record in caplog.records
        )
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.parametrize("model", ["fluxonium", "transmon"])
@pytest.mark.parametrize(
    "value",
    [
        1e-13,
        5e-324,
        1e-200,
        1.2345678901234567,
        1e200,
        1.7976931348623157e308,
        -1e200,
        None,
    ],
)
def test_dialog_parameters_and_cache_preserve_finite_float_values(
    qapp, tmp_path: Path, model, value
) -> None:
    trial = (
        FluxoniumParams(value, None, value)
        if model == "fluxonium"
        else TransmonParams(value, None, 0.12345678901234567)
    )
    port = PredictorControlFacet(PredictorService())
    dialog = PredictorDialog(port, params=FluxoniumParams())
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path)
        dialog.set_parameters(trial)
        assert dialog.parameters() == trial
    finally:
        dialog.close()
        dialog.deleteLater()
    reopened = PredictorDialog(port, params=FluxoniumParams())
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path)
        assert reopened.parameters() == trial
    finally:
        reopened.close()
        reopened.deleteLater()


@pytest.mark.parametrize(
    ("field", "valid", "invalid"),
    [
        ("predictor_input_fluxonium_EC", "1.0", "oops"),
        ("predictor_offset", "0.2", "nan"),
        ("predictor_value_coil", "0.3", "nan"),
    ],
    ids=["energy", "offset", "native"],
)
def test_invalid_drafts_do_not_overwrite_restored_v2_cache(
    qapp, tmp_path: Path, field: str, valid: str, invalid: str
) -> None:
    cache = tmp_path / "predictor.json"
    cache.write_text(
        json.dumps(
            {
                "version": 2,
                "model": "fluxonium",
                "params": {"EJ": None, "EC": 1.0, "EL": None},
                "flux": {
                    "offset": 0.2,
                    "period": {"coil": 2.0, "trim": -4.0},
                    "unit": {"coil": "A", "trim": "V"},
                },
                "device_values": {"coil": 0.3},
                "axis_device": "trim",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    evidence = Path(os.environ.get("PREDICTOR_CACHE_EVIDENCE", str(tmp_path))) / field
    port = PredictorControlFacet(PredictorService())
    dialog = PredictorDialog(port)
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path)
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.screenshot(evidence / "01-restored-v2.png")
        assert ui.observe(object_name=field).text == valid
        assert ui.observe(object_name="predictor_axis_trim").checked
        assert ui.observe(object_name="predictor_value_trim").text == ""
        previous = cache.read_bytes()
        ui.enter_text(object_name=field, value=invalid, commit=False)
        ui.screenshot(evidence / "02-invalid-draft.png")
        assert ui.observe(object_name=field).text == invalid
        assert not ui.observe("Calibrate").enabled
        assert cache.read_bytes() == previous
        ui.click("Close")
        assert cache.read_bytes() == previous
    finally:
        dialog.close()
        dialog.deleteLater()
    reopened = PredictorDialog(port)
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path)
        reopened.show()
        qapp.processEvents()
        ui = ComponentDriver(reopened)
        ui.screenshot(evidence / "03-reopened-last-valid.png")
        assert ui.observe(object_name=field).text == valid
        assert ui.observe(object_name="predictor_axis_trim").checked
        assert ui.observe(object_name="predictor_value_trim").text == ""
        assert reopened.parameters() == FluxoniumParams(None, 1.0, None)
        ui.click("Close")
    finally:
        reopened.close()
        reopened.deleteLater()


@pytest.fixture
def component_context(tmp_path: Path):
    """Provide a real point whose unchanged document is the isolation oracle."""
    added = "qubit/fluxonium" not in component_registry.kinds()
    if added:
        component_registry.register("qubit/fluxonium", components.Qubit)
    try:
        entry = ResultEntry.create(
            "trial",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind="qubit/fluxonium", EJ=5.0, EC=1.0, EL=0.5)
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        context.new_point(PointCreateRequest("first", "", CommandOrigin("gui")))
        yield context
    finally:
        if added:
            component_registry.unregister("qubit/fluxonium")


def test_dialog_cache_restores_transmon_and_fluxonium_trials(
    qapp, tmp_path: Path
) -> None:
    port = PredictorControlFacet(PredictorService())
    trial = TransmonParams(EJmax=20.0, EC=0.3, d=0.0)
    dialog = PredictorDialog(port)
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path)
        dialog.set_parameters(trial)
    finally:
        dialog.close()
        dialog.deleteLater()
    reopened = PredictorDialog(port)
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path)
        assert reopened.parameters() == trial
        assert reopened.capture_persisted_state().model == "transmon"
        reopened.set_parameters(FluxoniumParams(EJ=4.1))
    finally:
        reopened.close()
        reopened.deleteLater()
    fluxonium = PredictorDialog(port)
    try:
        fluxonium.enable_parameter_cache(cache_dir=tmp_path)
        assert fluxonium.parameters() == FluxoniumParams(EJ=4.1)
    finally:
        fluxonium.close()
        fluxonium.deleteLater()


@pytest.mark.parametrize(
    "target, clamped", [(1707.3748654723192, False), (-100.0, True), (20000.0, True)]
)
def test_dialog_displays_actual_calibration_and_clamping_without_point_writes(
    qapp, component_context, target, clamped
):
    from qtpy.QtWidgets import QLabel

    query = PredictorService(ContextControlFacet(context=component_context))
    before = component_context.read_context().document()
    dialog = PredictorDialog(
        PredictorControlFacet(query), params=TransmonParams(20.0, 0.3, 0.0)
    )
    try:
        dialog.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.1})
        result = dialog.calibrate(target)
        actual = dialog.predict_frequency()
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert f"clamped={str(clamped).lower()}" in text
        assert f"{actual:.6g} MHz" in text
        assert dialog.flux().offset == pytest.approx(0.2 + result.offset_correction)
        assert component_context.read_context().document() == before
    finally:
        dialog.close()
        dialog.deleteLater()
