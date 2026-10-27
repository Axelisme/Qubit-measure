"""Detached energy/frame trials, nullable drafts, cache and calibration displays."""

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


def test_dialog_missing_frame_requires_explicit_source(qapp) -> None:
    dialog = PredictorDialog(
        PredictorControlFacet(PredictorService()), params=TransmonParams(20.0, 0.3, 0.1)
    )
    try:
        with pytest.raises(ValueError, match="flux frame"):
            dialog.predict_frequency()
        dialog.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {})
        with pytest.raises(ValueError, match="coil"):
            dialog.predict_frequency()
    finally:
        dialog.close()
        dialog.deleteLater()


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
