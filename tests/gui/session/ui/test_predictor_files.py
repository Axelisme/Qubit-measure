"""Predictor file selection and detached editing through visible keyboard/mouse UI."""

import os
from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    load_predictor_file,
    save_predictor_file,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.transmon import TransmonParams

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.session.ui._predictor_support import make_device_control


def test_save_as_writes_v2_energy_and_multidevice_frame(qapp, tmp_path: Path):
    from zcu_tools.resources.predictor_file import PredictorFlux, load_predictor_file

    flux = PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"})
    params = TransmonParams(20.0, 0.3, 0.1)
    dialog = PredictorDialog(PredictorControlFacet(PredictorService()), params=params)
    try:
        dialog.set_flux(flux, {"coil": 0.3, "trim": -0.4})
        destination = tmp_path / "saved.json"
        dialog.save_as(destination)
        saved = load_predictor_file(destination)
        assert saved.params == params
        assert saved.flux == flux
    finally:
        dialog.close()
        dialog.deleteLater()


def test_entry_files_load_and_remain_manually_editable(qapp, tmp_path: Path):
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    entry = ResultEntry.create(
        "ui",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    entry.predictors_path.mkdir()
    for filename, params in (
        ("fluxonium.json", FluxoniumParams(EJ=4.5)),
        ("transmon.json", TransmonParams(EJmax=17.0)),
    ):
        save_predictor_file(
            entry.predictors_path / filename,
            PredictorFile(
                params.model,
                params,
                PredictorFlux(
                    0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}
                ),
                PredictorSource(None, None),
            ),
        )
    context.install_entry(entry)
    dialog = PredictorDialog(
        PredictorControlFacet(PredictorService(ContextControlFacet(context=context))),
        params=FluxoniumParams(),
    )
    evidence = Path(os.environ.get("PREDICTOR_FILES_EVIDENCE", str(tmp_path)))
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.screenshot(evidence / "01-entry-files.png")
        assert ui.observe(object_name="predictor_file").text == "fluxonium.json"
        assert ui.observe("Load").enabled
        ui.click("Load")
        ui.screenshot(evidence / "02-loaded-fluxonium.png")
        assert ui.observe(object_name="predictor_input_EJ").text == "4.5"
        el = ui.observe(object_name="predictor_input_EL")
        assert el.text == "" and el.enabled
        assert ui.observe(object_name="predictor_offset").text == "0.2"
        assert (
            ui.observe(object_name="predictor_status").text
            == "Loaded predictor file into local trial"
        )
        ui.enter_text(object_name="predictor_input_EJ", value="6.25", commit=False)
        ui.screenshot(evidence / "03-manual-edit.png")
        assert ui.observe(object_name="predictor_input_EJ").text == "6.25"
        ui.click("Load")
        ui.screenshot(evidence / "04-reloaded.png")
        assert ui.observe(object_name="predictor_input_EJ").text == "4.5"
        ui.select_combo("transmon.json", object_name="predictor_file")
        ui.screenshot(evidence / "05-selected-transmon.png")
        assert ui.observe(object_name="predictor_file").text == "transmon.json"
        assert ui.observe(object_name="predictor_input_EJ").text == "4.5"
        ui.click("Load")
        ui.screenshot(evidence / "06-loaded-transmon.png")
        assert ui.observe(object_name="predictor_input_EJmax").text == "17.0"
        asymmetry = ui.observe(object_name="predictor_input_d")
        assert asymmetry.text == "" and asymmetry.enabled
        assert ui.observe(object_name="predictor_offset").text == "0.2"
        ui.click("Refresh files")
        ui.screenshot(evidence / "07-refreshed.png")
        assert ui.observe(object_name="predictor_file").text == "transmon.json"
        ui.click("Close")
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.fixture
def multidevice_trial(qapp, tmp_path: Path):
    """Open a real entry and give each required instrument a cached value."""
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    entry = ResultEntry.create(
        "multi", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.predictors_path.mkdir()
    source = entry.predictors_path / "two.json"
    params = TransmonParams(20.0, 0.3, 0.1)
    save_predictor_file(
        source,
        PredictorFile(
            "transmon",
            params,
            PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}),
            PredictorSource(None, None),
        ),
    )
    context.install_entry(entry)
    dialog = PredictorDialog(
        PredictorControlFacet(PredictorService(ContextControlFacet(context=context))),
        params=FluxoniumParams(),
        device=make_device_control(cached={"coil": 0.3, "trim": -0.4}),
    )
    try:
        dialog.show()
        qapp.processEvents()
        yield ComponentDriver(dialog), source, params
    finally:
        dialog.close()
        dialog.deleteLater()


def _save_through_chooser(
    ui: ComponentDriver, destination: Path, evidence: Path
) -> None:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication, QFileDialog

    errors: list[Exception] = []

    def save_in_chooser():
        try:
            chooser = QApplication.activeModalWidget()
            assert isinstance(chooser, QFileDialog)
            driver = ComponentDriver(chooser)
            driver.screenshot(evidence / "07-save-as-chooser.png")
            driver.enter_text(
                object_name="fileNameEdit", value=str(destination), commit=False
            )
            driver.click("&Save")
        except (
            AssertionError,
            LookupError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(exc)
            if (modal := QApplication.activeModalWidget()) is not None:
                modal.close()

    QTimer.singleShot(0, save_in_chooser)
    ui.click("Save as…")
    assert not errors, errors


def test_multidevice_calibrate_and_save_as_visible_flow(
    multidevice_trial, tmp_path: Path
):
    ui, source, params = multidevice_trial
    original = source.read_bytes()
    evidence = Path(os.environ.get("PREDICTOR_MULTI_EVIDENCE", str(tmp_path)))
    destination = tmp_path / "saved-trial.json"
    ui.screenshot(evidence / "01-before-load.png")
    ui.click("Load")
    qt_until(
        lambda: any(axis.visible_curves for axis in ui.plots().axes),
        label="loaded curves",
    )
    ui.screenshot(evidence / "02-two-devices.png")
    assert ui.observe(object_name="predictor_value_coil").text == "0.3"
    assert ui.observe(object_name="predictor_value_trim").text == "-0.4"
    assert ui.observe(object_name="predictor_axis_coil").checked
    initial_frequency = ui.table_rows(object_name="predictor_transitions")[0].columns[1]
    ui.enter_text(object_name="predictor_value_trim", value="-0.6", commit=False)
    qt_until(
        lambda: (
            any(axis.visible_curves for axis in ui.plots().axes)
            and ui.table_rows(object_name="predictor_transitions")[0].columns[1]
            not in (initial_frequency, "—")
        ),
        label="fixed-line effect",
    )
    ui.screenshot(evidence / "03-fixed-line-changed.png")
    assert any(axis.visible_curves for axis in ui.plots().axes)
    ui.click(object_name="predictor_axis_trim")
    qt_until(
        lambda: any(axis.xlabel == "trim (V)" for axis in ui.plots().axes),
        label="trim axis",
    )
    ui.screenshot(evidence / "04-trim-axis.png")
    assert ui.observe(object_name="predictor_axis_coil").checked is False
    assert ui.observe(object_name="predictor_axis_trim").checked is True
    ui.enter_text(object_name="predictor_value_trim", value="-0.4", commit=False)
    qt_until(
        lambda: any(axis.visible_curves for axis in ui.plots().axes),
        label="restored trim curves",
    )
    ui.select_table_row("0→1", object_name="predictor_transitions")
    measured = params.build_predictor().predict_freq(0.48, (0, 1))
    ui.enter_text(
        object_name="predictor_calibration_frequency_expression",
        value=repr(measured),
        commit=False,
    )
    ui.screenshot(evidence / "05-ready-to-calibrate.png")
    assert ui.observe("Calibrate").enabled
    ui.click("Calibrate")
    qt_until(
        lambda: "Calibrated offset=" in ui.observe(object_name="predictor_status").text,
        label="calibration",
    )
    qt_until(
        lambda: any(axis.visible_curves for axis in ui.plots().axes),
        label="calibrated curves",
    )
    ui.screenshot(evidence / "06-calibrated-unsaved.png")
    assert abs(float(ui.observe(object_name="predictor_offset").text) - 0.23) < 1e-5
    assert source.read_bytes() == original
    _save_through_chooser(ui, destination, evidence)
    ui.screenshot(evidence / "08-saved-v2.png")
    saved = load_predictor_file(destination)
    assert saved.params == params
    assert abs(saved.flux.offset - 0.23) < 1e-5
    assert dict(saved.flux.period) == {"coil": 2.0, "trim": -4.0}
    assert source.read_bytes() == original


def test_no_entry_shows_empty_file_choice_and_hint(qapp, tmp_path: Path):
    dialog = PredictorDialog(
        PredictorControlFacet(PredictorService()), params=FluxoniumParams()
    )
    evidence = Path(os.environ.get("PREDICTOR_FILES_EVIDENCE", str(tmp_path)))
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.screenshot(evidence / "08-no-entry.png")
        assert ui.observe(object_name="predictor_file").text == ""
        assert not ui.observe("Load").enabled
        assert ui.observe(
            "Open an entry with predictors/*.json, or add predictor files."
        ).visible
        ui.click("Refresh files")
        ui.screenshot(evidence / "09-no-entry-refreshed.png")
        assert not ui.observe("Load").enabled
        ui.click("Close")
    finally:
        dialog.close()
        dialog.deleteLater()
