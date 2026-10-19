"""Predictor file selection and detached editing through visible keyboard/mouse UI."""

import os
from pathlib import Path

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
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.transmon import TransmonParams

from tests.gui._user_input import ComponentDriver


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
        ("transmon.json", TransmonParams(EJmax=17.0, d=0.1)),
    ):
        save_predictor_file(
            entry.predictors_path / filename,
            PredictorFile(
                params.model,
                params,
                None,
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
        assert ui.observe(object_name="predictor_input_d").text == "0.1"
        ui.click("Refresh files")
        ui.screenshot(evidence / "07-refreshed.png")
        assert ui.observe(object_name="predictor_file").text == "transmon.json"
        ui.click("Close")
    finally:
        dialog.close()
        dialog.deleteLater()


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
