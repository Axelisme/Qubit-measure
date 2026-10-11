"""Independent dialog trial parameters never alter a component or source file."""

from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.qubit_params import FluxDepFit, ParamsProject, QubitParams
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

import zcu_lab.components as components


def test_dialog_calibration_changes_only_local_trial(qapp) -> None:
    service = PredictorService()
    original = FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
    dialog = PredictorDialog(PredictorControlFacet(service), params=original)
    try:
        frequency = dialog.predict_frequency(0.1, (0, 1))
        assert frequency == pytest.approx(1707.3748654723192, abs=1e-7)
        bias = dialog.calibrate(0.101, frequency, (0, 1))
        assert bias == pytest.approx(-0.001, abs=1e-4)
        assert dialog.parameters().flux_bias == pytest.approx(bias)
        assert original.flux_bias == 0.0
        # A second dialog starts from its own explicit params, not first's edit.
        second = PredictorDialog(PredictorControlFacet(service), params=original)
        try:
            assert second.parameters() == original
        finally:
            second.close()
    finally:
        dialog.close()


def test_dialog_trial_calibration_leaves_real_point_unchanged(
    qapp, tmp_path: Path
) -> None:
    before_kinds = component_registry.kinds()
    added = "qubit/fluxonium" not in before_kinds
    if added:
        component_registry.register("qubit/fluxonium", components.QubitSchema)
    installed = component_registry.kinds()
    try:
        entry = ResultEntry.create(
            "trial",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component(
            "Q1",
            kind="qubit/fluxonium",
            EJ=5.0,
            EC=1.0,
            EL=0.5,
            flux_half=0.0,
            flux_period=1.0,
            ext={"flux_bias": 0.0},
        )
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        context.new_point(PointCreateRequest("first", "", CommandOrigin("gui")))
        query = PredictorService(context, components.params_for)
        before = context.read_context().document()
        dialog = PredictorDialog(
            PredictorControlFacet(query), params=query.params_for_qubit("Q1")
        )
        try:
            measured = dialog.predict_frequency(0.1)
            bias = dialog.calibrate(0.101, measured)
            assert bias == pytest.approx(-0.001, abs=1e-4)
            assert dialog.parameters().flux_bias == pytest.approx(bias)
            assert query.params_for_qubit("Q1").flux_bias == 0.0
            assert context.read_context().document() == before
        finally:
            dialog.close()
        assert component_registry.kinds() == installed
    finally:
        if added:
            component_registry.unregister("qubit/fluxonium")
        assert component_registry.kinds() == before_kinds


def test_dialog_file_load_only_prefills_local_parameters(qapp, tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    file = QubitParams(path)
    file.ensure_project(ParamsProject("trial", "trial"))
    file.set_fluxdep_fit(
        FluxDepFit(
            EJ=4.5, EC=1.2, EL=0.9, flux_half=0.31, flux_int=0.71, flux_period=0.8
        )
    )
    before = path.read_bytes()
    dialog = PredictorDialog(
        PredictorControlFacet(PredictorService()),
        params=FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0),
    )
    try:
        dialog.load_parameters(str(path))
        assert dialog.parameters() == FluxoniumParams(4.5, 1.2, 0.9, 0.31, 0.8, 0.0)
        assert path.read_bytes() == before
    finally:
        dialog.close()
