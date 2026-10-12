"""Independent dialog trial parameters never alter a component or source file."""

from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
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
from zcu_tools.simulate.transmon import TransmonParams

import tests.gui._component_fakes as components
from tests.gui._predictor_fakes import FailingPredictorService


def test_dialog_logs_unexpected_value_error_with_traceback(qapp, caplog) -> None:
    failure = ValueError("solver invariant")
    service = FailingPredictorService(failure)
    with caplog.at_level("ERROR"):
        dialog = PredictorDialog(
            PredictorControlFacet(service),
            params=FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0),
        )
    try:
        assert service.requests
        assert any(
            record.exc_info is not None and record.exc_info[1] is failure
            for record in caplog.records
        )
    finally:
        dialog.close()


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
    # Leave EC unknown to avoid solving a spectrum while exercising editor range.
    trial = (
        FluxoniumParams(value, None, value, value, value, value)
        if model == "fluxonium"
        else TransmonParams(value, None, 0.12345678901234567, value, value, value)
    )
    port = PredictorControlFacet(PredictorService())
    dialog = PredictorDialog(port, params=FluxoniumParams())
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path)
        dialog.set_parameters(trial)
        assert dialog.parameters() == trial
    finally:
        dialog.close()
    reopened = PredictorDialog(port, params=FluxoniumParams())
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path)
        assert reopened.parameters() == trial
    finally:
        reopened.close()


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
        component_registry.register("qubit/fluxonium", components.Qubit)
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


@pytest.fixture
def component_context(tmp_path: Path):
    """Provide a real point with a complete and an incomplete qubit."""
    added = "qubit/fluxonium" not in component_registry.kinds()
    if added:
        component_registry.register("qubit/fluxonium", components.Qubit)
    try:
        entry = ResultEntry.create(
            "prefill",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        for name, el in (("Q1", 0.5), ("Q2", None)):
            entry.setup.add_component(
                name,
                kind="qubit/fluxonium",
                EJ=5.0,
                EC=1.0,
                EL=el,
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
        context.set_alias_rows((AliasRow("qubit", "Q2"),))
        yield context
    finally:
        if added:
            component_registry.unregister("qubit/fluxonium")


def test_dialog_component_prefill_defaults_to_alias_and_stays_detached(
    qapp, component_context: ContextService
) -> None:
    context = component_context
    query = PredictorService(context, components.params_for)
    before = context.read_context().document()
    dialog = PredictorDialog(
        PredictorControlFacet(query),
        parameter_provider=context.read_context,
        params=FluxoniumParams(),
    )
    try:
        assert dialog.selected_component() == "Q2"
        dialog.load_component()
        assert dialog.parameters() == FluxoniumParams(5.0, 1.0, None, 0.0, 1.0, 0.0)
        assert dialog.parameters().missing() == ("EL",)
        with pytest.raises(ValueError, match="Missing parameters: EL"):
            dialog.predict_frequency(0.1)
        dialog.set_parameters(replace(dialog.parameters(), EL=0.5))
        frequency = dialog.predict_frequency(0.1)
        bias = dialog.calibrate(0.101, frequency)
        assert bias == pytest.approx(-0.001, abs=1e-4)
        # Loading again reads the current component, not edited dialog parameters.
        dialog.load_component("Q1")
        assert dialog.parameters().flux_bias == 0.0
        assert context.read_context().document() == before
        q2 = query.params_for_qubit("Q2")
        assert isinstance(q2, FluxoniumParams)
        assert q2.EL is None
    finally:
        dialog.close()


def test_dialog_incomplete_parameters_preserve_unknown_affine_inputs(qapp) -> None:
    params = FluxoniumParams(EJ=5.0, EC=1.0, EL=0.5)
    dialog = PredictorDialog(PredictorControlFacet(PredictorService()))
    try:
        dialog.set_parameters(params)
        assert dialog.parameters() == params
        with pytest.raises(ValueError, match="flux_half, flux_period, flux_bias"):
            dialog.predict_frequency(0.1)
        dialog.set_parameters(
            replace(params, flux_half=0.0, flux_period=1.0, flux_bias=0.0)
        )
        assert dialog.predict_frequency(0.1) == pytest.approx(
            1707.3748654723192, abs=1e-7
        )
    finally:
        dialog.close()


def test_dialog_cache_restores_partial_trial_without_changing_component(
    qapp, tmp_path: Path, component_context: ContextService
) -> None:
    context = component_context
    query = PredictorService(context, components.params_for)
    before = context.read_context().document()
    dialog = PredictorDialog(
        PredictorControlFacet(query), parameter_provider=context.read_context
    )
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path / "cache")
        dialog.load_component()
        dialog.set_parameters(replace(dialog.parameters(), flux_bias=0.012))
        saved = dialog.parameters()
    finally:
        dialog.close()
    reopened = PredictorDialog(
        PredictorControlFacet(query), parameter_provider=context.read_context
    )
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path / "cache")
        assert reopened.parameters() == saved
        assert reopened.parameters().missing() == ("EL",)
        assert context.read_context().document() == before
        assert query.params_for_qubit("Q2").flux_bias == 0.0
    finally:
        reopened.close()


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


def test_dialog_cache_restores_transmon_and_fluxonium_trials(
    qapp, tmp_path: Path
) -> None:
    from zcu_tools.simulate.transmon import TransmonParams

    port = PredictorControlFacet(PredictorService())
    trial = TransmonParams(EJmax=20.0, EC=0.3, d=0.0)
    dialog = PredictorDialog(port)
    try:
        dialog.enable_parameter_cache(cache_dir=tmp_path)
        dialog.set_parameters(trial)
        assert dialog.parameters() == trial
        with pytest.raises(ValueError, match="flux_half, flux_period, flux_bias"):
            dialog.predict_frequency(0.0)
    finally:
        dialog.close()
    reopened = PredictorDialog(port)
    try:
        reopened.enable_parameter_cache(cache_dir=tmp_path)
        assert reopened.parameters() == trial
        assert reopened.capture_persisted_state().model == "transmon"
        reopened.set_parameters(FluxoniumParams(EJ=4.1))
    finally:
        reopened.close()
    fluxonium = PredictorDialog(port)
    try:
        fluxonium.enable_parameter_cache(cache_dir=tmp_path)
        assert fluxonium.parameters() == FluxoniumParams(EJ=4.1)
    finally:
        fluxonium.close()


@pytest.mark.parametrize("model", ["fluxonium", "transmon"])
@pytest.mark.parametrize(
    "target, clamped", [(1707.3748654723192, False), (-100.0, True), (20000.0, True)]
)
def test_dialog_displays_actual_calibration_and_clamping_without_point_writes(
    qapp, component_context, model, target, clamped
):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.simulate.transmon import TransmonParams

    query = PredictorService(component_context, components.params_for)
    before = component_context.read_context().document()
    params = (
        FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
        if model == "fluxonium"
        else TransmonParams(20.0, 0.3, 0.0, 0.0, 1.0, 0.0)
    )
    dialog = PredictorDialog(PredictorControlFacet(query), params=params)
    try:
        bias = dialog.calibrate(0.101, target)
        actual = dialog.predict_frequency(0.101)
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert f"clamped={str(clamped).lower()}" in text
        assert f"{actual:.6g} MHz" in text
        assert dialog.parameters().flux_bias == pytest.approx(bias)
        assert component_context.read_context().document() == before
    finally:
        dialog.close()
