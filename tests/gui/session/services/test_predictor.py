"""Read-only predictor queries from explicit parameters.

Queries do not require a session-installed predictor. Numerical service
contracts remain here; physics goldens belong to simulate.
"""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, PointCreateRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.predictor import (
    CalibrateFluxBiasRequest,
    PredictCurveRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictorService,
)
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase
from zcu_tools.simulate.fluxonium import predict

import tests.gui._component_fakes as components


@pytest.fixture
def context(tmp_path: Path) -> Generator[ContextService]:
    before = component_registry.kinds()
    added = "qubit/fluxonium" not in before
    if added:
        component_registry.register("qubit/fluxonium", components.Qubit)
    installed = component_registry.kinds()
    try:
        entry = ResultEntry.create(
            "queries",
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
        entry.setup.add_component(
            "Q2",
            kind="qubit/fluxonium",
            EJ=4.1,
            EC=1.2,
            flux_half=1.75,
            flux_period=3.5,
            ext={"flux_bias": 0.012},
        )
        state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
        owner = ContextService(
            state,
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
            alias_rows=(AliasRow("qubit", "Q2"),),
        )
        owner.install_entry(entry)
        owner.new_point(PointCreateRequest("first", "", CommandOrigin("gui")))
        yield owner
        assert component_registry.kinds() == installed
    finally:
        if added:
            component_registry.unregister("qubit/fluxonium")
        assert component_registry.kinds() == before


def test_parameter_queries_do_not_change_source(context: ContextService) -> None:
    svc = PredictorService(ContextControlFacet(context=context))
    params = predict.FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
    before = context.read_context().document()
    freq = svc.predict_freq(params, PredictFreqRequest(value=0.1, transition=(0, 1)))
    assert freq == pytest.approx(1707.3748654723192, abs=1e-7)
    result = svc.calibrate_flux_bias(
        params, CalibrateFluxBiasRequest(value=0.101, frequency_mhz=freq)
    )
    assert result.flux_bias == pytest.approx(-0.001, abs=1e-4)
    assert params.flux_bias == 0.0
    assert context.read_context().document() == before
    # A separate trial does not replace the original explicit parameters.
    trial = predict.FluxoniumParams(3.0, 1.5, 0.3, 0.0, 1.0, 0.0)
    assert svc.predict_freq(trial, PredictFreqRequest(0.1, (0, 1))) != pytest.approx(
        freq, abs=1.0
    )
    assert svc.predict_freq(params, PredictFreqRequest(0.1, (0, 1))) == freq


def test_calibrate_flux_bias_invalid_transition_raises() -> None:
    svc = PredictorService()
    params = predict.FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
    with pytest.raises(InvalidInputError, match="from-level"):
        svc.calibrate_flux_bias(params, CalibrateFluxBiasRequest(0.25, 4567.0, (1, 1)))


def test_partial_params_refuse_calculation() -> None:
    svc = PredictorService()
    params = predict.FluxoniumParams(4.1, 1.2, None, 1.75, 3.5, 0.012)
    with pytest.raises(ValueError, match="EL"):
        svc.predict_freq(params, PredictFreqRequest(1.8, (0, 1)))
    with pytest.raises(ValueError, match="EL"):
        svc.calibrate_flux_bias(params, CalibrateFluxBiasRequest(1.8, 512.3))


def test_predict_freq_curve_shape_labels_and_single_point() -> None:
    svc = PredictorService()
    params = predict.FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
    values = np.array([0.1, 0.2, 0.1])
    result = svc.predict_freq_curve(
        params, PredictCurveRequest(values, ((0, 1), (0, 3)))
    )
    assert result.labels == ("0→1", "0→3")
    assert result.freqs_mhz.shape == (2, 3)
    np.testing.assert_array_equal(result.values, values)
    np.testing.assert_allclose(result.fluxs, [0.6, 0.7, 0.6])
    assert result.freqs_mhz[0, 0] == pytest.approx(
        svc.predict_freq(params, PredictFreqRequest(0.1, (0, 1))), abs=1e-7
    )
    assert result.freqs_mhz[1, 0] > result.freqs_mhz[0, 0]


@pytest.mark.parametrize("operator", ["n", "phi"])
def test_matrix_curves_support_high_levels(operator) -> None:
    svc = PredictorService()
    params = predict.FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
    result = svc.predict_matrix_element_curve(
        params,
        PredictMatrixCurveRequest(np.array([0.1, 0.2]), ((0, 1), (0, 3)), operator),
    )
    assert result.labels == ("0→1", "0→3")
    assert result.mags.shape == (2, 2)
    assert np.all(result.mags > 0)


@pytest.mark.parametrize("transitions", [(), ((-1, 1),), ((2, 1),)])
def test_curve_rejects_invalid_transitions(transitions) -> None:
    svc = PredictorService()
    params = predict.FluxoniumParams(5.0, 1.0, 0.5, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="transitions|Transition"):
        svc.predict_freq_curve(
            params, PredictCurveRequest(np.array([0.1]), transitions)
        )
    with pytest.raises(ValueError, match="transitions|Transition"):
        svc.predict_matrix_element_curve(
            params, PredictMatrixCurveRequest(np.array([0.1]), transitions, "n")
        )
