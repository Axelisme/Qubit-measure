from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.feedback import build_feedback_runtime
from zcu_tools.gui.app.autofluxdep.nodes import qubit_freq_recovery as recovery_mod
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq_recovery import (
    PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
    QubitFreqRecoveryState,
    TrustedFrequencyPoint,
    on_fit_failed,
    on_fit_succeeded,
    select_fit_points,
)
from zcu_tools.gui.app.autofluxdep.tools import FluxoniumPredictorAdapter, Tools
from zcu_tools.simulate.fluxonium import FluxoniumPredictor
from zcu_tools.simulate.fluxonium.physical_fit import (
    FluxoniumLocalFitResult,
    FluxoniumModelSnapshot,
)


@dataclass
class _Provider:
    name: str
    builder: Any
    schema: Any


@dataclass
class _FakePredictor:
    offset_mhz: float = 0.0

    def predict_freq(self, flux: float) -> float:
        return 1000.0 + 10.0 * float(flux) + self.offset_mhz

    def predict_matrix_element(self, flux: float) -> float:
        del flux
        return 0.2

    def calibrate(self, flux: float, measured_freq: float) -> None:
        del flux, measured_freq

    def supports_physical_recovery(self) -> bool:
        return True

    def physical_snapshot(self) -> FluxoniumModelSnapshot:
        return FluxoniumModelSnapshot((8.0, 1.0, 1.0), 0.0, 1.0, self.offset_mhz)

    def clone_physical(self) -> _FakePredictor:
        return _FakePredictor(self.offset_mhz)

    def overlay_physical(self, snapshot: FluxoniumModelSnapshot) -> _FakePredictor:
        return _FakePredictor(float(snapshot.flux_bias))


def _schema():
    return (
        QubitFreqBuilder()
        .make_default_schema()
        .with_overrides(
            {
                "physical_recovery_mode": PHYSICAL_RECOVERY_MODE_FAIL_TRIGGERED_FIT,
                "physical_recovery_max_rms_mhz": 20.0,
            }
        )
    )


def _env(
    *,
    flux: float,
    flux_idx: int,
    schema,
    tools: Tools,
    feedback_view,
) -> RunEnv:
    return RunEnv(
        flux=flux,
        flux_idx=flux_idx,
        schema=schema,
        tools=tools,
        feedback=feedback_view,
    )


def test_select_fit_points_enforces_bounds_and_spreads_history():
    history = tuple(
        TrustedFrequencyPoint(float(flux), 5000.0 + float(flux))
        for flux in np.linspace(0.0, 1.0, 101)
    )

    selected = select_fit_points(history, min_points=10, max_points=30)

    assert 25 <= len(selected) <= 30
    assert selected[0].flux == pytest.approx(0.0)
    assert selected[-1].flux == pytest.approx(1.0)
    diffs = np.diff([point.flux for point in selected])
    assert float(np.min(diffs)) >= 0.03


@pytest.mark.parametrize("n_points", [12, 31])
def test_select_fit_points_does_not_count_duplicate_flux_as_fit_points(n_points):
    history = tuple(
        TrustedFrequencyPoint(0.25, 5000.0 + float(idx)) for idx in range(n_points)
    )

    assert select_fit_points(history, min_points=10, max_points=30) == ()


def test_select_fit_points_sorts_unique_flux_and_keeps_latest_duplicate():
    fluxes = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.4)
    history = tuple(
        TrustedFrequencyPoint(float(flux), 5000.0 + float(idx))
        for idx, flux in enumerate(fluxes)
    )

    selected = select_fit_points(history, min_points=10, max_points=30)

    assert [point.flux for point in selected] == sorted(set(fluxes))
    duplicate_flux_point = next(point for point in selected if point.flux == 0.4)
    assert duplicate_flux_point.frequency_mhz == pytest.approx(5011.0)


def test_select_fit_points_returns_empty_until_minimum_history():
    history = tuple(
        TrustedFrequencyPoint(float(idx), 5000.0 + float(idx)) for idx in range(9)
    )

    assert select_fit_points(history, min_points=10, max_points=30) == ()


def test_fluxonium_predictor_adapter_overlay_does_not_mutate_raw_predictor():
    raw = FluxoniumPredictor((8.0, 1.0, 1.0), 0.0, 1.0, 0.0)
    adapter = FluxoniumPredictorAdapter(raw)

    overlay = adapter.overlay_physical(
        FluxoniumModelSnapshot((8.2, 1.1, 0.9), 0.0, 1.0, 0.25)
    )

    assert tuple(float(value) for value in raw.params) == pytest.approx((8.0, 1.0, 1.0))
    assert raw.flux_bias == pytest.approx(0.0)
    assert adapter.physical_snapshot().flux_bias == pytest.approx(0.0)
    assert overlay.physical_snapshot().params == pytest.approx((8.2, 1.1, 0.9))
    assert overlay.physical_snapshot().flux_bias == pytest.approx(0.25)


def test_fail_triggered_recovery_fits_first_fail_and_first_success(monkeypatch):
    builder = QubitFreqBuilder()
    schema = _schema()
    feedback = build_feedback_runtime([_Provider("qubit_freq", builder, schema)])
    feedback_view = feedback.view_for("qubit_freq")
    tools = Tools(predictor=_FakePredictor(), feedback=feedback)
    fit_offsets = [5.0, 6.0]
    fit_calls: list[int] = []

    def fake_fit(base, measured_points):
        points = tuple(measured_points)
        offset = fit_offsets[len(fit_calls)]
        fit_calls.append(len(points))
        fitted = FluxoniumModelSnapshot(
            base.params, base.flux_half, base.flux_period, offset
        )
        return FluxoniumLocalFitResult(
            accepted=True,
            reason="accepted",
            base=base,
            fitted=fitted,
            predictor=None,
            n_points=len(points),
            base_rms_mhz=8.0,
            fitted_rms_mhz=3.0,
        )

    monkeypatch.setattr(recovery_mod, "fit_local_fluxonium_model", fake_fit)

    for idx, flux in enumerate(np.linspace(0.0, 0.9, 10)):
        env = _env(
            flux=float(flux),
            flux_idx=idx,
            schema=schema,
            tools=tools,
            feedback_view=feedback_view,
        )
        on_fit_succeeded(
            env,
            1000.0 + 10.0 * float(flux) + 8.0,
            snapshot_predict_freq=1000.0 + 10.0 * float(flux),
            estimator_key="predict_freq_correction",
        )

    fail_env = _env(
        flux=1.0,
        flux_idx=10,
        schema=schema,
        tools=tools,
        feedback_view=feedback_view,
    )
    on_fit_failed(
        fail_env,
        snapshot_predict_freq=1010.0,
        estimator_key="predict_freq_correction",
    )

    state = tools.peek_recovery_state("qubit_freq", QubitFreqRecoveryState)
    assert state is not None
    assert state.fail_streak == 1
    assert state.overlay is not None
    assert state.overlay.predict_freq(0.0) == pytest.approx(1005.0)
    assert fit_calls == [10]
    assert state.last_attempt is not None
    assert state.last_attempt.trigger == "first_fail"
    assert state.last_attempt.accepted

    on_fit_failed(
        fail_env,
        snapshot_predict_freq=1010.0,
        estimator_key="predict_freq_correction",
    )

    assert state.fail_streak == 2
    assert fit_calls == [10]

    success_env = _env(
        flux=1.1,
        flux_idx=11,
        schema=schema,
        tools=tools,
        feedback_view=feedback_view,
    )
    on_fit_succeeded(
        success_env,
        1000.0 + 10.0 * 1.1 + 8.0,
        snapshot_predict_freq=1011.0,
        estimator_key="predict_freq_correction",
    )

    assert state.fail_streak == 0
    assert state.overlay is not None
    assert state.overlay.predict_freq(0.0) == pytest.approx(1006.0)
    assert fit_calls == [10, 11]
    assert state.last_attempt is not None
    assert state.last_attempt.trigger == "first_success_after_fail"
    assert state.last_attempt.accepted

    estimator = feedback_view.estimator("predict_freq_correction")
    assert estimator is not None
    estimate = estimator.estimate(0.5)
    assert estimate is not None
    assert estimate.confidence == pytest.approx(1.0)
