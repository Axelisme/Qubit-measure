"""RB-1: qubit_freq end-to-end real acquire against the flux-aware MockSoc.

This is the load-bearing Phase RB test. It runs the qubit_freq Node's *real*
acquire path (set flux device -> setup_devices -> Schedule-backed acquire -> fit ->
feedback) at several flux points and asserts the fitted qubit frequency CHANGES
with flux. If the picked flux device never received the value (the name/label
silent-miss), the SimEngine would stay pinned at one operating point and the fit
freq would be constant -- so a constant fit fails this test.

The GUI predictor is derived from the same SimParams as MockSoc, so the predicted
drive centre tracks the SimEngine's actual f01 and the dip lands in the detune
window. Flux device values are chosen inside the mock generator range and at
distinct reduced-flux points, so the fitted line must move.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.feedback import build_feedback_runtime
from zcu_tools.gui.cfg import CenteredSweepValue
from zcu_tools.gui.session.services.mock_flux import FAKE_FLUX_DEVICE_NAME

from .._helpers import (
    connect_mock,
    high_snr_simparams,
    mock_flux_predictor,
    run_controller_to_completion,
)

# A readout module near the dressed resonator (~6 GHz under DEFAULT_SIMPARAM).
_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 6000.0,
        "gain": 1.0,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {"ro_ch": 0, "ro_freq": 6000.0, "ro_length": 0.9, "trig_offset": 0.6},
}


class _Provider:
    def __init__(self, name, builder, schema) -> None:
        self.name = name
        self.builder = builder
        self.schema = schema


class _TrackingPredictor:
    def __init__(self, base: float = 600.0) -> None:
        self.base = float(base)
        self.calibrations: list[tuple[float, float]] = []

    def predict_freq(self, flux: float) -> float:
        del flux
        return self.base

    def predict_matrix_element(self, flux: float) -> float:
        del flux
        return 1.0

    def calibrate(self, flux: float, measured_freq: float) -> None:
        self.calibrations.append((float(flux), float(measured_freq)))
        self.base = float(measured_freq)


class _RecoverablePredictor:
    def __init__(self, base: float = 600.0, bias: float = 0.0) -> None:
        self.base = float(base)
        self.bias = float(bias)
        self.calibrations: list[tuple[float, float]] = []

    def predict_freq(self, flux: float) -> float:
        del flux
        return self.base + self.bias

    def predict_matrix_element(self, flux: float) -> float:
        del flux
        return 1.0

    def calibrate(self, flux: float, measured_freq: float) -> None:
        self.calibrations.append((float(flux), float(measured_freq)))

    def supports_physical_recovery(self) -> bool:
        return True

    def physical_snapshot(self):
        from zcu_tools.simulate.fluxonium.physical_fit import FluxoniumModelSnapshot

        return FluxoniumModelSnapshot((8.0, 1.0, 1.0), 0.0, 1.0, self.bias)

    def overlay_physical(self, snapshot):
        return _RecoverablePredictor(self.base, float(snapshot.flux_bias))


class _RecordingEstimator:
    def __init__(self) -> None:
        self.observe_calls: list[tuple[float, float]] = []
        self.replace_calls: list[tuple[tuple[float, float], ...]] = []

    def observe(self, flux: float, value: float) -> None:
        self.observe_calls.append((float(flux), float(value)))

    def replace_observations(self, observations) -> None:
        self.replace_calls.append(
            tuple((float(flux), float(value)) for flux, value in observations)
        )

    def estimate(self, flux: float):
        del flux
        return None


class _RecordingFeedbackView:
    def __init__(self, estimator: _RecordingEstimator) -> None:
        self._estimator = estimator

    def estimator(self, key: str):
        assert key == "predict_freq_correction"
        return self._estimator


def _configure_context(ctrl) -> None:
    """Populate the active context's ml so qubit_freq's make_cfg has a readout +
    drive waveform.

    The predicted-centre predictor is not built here: connect_mock has already run
    the MockFluxProvisioner, which installs a FluxoniumPredictor derived from the
    mock soc's SimParams (matching the SimEngine's f01). Relying on that provisioned
    predictor exercises the real mock-connect path rather than a hand-built copy."""
    ml = ctrl.state.exp_context.ml
    ml.register_waveform(
        qub_drive={"style": "const", "length": 1.0},
    )
    ml.register_module(readout=_READOUT)


@pytest.mark.filterwarnings(
    "ignore:fit_func failed; returning init_p fallback with infinite covariance:RuntimeWarning"
)
def test_qubit_freq_acquire_fit_varies_with_flux():
    ctrl = build_core()
    sim_params = high_snr_simparams()
    connect_mock(ctrl, sim_params=sim_params)
    _configure_context(ctrl)
    ctrl.set_flux_device(FAKE_FLUX_DEVICE_NAME)

    node = ctrl.add_node_by_type("qubit_freq")
    node.schema.with_overrides(
        {
            "qub_ch": 1,
            "qub_nqz": 1,
            "qub_gain": 0.3,
            "qub_length": 1.0,
            "reps": 100,
            "rounds": 1,
            "relax_delay": 0.0,
            # wide detune window around the predicted centre so the dip is caught.
            "detune_sweep": CenteredSweepValue(center=0.0, span=120.0, expts=121),
        }
    )

    # Flux device values within the realistic range that map to DISTINCT reduced
    # flux under DEFAULT_SIMPARAM (flux_period=5e-3, flux_half=0):
    #   0.0 -> 0.5 (sweet spot), 1.5e-3 -> 0.8, 2.5e-3 -> 1.0 (integer flux).
    # f01 rises from the sweet-spot minimum, sub-Nyquist and clearly separated.
    # (Avoid exact multiples of flux_period, which alias back to the same flux.)
    flux_values = [0.0, 1.5e-3, 2.5e-3]
    ctrl.set_flux_values(flux_values)
    run_controller_to_completion(ctrl, timeout=15.0)

    res = ctrl.state.run_results["qubit_freq"]
    fit = np.asarray(res.fit_freq, dtype=np.float64)
    good = fit[~np.isnan(fit)]
    # at least two points fit, and the fitted qubit frequency MOVES with flux
    # (the flux genuinely reached fake_flux -> SimEngine).
    assert good.size >= 2, f"too few good fits: {fit}"
    assert float(np.ptp(good)) > 200.0, (
        f"fit_freq did not vary with flux (flux likely never reached the device): {fit}"
    )


def test_plotter_update_runs_after_a_real_produce():
    # build qubit_freq's Result + Plotter, fill a row via a real acquire produce,
    # then redraw — the LivePlot-backed update path must not raise (existed_axes +
    # host draw). Uses the same flux-aware mock context as the fit test above.
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqBuilder
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot

    ctrl = build_core()
    sim_params = high_snr_simparams()
    connect_mock(ctrl, sim_params=sim_params)
    _configure_context(ctrl)
    ctrl.set_flux_device(FAKE_FLUX_DEVICE_NAME)

    builder = QubitFreqBuilder()
    flux = np.linspace(0.0, 0.1, 3)
    params = {
        "qub_ch": 1,
        "qub_nqz": 1,
        "qub_gain": 0.3,
        "qub_length": 1.0,
        "reps": 100,
        "rounds": 1,
        "relax_delay": 0.0,
        "detune_sweep": CenteredSweepValue(center=0.0, span=120.0, expts=121),
    }
    schema = builder.make_default_schema().with_overrides(params)
    result = builder.make_init_result(schema, flux)
    figure = Figure()
    plotter = builder.make_plotter(figure)
    ctx = ctrl.state.exp_context
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        soc=ctx.soc,
        soccfg=ctx.soccfg,
        ml=ctx.ml,
        flux_device=FAKE_FLUX_DEVICE_NAME,
        result=result,
    )
    builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "qfw_factor": None}, modules={"readout": _READOUT}
        )
    )
    plotter.update(result, 0)  # must not raise


def test_good_fit_observes_prediction_residual_by_default():
    # Fixed-bias default: a good real-acquire fit leaves the raw predictor alone
    # and records the run-local residual correction in the generic estimator.
    from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqBuilder
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor, Tools

    ctrl = build_core()
    sim_params = high_snr_simparams()
    connect_mock(ctrl, sim_params=sim_params)
    _configure_context(ctrl)
    ctrl.set_flux_device(FAKE_FLUX_DEVICE_NAME)

    builder = QubitFreqBuilder()
    params = {
        "qub_ch": 1,
        "qub_nqz": 1,
        "qub_gain": 0.3,
        "qub_length": 1.0,
        "reps": 1000,
        "rounds": 3,
        "earlystop_snr": None,
        "drive_gain_mode": "fixed",
        "relax_delay": 0.0,
        "detune_sweep": CenteredSweepValue(center=0.0, span=120.0, expts=121),
    }
    schema = builder.make_default_schema().with_overrides(params)
    result = builder.make_init_result(schema, np.array([0.0]))
    predictor = SimplePredictor(
        base=float(mock_flux_predictor(sim_params).predict_freq(0.0)), slope=50.0
    )
    before = predictor.predict_freq(0.0)
    ctx = ctrl.state.exp_context
    feedback = build_feedback_runtime([_Provider("qubit_freq", builder, schema)])
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        soc=ctx.soc,
        soccfg=ctx.soccfg,
        ml=ctx.ml,
        flux_device=FAKE_FLUX_DEVICE_NAME,
        result=result,
        tools=Tools(predictor=predictor, feedback=feedback),
        feedback=feedback.view_for("qubit_freq"),
    )
    patch = builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": before, "qfw_factor": None},
            modules={"readout": _READOUT},
        )
    )
    values = patch.values()
    assert "qubit_freq" in values  # a good fit
    assert "qfw_factor" in values
    assert abs(values["qfw_factor"] - values["fit_kappa"] / 0.3) < 1e-9
    # SimplePredictor has no physical calibration loop; the generic estimator owns
    # the residual correction and composes with the base predictor.
    assert predictor.predict_freq(0.0) == before
    correction = env.feedback.estimator("predict_freq_correction")
    assert correction is not None
    estimate = correction.estimate(0.0)
    assert estimate is not None
    assert (
        abs(
            predictor.predict_freq(0.0)
            + estimate.confidence * estimate.value
            - values["qubit_freq"]
        )
        < 1e-6
    )


def _mocked_qubit_freq_produce_env(
    monkeypatch,
    real,
    fit_return,
    *,
    predictor: Any | None = None,
    schema_overrides: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    import zcu_tools.gui.app.autofluxdep.experiments.qubit_freq as qf_mod
    from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqBuilder
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor, Tools

    ctrl = build_core()
    connect_mock(ctrl)
    _configure_context(ctrl)

    builder = QubitFreqBuilder()
    overrides: dict[str, Any] = {
        "qub_ch": 1,
        "qub_nqz": 1,
        "qub_gain": 0.25,
        "qub_length": 1.0,
        "reps": 1,
        "rounds": 1,
        "relax_delay": 0.0,
        "detune_sweep": CenteredSweepValue(center=0.0, span=10.0, expts=11),
    }
    if schema_overrides is not None:
        overrides.update(schema_overrides)
    schema = builder.make_default_schema().with_overrides(overrides)
    result = builder.make_init_result(schema, np.array([0.0]))

    class _DummyModularProgram:
        def __init__(self, _soccfg, cfg, *, modules, sweep):
            del modules, sweep
            self.cfg_model = cfg

        def acquire(self, *args, **kwargs):
            del args
            raw = [[np.zeros((result.n_detune, 2), dtype=np.float64)]]
            kwargs["round_hook"](1, raw, kwargs["cancel_flag"])
            return raw

        def acquire_decimated(self, *args, **kwargs):
            del args, kwargs
            raise NotImplementedError

    monkeypatch.setattr(qf_mod, "ModularProgramV2", _DummyModularProgram)
    monkeypatch.setattr(qf_mod, "setup_flux_point", lambda *args, **kwargs: None)
    monkeypatch.setattr(qf_mod, "_signal2real", lambda _signals: real)
    monkeypatch.setattr(qf_mod, "fit_qubit_freq", lambda _freqs, _real: fit_return)

    if predictor is None:
        predictor = SimplePredictor(base=600.0, slope=0.0)
    feedback = build_feedback_runtime([_Provider("qubit_freq", builder, schema)])
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        soc=ctrl.state.exp_context.soc,
        soccfg=ctrl.state.exp_context.soccfg,
        ml=ctrl.state.exp_context.ml,
        flux_device=FAKE_FLUX_DEVICE_NAME,
        result=result,
        tools=Tools(predictor=predictor, feedback=feedback),
        feedback=feedback.view_for("qubit_freq"),
    )
    return builder, env, result, predictor


def test_medium_fit_observes_residual_without_hard_calibration(monkeypatch):
    # A fit between the frequency and linewidth gates is still useful for centring
    # the run-local residual estimator, but qubit_freq must not calibrate the raw
    # predictor and its FWHM must not drive qfw_factor feedback.
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot

    fit_curve = np.linspace(0.0, 1.0, 11)
    real = fit_curve + 0.15  # residual passes 0.2 gate, fails 0.1 gate.
    predictor = _TrackingPredictor(base=600.0)
    builder, env, result, _returned_predictor = _mocked_qubit_freq_produce_env(
        monkeypatch,
        real,
        (605.0, 0.0, 4.0, 0.0, fit_curve, None),
        predictor=predictor,
    )
    before = predictor.predict_freq(0.0)
    patch = builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "qfw_factor": None}, modules={"readout": _READOUT}
        )
    )

    values = patch.values()
    assert values["qubit_freq"] == 605.0
    assert values["fit_detune"] == 5.0
    assert "fit_kappa" not in values
    assert "qfw_factor" not in values
    assert result.fit_freq[0] == 605.0
    assert predictor.predict_freq(0.0) == before
    assert predictor.calibrations == []
    correction = env.feedback.estimator("predict_freq_correction")
    assert correction is not None
    estimate = correction.estimate(0.0)
    assert estimate is not None
    assert (
        abs(predictor.predict_freq(0.0) + estimate.confidence * estimate.value - 605.0)
        < 1e-6
    )


def test_success_after_fail_reseed_skips_duplicate_residual_observe(monkeypatch):
    import zcu_tools.gui.app.autofluxdep.experiments.qubit_freq as recovery_mod
    from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import (
        QubitFreqRecoveryState,
        TrustedFrequencyPoint,
    )
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.simulate.fluxonium.physical_fit import (
        FluxoniumLocalFitResult,
        FluxoniumModelSnapshot,
    )

    fit_curve = np.linspace(0.0, 1.0, 11)
    real = fit_curve + 0.15
    predictor = _RecoverablePredictor(base=600.0)
    builder, env, _result, _returned_predictor = _mocked_qubit_freq_produce_env(
        monkeypatch,
        real,
        (605.0, 0.0, 4.0, 0.0, fit_curve, None),
        predictor=predictor,
        schema_overrides={
            "drive_gain_mode": "fixed",
            "physical_recovery_mode": "fail_triggered_fit",
        },
    )
    estimator = _RecordingEstimator()
    env.feedback = _RecordingFeedbackView(estimator)
    assert env.tools is not None
    state = env.tools.recovery_state("qubit_freq", QubitFreqRecoveryState)
    state.fail_streak = 1
    state.history = [
        TrustedFrequencyPoint(float(flux), 605.0) for flux in np.linspace(0.0, 0.9, 10)
    ]

    def fake_fit(base, measured_points):
        points = tuple(measured_points)
        return FluxoniumLocalFitResult(
            accepted=True,
            reason="accepted",
            base=base,
            fitted=FluxoniumModelSnapshot(
                base.params, base.flux_half, base.flux_period, 5.0
            ),
            predictor=None,
            n_points=len(points),
            base_rms_mhz=8.0,
            fitted_rms_mhz=3.0,
        )

    monkeypatch.setattr(recovery_mod, "fit_local_fluxonium_model", fake_fit)

    patch = builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "qfw_factor": None}, modules={"readout": _READOUT}
        )
    )

    assert patch.values()["qubit_freq"] == 605.0
    assert len(estimator.replace_calls) == 1
    assert estimator.observe_calls == []


def test_poor_fit_skips_patch_and_predictor_calibration(monkeypatch):
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot

    fit_curve = np.linspace(0.0, 1.0, 11)
    real = fit_curve + 0.25  # residual fails the 0.2 frequency gate.
    builder, env, result, predictor = _mocked_qubit_freq_produce_env(
        monkeypatch,
        real,
        (605.0, 0.0, 4.0, 0.0, fit_curve, None),
    )
    before = predictor.predict_freq(0.0)

    patch = builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "qfw_factor": None}, modules={"readout": _READOUT}
        )
    )

    assert patch.values() == {}
    assert np.isnan(result.fit_freq[0])
    assert predictor.predict_freq(0.0) == before
    correction = env.feedback.estimator("predict_freq_correction")
    assert correction is not None
    assert correction.estimate(0.0) is None


def test_linewidth_gate_rejects_invalid_width_and_window_bounds():
    from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import (
        _is_trusted_linewidth_fit,
    )

    detunes = np.linspace(-5.0, 5.0, 11)
    fit_curve = np.linspace(0.0, 1.0, detunes.size)
    real = fit_curve

    assert _is_trusted_linewidth_fit(4.0, detunes, real, fit_curve)
    assert not _is_trusted_linewidth_fit(0.0, detunes, real, fit_curve)
    assert not _is_trusted_linewidth_fit(-1.0, detunes, real, fit_curve)
    assert not _is_trusted_linewidth_fit(float("nan"), detunes, real, fit_curve)
    assert not _is_trusted_linewidth_fit(11.0, detunes, real, fit_curve)

    nonfinite_detunes = detunes.copy()
    nonfinite_detunes[0] = np.nan
    assert not _is_trusted_linewidth_fit(4.0, nonfinite_detunes, real, fit_curve)
    assert not _is_trusted_linewidth_fit(
        4.0, np.array([0.0], dtype=np.float64), real[:1], fit_curve[:1]
    )
