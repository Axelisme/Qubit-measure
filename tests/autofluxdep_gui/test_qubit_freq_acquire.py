"""RB-1: qubit_freq end-to-end real acquire against the flux-aware MockSoc.

This is the load-bearing Phase RB test. It runs the qubit_freq Node's *real*
acquire path (set flux device -> setup_devices -> TwoToneProgram.acquire -> fit ->
calibrate) at several flux points and asserts the fitted qubit frequency CHANGES
with flux. If the picked flux device never received the value (the name/label
silent-miss), the SimEngine would stay pinned at one operating point and the fit
freq would be constant -- so a constant fit fails this test.

The GUI predictor is a FluxoniumPredictor matching DEFAULT_SIMPARAM, so the
predicted drive centre tracks the SimEngine's actual f01 and the dip lands in the
detune window. Flux device values are chosen sub-Nyquist (f01 < 3072 MHz) so the
dip is recoverable.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import SweepValue
from zcu_tools.gui.app.autofluxdep.feedback import build_feedback_runtime
from zcu_tools.gui.session.services.mock_flux import FAKE_FLUX_DEVICE_NAME

from ._helpers import connect_mock, run_controller_to_completion

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
    "ro_cfg": {"ro_ch": 0, "ro_length": 0.9, "trig_offset": 0.6},
}


class _Provider:
    def __init__(self, name, builder, schema):
        self.name = name
        self.builder = builder
        self.schema = schema


def _configure_context(ctrl) -> None:
    """Populate the active context's ml so qubit_freq's make_cfg has a readout +
    drive waveform.

    The predicted-centre predictor is NOT built here: connect_mock has already run
    the MockFluxProvisioner, which installs a FluxoniumPredictor derived from the
    mock soc's SimParams (matching the SimEngine's f01). Relying on that provisioned
    predictor exercises the real mock-connect path rather than a hand-built copy."""
    ml = ctrl.state.exp_context.ml
    ml.register_waveform(
        qub_drive={"style": "const", "length": 1.0},
    )
    ml.register_module(readout=_READOUT)


def test_qubit_freq_acquire_fit_varies_with_flux():
    ctrl = build_core()
    connect_mock(ctrl)
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
            "detune_sweep": SweepValue(start=-60.0, stop=60.0, expts=121),
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
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder

    ctrl = build_core()
    connect_mock(ctrl)
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
        "detune_sweep": SweepValue(start=-60.0, stop=60.0, expts=121),
    }
    schema = builder.make_default_schema().with_overrides(params)
    result = builder.make_init_result(schema, flux)
    figure = Figure()
    plotter = builder.make_plotter(figure)
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
    )
    builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "qfw_factor": None}, modules={"readout": _READOUT}
        )
    )
    plotter.update(result, 0)  # must not raise


def test_good_fit_calibrates_the_predictor():
    # the closed-loop trigger: a good real-acquire fit feeds predictor.calibrate, so
    # the predictor's prediction at the measured flux moves toward the measurement.
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor, Tools

    ctrl = build_core()
    connect_mock(ctrl)
    _configure_context(ctrl)
    ctrl.set_flux_device(FAKE_FLUX_DEVICE_NAME)

    builder = QubitFreqBuilder()
    params = {
        "qub_ch": 1,
        "qub_nqz": 1,
        "qub_gain": 0.3,
        "qub_length": 1.0,
        "reps": 100,
        "rounds": 1,
        "relax_delay": 0.0,
        "detune_sweep": SweepValue(start=-60.0, stop=60.0, expts=121),
    }
    schema = builder.make_default_schema().with_overrides(params)
    result = builder.make_init_result(schema, np.array([0.0]))
    predictor = SimplePredictor(base=600.0, slope=50.0)
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
            {"predict_freq": 600.0, "qfw_factor": None}, modules={"readout": _READOUT}
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
    assert abs(predictor.predict_freq(0.0) + estimate - values["qubit_freq"]) < 1e-6


def _mocked_qubit_freq_produce_env(monkeypatch, real, fit_return):
    from zcu_tools.gui.app.autofluxdep.nodes import qubit_freq as qf_mod
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor, Tools

    ctrl = build_core()
    connect_mock(ctrl)
    _configure_context(ctrl)

    builder = QubitFreqBuilder()
    schema = builder.make_default_schema().with_overrides(
        {
            "qub_ch": 1,
            "qub_nqz": 1,
            "qub_gain": 0.25,
            "qub_length": 1.0,
            "reps": 1,
            "rounds": 1,
            "relax_delay": 0.0,
            "detune_sweep": SweepValue(start=-5.0, stop=5.0, expts=11),
        }
    )
    result = builder.make_init_result(schema, np.array([0.0]))

    class _DummyTwoToneProgram:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def acquire(self, *args, **kwargs):
            del args, kwargs
            return [[np.zeros((result.n_detune, 2), dtype=np.float64)]]

    monkeypatch.setattr(qf_mod, "TwoToneProgram", _DummyTwoToneProgram)
    monkeypatch.setattr(qf_mod, "setup_devices", lambda *args, **kwargs: None)
    monkeypatch.setattr(qf_mod, "set_flux_by_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(qf_mod, "_signal2real", lambda _signals: real)
    monkeypatch.setattr(qf_mod, "fit_qubit_freq", lambda _freqs, _real: fit_return)

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


def test_medium_fit_calibrates_predictor_without_linewidth_feedback(monkeypatch):
    # A fit between the frequency and linewidth gates is still useful for centring
    # the predictor, but its FWHM must not drive qfw_factor/gain feedback.
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot

    fit_curve = np.linspace(0.0, 1.0, 11)
    real = fit_curve + 0.15  # residual passes 0.2 gate, fails 0.1 gate.
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

    values = patch.values()
    assert values["qubit_freq"] == 605.0
    assert values["fit_detune"] == 5.0
    assert "fit_kappa" not in values
    assert "qfw_factor" not in values
    assert result.fit_freq[0] == 605.0
    assert predictor.predict_freq(0.0) == before
    correction = env.feedback.estimator("predict_freq_correction")
    assert correction is not None
    estimate = correction.estimate(0.0)
    assert estimate is not None
    assert abs(predictor.predict_freq(0.0) + estimate - 605.0) < 1e-6


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
    from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import _is_trusted_linewidth_fit

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
