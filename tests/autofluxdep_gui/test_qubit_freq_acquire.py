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
from zcu_tools.gui.session.services.mock_flux import FAKE_FLUX_DEVICE_NAME

from ._helpers import connect_mock

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
    node.params.update(
        {
            "qub_waveform": "qub_drive",
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

    # Flux device values placing f01 sub-Nyquist and clearly separated:
    #   value 0.0 -> f01 ~582 MHz, 0.06 -> ~1673 MHz, 0.1 -> ~2640 MHz.
    flux_values = [0.0, 0.06, 0.1]
    ctrl.set_flux_values(flux_values)
    ctrl.start_run()

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
        "qub_waveform": "qub_drive",
        "qub_ch": 1,
        "qub_nqz": 1,
        "qub_gain": 0.3,
        "qub_length": 1.0,
        "reps": 100,
        "rounds": 1,
        "relax_delay": 0.0,
        "detune_sweep": SweepValue(start=-60.0, stop=60.0, expts=121),
    }
    result = builder.make_init_result(params, flux)
    figure = Figure()
    plotter = builder.make_plotter(figure)
    ctx = ctrl.state.exp_context
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params=params,
        soc=ctx.soc,
        soccfg=ctx.soccfg,
        ml=ctx.ml,
        flux_device=FAKE_FLUX_DEVICE_NAME,
        result=result,
    )
    builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "fit_kappa": 0.05}, modules={"readout": _READOUT}
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
        "qub_waveform": "qub_drive",
        "qub_ch": 1,
        "qub_nqz": 1,
        "qub_gain": 0.3,
        "qub_length": 1.0,
        "reps": 100,
        "rounds": 1,
        "relax_delay": 0.0,
        "detune_sweep": SweepValue(start=-60.0, stop=60.0, expts=121),
    }
    result = builder.make_init_result(params, np.array([0.0]))
    predictor = SimplePredictor(base=600.0, slope=50.0)
    before = predictor.predict_freq(0.0)
    ctx = ctrl.state.exp_context
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params=params,
        soc=ctx.soc,
        soccfg=ctx.soccfg,
        ml=ctx.ml,
        flux_device=FAKE_FLUX_DEVICE_NAME,
        result=result,
        tools=Tools(predictor=predictor),
    )
    patch = builder.build_node(env).produce(
        Snapshot(
            {"predict_freq": 600.0, "fit_kappa": 0.05}, modules={"readout": _READOUT}
        )
    )
    assert "qubit_freq" in patch.values()  # a good fit
    # the predictor was calibrated at flux 0 toward the measured frequency
    assert predictor.predict_freq(0.0) != before
    assert abs(predictor.predict_freq(0.0) - patch.values()["qubit_freq"]) < 1e-6
