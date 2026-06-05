"""Node run-body tests — synthetic signal → real fit recovers planted params.

The qubit_freq dry-run body synthesises a Lorentzian dip and fits it with the
real ``fit_qubit_freq``; the fitted freq/kappa must recover the planted values.
This exercises the full acquire→fit→produce path without hardware (MockSoc gives
only noise, so the body synthesises instead of calling soc.acquire).
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QUBIT_FREQ_SPEC


def _run_qubit_freq(predict_freq: float, flux_idx: int = 0):
    snap = Snapshot(
        {"predict_freq": predict_freq, "fit_kappa": 0.05, "flux_idx": flux_idx},
        modules={"readout": "<ro>"},
    )
    assert QUBIT_FREQ_SPEC.build_cfg is not None
    assert QUBIT_FREQ_SPEC.run_body is not None
    cfg = QUBIT_FREQ_SPEC.build_cfg(snap, {"detune_sweep": "-20,50,0.5"}, None)
    return QUBIT_FREQ_SPEC.run_body(cfg, snap, None, "<mock-soc>")


def test_run_body_recovers_planted_freq_and_kappa():
    patch = _run_qubit_freq(5000.0)
    vals = patch.values()
    # planted: true_freq = predict + 1.5, true_fwhm = 2.0
    assert abs(vals["qubit_freq"] - 5001.5) < 0.5
    assert abs(vals["fit_detune"] - 1.5) < 0.5
    assert abs(vals["fit_kappa"] - 2.0) < 0.6


def test_run_body_produces_exactly_the_declared_keys():
    patch = _run_qubit_freq(5000.0)
    assert set(patch.values()) == {"qubit_freq", "fit_detune", "fit_kappa"}
    assert patch.modules() == {}  # qubit_freq produces no module


def test_run_body_tracks_the_predicted_frequency():
    # the resonance sits 1.5 MHz above the prediction → fitted freq follows it
    a = _run_qubit_freq(5000.0).values()["qubit_freq"]
    b = _run_qubit_freq(5200.0).values()["qubit_freq"]
    assert abs((b - a) - 200.0) < 1.0


def test_run_body_deterministic_per_flux_index():
    # same flux index → same synthetic noise seed → identical fit
    assert (
        _run_qubit_freq(5000.0, flux_idx=3).values()
        == _run_qubit_freq(5000.0, flux_idx=3).values()
    )


def test_controller_run_drives_real_run_body():
    # a full controller run (no Qt) should drive qubit_freq's run_body and leave
    # a fitted qubit_freq in the final InfoStore — not the fake fallback's 1.0.
    from zcu_tools.gui.app.autofluxdep.app import build_core

    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    ctrl.setup(use_mock=True)
    ctrl.set_flux_values([0.0, 1.0])
    info = ctrl.start_run()
    # predict_freq at idx 1 = 5000 + 50 = 5050 → resonance at 5051.5
    assert abs(info.point["qubit_freq"] - 5051.5) < 1.0
    assert info.point["qubit_freq"] != 1.0  # not the fake fallback
