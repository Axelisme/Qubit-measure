"""Full-workflow integration — all 7 experiments sweep with deps flowing.

Drives the controller over the whole registered workflow (predictor Service +
qubit_freq → lenrabi → ro_optimize → t1 → t2ramsey → t2echo → mist) on synthetic
signals, and asserts the dependency chain flows: scalars produced upstream are
consumed downstream, modules (pi_pulse / pi2_pulse / opt_readout) are produced
and flow, smoothing is projected, and every provider's sweep Result is filled.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core

from ._helpers import connect_mock

_ALL = ["qubit_freq", "lenrabi", "ro_optimize", "t1", "t2ramsey", "t2echo", "mist"]


def _run_all(flux_values):
    ctrl = build_core()
    for t in _ALL:
        ctrl.add_node_by_type(t)
    # add_node_by_type seeds a GUI-pacing acquire_delay; zero it so the headless
    # integration run is instant (the delay is exercised in test_synth).
    for node in ctrl.state.nodes:
        node.params["acquire_delay"] = 0
    connect_mock(ctrl)
    ctrl.set_flux_values(flux_values)
    info = ctrl.start_run()
    return ctrl, info


def test_full_workflow_produces_every_scalar():
    _ctrl, info = _run_all([0.0, 0.5, 1.0])
    point = info.point
    # predictor Service + every measurement scalar present at the last point
    for key in (
        "predict_freq",
        "cur_m",
        "qubit_freq",
        "pi_length",
        "pi2_length",
        "rabi_freq",
        "best_ro_freq",
        "best_ro_gain",
        "t1",
        "t2r",
        "t2r_detune",
        "t2e",
        "success",
    ):
        assert key in point, f"missing {key}"
        assert not np.isnan(point[key])


def test_full_workflow_flows_modules():
    _ctrl, info = _run_all([0.0, 0.5, 1.0])
    # lenrabi produced pi/pi2 pulses; ro_optimize produced the tuned readout
    assert set(info.module_point) == {"pi_pulse", "pi2_pulse", "opt_readout"}
    assert info.module_point["opt_readout"]["freq"] == info.point["best_ro_freq"]


def test_full_workflow_projects_smoothed_values():
    _ctrl, info = _run_all([0.0, 0.5, 1.0])
    # consumer-declared smoothing fired for the keys downstream reads smoothed
    for key in ("t1", "t2r", "t2e", "fit_kappa"):
        assert key in info.point_smoothed


def test_full_workflow_fills_every_result():
    ctrl, _info = _run_all([0.0, 0.5, 1.0])
    results = ctrl.state.run_results
    assert set(results) == set(_ALL)  # predictor (a Service) has no Result
    for name, res in results.items():
        # the last flux row's signal is fully filled (the sweep ran to the end)
        assert not np.isnan(res.signal[-1]).any(), f"{name} last row not filled"


def test_qubit_freq_tracks_predictor():
    # closed-loop feedback: the true resonance drifts with flux (flux_drift), so
    # flux 0.0's measurement calibrates the predictor and flux 1.0's predict_freq
    # adapts ABOVE the bare linear base 5050 (tracking the measured drift) rather
    # than staying fixed. qubit_freq stays a small offset above its (now adapted)
    # predict — near the drifted resonance.
    _ctrl, info = _run_all([0.0, 1.0])
    assert info.point["predict_freq"] > 5050.0  # adapted, not the bare linear 5050
    # qubit_freq sits within the detune window above the adapted predict
    offset = info.point["qubit_freq"] - info.point["predict_freq"]
    assert 0.0 < offset < 50.0  # the drifted resonance is within the sweep window
