"""Per-experiment Builder tests — synth → real fit/argmax → fill Result → Patch.

Each migrated experiment Builder synthesises a physical signal, recovers the
planted parameters with the real fitter (or argmax / variance for the fit-less
ones), fills its sweep Result row in place, fires the round_hook, and returns the
declared Patch. One parametrised set covers the 1D fit experiments; ro_optimize
(2D argmax) and mist (variance, no fit) are checked separately for their distinct
shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder


def _run(builder, snapshot_data, snapshot_modules, params=None, flux_idx=1):
    """Build the Result + Node, run produce; return (patch, result, fired).

    Uses flux_arr[flux_idx] as the flux value (not float(flux_idx)) so that the
    fit-quality gate always sees a deterministic SNR for the chosen array index.
    flux_idx=1 → flux=0.1 (SNR≈0.97), a high-SNR point safe for recovery tests.
    """
    params = params or {}
    flux_arr = np.linspace(0.0, 1.0, 11)  # 11-point sweep; idx=1→flux=0.1 (SNR≈0.97)
    result = builder.make_init_result(params, flux_arr)
    fired: list = []
    env = RunEnv(
        flux=float(flux_arr[flux_idx]),
        flux_idx=flux_idx,
        params=params,
        result=result,
        round_hook=lambda payload: fired.append(payload),
    )
    node = builder.build_node(env)
    snap = Snapshot(snapshot_data, modules=snapshot_modules)
    patch = node.produce(snap)
    return patch, result, fired


# --- 1D fit experiments (t1 / lenrabi / t2ramsey / t2echo) ---


def test_t1_recovers_planted_decay():
    # flux=linspace(0,1,11)[1]=0.1 → flux_drift(0.1,10,40)=16.4 us
    patch, result, fired = _run(
        T1Builder(),
        {"t1": 10.0},
        {"pi_pulse": "<pi>", "opt_readout": None},
    )
    assert set(patch.values()) == {"t1"}
    assert abs(patch.values()["t1"] - 16.4) < 2.0
    assert not np.isnan(result.signal[1]).any()  # row filled
    assert not np.isnan(result.fit_value[1])  # fitted scalar present
    assert np.isnan(result.fit_value[0])  # untouched row stays nan
    assert len(fired) == 2  # raw fill + fit fill


def test_lenrabi_recovers_pi_lengths_and_produces_pulse_modules():
    # flux=0.1 → rabi_freq=flux_drift(0.1,0.5,0.4)=0.564, pi≈0.886 us
    patch, result, _fired = _run(
        LenRabiBuilder(),
        {"qubit_freq": 5000.0, "smooth_pi_product": 0.3},
        {"opt_readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"pi_length", "pi2_length", "rabi_freq"}
    assert abs(vals["rabi_freq"] - 0.564) < 0.07  # planted rabi_freq at flux=0.1
    assert abs(vals["pi_length"] - 0.887) < 0.12  # pi at 1/(2*0.564) ≈ 0.887 us
    # produces the pi/pi2 pulse modules for downstream Nodes
    mods = patch.modules()
    assert set(mods) == {"pi_pulse", "pi2_pulse"}
    assert mods["pi_pulse"]["length"] == pytest.approx(vals["pi_length"])
    assert not np.isnan(result.fit_value[1])


def test_t2ramsey_recovers_t2_and_detune():
    # flux=0.1 → flux_drift(0.1,5.0,15.0)=7.4 us
    patch, _result, _fired = _run(
        T2RamseyBuilder(),
        {"t1": 10.0, "t2r": 5.0},
        {"pi2_pulse": "<pi2>", "opt_readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"t2r", "t2r_detune"}
    assert abs(vals["t2r"] - 7.4) < 2.0  # planted t2 at flux=0.1
    assert abs(vals["t2r_detune"] - 0.3) < 0.05  # planted fringe = 0.3


def test_t2echo_recovers_t2():
    # flux=0.1 → flux_drift(0.1,6.0,15.0)=8.4 us
    patch, _result, _fired = _run(
        T2EchoBuilder(),
        {"t1": 10.0, "t2e": 5.0},
        {"pi_pulse": "<pi>", "pi2_pulse": "<pi2>", "opt_readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"t2e"}
    assert abs(vals["t2e"] - 8.4) < 2.0  # planted t2 at flux=0.1


# --- 2D argmax (ro_optimize) ---


def test_ro_optimize_recovers_peak_and_produces_readout_module():
    patch, result, _fired = _run(
        RoOptimizeBuilder(),
        {"t1": 10.0, "best_ro_freq": 5000.0, "best_ro_gain": 0.5},
        {"pi_pulse": "<pi>", "readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"best_ro_freq", "best_ro_gain"}
    # peak planted at prev_freq + 0.2 = 5000.2, gain clamped to prev 0.5
    assert abs(vals["best_ro_freq"] - 5000.2) < 0.3
    assert abs(vals["best_ro_gain"] - 0.5) < 0.05
    # 3D Result: this flux row's landscape is filled
    assert result.signal.ndim == 3
    assert not np.isnan(result.signal[1]).any()
    assert not np.isnan(result.best_freq[1])
    # produces the tuned opt_readout module
    mods = patch.modules()
    assert set(mods) == {"opt_readout"}
    assert mods["opt_readout"]["freq"] == pytest.approx(vals["best_ro_freq"])


# --- variance, no fit (mist) ---


def test_mist_fills_variance_and_reports_success():
    patch, result, _fired = _run(
        MistBuilder(),
        {},
        {"pi_pulse": "<pi>", "opt_readout": None},
    )
    assert patch.values() == {"success": 1.0}
    assert patch.modules() == {}
    # the variance curve is filled; mist has no fit, so fit_value stays nan
    assert not np.isnan(result.signal[1]).any()
    assert np.isnan(result.fit_value[1])
    assert np.isnan(result.fit_curve[1]).all()


# --- fit-quality gate (SNR-trough points yield empty Patch + nan fit_value) ---

# flux=linspace(0,1,11): idx=6 (flux=0.6) is the SNR dead point (snr≈0.001),
# so the gating logic must discard the fit and return an empty Patch.  idx=8
# (flux=0.8, snr≈0.97) is a healthy point that must produce the full key-set.

_DEAD_IDX = 6  # flux=0.6: SNR ≈ 0.001 → gated out
_GOOD_IDX = 8  # flux=0.8: SNR ≈ 0.965 → passes gate


def _run_sweep(builder, snap_data, snap_modules, flux_idx):
    """Run one flux point over an 11-point sweep; return (patch, result)."""
    flux_arr = np.linspace(0.0, 1.0, 11)
    params = {"rounds": 4, "acquire_delay": 0}
    result = builder.make_init_result(params, flux_arr)
    env = RunEnv(
        flux=float(flux_arr[flux_idx]),
        flux_idx=flux_idx,
        params=params,
        result=result,
    )
    snap = Snapshot(snap_data, modules=snap_modules)
    patch = builder.build_node(env).produce(snap)
    return patch, result


def test_t1_snr_gate_dead_point_yields_empty_patch():
    patch, result = _run_sweep(
        T1Builder(),
        {"t1": 10.0},
        {"pi_pulse": "<pi>", "opt_readout": None},
        _DEAD_IDX,
    )
    assert patch.values() == {}  # no t1 produced
    assert np.isnan(result.fit_value[_DEAD_IDX])


def test_t1_snr_gate_good_point_yields_t1():
    patch, result = _run_sweep(
        T1Builder(),
        {"t1": 10.0},
        {"pi_pulse": "<pi>", "opt_readout": None},
        _GOOD_IDX,
    )
    assert "t1" in patch.values()
    assert not np.isnan(result.fit_value[_GOOD_IDX])


def test_lenrabi_snr_gate_dead_point_yields_empty_patch():
    patch, result = _run_sweep(
        LenRabiBuilder(),
        {"qubit_freq": 5000.0, "smooth_pi_product": 0.3},
        {"opt_readout": None},
        _DEAD_IDX,
    )
    assert patch.values() == {}
    assert patch.modules() == {}  # no pi_pulse/pi2_pulse modules either
    assert np.isnan(result.fit_value[_DEAD_IDX])


def test_lenrabi_snr_gate_good_point_yields_pi_lengths():
    patch, result = _run_sweep(
        LenRabiBuilder(),
        {"qubit_freq": 5000.0, "smooth_pi_product": 0.3},
        {"opt_readout": None},
        _GOOD_IDX,
    )
    assert {"pi_length", "pi2_length", "rabi_freq"} == set(patch.values())
    assert {"pi_pulse", "pi2_pulse"} == set(patch.modules())
    assert not np.isnan(result.fit_value[_GOOD_IDX])


def test_t2ramsey_snr_gate_dead_point_yields_empty_patch():
    patch, result = _run_sweep(
        T2RamseyBuilder(),
        {"t1": 10.0, "t2r": 5.0},
        {"pi2_pulse": "<pi2>", "opt_readout": None},
        _DEAD_IDX,
    )
    assert patch.values() == {}
    assert np.isnan(result.fit_value[_DEAD_IDX])


def test_t2ramsey_snr_gate_good_point_yields_t2r():
    patch, result = _run_sweep(
        T2RamseyBuilder(),
        {"t1": 10.0, "t2r": 5.0},
        {"pi2_pulse": "<pi2>", "opt_readout": None},
        _GOOD_IDX,
    )
    assert {"t2r", "t2r_detune"} == set(patch.values())
    assert not np.isnan(result.fit_value[_GOOD_IDX])


def test_t2echo_snr_gate_dead_point_yields_empty_patch():
    patch, result = _run_sweep(
        T2EchoBuilder(),
        {"t1": 10.0, "t2e": 5.0},
        {"pi_pulse": "<pi>", "pi2_pulse": "<pi2>", "opt_readout": None},
        _DEAD_IDX,
    )
    assert patch.values() == {}
    assert np.isnan(result.fit_value[_DEAD_IDX])


def test_t2echo_snr_gate_good_point_yields_t2e():
    patch, result = _run_sweep(
        T2EchoBuilder(),
        {"t1": 10.0, "t2e": 5.0},
        {"pi_pulse": "<pi>", "pi2_pulse": "<pi2>", "opt_readout": None},
        _GOOD_IDX,
    )
    assert {"t2e"} == set(patch.values())
    assert not np.isnan(result.fit_value[_GOOD_IDX])


# --- registry exposes all migrated measurement types ---


def test_registry_exposes_all_experiments():
    from zcu_tools.gui.app.autofluxdep.registry import available_node_types

    types = set(available_node_types())
    assert types == {
        "qubit_freq",
        "lenrabi",
        "ro_optimize",
        "t1",
        "t2ramsey",
        "t2echo",
        "mist",
    }


# --- liveplot alignment: each Builder builds the runner module's subplot layout ---


@pytest.mark.parametrize(
    ("type_name", "n_axes"),
    [
        ("qubit_freq", 3),  # fit_freq (1) + detune 2DwithLine (2)
        ("lenrabi", 2),  # rabi_curve 2DwithLine (2D + line)
        ("ro_optimize", 1),  # snr 2D landscape
        ("t1", 2),  # scalar scatter + current-point curve
        ("t2ramsey", 2),
        ("t2echo", 2),
        ("mist", 2),  # flux×gain 2DwithLine
    ],
)
def test_make_plotter_builds_aligned_subplots(type_name, n_axes):
    # each experiment's Plotter embeds the same LivePlot panels the runner module
    # draws, so the figure has the matching number of axes.
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.registry import create_placement

    builder = create_placement(type_name).builder
    figure = Figure()
    plotter = builder.make_plotter(figure)
    assert plotter is not None
    assert len(figure.axes) == n_axes


def test_plotter_update_runs_after_a_real_produce():
    # build qubit_freq's Result + Plotter, fill a row via produce, redraw — the
    # LivePlot-backed update path must not raise (existed_axes + host draw).
    from matplotlib.figure import Figure

    builder = QubitFreqBuilder()
    flux = np.linspace(0.0, 1.0, 3)
    result = builder.make_init_result({"detune_sweep": "-20,50,0.5"}, flux)
    figure = Figure()
    plotter = builder.make_plotter(figure)
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={"rounds": 2, "acquire_delay": 0},
        result=result,
    )
    builder.build_node(env).produce(
        Snapshot({"predict_freq": 5000.0, "fit_kappa": 0.05}, modules={"readout": None})
    )
    plotter.update(result, 0)  # must not raise
