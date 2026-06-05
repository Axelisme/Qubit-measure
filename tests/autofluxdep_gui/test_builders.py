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
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder


def _run(builder, snapshot_data, snapshot_modules, params=None, flux_idx=1):
    """Build the Result + Node, run produce; return (patch, result, fired)."""
    params = params or {}
    result = builder.make_init_result(params, n_flux=4)
    fired: list = []
    env = RunEnv(
        flux=float(flux_idx),
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
    # plant t1 = prev * 1.1 = 10 * 1.1 = 11
    patch, result, fired = _run(
        T1Builder(),
        {"t1": 10.0},
        {"pi_pulse": "<pi>", "opt_readout": None},
    )
    assert set(patch.values()) == {"t1"}
    assert abs(patch.values()["t1"] - 11.0) < 1.5
    assert not np.isnan(result.signal[1]).any()  # row filled
    assert not np.isnan(result.fit_value[1])  # fitted scalar present
    assert np.isnan(result.fit_value[0])  # untouched row stays nan
    assert len(fired) == 2  # raw fill + fit fill


def test_lenrabi_recovers_pi_lengths_and_produces_pulse_modules():
    patch, result, _fired = _run(
        LenRabiBuilder(),
        {"qubit_freq": 5000.0, "smooth_pi_product": 0.3},
        {"opt_readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"pi_length", "pi2_length", "rabi_freq"}
    assert abs(vals["rabi_freq"] - 0.5) < 0.05  # planted rabi_freq = 0.5
    assert abs(vals["pi_length"] - 1.0) < 0.1  # pi at 1/(2*freq) = 1.0 us
    # produces the pi/pi2 pulse modules for downstream Nodes
    mods = patch.modules()
    assert set(mods) == {"pi_pulse", "pi2_pulse"}
    assert mods["pi_pulse"]["length"] == pytest.approx(vals["pi_length"])
    assert not np.isnan(result.fit_value[1])


def test_t2ramsey_recovers_t2_and_detune():
    patch, _result, _fired = _run(
        T2RamseyBuilder(),
        {"t1": 10.0, "t2r": 5.0},
        {"pi2_pulse": "<pi2>", "opt_readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"t2r", "t2r_detune"}
    assert abs(vals["t2r"] - 5.5) < 1.0  # planted t2 = 5.0 * 1.1
    assert abs(vals["t2r_detune"] - 0.3) < 0.05  # planted fringe = 0.3


def test_t2echo_recovers_t2():
    patch, _result, _fired = _run(
        T2EchoBuilder(),
        {"t1": 10.0, "t2e": 5.0},
        {"pi_pulse": "<pi>", "pi2_pulse": "<pi2>", "opt_readout": None},
    )
    vals = patch.values()
    assert set(vals) == {"t2e"}
    assert abs(vals["t2e"] - 5.5) < 1.0  # planted t2 = 5.0 * 1.1


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
