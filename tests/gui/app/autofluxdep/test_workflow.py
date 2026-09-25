"""Full-workflow dependency-model integration — deps flow across providers.

Drives the controller over a workflow that mirrors the real one's dependency
graph (predictor Service + qubit_freq -> lenrabi -> ro_optimize -> t1 -> t2ramsey
-> t2echo -> mist) and asserts the chain flows: scalars produced upstream are
consumed downstream, modules (pi_pulse / pi2_pulse / opt_readout) are produced and
flow, consumer-declared smoothing is projected, and every provider's sweep Result
is filled.

The Nodes here are FAKE doubles (deterministic ``make_builder`` produce, no
acquire): this isolates the dependency-model wiring under test from experiment
physics, which is covered separately against the flux-aware MockSoc by the
``test_*_acquire.py`` integration tests. Each fake declares the SAME
provides/requires/modules/smoothing as its real counterpart, so the graph the
orchestrator resolves is identical.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.experiments._support.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    Dependency,
    ModuleDep,
    ModuleFallback,
    Need,
)

from ._helpers import make_builder, run_controller_to_completion


def _filling_result(params, flux):
    del params
    return Sweep1DResult.allocate(
        np.asarray(flux, dtype=float), np.linspace(0.0, 1.0, 4), x_label="x"
    )


def _fill_row(env) -> None:
    env.result.signal[env.flux_idx] = np.ones(env.result.n_x)


def _builders():
    """Fake doubles mirroring the real workflow's dependency graph."""

    def qf_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("qubit_freq", 5001.0)
        p.set("fit_detune", 1.0)
        p.set("fit_kappa", 0.05)
        p.set("qfw_factor", 1.0)
        return p

    qubit_freq = make_builder(
        "qubit_freq",
        provides=("qubit_freq", "fit_detune", "fit_kappa", "qfw_factor"),
        requires=(Dependency("predict_freq"),),
        optional=(
            Dependency("qfw_factor", smooth="step_weighted", default=lambda: None),
        ),
        optional_modules=(ModuleDep("readout", default=lambda: None),),
        produce_fn=qf_produce,
        result_factory=_filling_result,
    )

    def lenrabi_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("pi_length", 0.1)
        p.set("pi2_length", 0.05)
        p.set("rabi_freq", 0.5)
        p.set("pi_product", 0.03)
        p.set_module("pi_pulse", {"length": 0.1})
        p.set_module("pi2_pulse", {"length": 0.05})
        return p

    lenrabi = make_builder(
        "lenrabi",
        provides=("pi_length", "pi2_length", "rabi_freq", "pi_product"),
        provides_modules=("pi_pulse", "pi2_pulse"),
        requires=(Dependency("qubit_freq", need=Need.NOW),),
        optional=(
            Dependency("t1", smooth="ewma", default=lambda: None),
            Dependency("pi_length", default=lambda: None),
            Dependency("pi_product", smooth="step_weighted", default=lambda: None),
        ),
        optional_modules=(ModuleDep("opt_readout", default=lambda: None),),
        produce_fn=lenrabi_produce,
        result_factory=_filling_result,
    )

    def ro_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("best_ro_freq", 7444.6)
        p.set("best_ro_gain", 0.5)
        p.set_module("opt_readout", {"freq": 7444.6, "gain": 0.5})
        return p

    ro_optimize = make_builder(
        "ro_optimize",
        provides=("best_ro_freq", "best_ro_gain"),
        provides_modules=("opt_readout",),
        optional=(
            Dependency("t1", smooth="ewma", default=lambda: None),
            Dependency("best_ro_freq", default=lambda: None),
            Dependency("best_ro_gain", default=lambda: None),
        ),
        requires_modules=(
            ModuleDep("pi_pulse", need=Need.NOW, fallback=ModuleFallback.NONE),
        ),
        optional_modules=(ModuleDep("readout", default=lambda: None),),
        produce_fn=ro_produce,
        result_factory=_filling_result,
    )

    def t1_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("t1", 12.0)
        return p

    t1 = make_builder(
        "t1",
        provides=("t1",),
        optional=(Dependency("t1", smooth="ewma", default=lambda: None),),
        requires_modules=(ModuleDep("pi_pulse"),),
        optional_modules=(ModuleDep("opt_readout", default=lambda: None),),
        produce_fn=t1_produce,
        result_factory=_filling_result,
    )

    def t2r_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("t2r", 8.0)
        p.set("t2r_detune", 0.3)
        return p

    t2ramsey = make_builder(
        "t2ramsey",
        provides=("t2r", "t2r_detune"),
        optional=(
            Dependency("t1", smooth="ewma", default=lambda: None),
            Dependency("t2r", smooth="ewma", default=lambda: None),
        ),
        requires_modules=(ModuleDep("pi2_pulse"),),
        optional_modules=(ModuleDep("opt_readout", default=lambda: None),),
        produce_fn=t2r_produce,
        result_factory=_filling_result,
    )

    def t2e_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("t2e", 9.0)
        return p

    t2echo = make_builder(
        "t2echo",
        provides=("t2e",),
        optional=(
            Dependency("t1", smooth="ewma", default=lambda: None),
            Dependency("t2e", smooth="ewma", default=lambda: None),
        ),
        requires_modules=(ModuleDep("pi_pulse"), ModuleDep("pi2_pulse")),
        optional_modules=(ModuleDep("opt_readout", default=lambda: None),),
        produce_fn=t2e_produce,
        result_factory=_filling_result,
    )

    def mist_produce(env, snap):
        _fill_row(env)
        p = Patch()
        p.set("success", 1.0)
        return p

    mist = make_builder(
        "mist",
        provides=("success",),
        requires_modules=(ModuleDep("pi_pulse"),),
        optional_modules=(ModuleDep("opt_readout", default=lambda: None),),
        produce_fn=mist_produce,
        result_factory=_filling_result,
    )

    return [qubit_freq, lenrabi, ro_optimize, t1, t2ramsey, t2echo, mist]


_ALL = ["qubit_freq", "lenrabi", "ro_optimize", "t1", "t2ramsey", "t2echo", "mist"]


def _run_all(flux_values):
    ctrl = build_core()
    for b in _builders():
        ctrl.add_node(b)
    ctrl.set_flux_values(flux_values)
    info = run_controller_to_completion(ctrl)
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
        "pi_product",
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
    for key in ("t1", "t2r", "t2e", "qfw_factor", "pi_product"):
        assert key in info.point_smoothed


def test_full_workflow_fills_every_result():
    ctrl, _info = _run_all([0.0, 0.5, 1.0])
    results = ctrl.state.run_results
    assert set(results) == set(_ALL)  # predictor (a Service) has no Result
    for name, res in results.items():
        # the last flux row's signal is fully filled (the sweep ran to the end)
        assert not np.isnan(res.signal[-1]).any(), f"{name} last row not filled"


def test_predict_freq_flows_to_qubit_freq_consumer():
    # the prepended predictor Service produces predict_freq each point and the
    # qubit_freq fake consumes it (its produce ran → its provides are present),
    # proving the Service-to-Node dependency edge resolved.
    _ctrl, info = _run_all([0.0, 1.0])
    assert "predict_freq" in info.point
    assert "qubit_freq" in info.point  # the consumer resolved + ran
