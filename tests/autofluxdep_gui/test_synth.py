"""Synthetic-acquire helpers — the delay that paces the liveplot.

The synthetic sweep would finish in milliseconds; to make the liveplot advance
visibly, each Node sleeps a per-flux-point ``acquire_delay`` (worker thread, so
the UI never freezes). The default is seeded into a GUI-placed Node by the
controller; a directly-constructed Node (no params) sleeps zero so tests run
instantly.
"""

from __future__ import annotations

import time

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    DEFAULT_ACQUIRE_DELAY,
    resolve_acquire_delay,
    simulate_acquire_delay,
)

# --- resolve_acquire_delay: params → seconds (0 when unset/bad) ---


def test_resolve_missing_is_zero():
    assert resolve_acquire_delay({}) == 0.0


def test_resolve_none_or_blank_is_zero():
    assert resolve_acquire_delay({"acquire_delay": None}) == 0.0
    assert resolve_acquire_delay({"acquire_delay": ""}) == 0.0


def test_resolve_parses_number_and_text():
    assert resolve_acquire_delay({"acquire_delay": 0.05}) == 0.05
    assert resolve_acquire_delay({"acquire_delay": "0.2"}) == 0.2


def test_resolve_clamps_negative_to_zero():
    assert resolve_acquire_delay({"acquire_delay": -1.0}) == 0.0


def test_resolve_bad_value_degrades_to_zero():
    assert resolve_acquire_delay({"acquire_delay": "abc"}) == 0.0


# --- simulate_acquire_delay: sleeps only for a positive value ---


def test_simulate_zero_is_instant():
    t0 = time.perf_counter()
    simulate_acquire_delay(0)
    assert time.perf_counter() - t0 < 0.02


def test_simulate_positive_sleeps():
    t0 = time.perf_counter()
    simulate_acquire_delay(0.05)
    assert time.perf_counter() - t0 >= 0.05


# --- the GUI-placement seed: add_node_by_type seeds the default delay ---


def test_add_node_by_type_seeds_default_delay():
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    # qubit_freq exposes acquire_delay → seeded so the GUI run paces visibly
    assert node.params["acquire_delay"] == DEFAULT_ACQUIRE_DELAY


def test_add_node_directly_has_no_delay():
    # add_node (programmatic / tests) does NOT seed the delay — explicit only
    ctrl = build_core()
    node = ctrl.add_node(QubitFreqBuilder())
    assert "acquire_delay" not in node.params
    # …and a Node with no delay param resolves to zero (runs instantly)
    assert resolve_acquire_delay(node.params) == 0.0


# --- a produce with a positive delay actually sleeps ---


def test_produce_sleeps_for_positive_delay():
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot

    builder = QubitFreqBuilder()
    result = builder.make_init_result({"detune_sweep": "-20,50,0.5"}, n_flux=2)
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={"acquire_delay": 0.05},
        result=result,
    )
    node = builder.build_node(env)
    snap = Snapshot(
        {"predict_freq": 5000.0, "fit_kappa": 0.05}, modules={"readout": None}
    )
    t0 = time.perf_counter()
    node.produce(snap)
    assert time.perf_counter() - t0 >= 0.05
