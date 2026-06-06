"""Synthetic-acquire helpers — the delay + per-round accumulation.

The synthetic sweep would finish in milliseconds; to make the liveplot advance
visibly, each Node sleeps a per-flux-point ``acquire_delay`` (worker thread, so
the UI never freezes) and emulates a multi-round acquire — running-averaging
``rounds`` noisy passes so the row settles round by round, exactly like a real
acquire. Both defaults are seeded into a GUI-placed Node by the controller; a
directly-constructed Node (no params) sleeps zero and runs a single round, so
tests run instantly.
"""

from __future__ import annotations

import time

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    DEFAULT_ACQUIRE_DELAY,
    DEFAULT_ROUNDS,
    accumulate_rounds,
    exp_decay,
    resolve_acquire_delay,
    resolve_rounds,
    signal_to_real,
    simulate_acquire_delay,
)
from zcu_tools.utils.fitting import fit_decay

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
    result = builder.make_init_result(
        {"detune_sweep": "-20,50,0.5"}, np.linspace(0.0, 1.0, 2)
    )
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


# --- resolve_rounds: params → round count (1 when unset/bad) ---


def test_resolve_rounds_missing_is_one():
    assert resolve_rounds({}) == 1


def test_resolve_rounds_parses_and_clamps():
    assert resolve_rounds({"rounds": 10}) == 10
    assert resolve_rounds({"rounds": "5"}) == 5
    assert resolve_rounds({"rounds": 0}) == 1  # clamped to at least 1
    assert resolve_rounds({"rounds": "x"}) == 1  # bad value → 1


# --- accumulate_rounds: running-average settles + calls on_round per round ---


def test_accumulate_calls_on_round_per_round():
    seen = []
    accumulate_rounds(
        lambda k: np.array([float(k)]), 4, lambda avg, k: seen.append(k), delay=0.0
    )
    assert seen == [0, 1, 2, 3]  # one call per round, in order


def test_accumulate_running_average_settles_toward_clean():
    times = np.linspace(0.5, 60, 101)
    clean = exp_decay(times, 15.0, noise=0.0, seed=0)
    errors = []

    def make_round(k):
        return exp_decay(times, 15.0, noise=0.05, seed=100 + k)

    final = accumulate_rounds(
        make_round, 10, lambda avg, k: errors.append(np.abs(avg - clean).mean())
    )
    # noise ∝ 1/√k: the last round is markedly cleaner than the first
    assert errors[-1] < errors[0] * 0.6
    # the fully-averaged signal still fits the planted t1
    t1, _, _, _ = fit_decay(times, signal_to_real(final))
    assert abs(t1 - 15.0) < 1.5


def test_accumulate_single_round_is_one_pass():
    calls = []
    out = accumulate_rounds(
        lambda k: np.array([1.0, 2.0]), 1, lambda avg, k: calls.append(k)
    )
    assert calls == [0]  # exactly one pass
    assert list(out) == [1.0, 2.0]  # the single round's signal, unaveraged


# --- the GUI seed also seeds rounds; produce notifies once per round + fit ---


def test_add_node_by_type_seeds_default_rounds():
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    assert node.params["rounds"] == DEFAULT_ROUNDS


def test_produce_notifies_once_per_round_plus_fit():
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot

    builder = QubitFreqBuilder()
    result = builder.make_init_result(
        {"detune_sweep": "-20,50,0.5"}, np.linspace(0.0, 1.0, 2)
    )
    notifies = []
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={"rounds": 6, "acquire_delay": 0},
        result=result,
        round_hook=lambda idx: notifies.append(idx),
    )
    node = builder.build_node(env)
    snap = Snapshot(
        {"predict_freq": 5000.0, "fit_kappa": 0.05}, modules={"readout": None}
    )
    node.produce(snap)
    # 6 per-round redraws (the row settling) + 1 after the fit fills the curve
    assert len(notifies) == 7
