"""Calibrated gain reuse through scoped kernel preparation and density playback."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.cache import PrepareCache
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel

from ..contract.kernel_suite import synthetic_model


def scan(gains, *, frequency=0.05, port=0):
    count = len(gains)
    return SegmentTable(
        timeline=np.arange(count),
        start_ns=np.zeros(count),
        duration_ns=np.full(count, 23.7),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.full((count, 1), port, dtype=np.int64),
        tone_frequency_ghz=np.full((count, 1), frequency),
        tone_amplitude=np.array(gains)[:, None],
        tone_theta_rad=np.full((count, 1), 1.73),
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )


@pytest.fixture
def source():
    model = synthetic_model()
    model = replace(
        model,
        energies_ghz=np.array([0.0, 0.8, 1.6]),
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
        drives=(model.drives[0], replace(model.drives[0], name="other")),
    )
    return ModelSource(np.ones((3, 3), dtype=np.complex128) / 3, lambda level: model)


def scoped(table, source):
    factory = getattr(FloquetPeriodKernel(), "evolve_scope", None)
    assert callable(factory), "period preparation needs an evolve-local scope"
    return FloquetPeriodKernel().evolve_scope(table, source)


def prepare(kernel, table, source, budget=1e-3):
    rows = np.arange(len(table.timeline), dtype=np.int64)
    return kernel.prepare(kernel.prepare_keys(table, rows), table, rows, source, budget)


def playback(kernel, prepared, table, source, budget=1e-3):
    rows = np.arange(len(table.timeline), dtype=np.int64)
    states = np.repeat(source.initial_density[None], len(rows), axis=0)
    kernel.apply(prepared, rows, table, rows, states, budget)
    return states


def test_gain_predictor_is_checked_and_matches_independent_playback(source):
    table = scan(np.arange(6) * 1e-5 + 0.003)
    kernel, workspace = scoped(table, source)
    cache = PrepareCache(0)
    with cache.track_workspace(workspace):
        assert cache.live_bytes == 0
        prepared = prepare(kernel, table, source)
        assert cache.live_bytes == workspace.nbytes > 0
    assert cache.live_bytes == 0
    assert workspace.attempts >= 1
    assert workspace.hits >= 1
    assert workspace.nbytes > 0
    independent = prepare(FloquetPeriodKernel(), table, source)
    # Include translated phase, a cross-period tail, and coherent input.
    np.testing.assert_allclose(
        playback(kernel, prepared, table, source),
        playback(FloquetPeriodKernel(), independent, table, source),
        atol=1e-4,
        rtol=0,
    )
    # A new evolve does not inherit the previous call's retained tables.
    _, fresh = scoped(table, source)
    assert fresh.nbytes == fresh.attempts == fresh.hits == 0


def test_rejected_predictor_preserves_independent_result_bitwise(source):
    table = scan([0.003, 0.006, 0.009, 0.012])
    kernel, workspace = scoped(table, source)
    actual = prepare(kernel, table, source, 1e-10)
    assert workspace.attempts > 0
    assert workspace.hits == 0
    independent = prepare(FloquetPeriodKernel(), table, source, 1e-10)
    np.testing.assert_array_equal(
        playback(kernel, actual, table, source, 1e-10),
        playback(FloquetPeriodKernel(), independent, table, source, 1e-10),
    )


@pytest.mark.parametrize("changed", ["carrier", "budget", "model", "port"])
def test_history_never_crosses_physical_identity_or_budget(source, changed):
    table = scan([0.003, 0.00301])
    kernel, workspace = scoped(table, source)
    prepare(kernel, table, source)
    target = scan([0.00302])
    budget = 1e-3
    if changed == "carrier":
        target = scan([0.00302], frequency=-0.05)
    elif changed == "port":
        target = scan([0.00302], port=1)
    elif changed == "budget":
        budget = 1e-4
    else:
        model = replace(source.model((0,)), identity="another-model")
        source = ModelSource(source.initial_density, lambda level: model)
    actual = prepare(kernel, target, source, budget)
    assert workspace.attempts == 0
    independent = prepare(FloquetPeriodKernel(), target, source, budget)
    np.testing.assert_array_equal(
        playback(kernel, actual, target, source, budget),
        playback(FloquetPeriodKernel(), independent, target, source, budget),
    )
