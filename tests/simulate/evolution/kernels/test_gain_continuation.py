"""Calibrated gain reuse through scoped kernel preparation and density playback."""

from concurrent.futures import ThreadPoolExecutor
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
    return FloquetPeriodKernel().evolve_scope(table, source)


def prepare(kernel, table, source, budget=1e-3, *, rows=None):
    if rows is None:
        rows = np.arange(len(table.timeline), dtype=np.int64)
    return kernel.prepare(kernel.prepare_keys(table, rows), table, rows, source, budget)


def playback(kernel, prepared, table, source, budget=1e-3, *, rows=None):
    if rows is None:
        rows = np.arange(len(table.timeline), dtype=np.int64)
    states = np.repeat(source.initial_density[None], len(rows), axis=0)
    kernel.apply(prepared, np.arange(len(rows)), table, rows, states, budget)
    return states


def test_gain_predictor_is_checked_and_matches_independent_playback(source):
    table = scan(np.arange(6) * 1e-5 + 0.003)
    kernel, workspace = scoped(table, source)
    cache = PrepareCache(0)
    with cache.track_workspace(workspace):
        assert cache.live_bytes == 0
        first_rows, last_rows = np.arange(3), np.arange(3, 6)
        first = prepare(kernel, table, source, rows=first_rows)
        assert cache.live_bytes == workspace.nbytes > 0
        last = prepare(kernel, table, source, rows=last_rows)
        assert cache.live_bytes == workspace.nbytes == 0
    assert cache.live_bytes == 0
    assert workspace.attempts >= 1
    assert workspace.hits >= 1
    independent = prepare(FloquetPeriodKernel(), table, source)
    # Include translated phase, a cross-period tail, and coherent input.
    actual = np.concatenate(
        [
            playback(kernel, first, table, source, rows=first_rows),
            playback(kernel, last, table, source, rows=last_rows),
        ]
    )
    np.testing.assert_allclose(
        actual,
        playback(FloquetPeriodKernel(), independent, table, source),
        atol=1e-4,
        rtol=0,
    )
    # A new evolve does not inherit the previous call's retained tables.
    _, fresh = scoped(table, source)
    assert fresh.nbytes == fresh.attempts == fresh.hits == 0


@pytest.mark.parametrize("order", ["thread-handoff", "reverse"])
def test_scoped_preparation_is_bitwise_invariant_to_worker_and_key_order(source, order):
    table = scan(np.arange(8) * 1e-5 + 0.003)
    reference_kernel, _ = scoped(table, source)
    reference = playback(
        reference_kernel, prepare(reference_kernel, table, source), table, source
    )
    kernel, _ = scoped(table, source)
    actual = np.empty_like(reference)

    def prepare_rows(rows):
        prepared = kernel.prepare(
            kernel.prepare_keys(table, rows), table, rows, source, 1e-3
        )
        states = np.repeat(source.initial_density[None], len(rows), axis=0)
        kernel.apply(prepared, np.arange(len(rows)), table, rows, states, 1e-3)
        actual[rows] = states

    if order == "thread-handoff":
        prepare_rows(np.arange(4))
        # A separate worker necessarily has a different identity from the
        # calling thread. Await it explicitly, without relying on timing.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(prepare_rows, np.arange(4, 8)).result(timeout=25)
    else:
        for row in range(7, -1, -1):
            prepare_rows(np.array([row]))
    np.testing.assert_array_equal(actual, reference)


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
