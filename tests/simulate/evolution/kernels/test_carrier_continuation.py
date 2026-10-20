"""Constant CF4 carrier prediction through complete planning and kernel playback."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution._dispatch import create_work_plan
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import synthetic_model


def scan(frequencies):
    count = len(frequencies)
    return SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=float),
        duration_ns=np.linspace(20.7, 51.3, count),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.zeros((count, 1), dtype=np.int64),
        tone_frequency_ghz=np.asarray(frequencies)[:, None],
        tone_amplitude=np.full((count, 1), 0.003),
        tone_theta_rad=np.linspace(-3.1, 5.9, count)[:, None],
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )


@pytest.fixture
def source():
    model = synthetic_model()
    model = replace(
        model,
        energies_ghz=np.array([0.0, 0.9, 1.8]),
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
    )
    source = ModelSource(np.ones((3, 3), dtype=np.complex128) / 3, lambda level: model)
    # Compile the numerical boundary before observing NumPy; replacing a NumPy
    # function while Numba first types its callers would change compilation.
    playback(scan([0.05]), KernelContext(source, 1e-3, WorkPlan()))
    return source


def playback(table, context, *, order=None):
    kernel = FloquetPeriodKernel()
    states = np.repeat(
        context.source.initial_density[None], len(table.timeline), axis=0
    )
    if order is None:
        order = [np.arange(len(states))]
    for rows in order:
        prepared = kernel.prepare(
            kernel.prepare_keys(table, rows, context), table, rows, context
        )
        values = states[rows].copy()
        kernel.apply(
            prepared,
            np.arange(len(rows)),
            table,
            rows,
            values,
            ApplyContext(context.budget, work_plan=context.work_plan),
        )
        states[rows] = values
    return states


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_narrow_carriers_use_four_anchors_and_independent_coarse_checks(
    source, monkeypatch, sign
):
    table = scan(sign * np.linspace(0.05, 0.0504, 16))
    defects = []
    norm = np.linalg.norm

    def observe(matrix, *args, **kwargs):
        result = norm(matrix, *args, **kwargs)
        if matrix.shape == (3, 3) and not args and not kwargs:
            defects.append(float(result))
        return result

    context = KernelContext(source, 1e-3, create_work_plan(table, source))
    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "norm", observe)
        actual = playback(table, context)
    # Four real anchors build independently. Every other key checks all nine
    # original coarse probes; prediction is never an unchecked cache hit.
    assert len(defects) == (16 - 4) * 9
    assert max(defects) <= 1e-4
    independent = playback(table, KernelContext(source, 1e-3, WorkPlan()))
    # Playback stays within this budget's sampled CF4 guard tolerance:
    # min(1e-4, budget * 0.1). This is not a global density error certificate.
    np.testing.assert_allclose(actual, independent, atol=1e-4, rtol=0)
    assert not np.array_equal(actual, independent)
    anchors = np.linspace(0, 15, 4).round().astype(int)
    np.testing.assert_array_equal(actual[anchors], independent[anchors])
    context.work_plan.close()


def test_rejected_carrier_prediction_refines_independently_bitwise(source, monkeypatch):
    table = scan(np.linspace(0.05, 0.0504, 16))
    defects = []
    norm = np.linalg.norm

    def observe(matrix, *args, **kwargs):
        result = norm(matrix, *args, **kwargs)
        if matrix.shape == (3, 3) and not args and not kwargs:
            defects.append(float(result))
        return result

    context = KernelContext(source, 1e-10, create_work_plan(table, source))
    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "norm", observe)
        actual = playback(table, context)
    assert len(defects) == (16 - 4) * 9
    assert max(defects) > 1e-11
    independent = playback(table, KernelContext(source, 1e-10, WorkPlan()))
    np.testing.assert_array_equal(actual, independent)
    context.work_plan.close()


def test_carrier_predictions_do_not_depend_on_split_or_reordered_consumers(source):
    table = scan(np.linspace(0.05, 0.0504, 16))
    reference = playback(
        table, KernelContext(source, 1e-3, create_work_plan(table, source))
    )
    reordered = playback(
        table,
        KernelContext(source, 1e-3, create_work_plan(table, source)),
        order=[np.array([i]) for i in range(15, -1, -1)],
    )
    np.testing.assert_array_equal(reference, reordered)


@pytest.mark.parametrize(
    "frequencies",
    [
        np.linspace(0.05, 0.0504, 15),
        np.linspace(0.05, 0.051, 16),
        np.r_[-0.05, np.linspace(0.05, 0.0504, 15)],
        np.linspace(1.0, 1.004, 16),
        np.linspace(0.03, 0.0301, 16),  # Sambe, not CF4, at this spectral width.
    ],
)
def test_ineligible_carrier_groups_keep_independent_playback_bitwise(
    source, frequencies
):
    table = scan(frequencies)
    context = KernelContext(source, 1e-3, create_work_plan(table, source))
    actual = playback(table, context)
    independent = playback(table, KernelContext(source, 1e-3, WorkPlan()))
    np.testing.assert_array_equal(actual, independent)
    context.work_plan.close()


def test_carrier_resources_release_after_last_actual_consumer(source):
    from zcu_tools.simulate.evolution._gain_plan import admit_shared_builds

    table = scan(np.linspace(0.05, 0.0504, 16))
    context = KernelContext(source, 1e-3, create_work_plan(table, source))
    groups = getattr(context.work_plan, "cf4_carrier_groups", ())
    assert len(groups) == 1, "complete carrier inventory must publish fixed supports"
    from zcu_tools.simulate.evolution._gain_plan import carrier_anchor_resource

    group = groups[0]
    handles = [carrier_anchor_resource(context, group, f) for f in group.anchors_ghz]
    # One caller admission seals all supports. A prefix subset must not seal them.
    admit_shared_builds(context, table, np.arange(16), np.empty(0, dtype=np.int64))
    assert all(h.sealed and not h.released for h in handles)
    playback(table, context, order=[np.array([i]) for i in range(15, -1, -1)])
    assert all(h.released and not h.remaining_consumers for h in handles)
    assert all(h.actual_consumers == h.planned_consumers for h in handles)
    context.work_plan.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_engine_carrier_results_are_stable_across_warm_cache_and_worker_width(
    source, workers
):
    from contextlib import closing
    from threading import Event

    from zcu_tools.simulate.evolution import Engine
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    from .._engine import batch, track

    count = 16
    pulse = replace(
        track(count, EnvelopeSpec("constant"), duration=31.7),
        frequency_ghz=np.linspace(0.05, 0.0504, count),
        amplitude=np.full(count, 0.003),
        phase_rad=np.linspace(-3.1, 5.9, count),
    )
    timeline = replace(batch(count, duration=31.7), tracks=(pulse,))
    changed_pulse = replace(pulse, phase_rad=pulse.phase_rad + 0.7)
    changed = replace(timeline, tracks=(changed_pulse,))
    # Isolate floquet_period continuation from the constant single-tone RWA route.
    kernels = tuple(k for k in REGISTRY if k.name != "single_tone_rwa")
    with closing(Engine(workers, 10_000_000, kernels=kernels)) as engine:
        first = engine.evolve(source, (timeline,), cancel=Event())
        # Changed phase misses exact-result memo but hits identical period keys.
        warm = engine.evolve(source, (changed,), cancel=Event())
    with closing(Engine(1, 0, kernels=kernels)) as engine:
        cold = engine.evolve(source, (changed,), cancel=Event())
    assert first.paths.segments_by_kernel == {"floquet_period": count}
    assert warm.paths.segments_by_kernel == cold.paths.segments_by_kernel
    np.testing.assert_array_equal(warm.densities, cold.densities)


def spectral_source(source, geometry):
    model = source.model((0,))
    if geometry == "degenerate":
        drive = replace(
            model.drives[0],
            in_phase_ghz=np.diag([0.0, 0.0, 0.02]).astype(complex),
            quadrature_ghz=np.zeros((3, 3), dtype=complex),
        )
        model = replace(model, energies_ghz=np.array([0.0, 0.0, 1.8]), drives=(drive,))
    elif geometry == "grid-boundary":
        drive = model.drives[0]
        # The narrow carrier inventory straddles the 56/64 coarse-grid boundary.
        span = 56 * 0.0502 / 1.5 - 2 * 0.003 * (
            np.linalg.norm(drive.in_phase_ghz) + np.linalg.norm(drive.quadrature_ghz)
        )
        model = replace(model, energies_ghz=np.array([0.0, 0.9, span]))
    return ModelSource(source.initial_density, lambda level: model)


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("geometry", ["ordinary", "degenerate", "grid-boundary"])
def test_carrier_coarse_guard_preserves_density_bitwise(source, sign, geometry):
    from hashlib import sha256

    source = spectral_source(source, geometry)
    table = scan(sign * np.linspace(0.05, 0.0504, 16))
    context = KernelContext(source, 1e-3, create_work_plan(table, source))
    actual = playback(table, context)
    # Characterize the original Padé guard's accepted predictions and fallbacks.
    # Coherent inputs and phase-translated fractional tails make this sensitive
    # to a changed interpolation admission, not just to population conservation.
    expected = {
        (
            "ordinary",
            1.0,
        ): "b75ac730091debd31b638ccba89278218f634b30995d0cd7cefc04b34d542227",
        (
            "ordinary",
            -1.0,
        ): "d7abb23390d49e8b8619f804d5396d1e2e628019d47924f860b37ba13a325725",
        (
            "degenerate",
            1.0,
        ): "5956ff7500cfa1a274da68a865d4bd5fc4c8667d565f90e46c74a57bafc4ba9a",
        (
            "degenerate",
            -1.0,
        ): "a9f639a2678e59567e6edf1d61815f0674533580120bbae7fef99c181cad21c0",
        (
            "grid-boundary",
            1.0,
        ): "ef62fe92cddf4b4002d1a3930e4cf49083c036c405c7174323ec1650c55088e6",
        (
            "grid-boundary",
            -1.0,
        ): "90057d7cfaf52298d4c8e2a2e696f160a2922dad9e99a426a902ddbbdec2e501",
    }
    assert sha256(actual.tobytes()).hexdigest() == expected[geometry, sign]
    np.testing.assert_allclose(np.trace(actual, axis1=1, axis2=2), 1, atol=1e-12)
    context.work_plan.close()


def test_carrier_coarse_nodes_are_built_once_for_concurrent_consumers(
    source, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    table = scan(np.linspace(0.05, 0.0504, 16))
    context = KernelContext(source, 1e-3, create_work_plan(table, source))
    batches = []
    eigh = np.linalg.eigh

    def observe(nodes, *args, **kwargs):
        if nodes.ndim == 4:
            batches.append(nodes.shape)
        return eigh(nodes, *args, **kwargs)

    def consume(rows):
        return playback(table, context, order=[rows])[rows]

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "eigh", observe)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(consume, [np.arange(8), np.arange(8, 16)]))
    assert len(batches) == 1, "one single-flight node eigensystem per phase grid"
    assert batches[0][1:] == (2, 3, 3)
    reference = playback(
        table, KernelContext(source, 1e-3, create_work_plan(table, source))
    )
    np.testing.assert_array_equal(np.concatenate(results), reference)
    context.work_plan.close()
