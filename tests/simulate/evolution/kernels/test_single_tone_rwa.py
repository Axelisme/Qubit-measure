"""Constant RWA blocks at the Engine and density-array seams."""

from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import fields, replace
from threading import Event

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel
from zcu_tools.simulate.evolution.kernels.multitone_rwa import rwa_coefficients
from zcu_tools.simulate.evolution.workplan import FactKey, WorkPlan

from tests.simulate.evolution.contract.kernel_suite import KernelCase, reference


def source_for(energies, x):
    size = len(energies)
    rates = np.zeros((size, size))
    rates[0, 1] = 0.0002
    model = SecularModel(
        (0,),
        "constant-block-test",
        np.array(energies),
        np.eye(size, dtype=complex),
        (
            SecularDrive(
                "drive", np.array(x, dtype=complex), np.zeros((size, size), complex)
            ),
        ),
        rates,
        np.zeros((size, size)),
    )
    return ModelSource(
        np.diag([1.0] + [0.0] * (size - 1)).astype(complex),
        lambda level: replace(model, level=level),
    )


def table_for(frequency=0.3, gain=0.5, duration=200.0, theta=0.7):
    return SegmentTable(
        np.array([0]),
        np.array([0.0]),
        np.array([duration]),
        np.array([[0]]),
        np.array([1]),
        np.array([[0]]),
        np.array([[frequency]]),
        np.array([[gain]]),
        np.array([[theta]]),
        np.array([[0]]),
        np.array([[0.0]]),
        (EnvelopeSpec("constant"),),
    )


def batch_for(table):
    track = DriveTrack(
        "drive",
        table.start_ns,
        table.duration_ns,
        table.tone_frequency_ghz[:, 0],
        table.tone_theta_rad[:, 0],
        table.tone_amplitude[:, 0],
        table.envelopes,
        np.array([0]),
        np.array([True]),
    )
    return TimelineBatch(
        np.array([0]),
        (track,),
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        table.duration_ns.copy(),
    )


def family():
    result = next((k for k in REGISTRY if k.name == "single_tone_rwa"), None)
    assert result is not None, "constant RWA must have a visible registered family"
    return result


class CountingWorkPlan(WorkPlan):
    """Observe lazy numerical builds through the public fact publication seam."""

    def __init__(self):
        super().__init__()
        self.builds: Counter[str] = Counter()

    def fact[T](self, key: FactKey[T], build: Callable[[], T]) -> T:
        def observed() -> T:
            value = build()
            name = key.name
            if isinstance(name, tuple):
                self.builds[str(name[0])] += 1
            return value

        return super().fact(key, observed)


def play_with_plan(src, table, plan):
    kernel = family()
    rows = np.array([0])
    context = KernelContext(src, 0.001, plan)
    assert kernel.guard(table, rows, context).tolist() == [True]
    keys = kernel.prepare_keys(table, rows, context)
    prepared = kernel.prepare(keys, table, rows, context)
    states = src.initial_density[None].copy()
    kernel.apply(prepared, rows, table, rows, states, ApplyContext(0.001))
    return states


def test_carriers_share_exact_graph_and_topology_but_keep_their_stark_shifts():
    src = source_for([0.0, 0.3, 0.91], [[0, 0.01, 0], [0.01, 0, 0.002], [0, 0.002, 0]])
    with closing(CountingWorkPlan()) as plan:
        outputs = []
        for frequency in (0.3, 0.301):
            table = table_for(frequency)
            actual = play_with_plan(src, table, plan)
            with closing(WorkPlan()) as isolated:
                expected = play_with_plan(src, table, isolated)
            np.testing.assert_array_equal(actual, expected)
            outputs.append(actual)
        assert plan.builds["joint-rwa-phase-graph"] == 1
        assert plan.builds["rwa-topology"] == 1
        assert plan.builds["multitone-rwa-retained-norm"] == 1
        assert plan.builds["model-quadrature-norms"] == 1
        assert plan.builds["packed-model"] == 1
        assert plan.builds["secular-bath"] == 1
    assert not np.allclose(outputs[0], outputs[1], atol=1e-8, rtol=0)


@pytest.mark.parametrize("changed", ["window", "retained", "model", "level", "port"])
def test_graph_sharing_separates_final_masks_and_full_physics(changed):
    src = source_for([0.0, 7.11], [[0, 0.012], [0.012, 0]])
    model = src.model((0,))
    drive = model.drives[0]
    model = replace(model, drives=(drive, replace(drive, name="other")))
    src = ModelSource(src.initial_density, lambda level: replace(model, level=level))
    table = table_for(7.0, 1.0, 2.0)
    if changed == "window":
        other_table = table_for(7.1, 1.0, 2.0)
    elif changed == "retained":
        other_table = table_for(7.0, 1.0, 30.0)
    elif changed == "level":
        other_table = replace(table, level=np.array([[1]]))
    elif changed == "port":
        other_table = replace(table, tone_port=np.array([[1]]))
    else:
        other_table = table
    other_source = src
    if changed == "model":
        other = replace(
            model, identity="other-physics", rates_per_ns=model.rates_per_ns * 2
        )
        other_source = ModelSource(
            src.initial_density, lambda level: replace(other, level=level)
        )
    with closing(CountingWorkPlan()) as plan:
        play_with_plan(src, table, plan)
        actual = play_with_plan(other_source, other_table, plan)
        with closing(WorkPlan()) as isolated:
            expected = play_with_plan(other_source, other_table, isolated)
        np.testing.assert_array_equal(actual, expected)
        assert plan.builds["joint-rwa-phase-graph"] == 2
        assert plan.builds["rwa-topology"] == 2
        assert plan.builds["multitone-rwa-retained-norm"] == 2


def test_graph_fact_hits_do_not_skip_strength_phase_or_budget_admission():
    src = source_for([0.0, 0.3], [[0, 0.01], [0.01, 0]])
    table = table_for()
    rows = np.array([0])
    kernel = family()
    with closing(WorkPlan()) as plan:
        context = KernelContext(src, 0.001, plan)
        assert kernel.guard(table, rows, context).tolist() == [True]
        for other in (table_for(gain=5.0), table_for(duration=10000.0)):
            assert kernel.guard(other, rows, context).tolist() == [False]
        assert kernel.guard(table, rows, replace(context, budget=1e-5)).tolist() == [
            False
        ]


def test_constant_resonant_probe_uses_block_family():
    src = source_for([0.0, 0.3, 0.91], [[0, 0.01, 0], [0.01, 0, 0.002], [0, 0.002, 0]])
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(src, (batch_for(table_for()),), cancel=Event())
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    # The independent oracle only needs full lab DOP853 evolution, not JIT
    # compilation of Floquet/Magnus or a second production integrator. Its
    # local controls are tighter than this strict case's 1e-7 allowance.
    case = KernelCase(
        DOP853Kernel(), src, table_for(), src.initial_density[None].copy(), budget=1e-7
    )
    strict = reference(case, 0, case.states[0])[None]
    np.testing.assert_allclose(result.densities, strict, atol=0.002, rtol=0)


@pytest.mark.parametrize("complex_input", [False, True])
def test_blocks_preserve_cross_component_coherences_and_complex_residuals(
    complex_input,
):
    src = source_for([0.0, 0.3, 0.91], [[0, 0.01, 0], [0.01, 0, 0.002], [0, 0.002, 0]])
    table = table_for()
    kernel = family()
    context = KernelContext(src, 0.001, WorkPlan())
    rows = np.array([0])
    assert kernel.guard(table, rows, context).tolist() == [True]
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows, context), table, rows, context
    )
    rho = np.array([[0.6, 0.1j, 0], [-0.1j, 0.3, 0], [0, 0, 0.1]], complex)
    if complex_input:
        rho += np.array(
            [[1e-8j, 0, 0.03 + 0.04j], [0, 0, -0.02j], [0.01, 0.02, 0]], complex
        )
    states = rho[None].copy()
    kernel.apply(prepared, rows, table, rows, states, ApplyContext(0.001))
    # Independent dense Liouvillian reference, including all cross-block entries.
    model = src.model((0,))
    coeff = rwa_coefficients(model, (0,), (0.3,))
    q = np.array([1, 0, 0])
    h = np.diag(model.energies_ghz + q * 0.3 + coeff.diagonal[0] * 0.5**2).astype(
        complex
    )
    h += 0.5 * (coeff.positive[0] + coeff.negative[0])
    outgoing = model.rates_per_ns.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2

    def rhs(v):
        r = v.reshape(3, 3)
        d = -2j * np.pi * (h @ r - r @ h) - decay * r
        d[np.diag_indices(3)] += model.rates_per_ns @ np.diag(r)
        return d.ravel()

    generator = np.column_stack([rhs(v) for v in np.eye(9, dtype=complex)])
    start = np.exp(1j * q * 0.7)
    rotated = rho * start.conj()[:, None] * start[None, :]
    expected = (expm(generator * 200) @ rotated.ravel()).reshape(3, 3)
    end = np.exp(1j * q * (0.7 + 2 * np.pi * 0.3 * 200))
    expected *= end[:, None] * end.conj()[None, :]
    np.testing.assert_allclose(states[0], expected, atol=2e-12, rtol=0)


def dense_rwa_reference(src, rho, frequency, theta, duration, charges):
    """Independent full Liouvillian, without component or Hermitian splitting."""
    model = src.model((0,))
    coeff = rwa_coefficients(model, (0,), (frequency,))
    h = np.diag(
        model.energies_ghz + charges * frequency + coeff.diagonal[0] * 0.5**2
    ).astype(complex)
    h += 0.5 * (coeff.positive[0] + coeff.negative[0])
    outgoing = model.rates_per_ns.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2
    size = len(charges)

    def rhs(v):
        r = v.reshape(size, size)
        d = -2j * np.pi * (h @ r - r @ h) - decay * r
        d[np.diag_indices(size)] += model.rates_per_ns @ np.diag(r)
        return d.ravel()

    generator = np.column_stack([rhs(v) for v in np.eye(size**2, dtype=complex)])
    start = np.exp(1j * charges * theta)
    rotated = rho * start.conj()[:, None] * start[None, :]
    expected = (expm(generator * duration) @ rotated.ravel()).reshape(size, size)
    end = np.exp(1j * charges * (theta + 2 * np.pi * frequency * duration))
    return expected * end[:, None] * end.conj()[None, :]


@pytest.mark.parametrize("frequency", [0.3, -0.3])
@pytest.mark.parametrize("duration", [0.0, 17.3, 200.0])
@pytest.mark.parametrize("coupled", [False, True])
def test_complex_playback_matches_dense_reference_across_components(
    frequency, duration, coupled
):
    # Two dimers and two singletons, or an entirely scalar graph. The bath
    # transfers population across components, independently of their topology.
    x = np.zeros((6, 6), dtype=complex)
    if coupled:
        x[0, 1] = x[1, 0] = 0.003
        x[2, 3] = x[3, 2] = 0.004
    src = source_for([0.0, 0.3, 0.9, 1.2, 1.8, 2.4], x)
    table = table_for(frequency=frequency, duration=duration, theta=-0.83)
    kernel = family()
    rows = np.array([0])
    context = KernelContext(src, 0.001, WorkPlan())
    assert kernel.guard(table, rows, context).tolist() == [True]
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows, context), table, rows, context
    )
    charges = np.zeros(6, dtype=np.int64)
    if coupled:
        charges[[0, 2] if frequency > 0 else [1, 3]] = 1
    rng = np.random.default_rng(41)
    dense = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    reverse_only = np.zeros((6, 6), dtype=complex)
    # Below any useful tolerance, but not zero. The nonzero block is only in
    # the reverse direction, so scanning forward RHS alone loses this signal.
    reverse_only[3, 0] = 1e-90 * (1 + 2j)
    reverse_only[5, 4] = 1e-90j
    for rho in (dense, reverse_only, src.initial_density):
        states = rho[None].copy()
        exceeded = kernel.apply(
            prepared, rows, table, rows, states, ApplyContext(0.001)
        )
        assert not np.any(exceeded)
        expected = dense_rwa_reference(src, rho, frequency, -0.83, duration, charges)
        scale = float(np.max(np.abs(rho)))
        np.testing.assert_allclose(states[0], expected, atol=2e-12 * scale, rtol=0)


@pytest.mark.parametrize("kind", ["outside-one-photon", "same-tone-two-photon"])
def test_omitted_resonances_decline_and_retain_original_route(kind):
    if kind == "outside-one-photon":
        src = source_for([0.0, 7.051], [[0, 0.04], [0.04, 0]])
        table = table_for(7.0, 1.0, 129 / (2 * np.sqrt(0.051**2 + 0.04**2)))
    else:
        src = source_for([0.0, 1.4, 2.0], [[0, 0.02, 0], [0.02, 0, 0.02], [0, 0.02, 0]])
        f = 1.0
        for _ in range(8):
            shifts = rwa_coefficients(src.model((0,)), (0,), (f,)).diagonal[0]
            f = float((2 + shifts[2] - shifts[0]) / 2)
        table = table_for(f, 1.0, 250.0)
    kernel = family()
    assert kernel.guard(
        table, np.array([0]), KernelContext(src, 0.001, WorkPlan())
    ).tolist() == [False]
    original = tuple(k for k in REGISTRY if k.name != kernel.name)
    with closing(Engine(1, 1000000, kernels=original)) as engine:
        expected = engine.evolve(src, (batch_for(table),), cancel=Event())
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == expected.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, expected.densities)


@pytest.mark.parametrize("detuning", [0.07, 0.11])
def test_unomittable_single_photon_is_retained_beyond_initial_window(detuning):
    src = source_for([0.0, 7.0 + detuning], [[0, 0.012], [0.012, 0]])
    table = table_for(7.0, 1.0, 30.0)
    kernel = family()
    assert kernel.guard(
        table, np.array([0]), KernelContext(src, 0.001, WorkPlan())
    ).tolist() == [True]
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    with closing(Engine(1, 1000000)) as engine:
        reference = engine.evolve(src, (batch_for(table),), budget=1e-7, cancel=Event())
    np.testing.assert_allclose(
        result.densities, reference.densities, atol=0.002, rtol=0
    )


def test_unomittable_two_photon_path_restores_its_first_order_edges():
    src = source_for([0.0, 7.4, 14.0], [[0, 0.02, 0], [0.02, 0, 0.02], [0, 0.02, 0]])
    table = table_for(7.0, 1.0, 200.0)
    kernel = family()
    assert kernel.guard(
        table, np.array([0]), KernelContext(src, 0.001, WorkPlan())
    ).tolist() == [True]
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    with closing(Engine(1, 1000000)) as engine:
        reference = engine.evolve(src, (batch_for(table),), budget=1e-7, cancel=Event())
    np.testing.assert_allclose(
        result.densities, reference.densities, atol=0.002, rtol=0
    )


def test_prepare_identity_tracks_final_edges_not_duration_itself():
    src = source_for([0.0, 7.11], [[0, 0.012], [0.012, 0]])
    kernel = family()
    context = KernelContext(src, 0.001, WorkPlan())
    rows = np.array([0])
    short = table_for(7.0, 1.0, 2.0)
    restored = table_for(7.0, 1.0, 30.0)
    same_edges = table_for(7.0, 1.0, 40.0)
    for table in (short, restored, same_edges):
        assert kernel.guard(table, rows, context).tolist() == [True]
    short_key = kernel.prepare_keys(short, rows, context)
    restored_key = kernel.prepare_keys(restored, rows, context)
    assert short_key[0] != restored_key[0]
    np.testing.assert_array_equal(
        restored_key, kernel.prepare_keys(same_edges, rows, context)
    )


def test_retained_inventory_is_row_local_and_split_order_independent():
    src = source_for([0.0, 7.11], [[0, 0.012], [0.012, 0]])
    base = table_for(7.0, 1.0, 30.0)
    arrays = {
        field.name: np.repeat(value, 2, axis=0)
        for field in fields(base)
        if isinstance(value := getattr(base, field.name), np.ndarray)
    }
    table = replace(base, **arrays)
    table = replace(table, duration_ns=np.array([2.0, 30.0]))
    kernel = family()
    context = KernelContext(src, 0.001, WorkPlan())
    rows = np.array([1, 0])
    assert kernel.guard(table, rows, context).tolist() == [True, True]
    keys = kernel.prepare_keys(table, rows, context)
    assert keys[0] != keys[1], "one row must not promote another row's edges"
    prepared = kernel.prepare(keys, table, rows, context)
    states = np.repeat(src.initial_density[None], 2, axis=0)
    kernel.apply(prepared, np.array([0, 1]), table, rows, states, ApplyContext(0.001))
    for index, row in enumerate(rows):
        split = np.array([row])
        split_context = KernelContext(src, 0.001, WorkPlan())
        assert kernel.guard(table, split, split_context).tolist() == [True]
        split_keys = kernel.prepare_keys(table, split, split_context)
        np.testing.assert_array_equal(split_keys, keys[index : index + 1])
        split_prepared = kernel.prepare(split_keys, table, split, split_context)
        expected = src.initial_density[None].copy()
        kernel.apply(
            split_prepared, np.array([0]), table, split, expected, ApplyContext(0.001)
        )
        np.testing.assert_array_equal(states[index], expected[0])


def test_restored_graph_over_block_limit_keeps_original_route():
    size = 9
    x = 0.012 * (np.eye(size, k=1) + np.eye(size, k=-1))
    src = source_for(np.arange(size) * 7.07, x)
    table = table_for(7.0, 1.0, 10.0)
    kernel = family()
    assert kernel.guard(
        table, np.array([0]), KernelContext(src, 0.001, WorkPlan())
    ).tolist() == [False]
    original = tuple(k for k in REGISTRY if k.name != kernel.name)
    with closing(Engine(1, 1000000, kernels=original)) as engine:
        expected = engine.evolve(src, (batch_for(table),), cancel=Event())
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == expected.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, expected.densities)
