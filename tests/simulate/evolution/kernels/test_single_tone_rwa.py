"""Constant RWA blocks at the Engine and density-array seams."""

from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import fields, replace
from threading import Event

import numpy as np
import pytest
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
from zcu_tools.simulate.evolution.kernels._rwa_numeric import rwa_coefficients
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel
from zcu_tools.simulate.evolution.workplan import FactKey, WorkPlan

from tests.simulate.evolution.contract.kernel_suite import KernelCase, reference
from tests.simulate.evolution.contract.reference import (
    converged_reference,
    density_error,
)


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


def test_off_window_tone_preserves_full_density_at_readout():
    # The omitted off-resonant edge contributes first-order coherence even
    # when its population-transfer proxy passes the existing guard.
    src = source_for([0.0, 0.71], [[0.002, 0.006], [0.006, -0.002]])
    batch = batch_for(table_for(0.6, 1.0, 24.0, 0.73))
    strict = converged_reference(src, batch)
    with closing(Engine(1, families=("single_tone_rwa",))) as engine:
        result = engine.evolve(src, (batch,))
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    error = density_error(result.densities, strict.density)
    assert error <= 2e-3, f"off-window full density error {error:.9g} > 2e-3"


def test_constant_resonant_probe_uses_block_family():
    src = source_for([0.0, 0.3, 0.91], [[0, 0.01, 0], [0.01, 0, 0.002], [0, 0.002, 0]])
    with closing(Engine(1)) as engine:
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


@pytest.mark.parametrize("frequency", [0.3, -0.3])
@pytest.mark.parametrize("duration", [0.0, 17.3, 200.0])
@pytest.mark.parametrize("coupled", [False, True])
def test_complex_coherences_across_components_match_full_hamiltonian(
    frequency, duration, coupled
):
    # Two dimers and two singletons, or an entirely scalar graph. Compare the
    # complete physical density, not the old bare-RWA approximation's output.
    x = np.zeros((6, 6), dtype=complex)
    if coupled:
        x[0, 1] = x[1, 0] = 0.003
        x[2, 3] = x[3, 2] = 0.004
    src = source_for([0.0, 0.3, 0.9, 1.2, 1.8, 2.4], x)
    rng = np.random.default_rng(41)
    vector = rng.normal(size=6) + 1j * rng.normal(size=6)
    vector /= np.linalg.norm(vector)
    src = ModelSource(np.outer(vector, vector.conj()), src.model)
    batch = batch_for(table_for(frequency=frequency, duration=duration, theta=-0.83))
    strict = converged_reference(src, batch)
    with closing(Engine(1, families=("single_tone_rwa",))) as engine:
        result = engine.evolve(src, (batch,))
    if duration == 0:
        # Engine does not schedule an empty span; it returns the initial density.
        assert not result.paths.segments_by_kernel
        np.testing.assert_array_equal(result.densities[0], src.initial_density)
    else:
        assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    assert density_error(result.densities, strict.density) <= 2e-3


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
    with closing(Engine(1, families=("dyson", "floquet_period", "t6"))) as engine:
        expected = engine.evolve(src, (batch_for(table),), cancel=Event())
    with closing(Engine(1)) as engine:
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
    with closing(Engine(1)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    with closing(Engine(1)) as engine:
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
    with closing(Engine(1)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    with closing(Engine(1)) as engine:
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
    with closing(Engine(1, families=("dyson", "floquet_period", "t6"))) as engine:
        expected = engine.evolve(src, (batch_for(table),), cancel=Event())
    with closing(Engine(1)) as engine:
        result = engine.evolve(src, (batch_for(table),), cancel=Event())
    assert result.paths.segments_by_kernel == expected.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, expected.densities)
