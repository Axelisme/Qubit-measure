"""Identical driven histories share their actual input at the Engine seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.integrate import solve_ivp
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
from zcu_tools.simulate.evolution.kernel import ApplyContext, KernelContext
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel


class ObservedDyson(DysonKernel):
    """Delegate all numerics and record the public input-binding boundary."""

    def __init__(self):
        self.inputs = []

    def bind_input(
        self, prepared, key_index, table, rows, source_initial, context: KernelContext
    ):
        self.inputs.append((table.start_ns[rows].copy(), source_initial.copy()))
        return super().bind_input(
            prepared, key_index, table, rows, source_initial, context
        )


def source(initial=None):
    x = np.array([[0, 0.5], [0.5, 0]], complex)
    model = SecularModel(
        (0,),
        "prefix-model",
        np.array([0.0, 0.1]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros((2, 2), complex)),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    return ModelSource(
        np.diag([1.0, 0.0]).astype(complex) if initial is None else initial,
        lambda level: replace(model, level=level, identity=str(level)),
    )


def track(start, duration, frequency, gain):
    return DriveTrack(
        "drive",
        np.full(2, start),
        np.full(2, duration),
        np.full(2, frequency),
        np.zeros(2),
        np.full(2, gain),
        (EnvelopeSpec("constant"),),
        np.zeros(2, dtype=np.int64),
        np.ones(2, dtype=bool),
    )


def batch():
    return TimelineBatch(
        np.arange(2),
        (track(0.0, 2.0, 0.1, 0.15), track(3.0, 3.0, 0.4, 1e-5)),
        FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64)),
        np.full(2, 6.0),
    )


def engine(kernel):
    return Engine(
        1, 100000, kernels=tuple(kernel if k.name == "dyson" else k for k in REGISTRY)
    )


def reference(src, b, end_ns=None):
    """Independent tiny unitary DOP853, stopped at pulse boundaries."""
    model = src.model((0,))
    state = src.initial_density.copy()
    stop = b.end_ns[0] if end_ns is None else end_ns
    boundaries = sorted(
        {
            0.0,
            float(stop),
            *(float(t.start_ns[0]) for t in b.tracks if t.start_ns[0] < stop),
            *(
                float(t.start_ns[0] + t.duration_ns[0])
                for t in b.tracks
                if t.start_ns[0] + t.duration_ns[0] < stop
            ),
        }
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):

        def rhs(t, flat, midpoint=(start + end) / 2):
            h = np.diag(model.energies_ghz).astype(complex)
            for tone in b.tracks:
                if (
                    tone.start_ns[0]
                    <= midpoint
                    < tone.start_ns[0] + tone.duration_ns[0]
                ):
                    h += (
                        tone.amplitude[0]
                        * np.cos(2 * np.pi * tone.frequency_ghz[0] * t)
                        * model.drives[0].in_phase_ghz
                    )
            rho = flat.reshape(2, 2)
            return (-2j * np.pi * (h @ rho - rho @ h)).ravel()

        state = (
            solve_ivp(
                rhs,
                (start, end),
                state.ravel(),
                method="DOP853",
                rtol=2e-12,
                atol=2e-14,
            )
            .y[:, -1]
            .reshape(2, 2)
        )
    return state


@pytest.mark.parametrize("history", ["distinct", "source", "no_drive", "shared"])
def test_plateau_routing_preserves_reusable_dyson_inputs(history):
    src, b = source(), batch()
    if history == "distinct":
        pre = replace(b.tracks[0], phase_rad=np.array([0.0, 0.1]))
        b = replace(b, tracks=(pre, b.tracks[1]))
        expected = {"floquet_period": 4, "t4": 2}
    elif history == "source":
        probe = replace(b.tracks[1], start_ns=np.zeros(2))
        b = replace(b, tracks=(probe,), end_ns=np.full(2, 3.0))
        expected = {"dyson": 2}
    elif history == "no_drive":
        b = replace(b, tracks=(b.tracks[1],))
        expected = {"dyson": 2, "t4": 2}
    else:
        expected = {"floquet_period": 2, "dyson": 2, "t4": 2}
    # Isolate period/Dyson input reuse from the constant single-tone RWA route.
    kernels = tuple(k for k in REGISTRY if k.name != "single_tone_rwa")
    with closing(Engine(1, 100000, kernels=kernels)) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
    assert result.paths.segments_by_kernel == expected


@pytest.mark.parametrize("outside_period_domain", ["carrier_floor", "duration"])
def test_distinct_plateau_retains_dyson_outside_period_routing_domain(
    outside_period_domain,
):
    src, b = source(), batch()
    pre = replace(b.tracks[0], phase_rad=np.array([0.0, 0.1]))
    if outside_period_domain == "carrier_floor":
        probe = replace(b.tracks[1], frequency_ghz=np.full(2, 0.025))
    else:
        probe = replace(
            b.tracks[1], duration_ns=np.full(2, 10001.0), amplitude=np.full(2, 1e-10)
        )
        b = replace(b, end_ns=np.full(2, 10004.0))
    b = replace(b, tracks=(pre, probe))
    with closing(Engine(1, 100000)) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
    assert result.paths.segments_by_kernel == {"floquet_period": 2, "dyson": 2, "t4": 2}


def test_shared_driven_input_matches_dop853_and_stays_evolve_local():
    src, b, kernel = source(), batch(), ObservedDyson()
    with closing(engine(kernel)) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
        shared = [rho for starts, rho in kernel.inputs if np.all(starts == 3.0)]
        assert len(shared) == 1
        assert abs(shared[0][0, 1]) > 0.1
        np.testing.assert_allclose(
            result.densities,
            np.repeat(reference(src, b)[None], 2, axis=0),
            atol=1e-3,
            rtol=0,
        )
        kernel.inputs.clear()
        other = source(np.diag([0.0, 1.0]).astype(complex))
        changed = solver.evolve(other, (b,), cancel=Event())
        other_shared = [rho for starts, rho in kernel.inputs if np.all(starts == 3.0)]
        assert len(other_shared) == 1
        assert np.max(abs(other_shared[0] - shared[0])) > 0.1
        np.testing.assert_allclose(
            changed.densities,
            np.repeat(reference(other, b)[None], 2, axis=0),
            atol=1e-3,
            rtol=0,
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "start_ns",
        "duration_ns",
        "frequency_ghz",
        "amplitude",
        "phase_rad",
        "envelope",
        "level",
    ],
)
def test_any_changed_prefix_parameter_prevents_input_sharing(parameter):
    src, b, kernel = source(), batch(), ObservedDyson()
    with closing(engine(kernel)) as solver:
        solver.evolve(src, (b,), cancel=Event())
        assert any(np.all(starts == 3.0) for starts, _ in kernel.inputs)
        kernel.inputs.clear()
        pre = b.tracks[0]
        if parameter == "envelope":
            pre = replace(
                pre,
                envelopes=(
                    EnvelopeSpec("constant"),
                    EnvelopeSpec(
                        "sampled", np.array([1.0, 0.9], complex), np.array([0.0, 2.0])
                    ),
                ),
                envelope_index=np.array([0, 1]),
            )
        elif parameter == "level":
            b = replace(
                b,
                flux=FluxSchedule(
                    np.array([0, 2, 4]),
                    np.tile([0.0, 2.0], 2),
                    np.array([[0], [0], [1], [0]]),
                ),
            )
        else:
            values = getattr(pre, parameter).copy()
            values[1] += 0.01
            pre = replace(pre, **{parameter: values})
        b = replace(b, tracks=(pre, b.tracks[1]))
        solver.evolve(src, (b,), cancel=Event())
        assert not any(np.all(starts == 3.0) for starts, _ in kernel.inputs)


@pytest.mark.parametrize("gain, accepted", [(0.0005, True), (0.01, False)])
def test_driven_prefix_population_bound_uses_the_actual_coherent_input(gain, accepted):
    src, b, kernel = source(), batch(), ObservedDyson()
    b = replace(
        b, tracks=(b.tracks[0], replace(b.tracks[1], amplitude=np.full(2, gain)))
    )
    with closing(engine(kernel)) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
    assert (result.paths.segments_by_kernel.get("dyson", 0) == 2) == accepted
    assert any(np.all(starts == 3.0) for starts, _ in kernel.inputs) == accepted


def test_nested_shared_prefixes_use_their_own_known_input():
    from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    applications = []

    def record(table, rows, states):
        applications.extend(zip(table.start_ns[rows], states.copy(), strict=True))

    class NestedDyson(ObservedDyson):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            record(table, rows, states)
            return super().apply(prepared, key_index, table, rows, states, context)

    class NestedPeriod(FloquetPeriodKernel):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            record(table, rows, states)
            return super().apply(prepared, key_index, table, rows, states, context)

    class NestedIdle(T4):
        def apply(
            self, prepared, key_index, table, rows, states, context: ApplyContext
        ):
            record(table, rows, states)
            return super().apply(prepared, key_index, table, rows, states, context)

    src, b, kernel = source(), batch(), NestedDyson()
    b = replace(
        b,
        tracks=(
            b.tracks[0],
            replace(b.tracks[1], duration_ns=np.ones(2)),
            track(4.5, 1.5, 0.45, 1e-5),
        ),
    )
    replacements = {k.name: k for k in (kernel, NestedPeriod(), NestedIdle())}
    with closing(
        Engine(1, 100000, kernels=tuple(replacements.get(k.name, k) for k in REGISTRY))
    ) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
    inputs = {float(start): rho for start, rho in applications}
    bound = {float(starts[0]): rho for starts, rho in kernel.inputs}
    for start in (3.0, 4.5):
        expected = reference(src, b, start)
        np.testing.assert_allclose(inputs[start], expected, atol=1e-3, rtol=0)
        if start in bound:
            np.testing.assert_allclose(bound[start], expected, atol=1e-3, rtol=0)
    assert 4.5 in bound
    # Each nested history segment runs once in a representative solve. Only
    # the final probe runs once per adopting timeline.
    for start in (0.0, 2.0, 3.0, 4.0):
        assert sum(time == start for time, _ in applications) == 1
    assert sum(time == 4.5 for time, _ in applications) == 2
    np.testing.assert_allclose(
        result.densities,
        np.repeat(reference(src, b)[None], 2, axis=0),
        atol=1e-3,
        rtol=0,
    )
    assert result.paths.segments_by_kernel == {"floquet_period": 2, "t4": 4, "dyson": 4}
    assert result.paths.points_by_class == {"floquet_period": 2, "t4": 2, "dyson": 2}


def test_diagonal_source_shared_output_keeps_its_bitwise_baseline():
    """Mode pruning must not change the existing diagonal-input table output."""
    b = replace(batch(), tracks=(track(0.0, 3.0, 0.4, 0.0005),), end_ns=np.full(2, 3.0))
    with closing(engine(ObservedDyson())) as solver:
        result = solver.evolve(source(), (b,), cancel=Event())
    expected = np.array(
        [
            [1.0 + 0j, 1.030056647916492e-4 + 0.0004755282581475768j],
            [1.030056647916492e-4 - 0.0004755282581475768j, 0.0 + 0j],
        ]
    )
    np.testing.assert_array_equal(
        result.densities, np.repeat(expected[None], 2, axis=0)
    )
