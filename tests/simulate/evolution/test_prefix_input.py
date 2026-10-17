"""Identical driven histories share their actual input at the Engine seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import (
    DriveTrack, Engine, EnvelopeSpec, FluxSchedule, ModelSource,
    SecularDrive, SecularModel, TimelineBatch,
)
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel


class ObservedDyson(DysonKernel):
    """Delegate all numerics and record the public input-binding boundary."""

    def __init__(self):
        self.inputs = []

    def bind_input(self, prepared, key_index, table, rows, initial, budget):
        self.inputs.append((table.start_ns[rows].copy(), initial.copy()))
        return super().bind_input(prepared, key_index, table, rows, initial, budget)


def source(initial=None):
    x = np.array([[0, .5], [.5, 0]], complex)
    model = SecularModel(
        (0,), "prefix-model", np.array([0., .1]), np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros((2, 2), complex)),),
        np.zeros((2, 2)), np.zeros((2, 2)),
    )
    return ModelSource(
        np.diag([1., 0.]).astype(complex) if initial is None else initial,
        lambda level: replace(model, level=level, identity=str(level)),
    )


def track(start, duration, frequency, gain):
    return DriveTrack(
        "drive", np.full(2, start), np.full(2, duration), np.full(2, frequency),
        np.zeros(2), np.full(2, gain), (EnvelopeSpec("constant"),),
        np.zeros(2, dtype=np.int64), np.ones(2, dtype=bool),
    )


def batch():
    return TimelineBatch(
        np.arange(2), (track(0., 2., .1, .15), track(3., 3., .4, 1e-5)),
        FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64)),
        np.full(2, 6.),
    )


def engine(kernel):
    return Engine(1, 100000, kernels=tuple(
        kernel if k.name == "dyson" else k for k in REGISTRY
    ))


def reference(src, b):
    """Independent tiny unitary DOP853, stopped at pulse boundaries."""
    model = src.model((0,))
    state = src.initial_density.copy()
    for start, end in [(0., 2.), (2., 3.), (3., 6.)]:
        def rhs(t, flat):
            h = np.diag(model.energies_ghz).astype(complex)
            for tone in b.tracks:
                if tone.start_ns[0] <= (start + end) / 2 < tone.start_ns[0] + tone.duration_ns[0]:
                    h += tone.amplitude[0] * np.cos(2*np.pi*tone.frequency_ghz[0]*t) * model.drives[0].in_phase_ghz
            rho = flat.reshape(2, 2)
            return (-2j*np.pi*(h @ rho - rho @ h)).ravel()
        state = solve_ivp(rhs, (start, end), state.ravel(), method="DOP853", rtol=2e-12, atol=2e-14).y[:, -1].reshape(2, 2)
    return state


def test_shared_driven_input_matches_dop853_and_stays_evolve_local():
    src, b, kernel = source(), batch(), ObservedDyson()
    with closing(engine(kernel)) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
        shared = [rho for starts, rho in kernel.inputs if np.all(starts == 3.)]
        assert len(shared) == 1
        assert abs(shared[0][0, 1]) > .1
        np.testing.assert_allclose(result.densities, np.repeat(reference(src, b)[None], 2, axis=0), atol=1e-3, rtol=0)
        kernel.inputs.clear()
        other = source(np.diag([0., 1.]).astype(complex))
        changed = solver.evolve(other, (b,), cancel=Event())
        other_shared = [rho for starts, rho in kernel.inputs if np.all(starts == 3.)]
        assert len(other_shared) == 1
        assert np.max(abs(other_shared[0] - shared[0])) > .1
        np.testing.assert_allclose(changed.densities, np.repeat(reference(other, b)[None], 2, axis=0), atol=1e-3, rtol=0)


@pytest.mark.parametrize("parameter", ["start_ns", "duration_ns", "frequency_ghz", "amplitude", "phase_rad", "envelope", "level"])
def test_any_changed_prefix_parameter_prevents_input_sharing(parameter):
    src, b, kernel = source(), batch(), ObservedDyson()
    with closing(engine(kernel)) as solver:
        solver.evolve(src, (b,), cancel=Event())
        assert any(np.all(starts == 3.) for starts, _ in kernel.inputs)
        kernel.inputs.clear()
        pre = b.tracks[0]
        if parameter == "envelope":
            pre = replace(pre, envelopes=(EnvelopeSpec("constant"), EnvelopeSpec("sampled", np.array([1., .9], complex), np.array([0., 2.]))), envelope_index=np.array([0, 1]))
        elif parameter == "level":
            b = replace(b, flux=FluxSchedule(np.array([0, 2, 4]), np.tile([0., 2.], 2), np.array([[0], [0], [1], [0]])))
        else:
            values = getattr(pre, parameter).copy()
            values[1] += .01
            pre = replace(pre, **{parameter: values})
        b = replace(b, tracks=(pre, b.tracks[1]))
        solver.evolve(src, (b,), cancel=Event())
        assert not any(np.all(starts == 3.) for starts, _ in kernel.inputs)


@pytest.mark.parametrize("gain, accepted", [(0.0005, True), (0.002, False)])
def test_driven_prefix_population_bound_uses_the_actual_coherent_input(gain, accepted):
    src, b, kernel = source(), batch(), ObservedDyson()
    b = replace(b, tracks=(b.tracks[0], replace(b.tracks[1], amplitude=np.full(2, gain))))
    with closing(engine(kernel)) as solver:
        result = solver.evolve(src, (b,), cancel=Event())
    assert (result.paths.segments_by_kernel.get("dyson", 0) == 2) == accepted
    assert any(np.all(starts == 3.) for starts, _ in kernel.inputs) == accepted


def test_nested_shared_prefixes_use_their_own_known_input():
    src, b, kernel = source(), batch(), ObservedDyson()
    b = replace(b, tracks=(b.tracks[0], replace(b.tracks[1], duration_ns=np.ones(2)), track(4.5, 1.5, .45, 1e-5)))
    with closing(engine(kernel)) as solver:
        solver.evolve(src, (b,), cancel=Event())
    inputs = {float(starts[0]): rho for starts, rho in kernel.inputs}
    assert 3. in inputs and 4.5 in inputs
    assert np.max(abs(inputs[3.] - inputs[4.5])) > .1
