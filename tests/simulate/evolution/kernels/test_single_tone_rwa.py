"""Constant RWA blocks at the Engine and density-array seams."""

from contextlib import closing
from dataclasses import replace
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
from zcu_tools.simulate.evolution.kernels.multitone_rwa import rwa_coefficients
from zcu_tools.simulate.evolution.workplan import WorkPlan


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


def test_constant_resonant_probe_uses_block_family():
    src = source_for([0.0, 0.3, 0.91], [[0, 0.01, 0], [0.01, 0, 0.002], [0, 0.002, 0]])
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(src, (batch_for(table_for()),), cancel=Event())
    assert result.paths.segments_by_kernel == {"single_tone_rwa": 1}
    with closing(Engine(1, 1000000)) as engine:
        strict = engine.evolve(
            src, (batch_for(table_for()),), budget=1e-7, cancel=Event()
        )
    np.testing.assert_allclose(result.densities, strict.densities, atol=0.002, rtol=0)


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


def test_dual_constant_uses_the_same_omitted_one_photon_admission():
    src = source_for([0.0, 7.051], [[0, 0.04], [0.04, 0]])
    model = src.model((0,))
    model = replace(
        model,
        drives=(
            *model.drives,
            SecularDrive(
                "spectator", np.zeros((2, 2), complex), np.zeros((2, 2), complex)
            ),
        ),
    )
    src = ModelSource(src.initial_density, lambda level: replace(model, level=level))
    table = table_for(7.0, 1.0, 1000.0)
    table = replace(
        table,
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([[7.0, 8.5]]),
        tone_amplitude=np.array([[1.0, 1.0]]),
        tone_theta_rad=np.array([[0.7, 0.0]]),
        tone_envelope=np.array([[0, 0]]),
        tone_envelope_offset_ns=np.array([[0.0, 0.0]]),
    )
    kernel = next(k for k in REGISTRY if k.name == "multitone_rwa")
    assert kernel.guard(
        table, np.array([0]), KernelContext(src, 0.001, WorkPlan())
    ).tolist() == [False]
