"""Continuous flux and quantized baths at the C5 prepare/apply seam."""

from dataclasses import replace
from threading import Event
from typing import Literal

import numpy as np
import pytest
import zcu_tools.simulate.evolution as evolution
from zcu_tools.simulate.evolution.kernel import SegmentTable

from tests.simulate.evolution.contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)


def flux_case(size: int = 3, sampled: bool = True) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.flux_dop853 import FluxDOP853Kernel

    model = synthetic_model(size)
    x = model.drives[0].in_phase_ghz
    y = model.drives[0].quadrature_ghz
    kinds: tuple[Literal["cos_2pi", "sin_2pi", "cos_pi", "sin_pi"], ...] = (
        "cos_2pi",
        "sin_2pi",
        "cos_pi",
        "sin_pi",
    )
    expansion = evolution.FluxExpansion(
        np.diag(model.energies_ghz).astype(complex),
        tuple(
            evolution.FluxTerm(loop, kind, scale * matrix)
            for kind, (loop, scale, matrix) in zip(
                kinds,
                (
                    (0, 0.03, x),
                    (0, 0.09, y),
                    (1, 0.02, y),
                    (1, 0.08, x),
                ),
                strict=True,
            )
        ),
    )

    def build(level: tuple[int, ...]):
        flux = np.array([0.19, -0.12]) + 0.001 * np.array(level)
        h = expansion.static.copy()
        for term in expansion.terms:
            angle = np.pi * flux[term.loop] * (2 if "2pi" in term.kind else 1)
            h += (
                np.cos(angle) if term.kind.startswith("cos") else np.sin(angle)
            ) * term.matrix
        energies, vectors = np.linalg.eigh(h)
        drive = evolution.SecularDrive(
            "drive", vectors.conj().T @ x @ vectors, vectors.conj().T @ y @ vectors
        )
        return replace(
            model,
            level=level,
            identity=f"flux-{size}-{level}",
            energies_ghz=energies,
            eigenvectors=vectors,
            drives=(drive,),
            rates_per_ns=model.rates_per_ns * (1 + abs(level[0]) / 20),
        )

    vector = np.exp(0.4j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    source = evolution.ModelSource(state, build, expansion)
    time = np.array([0.0, 0.35, 0.8, 1.4, 2.4])
    waveform = evolution.FluxWaveform(
        np.array([0, 5, 10, 15]),
        np.tile(time, 3),
        np.concatenate(
            [
                np.column_stack(
                    (
                        0.19 + k + 0.018 * np.sin(time * 2),
                        -0.12 + 0.012 * np.cos(time * 3),
                    )
                )
                for k in (0, 0.01, -0.02)
            ]
        ),
    )
    assert hasattr(SegmentTable, "flux_waveform"), (
        "C5 must carry the continuous waveform"
    )
    tones = np.array([1, 2, 3])
    playing = np.arange(3)[None, :] < tones[:, None]
    env = (
        evolution.EnvelopeSpec(
            "sampled",
            np.array([0.2 + 0.1j, 1 - 0.3j, 0.4 + 0.2j]),
            np.array([0.0, 0.6, 2.5]),
        )
        if sampled
        else evolution.EnvelopeSpec("constant")
    )
    table = SegmentTable(
        timeline=np.array([2, 0, 1]),
        start_ns=np.array([0.2, 1.3, 0.0]),
        duration_ns=np.array([1.0, 0.9, 0.0]),
        level=np.array([[0, 0], [8, -2], [-4, 1]]),
        tones=tones,
        tone_port=np.where(playing, 0, -1),
        tone_frequency_ghz=np.broadcast_to([0.17, -0.21, 0.43], (3, 3)),
        tone_amplitude=np.broadcast_to([0.12, -0.07, 0.04], (3, 3)),
        tone_theta_rad=np.broadcast_to([0.3, -1.0, 0.9], (3, 3)),
        tone_envelope=np.where(playing, 0, -1),
        tone_envelope_offset_ns=np.full((3, 3), 0.1),
        envelopes=(env,),
        flux_waveform=waveform,
    )
    return KernelCase(
        FluxDOP853Kernel(), source, table, np.repeat(state[None], 3, axis=0)
    )


@pytest.fixture(params=[(3, False), (3, True), (4, True)])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    return flux_case(*request.param)


class TestFluxDOP853(KernelContract):
    pass


def prepare(case: KernelCase):
    rows = np.arange(len(case.states))
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(case.table, rows),
        return_index=True,
        return_inverse=True,
    )
    return case.kernel.prepare(
        keys, case.table, rows[first], case.source, case.budget
    ), inverse


def test_source_keys_share_across_baths_and_apply_parameters(kernel_case: KernelCase):
    keys = kernel_case.kernel.prepare_keys(kernel_case.table, np.arange(3))
    assert len(np.unique(keys)) == 1


def test_chained_bath_changes_and_split_restart(kernel_case: KernelCase):
    case = replace(
        kernel_case,
        table=replace(
            kernel_case.table,
            timeline=np.array([1, 1, 1]),
            start_ns=np.array([0.0, 0.7, 1.3]),
            duration_ns=np.array([0.7, 0.6, 0.8]),
        ),
    )
    prepared, inverse = prepare(case)
    actual = case.states.copy()
    actual[1:] = 0
    case.kernel.apply(prepared, inverse, case.table, np.arange(3), actual, case.budget)
    expected = []
    previous = case.source.model(tuple(case.table.level[0]))
    state = case.states[0]
    for row in range(3):
        model = case.source.model(tuple(case.table.level[row]))
        overlap = model.eigenvectors.conj().T @ previous.eigenvectors
        state = reference(case, row, overlap @ state @ overlap.conj().T)
        expected.append(state)
        previous = model
    assert_budget(actual, np.stack(expected), case.budget)
    split = case.states.copy()
    case.kernel.apply(
        prepared, inverse[:1], case.table, np.array([0]), split[:1], case.budget
    )
    left, right = (case.source.model(tuple(case.table.level[i])) for i in (0, 1))
    overlap = right.eigenvectors.conj().T @ left.eigenvectors
    split[1] = overlap @ split[0] @ overlap.conj().T
    case.kernel.apply(
        prepared, inverse[1:], case.table, np.array([1, 2]), split[1:], case.budget
    )
    assert_budget(actual, split, 1e-8)


def test_missing_waveform_and_expansion_fail_explicitly():
    case = flux_case()
    with pytest.raises(ValueError, match="expansion"):
        case.kernel.prepare(
            case.kernel.prepare_keys(case.table, np.array([0])),
            case.table,
            np.array([0]),
            evolution.ModelSource(case.states[0], case.source.model),
            case.budget,
        )
    prepared, inverse = prepare(case)
    with pytest.raises(ValueError, match="waveform"):
        case.kernel.apply(
            prepared,
            inverse,
            replace(case.table, flux_waveform=None),
            np.arange(3),
            case.states.copy(),
            case.budget,
        )


def test_engine_flux_wiring_across_batches_and_no_drive_midpoint_budget():
    case = flux_case()
    assert hasattr(evolution.TimelineBatch, "flux_waveform"), (
        "C3 must accept the continuous waveform"
    )
    times = np.arange(0, 0.401, 0.02)
    midpoint = times + (times[1] - times[0]) / 2
    levels = np.column_stack(
        (np.rint(12 * np.sin(midpoint * 2)), np.rint(7 * np.sin(midpoint)))
    ).astype(np.int64)
    schedule = evolution.FluxSchedule(np.array([0, len(times)]), times, levels)
    waveform = evolution.FluxWaveform(
        np.array([0, len(times)]),
        times,
        np.column_stack(
            (0.19 + 0.012 * np.sin(times * 2), -0.12 + 0.007 * np.sin(times))
        ),
    )
    driven = evolution.DriveTrack(
        "drive",
        np.array([0.0]),
        np.array([0.4]),
        np.array([0.17]),
        np.array([0.3]),
        np.array([0.12]),
        (evolution.EnvelopeSpec("constant"),),
        np.array([0]),
        np.array([True]),
    )
    batches = tuple(
        evolution.TimelineBatch(
            np.array([p]),
            (driven,) if p else (),
            schedule,
            np.array([0.4]),
            flux_waveform=waveform,
        )
        for p in range(2)
    )
    engine = evolution.Engine(1, 1000000)
    try:
        result = engine.evolve(case.source, batches, cancel=Event())
    finally:
        engine.close()
    assert result.paths.segments_by_kernel["t4"] > 1
    assert result.paths.segments_by_kernel["flux_dop853"] > 1
    expected = []
    for p in range(2):
        table = replace(
            case.table,
            timeline=np.array([0, 0, 0]),
            start_ns=np.zeros(3),
            duration_ns=np.full(3, 0.4),
            level=np.zeros((3, 2), dtype=np.int64),
            tones=np.full(3, p),
            tone_port=np.array([[0 if p else -1, -1, -1]] * 3),
            tone_envelope=np.array([[0 if p else -1, -1, -1]] * 3),
            tone_envelope_offset_ns=np.zeros((3, 3)),
            envelopes=(evolution.EnvelopeSpec("constant"),),
            flux_waveform=waveform,
        )
        # The independent reference switches the bath in the fixed segment basis.
        expected.append(
            reference(
                replace(case, table=table),
                0,
                case.source.initial_density,
                bath=schedule,
            )
        )
    assert_budget(result.densities, np.stack(expected), 1e-3)


def test_reference_rotated_bath_coherence_hand_calculation():
    case = flux_case()
    size = 3
    angle = 0.47
    vectors = np.eye(size, dtype=complex)
    vectors[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    gamma = np.zeros((size, size))
    gamma[0, 1] = gamma[1, 0] = 0.31
    base = synthetic_model(size)

    def build(level):
        return replace(
            base,
            level=level,
            eigenvectors=vectors if level[0] else np.eye(size, dtype=complex),
            energies_ghz=np.zeros(size),
            rates_per_ns=np.zeros((size, size)),
            dephasing_per_ns=gamma,
        )

    rho = np.diag([0.7, 0.3, 0.0]).astype(complex)
    source = evolution.ModelSource(
        rho, build, evolution.FluxExpansion(np.zeros((size, size), complex), ())
    )
    wave = evolution.FluxWaveform(
        np.array([0, 2]), np.array([0.0, 2.0]), np.zeros((2, 2))
    )
    table = replace(
        case.table,
        timeline=np.zeros(3, dtype=np.int64),
        start_ns=np.zeros(3),
        duration_ns=np.full(3, 2.0),
        level=np.zeros((3, 2), dtype=np.int64),
        tones=np.zeros(3, dtype=np.int64),
        tone_port=np.full((3, 3), -1),
        tone_envelope=np.full((3, 3), -1),
        flux_waveform=wave,
    )
    bath = evolution.FluxSchedule(
        np.array([0, 2]), np.array([0.0, 0.8]), np.array([[0, 0], [1, 0]])
    )
    rotated = vectors.conj().T @ rho @ vectors
    rotated[0, 1] *= np.exp(-0.31 * 1.2)
    rotated[1, 0] *= np.exp(-0.31 * 1.2)
    expected = vectors @ rotated @ vectors.conj().T
    actual = reference(replace(case, source=source, table=table), 0, rho, bath=bath)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
