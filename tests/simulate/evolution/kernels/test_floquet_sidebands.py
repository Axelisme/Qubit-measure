"""Retained resonant bath harmonics through the public Floquet kernel seam."""

import numpy as np
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
    kernel,
)
from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel

from ..contract.kernel_suite import KernelCase, reference


def test_resonant_bath_sidebands_are_evolved_instead_of_averaged() -> None:
    zero = np.zeros((2, 2), complex)
    drive = np.array([[0.0, 0.02], [0.02, 0.0]], complex)
    model = SecularModel(
        (0,),
        "resonant-bath",
        np.array([0.0, 0.13]),
        np.eye(2, dtype=complex),
        (SecularDrive("stark", drive, zero), SecularDrive("probe", drive, zero)),
        np.array([[0.0, 0.001], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    table = kernel.SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([16.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([[0.13, 0.11]]),
        tone_amplitude=np.array([[0.5, 0.0001]]),
        tone_theta_rad=np.array([[0.0, 0.3]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    states = np.full((1, 2, 2), 0.5, complex)
    source = ModelSource(states[0], lambda level: model)
    solver = FloquetKernel()
    rows = np.array([0])
    assert solver.guard(table, rows, source, 0.001).all()
    case = KernelCase(solver, source, table, states.copy())
    expected = reference(case, 0, states[0])[None]
    prepared = solver.prepare(
        solver.prepare_keys(table, rows), table, rows, source, 0.001
    )
    solver.apply(prepared, rows, table, rows, states, 0.001)
    np.testing.assert_allclose(states, expected, atol=0.001, rtol=0)


def _adversarial_case(name: str) -> KernelCase:
    size = 3 if name == "ill-conditioned" else 2
    zero = np.zeros((size, size), complex)
    drive = zero.copy()
    drive[0, 1] = drive[1, 0] = 0.02
    rates = np.zeros((size, size))
    rates[0, 1] = 0.001
    if name == "ill-conditioned":
        # Equal-rate unidirectional population chain has a defective generator.
        rates[0, 1] = rates[1, 2] = 0.1
    if name == "exact-resonance":
        rates[:] = 0
    stark = zero if name in ("ill-conditioned", "exact-resonance") else drive
    model = SecularModel(
        (0,),
        name,
        np.arange(size) * 0.13,
        np.eye(size, dtype=complex),
        (SecularDrive("stark", stark, zero), SecularDrive("probe", drive, zero)),
        rates,
        np.zeros((size, size)),
    )
    table = kernel.SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([1e6 if name == "long-duration" else 16.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array(
            [[3.7, 0.13 if name == "exact-resonance" else 0.11]]
        ),
        tone_amplitude=np.array([[0.5, 0.0001]]),
        tone_theta_rad=np.array([[0.0, 0.3]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    states = np.full((1, size, size), 1 / size, complex)
    return KernelCase(
        FloquetKernel(), ModelSource(states[0], lambda level: model), table, states
    )


def test_ill_conditioned_floquet_frame_is_rejected() -> None:
    case = _adversarial_case("ill-conditioned")
    assert not case.kernel.guard(
        case.table, np.array([0]), case.source, case.budget
    ).any()


def test_long_duration_outside_modal_dynamic_range_is_rejected() -> None:
    case = _adversarial_case("long-duration")
    assert not case.kernel.guard(
        case.table, np.array([0]), case.source, case.budget
    ).any()


def test_exact_resonant_divisor_is_accepted_and_matches_lindblad() -> None:
    case = _adversarial_case("exact-resonance")
    rows = np.array([0])
    assert case.kernel.guard(case.table, rows, case.source, case.budget).all()
    expected = reference(case, 0, case.states[0])[None]
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(case.table, rows),
        case.table,
        rows,
        case.source,
        case.budget,
    )
    case.kernel.apply(prepared, rows, case.table, rows, case.states, case.budget)
    np.testing.assert_allclose(case.states, expected, atol=1e-3, rtol=0)
