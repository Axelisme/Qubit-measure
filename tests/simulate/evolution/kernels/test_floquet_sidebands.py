"""Retained resonant bath harmonics through the public Floquet kernel seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
    kernel,
)
from zcu_tools.simulate.evolution.kernel import KernelContext
from zcu_tools.simulate.evolution.kernels import _floquet_sidebands as sidebands
from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import KernelCase, reference


def test_certificate_bins_preserve_omission_data_and_plans() -> None:
    # Characterize the existing per-bin calculation before factoring common work.
    indices = np.arange(96).reshape(6, 4, 4)
    frame = sidebands.Frame(
        np.array([0.0, 0.12, -0.12, 0.1, 0.0, 0.12]),
        (np.sin(indices) + 1j * np.cos(indices)) * 1e-4,
        np.array([-1, 0, -1, 1, 0, 1]),
        np.array([0, -1, 0, 1, 1, -1]),
        np.array([1e-8, 2e-8, 3e-8]),
        0.13,
    )
    lam = np.array([0.0, -0.004, -0.002 + 0.13j, -0.002 - 0.13j])
    modes = np.eye(4, dtype=complex) + np.triu(np.full((4, 4), 0.1 + 0.02j), 1)
    inverse = np.linalg.inv(modes).astype(np.complex128)
    parameters = np.array(
        [
            [1.0, 0.2, 0.4, 0.019, 0.01, 0.3, 0],
            [16.0, 0.2, 0.4, 0.023, -0.02, 0.3, 0],
            [1.0, 0.2, 0.4, 0.019, 0.01, 0.3, 1],
            [16.0, 0.2, 0.4, 0.023, -0.02, 0.3, 1],
        ]
    )
    prepared = sidebands.prepare(frame, lam, modes, inverse, parameters)
    # q, r, e from independent per-bin certificates, including a nonnormal basis.
    expected = np.array(
        [
            [
                0.00518860504833984,
                0.00553678024903929,
                0.0004925929876322,
                0.000489682683096,
                0.0008016742980128,
                0.0008106471415903,
            ],
            [
                0.00518860504833984,
                0.00406443670493019,
                0.0004925929876322,
                0.0006143988282523,
                0.0008016742980128,
                0.00073826966370096,
            ],
            [
                0.00518860504833984,
                0.00435706592716922,
                0.0004925929876322,
                0.00056063665073664,
                0.0008016742980128,
                0.00080342962912296,
            ],
            [
                0.00518860504833984,
                0.00431179335500431,
                0.0004925929876322,
                0.00054252721212667,
                0.0008016742980128,
                0.00081825961355971,
            ],
        ]
    )
    actual = [np.concatenate((c.q, c.r, c.e)) for _, c in prepared.certificates]
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=1e-17)
    expected_bounds = [
        0.0105000361470822,
        0.01073737403872176,
        0.01047642560257535,
        0.01074722600596878,
    ]
    for row, expected_bound in zip(parameters, expected_bounds, strict=True):
        cached = sidebands.plan(prepared, lam, modes, inverse, row)
        independent = sidebands.plan(frame, lam, modes, inverse, row)
        np.testing.assert_allclose(cached.bound, expected_bound, rtol=5e-14)
        np.testing.assert_array_equal(cached.rows, independent.rows)
        np.testing.assert_array_equal(cached.columns, independent.columns)
        np.testing.assert_allclose(cached.values, independent.values, rtol=5e-14)
        np.testing.assert_allclose(cached.exponents, independent.exponents, rtol=5e-14)
    assert not frame.certificates
    for _, certificate in prepared.certificates:
        assert not certificate.q.flags.writeable
        assert not certificate.values.flags.writeable


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
        tone_amplitude=np.array([[0.2, 0.0001]]),
        tone_theta_rad=np.array([[0.0, 0.3]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    states = np.full((1, 2, 2), 0.5, complex)
    source = ModelSource(states[0], lambda level: model)
    solver = FloquetKernel()
    rows = np.array([0])
    assert solver.guard(table, rows, KernelContext(source, 0.001, WorkPlan())).all()
    case = KernelCase(solver, source, table, states.copy())
    expected = reference(case, 0, states[0])[None]
    prepared = solver.prepare(
        solver.prepare_keys(table, rows, KernelContext(source, 0.001, WorkPlan())),
        table,
        rows,
        KernelContext(source, 0.001, WorkPlan()),
    )
    solver.apply(prepared, rows, table, rows, states, case.context(rows))
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
    stark = zero if name == "ill-conditioned" else drive
    # The DC probe has nonzero diagonal modal coefficients with zero divisors.
    probe = drive + np.diag([0.02, -0.02]) if name == "exact-resonance" else drive
    model = SecularModel(
        (0,),
        name,
        np.arange(size, dtype=np.float64) * 0.13,
        np.eye(size, dtype=complex),
        (SecularDrive("stark", stark, zero), SecularDrive("probe", probe, zero)),
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
            [[0.13, 0.0] if name == "exact-resonance" else [3.7, 0.11]]
        ),
        tone_amplitude=np.array([[0.2 if name == "exact-resonance" else 0.5, 0.0001]]),
        tone_theta_rad=np.array([[0.0, 0.3]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    states = np.full((1, size, size), 1 / size, complex)
    return KernelCase(
        FloquetKernel(), ModelSource(states[0], lambda level: model), table, states
    )


@pytest.mark.parametrize(
    ("name", "accepted"),
    [
        ("ill-conditioned", [False, False, False]),
        ("long-duration", [False, False, False]),
        ("exact-resonance", [True, True, False]),
    ],
)
def test_multiple_probe_bins_preserve_row_admission(
    name: str, accepted: list[bool]
) -> None:
    case = _adversarial_case(name)
    table = case.table
    repeated = replace(
        table,
        timeline=np.arange(3),
        start_ns=np.repeat(table.start_ns, 3, axis=0),
        duration_ns=np.repeat(table.duration_ns, 3, axis=0),
        level=np.repeat(table.level, 3, axis=0),
        tones=np.repeat(table.tones, 3, axis=0),
        tone_port=np.repeat(table.tone_port, 3, axis=0),
        tone_frequency_ghz=np.column_stack(
            (np.repeat(table.tone_frequency_ghz[0, 0], 3), [0.0, 0.019, 0.023])
        ),
        tone_amplitude=np.repeat(table.tone_amplitude, 3, axis=0),
        tone_theta_rad=np.repeat(table.tone_theta_rad, 3, axis=0),
        tone_envelope=np.repeat(table.tone_envelope, 3, axis=0),
        tone_envelope_offset_ns=np.repeat(table.tone_envelope_offset_ns, 3, axis=0),
    )
    batch = case.kernel.guard(
        repeated, np.arange(3), KernelContext(case.source, case.budget, WorkPlan())
    )
    independent = [
        case.kernel.guard(
            repeated,
            np.array([row]),
            KernelContext(case.source, case.budget, WorkPlan()),
        )[0]
        for row in range(3)
    ]
    np.testing.assert_array_equal(batch, independent)
    np.testing.assert_array_equal(batch, accepted)


def test_ill_conditioned_floquet_frame_is_rejected() -> None:
    case = _adversarial_case("ill-conditioned")
    assert not case.kernel.guard(
        case.table, np.array([0]), KernelContext(case.source, case.budget, WorkPlan())
    ).any()


def test_long_duration_outside_modal_dynamic_range_is_rejected() -> None:
    case = _adversarial_case("long-duration")
    assert not case.kernel.guard(
        case.table, np.array([0]), KernelContext(case.source, case.budget, WorkPlan())
    ).any()


def test_exact_resonant_divisor_is_accepted_and_matches_lindblad() -> None:
    case = _adversarial_case("exact-resonance")
    rows = np.array([0])
    assert case.kernel.guard(
        case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
    ).all()
    expected = reference(case, 0, case.states[0])[None]
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(
            case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
        ),
        case.table,
        rows,
        KernelContext(case.source, case.budget, WorkPlan()),
    )
    case.kernel.apply(prepared, rows, case.table, rows, case.states, case.context(rows))
    np.testing.assert_allclose(case.states, expected, atol=1e-3, rtol=0)


def test_cached_sideband_domain_requires_every_probe_bin() -> None:
    case = _adversarial_case("exact-resonance")
    rows = np.array([0])
    context = KernelContext(case.source, case.budget, WorkPlan())
    built = case.kernel.guard_prepared(case.table, rows, context)
    prepared = built.preparations[0].prepared
    admitted = case.kernel.admit_prepared(prepared, case.table, rows, context)
    assert admitted is not None, "covered Floquet domain must admit without build"
    np.testing.assert_array_equal(admitted.accepted, built.accepted)
    # Phase and duration are current-row decisions, not part of the frame key.
    changed = replace(case.table, duration_ns=np.array([8.0]))
    admitted = case.kernel.admit_prepared(prepared, changed, rows, context)
    assert admitted is not None
    expected = case.kernel.guard_prepared(changed, rows, context)
    np.testing.assert_array_equal(admitted.accepted, expected.accepted)
    missing = replace(case.table, tone_frequency_ghz=np.array([[0.13, 0.023]]))
    assert case.kernel.admit_prepared(prepared, missing, rows, context) is None
