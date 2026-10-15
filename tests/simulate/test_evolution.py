"""Synthetic references through the independent evolution and population API."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
GROUND = np.diag([1, 0]).astype(complex)
PLUS = np.ones((2, 2), dtype=complex) / 2


def model(*, h=None, sources=(), collapse=()):
    from zcu_tools.simulate.evolution import Basis, NumericModel

    return NumericModel(
        identity="synthetic-v1",
        basis=Basis("computational", ("g", "e"), np.eye(2, dtype=complex)),
        hamiltonian_ghz=np.zeros((2, 2), complex) if h is None else h,
        sources=sources,
        collapse_operators=collapse,
    )


def run(m, rho, end=20.0, controls=None, start=0.0):
    from zcu_tools.simulate.evolution import (
        ControlSequence,
        DensityState,
        Evolution,
        SolverSettings,
    )

    return Evolution(m, SolverSettings(rtol=1e-11, atol=1e-12, max_step_ns=0.25)).evolve(
        DensityState(m.basis, rho),
        ControlSequence() if controls is None else controls,
        start_ns=start,
        end_ns=end,
    )


def test_closed_evolution_explicit_initial_state_and_phase():
    from zcu_tools.simulate.evolution import DensityState, Evolution

    m = model(h=0.013 * Z)
    engine = Evolution(m)
    initial = DensityState(m.basis, PLUS)
    result = engine.evolve(initial, start_ns=0, end_ns=20)
    u = expm(-2j * np.pi * 0.013 * Z * 20)
    assert_allclose(result.density, u @ PLUS @ u.conj().T, atol=2e-9)
    assert_allclose(engine.evolve(initial, start_ns=0, end_ns=20).density, result.density)
    assert_allclose(initial.density, PLUS)
    assert_allclose(engine.evolve(DensityState(m.basis, GROUND), end_ns=20).density, GROUND)


@pytest.mark.parametrize("kind", ["relaxation", "dephasing", "thermal"])
def test_lindblad_analytic_limits_and_density_invariants(kind):
    down, up = 0.031, 0.007
    lowering = np.array([[0, 1], [0, 0]], complex)
    if kind == "relaxation":
        collapse, initial = (np.sqrt(down) * lowering,), PLUS
        excited = 0.5 * np.exp(-down * 20)
        coherence = 0.5 * np.exp(-down * 20 / 2)
    elif kind == "dephasing":
        collapse, initial = (np.sqrt(down / 2) * Z,), PLUS
        excited, coherence = 0.5, 0.5 * np.exp(-down * 20)
    else:
        collapse = (np.sqrt(down) * lowering, np.sqrt(up) * lowering.T)
        equilibrium = up / (up + down)
        initial = np.diag([1 - equilibrium, equilibrium]).astype(complex)
        excited, coherence = equilibrium, 0.0
    result = run(model(collapse=collapse), initial).density
    assert_allclose(result, [[1 - excited, coherence], [coherence, excited]], atol=2e-9)
    assert_allclose(np.trace(result), 1, atol=2e-9)
    assert_allclose(result, result.conj().T, atol=2e-9)
    assert np.linalg.eigvalsh(result).min() >= -2e-9


def test_joint_noncommuting_sources_are_not_serial_rotations():
    from zcu_tools.simulate.evolution import ControlPulse, ControlSequence, DriveSource

    sources = (DriveSource("x", 0.012 * X), DriveSource("z", 0.019 * Z))
    controls = ControlSequence((ControlPulse("x", 0, 20), ControlPulse("z", 0, 20)))
    actual = run(model(sources=sources), GROUND, controls=controls).density
    u = expm(-2j * np.pi * (0.012 * X + 0.019 * Z) * 20)
    serial = expm(-2j * np.pi * 0.019 * Z * 20) @ expm(-2j * np.pi * 0.012 * X * 20)
    assert_allclose(actual, u @ GROUND @ u.conj().T, atol=2e-9)
    assert np.linalg.norm(actual - serial @ GROUND @ serial.conj().T) > 0.1


def test_absolute_quantization_union_and_uninterrupted_envelope():
    from zcu_tools.simulate.evolution import (
        ControlPulse,
        ControlSequence,
        DriveSource,
        Envelope,
    )

    gaussian = Envelope.gaussian(sigma_ns=3.0, center_ns=6.0)
    controls = ControlSequence(
        (
            ControlPulse("x", 0.1, 12.1, envelope=gaussian, breakpoints_ns=(3.1, 5.1)),
            ControlPulse("silent", 3.2, 5.2),
        )
    )
    assert controls.boundaries_ns(start_ns=0, end_ns=12) == (0.0, 4.0, 6.0, 12.0)
    m = model(sources=(DriveSource("x", 0.021 * X), DriveSource("silent", 0 * Z)))
    result = run(m, GROUND, end=12, controls=controls).density
    area = quad(lambda t: np.exp(-0.5 * ((t - 6) / 3) ** 2), 0, 12)[0]
    u = expm(-2j * np.pi * 0.021 * X * area)
    assert_allclose(result, u @ GROUND @ u.conj().T, atol=2e-9)
    plateau = ControlSequence((ControlPulse("x", 0, 5000),))
    assert plateau.boundaries_ns(start_ns=0, end_ns=5000) == (0.0, 5000.0)


def test_quantization_error_is_separate_from_solver_error():
    from zcu_tools.simulate.evolution import ControlPulse, ControlSequence, DriveSource

    controls = ControlSequence((ControlPulse("x", 0, 5.1),))
    actual = run(model(sources=(DriveSource("x", 0.03 * X),)), GROUND, end=8, controls=controls)
    quantized_u = expm(-2j * np.pi * 0.03 * X * 6)
    original_u = expm(-2j * np.pi * 0.03 * X * 5.1)
    assert_allclose(actual.density, quantized_u @ GROUND @ quantized_u.conj().T, atol=2e-9)
    assert np.linalg.norm(actual.density - original_u @ GROUND @ original_u.conj().T) > 0.1


def test_phase_reset_continuous_virtual_z_and_explicit_fragment_continuation():
    from zcu_tools.simulate.evolution import (
        ControlPulse,
        ControlSequence,
        DriveSource,
        PhaseEvent,
    )

    pulses = (
        ControlPulse("drive", 0, 8, frequency_ghz=0.037, phase_rad=0.2),
        ControlPulse("drive", 8, 20, frequency_ghz=0.021, phase_mode="continuous"),
        ControlPulse("drive", 20, 28, frequency_ghz=0.029, phase_mode="reset"),
    )
    controls = ControlSequence(pulses, (PhaseEvent("drive", 12, 0.7),))
    m = model(sources=(DriveSource("drive", 0.013 * X, 0.011 * Y),))

    def rhs(t, flat):
        if t < 8:
            phase = 2 * np.pi * 0.037 * t + 0.2
        elif t < 20:
            phase = 2 * np.pi * (0.037 * 8 + 0.021 * (t - 8))
        else:
            phase = 2 * np.pi * 0.029 * (t - 20)
        phase += 0.7 if t >= 12 else 0
        h = 0.013 * np.cos(phase) * X + 0.011 * np.sin(phase) * Y
        rho = flat.reshape(2, 2)
        return (-2j * np.pi * (h @ rho - rho @ h)).ravel()

    expected = PLUS.ravel()
    for a, b in zip((0, 8, 12, 20), (8, 12, 20, 28), strict=True):
        expected = solve_ivp(rhs, (a, b), expected, rtol=1e-12, atol=1e-13, max_step=0.05).y[:, -1]
    whole = run(m, PLUS, end=28, controls=controls)
    first = run(m, PLUS, end=12, controls=controls)
    second = run(m, first.density, start=12, end=28, controls=controls)
    assert_allclose(whole.density, expected.reshape(2, 2), atol=2e-8)
    assert_allclose(second.density, whole.density, atol=2e-9)


def test_period_plus_tail_and_binary_integer_power():
    from zcu_tools.simulate.evolution import (
        ControlPulse,
        ControlSequence,
        DriveSource,
        propagator_power,
    )

    u = expm(-0.012j * X)
    assert_allclose(propagator_power(u, 0), np.eye(2))
    assert_allclose(propagator_power(u, 1000003), expm(-0.012j * X * 1000003), atol=2e-9)
    with pytest.raises(ValueError, match="nonnegative integer"):
        propagator_power(u, -1)
    f, strength, duration = 0.073, 0.011, 82.0
    controls = ControlSequence((ControlPulse("x", 0, duration, frequency_ghz=f),))
    actual = run(model(sources=(DriveSource("x", strength * X),)), GROUND, end=duration, controls=controls)
    area = np.sin(2 * np.pi * f * duration) / (2 * np.pi * f)
    expected_u = expm(-2j * np.pi * strength * area * X)
    assert_allclose(actual.density, expected_u @ GROUND @ expected_u.conj().T, atol=2e-9)


def test_population_conversion_names_source_target_and_order():
    from zcu_tools.simulate.evolution import Basis, DensityState, populations

    computational = model().basis
    hadamard = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    target = Basis("x-eigenstates", ("plus", "minus"), hadamard)
    result = populations(DensityState(computational, PLUS), target)
    assert result.basis.identity == "x-eigenstates"
    assert result.basis.labels == ("plus", "minus")
    assert_allclose(result.values, [1, 0], atol=2e-9)
    reverse = populations(DensityState(target, GROUND), computational)
    assert_allclose(reverse.values, [0.5, 0.5], atol=2e-9)


def test_invalid_density_and_nonfinite_dynamic_model_fail_without_repair():
    from zcu_tools.simulate.evolution import DensityState, Evolution, NumericModel

    m = model()
    with pytest.raises(ValueError, match="positive"):
        DensityState(m.basis, np.diag([1.1, -0.1]).astype(complex))
    with pytest.raises(ValueError, match="trace"):
        DensityState(m.basis, 2 * GROUND)
    bad = NumericModel("bad", m.basis, m.hamiltonian_ghz, hamiltonian_at=lambda t: np.full((2, 2), np.nan))
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        Evolution(bad).evolve(DensityState(m.basis, GROUND), end_ns=4)
