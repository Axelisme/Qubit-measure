"""Property and cross-validation tests for the TLS Bloch propagator.

Complements the analytic-limit gate in ``test_bloch_limits.py``: basic
propagator properties (identity in the no-dynamics limit, thermal steady state)
and a qutip Lindblad cross-validation against the hand-written affine
propagator.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.program.v2.sim.bloch import (
    Segment,
    bloch_generator,
    evolve,
    excited_population,
    ground_state,
    segment_propagator,
)


def test_no_dynamics_is_identity() -> None:
    """With Om=Delta=0 and no decay the propagator is the 4x4 identity."""

    prop = segment_propagator(0.0, 0.0, 0.0, 1.7, None, None, 0.0)
    assert np.allclose(prop, np.eye(4), atol=1e-12)


def test_steady_state_is_preserved() -> None:
    """The thermal steady state is a fixed point of free evolution."""

    thermal = 0.15
    v_eq = ground_state(thermal)
    v = evolve(v_eq, [Segment(0.0, 0.0, 0.0, 5.0, 2.0, 1.0, thermal)])
    assert np.allclose(v, v_eq, atol=1e-9)


def test_free_evolution_converges_to_thermal() -> None:
    """Long free evolution from ground relaxes toward z_eq = 2*thermal - 1."""

    thermal = 0.2
    t1 = 0.5
    v0 = np.array([0.0, 0.0, -1.0])  # pure ground
    v = evolve(v0, [Segment(0.0, 0.0, 0.0, 50.0 * t1, t1, t1, thermal)])
    assert v == pytest.approx([0.0, 0.0, 2.0 * thermal - 1.0], abs=1e-9)


def test_excited_population_mapping() -> None:
    """P_e maps z=-1 -> 0, z=+1 -> 1, z=0 -> 0.5."""

    assert excited_population(np.array([0.0, 0.0, -1.0])) == pytest.approx(0.0)
    assert excited_population(np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0)
    assert excited_population(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.5)


def test_evolve_rejects_wrong_shape() -> None:
    """evolve fast-fails on a non-(3,) initial vector."""

    with pytest.raises(ValueError):
        evolve(np.zeros(4), [])


def test_infinite_coherence_has_no_decay_terms() -> None:
    """Passing None for T1/T2 zeroes the diagonal relaxation entries."""

    gen = bloch_generator(1.0, 2.0, 0.3, None, None, 0.0)
    assert gen[0, 0] == 0.0
    assert gen[1, 1] == 0.0
    assert gen[2, 2] == 0.0
    assert gen[2, 3] == 0.0  # b vector is zero when gamma1 = 0


@pytest.mark.slow
def test_qutip_cross_validation() -> None:
    """Hand-written propagator matches a qutip Lindblad mesolve integration.

    The qutip model encodes the same conventions: H = 0.5 (Delta Z + Om(cos X +
    sin Y)) with excited at z=+1 (Z = |e><e| - |g><g|), amplitude damping with
    down/up rates set so z_eq = 2*thermal - 1, and pure dephasing tuned so that
    1/T2 = 1/(2 T1) + gamma_phi.
    """

    qt = pytest.importorskip("qutip")

    omega = 2.0 * np.pi * 1.0
    delta = 2.0 * np.pi * 0.7
    phase = 0.3
    t1 = 3.0
    t2 = 1.5
    duration = 0.8
    thermal = 0.05

    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    big_z = e * e.dag() - g * g.dag()  # excited at z = +1
    big_x = e * g.dag() + g * e.dag()
    big_y = -1j * (e * g.dag() - g * e.dag())

    ham = 0.5 * (
        delta * big_z + omega * (np.cos(phase) * big_x + np.sin(phase) * big_y)
    )

    gamma1 = 1.0 / t1
    gamma2 = 1.0 / t2
    c_down = np.sqrt(gamma1 * (1.0 - thermal)) * (g * e.dag())
    c_up = np.sqrt(gamma1 * thermal) * (e * g.dag())
    gamma_phi = gamma2 - gamma1 / 2.0
    assert gamma_phi > 0.0, "test parameters must give a physical T2 <= 2 T1"
    c_phi = np.sqrt(2.0 * gamma_phi) * 0.5 * big_z
    collapse = [c_down, c_up, c_phi]

    rho0 = g * g.dag()  # pure ground
    result = qt.mesolve(
        ham, rho0, [0.0, duration], collapse, e_ops=[big_x, big_y, big_z]
    )
    v_qutip = np.array(
        [result.expect[0][-1], result.expect[1][-1], result.expect[2][-1]]
    )

    v0 = np.array([0.0, 0.0, -1.0])  # pure ground, matching qutip rho0
    v_bloch = evolve(v0, [Segment(omega, delta, phase, duration, t1, t2, thermal)])

    max_dev = float(np.max(np.abs(v_qutip - v_bloch)))
    assert max_dev < 1e-5, f"max Bloch deviation {max_dev} exceeds tolerance"
