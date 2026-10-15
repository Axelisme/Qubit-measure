"""Joint fixed-coordinate baths: stationarity and selected segment refinement."""

import numpy as np
from zcu_tools.simulate.evolution import Evolution, SolverSettings
from zcu_tools.simulate.ftf import (
    FluxoniumCircuit,
    JointModel,
    JointParameters,
    Resonator,
    SquidCircuit,
)


def _joint():
    f1 = FluxoniumCircuit(
        3.0, 0.8, 0.6, 10, 3, q_cap=1e4, q_ind=2e5, xqp=1e-7, aphi=0.001
    )
    f2 = FluxoniumCircuit(3.3, 0.7, 0.55, 10, 3, aphi=0.002)
    coupler = SquidCircuit(10.0, 0.3, 0.2, 4, 3, q_cap=2e4, aphi=0.001)
    return JointModel(
        JointParameters(f1, coupler, f2, 0.05, 0.06, -0.04, 8),
        np.array([0.35, 0.22, 0.4]),
        temperature_k=0.08,
    )


def _resonators():
    return (
        Resonator(6.0, (0.03, 0.01, 0.0), 0.001, 0.0001, 0.01),
        Resonator(6.2, (0.0, 0.01, 0.03), 0.001, 0.0002, 0.02),
    )


def test_coupled_gibbs_stationary_under_all_enabled_joint_baths() -> None:
    joint = _joint()
    initial = joint.initial_state()
    weights = np.exp(
        -0.04799243073366221 * (joint.energies_ghz - joint.energies_ghz[0]) / 0.08
    )
    weights /= weights.sum()
    np.testing.assert_allclose(initial.density, np.diag(weights), atol=1e-14)
    model = joint.numeric_model(
        lambda _t: joint.dc_flux,
        (),
        end_ns=12.0,
        bath_segment_ns=2.0,
        resonators=_resonators(),
    )
    final = Evolution(model, SolverSettings(1e-10, 1e-12, 0.05)).evolve(
        initial, end_ns=12.0
    )
    np.testing.assert_allclose(final.density, initial.density, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.trace(final.density), 1.0, atol=1e-10)
    assert np.linalg.eigvalsh(final.density).min() > -1e-9


def test_dynamic_joint_bath_segment_refinement_in_same_fixed_basis() -> None:
    joint = _joint()

    def flux(t):
        return (
            joint.dc_flux
            + np.array([0.002, 0.005, -0.001]) * np.sin(np.pi * t / 12) ** 2
        )

    outputs = []
    for segment in (4.0, 2.0):
        model = joint.numeric_model(
            flux, (), end_ns=12.0, bath_segment_ns=segment, resonators=_resonators()
        )
        outputs.append(
            Evolution(model, SolverSettings(1e-10, 1e-12, 0.05))
            .evolve(joint.initial_state(), end_ns=12.0)
            .density
        )
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5, rtol=0)
    for rho in outputs:
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-9)
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-9)
        assert np.linalg.eigvalsh(rho).min() > -1e-9
