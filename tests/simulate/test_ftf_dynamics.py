"""Independent larger-parent dynamic reference, not a second production backend."""

import numpy as np
from scipy.integrate import solve_ivp


def _lift(operator, index):
    factors = [np.eye(3, dtype=complex) for _ in range(3)]
    factors[index] = operator
    return np.kron(np.kron(factors[0], factors[1]), factors[2])


def test_dynamic_flux_signed_complex_drive_against_larger_fixed_parent() -> None:
    import scqubits as scq
    from zcu_tools.simulate.evolution import ControlPulse, ControlSequence, Envelope, Evolution, DriveSource, SolverSettings
    from zcu_tools.simulate.ftf import FluxoniumCircuit, SquidCircuit, JointParameters, JointModel, Resonator

    f1 = FluxoniumCircuit(3., .8, .6, 12, 3)
    f2 = FluxoniumCircuit(3.3, .7, .55, 12, 3)
    c = SquidCircuit(10., .3, .2, 4, 3)
    parameters = JointParameters(f1, c, f2, .05, .06, -.04, 20)
    dc = np.array([.35, .22, .4])
    joint = JointModel(parameters, dc, temperature_k=0.)
    independent = [scq.Fluxonium(EJ=3., EC=.8, EL=.6, flux=dc[0], cutoff=12, truncated_dim=3),
                   scq.Fluxonium(EJ=3.3, EC=.7, EL=.55, flux=dc[2], cutoff=12, truncated_dim=3)]
    # Independent circuit construction uses scqubits' native fluxonium matrices,
    # charge-coordinate SQUID, and an untruncated 27-state local product reference.
    charge = np.diag(np.arange(-4, 5)).astype(complex)
    shift = np.eye(9, k=1, dtype=complex)
    cosine, sine = (shift + shift.T) / 2, (shift - shift.T) / (2j)
    def squid_h(flux):
        return 4*.3*charge@charge - 10*np.cos(np.pi*flux)*cosine + 2*np.sin(np.pi*flux)*sine
    h_native = (independent[0].hamiltonian(), squid_h(dc[1]), independent[1].hamiltonian())
    local_vectors = tuple(np.linalg.eigh(h)[1][:, :3] for h in h_native)
    native_n = (independent[0].n_operator(), charge, independent[1].n_operator())
    operators = tuple(_lift(v.conj().T @ op @ v, i) for i, (op, v) in enumerate(zip(native_n, local_vectors, strict=True)))
    n1, nc, n2 = operators
    coupling = .05*n1@nc + .06*n2@nc - .04*n1@n2
    def flux_at(t):
        return dc + np.array([.002, .005, -.001])*np.sin(np.pi*t/8)**2
    def parent_h(flux):
        independent[0].flux, independent[1].flux = flux[0], flux[2]
        matrices = (independent[0].hamiltonian(), squid_h(flux[1]), independent[1].hamiltonian())
        return coupling + sum((_lift(v.conj().T @ h @ v, i) for i, (h, v) in enumerate(zip(matrices, local_vectors, strict=True))), np.zeros((27, 27), dtype=complex))
    energy, vectors = np.linalg.eigh(parent_h(dc))
    np.testing.assert_allclose(joint.energies_ghz, energy[:20], atol=1e-12)
    coefficients = (.03+.01j, 0j, -.004+.003j)
    envelope = Envelope(lambda t: complex(-.2*np.exp(-.5*((t-4)/2)**2)*(1+.3j*np.sin(t))))
    controls = ControlSequence((ControlPulse("mw", 0., 8., envelope, .4, .3),))
    def drive(t):
        z = envelope.at(t)*np.exp(2j*np.pi*.4*t+.3j)
        return sum(((coefficient*z).real*operator for coefficient, operator in zip(coefficients, operators, strict=True)), np.zeros((27,27), dtype=complex))
    reference = solve_ivp(lambda t, state: -2j*np.pi*(parent_h(flux_at(t))+drive(t))@state,
                          (0., 8.), vectors[:, 0], method="DOP853", rtol=1e-11, atol=1e-13, max_step=.05)
    assert reference.success
    projected = vectors[:, :20].conj().T @ reference.y[:, -1]
    expected = np.outer(projected, projected.conj())
    zero = np.zeros((20,20), dtype=complex)
    x = sum((co.real*joint.charge_operator(i) for i, co in enumerate(coefficients)), zero.copy())
    y = sum((-co.imag*joint.charge_operator(i) for i, co in enumerate(coefficients)), zero.copy())
    resonators = (Resonator(6., (0.,0.,0.), .001, .001, 0.), Resonator(6.2, (0.,0.,0.), .001, .001, 0.))
    model = joint.numeric_model(flux_at, (DriveSource("mw", x, y),), end_ns=8., bath_segment_ns=2., resonators=resonators)
    actual = Evolution(model, SolverSettings(1e-10, 1e-12, .05)).evolve(joint.initial_state(), controls, end_ns=8.)
    np.testing.assert_allclose(np.abs(actual.density), np.abs(expected), atol=1e-5, rtol=0)
    assert abs(np.trace(actual.density)-1) < 1e-9
    assert np.linalg.eigvalsh(actual.density).min() >= -1e-9


def test_joint_static_coherence_phase_uses_coupled_energies() -> None:
    from zcu_tools.simulate.evolution import DensityState, Evolution, NumericModel
    from zcu_tools.simulate.ftf import FluxoniumCircuit, JointModel, JointParameters, SquidCircuit

    f = FluxoniumCircuit(3., .8, .6, 10, 3)
    joint = JointModel(JointParameters(f, SquidCircuit(10., .3, .2, 4, 3), f, .05, .06, -.04, 8), np.array([.35,.22,.4]), temperature_k=0.)
    rho = np.zeros((8,8), dtype=complex)
    rho[:2,:2] = .5
    result = Evolution(NumericModel("static", joint.basis, joint.hamiltonian(joint.dc_flux))).evolve(DensityState(joint.basis, rho), end_ns=12.)
    expected = .5*np.exp(-2j*np.pi*(joint.energies_ghz[0]-joint.energies_ghz[1])*12.)
    np.testing.assert_allclose(result.density[0,1], expected, atol=1e-9)
