"""Joint fixed-coordinate baths: stationarity and selected segment refinement."""

import numpy as np
from zcu_tools.simulate.evolution import Evolution, SolverSettings
from zcu_tools.simulate.ftf import FluxoniumCircuit, JointModel, JointParameters, Resonator, SquidCircuit


def _joint():
    f1 = FluxoniumCircuit(3., .8, .6, 10, 3, q_cap=1e4, q_ind=2e5, xqp=1e-7, aphi=.001)
    f2 = FluxoniumCircuit(3.3, .7, .55, 10, 3, aphi=.002)
    coupler = SquidCircuit(10., .3, .2, 4, 3, q_cap=2e4, aphi=.001)
    return JointModel(JointParameters(f1,coupler,f2,.05,.06,-.04,8), np.array([.35,.22,.4]), temperature_k=.08)


def _resonators():
    return (Resonator(6.,(.03,.01,0.),.001,.0001,.01), Resonator(6.2,(0.,.01,.03),.001,.0002,.02))


def test_coupled_gibbs_stationary_under_all_enabled_joint_baths() -> None:
    joint = _joint()
    initial = joint.initial_state()
    weights = np.exp(-.04799243073366221*(joint.energies_ghz-joint.energies_ghz[0])/.08)
    weights /= weights.sum()
    np.testing.assert_allclose(initial.density, np.diag(weights), atol=1e-14)
    model = joint.numeric_model(lambda _t: joint.dc_flux, (), end_ns=12., bath_segment_ns=2., resonators=_resonators())
    final = Evolution(model, SolverSettings(1e-10,1e-12,.05)).evolve(initial, end_ns=12.)
    np.testing.assert_allclose(final.density, initial.density, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.trace(final.density), 1., atol=1e-10)
    assert np.linalg.eigvalsh(final.density).min() > -1e-9


def test_dynamic_joint_bath_segment_refinement_in_same_fixed_basis() -> None:
    joint = _joint()
    def flux(t):
        return joint.dc_flux+np.array([.002,.005,-.001])*np.sin(np.pi*t/12)**2
    outputs = []
    for segment in (4.,2.):
        model = joint.numeric_model(flux, (), end_ns=12., bath_segment_ns=segment, resonators=_resonators())
        outputs.append(Evolution(model, SolverSettings(1e-10,1e-12,.05)).evolve(joint.initial_state(), end_ns=12.).density)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5, rtol=0)
    for rho in outputs:
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-9)
        np.testing.assert_allclose(np.trace(rho), 1., atol=1e-9)
        assert np.linalg.eigvalsh(rho).min() > -1e-9
