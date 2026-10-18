"""FTF producer contracts against fixed-coordinate Hamiltonian/Lindblad references."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.simulate import ftf
from zcu_tools.simulate.evolution import DriveSource


@pytest.fixture
def joint() -> ftf.JointModel:
    f1 = ftf.FluxoniumCircuit(3.1, 0.8, 0.7, 8, 2, q_cap=2e4, aphi=2e-4)
    f2 = ftf.FluxoniumCircuit(3.7, 0.9, 0.6, 8, 2, q_ind=3e4, xqp=1e-6)
    c = ftf.SquidCircuit(9, 0.4, 0.17, 3, 2, q_cap=3e4, aphi=1e-4)
    return ftf.JointModel(
        ftf.JointParameters(f1, c, f2, 0.08, -0.06, 0.02, 5),
        np.array([0.34, 0.21, 0.43]),
        temperature_k=0.06,
    )


@pytest.fixture
def resonators() -> tuple[ftf.Resonator, ftf.Resonator]:
    return (
        ftf.Resonator(6.2, (0.01, 0.004, 0), 0.002, 0.001, 0.03),
        ftf.Resonator(7.3, (0, 0.003, 0.02), 0.003, 0.001, 0.02),
    )


def test_secular_evolution_matches_full_collapse_operators(joint, resonators):
    x = joint.charge_operator(0)
    y = joint.charge_operator(2)
    level = (7, -3, 5)
    model = ftf.secular_model(
        joint, level, (DriveSource("mw", x, y),), resonators=resonators
    )
    v = model.eigenvectors
    flux = joint.dc_flux + 1e-3 * np.array(level)
    np.testing.assert_allclose(
        v @ np.diag(model.energies_ghz) @ v.conj().T,
        joint.hamiltonian(flux),
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        model.drives[0].in_phase_ghz, v.conj().T @ x @ v, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        model.drives[0].quadrature_ghz, v.conj().T @ y @ v, atol=1e-12, rtol=0
    )
    m = len(model.energies_ghz)
    rho = np.ones((m, m), dtype=complex) / m
    h = np.diag(model.energies_ghz) + 0.12 * model.drives[0].in_phase_ghz
    h += -0.08 * model.drives[0].quadrature_ghz
    collapse = tuple(
        v.conj().T @ c @ v for c in joint.collapse_operators(flux, resonators)
    )

    def full(r):
        result = -2j * np.pi * (h @ r - r @ h)
        for c in collapse:
            loss = c.conj().T @ c
            result += c @ r @ c.conj().T - (loss @ r + r @ loss) / 2
        return result

    def secular(r):
        rates = model.rates_per_ns
        outgoing = rates.sum(axis=0)
        result = -2j * np.pi * (h @ r - r @ h)
        result -= (
            (outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns
        ) * r
        result += np.diag(rates @ np.diag(r))
        return result

    units = np.eye(m * m).reshape(m * m, m, m).astype(complex)
    a = np.column_stack([full(r).ravel() for r in units])
    b = np.column_stack([secular(r).ravel() for r in units])
    error = np.max(np.abs(expm(40 * a) @ rho.ravel() - expm(40 * b) @ rho.ravel()))
    assert error < 1e-10
    assert np.max(model.dephasing_per_ns) > 0
    assert np.max(model.rates_per_ns) > 0


def test_flux_expansion_exactly_reconstructs_joint_hamiltonian(joint):
    expansion = ftf.flux_expansion(joint)
    for flux in (
        joint.dc_flux,
        np.array([-0.11, 0.57, 0.81]),
        np.array([1.2, -0.3, 0.5]),
    ):
        h = expansion.static.copy()
        for term in expansion.terms:
            angle = flux[term.loop] * np.pi * (2 if "2pi" in term.kind else 1)
            value = np.cos(angle) if term.kind.startswith("cos") else np.sin(angle)
            h += value * term.matrix
        np.testing.assert_allclose(h, joint.hamiltonian(flux), atol=1e-12, rtol=0)


def test_source_gibbs_cache_and_content_identity_include_absolute_dc(joint, resonators):
    source = ftf.model_source(joint, (), resonators=resonators)
    model = source.model((0, 0, 0))
    assert source.model((0, 0, 0)) is model
    weights = np.exp(
        -0.04799243073366221 * (model.energies_ghz - model.energies_ghz[0]) / 0.06
    )
    np.testing.assert_allclose(
        source.initial_density, np.diag(weights / weights.sum()), atol=1e-12
    )
    assert source.flux_expansion().terms
    same = ftf.model_source(joint, (), resonators=resonators)
    assert same.model((0, 0, 0)).identity == model.identity
    other = ftf.JointModel(
        joint.parameters, joint.dc_flux + [0.001, 0, 0], temperature_k=0.06
    )
    assert (
        ftf.model_source(other, (), resonators=resonators).model((0, 0, 0)).identity
        != model.identity
    )
    changed = ftf.JointModel(
        replace(joint.parameters, g12_ghz=0.03), joint.dc_flux, temperature_k=0.06
    )
    assert (
        ftf.model_source(changed, (), resonators=resonators).model((0, 0, 0)).identity
        != model.identity
    )
    assert source.model((1, 0, 0)).identity != model.identity


def test_active_degenerate_bath_names_level_and_secular_requirement(joint, resonators):
    class DegenerateJoint(ftf.JointModel):
        def eigensystem(self, flux):
            energy, v = super().eigensystem(flux)
            return np.zeros_like(energy), v

    degenerate = DegenerateJoint(joint.parameters, joint.dc_flux, temperature_k=0.06)
    with pytest.raises(ValueError, match=r"secular.*\(2, -1, 0\)"):
        ftf.secular_model(degenerate, (2, -1, 0), (), resonators=resonators)


def test_readout_uses_dc_joint_state_order_and_baths(joint, resonators):
    model = ftf.readout_model(joint, resonators)
    np.testing.assert_allclose(
        model.frequencies_ghz,
        joint.readout_frequencies_ghz(joint.dc_flux, resonators),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.rates_per_ns, joint.rates(joint.dc_flux), atol=1e-15
    )
    np.testing.assert_array_equal(model.kappa_external_per_ns, [0.002, 0.003])
    np.testing.assert_array_equal(model.kappa_internal_per_ns, [0.001, 0.001])
    assert model.state_labels == joint.basis.labels


@pytest.mark.parametrize("subsystem", [0, 1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_flux_operator_is_dc_derivative_in_retained_basis(
    joint: ftf.JointModel, subsystem: int, symmetric: bool
):
    if symmetric:
        parameters = replace(
            joint.parameters, coupler=replace(joint.parameters.coupler, asymmetry=0.0)
        )
        joint = ftf.JointModel(parameters, np.zeros(3), temperature_k=0.0)
    actual = joint.flux_operator(subsystem)
    delta = np.eye(3)[subsystem] * 1e-5
    expected = (
        joint.hamiltonian(joint.dc_flux + delta)
        - joint.hamiltonian(joint.dc_flux - delta)
    ) / 2e-5
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0, atol=1e-12)

    # Independent native potential, contracted through the public embedding.
    # No displaced eigensystem is used: both differences keep the DC basis.
    from scipy.linalg import cosm

    p = (joint.parameters.f1, joint.parameters.coupler, joint.parameters.f2)[subsystem]
    f = joint.dc_flux[subsystem]
    if isinstance(p, ftf.FluxoniumCircuit):
        a = np.diag(np.sqrt(np.arange(1, p.native_cutoff)), 1)
        phi = (2 * p.ec_ghz / p.el_ghz) ** 0.25 * (a + a.T)

        def fluxonium_potential(flux):
            return -p.ej_ghz * cosm(phi + 2 * np.pi * flux * np.eye(len(phi)))

        potential = fluxonium_potential
    else:
        shift = np.eye(2 * p.charge_cutoff + 1, k=1)
        cosine, sine = (shift + shift.T) / 2, (shift - shift.T) / (2j)

        def squid_potential(flux):
            return p.ejmax_ghz * (
                -np.cos(np.pi * flux) * cosine
                + p.asymmetry * np.sin(np.pi * flux) * sine
            )

        potential = squid_potential

    derivative = (potential(f + 1e-5) - potential(f - 1e-5)) / 2e-5
    embedded = np.moveaxis(joint.native_embedding(), subsystem, 0)
    expected_native = np.einsum(
        "aijm,ab,bijn->mn", embedded.conj(), derivative, embedded
    )
    np.testing.assert_allclose(actual, expected_native, rtol=2e-8, atol=2e-9)
    if not symmetric:
        assert np.max(np.abs(actual - np.diag(np.diag(actual)))) > 1e-3


@pytest.mark.parametrize("subsystem", [-1, 3])
def test_flux_operator_rejects_unknown_subsystem(joint: ftf.JointModel, subsystem: int):
    with pytest.raises(ValueError, match="subsystem"):
        joint.flux_operator(subsystem)
