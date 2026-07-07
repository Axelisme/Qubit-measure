"""Tests for the perturbative TLS resonance map over a no-TLS Floquet basis.

``calc_tls_resonance_map`` is the cheap replacement for a full
``calc_branch_infos_with_tls`` frequency scan (mist_tls_analysis.md deliverable
#2): it post-processes an already-computed ``fbasis_n`` algebraically, so these
tests verify the physics invariants (Fourier-component hermiticity, resonance
peaked at the tracked quasi-energy difference, cross-consistency with
``calc_floquet_fourier_melem``) and the fast-fail input validation. A small
``qub_dim=8`` basis over a handful of photons keeps the whole module ~1s.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from zcu_tools.simulate.fluxonium.branch.floquet import (
    FloquetBranchAnalysis,
    calc_floquet_fourier_melem,
    calc_tls_resonance_map,
)

_PARAMS = (4.5, 1.0, 0.5)
_R_F = 5.927
_G = 0.11
_FLUX = 0.5
_QUB_DIM = 8
_PHOTONS = np.array([0.0, 5.0, 10.0, 20.0])


def _build_floquet_setup() -> dict:
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*_PARAMS, flux=_FLUX, cutoff=40, truncated_dim=_QUB_DIM)
    esys = fluxonium.eigensys(evals_count=_QUB_DIM)
    n_op = np.asarray(fluxonium.n_operator(energy_esys=esys), dtype=np.complex128)
    # scqubits' eigenbasis operator carries ~1e-10 anti-hermitian numerical
    # noise; hermitize so the hermiticity invariant below is exact.
    n_op = (n_op + n_op.conj().T) / 2

    fb_analysis = FloquetBranchAnalysis(
        _PARAMS,
        _R_F,
        _G,
        flux=_FLUX,
        qub_dim=_QUB_DIM,
        qub_cutoff=40,
        esys=esys,
        solver_options=None,
    )
    fbasis_n: list[qt.FloquetBasis] = [
        fb_analysis.make_floquet_basis(photon) for photon in _PHOTONS
    ]
    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, [0, 1], progress=False)
    branch_energies = fb_analysis.calc_branch_energies(fbasis_n, branch_infos)
    ts = np.linspace(0.0, 2 * np.pi / _R_F, 32, endpoint=False)
    return dict(
        fbasis_n=fbasis_n,
        branch_infos=branch_infos,
        branch_energies=branch_energies,
        n_op=n_op,
        ts=ts,
    )


@pytest.fixture(scope="module")
def floquet_setup() -> dict:
    return _build_floquet_setup()


def test_fourier_melem_hermiticity(floquet_setup: dict) -> None:
    # n̂ is hermitian, so M_{i->j}^{(k)} == conj(M_{j->i}^{(-k)}). The relation is
    # exact for a fixed modes evaluation; 1e-8 sits well above qutip solver noise
    # and well below the dominant |M| components (~5e-2 and ~3e-4 here).
    fbasis = floquet_setup["fbasis_n"][1]
    n_op, ts = floquet_setup["n_op"], floquet_setup["ts"]
    # Raw mode columns 0/1: the invariant holds for any index pair, and this
    # pair has a dominant k=-1 component (~5e-2) at this drive strength.
    i, j = 0, 1
    fwd = calc_floquet_fourier_melem(fbasis, n_op, i, j, [-1, 0, 1], ts, _R_F)
    bwd = calc_floquet_fourier_melem(fbasis, n_op, j, i, [-1, 0, 1], ts, _R_F)
    assert abs(fwd[-1]) > 1e-3  # the invariant is checked on non-trivial components
    for k in (-1, 0, 1):
        assert fwd[k] == pytest.approx(np.conj(bwd[-k]), abs=1e-8)


def test_fourier_melem_cold_cache_is_repeatable() -> None:
    # qutip's Propagator mutates its memoized time grid when a new t is requested.
    # The Fourier helper must hide that cold-cache detail from callers.
    setup = _build_floquet_setup()
    fbasis = setup["fbasis_n"][1]
    n_op, ts = setup["n_op"], setup["ts"]
    i_from = setup["branch_infos"][1][1]
    i_to = setup["branch_infos"][0][1]

    first = calc_floquet_fourier_melem(fbasis, n_op, i_from, i_to, [1], ts, _R_F)
    second = calc_floquet_fourier_melem(fbasis, n_op, i_from, i_to, [1], ts, _R_F)

    np.testing.assert_allclose(first[1], second[1], rtol=0.0, atol=0.0)


def test_resonance_map_peaks_at_quasi_energy_difference(floquet_setup: dict) -> None:
    # With a single (0,1) pair and k=0, the map along E_tls must peak at the
    # tracked quasi-energy difference dE01(n), within the axis grid step.
    branch_energies = floquet_setup["branch_energies"]
    dE01 = np.array(branch_energies[1]) - np.array(branch_energies[0])
    E_axis = np.linspace(dE01.min() - 0.2, dE01.max() + 0.2, 401)
    res_map = calc_tls_resonance_map(
        floquet_setup["fbasis_n"],
        branch_energies,
        floquet_setup["branch_infos"],
        floquet_setup["n_op"],
        branch_pairs=[(0, 1)],
        harmonics=[0],
        E_tls_axis=E_axis,
        g_tls=1e-3,
        r_f=_R_F,
        ts=floquet_setup["ts"],
        gamma=2e-3,
        progress=False,
    )
    assert res_map.shape == (len(E_axis), len(_PHOTONS))
    step = E_axis[1] - E_axis[0]
    for n in range(len(_PHOTONS)):
        peak = E_axis[int(np.argmax(res_map[:, n]))]
        assert peak == pytest.approx(dE01[n], abs=step)


def test_resonance_map_matches_fourier_melem(floquet_setup: dict) -> None:
    # Cross-check: a single-pair single-harmonic map equals the Lorentzian built
    # by hand from calc_floquet_fourier_melem.
    branch_infos = floquet_setup["branch_infos"]
    branch_energies = floquet_setup["branch_energies"]
    n_op, ts = floquet_setup["n_op"], floquet_setup["ts"]
    E_axis = np.linspace(0.2, 1.5, 50)
    g_tls, gamma, k = 1.3e-3, 5e-3, 1
    res_map = calc_tls_resonance_map(
        floquet_setup["fbasis_n"],
        branch_energies,
        branch_infos,
        n_op,
        branch_pairs=[(1, 0)],
        harmonics=[k],
        E_tls_axis=E_axis,
        g_tls=g_tls,
        r_f=_R_F,
        ts=ts,
        gamma=gamma,
        progress=False,
    )
    for n, fbasis in enumerate(floquet_setup["fbasis_n"]):
        Mk = calc_floquet_fourier_melem(
            fbasis, n_op, branch_infos[1][n], branch_infos[0][n], [k], ts, _R_F
        )[k]
        dE = branch_energies[0][n] - branch_energies[1][n]
        expected = np.abs(g_tls * Mk) ** 2 / ((dE + k * _R_F - E_axis) ** 2 + gamma**2)
        np.testing.assert_allclose(res_map[:, n], expected, rtol=1e-12)


def test_fast_fail_validation(floquet_setup: dict) -> None:
    kwargs: dict = dict(
        branch_pairs=[(0, 1)],
        harmonics=[0],
        E_tls_axis=np.linspace(0.2, 1.5, 10),
        g_tls=1e-3,
        r_f=_R_F,
        ts=floquet_setup["ts"],
        gamma=5e-3,
        progress=False,
    )
    args = (
        floquet_setup["fbasis_n"],
        floquet_setup["branch_energies"],
        floquet_setup["branch_infos"],
        floquet_setup["n_op"],
    )
    with pytest.raises(ValueError, match="gamma"):
        calc_tls_resonance_map(*args, **{**kwargs, "gamma": 0.0})
    with pytest.raises(ValueError, match="E_tls_axis"):
        calc_tls_resonance_map(*args, **{**kwargs, "E_tls_axis": np.array([])})
    with pytest.raises(ValueError, match="ts"):
        calc_tls_resonance_map(*args, **{**kwargs, "ts": np.array([])})
    with pytest.raises(KeyError, match="branch 2"):
        calc_tls_resonance_map(*args, **{**kwargs, "branch_pairs": [(1, 2)]})
    truncated = {b: infos[:-1] for b, infos in floquet_setup["branch_infos"].items()}
    with pytest.raises(ValueError, match="tracking length"):
        calc_tls_resonance_map(args[0], args[1], truncated, args[3], **kwargs)
    with pytest.raises(ValueError, match="ts"):
        calc_floquet_fourier_melem(
            floquet_setup["fbasis_n"][0],
            floquet_setup["n_op"],
            0,
            1,
            [0],
            np.array([]),
            _R_F,
        )
