from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import qutip as qt
from numpy.typing import NDArray
from tqdm.auto import tqdm

# Relaxed FloquetBasis ODE tolerance used only on the ge-SNR design-search path.
# The design metric is sort(snr)[-3], a relative ranking, so the ~6e-5 relative
# error this introduces is far below the parameter-grid step. Callers that need
# the strict (qutip-default) solver pass solver_options=None explicitly.
SNR_SOLVER_OPTIONS: dict[str, Any] = {"rtol": 1e-3, "atol": 1e-5}


class _BaseFloquetBranchAnalysis:
    def __init__(
        self, r_f: float, *, solver_options: dict[str, Any] | None = None
    ) -> None:
        self.r_f = r_f
        self.solver_options = solver_options
        self.H_with_drive: Any = None

    def _floquet_args(self, photon: float) -> dict[str, float]:
        raise NotImplementedError

    def _qub_dim(self, state_dim: int) -> int:
        return state_dim

    def _bare_label_state(self, state_dim: int, branch: int) -> qt.Qobj:
        return qt.basis(self._qub_dim(state_dim), branch)

    def _dag_weighted_bare_states(self, state_dim: int) -> NDArray[np.complex128]:
        qub_dim = self._qub_dim(state_dim)
        return np.array(
            [np.sqrt(j) * qt.basis(qub_dim, j).dag().full() for j in range(qub_dim)]
        )

    def make_floquet_basis(
        self, photon: float, precompute: NDArray[np.float64] | None = None
    ) -> qt.FloquetBasis:
        return qt.FloquetBasis(
            self.H_with_drive,
            2 * np.pi / self.r_f,
            args=self._floquet_args(photon),
            options=self.solver_options,  # type: ignore[arg-type]  # qutip stubs declare dict but accept None
            precompute=precompute,  # type: ignore[arg-type]  # qutip stubs declare ArrayLike but accept None
        )

    def calc_branch_infos(
        self,
        fbasis_n: Sequence[qt.FloquetBasis],
        branchs: list[int],
        progress: bool = True,
    ) -> dict[int, list[int]]:
        branch_infos = {b: [] for b in branchs}
        # qutip stubs have an overloaded `state` that doesn't accept int `t`;
        # the actual API accepts t=0 for the initial-time state.
        fstate_n = [fbasis.state(t=0) for fbasis in fbasis_n]  # type: ignore[call-overload]
        state_dim = len(fstate_n[0])

        for n in tqdm(
            range(len(fstate_n)), desc="Computing branch infos", disable=not progress
        ):
            for b, b_list in branch_infos.items():
                if n == 0:
                    # the first element is always the bare label state
                    last_state = self._bare_label_state(state_dim, b)
                else:
                    last_state = fstate_n[n - 1][b_list[n - 1]]
                # the next state is the one with the largest overlap with the last state
                dists = [np.abs(last_state.dag() @ fstate) for fstate in fstate_n[n]]
                b_list.append(int(np.argmax(dists)))

        return branch_infos

    def calc_branch_energies(
        self, fbasis_n: list[qt.FloquetBasis], branch_infos: dict[int, list[int]]
    ) -> dict[int, list[float]]:
        branch_energies = {b: [] for b in branch_infos.keys()}
        for b, b_list in branch_infos.items():
            for n, b_idx in enumerate(b_list):
                branch_energies[b].append(fbasis_n[n].e_quasi[b_idx])
        return branch_energies

    def calc_branch_populations(
        self,
        fbasis_n: list[qt.FloquetBasis],
        branch_infos: dict[int, list[int]],
        avg_times: NDArray[np.float64] | None = None,
        progress: bool = True,
    ) -> dict[int, list[float]]:
        if avg_times is None:
            avg_times = np.array([0.0])

        fstates_t_n = [
            np.array([fbasis.state(t=t, data=True).to_array() for t in avg_times])  # type: ignore
            for fbasis in tqdm(
                fbasis_n, desc="Computing time dependent states", disable=not progress
            )
        ]
        state_dim = len(fstates_t_n[0][0])

        # time average over one period on the expectation value of population
        dag_weighted_bare_states = self._dag_weighted_bare_states(state_dim)

        def calc_pop(fstates_t, i) -> float:
            return np.sum(
                np.abs(np.dot(dag_weighted_bare_states, fstates_t[..., i].T)) ** 2
            ) / len(fstates_t)

        # Serial: branchs is a 2-element list, so this is a handful of tasks off
        # the snr hot path; kept serial for consistency with the photon layer.
        return {
            b: [
                calc_pop(fstates_t, i)
                for fstates_t, i in zip(fstates_t_n, branch_infos[b])
            ]
            for b in branch_infos.keys()
        }


class FloquetBranchAnalysis(_BaseFloquetBranchAnalysis):
    def __init__(
        self,
        params: tuple[float, float, float],
        r_f: float,
        g: float,
        flux: float = 0.5,
        qub_dim: int = 40,
        qub_cutoff: int = 60,
        esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(r_f, solver_options=solver_options)
        self.g = g

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        fluxonium = Fluxonium(
            *params, flux=flux, cutoff=qub_cutoff, truncated_dim=qub_dim
        )
        if esys is None:
            esys = fluxonium.eigensys(evals_count=qub_dim)
        H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
        n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
        self.H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]

    def _floquet_args(self, photon: float) -> dict[str, float]:
        return {"amp": 2 * self.g * np.sqrt(photon)}


def calc_branch_infos(
    branchs: list[int],
    *,
    params: tuple[float, float, float],
    r_f: float,
    g: float,
    flux: float,
    qub_dim: int,
    qub_cutoff: int,
    photons: NDArray[np.float64],
    avg_times: NDArray[np.float64] | None = None,
    progress: bool = True,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
    solver_options: dict[str, Any] | None = SNR_SOLVER_OPTIONS,
) -> tuple[dict[int, list[int]], list[qt.FloquetBasis]]:
    # This is a ge-SNR entry point, so the relaxed tolerance is the default;
    # pass solver_options=None for the strict (qutip-default) solver.
    fb_analysis = FloquetBranchAnalysis(
        params,
        r_f,
        g,
        flux=flux,
        qub_dim=qub_dim,
        qub_cutoff=qub_cutoff,
        esys=esys,
        solver_options=solver_options,
    )

    # Serial build: each FloquetBasis is ~3ms, far below loky dispatch+pickle
    # overhead, so process-level parallelism here is a net loss (measured serial
    # 851ms vs Parallel 1386ms for 263 photons). Cell-level parallelism in the
    # search.py caller owns the concurrency instead (no nested oversubscription).
    fbasis_n: list[qt.FloquetBasis] = [
        fb_analysis.make_floquet_basis(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    ]

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs, progress=progress)
    return branch_infos, fbasis_n


def calc_ge_snr(
    params: tuple[float, float, float],
    flux: float,
    r_f: float,
    rf_w: float,
    g: float,
    qub_dim: int,
    qub_cutoff: int,
    max_photon: int,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
    solver_options: dict[str, Any] | None = SNR_SOLVER_OPTIONS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    branchs = [0, 1]

    amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
    photons = (amps / (2 * g)) ** 2

    branch_infos, fbasis_n = calc_branch_infos(
        branchs=branchs,
        params=params,
        r_f=r_f,
        g=g,
        flux=flux,
        qub_dim=qub_dim,
        qub_cutoff=qub_cutoff,
        photons=photons,
        progress=False,
        esys=esys,
        solver_options=solver_options,
    )

    branch_energies = np.array(
        [
            [fbasis.e_quasi[branch_infos[b][n]] for n, fbasis in enumerate(fbasis_n)]
            for b in branchs
        ]
    )

    f01_over_n = branch_energies[1] - branch_energies[0]
    chi_over_n = (f01_over_n[1:] - f01_over_n[:-1]) / (photons[1:] - photons[:-1])

    def signal_diff(x: NDArray) -> NDArray:
        return 1 - np.exp(-(x**2) / (2 * rf_w**2))

    snrs = np.abs(signal_diff(chi_over_n) * np.sqrt(photons[:-1]))

    return photons[:-1], snrs


class FloquetWithTLSBranchAnalysis(_BaseFloquetBranchAnalysis):
    def __init__(
        self,
        params: tuple[float, float, float],
        r_f: float,
        g: float,
        E_tls: float,
        g_tls: float,
        flux: float = 0.5,
        qub_dim: int = 40,
        qub_cutoff: int = 60,
        esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(r_f, solver_options=solver_options)
        self.g = g

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        fluxonium = Fluxonium(
            *params, flux=flux, cutoff=qub_cutoff, truncated_dim=qub_dim
        )
        if esys is None:
            esys = fluxonium.eigensys(evals_count=qub_dim)
        H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
        n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))

        H = (
            qt.tensor(H, qt.identity(2))
            + qt.tensor(qt.identity(qub_dim), 0.5 * E_tls * qt.sigmaz())
            + g_tls * qt.tensor(n_op, qt.sigmax())
        )
        n_op = qt.tensor(n_op, qt.identity(2))

        self.H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]

    def _floquet_args(self, photon: float) -> dict[str, float]:
        return {"amp": 2 * self.g * np.sqrt(photon)}

    def _qub_dim(self, state_dim: int) -> int:
        return state_dim // 2

    def _bare_label_state(self, state_dim: int, branch: int) -> qt.Qobj:
        qub_dim = self._qub_dim(state_dim)
        return qt.tensor(qt.basis(qub_dim, branch), qt.basis(2, 0))

    def _dag_weighted_bare_states(self, state_dim: int) -> NDArray[np.complex128]:
        qub_dim = self._qub_dim(state_dim)
        return np.array(
            [
                np.sqrt(j)
                * qt.tensor(qt.basis(qub_dim, j), qt.basis(2, z)).dag().full()
                for j in range(qub_dim)
                for z in range(2)
            ]
        )


def calc_branch_infos_with_tls(
    branchs: list[int],
    *,
    params: tuple[float, float, float],
    r_f: float,
    g: float,
    E_tls: float,
    g_tls: float,
    flux: float,
    qub_dim: int,
    qub_cutoff: int,
    photons: NDArray[np.float64],
    avg_times: NDArray[np.float64] | None = None,
    progress: bool = True,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
    solver_options: dict[str, Any] | None = None,
) -> tuple[dict[int, list[int]], list[qt.FloquetBasis]]:
    # Unlike calc_branch_infos (the design-search snr entry), the TLS path is not
    # on the snr hot path and feeds precision-sensitive mist analyses, so the
    # default here is the strict (qutip-default) solver; pass solver_options to
    # opt into relaxed tolerance.
    fb_analysis = FloquetWithTLSBranchAnalysis(
        params,
        r_f,
        g,
        E_tls,
        g_tls,
        flux=flux,
        qub_dim=qub_dim,
        qub_cutoff=qub_cutoff,
        esys=esys,
        solver_options=solver_options,
    )

    # Serial build for the same reason as calc_branch_infos: per-photon work is
    # too small for process-level parallelism to pay off; concurrency lives at
    # the cell layer.
    fbasis_n: list[qt.FloquetBasis] = [
        fb_analysis.make_floquet_basis(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    ]

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs, progress=progress)
    return branch_infos, fbasis_n


def calc_floquet_fourier_melem(
    fbasis: qt.FloquetBasis,
    n_op: NDArray[np.complex128],
    i_from: int,
    i_to: int,
    harmonics: Sequence[int],
    ts: NDArray[np.float64],
    omega: float,
) -> dict[int, complex]:
    """Time-averaged Fourier components of ``<u_{i_to}(t)| n |u_{i_from}(t)>``.

    ``ts`` must sample exactly one drive period ``[0, 2*pi/omega)`` uniformly
    with the endpoint excluded, so the plain mean below is the period average.
    Modes come from ``fbasis.mode(t)``: the columns of the returned matrix are
    the periodic Floquet modes ``u_b(t)``.
    """
    if len(ts) == 0:
        raise ValueError("ts must contain at least one sample of the drive period")
    modes = _sample_floquet_modes(fbasis, ts)
    phases = _floquet_harmonic_phases(harmonics, ts, omega)
    return _floquet_fourier_melem_from_modes(
        modes, n_op, i_from=i_from, i_to=i_to, harmonic_phases=phases
    )


def _floquet_harmonic_phases(
    harmonics: Sequence[int], ts: NDArray[np.float64], omega: float
) -> dict[int, NDArray[np.complex128]]:
    return {
        k: np.asarray(np.exp(1j * k * omega * ts), dtype=np.complex128)
        for k in harmonics
    }


def _floquet_fourier_melem_from_modes(
    modes: NDArray[np.complex128],
    n_op: NDArray[np.complex128],
    *,
    i_from: int,
    i_to: int,
    harmonic_phases: dict[int, NDArray[np.complex128]],
) -> dict[int, complex]:
    ket = modes[:, :, i_from]  # (nt, D)
    bra = modes[:, :, i_to].conj()  # (nt, D)
    mel = np.einsum("ti,ij,tj->t", bra, n_op, ket)  # (nt,)
    return {k: complex(np.mean(phase * mel)) for k, phase in harmonic_phases.items()}


def _sample_floquet_modes(
    fbasis: qt.FloquetBasis, ts: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Sample periodic Floquet modes after priming qutip's propagator cache."""
    for t in ts:
        fbasis.U(float(t % fbasis.T))
    return np.stack([fbasis.mode(t=float(t), data=True).to_array() for t in ts])


def calc_tls_resonance_map(
    fbasis_n: Sequence[qt.FloquetBasis],
    branch_energies: dict[int, list[float]],
    branch_infos: dict[int, list[int]],
    n_op: NDArray[np.complex128],
    *,
    branch_pairs: Sequence[tuple[int, int]],
    harmonics: Sequence[int],
    E_tls_axis: NDArray[np.float64],
    g_tls: float,
    r_f: float,
    ts: NDArray[np.float64],
    gamma: float,
    progress: bool = True,
) -> NDArray[np.float64]:
    """Detuning-weighted TLS resonance strength, shape ``(len(E_tls_axis), len(fbasis_n))``.

    Perturbative E_tls sweep over an already-computed no-TLS Floquet basis: for
    each photon column ``n``, branch pair ``(b_from, b_to)`` and drive harmonic
    ``k``, the Lorentzian-weighted ``|g_tls * M_k|**2`` is accumulated with
    detuning ``(E[b_to] - E[b_from]) + k*r_f - E_tls``, where ``M_k`` is the
    period-averaged Fourier component of ``<u_{b_to}(t)| n |u_{b_from}(t)>``.
    This is pure algebraic post-processing (no new ODE integration), so the
    E_tls axis is swept for free — the cheap replacement for a full
    ``calc_branch_infos_with_tls`` frequency scan.

    ``g_tls`` only sets the overall scale of the map; ``gamma`` is the
    Lorentzian visualization width. Branch labels are mapped to per-photon
    Floquet indices via ``branch_infos`` so tracking stays consistent with the
    quasi-energy curves. ``ts`` must sample one drive period as in
    ``calc_floquet_fourier_melem``; both directions of a pair carry distinct
    physics (``dE`` is signed), so include ``(1, 0)`` as well as ``(0, 1)``.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if len(E_tls_axis) == 0:
        raise ValueError("E_tls_axis must not be empty")
    if len(ts) == 0:
        raise ValueError("ts must contain at least one sample of the drive period")
    n_photons = len(fbasis_n)
    for b in {b for pair in branch_pairs for b in pair}:
        if b not in branch_infos or b not in branch_energies:
            raise KeyError(f"branch {b} missing from branch_infos/branch_energies")
        if len(branch_infos[b]) != n_photons or len(branch_energies[b]) != n_photons:
            raise ValueError(
                f"branch {b} tracking length does not match len(fbasis_n)={n_photons}"
            )

    E_tls_axis = np.asarray(E_tls_axis, dtype=np.float64)
    strength = np.zeros((len(E_tls_axis), n_photons))
    harmonic_phases = _floquet_harmonic_phases(harmonics, ts, r_f)
    for n in tqdm(range(n_photons), desc="TLS resonance map", disable=not progress):
        modes = _sample_floquet_modes(fbasis_n[n], ts)
        for b_from, b_to in branch_pairs:
            fourier_melems = _floquet_fourier_melem_from_modes(
                modes,
                n_op,
                i_from=branch_infos[b_from][n],
                i_to=branch_infos[b_to][n],
                harmonic_phases=harmonic_phases,
            )
            dE = branch_energies[b_to][n] - branch_energies[b_from][n]
            for k in harmonics:
                Mk = fourier_melems[k]
                det = dE + k * r_f - E_tls_axis  # (nE,)
                strength[:, n] += (np.abs(g_tls * Mk) ** 2) / (det**2 + gamma**2)
    return strength


class FloquetDualCouplingBranchAnalysis(_BaseFloquetBranchAnalysis):
    def __init__(
        self,
        params: tuple[float, float, float],
        r_f: float,
        g1: float,
        g2: float,
        flux: float = 0.5,
        qub_dim: int = 40,
        qub_cutoff: int = 60,
        esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(r_f, solver_options=solver_options)
        self.g1 = g1
        self.g2 = g2

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        fluxonium = Fluxonium(
            *params, flux=flux, cutoff=qub_cutoff, truncated_dim=qub_dim
        )
        if esys is None:
            esys = fluxonium.eigensys(evals_count=qub_dim)
        H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
        n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
        phi_op = qt.Qobj(fluxonium.phi_operator(energy_esys=esys))
        self.H_with_drive = [
            H,
            [n_op, lambda t, amp1, amp2: amp1 * np.cos(r_f * t)],
            [phi_op, lambda t, amp1, amp2: amp2 * np.sin(r_f * t)],
        ]

    def _floquet_args(self, photon: float) -> dict[str, float]:
        return {
            "amp1": 2 * self.g1 * np.sqrt(photon),
            "amp2": 2 * self.g2 * np.sqrt(photon),
        }
