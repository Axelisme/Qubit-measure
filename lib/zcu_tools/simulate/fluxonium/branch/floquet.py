from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import qutip as qt
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class FloquetBranchAnalysis:
    def __init__(
        self,
        params: Tuple[float, float, float],
        r_f: float,
        g: float,
        flx: float = 0.5,
        qub_dim: int = 40,
        qub_cutoff: int = 60,
        esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.r_f = r_f
        self.g = g

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        fluxonium = Fluxonium(
            *params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim
        )
        if esys is None:
            esys = fluxonium.eigensys(evals_count=qub_dim)
        H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
        n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
        self.H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]

    def make_floquet_basis(
        self, photon: float, precompute: Optional[np.ndarray] = None
    ) -> qt.FloquetBasis:
        return qt.FloquetBasis(
            self.H_with_drive,
            2 * np.pi / self.r_f,
            args=dict(amp=2 * self.g * np.sqrt(photon)),
            precompute=precompute,
        )

    def calc_branch_infos(
        self,
        fbasis_n: Sequence[qt.FloquetBasis],
        branchs: List[int],
        progress: bool = True,
    ) -> Dict[int, List[int]]:
        branch_infos = {b: [] for b in branchs}
        fstate_n = [fbasis.state(t=0) for fbasis in fbasis_n]
        qub_dim = len(fstate_n[0])

        for n in tqdm(
            range(len(fstate_n)), desc="Computing branch infos", disable=not progress
        ):
            for b, b_list in branch_infos.items():
                if n == 0:
                    # the first element is always the bare label state
                    last_state = qt.basis(qub_dim, b)
                else:
                    last_state = fstate_n[n - 1][b_list[n - 1]]
                # the next state is the one with the largest overlap with the last state
                dists = [np.abs(last_state.dag() @ fstate) for fstate in fstate_n[n]]
                b_list.append(np.argmax(dists))

        return branch_infos

    def calc_branch_energies(
        self, fbasis_n: List[qt.FloquetBasis], branch_infos: Dict[int, List[int]]
    ) -> Dict[int, List[float]]:
        branch_energies = {b: [] for b in branch_infos.keys()}
        for b, b_list in branch_infos.items():
            for n, b_idx in enumerate(b_list):
                branch_energies[b].append(fbasis_n[n].e_quasi[b_idx])
        return branch_energies

    def calc_branch_populations(
        self,
        fbasis_n: List[qt.FloquetBasis],
        branch_infos: Dict[int, List[int]],
        avg_times: np.ndarray,
        progress: bool = True,
    ) -> Dict[int, List[float]]:
        fstates_t_n = [
            np.array([fbasis.state(t=t, data=True).to_array() for t in avg_times])  # type: ignore
            for fbasis in tqdm(
                fbasis_n, desc="Computing time dependent states", disable=not progress
            )
        ]
        qub_dim = len(fstates_t_n[0][0])

        # time average over one period on the expectation value of population
        dag_weighted_bare_states = np.array(
            [np.sqrt(j) * qt.basis(qub_dim, j).dag().full() for j in range(qub_dim)]
        )

        def calc_pop(fstates_t, i) -> float:
            return np.sum(
                np.abs(np.dot(dag_weighted_bare_states, fstates_t[..., i].T)) ** 2
            ) / len(fstates_t)

        branch_populations = Parallel(n_jobs=-1)(
            delayed(
                lambda b: (
                    b,
                    [
                        calc_pop(fstates_t, i)
                        for fstates_t, i in zip(fstates_t_n, branch_infos[b])
                    ],
                )
            )(b)
            for b in branch_infos.keys()
        )
        branch_populations = dict(branch_populations)  # type: ignore
        branch_populations = cast(Dict[int, List[float]], branch_populations)

        return branch_populations


def calc_branch_infos(
    branchs: List[int],
    *,
    params: Tuple[float, float, float],
    r_f: float,
    g: float,
    flx: float,
    qub_dim: int,
    qub_cutoff: int,
    photons: np.ndarray,
    avg_times: Optional[np.ndarray] = None,
    progress: bool = True,
    esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict[int, List[int]], List[qt.FloquetBasis]]:
    fb_analysis = FloquetBranchAnalysis(
        params, r_f, g, flx=flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff, esys=esys
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    )
    assert isinstance(fbasis_n, list)
    fbasis_n = cast(List[qt.FloquetBasis], fbasis_n)

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs, progress=progress)
    return branch_infos, fbasis_n


def calc_ge_snr(
    params: Tuple[float, float, float],
    flx: float,
    r_f: float,
    rf_w: float,
    g: float,
    qub_dim: int,
    qub_cutoff: int,
    max_photon: int,
    esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    branchs = [0, 1]

    amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
    photons = (amps / (2 * g)) ** 2

    branch_infos, fbasis_n = calc_branch_infos(
        branchs=branchs,
        params=params,
        r_f=r_f,
        g=g,
        flx=flx,
        qub_dim=qub_dim,
        qub_cutoff=qub_cutoff,
        photons=photons,
        progress=False,
        esys=esys,
    )

    branch_energies = np.array(
        [
            [fbasis.e_quasi[branch_infos[b][n]] for n, fbasis in enumerate(fbasis_n)]
            for b in branchs
        ]
    )

    f01_over_n = branch_energies[1] - branch_energies[0]
    chi_over_n = (f01_over_n[1:] - f01_over_n[:-1]) / (photons[1:] - photons[:-1])

    def signal_diff(x: np.ndarray) -> np.ndarray:
        return 1 - np.exp(-(x**2) / (2 * rf_w**2))

    snrs = np.abs(signal_diff(chi_over_n) * np.sqrt(photons[:-1]))

    return photons[:-1], snrs


class FloquetWithTLSBranchAnalysis:
    def __init__(
        self,
        params: Tuple[float, float, float],
        r_f: float,
        g: float,
        E_tls: float,
        g_tls: float,
        flx: float = 0.5,
        qub_dim: int = 40,
        qub_cutoff: int = 60,
        esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.r_f = r_f
        self.g = g

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        fluxonium = Fluxonium(
            *params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim
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

    def make_floquet_basis(
        self, photon: float, precompute: Optional[np.ndarray] = None
    ) -> qt.FloquetBasis:
        return qt.FloquetBasis(
            self.H_with_drive,
            2 * np.pi / self.r_f,
            args=dict(amp=2 * self.g * np.sqrt(photon)),
            precompute=precompute,
        )

    def calc_branch_infos(
        self, fbasis_n: List[qt.FloquetBasis], branchs: List[int], progress: bool = True
    ) -> Dict[int, List[int]]:
        branch_infos = {b: [] for b in branchs}
        fstate_n = [fbasis.state(t=0) for fbasis in fbasis_n]
        qub_dim = len(fstate_n[0]) // 2

        for n in tqdm(
            range(len(fstate_n)), desc="Computing branch infos", disable=not progress
        ):
            for b, b_list in branch_infos.items():
                if n == 0:
                    # the first element is always the bare label state
                    last_state = qt.tensor(qt.basis(qub_dim, b), qt.basis(2, 0))
                else:
                    last_state = fstate_n[n - 1][b_list[n - 1]]
                # the next state is the one with the largest overlap with the last state
                dists = [np.abs(last_state.dag() @ fstate) for fstate in fstate_n[n]]
                b_list.append(np.argmax(dists))

        return branch_infos

    def calc_branch_populations(
        self,
        fbasis_n: List[qt.FloquetBasis],
        branch_infos: Dict[int, List[int]],
        avg_times: Optional[np.ndarray] = None,
        progress: bool = True,
    ) -> Dict[int, List[float]]:
        if avg_times is None:
            avg_times = np.array([0.0])

        fstates_t_n = [
            np.array([fbasis.state(t=t, data=True).to_array() for t in avg_times])  # type: ignore
            for fbasis in tqdm(
                fbasis_n,
                desc="Computing time dependent states",
                disable=not progress,
            )
        ]
        qub_dim = len(fstates_t_n[0][0]) // 2

        # time average over one period on the expectation value of population
        dag_weighted_bare_states = np.array(
            [
                np.sqrt(j)
                * qt.tensor(qt.basis(qub_dim, j), qt.basis(2, z)).dag().full()
                for j in range(qub_dim)
                for z in range(2)
            ]
        )

        def calc_pop(fstates_t, i) -> float:
            return np.sum(
                np.abs(np.dot(dag_weighted_bare_states, fstates_t[..., i].T)) ** 2
            ) / len(fstates_t)

        branch_populations = Parallel(n_jobs=-1)(
            delayed(
                lambda b: (
                    b,
                    [
                        calc_pop(fstates_t, i)
                        for fstates_t, i in zip(fstates_t_n, branch_infos[b])
                    ],
                )
            )(b)
            for b in branch_infos.keys()
        )
        assert isinstance(branch_populations, list)

        branch_populations = dict(branch_populations)  # type: ignore
        branch_populations = cast(Dict[int, List[float]], branch_populations)

        return branch_populations


def calc_branch_infos_with_tls(
    branchs: List[int],
    *,
    params: Tuple[float, float, float],
    r_f: float,
    g: float,
    E_tls: float,
    g_tls: float,
    flx: float,
    qub_dim: int,
    qub_cutoff: int,
    photons: np.ndarray,
    avg_times: Optional[np.ndarray] = None,
    progress: bool = True,
    esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict[int, List[int]], List[qt.FloquetBasis]]:
    fb_analysis = FloquetWithTLSBranchAnalysis(
        params,
        r_f,
        g,
        E_tls,
        g_tls,
        flx=flx,
        qub_dim=qub_dim,
        qub_cutoff=qub_cutoff,
        esys=esys,
    )

    fbasis_n = Parallel(n_jobs=-1)(
        delayed(fb_analysis.make_floquet_basis)(photon, precompute=avg_times)
        for photon in tqdm(
            photons, desc="Computing Floquet basis", disable=not progress
        )
    )
    assert isinstance(fbasis_n, list)
    fbasis_n = cast(List[qt.FloquetBasis], fbasis_n)

    branch_infos = fb_analysis.calc_branch_infos(fbasis_n, branchs, progress=progress)
    return branch_infos, fbasis_n
