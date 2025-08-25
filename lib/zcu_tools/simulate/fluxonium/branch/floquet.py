from typing import Callable, Dict, List, Optional, Tuple

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
    ) -> None:
        self.params = params
        self.r_f = r_f
        self.g = g
        self.flx = flx
        self.qub_dim = qub_dim
        self.qub_cutoff = qub_cutoff

        import scqubits as scq  # lazy import

        fluxonium = scq.Fluxonium(
            *params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim
        )
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
        self, fbasis_n: List[qt.FloquetBasis], branchs: List[int], progress: bool = True
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

    def calc_branch_populations(
        self,
        fbasis_n: List[qt.FloquetBasis],
        branch_infos: Dict[int, List[int]],
        avg_times: np.ndarray,
        progress: bool = True,
    ) -> Dict[int, List[float]]:
        fstates_t_n = [
            np.array([fbasis.state(t=t, data=True).to_array() for t in avg_times])
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
        branch_populations = dict(branch_populations)

        return branch_populations


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
    ) -> None:
        self.params = params
        self.r_f = r_f
        self.g = g
        self.E_tls = E_tls
        self.flx = flx
        self.qub_dim = qub_dim
        self.qub_cutoff = qub_cutoff

        import scqubits as scq  # lazy import

        fluxonium = scq.Fluxonium(
            *params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim
        )
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
            avg_times = [0.0]

        fstates_t_n = [
            np.array([fbasis.state(t=t, data=True).to_array() for t in avg_times])
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
        branch_populations = dict(branch_populations)

        return branch_populations
