from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import qutip as qt
from tqdm.auto import tqdm


def encode_floquet_hamiltonian(
    params: Tuple[float, float, float],
    r_f: float,
    g: float,
    flx: float = 0.5,
    qub_dim: int = 40,
    qub_cutoff: int = 60,
) -> Callable[[float, Optional[np.ndarray]], qt.FloquetBasis]:
    import scqubits as scq  # lazy import

    fluxonium = scq.Fluxonium(
        *params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim
    )
    esys = fluxonium.eigensys(evals_count=qub_dim)

    H = qt.Qobj(fluxonium.hamiltonian(energy_esys=esys))
    n_op = qt.Qobj(fluxonium.n_operator(energy_esys=esys))
    H_with_drive = [H, [n_op, lambda t, amp: amp * np.cos(r_f * t)]]

    def make_floquet_basis(
        photon: float, precompute: Optional[np.ndarray] = None
    ) -> qt.FloquetBasis:
        return qt.FloquetBasis(
            H_with_drive,
            2 * np.pi / r_f,
            args={"amp": 2 * g * np.sqrt(photon)},
            precompute=precompute,
        )

    return make_floquet_basis


def calc_branch_infos(
    fbasis_n: List[qt.FloquetBasis], branchs: List[int], progress: bool = True
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
    branch_populations = {
        b: [
            np.sum(np.abs(np.dot(dag_weighted_bare_states, fstates_t[..., i].T)) ** 2)
            / len(fstates_t)
            for fstates_t, i in zip(fstates_t_n, branch_infos[b])
        ]
        for b in branch_infos.keys()
    }

    return branch_populations
