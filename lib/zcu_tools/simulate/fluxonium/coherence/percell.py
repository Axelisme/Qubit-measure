from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.oscillator import Oscillator

# -----------------------------
# The equations:
# $$
# P_{res} (n) = (1 - exp(-\beta \hbar \omega_r)) exp(-n \beta \hbar \omega_r) \\
# n_{th}(\omega_j) = \frac{1}{exp(\beta \hbar \omega_j) - 1} \\
# \Gamma^{up}_{l->l'} = \sum_{n,n'} P_{res}(n)\kappa n_{th} (\omega_{l',n'} - \omega_{l,n})\left|\langle l',n'\left|a^\dagger\right|l,n \rangle\right|^2 \\
# \Gamma^{down}_{l->l'} = \sum_{n,n'} P_{res}(n)\kappa (n_{th} (-\omega_{l',n'} + \omega_{l,n}) + 1)\left|\langle l',n'\left|a\right|l,n \rangle\right|^2 \\
# $$
# -----------------------------


def percell(
    Egn: NDArray[np.float64],
    Vec_gn: NDArray[np.complex128],
    Een: NDArray[np.float64],
    Vec_en: NDArray[np.complex128],
    Vec_n: NDArray[np.complex128],
    r_f: float,
    kappa: float,
    Temp: float,
    ns: NDArray[np.int64],
    fluxonium: Fluxonium,
    resonator: Oscillator,
) -> NDArray[np.float64]:
    """
    Calculate the transition rate of 0-1 caused by percell effect.
    """

    from scqubits.utils.spectrum_utils import identity_wrap

    beta_hbar = sc.hbar / (sc.k * Temp) * 1e9

    def P_res(n: NDArray[np.int64]) -> NDArray[np.float64]:
        return (1 - np.exp(-beta_hbar * r_f)) * np.exp(-n * beta_hbar * r_f)

    def n_th(w_j: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (np.exp(beta_hbar * w_j) - 1)

    P_res_ns = P_res(ns)

    E_1n0m = Egn[:, None] - Een[None, :]  # (ns, ns), from |1, n> to |0, m>

    # calculate the transition rate of 0-1 caused by up percell effect

    up_mask = E_1n0m > 0
    E_1n0m_up = E_1n0m.copy()
    E_1n0m_up[~up_mask] = np.inf
    n_ths = n_th(E_1n0m_up)  # (ns, ns)
    # calculate <0, n'|a^dag|1, n>
    ad_op = identity_wrap(
        resonator.creation_operator, resonator, [fluxonium, resonator], evecs=Vec_n
    )
    ad_1n0n = np.zeros((len(ns), len(ns)), dtype=complex)
    for ng in ns:
        for ne in ns:
            ad_1n0n[ng, ne] = Vec_gn[ng].dag() * ad_op * Vec_en[ne]

    Percell_up = np.nansum(P_res_ns[None, :] * kappa * n_ths * np.abs(ad_1n0n) ** 2)

    # calculate the transition rate of 0-1 caused by down percell effect

    down_mask = E_1n0m < 0
    E_1n0n_down = -E_1n0m.copy()
    E_1n0n_down[~down_mask] = np.inf
    n_ths = n_th(E_1n0n_down)  # (ns, ns)
    # calculate <0, n'|a|1, n>
    a_op = identity_wrap(
        resonator.annihilation_operator, resonator, [fluxonium, resonator], evecs=Vec_n
    )
    a_1n0n = np.zeros((len(ns), len(ns)), dtype=complex)
    for ng in ns:
        for ne in ns:
            a_1n0n[ng, ne] = Vec_gn[ng].dag() * a_op * Vec_en[ne]

    Percell_down = np.nansum(P_res_ns[None, :] * kappa * n_ths * np.abs(a_1n0n) ** 2)

    Percell_total = np.asarray(1 / (Percell_up + Percell_down), dtype=np.float64)

    return Percell_total


def calculate_percell_t1_vs_flx(
    flxs: NDArray[np.float64],
    r_f: float,
    kappa: float,
    g: float,
    Temp: float,
    params: Tuple[float, float, float],
) -> NDArray[np.float64]:
    Nf = 10
    Nr = 10
    ns = np.arange(0, Nr)

    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator
    from scqubits.core.param_sweep import ParameterSweep

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=40, truncated_dim=Nf)
    resonator = Oscillator(E_osc=r_f, truncated_dim=Nr)
    hilbertspace = HilbertSpace([fluxonium, resonator])
    hilbertspace.add_interaction(
        g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
    )

    def update_hilbertspace(flx: float) -> None:
        fluxonium.flux = flx

    sweep = ParameterSweep(
        hilbertspace=hilbertspace,
        paramvals_by_name={"flux": flxs},
        update_hilbertspace=update_hilbertspace,
        evals_count=Nf * Nr,
        subsys_update_info={"flux": [fluxonium]},
    )

    def get_percell_t1(
        paramsweep: ParameterSweep, paramindex_tuple: tuple, **kwargs
    ) -> NDArray[np.float64]:
        fluxonium = paramsweep.get_subsys(0)
        resonator = paramsweep.get_subsys(1)
        assert isinstance(fluxonium, Fluxonium)
        assert isinstance(resonator, Oscillator)

        evals = paramsweep["evals"][paramindex_tuple]
        evecs = paramsweep["evecs"][paramindex_tuple]
        Vec_n = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]

        def get_esys(state: int) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
            idxs = [paramsweep.dressed_index((state, n), paramindex_tuple) for n in ns]
            mask = np.array([idx is None for idx in idxs])

            # mask the None index
            idxs = np.array([idx if idx is not None else 0 for idx in idxs])

            En = evals[idxs]
            Vec_n = evecs[idxs]

            # fill the None index with nan
            En[mask] = np.nan

            return En, Vec_n

        # calculate the transition rate of 0-1 caused by percell effect
        Egn, Vec_gn = get_esys(0)
        Een, Vec_en = get_esys(1)

        return percell(
            Egn,
            Vec_gn,
            Een,
            Vec_en,
            Vec_n,
            r_f=r_f,
            kappa=kappa,
            Temp=Temp,
            ns=ns,
            fluxonium=fluxonium,
            resonator=resonator,
        )

    sweep.add_sweep(get_percell_t1, sweep_name="percell_t1")

    return sweep["percell_t1"]
