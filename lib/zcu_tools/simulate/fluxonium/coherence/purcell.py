from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium.scq_settings import scq_progress

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.oscillator import Oscillator

# -----------------------------
# Reference: Zhang et al., Universal Fast-Flux Control of a Coherent,
# Low-Frequency Qubit, PRX 11, 011010 (2021).
# DOI: https://doi.org/10.1103/PhysRevX.11.011010
# https://arxiv.org/abs/2002.10653v1, Appendix G.5, Eqs. (G21)-(G22).
# The flux sweep keeps only 0↔1, not the paper's higher-state channels.
# The equations use angular frequencies/rates: omega = 2*pi*nu and
# kappa_rate = 2*pi*kappa, where nu and the input linewidth kappa are in GHz.
# The original frequency-selected bath uses a† for upward energy exchange and
# a for downward exchange, not the full quadrature a+a† (ADR-0071).
# $$
# Thermal resonator occupation probability: P_{res} (n) = (1 - exp(-\beta \hbar \omega_r)) exp(-n \beta \hbar \omega_r) \\
# Thermal mean photon number: n_{th}(\omega_j) = \frac{1}{exp(\beta \hbar \omega_j) - 1} \\
# Upward transition rate from l to l': \Gamma^{up}_{l->l'} = \sum_{n,n'} P_{res}(n)\kappa n_{th} (\omega_{l',n'} - \omega_{l,n})\left|\langle l',n'\left|a^\dagger\right|l,n \rangle\right|^2 \\
# Downward transition rate from l to l': \Gamma^{down}_{l->l'} = \sum_{n,n'} P_{res}(n)\kappa (n_{th} (-\omega_{l',n'} + \omega_{l,n}) + 1)\left|\langle l',n'\left|a\right|l,n \rangle\right|^2 \\
# $$
# -----------------------------


def purcell(
    Egn: NDArray[np.float64],
    Vec_gn: NDArray[np.complex128],
    Een: NDArray[np.float64],
    Vec_en: NDArray[np.complex128],
    Vec_n: NDArray[np.complex128],
    bare_rf: float,
    kappa: float,
    Temp: float,
    ns: NDArray[np.int64],
    fluxonium: Fluxonium,
    resonator: Oscillator,
) -> NDArray[np.float64]:
    """Return a thermal-averaged escape lifetime from the initial to final group.

    Egn and Vec_gn describe final dressed states; Een and Vec_en describe
    initial dressed states. Energies are energy/h in GHz, indexed by resonator
    photon labels ns. The vector arrays contain composite QuTiP eigenstates;
    Vec_n contains the bare resonator eigenvectors used to wrap its operators.
    fluxonium and resonator define the composite space in that order.

    bare_rf is the positive bare resonator frequency in GHz. kappa is the
    linewidth kappa_rate/(2*pi) in GHz (0.001 means 1 MHz), not a decay rate;
    the rate sum uses 2*pi*kappa in ns^-1. Temp is the bath temperature in K,
    including zero. Thermal populations use h*nu/(k_B*Temp).

    Positive final-minus-initial energy differences use a† and n_B; negative
    differences use a and n_B+1. Only the initial group is thermally averaged
    over resonator photons. Swapping groups evaluates the reverse escape.
    This directional lifetime is not equilibrium relaxation 1/(Gamma_eg+Gamma_ge).
    The sweep combines both directional rates to obtain equilibrium T1.
    Return a float64 scalar array in physical ns, or infinity when no allowed
    decay rate remains.
    Missing dressed labels (NaN energies) contribute no rate.
    """

    from scqubits.utils.spectrum_utils import identity_wrap

    beta_h = sc.h / (sc.k * Temp) * 1e9 if Temp != 0 else np.inf
    kappa_rate = 2 * np.pi * kappa

    if Temp == 0:
        P_res_ns = np.asarray(ns == 0, dtype=np.float64)
    else:
        resonator_exponent = beta_h * bare_rf
        P_res_ns = -np.expm1(-resonator_exponent) * np.exp(-ns * resonator_exponent)

    def n_th(frequency: NDArray[np.float64]) -> NDArray[np.float64]:
        if Temp == 0:
            return np.zeros_like(frequency)
        # Equivalent to 1/expm1(x), without overflow for a cold bath.
        exponent = beta_h * frequency
        return np.exp(-exponent) / -np.expm1(-exponent)

    E_1n0m = Egn[:, None] - Een[None, :]  # (final photons, initial photons)

    # calculate the transition rate of 0-1 caused by up Purcell effect

    up_mask = E_1n0m > 0
    E_1n0m_up = E_1n0m.copy()
    E_1n0m_up[~up_mask] = np.inf
    n_ths = n_th(E_1n0m_up)  # (ns, ns)
    # calculate <final, n'|a^dag|initial, n>
    ad_op = identity_wrap(
        resonator.creation_operator, resonator, [fluxonium, resonator], evecs=Vec_n
    )
    ad_1n0n = np.zeros((len(ns), len(ns)), dtype=complex)
    for ng in ns:
        for ne in ns:
            ad_1n0n[ng, ne] = Vec_gn[ng].dag() * ad_op * Vec_en[ne]  # type: ignore[attr-defined]
            # Vec_gn entries are QuTiP Qobj (not plain ndarray); .dag() is valid
            # at runtime but scqubits/qutip lack typed stubs.

    purcell_up = np.nansum(
        P_res_ns[None, :] * kappa_rate * n_ths * np.abs(ad_1n0n) ** 2
    )

    # calculate the transition rate of 0-1 caused by down Purcell effect

    down_mask = E_1n0m < 0
    E_1n0n_down = -E_1n0m.copy()
    E_1n0n_down[~down_mask] = np.inf
    n_ths = n_th(E_1n0n_down)  # (ns, ns)
    # calculate <final, n'|a|initial, n>
    a_op = identity_wrap(
        resonator.annihilation_operator, resonator, [fluxonium, resonator], evecs=Vec_n
    )
    a_1n0n = np.zeros((len(ns), len(ns)), dtype=complex)
    for ng in ns:
        for ne in ns:
            a_1n0n[ng, ne] = Vec_gn[ng].dag() * a_op * Vec_en[ne]  # type: ignore[attr-defined]
            # Vec_gn entries are QuTiP Qobj; .dag() is valid at runtime.

    # The spontaneous +1 belongs only to downward dressed-energy channels.
    downward_population = np.where(down_mask, n_ths + 1, 0.0)
    purcell_down = np.nansum(
        P_res_ns[None, :] * kappa_rate * downward_population * np.abs(a_1n0n) ** 2
    )

    escape_rate = purcell_up + purcell_down
    return np.asarray(np.inf if escape_rate == 0 else 1 / escape_rate, dtype=np.float64)


def calculate_purcell_t1_vs_flux(
    fluxs: NDArray[np.float64],
    bare_rf: float,
    kappa: float,
    g: float,
    Temp: float,
    params: tuple[float, float, float],
    *,
    progress: bool = True,
) -> NDArray[np.float64]:
    """Return two-level Purcell equilibrium T1 along normalized flux Phi/Phi0.

    fluxs is a one-dimensional flux array; params is (EJ, EC, EL) in GHz.
    bare_rf is the positive bare resonator frequency and g is the charge-mode
    coupling in GHz. kappa is the resonator linewidth kappa_rate/(2*pi) in GHz,
    so 0.001 denotes 1 MHz and contributes 2*pi*0.001 ns^-1 to the rate sum.
    Temp is the bath temperature in K, including zero.

    Average e-to-g and g-to-e rates separately over initial resonator thermal
    populations using h*nu/(k_B*Temp). Return T1=1/(Gamma_eg+Gamma_ge) for the
    0↔1 subset, excluding leakage to higher qubit states. Dressed-energy upward
    channels use a†/n_B; downward channels use a/(n_B+1), retaining spontaneous
    decay at zero temperature. These bath directions describe dressed energy
    exchange, not the qubit transition direction.

    Return float64 lifetimes in physical ns with the same shape as fluxs;
    no allowed decay gives infinity. Missing dressed labels contribute no
    rate. progress controls scqubits sweep progress and restores its prior
    global setting on return or failure. Diagonalization errors propagate.
    """

    Nf = 10
    Nr = 10
    ns = np.arange(0, Nr)

    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator
    from scqubits.core.param_sweep import ParameterSweep

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=40, truncated_dim=Nf)
    resonator = Oscillator(E_osc=bare_rf, truncated_dim=Nr)
    hilbertspace = HilbertSpace([fluxonium, resonator])
    hilbertspace.add_interaction(
        g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
    )

    def update_hilbertspace(flux: float) -> None:
        fluxonium.flux = flux

    with scq_progress(progress=progress):
        sweep = ParameterSweep(
            hilbertspace=hilbertspace,
            paramvals_by_name={"flux": fluxs},
            update_hilbertspace=update_hilbertspace,
            evals_count=Nf * Nr,
            subsys_update_info={"flux": [fluxonium]},
        )

        def get_purcell_t1(
            paramsweep: ParameterSweep, paramindex_tuple: tuple[int, ...], **kwargs
        ) -> NDArray[np.float64]:
            fluxonium = paramsweep.get_subsys(0)
            resonator = paramsweep.get_subsys(1)
            assert isinstance(fluxonium, Fluxonium)
            assert isinstance(resonator, Oscillator)

            evals = paramsweep["evals"][paramindex_tuple]
            evecs = paramsweep["evecs"][paramindex_tuple]
            Vec_n = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]

            def get_esys(
                state: int,
            ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
                idxs = [
                    paramsweep.dressed_index((state, n), paramindex_tuple) for n in ns
                ]
                mask = np.array([idx is None for idx in idxs])

                # mask the None index
                idxs = np.array([idx if idx is not None else 0 for idx in idxs])

                En = evals[idxs]
                Vec_n = evecs[idxs]

                # fill the None index with nan
                En[mask] = np.nan

                return En, Vec_n

            # Reuse these dressed states for both qubit directions.
            Egn, Vec_gn = get_esys(0)
            Een, Vec_en = get_esys(1)

            t1_eg = purcell(
                Egn,
                Vec_gn,
                Een,
                Vec_en,
                Vec_n,
                bare_rf=bare_rf,
                kappa=kappa,
                Temp=Temp,
                ns=ns,
                fluxonium=fluxonium,
                resonator=resonator,
            )
            t1_ge = purcell(
                Een,
                Vec_en,
                Egn,
                Vec_gn,
                Vec_n,
                bare_rf=bare_rf,
                kappa=kappa,
                Temp=Temp,
                ns=ns,
                fluxonium=fluxonium,
                resonator=resonator,
            )
            # An infinite directional lifetime contributes zero rate.
            equilibrium_rate = 1 / t1_eg + 1 / t1_ge
            return np.asarray(
                np.inf if equilibrium_rate == 0 else 1 / equilibrium_rate,
                dtype=np.float64,
            )

        sweep.add_sweep(get_purcell_t1, sweep_name="purcell_t1")

    return sweep["purcell_t1"]
