from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scqubits.core.param_sweep import ParameterSweep

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium


def calculate_dispersive(
    params: Tuple[float, float, float], flx: float, r_f: float, g: float
) -> Tuple[float, ...]:
    """
    Calculate the dispersive shift of ground and excited state
    """

    resonator_dim = 10
    cutoff = 30
    evals_count = 10

    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator

    resonator = Oscillator(r_f, truncated_dim=resonator_dim)
    fluxonium = Fluxonium(*params, flux=flx, cutoff=cutoff, truncated_dim=evals_count)
    hilbertspace = HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
    )
    hilbertspace.generate_lookup(ordering="LX")

    idx_00 = hilbertspace.dressed_index((0, 0))
    idx_10 = hilbertspace.dressed_index((1, 0))
    idx_01 = hilbertspace.dressed_index((0, 1))
    idx_11 = hilbertspace.dressed_index((1, 1))
    max_idx = max(idx_00, idx_10, idx_01, idx_11)

    evals = hilbertspace.eigenvals(evals_count=max_idx + 1)
    rf_0 = evals[idx_10] - evals[idx_00]
    rf_1 = evals[idx_11] - evals[idx_01]

    return rf_0, rf_1


def calculate_dispersive_sweep(
    sweep_list: NDArray[np.float64],
    update_fn: Callable[[Fluxonium, Any], None],
    g: float,
    r_f: float,
    progress: bool = True,
    res_dim: int = 5,
    qub_cutoff: int = 30,
    qub_dim: int = 20,
    return_dim: int = 2,
) -> Tuple[NDArray[np.float64], ...]:
    """
    Calculate the dispersive shift of ground and excited state vs. params of fluxonium
    """

    # import scqubits as scq  # lazy import
    import scqubits.settings as scq_settings
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator
    from scqubits.core.param_sweep import ParameterSweep

    resonator = Oscillator(r_f, truncated_dim=res_dim)
    fluxonium = Fluxonium(
        *(1.0, 1.0, 1.0), flux=0.5, cutoff=qub_cutoff, truncated_dim=qub_dim
    )
    hilbertspace = HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
    )

    def update_hilbertspace(sweep_param: Any) -> None:
        update_fn(fluxonium, sweep_param)

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = not progress
    sweep = ParameterSweep(
        hilbertspace,
        {"params": sweep_list},
        update_hilbertspace=update_hilbertspace,
        evals_count=res_dim * qub_dim,
        subsys_update_info={"params": [fluxonium]},
        labeling_scheme="LX",
    )
    scq_settings.PROGRESSBAR_DISABLED = old

    evals = sweep["evals"].toarray()

    rf_list = []
    idxs = np.arange(len(sweep_list))
    for i in range(return_dim):
        idx_0i = sweep.dressed_index((0, i)).toarray()
        idx_1i = sweep.dressed_index((1, i)).toarray()
        rf_list.append(evals[idxs, idx_1i] - evals[idxs, idx_0i])
    return tuple(rf_list)


def calculate_dispersive_vs_flx(
    params: Tuple[float, float, float],
    flxs: NDArray[np.float64],
    r_f: float,
    g: float,
    progress: bool = True,
    res_dim: int = 10,
    qub_cutoff: int = 30,
    qub_dim: int = 10,
    return_dim: int = 2,
) -> Tuple[NDArray[np.float64], ...]:
    """
    Calculate the dispersive shift of ground and excited state vs. flux
    """

    def update_hilbertspace(fluxonium: Fluxonium, flux: float) -> None:
        fluxonium.flux = flux
        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]

    return calculate_dispersive_sweep(
        flxs,
        update_hilbertspace,
        g,
        r_f,
        progress,
        res_dim,
        qub_cutoff,
        qub_dim,
        return_dim,
    )


def calculate_chi_sweep(
    sweep_list: NDArray,
    update_fn: Callable[[Fluxonium, Any], None],
    g: float,
    r_f: float,
    progress: bool = True,
    resonator_dim: int = 5,
    cutoff: int = 30,
    evals_count: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the chi of ground and excited state vs. params of fluxonium
    """

    import scqubits.settings as scq_settings
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator

    resonator = Oscillator(r_f, truncated_dim=resonator_dim)
    fluxonium = Fluxonium(
        *(1.0, 1.0, 1.0), flux=0.5, cutoff=cutoff, truncated_dim=evals_count
    )
    hilbertspace = HilbertSpace([fluxonium, resonator])
    hilbertspace.add_interaction(
        g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
    )

    def update_hilbertspace(sweep_param: Any) -> None:
        update_fn(fluxonium, sweep_param)

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = not progress
    sweep = ParameterSweep(
        hilbertspace,
        {"params": sweep_list},
        update_hilbertspace=update_hilbertspace,
        evals_count=resonator_dim * evals_count,
        subsys_update_info={"params": [fluxonium]},
        labeling_scheme="LX",
    )
    scq_settings.PROGRESSBAR_DISABLED = old

    return sweep["chi"]["subsys1":0, "subsys2":1]


def calculate_chi_vs_flx(
    params: Tuple[float, float, float],
    flxs: NDArray[np.float64],
    r_f: float,
    g: float,
    progress: bool = True,
    res_dim: int = 5,
    qub_cutoff: int = 30,
    qub_dim: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the dispersive shift of ground and excited state vs. flux
    """

    def update_hilbertspace(fluxonium: Fluxonium, flux: float) -> None:
        fluxonium.flux = flux
        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]

    return calculate_chi_sweep(
        flxs,
        update_hilbertspace,
        g,
        r_f,
        progress,
        res_dim,
        qub_cutoff,
        qub_dim,
    )
