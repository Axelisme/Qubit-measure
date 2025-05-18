from typing import Any, Callable, List, Tuple

import numpy as np
import scqubits as scq


def calculate_dispersive(
    params: Tuple[float, float, float],
    flx: float,
    r_f: float,
    g: float,
    cutoff: int = 40,
    evals_count: int = 20,
) -> Tuple[float, float]:
    """
    Calculate the dispersive shift of ground and excited state
    """

    resonator = scq.Oscillator(r_f, truncated_dim=2, id_str="resonator")
    fluxonium = scq.Fluxonium(
        *params,
        flux=flx,
        cutoff=cutoff,
        truncated_dim=evals_count,
        id_str="qubit",
    )
    hilbertspace = scq.HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g,
        op1=resonator.creation_operator,
        op2=fluxonium.n_operator,
        add_hc=True,
        id_str="q-r coupling",
    )
    hilbertspace.generate_lookup(ordering="LX")

    evals, _ = hilbertspace.eigensys(evals_count=2 * evals_count)

    idx_00 = hilbertspace.dressed_index((0, 0))
    idx_10 = hilbertspace.dressed_index((1, 0))
    idx_01 = hilbertspace.dressed_index((0, 1))
    idx_11 = hilbertspace.dressed_index((1, 1))

    rf_0 = evals[idx_10] - evals[idx_00]
    rf_1 = evals[idx_11] - evals[idx_01]

    return rf_0, rf_1


def calculate_dispersive_sweep(
    sweep_list: List[Any],
    update_fn: Callable[[scq.Fluxonium, Any], None],
    g: float,
    r_f: float,
    cutoff: int = 40,
    evals_count: int = 20,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dispersive shift of ground and excited state vs. params
    """

    resonator = scq.Oscillator(r_f, truncated_dim=2, id_str="resonator")
    fluxonium = scq.Fluxonium(
        *(1.0, 1.0, 1.0),
        flux=0.5,
        cutoff=cutoff,
        truncated_dim=evals_count,
        id_str="qubit",
    )
    hilbertspace = scq.HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g,
        op1=resonator.creation_operator,
        op2=fluxonium.n_operator,
        add_hc=True,
        id_str="q-r coupling",
    )

    def update_hilbertspace(sweep_param: Any) -> None:
        update_fn(fluxonium, sweep_param)

    old, scq.settings.PROGRESSBAR_DISABLED = (
        scq.settings.PROGRESSBAR_DISABLED,
        not progress,
    )
    sweep = scq.ParameterSweep(
        hilbertspace,
        {"params": sweep_list},
        update_hilbertspace=update_hilbertspace,
        evals_count=2 * evals_count,
        subsys_update_info={"params": [fluxonium]},
        labeling_scheme="LX",
    )
    scq.settings.PROGRESSBAR_DISABLED = old

    evals = sweep["evals"].toarray()

    idxs = np.arange(len(sweep_list))
    idx_00 = sweep.dressed_index((0, 0)).toarray()
    idx_10 = sweep.dressed_index((1, 0)).toarray()
    idx_01 = sweep.dressed_index((0, 1)).toarray()
    idx_11 = sweep.dressed_index((1, 1)).toarray()

    rf_0 = evals[idxs, idx_10] - evals[idxs, idx_00]
    rf_1 = evals[idxs, idx_11] - evals[idxs, idx_01]

    return rf_0, rf_1


def calculate_dispersive_vs_flx(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    r_f: float,
    g: float,
    cutoff: int = 40,
    evals_count: int = 20,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dispersive shift of ground and excited state vs. flux
    """

    def update_hilbertspace(fluxonium: scq.Fluxonium, flux: float) -> None:
        fluxonium.flux = flux
        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]

    return calculate_dispersive_sweep(
        flxs, update_hilbertspace, g, r_f, cutoff, evals_count, progress
    )
