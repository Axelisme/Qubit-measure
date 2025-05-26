from typing import Optional, Tuple

import numpy as np
import scqubits as scq


def calculate_n_oper(
    params: Tuple[float, float, float],
    flx: float,
    return_dim: int = 4,
    esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Calculate the matrix elements of the fluxonium
    """

    cutoff = 30

    fluxonium = scq.Fluxonium(
        *params, flux=flx, cutoff=cutoff, truncated_dim=return_dim
    )
    if esys is None:
        esys = fluxonium.eigensys(evals_count=return_dim)

    matrix_elements = fluxonium.n_operator(energy_esys=esys)

    return matrix_elements[:return_dim, :return_dim]


def calculate_n_oper_vs_flx(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    return_dim: int = 4,
    spectrum_data: Optional[scq.SpectrumData] = None,
) -> np.ndarray:
    """
    Calculate the matrix elements of the fluxonium vs. a parameter
    """

    if spectrum_data is None:
        cutoff = 40

        fluxonium = scq.Fluxonium(
            *params, flux=0.5, cutoff=cutoff, truncated_dim=return_dim
        )
        spectrum_data = fluxonium.get_matelements_vs_paramvals(
            operator="n_operator",
            param_name="flux",
            param_vals=flxs,
            evals_count=return_dim,
        )
    matrix_elements = spectrum_data.matrixelem_table

    return matrix_elements[:, :return_dim, :return_dim]


def calculate_system_n_oper_vs_flx(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    r_f: float,
    g: float,
    return_dim: int = 4,
    progress: bool = True,
) -> np.ndarray:
    """
    Calculate the matrix elements of the system vs. a parameter
    """

    cutoff = 40
    evals_count = 5
    resonator_dim = 5

    resonator = scq.Oscillator(r_f, truncated_dim=resonator_dim)
    fluxonium = scq.Fluxonium(
        *params, flux=0.5, cutoff=cutoff, truncated_dim=evals_count
    )
    hilbertspace = scq.HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
    )

    def update_hilbertspace(flx: float) -> None:
        fluxonium.flux = flx

    def get_n_oper(
        paramsweep: scq.ParameterSweep,
        paramindex_tuple: Tuple[int, int],
        paramvals_tuple: Tuple[float, float],
        **kwargs,
    ) -> np.ndarray:
        return paramsweep.op_in_dressed_eigenbasis(fluxonium.n_operator)

    old = scq.settings.PROGRESSBAR_DISABLED
    scq.settings.PROGRESSBAR_DISABLED = not progress
    sweep = scq.ParameterSweep(
        hilbertspace,
        {"params": flxs},
        update_hilbertspace=update_hilbertspace,
        evals_count=resonator_dim * evals_count,
        subsys_update_info={"params": [fluxonium]},
        labeling_scheme="LX",
    )
    scq.settings.PROGRESSBAR_DISABLED = old

    sweep.add_sweep(get_n_oper, "n_oper")

    return sweep["n_oper"]
