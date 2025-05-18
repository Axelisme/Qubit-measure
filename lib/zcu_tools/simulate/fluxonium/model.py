from typing import Tuple

import numpy as np
from scqubits import Fluxonium


def calculate_energy(
    params: Tuple[float, float, float],
    flx: float,
    cutoff: int = 40,
    evals_count: int = 20,
) -> np.ndarray:
    """
    Calculate the energy of a fluxonium qubit.
    """

    fluxonium = Fluxonium(*params, flux=flx, cutoff=cutoff, truncated_dim=evals_count)
    return fluxonium.eigenvals(evals_count=evals_count)


def calculate_energy_vs_flx(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    cutoff: int = 40,
    evals_count: int = 20,
) -> np.ndarray:
    """
    Calculate the energy of a fluxonium qubit.
    """

    # because energy is periodic, remove repeated values and record index
    flxs = flxs % 1.0
    flxs = np.where(flxs < 0.5, flxs, 1.0 - flxs)

    flxs, uni_idxs = np.unique(flxs, return_inverse=True)
    sort_idxs = np.argsort(flxs)
    flxs = flxs[sort_idxs]

    # calculate energy vs flux
    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=evals_count
    )
    energies = spectrum.energy_table

    # rearrange energies to match the original order of flxs
    energies[sort_idxs, :] = energies
    energies = energies[uni_idxs, :]

    return energies
