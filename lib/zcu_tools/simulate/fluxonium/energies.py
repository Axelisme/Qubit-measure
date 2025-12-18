from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData


def calculate_energy(
    params: Tuple[float, float, float],
    flx: float,
    cutoff: int = 40,
    evals_count: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the energy of a fluxonium qubit.
    """

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    fluxonium = Fluxonium(*params, flux=flx, cutoff=cutoff, truncated_dim=evals_count)
    return fluxonium.eigenvals(evals_count=evals_count)


def calculate_energy_vs_flx(
    params: Tuple[float, float, float],
    flxs: NDArray[np.float64],
    cutoff: int = 40,
    evals_count: int = 20,
    spectrum_data: Optional[SpectrumData] = None,
) -> Tuple[SpectrumData, NDArray[np.float64]]:
    """
    Calculate the energy of a fluxonium qubit.
    """

    from scqubits.core.fluxonium import Fluxonium

    if spectrum_data is not None:
        energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)
        return spectrum_data, energies  # early return

    # because energy is periodic, remove repeated values and record index
    flxs = flxs % 1.0
    flxs = np.where(flxs < 0.5, flxs, 1.0 - flxs)

    flxs, uni_idxs = np.unique(flxs, return_inverse=True)
    sort_idxs = np.argsort(flxs)
    flxs = flxs[sort_idxs]

    # calculate energy vs flux
    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=evals_count
    )
    energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)

    # rearrange energies to match the original order of flxs
    energies[sort_idxs, :] = energies
    energies = energies[uni_idxs, :]

    return spectrum_data, energies
