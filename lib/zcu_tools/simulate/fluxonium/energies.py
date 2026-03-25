from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData


def calculate_energy(
    params: tuple[float, float, float],
    flux: float,
    cutoff: int = 40,
    evals_count: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the energy of a fluxonium qubit.
    """

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    fluxonium = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=evals_count)
    return fluxonium.eigenvals(evals_count=evals_count)


def calculate_energy_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    cutoff: int = 40,
    evals_count: int = 20,
    spectrum_data: Optional[SpectrumData] = None,
) -> tuple[SpectrumData, NDArray[np.float64]]:
    """
    Calculate the energy of a fluxonium qubit.
    """

    from scqubits.core.fluxonium import Fluxonium

    if spectrum_data is not None:
        energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)
        return spectrum_data, energies  # early return

    # because energy is periodic, remove repeated values and record index
    fluxs = fluxs % 1.0
    fluxs = np.where(fluxs < 0.5, fluxs, 1.0 - fluxs)

    fluxs, uni_idxs = np.unique(fluxs, return_inverse=True)
    sort_idxs = np.argsort(fluxs)
    fluxs = fluxs[sort_idxs]

    # calculate energy vs flux
    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux", fluxs, evals_count=evals_count
    )
    energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)

    # rearrange energies to match the original order of fluxs
    energies[sort_idxs, :] = energies
    energies = energies[uni_idxs, :]

    return spectrum_data, energies
