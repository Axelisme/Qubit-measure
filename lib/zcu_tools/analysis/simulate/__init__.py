import warnings
from typing import Optional

import numpy as np
from scqubits import Fluxonium

from zcu_tools.config import config

from .predict import FluxoniumPredictor


def calculate_energy(
    flxs: np.ndarray,
    EJ: float,
    EC: float,
    EL: float,
    cutoff: Optional[int] = None,
    evals_count: Optional[int] = None,
    fluxonium: Optional[Fluxonium] = None,
) -> np.ndarray:
    """
    Calculate the energy of a fluxonium qubit.
    """

    if evals_count is None:
        evals_count = config.DEFAULT_EVALS_COUNT

    # because energy is periodic, remove repeated values and record index
    flxs = flxs % 1.0
    flxs = np.where(flxs < 0.5, flxs, 1.0 - flxs)

    flxs, uni_idxs = np.unique(flxs, return_inverse=True)
    sort_idxs = np.argsort(flxs)
    flxs = flxs[sort_idxs]

    if fluxonium is None:
        if cutoff is None:
            cutoff = config.DEFAULT_CUTOFF

        fluxonium = Fluxonium(
            EJ, EC, EL, flux=0.0, cutoff=cutoff, truncated_dim=evals_count
        )
    else:
        if cutoff is not None:
            warnings.warn(
                "cutoff are ignored when fluxonium is provided, use fluxonium.cutoff instead"
            )
        fluxonium.EJ = EJ
        fluxonium.EC = EC
        fluxonium.EL = EL
    energies = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=evals_count
    ).energy_table

    # rearrange energies to match the original order
    energies[sort_idxs, :] = energies
    energies = energies[uni_idxs, :]

    return energies


__all__ = ["calculate_energy", "FluxoniumPredictor"]
