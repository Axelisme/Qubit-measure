import json
import warnings
from typing import Optional

import numpy as np
from scqubits import Fluxonium

from zcu_tools.config import config


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


def predict01(result_path: str, mA: float, mA_c: Optional[float] = None):
    """
    Predict the 0-1 transition frequency of a fluxonium qubit.
    Args:
        result_path (str): Path to the result file.
        mA (float): Current in mA.
        mA_c (float, optional): Overwrite the mA_c value in the result file. Defaults to None.
    Returns:
        f01 (float): 0-1 transition frequency in GHz.
    """
    with open(result_path, "r") as f:
        data = json.load(f)
    params = np.array(
        [data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]]
    )
    _mA_c = data["half flux"]
    period = data["period"]

    if mA_c is not None:
        _mA_c = mA_c
    # Convert mA to flux
    flx = (mA - _mA_c) / period + 0.5

    fluxonium = Fluxonium(*params, flux=flx, cutoff=40, truncated_dim=2)
    energies = fluxonium.eigenvals(evals_count=2)

    return float(energies[1] - energies[0])
