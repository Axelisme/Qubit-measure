import json

import numpy as np
from scqubits import Fluxonium


def predict01(result_path: str, mA: float, mA_c: float = None):
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
