from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import scipy.constants as sc
from scipy.optimize import curve_fit


def boltzmann_distribution(
    freq_MHz: NDArray[np.float64], eff_T: float
) -> NDArray[np.float64]:
    exp_term = np.exp(-1e6 * sc.h * freq_MHz / (sc.k * 1e-3 * eff_T))
    return exp_term / np.sum(exp_term)


def effective_temperature(population: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the effective temperature of a population of qubits.

    Parameters
    ----------
    population : List[Tuple[float, float]]
        A list of tuples of (population, energy in MHz).

    Returns
    -------
    Tuple[float, float]
        The effective temperature in mK and its error.
    """

    # calculate the effective temperature
    if len(population) < 2:
        raise ValueError(
            "At least two qubits are required to calculate effective temperature."
        )

    pops, freqs = zip(*population)
    pops, freqs = np.array(pops), np.array(freqs)

    # directly calculate from two points
    eff_T = 1e9 * sc.h * (freqs[1] - freqs[0]) / (sc.k * np.log(pops[0] / pops[1]))
    err_T = 0.0
    if len(population) > 2:
        # fit the boltzmann distribution
        pOpt, err = curve_fit(boltzmann_distribution, freqs, pops, p0=(eff_T,))
        eff_T = pOpt[0]
        err_T = np.sqrt(np.diag(err))[0]

    return eff_T, err_T
