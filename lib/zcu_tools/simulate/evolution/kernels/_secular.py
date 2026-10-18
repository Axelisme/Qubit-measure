"""Secular rate algebra shared by static, split, modal and ODE propagators.

Inputs are validated model arrays in the secular basis. Rates[i,j] transfer
population from j to i, in inverse ns; pure dephasing is a pairwise (M,M)
array in inverse ns. Helpers return new mutable arrays, never alter inputs,
and leave time steps, Hamiltonian phases and frames to their callers.
"""

import numpy as np
from numpy.typing import NDArray

from ..secular import SecularModel


def population_generator(rates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the (M,M) population generator Q, with outgoing column sums removed."""
    return rates - np.diag(rates.sum(axis=0))


def relaxation_decay(rates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return pair coherence loss from population transfer alone, in inverse ns."""
    outgoing = rates.sum(axis=0)
    return (outgoing[:, None] + outgoing[None, :]) / 2


def coherence_decay(model: SecularModel) -> NDArray[np.float64]:
    """Return total (M,M) coherence loss, including the model's pure dephasing.

    The diagonal is not a population generator; callers propagate populations
    separately with Q and ignore diagonal entries of this array.
    """
    return relaxation_decay(model.rates_per_ns) + model.dephasing_per_ns


def secular_operators(
    model: SecularModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Q and total coherence loss for a model, both in inverse ns."""
    return population_generator(model.rates_per_ns), coherence_decay(model)
