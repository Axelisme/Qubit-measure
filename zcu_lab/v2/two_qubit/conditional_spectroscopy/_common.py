"""Numerical mechanics shared by the two spectroscopy entries."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance

from .._sums import qubit_marginal_axis


def excitation_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return terminal Qubit1/Qubit2 P(e) SEM; counts end in 3 × 3 (g/e/other).

    Other contributes to N. Fewer than two shots and unmeasured points give NaN.
    """
    errors = []
    for target in ("qubit1", "qubit2"):
        marginal = counts.sum(axis=qubit_marginal_axis(target))
        errors.append(np.sqrt(class_probability_covariance(marginal)[..., 1, 1]))
    return np.stack(errors, axis=-1)
