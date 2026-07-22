from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def least_squares_cost(residuals: NDArray[np.float64]) -> float:
    return float(0.5 * np.sum(np.asarray(residuals, dtype=np.float64) ** 2))


def reduced_chi2_from_cost(
    cost: float,
    observation_count: int | float,
    free_parameter_count: int,
) -> float:
    dof = max(float(observation_count) - float(free_parameter_count), 1.0)
    return float(2.0 * cost / dof)
