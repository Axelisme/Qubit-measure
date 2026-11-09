"""Exact 65-feature moments of cycle models, for tests only.

Production has no use for exact moments: fit_cycle works on measured means and
model-covariance weights. Tests use these to plant noise-free sums and to
compare closed-form Jacobians with numerical differences, through the same
kernels the fit uses.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from zcu_tools.analysis.readout_chain import CycleModel, CycleNuisances, RateNuisances
from zcu_tools.analysis.readout_chain._cycle_coordinates import (
    NUISANCE_NAMES,
    joint_mean_jacobian,
    joint_table,
    pack_nuisance,
    parameter_vector,
)
from zcu_tools.analysis.readout_chain._cycle_engine import covariance
from zcu_tools.analysis.readout_chain._cycle_layout import SHARED
from zcu_tools.analysis.readout_chain._rate_coordinates import (
    RATE_NAMES,
    RateCoordinates,
    dominant_slots,
)

Curve = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]


@dataclass(frozen=True)
class ExactMoments:
    """Per-cycle moments of the 65 numbers at F points.

    means: Shape (F,65). covariance: Shape (F,65,65), per-cycle model
    covariance. jacobian: Shape (F,65,P), analytic derivatives of means by the
    coordinates named in parameter_names. parameter_names: P column names.
    """

    means: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray
    parameter_names: tuple[str, ...]


def curve_moments(
    model: CycleModel,
    axis: np.ndarray,
    parameters: Mapping[str, float],
    nuisances: CycleNuisances,
) -> ExactMoments:
    """Return exact moments of a curve model; ValueError for invalid parameters."""
    q = np.concatenate([parameter_vector(model, parameters), pack_nuisance(nuisances)])
    means, jacobian = joint_mean_jacobian(model, axis, q)
    drive, rows = joint_table(model, axis, q)
    return ExactMoments(
        means,
        covariance(drive, rows, means),
        jacobian,
        model.parameter_names + NUISANCE_NAMES,
    )


def rate_moments(
    axis: np.ndarray, parameters: Mapping[str, float], nuisances: RateNuisances
) -> ExactMoments:
    """Return exact moments of the rate model in its dominant-slot D0 chart."""
    layout = RateCoordinates(axis, dominant_slots(nuisances.drive_at_zero))
    rates = np.array([parameters[name] for name in RATE_NAMES])
    curve_nuisance = CycleNuisances(
        np.repeat(nuisances.drive_at_zero[:, None, :], 2, axis=1),
        nuisances.centers,
        nuisances.variance,
        nuisances.thermal,
        nuisances.read_retention,
        nuisances.read_landing,
    )
    q = np.r_[
        rates,
        layout.pack_d0(nuisances.drive_at_zero),
        pack_nuisance(curve_nuisance)[SHARED],
    ]
    means, jacobian = layout.mean_jacobian(q)
    drive, rows = layout.table(q)
    return ExactMoments(means, covariance(drive, rows, means), jacobian, layout.names)


def repeated_curve(
    curve: Curve,
    parameter_names: tuple[str, ...],
    bounds: Sequence[tuple[float, float]],
    *,
    derivative: Curve,
    oscillation_frequency: str | None = None,
) -> CycleModel:
    """Return a CycleModel that repeats a scalar curve for each state.

    curve receives (axis, parameters) and returns shape (F,); derivative
    returns shape (F, len(parameter_names)). Names, bounds and
    oscillation_frequency are as on CycleModel.
    """

    def repeated(axis: np.ndarray, parameters: Mapping[str, float]) -> np.ndarray:
        return np.repeat(curve(axis, parameters)[:, None], 3, axis=1)

    def repeated_derivative(
        axis: np.ndarray, parameters: Mapping[str, float]
    ) -> np.ndarray:
        return np.repeat(derivative(axis, parameters)[:, None, :], 3, axis=1)

    return CycleModel(
        repeated,
        parameter_names,
        bounds,
        oscillation_frequency,
        derivative=repeated_derivative,
    )
