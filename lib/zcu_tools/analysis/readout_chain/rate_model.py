"""Rate-equation T1 model and its physical nuisance/result contracts."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._cycle_coordinates import pack_nuisance
from ._cycle_engine import mean_jacobian
from ._cycle_layout import SHARED
from ._rate_coordinates import (
    RATE_MAX,
    RATE_NAMES,
    RateCoordinates,
    dominant_slots,
)
from .model import Array, CycleNuisances


@dataclass(frozen=True)
class RateNuisances:
    """Shared physical quantities for D(x) = D0 exp(Gx).

    drive_at_zero: Stochastic D0, shape (3,3), dimensionless. It acts before
        relaxation, on population row vectors multiplied from the left.
    centers: Cloud I/Q centers, shape (3,2), in ADC units.
    variance: Common isotropic Gaussian noise variance, positive, ADC units².
    thermal: Cycle-start and stationary probabilities, shape (3,), strictly
        positive. The fit bounds thermal simplex coordinates to [1e-6,1-1e-6]
        because the symmetric spectral formula divides by sqrt(thermal).
    read_retention/read_landing: Shape (3,), dimensionless; same read/emission
        law as CycleNuisances. Labels are geometric states, not g/e/f.

    Construction and model evaluation reject malformed or nonfinite arrays,
    nonstochastic D0, nonpositive variance/thermal and invalid read probabilities.
    Arrays remain mutable although the dataclass is frozen.
    """

    drive_at_zero: Array
    centers: Array
    variance: float
    thermal: Array
    read_retention: Array
    read_landing: Array

    def __post_init__(self) -> None:
        # Building the CycleNuisances validates every field; only its raising matters here.
        _as_cycle_nuisances(self)


def _as_cycle_nuisances(nuisances: RateNuisances) -> CycleNuisances:
    if nuisances.drive_at_zero.shape != (3, 3):
        raise ValueError("drive_at_zero must have shape (3,3)")
    result = CycleNuisances(
        np.repeat(nuisances.drive_at_zero[:, None, :], 2, axis=1),
        nuisances.centers,
        nuisances.variance,
        nuisances.thermal,
        nuisances.read_retention,
        nuisances.read_landing,
    )
    if np.any(nuisances.thermal <= 0):
        raise ValueError("rate model thermal probabilities must be positive")
    return result


@dataclass(frozen=True)
class CycleRelaxation:
    """Two eigen times and the larger-amplitude T1, in axis time units.

    eigen_times/eigen_time_errors: Shape (2,), slow then fast times and marginal
        Wald errors. Zero decay rates give infinite times/errors. Errors use
        analytic Hellmann–Feynman gradients and the joint generalized covariance.
        An error is NaN when that eigen time involves an unconstrained direction;
        unrelated nuisance rank loss does not remove errors for estimable times.
    amplitudes: Shape (2,), thermal-weighted sum over rows of the L1 population
        change carried by each exponential: A_k = sum_i pi_i ||(D0 P_k)_i||_1,
        where P_k is the generator's spectral projector. Dimensionless.
    t1/t1_error: Eigen time with the larger amplitude and its Wald error. A tie
        selects the slow time. There is no amplitude pre-test or fallback.
    secondary_time/secondary_error: Other eigen time and its Wald error.

    Near repeated eigenvalues or equal amplitudes, the selected mode is not a
    regular parameter. These local Wald errors do not include selection
    uncertainty, repair nonidentification, or guarantee finite-sample coverage.
    """

    eigen_times: Array
    eigen_time_errors: Array
    amplitudes: Array
    t1: float
    t1_error: float
    secondary_time: float
    secondary_error: float


@dataclass(frozen=True)
class RateModel:
    """Detailed-balance T1 family with three nonnegative pair rates.

    D(x) = D0 exp(Gx), G[i,j] = rate_ij*pi[j]/(pi[i]+pi[j]), with row sums zero.
    D0 is free stochastic and pi is the fitted thermal distribution. Parameters
    are exactly rate_01, rate_02, rate_12, in inverse axis time units, bounded
    [0,1000]. Indices name geometric states and have no g/e/f meaning. Delays
    must be finite and nonnegative. fit_cycle takes init={"t1": time}, not
    these three rates: it seeds all three at 1/time.
    """

    def populations(
        self, axis: Array, parameters: Mapping[str, float], nuisances: RateNuisances
    ) -> Array:
        """Return first-read pre-emission populations (F,3), pi D0 exp(Gx).

        axis holds finite nonnegative delays; parameters holds exactly rate_01,
        rate_02 and rate_12 in [0,1000]. Invalid inputs raise ValueError. Inputs
        are unchanged.
        """
        coordinates, q = self._coordinates(axis, parameters, nuisances)
        return np.einsum("i,xij->xj", nuisances.thermal, coordinates.table(q)[0])

    def feature_means(
        self, axis: Array, parameters: Mapping[str, float], nuisances: RateNuisances
    ) -> Array:
        """Predict per-cycle feature means on nonnegative delays without fitting.

        Columns follow CycleSums in feature units, shape (F,65). Axis, rates
        and nuisances have the same contract as for populations. Invalid inputs
        raise ValueError; inputs are unchanged.
        """
        coordinates, q = self._coordinates(axis, parameters, nuisances)
        # Reuse the fit's analytic moment kernel; the plot grid only needs means.
        means, _ = mean_jacobian(*coordinates.table(q))
        return means

    def _coordinates(
        self, axis: Array, parameters: Mapping[str, float], nuisances: RateNuisances
    ) -> tuple[RateCoordinates, Array]:
        validate_delays(axis)
        curve_nuisance = _as_cycle_nuisances(nuisances)
        if set(parameters) != set(RATE_NAMES):
            raise ValueError("parameters must name rate_01, rate_02 and rate_12")
        rates = np.array([parameters[name] for name in RATE_NAMES])
        if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > RATE_MAX)):
            raise ValueError("pair rates must be finite and in [0,1000]")
        layout = RateCoordinates(axis, dominant_slots(nuisances.drive_at_zero))
        q = np.r_[
            rates,
            layout.pack_d0(nuisances.drive_at_zero),
            pack_nuisance(curve_nuisance)[SHARED],
        ]
        return layout, q


def validate_delays(axis: Array) -> None:
    """Reject a non-vector, empty, nonfinite or negative delay axis."""
    if (
        axis.ndim != 1
        or not axis.size
        or not np.isfinite(axis).all()
        or np.any(axis < 0)
    ):
        raise ValueError("delay axis must be finite, nonempty and nonnegative")
