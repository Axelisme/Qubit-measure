"""Three-state affine cycle model and exact-moment interface."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ._cycle_coordinates import (
    NUISANCE_NAMES,
    joint_table,
    pack_nuisance,
    parameter_vector,
)
from ._cycle_engine import mean_jacobian

Array = NDArray[np.float64]


def _validate_axis(axis: Array) -> None:
    if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all():
        raise ValueError("axis must be finite, nonempty and shape (points,)")


def _validate_nuisances(nuisances: "CycleNuisances") -> None:
    _require_finite("drive_endpoints", nuisances.drive_endpoints, (3, 2, 3))
    _require_probabilities("drive_endpoints", nuisances.drive_endpoints)
    _require_finite("centers", nuisances.centers, (3, 2))
    if not np.isfinite(nuisances.variance) or nuisances.variance <= 0:
        raise ValueError("nuisance variance must be finite and positive")
    _require_finite("thermal", nuisances.thermal, (3,))
    _require_probabilities("thermal", nuisances.thermal)
    _require_finite("read_retention", nuisances.read_retention, (3,))
    _require_unit_interval("read_retention", nuisances.read_retention)
    _require_finite("read_landing", nuisances.read_landing, (3,))
    _require_probabilities("read_landing", nuisances.read_landing)


def _require_finite(name: str, value: Array, shape: tuple[int, ...]) -> None:
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"nuisance {name} must be finite with shape {shape}")


def _require_unit_interval(name: str, value: Array) -> None:
    if np.any((value < 0) | (value > 1)):
        raise ValueError(f"nuisance {name} must be in [0, 1]")


def _require_probabilities(name: str, value: Array) -> None:
    """Each last-axis row is in [0, 1] and sums to one."""
    _require_unit_interval(name, value)
    if not np.allclose(value.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError(f"nuisance {name} must sum to one")


Curve = Callable[[Array, Mapping[str, float]], Array]


@dataclass(frozen=True)
class CycleNuisances:
    """Shared physical quantities of a three-state cycle.

    drive_endpoints: Shape (3,2,3), dimensionless row probabilities. For state i,
        D[i,:] = (1-f_i) * drive_endpoints[i,0,:] + f_i * drive_endpoints[i,1,:].
        Each endpoint is a simplex, allowing both branching fractions to vary.
    centers: Shape (3,2), cloud I,Q centers in ADC units, in the same state order.
    variance: Common isotropic Gaussian read noise variance in ADC units squared, >0.
    thermal: Shape (3,), initial thermal simplex before the first D, dimensionless.
    read_retention: Shape (3,), retention coordinates in [0,1]. The read matrix is
        diag(read_retention) + outer(1-read_retention, read_landing).
    read_landing: Shape (3,), landing simplex for read transitions, dimensionless.
        On an i->j transition, IQ has mean c_j + u*(c_i-c_j), where u on [0,1]
        follows a truncated exponential with rate -log(1-R[i,j]). Noise is
        independent between reads and cycles. No transition gives center c_i.

    State labels are arbitrary and may be permuted together. Construction
    raises ValueError for a wrong shape, a non-finite value, a probability
    outside [0, 1], a simplex that does not sum to one or a nonpositive
    variance. Fields are frozen; their arrays remain mutable, so readers check
    them again. Exactly deterministic off-diagonal read jumps are nonregular:
    the fit rejects them because their emission Jacobian is not finite.
    """

    drive_endpoints: Array
    centers: Array
    variance: float
    thermal: Array
    read_retention: Array
    read_landing: Array

    def __post_init__(self) -> None:
        _validate_nuisances(self)


@dataclass(frozen=True)
class CycleModel:
    """Author-supplied per-state curve for K(DRR)^8.

    curve: Callable (axis, parameter mapping) returning finite shape (F,3) in [0,1].
        Axis units are caller-defined. The curve does not include readout nuisances.
    parameter_names: Unique nonempty curve parameter names, in Jacobian order.
    bounds: One (lower,upper) pair per curve parameter, strictly increasing.
        Infinite bounds are allowed; parameter values must be finite.
    oscillation_frequency: Optional curve parameter name for an oscillation
        frequency in cycles per axis unit, used only for the R diagnostic.
        Raise ValueError at construction if it is not in parameter_names.

    derivative: Analytic derivative callable (axis, parameters) returning finite
        shape (F,3,len(parameter_names)), in the same parameter order.

    The drive row of state i is (1 - f_i(x)) * endpoint 0 + f_i(x) * endpoint 1,
    as on CycleNuisances. All Jacobian columns are analytic.
    """

    curve: Curve
    parameter_names: tuple[str, ...]
    bounds: Sequence[tuple[float, float]]
    oscillation_frequency: str | None = None
    derivative: Curve = field(kw_only=True)

    def __post_init__(self) -> None:
        names = self.parameter_names
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError("curve parameter names must be nonempty and unique")
        if set(names).intersection(NUISANCE_NAMES):
            raise ValueError("curve parameter names must not duplicate nuisance names")
        bounds = np.asarray(self.bounds, dtype=float)
        if (
            bounds.shape != (len(names), 2)
            or np.isnan(bounds).any()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError(
                "bounds must contain one increasing (lower, upper) pair per parameter"
            )
        if (
            self.oscillation_frequency is not None
            and self.oscillation_frequency not in names
        ):
            raise ValueError("oscillation_frequency must name a curve parameter")

    def populations(
        self, axis: Array, parameters: Mapping[str, float], nuisances: CycleNuisances
    ) -> Array:
        """Return the first-read populations before emission, shape (F,3).

        The thermal populations pass one drive D(x), whose row for state i is
        (1 - f_i(x)) * drive_endpoints[i,0] + f_i(x) * drive_endpoints[i,1].
        axis is a finite nonempty (F,) array; parameters contains exactly all
        named curve values within bounds; nuisances are shared over axis. For a
        fit, pass CycleFit.parameters and CycleFit.nuisances. Columns follow the
        state order of nuisances. Raise ValueError for invalid inputs. Inputs
        are not changed.
        """
        drive, _ = self._table(axis, parameters, nuisances)
        return np.einsum("i,xij->xj", nuisances.thermal, drive)

    def feature_means(
        self, axis: Array, parameters: Mapping[str, float], nuisances: CycleNuisances
    ) -> Array:
        """Predict per-cycle means on a finite nonempty axis with no additional fit.

        Columns follow CycleSums in feature units, shape (F,65). Parameters
        and nuisances have the same contract as for populations. Invalid inputs
        raise ValueError; inputs are unchanged.
        """
        means, _ = mean_jacobian(*self._table(axis, parameters, nuisances))
        return means

    def _table(
        self, axis: Array, parameters: Mapping[str, float], nuisances: CycleNuisances
    ) -> tuple[Array, Array]:
        _validate_axis(axis)
        _validate_nuisances(nuisances)
        theta = parameter_vector(self, parameters)
        q = np.concatenate([theta, pack_nuisance(nuisances)])
        return joint_table(self, axis, q)
