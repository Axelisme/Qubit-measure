"""Three-state affine cycle model and exact-moment interface."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._cycle_coordinates import (
    NUISANCE_NAMES,
    joint_mean_jacobian,
    joint_table,
    pack_nuisance,
    parameter_vector,
)
from ._cycle_engine import covariance

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
    variance. Fields are frozen; their arrays remain mutable, so moments checks
    them again. Exactly deterministic off-diagonal read jumps are nonregular:
    moments raises ValueError because their emission Jacobian is not finite.
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
class CycleMoments:
    """Moments of the saved 65 features, in axis order.

    means: Shape (F,65), per-cycle means, in ADC or ADC-squared feature units.
    covariance: Shape (F,65,65), per-cycle model covariance, in product feature units.
    jacobian: Shape (F,65,P), derivatives of means by the named coordinates.
    parameter_names: P column names. Curve coordinates come first. Simplexes use
        (s,f): probabilities (s*f, s*(1-f), 1-s). Endpoint coordinates follow in
        state, endpoint, then s/f order, followed by center I/Q in state order,
        variance, thermal s/f, three retentions, landing s/f. Other coordinates
        use the units on CycleNuisances; s,f and retentions are dimensionless.
    """

    means: Array
    covariance: Array
    jacobian: Array
    parameter_names: tuple[str, ...]


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

    The drive row of state i is (1 - f_i(x)) * endpoint 0 + f_i(x) * endpoint 1,
    as on CycleNuisances. Jacobian columns of the curve parameters use bounded
    central differences of curve, because curve is a caller callable; all other
    columns are analytic. Use identical to repeat one scalar curve across the
    three states.
    """

    curve: Curve
    parameter_names: tuple[str, ...]
    bounds: Sequence[tuple[float, float]]
    oscillation_frequency: str | None = None

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

    @classmethod
    def identical(
        cls,
        curve: Curve,
        parameter_names: tuple[str, ...],
        bounds: Sequence[tuple[float, float]],
        *,
        oscillation_frequency: str | None = None,
    ) -> "CycleModel":
        """Repeat a scalar curve returning shape (F,) for each state.

        curve receives the same axis and parameter mapping as CycleModel.curve.
        Names, bounds and oscillation_frequency have the meanings on CycleModel.
        """

        def repeated(axis: Array, parameters: Mapping[str, float]) -> Array:
            return np.repeat(curve(axis, parameters)[:, None], 3, axis=1)

        return cls(repeated, parameter_names, bounds, oscillation_frequency)

    def moments(
        self, axis: Array, parameters: Mapping[str, float], nuisances: CycleNuisances
    ) -> CycleMoments:
        """Return closed-form cycle means, covariance and mean Jacobian.

        axis is a finite nonempty (F,) array; parameters contains exactly all
        named curve values within bounds. Nuisances are shared over axis.
        The matrix-chain, noise and transition Jacobians are analytic. Only
        derivatives of the author's callable curve use bounded central differences.
        Raise ValueError for invalid inputs or a nonregular deterministic read
        jump. Inputs are not changed.
        """
        _validate_axis(axis)
        _validate_nuisances(nuisances)
        theta = parameter_vector(self, parameters)
        q = np.concatenate([theta, pack_nuisance(nuisances)])
        means, jacobian = joint_mean_jacobian(self, axis, q)
        drive, rows = joint_table(self, axis, q)
        return CycleMoments(
            means,
            covariance(drive, rows, means),
            jacobian,
            self.parameter_names + NUISANCE_NAMES,
        )

    def populations(
        self, axis: Array, parameters: Mapping[str, float], nuisances: CycleNuisances
    ) -> Array:
        """Return the first-read populations before emission, shape (F,3).

        The thermal populations pass one drive D(x), whose row for state i is
        (1 - f_i(x)) * drive_endpoints[i,0] + f_i(x) * drive_endpoints[i,1].
        axis, parameters and nuisances are as for moments; for a fit, pass
        CycleFit.parameters and CycleFit.nuisances. Columns follow the state
        order of nuisances. Raise ValueError for invalid inputs. Inputs are not
        changed.
        """
        _validate_axis(axis)
        _validate_nuisances(nuisances)
        theta = parameter_vector(self, parameters)
        q = np.concatenate([theta, pack_nuisance(nuisances)])
        drive, _ = joint_table(self, axis, q)
        return np.einsum("i,xij->xj", nuisances.thermal, drive)
