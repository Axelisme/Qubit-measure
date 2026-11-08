"""Joint model-covariance GMM and nested cycle diagnostics."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ._cycle_checks import inner, populations, solve_shared
from ._cycle_coordinates import (
    NUISANCE_NAMES,
    endpoints_of,
    joint_table,
    nuisance_bounds,
    parameter_vector,
)
from ._cycle_emission import simplex
from ._cycle_solver import local_table, solve_joint, starting_nuisance
from .cycle import CycleSums
from .model import Array, CycleModel, CycleNuisances


@dataclass(frozen=True)
class CycleCheck:
    """One nested-model lack-of-fit check.

    J: Weighted residual sum of squares (within-point), or its increase under
        sharing/curve constraints, dimensionless. All comparisons use the same
        last-stage joint-model covariance weights.
    df: Nominal number of restrictions, 45*F for within-point, 14*(F-1) for
        sharing, 6*F-12-number_of_curve_parameters for the curve.
    p: Chi-square survival probability for within-point/curve, or None when df
        is nonpositive. Always None for sharing: its chi-square reference is
        miscalibrated when fits reach boundaries. Calibration needs a model
        parametric bootstrap, which fit_cycle does not run. Passing a check
        does not establish small bias. Other p values are asymptotic references,
        not finite-sample guarantees, particularly near boundaries.
    """

    J: float
    df: int
    p: float | None


@dataclass(frozen=True)
class CycleFit:
    """Cycle fit estimates and diagnostics, without acceptance decisions.

    parameters: Curve parameter estimates by the author's name and units.
    errors: Marginal Wald standard errors by curve name, in parameter units.
    nuisances: Shared endpoint, geometry, thermal and read estimates. State
        labels are arbitrary; comparisons must allow a simultaneous permutation.
    covariance: Joint covariance of curve and nuisance coordinates, shape (P,P),
        Moore-Penrose inverse fixed-weight information (relative cutoff 1e-12),
        without residual-variance rescaling. Rank loss and active bounds invalidate
        ordinary Wald coverage; the pseudoinverse does not restore identification.
    parameter_names: Covariance column names, ordered as on CycleMoments.
        Simplexes use (s,f), not redundant three-probability coordinates.
    within_point: Unrestricted per-point three-state model residual check.
    sharing: Shared-nuisance versus fully pointwise check; no calibrated p.
    curve: Affine author-curve versus free pointwise drive check.
    populations: Shape (F,3), independently fitted first-read pre-emission
        populations from the shared-nuisance fit with free D at each x.
    population_errors: Shape (F,3), marginal Wald errors of populations,
        including the shared-nuisance covariance via the Schur complement.
    reliability: Research R = 1/(2*pi*axis_span*SE(frequency)), dimensionless.
        Frequency uses cycles per axis unit, converted here to angular frequency.
        None unless CycleModel.oscillation_frequency names a curve parameter;
        R applies only to oscillating curves and was calibrated on Rabi. A zero
        axis span or zero SE gives infinity. No threshold or choice is applied.
    J: Joint constrained weighted residual sum of squares, dimensionless.
    df: Joint residual degrees of freedom, 65*F-P.
    """

    parameters: Mapping[str, float]
    errors: Mapping[str, float]
    nuisances: CycleNuisances
    covariance: Array
    parameter_names: tuple[str, ...]
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    populations: Array
    population_errors: Array
    reliability: float | None
    J: float
    df: int


def fit_cycle(
    sums: CycleSums, model: CycleModel, init: Mapping[str, float]
) -> CycleFit:
    """Fit the saved cycle feature means with a shared three-state affine model.

    init must map every curve name to one finite, in-bounds starting value.
    Nuisances start from closed-form moment geometry and fixed physical seeds,
    then fit with curve parameters held at init before the joint fit begins.
    Use bounded, single-start least squares and two model-covariance weight
    stages. Pooled measured products do not supply fit weights. The last-stage
    joint-model weights are held fixed for the three nested fits.
    Independent Gaussian read noise, Markov states, independent thermalised
    cycles and the emission law on CycleNuisances are assumed.

    Return all fields described on CycleFit without changing inputs. Raise
    ValueError for malformed sums/model/curve/init; raise RuntimeError when
    an optimiser exhausts its evaluation budget. A non-positive-definite model
    covariance raises numpy.linalg.LinAlgError. No multistart, model selection,
    bootstrap or profile likelihood is run. Wald errors are local asymptotic
    approximations and do not account for competing minima or active bounds.
    At low state contrast and few cycles per x, Wald intervals can under-cover.
    Estimating the additional affine-row nuisances widens and can skew the
    parameter distribution. More cycles do not guarantee a particular coverage.
    Optimisation keeps read-landing coordinates one machine epsilon inside the
    simplex-coordinate bounds to avoid nonregular deterministic read jumps.
    """
    # Revalidate mutable saved arrays through the record's public contract.
    CycleSums(sums.axis, sums.cycle_count, sums.feature_sums, sums.product_sum)
    theta = parameter_vector(model, init)
    means = sums.feature_sums / sums.cycle_count[:, None]
    q0 = np.r_[theta, starting_nuisance(means)]
    solution = solve_joint(model, sums.axis, means, sums.cycle_count, q0)
    n = len(model.parameter_names)
    q = solution.q
    nuisance = _nuisances(q[n:])
    table = local_table(*joint_table(model, sums.axis, q))
    nuisance_lo, nuisance_hi = nuisance_bounds()
    lo = np.r_[np.zeros(6), nuisance_lo[12:]]
    hi = np.r_[np.ones(6), nuisance_hi[12:]]
    shared = solve_shared(table, means, solution.weights, (lo, hi))
    _, unrestricted_residual = inner(
        shared.table, means, solution.weights, np.arange(20), (lo, hi), steps=60
    )
    within_J = float(np.sum(unrestricted_residual**2))
    points = len(sums.axis)
    values, errors = populations(shared)
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(solution.covariance)))
    reliability = None
    if model.oscillation_frequency is not None:
        index = model.parameter_names.index(model.oscillation_frequency)
        denominator = 2 * np.pi * np.ptp(sums.axis) * standard_errors[index]
        reliability = float(1 / denominator) if denominator else float("inf")
    return CycleFit(
        dict(zip(model.parameter_names, q[:n], strict=True)),
        dict(zip(model.parameter_names, standard_errors[:n], strict=True)),
        nuisance,
        solution.covariance,
        model.parameter_names + NUISANCE_NAMES,
        _check(within_J, 45 * points),
        CycleCheck(max(0.0, shared.J - within_J), 14 * (points - 1), None),
        _check(max(0.0, solution.J - shared.J), 6 * points - 12 - n),
        values,
        errors,
        reliability,
        solution.J,
        65 * points - len(q),
    )


def _check(J: float, df: int) -> CycleCheck:
    return CycleCheck(J, df, float(chi2.sf(J, df)) if df > 0 else None)


def _nuisances(q: Array) -> CycleNuisances:
    return CycleNuisances(
        endpoints_of(q),
        q[12:18].reshape(3, 2).copy(),
        float(q[18]),
        simplex(*q[19:21]),
        q[21:24].copy(),
        simplex(*q[24:26]),
    )
