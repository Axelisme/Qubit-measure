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
    parameter_vector,
    table_bounds,
)
from ._cycle_emission import simplex
from ._cycle_layout import (
    CENTERS,
    LANDING,
    RETENTION,
    TABLE_SIZE,
    THERMAL,
    VARIANCE,
)
from ._cycle_solver import local_table, solve_joint, starting_nuisance
from ._rate_solver import solve_rate
from .cycle import CycleSums, validate_cycle_sums
from .model import Array, CycleModel, CycleNuisances
from .rate_model import CycleRelaxation, RateModel, RateNuisances


@dataclass(frozen=True)
class CycleCheck:
    """One nested-model lack-of-fit check.

    J: Weighted residual sum of squares (within-point), or its increase under
        sharing/curve constraints, dimensionless. All comparisons use the same
        last-stage joint-model covariance weights. NaN for an unconverged
        within-point fit and the sharing comparison that depends on it.
    df: Nominal number of restrictions, 45*F for within-point, 14*(F-1) for
        sharing, 6*F-12-number_of_curve_parameters for the curve.
    p: Chi-square survival probability for within-point/curve, or None when df
        is nonpositive. Always None for sharing: its chi-square reference is
        miscalibrated when fits reach boundaries. Calibration needs a model
        parametric bootstrap, which fit_cycle does not run. Passing a check
        does not establish small bias. Other p values are asymptotic references,
        not finite-sample guarantees, particularly near boundaries. Also None
        when within-point did not converge; its J and the dependent sharing J
        are NaN in that case, unlike the finite-J cases without a calibrated p.
    """

    J: float
    df: int
    p: float | None


@dataclass(frozen=True)
class CycleFit:
    """Cycle fit estimates and diagnostics, without acceptance decisions.

    parameters: Curve parameter estimates by the author's name and units, or
        rate_01/rate_02/rate_12 in inverse time units for RateModel. Rate indices
        name geometric states, with no g/e/f meaning.
    errors: Marginal Wald standard errors by curve/rate name, in parameter units.
        NaN means the data do not constrain that parameter locally.
    relaxation: Rate-model eigen times and T1; None for curve models.
    nuisances: Shared endpoint/D0, geometry, thermal and read estimates. State
        labels are arbitrary; comparisons must allow a simultaneous permutation.
    covariance: Joint covariance of curve and nuisance coordinates, shape (P,P),
        column-normalized generalized inverse of fixed-weight information (relative
        cutoff 1e-12), transformed back to parameter units without residual-variance
        rescaling. Curve, rate and nested fits normalize Jacobian columns so the
        cutoff describes collinearity rather than units. Rank loss and active
        bounds invalidate ordinary Wald coverage; the generalized inverse does not
        restore identification. Its null-space zeros are not standard errors.
        Parameter and derived errors involving that null space are NaN; other
        estimable quantities retain their errors.
    parameter_names: Covariance column names. Curve or rate parameters come
        first. Curve models continue with the drive endpoints (state, endpoint,
        s/f); rate models with the D0 rows (s,f), whose names encode each row's
        slot order. Both end with center I/Q by state, variance, thermal s/f,
        three retentions and landing s/f. Simplexes use (s,f), not redundant
        three-probability coordinates; probabilities are (s*f, s*(1-f), 1-s).
    within_point: Unrestricted per-point three-state model residual check.
        If any point fails to converge, J is NaN, p is None, and df is unchanged.
        Other fitted quantities and the independent curve check remain available.
    sharing: Shared-nuisance versus fully pointwise check; no calibrated p.
        J is NaN when within_point is unavailable, otherwise finite. Thus p=None
        can mean df<=0, uncalibrated sharing, or an unavailable within-point fit;
        the last case is distinguished by NaN J, not unidentifiable Wald errors.
    curve: Constrained model versus free pointwise drive check. RateModel
        has 6*F-9 restrictions; a curve model has 6*F-12-number_of_parameters.
    populations: Shape (F,3), independently fitted first-read pre-emission
        populations from the shared-nuisance fit with free D at each x.
    population_errors: Shape (F,3), marginal Wald errors of populations,
        including the shared-nuisance covariance via the Schur complement. NaN
        means that population involves an unconstrained local or shared direction.
        An unrelated unidentifiable nuisance does not make all errors NaN.
    reliability: R = 1 / (2*pi * axis span * SE(frequency)), dimensionless.
        frequency is the CycleModel.oscillation_frequency parameter in cycles
        per axis unit and SE its Wald standard error, so 2*pi*SE is the
        angular-frequency error; a larger R means a smaller frequency error
        relative to the sweep span. Its scale was calibrated on Rabi sweeps
        only. None when the model has no oscillation_frequency or the named
        frequency is unidentifiable. Other nuisance rank loss does not remove R.
        An estimable frequency with zero axis span or zero SE gives infinity.
        No threshold or choice is applied.
    J: Joint constrained weighted residual sum of squares, dimensionless.
    df: Joint residual degrees of freedom, 65*F-P.
    """

    parameters: Mapping[str, float]
    errors: Mapping[str, float]
    nuisances: CycleNuisances | RateNuisances
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
    relaxation: CycleRelaxation | None


def fit_cycle(
    sums: CycleSums, model: CycleModel | RateModel, init: Mapping[str, float]
) -> CycleFit:
    """Fit saved cycle feature means with a three-state curve or rate model.

    For CycleModel, init maps every curve name to a finite in-bounds starting
    value. For RateModel, init is exactly {"t1": textbook_time}, a finite time
    >= 1e-3 in axis units; all three pair rates start at its inverse. No
    textbook fit is run internally. Nuisances start from closed-form moment
    geometry and fixed physical seeds. Every Jacobian column is analytic.
    Fit nuisances at init for each of the six assignments of geometric states
    to curve rows, then run the joint fit from the lowest-objective assignment.
    Assignment scores use common initial model-covariance weights and feasible
    last iterates if a nuisance-only fit reaches its evaluation budget.
    Use bounded least squares and two joint model-covariance weight stages.
    Pooled measured products do not supply fit weights. The last-stage joint-model
    weights are held fixed for the three nested fits.
    Independent Gaussian read noise, Markov states, independent thermalised
    cycles and the emission law on CycleNuisances are assumed.

    Return all fields described on CycleFit without changing inputs. Raise
    ValueError for malformed sums/model/curve/init; raise RuntimeError when
    the joint or shared optimiser exhausts its evaluation budget, or the
    sharing/curve inner fit fails optimality. An
    unconverged within-point fit instead leaves its J and sharing J as NaN,
    with p=None and unchanged df; parameters, errors, reliability, populations
    and the independent curve check still return. A negative nested J
    increase beyond floating-point summation roundoff also raises RuntimeError;
    only increases within that tolerance are rounded to zero. A non-positive-definite
    model covariance raises numpy.linalg.LinAlgError. No multistart, model selection,
    bootstrap or profile likelihood is run. Wald errors are local asymptotic
    approximations and do not account for competing minima or active bounds.
    At low state contrast and few cycles per x, Wald intervals can under-cover.
    Estimating the additional affine-row nuisances widens and can skew the
    parameter distribution. More cycles do not guarantee a particular coverage.
    Optimisation keeps read-landing coordinates one machine epsilon inside the
    simplex-coordinate bounds to avoid nonregular deterministic read jumps.
    """
    validate_cycle_sums(sums)
    means = sums.feature_sums / sums.cycle_count[:, None]
    relaxation = None
    nuisance: CycleNuisances | RateNuisances
    if isinstance(model, RateModel):
        fitted = solve_rate(sums.axis, means, sums.cycle_count, init)
        solution = fitted.joint
        nuisance, relaxation = fitted.nuisances, fitted.relaxation
        names = fitted.coordinates.names
        n, drive_parameters = 3, 9
        table = local_table(*fitted.coordinates.table(solution.q))
    else:
        theta = parameter_vector(model, init)
        q0 = np.r_[theta, starting_nuisance(means)]
        solution = solve_joint(model, sums.axis, means, sums.cycle_count, q0)
        n = len(model.parameter_names)
        nuisance = _nuisances(solution.q[n:])
        names = model.parameter_names + NUISANCE_NAMES
        drive_parameters = 12 + n
        table = local_table(*joint_table(model, sums.axis, solution.q))
    q = solution.q
    limits = table_bounds()
    shared = solve_shared(table, means, solution.weights, limits)
    unrestricted = inner(
        shared.table, means, solution.weights, np.arange(TABLE_SIZE), limits
    )
    points = len(sums.axis)
    if unrestricted.converged:
        within = _check(float(np.sum(unrestricted.residual**2)), 45 * points)
        sharing_J = _increase(shared.J, within.J, points, "sharing")
    else:
        # A failed point leaves partial residuals: neither their sum nor the
        # dependent J difference is a valid check. Do not send NaN to _increase.
        within = CycleCheck(float("nan"), 45 * points, None)
        sharing_J = float("nan")
    values, errors = populations(shared)
    standard_errors = solution.information.errors(np.eye(len(q)))
    reliability = None
    if isinstance(model, CycleModel) and model.oscillation_frequency is not None:
        index = model.parameter_names.index(model.oscillation_frequency)
        denominator = 2 * np.pi * np.ptp(sums.axis) * standard_errors[index]
        if not np.isnan(standard_errors[index]):
            reliability = float(1 / denominator) if denominator else float("inf")
    return CycleFit(
        dict(zip(names[:n], q[:n], strict=True)),
        dict(zip(names[:n], standard_errors[:n], strict=True)),
        nuisance,
        solution.covariance,
        names,
        within,
        CycleCheck(sharing_J, 14 * (points - 1), None),
        _check(
            _increase(solution.J, shared.J, points, "curve"),
            6 * points - drive_parameters,
        ),
        values,
        errors,
        reliability,
        solution.J,
        65 * points - len(q),
        relaxation,
    )


def _increase(constrained: float, free: float, points: int, check: str) -> float:
    increase = constrained - free
    # Each J sums 65*F squared residuals. Allow first-order roundoff for the
    # products, sums and subtraction: (65*F+2)*eps times the combined objective
    # scale, with a unit absolute floor near zero. This is not an optimisation
    # tolerance: a genuinely worse free fit must not produce a passing p.
    tolerance = (
        (65 * points + 2) * np.finfo(float).eps * max(1.0, abs(constrained) + abs(free))
    )
    if increase < -tolerance:
        raise RuntimeError(
            f"{check} cycle check has negative J increase {increase:g} "
            f"beyond roundoff tolerance {tolerance:g}"
        )
    return max(0.0, increase)


def _check(J: float, df: int) -> CycleCheck:
    return CycleCheck(J, df, float(chi2.sf(J, df)) if df > 0 else None)


def _nuisances(nuisance: Array) -> CycleNuisances:
    return CycleNuisances(
        drive_endpoints=endpoints_of(nuisance),
        centers=nuisance[CENTERS].reshape(3, 2).copy(),
        variance=float(nuisance[VARIANCE]),
        thermal=simplex(*nuisance[THERMAL]),
        read_retention=nuisance[RETENTION].copy(),
        read_landing=simplex(*nuisance[LANDING]),
    )
