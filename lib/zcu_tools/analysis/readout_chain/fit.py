"""Joint model-covariance GMM for saved K-cycle features."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._cycle_coordinates import NUISANCE_NAMES, endpoints_of, parameter_vector
from ._cycle_emission import simplex
from ._cycle_layout import CENTERS, LANDING, RETENTION, THERMAL, VARIANCE
from ._cycle_solver import solve_joint, starting_nuisance
from ._rate_solver import solve_rate
from .cycle import CYCLE_FEATURE_COUNT, CycleSums, validate_cycle_sums
from .model import Array, CycleModel, CycleNuisances
from .rate_model import CycleRelaxation, RateModel, RateNuisances


@dataclass(frozen=True)
class CycleFit:
    """Joint cycle fit estimates and residual numbers, without acceptance decisions.

    parameters: Curve estimates by the author's name and units, or
        rate_01/rate_02/rate_12 in inverse time units for RateModel. Rate indices
        name geometric states, with no g/e/f meaning.
    errors: Marginal Wald standard errors by curve/rate name, in parameter units.
        NaN means the data do not constrain that parameter locally.
    nuisances: Shared endpoint/D0, geometry, thermal and read estimates. State
        labels are arbitrary; comparisons must allow a simultaneous permutation.
    covariance: Joint covariance of curve/rate and nuisance coordinates, shape
        (P,P). A column-normalized generalized inverse of fixed-weight information
        (relative cutoff 1e-12) is transformed back to parameter units without
        residual-variance rescaling. Column normalization makes the cutoff
        describe collinearity rather than units. Rank loss and active bounds
        invalidate ordinary Wald coverage; the generalized inverse does not
        restore identification. Parameter and derived errors involving its null
        space are NaN; other estimable quantities retain their errors.
    parameter_names: Covariance column names. Curve or rate parameters come
        first. Curve models continue with the drive endpoints (state, endpoint,
        s/f); rate models with the D0 rows (s/f), whose names encode each row's
        slot order. Both end with center I/Q by state, variance, thermal s/f,
        three retentions and landing s/f. Simplexes use (s, f), not redundant
        three-probability coordinates; probabilities are (s*f, s*(1-f), 1-s).
    reliability: R = 1 / (2*pi * axis span * SE(frequency)), dimensionless.
        frequency is the CycleModel.oscillation_frequency parameter in cycles
        per axis unit, and SE is its Wald standard error. Its scale was
        calibrated on Rabi sweeps only. None when there is no named frequency
        or it is unidentifiable. An estimable frequency with zero axis span or
        zero SE gives infinity. No threshold or choice is applied.
    J: Sum of squares of W(measured - predicted), dimensionless. W is the
        fixed model covariance whitening of the last joint stage.
    df: Nominal joint residual degrees of freedom, 65*F-P for F sweep points.
    j_per_df: J / df, or NaN if df is nonpositive.
    expected_band: Approximate chi-square reference endpoints 1 ± 2√(2/df),
        not acceptance limits. Both are NaN if df is nonpositive.
    tail_fraction: Fraction of all 65*F final whitened components with |r| > 3.
    predicted_means: Joint model at the acquired points, shape (F,65), in
        CycleSums feature units and order. Use for plots, not analysis summaries.
    relaxation: Rate-model eigen times and T1; None for curve models.
    """

    parameters: Mapping[str, float]
    errors: Mapping[str, float]
    nuisances: CycleNuisances | RateNuisances
    covariance: Array
    parameter_names: tuple[str, ...]
    reliability: float | None
    J: float
    df: int
    j_per_df: float
    expected_band: tuple[float, float]
    tail_fraction: float
    predicted_means: Array
    relaxation: CycleRelaxation | None


def fit_cycle(
    sums: CycleSums, model: CycleModel | RateModel, init: Mapping[str, float]
) -> CycleFit:
    """Fit saved cycle feature means with a three-state curve or rate model.

    For CycleModel, init maps every curve name to a finite in-bounds start.
    For RateModel, init is exactly {"t1": textbook_time}, a finite time >= 1e-3
    in axis units; all three pair rates start at its inverse. No textbook fit
    runs internally. Nuisances start from moment geometry and physical seeds.
    Optimize them at init for the six assignments of geometric states to curve
    rows, using common weights to compare even budget-limited last iterates.
    Run two bounded joint model-covariance weight stages from the best
    assignment, each capped at 1,000 residual evaluations. Every Jacobian
    column is analytic. Pooled measured products do not supply fit weights.
    There are no secondary fits or fit-quality thresholds.

    Return CycleFit without changing inputs. Raise ValueError for malformed
    sums/model/curve/init, RuntimeError if a joint optimizer stage does not
    converge, and numpy.linalg.LinAlgError for non-positive-definite model
    covariance. The measured covariance used by cycle_residual_projection
    is not needed by this fit.

    The model assumes independent Gaussian read noise, Markov states,
    independent thermalized cycles and the emission law on CycleNuisances.
    There is no search across curve-parameter starts, model selection,
    bootstrap or profile likelihood. Wald errors are local asymptotic
    approximations; competing minima, active bounds, low contrast and few
    cycles can invalidate coverage. Optimization keeps read-landing coordinates
    one machine epsilon inside bounds to avoid nonregular deterministic jumps.
    """
    validate_cycle_sums(sums)
    means = sums.feature_sums / sums.cycle_count[:, None]
    relaxation = None
    nuisance: CycleNuisances | RateNuisances
    if isinstance(model, RateModel):
        fitted = solve_rate(sums.axis, means, sums.cycle_count, init)
        solution = fitted.joint
        nuisance, relaxation = fitted.nuisances, fitted.relaxation
        names = fitted.parameter_names
        n = 3
    else:
        theta = parameter_vector(model, init)
        q0 = np.r_[theta, starting_nuisance(means)]
        solution = solve_joint(model, sums.axis, means, sums.cycle_count, q0)
        n = len(model.parameter_names)
        nuisance = _nuisances(solution.q[n:])
        names = model.parameter_names + NUISANCE_NAMES
    q = solution.q
    standard_errors = solution.information.errors(np.eye(len(q)))
    reliability = None
    if isinstance(model, CycleModel) and model.oscillation_frequency is not None:
        index = model.parameter_names.index(model.oscillation_frequency)
        denominator = 2 * np.pi * np.ptp(sums.axis) * standard_errors[index]
        if not np.isnan(standard_errors[index]):
            reliability = float(1 / denominator) if denominator else float("inf")
    J = float(np.sum(solution.residual**2))
    df = CYCLE_FEATURE_COUNT * len(sums.axis) - len(q)
    half_width = 2 * (2 / df) ** 0.5 if df > 0 else float("nan")
    return CycleFit(
        parameters=dict(zip(names[:n], map(float, q[:n]), strict=True)),
        errors=dict(zip(names[:n], map(float, standard_errors[:n]), strict=True)),
        nuisances=nuisance,
        covariance=solution.information.covariance,
        parameter_names=names,
        reliability=reliability,
        J=J,
        df=df,
        j_per_df=J / df if df > 0 else float("nan"),
        expected_band=(1 - half_width, 1 + half_width),
        tail_fraction=float(np.mean(np.abs(solution.residual) > 3)),
        predicted_means=solution.predicted_means,
        relaxation=relaxation,
    )


def _nuisances(nuisance: Array) -> CycleNuisances:
    return CycleNuisances(
        drive_endpoints=endpoints_of(nuisance),
        centers=nuisance[CENTERS].reshape(3, 2).copy(),
        variance=float(nuisance[VARIANCE]),
        thermal=simplex(*nuisance[THERMAL]),
        read_retention=nuisance[RETENTION].copy(),
        read_landing=simplex(*nuisance[LANDING]),
    )
