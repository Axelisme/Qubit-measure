"""Bounded rate-model GMM stages and propagation of eigen-time uncertainty."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from ._cycle_coordinates import endpoints_of
from ._cycle_emission import simplex
from ._cycle_layout import (
    ROW_CENTERS,
    ROW_LANDING,
    ROW_RETENTION,
    ROW_THERMAL,
    ROW_VARIANCE,
    SHARED,
)
from ._cycle_solver import (
    JointSolution,
    ScaledInformation,
    apply_weights,
    assigned_seed,
    bounded_least_squares,
    model_weights,
    scaled_information,
    starting_nuisance,
)
from ._rate_coordinates import (
    RATE_MAX,
    SHARED_RATE,
    SIZE,
    RateCoordinates,
    dominant_slots,
)
from ._rate_equation import mode_amplitude
from .model import Array
from .rate_model import CycleRelaxation, RateNuisances, validate_delays


@dataclass(frozen=True)
class RateSolution:
    """Joint solution with covariance column names and physical estimates."""

    joint: JointSolution
    parameter_names: tuple[str, ...]
    nuisances: RateNuisances
    relaxation: CycleRelaxation


def solve_rate(
    axis: Array, means: Array, counts: Array, init: Mapping[str, float]
) -> RateSolution:
    """Seed at inverse init T1, fit six nuisance assignments, then two joint stages."""
    validate_delays(axis)
    if set(init) != {"t1"} or not np.isfinite(init["t1"]) or init["t1"] < 1 / RATE_MAX:
        raise ValueError("rate init must contain exactly finite t1 >= 1e-3 axis units")
    rate = 1 / init["t1"]
    nuisance = starting_nuisance(means)
    layout, q = _seed(axis, rate, nuisance)
    weights = model_weights(*layout.table(q), counts)
    candidates = []
    for order in permutations(range(3)):
        candidate_layout, start = _seed(axis, rate, assigned_seed(nuisance, order))
        fitted = _stage(candidate_layout, means, weights, start, fixed=3)
        score = np.sum(
            apply_weights(weights, means - candidate_layout.mean_jacobian(fitted)[0])
            ** 2
        )
        candidates.append((float(score), candidate_layout, fitted))
    _, layout, q = min(candidates, key=lambda item: item[0])
    for _ in range(2):
        layout, q = layout.reslot(q)
        weights = model_weights(*layout.table(q), counts)
        q = _stage(layout, means, weights, q)
    predicted, jacobian = layout.mean_jacobian(q)
    weighted = apply_weights(weights, jacobian).reshape(-1, SIZE)
    information = scaled_information(weighted)
    residual = apply_weights(weights, means - predicted)
    joint = JointSolution(q, predicted, residual, information)
    return RateSolution(
        joint, layout.names, _nuisances(layout, q), _relaxation(layout, q, information)
    )


def _seed(axis: Array, rate: float, nuisance: Array) -> tuple[RateCoordinates, Array]:
    d0 = endpoints_of(nuisance)[:, 1]
    layout = RateCoordinates(axis, dominant_slots(d0))
    return layout, np.r_[np.full(3, rate), layout.pack_d0(d0), nuisance[SHARED]]


def _stage(
    layout: RateCoordinates, means: Array, weights: Array, q: Array, *, fixed: int = 0
) -> Array:
    lo, hi = layout.bounds()
    previous = np.full_like(q, np.nan)
    cached_mean = np.empty_like(means)
    cached_jac = np.empty((len(layout.axis), 65, SIZE))

    def evaluate(v: Array) -> tuple[Array, Array]:
        nonlocal previous, cached_mean, cached_jac
        if not np.array_equal(previous, v):
            cached_mean, cached_jac = layout.mean_jacobian(v)
            previous = v.copy()
        return cached_mean, cached_jac

    def residual(v: Array) -> Array:
        return apply_weights(weights, means - evaluate(np.r_[q[:fixed], v])[0]).ravel()

    def derivative(v: Array) -> Array:
        jacobian = evaluate(np.r_[q[:fixed], v])[1][:, :, fixed:]
        return -apply_weights(weights, jacobian).reshape(-1, len(v))

    result = bounded_least_squares(
        residual,
        q[fixed:],
        derivative,
        (lo[fixed:], hi[fixed:]),
        nuisance_only=bool(fixed),
    )
    if not result.success and not fixed:
        raise RuntimeError(f"joint rate cycle fit did not converge: {result.message}")
    return np.r_[q[:fixed], result.x]


def _nuisances(layout: RateCoordinates, q: Array) -> RateNuisances:
    shared = q[SHARED_RATE]
    return RateNuisances(
        layout.d0(q),
        shared[ROW_CENTERS].reshape(3, 2).copy(),
        float(shared[ROW_VARIANCE]),
        simplex(*shared[ROW_THERMAL]),
        shared[ROW_RETENTION].copy(),
        simplex(*shared[ROW_LANDING]),
    )


def _relaxation(
    layout: RateCoordinates, q: Array, information: ScaledInformation
) -> CycleRelaxation:
    spec = layout.spectrum(q)
    times, errors = [], []
    for mode in (1, 2):
        if spec.values[mode] >= 0:
            times.append(float("inf"))
            errors.append(float("inf"))
        else:
            gradient = layout.eigen_time_gradient(q, mode)
            times.append(float(-1 / spec.values[mode]))
            errors.append(float(information.errors(gradient[None, :])[0]))
    amplitudes = np.array([mode_amplitude(spec, layout.d0(q), k) for k in (1, 2)])
    winner = int(np.argmax(amplitudes))
    return CycleRelaxation(
        np.array(times),
        np.array(errors),
        amplitudes,
        times[winner],
        errors[winner],
        times[1 - winner],
        errors[1 - winner],
    )
