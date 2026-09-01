from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import cast

import numpy as np
from iminuit import Minuit
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.special import expit

from zcu_tools.utils.fitting.singleshot import (
    transition_state_bin_probabilities,
    transition_state_circle_probabilities,
)

_PARAMETER_NAMES = (
    "p_e0_u",
    "p_inf_u",
    "center_mid",
    "log_separation",
    "log_sigma",
    "p_avg_u",
    "log_length_ratio",
    "log_t_r",
    "log_omega",
)
_PENALTY = 1e300


@dataclass(frozen=True)
class LenRabiProjection:
    projected: NDArray[np.float64]
    bin_edges: NDArray[np.float64]
    counts: NDArray[np.int64]
    origin: complex
    axis: complex
    perpendicular_offset: float


@dataclass(frozen=True)
class LenRabiPhysicalParams:
    p_e0: float
    p_inf: float
    center_g: float
    center_e: float
    sigma: float
    p_avg: float
    length_ratio: float
    t_r: float
    omega: float


@dataclass(frozen=True)
class LenRabiBackendResult:
    parameter_names: tuple[str, ...]
    values: NDArray[np.float64]
    covariance: NDArray[np.float64]
    valid: bool
    covariance_accurate: bool
    reached_call_limit: bool
    hesse_failed: bool
    edm: float
    nll: float
    calls: int


@dataclass(frozen=True)
class LenRabiJointFitResult:
    initial_populations: NDArray[np.float64]
    p_inf: float
    t_r: float
    omega: float
    projected_g_center: float
    projected_e_center: float
    sigma: float
    p_avg: float
    length_ratio: float
    g_center: complex
    e_center: complex
    radius: float
    confusion_matrix: NDArray[np.float64]
    condition_number: float
    measured_populations: NDArray[np.float64]
    fitted_populations: NDArray[np.float64]
    projection: LenRabiProjection
    backend: LenRabiBackendResult


def project_len_rabi_iq(
    signals: NDArray[np.complex128],
) -> LenRabiProjection:
    raw = np.asarray(signals, dtype=np.complex128)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError("Len Rabi joint fit requires a two-dimensional raw-IQ sweep")
    if np.any(~np.isfinite(raw)):
        raise ValueError("Len Rabi joint fit requires finite raw IQ")

    points = np.column_stack((raw.real.ravel(), raw.imag.ravel()))
    mean = points.mean(axis=0)
    centered = points - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(np.linalg.norm(centered)), 1.0)
    if singular_values.size == 0 or singular_values[0] <= np.finfo(float).eps * scale:
        raise ValueError("Len Rabi pooled PCA has no identifiable projection axis")

    axis_vector = right[0]
    projected = (centered @ axis_vector).reshape(raw.shape)

    # PCA eigenvectors have arbitrary sign. Cluster the pooled projection, then
    # assign the cluster that dominates the first length to ground and orient it
    # below excited. This remains deterministic when row and pooled means tie.
    cluster_centers = np.asarray(np.quantile(projected, [0.25, 0.75]), dtype=np.float64)
    labels = np.empty(projected.shape, dtype=np.intp)
    for _ in range(16):
        labels = np.argmin(
            np.abs(projected[..., None] - cluster_centers[None, None, :]), axis=2
        )
        updated = cluster_centers.copy()
        for index in range(2):
            members = projected[labels == index]
            if members.size:
                updated[index] = float(members.mean())
        if np.allclose(updated, cluster_centers, rtol=0.0, atol=1e-12):
            break
        cluster_centers = updated
    initial_counts = np.bincount(labels[0], minlength=2)
    ground_cluster = int(np.argmax(initial_counts))
    if cluster_centers[ground_cluster] > cluster_centers[1 - ground_cluster]:
        axis_vector = -axis_vector
        projected = -projected

    bin_edges = np.asarray(
        np.histogram_bin_edges(projected.ravel(), bins="auto"), dtype=np.float64
    )
    if bin_edges.size < 3 or np.any(np.diff(bin_edges) <= 0.0):
        raise ValueError("Len Rabi projected histogram bins are not identifiable")
    counts = np.stack(
        [np.histogram(row, bins=bin_edges)[0] for row in projected], axis=0
    ).astype(np.int64)
    perpendicular = np.array([-axis_vector[1], axis_vector[0]])
    return LenRabiProjection(
        projected=np.asarray(projected, dtype=np.float64),
        bin_edges=bin_edges,
        counts=counts,
        origin=complex(float(mean[0]), float(mean[1])),
        axis=complex(float(axis_vector[0]), float(axis_vector[1])),
        perpendicular_offset=float(np.dot(mean, perpendicular)),
    )


def rabi_excited_population(
    lengths: NDArray[np.float64], params: LenRabiPhysicalParams
) -> NDArray[np.float64]:
    times = np.asarray(lengths, dtype=np.float64)
    return params.p_inf + (params.p_e0 - params.p_inf) * np.exp(
        -times / params.t_r
    ) * np.cos(params.omega * times)


def _unpack(values: NDArray[np.float64]) -> LenRabiPhysicalParams:
    separation = float(np.exp(values[3]))
    return LenRabiPhysicalParams(
        p_e0=0.5 * float(expit(values[0])),
        p_inf=float(expit(values[1])),
        center_g=float(values[2] - 0.5 * separation),
        center_e=float(values[2] + 0.5 * separation),
        sigma=float(np.exp(values[4])),
        p_avg=float(expit(values[5])),
        length_ratio=float(np.exp(values[6])),
        t_r=float(np.exp(values[7])),
        omega=float(np.exp(values[8])),
    )


def model_bin_probabilities(
    lengths: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    params: LenRabiPhysicalParams,
) -> NDArray[np.float64]:
    p_e = rabi_excited_population(lengths, params)
    if np.any(~np.isfinite(p_e)) or np.any((p_e < 0.0) | (p_e > 1.0)):
        return np.full((lengths.size, bin_edges.size - 1), np.nan)
    try:
        qg, qe = transition_state_bin_probabilities(
            bin_edges,
            params.center_g,
            params.center_e,
            params.sigma,
            params.p_avg,
            params.length_ratio,
        )
    except ValueError:
        return np.full((lengths.size, bin_edges.size - 1), np.nan)
    probabilities = (1.0 - p_e[:, None]) * qg[None, :] + p_e[:, None] * qe[None, :]
    probabilities = np.clip(probabilities, np.finfo(float).tiny, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.asarray(probabilities, dtype=np.float64)


def multinomial_nll(
    counts: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float:
    observed = np.asarray(counts)
    predicted = np.asarray(probabilities, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("counts and probabilities must have the same shape")
    if observed.ndim != 2 or np.any(observed < 0):
        raise ValueError("counts must be a non-negative two-dimensional array")
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return np.inf
    if not np.allclose(predicted.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probability rows must be normalized")
    return float(-np.sum(observed * np.log(predicted)))


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return log(clipped / (1.0 - clipped))


def _initial_values(
    lengths: NDArray[np.float64], projection: LenRabiProjection
) -> NDArray[np.float64]:
    projected = projection.projected
    lower, upper = np.quantile(projected, [0.2, 0.8])
    separation = max(float(upper - lower), float(np.ptp(projected)) * 0.2)
    sigma = max(0.12 * separation, float(np.min(np.diff(projection.bin_edges))))
    threshold = 0.5 * (lower + upper)
    crude_e = np.mean(projected > threshold, axis=1)
    p_e0 = float(np.clip(crude_e[0], 0.02, 0.45))
    p_inf = float(np.clip(np.mean(crude_e), 0.05, 0.95))

    span = float(np.ptp(lengths))
    centered = crude_e - crude_e.mean()
    omega_grid = np.linspace(0.5 * np.pi / span, 8.0 * np.pi / span, 96)
    scores = np.abs(np.exp(-1j * np.outer(omega_grid, lengths)) @ centered)
    omega = float(omega_grid[int(np.argmax(scores))])
    return np.array(
        [
            _logit(2.0 * p_e0),
            _logit(p_inf),
            0.5 * float(lower + upper),
            log(separation),
            log(sigma),
            _logit(0.2),
            log(0.1),
            log(max(span, np.finfo(float).eps)),
            log(omega),
        ],
        dtype=np.float64,
    )


def _fit_backend(
    lengths: NDArray[np.float64],
    projection: LenRabiProjection,
    *,
    max_calls: int | None,
) -> tuple[LenRabiPhysicalParams, LenRabiBackendResult]:
    initial = _initial_values(lengths, projection)
    calls = 0

    def objective(*args: float) -> float:
        nonlocal calls
        calls += 1
        values = np.asarray(args, dtype=np.float64)
        probabilities = model_bin_probabilities(
            lengths, projection.bin_edges, _unpack(values)
        )
        value = multinomial_nll(projection.counts, probabilities)
        return value if np.isfinite(value) else _PENALTY

    fit = Minuit(objective, *initial, name=_PARAMETER_NAMES)
    fit.errordef = Minuit.LIKELIHOOD
    projected_span = float(np.ptp(projection.projected))
    minimum_bin = float(np.min(np.diff(projection.bin_edges)))
    time_span = float(np.ptp(lengths))
    minimum_step = float(np.min(np.diff(lengths)))
    fit.limits["p_e0_u"] = (-12.0, 12.0)
    fit.limits["p_inf_u"] = (-12.0, 12.0)
    fit.limits["p_avg_u"] = (-12.0, 12.0)
    fit.limits["center_mid"] = (
        float(projection.bin_edges[0] - projected_span),
        float(projection.bin_edges[-1] + projected_span),
    )
    fit.limits["log_separation"] = (
        log(max(0.25 * minimum_bin, projected_span * 1e-5)),
        log(4.0 * projected_span),
    )
    fit.limits["log_sigma"] = (
        log(max(0.1 * minimum_bin, projected_span * 1e-6)),
        log(2.0 * projected_span),
    )
    fit.limits["log_length_ratio"] = (-12.0, 4.0)
    fit.limits["log_t_r"] = (log(time_span * 1e-3), log(time_span * 1e3))
    fit.limits["log_omega"] = (
        log(2.0 * np.pi / (time_span * 100.0)),
        log(4.0 * np.pi / minimum_step),
    )
    fit.migrad(ncall=max_calls)
    fit.hesse()
    fmin = fit.fmin
    fval = fit.fval
    if fmin is None or fval is None:
        raise RuntimeError("iminuit did not return a Len Rabi fit minimum")

    values = np.array([fit.values[name] for name in _PARAMETER_NAMES], dtype=np.float64)
    covariance = np.full((len(_PARAMETER_NAMES), len(_PARAMETER_NAMES)), np.nan)
    if fit.covariance is not None:
        for row, row_name in enumerate(_PARAMETER_NAMES):
            for col, col_name in enumerate(_PARAMETER_NAMES):
                covariance[row, col] = fit.covariance[row_name, col_name]
    backend = LenRabiBackendResult(
        parameter_names=_PARAMETER_NAMES,
        values=values,
        covariance=covariance,
        valid=bool(fmin.is_valid),
        covariance_accurate=bool(fmin.has_accurate_covar),
        reached_call_limit=bool(fmin.has_reached_call_limit),
        hesse_failed=bool(fmin.hesse_failed),
        edm=float(fmin.edm),
        nll=float(fval),
        calls=calls,
    )
    return _unpack(values), backend


def _failed_backend() -> LenRabiBackendResult:
    size = len(_PARAMETER_NAMES)
    return LenRabiBackendResult(
        parameter_names=_PARAMETER_NAMES,
        values=np.full(size, np.nan),
        covariance=np.full((size, size), np.nan),
        valid=False,
        covariance_accurate=False,
        reached_call_limit=False,
        hesse_failed=True,
        edm=np.inf,
        nll=np.inf,
        calls=0,
    )


def _failed_result(
    projection: LenRabiProjection,
    backend: LenRabiBackendResult | None = None,
) -> LenRabiJointFitResult:
    rows = projection.projected.shape[0]
    return LenRabiJointFitResult(
        initial_populations=np.array([np.nan, np.nan, 0.0]),
        p_inf=np.nan,
        t_r=np.nan,
        omega=np.nan,
        projected_g_center=np.nan,
        projected_e_center=np.nan,
        sigma=np.nan,
        p_avg=np.nan,
        length_ratio=np.nan,
        g_center=complex(np.nan, np.nan),
        e_center=complex(np.nan, np.nan),
        radius=np.nan,
        confusion_matrix=np.full((3, 3), np.nan),
        condition_number=np.inf,
        measured_populations=np.full((rows, 3), np.nan),
        fitted_populations=np.full((rows, 3), np.nan),
        projection=projection,
        backend=_failed_backend() if backend is None else backend,
    )


def _mixture_populations(
    counts: NDArray[np.int64],
    qg: NDArray[np.float64],
    qe: NDArray[np.float64],
) -> NDArray[np.float64]:
    populations = np.empty((counts.shape[0], 3), dtype=np.float64)
    for index, row in enumerate(counts):

        def objective(p_e: float) -> float:
            probabilities = (1.0 - p_e) * qg + p_e * qe
            probabilities = np.clip(probabilities, np.finfo(float).tiny, None)
            return float(-np.sum(row * np.log(probabilities)))

        optimum = cast(
            OptimizeResult,
            minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded"),
        )
        p_e = float(optimum.x) if optimum.success else np.nan
        populations[index] = (1.0 - p_e, p_e, 0.0)
    return populations


def _confusion_matrix(
    params: LenRabiPhysicalParams,
) -> tuple[float, NDArray[np.float64], float]:
    max_radius = 0.5 * (params.center_e - params.center_g)

    def matrix_at(radius: float) -> NDArray[np.float64]:
        g_row, e_row = transition_state_circle_probabilities(
            params.center_g,
            params.center_e,
            params.sigma,
            params.p_avg,
            params.length_ratio,
            radius,
        )
        return np.vstack((g_row, e_row, np.array([0.0, 0.0, 1.0])))

    def objective(radius: float) -> float:
        condition = float(np.linalg.cond(matrix_at(radius)))
        return condition if np.isfinite(condition) else _PENALTY

    optimum = cast(
        OptimizeResult,
        minimize_scalar(
            objective,
            bounds=(max_radius * 1e-6, max_radius),
            method="bounded",
            options={"xatol": max(max_radius * 1e-8, 1e-12)},
        ),
    )
    radius = float(optimum.x)
    if not optimum.success or not np.isfinite(radius):
        raise RuntimeError("classification-radius optimization failed")
    matrix = matrix_at(radius)
    return radius, matrix, float(np.linalg.cond(matrix))


def fit_len_rabi_joint(
    lengths: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    max_calls: int | None = None,
) -> LenRabiJointFitResult:
    times = np.asarray(lengths, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or np.any(~np.isfinite(times)):
        raise ValueError("Len Rabi joint fit requires at least two finite lengths")
    if np.any(np.diff(times) <= 0.0) or float(np.ptp(times)) <= 0.0:
        raise ValueError("Len Rabi lengths must be strictly increasing")
    projection = project_len_rabi_iq(signals)
    if projection.projected.shape[0] != times.size:
        raise ValueError("Len Rabi length and raw-IQ row counts do not match")

    backend: LenRabiBackendResult | None = None
    try:
        params, backend = _fit_backend(times, projection, max_calls=max_calls)
        if not backend.valid:
            return _failed_result(projection, backend)
        physical = np.array(
            [
                params.p_e0,
                params.p_inf,
                params.center_g,
                params.center_e,
                params.sigma,
                params.p_avg,
                params.length_ratio,
                params.t_r,
                params.omega,
            ]
        )
        if np.any(~np.isfinite(physical)):
            return _failed_result(projection)
        p_e = rabi_excited_population(times, params)
        if np.any((p_e < 0.0) | (p_e > 1.0)):
            return _failed_result(projection)
        qg, qe = transition_state_bin_probabilities(
            projection.bin_edges,
            params.center_g,
            params.center_e,
            params.sigma,
            params.p_avg,
            params.length_ratio,
        )
        measured = _mixture_populations(projection.counts, qg, qe)
        fitted = np.column_stack((1.0 - p_e, p_e, np.zeros_like(p_e)))
        radius, confusion, condition = _confusion_matrix(params)
    except (FloatingPointError, RuntimeError, ValueError):
        return _failed_result(projection, backend)

    axis = projection.axis
    return LenRabiJointFitResult(
        initial_populations=np.array([1.0 - params.p_e0, params.p_e0, 0.0]),
        p_inf=params.p_inf,
        t_r=params.t_r,
        omega=params.omega,
        projected_g_center=params.center_g,
        projected_e_center=params.center_e,
        sigma=params.sigma,
        p_avg=params.p_avg,
        length_ratio=params.length_ratio,
        g_center=projection.origin + axis * params.center_g,
        e_center=projection.origin + axis * params.center_e,
        radius=radius,
        confusion_matrix=confusion,
        condition_number=condition,
        measured_populations=measured,
        fitted_populations=fitted,
        projection=projection,
        backend=backend,
    )
