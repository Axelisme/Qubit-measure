from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.special import iv, ndtr

_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(48)
_TRANSITION_POINTS = 0.5 * (_QUADRATURE_NODES + 1.0)
_TRANSITION_WEIGHTS = 0.5 * _QUADRATURE_WEIGHTS

from zcu_tools.utils.shot_classification import gaussian_region_probability

from .base import assign_init_p


def calc_fc(x: NDArray[np.float64], rA: float, rB: float) -> NDArray[np.float64]:
    f_c = np.zeros_like(x, dtype=float)

    mask_main = (x > 0) & (x <= 1)
    if np.any(mask_main):
        s = x[mask_main]

        # 計算兩個主要項
        z = 2.0 * np.sqrt(rA * rB * s * (1 - s))
        term1 = rA * iv(0, z)
        term2 = np.sqrt(rA * rB * (1 - s) / s) * iv(1, z)

        # 組合結果
        f_c[mask_main] = np.exp(-(rA * (1 - s) + rB * s)) * (term1 + term2)

    return f_c


def calc_noise_fc(
    xs: NDArray[np.float64], rA: float, rB: float, s: float
) -> NDArray[np.float64]:
    assert xs.ndim == 1 and xs.size > 0
    fc = calc_fc(xs, rA, rB)
    dx = (xs.max() - xs.min()) / (xs.size - 1)
    noise_kernel = stats.norm.pdf(np.arange(-2.5 * s, 2.5 * s, step=dx), loc=0, scale=s)
    noise_fc = np.convolve(fc, noise_kernel, mode="full") * dx
    noise_fc = noise_fc[len(noise_kernel) // 2 : len(noise_kernel) // 2 + len(xs)]
    return noise_fc


def calc_noise_f(
    xs: NDArray[np.float64], rA: float, rB: float, s: float
) -> NDArray[np.float64]:
    noise_f0 = np.exp(-rA) * stats.norm.pdf(xs, loc=0, scale=s)
    if rA != 0.0 or rB != 0.0:
        noise_fc = calc_noise_fc(xs, rA, rB, s)
    else:
        noise_fc = 0.0
    noise_f = noise_f0 + noise_fc
    return noise_f / np.sum(noise_f)


def calc_population_pdf(
    xs: NDArray[np.float64],
    sg: float,
    se: float,
    s: float,
    p0_g: float,
    p0_e: float,
    p_avg: float,
    length_ratio: float,
) -> NDArray[np.float64]:
    rg = p_avg * length_ratio
    re = (1 - p_avg) * length_ratio
    norm_s = s / abs(se - sg)
    norm_g_f = calc_noise_f((xs - sg) / (se - sg), rg, re, norm_s)
    norm_e_f = calc_noise_f((xs - se) / (sg - se), re, rg, norm_s)
    norm_f = p0_g * norm_g_f + p0_e * norm_e_f
    return norm_f / np.sum(norm_f) * (p0_g + p0_e)


def transition_state_bin_probabilities(
    bin_edges: NDArray[np.float64],
    sg: float,
    se: float,
    sigma: float,
    p_avg: float,
    length_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the existing readout-transition family over fixed histogram bins."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or np.any(~np.isfinite(edges)):
        raise ValueError("bin_edges must contain at least three finite values")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be strictly increasing")
    if not np.isfinite([sg, se, sigma, p_avg, length_ratio]).all():
        raise ValueError("readout-transition parameters must be finite")
    if se <= sg:
        raise ValueError("se must be greater than sg")
    if sigma <= 0.0 or not 0.0 <= p_avg <= 1.0 or length_ratio < 0.0:
        raise ValueError("invalid readout-transition parameter range")

    rg = p_avg * length_ratio
    re = (1.0 - p_avg) * length_ratio
    points = _TRANSITION_POINTS
    separation = se - sg

    def integrated_state(
        start: float,
        means: NDArray[np.float64],
        leave_rate: float,
        return_rate: float,
    ) -> NDArray[np.float64]:
        upper = ndtr((edges[1:, None] - means[None, :]) / sigma)
        lower = ndtr((edges[:-1, None] - means[None, :]) / sigma)
        continuous = (upper - lower) @ (
            calc_fc(points, leave_rate, return_rate) * _TRANSITION_WEIGHTS
        )
        atom = np.exp(-leave_rate) * (
            ndtr((edges[1:] - start) / sigma) - ndtr((edges[:-1] - start) / sigma)
        )
        probabilities = np.asarray(atom + continuous, dtype=np.float64)
        total = float(probabilities.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(
                "readout-transition bin probabilities are not normalizable"
            )
        probabilities /= total
        return probabilities

    qg = integrated_state(sg, sg + separation * points, rg, re)
    qe = integrated_state(se, se - separation * points, re, rg)
    return qg, qe


def transition_state_circle_probabilities(
    sg: float,
    se: float,
    sigma: float,
    p_avg: float,
    length_ratio: float,
    radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate conditional states over radius-limited nearest-center regions."""

    if se <= sg or sigma <= 0.0 or radius < 0.0:
        raise ValueError("invalid circle-classification geometry")

    rg = p_avg * length_ratio
    re = (1.0 - p_avg) * length_ratio
    points = _TRANSITION_POINTS
    weights_g = calc_fc(points, rg, re) * _TRANSITION_WEIGHTS
    weights_e = calc_fc(points, re, rg) * _TRANSITION_WEIGHTS
    separation = se - sg

    def circle_probability(
        atom_weight: float,
        atom_distance: float,
        path_distances: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        probabilities = gaussian_region_probability(
            np.concatenate(([atom_distance], path_distances)), sigma, radius, separation
        )
        atom = atom_weight * probabilities[0]
        continuous = float(np.dot(probabilities[1:], weights))
        return float(np.clip(atom + continuous, 0.0, 1.0))

    gg = circle_probability(np.exp(-rg), 0.0, separation * points, weights_g)
    ge = circle_probability(
        np.exp(-rg), separation, separation * (1.0 - points), weights_g
    )
    ee = circle_probability(np.exp(-re), 0.0, separation * points, weights_e)
    eg = circle_probability(
        np.exp(-re), separation, separation * (1.0 - points), weights_e
    )
    return (
        np.array([gg, ge, max(0.0, 1.0 - gg - ge)], dtype=np.float64),
        np.array([eg, ee, max(0.0, 1.0 - eg - ee)], dtype=np.float64),
    )


def gauss_func(xs: NDArray[np.float64], x_c: float, s: float) -> NDArray[np.float64]:
    """params: [x_c, s]"""
    f = stats.norm.pdf(xs, loc=x_c, scale=s)
    return f / np.sum(f)


def _fit_population_simplex(
    xs: NDArray[np.float64],
    data: NDArray[np.float64],
    model: Callable[..., NDArray[np.float64]],
    initial: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    fixed: Sequence[float | None],
    population_indices: tuple[int, int],
    sigma: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit legal populations and return covariance in physical coordinates.

    Free populations use total occupancy and conditional excited fraction.
    A single free population instead uses the remaining probability mass.
    Optimizer failures propagate; an initial guess is never a fit result.
    """
    g, e = population_indices
    values = np.asarray(initial, dtype=np.float64).copy()
    lower, upper = np.asarray(bounds, dtype=np.float64).copy()
    locked = np.array([value is not None for value in fixed])
    for index, value in enumerate(fixed):
        if value is not None:
            if not np.isfinite(value) or not lower[index] <= value <= upper[index]:
                raise ValueError("Fixed GE parameters must be finite and within bounds")
            values[index] = value
    if sum(values[index] for index in (g, e) if locked[index]) > 1.0:
        raise ValueError("Fixed GE populations must sum to at most one")
    if locked[g] and locked[e] and values[g] + values[e] == 0.0:
        raise ValueError("GE calibration requires positive modeled population")
    if not locked[g] and not locked[e]:
        total = max(float(values[g] + values[e]), np.finfo(float).eps)
        values[g], values[e] = min(total, 1.0), np.clip(values[e] / total, 0.0, 1.0)
    elif locked[g] != locked[e]:
        free_pop, fixed_pop = (e, g) if locked[g] else (g, e)
        remainder = 1.0 - values[fixed_pop]
        if remainder == 0.0:
            values[free_pop] = 0.0
            locked[free_pop] = True
        else:
            values[free_pop] = np.clip(values[free_pop] / remainder, 0.0, 1.0)
    free = np.flatnonzero(~locked)

    def decode(encoded: NDArray[np.float64]) -> NDArray[np.float64]:
        physical = encoded.copy()
        if not locked[g] and not locked[e]:
            physical[g] = encoded[g] * (1.0 - encoded[e])
            physical[e] = encoded[g] * encoded[e]
        elif locked[g] != locked[e]:
            free_pop, fixed_pop = (e, g) if locked[g] else (g, e)
            physical[free_pop] = encoded[free_pop] * (1.0 - encoded[fixed_pop])
        return physical

    def wrapped(x: NDArray[np.float64], *args: float) -> NDArray[np.float64]:
        encoded = values.copy()
        encoded[free] = args
        return model(x, *decode(encoded))

    covariance = np.zeros((values.size, values.size), dtype=np.float64)
    if free.size:
        fitted, free_cov = curve_fit(
            wrapped,
            xs,
            data,
            p0=np.clip(values[free], lower[free], upper[free]),
            bounds=(lower[free], upper[free]),
            sigma=sigma,
            max_nfev=5000,
        )
        if not np.isfinite(fitted).all() or not np.isfinite(free_cov).all():
            raise RuntimeError("GE fit returned non-finite parameters or covariance")
        values[free] = fitted
        covariance[np.ix_(free, free)] = free_cov
    jacobian = np.eye(values.size)
    if not locked[g] and not locked[e]:
        jacobian[g, g], jacobian[g, e] = 1.0 - values[e], -values[g]
        jacobian[e, g], jacobian[e, e] = values[e], values[g]
    elif locked[g] != locked[e]:
        free_pop, fixed_pop = (e, g) if locked[g] else (g, e)
        jacobian[free_pop, free_pop] = 1.0 - values[fixed_pop]
    return decode(values), jacobian @ covariance @ jacobian.T


def _swap_ge_parameters(values: Sequence[float | None]) -> list[float | None]:
    swapped = list(values)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    if swapped[5] is not None:
        swapped[5] = 1.0 - swapped[5]
    return swapped


def fit_singleshot(
    xs: NDArray[np.float64],
    g_pdfs: NDArray[np.float64],
    e_pdfs: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[tuple[float, float, float, float, float, float, float], NDArray[np.float64]]:
    """fitparams: [sg, se, s, p0_g, p0_e, p_avg, length_ratio]"""
    if (
        xs.ndim != 1
        or xs.size < 3
        or g_pdfs.shape != xs.shape
        or e_pdfs.shape != xs.shape
        or not np.isfinite([xs, g_pdfs, e_pdfs]).all()
        or np.any(np.diff(xs) <= 0)
        or np.any(g_pdfs < 0)
        or np.any(e_pdfs < 0)
        or g_pdfs.sum() <= 0
        or e_pdfs.sum() <= 0
    ):
        raise ValueError(
            "GE fit requires finite nonnegative histograms on an increasing axis"
        )
    if fixedparams is not None:
        if len(fixedparams) != 7:
            raise ValueError(
                "Fixed parameters must be a list of seven elements: [sg, se, s, p0_g, p0_e, p_avg, length_ratio]"
            )

        fixedparams = list(fixedparams)

        length_ratio = fixedparams[6]
        if length_ratio == 0.0:
            # p_avg not affecting the result
            fixedparams[5] = 0.5 if fixedparams[5] is None else fixedparams[5]

    if fitparams is not None and len(fitparams) != 7:
        raise ValueError("GE fit requires seven initial parameters")
    # Canonical numerical ordering makes exchanging physical labels exactly
    # equivalent, including starts, bounds and optimizer trajectories.
    if float(xs @ g_pdfs) > float(xs @ e_pdfs):
        fitted, covariance = fit_singleshot(
            xs,
            e_pdfs,
            g_pdfs,
            fitparams=None if fitparams is None else _swap_ge_parameters(fitparams),
            fixedparams=None
            if fixedparams is None
            else _swap_ge_parameters(fixedparams),
        )
        transform = np.eye(7)[[1, 0, 2, 3, 4, 5, 6]]
        transform[5, 5] = -1.0
        return cast(
            tuple[float, float, float, float, float, float, float],
            tuple(_swap_ge_parameters(fitted)),
        ), transform @ covariance @ transform.T

    if fitparams is None:
        fitparams = [None] * 7
    fitparams = list(fitparams)

    # guess initial parameters
    if any([p is None for p in fitparams]):
        # guess initial parameters
        sg = xs[np.argmax(g_pdfs)]
        se = xs[np.argmax(e_pdfs)]

        if sg < se:
            g_idxs = xs < sg
            e_idxs = xs > se
        else:
            g_idxs = xs > sg
            e_idxs = xs < se

        g_keep_pdf = g_pdfs[g_idxs]
        e_keep_pdf = e_pdfs[e_idxs]
        sigma_g = np.sum(g_keep_pdf * np.abs(xs[g_idxs] - sg)) / np.sum(g_keep_pdf)
        sigma_e = np.sum(e_keep_pdf * np.abs(xs[e_idxs] - se)) / np.sum(e_keep_pdf)
        s = 0.5 * (sigma_g + sigma_e)

        if sg == se:
            if np.sum(g_pdfs * xs) < np.sum(e_pdfs * xs):
                sg -= 0.2 * s
                se += 0.2 * s
            else:
                sg += 0.2 * s
                se -= 0.2 * s

        g_tran_pop = np.sum(g_pdfs[e_idxs])
        e_tran_pop = np.sum(e_pdfs[g_idxs])

        p0_e = min(0.5 * (g_tran_pop + e_tran_pop), 0.5)
        p0_g = 1 - p0_e
        transition_mass = g_tran_pop + e_tran_pop
        p_avg = float(g_tran_pop / transition_mass) if transition_mass > 0 else 0.5
        length_ratio = 0.01

        assign_init_p(
            fitparams,
            [
                float(sg),
                float(se),
                float(s),
                float(p0_g),
                float(p0_e),
                float(p_avg),
                length_ratio,
            ],
        )
    fitparams = cast(list[float], fitparams)
    if fixedparams is not None:
        for index, value in enumerate(fixedparams):
            if value is not None:
                fitparams[index] = value

    sg, se, s, p0_g, p0_e, p_avg, length_ratio = fitparams
    bounds = (
        [
            se if se < sg else np.min(xs),
            sg if sg < se else np.min(xs),
            xs[1] - xs[0],
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            se if se > sg else np.max(xs),
            sg if sg > se else np.max(xs),
            xs[-1] - xs[0],
            1.0,
            1.0,
            1.0,
            3.0,
        ],
    )

    if s <= xs[1] - xs[0]:
        raise ValueError("s is too small")

    # scipy requires p0 within bounds; data-derived guesses (sg, se, s, …) can fall
    # outside when the histogram is wide or pathological.  Clip each param into its
    # bound before handing off so curve_fit never sees an out-of-bounds p0.
    # lower < upper is guaranteed by the conditional derivation above, so np.clip is safe.
    lower = np.array(bounds[0], dtype=np.float64)
    upper = np.array(bounds[1], dtype=np.float64)
    fitparams = list(np.clip(np.array(fitparams, dtype=np.float64), lower, upper))

    cat_xs = np.concatenate([xs, xs])
    cat_pdfs = np.concatenate([g_pdfs, e_pdfs])

    def calc_cat_pdf(cat_xs: NDArray[np.float64], *args: float) -> NDArray[np.float64]:
        p0_g, p0_e = args[3], args[4]
        g_args = list(args)
        e_args = list(args)
        e_args[3], e_args[4] = p0_e, p0_g
        g_pdf = calc_population_pdf(cat_xs[: len(xs)], *g_args)
        e_pdf = calc_population_pdf(cat_xs[len(xs) :], *e_args)
        return np.concatenate([g_pdf, e_pdf])

    fixed = list(fixedparams) if fixedparams is not None else [None] * 7
    # Include both transition directions and multiple transition durations.
    candidates = []
    for avg_start, ratio_start in (
        (p_avg, length_ratio),
        (0.2, 0.5),
        (0.8, 0.5),
        (0.5, 1.5),
    ):
        initial = [float(value) for value in cast(list[float], fitparams)]
        initial[5], initial[6] = avg_start, ratio_start
        try:
            fitted, covariance = _fit_population_simplex(
                cat_xs, cat_pdfs, calc_cat_pdf, initial, bounds, fixed, (3, 4)
            )
        except RuntimeError:
            continue
        residual = calc_cat_pdf(cat_xs, *fitted) - cat_pdfs
        if not np.isfinite(residual).all():
            continue
        candidates.append((float(residual @ residual), fitted, covariance))
    if not candidates:
        raise RuntimeError("GE fit did not converge from any initial parameters")
    _, fitted, covariance = min(candidates, key=lambda item: item[0])
    return cast(
        tuple[float, float, float, float, float, float, float], tuple(fitted)
    ), covariance


def fit_singleshot_p0(
    xs: NDArray[np.float64],
    pdfs: NDArray[np.float64],
    init_p0_g: float,
    init_p0_e: float,
    ge_params: tuple[float, float, float, float, float, float, float],
    fit_length_ratio: bool = False,
) -> tuple[tuple[float, float, float], NDArray[np.float64]]:
    sg, se, s, _, _, p_avg, length_ratio = ge_params
    if sg > se:
        swapped = cast(
            tuple[float, float, float, float, float, float, float],
            tuple(_swap_ge_parameters(ge_params)),
        )
        fitted, covariance = fit_singleshot_p0(
            xs, pdfs, init_p0_e, init_p0_g, swapped, fit_length_ratio=fit_length_ratio
        )
        order = [1, 0, 2]
        return (fitted[1], fitted[0], fitted[2]), covariance[np.ix_(order, order)]

    def calc_pdf(
        xs: NDArray[np.float64], p0_g: float, p0_e: float, ratio: float
    ) -> NDArray[np.float64]:
        return calc_population_pdf(xs, sg, se, s, p0_g, p0_e, p_avg, ratio)

    fixed: list[float | None] = [None, None, None]
    if not fit_length_ratio or length_ratio == 0.0:
        fixed[2] = length_ratio
    weights = init_p0_g * gauss_func(xs, sg, s) + init_p0_e * gauss_func(xs, se, s)
    sigmas = 1 / np.sqrt(np.maximum(weights, np.finfo(float).tiny))
    fitted, covariance = _fit_population_simplex(
        xs,
        pdfs,
        calc_pdf,
        [init_p0_g, init_p0_e, length_ratio],
        ([0.0, 0.0, 0.5 * length_ratio], [1.0, 1.0, max(2.0 * length_ratio, 1e-12)]),
        fixed,
        (0, 1),
        sigma=sigmas,
    )
    return (float(fitted[0]), float(fitted[1]), float(fitted[2])), covariance
