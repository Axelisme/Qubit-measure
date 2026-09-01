from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from scipy.special import iv, ndtr

_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(48)
_TRANSITION_POINTS = 0.5 * (_QUADRATURE_NODES + 1.0)
_TRANSITION_WEIGHTS = 0.5 * _QUADRATURE_WEIGHTS

from .base import assign_init_p, fit_func


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
    """Integrate fitted conditional states over non-overlapping g/e circles."""
    from scipy.stats import ncx2

    if se <= sg or sigma <= 0.0 or radius < 0.0:
        raise ValueError("invalid circle-classification geometry")
    if radius > 0.5 * (se - sg):
        raise ValueError("classification circles must not overlap")

    rg = p_avg * length_ratio
    re = (1.0 - p_avg) * length_ratio
    points = _TRANSITION_POINTS
    weights_g = calc_fc(points, rg, re) * _TRANSITION_WEIGHTS
    weights_e = calc_fc(points, re, rg) * _TRANSITION_WEIGHTS
    separation = se - sg
    limit = (radius / sigma) ** 2

    def circle_probability(
        atom_weight: float,
        atom_distance: float,
        path_distances: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        atom = atom_weight * float(ncx2.cdf(limit, 2, (atom_distance / sigma) ** 2))
        continuous = float(
            np.dot(ncx2.cdf(limit, 2, (path_distances / sigma) ** 2), weights)
        )
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


def fit_singleshot(
    xs: NDArray[np.float64],
    g_pdfs: NDArray[np.float64],
    e_pdfs: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[tuple[float, float, float, float, float, float, float], NDArray[np.float64]]:
    """fitparams: [sg, se, s, p0_g, p0_e, p_avg, length_ratio]"""
    if fixedparams is not None:
        if len(fixedparams) != 7:
            raise ValueError(
                "Fixed parameters must be a list of six elements: [sg, se, s, p0_g, p0_e, p_avg, length_ratio]"
            )

        fixedparams = list(fixedparams)

        length_ratio = fixedparams[6]
        if length_ratio == 0.0:
            # p_avg not affecting the result
            fixedparams[5] = 0.5 if fixedparams[5] is None else fixedparams[5]

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

        p0_g = min(0.5 * (g_tran_pop + e_tran_pop), 0.5)
        p0_e = 1 - p0_g
        p_avg = min(e_tran_pop / (g_tran_pop + e_tran_pop), 0.5)
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
            0.5,
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

    pOpt, pCov = fit_func(
        cat_xs,
        cat_pdfs,
        calc_cat_pdf,
        init_p=fitparams,
        bounds=bounds,
        fixedparams=fixedparams,
    )
    pOpt = cast(tuple[float, float, float, float, float, float, float], pOpt)

    return pOpt, pCov


def fit_singleshot_p0(
    xs: NDArray[np.float64],
    pdfs: NDArray[np.float64],
    init_p0_g: float,
    init_p0_e: float,
    ge_params: tuple[float, float, float, float, float, float, float],
    fit_length_ratio: bool = False,
) -> tuple[tuple[float, float, float], NDArray[np.float64]]:
    sg, se, s, _, _, p_avg, length_ratio = ge_params

    def calc_pdf(
        xs: NDArray[np.float64], p0_g: float, p0_e: float, length_ratio: float
    ) -> NDArray[np.float64]:
        return calc_population_pdf(xs, sg, se, s, p0_g, p0_e, p_avg, length_ratio)

    fixedparams: list[float | None] = [None, None, None]
    if not fit_length_ratio:
        fixedparams[2] = length_ratio

    weights = init_p0_g * gauss_func(xs, sg, s) + init_p0_e * gauss_func(xs, se, s)
    sigmas = 1 / np.sqrt(weights)

    pOpt, pCov = fit_func(
        xs,
        pdfs,
        calc_pdf,
        init_p=[init_p0_g, init_p0_e, length_ratio],
        bounds=([0.0, 0.0, 0.5 * length_ratio], [1.0, 1.0, 2.0 * length_ratio]),
        sigma=sigmas,
        absolute_sigma=False,
    )
    pOpt = cast(tuple[float, float, float], pOpt)

    p0_g, p0_e, length_ratio = pOpt

    return (p0_g, p0_e, length_ratio), pCov
