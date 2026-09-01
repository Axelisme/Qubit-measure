from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import (
    decaycos,
    dual_expfunc,
    expfunc,
    fit_dualexp,
    fit_gauss,
    fitdecaycos,
    fitexp,
    gauss_func,
)
from .shared import FitTrace, ParameterSpec, fit_shared


def fit_decay(
    xs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    fit_params: tuple[float, float, float] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, float, float], NDArray[np.float64]],
]:
    """return [t1, t1err, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fitexp(xs, real_signals, fitparams=fit_params, fixedparams=fixedparams)

    fit_signals = expfunc(xs, *pOpt)

    t1: float = pOpt[2]
    t1err: float = np.sqrt(pCov[2, 2])
    return t1, t1err, fit_signals, (pOpt, pCov)


def fit_dual_decay(
    xs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    fit_params: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, float, float, float, float], NDArray[np.float64]],
]:
    """return [t1, t1err, t1b, t1berr, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fit_dualexp(
        xs, real_signals, fitparams=fit_params, fixedparams=fixedparams
    )

    # make sure t1 is the longer one
    if pOpt[4] > pOpt[2]:
        pOpt = (pOpt[0], pOpt[3], pOpt[4], pOpt[1], pOpt[2])
        new_pCov = pCov.copy()
        new_pCov[[1, 3], :] = new_pCov[[3, 1], :]
        new_pCov[[2, 4], :] = new_pCov[[4, 2], :]
        new_pCov[:, [1, 3]] = new_pCov[:, [3, 1]]
        new_pCov[:, [2, 4]] = new_pCov[:, [4, 2]]
        pCov = new_pCov

    fit_signals = dual_expfunc(xs, *pOpt)

    t1: float = pOpt[2]
    t1err: float = np.sqrt(pCov[2, 2])
    t1b: float = pOpt[4]
    t1berr: float = np.sqrt(pCov[4, 4])

    return t1, t1err, t1b, t1berr, fit_signals, (pOpt, pCov)


def fit_ge_decay(
    times: NDArray[np.float64],
    g_populations: NDArray[np.float64],
    e_populations: NDArray[np.float64],
    fit_params: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
    share_t1: bool = True,
) -> tuple[
    tuple[float, float, NDArray[np.float64], tuple[float, float, float]],
    tuple[float, float, NDArray[np.float64], tuple[float, float, float]],
]:
    """return [(g_t1, g_t1err, g_fit_signals, g_params), (e_t1, e_t1err, e_fit_signals, e_params)]"""
    g_params, g_pCov = fitexp(
        times, g_populations, fitparams=fit_params, fixedparams=fixedparams
    )  # (y0, yscale, decay)
    e_params, e_pCov = fitexp(
        times, e_populations, fitparams=fit_params, fixedparams=fixedparams
    )  # (y0, yscale, decay)

    if share_t1:
        fixed = (
            tuple(value is not None for value in fixedparams)
            if fixedparams is not None
            else (False, False, False)
        )
        result = fit_shared(
            traces=(
                FitTrace(times, g_populations, expfunc, ("g_y0", "g_scale", "t1")),
                FitTrace(times, e_populations, expfunc, ("e_y0", "e_scale", "t1")),
            ),
            parameters=(
                ParameterSpec("g_y0", g_params[0], fixed=fixed[0]),
                ParameterSpec(
                    "g_scale",
                    g_params[1],
                    fixed=fixed[1],
                    limits=(-2 * np.abs(g_params[1]), 2 * np.abs(g_params[1])),
                ),
                ParameterSpec(
                    "e_y0",
                    e_params[0],
                    fixed=fixed[0],
                ),
                ParameterSpec(
                    "e_scale",
                    e_params[1],
                    fixed=fixed[1],
                    limits=(-2 * np.abs(e_params[1]), 2 * np.abs(e_params[1])),
                ),
                ParameterSpec(
                    "t1",
                    0.5 * (g_params[2] + e_params[2]),
                    fixed=fixed[2],
                    limits=(0.0, None),
                ),
            ),
        )
        shared_t1 = result.values["t1"]
        g_params = (result.values["g_y0"], result.values["g_scale"], shared_t1)
        e_params = (result.values["e_y0"], result.values["e_scale"], shared_t1)
        shared_t1_error = np.sqrt(result.variance("t1"))
        g_t1err = shared_t1_error
        e_t1err = shared_t1_error
    else:
        g_t1err = np.sqrt(g_pCov[2, 2])
        e_t1err = np.sqrt(e_pCov[2, 2])

    g_t1 = g_params[2]
    e_t1 = e_params[2]

    if g_t1 > 0.8 * np.max(times) or g_t1 < 3 * (times[1] - times[0]):
        g_t1 = np.nan
        g_t1err = np.inf
    if e_t1 > 0.8 * np.max(times) or e_t1 < 3 * (times[1] - times[0]):
        e_t1 = np.nan
        e_t1err = np.inf

    g_fit_signals = expfunc(times, *g_params)
    e_fit_signals = expfunc(times, *e_params)

    return (
        (g_t1, g_t1err, g_fit_signals, g_params),
        (e_t1, e_t1err, e_fit_signals, e_params),
    )


def fit_decay_fringe(
    xs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    fit_params: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, float, float, float, float], NDArray[np.float64]],
]:
    """return [t2f, t2ferr, detune, detune_err, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fitdecaycos(
        xs, real_signals, fitparams=fit_params, fixedparams=fixedparams
    )

    fit_signals = decaycos(xs, *pOpt)

    t2f: float = pOpt[4]
    t2ferr: float = np.sqrt(pCov[4, 4])
    detune: float = pOpt[2]
    detune_err: float = np.sqrt(pCov[2, 2])

    pOpt = (pOpt[0], pOpt[1], pOpt[2], pOpt[3], pOpt[4])

    return t2f, t2ferr, detune, detune_err, fit_signals, (pOpt, pCov)


def fit_gauss_decay(
    xs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    fit_params: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, float, float, float], NDArray[np.float64]],
]:
    """return [t2g, t2gerr, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fit_gauss(
        xs, real_signals, fitparams=fit_params, fixedparams=fixedparams
    )

    fit_signals = gauss_func(xs, *pOpt)

    sigma = pOpt[3]
    sigma_err: float = np.sqrt(pCov[3, 3])

    # effective T2
    # f(0) * exp(-1) = f(0) * exp(-t2^2 / (2*sigma^2))
    t2 = np.sqrt(2) * sigma
    t2_err = np.sqrt(2) * sigma_err

    pOpt = (pOpt[0], pOpt[1], pOpt[2], pOpt[3])

    return t2, t2_err, fit_signals, (pOpt, pCov)
