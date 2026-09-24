from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TypeVar

import numpy as np
import scipy as sp
from numpy.typing import NDArray

Y_DataType = TypeVar("Y_DataType", bound=np.generic)


def with_fixed_params(
    fitfunc: Callable[..., NDArray[Y_DataType]],
    init_p: Sequence[float | None],
    bounds: tuple[Sequence[float], Sequence[float]] | None,
    fixedparams: Sequence[float | None],
) -> tuple[
    Callable[..., NDArray[Y_DataType]],
    Sequence[float | None],
    tuple[Sequence[float], Sequence[float]] | None,
]:
    fixedparams_array = np.asarray(fixedparams, dtype=np.float64)  # convert None to nan
    non_fixed_idxs = np.isnan(fixedparams_array)

    @wraps(fitfunc)
    def wrapped_func(xs: NDArray, *args) -> NDArray:
        if len(args) != np.sum(non_fixed_idxs):
            raise ValueError(
                f"Expected {np.sum(non_fixed_idxs)} arguments, got {len(args)}."
            )
        # assign the arguments to the parameters
        params = fixedparams_array.copy()
        params[non_fixed_idxs] = args

        return fitfunc(xs, *params)

    init_p_array = np.array(init_p)[non_fixed_idxs]
    init_p = list(init_p_array)

    if bounds is not None:
        bounds_array = np.array(bounds)[:, non_fixed_idxs]
        bounds = (list(bounds_array[0]), list(bounds_array[1]))
    else:
        bounds = None

    return wrapped_func, init_p, bounds


def add_fixed_params_back(
    pOpt: list[float], pCov: NDArray[np.float64], fixedparams: Sequence[float | None]
) -> tuple[list[float], NDArray[np.float64]]:
    _fixedparams = np.asarray(fixedparams, dtype=float)
    non_fixed_idxs = np.isnan(_fixedparams)

    pOpt_full = _fixedparams.copy()
    pOpt_full[non_fixed_idxs] = pOpt

    pCov_full = np.zeros((_fixedparams.size, _fixedparams.size))
    idx = np.where(non_fixed_idxs)[0]
    for i, row in enumerate(idx):
        for j, col in enumerate(idx):
            pCov_full[row, col] = pCov[i, j]

    return list(pOpt_full), pCov_full


def fit_func(
    xdata: NDArray,
    ydata: NDArray[Y_DataType],
    fitfunc: Callable[..., NDArray[Y_DataType]],
    init_p: Sequence[float | None] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    fixedparams: Sequence[float | None] | None = None,
    **kwargs,
) -> tuple[list[float], NDArray[np.float64]]:
    has_fixedparams = fixedparams is not None and any(
        p is not None for p in fixedparams
    )
    if has_fixedparams:
        assert fixedparams is not None
        if init_p is None:
            raise ValueError(
                "Initial parameters must be provided when fixed parameters are specified."
            )

        fitfunc, init_p, bounds = with_fixed_params(
            fitfunc, init_p, bounds, fixedparams
        )

    if bounds is None:
        bounds = (-np.inf, np.inf)  # type: ignore

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            fitfunc, xdata, ydata, p0=init_p, bounds=bounds, **kwargs
        )
    except RuntimeError as exc:
        if init_p is None:
            raise
        warnings.warn(
            "fit_func failed; returning init_p fallback with infinite covariance "
            f"({exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        pOpt = [p if p is not None else np.nan for p in init_p]
        pCov = np.full(shape=(len(init_p), len(init_p)), fill_value=np.inf)

    if has_fixedparams:
        assert fixedparams is not None
        pOpt, pCov = add_fixed_params_back(pOpt, pCov, fixedparams)

    return pOpt, pCov


def assign_init_p(
    fitparams: list[float | None], init_p: Sequence[float]
) -> list[float | None]:
    for i, p in enumerate(init_p):
        if fitparams[i] is None:
            fitparams[i] = p
    return fitparams


def fit_line(
    xdata: NDArray[np.float64], ydata: NDArray[np.float64]
) -> tuple[float, float]:
    """params: [a, b] -> y = a * x + b"""
    a, b, *_ = sp.stats.linregress(xdata, ydata)

    return a, b
