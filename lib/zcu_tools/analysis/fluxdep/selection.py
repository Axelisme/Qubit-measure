"""Brush-selection geometry for Flux-Dependence Analysis interactions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _span(lo: float, hi: float, name: str) -> float:
    span = float(hi - lo)
    if span == 0.0:
        raise ValueError(f"{name} span must be non-zero")
    return span


def points_in_normalized_brush(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    *,
    x: float,
    y: float,
    width: float,
    x_bound: tuple[float, float],
    y_bound: tuple[float, float],
) -> NDArray[np.bool_]:
    """Return points inside a circular brush in normalised axis coordinates."""

    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")
    if xs.ndim != 1:
        raise ValueError("xs and ys must be 1D arrays")
    if width < 0:
        raise ValueError("width must be non-negative")

    x_span = _span(x_bound[0], x_bound[1], "x_bound")
    y_span = _span(y_bound[0], y_bound[1], "y_bound")
    x_d = np.abs(xs - x) / x_span
    y_d = np.abs(ys - y) / y_span
    return x_d**2 + y_d**2 <= width**2


def toggle_near_mask(
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],
    mask: NDArray[np.bool_],
    x: float,
    y: float,
    width: float,
    select: bool,
) -> None:
    """In-place select/erase a 2D grid mask inside a normalised circular brush."""

    if dev_values.ndim != 1 or freqs.ndim != 1:
        raise ValueError("dev_values and freqs must be 1D axes")
    if mask.shape != (dev_values.size, freqs.size):
        raise ValueError("mask shape must match (len(dev_values), len(freqs))")
    if width < 0:
        raise ValueError("width must be non-negative")

    x_span = _span(float(dev_values[0]), float(dev_values[-1]), "dev_values")
    y_span = _span(float(freqs[0]), float(freqs[-1]), "freqs")
    x_d = np.abs(dev_values - x) / x_span
    y_d = np.abs(freqs - y) / y_span
    region = x_d[:, None] ** 2 + y_d[None, :] ** 2 <= width**2
    if select:
        mask |= region
    else:
        mask &= ~region
