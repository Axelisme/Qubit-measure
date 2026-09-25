"""Flux-line state and candidate calculation shared by interactive frontends.

Positions are in device-axis units. Selection and preview belong to each frontend;
this state contains only values that can affect the committed analysis result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.processing import diff_mirror

FluxLineRole: TypeAlias = Literal["half", "integer"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FluxPickState:
    flux_half: float
    flux_int: float
    conjugate: bool = False
    magnitude_only: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.flux_half):
            raise ValueError("flux_half must be finite")
        if not isfinite(self.flux_int):
            raise ValueError("flux_int must be finite")


@dataclass(frozen=True, slots=True)
class FluxPickInputs:
    """Owned, read-only spectrum and device/frequency axes for one picker."""

    signals: NDArray[np.complex128]
    dev_values: NDArray[np.float64]
    freqs: NDArray[np.float64]

    def __post_init__(self) -> None:
        signals = np.array(self.signals, dtype=np.complex128, copy=True)
        dev_values = np.array(self.dev_values, dtype=np.float64, copy=True)
        freqs = np.array(self.freqs, dtype=np.float64, copy=True)
        if dev_values.ndim != 1 or dev_values.size < 5:
            raise ValueError("dev_values must be a 1D axis with at least five values")
        if freqs.ndim != 1 or freqs.size < 2:
            raise ValueError("freqs must be a 1D axis with at least two values")
        if (
            signals.ndim != 2
            or signals.shape[0] != dev_values.size
            or not signals.shape[1]
        ):
            raise ValueError("signals shape must have device rows and frequency data")
        if not np.isfinite(dev_values).all() or dev_values[0] == dev_values[-1]:
            raise ValueError("dev_values must span finite distinct endpoints")
        if not np.isfinite(freqs).all() or freqs[0] == freqs[-1]:
            raise ValueError("freqs must span finite distinct endpoints")
        for name, values in (
            ("signals", signals),
            ("dev_values", dev_values),
            ("freqs", freqs),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def min_distance(self) -> float:
        """Minimum line separation in device-axis units (1% of axis span)."""
        return 0.01 * abs(float(self.dev_values[-1] - self.dev_values[0]))


def _mirror_inbounds_mask(
    dev_values: NDArray[np.float64], center: float
) -> NDArray[np.bool_]:
    """Rows whose mirror counterpart stays inside the device-value axis."""
    n = len(dev_values)
    c_idx = (n - 1) * (center - dev_values[0]) / (dev_values[-1] - dev_values[0])
    idxs = np.arange(n)
    mirror_idxs = np.round(2 * c_idx - idxs).astype(int)
    return (mirror_idxs >= 0) & (mirror_idxs < n)


def fold_initial_lines(
    dev_values: NDArray[np.float64],
    flux_half: float | None,
    flux_int: float | None,
) -> tuple[float, float]:
    """Initial ``(flux_half, flux_int)`` folded near the spectrum center."""
    if dev_values.ndim != 1:
        raise ValueError("dev_values must be a 1D axis")
    if dev_values.size < 5:
        raise ValueError("dev_values must contain at least five values")

    flux_center = (dev_values[0] + dev_values[-1]) / 2
    half = flux_center if flux_half is None else flux_half
    intg = dev_values[-5] if flux_int is None else flux_int
    if flux_half is not None and flux_int is not None:
        fix_period = 2 * abs(intg - half)
        if fix_period != 0.0:
            half = half - round((half - flux_center) / fix_period) * fix_period
            intg = intg - round((intg - flux_center) / fix_period) * fix_period
    return float(half), float(intg)


def find_best_mirror_position(
    dev_values: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    current_pos: float,
    search_width: float,
) -> float:
    """Position with minimal mean mirror loss within ``current_pos +/- width/2``."""
    if dev_values.ndim != 1:
        raise ValueError("dev_values must be a 1D axis")
    if real_signals.shape[0] != dev_values.size:
        raise ValueError("real_signals first dimension must match len(dev_values)")
    if search_width < 0:
        raise ValueError("search_width must be non-negative")

    lo = float(dev_values.min())
    hi = float(dev_values.max())
    precision = 0.25 * (hi - lo) / len(dev_values)
    if precision <= 0.0:
        return current_pos
    left_bound = max(lo, current_pos - search_width / 2)
    right_bound = min(hi, current_pos + search_width / 2)
    left_steps = int(np.floor((left_bound - lo) / precision))
    right_steps = int(np.ceil((right_bound - lo) / precision))
    candidates = [
        lo + i * precision
        for i in range(left_steps, right_steps + 1)
        if lo <= lo + i * precision <= hi
    ]
    if not candidates:
        return current_pos

    best_pos = current_pos
    min_loss = float("inf")
    for candidate in candidates:
        _image, mean_loss = mirror_loss_at(dev_values, real_signals, candidate)
        if not np.isnan(mean_loss) and mean_loss < min_loss:
            min_loss = mean_loss
            best_pos = candidate
    return best_pos


def mirror_loss_at(
    dev_values: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    center: float,
) -> tuple[NDArray[np.float64], float]:
    """Mirror-loss image and mean over valid device-axis rows at ``center``."""
    if dev_values.ndim != 1 or dev_values.size < 2:
        raise ValueError("dev_values must be a 1D axis")
    if real_signals.ndim != 2 or real_signals.shape[0] != dev_values.size:
        raise ValueError("real_signals must match the device axis")
    if not isfinite(center) or dev_values[0] == dev_values[-1]:
        raise ValueError("center and device axis span must be finite")
    image = diff_mirror(dev_values, real_signals, center)
    # ADR-0028: valid zero loss must not be confused with out-of-bounds zero fill.
    inbounds = _mirror_inbounds_mask(dev_values, center)
    mean_loss = float(np.mean(image[inbounds])) if inbounds.any() else float("nan")
    return image, mean_loss


def align_lines(
    state: FluxPickState,
    dev_values: NDArray[np.float64],
    real_signals: NDArray[np.float64],
) -> FluxPickState:
    """Return independently mirror-aligned line positions without mutating state."""
    search_width = abs(float(dev_values[-1] - dev_values[0])) / 20
    half = find_best_mirror_position(
        dev_values, real_signals, state.flux_half, search_width
    )
    integer = find_best_mirror_position(
        dev_values, real_signals, state.flux_int, search_width
    )
    return replace(state, flux_half=half, flux_int=integer)


def swap_lines(state: FluxPickState) -> FluxPickState:
    """Exchange half/integer positions without changing state or settings."""
    return replace(state, flux_half=state.flux_int, flux_int=state.flux_half)


def move_line(
    state: FluxPickState,
    role: FluxLineRole,
    position: float,
    *,
    min_distance: float,
) -> FluxPickState:
    """Return a candidate at ``position`` without changing ``state``.

    Clamp against the other line to keep the specified minimum separation.
    Conjugate moves translate both lines by the same effective displacement.
    Invalid roles, non-finite coordinates and invalid distances raise ValueError.
    Frontends use this calculation for both preview and committed actions.
    """
    if role not in ("half", "integer"):
        raise ValueError(f"unknown flux-line role: {role!r}")
    if not isfinite(position):
        raise ValueError("position must be finite")
    if not isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_distance must be finite and non-negative")

    picked = state.flux_half if role == "half" else state.flux_int
    other = state.flux_int if role == "half" else state.flux_half
    if position > other and position - other < min_distance:
        position = other + min_distance
    elif position < other and other - position < min_distance:
        position = other - min_distance

    if state.conjugate:
        delta = position - picked
        return replace(
            state,
            flux_half=state.flux_half + delta,
            flux_int=state.flux_int + delta,
        )
    if role == "half":
        return replace(state, flux_half=position)
    return replace(state, flux_int=position)
