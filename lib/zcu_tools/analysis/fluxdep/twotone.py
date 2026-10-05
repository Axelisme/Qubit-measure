"""Captured TwoTone brush state, transitions and deterministic point projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.utils.process import SmoothMethod

from .line_state import FluxPickInputs
from .stroke import BrushPoint

BrushMode = Literal["select", "erase"]


@dataclass(frozen=True)
class TwoToneInputs:
    """Owned read-only spectrum in native device/GHz axes.

    spectrum supplies device-major finite complex signals, with exactly one
    column per frequency value. Construction rejects mismatched/nonfinite data
    with ValueError and captures data independently of caller-owned arrays.
    """

    spectrum: FluxPickInputs

    def __post_init__(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TwoToneSettings:
    """Partial detector update; None means retain that committed setting.

    threshold is finite [1,20]; sigma is [0,5] for wavelet and [0.001,5]
    for gaussian. smooth_method is wavelet or gaussian. Transitions require
    at least one non-None field and validate the resulting method/sigma pair.
    """

    threshold: float | None = None
    sigma: float | None = None
    smooth_method: SmoothMethod | None = None


@dataclass(frozen=True)
class TwoToneTool:
    """Partial tool update; None retains the committed value.

    width is finite normalized radius [0,0.1]; mode is select or erase.
    At least one non-None field is required by transitions.
    """

    width: float | None = None
    mode: BrushMode | None = None


@dataclass(frozen=True)
class TwoToneStroke:
    """One complete gesture in native device/GHz coordinates.

    vertices is nonempty and finite. width is normalized radius [0,0.1].
    mode is select or erase. The kernel enforces its 10,000-sample limit;
    zero width is valid only for a stationary stroke.
    """

    vertices: tuple[BrushPoint, ...]
    width: float
    mode: BrushMode


@dataclass(frozen=True)
class TwoTonePickChange:
    """Bounded before-image and brush geometry for the latest analysis mutation.

    mask is the prior boolean device-major selection; threshold, sigma and
    smooth_method are the prior detector settings. vertices is empty for
    settings/fill mutations, otherwise the most recent native-axis stroke.
    width is that stroke's normalized radius, or zero without stroke geometry.
    """

    mask: NDArray[np.bool_]
    threshold: float
    sigma: float
    smooth_method: SmoothMethod
    vertices: tuple[BrushPoint, ...] = ()
    width: float = 0.0


@dataclass(frozen=True)
class TwoTonePickState:
    """Complete detached selection and tools for one captured TwoTone spectrum.

    mask has shape (N_device,N_frequency), dtype bool. threshold is [1,20],
    sigma is [0,5] for wavelet or [0.001,5] for gaussian, width is normalized
    [0,0.1], mode is select/erase. last_change is the bounded before-image for
    rendering the latest analysis mutation, or None before the first mutation.
    """

    mask: NDArray[np.bool_]
    threshold: float = 1.0
    sigma: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    width: float = 0.05
    mode: BrushMode = "select"
    last_change: TwoTonePickChange | None = None


@dataclass(frozen=True)
class TwoTonePickResult:
    """Derived points and normalized background, never a second committed state.

    real_signals is a finite float device-major matrix. dev_values/freqs are
    paired 1D float arrays in native device/GHz units, ascending by device.
    Empty point arrays are a successful empty selection.
    """

    real_signals: NDArray[np.float64]
    dev_values: NDArray[np.float64]
    freqs: NDArray[np.float64]


@dataclass(frozen=True)
class TwoTonePickView:
    """Detached projection of one exact captured state and its latest change.

    state is the projected committed snapshot; result is its detected points.
    added_points/removed_points are float (N,2) arrays with device/GHz columns.
    mask_added/mask_removed are nonnegative counts against last_change.mask.
    stroke_vertices/stroke_width describe the applicable latest gesture in
    native axes/normalized radius, or empty/zero without a matching stroke.
    Without last_change or explicit previous state, changes are empty/zero.
    """

    state: TwoTonePickState
    result: TwoTonePickResult
    added_points: NDArray[np.float64]
    removed_points: NDArray[np.float64]
    mask_added: int
    mask_removed: int
    stroke_vertices: tuple[BrushPoint, ...] = ()
    stroke_width: float = 0.0


def make_twotone_state(
    inputs: TwoToneInputs,
    threshold: float = 1.0,
    sigma: float = 1.0,
    smooth_method: SmoothMethod = "wavelet",
    width: float = 0.05,
    mode: BrushMode = "select",
) -> TwoTonePickState:
    """Return an all-selected owned seed; ValueError rejects invalid settings."""
    raise NotImplementedError


def set_twotone_settings(
    inputs: TwoToneInputs, state: TwoTonePickState, params: TwoToneSettings
) -> TwoTonePickState:
    """Replace supplied detector settings and record a bounded before-image.

    Return detached state without modifying inputs/state. ValueError rejects
    empty updates, bool/nonfinite/out-of-range values or invalid prior state.
    """
    raise NotImplementedError


def set_twotone_tool(
    inputs: TwoToneInputs, state: TwoTonePickState, params: TwoToneTool
) -> TwoTonePickState:
    """Return detached tools while retaining last_change; ValueError rejects invalid input."""
    raise NotImplementedError


def stroke_twotone_state(
    inputs: TwoToneInputs, state: TwoTonePickState, params: TwoToneStroke
) -> TwoTonePickState:
    """Paint once via apply_mask_stroke, update tools and record the before-image.

    Validate the complete sampling budget before painting. Return detached
    state; ValueError leaves state/inputs unchanged for every invalid gesture.
    """
    raise NotImplementedError


def fill_twotone_state(
    inputs: TwoToneInputs, state: TwoTonePickState, *, select: bool
) -> TwoTonePickState:
    """Return filled/cleared mask with prior selection recorded, retaining tools.

    select=True fills, False clears. ValueError rejects invalid state or a
    non-bool select value; caller-owned state and arrays are unchanged.
    """
    raise NotImplementedError


def analyze_twotone_pick(
    inputs: TwoToneInputs, state: TwoTonePickState
) -> TwoTonePickResult:
    """Compute sorted native points from exact state, never a preview cache.

    Reuse cast2real_and_norm and spectrum2d_findpoint. Empty mask is valid.
    ValueError rejects invalid state/inputs; calculation never modifies either.
    """
    raise NotImplementedError


def project_twotone_pick(
    inputs: TwoToneInputs,
    state: TwoTonePickState,
    *,
    previous: TwoTonePickState | None = None,
) -> TwoTonePickView:
    """Compute exact result and positional/mask changes from previous or last_change.

    An explicit previous snapshot overrides the stored before-image, including
    for Undo feedback. Keep gesture geometry only if last_change matches that
    before-image. Preserve added/removed counts separately and exact state.
    ValueError rejects invalid snapshots. This query never creates history.
    """
    raise NotImplementedError
