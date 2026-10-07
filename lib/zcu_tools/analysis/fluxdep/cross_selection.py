"""Captured joint-cloud filtering in calibrated flux/GHz, without caller mutation.

Transitions reject invalid shapes/dtypes, nonfinite values, boolean numbers,
ranges and gestures with ValueError. Radius/distance use the joint bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from .processing import downsample_points
from .stroke import (
    BrushMode,
    BrushPoint,
    BrushStroke,
    BrushTool,
    points_in_normalized_stroke,
)


@dataclass(frozen=True, slots=True)
class CrossSelectionBackground:
    """Read-only display background for one nonempty spectrum.

    name is its nonempty collection name. fluxs/freqs are finite float64 1-D
    calibrated-flux/GHz axes, strictly monotone with at least two samples.
    real_signals is a flux-major float64 matrix; masked/NaN samples represent
    missing data, infinity is invalid. Construction copies all arrays and raises
    ValueError on invalid data. Numerical preprocessing belongs to analysis.
    """

    name: str
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    real_signals: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("background name must be nonempty")
        fluxs = _capture_axis(self.fluxs, "flux axis")
        freqs = _capture_axis(self.freqs, "frequency axis")
        if self.real_signals.dtype != np.float64 or self.real_signals.shape != (
            fluxs.size,
            freqs.size,
        ):
            raise ValueError("real_signals must be a flux-major float64 matrix")
        signals = np.array(np.ma.filled(self.real_signals, np.nan), copy=True)
        if np.isinf(signals).any():
            raise ValueError("real_signals must not contain infinity")
        signals.setflags(write=False)
        object.__setattr__(self, "fluxs", fluxs)
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "real_signals", signals)


@dataclass(frozen=True, slots=True)
class CrossSelectionInputs:
    """Immutable full cloud and backgrounds, in service insertion order.

    fluxs/freqs are same-length finite float64 1-D arrays, at least one point.
    backgrounds includes only spectra with points; empty entries still belong to
    the app's source versions. flux_bound/freq_bound are finite (lower, upper)
    tuples with positive spans enclosing cloud and background axes. Construction
    copies all data read-only and raises ValueError on invalid data.
    """

    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    backgrounds: tuple[CrossSelectionBackground, ...]
    flux_bound: tuple[float, float]
    freq_bound: tuple[float, float]

    def __post_init__(self) -> None:
        fluxs = _capture_vector(self.fluxs, "cloud fluxs")
        freqs = _capture_vector(self.freqs, "cloud freqs")
        if fluxs.shape != freqs.shape:
            raise ValueError("cloud fluxs/freqs must have the same length")
        backgrounds = tuple(replace(background) for background in self.backgrounds)
        _validate_bound(self.flux_bound, "flux_bound")
        _validate_bound(self.freq_bound, "freq_bound")
        for xs, ys in [(fluxs, freqs)] + [
            (background.fluxs, background.freqs) for background in backgrounds
        ]:
            if xs.min() < self.flux_bound[0] or xs.max() > self.flux_bound[1]:
                raise ValueError("flux_bound must enclose cloud and backgrounds")
            if ys.min() < self.freq_bound[0] or ys.max() > self.freq_bound[1]:
                raise ValueError("freq_bound must enclose cloud and backgrounds")
        object.__setattr__(self, "fluxs", fluxs)
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "backgrounds", backgrounds)


@dataclass(frozen=True, slots=True)
class CrossSelectionChange:
    """One before-image, not an undo stack.

    selected is the prior full-cloud bool mask; min_distance is its normalized
    downsample distance. vertices is the latest calibrated flux/GHz stroke,
    empty for distance/fill mutations. width is its normalized radius or zero.
    """

    selected: NDArray[np.bool_]
    min_distance: float
    vertices: tuple[BrushPoint, ...] = ()
    width: float = 0.0


@dataclass(frozen=True, slots=True)
class CrossSelectionState:
    """Committed brush mask, downsample distance and tool for the full cloud.

    selected is a 1-D bool mask in captured cloud order. min_distance and width
    are finite normalized values [0,0.1]; width is radius, not diameter. mode is
    select or erase. last_change is one before-image/gesture or None on the seed.
    Transitions detach arrays and validate their length against captured inputs.
    """

    selected: NDArray[np.bool_]
    min_distance: float = 0.0
    width: float = 0.05
    mode: BrushMode = "select"
    last_change: CrossSelectionChange | None = None


@dataclass(frozen=True, slots=True)
class CrossSelectionResult:
    """Detached cloud and complete kept mask after deterministic downsampling.

    fluxs/freqs retain every captured point, including same-coordinate identities.
    selected is their full-length 1-D bool mask, not a subset-length mask.
    min_distance is the normalized distance used. An all-false mask is valid.
    """

    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    selected: NDArray[np.bool_]
    min_distance: float


@dataclass(frozen=True, slots=True)
class CrossSelectionView:
    """Detached exact snapshot, result and index-based point changes.

    state is the projected committed snapshot; result includes the complete mask.
    added_points/removed_points are (N,2) flux/GHz arrays comparing kept indices,
    preserving distinct points at identical coordinates. stroke_vertices and
    stroke_width describe the applicable gesture (empty/zero without one).
    """

    state: CrossSelectionState
    result: CrossSelectionResult
    added_points: NDArray[np.float64]
    removed_points: NDArray[np.float64]
    stroke_vertices: tuple[BrushPoint, ...] = ()
    stroke_width: float = 0.0


def make_cross_selection_state(
    inputs: CrossSelectionInputs,
    *,
    min_distance: float = 0.0,
    width: float = 0.05,
    mode: BrushMode = "select",
) -> CrossSelectionState:
    """Make an all-selected seed with finite [0,0.1] distance/radius.

    Reject bool numbers and modes other than select/erase with ValueError.
    """
    _number(min_distance, "min_distance")
    _number(width, "width")
    _validate_mode(mode)
    return CrossSelectionState(
        np.ones(inputs.fluxs.size, dtype=np.bool_), min_distance, width, mode
    )


def set_cross_selection_distance(
    inputs: CrossSelectionInputs, state: CrossSelectionState, min_distance: float
) -> CrossSelectionState:
    """Set finite [0,0.1] distance, retaining selection/tool and recording change.

    Reject bool numbers and invalid state with ValueError without mutation.
    """
    _validate_state(inputs, state)
    _number(min_distance, "min_distance")
    return replace(
        state,
        selected=state.selected.copy(),
        min_distance=min_distance,
        last_change=_before_image(state),
    )


def set_cross_selection_tool(
    inputs: CrossSelectionInputs, state: CrossSelectionState, tool: BrushTool
) -> CrossSelectionState:
    """Apply a nonempty partial width/mode update, retaining mask/last_change.

    None retains the field; radius width is finite [0,0.1], not bool; mode is
    select/erase. Empty/invalid update or state raises ValueError atomically.
    """
    _validate_state(inputs, state)
    if tool.width is None and tool.mode is None:
        raise ValueError("tool update must include width or mode")
    width = state.width if tool.width is None else tool.width
    mode = state.mode if tool.mode is None else tool.mode
    _number(width, "width")
    _validate_mode(mode)
    return replace(_detach_state(state), width=width, mode=mode)


def stroke_cross_selection_state(
    inputs: CrossSelectionInputs, state: CrossSelectionState, stroke: BrushStroke
) -> CrossSelectionState:
    """Apply one calibrated flux/GHz stroke using normalized joint spans.

    Share the kernel's 10,000-sample budget; zero radius requires stationary
    vertices. Set the gesture tool and record before-image/geometry. Invalid
    state, vertices, tool or sample-budget overflow raises ValueError atomically.
    """
    _validate_state(inputs, state)
    _number(stroke.width, "width")
    _validate_mode(stroke.mode)
    _validate_vertices(stroke.vertices)
    membership = points_in_normalized_stroke(
        inputs.fluxs,
        inputs.freqs,
        stroke=stroke.vertices,
        width=stroke.width,
        x_bound=inputs.flux_bound,
        y_bound=inputs.freq_bound,
    )
    selected = state.selected.copy()
    selected[membership] = stroke.mode == "select"
    return replace(
        state,
        selected=selected,
        width=stroke.width,
        mode=stroke.mode,
        last_change=replace(
            _before_image(state), vertices=stroke.vertices, width=stroke.width
        ),
    )


def fill_cross_selection_state(
    inputs: CrossSelectionInputs, state: CrossSelectionState, *, select: bool
) -> CrossSelectionState:
    """Select/erase every point, retaining distance/tool and recording change.

    select must be bool. Invalid state or select raises ValueError atomically.
    """
    _validate_state(inputs, state)
    _validate_boolean(select, "select")
    return replace(
        state,
        selected=np.full(inputs.fluxs.size, select, dtype=np.bool_),
        last_change=_before_image(state),
    )


def analyze_cross_selection(
    inputs: CrossSelectionInputs, state: CrossSelectionState
) -> CrossSelectionResult:
    """Compute a detached full kept mask with deterministic downsampling.

    Brush selection precedes downsampling in normalized joint bounds; distance
    zero keeps all brushed points. Empty selection is valid. Invalid state
    shape/dtype, tool or before-image raises ValueError without mutation.
    """
    _validate_state(inputs, state)
    selected = state.selected.copy()
    if state.min_distance > 0 and selected.any():
        xs = (inputs.fluxs[selected] - inputs.flux_bound[0]) / (
            inputs.flux_bound[1] - inputs.flux_bound[0]
        )
        ys = (inputs.freqs[selected] - inputs.freq_bound[0]) / (
            inputs.freq_bound[1] - inputs.freq_bound[0]
        )
        selected[selected] = downsample_points(xs, ys, state.min_distance)
    return CrossSelectionResult(
        inputs.fluxs.copy(), inputs.freqs.copy(), selected, state.min_distance
    )


def project_cross_selection(
    inputs: CrossSelectionInputs,
    state: CrossSelectionState,
    *,
    previous: CrossSelectionState | None = None,
) -> CrossSelectionView:
    """Project detached state/result and kept-index changes without mutation.

    Explicit previous overrides last_change, including gesture display, for Undo
    inverse. No before-image means empty changes. Same-coordinate identities stay
    distinct. Invalid state/previous raises ValueError; arrays detach.
    """
    result = analyze_cross_selection(inputs, state)
    vertices: tuple[BrushPoint, ...] = ()
    width = 0.0
    before: CrossSelectionResult | None = None
    if previous is not None:
        before = analyze_cross_selection(inputs, previous)
        if previous.last_change is not None:
            vertices = previous.last_change.vertices
            width = previous.last_change.width
    elif state.last_change is not None:
        change = state.last_change
        before = analyze_cross_selection(
            inputs,
            replace(
                state,
                selected=change.selected,
                min_distance=change.min_distance,
                last_change=None,
            ),
        )
        vertices, width = change.vertices, change.width
    points = np.column_stack((inputs.fluxs, inputs.freqs))
    added = np.empty((0, 2), dtype=np.float64)
    removed = np.empty((0, 2), dtype=np.float64)
    if before is not None:
        added = points[result.selected & ~before.selected]
        removed = points[before.selected & ~result.selected]
    return CrossSelectionView(
        _detach_state(state), result, added, removed, vertices, width
    )


def _capture_vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    if (
        np.ma.isMaskedArray(values)
        or values.dtype != np.float64
        or values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite float64 vector")
    captured = values.copy()
    captured.setflags(write=False)
    return captured


def _capture_axis(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    axis = _capture_vector(values, name)
    differences = np.diff(axis)
    if axis.size < 2 or not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"{name} must be strictly monotone with at least two samples")
    return axis


def _validate_bound(bound: tuple[float, float], name: str) -> None:
    if len(bound) != 2:
        raise ValueError(f"{name} must be a finite (lower, upper) tuple")
    for value in bound:
        _finite_number(value, name)
    span = bound[1] - bound[0]
    if not np.isfinite(span) or span <= 0:
        raise ValueError(f"{name} must have a finite positive span")


def _finite_number(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        finite = isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not finite:
        raise ValueError(f"{name} must be a finite number")


def _validate_boolean(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")


def _number(value: float, name: str) -> None:
    _finite_number(value, name)
    if not 0 <= value <= 0.1:
        raise ValueError(f"{name} must be in [0,0.1]")


def _validate_mode(mode: BrushMode) -> None:
    if mode not in ("select", "erase"):
        raise ValueError("mode must be select or erase")


def _validate_vertices(vertices: tuple[BrushPoint, ...]) -> None:
    if not vertices:
        raise ValueError("stroke vertices must be a nonempty tuple")
    for point in vertices:
        _finite_number(point.x, "stroke x")
        _finite_number(point.y, "stroke y")


def _validate_mask(mask: NDArray[np.bool_], size: int) -> None:
    if mask.dtype != np.bool_ or mask.shape != (size,) or np.ma.isMaskedArray(mask):
        raise ValueError("selected must be a complete 1-D bool cloud mask")


def _validate_state(inputs: CrossSelectionInputs, state: CrossSelectionState) -> None:
    _validate_mask(state.selected, inputs.fluxs.size)
    _number(state.min_distance, "min_distance")
    _number(state.width, "width")
    _validate_mode(state.mode)
    change = state.last_change
    if change is not None:
        _validate_mask(change.selected, inputs.fluxs.size)
        _number(change.min_distance, "min_distance")
        _number(change.width, "stroke width")
        if change.vertices:
            _validate_vertices(change.vertices)
            # Share stationary zero-radius and sample-budget validation with the kernel.
            points_in_normalized_stroke(
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                stroke=change.vertices,
                width=change.width,
                x_bound=inputs.flux_bound,
                y_bound=inputs.freq_bound,
            )
        elif change.width != 0:
            raise ValueError("stroke width without vertices must be zero")


def _before_image(state: CrossSelectionState) -> CrossSelectionChange:
    return CrossSelectionChange(state.selected.copy(), state.min_distance)


def _detach_state(state: CrossSelectionState) -> CrossSelectionState:
    change = state.last_change
    if change is not None:
        change = replace(change, selected=change.selected.copy())
    return replace(state, selected=state.selected.copy(), last_change=change)
