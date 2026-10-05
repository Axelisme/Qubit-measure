"""Captured TwoTone brush state, transitions and deterministic point projections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.utils.process import SmoothMethod

from .line_state import FluxPickInputs
from .processing import cast2real_and_norm, spectrum2d_findpoint
from .stroke import BrushMode, BrushPoint, BrushStroke, BrushTool, apply_mask_stroke


@dataclass(frozen=True)
class TwoToneInputs:
    """Owned read-only spectrum in native device/GHz axes.

    spectrum supplies device-major finite complex signals, with exactly one
    column per frequency value. Construction rejects mismatched/nonfinite data
    with ValueError and captures data independently of caller-owned arrays.
    """

    spectrum: FluxPickInputs

    def __post_init__(self) -> None:
        spectrum = self.spectrum
        if spectrum.signals.shape != (spectrum.dev_values.size, spectrum.freqs.size):
            raise ValueError("signals shape must match device and frequency axes")
        if not np.isfinite(spectrum.signals).all():
            raise ValueError("signals must contain only finite complex values")
        object.__setattr__(
            self,
            "spectrum",
            FluxPickInputs(spectrum.signals, spectrum.dev_values, spectrum.freqs),
        )


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
    state = TwoTonePickState(
        np.ones(inputs.spectrum.signals.shape, dtype=np.bool_),
        threshold,
        sigma,
        smooth_method,
        width,
        mode,
    )
    _validate_state(inputs, state)
    return state


def set_twotone_settings(
    inputs: TwoToneInputs, state: TwoTonePickState, params: TwoToneSettings
) -> TwoTonePickState:
    """Replace supplied detector settings and record a bounded before-image.

    Return detached state without modifying inputs/state. ValueError rejects
    empty updates, bool/nonfinite/out-of-range values or invalid prior state.
    """
    _validate_state(inputs, state)
    if (
        params.threshold is None
        and params.sigma is None
        and params.smooth_method is None
    ):
        raise ValueError("settings must provide threshold, sigma or smooth_method")
    candidate = replace(
        deepcopy(state),
        threshold=state.threshold if params.threshold is None else params.threshold,
        sigma=state.sigma if params.sigma is None else params.sigma,
        smooth_method=state.smooth_method
        if params.smooth_method is None
        else params.smooth_method,
        last_change=_before_image(state),
    )
    _validate_state(inputs, candidate)
    return candidate


def set_twotone_tool(
    inputs: TwoToneInputs, state: TwoTonePickState, params: BrushTool
) -> TwoTonePickState:
    """Return detached tools while retaining last_change; ValueError rejects invalid input."""
    _validate_state(inputs, state)
    if params.width is None and params.mode is None:
        raise ValueError("tool must provide width or mode")
    candidate = replace(
        deepcopy(state),
        width=state.width if params.width is None else params.width,
        mode=state.mode if params.mode is None else params.mode,
    )
    _validate_state(inputs, candidate)
    return candidate


def stroke_twotone_state(
    inputs: TwoToneInputs, state: TwoTonePickState, params: BrushStroke
) -> TwoTonePickState:
    """Paint once via apply_mask_stroke, update tools and record the before-image.

    Validate the complete sampling budget before painting. Return detached
    state; ValueError leaves state/inputs unchanged for every invalid gesture.
    """
    _number(params.width, "width", 0, 0.1)
    if params.mode not in ("select", "erase"):
        raise ValueError("mode must be select or erase")
    candidate = set_twotone_tool(inputs, state, BrushTool(params.width, params.mode))
    _validate_vertices(params.vertices)
    spectrum = inputs.spectrum
    apply_mask_stroke(
        spectrum.dev_values,
        spectrum.freqs,
        candidate.mask,
        params.vertices,
        params.width,
        select=params.mode == "select",
    )
    return replace(
        candidate,
        last_change=replace(
            _before_image(state), vertices=tuple(params.vertices), width=params.width
        ),
    )


def fill_twotone_state(
    inputs: TwoToneInputs, state: TwoTonePickState, *, select: bool
) -> TwoTonePickState:
    """Return filled/cleared mask with prior selection recorded, retaining tools.

    select=True fills, False clears. ValueError rejects invalid state or a
    non-bool select value; caller-owned state and arrays are unchanged.
    """
    _validate_state(inputs, state)
    if type(select) is not bool:
        raise ValueError("select must be a bool")
    candidate = deepcopy(state)
    candidate.mask.fill(select)
    return replace(candidate, last_change=_before_image(state))


def analyze_twotone_pick(
    inputs: TwoToneInputs, state: TwoTonePickState
) -> TwoTonePickResult:
    """Compute sorted native points from exact state, never a preview cache.

    Reuse cast2real_and_norm and spectrum2d_findpoint. Empty mask is valid.
    ValueError rejects invalid state/inputs; calculation never modifies either.
    """
    _validate_state(inputs, state)
    spectrum = inputs.spectrum
    real_signals = np.asarray(
        cast2real_and_norm(
            spectrum.signals, sigma=state.sigma, smooth_method=state.smooth_method
        ),
        dtype=np.float64,
    )
    if not np.isfinite(real_signals).all():
        raise ValueError("signals normalization produced nonfinite values")
    devs, freqs = spectrum2d_findpoint(
        spectrum.dev_values,
        spectrum.freqs,
        real_signals,
        state.threshold,
        weight=state.mask,
    )
    order = np.argsort(devs, kind="stable")
    return TwoTonePickResult(real_signals.copy(), devs[order], freqs[order])


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
    result = analyze_twotone_pick(inputs, state)
    change = state.last_change
    if previous is None and change is not None:
        previous = replace(
            state,
            mask=change.mask,
            threshold=change.threshold,
            sigma=change.sigma,
            smooth_method=change.smooth_method,
            last_change=None,
        )
    added = np.empty((0, 2), dtype=np.float64)
    removed = np.empty((0, 2), dtype=np.float64)
    mask_added = mask_removed = 0
    vertices: tuple[BrushPoint, ...] = ()
    width = 0.0
    if previous is not None:
        prior = analyze_twotone_pick(inputs, previous)
        current_points = list(
            zip(result.dev_values.tolist(), result.freqs.tolist(), strict=True)
        )
        prior_points = list(
            zip(prior.dev_values.tolist(), prior.freqs.tolist(), strict=True)
        )
        current_set, prior_set = set(current_points), set(prior_points)
        added = np.array(
            [p for p in current_points if p not in prior_set], dtype=np.float64
        ).reshape(-1, 2)
        removed = np.array(
            [p for p in prior_points if p not in current_set], dtype=np.float64
        ).reshape(-1, 2)
        mask_added = int(np.count_nonzero(state.mask & ~previous.mask))
        mask_removed = int(np.count_nonzero(previous.mask & ~state.mask))
        if change is not None and (
            np.array_equal(change.mask, previous.mask)
            and change.threshold == previous.threshold
            and change.sigma == previous.sigma
            and change.smooth_method == previous.smooth_method
        ):
            vertices, width = change.vertices, change.width
    return TwoTonePickView(
        deepcopy(state),
        result,
        added,
        removed,
        mask_added,
        mask_removed,
        vertices,
        width,
    )


def _number(value: object, name: str, lower: float, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    # Compare bounds before conversion so arbitrarily large ints fail without overflow.
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")


def _validate_detector(threshold: float, sigma: float, method: SmoothMethod) -> None:
    _number(threshold, "threshold", 1, 20)
    if method not in ("wavelet", "gaussian"):
        raise ValueError("smooth_method must be wavelet or gaussian")
    _number(sigma, "sigma", 0.001 if method == "gaussian" else 0, 5)


def _validate_vertices(vertices: tuple[BrushPoint, ...]) -> None:
    if not vertices:
        raise ValueError("vertices must be nonempty")
    for point in vertices:
        _finite_coordinate(point.x, "vertices.x")
        _finite_coordinate(point.y, "vertices.y")


def _finite_coordinate(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric coordinates")
    try:
        finite = np.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite numeric coordinates") from exc
    if not finite:
        raise ValueError(f"{name} must be finite numeric coordinates")


def _validate_state(inputs: TwoToneInputs, state: TwoTonePickState) -> None:
    def validate_mask(mask: NDArray[np.bool_]) -> None:
        if mask.dtype != np.bool_ or mask.shape != inputs.spectrum.signals.shape:
            raise ValueError("mask must be boolean and match signals shape")

    validate_mask(state.mask)
    _validate_detector(state.threshold, state.sigma, state.smooth_method)
    _number(state.width, "width", 0, 0.1)
    if state.mode not in ("select", "erase"):
        raise ValueError("mode must be select or erase")
    if state.last_change is not None:
        change = state.last_change
        validate_mask(change.mask)
        _validate_detector(change.threshold, change.sigma, change.smooth_method)
        _number(change.width, "width", 0, 0.1)
        if change.vertices:
            _validate_vertices(change.vertices)


def _before_image(state: TwoTonePickState) -> TwoTonePickChange:
    return TwoTonePickChange(
        state.mask.copy(), state.threshold, state.sigma, state.smooth_method
    )
