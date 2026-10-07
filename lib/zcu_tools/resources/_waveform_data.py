"""Waveform data carriers, payload validation and preview arithmetic.

Recipe annotations are type-only; formula parsing and rendering depend on this
validated array layer, not the other way around.
"""

from __future__ import annotations

import math
from collections.abc import Buffer
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._waveform_formula import FormulaRecipe

MAX_ARB_WAVEFORM_SAMPLES = 1_000_001


class ArbWaveformError(ValueError):
    """Expected arbitrary-waveform validation/storage failure."""

    def __init__(
        self, message: str, *, reason: str, data: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.data = data or {}


@dataclass(frozen=True)
class ArbWaveformData:
    idata: NDArray[np.float64]
    qdata: NDArray[np.float64]
    time: NDArray[np.float64]
    recipe: FormulaRecipe | None = None

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def peak_abs(self) -> float:
        return peak_abs(self.idata, self.qdata)

    @property
    def has_q(self) -> bool:
        return bool(np.any(self.qdata != 0))


@dataclass(frozen=True)
class ArbWaveformListEntry:
    data_key: str
    mtime: float
    file_size: int


@dataclass(frozen=True)
class ArbWaveformInfo:
    data_key: str
    duration: float
    sample_count: int
    has_q: bool
    peak_abs: float
    has_recipe: bool
    mtime: float
    file_size: int
    recipe: FormulaRecipe | None = None


@dataclass(frozen=True)
class ArbWaveformPreview:
    """Peak-normalized (optional) I/Q/Abs series ready for preview rendering.

    Keeps GUI and agent-PNG callers consistent without each reimplementing the
    normalization arithmetic.  All arrays share the same sample count as the
    source ArbWaveformData. Preview series generation belongs to this module.
    """

    time: NDArray[np.float64]
    idata: NDArray[np.float64]
    qdata: NDArray[np.float64]
    abs_data: NDArray[np.float64]


def prepare_preview_series(
    data: ArbWaveformData, *, normalize: bool
) -> ArbWaveformPreview:
    """Peak-normalize (optional) and derive I/Q/Abs series for preview rendering.

    When normalize=True and peak>0, each channel is divided by the peak |IQ|.
    When peak==0 the raw (all-zero) data is returned unchanged to avoid division
    by zero.  abs_data is always computed from the (possibly normalized) I/Q.
    """
    idata = np.asarray(data.idata, dtype=np.float64)
    qdata = np.asarray(data.qdata, dtype=np.float64)
    if normalize:
        peak = float(np.max(np.hypot(idata, qdata)))
        if peak > 0.0:
            idata = idata / peak
            qdata = qdata / peak
    abs_data = np.hypot(idata, qdata)
    return ArbWaveformPreview(
        time=np.asarray(data.time, dtype=np.float64),
        idata=idata,
        qdata=qdata,
        abs_data=abs_data,
    )


@dataclass(frozen=True)
class FormulaValidationResult:
    ok: bool
    data: ArbWaveformData | None = None
    reason: str | None = None
    message: str | None = None
    error_data: dict[str, object] | None = None


def validate_payload(
    *,
    idata: object,
    qdata: object,
    time: object,
    recipe: FormulaRecipe | None,
) -> ArbWaveformData:
    idata_arr = _coerce_1d_float_array(idata, name="idata")
    qdata_arr = _coerce_1d_float_array(qdata, name="qdata")
    time_arr = _coerce_1d_float_array(time, name="time")

    if idata_arr.shape != qdata_arr.shape or idata_arr.shape != time_arr.shape:
        raise ArbWaveformError(
            "idata, qdata, and time must have the same shape",
            reason="array_shape_mismatch",
            data={
                "idata_shape": idata_arr.shape,
                "qdata_shape": qdata_arr.shape,
                "time_shape": time_arr.shape,
            },
        )
    sample_count = int(time_arr.size)
    if sample_count < 2:
        raise ArbWaveformError(
            "arbitrary waveform must contain at least 2 samples",
            reason="sample_count_too_small",
            data={"sample_count": sample_count},
        )
    if sample_count > MAX_ARB_WAVEFORM_SAMPLES:
        raise ArbWaveformError(
            f"arbitrary waveform has {sample_count} samples, exceeding "
            f"{MAX_ARB_WAVEFORM_SAMPLES}",
            reason="sample_count_too_large",
            data={
                "sample_count": sample_count,
                "max_sample_count": MAX_ARB_WAVEFORM_SAMPLES,
            },
        )
    if not np.isclose(time_arr[0], 0.0, rtol=0.0, atol=0.0):
        raise ArbWaveformError(
            "time[0] must be 0 us",
            reason="time_must_start_at_zero",
            data={"time0": float(time_arr[0])},
        )
    if not np.all(np.diff(time_arr) > 0):
        raise ArbWaveformError(
            "time array must be strictly increasing",
            reason="time_not_strictly_increasing",
        )
    peak = peak_abs(idata_arr, qdata_arr)
    if peak > 1.0:
        raise ArbWaveformError(
            f"waveform magnitude peak ({peak:.6g}) exceeds 1.0",
            reason="amplitude_out_of_range",
            data={"peak_abs": peak},
        )
    return ArbWaveformData(
        idata=idata_arr, qdata=qdata_arr, time=time_arr, recipe=recipe
    )


def coerce_positive_float(value: object, *, path: str) -> float:
    """Convert a recipe number to a finite positive float, rejecting bool.

    path identifies the recipe field in invalid_recipe error details. Text,
    buffers and numeric conversion protocols retain Python float semantics.
    """
    if isinstance(value, bool):
        raise ArbWaveformError(
            f"{path} must be a positive number",
            reason="invalid_recipe",
            data={"path": path},
        )
    try:
        # float accepts text, buffers and the two numeric conversion protocols.
        if not isinstance(value, (str, Buffer, SupportsFloat, SupportsIndex)):
            raise TypeError(
                "float() argument must be a string or a real number, "
                f"not '{type(value).__name__}'"
            )
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ArbWaveformError(
            f"{path} must be a positive number",
            reason="invalid_recipe",
            data={"path": path},
        ) from exc
    if not math.isfinite(coerced) or coerced <= 0:
        raise ArbWaveformError(
            f"{path} must be > 0",
            reason="invalid_recipe",
            data={"path": path, "value": coerced},
        )
    return coerced


def _coerce_1d_float_array(value: object, *, name: str) -> NDArray[np.float64]:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArbWaveformError(
            f"{name} must be a numeric array",
            reason="array_not_numeric",
            data={"name": name},
        ) from exc
    if arr.ndim != 1:
        raise ArbWaveformError(
            f"{name} must be a 1D array",
            reason="array_not_1d",
            data={"name": name, "ndim": arr.ndim},
        )
    if not np.all(np.isfinite(arr)):
        raise ArbWaveformError(
            f"{name} must contain only finite values",
            reason="array_non_finite",
            data={"name": name},
        )
    return arr.astype(np.float64, copy=False)


def peak_abs(idata: NDArray[np.float64], qdata: NDArray[np.float64]) -> float:
    """Return max |I + iQ| for nonempty, equal-shaped finite channels."""
    return float(np.max(np.hypot(idata, qdata)))
