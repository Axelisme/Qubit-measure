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

from zcu_tools.core.waveform_format import (
    MAX_ARB_WAVEFORM_SAMPLES as MAX_ARB_WAVEFORM_SAMPLES,
)
from zcu_tools.core.waveform_format import (
    WaveformFormatError,
    validate_samples,
)


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
    try:
        samples = validate_samples(time=time, idata=idata, qdata=qdata)
    except WaveformFormatError as exc:
        raise ArbWaveformError(
            str(exc), reason=exc.reason, data=dict(exc.data)
        ) from exc
    return ArbWaveformData(
        idata=samples.idata,
        qdata=np.zeros_like(samples.idata) if samples.qdata is None else samples.qdata,
        time=samples.time,
        recipe=recipe,
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


def peak_abs(idata: NDArray[np.float64], qdata: NDArray[np.float64]) -> float:
    """Return max |I + iQ| for nonempty, equal-shaped finite channels."""
    return float(np.max(np.hypot(idata, qdata)))
