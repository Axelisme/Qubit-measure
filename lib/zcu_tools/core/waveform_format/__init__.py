"""Stateless NPZ waveform samples and opaque JSON recipe format.

No path I/O, repository lookup, or formula interpretation is performed here.
Consumers read bytes and pass them to decode_npz. Resources alone interprets
formula recipes; program and simulation play the saved samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, NoReturn, TypedDict

import numpy as np
from numpy.typing import NDArray

MAX_ARB_WAVEFORM_SAMPLES = 1_000_001

WaveformErrorReason = Literal[
    "array_not_numeric",
    "array_not_1d",
    "array_non_finite",
    "array_shape_mismatch",
    "sample_count_too_small",
    "sample_count_too_large",
    "time_must_start_at_zero",
    "time_not_strictly_increasing",
    "amplitude_out_of_range",
    "unknown_npz_key",
    "missing_npz_key",
    "invalid_recipe_json",
    "npz_read_failed",
]


class WaveformErrorData(TypedDict, total=False):
    """Optional diagnostics for the failing format constraint.

    name identifies idata/qdata/time; ndim is its observed dimensionality.
    *_shape are observed array shapes. sample_count is the observed count,
    max_sample_count the upper limit. time0 is the first time in microseconds;
    peak_abs is the observed I/Q magnitude. path identifies recipe_json.
    unknown_keys/missing_keys list the offending NPZ key names.
    Only fields relevant to the reason are present.
    """

    name: str
    ndim: int
    idata_shape: tuple[int, ...]
    qdata_shape: tuple[int, ...]
    time_shape: tuple[int, ...]
    sample_count: int
    max_sample_count: int
    time0: float
    peak_abs: float
    path: str
    unknown_keys: list[str]
    missing_keys: list[str]


class WaveformFormatError(ValueError):
    """Malformed sample arrays, NPZ bytes, or recipe JSON.

    reason is a stable format classification. data contains only relevant
    typed diagnostics, not paths to assets or repository state. The decoder
    retains an archive/JSON cause when translating its failure.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: WaveformErrorReason,
        data: WaveformErrorData | None = None,
    ) -> None:
        """Describe a format failure without interpreting repository policy."""
        super().__init__(message)
        self.reason = reason
        self.data: WaveformErrorData = {} if data is None else data


@dataclass(frozen=True)
class WaveformSamples:
    """One isolated read of normalized finite one-dimensional float64 arrays.

    time is the reference axis in microseconds, starts at zero, strictly
    increases, and contains 2..1,000,001 samples. idata is normalized real I;
    qdata is real Q of the same length or None when absent/all zero.
    The magnitude hypot(I,Q) is at most one. Each validated read owns its
    arrays, but callers can modify them; asset bytes are never frozen.
    """

    time: NDArray[np.float64]
    idata: NDArray[np.float64]
    qdata: NDArray[np.float64] | None


@dataclass(frozen=True)
class WaveformFile:
    """Decoded samples and optional opaque recipe text.

    samples obeys WaveformSamples constraints. recipe_json is a scalar JSON
    string or None when no recipe was supplied. JSON syntax is validated,
    but segments, normalization, and formula semantics are not interpreted.
    """

    samples: WaveformSamples
    recipe_json: str | None = None


def validate_samples(*, time: object, idata: object, qdata: object) -> WaveformSamples:
    """Validate and isolate array-like time/I/Q using WaveformSamples rules.

    qdata=None means zero Q. Return None for absent or entirely zero Q.
    Raise WaveformFormatError for nonnumeric, nonfinite, dimensionality,
    length, reference time, sample count, or magnitude violations.
    The returned arrays do not share caller-owned storage.
    """
    time_arr = _isolate_array(time, name="time")
    idata_arr = _isolate_array(idata, name="idata")
    qdata_arr = None if qdata is None else _isolate_array(qdata, name="qdata")
    qdata_shape = idata_arr.shape if qdata_arr is None else qdata_arr.shape
    if idata_arr.shape != time_arr.shape or idata_arr.shape != qdata_shape:
        raise WaveformFormatError(
            "idata, qdata, and time must have the same shape",
            reason="array_shape_mismatch",
            data={
                "idata_shape": idata_arr.shape,
                "qdata_shape": qdata_shape,
                "time_shape": time_arr.shape,
            },
        )
    sample_count = int(time_arr.size)
    if sample_count < 2:
        raise WaveformFormatError(
            "arbitrary waveform must contain at least 2 samples",
            reason="sample_count_too_small",
            data={"sample_count": sample_count},
        )
    if sample_count > MAX_ARB_WAVEFORM_SAMPLES:
        raise WaveformFormatError(
            f"arbitrary waveform has {sample_count} samples, exceeding "
            f"{MAX_ARB_WAVEFORM_SAMPLES}",
            reason="sample_count_too_large",
            data={
                "sample_count": sample_count,
                "max_sample_count": MAX_ARB_WAVEFORM_SAMPLES,
            },
        )
    if time_arr[0] != 0.0:
        raise WaveformFormatError(
            "time[0] must be 0 us",
            reason="time_must_start_at_zero",
            data={"time0": float(time_arr[0])},
        )
    if not np.all(np.diff(time_arr) > 0):
        raise WaveformFormatError(
            "time array must be strictly increasing",
            reason="time_not_strictly_increasing",
        )
    magnitude = (
        np.abs(idata_arr) if qdata_arr is None else np.hypot(idata_arr, qdata_arr)
    )
    peak = float(np.max(magnitude))
    if peak > 1.0:
        raise WaveformFormatError(
            f"waveform magnitude peak ({peak:.6g}) exceeds 1.0",
            reason="amplitude_out_of_range",
            data={"peak_abs": peak},
        )
    if qdata_arr is not None and not np.any(qdata_arr):
        qdata_arr = None
    return WaveformSamples(time=time_arr, idata=idata_arr, qdata=qdata_arr)


def decode_npz(payload: bytes) -> WaveformFile:
    """Decode isolated samples from NPZ bytes with allow_pickle=False.

    Require idata/qdata/time; accept only optional recipe_json in addition.
    Validate arrays and scalar JSON syntax, not formula semantics. Reject
    unsupported/missing keys and malformed content with WaveformFormatError.
    No filesystem or repository access occurs.
    """
    try:
        loaded = np.load(BytesIO(payload), allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("Waveform payload must be an NPZ archive")
        with loaded as archive:
            keys = set(archive.files)
            extra = keys - {"idata", "qdata", "time", "recipe_json"}
            if extra:
                raise WaveformFormatError(
                    f"Unsupported NPZ key(s): {sorted(extra)}",
                    reason="unknown_npz_key",
                    data={"unknown_keys": sorted(extra)},
                )
            missing = {"idata", "qdata", "time"} - keys
            if missing:
                raise WaveformFormatError(
                    f"Missing required NPZ key(s): {sorted(missing)}",
                    reason="missing_npz_key",
                    data={"missing_keys": sorted(missing)},
                )
            recipe_json = None
            if "recipe_json" in keys:
                raw_recipe = archive["recipe_json"]
                if raw_recipe.shape != ():
                    raise WaveformFormatError(
                        "recipe_json must be a scalar JSON string",
                        reason="invalid_recipe_json",
                        data={"path": "recipe_json"},
                    )
                recipe_json = _validate_recipe_json(raw_recipe.item())
            samples = validate_samples(
                time=archive["time"], idata=archive["idata"], qdata=archive["qdata"]
            )
            return WaveformFile(samples=samples, recipe_json=recipe_json)
    except WaveformFormatError:
        raise
    except Exception as exc:
        raise WaveformFormatError(
            f"Failed to decode arbitrary waveform NPZ: {exc}",
            reason="npz_read_failed",
        ) from exc


def encode_npz(data: WaveformFile) -> bytes:
    """Validate samples/opaque JSON and encode idata/qdata/time to NPZ bytes.

    Always write qdata, including zero Q. Write recipe_json only when present.
    Preserve recipe text without interpreting it. Raise WaveformFormatError
    for invalid samples or JSON; do not write files or alter the input.
    """
    samples = validate_samples(
        time=data.samples.time, idata=data.samples.idata, qdata=data.samples.qdata
    )
    recipe_json = (
        None if data.recipe_json is None else _validate_recipe_json(data.recipe_json)
    )
    qdata = samples.qdata
    if qdata is None:
        qdata = np.zeros_like(samples.idata)
    stream = BytesIO()
    if recipe_json is None:
        np.savez(stream, idata=samples.idata, qdata=qdata, time=samples.time)
    else:
        np.savez(
            stream,
            idata=samples.idata,
            qdata=qdata,
            time=samples.time,
            recipe_json=np.array(recipe_json),
        )
    return stream.getvalue()


def _isolate_array(value: object, *, name: str) -> NDArray[np.float64]:
    """Convert a real array-like to owned float64 storage, rejecting complex input."""
    try:
        raw = np.asarray(value)
        # Object arrays can hide NumPy complex scalars that otherwise cast with a warning.
        has_object_complex = raw.dtype.kind == "O" and any(
            isinstance(item, (complex, np.complexfloating)) for item in raw.flat
        )
        if np.iscomplexobj(raw) or has_object_complex:
            raise ValueError("I/Q and time arrays must be real")
        arr = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WaveformFormatError(
            f"{name} must be a real numeric array",
            reason="array_not_numeric",
            data={"name": name},
        ) from exc
    if arr.ndim != 1:
        raise WaveformFormatError(
            f"{name} must be a 1D array",
            reason="array_not_1d",
            data={"name": name, "ndim": arr.ndim},
        )
    if not np.all(np.isfinite(arr)):
        raise WaveformFormatError(
            f"{name} must contain only finite values",
            reason="array_non_finite",
            data={"name": name},
        )
    return arr


def _reject_json_constant(constant: str) -> NoReturn:
    """Reject Python decoder extensions that are not part of JSON syntax."""
    raise ValueError(f"{constant} is not a valid JSON constant")


def _validate_recipe_json(value: object) -> str:
    """Validate opaque JSON text, without normalizing it or interpreting a recipe."""
    if not isinstance(value, str):
        raise WaveformFormatError(
            "recipe_json must be a scalar JSON string",
            reason="invalid_recipe_json",
            data={"path": "recipe_json"},
        )
    try:
        json.loads(value, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise WaveformFormatError(
            f"recipe_json is not valid JSON: {exc}",
            reason="invalid_recipe_json",
            data={"path": "recipe_json"},
        ) from exc
    return value
