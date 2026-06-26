from __future__ import annotations

import ast
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike, NDArray
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

MAX_ARB_WAVEFORM_SAMPLES = 1_000_001
ARB_WAVEFORM_RENDER_SAMPLES_PER_US = 1000.0

_DATA_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ALLOWED_FUNCTIONS = frozenset(
    {"sin", "cos", "tan", "exp", "sqrt", "Abs", "abs", "erf"}
)
_ALLOWED_NAMES = frozenset({"t", "T", "pi", "e", "I", *_ALLOWED_FUNCTIONS})
_NPZ_KEYS = frozenset({"idata", "qdata", "time", "recipe_json"})


NormalizeMode = Literal["none", "peak"]


class ArbWaveformError(ValueError):
    """Expected arbitrary-waveform validation/storage failure."""

    def __init__(
        self, message: str, *, reason: str, data: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.data = data or {}


@dataclass(frozen=True)
class FormulaSegment:
    duration: float
    formula: str

    @classmethod
    def from_raw(cls, raw: object, *, index: int) -> FormulaSegment:
        if not isinstance(raw, dict):
            raise ArbWaveformError(
                f"segments[{index}] must be an object",
                reason="invalid_recipe",
                data={"path": f"segments[{index}]"},
            )
        extra = set(raw) - {"duration", "formula"}
        if extra:
            raise ArbWaveformError(
                f"segments[{index}] has unsupported key(s): {sorted(extra)}",
                reason="invalid_recipe",
                data={"path": f"segments[{index}]", "unknown_keys": sorted(extra)},
            )
        if "duration" not in raw:
            raise ArbWaveformError(
                f"segments[{index}].duration is required",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].duration"},
            )
        if "formula" not in raw:
            raise ArbWaveformError(
                f"segments[{index}].formula is required",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].formula"},
            )
        duration = _coerce_positive_float(
            raw["duration"], path=f"segments[{index}].duration"
        )
        formula = raw["formula"]
        if not isinstance(formula, str):
            raise ArbWaveformError(
                f"segments[{index}].formula must be a string",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].formula"},
            )
        if not formula.strip():
            raise ArbWaveformError(
                f"segments[{index}].formula must not be empty",
                reason="invalid_recipe",
                data={"path": f"segments[{index}].formula"},
            )
        return cls(duration=duration, formula=formula)

    def to_dict(self) -> dict[str, object]:
        return {"duration": self.duration, "formula": self.formula}


@dataclass(frozen=True)
class FormulaRecipe:
    segments: tuple[FormulaSegment, ...]
    normalize: NormalizeMode

    @classmethod
    def from_raw(cls, raw: object) -> FormulaRecipe:
        if isinstance(raw, FormulaRecipe):
            return raw
        if not isinstance(raw, dict):
            raise ArbWaveformError(
                "recipe must be an object",
                reason="invalid_recipe",
                data={"path": ""},
            )
        extra = set(raw) - {"segments", "normalize"}
        if extra:
            raise ArbWaveformError(
                f"recipe has unsupported key(s): {sorted(extra)}",
                reason="invalid_recipe",
                data={"path": "", "unknown_keys": sorted(extra)},
            )
        if "segments" not in raw:
            raise ArbWaveformError(
                "recipe.segments is required",
                reason="invalid_recipe",
                data={"path": "segments"},
            )
        if "normalize" not in raw:
            raise ArbWaveformError(
                "recipe.normalize is required",
                reason="invalid_recipe",
                data={"path": "normalize"},
            )
        segments_raw = raw["segments"]
        if not isinstance(segments_raw, list):
            raise ArbWaveformError(
                "recipe.segments must be a list",
                reason="invalid_recipe",
                data={"path": "segments"},
            )
        if not segments_raw:
            raise ArbWaveformError(
                "recipe.segments must contain at least one segment",
                reason="invalid_recipe",
                data={"path": "segments"},
            )
        normalize = raw["normalize"]
        if normalize not in ("none", "peak"):
            raise ArbWaveformError(
                "recipe.normalize must be 'none' or 'peak'",
                reason="invalid_recipe",
                data={"path": "normalize"},
            )
        return cls(
            segments=tuple(
                FormulaSegment.from_raw(item, index=index)
                for index, item in enumerate(segments_raw)
            ),
            normalize=normalize,
        )

    @classmethod
    def from_json(cls, text: str) -> FormulaRecipe:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArbWaveformError(
                f"recipe_json is not valid JSON: {exc.msg}",
                reason="invalid_recipe_json",
                data={"pos": exc.pos},
            ) from exc
        try:
            return cls.from_raw(raw)
        except ArbWaveformError as exc:
            if exc.reason == "invalid_recipe":
                raise ArbWaveformError(
                    str(exc), reason="invalid_recipe_json", data=exc.data
                ) from exc
            raise

    def to_dict(self) -> dict[str, object]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "normalize": self.normalize,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


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
        return _peak_abs(self.idata, self.qdata)

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
    source ArbWaveformData.  ADR-0034: preview series generation is domain logic.
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


class ArbWaveformDatabase:
    """Qubit-scoped single-file arbitrary waveform asset repository."""

    _database_path: ClassVar[Path | None] = None

    @classmethod
    def init(cls, path: str | Path) -> None:
        """Set and create the database directory for arbitrary waveforms."""
        cls._database_path = Path(path)
        cls._database_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _check_initialized(cls) -> Path:
        if cls._database_path is None:
            raise RuntimeError(
                "ArbWaveformDatabase is not initialized. "
                "Call ArbWaveformDatabase.init(path) first."
            )
        cls._database_path.mkdir(parents=True, exist_ok=True)
        return cls._database_path

    @classmethod
    def path_for(cls, data_key: str) -> Path:
        db_path = cls._check_initialized()
        _validate_data_key(data_key)
        return db_path / f"{data_key}.npz"

    @classmethod
    def exists(cls, data_key: str) -> bool:
        return cls.path_for(data_key).exists()

    @classmethod
    def list(cls) -> list[str]:
        """List all available data keys without opening each archive."""
        return [entry.data_key for entry in cls.list_entries()]

    @classmethod
    def list_entries(cls) -> list[ArbWaveformListEntry]:
        """List cheap directory entries without opening each `.npz` payload."""
        db_path = cls._check_initialized()
        entries: list[ArbWaveformListEntry] = []
        for path in db_path.glob("*.npz"):
            data_key = path.stem
            if not _DATA_KEY_RE.fullmatch(data_key):
                continue
            stat = path.stat()
            entries.append(
                ArbWaveformListEntry(
                    data_key=data_key, mtime=stat.st_mtime, file_size=stat.st_size
                )
            )
        return sorted(entries, key=lambda item: item.data_key)

    @classmethod
    def load(cls, data_key: str) -> ArbWaveformData:
        path = cls.path_for(data_key)
        if not path.exists():
            raise cls._not_found(data_key, path)
        return _load_npz(path)

    @classmethod
    def get(cls, name: str) -> tuple[NDArray, NDArray | None, NDArray]:
        """Load a waveform by key for the program layer.

        Returns qdata as None when the stored Q channel is all zero, preserving the
        old runtime contract while using the new single-file asset layout.
        """
        data = cls.load(name)
        qdata_out: NDArray | None = None if not data.has_q else data.qdata
        return data.idata, qdata_out, data.time

    @classmethod
    def load_recipe(cls, data_key: str) -> FormulaRecipe | None:
        return cls.load(data_key).recipe

    @classmethod
    def inspect(cls, data_key: str) -> ArbWaveformInfo:
        path = cls.path_for(data_key)
        if not path.exists():
            raise cls._not_found(data_key, path)
        data = _load_npz(path)
        stat = path.stat()
        return ArbWaveformInfo(
            data_key=data_key,
            duration=data.duration,
            sample_count=int(data.time.size),
            has_q=data.has_q,
            peak_abs=data.peak_abs,
            has_recipe=data.recipe is not None,
            mtime=stat.st_mtime,
            file_size=stat.st_size,
            recipe=data.recipe,
        )

    @classmethod
    def save(
        cls,
        name: str,
        idata: NDArray,
        time: NDArray,
        qdata: NDArray | None = None,
    ) -> None:
        """Notebook-friendly raw save wrapper.

        For the explicit collision policy use `import_data(..., overwrite=...)`.
        This historical convenience API overwrites the named raw asset.
        """
        qdata_save = np.zeros_like(idata) if qdata is None else qdata
        cls.import_data(name, idata=idata, qdata=qdata_save, time=time, overwrite=True)

    @classmethod
    def import_data(
        cls,
        data_key: str,
        *,
        idata: NDArray,
        qdata: NDArray | None,
        time: NDArray,
        overwrite: bool = False,
    ) -> ArbWaveformInfo:
        path = cls.path_for(data_key)
        if path.exists() and not overwrite:
            raise ArbWaveformError(
                f"Arbitrary waveform {data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": data_key},
            )
        qdata_save = np.zeros_like(idata) if qdata is None else qdata
        data = validate_payload(idata=idata, qdata=qdata_save, time=time, recipe=None)
        _write_npz(path, data)
        return cls.inspect(data_key)

    @classmethod
    def import_file(
        cls, data_key: str, source_path: str | Path, *, overwrite: bool = False
    ) -> ArbWaveformInfo:
        source = Path(source_path)
        if source.suffix.lower() != ".npz":
            raise ArbWaveformError(
                "Only .npz arbitrary waveform imports are supported",
                reason="unsupported_import_format",
                data={"path": str(source)},
            )
        path = cls.path_for(data_key)
        if path.exists() and not overwrite:
            raise ArbWaveformError(
                f"Arbitrary waveform {data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": data_key},
            )
        data = _load_npz(source)
        _write_npz(path, data)
        return cls.inspect(data_key)

    @classmethod
    def create_from_formula(
        cls,
        data_key: str,
        recipe: FormulaRecipe | dict[str, object],
        *,
        overwrite: bool = False,
    ) -> ArbWaveformInfo:
        path = cls.path_for(data_key)
        if path.exists() and not overwrite:
            raise ArbWaveformError(
                f"Arbitrary waveform {data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": data_key},
            )
        data = render_formula_recipe(recipe)
        _write_npz(path, data)
        return cls.inspect(data_key)

    @classmethod
    def update_formula(
        cls, data_key: str, recipe: FormulaRecipe | dict[str, object]
    ) -> ArbWaveformInfo:
        path = cls.path_for(data_key)
        if not path.exists():
            raise cls._not_found(data_key, path)
        data = render_formula_recipe(recipe)
        _write_npz(path, data)
        return cls.inspect(data_key)

    @classmethod
    def validate_formula(
        cls, recipe: FormulaRecipe | dict[str, object]
    ) -> FormulaValidationResult:
        try:
            data = render_formula_recipe(recipe)
        except ArbWaveformError as exc:
            return FormulaValidationResult(
                ok=False,
                reason=exc.reason,
                message=str(exc),
                error_data=exc.data,
            )
        return FormulaValidationResult(ok=True, data=data)

    @classmethod
    def delete(cls, data_key: str) -> None:
        path = cls.path_for(data_key)
        if not path.exists():
            raise cls._not_found(data_key, path)
        path.unlink()

    @classmethod
    def rename(cls, old_data_key: str, new_data_key: str) -> None:
        old_path = cls.path_for(old_data_key)
        new_path = cls.path_for(new_data_key)
        if not old_path.exists():
            raise cls._not_found(old_data_key, old_path)
        if new_path.exists():
            raise ArbWaveformError(
                f"Arbitrary waveform {new_data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": new_data_key},
            )
        old_path.rename(new_path)

    @classmethod
    def _not_found(cls, data_key: str, path: Path) -> ArbWaveformError:
        available = cls.list()
        return ArbWaveformError(
            f"Arbitrary waveform {data_key!r} not found at {path}. "
            f"Available waveforms: {available}",
            reason="data_key_not_found",
            data={"data_key": data_key, "available": available},
        )


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
    peak = _peak_abs(idata_arr, qdata_arr)
    if peak > 1.0:
        raise ArbWaveformError(
            f"waveform magnitude peak ({peak:.6g}) exceeds 1.0",
            reason="amplitude_out_of_range",
            data={"peak_abs": peak},
        )
    return ArbWaveformData(
        idata=idata_arr, qdata=qdata_arr, time=time_arr, recipe=recipe
    )


def render_formula_recipe(recipe: FormulaRecipe | dict[str, object]) -> ArbWaveformData:
    parsed = FormulaRecipe.from_raw(recipe)
    total_duration = math.fsum(segment.duration for segment in parsed.segments)
    sample_count = round(total_duration * ARB_WAVEFORM_RENDER_SAMPLES_PER_US) + 1
    if sample_count < 2:
        raise ArbWaveformError(
            "formula duration is too short for the internal render resolution",
            reason="sample_count_too_small",
            data={"sample_count": sample_count, "total_duration": total_duration},
        )
    if sample_count > MAX_ARB_WAVEFORM_SAMPLES:
        raise ArbWaveformError(
            f"formula render would produce {sample_count} samples, exceeding "
            f"{MAX_ARB_WAVEFORM_SAMPLES}",
            reason="sample_count_too_large",
            data={
                "sample_count": sample_count,
                "max_sample_count": MAX_ARB_WAVEFORM_SAMPLES,
                "total_duration": total_duration,
            },
        )

    time = np.linspace(0.0, total_duration, sample_count, dtype=np.float64)
    boundaries = np.cumsum(
        [segment.duration for segment in parsed.segments], dtype=np.float64
    )
    segment_index = np.searchsorted(boundaries, time, side="right")
    segment_index = np.minimum(segment_index, len(parsed.segments) - 1)
    starts = np.concatenate(([0.0], boundaries[:-1]))

    idata = np.zeros(sample_count, dtype=np.float64)
    qdata = np.zeros(sample_count, dtype=np.float64)
    for index, segment in enumerate(parsed.segments):
        mask = segment_index == index
        if not np.any(mask):
            continue
        local_t = time[mask] - starts[index]
        values = _evaluate_segment(
            segment.formula, local_t=local_t, global_t=time[mask]
        )
        try:
            values = np.broadcast_to(values, local_t.shape)
        except ValueError as exc:
            raise ArbWaveformError(
                f"segments[{index}].formula output cannot broadcast to segment samples",
                reason="formula_shape_mismatch",
                data={
                    "path": f"segments[{index}].formula",
                    "segment_index": index,
                    "output_shape": values.shape,
                    "sample_shape": local_t.shape,
                },
            ) from exc
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ArbWaveformError(
                f"segments[{index}].formula produced non-finite values",
                reason="formula_non_finite",
                data={"path": f"segments[{index}].formula", "segment_index": index},
            )
        idata[mask] = values.real.astype(np.float64, copy=False)
        qdata[mask] = values.imag.astype(np.float64, copy=False)

    peak = _peak_abs(idata, qdata)
    if parsed.normalize == "peak" and peak > 0.0:
        idata = idata / peak
        qdata = qdata / peak

    return validate_payload(idata=idata, qdata=qdata, time=time, recipe=parsed)


def _load_npz(path: Path) -> ArbWaveformData:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            extra = keys - _NPZ_KEYS
            if extra:
                raise ArbWaveformError(
                    f"{path} contains unsupported key(s): {sorted(extra)}",
                    reason="unknown_npz_key",
                    data={"path": str(path), "unknown_keys": sorted(extra)},
                )
            missing = {"idata", "qdata", "time"} - keys
            if missing:
                raise ArbWaveformError(
                    f"{path} is missing required key(s): {sorted(missing)}",
                    reason="missing_npz_key",
                    data={"path": str(path), "missing_keys": sorted(missing)},
                )
            recipe = None
            if "recipe_json" in keys:
                recipe = FormulaRecipe.from_json(
                    _read_recipe_json(archive["recipe_json"])
                )
            return validate_payload(
                idata=archive["idata"],
                qdata=archive["qdata"],
                time=archive["time"],
                recipe=recipe,
            )
    except ArbWaveformError:
        raise
    except Exception as exc:
        raise ArbWaveformError(
            f"Failed to read arbitrary waveform asset {path}: {exc}",
            reason="npz_read_failed",
            data={"path": str(path)},
        ) from exc


def _write_npz(path: Path, data: ArbWaveformData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, ArrayLike] = {
        "idata": data.idata,
        "qdata": data.qdata,
        "time": data.time,
    }
    if data.recipe is not None:
        payload["recipe_json"] = np.array(data.recipe.to_json())
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            np.savez(tmp, **payload)
        Path(tmp_name).replace(path)
    except Exception:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _read_recipe_json(raw: NDArray[Any]) -> str:
    if raw.shape != ():
        raise ArbWaveformError(
            "recipe_json must be a scalar JSON string",
            reason="invalid_recipe_json",
            data={"path": "recipe_json"},
        )
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ArbWaveformError(
            "recipe_json must be a scalar JSON string",
            reason="invalid_recipe_json",
            data={"path": "recipe_json"},
        )
    return value


def _validate_data_key(data_key: str) -> None:
    if not isinstance(data_key, str) or not _DATA_KEY_RE.fullmatch(data_key):
        raise ArbWaveformError(
            "Arbitrary waveform data key must match "
            r"^[A-Za-z][A-Za-z0-9_]*$",
            reason="invalid_data_key",
            data={"data_key": data_key},
        )


def _coerce_positive_float(value: object, *, path: str) -> float:
    if isinstance(value, bool):
        raise ArbWaveformError(
            f"{path} must be a positive number",
            reason="invalid_recipe",
            data={"path": path},
        )
    try:
        coerced = float(value)  # type: ignore[arg-type]
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


def _peak_abs(idata: NDArray[np.float64], qdata: NDArray[np.float64]) -> float:
    return float(np.max(np.hypot(idata, qdata)))


def _evaluate_segment(
    formula: str, *, local_t: NDArray[np.float64], global_t: NDArray[np.float64]
) -> NDArray[np.complex128]:
    expr = _parse_formula(formula)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    func = sp.lambdify(
        (_SYM_T, _SYM_GLOBAL_T),
        expr,
        modules=[{"Abs": np.abs, "erf": erf_vec}, "numpy"],
    )
    try:
        raw = func(local_t, global_t)
    except Exception as exc:
        raise ArbWaveformError(
            f"formula evaluation failed: {exc}",
            reason="formula_evaluation_failed",
            data={"formula": formula},
        ) from exc
    try:
        arr = np.asarray(raw, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ArbWaveformError(
            "formula output must be numeric",
            reason="formula_not_numeric",
            data={"formula": formula},
        ) from exc
    return arr


_SYM_T = sp.Symbol("t")
_SYM_GLOBAL_T = sp.Symbol("T")
_LOCAL_DICT: dict[str, object] = {
    "t": _SYM_T,
    "T": _SYM_GLOBAL_T,
    "pi": sp.pi,
    "e": sp.E,
    "I": sp.I,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "erf": sp.erf,
}
_GLOBAL_DICT: dict[str, object] = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
}


def _parse_formula(formula: str) -> sp.Expr:
    stripped = formula.strip()
    _validate_formula_ast(stripped)
    try:
        expr = parse_expr(
            stripped,
            local_dict=_LOCAL_DICT,
            global_dict=_GLOBAL_DICT,
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise ArbWaveformError(
            f"formula parse failed: {exc}",
            reason="formula_parse_failed",
            data={"formula": formula},
        ) from exc
    if not isinstance(expr, sp.Expr):
        raise ArbWaveformError(
            "formula must parse to a SymPy expression",
            reason="formula_parse_failed",
            data={"formula": formula},
        )
    symbols = {str(symbol) for symbol in expr.free_symbols}
    unsupported = symbols - {"t", "T"}
    if unsupported:
        raise ArbWaveformError(
            f"formula uses unsupported symbol(s): {sorted(unsupported)}",
            reason="formula_unknown_symbol",
            data={"symbols": sorted(unsupported)},
        )
    if expr.has(sp.Piecewise) or expr.atoms(sp.core.relational.Relational):
        raise ArbWaveformError(
            "formula conditionals and comparisons are not supported",
            reason="formula_conditional_not_supported",
            data={"formula": formula},
        )
    return expr


def _validate_formula_ast(formula: str) -> None:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ArbWaveformError(
            f"formula syntax error: {exc.msg}",
            reason="formula_parse_failed",
            data={"formula": formula},
        ) from exc
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.UAdd,
                ast.USub,
            ),
        ):
            continue
        raise ArbWaveformError(
            f"formula syntax is not supported: {type(node).__name__}",
            reason="formula_syntax_not_supported",
            data={"node": type(node).__name__},
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ArbWaveformError(
                f"formula uses unsupported name {node.id!r}",
                reason="formula_unknown_symbol",
                data={"symbol": node.id},
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCTIONS
            ):
                name = getattr(node.func, "id", type(node.func).__name__)
                raise ArbWaveformError(
                    f"formula uses unsupported function {name!r}",
                    reason="formula_unknown_function",
                    data={"function": name},
                )
            if node.keywords:
                raise ArbWaveformError(
                    "formula function calls do not support keyword arguments",
                    reason="formula_syntax_not_supported",
                    data={"node": "keyword"},
                )
