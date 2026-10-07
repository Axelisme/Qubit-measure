"""Canonical arbitrary waveform API and scope binding. See waveform_assets.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zcu_tools.cfg_model import WaveformAssetInfo, WaveformSamples

from ._waveform_data import (
    MAX_ARB_WAVEFORM_SAMPLES,
    ArbWaveformData,
    ArbWaveformError,
    ArbWaveformInfo,
    ArbWaveformListEntry,
    ArbWaveformPreview,
    FormulaValidationResult,
    prepare_preview_series,
    validate_payload,
)
from ._waveform_formula import (
    ARB_WAVEFORM_RENDER_SAMPLES_PER_US,
    FormulaRecipe,
    FormulaSegment,
    render_formula_recipe,
)
from ._waveform_npz import load_npz, write_npz

_DATA_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


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
    def bound_info(cls) -> WaveformAssetInfo:
        """Bind asset reads to the currently initialized repository directory.

        Raises RuntimeError if init() has not set an explicit scope. A later
        init() does not change this binding. duration() and samples() load
        current asset bytes on every call and propagate ArbWaveformError for
        missing keys, invalid archives or I/O failures. No data is cached.
        """
        return cls.info_at(cls._check_initialized())

    @staticmethod
    def info_at(directory: Path) -> WaveformAssetInfo:
        """Bind asset reads to an explicit directory without changing global scope.

        directory is the asset folder, not the entry's database root. Binding
        resolves it to an absolute path but creates no directory and reads no
        archive. Later duration/samples calls load current bytes and propagate
        ArbWaveformError for missing/invalid assets or I/O failure.
        """
        return _BoundWaveformAssetInfo(directory.resolve())

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
        return _load_asset(cls.path_for(data_key))

    @classmethod
    def get(
        cls, name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64]]:
        """Read isolated 1D (I, Q, time) arrays, with time in microseconds.

        Q is None for zero Q; missing/invalid assets or I/O raise ArbWaveformError.
        """
        samples = cls.bound_info().samples(name)
        return samples.idata, samples.qdata, samples.time

    @classmethod
    def load_recipe(cls, data_key: str) -> FormulaRecipe | None:
        return cls.load(data_key).recipe

    @classmethod
    def inspect(cls, data_key: str) -> ArbWaveformInfo:
        path = cls.path_for(data_key)
        data = _load_asset(path)
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
        idata: ArrayLike,
        time: ArrayLike,
        qdata: ArrayLike | None = None,
    ) -> None:
        """Overwrite a named raw asset with normalized I/Q arrays and time in us.

        Arrays must be finite, one-dimensional and equal in length. Time starts
        at zero and increases strictly; I/Q magnitude cannot exceed one. Omitted
        qdata becomes zeros. This writes no recipe and returns None.
        ArbWaveformError reports an invalid key or payload; I/O errors propagate.
        For an explicit collision policy, use import_data instead.
        """
        qdata_save = np.zeros_like(idata) if qdata is None else qdata
        cls.import_data(name, idata=idata, qdata=qdata_save, time=time, overwrite=True)

    @classmethod
    def import_data(
        cls,
        data_key: str,
        *,
        idata: ArrayLike,
        qdata: ArrayLike | None,
        time: ArrayLike,
        overwrite: bool = False,
    ) -> ArbWaveformInfo:
        """Write raw I/Q arrays and return metadata from the saved asset.

        data_key names one asset. Arrays must be finite, one-dimensional and
        equal in length, with 2 to MAX_ARB_WAVEFORM_SAMPLES elements. Time is in
        us, starts at zero and increases strictly. I/Q magnitude cannot exceed
        one; qdata=None supplies zeros. No recipe is stored.
        An existing key raises ArbWaveformError unless overwrite=True. Invalid
        keys or payloads also raise ArbWaveformError; I/O errors propagate.
        Metadata inspection can fail after the write has completed.
        """
        path = cls.path_for(data_key)
        if path.exists() and not overwrite:
            raise ArbWaveformError(
                f"Arbitrary waveform {data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": data_key},
            )
        qdata_save = np.zeros_like(idata) if qdata is None else qdata
        data = validate_payload(idata=idata, qdata=qdata_save, time=time, recipe=None)
        write_npz(path, data)
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
        data = load_npz(source)
        write_npz(path, data)
        return cls.inspect(data_key)

    @classmethod
    def create_from_formula(
        cls,
        data_key: str,
        recipe: FormulaRecipe | dict[str, object],
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist a recipe and its arrays; inspect/preview are separate reads."""
        path = cls.path_for(data_key)
        if path.exists() and not overwrite:
            raise ArbWaveformError(
                f"Arbitrary waveform {data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": data_key},
            )
        data = render_formula_recipe(recipe)
        write_npz(path, data)

    @classmethod
    def update_formula(
        cls, data_key: str, recipe: FormulaRecipe | dict[str, object]
    ) -> None:
        """Replace an existing recipe asset without a post-write inspection."""
        path = cls.path_for(data_key)
        if not path.exists():
            raise _asset_not_found(data_key, path)
        data = render_formula_recipe(recipe)
        write_npz(path, data)

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
            raise _asset_not_found(data_key, path)
        path.unlink()

    @classmethod
    def rename(cls, old_data_key: str, new_data_key: str) -> None:
        old_path = cls.path_for(old_data_key)
        new_path = cls.path_for(new_data_key)
        if not old_path.exists():
            raise _asset_not_found(old_data_key, old_path)
        if new_path.exists():
            raise ArbWaveformError(
                f"Arbitrary waveform {new_data_key!r} already exists",
                reason="data_key_exists",
                data={"data_key": new_data_key},
            )
        old_path.rename(new_path)


def _load_asset(path: Path) -> ArbWaveformData:
    """Read a validated-key asset path, preserving owner missing-key errors."""
    if not path.exists():
        raise _asset_not_found(path.stem, path)
    return load_npz(path)


def _asset_not_found(data_key: str, path: Path) -> ArbWaveformError:
    available = sorted(
        entry.stem
        for entry in path.parent.glob("*.npz")
        if _DATA_KEY_RE.fullmatch(entry.stem)
    )
    return ArbWaveformError(
        f"Arbitrary waveform {data_key!r} not found at {path}. "
        f"Available waveforms: {available}",
        reason="data_key_not_found",
        data={"data_key": data_key, "available": available},
    )


@dataclass(frozen=True)
class _BoundWaveformAssetInfo:
    """Asset owner adapter with an absolute directory, never cached bytes."""

    directory: Path

    def duration(self, data_key: str) -> float:
        """Read current asset duration in microseconds, propagating read errors."""
        return float(self.samples(data_key).time[-1])

    def samples(self, data_key: str) -> WaveformSamples:
        """Read validated, isolated arrays from this binding's fixed directory."""
        _validate_data_key(data_key)
        data = _load_asset(self.directory / f"{data_key}.npz")
        return WaveformSamples(
            time=data.time,
            idata=data.idata,
            qdata=data.qdata if data.has_q else None,
        )


def _validate_data_key(data_key: object) -> None:
    if not isinstance(data_key, str) or not _DATA_KEY_RE.fullmatch(data_key):
        raise ArbWaveformError(
            "Arbitrary waveform data key must match "
            r"^[A-Za-z][A-Za-z0-9_]*$",
            reason="invalid_data_key",
            data={"data_key": data_key},
        )


__all__ = [
    "ARB_WAVEFORM_RENDER_SAMPLES_PER_US",
    "MAX_ARB_WAVEFORM_SAMPLES",
    "ArbWaveformData",
    "ArbWaveformDatabase",
    "ArbWaveformError",
    "ArbWaveformInfo",
    "ArbWaveformListEntry",
    "ArbWaveformPreview",
    "FormulaRecipe",
    "FormulaSegment",
    "FormulaValidationResult",
    "prepare_preview_series",
    "render_formula_recipe",
    "validate_payload",
]
