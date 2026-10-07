"""NPZ archive reads, atomic writes and embedded recipe JSON decoding."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._waveform_data import ArbWaveformData, ArbWaveformError, validate_payload
from ._waveform_formula import FormulaRecipe

_NPZ_KEYS = frozenset({"idata", "qdata", "time", "recipe_json"})


def load_npz(path: Path) -> ArbWaveformData:
    """Read and validate playback arrays and optional recipe from an NPZ path.

    Reject unknown keys and malformed payloads with ArbWaveformError; translate
    archive and I/O failures to npz_read_failed while preserving their cause.
    """
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


def write_npz(path: Path, data: ArbWaveformData) -> None:
    """Replace an NPZ path with validated data and its optional recipe JSON.

    Create the parent directory and use a same-directory temporary file. On
    write failure remove that temporary file and propagate the original error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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
            if data.recipe is None:
                np.savez(tmp, idata=data.idata, qdata=data.qdata, time=data.time)
            else:
                np.savez(
                    tmp,
                    idata=data.idata,
                    qdata=data.qdata,
                    time=data.time,
                    recipe_json=np.array(data.recipe.to_json()),
                )
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
