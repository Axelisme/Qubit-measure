"""NPZ archive reads, atomic writes and embedded recipe JSON decoding."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from zcu_tools.core.waveform_format import (
    WaveformFile,
    WaveformFormatError,
    WaveformSamples,
    decode_npz,
    encode_npz,
)

from ._waveform_data import ArbWaveformData, ArbWaveformError
from ._waveform_formula import FormulaRecipe


def load_npz(path: Path) -> ArbWaveformData:
    """Read and validate playback arrays and optional recipe from an NPZ path.

    Reject unknown keys and malformed payloads with ArbWaveformError; translate
    archive and I/O failures to npz_read_failed while preserving their cause.
    """
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArbWaveformError(
            f"Failed to read arbitrary waveform asset {path}: {exc}",
            reason="npz_read_failed",
            data={"path": str(path)},
        ) from exc
    try:
        decoded = decode_npz(payload)
    except WaveformFormatError as exc:
        diagnostics: dict[str, object] = dict(exc.data)
        if exc.reason in {"npz_read_failed", "unknown_npz_key", "missing_npz_key"}:
            diagnostics["path"] = str(path)
        # Keep underlying I/O errors directly visible at the repository boundary.
        cause = exc.__cause__ if isinstance(exc.__cause__, OSError) else exc
        raise ArbWaveformError(str(exc), reason=exc.reason, data=diagnostics) from cause
    recipe = (
        None
        if decoded.recipe_json is None
        else FormulaRecipe.from_json(decoded.recipe_json)
    )
    samples = decoded.samples
    return ArbWaveformData(
        idata=samples.idata,
        qdata=np.zeros_like(samples.idata) if samples.qdata is None else samples.qdata,
        time=samples.time,
        recipe=recipe,
    )


def write_npz(path: Path, data: ArbWaveformData) -> None:
    """Replace an NPZ path with validated data and its optional recipe JSON.

    Create the parent directory and use a same-directory temporary file. On
    write failure remove that temporary file and propagate the original error.
    """
    try:
        payload = encode_npz(
            WaveformFile(
                samples=WaveformSamples(
                    time=data.time, idata=data.idata, qdata=data.qdata
                ),
                recipe_json=None if data.recipe is None else data.recipe.to_json(),
            )
        )
    except WaveformFormatError as exc:
        raise ArbWaveformError(
            str(exc), reason=exc.reason, data=dict(exc.data)
        ) from exc
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
            tmp.write(payload)
        Path(tmp_name).replace(path)
    except Exception:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise
