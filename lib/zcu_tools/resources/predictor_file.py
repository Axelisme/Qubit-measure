"""Typed zcu.predictor v1 JSON I/O, separate from numeric model construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from zcu_tools.simulate.parameters import ModelName, ModelParams, params_class
from zcu_tools.utils import atomic_replace


@dataclass(frozen=True)
class PredictorSource:
    """Authored provenance, with no generated values or ledger lookup.

    run_id is a nonblank opaque run identifier, or None when unknown.
    created_at is None when unknown, otherwise an extended calendar timestamp:
    YYYY-MM-DDTHH:MM[:SS[.ffffff]] followed by Z or an offset of ±HH:MM.
    Only uppercase T separates date and time; fractional seconds use 1–6 digits.
    Calendar, time and offset values must be valid; other ISO forms are rejected.
    Text is preserved verbatim. The file I/O functions validate these fields.
    """

    run_id: str | None
    created_at: str | None


@dataclass(frozen=True)
class PredictorFile:
    """Independent numeric inputs and metadata for one predictor file.

    model selects the built-in 'fluxonium' or 'transmon' parameter type.
    params holds that model's GHz/native-flux inputs; None leaves a value unknown.
    flux_unit names the native flux unit as nonblank text, or None when unknown.
    source holds authored provenance, without checking whether the run exists.
    The I/O functions validate representation, not completeness or physical
    constraints. Call params.build_predictor() to validate and build a model.
    """

    model: ModelName
    params: ModelParams
    flux_unit: str | None
    source: PredictorSource


def _fields(value: object, expected: tuple[str, ...], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    missing = set(expected) - result.keys()
    if missing:
        raise ValueError(f"{name} missing fields: {', '.join(sorted(missing))}")
    unknown = result.keys() - set(expected)
    if unknown:
        raise ValueError(f"{name} unknown fields: {', '.join(sorted(unknown))}")
    return result


def _nullable_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string or null")
    return value


def _decode(value: object) -> PredictorFile:
    document = _fields(
        value,
        ("format", "version", "model", "params", "flux_unit", "source"),
        "predictor",
    )
    if document["format"] != "zcu.predictor":
        raise ValueError("format must be 'zcu.predictor'")
    version = document["version"]
    if type(version) is not int or version != 1:
        raise ValueError("version must be the integer 1")
    model = document["model"]
    if model not in ("fluxonium", "transmon"):
        raise ValueError("model must be 'fluxonium' or 'transmon'")
    model_name: ModelName = "fluxonium" if model == "fluxonium" else "transmon"
    raw_params = document["params"]
    if not isinstance(raw_params, dict):
        raise ValueError("params must be a JSON object")
    # Numeric owners validate parameter names and finite numbers, not this envelope.
    params = params_class(model_name).from_mapping(raw_params)
    flux_unit = _nullable_text(document["flux_unit"], "flux_unit")
    source = _fields(document["source"], ("run_id", "created_at"), "source")
    run_id = _nullable_text(source["run_id"], "source.run_id")
    created_at = _nullable_text(source["created_at"], "source.created_at")
    if created_at is not None:
        if (
            re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"
                r"(?::[0-9]{2}(?:\.[0-9]{1,6})?)?(?:Z|[+-][0-9]{2}:[0-9]{2})",
                created_at,
            )
            is None
        ):
            raise ValueError(
                "source.created_at must be YYYY-MM-DDTHH:MM[:SS[.ffffff]] "
                "with Z or ±HH:MM"
            )
        timestamp = datetime.fromisoformat(created_at)
        if timestamp.utcoffset() is None:
            raise ValueError("source.created_at must include a timezone")
    return PredictorFile(
        model_name, params, flux_unit, PredictorSource(run_id, created_at)
    )


def load_predictor_file(path: Path) -> PredictorFile:
    """Read existing UTF-8 zcu.predictor v1 JSON into independent typed inputs.

    All envelope/source fields are required, including nullable metadata. Extra
    envelope/source fields, unknown models/parameters, invalid metadata, bool or
    nonfinite numeric inputs raise ValueError. Numeric fields may be omitted or
    null. Version must be integer 1, not bool/float. Timestamps need a timezone.
    Parsing errors and filesystem errors propagate; no file or directory is
    created or modified. This does not build a numeric model or read a component.
    """
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    return _decode(value)


def save_predictor_file(path: Path, predictor: PredictorFile) -> None:
    """Validate and atomically replace path with zcu.predictor v1 UTF-8 JSON.

    The parent directory must exist. Model and parameter type must agree;
    representation errors raise ValueError before writing, using the same rules
    as load_predictor_file. Unknown numeric values are saved as null. No values,
    units, run IDs or timestamps are inferred, and predictor is not mutated.
    I/O errors propagate; a failed replacement retains the existing destination
    and removes the sibling temporary. There is no multiwriter merge, lock or
    power-loss guarantee.
    """
    if predictor.model != predictor.params.model:
        raise ValueError("model and params type must agree")
    document = {
        "format": "zcu.predictor",
        "version": 1,
        "model": predictor.model,
        "params": predictor.params.to_dict(),
        "flux_unit": predictor.flux_unit,
        "source": {
            "run_id": predictor.source.run_id,
            "created_at": predictor.source.created_at,
        },
    }
    _decode(document)
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        atomic_replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
