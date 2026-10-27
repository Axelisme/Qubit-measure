"""Typed zcu.predictor v2 JSON I/O, separate from numeric model construction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import TypedDict

from zcu_tools.analysis.flux import DevicePeriod, FluxFrame, offset_for_flux
from zcu_tools.simulate.fluxonium.predict import FluxoniumParameterValues
from zcu_tools.simulate.parameters import ModelName, ModelParams, params_class
from zcu_tools.simulate.predictor import parameter_number
from zcu_tools.simulate.transmon.predict import TransmonParameterValues
from zcu_tools.utils import atomic_replace


@dataclass(frozen=True)
class PredictorSource:
    """Authored provenance, with no generated values or ledger lookup.

    run_id is a nonblank opaque run identifier, or None when unknown.
    created_at is None when unknown, otherwise an extended calendar timestamp:
    YYYY-MM-DDTHH:MM[:SS[.ffffff]] followed by Z or an offset of ±HH:MM.
    Only uppercase T separates date and time; fractional seconds use 1–6 digits.
    Calendar, time and offset values must be valid; other ISO forms are rejected.
    Text is preserved verbatim. Validation and I/O functions check these fields.
    """

    run_id: str | None
    created_at: str | None


class PredictorFluxValues(TypedDict):
    """Detached JSON representation of recorded flux coordinates.

    offset is the finite zero-output flux in Phi0, without period folding.
    period maps device names to signed finite nonzero native units per Phi0.
    unit maps exactly the same device names to nonblank native unit strings.
    Empty period/unit mappings describe constant flux.
    """

    offset: float
    period: dict[str, float]
    unit: dict[str, str]


@dataclass(frozen=True)
class PredictorFlux:
    """Recorded zero-output offset (Phi0) and signed device periods with units.

    period maps instrument names to finite nonzero native units per Phi0.
    unit has exactly the same keys and nonblank unit strings. Empty mappings
    describe constant flux. Construction snapshots mappings and raises ValueError
    for invalid names, offset, periods or units, using analysis.flux rules.
    This records one dataset's alignment, not the current component calibration.
    """

    offset: float
    period: Mapping[str, float]
    unit: Mapping[str, str]

    def __post_init__(self) -> None:
        self.to_frame()
        object.__setattr__(self, "period", MappingProxyType(dict(self.period)))
        object.__setattr__(self, "unit", MappingProxyType(dict(self.unit)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PredictorFlux:
        """Decode exact v2 offset/period/unit fields, snapshotting input mappings.

        Nested period/unit values must be JSON objects. Missing/extra fields,
        invalid names, nonfinite/bool/null numbers, zero periods, mismatched
        device keys or blank units raise ValueError. No I/O or model building.
        """
        return _decode_flux(dict(value))

    def to_dict(self) -> PredictorFluxValues:
        """Return offset/period/unit with fresh mutable dictionaries and no I/O."""
        return {
            "offset": self.offset,
            "period": dict(self.period),
            "unit": dict(self.unit),
        }

    def to_frame(self) -> FluxFrame:
        """Return a validated analysis frame; mismatched unit keys raise ValueError."""
        if self.period.keys() != self.unit.keys():
            raise ValueError(
                "flux.period and flux.unit must have identical device keys"
            )
        return FluxFrame(
            self.offset,
            {
                name: DevicePeriod(period, self.unit[name])
                for name, period in self.period.items()
            },
        )


@dataclass(frozen=True)
class PredictorFile:
    """Independent numeric inputs and metadata for one predictor file.

    model selects the built-in 'fluxonium' or 'transmon' parameter type.
    params holds only that model's energy/asymmetry inputs; None is unknown.
    flux records offset, signed device periods and explicit native units.
    source holds authored provenance, without checking whether the run exists.
    validate_predictor_file and the I/O functions check representation, not
    completeness or physical constraints. Call params.build_predictor() to
    validate and build a model.
    """

    model: ModelName
    params: ModelParams
    flux: PredictorFlux
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
    if (
        isinstance(value, dict)
        and type(value.get("version")) is int
        and value.get("version") == 1
    ):
        raise ValueError(
            "zcu.predictor v1 requires upgrade: python scripts/upgrade_parameters.py <entry> (preview); add --apply to back up and replace"
        )
    document = _fields(
        value,
        ("format", "version", "model", "params", "flux", "source"),
        "predictor",
    )
    if document["format"] != "zcu.predictor":
        raise ValueError("format must be 'zcu.predictor'")
    version = document["version"]
    if type(version) is not int or version != 2:
        raise ValueError("version must be the integer 2")
    model = document["model"]
    if model not in ("fluxonium", "transmon"):
        raise ValueError("model must be 'fluxonium' or 'transmon'")
    model_name: ModelName = "fluxonium" if model == "fluxonium" else "transmon"
    raw_params = document["params"]
    if not isinstance(raw_params, dict):
        raise ValueError("params must be a JSON object")
    # Numeric owners validate parameter names and finite numbers, not this envelope.
    params = params_class(model_name).from_mapping(raw_params)
    flux = _decode_flux(document["flux"])
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
    return PredictorFile(model_name, params, flux, PredictorSource(run_id, created_at))


def _decode_flux(value: object) -> PredictorFlux:
    flux = _fields(value, ("offset", "period", "unit"), "flux")
    periods, units = flux["period"], flux["unit"]
    if not isinstance(periods, dict) or not isinstance(units, dict):
        raise ValueError("flux.period and flux.unit must be objects")
    decoded_periods: dict[str, float] = {}
    decoded_units: dict[str, str] = {}
    for name, period in periods.items():
        if not isinstance(name, str):
            raise ValueError("flux device names must be strings")
        number = parameter_number({"period": period}, "period")
        if number is None:
            raise ValueError("flux period cannot be null")
        decoded_periods[name] = number
    for name, unit in units.items():
        if not isinstance(name, str) or not isinstance(unit, str):
            raise ValueError("flux units must map string names to strings")
        decoded_units[name] = unit
    offset = parameter_number(flux, "offset")
    if offset is None:
        raise ValueError("flux offset cannot be null")
    return PredictorFlux(offset, decoded_periods, decoded_units)


def upgrade_predictor_document(value: object) -> PredictorFile:
    """Purely decode v1 or v2 into detached v2 inputs, without writing files.

    V1 requires finite half/bias and nonzero signed period in params.
    offset = 0.5 + (bias - half)/period is not folded. The device name is
    unnamed_device; absent/null flux_unit becomes native. Energy nulls and source
    text are preserved. Malformed, incomplete affine or future input raises
    ValueError; the original document is unchanged even on failure.
    """
    if not isinstance(value, dict):
        raise ValueError("predictor must be a JSON object")
    if type(value.get("version")) is int and value.get("version") == 2:
        return _decode(value)
    document = _fields(
        {"flux_unit": None, **value},
        ("format", "version", "model", "params", "flux_unit", "source"),
        "predictor",
    )
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("upgrade requires integer version 1 or 2")
    raw_params = document["params"]
    if not isinstance(raw_params, dict):
        raise ValueError("params must be an object")
    affine = {
        name: parameter_number(raw_params, name)
        for name in ("flux_half", "flux_period", "flux_bias")
    }
    half, period, bias = affine["flux_half"], affine["flux_period"], affine["flux_bias"]
    if half is None or period is None or bias is None:
        raise ValueError("v1 upgrade requires flux_half, flux_period and flux_bias")
    if period == 0:
        raise ValueError("flux_period must be nonzero")
    unit = _nullable_text(document["flux_unit"], "flux_unit") or "native"
    return _decode(
        {
            "format": document["format"],
            "version": 2,
            "model": document["model"],
            "params": {
                name: number
                for name, number in raw_params.items()
                if name not in affine
            },
            "flux": {
                "offset": offset_for_flux(
                    {"unnamed_device": DevicePeriod(period, unit)},
                    {"unnamed_device": half - bias},
                    flux=0.5,
                ),
                "period": {"unnamed_device": period},
                "unit": {"unnamed_device": unit},
            },
            "source": document["source"],
        }
    )


def load_predictor_file(path: Path) -> PredictorFile:
    """Read existing UTF-8 zcu.predictor v2 JSON into independent typed inputs.

    All envelope/source fields are required, including nullable metadata. Extra
    envelope/source fields, unknown models/parameters, invalid metadata, bool or
    nonfinite numeric inputs raise ValueError. Numeric fields may be omitted or
    null. Version must be integer 2, not bool/float. Timestamps need a timezone.
    V1 is rejected with the explicit entry upgrade command. Flux fields are
    complete and validated by analysis.flux; device/unit keys must match.
    Parsing errors and filesystem errors propagate; no file or directory is
    created or modified. This does not build a numeric model or read a component.
    """
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    return _decode(value)


class PredictorSourceValues(TypedDict):
    """Authored nullable run_id and timezone-aware created_at text, unchanged."""

    run_id: str | None
    created_at: str | None


class PredictorDocument(TypedDict):
    """Detached v2 wire envelope for JSON encoding or semantic comparison.

    format/version identify zcu.predictor v2. model selects the matching numeric
    params in GHz (d is dimensionless); None means unknown. flux records native
    signed periods and Phi0 offset. source retains authored nullable provenance.
    """

    format: str
    version: int
    model: ModelName
    params: FluxoniumParameterValues | TransmonParameterValues
    flux: PredictorFluxValues
    source: PredictorSourceValues


def encode_predictor_document(predictor: PredictorFile) -> PredictorDocument:
    """Validate and encode detached zcu.predictor v2 JSON values without I/O.

    The result uses exactly the save representation, including null unknowns and
    signed native periods. Mutating its nested mappings cannot change predictor.
    Model/type disagreement, invalid metadata or numeric representation raises
    ValueError before returning; incomplete/unphysical energies remain allowed.
    No model is built and no coordinates or provenance are inferred.
    """
    if predictor.model != predictor.params.model:
        raise ValueError("model and params type must agree")
    document: PredictorDocument = {
        "format": "zcu.predictor",
        "version": 2,
        "model": predictor.model,
        "params": predictor.params.to_dict(),
        "flux": predictor.flux.to_dict(),
        "source": {
            "run_id": predictor.source.run_id,
            "created_at": predictor.source.created_at,
        },
    }
    _decode(document)
    return document


def validate_predictor_file(predictor: PredictorFile) -> None:
    """Validate typed inputs using the load/save representation rules, without I/O.

    Model and parameter type must agree. Invalid metadata or numeric inputs raise
    ValueError. Unknown numbers and metadata remain unknown; incomplete or
    physically invalid inputs are accepted without building a numeric model.
    The input is not mutated. This checks the candidate for dry-run callers;
    save_predictor_file repeats validation before writing.
    """
    encode_predictor_document(predictor)


def save_predictor_file(path: Path, predictor: PredictorFile) -> None:
    """Validate and atomically replace path with zcu.predictor v2 UTF-8 JSON.

    The parent directory must exist. Model and parameter type must agree;
    representation errors raise ValueError before writing, using the same rules
    as load_predictor_file. Unknown numeric values are saved as null. No values,
    units, run IDs or timestamps are inferred, and predictor is not mutated.
    I/O errors propagate; a failed replacement retains the existing destination
    and removes the sibling temporary. There is no multiwriter merge, lock or
    power-loss guarantee.
    """
    document = encode_predictor_document(predictor)
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
