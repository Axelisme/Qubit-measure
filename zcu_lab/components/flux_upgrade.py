"""Retained, pure component flux v1 to v2 conversion, without file I/O."""

from copy import deepcopy
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.format_version import FormatVersion, YamlMap, validate_header
from zcu_tools.resources.entry.aliases import parse_entry_aliases
from zcu_tools.resources.entry.schema import (
    PointDocument,
    PointGeneral,
    SetupGeneral,
)
from zcu_tools.resources.parameter_metadata import validate_parameter_header
from zcu_tools.resources.parameter_upgrade import ParameterDocument

from .qubit import QubitFluxCalibration

_LOCAL = ("flux_half", "flux_int", "flux_period", "flux_unit")
_COMPONENTS = TypeAdapter(dict[str, YamlMap])


@dataclass(frozen=True)
class FluxClassification:
    """Offline shape classification for one complete parameter document.

    status is legacy, current, partial or corrupt. diagnostics names failed
    component paths/checks, or explains the assumed references on legacy input.
    Classification checks syntax, not whether a named device exists or is wired.
    """

    status: Literal["legacy", "current", "partial", "corrupt"]
    diagnostics: tuple[str, ...]


class _PartialFlux(ValueError):
    """An incomplete calibration or mixed old/new document needs manual repair."""


def _calibration(raw: YamlMap, path: str, *, legacy: bool) -> YamlMap:
    keys = ("half", "int", "period") if legacy else ("half", "integer", "period")
    unknown = raw.keys() - set(keys)
    if unknown:
        raise ValueError(f"{path}: unknown calibration fields {sorted(unknown)}")
    values = [raw.get(key) for key in keys]
    if any(value is None for value in values):
        raise _PartialFlux(f"{path}: partial calibration; require {', '.join(keys)}")
    renamed: YamlMap = dict(zip(("half", "integer", "period"), values, strict=True))
    try:
        model = QubitFluxCalibration.model_validate(renamed)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isclose(
        abs(model.integer - model.half),
        abs(model.period) / 2,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{path}: require |integer - half| = |period| / 2")
    return renamed


def _source(fields: YamlMap, path: str) -> str:
    source = fields.get("flux_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            f"{path}.flux_source: manual repair required; no usable explicit device name, cannot guess"
        )
    try:
        validate_device_name(source)
    except ValueError as error:
        raise ValueError(f"{path}.flux_source: {error}") from error
    return source


def _validate_container(
    document: ParameterDocument, components: dict[str, YamlMap]
) -> None:
    source = document.source_path or Path("point.yaml")
    if document.data.get("format") == "zcu.entry-template":
        validate_header(
            document.data,
            expected_format="zcu.entry-template",
            supported_version=FormatVersion(1, 0),
            source=source,
        )
        if "aliases" not in document.data:
            raise ValueError("missing aliases section")
        parse_entry_aliases(document.data["aliases"], source=source)
    else:
        version = validate_parameter_header(document.data, source=source)
        general = document.data.get("general")
        if source.name == "setup.yaml":
            SetupGeneral.model_validate(general)
        else:
            PointGeneral.model_validate(general)
        allowed = {
            "format",
            "format_version",
            "schema_version",
            "general",
            "components",
            "metadata",
        }
        unknown = document.data.keys() - allowed
        if unknown and version.minor == 0:
            raise ValueError(f"unknown parameter sections: {sorted(unknown)}")
        TypeAdapter(dict[str, YamlMap]).validate_python(
            document.data.get("metadata", {}), strict=True
        )
    # Validate current non-flux fields too. Strip retired fields so this retained
    # validator continues working when the runtime removes its old declarations.
    current = deepcopy(components)
    for fields in current.values():
        if str(fields.get("kind", "")).startswith("qubit/"):
            for key in _LOCAL:
                fields.pop(key, None)
        if fields.get("kind") == "group/mux":
            fields.pop("coil_flux", None)
    PointDocument.validate_components(current)


def _add_flux(
    components: dict[str, YamlMap], qubit: str, device: str, calibration: YamlMap
) -> None:
    fields = components[qubit]
    flux = fields.setdefault("flux", {})
    if not isinstance(flux, dict):
        raise ValueError(f"components.{qubit}.flux: expected mapping")
    if device in flux:
        raise ValueError(
            f"components.{qubit}.flux.{device}: conflicting local/mux calibration; manual repair required"
        )
    flux[device] = calibration


def _convert_components(components: dict[str, YamlMap]) -> None:
    for name, fields in components.items():
        path = f"components.{name}"
        kind = fields.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path}.kind: required string")
        if fields.get("flux_source") is not None:
            _source(fields, path)
        if not kind.startswith("qubit/"):
            continue
        _convert_local(components, name, fields)
    for name, fields in components.items():
        if fields.get("kind") != "group/mux":
            continue
        _convert_mux(components, name, fields)


def _convert_local(components: dict[str, YamlMap], name: str, fields: YamlMap) -> None:
    path = f"components.{name}"
    if fields.get("flux_unit") not in (None, "A", "V"):
        raise ValueError(f"{path}.flux_unit: expected A, V or null")
    flux = fields.get("flux", {})
    if not isinstance(flux, dict):
        raise ValueError(f"{path}.flux: expected mapping")
    for device, raw in flux.items():
        validate_device_name(device)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}.flux.{device}: expected calibration mapping")
        _calibration(raw, f"{path}.flux.{device}", legacy=False)
    raw_local: YamlMap = {
        "half": fields.get("flux_half"),
        "int": fields.get("flux_int"),
        "period": fields.get("flux_period"),
    }
    if any(value is not None for value in raw_local.values()):
        calibration = _calibration(raw_local, path, legacy=True)
        _add_flux(components, name, _source(fields, path), calibration)
    for key in _LOCAL:
        fields.pop(key, None)
    fields.setdefault("flux", {})


def _convert_mux(components: dict[str, YamlMap], name: str, fields: YamlMap) -> None:
    coil = fields.pop("coil_flux", {})
    if not isinstance(coil, dict):
        raise ValueError(f"components.{name}.coil_flux: expected mapping")
    qubits = fields.get("qubits") or []
    if not isinstance(qubits, list):
        raise ValueError(f"components.{name}.qubits: expected list")
    for qubit, raw in coil.items():
        path = f"components.{name}.coil_flux.{qubit}"
        target = components.get(qubit)
        if (
            qubit not in qubits
            or target is None
            or not str(target.get("kind", "")).startswith("qubit/")
        ):
            raise ValueError(f"{path}: key must name a qubit in this mux group")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected calibration mapping")
        calibration = _calibration(raw, path, legacy=True)
        _add_flux(components, qubit, _source(fields, f"components.{name}"), calibration)


def _shape(components: dict[str, YamlMap], version: object) -> None:
    old_values = False
    new_values = False
    old_fields = False
    for fields in components.values():
        kind = str(fields.get("kind", ""))
        if kind.startswith("qubit/"):
            old_fields |= any(key in fields for key in _LOCAL)
            old_values |= any(fields.get(key) is not None for key in _LOCAL[:3])
            new_values |= bool(fields.get("flux"))
        if kind == "group/mux":
            old_fields |= "coil_flux" in fields
            old_values |= bool(fields.get("coil_flux"))
    if version == 2 and old_fields:
        raise _PartialFlux(
            "schema_version 2 contains retired flux fields; partial migration"
        )
    if old_values and new_values:
        raise _PartialFlux(
            "document mixes old calibrations and new flux entries; partial migration"
        )
    if version != 2 and new_values and not old_values:
        raise _PartialFlux(
            "new flux entries without schema_version 2; partial migration, verify the entire document"
        )


def classify_flux_document(document: ParameterDocument) -> FluxClassification:
    """Classify a complete detached setup, point or template without mutation/I/O.

    Check container/aliases, all registered components, calibration completeness,
    finite numbers, half/integer spacing, mux membership and device name syntax.
    Missing source names require manual repair, never a guessed device. Unknown
    device existence is not checked: no catalog is present in these documents.
    Caller must bootstrap component models before calling. Invalid data returns
    partial/corrupt with source and failed checks instead of raising ValueError.
    """
    source = str(document.source_path or "<parameter-document>")
    try:
        version = document.data.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"unsupported schema_version {version!r}")
        if "components" not in document.data:
            raise ValueError("missing components section")
        components = _COMPONENTS.validate_python(
            document.data["components"], strict=True
        )
        _shape(components, version)
        converted = deepcopy(components)
        _convert_components(converted)
        _validate_container(document, converted)
    except _PartialFlux as error:
        return FluxClassification("partial", (f"{source}: {error}",))
    except ValueError as error:
        return FluxClassification("corrupt", (f"{source}: {error}",))
    if version == 2:
        return FluxClassification("current", ())
    return FluxClassification(
        "legacy",
        (
            "Converted reference points are assumed; recalibrate with the other contributing devices at their integer references.",
        ),
    )


def convert_flux_v1_to_v2(document: ParameterDocument) -> ParameterDocument:
    """Return a detached v2 document from completely classified v1/unmarked data.

    Preserve unrelated fields, flux_source and original working-unit numbers.
    Move local and mux calibrations to qubit flux entries keyed by their explicit
    source device. Remove replaced fields; never synthesize a record or a frame.
    Missing source, partial/corrupt input or non-v1 data raises located ValueError.
    No I/O, registration, unit inference or device existence check occurs.
    """
    classification = classify_flux_document(document)
    if classification.status != "legacy":
        raise ValueError(
            f"{classification.status}: {'; '.join(classification.diagnostics)}"
        )
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    _convert_components(components)
    result.data["components"] = {name: fields for name, fields in components.items()}
    result.data["schema_version"] = 2
    return result


def convert_storage_flux_batch(document: ParameterDocument) -> ParameterDocument:
    """Convert a complete legacy-import seed/value batch before current validation.

    The storage caller may overlay historical fields onto a current seed. This
    batch is not a user-authored half-migrated document. Validate/convert all
    supplied calibrations using the same pure conversion as the retained step.
    Reject conflicts or missing explicit sources instead of overwriting them.
    The migration owner isolates failed groups as pending and preserves raw input.
    Output is v2; runtime v1 cannot read it. No I/O or runtime policy change occurs.
    """
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    _convert_components(components)
    _validate_container(document, components)
    result.data["components"] = {name: fields for name, fields in components.items()}
    result.data["schema_version"] = 2
    return result
