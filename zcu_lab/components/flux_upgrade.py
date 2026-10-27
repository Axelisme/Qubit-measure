"""Retained, pure component flux v1 to v2 conversion, without file I/O."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
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

from .qubit import QubitFluxCalibration, QubitFluxSchema

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


def _calibration(raw: YamlMap, path: str, *, device_name: str, legacy: bool) -> YamlMap:
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
        model.validate_spacing(device_name)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return renamed


def _source(fields: YamlMap, path: str) -> str:
    source = fields.get("flux_dev")
    if not isinstance(source, str):
        raise ValueError(f"{path}.flux_dev: expected a device name")
    try:
        validate_device_name(source)
    except ValueError as error:
        raise ValueError(f"{path}.flux_dev: {error}") from error
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


@dataclass
class _FluxConversion:
    """One qubit's current frame plus incoming legacy calibration evidence."""

    frame: QubitFluxSchema
    local_calibrated: bool
    incoming: set[str] = field(default_factory=set)
    coil_offsets: dict[str, float] = field(default_factory=dict)


def _convert_components(components: dict[str, YamlMap]) -> tuple[str, ...]:
    conversions: dict[str, _FluxConversion] = {}
    notes: list[str] = []
    for name, fields in components.items():
        path = f"components.{name}"
        kind = fields.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path}.kind: required string")
        if "flux_source" in fields:
            if "flux_dev" in fields:
                raise _PartialFlux(f"{path}: mixes v1 flux_source and v2 flux_dev")
            fields["flux_dev"] = fields.pop("flux_source")
        if fields.get("flux_dev") is not None:
            _source(fields, path)
        if kind.startswith("qubit/"):
            frame = QubitFluxSchema.model_validate(fields.get("flux", {}))
            local = _convert_local(name, fields, frame)
            conversion = _FluxConversion(frame, local)
            if local:
                conversion.incoming.add(_source(fields, path))
            conversions[name] = conversion
    for name, fields in components.items():
        if fields.get("kind") == "group/mux":
            _convert_mux(name, fields, conversions, notes)
    for name, conversion in conversions.items():
        _select_offset(name, components[name], conversion, notes)
        components[name]["flux"] = _COMPONENTS.validate_python(
            {name: conversion.frame.model_dump(mode="python")}
        )[name]
        notes.extend(_placeholder_notes(name, components[name], conversion.frame))
    return tuple(notes)


def _placeholder_notes(name: str, fields: YamlMap, frame: QubitFluxSchema) -> list[str]:
    paths = []
    if fields.get("flux_dev") == "unnamed_device":
        paths.append(f"components.{name}.flux_dev")
    for device in ("unnamed_device", "unnamed_coil"):
        if device in frame.period:
            paths.append(f"components.{name}.flux.period.{device}")
    return [f"{path}: placeholder; rename to the actual device name." for path in paths]


def _convert_local(name: str, fields: YamlMap, frame: QubitFluxSchema) -> bool:
    path = f"components.{name}"
    if fields.get("flux_unit") not in (None, "A", "V"):
        raise ValueError(f"{path}.flux_unit: expected A, V or null")
    raw_local: YamlMap = {
        "half": fields.get("flux_half"),
        "int": fields.get("flux_int"),
        "period": fields.get("flux_period"),
    }
    calibrated = any(value is not None for value in raw_local.values())
    if calibrated:
        if fields.get("flux_dev") is None:
            fields["flux_dev"] = "unnamed_device"
        device = _source(fields, path)
        raw = _calibration(raw_local, path, device_name=device, legacy=True)
        points = QubitFluxCalibration.model_validate(raw)
        frame.offset = 0.5 - points.half / points.period
        frame.period[device] = points.period
    for key in _LOCAL:
        fields.pop(key, None)
    return calibrated


def _convert_mux(
    name: str,
    fields: YamlMap,
    conversions: dict[str, _FluxConversion],
    notes: list[str],
) -> None:
    coil = fields.pop("coil_flux", {})
    if not isinstance(coil, dict):
        raise ValueError(f"components.{name}.coil_flux: expected mapping")
    qubits = fields.get("qubits") or []
    if not isinstance(qubits, list):
        raise ValueError(f"components.{name}.qubits: expected list")
    for qubit, raw in coil.items():
        path = f"components.{name}.coil_flux.{qubit}"
        if qubit not in qubits or qubit not in conversions:
            raise ValueError(f"{path}: key must name a qubit in this mux group")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected calibration mapping")
        device = (
            "unnamed_coil"
            if fields.get("flux_dev") is None
            else _source(fields, f"components.{name}")
        )
        points = QubitFluxCalibration.model_validate(
            _calibration(raw, path, device_name=device, legacy=True)
        )
        conversion = conversions[qubit]
        if device in conversion.incoming:
            raise ValueError(
                f"{path}: conflicting local/mux calibration for {device!r}"
            )
        conversion.incoming.add(device)
        conversion.frame.period[device] = points.period
        candidate = 0.5 - points.half / points.period
        conversion.coil_offsets[device] = candidate
        notes.append(f"{path}: candidate offset={candidate!r} from device {device!r}.")
        notes.append(
            f"{path}: dropped half={points.half!r}, int={points.integer!r}; retained period at components.{qubit}.flux.period.{device}."
        )


def _select_offset(
    name: str, fields: YamlMap, conversion: _FluxConversion, notes: list[str]
) -> None:
    if conversion.local_calibrated or not conversion.coil_offsets:
        return
    source = fields.get("flux_dev")
    if isinstance(source, str) and source in conversion.coil_offsets:
        conversion.frame.offset = conversion.coil_offsets[source]
    elif len(conversion.coil_offsets) == 1:
        conversion.frame.offset = next(iter(conversion.coil_offsets.values()))
    else:
        conversion.frame.offset = None
        notes.append(
            f"components.{name}.flux.offset: multiple coils without a designated source; "
            "left null. Recalibrate offset; all device periods are retained."
        )


def _shape(components: dict[str, YamlMap], version: object) -> None:
    old_values = False
    new_values = False
    old_fields = False
    for fields in components.values():
        old_fields |= "flux_source" in fields
        kind = str(fields.get("kind", ""))
        if kind.startswith("qubit/"):
            old_fields |= any(key in fields for key in _LOCAL)
            old_values |= any(fields.get(key) is not None for key in _LOCAL[:3])
            flux = fields.get("flux")
            new_values |= isinstance(flux, dict) and (
                flux.get("offset") is not None or bool(flux.get("period"))
            )
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
    Missing names receive distinct local/coil placeholders. Present illegal names
    remain corrupt. Unknown
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
            "Offset is derived from the qubit local half/period, or coil half/period when no local calibration exists.",
        ),
    )


def convert_flux_v1_to_v2(document: ParameterDocument) -> ParameterDocument:
    """Return a detached v2 document from completely classified v1/unmarked data.

    Rename v1 flux_source to flux_dev, preserving its value, unrelated fields
    and original working-unit numbers.
    Derive offset from local half/period (coil only when local is absent), keep
    signed device periods and report dropped coil reference points. Missing names
    receive distinct placeholders with located rename notes. Never create records.
    Partial/corrupt input or non-v1 data raises located ValueError.
    No I/O, registration, unit inference or device existence check occurs.
    """
    classification = classify_flux_document(document)
    if classification.status != "legacy":
        raise ValueError(
            f"{classification.status}: {'; '.join(classification.diagnostics)}"
        )
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    result = replace(result, notes=_convert_components(components))
    result.data["components"] = {name: fields for name, fields in components.items()}
    result.data["schema_version"] = 2
    return result


def convert_storage_flux_batch(document: ParameterDocument) -> ParameterDocument:
    """Convert a complete legacy-import seed/value batch before current validation.

    The storage caller may overlay historical fields onto a current seed. This
    batch is not a user-authored half-migrated document. Validate/convert all
    supplied calibrations using the same pure conversion as the retained step.
    A complete valid historical calibration with an explicit source replaces
    the same device's inherited seed calibration once, before conversion; other
    periods stay unchanged. A source explicitly declared in the complete overlay
    includes setup/mapping inheritance; missing names receive placeholders.
    Conflicts within the incoming local/mux batch, partial calibrations and invalid
    names/spacing still fail.
    Native upgrade classification is unchanged and rejects mixed documents.
    The migration owner isolates failed groups as pending and preserves raw input.
    Output is v2. No I/O or runtime policy change occurs.
    """
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    result = replace(result, notes=_convert_components(components))
    _validate_container(document, components)
    result.data["components"] = {name: fields for name, fields in components.items()}
    result.data["schema_version"] = 2
    return result
