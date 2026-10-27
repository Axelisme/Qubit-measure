"""Retained pure component flux v1→v2→v3 conversions, without file I/O."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.format_version import FormatVersion, YamlMap, validate_header
from zcu_tools.resources.entry.aliases import parse_entry_aliases
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral, SetupGeneral
from zcu_tools.resources.parameter_metadata import validate_parameter_header
from zcu_tools.resources.parameter_upgrade import ParameterDocument

from .qubit import QubitFluxCalibration, QubitFluxSchema

_LOCAL = ("flux_half", "flux_int", "flux_period", "flux_unit")
_COMPONENTS = TypeAdapter(dict[str, YamlMap])


@dataclass(frozen=True)
class FluxClassification:
    """Offline classification of one complete parameter document.

    status is legacy (v1 or v2), current (v3), partial or corrupt. diagnostics
    names failed paths/checks. Device existence is not checked without a catalog.
    """

    status: Literal["legacy", "current", "partial", "corrupt"]
    diagnostics: tuple[str, ...]


class _PartialFlux(ValueError):
    """An incomplete calibration or mixed-version document needs manual repair."""


def _calibration(
    raw: YamlMap, path: str, *, device_name: str, legacy: bool = True
) -> QubitFluxCalibration:
    keys = ("half", "int", "period") if legacy else ("half", "integer", "period")
    if unknown := raw.keys() - set(keys):
        raise ValueError(f"{path}: unknown calibration fields {sorted(unknown)}")
    values = [raw.get(key) for key in keys]
    if any(value is None for value in values):
        raise _PartialFlux(f"{path}: partial calibration; require {', '.join(keys)}")
    try:
        points = QubitFluxCalibration.model_validate(
            dict(zip(("half", "integer", "period"), values, strict=True))
        )
        points.validate_spacing(device_name)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return points


def _source(fields: YamlMap, path: str) -> str:
    source = fields.get("flux_source")
    if not isinstance(source, str):
        raise ValueError(f"{path}.flux_source: expected a device name")
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
        if (unknown := document.data.keys() - allowed) and version.minor == 0:
            raise ValueError(f"unknown parameter sections: {sorted(unknown)}")
        TypeAdapter(dict[str, YamlMap]).validate_python(
            document.data.get("metadata", {}), strict=True
        )
    # Historical flux is already projected to v3; validate all other fields with
    # the registered current models, without teaching runtime about old shapes.
    PointDocument.validate_components(components)


@dataclass
class _FluxConversion:
    """One v2 qubit frame and incoming v1 source-selection evidence."""

    frame: dict[str, QubitFluxCalibration] = field(default_factory=dict)
    local_calibrated: bool = False
    coil_offsets: dict[str, float] = field(default_factory=dict)


def _convert_components_v1(components: dict[str, YamlMap]) -> tuple[str, ...]:
    conversions: dict[str, _FluxConversion] = {}
    notes: list[str] = []
    for name, fields in components.items():
        path = f"components.{name}"
        kind = fields.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path}.kind: required string")
        if fields.get("flux_source") is not None:
            _source(fields, path)
        if kind.startswith("qubit/"):
            conversion = _FluxConversion()
            conversion.local_calibrated = _convert_local(name, fields, conversion.frame)
            conversions[name] = conversion
    for name, fields in components.items():
        if fields.get("kind") == "group/mux":
            _convert_mux(name, fields, conversions, notes)
    for name, conversion in conversions.items():
        _select_source(name, components[name], conversion, notes)
        components[name]["flux"] = {
            device: _COMPONENTS.validate_python(
                {device: points.model_dump(mode="python")}
            )[device]
            for device, points in conversion.frame.items()
        }
    return tuple(notes)


def _convert_local(
    name: str, fields: YamlMap, frame: dict[str, QubitFluxCalibration]
) -> bool:
    path = f"components.{name}"
    if fields.get("flux_unit") not in (None, "A", "V"):
        raise ValueError(f"{path}.flux_unit: expected A, V or null")
    raw: YamlMap = {
        "half": fields.get("flux_half"),
        "int": fields.get("flux_int"),
        "period": fields.get("flux_period"),
    }
    calibrated = any(value is not None for value in raw.values())
    if calibrated:
        if fields.get("flux_source") is None:
            fields["flux_source"] = "unnamed_device"
        device = _source(fields, path)
        frame[device] = _calibration(raw, path, device_name=device)
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
            if fields.get("flux_source") is None
            else _source(fields, f"components.{name}")
        )
        points = _calibration(raw, path, device_name=device)
        conversion = conversions[qubit]
        if device in conversion.frame:
            raise ValueError(
                f"{path}: conflicting local/mux calibration for {device!r}"
            )
        # V1 coil reference points do not establish a shared reference frame.
        # Retain only its signed period; neutral references prevent v3 from
        # silently recovering the discarded offset contribution.
        conversion.frame[device] = QubitFluxCalibration(
            half=points.period / 2, integer=0.0, period=points.period
        )
        candidate = 0.5 - points.half / points.period
        conversion.coil_offsets[device] = candidate
        notes.append(f"{path}: candidate offset={candidate!r} from device {device!r}.")
        notes.append(
            f"{path}: dropped half={points.half!r}, int={points.integer!r}; retained period at components.{qubit}.flux.period.{device}."
        )


def _select_source(
    name: str, fields: YamlMap, conversion: _FluxConversion, notes: list[str]
) -> None:
    if conversion.local_calibrated or not conversion.coil_offsets:
        return
    source = fields.get("flux_source")
    if isinstance(source, str) and source in conversion.coil_offsets:
        return
    if len(conversion.coil_offsets) == 1:
        fields["flux_source"] = next(iter(conversion.coil_offsets))
    else:
        fields.pop("flux_source", None)
        notes.append(
            f"components.{name}.flux.offset: multiple coils without a designated source; left null. Recalibrate offset; all device periods are retained."
        )


def _v2_frame(raw: object, path: str) -> dict[str, QubitFluxCalibration]:
    entries = _COMPONENTS.validate_python(raw, strict=True)
    result = {}
    for device, calibration in entries.items():
        validate_device_name(device)
        result[device] = _calibration(
            calibration, f"{path}.{device}", device_name=device, legacy=False
        )
    return result


def _convert_components_v2(components: dict[str, YamlMap]) -> tuple[str, ...]:
    notes: list[str] = []
    for name, fields in components.items():
        kind = str(fields.get("kind", ""))
        if not (kind.startswith("qubit/") or kind in ("group/mux", "amplifier/jpa")):
            continue
        if fields.get("flux_source") is not None:
            _source(fields, f"components.{name}")
        if "flux_source" in fields:
            fields["flux_dev"] = fields.pop("flux_source")
        if not kind.startswith("qubit/"):
            continue
        frame = _v2_frame(fields.get("flux", {}), f"components.{name}.flux")
        source = fields.get("flux_dev")
        offset = None
        if isinstance(source, str) and source in frame:
            main = frame[source]
            offset = (
                0.5
                - main.half / main.period
                - sum(
                    points.integer / points.period
                    for device, points in frame.items()
                    if device != source
                )
            )
        current = QubitFluxSchema(
            offset=offset,
            period={device: points.period for device, points in frame.items()},
        )
        fields["flux"] = _COMPONENTS.validate_python(
            {name: current.model_dump(mode="python")}
        )[name]
        paths = []
        if source in ("unnamed_device", "unnamed_coil"):
            paths.append(f"components.{name}.flux_dev")
        for device in ("unnamed_device", "unnamed_coil"):
            if device in current.period:
                paths.append(f"components.{name}.flux.period.{device}")
        notes.extend(
            f"{path}: placeholder; rename to the actual device name." for path in paths
        )
    return tuple(notes)


def _shape(components: dict[str, YamlMap], version: int) -> None:
    for name, fields in components.items():
        kind = str(fields.get("kind", ""))
        path = f"components.{name}"
        if version >= 2 and (
            (kind.startswith("qubit/") and any(key in fields for key in _LOCAL))
            or (kind == "group/mux" and "coil_flux" in fields)
        ):
            raise _PartialFlux(
                f"{path}: schema_version {version} contains retired v1 flux fields"
            )
        has_source = kind.startswith("qubit/") or kind in ("group/mux", "amplifier/jpa")
        if has_source and (
            (version < 3 and "flux_dev" in fields)
            or (version == 3 and "flux_source" in fields)
        ):
            raise _PartialFlux(
                f"{path}: source field does not match schema_version {version}"
            )
        if kind.startswith("qubit/"):
            flux = fields.get("flux", {})
            if version == 1 and not isinstance(flux, dict):
                raise ValueError(f"{path}.flux: expected mapping")
            if version == 1 and flux:
                raise _PartialFlux(
                    f"{path}: new flux entries without schema_version 2/3; partial migration"
                )


def classify_flux_document(document: ParameterDocument) -> FluxClassification:
    """Classify detached v1/v2/v3 setup, point or template without mutation/I/O.

    Validate container, aliases, registered components, calibration completeness,
    finite numbers, historical spacing, mux membership and device name syntax.
    Missing v1 names receive distinct placeholders; illegal names remain corrupt.
    Bootstrap component models first. Invalid input returns located diagnostics.
    """
    source = str(document.source_path or "<parameter-document>")
    try:
        version = document.data.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2, 3):
            raise ValueError(f"unsupported schema_version {version!r}")
        if "components" not in document.data:
            raise ValueError("missing components section")
        components = _COMPONENTS.validate_python(
            document.data["components"], strict=True
        )
        _shape(components, version)
        converted = deepcopy(components)
        if version == 1:
            _convert_components_v1(converted)
        if version < 3:
            _convert_components_v2(converted)
        _validate_container(document, converted)
    except _PartialFlux as error:
        return FluxClassification("partial", (f"{source}: {error}",))
    except ValueError as error:
        return FluxClassification("corrupt", (f"{source}: {error}",))
    return FluxClassification("current" if version == 3 else "legacy", ())


def convert_flux_v1_to_v2(document: ParameterDocument) -> ParameterDocument:
    """Return detached main-v2 per-device half/integer/period from v1 input.

    Keep flux_source and the local calibration unchanged. Coils retain signed
    period with integer=0 and half=period/2; original references enter notes only.
    Select local, designated coil, or unique coil; ambiguous coils leave no source.
    Missing names get distinct placeholders. Invalid/non-v1 input raises ValueError.
    No I/O, registration, unit inference or device existence check occurs.
    """
    classification = classify_flux_document(document)
    if document.data.get("schema_version", 1) != 1 or classification.status != "legacy":
        raise ValueError(
            f"{classification.status}: {'; '.join(classification.diagnostics)}"
        )
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    result = replace(
        result, notes=(*document.notes, *_convert_components_v1(components))
    )
    result.data["components"] = dict(components)
    result.data["schema_version"] = 2
    return result


def convert_flux_v2_to_v3(document: ParameterDocument) -> ParameterDocument:
    """Return detached offset/period v3 from completely validated main-v2 data.

    Rename qubit/mux/JPA flux_source to flux_dev. With a main entry, offset is
    0.5-half_main/period_main minus every other integer/period contribution.
    Preserve that half branch without mod1 and all signed periods. Missing source
    or main entry gives null offset; empty flux gives null offset and empty periods.
    Preserve prior notes and report placeholder destinations. Invalid/non-v2 input
    raises ValueError. No I/O, registration or device existence check occurs.
    """
    classification = classify_flux_document(document)
    if document.data.get("schema_version") != 2 or classification.status != "legacy":
        raise ValueError(
            f"{classification.status}: {'; '.join(classification.diagnostics)}"
        )
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    result = replace(
        result, notes=(*document.notes, *_convert_components_v2(components))
    )
    result.data["components"] = dict(components)
    result.data["schema_version"] = 3
    return result


def convert_storage_flux_batch(document: ParameterDocument) -> ParameterDocument:
    """Convert a legacy-import overlay to v3 before current schema validation.

    Effective flux_source/flux_dev includes setup/mapping inheritance. Complete
    incoming calibration replaces offset and its device periods, retaining other
    inherited periods. No calibration retains the inherited frame unchanged.
    Missing names get placeholders; invalid names/spacing, partial calibration
    and same-batch device collisions fail. Native mixed documents remain rejected.
    No I/O occurs; the migration owner isolates failures and preserves raw input.
    """
    result = deepcopy(document)
    components = _COMPONENTS.validate_python(result.data["components"], strict=True)
    inherited: dict[str, QubitFluxSchema] = {}
    for name, fields in components.items():
        if "flux_dev" in fields:
            if "flux_source" in fields and fields["flux_source"] != fields["flux_dev"]:
                raise _PartialFlux(
                    f"components.{name}: conflicting inherited flux sources"
                )
            fields["flux_source"] = fields.pop("flux_dev")
        if str(fields.get("kind", "")).startswith("qubit/"):
            inherited[name] = QubitFluxSchema.model_validate(fields.pop("flux", {}))
    notes = _convert_components_v1(components)
    notes += _convert_components_v2(components)
    for name, previous in inherited.items():
        fields = components[name]
        incoming = QubitFluxSchema.model_validate(fields["flux"])
        if incoming.offset is not None:
            previous.offset = incoming.offset
        previous.period.update(incoming.period)
        fields["flux"] = _COMPONENTS.validate_python(
            {name: previous.model_dump(mode="python")}
        )[name]
    _validate_container(document, components)
    result = replace(result, notes=(*document.notes, *notes))
    result.data["components"] = dict(components)
    result.data["schema_version"] = 3
    return result
