"""Component flux groups and cross-file evidence; never predictor/CSV units."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import EditView, document_value, snapshot
from zcu_tools.resources.parameter_upgrade import ParameterDocument

from ._meta_info import preserve_meta_field
from .evidence import has_legacy_expression
from .models import KeyMappingItem, KeyRule, MigrationMapping
from .parameter_batch import ParameterValue, prepare_parameter_batch
from .state import (
    MigrationSession,
    clear_pending,
    record_key_mapping,
    record_parameter_notes,
    record_pending,
)

_JSON = TypeAdapter(JsonObject)
_YAML = TypeAdapter(YamlValue)
_MAP = TypeAdapter(YamlMap)
_UNIT_KEYS = ("flux_unit", "jpa_flux_unit")


@dataclass(frozen=True)
class ResolvedFluxUnits:
    """Isolated values plus adopted source labels and blocked group diagnostics.

    source_values keeps actual source JSON, never the null used for blocked evidence.
    values adds accepted active unit keys, or null for blocked evidence.
    reasons names the actual source file or operator assumption by unit key.
    blocked names missing, invalid or conflicting evidence by unit key.
    local_flux_unit is a scalar value, not a numeric group's evidence.
    """

    source_values: JsonObject
    values: JsonObject
    reasons: dict[str, str]
    blocked: dict[str, str]


def _read_units(session: MigrationSession, path: Path) -> JsonObject:
    if not path.is_file():
        return {}
    session.baseline(path)
    return _JSON.validate_json(path.read_text(encoding="utf-8"), strict=True)


def _unit_evidence(
    key: str,
    source: Path,
    values: JsonObject,
    root: Path,
    root_meta: JsonObject,
    params: JsonObject,
) -> list[tuple[Path, object]]:
    # A point's explicit unit defines its own numbers, not an entry-level conflict.
    if source != root / "meta_info.json" and key in values:
        return [(source, values[key])]
    evidence: list[tuple[Path, object]] = []
    if key in root_meta:
        evidence.append((root / "meta_info.json", root_meta[key]))
    if key == "flux_unit" and key in params:
        evidence.append((root / "params.json", params[key]))
    return evidence


def resolve_flux_units(
    session: MigrationSession,
    source: Path,
    values: JsonObject,
    mapping: MigrationMapping,
) -> ResolvedFluxUnits:
    """Resolve active global/JPA groups without writes or numeric conversion.

    Global keys use flux_unit: same-file, root meta, params root, assumption.
    JPA uses jpa_flux_unit: same-file, root meta, assume_jpa_flux_unit, never
    qubit params or assume_flux_unit.
    Point explicit units take precedence; conflicting entry-level explicit units
    block only that group. Missing/invalid evidence stays pending. Source files
    are baseline-checked. local_flux_unit remains an independent scalar value.
    """
    root = session.manifest.report.source.result_path
    root_meta = (
        values
        if source == root / "meta_info.json"
        else _read_units(session, root / "meta_info.json")
    )
    params = _read_units(session, root / "params.json")
    active = {
        key
        for rule in mapping.rules
        if rule.old_key in values
        for key in (rule.old_key, *rule.requires_keys)
        if key in _UNIT_KEYS
    }
    result = dict(values)
    reasons: dict[str, str] = {}
    blocked: dict[str, str] = {}
    for key in sorted(active):
        declarations = session.manifest.report.declarations
        assumed = (
            declarations.assume_jpa_flux_unit
            if key == "jpa_flux_unit"
            else declarations.assume_flux_unit
        )
        evidence = _unit_evidence(key, source, values, root, root_meta, params)
        if evidence:
            selected = evidence[0][1]
            locations = "; ".join(f"{path}#{key}={value!r}" for path, value in evidence)
            if any(value != selected for _, value in evidence):
                blocked[key] = (
                    f"Conflicting explicit flux units: {locations}; no overwrite"
                )
            elif not isinstance(selected, str) or selected not in (
                "A",
                "mA",
                "V",
                "mV",
            ):
                blocked[key] = f"Invalid explicit flux unit: {locations}; no assumption"
            else:
                result[key] = selected
                reasons[key] = f"Unit evidence from {evidence[0][0]}#{key}"
        elif assumed is not None:
            result[key] = assumed
            reasons[key] = (
                f"假設單位: {result[key]}; operator supplied, no numeric conversion"
            )
        else:
            blocked[key] = f"Missing explicit evidence keys: {key}"
        if key in blocked:
            result[key] = None
    return ResolvedFluxUnits(dict(values), result, reasons, blocked)


def _clear_value(view: EditView, path: str) -> tuple[str, YamlValue]:
    if ".ext." in path:
        parent, key = path.rsplit(".", 1)
        current = document_value(snapshot(view), parent)
        if not isinstance(current, dict):
            raise ValueError(f"{parent}: expected extension mapping")
        value = {name: item for name, item in current.items() if name != key}
        view.set(parent, value)
        return parent, value
    view.set(path, None)
    return path, None


def _converted_updates(
    view: EditView,
    candidate: EditView,
    updates: list[tuple[str, YamlValue]],
    mapping: MigrationMapping,
    *,
    is_point: bool,
) -> tuple[list[tuple[str, YamlValue]], tuple[str, ...]]:
    before = _MAP.validate_python(
        snapshot(view).model_dump(
            mode="python", serialize_as_any=True, exclude_unset=True
        )
    )
    raw = _MAP.validate_python(
        snapshot(candidate).model_dump(
            mode="python", serialize_as_any=True, exclude_unset=True
        )
    )
    converted = prepare_parameter_batch(
        ParameterDocument(raw, Path("point.yaml" if is_point else "setup.yaml")),
        transform=mapping.parameter_transform,
        values=tuple(ParameterValue(path, value) for path, value in updates),
    )
    after = _MAP.validate_python(
        converted.document.model_dump(
            mode="python", serialize_as_any=True, exclude_unset=True
        )
    )
    changes: list[tuple[str, YamlValue]] = []
    for section in ("general", "components"):
        old_section, new_section = before[section], after[section]
        if not isinstance(old_section, dict) or not isinstance(new_section, dict):
            raise ValueError(f"{section}: expected mapping")
        if section == "general":
            changes.extend(
                (f"general.{key}", value)
                for key, value in new_section.items()
                if old_section.get(key) != value
            )
        else:
            for name, fields in new_section.items():
                old_fields = old_section.get(name)
                if not isinstance(fields, dict) or not isinstance(old_fields, dict):
                    raise ValueError(f"{name}: conversion cannot add/remove components")
                changes.extend(
                    (f"{name}.{key}", value)
                    for key, value in fields.items()
                    if old_fields.get(key) != value
                )
    return changes, converted.notes


def _stage_group(
    view: EditView,
    mapping: MigrationMapping,
    unit_rule: KeyRule,
    units: ResolvedFluxUnits,
    *,
    is_point: bool,
) -> tuple[list[tuple[str, YamlValue]], list[str], tuple[str, ...]]:
    rules = [rule for rule in mapping.rules if unit_rule.old_key in rule.requires_keys]
    candidate = EditView(snapshot(view))
    updates: list[tuple[str, YamlValue]] = []
    cleared: list[str] = []
    unit_path = unit_rule.target_path
    if unit_path is None:
        raise ValueError("A flux unit rule requires its explicit destination")
    try:
        previous = document_value(snapshot(candidate), unit_path)
    except AttributeError:
        previous = None
    unit = units.values[unit_rule.old_key]
    if is_point and previous != unit:
        for rule in rules:
            if rule.old_key not in units.values and rule.target_path is not None:
                try:
                    document_value(snapshot(candidate), rule.target_path)
                except AttributeError:
                    continue
                updates.append(_clear_value(candidate, rule.target_path))
                candidate.set_meta(rule.target_path, {})
                cleared.append(rule.target_path)
    for rule in (unit_rule, *rules):
        if rule.old_key not in units.values:
            continue
        value = units.values[rule.old_key]
        if has_legacy_expression(value):
            raise ValueError(f"{rule.old_key}: legacy expression; no evaluation")
        if rule.target_path is None:
            raise ValueError(f"{rule.old_key}: missing flux destination")
        accepted = _YAML.validate_python(value, strict=True)
        updates.append((rule.target_path, accepted))
    updates, notes = _converted_updates(
        view, candidate, updates, mapping, is_point=is_point
    )
    return updates, cleared, notes


def apply_flux_groups(
    session: MigrationSession,
    source: Path,
    target: Path,
    view: EditView,
    mapping: MigrationMapping,
    units: ResolvedFluxUnits,
) -> set[str]:
    """Stage each active flux group before accepting it into the context draft.

    units.source_values holds the actual JSON before unit resolution. Blocked
    source fields stay in general.ext.legacy_meta_info with literal report keys;
    synthetic evidence keys are never stored as original content.
    Owner-invalid units/values become located pending for the whole group.
    On a point unit change, clear inherited, unreplaced values and metadata only
    in that group. Accepted group updates share the caller's single edit commit.
    No runtime clone policy, source bytes, predictors or samples are changed.
    Return consumed source keys so ordinary scalar mapping does not reapply them.
    """
    consumed: set[str] = set()
    source_values = units.source_values
    is_point = source.parent != session.manifest.report.source.result_path
    for key in _UNIT_KEYS:
        unit_rule = next((rule for rule in mapping.rules if rule.old_key == key), None)
        if key not in units.values or unit_rule is None:
            continue
        rules = [rule for rule in mapping.rules if key in rule.requires_keys]
        present = [rule for rule in (unit_rule, *rules) if rule.old_key in units.values]
        consumed.update(rule.old_key for rule in present)
        error = units.blocked.get(key)
        if error is None:
            try:
                updates, cleared, notes = _stage_group(
                    view, mapping, unit_rule, units, is_point=is_point
                )
            except ValueError as exc:
                # Isolate owner-invalid legacy groups; accepted unrelated values remain usable.
                error = f"Flux group pending without conversion: {exc}"
            else:
                for path, value in updates:
                    view.set(path, value)
                for path in cleared:
                    view.set_meta(path, {})
                record_parameter_notes(session, target, notes)
        for rule in present:
            destination_keys = None
            reason = error or units.reasons[key]
            if error is None:
                clear_pending(session, source, rule.old_key)
            else:
                reason = (
                    f"{error}; blocked source group preserved at "
                    "general.ext.legacy_meta_info"
                )
                if rule.old_key in source_values:
                    destination_keys = preserve_meta_field(
                        view, rule.old_key, source_values[rule.old_key], legacy=True
                    )
                    reason = (
                        f"{error}; source preserved at "
                        f"general.ext.legacy_meta_info[{rule.old_key!r}]"
                    )
                record_pending(session, source, rule.old_key, reason)
            record_key_mapping(
                session,
                KeyMappingItem(
                    old_file=source,
                    old_key=rule.old_key,
                    new_file=target,
                    new_path=None if destination_keys else rule.target_path,
                    action="preserved"
                    if destination_keys
                    else "pending"
                    if error
                    else "value",
                    reason=reason,
                    source_keys=(rule.old_key,)
                    if rule.old_key in source_values
                    else None,
                    destination_keys=destination_keys,
                ),
            )
    return consumed
