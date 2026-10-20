"""Offline migration parameters ownership."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from zcu_tools.datafile import (
    JsonObject,
)
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import (
    EditView,
    ResultEntry,
    document_value,
    snapshot,
)
from zcu_tools.resources.entry.schema import SetupDocument

from ._flux_units import apply_flux_groups, resolve_flux_units
from ._module_cfg import convert_module_cfg
from .errors import MigrationInputError
from .evidence import has_legacy_expression
from .models import (
    KeyMappingItem,
    KeyRule,
    MappingAction,
    MigrationMapping,
)
from .params_json import convert_params_json
from .paths import contained_path, validate_segment
from .samples import convert_samples
from .state import MigrationSession, clear_pending, record_key_mapping, record_pending

_YAML_VALUE = TypeAdapter(YamlValue)
_YAML_MAP = TypeAdapter(YamlMap)
_JSON = TypeAdapter(JsonObject)


def _seed_components(entry: ResultEntry, mapping: MigrationMapping) -> None:
    for name, seed in mapping.components.items():
        fields = _YAML_MAP.validate_python(seed, strict=True)
        kind = fields.pop("kind", None)
        if not isinstance(kind, str):
            raise MigrationInputError(f"{name}: seed requires kind")
        # Interrupted parameter work can already have published this seed.
        try:
            getattr(entry.setup, name)
        except AttributeError:
            entry.setup.add_component(name, kind=kind, **fields)


def _missing_evidence(rule: KeyRule, values: JsonObject) -> tuple[str, ...]:
    return tuple(
        key
        for key in rule.requires_keys
        if key not in values
        or values[key] is None
        or has_legacy_expression(values[key])
    )


def _value_accepted(path: str, values: JsonObject, mapping: MigrationMapping) -> bool:
    """Whether this mapping's value rule for path accepts its explicit value."""
    rule = next(
        (
            item
            for item in mapping.rules
            if item.action == "value" and item.target_path == path
        ),
        None,
    )
    return (
        rule is not None
        and rule.old_key in values
        and not has_legacy_expression(values[rule.old_key])
        and not _missing_evidence(rule, values)
    )


def _accept_parameter(
    view: EditView,
    rule: KeyRule,
    value: JsonValue,
    values: JsonObject,
    mapping: MigrationMapping,
) -> str | None:
    missing = _missing_evidence(rule, values)
    if missing:
        return f"Missing explicit evidence keys: {', '.join(missing)}"
    path = rule.target_path
    if path is None:
        raise MigrationInputError(f"{rule.old_key}: missing target path")
    if rule.action == "stderr":
        if not _value_accepted(path, values, mapping):
            return "No accepted explicit target value for stderr"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MigrationInputError(f"{rule.old_key}: stderr must be numeric")
        view.set_meta(path, {**(view.meta(path) or {}), "stderr": float(value)})
    else:
        accepted = {rule.wrap_key: value} if rule.wrap_key is not None else value
        try:
            view.set(path, _YAML_VALUE.validate_python(accepted, strict=True))
        except ValueError as exc:
            if rule.old_key != "local_flux_unit":
                raise
            return f"Local unit incompatible with component owner; no conversion: {exc}"
    return None


def _map_module_scalar(
    session: MigrationSession,
    source: Path,
    rule: KeyRule | None,
    mapping: MigrationMapping,
    view: EditView,
    value: JsonValue,
) -> tuple[MappingAction, str | None, str]:
    if rule is None:
        raise MigrationInputError("A module scalar requires its declared key rule")
    path = _module_scalar_destination(session, source, rule, mapping)
    if path is None:
        reason = f"{rule.reason}; module_cfg conversion owns this value"
        record_pending(session, source, rule.old_key, reason)
        return "module", rule.target_path, reason
    view.set(path, _YAML_VALUE.validate_python(value, strict=True))
    clear_pending(session, source, rule.old_key)
    return (
        "value",
        path,
        "No converted matching module; preserve scalar in component.ext",
    )


def _map_parameters(
    session: MigrationSession,
    source: Path,
    target: Path,
    values: JsonObject,
    mapping: MigrationMapping,
    view: EditView,
) -> None:
    ext: YamlMap = {}
    units = resolve_flux_units(session, source, values, mapping)
    values = units.values
    consumed = apply_flux_groups(session, source, target, view, mapping, units)
    stderr_keys = {rule.old_key for rule in mapping.rules if rule.action == "stderr"}
    # Values precede uncertainties even if JSON key order is reversed.
    for key in sorted(set(values) - consumed, key=lambda key: key in stderr_keys):
        value = values[key]
        rule = next((item for item in mapping.rules if item.old_key == key), None)
        action = rule.action if rule is not None else "pending"
        path = rule.target_path if rule is not None else None
        reason = (
            rule.reason if rule is not None else "Unmapped key preserved in general.ext"
        )
        if key == "qub_ch" and "qub_0_1_ch" in values:
            action, path, reason = (
                "remove",
                None,
                "qub_0_1_ch takes precedence over qub_ch",
            )
        if has_legacy_expression(value):
            action, path, reason = (
                "pending",
                None,
                "Legacy expression preserved without evaluation",
            )
        if action == "module":
            action, path, reason = _map_module_scalar(
                session, source, rule, mapping, view, value
            )
        elif action == "pending":
            ext[key] = _YAML_VALUE.validate_python(value, strict=True)
            record_pending(session, source, key, reason)
        elif action in ("value", "stderr") and rule is not None:
            unresolved = _accept_parameter(view, rule, value, values, mapping)
            if unresolved is None:
                clear_pending(session, source, key)
            else:
                action = "pending"
                reason = unresolved
                record_pending(session, source, key, reason)
        new_file = (
            target.parent / "module_cfg.yaml"
            if action == "module"
            else target
            if action != "remove"
            else None
        )
        record_key_mapping(
            session,
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=new_file,
                new_path=f"{path}.{rule.wrap_key}"
                if path is not None and rule is not None and rule.wrap_key is not None
                else path
                if path is not None
                else "general.ext"
                if action == "pending"
                else None,
                action=action,
                reason=reason,
            ),
        )

    if ext:
        current = document_value(snapshot(view), "general.ext")
        view.set(
            "general.ext", {**(current if isinstance(current, dict) else {}), **ext}
        )


def _module_scalar_destination(
    session: MigrationSession,
    source: Path,
    rule: KeyRule | None,
    mapping: MigrationMapping,
) -> str | None:
    if rule is None or rule.target_path is None:
        return None
    segments = rule.target_path.split(".")
    name = mapping.roles.get(segments[0], segments[0])
    slot = ".".join((name, *segments[1:3]))
    converted = {
        row.new_path
        for row in session.manifest.report.key_mappings
        if row.old_file == source.with_name("module_cfg.yaml")
        and row.library_kind == "module"
        and row.action == "module"
    }
    if any(
        item.reference_path == slot and item.target_path in converted
        for item in mapping.module_rules
    ):
        return None
    if name not in mapping.components:
        return None
    return f"{name}.ext.{rule.old_key}"


def _retain_point_slots(
    entry: ResultEntry, label: str, mapping: MigrationMapping
) -> None:
    # New points copy setup YAML but deliberately start with an empty library.
    # Keep only references that this point's own published library can resolve.
    published = entry.module_library(label).snapshot().module_keys()
    with entry.use_point(label).edit() as draft:
        document = snapshot(draft)
        for name in mapping.components:
            path = f"{name}.module"
            try:
                slots = document_value(document, path)
            except AttributeError:
                continue  # Components need not declare a module-slot container.
            if isinstance(slots, dict):
                retained: YamlMap = {
                    slot: key for slot, key in slots.items() if key in published
                }
                if retained != slots:
                    draft.set(path, retained)


def _convert_context(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
    candidate: Path,
    dry_setup: EditView | None,
) -> None:
    source_root = session.manifest.report.source.result_path
    destination_root = session.manifest.report.destination.result_path
    contained_path(candidate, source_root)
    source = contained_path(candidate / "meta_info.json", source_root)
    is_setup = candidate == source_root
    label = candidate.name
    validate_segment(label, "context label")
    target = (
        destination_root / "setup.yaml"
        if is_setup
        else destination_root / "points" / label / "point.yaml"
    )
    view = None
    if entry is not None and not session.dry_run:
        view = (
            entry.setup
            if is_setup
            else entry.use_point(label)
            if label in entry.list_points()
            else entry.new_point(label)
        )
        if not is_setup:
            _retain_point_slots(entry, label, mapping)
    dry_context = (
        (dry_setup if is_setup else EditView(snapshot(dry_setup)))
        if dry_setup is not None
        else None
    )
    convert_module_cfg(
        session,
        entry=entry,
        label=None if is_setup else label,
        source=candidate / "module_cfg.yaml",
        mapping=mapping,
        dry_view=dry_context,
    )
    if source.is_file():
        session.baseline(source)
        values = _JSON.validate_json(source.read_text(encoding="utf-8"), strict=True)
        if view is not None:
            with view.edit() as draft:
                _map_parameters(session, source, target, values, mapping, draft)
        elif dry_context is not None:
            _map_parameters(session, source, target, values, mapping, dry_context)
        else:
            raise MigrationInputError(
                "Context mapping requires an entry or dry-run draft"
            )
        session.baseline(source)


def _context_directories(source_root: Path) -> list[Path]:
    children = sorted(path.parent for path in source_root.glob("*/meta_info.json"))
    labels = {path.name: path for path in children}
    for child in sorted((source_root / "exps").glob("*")):
        if not child.is_dir():
            continue
        if child.name in labels:
            raise MigrationInputError(
                f"Duplicate point label {child.name!r}: {labels[child.name]} and {child}"
            )
        children.append(child)
    if any(
        (source_root / name).is_file() for name in ("meta_info.json", "module_cfg.yaml")
    ):
        children.insert(0, source_root)
    return children


def convert_parameters(
    session: MigrationSession, entry: ResultEntry | None, mapping: MigrationMapping
) -> None:
    """Convert explicit legacy meta keys into independent complete entry points.

    session owns progress/report and sources; entry is its handle or None in a new
    dry-run; mapping provides registered seeds and working-unit key decisions.
    Root metadata seeds setup; direct metadata children and every exps child
    become points. Reject duplicate labels with both source locations before
    point writes. Preserve unknown/expressions in ext, convert typed module cfg
    and convert samples with explicit independent CSV declarations. Original
    params bytes remain only in the complete verified backup, not an entry copy.
    Completed parameters are not rewritten. Checkpoint completion; input/schema
    and I/O failures propagate without modifying the old result tree."""
    if session.manifest.parameters_complete:
        return
    source_root = session.manifest.report.source.result_path
    if entry is not None and not session.dry_run:
        _seed_components(entry, mapping)
    convert_params_json(session, entry, mapping)
    dry_setup = None
    if session.dry_run:
        dry_setup = EditView(
            SetupDocument.model_validate(
                {
                    "format": "zcu.parameter-container",
                    "format_version": "2.0",
                    "general": {
                        "entry_id": str(session.manifest.identity.entry_id),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "components": dict(mapping.components),
                }
            )
        )
    for candidate in _context_directories(source_root):
        _convert_context(session, entry, mapping, candidate, dry_setup)
    convert_samples(session, entry, mapping)
    session.manifest = replace(session.manifest, parameters_complete=True)
    session.checkpoint()
