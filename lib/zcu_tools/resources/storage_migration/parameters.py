"""Offline migration parameters ownership."""

from dataclasses import replace
from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.datafile import (
    JsonObject,
)
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import (
    PointView,
    ResultEntry,
    SetupView,
)

from .errors import MigrationInputError
from .evidence import has_legacy_expression
from .models import (
    KeyMappingItem,
    KeyRule,
    MigrationMapping,
)
from .paths import contained_path, validate_segment
from .state import MigrationSession, clear_pending, record_pending

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


def _accept_parameter(
    view: SetupView | PointView | None,
    rule: KeyRule,
    value: JsonValue,
    values: JsonObject,
    mapping: MigrationMapping,
) -> str | None:
    missing = tuple(
        key
        for key in rule.requires_keys
        if key not in values
        or values[key] is None
        or has_legacy_expression(values[key])
    )
    if missing:
        return f"Missing explicit evidence keys: {', '.join(missing)}"
    if view is None:
        return None
    path = rule.target_path
    if path is None:
        raise MigrationInputError(f"{rule.old_key}: missing target path")
    with view.edit() as draft:
        if rule.action == "stderr":
            provenance = view.meta(path)
            value_key = next(
                (
                    item.old_key
                    for item in mapping.rules
                    if item.action == "value" and item.target_path == path
                ),
                None,
            )
            if (
                provenance is None
                or value_key is None
                or value_key not in values
                or has_legacy_expression(values[value_key])
            ):
                return "No accepted explicit target value for stderr"
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise MigrationInputError(f"{rule.old_key}: stderr must be numeric")
            draft.set(
                path,
                _YAML_VALUE.validate_python(values[value_key], strict=True),
                provenance=replace(provenance, stderr=float(value)),
            )
        else:
            accepted = {rule.wrap_key: value} if rule.wrap_key is not None else value
            draft.set(path, _YAML_VALUE.validate_python(accepted, strict=True))
    return None


def _map_parameters(
    session: MigrationSession,
    source: Path,
    target: Path,
    values: JsonObject,
    mapping: MigrationMapping,
    view: SetupView | PointView | None,
) -> None:
    ext: YamlMap = {}
    stderr_keys = {rule.old_key for rule in mapping.rules if rule.action == "stderr"}
    # Values precede uncertainties even if JSON key order is reversed.
    for key in sorted(values, key=lambda key: key in stderr_keys):
        value = values[key]
        rule = next((item for item in mapping.rules if item.old_key == key), None)
        action = rule.action if rule is not None else "pending"
        path = rule.target_path if rule is not None else None
        reason = (
            rule.reason if rule is not None else "Unmapped key preserved in general.ext"
        )
        if has_legacy_expression(value):
            action, path, reason = (
                "pending",
                None,
                "Legacy expression preserved without evaluation",
            )
        if action == "module":
            reason = f"{reason}; 由 4a 轉換"
            record_pending(session, source, key, reason)
        elif action == "pending":
            ext[key] = _YAML_VALUE.validate_python(value, strict=True)
            record_pending(session, source, key, reason)
        elif action in ("value", "stderr") and rule is not None:
            unresolved = _accept_parameter(view, rule, value, values, mapping)
            if unresolved is None:
                clear_pending(session, source, key)
            else:
                reason = unresolved
                record_pending(session, source, key, reason)
        report = session.manifest.report
        items = tuple(
            item
            for item in report.key_mappings
            if (item.old_file, item.old_key) != (source, key)
        )
        new_file = (
            target.parent / "module_cfg.yaml"
            if action == "module"
            else target
            if action != "remove"
            else None
        )
        session.report(
            replace(
                report,
                key_mappings=(
                    *items,
                    KeyMappingItem(
                        old_file=source,
                        old_key=key,
                        new_file=new_file,
                        new_path=f"{path}.{rule.wrap_key}"
                        if path is not None
                        and rule is not None
                        and rule.wrap_key is not None
                        else path
                        if path is not None
                        else "general.ext"
                        if action == "pending"
                        else None,
                        action=action,
                        reason=reason,
                    ),
                ),
            )
        )

    if ext and view is not None:
        with view.edit() as draft:
            draft.set("general.ext", ext)


def _defer_modules(
    session: MigrationSession,
    source: Path,
    target: Path,
    mapping: MigrationMapping,
    view: SetupView | PointView | None,
) -> None:
    if not source.is_file():
        return
    source = contained_path(source, session.manifest.report.source.result_path)
    session.baseline(source)
    modules = _YAML_MAP.validate_python(
        YAML(typ="safe").load(source.read_text(encoding="utf-8")), strict=True
    )
    references: dict[str, str] = {}
    for rule in mapping.module_rules:
        if rule.old_name in modules and rule.reference_path is not None:
            references.setdefault(rule.reference_path, rule.target_path)
    if view is not None and references:
        with view.edit() as draft:
            for path, reference in references.items():
                draft.set(path, reference)
    for name in modules:
        rule = next(
            (rule for rule in mapping.module_rules if rule.old_name == name), None
        )
        reason = (
            f"{rule.reason}; 由 4a 轉換"
            if rule is not None
            else ("Unknown legacy module; no inferred destination; 由 4a 轉換")
        )
        if rule is not None and rule.reference_path is not None:
            reason += (
                f"; {rule.reference_path} references {references[rule.reference_path]}"
            )
        record_pending(session, source, name, reason)
        report = session.manifest.report
        items = tuple(
            item
            for item in report.key_mappings
            if (item.old_file, item.old_key) != (source, name)
        )
        session.report(
            replace(
                report,
                key_mappings=(
                    *items,
                    KeyMappingItem(
                        old_file=source,
                        old_key=name,
                        new_file=target,
                        new_path=rule.target_path if rule is not None else None,
                        action="module",
                        reason=reason,
                    ),
                ),
            )
        )
    session.baseline(source)


def _convert_context(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
    candidate: Path,
) -> None:
    source_root = session.manifest.report.source.result_path
    destination_root = session.manifest.report.destination.result_path
    source = contained_path(candidate, source_root)
    session.baseline(source)
    values = _JSON.validate_json(source.read_text(encoding="utf-8"), strict=True)
    is_setup = source == source_root / "meta_info.json"
    label = candidate.parent.name
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
    _map_parameters(session, source, target, values, mapping, view)
    _defer_modules(
        session,
        candidate.parent / "module_cfg.yaml",
        target.parent / "module_cfg.yaml",
        mapping,
        view,
    )
    session.baseline(source)


def convert_parameters(
    session: MigrationSession, entry: ResultEntry | None, mapping: MigrationMapping
) -> None:
    """Convert explicit legacy meta keys into independent complete entry points.

    session owns progress/report and sources; entry is its handle or None in a new
    dry-run; mapping provides registered seeds and working-unit key decisions.
    Preserve unknown/expressions in ext, defer module cfg and copy params bytes.
    Completed parameters are not rewritten. Checkpoint completion; input/schema
    and I/O failures propagate without modifying the old result tree."""
    if session.manifest.parameters_complete:
        return
    source_root = session.manifest.report.source.result_path
    destination_root = session.manifest.report.destination.result_path
    if entry is not None and not session.dry_run:
        _seed_components(entry, mapping)
    sources = sorted(source_root.glob("*/meta_info.json"))
    root_meta = source_root / "meta_info.json"
    if root_meta.is_file():
        sources.insert(0, root_meta)
    for candidate in sources:
        _convert_context(session, entry, mapping, candidate)
    params = source_root / "params.json"
    if params.is_file():
        state = session.publish(
            contained_path(params, source_root),
            destination_root / "params.json",
            operation="copy",
        )
        session.completed(state)
    session.manifest = replace(session.manifest, parameters_complete=True)
    session.checkpoint()
