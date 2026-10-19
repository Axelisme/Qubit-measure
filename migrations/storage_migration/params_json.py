"""Offline projection of the legacy analysis handoff into entry parameters."""

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import (
    PointView,
    ResultEntry,
    SetupView,
    component_registry,
    document_value,
    snapshot,
)

from .models import KeyMappingItem, MigrationMapping
from .paths import contained_path
from .state import MigrationSession, record_pending

_JSON = TypeAdapter(JsonObject)
_YAML = TypeAdapter(YamlValue)
_ENERGIES = ("EJ", "EC", "EL")
_FLUX_FIELDS = ("flux_half", "flux_int", "flux_period")


@dataclass
class _Context:
    """One destination and its already accepted meta_info values for dry-run."""

    target: Path
    view: SetupView | PointView | None
    values: YamlMap
    origins: dict[str, Path]

    def value(self, path: str) -> YamlValue:
        if self.view is not None:
            try:
                return document_value(snapshot(self.view), path)
            except AttributeError:
                return None
        return self.values.get(path)

    def set(self, path: str, value: YamlValue) -> None:
        if self.view is not None:
            with self.view.edit() as draft:
                draft.set(path, value)
        self.values[path] = value


def _setup_context(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
    name: str,
) -> _Context:
    seed = mapping.components[name]
    kind = seed["kind"]
    if not isinstance(kind, str):
        raise ValueError(f"{name}: seed requires kind")
    model = component_registry.get(kind).model_validate(seed)
    return _Context(
        target=session.manifest.report.destination.result_path / "setup.yaml",
        view=entry.setup if entry is not None and not session.dry_run else None,
        values={
            f"{name}.{key}": _YAML.validate_python(value, strict=True)
            for key, value in model.model_dump().items()
        },
        origins={},
    )


def _flux_unit(values: JsonObject, fit: JsonObject) -> tuple[str | None, str | None]:
    root_unit, fit_unit = values.get("flux_unit"), fit.get("flux_unit")
    if "flux_unit" in values and "flux_unit" in fit and root_unit != fit_unit:
        return None, "params.json flux_unit and fluxdep_fit.flux_unit disagree"
    unit = root_unit if "flux_unit" in values else fit_unit
    if not isinstance(unit, str) or unit not in ("A", "mA", "V", "mV"):
        return None, "Missing or unsupported explicit flux_unit (A/mA/V/mV)"
    return unit, None


def _put_field(
    context: _Context, path: str, value: JsonValue, source: Path
) -> str | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        return "Expected a finite numeric value; no expression evaluation or coercion"
    current = context.value(path)
    if current is not None and current != value:
        origin = context.origins.get(path, context.target)
        return f"Keep {origin}: {path}={current!r}; params source {source}: {value!r}"
    if current is None:
        context.set(path, value)
    return None


def _project_fit(
    context: _Context,
    source: Path,
    name: str,
    values: JsonObject,
    fields: set[str],
) -> list[KeyMappingItem]:
    fit = values.get("fluxdep_fit")
    if not isinstance(fit, dict):
        return []
    energies = fit.get("params")
    unit, unit_error = _flux_unit(values, fit)
    rows: list[KeyMappingItem] = []
    for field in (*_ENERGIES, *_FLUX_FIELDS):
        container = energies if field in _ENERGIES else fit
        if not isinstance(container, dict) or field not in container:
            continue
        key = (
            f"fluxdep_fit.params.{field}"
            if field in _ENERGIES
            else f"fluxdep_fit.{field}"
        )
        path = f"{name}.{field}"
        reason = (
            f"No formal field {path} in the destination component"
            if field not in fields
            else None
        )
        if reason is None and field in _FLUX_FIELDS:
            reason = unit_error or _context_unit_conflict(context, source, unit)
        if reason is None:
            reason = _put_field(context, path, container[field], source)
            if (
                reason is None
                and field in _FLUX_FIELDS
                and context.value("general.ext.flux_unit") is None
            ):
                context.set("general.ext.flux_unit", unit)
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=context.target,
                new_path=path,
                action="pending" if reason else "value",
                reason=reason
                or "Fill missing entry value from explicit legacy analysis result",
            )
        )
    return rows


def _context_unit_conflict(
    context: _Context, source: Path, unit: str | None
) -> str | None:
    current = context.value("general.ext.flux_unit")
    if current is None or current == unit:
        return None
    origin = context.origins.get("general.ext.flux_unit", context.target)
    return f"Keep {origin}: flux_unit={current!r}; params source {source}: flux_unit={unit!r}; do not reinterpret flux values or convert units"


def _preserve_sections(
    context: _Context, source: Path, name: str, values: JsonObject, *, has_ext: bool
) -> list[KeyMappingItem]:
    rows: list[KeyMappingItem] = []
    ext: YamlMap = {}
    current = context.value(f"{name}.ext")
    if isinstance(current, dict):
        ext.update(current)
    legacy: YamlMap = {}
    existing = ext.get("legacy_params")
    if isinstance(existing, dict):
        legacy.update(existing)
    for key, value in values.items():
        legacy[key] = _YAML.validate_python(value, strict=True)
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=context.target if has_ext else None,
                new_path=f"{name}.ext.legacy_params.{key}" if has_ext else None,
                action="pending",
                reason="Legacy analysis section and metadata preserved verbatim; review before formal use"
                if has_ext
                else "Destination component has no ext field; original handoff remains preserved",
            )
        )
    if has_ext:
        ext["legacy_params"] = legacy
        context.set(f"{name}.ext", ext)
    return rows


def convert_params_json(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
) -> None:
    """Fill setup analysis fields from params.json after component seeding.

    session owns baselines and report; entry is its destination handle or None in
    dry-run. mapping supplies seeds with exactly one qubit/* kind; otherwise the
    whole handoff stays pending. Run before meta contexts so new points inherit
    these defaults. Existing non-null seed values win. Meta conversion must call
    reconcile_params_meta after applying each context. Flux numbers require explicit
    units; no conversion or expression evaluation occurs. Preserve complete raw
    sections (including metadata) in component.ext.legacy_params with located
    pending rows. No publication/completion checkpoint is owned here. Validation
    and I/O errors propagate; old sources are never changed.
    """
    root = session.manifest.report.source.result_path
    candidate = root / "params.json"
    if not candidate.is_file():
        return
    source = contained_path(candidate, root)
    session.baseline(source)
    values = _JSON.validate_json(source.read_text(encoding="utf-8"), strict=True)
    names = [
        name
        for name, seed in mapping.components.items()
        if isinstance(seed.get("kind"), str) and str(seed["kind"]).startswith("qubit/")
    ]
    if len(names) != 1:
        record_pending(
            session,
            source,
            "params.json",
            "No unique migrated qubit component; keep the whole handoff pending without guessing",
        )
        return
    name = names[0]
    kind = mapping.components[name]["kind"]
    if not isinstance(kind, str):
        raise ValueError(f"{name}: seed requires kind")
    fields = set(component_registry.get(kind).model_fields)
    context = _setup_context(session, entry, mapping, name)
    rows = _project_fit(context, source, name, values, fields)
    rows.extend(
        _preserve_sections(context, source, name, values, has_ext="ext" in fields)
    )
    report = session.manifest.report
    session.report(
        replace(
            report,
            key_mappings=(
                *[row for row in report.key_mappings if row.old_file != source],
                *rows,
            ),
        )
    )
    reasons: dict[str, list[str]] = {}
    for row in rows:
        if row.action == "pending":
            reasons.setdefault(row.old_key, []).append(f"{row.new_file}: {row.reason}")
    for key, messages in reasons.items():
        record_pending(session, source, key, "; ".join(messages))
    session.baseline(source)


def _original_value(values: JsonObject, key: str) -> JsonValue:
    value: JsonValue = values
    for part in key.split("."):
        if not isinstance(value, dict):
            raise ValueError(f"{key}: invalid analysis field location")
        value = value[part]
    return value


def _conflict_pending(
    session: MigrationSession, source: Path, key: str, reason: str
) -> None:
    previous = next(
        (
            item.reason
            for item in session.manifest.report.pending
            if item.source == source and item.location == key
        ),
        None,
    )
    record_pending(
        session, source, key, f"{previous}; {reason}" if previous else reason
    )


def reconcile_params_meta(
    session: MigrationSession,
    source: Path,
    view: SetupView | PointView | None,
) -> None:
    """Report accepted meta overrides and remove inherited flux with different units.

    Call immediately after mapping one meta_info context. session's report is
    the parameter owner's acceptance record; source is that absolute meta file.
    view is the updated context, or None for dry-run. Meta values always win.
    When meta explicitly changes global flux_unit, clear only params-derived
    flux fields not explicitly supplied by that meta file. Never convert units.
    Keep params and meta source paths and values in located pending reasons.
    No source writes or completion checkpoint occur; I/O/validation propagates.
    """
    report = session.manifest.report
    params = report.source.result_path / "params.json"
    projected = {
        row.new_path: row
        for row in report.key_mappings
        if row.old_file == params and row.action == "value"
    }
    if not projected:
        return
    accepted = [
        row
        for row in report.key_mappings
        if row.old_file == source
        and row.action == "value"
        and not any(
            item.source == source and item.location == row.old_key
            for item in report.pending
        )
    ]
    if not accepted:
        return
    values = _JSON.validate_json(source.read_text(encoding="utf-8"), strict=True)
    original = _JSON.validate_json(params.read_text(encoding="utf-8"), strict=True)
    mapped = {row.new_path: values[row.old_key] for row in accepted}
    for path, row in projected.items():
        if path in mapped and mapped[path] != _original_value(original, row.old_key):
            _conflict_pending(
                session,
                params,
                row.old_key,
                f"Keep {source}: {path}={mapped[path]!r}; params source {params}: {_original_value(original, row.old_key)!r}",
            )
    unit_row = next(
        (row for row in accepted if row.new_path == "general.ext.flux_unit"), None
    )
    fit = original.get("fluxdep_fit")
    if unit_row is None or not isinstance(fit, dict):
        return
    unit, error = _flux_unit(original, fit)
    meta_unit = values[unit_row.old_key]
    if error is not None or meta_unit == unit:
        return
    for path, row in projected.items():
        if path is None or row.old_key not in {
            f"fluxdep_fit.{field}" for field in _FLUX_FIELDS
        }:
            continue
        _conflict_pending(
            session,
            params,
            row.old_key,
            f"Keep {source}: flux_unit={meta_unit!r}; params source {params}: flux_unit={unit!r}; no unit conversion",
        )
        if path not in mapped and view is not None and not session.dry_run:
            with view.edit() as draft:
                draft.set(path, None)
