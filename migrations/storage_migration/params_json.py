"""Offline conversion of legacy analysis handoffs into entry-scoped resources."""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from zcu_tools.datafile import JsonObject, json_values_equal
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import (
    ResultEntry,
    SetupView,
    component_registry,
    document_value,
    snapshot,
)
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorSource,
    save_predictor_file,
    validate_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from .analysis_files import convert_analysis_files, proven_created_at
from .backup import backup_source_path
from .documents import edit_parameter, publish_document
from .models import KeyMappingItem, MigrationMapping
from .paths import contained_path
from .state import MigrationSession, record_pending

_JSON = TypeAdapter(JsonObject)
_YAML = TypeAdapter(YamlValue)
_ENERGIES = ("EJ", "EC", "EL")
_FLUX_FIELDS = ("flux_half", "flux_period")


@dataclass
class _Context:
    """Setup metadata, including seed values when no dry-run handle exists."""

    target: Path
    session: MigrationSession
    source: Path
    view: SetupView | None
    values: YamlMap

    def value(self, path: str) -> YamlValue:
        if self.view is not None:
            try:
                return document_value(snapshot(self.view), path)
            except AttributeError:
                return None
        return self.values.get(path)

    def set(self, path: str, value: YamlValue) -> None:
        if self.view is not None:
            with edit_parameter(self.session, self.source, self.view) as draft:
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
        session=session,
        source=session.manifest.report.source.result_path / "params.json",
        view=entry.setup if entry is not None and not session.dry_run else None,
        values={
            f"{name}.{key}": _YAML.validate_python(value, strict=True)
            for key, value in model.model_dump().items()
        },
    )


def _flux_unit(
    values: JsonObject, fit: JsonObject, assumed: str | None
) -> tuple[str | None, str | None]:
    if "flux_unit" not in values and "flux_unit" not in fit and assumed is not None:
        return assumed, None
    root_unit, fit_unit = values.get("flux_unit"), fit.get("flux_unit")
    if "flux_unit" in values and "flux_unit" in fit and root_unit != fit_unit:
        return None, "params.json flux_unit and fluxdep_fit.flux_unit disagree"
    unit = root_unit if "flux_unit" in values else fit_unit
    if not isinstance(unit, str) or unit not in ("A", "mA", "V", "mV"):
        return None, "Missing or unsupported explicit flux_unit (A/mA/V/mV)"
    return unit, None


def _project_model_params(
    source: Path, target: Path, fit: JsonObject, prefix: tuple[str, ...]
) -> tuple[FluxoniumParams, list[KeyMappingItem]]:
    """Project supported numeric fields through the model owner, recording exact source subtrees."""
    energies = fit.get("params")
    params: JsonObject = {"flux_bias": 0.0}
    rows: list[KeyMappingItem] = []
    if energies == {}:
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=".".join((*prefix, "params")),
                source_keys=(*prefix, "params"),
                new_file=target,
                new_path="params",
                action="value",
                reason="Empty supported parameter container transferred to predictor",
            )
        )
    flux_fields = (
        tuple((field, field) for field in _FLUX_FIELDS)
        if prefix
        else (("half flux", "flux_half"), ("period", "flux_period"))
    )
    for source_field, field in (
        *((energy, energy) for energy in _ENERGIES),
        *flux_fields,
    ):
        container = energies if field in _ENERGIES else fit
        if not isinstance(container, dict) or source_field not in container:
            continue
        value = container[source_field]
        reason = None
        try:
            FluxoniumParams.from_mapping({field: value})
        except ValueError:
            reason = "Expected a finite numeric value or null; no expression evaluation or coercion"
        else:
            params[field] = value
        keys = (
            (*prefix, "params", source_field)
            if field in _ENERGIES
            else (*prefix, source_field)
        )
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=".".join(keys),
                source_keys=keys,
                new_file=target,
                new_path=f"params.{field}",
                action="pending" if reason else "value",
                reason=reason or "Preserve legacy fluxonium parameter in native units",
            )
        )
    return FluxoniumParams.from_mapping(params), rows


def _unsupported_model_params(energies: JsonValue) -> bool:
    return (
        isinstance(energies, dict)
        and not {"EJ", "EL"}.intersection(energies)
        and bool(energies.keys() - set(_ENERGIES))
    )


def _project_fit(
    session: MigrationSession,
    source: Path,
    name: str,
    values: JsonObject,
) -> list[KeyMappingItem]:
    fit = values.get("fluxdep_fit")
    prefix = ("fluxdep_fit",)
    if "fluxdep_fit" not in values and (
        "params" in values or "half flux" in values or "period" in values
    ):
        fit = values
        prefix = ()
    if not isinstance(fit, dict):
        return []
    target = (
        session.manifest.report.destination.result_path / "predictors" / f"{name}.json"
    )
    energies = fit.get("params")
    unsupported = None
    if "model" in fit:
        unsupported = ".".join((*prefix, "model"))
    elif _unsupported_model_params(energies):
        # Unsupported model shapes are preserved, not missing-evidence pending.
        return []
    if unsupported is not None:
        return [
            KeyMappingItem(
                old_file=source,
                old_key=unsupported,
                new_file=None,
                new_path=None,
                action="pending",
                reason="Model or parameter names have no supported legacy fluxdep_fit contract; preserve without guessing",
            )
        ]
    params, rows = _project_model_params(source, target, fit, prefix)
    assumed = session.manifest.report.declarations.assume_flux_unit
    unit, unit_error = _flux_unit(values, fit, assumed)
    created_at = proven_created_at(fit.get("timestamp"))
    backup = session.manifest.report.backup_path
    original = (
        backup_source_path(backup, session.manifest.report.source, source)
        if backup is not None
        else source
    )
    unit_key = (
        "flux_unit"
        if unit_error is None and "flux_unit" in values
        else ".".join((*prefix, "flux_unit"))
    )
    unit_source_keys = (
        ("flux_unit",) if "flux_unit" in values else (*prefix, "flux_unit")
    )
    timestamp_keys = (*prefix, "timestamp")
    timestamp_key = ".".join(timestamp_keys)
    for key, path, reason in (
        (unit_key, "flux_unit", unit_error),
        (
            timestamp_key,
            "source.created_at",
            None
            if created_at is not None
            else f"No proven timezone-aware ISO 8601 timestamp; original at {original}#{timestamp_key}",
        ),
    ):
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=target,
                new_path=path,
                action="pending" if reason else "value",
                source_keys=unit_source_keys if path == "flux_unit" else timestamp_keys,
                reason=reason
                or (
                    f"假設單位: {assumed}; operator supplied, no numeric conversion"
                    if path == "flux_unit"
                    and assumed is not None
                    and "flux_unit" not in values
                    and "flux_unit" not in fit
                    else "Preserve explicit legacy source metadata"
                ),
            )
        )
    if prefix and unit_error is None and "flux_unit" in values and "flux_unit" in fit:
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key="fluxdep_fit.flux_unit",
                new_file=target,
                new_path="flux_unit",
                source_keys=("fluxdep_fit", "flux_unit"),
                action="value",
                reason="Matching explicit fit unit shares the predictor unit destination",
            )
        )
    predictor = PredictorFile(
        model="fluxonium",
        params=params,
        flux_unit=unit,
        source=PredictorSource(run_id=None, created_at=created_at),
    )
    validate_predictor_file(predictor)
    if not session.dry_run:
        publish_document(
            session,
            source,
            target,
            "predictor",
            lambda path: save_predictor_file(path, predictor),
        )
    return rows


def _compare_source_values(
    formal: KeyMappingItem,
    old_keys: tuple[str, ...],
    newer_keys: tuple[str, ...],
    old: JsonValue,
    newer: JsonValue,
    *,
    newer_present: bool = True,
) -> list[KeyMappingItem]:
    """Compare corresponding source leaves, never unrelated equal numbers."""
    if isinstance(old, dict) and old and isinstance(newer, dict):
        rows: list[KeyMappingItem] = []
        for key, value in old.items():
            rows.extend(
                _compare_source_values(
                    formal,
                    (*old_keys, key),
                    (*newer_keys, key),
                    value,
                    newer.get(key),
                    newer_present=key in newer,
                )
            )
        return rows
    equal = newer_present and json_values_equal(old, newer)
    newer_display = repr(newer) if newer_present else "<absent>"
    return [
        KeyMappingItem(
            old_file=formal.old_file,
            old_key=".".join(old_keys),
            source_keys=old_keys,
            compared_source_keys=newer_keys,
            new_file=formal.new_file if equal else None,
            new_path=formal.new_path if equal else None,
            action="remove" if equal else "preserved",
            reason=(
                f"{'.'.join(old_keys)}={old!r} "
                f"{'matches' if equal else 'differs from'} "
                f"{'.'.join(newer_keys)}={newer_display}; "
                + (
                    "deduplicated against its formal destination"
                    if equal
                    else "newer source takes precedence; preserve historical value verbatim"
                )
            ),
        )
    ]


def _compare_top_sources(
    values: JsonObject, converted: list[KeyMappingItem]
) -> list[KeyMappingItem]:
    fit = values.get("fluxdep_fit")
    if not isinstance(fit, dict):
        return []
    rows: list[KeyMappingItem] = []
    pairs = (
        *((("params", field), ("fluxdep_fit", "params", field)) for field in _ENERGIES),
        (("params",), ("fluxdep_fit", "params")),
        (("half flux",), ("fluxdep_fit", "flux_half")),
        (("period",), ("fluxdep_fit", "flux_period")),
        (("allows",), ("fluxdep_fit", "plot_transitions")),
    )
    for old_keys, newer_keys in pairs:
        formal = next(
            (
                row
                for row in converted
                if row.action == "value" and row.source_keys == newer_keys
            ),
            None,
        )
        old_container = values.get("params") if len(old_keys) == 2 else values
        newer_container = fit.get("params") if len(newer_keys) == 3 else fit
        if (
            formal is not None
            and isinstance(old_container, dict)
            and isinstance(newer_container, dict)
            and old_keys[-1] in old_container
            and newer_keys[-1] in newer_container
        ):
            rows.extend(
                _compare_source_values(
                    formal,
                    old_keys,
                    newer_keys,
                    old_container[old_keys[-1]],
                    newer_container[newer_keys[-1]],
                )
            )
    return rows


def _unconverted_values(
    values: JsonObject, converted: list[KeyMappingItem]
) -> JsonObject:
    """Subtract only successfully transferred source subtrees and prune their empty parents."""
    remaining = deepcopy(values)
    for row in converted:
        if not row.source_keys or not (
            row.action == "value"
            or (row.action == "remove" and row.compared_source_keys is not None)
        ):
            continue
        cursor = remaining
        parents: list[tuple[JsonObject, str]] = []
        for key in row.source_keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                break
            parents.append((cursor, key))
            cursor = child
        else:
            cursor.pop(row.source_keys[-1], None)
            for parent, key in reversed(parents):
                if parent[key] == {}:
                    del parent[key]
    return remaining


def _preserve_sections(
    context: _Context,
    source: Path,
    name: str,
    values: JsonObject,
    converted: list[KeyMappingItem],
    *,
    has_ext: bool,
) -> list[KeyMappingItem]:
    remaining = _unconverted_values(values, converted)
    rows: list[KeyMappingItem] = []
    ext: YamlMap = {}
    current = context.value(f"{name}.ext")
    if isinstance(current, dict):
        ext.update(current)
    legacy: YamlMap = {}
    existing = ext.get("legacy_params")
    if isinstance(existing, dict):
        legacy.update(existing)
    for key in values:
        legacy.pop(key, None)
    for key, value in remaining.items():
        legacy[key] = _YAML.validate_python(value, strict=True)

    def record(value: JsonValue, keys: tuple[str, ...]) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                record(child, (*keys, key))
            return
        comparison = next(
            (
                row
                for row in converted
                if row.action == "preserved"
                and row.source_keys is not None
                and keys[: len(row.source_keys)] == row.source_keys
            ),
            None,
        )
        model_prefix = ("fluxdep_fit",) if "fluxdep_fit" in values else ()
        fit = values.get("fluxdep_fit") if model_prefix else values
        unsupported = (
            isinstance(fit, dict)
            and "model" not in fit
            and _unsupported_model_params(fit.get("params"))
            and keys[: len(model_prefix) + 1] == (*model_prefix, "params")
        )
        reason = (
            comparison.reason
            if comparison is not None
            else "Unsupported model parameter shape; original value preserved verbatim"
            if unsupported
            else "Unconverted legacy value preserved verbatim; not semantic conversion"
        )
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=".".join(keys),
                source_keys=keys,
                compared_source_keys=comparison.compared_source_keys
                if comparison is not None
                else None,
                new_file=context.target if has_ext else None,
                new_path=None,
                destination_keys=(name, "ext", "legacy_params", *keys)
                if has_ext
                else None,
                action="preserved" if has_ext else "pending",
                reason=reason
                if has_ext
                else "Destination component has no ext field; original handoff remains in source backup",
            )
        )

    for key, value in remaining.items():
        record(value, (key,))
    if has_ext:
        if legacy:
            ext["legacy_params"] = legacy
        else:
            ext.pop("legacy_params", None)
        context.set(f"{name}.ext", ext)
    return rows


def convert_params_json(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
) -> None:
    """Preserve metadata and publish entry predictor/dispersive/T1 fit files.

    session owns baselines and report; entry is its destination handle or None in
    dry-run. mapping supplies seeds with exactly one qubit/* kind; otherwise the
    whole handoff stays pending. Call before points so they inherit raw residuals,
    then apply each context's meta_info normally. Legacy EJ/EC/EL fits use the
    documented fluxonium model and zero bias; flux_int is alignment-only. An
    explicit model is unsupported and stays pending. Earlier root params.EJ/EC/EL,
    half flux and period use the same predictor owner, without numeric conversion.
    EJ or EL identifies fluxonium even with extra parameters; only EC plus unknown
    parameters is an unsupported model, preserved without pending. If both shapes
    occur, fluxdep_fit takes precedence; corresponding equal old leaves are removed
    from raw residuals, while different values retain both-source comparison rows.
    Unknown/invalid numbers
    remain unknown, with invalid values located in pending. Units only come from
    explicit root/fit flux_unit, or the declared assumption only when both keys
    are absent. Otherwise missing/conflicting units become null and pending,
    without conversion. Source run_id is unknown; only a timezone-aware ISO 8601
    timestamp supplies created_at. Preserve only unconverted source values in
    component.ext.legacy_params. Successful formal mappings remove their source
    subtrees from that raw container, including supported empty containers, and
    prune emptied parents. Unknown empty objects/lists remain original values;
    unsupported or invalid content is never removed merely because a file exists.
    Report each remaining leaf, list or empty object with a typed preserved action
    and literal source_keys/destination_keys; raw preservation alone is not pending.
    Without an ext field, preservation stays pending and reports no destination.
    Predictors remain entry
    resources and never participate in setup/point inheritance. Dispersive and T1
    file owners validate partial data and source evidence; invalid content stays
    pending with its original ext copy. plot_transitions joins the dispersive
    file without inventing fit values. Earlier allows transfers unchanged as opaque
    plot_transitions annotations. Newer fluxdep_fit.plot_transitions wins when both
    occur; only different/unmatched allows leaves remain raw. Dry-run reports
    the same plan without writes. The parameters owner checkpoints completion;
    input/I/O failures propagate and sources are never changed here.
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
    rows = _project_fit(session, source, name, values)
    rows.extend(convert_analysis_files(session, entry, source, name, values))
    rows.extend(_compare_top_sources(values, rows))
    preserved = _preserve_sections(
        context, source, name, values, rows, has_ext="ext" in fields
    )
    # Finalize comparison differences through the same ext capability/location
    # check as all other raw residuals, without duplicate diagnostic rows.
    rows = [row for row in rows if row.action != "preserved"] + preserved
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
