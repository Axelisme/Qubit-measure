"""Typed setup/point legacy library conversion in the parameter migration step."""

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter, ValidationError
from ruamel.yaml import YAML
from zcu_tools.cfg_model import (
    partition_module_spec,
    partition_waveform_spec,
    remap_module_references,
    remap_waveform_references,
)
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.context.library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import ResultEntry, document_value, snapshot
from zcu_tools.resources.entry.views import EditView

from ._legacy_modules import legacy_module_candidate
from .documents import edit_parameter, publish_document
from .errors import MigrationInputError
from .models import (
    FileMigrationItem,
    KeyMappingItem,
    MigrationMapping,
    ModuleFilePublication,
    ModuleRule,
)
from .paths import contained_path
from .state import MigrationSession, clear_pending, file_hash, record_pending

_JSON = TypeAdapter(JsonObject)
_PARTITIONS: tuple[tuple[Literal["module", "waveform"], str], ...] = (
    ("module", "modules"),
    ("waveform", "waveforms"),
)


@dataclass(frozen=True)
class _EntrySource:
    source: Path
    kind: Literal["module", "waveform"]
    keys: tuple[str, str]
    raw: JsonValue


def _record_decision(session: MigrationSession, decision: KeyMappingItem) -> None:
    report = session.manifest.report
    items = tuple(
        item
        for item in report.key_mappings
        if (
            item.old_file,
            item.old_key,
            item.library_kind,
            item.source_keys,
            item.action == "preserved",
        )
        != (
            decision.old_file,
            decision.old_key,
            decision.library_kind,
            decision.source_keys,
            decision.action == "preserved",
        )
    )
    session.report(replace(report, key_mappings=(*items, decision)))


class _SpecInputError(Exception):
    """Only input/remap failures may be isolated from library publication errors."""

    def __init__(self, cause: ValueError | KeyError) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _convert_spec(
    candidate: LibrarySnapshot,
    raw: JsonObject,
    rule: ModuleRule,
    mapping: MigrationMapping,
) -> tuple[LibrarySnapshot, JsonObject]:
    """Convert a supported entry and return only its unexpressed original fields.

    Spec/reference failures propagate for entry-level isolation by the caller.
    Destination document validation remains a whole-operation failure.
    """
    modules = {
        item.old_name: item.target_path
        for item in mapping.module_rules
        if item.kind == "module"
    }
    waveforms = {
        item.old_name: item.target_path
        for item in mapping.module_rules
        if item.kind == "waveform"
    }
    if rule.kind == "module":
        try:
            legacy = legacy_module_candidate(raw)
            projection = partition_module_spec(legacy.values)
            module = remap_module_references(
                projection.spec, modules=modules, waveforms=waveforms
            )
        except (ValueError, KeyError) as exc:
            raise _SpecInputError(exc) from exc
        return candidate.replaced_module(
            rule.target_path, module
        ), legacy.original_unknown(projection.unrecognized)
    try:
        projection_waveform = partition_waveform_spec(raw)
        waveform = remap_waveform_references(
            projection_waveform.spec, waveforms=waveforms
        )
    except (ValueError, KeyError) as exc:
        raise _SpecInputError(exc) from exc
    return candidate.replaced_waveform(
        rule.target_path, waveform
    ), projection_waveform.unrecognized


def _preserved_paths(
    original: JsonValue, retained: JsonValue, keys: tuple[str, ...]
) -> list[tuple[str, ...]]:
    if original == retained or not isinstance(retained, dict):
        return [keys]
    if not isinstance(original, dict):
        raise ValueError("Retained module fields must belong to the original tree")
    return [
        path
        for key, value in retained.items()
        for path in _preserved_paths(original[key], value, (*keys, key))
    ]


def _record_preserved(
    session: MigrationSession,
    source: Path,
    original: JsonObject,
    retained: JsonObject,
    target: Path,
) -> None:
    paths: list[tuple[str, ...]] = []
    for section, value in retained.items():
        section_original = original[section]
        if (
            section in {"modules", "waveforms"}
            and isinstance(value, dict)
            and isinstance(section_original, dict)
        ):
            for name, entry_raw in value.items():
                paths.extend(
                    _preserved_paths(section_original[name], entry_raw, (section, name))
                )
        else:
            paths.extend(_preserved_paths(section_original, value, (section,)))
    for keys in paths:
        _record_decision(
            session,
            KeyMappingItem(
                old_file=source,
                old_key=".".join(keys),
                new_file=target,
                new_path=None,
                action="preserved",
                source_keys=keys,
                destination_keys=("general", "ext", "legacy_module_cfg", *keys),
                reason="Unexpressed legacy module fields retained verbatim",
            ),
        )


def _prepare_entry(
    session: MigrationSession,
    candidate: LibrarySnapshot,
    item: _EntrySource,
    mapping: MigrationMapping,
    target: Path | None,
) -> tuple[LibrarySnapshot, JsonValue, bool]:
    """Isolate one unsupported/invalid entry without blocking valid siblings."""
    source, keys, raw = item.source, item.keys, item.raw
    location = ".".join(keys)
    rule = next(
        (
            rule
            for rule in mapping.module_rules
            if (rule.kind, rule.old_name) == (item.kind, keys[1])
        ),
        None,
    )
    if rule is None or not isinstance(raw, dict):
        return candidate, raw, False
    try:
        updated, remainder = _convert_spec(candidate, raw, rule, mapping)
    except _SpecInputError as exc:
        cause = exc.cause
        # Unknown discriminators are unsupported, not missing conversion evidence.
        if isinstance(cause, ValidationError) and all(
            error["type"] in {"union_tag_invalid", "union_tag_not_found"}
            for error in cause.errors()
        ):
            clear_pending(session, source, location)
            return candidate, raw, False
        reason = (
            f"Invalid or missing supported Spec evidence or child reference: {cause}"
        )
    else:
        clear_pending(session, source, location)
        _record_decision(
            session,
            KeyMappingItem(
                old_file=source,
                old_key=keys[1],
                library_kind=rule.kind,
                new_file=target,
                new_path=rule.target_path,
                action="module",
                source_keys=keys,
                reason=rule.reason,
            ),
        )
        return updated, remainder, True
    record_pending(session, source, location, reason)
    _record_decision(
        session,
        KeyMappingItem(
            old_file=source,
            old_key=keys[1],
            library_kind=rule.kind,
            new_file=target,
            new_path=rule.target_path,
            action="pending",
            source_keys=keys,
            reason=reason,
        ),
    )
    return candidate, raw, False


def _prepare_library(
    session: MigrationSession,
    source: Path,
    values: JsonObject,
    mapping: MigrationMapping,
    target: Path | None,
) -> tuple[LibrarySnapshot, JsonObject]:
    candidate = LibrarySnapshot(
        ModuleLibraryDocument(
            format=MODULE_LIBRARY_FORMAT,
            format_version=f"{MODULE_LIBRARY_VERSION.major}.{MODULE_LIBRARY_VERSION.minor}",
        )
    )
    retained: JsonObject = {
        key: value
        for key, value in values.items()
        if key not in ("modules", "waveforms")
    }
    for kind, partition in _PARTITIONS:
        if partition not in values:
            continue
        entries = values[partition]
        if not isinstance(entries, dict):
            record_pending(
                session, source, partition, "Expected a legacy library mapping"
            )
            continue
        for name, raw in entries.items():
            item = _EntrySource(source, kind, (partition, name), raw)
            candidate, remainder, converted = _prepare_entry(
                session, candidate, item, mapping, target
            )
            if remainder or not converted:
                branch = retained.setdefault(partition, {})
                if not isinstance(branch, dict):
                    raise ValueError("Expected retained library section mapping")
                branch[name] = remainder
    return candidate, retained


def _preserve_in_context(view: EditView, retained: JsonObject) -> None:
    if retained:
        current = document_value(snapshot(view), "general.ext")
        if not isinstance(current, dict):
            raise ValueError("Context general.ext must be a mapping")
        view.set("general.ext", {**current, "legacy_module_cfg": retained})


def _set_slots(
    session: MigrationSession,
    entry: ResultEntry,
    label: str | None,
    published: LibrarySnapshot,
    mapping: MigrationMapping,
) -> None:
    modules = published.module_keys()
    slots: set[str] = set()
    view = entry.setup if label is None else entry.use_point(label)
    with edit_parameter(
        session, session.manifest.report.source.result_path, view
    ) as draft:
        for rule in mapping.module_rules:
            if (
                rule.kind == "module"
                and rule.target_path in modules
                and rule.reference_path is not None
                and rule.reference_path not in slots
            ):
                draft.set(rule.reference_path, rule.target_path)
                slots.add(rule.reference_path)


def convert_module_cfg(
    session: MigrationSession,
    *,
    entry: ResultEntry | None,
    label: str | None,
    source: Path,
    mapping: MigrationMapping,
    dry_view: EditView | None = None,
) -> LibrarySnapshot | None:
    """Prepare/report one legacy library and commit its complete typed Spec file.

    session owns report, source baselines and manifest checkpoints. source is
    the absolute legacy module_cfg.yaml under its result root; an absent file
    returns None. Return the owner-validated same-context snapshot for scalar
    comparison: the isolated plan in dry-run, or the published library otherwise.
    label is the validated point label, or None for root metadata with setup
    destination. entry is the matching handle, or None only in dry-run.
    mapping supplies explicit module/waveform source keys and canonical targets.
    The legacy envelope requires modules and waveforms mappings; malformed
    envelopes raise MigrationInputError and retain a module-step failure diagnostic.
    Root entries use the setup library and set setup slots through the entry owner.
    Slots choose the first successfully published module rule per reference path.
    Known fields with expressions, invalid/missing values or unmapped child
    references remain located pending, with the complete entry retained in the
    context's general.ext.legacy_module_cfg. Unsupported families/entries/sections
    are preserved there without pending; successful entries retain only unknown
    fields, in their original nesting. The cfg_model partition owner keeps strict
    Spec validation. No expression is evaluated. Document/schema or I/O commit
    failures are recorded then propagate.

    dry_view is the isolated context draft required in dry-run, or None for
    execution. Dry-run validates retained fields in that draft and prepares an
    isolated typed library without destination/manifest writes.
    Normal execution uses one ModuleLibrary.edit, then records source/final hashes
    and converted file in the existing manifest/report checkpoint. Recorded module
    commits are not rewritten; resume verifies their hashes. Remaining pending
    entries do not undo module-step completion. No sources/data/assets are moved,
    no ledger events are invented, and no unrecorded-write recovery is provided.
    """
    if not source.is_file():
        return None
    source = contained_path(source, session.manifest.report.source.result_path)
    if entry is not None and any(
        item.source == source for item in session.manifest.module_files
    ):
        # The recovery owner already verified the fixed document baseline.
        return entry.module_library(label).snapshot()
    try:
        source_hash = session.baseline(source)
        with source.open("r", encoding="utf-8") as stream:
            envelope: object = YAML(typ="safe").load(stream)
        if (
            not isinstance(envelope, dict)
            or not isinstance(envelope.get("modules"), dict)
            or not isinstance(envelope.get("waveforms"), dict)
        ):
            raise MigrationInputError(
                f"{source}: expected waveforms/modules mapping envelope"
            )
        values = _JSON.validate_python(envelope, strict=True)
        context_target = session.manifest.report.destination.result_path / (
            "setup.yaml" if label is None else f"points/{label}/point.yaml"
        )
        if session.dry_run:
            candidate, retained = _prepare_library(
                session, source, values, mapping, None
            )
            if dry_view is None:
                raise ValueError(
                    "A dry-run library requires its isolated context draft"
                )
            _preserve_in_context(dry_view, retained)
            _record_preserved(session, source, values, retained, context_target)
            return candidate
        if entry is None:
            raise ValueError("A library commit requires its ResultEntry handle")
        library = entry.module_library(label)
        candidate, retained = _prepare_library(
            session, source, values, mapping, library.source_path
        )
        owned = (session.manifest.documents or {}).get(str(library.source_path))
        if owned is None or owned.source != source:

            def build(path: Path) -> None:
                shutil.copyfile(library.source_path, path)
                with DocumentStore(
                    path,
                    ModuleLibraryDocument,
                    format=MODULE_LIBRARY_FORMAT,
                    supported_version=MODULE_LIBRARY_VERSION,
                ).edit() as document:
                    planned = candidate.document()
                    document.components = planned.components
                    document.provenance = planned.provenance

            publish_document(session, source, library.source_path, "library", build)
            library.refresh()
        else:
            candidate = library.snapshot()
        _set_slots(session, entry, label, candidate, mapping)
        view = entry.setup if label is None else entry.use_point(label)
        with edit_parameter(session, source, view) as draft:
            _preserve_in_context(draft, retained)
        _record_preserved(session, source, values, retained, context_target)
        destination_hash = file_hash(library.source_path)
        publication = ModuleFilePublication(
            source=source,
            destination=library.source_path,
            source_hash=source_hash,
            destination_hash=destination_hash,
        )
        session.manifest = replace(
            session.manifest, module_files=(*session.manifest.module_files, publication)
        )
        report = session.manifest.report
        converted = tuple(
            item for item in report.converted_files if item.source != source
        )
        session.report(
            replace(
                report,
                converted_files=(
                    *converted,
                    FileMigrationItem(
                        source=source,
                        destination=library.source_path,
                        source_hash=source_hash,
                        destination_hash=destination_hash,
                        status="converted",
                    ),
                ),
            )
        )
        session.failure(source, "module_cfg", None)
        return candidate
    except Exception as exc:
        # This boundary persists module-step diagnostics, then propagates the failure.
        session.failure(source, "module_cfg", exc)
        raise
