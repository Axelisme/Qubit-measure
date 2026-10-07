"""Point-local typed legacy library conversion in the parameter migration step."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.cfg_model import (
    module_spec_from_raw,
    remap_module_references,
    remap_waveform_references,
    waveform_spec_from_raw,
)
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.context.library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import ResultEntry

from .errors import MigrationInputError
from .evidence import has_legacy_expression
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


def _record_decision(session: MigrationSession, decision: KeyMappingItem) -> None:
    report = session.manifest.report
    items = tuple(
        item
        for item in report.key_mappings
        if (item.old_file, item.old_key, item.library_kind)
        != (decision.old_file, decision.old_key, decision.library_kind)
    )
    session.report(replace(report, key_mappings=(*items, decision)))


def _convert_spec(
    candidate: LibrarySnapshot,
    raw: JsonValue,
    rule: ModuleRule,
    mapping: MigrationMapping,
) -> tuple[LibrarySnapshot, str | None]:
    """Return the updated candidate and any entry-level deferral diagnostic."""
    if not isinstance(raw, dict):
        return candidate, "Unsupported Spec: expected a mapping"
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
    # Only Spec/reference errors are entry-level deferrals. Destination
    # document validation must fail the whole module operation.
    if rule.kind == "module":
        try:
            module = remap_module_references(
                module_spec_from_raw(raw), modules=modules, waveforms=waveforms
            )
        except (ValueError, KeyError) as exc:
            return candidate, f"Unsupported Spec or unmapped child reference: {exc}"
        return candidate.replaced_module(rule.target_path, module), None
    try:
        waveform = remap_waveform_references(
            waveform_spec_from_raw(raw), waveforms=waveforms
        )
    except (ValueError, KeyError) as exc:
        return candidate, f"Unsupported Spec or unmapped child reference: {exc}"
    return candidate.replaced_waveform(rule.target_path, waveform), None


def _prepare_library(
    session: MigrationSession,
    source: Path,
    values: JsonObject,
    mapping: MigrationMapping,
    target: Path | None,
    label: str | None,
) -> LibrarySnapshot:
    candidate = LibrarySnapshot(
        ModuleLibraryDocument(
            format=MODULE_LIBRARY_FORMAT,
            format_version=str(MODULE_LIBRARY_VERSION),
        )
    )
    for key in values:
        if key not in ("modules", "waveforms"):
            record_pending(session, source, key, "Unsupported legacy library section")
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
            location = f"{partition}.{name}"
            rule = next(
                (
                    rule
                    for rule in mapping.module_rules
                    if (rule.kind, rule.old_name) == (kind, name)
                ),
                None,
            )
            reason = _pending_reason(raw, rule, label)
            if reason is None and rule is not None:
                candidate, reason = _convert_spec(candidate, raw, rule, mapping)
            if reason is not None:
                record_pending(session, source, location, reason)
            else:
                clear_pending(session, source, location)
                reason = rule.reason if rule is not None else "Converted typed Spec"
            _record_decision(
                session,
                KeyMappingItem(
                    old_file=source,
                    old_key=name,
                    library_kind=kind,
                    new_file=target,
                    new_path=rule.target_path if rule is not None else None,
                    action="module"
                    if rule is not None and label is not None
                    else "pending",
                    reason=reason,
                ),
            )
    return candidate


def _pending_reason(
    raw: JsonValue, rule: ModuleRule | None, label: str | None
) -> str | None:
    if label is None:
        return "Root library has no point destination; explicit point ownership is required"
    if rule is None:
        return "No explicit library mapping"
    if has_legacy_expression(raw):
        return "Legacy expression preserved without evaluation"
    return None


def _set_slots(
    entry: ResultEntry, label: str, values: JsonObject, mapping: MigrationMapping
) -> None:
    modules = values.get("modules")
    if not isinstance(modules, dict):
        return
    slots: set[str] = set()
    with entry.use_point(label).edit() as draft:
        for rule in mapping.module_rules:
            if (
                rule.kind == "module"
                and rule.old_name in modules
                and rule.reference_path is not None
                and rule.reference_path not in slots
            ):
                # Slot priority is presence-based, even for unsupported Specs.
                draft.set(rule.reference_path, rule.target_path)
                slots.add(rule.reference_path)


def convert_module_cfg(
    session: MigrationSession,
    *,
    entry: ResultEntry | None,
    label: str | None,
    source: Path,
    mapping: MigrationMapping,
) -> None:
    """Prepare/report one legacy library and commit a point's complete Spec file.

    session owns report, source baselines and manifest checkpoints. source is
    the absolute legacy module_cfg.yaml under its result root; an absent file is
    a no-op. label is the validated point label, or None for root metadata with
    no point destination. entry is the matching handle, or None only in dry-run.
    mapping supplies explicit module/waveform source keys and canonical targets.
    The legacy envelope requires modules and waveforms mappings; malformed
    envelopes raise MigrationInputError and retain a module-step failure diagnostic.
    Root entries remain located pending with new_file=None and do not set slots.
    Child slots retain first-present module-rule priority. Unsupported expressions,
    Specs and unmapped child references remain located pending without evaluation.
    Document/schema or I/O commit failures are recorded then propagate.

    Dry-run prepares an isolated typed library without destination/manifest writes.
    Normal execution uses one ModuleLibrary.edit, then records source/final hashes
    and converted file in the existing manifest/report checkpoint. Recorded module
    commits are not rewritten; resume verifies their hashes. Remaining pending
    entries do not undo module-step completion. No sources/data/assets are moved,
    no ledger events are invented, and no unrecorded-write recovery is provided.
    """
    if not source.is_file():
        return
    source = contained_path(source, session.manifest.report.source.result_path)
    if any(item.source == source for item in session.manifest.module_files):
        # The manifest recovery owner already verified completed source/destination hashes.
        return
    try:
        source_hash = session.baseline(source)
        with source.open("r", encoding="utf-8") as stream:
            values = _JSON.validate_python(YAML(typ="safe").load(stream), strict=True)
        if not isinstance(values.get("modules"), dict) or not isinstance(
            values.get("waveforms"), dict
        ):
            raise MigrationInputError(
                f"{source}: expected waveforms/modules mapping envelope"
            )
        if label is None or session.dry_run:
            # No destination handle has been acquired; point dry-runs still
            # validate candidates and retain their canonical key decisions.
            _prepare_library(session, source, values, mapping, None, label)
            return
        if entry is None:
            raise ValueError("A point library commit requires its ResultEntry handle")
        library = entry.module_library(label)
        candidate = _prepare_library(
            session, source, values, mapping, library.source_path, label
        )
        _set_slots(entry, label, values, mapping)
        with library.edit() as draft:
            for path in candidate.module_keys():
                draft.set_module(path, candidate.module_spec(path))
            for path in candidate.waveform_keys():
                draft.set_waveform(path, candidate.waveform_spec(path))
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
    except Exception as exc:
        # This boundary persists module-step diagnostics, then propagates the failure.
        session.failure(source, "module_cfg", exc)
        raise
