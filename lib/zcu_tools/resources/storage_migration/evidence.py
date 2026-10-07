"""Explicit JSON acquisition evidence; no filename inference or live capture."""

import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from zcu_tools.datafile import JsonObject, json_values_equal

from .errors import MigrationInputError
from .models import LegacyRunEvidence, MigrationRequest, MigrationRunEvidenceDocument
from .paths import source_identity, source_removed
from .state import MigrationSession

_JSON = TypeAdapter(JsonObject)


@dataclass(frozen=True, kw_only=True)
class _EvidenceEnvelope:
    format: str
    format_version: str
    entries: tuple[LegacyRunEvidence, ...]


_ENVELOPE = TypeAdapter(_EvidenceEnvelope)


def load_run_evidence(source: Path) -> MigrationRunEvidenceDocument:
    """Read a caller-selected UTF-8 JSON acquisition evidence document.

    Accept zcu.migration-run-evidence with major 1 and any nonnegative minor.
    Return typed entries and the full detached raw document, including unknown
    root/nested fields. Reject missing/wrong types, duplicate relative sources,
    absolute/'..'/empty paths, invalid lowercase SHA256 or non-UTC acquisition
    times with MigrationInputError identifying source and field. Never infer
    experiment, point, completion or acquisition time from paths or file dates.
    Missing cfg or missing/null/empty cfg_type projects cfg=None, retaining the
    unresolved raw JSON for the converter to report pending, never a fallback.
    Actual source containment (including symlinks), hash/metadata agreement and
    declared experiment schema checks belong to migrate_storage.
    Propagate filesystem errors; do not return an empty success on read failure.
    """
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise MigrationInputError(f"{source}: {exc}") from exc
    return parse_run_evidence(text, source=source)


def parse_run_evidence(text: str, *, source: Path) -> MigrationRunEvidenceDocument:
    """Validate UTF-8 JSON text with the load_run_evidence contract.

    source labels diagnostics only; no filesystem access occurs. Return typed
    entries plus complete raw fields. Malformed JSON, unsupported headers and
    invalid entry identities/times raise MigrationInputError naming source.
    """
    try:
        raw = _JSON.validate_python(
            json.loads(
                text,
                parse_constant=_reject_json_constant,
                parse_float=_finite_json_float,
            ),
            strict=True,
        )
        envelope = _ENVELOPE.validate_json(
            json.dumps(_evidence_projection(raw)), strict=True
        )
        _validate_header(envelope)
        seen: set[Path] = set()
        for index, entry in enumerate(envelope.entries):
            _validate_entry(entry, index)
            if entry.source in seen:
                raise ValueError(f"entries[{index}].source: duplicate {entry.source}")
            seen.add(entry.source)
    except (UnicodeError, ValidationError, ValueError) as exc:
        raise MigrationInputError(f"{source}: {exc}") from exc
    return MigrationRunEvidenceDocument(
        format=envelope.format,
        format_version=envelope.format_version,
        entries=envelope.entries,
        raw=raw,
    )


def _evidence_projection(raw: JsonObject) -> JsonObject:
    projected = dict(raw)
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return projected
    projected_entries: list[JsonValue] = []
    for entry in entries:
        if not isinstance(entry, dict):
            projected_entries.append(entry)
            continue
        cfg = entry.get("cfg")
        if cfg is None or (isinstance(cfg, dict) and cfg.get("cfg_type") in (None, "")):
            if isinstance(cfg, dict) and "values" in cfg:
                try:
                    _JSON.validate_python(cfg["values"], strict=True)
                except ValidationError as exc:
                    raise ValueError(f"cfg.values: {exc}") from exc
            projected_entries.append({**entry, "cfg": None})
        else:
            projected_entries.append(entry)
    projected["entries"] = projected_entries
    return projected


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON: nonfinite numeric value {value}")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON: nonfinite numeric constant {value}")


def _validate_header(envelope: _EvidenceEnvelope) -> None:
    if envelope.format != "zcu.migration-run-evidence":
        raise ValueError("format: expected zcu.migration-run-evidence")
    version = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)", envelope.format_version)
    if version is None:
        raise ValueError("format_version: expected major.minor")
    if version[1] != "1":
        raise ValueError(f"format_version: unsupported major {version[1]}")


def _utc_time(value: str, location: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{location}: expected UTC ISO acquisition time") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{location}: expected UTC ISO acquisition time")
    return parsed


def _validate_entry(entry: LegacyRunEvidence, index: int) -> None:
    location = f"entries[{index}]"
    if (
        entry.source.is_absolute()
        or ".." in entry.source.parts
        or not entry.source.parts
        or entry.source.anchor
    ):
        raise ValueError(
            f"{location}.source: expected a nonempty contained relative path"
        )
    if re.fullmatch(r"[0-9a-f]{64}", entry.source_hash) is None:
        raise ValueError(f"{location}.source_hash: expected lowercase SHA256")
    if not entry.experiment:
        raise ValueError(f"{location}.experiment: expected a declared nonempty tag")
    started = _utc_time(entry.started_at, f"{location}.started_at")
    if entry.finished_at is not None:
        finished = _utc_time(entry.finished_at, f"{location}.finished_at")
        if finished < started:
            raise ValueError(f"{location}.finished_at: precedes started_at")


def has_legacy_expression(value: JsonValue) -> bool:
    """Detect recognized legacy expression strings recursively without evaluating them."""
    if isinstance(value, str):
        return value.startswith(("=", "${", "md.", "ml."))
    if isinstance(value, dict):
        return any(has_legacy_expression(child) for child in value.values())
    if isinstance(value, list):
        return any(has_legacy_expression(child) for child in value)
    return False


def _typed_projection(document: MigrationRunEvidenceDocument) -> JsonObject:
    envelope = _EvidenceEnvelope(
        format=document.format,
        format_version=document.format_version,
        entries=document.entries,
    )
    return _JSON.validate_python(
        _ENVELOPE.dump_python(envelope, mode="json"), strict=True
    )


def select_run_evidence(
    request: MigrationRequest,
    session: MigrationSession,
) -> MigrationRunEvidenceDocument | None:
    """Select stored/supplemental historical evidence and checkpoint its complete raw document.

    request may extend only unplanned sources; session owns the existing snapshot.
    Reject changed planned evidence, typed/raw disagreement, escaped or missing
    sources with MigrationInputError. Return the selected document or None.
    dry_run updates memory only; filesystem/checkpoint failures propagate."""
    stored = session.manifest.evidence
    incoming = request.run_evidence
    if incoming is None:
        return (
            None
            if stored is None
            else parse_run_evidence(
                json.dumps(stored, allow_nan=False), source=session.path
            )
        )
    checked = parse_run_evidence(
        json.dumps(incoming.raw, allow_nan=False), source=session.path
    )
    if not json_values_equal(_typed_projection(checked), _typed_projection(incoming)):
        raise MigrationInputError(
            "run_evidence: typed projection disagrees with raw document"
        )
    if stored is not None:
        old = parse_run_evidence(
            json.dumps(stored, allow_nan=False), source=session.path
        )
        old_entries = stored["entries"]
        new_entries = incoming.raw["entries"]
        if not isinstance(old_entries, list) or not isinstance(new_entries, list):
            raise MigrationInputError("run_evidence: entries must be a list")
        merged_entries = list(old_entries)
        for index, entry in enumerate(checked.entries):
            old_index = next(
                (
                    i
                    for i, prior in enumerate(old.entries)
                    if prior.source == entry.source
                ),
                None,
            )
            planned = any(
                item.source
                == (session.manifest.report.source.database_path / entry.source)
                for item in session.manifest.files
            )
            if old_index is not None:
                if planned and not json_values_equal(
                    old_entries[old_index], new_entries[index]
                ):
                    raise MigrationInputError(
                        f"{entry.source}: evidence for a planned source cannot change"
                    )
                merged_entries[old_index] = new_entries[index]
            else:
                if planned:
                    raise MigrationInputError(
                        f"{entry.source}: cannot add evidence to a planned source"
                    )
                merged_entries.append(new_entries[index])
        raw = dict(stored)
        raw.update(incoming.raw)
        raw["entries"] = merged_entries
        checked = parse_run_evidence(
            json.dumps(raw, allow_nan=False), source=session.path
        )
    source_root = session.manifest.report.source.database_path
    for entry in checked.entries:
        source = source_identity(source_root / entry.source, session.manifest)
        if source_removed(source, session.manifest):
            continue
        if not source.is_file() and not any(
            state.source == source
            and state.operation == "move_labber"
            and state.phase in ("published", "source_removed")
            for state in session.manifest.files
        ):
            raise MigrationInputError(f"{source}: evidence source does not exist")
    session.manifest = replace(session.manifest, evidence=checked.raw)
    session.checkpoint()
    return checked
