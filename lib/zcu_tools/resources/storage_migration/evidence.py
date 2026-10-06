"""Explicit JSON acquisition evidence; no filename inference or live capture."""

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from zcu_tools.datafile import JsonObject

from .errors import MigrationInputError
from .models import LegacyRunEvidence, MigrationRunEvidenceDocument

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
    Actual source containment (including symlinks), hash/metadata agreement and
    declared experiment schema checks belong to migrate_storage.
    Propagate filesystem errors; do not return an empty success on read failure.
    """
    try:
        text = source.read_text(encoding="utf-8")
        raw = _JSON.validate_python(
            json.loads(
                text,
                parse_constant=_reject_json_constant,
                parse_float=_finite_json_float,
            ),
            strict=True,
        )
        envelope = _ENVELOPE.validate_json(text, strict=True)
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
