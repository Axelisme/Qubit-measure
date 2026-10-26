"""Operator-declared device attribution for complete JSONL record histories."""

import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import uuid4

from filelock import FileLock, Timeout
from pydantic import JsonValue, TypeAdapter

from zcu_tools.cfg_model import validate_device_name
from zcu_tools.utils import atomic_replace


@dataclass(frozen=True)
class RecordRowUpgrade:
    """Classification of one physical line, numbered from one.

    classification is old (complete movable pair), new (no old pair), partial
    (missing pair member or conflict), or corrupt (invalid JSON/envelope/value).
    error explains a rejected line; None denotes a valid old or new row.
    """

    line_number: int
    classification: Literal["old", "new", "partial", "corrupt"]
    error: str | None = None


@dataclass(frozen=True)
class RecordUpgradeResult:
    """One file result with the exact operator-declared device_name.

    path is the requested source. status is dry-run, applied, skipped (no old
    rows), or rejected (no replacement). rows lists every physical line.
    backup_path names a completed original-bytes backup, even if replacement
    fails. error is a file-level I/O diagnostic, otherwise None.
    """

    path: Path
    device_name: str
    status: Literal["dry-run", "applied", "skipped", "rejected"]
    rows: tuple[RecordRowUpgrade, ...] = ()
    backup_path: Path | None = None
    error: str | None = None


def upgrade_record_files(
    paths: Sequence[Path], *, device_name: str, apply: bool = False
) -> tuple[RecordUpgradeResult, ...]:
    """Attribute old dev_value/dev_unit pairs to an operator-declared device.

    paths selects exact JSONL histories, relative to cwd when not absolute.
    device_name must be nonblank without ASCII dots, otherwise raise ValueError
    before I/O; preserve valid names verbatim. Old/target pairs require a finite
    non-boolean real value and nonblank string unit, preserved without conversion.
    Target metadata is retained; old fields coexisting with target pair fields
    conflict, never overwrite. Other devices and no-device rows stay opaque.
    Default dry-run creates no files. apply backs up original bytes before atomic
    replacement. Reject
    a whole file on any partial/corrupt row, report physical line numbers, and
    continue other files. Preserve all rows, duplicate run IDs and opaque fields;
    never append, infer component/frame, or change component schema versions.
    File/lock failures return rejected results. There is no multi-file transaction
    or power-loss guarantee. Stop concurrent writers before applying.
    """
    validate_device_name(device_name)
    return tuple(_upgrade_file(path, device_name, apply=apply) for path in paths)


class _PartialRecord(ValueError):
    """Incomplete upgrade pair or coexisting old/target fields."""


def _check_pair(
    fields: dict[str, JsonValue], value_key: str, unit_key: str, location: str
) -> bool:
    has_value, has_unit = value_key in fields, unit_key in fields
    if has_value != has_unit:
        missing = unit_key if has_value else value_key
        raise _PartialRecord(f"{location}: missing {missing}")
    if not has_value:
        return False
    value, unit = fields[value_key], fields[unit_key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not isfinite(value))
    ):
        raise ValueError(f"{location}.{value_key}: expected finite non-boolean real")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"{location}.{unit_key}: expected nonblank unit string")
    return True


def _convert_line(
    raw: bytes, number: int, device_name: str
) -> tuple[bytes, RecordRowUpgrade]:
    try:
        if not raw.endswith(b"\n"):
            raise ValueError("incomplete tail: expected LF")
        record = TypeAdapter(dict[str, JsonValue]).validate_json(raw)
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        devices = record.get("devices", {})
        if not isinstance(devices, dict):
            raise ValueError("devices must be an object")
        target = devices.get(device_name, {})
        if not isinstance(target, dict):
            raise ValueError(f"devices[{device_name!r}] must be an object")
        has_old = "dev_value" in record or "dev_unit" in record
        if has_old and ("value" in target or "value_unit" in target):
            raise _PartialRecord(
                f"conflict: old fields coexist with devices[{device_name!r}] value/value_unit"
            )
        old = _check_pair(record, "dev_value", "dev_unit", "row")
        _check_pair(target, "value", "value_unit", f"devices[{device_name!r}]")
    except _PartialRecord as error:
        return raw, RecordRowUpgrade(number, "partial", str(error))
    except ValueError as error:
        return raw, RecordRowUpgrade(number, "corrupt", str(error))
    if not old:
        return raw, RecordRowUpgrade(number, "new")
    target["value"] = record.pop("dev_value")
    target["value_unit"] = record.pop("dev_unit")
    devices[device_name] = target
    record["devices"] = devices
    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    return encoded, RecordRowUpgrade(number, "old")


def _replace_history(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        atomic_replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _upgrade_file(path: Path, device_name: str, *, apply: bool) -> RecordUpgradeResult:
    resolved = path.resolve()
    backup: Path | None = None
    rows: list[RecordRowUpgrade] = []
    try:
        resolved.parent.stat()
        lock = FileLock(f"{resolved}.lock", timeout=5) if apply else nullcontext()
        with lock:
            original = resolved.read_bytes()
            output: list[bytes] = []
            changed = False
            lines = original.split(b"\n")
            for number, chunk in enumerate(lines, start=1):
                if number == len(lines) and not chunk:
                    break
                line = chunk + b"\n" if number < len(lines) else chunk
                converted, row = _convert_line(line, number, device_name)
                rows.append(row)
                output.append(converted)
                changed = changed or row.classification == "old"
            if any(row.error is not None for row in rows):
                return RecordUpgradeResult(path, device_name, "rejected", tuple(rows))
            if not changed:
                return RecordUpgradeResult(path, device_name, "skipped", tuple(rows))
            if not apply:
                return RecordUpgradeResult(path, device_name, "dry-run", tuple(rows))
            destination = resolved.with_name(f"{resolved.name}.{uuid4().hex}.bak")
            with destination.open("xb") as stream:
                stream.write(original)
            backup = destination
            _replace_history(resolved, b"".join(output))
            return RecordUpgradeResult(
                path, device_name, "applied", tuple(rows), backup
            )
    except (OSError, ValueError, Timeout) as error:
        return RecordUpgradeResult(
            path, device_name, "rejected", tuple(rows), backup, str(error)
        )
