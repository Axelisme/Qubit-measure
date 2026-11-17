"""Historical entry identity queries do not depend on component definitions."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from zcu_tools.resources.entry import (
    AcquiredPayload,
    EntryIdentity,
    LedgerEvent,
    RecordsLedger,
    ResultEntry,
    UnknownKindError,
)

_ID = "340b3db4-a422-4b1c-98b5-00413aed0d1d"


@pytest.fixture
def global_parameters(tmp_path: Path) -> Path:
    source = tmp_path / "entry" / "parameters.yaml"
    source.parent.mkdir()
    source.write_text(
        'format: zcu.parameter-container\nformat_version: "2.0"\n'
        f'general:\n  entry_id: "{_ID}"\n  created_at: "2026-10-23T00:00:00Z"\n'
        "components:\n  Q1:\n    kind: not-registered-for-browser/qubit\n"
        "    historical_field: 123\n",
        encoding="utf-8",
    )
    return source


def test_identity_does_not_resolve_component_kinds_or_publish_files(
    global_parameters: Path,
) -> None:
    before = {
        p: p.read_bytes() for p in global_parameters.parent.rglob("*") if p.is_file()
    }
    try:
        identity = ResultEntry.read_identity(
            "entry", result_root=global_parameters.parent.parent
        )
    except UnknownKindError as error:
        pytest.fail(f"identity query must not require component registration: {error}")
    assert identity == EntryIdentity(
        created_at=datetime(2026, 10, 23, tzinfo=timezone.utc)
    )
    assert {
        p: p.read_bytes() for p in global_parameters.parent.rglob("*") if p.is_file()
    } == before


def test_event_snapshot_does_not_resolve_components_or_publish_files(
    global_parameters: Path,
) -> None:
    records = global_parameters.parent / "records"
    records.mkdir()
    event = LedgerEvent(
        id=str(uuid4()),
        kind="acquired",
        at="2026-10-23T01:00:00Z",
        origin="gui",
        payload=AcquiredPayload(
            run_id="opaque-run",
            experiment="synthetic/tag",
            point="historical",
            roles={"qubit": "Q1"},
        ),
    )
    RecordsLedger(records).append(event)
    for lock in records.glob("*.lock"):
        lock.unlink()
    before = {
        p: p.read_bytes() for p in global_parameters.parent.rglob("*") if p.is_file()
    }
    assert ResultEntry.read_events_snapshot(
        "entry", result_root=global_parameters.parent.parent
    ) == (event,)
    assert {
        p: p.read_bytes() for p in global_parameters.parent.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("missing", ["entry", "records", "ledger"])
def test_missing_event_history_is_empty_without_creating_storage(
    global_parameters: Path, missing: str
) -> None:
    root = global_parameters.parent.parent
    name = "absent" if missing == "entry" else "entry"
    if missing == "ledger":
        (global_parameters.parent / "records").mkdir()
    before = set(root.rglob("*"))
    assert ResultEntry.read_events_snapshot(name, result_root=root) == ()
    assert ResultEntry.read_contents_snapshot(name, result_root=root) == ()
    assert set(root.rglob("*")) == before


@pytest.mark.parametrize("redirect", ["root", "entry", "records", "ledger.jsonl"])
def test_event_snapshot_rejects_redirected_storage(
    global_parameters: Path, redirect: str
) -> None:
    root = global_parameters.parent.parent
    records = global_parameters.parent / "records"
    records.mkdir()
    ledger = records / "ledger.jsonl"
    ledger.touch()
    source = {
        "root": root,
        "entry": global_parameters.parent,
        "records": records,
        "ledger.jsonl": ledger,
    }[redirect]
    moved = root.with_name(f"{root.name}-detached")
    source.rename(moved)
    try:
        source.symlink_to(moved, target_is_directory=redirect != "ledger.jsonl")
        with pytest.raises(ValueError, match="escapes"):
            ResultEntry.read_events_snapshot("entry", result_root=root)
        with pytest.raises(ValueError, match="escapes"):
            ResultEntry.read_contents_snapshot("entry", result_root=root)
    finally:
        source.unlink()
        moved.rename(source)


@pytest.mark.parametrize("field,bad", [("created_at", "2026-10-23")])
def test_identity_rejects_invalid_identity(
    global_parameters: Path, field: str, bad: str
) -> None:
    valid = "2026-10-23T00:00:00Z"
    global_parameters.write_text(
        global_parameters.read_text(encoding="utf-8").replace(valid, bad),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=field):
        ResultEntry.read_identity("entry", result_root=global_parameters.parent.parent)


def test_identity_rejects_entry_symlink_escape(
    global_parameters: Path, tmp_path: Path
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    (root / "entry").symlink_to(global_parameters.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        ResultEntry.read_identity("entry", result_root=root)
