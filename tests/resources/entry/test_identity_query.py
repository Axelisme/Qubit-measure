"""Historical entry identity queries do not depend on component definitions."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

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
def setup(tmp_path: Path) -> Path:
    source = tmp_path / "entry" / "setup.yaml"
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
    setup: Path,
) -> None:
    before = {p: p.read_bytes() for p in setup.parent.rglob("*") if p.is_file()}
    try:
        identity = ResultEntry.read_identity("entry", result_root=setup.parent.parent)
    except UnknownKindError as error:
        pytest.fail(f"identity query must not require component registration: {error}")
    assert identity == EntryIdentity(
        entry_id=UUID(_ID), created_at=datetime(2026, 10, 23, tzinfo=timezone.utc)
    )
    assert {p: p.read_bytes() for p in setup.parent.rglob("*") if p.is_file()} == before


def test_event_snapshot_does_not_resolve_components_or_publish_files(
    setup: Path,
) -> None:
    records = setup.parent / "records"
    records.mkdir()
    event = LedgerEvent(
        id=str(uuid4()),
        entry_id=_ID,
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
    RecordsLedger(records, entry_id=_ID).append(event)
    for lock in records.glob("*.lock"):
        lock.unlink()
    before = {p: p.read_bytes() for p in setup.parent.rglob("*") if p.is_file()}
    assert ResultEntry.read_events_snapshot(
        "entry", result_root=setup.parent.parent
    ) == (event,)
    assert {p: p.read_bytes() for p in setup.parent.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("missing", ["entry", "records", "ledger"])
def test_missing_event_history_is_empty_without_creating_storage(
    setup: Path, missing: str
) -> None:
    root = setup.parent.parent
    name = "absent" if missing == "entry" else "entry"
    if missing == "ledger":
        (setup.parent / "records").mkdir()
    before = set(root.rglob("*"))
    assert ResultEntry.read_events_snapshot(name, result_root=root) == ()
    assert set(root.rglob("*")) == before


@pytest.mark.parametrize("redirect", ["root", "entry", "records", "ledger.jsonl"])
def test_event_snapshot_rejects_redirected_storage(setup: Path, redirect: str) -> None:
    root = setup.parent.parent
    records = setup.parent / "records"
    records.mkdir()
    ledger = records / "ledger.jsonl"
    ledger.touch()
    source = {
        "root": root,
        "entry": setup.parent,
        "records": records,
        "ledger.jsonl": ledger,
    }[redirect]
    moved = root.with_name(f"{root.name}-detached")
    source.rename(moved)
    try:
        source.symlink_to(moved, target_is_directory=redirect != "ledger.jsonl")
        with pytest.raises(ValueError, match="escapes"):
            ResultEntry.read_events_snapshot("entry", result_root=root)
    finally:
        source.unlink()
        moved.rename(source)


@pytest.mark.parametrize(
    "field,bad", [("entry_id", "bad-id"), ("created_at", "2026-10-23")]
)
def test_identity_rejects_invalid_identity(setup: Path, field: str, bad: str) -> None:
    valid = _ID if field == "entry_id" else "2026-10-23T00:00:00Z"
    setup.write_text(
        setup.read_text(encoding="utf-8").replace(valid, bad), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=field):
        ResultEntry.read_identity("entry", result_root=setup.parent.parent)


def test_identity_rejects_entry_symlink_escape(setup: Path, tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    (root / "entry").symlink_to(setup.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        ResultEntry.read_identity("entry", result_root=root)
