"""Storage capture through bound point and ledger public contracts."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from zcu_tools.resources.entry import (
    AcquiredPayload,
    LedgerEvent,
    ResultEntry,
    capture_point_storage,
    snapshot,
)


def test_capture_retains_cached_context_and_exact_entry_attribution(
    tmp_path: Path,
) -> None:
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    point = entry.new_point("sweet", description="cached")
    newer = entry.use_point("sweet")
    newer.description = "published later"
    before = tuple(sorted(tmp_path.rglob("*")))
    captured = capture_point_storage(point)

    assert captured.entry_id == UUID(snapshot(entry.setup).general.entry_id)
    assert captured.entry_name == "entry"
    assert captured.point == "sweet"
    assert captured.description == "cached"
    assert captured.result_path == entry.result_path
    assert captured.database_path == entry.database_path
    assert tuple(sorted(tmp_path.rglob("*"))) == before

    event = LedgerEvent(
        id=str(uuid4()),
        kind="acquired",
        at=datetime.now(UTC).isoformat(),
        entry_id=str(captured.entry_id),
        origin="notebook",
        payload=AcquiredPayload(
            run_id="captured-run",
            tab="synthetic",
            experiment="test/saving",
            cfg_summary={},
            point=captured.point,
            roles={},
        ),
    )
    captured.ledger.append(event)
    assert entry.ledger.get(event.id).event == event
