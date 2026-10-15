"""Opened-entry ledger Load query contracts through the shipped Save facet."""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.resources.entry import (
    AcquiredPayload,
    LedgerEvent,
    ResultEntry,
    SavedOutput,
    SavedPayload,
)

from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    native_save_session,
)


@pytest.fixture
def session(tmp_path: Path, qapp: QApplication) -> Iterator[NativeSaveSession]:
    with native_save_session(tmp_path, qapp) as active:
        yield active


def acquired(
    session: NativeSaveSession,
    run_id: str,
    *,
    point: str = "p",
    experiment: str = "scan",
    at: str = "2026-10-08T00:01:00Z",
) -> None:
    session.entry.ledger.append(
        LedgerEvent(
            id=str(uuid4()),
            kind="acquired",
            at=at,
            entry_id=session.entry.entry_id,
            origin="gui",
            payload=AcquiredPayload(
                run_id=run_id,
                tab="scan",
                experiment=experiment,
                point=point,
                roles={},
            ),
        )
    )


def saved(
    session: NativeSaveSession,
    run_id: str,
    path: Path,
    *,
    status: str = "saved",
    point: str = "p",
    experiment: str = "scan",
    at: str = "2026-10-08T00:02:00Z",
) -> str:
    output = SavedOutput.model_validate(
        {
            "artifact": "measurement_data",
            "member": "data",
            "format": "data_h5",
            "path": str(path),
            "status": status,
            "error": "Permission denied" if status == "failed" else None,
        }
    )
    event_id = str(uuid4())
    session.entry.ledger.append(
        LedgerEvent(
            id=event_id,
            kind="saved",
            at=at,
            entry_id=session.entry.entry_id,
            origin="gui",
            payload=SavedPayload(run_id=run_id, outputs=(output,)),
        ),
        record={"point": point, "experiment": experiment, "outputs": []},
    )
    return event_id


def test_query_merges_native_save_history_without_opening_data(
    session: NativeSaveSession, tmp_path: Path
) -> None:
    acquired(session, "opaque-run")
    first = tmp_path / "missing-first.h5"
    latest = tmp_path / "missing-latest.h5"
    saved(session, "opaque-run", first)
    saved(session, "opaque-run", latest)
    saved(session, "opaque-run", first, status="failed")
    before = session.entry.ledger.events()
    rows = session.control.list_saved_runs()
    assert len(rows) == 1
    row = rows[0]
    assert (row.run_id, row.point, row.experiment, row.recorded_at) == (
        "opaque-run",
        "p",
        "scan",
        "2026-10-08T00:01:00Z",
    )
    assert row.data_path == latest
    assert session.entry.ledger.events() == before
    assert not (session.entry.database_path / "runs").exists()


def test_query_saved_only_and_acquired_only_with_exact_point_filter(
    session: NativeSaveSession, tmp_path: Path
) -> None:
    saved(session, "saved-only", tmp_path / "gone.h5", point="p2", experiment="fit")
    acquired(session, "not-saved")
    rows = session.control.list_saved_runs()
    assert [row.run_id for row in rows] == ["saved-only", "not-saved"]
    assert rows[0].recorded_at == "2026-10-08T00:02:00Z"
    assert rows[0].experiment == "fit"
    assert rows[1].data_path is None
    assert session.control.list_saved_runs(point="p2") == (rows[0],)
    assert session.control.list_saved_runs(point="unknown") == ()


def test_query_acquired_facts_win_even_when_saved_event_came_first(
    session: NativeSaveSession, tmp_path: Path
) -> None:
    saved(session, "run", tmp_path / "gone.h5", point="other", experiment="other")
    acquired(session, "run", at="2026-10-08T00:03:00Z")
    acquired(session, "run", point="later", at="2026-10-08T00:04:00Z")
    (row,) = session.control.list_saved_runs()
    assert (row.point, row.experiment, row.recorded_at) == (
        "p",
        "scan",
        "2026-10-08T00:03:00Z",
    )


def test_query_real_shared_saver_attachment_is_consumed(
    session: NativeSaveSession,
) -> None:
    report = session.save()
    (row,) = session.control.list_saved_runs()
    assert row.run_id == session.metadata.run_id
    assert row.point == session.metadata.snapshot.point
    assert row.experiment == session.metadata.experiment
    native = next(
        item.output.path
        for item in report.outputs
        if item.output.format == "data_h5" and item.status == "saved"
    )
    assert row.data_path == native
    saved_event = session.entry.ledger.events()[0]
    assert row.recorded_at == saved_event.at


def test_query_empty_draft_and_no_opened_entry(session: NativeSaveSession) -> None:
    assert session.control.list_saved_runs() == ()
    session.context.install_entry(session.entry)
    assert session.control.list_saved_runs() == ()
    session.context.clear_context()
    with pytest.raises(FailedPreconditionError) as error:
        session.control.list_saved_runs()
    assert error.value.reason_code == "no_entry"


def test_query_ignores_non_native_outputs_and_follows_current_opened_entry(
    session: NativeSaveSession,
    tmp_path: Path,
) -> None:
    event_id = str(uuid4())
    session.entry.ledger.append(
        LedgerEvent(
            id=event_id,
            kind="saved",
            at="2026-10-08T00:02:00Z",
            entry_id=session.entry.entry_id,
            origin="gui",
            payload=SavedPayload(
                run_id="png-only",
                outputs=(
                    SavedOutput(
                        artifact="fit",
                        member="figure",
                        format="png",
                        path=str(tmp_path / "fit.png"),
                        status="saved",
                    ),
                ),
            ),
        ),
        record={"point": "p", "experiment": "scan"},
    )
    (row,) = session.control.list_saved_runs()
    assert row.data_path is None
    other = ResultEntry.create(
        "other",
        result_root=tmp_path / "Result",
        database_root=tmp_path / "Database",
    )
    session.context.install_entry(other)
    assert session.control.list_saved_runs() == ()
    assert len(session.entry.ledger.events()) == 1


@pytest.mark.parametrize(
    "failure",
    ["corrupt-ledger", "missing-record", "malformed-record", "invalid-fields"],
)
def test_query_persistence_faults_are_not_an_empty_list(
    session: NativeSaveSession, tmp_path: Path, failure: str
) -> None:
    event_id = saved(session, "run", tmp_path / "gone.h5")
    records = session.entry.result_path / "records"
    if failure == "corrupt-ledger":
        (records / "ledger.jsonl").write_text("not-json\n", encoding="utf-8")
    elif failure == "missing-record":
        (records / f"{event_id}.json").unlink()
    elif failure == "malformed-record":
        (records / f"{event_id}.json").write_text("not-json", encoding="utf-8")
    else:
        (records / f"{event_id}.json").write_text(
            '{"point": 42, "experiment": "scan"}', encoding="utf-8"
        )
    expected = FileNotFoundError if failure == "missing-record" else ValueError
    with pytest.raises(expected):
        session.control.list_saved_runs()
