"""File-backed Load listing through the shipped Save control and Load seams."""

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import h5py as h5
import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QTableWidget
from zcu_tools.datafile import save_run_data
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog
from zcu_tools.gui.app.measure.writeback import WritebackService
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.resources.entry import (
    AcquiredPayload,
    LedgerEvent,
    ResultEntry,
    SavedPayload,
)

from tests._native_support import native_cfg, native_payload
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    native_save_session,
)


@pytest.fixture
def session(tmp_path: Path, qapp: QApplication) -> Iterator[NativeSaveSession]:
    with native_save_session(tmp_path, qapp) as active:
        yield active


def native_file(session: NativeSaveSession, *, completion: str = "complete") -> Path:
    """Write a synthetic native run at its canonical entry path."""
    path = (
        session.entry.database_path
        / "runs"
        / f"scan_{session.metadata.run_id}"
        / "data.h5"
    )
    path.parent.mkdir(parents=True)
    metadata = session.metadata
    if completion == "unknown":
        metadata = replace(
            metadata, started_at=None, finished_at=None, completion="unknown"
        )
    save_run_data(path, native_payload(), metadata, cfg=native_cfg())
    return path


def acquired(session: NativeSaveSession, run_id: str) -> None:
    session.entry.ledger.append(
        LedgerEvent(
            id=str(uuid4()),
            kind="acquired",
            at="2026-10-08T00:02:00Z",
            entry_id=session.entry.entry_id,
            origin="gui",
            payload=AcquiredPayload(
                run_id=run_id,
                tab="scan",
                experiment="ledger/tag",
                point="other",
                roles={},
            ),
        )
    )


def test_query_lists_native_files_without_ledger_and_keeps_unknown_history(
    session: NativeSaveSession,
) -> None:
    path = native_file(session, completion="unknown")
    rows = session.control.list_saved_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row.data_path == path
    assert (row.run_id, row.point, row.experiment) == (
        session.metadata.run_id,
        "p",
        session.metadata.experiment,
    )
    assert row.started_at is None
    assert row.finished_at is None
    assert row.completion == "unknown"
    assert row.metadata_error is None
    assert session.entry.ledger.events() == ()
    assert session.control.list_saved_runs(point="p") == (row,)
    assert session.control.list_saved_runs(point="other") == ()


def test_query_uses_native_history_for_rows_despite_unrelated_ledger_events(
    session: NativeSaveSession,
) -> None:
    path = native_file(session)
    acquired(session, "ledger-only")
    acquired(session, session.metadata.run_id)
    before = session.entry.ledger.events()
    rows = session.control.list_saved_runs()
    assert [row.data_path for row in rows] == [path]
    assert (
        rows[0].point,
        rows[0].experiment,
        rows[0].started_at,
        rows[0].finished_at,
        rows[0].completion,
    ) == (
        "p",
        session.metadata.experiment,
        session.metadata.started_at,
        session.metadata.finished_at,
        "complete",
    )
    assert session.entry.ledger.events() == before


@pytest.mark.parametrize(
    "fault", ["invalid-hdf5", "invalid-metadata", "missing-payload"]
)
def test_query_metadata_and_load_validity_are_independent(
    session: NativeSaveSession,
    qapp: QApplication,
    fault: str,
) -> None:
    path = native_file(session)
    if fault == "invalid-hdf5":
        path.write_bytes(b"not HDF5")
    else:
        with h5.File(path, "r+") as file:
            if fault == "invalid-metadata":
                del file.attrs["experiment"]
            else:
                del file["data"]
    rows = session.control.list_saved_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row.data_path == path
    if fault == "missing-payload":
        assert row.metadata_error is None
        assert row.experiment == session.metadata.experiment
    else:
        assert row.metadata_error
        assert row.run_id == ""
        assert row.experiment == ""
        assert row.started_at is None
        assert row.completion is None

    dialog = RunLoadDialog(rows, initial_point="", start_directory="")
    try:
        table = dialog.findChild(QTableWidget, "load_runs")
        load = dialog.findChild(QPushButton, "load_selected")
        assert table is not None and load is not None
        if row.metadata_error:
            texts = []
            for column in range(table.columnCount()):
                item = table.item(0, column)
                assert item is not None
                texts.append(item.text())
            assert any(row.metadata_error in text for text in texts)
        table.selectRow(0)
        assert load.isEnabled()
        load.click()
        assert dialog.selected_path() == str(path)
        service = LoadService(
            session.state,
            WritebackService(
                ContextControlFacet(context=session.context), arb_names=lambda: ()
            ),
            context=ContextControlFacet(context=session.context),
            captured_parameters=lambda _aliases: None,
        )
        with pytest.raises(LoadDataError) as error:
            service.load_result("tab", dialog.selected_path() or "")
        assert error.value.reason_code == (
            "data_file_read_failed" if fault == "invalid-hdf5" else "invalid_data_file"
        )
    finally:
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("fault", ["corrupt-ledger", "missing-attachment"])
def test_query_ledger_fault_does_not_hide_valid_or_broken_files(
    session: NativeSaveSession,
    fault: str,
) -> None:
    valid = native_file(session)
    broken = session.entry.database_path / "runs" / "broken" / "data.h5"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not HDF5")
    event_id = str(uuid4())
    session.entry.ledger.append(
        LedgerEvent(
            id=event_id,
            kind="saved",
            at="2026-10-08T00:03:00Z",
            entry_id=session.entry.entry_id,
            origin="gui",
            payload=SavedPayload(run_id=session.metadata.run_id, outputs=()),
        ),
        record={"point": "p", "experiment": "scan"},
    )
    records = session.entry.result_path / "records"
    if fault == "corrupt-ledger":
        (records / "ledger.jsonl").write_text("not-json\n", encoding="utf-8")
    else:
        (records / f"{event_id}.json").unlink()
    rows = session.control.list_saved_runs()
    assert {row.data_path for row in rows} == {valid, broken}
    errors = {row.data_path: row.metadata_error for row in rows}
    assert errors[valid] is None
    assert errors[broken]


def test_query_follows_current_entry_and_ignores_noncanonical_files(
    session: NativeSaveSession,
    tmp_path: Path,
) -> None:
    native_file(session)
    other = ResultEntry.create(
        "other",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    decoy = other.database_path / "Labber" / "data.h5"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"not a run")
    session.context.install_entry(other)
    assert session.control.list_saved_runs() == ()
    session.context.install_entry(session.entry)
    assert len(session.control.list_saved_runs()) == 1


def test_query_real_shared_saver_file_is_listed(session: NativeSaveSession) -> None:
    report = session.save()
    (row,) = session.control.list_saved_runs()
    native = next(
        item.output.path
        for item in report.outputs
        if item.output.format == "data_h5" and item.status == "saved"
    )
    assert row.data_path == native
    assert row.run_id == session.metadata.run_id
    assert row.point == session.metadata.snapshot.point
    assert row.experiment == session.metadata.experiment


def test_unreadable_metadata_selection_uses_exact_path_without_inferred_identity(
    session: NativeSaveSession, qapp: QApplication, tmp_path: Path
) -> None:
    path = native_file(session)
    with h5.File(path, "r+") as file:
        del file.attrs["experiment"]
    (row,) = session.control.list_saved_runs()
    dialog = RunLoadDialog((row,), initial_point="", start_directory="")
    images = Path(os.environ.get("LOAD_FALLBACK_EVIDENCE", str(tmp_path / "images")))
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.screenshot(images / "01-unreadable.png")
        (visible,) = driver.table_rows(object_name="load_runs")
        assert visible.columns[:6] == ("", "", "", "", "", "")
        assert row.metadata_error
        assert visible.columns[6] == f"Metadata error: {row.metadata_error}"
        assert str(path) in visible.tooltips[6]
        assert not driver.observe("Load selected").enabled

        driver.enter_text(object_name="load_started_at", value="2025", commit=False)
        driver.screenshot(images / "02-filtered.png")
        assert driver.table_rows(object_name="load_runs") == ()
        driver.enter_text(object_name="load_started_at", value="", commit=False)
        driver.screenshot(images / "03-restored.png")
        driver.select_table_row(visible.columns[6], object_name="load_runs")
        driver.screenshot(images / "04-selected.png")
        assert driver.observe("Load selected").enabled
        assert driver.table_rows(object_name="load_runs")[0].selected
        driver.click("Load selected")
        assert not driver.observe(object_name="load_runs").visible
        assert dialog.selected_path() == str(path)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_query_empty_entry_and_no_opened_entry(session: NativeSaveSession) -> None:
    assert session.control.list_saved_runs() == ()
    session.context.clear_context()
    with pytest.raises(FailedPreconditionError) as error:
        session.control.list_saved_runs()
    assert error.value.reason_code == "no_entry"
