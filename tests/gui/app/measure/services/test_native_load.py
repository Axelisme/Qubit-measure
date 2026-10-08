"""Native Load retains history and rejects unsafe saves through public services."""

from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import ResultEntry

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._native_save_support import (
    declare_panes,
    native_save_session,
)


@pytest.mark.parametrize("completion", ["complete", "partial", "stopped"])
def test_native_load_retains_history_and_resaves_captured_source(
    tmp_path: Path,
    qapp: QApplication,
    completion: Literal["complete", "partial", "stopped"],
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.metadata = replace(session.metadata, completion=completion)
        session.state.update_tab_result(
            "tab", session.state.get_tab("tab").run.result, metadata=session.metadata
        )
        session.save()
        path = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        stored = load_run_data(path)
        events = session.entry.ledger.events()
        original_capture = session.state.get_tab("tab").save_context
        other = ResultEntry.create(
            "other",
            result_root=tmp_path / "Elsewhere",
            database_root=tmp_path / "ElseDB",
        )
        other.new_point("other-point")
        session.context.install_entry(other)
        ContextControlFacet(context=session.context).use_point(
            PointUseRequest("other-point")
        )
        service = LoadService(session.state, MagicMock(), provide_options=lambda _: ())

        outcome = service.load_result("tab", str(path))

        tab = session.state.get_tab("tab")
        record = tab.run.result
        assert isinstance(record, RunRecord)
        assert isinstance(record.cfg, SavingCfg)
        assert isinstance(record.result, SavingResult)
        np.testing.assert_array_equal(record.result.signal, [1 + 2j, 3 - 4j, 5 + 6j])
        assert outcome.result_type == "RunRecord"
        assert tab.run.metadata == stored.metadata
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.save_context is original_capture
        assert session.entry.ledger.events() == events

        report = session.save()

        assert report.run_id == stored.metadata.run_id
        saved = load_run_data(path).metadata
        assert (
            replace(saved, labber_path=stored.metadata.labber_path) == stored.metadata
        )
        assert saved.labber_path != stored.metadata.labber_path
        assert all(event.kind == "saved" for event in session.entry.ledger.events())
        assert not other.ledger.events()


@pytest.mark.parametrize("foreign_entry", [False, True])
@pytest.mark.parametrize("save_all", [False, True])
def test_loaded_foreign_origin_save_rejects_before_mapping_or_io(
    tmp_path: Path,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    foreign_entry: bool,
    save_all: bool,
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        source = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        if foreign_entry:
            destination = ResultEntry.create(
                "destination",
                result_root=tmp_path / "Destination",
                database_root=tmp_path / "DestinationDB",
            )
            destination.new_point("destination-point")
            session.context.install_entry(destination)
        else:
            destination = session.entry
            destination.new_point("destination-point")
        control = ContextControlFacet(context=session.context)
        control.use_point(PointUseRequest("destination-point"))
        tab = session.state.get_tab("tab")
        tab.parameters = control.read_context()
        tab.save_context = tab.parameters
        captured = tab.save_context.capture_storage()
        service = LoadService(session.state, MagicMock(), provide_options=lambda _: ())
        service.load_result("tab", str(source))
        assert tab.run.metadata == load_run_data(source).metadata
        before = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }

        def reject_mapping(result: object, *, comment: str) -> None:
            raise AssertionError("Origin rejection must precede data mapping")

        monkeypatch.setattr(tab.adapter, "prepare_save_data", reject_mapping)
        save = session.save_artifacts if save_all else session.save
        with pytest.raises(FailedPreconditionError) as error:
            save()

        message = str(error.value)
        assert str(session.metadata.snapshot.entry_id) in message
        assert session.metadata.snapshot.point in message
        assert str(captured.entry_id) in message
        assert captured.point in message
        assert "save" in message.lower()
        after = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }
        assert after == before
        assert tab.run.confirmed_run_id is None


@pytest.mark.parametrize("save_all", [False, True])
def test_draft_capture_load_retains_history_and_save_rejects_without_writes(
    tmp_path: Path, qapp: QApplication, save_all: bool
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        declare_panes(session)
        path = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        stored = load_run_data(path)
        session.context.install_entry(session.entry)
        control = ContextControlFacet(context=session.context)
        draft = control.read_context()
        assert draft.storage is None
        tab = session.state.get_tab("tab")
        tab.parameters = draft
        tab.save_context = draft
        before = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }
        events = session.entry.ledger.events()
        service = LoadService(session.state, MagicMock(), provide_options=lambda _: ())

        outcome = service.load_result("tab", str(path))

        assert outcome.result_type == "RunRecord"
        assert isinstance(tab.run.result, RunRecord)
        assert tab.run.metadata == stored.metadata
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.save_context is draft
        assert tab.analysis.result is None
        assert tab.post_analysis.result is None
        assert tab.artifacts.snapshot() == ()
        # A valid active focus cannot fill this tab's missing captured destination.
        control.use_point(PointUseRequest("p"))
        save = session.save_artifacts if save_all else session.save
        with pytest.raises(FailedPreconditionError):
            save()
        assert session.entry.ledger.events() == events
        assert {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        } == before


def test_native_experiment_mismatch_keeps_previous_result_and_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        tab = session.state.get_tab("tab")
        session.state.update_tab_result(
            "tab",
            tab.run.result,
            metadata=replace(session.metadata, experiment="test/different"),
        )
        session.save()
        path = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        old_run = tab.run
        service = LoadService(session.state, MagicMock(), provide_options=lambda _: ())

        with pytest.raises(LoadDataError) as error:
            service.load_result("tab", str(path))

        assert error.value.reason_code == "invalid_data_file"
        assert "test/different" in str(error.value)
        assert tab.run is old_run


def test_unreadable_native_load_keeps_previous_result_and_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        tab = session.state.get_tab("tab")
        old_run = tab.run
        service = LoadService(session.state, MagicMock(), provide_options=lambda _: ())

        with pytest.raises(LoadDataError) as error:
            service.load_result("tab", str(tmp_path / "missing.h5"))

        assert error.value.reason_code == "data_file_read_failed"
        assert tab.run is old_run
        assert tab.run.metadata == session.metadata
