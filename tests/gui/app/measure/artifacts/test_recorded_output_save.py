"""Recorded-name and storage-free native Save output contracts."""

from dataclasses import replace
from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
from zcu_tools.datafile import load
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.gui.app.measure.artifacts import SaveStatus
from zcu_tools.gui.app.measure.services.pane_commit import commit_run
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.resources.entry import ResultEntry

from tests.gui.app.measure._native_save_support import NativeSaveSession, declare_panes
from tests.gui.app.measure._record_support import require_record


@pytest.mark.parametrize(
    "entry_name,point",
    [(None, None), ("recorded", None), (None, "historic"), ("absent", "gone")],
)
@pytest.mark.parametrize("fail_figure", [False, True])
def test_storage_free_save_keeps_declarations_and_publishes_recorded_outputs(
    native_session: NativeSaveSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_name: str | None,
    point: str | None,
    fail_figure: bool,
) -> None:
    session = native_session
    metadata = session.metadata.model_copy(
        update={"snapshot": RunSnapshot(entry_name=entry_name, point=point, roles={})}
    )
    original = require_record(session.state.get_tab("tab").run)
    commit_run(
        session.state, "tab", RunPaneState(result=replace(original, metadata=metadata))
    )
    plots = declare_panes(session)
    before = session.state.get_artifact_snapshots("tab")
    assert {section.section for section in before} == {
        "run",
        "analysis",
        "post_analysis",
    }
    assert all(
        artifact.members and artifact.status is SaveStatus.NOT_SAVED
        for section in before
        for artifact in section.artifacts
    )
    assert all(
        not member.outputs
        for section in before
        for artifact in section.artifacts
        for member in artifact.members
    )
    source_bytes = {
        path: path.read_bytes()
        for path in session.entry.result_path.rglob("*")
        if path.is_file()
    }

    if fail_figure:

        def failed_png(*args: object, **kwargs: object) -> bytes:
            raise OSError("synthetic PNG failure")

        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.artifacts.save.render_save_figure_png",
            failed_png,
        )
    try:
        report = session.save_artifacts()
        native = next(
            item for item in report.outputs if item.output.format == "data_h5"
        )
        run_dir = (
            tmp_path
            / "Database"
            / (entry_name or "unnamed_entry")
            / "runs"
            / f"saving_{metadata.run_id}"
        )
        assert native.status == "saved", native.error
        assert native.output.path == run_dir / "data.h5"
        restored = load(native.output.path)
        assert metadata_from_container(restored) == metadata
        variable = next(iter(restored.vars.values()))
        np.testing.assert_allclose(variable.values, original.result.signal)
        np.testing.assert_allclose(variable.axes[0].values, [1e9, 1.001e9, 1.002e9])
        assert list((run_dir / "analysis").glob("*.json"))
        figures = [item for item in report.outputs if item.output.format == "png"]
        assert figures
        assert {item.status for item in figures} == (
            {"failed"} if fail_figure else {"saved"}
        )
        copies = [
            item
            for item in figures
            if item.output.path.is_relative_to(tmp_path / "result")
        ]
        assert copies
        assert all(
            item.output.path.parent
            == tmp_path
            / "result"
            / (entry_name or "unnamed_entry")
            / "points"
            / (point or "unnamed_point")
            / "figures"
            for item in copies
        )
        after = session.state.get_artifact_snapshots("tab")
        assert all(
            member.outputs
            for section in after
            for artifact in section.artifacts
            for member in artifact.members
        )
        assert {
            output.status
            for section in after
            for artifact in section.artifacts
            for member in artifact.members
            for output in member.outputs
        } == ({"saved", "failed"} if fail_figure else {"saved"})
        assert all(
            path.read_bytes() == content for path, content in source_bytes.items()
        )
        assert session.entry.ledger.events() == ()
        second = session.save_artifacts()
        assert (
            next(
                item.output.path
                for item in second.outputs
                if item.output.format == "data_h5"
            )
            == native.output.path
        )
        assert metadata_from_container(load(native.output.path)) == metadata
    finally:
        plots.release()


def test_open_with_only_setup_and_points_saves_before_first_ledger_append(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    records = session.entry.result_path / "records"
    if records.exists():
        rmtree(records)
    (session.entry.result_path / "module_cfg.yaml").unlink()
    opened = ResultEntry.open(
        session.entry.result_path.name,
        result_root=session.entry.result_path.parent,
        database_root=session.entry.database_path.parent,
    )
    assert opened.ledger.events() == ()
    assert not records.exists()
    report = session.save()
    assert all(item.status == "saved" for item in report.outputs)
    assert [event.kind for event in opened.ledger.events()] == ["saved"]
    assert records.is_dir()
    native = next(
        item.output.path for item in report.outputs if item.output.format == "data_h5"
    )
    assert metadata_from_container(load(native)) == session.metadata
