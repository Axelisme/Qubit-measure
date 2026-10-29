"""Output previews and first Save through the real Controller composition."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.datafile import load, save
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.session.context_types import CommandOrigin

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    PaneParams,
)


@pytest.fixture
def native_controller(
    native_session: NativeSaveSession, tmp_path: Path
) -> Generator[tuple[Controller, str]]:
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    ctrl = Controller(
        native_session.state,
        registry,
        None,
        native_session.bus,
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    tab_id = ctrl.tab_control.new_tab("saving")
    try:
        yield ctrl, tab_id
    finally:
        ctrl.close()
        ctrl.tab_control.close_tab(tab_id)


@pytest.mark.parametrize("malformed_snapshot", [False, True])
def test_controller_load_previews_without_rejecting_unreadable_snapshot(
    native_session: NativeSaveSession,
    native_controller: tuple[Controller, str],
    malformed_snapshot: bool,
) -> None:
    session = native_session
    ctrl, tab_id = native_controller
    session.save()
    source = session.native_data_path
    if malformed_snapshot:
        stored = load(source)
        stored.docs["metadata"]["snapshot"] = "malformed"
        save(source, stored, replace=True)
    original_bytes = source.read_bytes()
    ctrl.run_analyze_control.load_tab_result(tab_id, str(source))
    loaded = ctrl.tab_control.get_tab_snapshot(tab_id)
    outputs = [
        output
        for section in loaded.artifacts
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    ]
    assert loaded.run is not None and loaded.run.run_id is None
    if malformed_snapshot:
        assert outputs == []
        with pytest.raises(ValueError, match="snapshot"):
            ctrl.save_control.save_artifacts(tab_id, origin=CommandOrigin("gui"))
        assert source.read_bytes() == original_bytes
        return
    assert {output.path for output in outputs if output.format == "data_h5"} == {source}
    ctrl.run_analyze_control.analyze(tab_id, PaneParams())
    qt_until(lambda: ctrl.active_operation_count() == 0, label="primary analysis")
    ctrl.run_analyze_control.start_post_analyze(tab_id, PaneParams())
    qt_until(lambda: ctrl.active_operation_count() == 0, label="post analysis")
    analyzed = ctrl.tab_control.get_tab_snapshot(tab_id)
    planned = {
        output.path
        for section in analyzed.artifacts
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    }
    assert all(
        member.outputs
        for section in analyzed.artifacts
        for artifact in section.artifacts
        for member in artifact.members
    )
    report = ctrl.save_control.save_artifacts(tab_id, origin=CommandOrigin("gui"))
    assert planned == {item.output.path for item in report.outputs}
    assert all(item.status == "saved" for item in report.outputs)
    assert all(path.is_file() for path in planned)
