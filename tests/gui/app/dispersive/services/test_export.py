"""Dispersive export uses the entry artifact and declared component fields."""

import pytest
from zcu_tools.gui.app.dispersive.services.export import ExportService
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.entry import ResultEntry, snapshot


def test_export_writes_entry_artifact_without_adding_component_fields(analysis_target):
    state = DispersiveState(analysis_target)
    state.set_disp_result(g=0.068, bare_rf=5.35, res_dim=4)
    before = snapshot(analysis_target.resource.setup)

    path = ExportService(state).export_params()

    expected = analysis_target.resource.dispersive_path / "q1.json"
    assert path == str(expected)
    saved = load_dispersive_file(expected)
    assert saved.params.g == 0.068
    assert saved.params.bare_rf == 5.35
    assert saved.source.run_id is None
    assert saved.source.created_at is not None
    assert snapshot(analysis_target.resource.setup) == before


def test_export_updates_only_declared_component_fields(analysis_target):
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget

    target = AnalysisTarget.open(
        analysis_target.result_root.parent, analysis_target.entry, "q2"
    )
    state = DispersiveState(target)
    state.set_disp_result(g=0.068, bare_rf=5.35, res_dim=4)

    ExportService(state).export_params()

    reopened = ResultEntry.open(
        target.entry, result_root=target.result_root, database_root=target.result_root
    )
    assert snapshot(reopened.setup).components["q2"].model_dump()["g"] == 0.068
    assert (
        snapshot(reopened.setup).components["q1"]
        == snapshot(target.resource.setup).components["q1"]
    )


def test_export_without_result_fast_fails(analysis_target):
    with pytest.raises(RuntimeError, match="no dispersive fit result"):
        ExportService(DispersiveState(analysis_target)).export_params()


def test_export_without_entry_fast_fails():
    state = DispersiveState()
    state.set_disp_result(g=0.068, bare_rf=5.35, res_dim=4)
    with pytest.raises(RuntimeError, match="entry"):
        ExportService(state).export_params()
