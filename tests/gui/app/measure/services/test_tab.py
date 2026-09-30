from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ContextReadiness,
    SessionEnv,
)
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKind, SaveStatus
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
    SavePaneState,
    Session,
    State,
)

from tests.gui.app.measure._cfg_fakes import cfg_resources


def test_tab_snapshot_is_single_pure_render_model() -> None:
    state = State(
        SessionEnv(
            md=MagicMock(),
            ml=MagicMock(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )
    analyze_params = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock(), params=analyze_params),
            save=SavePaneState(data_path_override="data.h5", comment="draft note"),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    # TabService's render model depends only on State + a writeback query port;
    # readiness / save paths come off State's aggregates, not sibling
    # app-services. The registry is unused by get_snapshot.
    service = TabService(state, MagicMock(), writeback, cfg_resources(state))

    snapshot = service.get_snapshot("tab")

    assert snapshot.tab_id == "tab"
    assert snapshot.interaction is not None  # render path fills every live field
    assert snapshot.interaction.has_run_result is True
    assert snapshot.interaction.has_active_context is True  # ctx.readiness=ACTIVE
    assert snapshot.analysis is not None
    assert snapshot.analysis.params is analyze_params
    assert snapshot.paths is not None and snapshot.paths.data.path == "data.h5"
    assert snapshot.save is not None
    assert asdict(snapshot.save)["comment"] == "draft note"
    assert state.get_tab("tab").analysis.params is analyze_params


def test_snapshot_projects_empty_writeback_draft_existence() -> None:
    state = _active_state()
    draft = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
            analysis=AnalysisPaneState(writeback_draft=draft),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []

    snapshot = TabService(
        state, MagicMock(), writeback, cfg_resources(state)
    ).get_snapshot("tab")

    assert snapshot.analysis is not None
    assert snapshot.analysis.has_writeback_draft is True
    assert snapshot.analysis.writeback_items == ()


def test_snapshot_propagates_writeback_preview_failure() -> None:
    state = _active_state()
    draft = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
            analysis=AnalysisPaneState(writeback_draft=draft),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.side_effect = RuntimeError("broken draft")

    with pytest.raises(RuntimeError, match="broken draft"):
        TabService(state, MagicMock(), writeback, cfg_resources(state)).get_snapshot(
            "tab"
        )


def test_snapshot_projects_running_owner_without_session_run_flag() -> None:
    state = _active_state()
    for tab_id in ("running", "idle"):
        state.add_tab(
            tab_id,
            Session(
                adapter_name="fake",
                adapter=MagicMock(),
                cfg=MagicMock(),
            ),
        )
    state.set_tab_running("running", True)
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    service = TabService(state, MagicMock(), writeback, cfg_resources(state))

    running = service.get_snapshot("running").interaction
    idle = service.get_snapshot("idle").interaction

    assert running is not None
    assert running.is_running is True
    assert running.global_run_active is False
    assert idle is not None
    assert idle.is_running is False
    assert idle.global_run_active is True


def _active_state() -> State:
    return State(
        SessionEnv(
            md=MagicMock(),
            ml=MagicMock(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )


def test_snapshot_carries_post_analyze_fields() -> None:
    state = _active_state()
    post_params = object()
    post_fig = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=MagicMock(),
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock()),
            post_analysis=PostAnalysisPaneState(
                result=MagicMock(),
                params=post_params,
                figure=post_fig,  # type: ignore[arg-type]
            ),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    service = TabService(state, MagicMock(), writeback, cfg_resources(state))

    snapshot = service.get_snapshot("tab")

    assert snapshot.post_analysis is not None
    assert snapshot.post_analysis.params is post_params
    assert snapshot.post_analysis.figure is post_fig
    assert snapshot.interaction is not None
    assert snapshot.interaction.has_post_analyze_result is True


def test_initialize_post_analyze_params_seeds_from_primary_result() -> None:
    state = _active_state()
    adapter = MagicMock()
    built = object()
    adapter.get_post_analyze_params.return_value = built
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=adapter,
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock()),  # primary result present
        ),
    )
    service = TabService(state, MagicMock(), MagicMock(), cfg_resources(state))

    out = service.initialize_tab_post_analyze_params("tab")

    assert out is built
    assert state.get_tab("tab").post_analysis.params is built


def test_initialize_post_analyze_params_fast_fails_without_primary_result() -> None:
    import pytest

    state = _active_state()
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=MagicMock(),
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=None),  # no primary analyze result
        ),
    )
    service = TabService(state, MagicMock(), MagicMock(), cfg_resources(state))

    with pytest.raises(RuntimeError, match="primary analyze result"):
        service.initialize_tab_post_analyze_params("tab")


def test_tab_snapshot_loaded_artifacts_have_no_success_record() -> None:
    state = _active_state()
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True
    )
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg=MagicMock(),
            save=SavePaneState(data_path_override="data.hdf5"),
            analysis=AnalysisPaneState(image_path_override="analysis.png"),
        ),
    )
    state.update_tab_loaded_result("tab", object(), "existing.hdf5")

    snapshot = TabService(
        state, MagicMock(), MagicMock(), cfg_resources(state)
    ).get_snapshot("tab")

    assert snapshot.run is not None and snapshot.run.source_path == "existing.hdf5"
    assert [(a.kind, a.status) for a in snapshot.artifacts] == [
        (ArtifactKind.DATA, SaveStatus.NOT_SAVED),
        (ArtifactKind.ANALYSIS, SaveStatus.NO_RESULT),
        (ArtifactKind.POST_ANALYSIS, SaveStatus.NO_RESULT),
    ]
    data = snapshot.artifacts[0]
    assert data.default_path == "data.hdf5"
    assert data.is_saveable is True
    assert data.last_saved_path is None


def test_tab_snapshot_tracks_actual_success_and_draft_or_result_drift() -> None:
    state = _active_state()
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg=MagicMock(),
            save=SavePaneState(data_path_override="data.hdf5"),
        ),
    )
    state.update_tab_result("tab", object())
    state.update_tab_comment("tab", "initial")
    service = TabService(state, MagicMock(), MagicMock(), cfg_resources(state))
    assert service.get_snapshot("tab").artifacts[0].status is SaveStatus.NOT_SAVED

    state.get_tab("tab").artifacts.started(ArtifactKind.DATA)
    state.update_tab_comment("tab", "edited during save")
    state.get_tab("tab").artifacts.succeeded(ArtifactKind.DATA, "data_1.hdf5")
    data = service.get_snapshot("tab").artifacts[0]
    assert data.status is SaveStatus.UNSAVED_CHANGES
    assert data.default_path == "data.hdf5"
    assert data.last_saved_path == "data_1.hdf5"

    state.update_tab_comment("tab", "initial")
    assert service.get_snapshot("tab").artifacts[0].status is SaveStatus.SAVED
    state.update_tab_result("tab", object())
    assert service.get_snapshot("tab").artifacts[0].status is SaveStatus.UNSAVED_CHANGES
    state.update_tab_loaded_result("tab", object(), "imported.hdf5")
    loaded = service.get_snapshot("tab").artifacts[0]
    assert loaded.status is SaveStatus.NOT_SAVED
    assert loaded.last_saved_path is None


def test_artifact_failure_preserves_previous_success_but_does_not_mark_new_draft_saved() -> (
    None
):
    state = _active_state()
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg=MagicMock(),
            save=SavePaneState(data_path_override="first.hdf5"),
        ),
    )
    state.update_tab_result("tab", object())
    service = TabService(state, MagicMock(), MagicMock(), cfg_resources(state))
    assert service.get_snapshot("tab").artifacts[0].status is SaveStatus.NOT_SAVED
    tracker = state.get_tab("tab").artifacts
    tracker.started(ArtifactKind.DATA)
    tracker.succeeded(ArtifactKind.DATA, "first_1.hdf5")
    assert service.get_snapshot("tab").artifacts[0].status is SaveStatus.SAVED

    state.update_tab_data_path_override("tab", "next.hdf5")
    assert service.get_snapshot("tab").artifacts[0].status is SaveStatus.UNSAVED_CHANGES
    tracker.started(ArtifactKind.DATA)
    tracker.failed(ArtifactKind.DATA)
    after_failure = service.get_snapshot("tab").artifacts[0]
    assert after_failure.status is SaveStatus.UNSAVED_CHANGES
    assert after_failure.last_saved_path == "first_1.hdf5"
    assert after_failure.default_path == "next.hdf5"
