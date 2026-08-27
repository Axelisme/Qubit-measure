from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.app.main.services.tab import TabService
from zcu_tools.gui.app.main.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
    SavePaneState,
    Session,
    State,
)


def test_tab_snapshot_is_single_pure_render_model() -> None:
    state = State(
        ExpContext(
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
            cfg_schema=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock(), params=analyze_params),
            save=SavePaneState(data_path_override="data.h5"),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    # TabService's render model depends only on State + a writeback query port;
    # readiness / save paths come off State's aggregates, not sibling
    # app-services. The registry is unused by get_snapshot.
    service = TabService(state, MagicMock(), writeback)

    snapshot = service.get_snapshot("tab")

    assert snapshot.tab_id == "tab"
    assert snapshot.interaction is not None  # render path fills every live field
    assert snapshot.interaction.has_run_result is True
    assert snapshot.interaction.has_active_context is True  # ctx.readiness=ACTIVE
    assert snapshot.analysis is not None
    assert snapshot.analysis.params is analyze_params
    assert snapshot.paths is not None and snapshot.paths.data.path == "data.h5"
    assert state.get_tab("tab").analysis.params is analyze_params


def test_snapshot_projects_empty_writeback_draft_existence() -> None:
    state = _active_state()
    draft = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg_schema=MagicMock(),
            analysis=AnalysisPaneState(writeback_draft=draft),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []

    snapshot = TabService(state, MagicMock(), writeback).get_snapshot("tab")

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
            cfg_schema=MagicMock(),
            analysis=AnalysisPaneState(writeback_draft=draft),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.side_effect = RuntimeError("broken draft")

    with pytest.raises(RuntimeError, match="broken draft"):
        TabService(state, MagicMock(), writeback).get_snapshot("tab")


def test_snapshot_projects_running_owner_without_session_run_flag() -> None:
    state = _active_state()
    for tab_id in ("running", "idle"):
        state.add_tab(
            tab_id,
            Session(
                adapter_name="fake",
                adapter=MagicMock(),
                cfg_schema=MagicMock(),
            ),
        )
    state.set_tab_running("running", True)
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    service = TabService(state, MagicMock(), writeback)

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
        ExpContext(
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
            cfg_schema=MagicMock(),
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
    service = TabService(state, MagicMock(), writeback)

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
            cfg_schema=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock()),  # primary result present
        ),
    )
    service = TabService(state, MagicMock(), MagicMock())

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
            cfg_schema=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=None),  # no primary analyze result
        ),
    )
    service = TabService(state, MagicMock(), MagicMock())

    with pytest.raises(RuntimeError, match="primary analyze result"):
        service.initialize_tab_post_analyze_params("tab")
