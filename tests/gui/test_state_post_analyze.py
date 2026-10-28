"""Unit tests for the post-analysis layer on State (方案 A parallel fields).

Covers: recording a post result requires a primary analyze result; and the
invalidation out-edges (run start / re-run / re-analyze) clear the post fields.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.services.pane_commit import (
    commit_analysis,
    commit_post_analysis,
    commit_run,
)
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
    Session,
    State,
)
from zcu_tools.gui.form_adapter import Form
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._form_fakes import make_form


def _plots_for(figure: Figure) -> Plots:
    plots = Plots(NonPresentingHost())
    plots.adopt("fit", figure)
    plots.finish()
    plots.release()
    return plots


def _make_state(tab_id: str = "t1") -> State:
    state = State(MagicMock())
    state.add_tab(
        tab_id,
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=make_form(Form(ConfigBase, {}), state=state),
        ),
    )
    return state


def _seed_analyze(state: State, tab_id: str = "t1") -> None:
    """Put the tab in a state where a primary analyze result exists."""
    commit_run(state, tab_id, RunPaneState(result=RunRecord(None, object())))
    commit_analysis(
        state,
        tab_id,
        AnalysisPaneState(
            result=MagicMock(),
            plots=_plots_for(Figure()),
            params=state.get_tab(tab_id).analysis.params,
        ),
    )


def test_update_post_analyze_requires_primary_result() -> None:
    state = _make_state()
    commit_run(
        state, "t1", RunPaneState(result=RunRecord(None, object()))
    )  # run result but no analyze result
    with pytest.raises(RuntimeError, match="no primary analyze result"):
        commit_post_analysis(
            state,
            "t1",
            PostAnalysisPaneState(
                result=MagicMock(),
                plots=_plots_for(Figure()),
                params=state.get_tab("t1").post_analysis.params,
            ),
        )


def test_update_post_analyze_records_result_and_figure() -> None:
    state = _make_state()
    _seed_analyze(state)
    fig = Figure()
    post_result = MagicMock()
    commit_post_analysis(
        state,
        "t1",
        PostAnalysisPaneState(
            result=post_result,
            plots=_plots_for(fig),
            params=state.get_tab("t1").post_analysis.params,
        ),
    )

    tab = state.get_tab("t1")
    assert tab.post_analysis.result is post_result
    assert tab.post_analysis.plots is not None
    assert tab.post_analysis.plots["fit"] is fig
    assert tab.has_post_analyze_result() is True


def test_post_analyze_invalidated_on_reanalyze() -> None:
    state = _make_state()
    _seed_analyze(state)
    commit_post_analysis(
        state,
        "t1",
        PostAnalysisPaneState(
            result=MagicMock(),
            plots=_plots_for(Figure()),
            params=state.get_tab("t1").post_analysis.params,
        ),
    )
    assert state.get_tab("t1").has_post_analyze_result() is True

    # A re-analyze replaces the primary result the post built on → post cleared.
    commit_analysis(
        state,
        "t1",
        AnalysisPaneState(
            result=MagicMock(),
            plots=_plots_for(Figure()),
            params=state.get_tab("t1").analysis.params,
        ),
    )
    tab = state.get_tab("t1")
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None
    assert tab.has_post_analyze_result() is False


def test_post_analyze_invalidated_on_rerun() -> None:
    state = _make_state()
    _seed_analyze(state)
    commit_post_analysis(
        state,
        "t1",
        PostAnalysisPaneState(
            result=MagicMock(),
            plots=_plots_for(Figure()),
            params=state.get_tab("t1").post_analysis.params,
        ),
    )

    # A new run result clears both the primary analyze and the post-analysis.
    commit_run(state, "t1", RunPaneState(result=RunRecord(None, object())))
    tab = state.get_tab("t1")
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None


def test_post_analyze_invalidated_on_clear_results() -> None:
    state = _make_state()
    _seed_analyze(state)
    commit_post_analysis(
        state,
        "t1",
        PostAnalysisPaneState(
            result=MagicMock(),
            plots=_plots_for(Figure()),
            params=state.get_tab("t1").post_analysis.params,
        ),
    )

    commit_run(state, "t1", RunPaneState())
    tab = state.get_tab("t1")
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None


def test_post_analyze_param_instance_round_trip() -> None:
    state = _make_state()
    sentinel = object()
    state.update_tab_post_analyze_param_instance("t1", sentinel)
    assert state.get_tab("t1").post_analysis.params is sentinel
