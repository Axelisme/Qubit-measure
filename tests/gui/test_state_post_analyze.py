"""Unit tests for the post-analysis layer on State (方案 A parallel fields).

Covers: recording a post result requires a primary analyze result; and the
invalidation out-edges (run start / re-run / re-analyze) clear the post fields.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._cfg_fakes import make_cfg


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
            cfg=make_cfg(CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())),
        ),
    )
    return state


def _seed_analyze(state: State, tab_id: str = "t1") -> None:
    """Put the tab in a state where a primary analyze result exists."""
    state.update_tab_result(tab_id, object())
    state.update_tab_analyze(tab_id, MagicMock(), _plots_for(Figure()))


def test_update_post_analyze_requires_primary_result() -> None:
    state = _make_state()
    state.update_tab_result("t1", object())  # run result but no analyze result
    with pytest.raises(RuntimeError, match="no primary analyze result"):
        state.update_tab_post_analyze("t1", MagicMock(), _plots_for(Figure()))


def test_update_post_analyze_records_result_and_figure() -> None:
    state = _make_state()
    _seed_analyze(state)
    fig = Figure()
    post_result = MagicMock()
    state.update_tab_post_analyze("t1", post_result, _plots_for(fig))

    tab = state.get_tab("t1")
    assert tab.post_analysis.result is post_result
    assert tab.post_analysis.plots is not None
    assert tab.post_analysis.plots["fit"] is fig
    assert tab.has_post_analyze_result() is True


def test_post_analyze_invalidated_on_reanalyze() -> None:
    state = _make_state()
    _seed_analyze(state)
    state.update_tab_post_analyze("t1", MagicMock(), _plots_for(Figure()))
    assert state.get_tab("t1").has_post_analyze_result() is True

    # A re-analyze replaces the primary result the post built on → post cleared.
    state.update_tab_analyze("t1", MagicMock(), _plots_for(Figure()))
    tab = state.get_tab("t1")
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None
    assert tab.has_post_analyze_result() is False


def test_post_analyze_invalidated_on_rerun() -> None:
    state = _make_state()
    _seed_analyze(state)
    state.update_tab_post_analyze("t1", MagicMock(), _plots_for(Figure()))

    # A new run result clears both the primary analyze and the post-analysis.
    state.update_tab_result("t1", object())
    tab = state.get_tab("t1")
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None


def test_post_analyze_invalidated_on_clear_results() -> None:
    state = _make_state()
    _seed_analyze(state)
    state.update_tab_post_analyze("t1", MagicMock(), _plots_for(Figure()))

    state.clear_tab_results("t1")
    tab = state.get_tab("t1")
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None


def test_post_analyze_param_instance_round_trip() -> None:
    state = _make_state()
    sentinel = object()
    state.update_tab_post_analyze_param_instance("t1", sentinel)
    assert state.get_tab("t1").post_analysis.params is sentinel
