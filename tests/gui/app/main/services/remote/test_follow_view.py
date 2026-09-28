"""Explicit write commands select the view before starting their mutation."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import AnalysisMode
from zcu_tools.gui.app.main.services.remote.handlers.analysis import (
    h_tab_analyze,
    h_tab_post_analyze,
)
from zcu_tools.gui.app.main.services.remote.handlers.run_save import (
    h_tab_run_start,
    h_tab_save_artifacts,
)
from zcu_tools.gui.app.main.services.remote.handlers.tab import h_tab_set_cfg


@dataclass
class Params:
    threshold: float = 0.5


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize(
    "handler,pane,method",
    [
        (h_tab_run_start, "run", "start_run"),
        (h_tab_analyze, "analysis", "analyze"),
        (h_tab_post_analyze, "post_analysis", "start_post_analyze"),
        (h_tab_set_cfg, "run", "cfg_editor_set_fields"),
    ],
)
def test_write_follow_precedes_mutation_and_headless_still_works(
    handler, pane, method, headless
):
    adapter = MagicMock()
    control = adapter.run_analyze_control
    control.get_tab_snapshot.return_value = SimpleNamespace(
        interaction=None,
        capabilities=SimpleNamespace(analysis=AnalysisMode.FIT),
        analysis=SimpleNamespace(params=Params(), has_writeback_draft=False),
        post_analysis=SimpleNamespace(
            params=Params(), has_writeback_draft=False, result=None
        ),
    )
    adapter.tab_control.get_running_tab_id.return_value = None
    owner = adapter.ctrl if method == "cfg_editor_set_fields" else control
    order = []
    if headless:
        adapter.render_view = None
    else:
        adapter.render_view.select_tab_pane.side_effect = lambda tab, selected: (
            order.append((tab, selected))
        )
    getattr(owner, method).side_effect = lambda *args, **kwargs: (
        order.append("mutation") or MagicMock()
    )
    handler(adapter, {"tab_id": "t", "updates": {}, "edits": [], "agent_edit": True})
    assert order == (["mutation"] if headless else [("t", pane), "mutation"])


@pytest.mark.parametrize("headless", [False, True])
def test_save_selects_data_before_start_and_supports_headless(headless):
    adapter = MagicMock()
    order = []
    if headless:
        adapter.render_view = None
    else:
        adapter.render_view.select_tab_pane.side_effect = lambda tab, pane: (
            order.append((tab, pane))
        )

    def save(*args, **kwargs):
        order.append("save")
        return SimpleNamespace(operation_id=7, destinations=())

    adapter.save_control.save_artifacts.side_effect = save
    result = h_tab_save_artifacts(
        adapter,
        {
            "tab_id": "t",
            "artifacts": "all",
            "paths": {},
            "comment": None,
        },
    )
    assert result == {"operation_id": 7, "destinations": {}}
    assert order == (["save"] if headless else [("t", "data"), "save"])


def test_failed_save_view_selection_does_not_start_save():
    adapter = MagicMock()
    adapter.render_view.select_tab_pane.side_effect = ValueError("unavailable pane")
    with pytest.raises(ValueError, match="unavailable pane"):
        h_tab_save_artifacts(
            adapter,
            {
                "tab_id": "t",
                "artifacts": "all",
                "paths": {},
                "comment": None,
            },
        )
    adapter.save_control.save_artifacts.assert_not_called()


def test_failed_view_selection_does_not_start_an_operation():
    adapter = MagicMock()
    failure = ValueError("unavailable pane")
    adapter.render_view.select_tab_pane.side_effect = failure
    with pytest.raises(ValueError, match="unavailable pane") as caught:
        h_tab_run_start(adapter, {"tab_id": "t"})
    assert caught.value is failure
    adapter.run_analyze_control.start_run.assert_not_called()
